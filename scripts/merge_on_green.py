#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serialize authorized squash merges using exact-head evidence and release freezes."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The privileged workflow invokes this file with Python -I. Load only the
# adjacent reviewed policy; never import the checkout's scripts package or let
# a root-level module shadow standard-library imports on the token-bearing job.
_policy_spec = importlib.util.spec_from_file_location(
    "_rapid_catalog_trust", Path(__file__).with_name("catalog_trust.py")
)
assert _policy_spec is not None and _policy_spec.loader is not None
_policy = importlib.util.module_from_spec(_policy_spec)
_policy_spec.loader.exec_module(_policy)
CATALOGS = _policy.CATALOGS
catalog_changes = _policy.catalog_changes
is_trust_control = _policy.is_trust_control
parse_catalog = _policy.parse_catalog


class APIError(RuntimeError):
    """An API request failed; callers must fail closed."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


@dataclass
class GitHub:
    repo: str
    token: str
    pat: str = ""

    def request(
        self, path: str, method: str = "GET", data: Any = None, *, pat: bool = False
    ) -> Any:
        token = self.pat if pat else self.token
        if not token:
            raise APIError(
                "RELEASE_PAT is required for merge and revert writes"
                if pat
                else "GitHub read token is required"
            )
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}/{path}",
            method=method,
            data=json.dumps(data).encode() if data is not None else None,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except (OSError, ValueError) as exc:
            raise APIError(
                f"GitHub API {method} {path} failed: {exc}", getattr(exc, "code", 0)
            ) from exc

    def items(
        self, path: str, key: str | None = None, *, max_pages: int | None = None
    ) -> list[dict]:
        """List all records, or a bounded history when max_pages is specified."""
        records = []
        separator = "&" if "?" in path else "?"
        for page in range(1, (1000 if max_pages is None else max_pages) + 1):
            payload = self.request(f"{path}{separator}per_page=100&page={page}")
            batch = payload[key] if key else payload
            if not isinstance(batch, list) or not all(
                isinstance(x, dict) for x in batch
            ):
                raise APIError(f"Malformed list from {path}")
            records.extend(batch)
            if len(batch) < 100:
                return records
        if max_pages is not None:
            return records
        raise APIError(f"Pagination limit exceeded for {path}")


def freeze(api: GitHub, manual: str = "") -> str:
    """An unavailable or undetermined release state always freezes merges."""
    if manual.lower() == "true":
        return "RELEASE_FREEZE is true"
    try:
        runs = api.items(
            "actions/workflows/auto-release.yml/runs",
            "workflow_runs",
            max_pages=1,
        )
        # Keep the recent snapshot across status transitions, while finding old
        # active reruns without walking the entire unfiltered workflow history.
        candidates = {run["id"]: run for run in runs}
        for status in ("queued", "in_progress", "waiting", "pending", "requested"):
            for run in api.items(
                f"actions/workflows/auto-release.yml/runs?status={status}",
                "workflow_runs",
            ):
                candidates[run["id"]] = run
        for run in candidates.values():
            if run["status"] == "completed":
                continue
            run = api.request(f"actions/runs/{run['id']}")
            if run["status"] == "completed":
                continue
            jobs = api.items(f"actions/runs/{run['id']}/jobs", "jobs")
            detect = [j for j in jobs if j["name"] == "detect"]
            if not detect or any(j["status"] != "completed" for j in detect):
                return f"Release run {run['id']} detection undetermined"
            if any(
                j["name"] != "detect"
                and (j["status"] != "completed" or j["conclusion"] != "skipped")
                for j in jobs
            ):
                return f"Release run {run['id']} is releasing"
            # With only detect visible, downstream jobs may not exist yet.
            if not any(j["name"] != "detect" for j in jobs):
                return f"Release run {run['id']} downstream jobs undetermined"
        head = api.request("commits/main")
        if head["commit"]["message"].startswith("chore: bump version to "):
            runs = api.items(
                "actions/workflows/auto-release.yml/runs?event=push"
                f"&head_sha={head['sha']}",
                "workflow_runs",
            )
            if not any(
                r["head_sha"] == head["sha"]
                and r["event"] == "push"
                and r["status"] == "completed"
                for r in runs
            ):
                return "Version-bump main head awaits completed auto-release push run"
        return ""
    except (APIError, KeyError, TypeError, ValueError) as exc:
        return f"Release freeze evaluation failed: {exc}"


CHECK_WORKFLOWS = {
    "tests": ".github/workflows/ci.yml",
    "type-check": ".github/workflows/ci.yml",
    "merge-lane-mac": ".github/workflows/ci.yml",
    "merge-lane-no-mac": ".github/workflows/ci.yml",
    "desktop-tests": ".github/workflows/rapid-mac-ci.yml",
    "version-bump-guard": ".github/workflows/version-check.yml",
}


def catalog_review_gate(api: GitHub, pr: dict) -> str:
    """Trusted merge controller independently checks catalog identity and approval."""
    try:
        files = api.items(f"pulls/{pr['number']}/files")
        if len(files) != pr.get("changed_files"):
            return "Cannot establish complete changed-file inventory for trust review"
        needs_review = False
        head_tree = None
        for entry in files:
            path = entry["filename"]
            previous = entry.get("previous_filename", "")
            needs_review |= is_trust_control(path) or is_trust_control(previous)
            if previous in CATALOGS and previous != path:
                needs_review = True
            if path not in CATALOGS:
                continue

            def read(ref, path=path):
                payload = api.request(f"contents/{path}?ref={ref}")
                if payload.get("encoding") != "base64":
                    raise ValueError("Catalog content is unavailable")
                return parse_catalog(
                    base64.b64decode(payload["content"]).decode("utf-8")
                )

            # Contents API dereferences symlinks. Check immutable Git modes
            # first so an alias cannot redirect future edits into an unguarded file.
            if entry["status"] != "removed":
                if head_tree is None:
                    tree = api.request(f"git/trees/{pr['head']['sha']}?recursive=1")
                    if tree.get("truncated") is not False:
                        raise ValueError("Catalog Git tree is incomplete")
                    head_tree = {item["path"]: item for item in tree["tree"]}
                item = head_tree[path]
                if item.get("type") != "blob" or item.get("mode") not in {
                    "100644",
                    "100755",
                }:
                    raise ValueError(
                        "Catalog must be a regular Git blob, not a symlink"
                    )
            renamed = entry["status"] == "renamed"
            needs_review |= renamed
            # A file entering the protected catalog set introduces new
            # identities; unreviewed data cannot grandfather missing pins.
            before = (
                {}
                if entry["status"] == "added" or (renamed and previous not in CATALOGS)
                else read(pr["base"]["sha"], previous if renamed else path)
            )
            after = {} if entry["status"] == "removed" else read(pr["head"]["sha"])
            changes, errors = catalog_changes(before, after, path)
            if errors:
                return "Catalog identity policy: " + "; ".join(errors)
            needs_review |= bool(changes)
        if not needs_review:
            return ""
        reviews = api.items(f"pulls/{pr['number']}/reviews")
        latest = {}
        for review in sorted(reviews, key=lambda item: item["id"]):
            user = review.get("user") or {}
            login = user.get("login")
            if (
                not login
                or user.get("type") != "User"
                or login == pr.get("user", {}).get("login")
                or review.get("state")
                not in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
            ):
                continue
            latest[login] = review
        approved = False
        for login, review in latest.items():
            permission = api.request(f"collaborators/{login}/permission").get(
                "permission"
            )
            if permission not in {"admin", "maintain", "write"}:
                continue
            if review["state"] == "CHANGES_REQUESTED":
                return "Catalog/trust change has outstanding human change requests"
            if (
                review["state"] == "APPROVED"
                and review.get("commit_id") == pr["head"]["sha"]
            ):
                approved = True
        if not approved:
            return "Catalog/trust change requires human maintainer approval of the current head"
        return ""
    except (APIError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        return f"Catalog/trust review could not be verified: {exc}"


def eligibility(api: GitHub, pr: dict) -> str:
    """Return the first rejection reason, or an empty string for eligible heads."""
    labels = {x["name"] for x in pr["labels"]}
    ready = labels & {"merge-ready", "merge-ready-mac"}
    if pr["state"] != "open" or pr["base"]["ref"] != "main":
        return "PR must be open and target main"
    if (
        pr["draft"]
        or pr["head"]["repo"] is None
        or (
            pr["head"]["repo"]["full_name"] != pr["base"]["repo"]["full_name"]
            or pr["base"]["repo"]["full_name"] != api.repo
        )
    ):
        return "Draft or fork PR"
    if len(ready) != 1:
        return "Apply exactly one ready label"
    if labels & {"version-bump", "skip-version-bump"} or pr["title"].startswith(
        "chore: bump version to "
    ):
        return "Version-bump policy excludes automatic merge"
    if pr["mergeable"] is not True or pr["mergeable_state"] not in {
        "clean",
        "unstable",
        "has_hooks",
    }:
        return "Mergeability not confirmed"
    if reason := catalog_review_gate(api, pr):
        return reason
    sha = pr["head"]["sha"]
    workflows = {
        run["check_suite_id"]: run["path"].split("@", 1)[0]
        for run in api.items(f"actions/runs?head_sha={sha}", "workflow_runs")
    }
    latest = {}
    for check in api.items(f"commits/{sha}/check-runs?filter=all", "check_runs"):
        name = check["name"]
        if (
            check["app"]["slug"] == "github-actions"
            and name in CHECK_WORKFLOWS
            and workflows.get(check.get("check_suite", {}).get("id"))
            == CHECK_WORKFLOWS[name]
        ):
            if name not in latest or check["id"] > latest[name]["id"]:
                latest[name] = check
    required = [
        "tests",
        "desktop-tests",
        "version-bump-guard",
        "merge-lane-no-mac" if "merge-ready" in ready else "merge-lane-mac",
    ]
    for name in required + (["type-check"] if "type-check" in latest else []):
        check = latest.get(name, {})
        allowed = {"success", "skipped"} if name == "type-check" else {"success"}
        if check.get("status") != "completed" or check.get("conclusion") not in allowed:
            return f"{name} missing or not green on head"
    other_lane = "merge-lane-mac" if "merge-ready" in ready else "merge-lane-no-mac"
    other_check = latest.get(other_lane)
    if other_check is not None and (
        other_check.get("status") != "completed"
        or other_check.get("conclusion") != "skipped"
    ):
        return "label does not match CI lane"
    statuses = api.items(f"commits/{sha}/statuses")
    binding = [
        s
        for s in statuses
        if s["context"] == "merge-ready-head"
        and s.get("creator", {}).get("login") == "github-actions[bot]"
    ]
    if not binding or max(binding, key=lambda s: s["id"])["state"] != "success":
        return "merge-ready-head missing or not successful on head"
    return ""


def live_pr(api: GitHub, number: int) -> dict:
    for attempt in range(3):
        pr = api.request(f"pulls/{number}")
        if pr["mergeable"] is not None and pr["mergeable_state"] not in {
            "unknown",
            None,
        }:
            return pr
        if attempt < 2:
            time.sleep(2)
    return pr


def merge(
    api: GitHub,
    *,
    number: int | None = None,
    dry_run: bool = False,
    manual_freeze: str = "",
) -> list[tuple[str, str, str]]:
    rows = []
    frozen = freeze(api, manual_freeze)
    if frozen:
        return [("all", "frozen", frozen)]
    candidates = (
        [{"number": number}] if number else api.items("pulls?state=open&base=main")
    )
    for candidate in sorted(candidates, key=lambda p: p["number"]):
        n = candidate["number"]
        try:
            pr = live_pr(api, n)
            if not number and not {x["name"] for x in pr["labels"]} & {
                "merge-ready",
                "merge-ready-mac",
            }:
                continue
            reason = eligibility(api, pr)
            if reason:
                rows.append((str(n), "skip", reason))
                continue
            if dry_run:
                rows.append((str(n), "eligible", "dry run; no merge"))
                continue
            frozen = freeze(api, manual_freeze)
            if frozen:
                rows.append((str(n), "frozen", frozen))
                break
            live = live_pr(api, n)
            reason = eligibility(api, live)
            if live["head"]["sha"] != pr["head"]["sha"] or reason:
                rows.append((str(n), "skip", reason or "Head moved"))
                continue
            result = api.request(
                f"pulls/{n}/merge",
                "PUT",
                {
                    "merge_method": "squash",
                    "sha": pr["head"]["sha"],
                    "commit_title": f"{live['title']} (#{n})",
                },
                pat=True,
            )
            rows.append(
                (
                    str(n),
                    "merged" if result.get("merged") is True else "skip",
                    result.get("message", "Merge API refused"),
                )
            )
            frozen = freeze(api, manual_freeze)
            if frozen:
                rows.append(("all", "frozen", frozen))
                break
        except APIError as exc:
            if exc.status not in {405, 409}:
                raise
            rows.append((str(n), "skip", str(exc)))
    return rows


def summary(rows: list[tuple[str, str, str]]) -> None:
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    content = "| PR | Decision | Reason |\n| --- | --- | --- |\n"
    content += "".join("| " + " | ".join(map(cell, row)) + " |\n" for row in rows)
    print(content)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
            output.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY"), required=False
    )
    parser.add_argument("--pr", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo is required")
    api = GitHub(
        args.repo, os.environ.get("GH_TOKEN", ""), os.environ.get("RELEASE_PAT", "")
    )
    try:
        summary(
            merge(
                api,
                number=args.pr,
                dry_run=args.dry_run,
                manual_freeze=os.environ.get("RELEASE_FREEZE", ""),
            )
        )
    except (APIError, KeyError, TypeError, ValueError) as exc:
        summary([("all", "error", str(exc))])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
