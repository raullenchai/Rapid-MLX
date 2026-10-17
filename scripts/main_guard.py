#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rerun main failures once, propose attributable reverts, and track nightlies."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.merge_on_green import APIError, GitHub

NIGHTLIES = ("ci.yml", "rapid-mac-ci.yml")


def ensure_label(api: GitHub, label: str) -> None:
    try:
        api.request(f"labels/{label}")
    except APIError as exc:
        if exc.status != 404:
            raise
        try:
            api.request("labels", "POST", {"name": label, "color": "b60205"})
        except APIError as create_error:
            if create_error.status != 422:
                raise


def issue(api: GitHub, label: str, title: str, body: str) -> None:
    ensure_label(api, label)
    path = f"issues?state=open&labels={label}"
    issues = [i for i in api.items(path) if "pull_request" not in i]
    body = f"<!-- {label} -->\n{body}"
    if not issues:
        api.request("issues", "POST", {"title": title, "body": body, "labels": [label]})
        # Another guard may have created the same tracking issue concurrently.
        issues = [i for i in api.items(path) if "pull_request" not in i]
    if issues:
        issues.sort(key=lambda i: i["number"])
        kept = issues[0]["number"]
        api.request(f"issues/{kept}", "PATCH", {"body": body})
        for duplicate in issues[1:]:
            number = duplicate["number"]
            api.request(
                f"issues/{number}/comments", "POST", {"body": f"Duplicate of #{kept}"}
            )
            api.request(f"issues/{number}", "PATCH", {"state": "closed"})


def runs(api: GitHub, workflow: str, event: str) -> list[dict]:
    return sorted(
        (
            r
            for r in api.items(
                f"actions/workflows/{workflow}/runs?branch=main&event={event}",
                "workflow_runs",
                max_pages=3,
            )
            if r["head_branch"] == "main" and r["event"] == event
        ),
        key=lambda r: (r["id"], r["run_attempt"]),
        reverse=True,
    )


def nightly(api: GitHub, run: dict) -> str:
    if run["conclusion"] == "failure":
        previous = next(
            (
                r
                for r in runs(api, str(run["workflow_id"]), "schedule")
                if r["id"] < run["id"] and r["conclusion"] == "success"
            ),
            None,
        )
        span = (
            f"{previous['head_sha']}..{run['head_sha']}"
            if previous
            else "no green nightly on record"
        )
        issue(
            api,
            "nightly-failure",
            "Nightly main is failing",
            f"{run['name']}: {run['html_url']}\nSHA: {run['head_sha']}\nRange: {span}",
        )
        return "Nightly failure recorded"
    if run["conclusion"] == "success":
        for workflow in NIGHTLIES:
            history = runs(api, workflow, "schedule")
            if (
                not history
                or history[0]["status"] != "completed"
                or history[0]["conclusion"] != "success"
            ):
                return "Nightly recovery pending"
        for item in api.items("issues?state=open&labels=nightly-failure"):
            if "pull_request" not in item:
                api.request(
                    f"issues/{item['number']}/comments",
                    "POST",
                    {"body": "Every nightly workflow is green again."},
                )
                api.request(f"issues/{item['number']}", "PATCH", {"state": "closed"})
        return "Nightly recovery recorded"
    return "Nightly conclusion ignored"


def attributable(api: GitHub, run: dict) -> tuple[bool, str]:
    """Find the nearest completed non-cancelled run along first-parent ancestry."""
    history = runs(api, str(run["workflow_id"]), "push")
    sha = run["head_sha"]
    for distance in range(1, 31):
        commit = api.request(f"commits/{sha}")
        if not commit["parents"]:
            break
        sha = commit["parents"][0]["sha"]
        ancestor = next(
            (
                r
                for r in history
                if r["head_sha"] == sha
                and r["status"] == "completed"
                and r["conclusion"] != "cancelled"
            ),
            None,
        )
        if ancestor:
            return (
                distance == 1 and ancestor["conclusion"] == "success",
                f"{sha}..{run['head_sha']} (previous result: {ancestor['conclusion']})",
            )
    return (
        False,
        f"No completed ancestor run within 30 first-parent commits of {run['head_sha']}",
    )


def revert_commit(api: GitHub, sha: str, branch: str) -> None:
    """Revert on freshly fetched main; never execute repository or PR code."""
    if not api.pat:
        raise APIError("RELEASE_PAT is required for revert writes")
    credential = base64.b64encode(f"x-access-token:{api.pat}".encode()).decode()
    env = dict(
        os.environ,
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {credential}",
    )
    with tempfile.TemporaryDirectory(
        prefix="main-guard-revert-", dir=os.environ.get("RUNNER_TEMP") or None
    ) as directory:
        commands = [
            [
                "clone",
                "--single-branch",
                "--branch",
                "main",
                f"https://github.com/{api.repo}.git",
                ".",
            ],
            ["config", "user.name", "Rapid-MLX main guard"],
            ["config", "user.email", "main-guard@users.noreply.github.com"],
            ["checkout", "-b", branch],
            ["revert", "--no-edit", sha],
            ["push", "origin", f"HEAD:refs/heads/{branch}"],
        ]
        for command in commands:
            result = subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", *command],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if result.returncode:
                # Do not include git output: credential helpers may echo credentials.
                raise APIError(
                    f"Revert preparation failed at git {command[0]}; manual recovery required"
                )


def notify_author(api: GitHub, original: dict, revert: dict, run: dict) -> None:
    marker = f"<!-- main-guard:{run['head_sha']} -->"
    comments = api.items(f"issues/{original['number']}/comments")
    if not any(marker in c["body"] for c in comments):
        api.request(
            f"issues/{original['number']}/comments",
            "POST",
            {
                "body": f"{marker}\n@{original['user']['login']} main CI failed: {run['html_url']}. "
                f"Proposed revert: {revert['html_url']} (NOT auto-merged)."
            },
        )


def guard(api: GitHub, run: dict) -> str:
    if run["head_branch"] != "main":
        return "Non-main run ignored"
    if run["event"] == "schedule":
        return nightly(api, run)
    if run["event"] != "push" or run["conclusion"] != "failure":
        return "Run ignored"
    jobs = api.items(
        f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs"
    )
    failed = [
        j
        for j in jobs
        if j["conclusion"] == "failure"
        and j.get("name") not in {"tests", "desktop-tests"}
    ]
    failed_steps = [
        s for j in failed for s in j.get("steps", []) if s["conclusion"] == "failure"
    ]
    if failed and all(
        s["name"] in {"Set up job", "Set up runner", "Initialize containers"}
        for s in failed_steps
    ):
        return "Infrastructure-only failure ignored"
    if run["run_attempt"] == 1:
        api.request(f"actions/runs/{run['id']}/rerun-failed-jobs", "POST")
        return "Failed jobs rerun once"
    blame, span = attributable(api, run)
    sha = run["head_sha"]
    title = api.request(f"commits/{sha}")["commit"]["message"].splitlines()[0]
    originals = [
        p
        for p in api.items(f"commits/{sha}/pulls")
        if p["merged_at"]
        and p["base"]["ref"] == "main"
        and p["merge_commit_sha"] == sha
    ]
    detail = f"{run['name']}: {run['html_url']}\nRange: {span}"
    if (
        not blame
        or title.startswith(("chore: bump version to ", 'Revert "'))
        or not originals
    ):
        issue(api, "main-red", "Main CI is failing", detail)
        return "Ambiguous or protected commit recorded"
    branch = f"revert/{sha}"
    existing = api.items(f"pulls?state=all&head={api.repo.split('/')[0]}:{branch}")
    if existing:
        notify_author(api, originals[0], existing[0], run)
        return "Revert PR already exists"
    if not api.pat:
        raise APIError("RELEASE_PAT is required for revert writes")
    try:
        api.request(f"git/ref/heads/{branch}")
    except APIError as exc:
        if exc.status != 404:
            raise
        try:
            revert_commit(api, sha, branch)
        except (APIError, OSError, subprocess.TimeoutExpired) as exc:
            issue(api, "main-red", "Main CI is failing", f"{detail}\n{exc}")
            return "Revert conflict or preparation failure recorded"
    try:
        revert = api.request(
            "pulls",
            "POST",
            {
                "head": branch,
                "base": "main",
                "title": f'Revert "{title}"',
                "body": f"Reverts #{originals[0]['number']}.\nFailing run: {run['html_url']}\n"
                "This revert PR is NOT auto-merged. Review and merge manually.",
            },
            pat=True,
        )
    except APIError as exc:
        issue(api, "main-red", "Main CI is failing", f"{detail}\n{exc}")
        return "Revert PR creation failure recorded"
    ensure_label(api, "post-merge-revert")
    api.request(
        f"issues/{revert['number']}/labels", "POST", {"labels": ["post-merge-revert"]}
    )
    notify_author(api, originals[0], revert, run)
    return "Revert PR proposed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--event-path", default=os.environ.get("GITHUB_EVENT_PATH"))
    args = parser.parse_args(argv)
    if not args.repo or not args.event_path:
        parser.error("--repo and --event-path are required")
    try:
        event = json.loads(Path(args.event_path).read_text())
        api = GitHub(
            args.repo, os.environ.get("GH_TOKEN", ""), os.environ.get("RELEASE_PAT", "")
        )
        # Re-fetch rather than trust stale workflow_run delivery or reruns of this guard.
        run = api.request(f"actions/runs/{event['workflow_run']['id']}")
        if run["status"] != "completed":
            print("Run no longer completed")
            return 0
        print(guard(api, run))
    except (APIError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Main guard failed closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
