# SPDX-License-Identifier: Apache-2.0
"""Opt-in handoff from completed local review to the existing managed queue.

This is a single operator helper, not a second queue controller. GitHub owns
checks and exact-head authorization; Mergify owns integration and merging.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any

from .context import Context

READY = {"merge-ready", "merge-ready-mac"}
REQUIRED = {"tests", "desktop-tests", "validate", "version-bump-guard"}


def gh(*args: str) -> Any:
    result = subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True, timeout=60
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def pages(endpoint: str, key: str | None = None) -> list[dict[str, Any]]:
    data = gh("api", endpoint, "--paginate", "--slurp")
    return [item for page in data for item in (page[key] if key else page)]


def check_pull(ctx: Context) -> dict[str, Any]:
    pull = gh("api", f"repos/{ctx.repo}/pulls/{ctx.pr_number}")
    labels = {label["name"] for label in pull["labels"]}
    if (
        pull["state"] != "open"
        or pull["draft"]
        or pull["head"]["sha"] != ctx.head_sha
        or pull["head"]["repo"]["full_name"] != ctx.repo
        or pull["base"]["ref"] != "main"
        or pull["head"]["ref"].startswith("mergify/")
        or labels & {"version-bump", "skip-version-bump", "dequeued"}
        or pull["title"].startswith("chore: bump version to ")
        or pull["title"] != ctx.pr_title
        or (pull.get("body") or "") != ctx.pr_body
    ):
        raise ValueError("PR changed or requires a separate recovery/release path")
    return pull


def existing_request(ctx: Context) -> bool:
    # Fully paginated. A previous request is never silently treated as license
    # for a new cycle, even if it belongs to an old head or was later cancelled.
    comments = pages(f"repos/{ctx.repo}/issues/{ctx.pr_number}/comments?per_page=100")
    for comment in comments:
        if comment.get("user", {}).get("login") != "mergify[bot]":
            continue
        # An automatically admitted old cycle may have no owner command.
        # Any provider receipt with an admission timestamp counts as history.
        match = re.search(
            r"-\*- Mergify Payload -\*-\s*(\{.*?\})\s*-\*- Mergify Payload End -\*-",
            comment.get("body") or "",
            re.DOTALL,
        )
        if match:
            payload = json.loads(match.group(1))
            if payload.get("queued_at"):
                return True
    return any(
        re.search(r"(?m)^\s*@mergifyio\s+queue(?:\s|$)", comment.get("body") or "")
        for comment in comments
    )


def hosted_lane(ctx: Context) -> str | None:
    checks = pages(
        f"repos/{ctx.repo}/commits/{ctx.head_sha}/check-runs?per_page=100&filter=latest",
        "check_runs",
    )
    latest: dict[str, dict[str, Any]] = {}
    for check in sorted(checks, key=lambda item: item["id"]):
        if check.get("app", {}).get("slug") == "github-actions":
            latest[check["name"]] = check
    lanes = [
        name
        for name in ("merge-lane-mac", "merge-lane-no-mac")
        if latest.get(name, {}).get("conclusion") == "success"
    ]
    if len(lanes) > 1:
        raise ValueError("ambiguous merge lane")
    names = REQUIRED | {"changed-lines-coverage"}
    for name in names:
        check = latest.get(name, {})
        if check.get("status") != "completed":
            return None
        # Source preflight intentionally defers patch coverage to the full
        # candidate. Its tests aggregate must still succeed; no other required
        # check can use a skipped conclusion as success.
        allowed = (
            {"success", "skipped"} if name == "changed-lines-coverage" else {"success"}
        )
        if check.get("conclusion") not in allowed:
            raise ValueError(f"hosted gate {name} did not succeed")
    if not lanes:
        return None
    lane = lanes[0]
    if latest[lane].get("status") != "completed":
        return None
    return "mac-batch" if lane == "merge-lane-mac" else "no-mac-batch"


def authorized(ctx: Context) -> bool:
    statuses = pages(f"repos/{ctx.repo}/commits/{ctx.head_sha}/statuses?per_page=100")
    matching = [s for s in statuses if s["context"] == "merge-ready-head"]
    if not matching:
        return False
    latest = max(matching, key=lambda item: item["id"])
    if latest["state"] in {"failure", "error"}:
        raise ValueError("exact-head readiness authorization failed")
    return latest["state"] == "success"


def queue_validated_head(
    ctx: Context, expected: list[str], *, wait_seconds: int = 1800
) -> None:
    results = {result.name: result.status for result in ctx.results}
    if (
        len(results) != len(ctx.results)
        or set(results) != set(expected)
        or any(status not in {"pass", "skip"} for status in results.values())
        or results.get("fetch") != "pass"
        or results.get("codex_review") != "pass"
        or not re.fullmatch(r"[0-9a-f]{40}", ctx.head_sha)
        or ctx.base_strategy == "tip-fallback"
    ):
        raise ValueError(
            "queue handoff requires complete validation and executed review"
        )
    pull = check_pull(ctx)
    if "queued" in {item["name"] for item in pull["labels"]} or existing_request(ctx):
        ctx.run_log("Already queued or previously requested; no new command issued")
        return
    actor = gh("api", "user")["login"]
    permission = gh("api", f"repos/{ctx.repo}/collaborators/{actor}/permission")
    if permission.get("permission") not in {"admin", "maintain", "write"}:
        raise ValueError("queue handoff requires a repository writer")

    # All worktrees on this operator's clone share one durable receipt. Write
    # exclusively before any mutation and retain it on error or transport
    # ambiguity. Across hosts there must still be one designated PR operator.
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ctx.repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    directory = (ctx.repo_root / common).resolve() / "rapid-queue-receipts"
    directory.mkdir(exist_ok=True)
    receipt = directory / f"{ctx.pr_number}-{ctx.head_sha}.json"
    record = {
        "pr": ctx.pr_number,
        "head": ctx.head_sha,
        "actor": actor,
        "state": "reserved",
    }
    with receipt.open("x") as stream:
        json.dump(record, stream)
    ctx.run_log(f"Queue issuance receipt: {receipt}")
    deadline = time.monotonic() + wait_seconds
    lane = None
    while time.monotonic() < deadline:
        pull = check_pull(ctx)
        if "queued" in {item["name"] for item in pull["labels"]} or existing_request(
            ctx
        ):
            ctx.run_log("Another admission request appeared; no new command issued")
            return
        lane = hosted_lane(ctx)
        if lane:
            break
        time.sleep(15)
    if not lane:
        raise RuntimeError(
            "hosted gate wait expired; receipt retained, no automatic retry"
        )
    label = "merge-ready-mac" if lane == "mac-batch" else "merge-ready"
    present = {item["name"] for item in pull["labels"]} & READY
    if present and present != {label}:
        raise ValueError("conflicting readiness label; do not reset automatically")
    if not present:
        check_pull(ctx)
        gh(
            "api",
            f"repos/{ctx.repo}/issues/{ctx.pr_number}/labels",
            "-f",
            f"labels[]={label}",
        )
    while time.monotonic() < deadline:
        pull = check_pull(ctx)
        if {item["name"] for item in pull["labels"]} & READY != {label}:
            raise ValueError("readiness label changed")
        if authorized(ctx):
            break
        time.sleep(15)
    else:
        raise RuntimeError("readiness wait expired; receipt retained")
    # Recheck source checks and head after the asynchronous authorization job.
    if hosted_lane(ctx) != lane:
        raise ValueError("hosted gates changed before queue request")
    pull = check_pull(ctx)
    if {item["name"] for item in pull["labels"]} & READY != {label} or not authorized(
        ctx
    ):
        raise ValueError("readiness changed before queue request")
    if "queued" in {item["name"] for item in pull["labels"]} or existing_request(ctx):
        return
    body = f"@mergifyio queue {lane}"
    record.update(state="issuing", command=body)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    # Never retry this mutation. Even a network error may mean GitHub accepted
    # the comment; the preserved receipt and live history must be reconciled.
    response = gh(
        "api", f"repos/{ctx.repo}/issues/{ctx.pr_number}/comments", "-f", f"body={body}"
    )
    record.update(state="issued", comment_id=response["id"], url=response["html_url"])
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    ctx.run_log(
        f"Queue requested once: {response['html_url']} (not proof of admission or merge)"
    )
