# SPDX-License-Identifier: Apache-2.0
"""Wait for GitHub Actions checks on the fetched PR head."""

import os
import re
import time

from ..base import Step, StepResult
from ..context import Context, env_truthy
from ..queue_ready import pages


def _fetch_checks(ctx: Context) -> list[dict]:
    return pages(
        f"repos/{ctx.repo}/commits/{ctx.head_sha}/check-runs?per_page=100&filter=latest",
        "check_runs",
    )


_sleep = time.sleep
_FAILURES = {
    "failure",
    "cancelled",
    "timed_out",
    "action_required",
    "startup_failure",
    "stale",
}


class HostedCIStep(Step):
    name = "hosted_ci"
    description = "GitHub CI green on the exact head (full unit suite, Apple Silicon tests, type-check, changed-lines coverage)"

    def should_run(self, ctx: Context) -> bool:
        return os.environ.get("GITHUB_ACTIONS") != "true" and not env_truthy(
            "PR_VALIDATE_NO_HOSTED_CI"
        )

    def run(self, ctx: Context) -> StepResult:
        head = ctx.head_sha or ""
        if not re.fullmatch(r"[0-9a-fA-F]{40}", head):
            return StepResult(self.name, "error", "invalid or missing PR head SHA")
        interval = float(os.environ.get("PR_VALIDATE_HOSTED_CI_POLL_S", "30"))
        timeout = float(os.environ.get("PR_VALIDATE_HOSTED_CI_TIMEOUT_S", "5400"))
        if interval <= 0 or timeout < 0:
            raise ValueError(
                "hosted CI poll interval must be positive and timeout non-negative"
            )
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                return StepResult(
                    self.name, "skip", "cancelled while waiting for hosted CI"
                )
            latest = {}
            for check in _fetch_checks(ctx):
                if (check.get("app") or {}).get("slug") != "github-actions":
                    continue
                key = ((check.get("check_suite") or {}).get("id"), check["name"])
                if key not in latest or check["id"] > latest[key]["id"]:
                    latest[key] = check
            checks = [
                c
                for c in latest.values()
                if c.get("conclusion") not in {"skipped", "neutral"}
            ]
            failures = [
                c
                for c in checks
                if c.get("status") == "completed" and c.get("conclusion") in _FAILURES
            ]
            if failures:
                return StepResult(
                    self.name,
                    "fail",
                    "hosted CI failed",
                    details="\n".join(
                        f"- {c['name']}: {c.get('html_url', '')}" for c in failures
                    ),
                )
            pending = {
                c["name"]
                for c in checks
                if c.get("status") != "completed" or c.get("conclusion") != "success"
            }
            if not any(
                c["name"] == "tests"
                and c.get("status") == "completed"
                and c.get("conclusion") == "success"
                for c in checks
            ):
                pending.add("tests")
            if checks and not pending:
                return StepResult(
                    self.name,
                    "pass",
                    f"{len(checks)} hosted checks green on {head[:12]}",
                )
            if pending != previous:
                ctx.run_log(f"Waiting for hosted CI: {', '.join(sorted(pending))}")
                previous = pending
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return StepResult(
                    self.name,
                    "fail",
                    "timed out waiting for hosted CI",
                    details="Pending checks: " + ", ".join(sorted(pending)),
                )
            seconds = min(interval, remaining)
            if ctx.cancel_event is None:
                _sleep(seconds)
            elif ctx.cancel_event.wait(seconds):
                return StepResult(
                    self.name, "skip", "cancelled while waiting for hosted CI"
                )
