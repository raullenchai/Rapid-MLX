# SPDX-License-Identifier: Apache-2.0
"""Step 4 — full unit test suite.

Skipped for low-blast PRs (docs, examples) — they can't break behavior.
For medium and high blast PRs, runs the same set we use locally:
``tests/`` minus integrations (those need a running server) and
``test_event_loop.py`` (long-running, separate gate).

Pre-existing failures on main are NOT filtered here — that's step 3's
job. This step validates "the suite as-is is still green"; if main is
broken that's a separate problem and we want to surface it loudly.
"""

from __future__ import annotations

import subprocess
import sys

from ..base import Step, StepResult
from ..context import Context


class FullUnitStep(Step):
    name = "full_unit"
    description = "full pytest suite (gated on blast radius)"

    def should_run(self, ctx: Context) -> bool:
        # Low-blast PRs get a skip — docs/examples can't break runtime.
        return ctx.blast_radius != "low"

    def run(self, ctx: Context) -> StepResult:
        log_path = ctx.artifact_path("full-unit.log")

        # Mirror what we run by hand. Two ignores: integrations needs a
        # live server (covered in step 5), and test_event_loop is the
        # long-running soak — separate budget.
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--ignore=tests/integrations",
            "--ignore=tests/test_event_loop.py",
            "-q",
            "--no-header",
            # Don't stop on first failure — we want the full count for
            # the scorecard ("3 failed, 2080 passed" is more actionable
            # than "1 failed, ???? passed").
        ]
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, cwd=str(ctx.repo_root)
        )
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""))

        # Pull the summary line: pytest ends with a line like
        # "==== 3 failed, 2080 passed, 17 skipped in 25.56s ===="
        summary_line = _last_summary_line(proc.stdout)

        if proc.returncode == 0:
            return StepResult(
                name=self.name,
                status="pass",
                summary=summary_line or "all tests passed",
                artifacts=[str(log_path)],
            )

        # On failure, extract the per-test FAILED lines for the
        # scorecard's detail block — much more useful than dumping the
        # whole log inline.
        failed = _extract_failed_lines(proc.stdout)
        details = ["**Failed tests:**\n```", *failed[:30], "```"]
        if len(failed) > 30:
            details.append(f"\n…and {len(failed) - 30} more — see {log_path}")

        return StepResult(
            name=self.name,
            status="fail",
            summary=summary_line or f"pytest exited {proc.returncode}",
            details="\n".join(details),
            artifacts=[str(log_path)],
        )


def _last_summary_line(stdout: str) -> str:
    """Pytest writes its overall summary as the very last non-empty
    line wrapped in '====' decorations. Return without decorations."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("=") and ("passed" in line or "failed" in line):
            return line.strip("= ").strip()
    return ""


def _extract_failed_lines(stdout: str) -> list[str]:
    """Return the lines pytest's short summary section labels FAILED."""
    out = []
    in_summary = False
    for line in (stdout or "").splitlines():
        if "short test summary" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("="):
                break
            if line.startswith("FAILED"):
                out.append(line)
    return out
