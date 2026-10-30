# SPDX-License-Identifier: Apache-2.0
"""Step 4 — full unit test suite, bounded by a process-group timeout.

Skipped for low-blast PRs (docs, examples) — they can't break behavior.
For medium and high blast PRs, runs the same set we use locally:
``tests/`` minus integrations (those need a running server) and
``test_event_loop.py`` (long-running, separate gate).

Pre-existing failures on main are NOT filtered here — that's step 3's
job. This step validates "the suite as-is is still green"; if main is
broken that's a separate problem and we want to surface it loudly.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from .._xdist import pytest_parallel_args
from ..base import Step, StepResult
from ..context import Context
from .diff_coverage import _run_group_bounded, _timeout_dump

_FULL_UNIT_TIMEOUT_S = 3600
_FULL_UNIT_SERIAL_TIMEOUT_S = 4 * 3600


class FullUnitStep(Step):
    name = "full_unit"
    description = "full pytest suite (gated on blast radius)"

    def should_run(self, ctx: Context) -> bool:
        # Low-blast PRs get a skip — docs/examples can't break runtime.
        return ctx.blast_radius != "low"

    def run(self, ctx: Context) -> StepResult:
        log_path = ctx.artifact_path("full-unit.log")

        parallel_args, note = pytest_parallel_args()
        ctx.run_log(note)
        raw_timeout = os.environ.get("PR_VALIDATE_FULL_UNIT_TIMEOUT_S")
        if raw_timeout is None:
            timeout = (
                _FULL_UNIT_TIMEOUT_S if parallel_args else _FULL_UNIT_SERIAL_TIMEOUT_S
            )
        else:
            try:
                timeout = int(raw_timeout)
                if timeout <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    "PR_VALIDATE_FULL_UNIT_TIMEOUT_S must be a positive integer"
                ) from None

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
            "-rfE",
            "-o",
            "faulthandler_timeout=600",
            *parallel_args,
            # Don't stop on first failure — we want the full count for
            # the scorecard ("3 failed, 2080 passed" is more actionable
            # than "1 failed, ???? passed").
        ]
        try:
            proc = _run_group_bounded(cmd, cwd=str(ctx.repo_root), timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(
                note + "\n" + _timeout_dump("full_unit pytest", cmd, exc)
            )
            return StepResult(
                name=self.name,
                status="fail",
                summary=f"full suite exceeded {timeout}s — process group killed (stuck test or leaked pipe holder; see faulthandler dump in log); {note}",
                artifacts=[str(log_path)],
            )
        log_path.write_text(note + "\n" + (proc.stdout or "") + (proc.stderr or ""))

        # Pull the summary line: pytest ends with a line like
        # "==== 3 failed, 2080 passed, 17 skipped in 25.56s ===="
        summary_line = _last_summary_line(proc.stdout)

        if proc.returncode == 0:
            return StepResult(
                name=self.name,
                status="pass",
                summary=f"{summary_line or 'all tests passed'}; {note}",
                details=note,
                artifacts=[str(log_path)],
            )

        # On failure, extract the per-test FAILED and ERROR lines for the
        # scorecard's detail block — much more useful than dumping the
        # whole log inline.
        failed = _extract_failed_lines(proc.stdout)
        details = [note, "**Failed tests:**\n```", *failed[:30], "```"]
        if len(failed) > 30:
            details.append(f"\n…and {len(failed) - 30} more — see {log_path}")

        return StepResult(
            name=self.name,
            status="fail",
            summary=f"{summary_line or f'pytest exited {proc.returncode}'}; {note}",
            details="\n".join(details),
            artifacts=[str(log_path)],
        )


def _last_summary_line(stdout: str) -> str:
    """Return the final outcome summary, with or without pytest decorations."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if re.search(
            r"\b\d+ (?:passed|failed|skipped|xfailed|xpassed|errors?)\b", line
        ) and re.search(r"\bin \d", line):
            return line.strip("= ").strip()
    return ""


def _extract_failed_lines(stdout: str) -> list[str]:
    """Return the lines pytest's short summary section labels FAILED or ERROR."""
    out = []
    in_summary = False
    for line in (stdout or "").splitlines():
        if "short test summary" in line:
            in_summary = True
            continue
        if in_summary:
            if line.startswith("="):
                break
            if line.startswith(("FAILED", "ERROR")):
                out.append(line)
    return out
