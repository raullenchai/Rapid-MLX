# SPDX-License-Identifier: Apache-2.0
"""Check tier — API contract + smoke matrix against a live server."""

from __future__ import annotations

import time

from ...doctor.runner import (
    REPO_ROOT,
    CheckResult,
    Status,
    python_executable,
    run_subprocess,
)


def check_smoke_matrix(port: int) -> CheckResult:
    """Bash smoke matrix: emoji/CJK/thinking toggle/special-token leaks."""
    t0 = time.perf_counter()
    script = REPO_ROOT / "tests" / "test_smoke_matrix.sh"
    rc, stdout, stderr = run_subprocess(
        ["bash", str(script), str(port)],
        timeout=180,
    )
    elapsed = time.perf_counter() - t0
    last_lines = "\n".join(stdout.strip().splitlines()[-12:])
    if rc == 0:
        return CheckResult(
            name="smoke_matrix",
            status=Status.PASS,
            duration_s=elapsed,
            detail=last_lines.split("\n")[-2] if last_lines else "",
        )
    return CheckResult(
        name="smoke_matrix",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=last_lines or stderr[-500:],
    )


def check_regression_suite(port: int) -> CheckResult:
    """API contract regression suite (10 cases: stop sequences, validation, etc.).

    Note: ``regression_suite.py`` always exits 0 — it only prints a
    "N/M tests passed" summary.  We must parse that line and fail
    whenever any case failed; otherwise regressions slip through.
    """
    import os
    import re

    t0 = time.perf_counter()
    script = REPO_ROOT / "tests" / "regression_suite.py"
    py = python_executable()
    env = os.environ.copy()
    env["RAPID_MLX_PORT"] = str(port)
    rc, stdout, stderr = run_subprocess(
        [py, str(script)],
        timeout=300,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    last_lines = "\n".join(stdout.strip().splitlines()[-6:])

    # Parse "N/M tests passed" from anywhere in the output.
    summary_re = re.compile(r"(\d+)\s*/\s*(\d+)\s+tests?\s+passed", re.IGNORECASE)
    passed = total = None
    for line in stdout.splitlines():
        m = summary_re.search(line)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))

    # Subprocess crash trumps the summary parse.
    if rc != 0:
        return CheckResult(
            name="regression_suite",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"script exited rc={rc}\n{last_lines or stderr[-500:]}",
        )

    if passed is None or total is None:
        # Couldn't find the summary — be conservative, treat as failure.
        return CheckResult(
            name="regression_suite",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"could not parse 'N/M tests passed' from output:\n{last_lines}",
        )

    if passed < total:
        return CheckResult(
            name="regression_suite",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"{passed}/{total} passed — see {total - passed} failure(s) above:\n{last_lines}",
        )

    return CheckResult(
        name="regression_suite",
        status=Status.PASS,
        duration_s=elapsed,
        detail=f"{passed}/{total} tests passed",
    )
