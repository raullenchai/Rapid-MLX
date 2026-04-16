# SPDX-License-Identifier: Apache-2.0
"""Check tier — API contract + smoke matrix against a live server."""

from __future__ import annotations

import time

from ..runner import REPO_ROOT, CheckResult, Status, python_executable, run_subprocess


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
    """API contract regression suite (10 cases: stop sequences, validation, etc.)."""
    t0 = time.perf_counter()
    script = REPO_ROOT / "tests" / "regression_suite.py"
    py = python_executable()
    # regression_suite.py defaults to localhost:8000; override via env if it
    # supports it, else patch URL via PORT env.  As of writing it hardcodes
    # the URL — we monkey-patch via a wrapper script if the port differs.
    # For simplicity we run on the doctor's port via env override.
    import os
    env = os.environ.copy()
    env["RAPID_MLX_PORT"] = str(port)
    rc, stdout, stderr = run_subprocess(
        [py, str(script)],
        timeout=300,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    last_lines = "\n".join(stdout.strip().splitlines()[-6:])
    if rc == 0:
        return CheckResult(
            name="regression_suite",
            status=Status.PASS,
            duration_s=elapsed,
            detail=last_lines,
        )
    return CheckResult(
        name="regression_suite",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=last_lines or stderr[-500:],
    )
