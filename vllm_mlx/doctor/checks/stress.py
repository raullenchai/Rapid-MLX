# SPDX-License-Identifier: Apache-2.0
"""Stress check — sustained load, concurrency, disconnect resilience."""

from __future__ import annotations

import re
import time

from ..runner import REPO_ROOT, CheckResult, Status, python_executable, run_subprocess


def check_stress(port: int) -> CheckResult:
    """Run scripts/stress_test.py against a live server.

    Parses the "N/8 passed" summary line to determine pass/fail.
    """

    t0 = time.perf_counter()
    script = REPO_ROOT / "scripts" / "stress_test.py"
    py = python_executable()
    rc, stdout, stderr = run_subprocess(
        [py, str(script), "--port", str(port)],
        timeout=600,
    )
    elapsed = time.perf_counter() - t0

    last_lines = "\n".join(stdout.strip().splitlines()[-15:])

    # Parse "N/M passed" from output
    summary_re = re.compile(r"(\d+)\s*/\s*(\d+)\s+passed", re.IGNORECASE)
    passed = total = None
    for line in stdout.splitlines():
        m = summary_re.search(line)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))

    if rc != 0 and passed is None:
        return CheckResult(
            name="stress",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"script exited rc={rc}\n{last_lines or stderr[-500:]}",
        )

    if passed is not None and total is not None:
        if passed >= total:
            return CheckResult(
                name="stress",
                status=Status.PASS,
                duration_s=elapsed,
                detail=f"{passed}/{total} stress tests passed",
            )
        else:
            return CheckResult(
                name="stress",
                status=Status.FAIL,
                duration_s=elapsed,
                detail=f"{passed}/{total} passed\n{last_lines}",
            )

    return CheckResult(
        name="stress",
        status=Status.FAIL,
        duration_s=elapsed,
        detail=f"could not parse results\n{last_lines}",
    )
