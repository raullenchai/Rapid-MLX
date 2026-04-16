# SPDX-License-Identifier: Apache-2.0
"""Check tier — performance metrics via autoresearch_bench.py."""

from __future__ import annotations

import json
import os
import time

from ..runner import REPO_ROOT, CheckResult, Status, python_executable, run_subprocess


def check_autoresearch(port: int, runs: int = 1) -> CheckResult:
    """Run autoresearch_bench --json and capture metrics for baseline diff.

    The script hardcodes ``http://127.0.0.1:8000`` — we override it via
    AUTORESEARCH_BASE env var (read by our patched script).  Falls back
    to a small in-process replacement if the script can't be retargeted.
    """
    t0 = time.perf_counter()
    script = REPO_ROOT / "scripts" / "autoresearch_bench.py"
    py = python_executable()

    env = os.environ.copy()
    env["AUTORESEARCH_BASE"] = f"http://127.0.0.1:{port}/v1/chat/completions"

    rc, stdout, stderr = run_subprocess(
        [py, str(script), "--json", "--runs", str(runs)],
        timeout=600,
        env=env,
    )
    elapsed = time.perf_counter() - t0

    if rc != 0:
        return CheckResult(
            name="autoresearch",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=stderr[-1000:] or stdout[-500:],
        )

    # autoresearch_bench prints the JSON dict on stdout in --json mode.
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        return CheckResult(
            name="autoresearch",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"could not parse JSON output: {e}\n{stdout[-500:]}",
        )

    # Lift numeric metrics to the top level for baseline comparison.
    metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}

    # Sanity: if every primary metric is zero/falsy the script almost
    # certainly failed silently (e.g. wrong model name → all requests
    # 404'd).  Treat that as a check failure instead of recording a
    # garbage baseline.
    primary_signals = ("decode_tps", "cold_tps", "cached_ttft_ms")
    if all(metrics.get(k, 0) == 0 for k in primary_signals):
        return CheckResult(
            name="autoresearch",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=(
                f"all primary metrics zero — server probably rejected "
                f"requests. last 200 chars of stdout: {stdout[-200:]}"
            ),
            metrics=metrics,
        )

    return CheckResult(
        name="autoresearch",
        status=Status.PASS,
        duration_s=elapsed,
        detail=f"captured {len(metrics)} metrics",
        metrics=metrics,
    )
