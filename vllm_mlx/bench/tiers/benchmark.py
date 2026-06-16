# SPDX-License-Identifier: Apache-2.0
"""Benchmark tier — capture a small set of metrics per (model, engine) cell."""

from __future__ import annotations

import json
import os
import time

from ...doctor.runner import (
    REPO_ROOT,
    CheckResult,
    Status,
    python_executable,
    run_subprocess,
)


def benchmark_one_cell(
    model: str,
    port: int,
    runs: int = 1,
) -> CheckResult:
    """Run autoresearch_bench against a live server, return key metrics.

    Captures the four metrics that fit in a scorecard cell:
      - decode_tps      throughput (tok/s)
      - cold_ttft_ms    time to first token, no cache
      - cached_ttft_ms  time to first token, cache warm
      - tc_success_rate tool calling reliability (0.0-1.0)

    The result.metrics dict is the source of truth for the scorecard
    renderer; this function does not format anything itself.
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
            name=f"bench[{model}]",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"autoresearch failed rc={rc}\n{(stderr or stdout)[-400:]}",
        )

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        return CheckResult(
            name=f"bench[{model}]",
            status=Status.FAIL,
            duration_s=elapsed,
            detail=f"could not parse JSON: {e}\n{stdout[-400:]}",
        )

    # Filter to numeric metrics only.
    metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}

    # Same all-zero-primary check as the check tier perf.py — catches
    # the silent-failure mode where every request was rejected.
    primary_signals = ("decode_tps", "cold_tps", "cached_ttft_ms")
    if all(metrics.get(k, 0) == 0 for k in primary_signals):
        return CheckResult(
            name=f"bench[{model}]",
            status=Status.FAIL,
            duration_s=elapsed,
            detail="all primary metrics zero — server probably rejected requests",
            metrics=metrics,
        )

    return CheckResult(
        name=f"bench[{model}]",
        status=Status.PASS,
        duration_s=elapsed,
        detail=(
            f"decode={metrics.get('decode_tps', 0):.1f} tok/s · "
            f"cold_ttft={metrics.get('cold_ttft_ms', 0):.0f}ms · "
            f"tc={metrics.get('tc_success_rate', 0):.0%}"
        ),
        metrics=metrics,
    )
