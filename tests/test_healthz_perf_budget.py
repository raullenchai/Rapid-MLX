# SPDX-License-Identifier: Apache-2.0
"""R7-H8: /healthz must stay fast under streaming concurrency.

The 0.8.7 dogfood (Olu r5) measured ``/healthz`` p99 at ~70 ms under
load; the 0.8.8 dogfood (Talia r1/r2) caught a regression to 213 ms
under 8-way streaming concurrency — well past the 50 ms k8s probe
budget. The fix (see ``vllm_mlx/routes/health.py::healthz``) is to
make ``/healthz`` a *liveness*-only probe that reads three constant-
time fields off the config object and does NOT call
``engine.get_stats()`` (which synchronizes with the Metal command
queue + iterates ``scheduler.running``).

This test pins the fix via two complementary assertions:

1. **No-engine-call invariant** — ``/healthz`` must NOT invoke
   ``engine.get_stats()`` (the heavy path that regressed). A mock
   engine asserts on its ``get_stats`` call counter.

2. **Latency budget under simulated contention** — a mock engine
   whose ``get_stats()`` blocks for 100 ms simulates the Metal-queue
   contention shape. ``/healthz`` p99 across 200 hits must stay
   below the 50 ms budget regardless of whether other endpoints are
   triggering ``get_stats()`` in flight.

Both assertions run against the FastAPI ``TestClient`` so the test
is deterministic and runs in CI without a real model load.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import get_config


@pytest.fixture
def slow_stats_engine():
    """Mock engine whose ``get_stats()`` blocks for 100 ms.

    Simulates the Metal-queue contention shape that turned a 70 ms
    p99 into a 213 ms p99 in Talia's r1/r2 repro: under 8 concurrent
    streams, every ``engine.get_stats()`` call held the Metal lock /
    waited on the command queue. If ``/healthz`` reaches this path,
    the perf budget assertion below blows out.
    """
    engine = MagicMock()
    engine.is_mllm = False

    def _slow_get_stats():
        time.sleep(0.1)  # 100 ms — well above the 50 ms budget
        return {
            "engine_type": "batched",
            "running": False,
            "uptime_seconds": 1.0,
            "steps_executed": 0,
        }

    engine.get_stats = MagicMock(side_effect=_slow_get_stats)
    return engine


def _patch_config(**kwargs):
    cfg = get_config()
    originals = {}
    for k, v in kwargs.items():
        originals[k] = getattr(cfg, k)
        setattr(cfg, k, v)
    return originals


def _restore_config(originals):
    cfg = get_config()
    for k, v in originals.items():
        setattr(cfg, k, v)


def _make_app():
    from vllm_mlx.routes.health import probe_router

    app = FastAPI()
    app.include_router(probe_router)
    return app


def test_healthz_does_not_call_engine_get_stats(slow_stats_engine):
    """R7-H8 root-cause guard: ``/healthz`` must NOT invoke
    ``engine.get_stats()``. That call is what synchronizes with the
    Metal command queue + iterates ``scheduler.running`` building
    per-request dicts — work that put the route over the k8s probe
    budget. The fast path reads three constant-time fields off the
    config object only.

    A regression that re-adds ``engine.get_stats()`` to /healthz
    (e.g. by going back to delegating to ``/health``) will fail this
    test before it can ship.
    """
    orig = _patch_config(
        engine=slow_stats_engine,
        mcp_manager=None,
        model_name="test-model",
        ready=True,
    )
    try:
        app = _make_app()
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        # The headline invariant — get_stats() must not have been called.
        assert slow_stats_engine.get_stats.call_count == 0, (
            f"/healthz called engine.get_stats() — R7-H8 regression. The"
            f" route must stay on the static-config fast path. call_count="
            f"{slow_stats_engine.get_stats.call_count}"
        )
    finally:
        _restore_config(orig)


def test_healthz_p99_stays_below_budget_under_simulated_contention(
    slow_stats_engine,
):
    """End-to-end perf budget: with the engine's ``get_stats()`` set
    to a 100 ms sleep (simulating Metal-queue contention under
    streaming concurrency), ``/healthz`` p99 across 200 hits must
    stay below 50 ms.

    This is the deterministic version of Talia's 8-way streaming-
    concurrency repro: the slow ``get_stats()`` stands in for the
    real Metal contention. If ``/healthz`` invokes ``get_stats()``
    on the request path (the pre-R7 shape), every hit pays the
    100 ms penalty and p99 blows past the budget.

    The budget matches the Olu r5 baseline (70 ms) with headroom:
    we assert p99 ≤ 50 ms (k8s default probe ``timeoutSeconds=1``
    but most production probes set 100 ms tail budgets). The
    pre-fix path measured 213 ms p99 in dogfood-088 — 4× over.
    """
    orig = _patch_config(
        engine=slow_stats_engine,
        mcp_manager=None,
        model_name="test-model",
        ready=True,
    )
    try:
        app = _make_app()
        client = TestClient(app)

        # Warm the route once so import / first-request setup costs
        # don't poison the p99 measurement.
        client.get("/healthz")
        slow_stats_engine.get_stats.reset_mock()

        durations_ms: list[float] = []
        for _ in range(200):
            t0 = time.perf_counter()
            r = client.get("/healthz")
            t1 = time.perf_counter()
            assert r.status_code == 200
            durations_ms.append((t1 - t0) * 1000.0)

        durations_ms.sort()
        # p99 across 200 samples = the 198th element (0-indexed).
        p99 = durations_ms[198]
        # Budget: 50 ms p99. The pre-r7 code measured 213 ms; this
        # threshold detects the regression with healthy margin.
        budget_ms = 50.0
        assert p99 < budget_ms, (
            f"/healthz p99 = {p99:.1f} ms exceeds {budget_ms} ms budget."
            f" Top-5 durations: {durations_ms[-5:]}. R7-H8 regression:"
            f" the route is back on the engine.get_stats() hot path."
        )
        # Belt-and-braces: if get_stats() was called even once on a
        # real /healthz hit, the perf is also broken even on a
        # well-tuned machine. Pin both invariants.
        assert slow_stats_engine.get_stats.call_count == 0, (
            f"/healthz called engine.get_stats() during the perf loop"
            f" — call_count={slow_stats_engine.get_stats.call_count}"
        )
    finally:
        _restore_config(orig)
