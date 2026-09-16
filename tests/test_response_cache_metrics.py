# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests: the response-cache hit/miss counters on ``/metrics``.

Mirrors ``tests/test_metrics_route.py`` — no real engine, just the
metrics router mounted on a throwaway FastAPI app. The counters live in
the ``rapid_mlx.response_cache`` module singleton (NOT the engine), so
they must render even when the engine is absent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def metrics_client():
    from rapid_mlx.config import reset_config
    from rapid_mlx.response_cache import reset_response_cache_for_tests
    from rapid_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    _reset_accumulator_for_tests()
    reset_response_cache_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()
    reset_response_cache_for_tests()


def _metric_lines(body: str) -> set[str]:
    """Split the Prometheus payload into whole sample lines so a value can
    be matched EXACTLY. Substring-matching ``... 2`` also accepts ``20`` /
    ``200``; whole-line membership does not."""
    return set(body.splitlines())


def test_response_cache_counters_present_even_without_engine(metrics_client):
    """Both series render (at zero) even when the engine is not loaded —
    the counters are engine-independent module state."""
    metrics_client.cfg.engine = None
    body = metrics_client.client.get("/metrics").text
    lines = _metric_lines(body)
    assert "rapid_mlx_response_cache_hits_total" in body
    assert "rapid_mlx_response_cache_misses_total" in body
    # Disabled cache → both EXACTLY zero (whole-line match, so a future
    # "...total 0.x" or "...total 10" cannot slip past).
    assert "rapid_mlx_response_cache_hits_total 0" in lines
    assert "rapid_mlx_response_cache_misses_total 0" in lines


def test_response_cache_counters_reflect_singleton_state(metrics_client):
    """Driving the singleton's counters must surface on the scrape."""
    from rapid_mlx.response_cache import get_response_cache

    c = get_response_cache()
    c.configure(4)
    ep = c.current_epoch()
    c.get("miss-a", ep)  # miss
    c.put("k", object(), ep)
    c.get("k", ep)  # hit
    c.get("k", ep)  # hit
    c.get("miss-b", ep)  # miss

    body = metrics_client.client.get("/metrics").text
    lines = _metric_lines(body)
    # EXACT whole-line match: substring "... 2" would also (wrongly) pass
    # for a rendered value of 20 / 200. Whole-line membership rejects those.
    assert "rapid_mlx_response_cache_hits_total 2" in lines
    assert "rapid_mlx_response_cache_misses_total 2" in lines
    # And prove the loose-substring trap would have been permissive here:
    # the exact-2 lines are present, the 20/200 lines are NOT.
    assert "rapid_mlx_response_cache_hits_total 20" not in lines
    assert "rapid_mlx_response_cache_misses_total 20" not in lines


def test_response_cache_counters_have_help_and_type_lines(metrics_client):
    """Prometheus HELP + TYPE metadata must accompany each series."""
    body = metrics_client.client.get("/metrics").text
    assert "# TYPE rapid_mlx_response_cache_hits_total counter" in body
    assert "# TYPE rapid_mlx_response_cache_misses_total counter" in body
    assert "# HELP rapid_mlx_response_cache_hits_total" in body
    assert "# HELP rapid_mlx_response_cache_misses_total" in body
