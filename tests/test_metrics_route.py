# SPDX-License-Identifier: Apache-2.0
"""Wire-level tests for the Prometheus ``/metrics`` endpoint (issue #701).

These tests do NOT spin up the real engine — they inject a fake engine
exposing the same ``get_stats()`` shape, which is the only contract the
route depends on. That keeps the suite at unit-test speed and avoids the
2-3 GB model download that the live engine would otherwise pull.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def metrics_client():
    """FastAPI TestClient mounting only the metrics router.

    Using a per-test app instance keeps the global ``ServerConfig``
    singleton in a known shape and avoids interfering with other tests
    that share the same process.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import router

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    cfg.api_key = "test-secret"  # auth IS set, but /metrics must ignore it.

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()


def _fake_engine(stats: dict[str, Any]):
    """Build a minimal engine stand-in with a configurable get_stats()."""
    return SimpleNamespace(get_stats=lambda: stats)


# ---------------------------------------------------------------------------
# Basic protocol surface
# ---------------------------------------------------------------------------


def test_metrics_returns_200_and_text_content_type(metrics_client):
    """200 OK with Prometheus text exposition content-type."""
    resp = metrics_client.client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    # version=0.0.4 is what every Prometheus 2.x scraper expects.
    assert "version=0.0.4" in resp.headers["content-type"]


def test_metrics_engine_not_loaded_still_returns_200(metrics_client):
    """No engine → 200 with only build_info, never 500.

    Prometheus drops a scrape target after a single non-2xx; if /metrics
    500'd between restarts the dashboard would lose continuity until the
    scrape interval after the engine finished warmup.
    """
    metrics_client.cfg.engine = None
    resp = metrics_client.client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "rapid_mlx_build_info" in body
    # Engine-dependent metrics must be absent (no fake zeros that imply
    # a running engine).
    assert "rapid_mlx_requests_processed_total" not in body


def test_metrics_engine_get_stats_raises_falls_back_to_build_info(metrics_client):
    """If get_stats() raises, /metrics must still serve build_info."""

    def _explode() -> dict[str, Any]:
        raise RuntimeError("engine half-initialized")

    metrics_client.cfg.engine = SimpleNamespace(get_stats=_explode)
    resp = metrics_client.client.get("/metrics")
    assert resp.status_code == 200
    assert "rapid_mlx_build_info" in resp.text


def test_metrics_unauthenticated_even_when_api_key_set(metrics_client):
    """/metrics ignores --api-key (Prometheus scrapers cannot send one).

    The fixture sets ``cfg.api_key = "test-secret"`` to assert that the
    handler itself is on a no-auth router and would still respond even
    with no Authorization header.
    """
    assert metrics_client.cfg.api_key == "test-secret"
    resp = metrics_client.client.get("/metrics")  # no Authorization header
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Exposition contents
# ---------------------------------------------------------------------------


_FULL_STATS = {
    "num_waiting": 2,
    "num_running": 3,
    "num_requests_processed": 17,
    "total_prompt_tokens": 1234,
    "total_completion_tokens": 5678,
    "steps_executed": 99,
    "uptime_seconds": 42.5,
    "metal_active_memory_gb": 1.5,
    "metal_peak_memory_gb": 2.0,
    "metal_cache_memory_gb": 0.25,
    "prefix_cache": {
        "hits": 10,
        "misses": 4,
        "evictions": 1,
        "tokens_saved": 256,
    },
}


def test_metrics_exposes_all_expected_series(metrics_client):
    """Every metric documented in issue #701 is present in the output."""
    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    resp = metrics_client.client.get("/metrics")
    body = resp.text

    expected_names = [
        "rapid_mlx_build_info",
        "rapid_mlx_requests_processed_total",
        "rapid_mlx_prompt_tokens_total",
        "rapid_mlx_completion_tokens_total",
        "rapid_mlx_requests_running",
        "rapid_mlx_requests_waiting",
        "rapid_mlx_steps_executed_total",
        "rapid_mlx_uptime_seconds",
        "rapid_mlx_metal_active_memory_bytes",
        "rapid_mlx_metal_peak_memory_bytes",
        "rapid_mlx_metal_cache_memory_bytes",
        "rapid_mlx_prefix_cache_hits_total",
        "rapid_mlx_prefix_cache_misses_total",
        "rapid_mlx_prefix_cache_evictions_total",
        "rapid_mlx_prefix_cache_tokens_saved_total",
    ]
    for name in expected_names:
        assert f"# HELP {name}" in body, f"missing HELP for {name}"
        assert f"# TYPE {name}" in body, f"missing TYPE for {name}"


def test_metrics_values_match_get_stats(metrics_client):
    """Snapshot values render verbatim — counters not silently rescaled."""
    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    body = metrics_client.client.get("/metrics").text

    assert "rapid_mlx_requests_processed_total 17" in body
    assert "rapid_mlx_prompt_tokens_total 1234" in body
    assert "rapid_mlx_completion_tokens_total 5678" in body
    assert "rapid_mlx_requests_running 3" in body
    assert "rapid_mlx_requests_waiting 2" in body
    assert "rapid_mlx_steps_executed_total 99" in body
    assert "rapid_mlx_uptime_seconds 42.5" in body
    # GB → bytes conversion verified.
    assert "rapid_mlx_metal_active_memory_bytes 1500000000" in body
    assert "rapid_mlx_metal_peak_memory_bytes 2000000000" in body
    # Prefix-cache pass-through.
    assert "rapid_mlx_prefix_cache_hits_total 10" in body
    assert "rapid_mlx_prefix_cache_misses_total 4" in body
    assert "rapid_mlx_prefix_cache_evictions_total 1" in body
    assert "rapid_mlx_prefix_cache_tokens_saved_total 256" in body


def test_metrics_build_info_labels_carry_version_and_model(metrics_client):
    """``rapid_mlx_build_info`` labels expose version + model_name."""
    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    body = metrics_client.client.get("/metrics").text

    # Find the build_info sample line (HELP/TYPE excluded).
    sample_line = next(
        line for line in body.splitlines() if line.startswith("rapid_mlx_build_info{")
    )
    assert 'model="qwen3.5-4b"' in sample_line
    assert 'version="' in sample_line
    assert sample_line.endswith(" 1")


def test_metrics_handles_none_metal_stats_as_zero(metrics_client):
    """``None`` metal fields render as 0 rather than dropping the series.

    Operators dashboards interpret a missing series as "no data" — which
    would mask a stuck-at-zero from a configuration regression. Render
    explicit zeros instead.
    """
    stats = dict(_FULL_STATS)
    stats["metal_active_memory_gb"] = None
    stats["metal_peak_memory_gb"] = None
    stats["metal_cache_memory_gb"] = None
    metrics_client.cfg.engine = _fake_engine(stats)

    body = metrics_client.client.get("/metrics").text
    assert "rapid_mlx_metal_active_memory_bytes 0" in body
    assert "rapid_mlx_metal_peak_memory_bytes 0" in body
    assert "rapid_mlx_metal_cache_memory_bytes 0" in body


def test_metrics_prefers_memory_aware_cache_when_present(metrics_client):
    """When multiple cache stats keys appear, prefer memory_aware_cache.

    Matches scheduler.get_stats() precedence (block_aware_cache and
    memory_aware_cache shadow prefix_cache when active). The order
    matters: if a deploy switches cache implementations, the metric
    series must not flap between two parallel sources of truth.
    """
    stats = {
        "num_waiting": 0,
        "num_running": 0,
        "num_requests_processed": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "steps_executed": 0,
        "uptime_seconds": 0,
        "memory_aware_cache": {
            "hits": 11,
            "misses": 7,
            "evictions": 2,
            "tokens_saved": 88,
        },
        "prefix_cache": {
            "hits": 999,
            "misses": 999,
            "evictions": 999,
            "tokens_saved": 999,
        },
    }
    metrics_client.cfg.engine = _fake_engine(stats)
    body = metrics_client.client.get("/metrics").text

    assert "rapid_mlx_prefix_cache_hits_total 11" in body
    assert "rapid_mlx_prefix_cache_misses_total 7" in body
    assert "rapid_mlx_prefix_cache_tokens_saved_total 88" in body
    # The shadowed prefix_cache values must NOT leak through.
    assert "rapid_mlx_prefix_cache_hits_total 999" not in body


def test_metrics_omits_cache_series_when_no_cache_active(metrics_client):
    """No cache stats key → cache series are silently omitted, not zero.

    A user who runs ``--disable-prefix-cache`` should not see an always-0
    prefix-cache-hits series implying the cache is "working but ineffective".
    Absence of the series is the correct signal that the feature is off.
    """
    stats = {
        "num_waiting": 0,
        "num_running": 0,
        "num_requests_processed": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "steps_executed": 0,
        "uptime_seconds": 0,
    }
    metrics_client.cfg.engine = _fake_engine(stats)
    body = metrics_client.client.get("/metrics").text

    assert "rapid_mlx_prefix_cache_hits_total" not in body
    # The non-cache series must still be there.
    assert "rapid_mlx_requests_processed_total 0" in body


def test_metrics_escapes_quotes_in_model_label(metrics_client):
    """Label values containing ``"`` are escaped per the exposition spec.

    Without escaping a malicious or unlucky model name like ``foo"bar``
    would break the parser at the scraper. We don't realistically expect
    that name, but the escape function is the only sensitive part of
    the renderer and deserves a regression test.
    """
    metrics_client.cfg.model_name = 'foo"bar\\baz'
    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    body = metrics_client.client.get("/metrics").text
    sample = next(
        line for line in body.splitlines() if line.startswith("rapid_mlx_build_info{")
    )
    assert 'model="foo\\"bar\\\\baz"' in sample


def test_metrics_body_ends_with_newline(metrics_client):
    """Prometheus text exposition requires a trailing newline."""
    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    body = metrics_client.client.get("/metrics").text
    assert body.endswith("\n")
