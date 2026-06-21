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
    that share the same process. Also resets the module-level sticky
    counter accumulator so each test sees a fresh ``(last_raw=0, baseline=0)``
    starting state.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    cfg.api_key = "test-secret"  # auth IS set, but /metrics must ignore it.
    _reset_accumulator_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()


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


# ---------------------------------------------------------------------------
# Counter monotonicity (sticky-counter accumulator — codex r1 MEDIUM)
# ---------------------------------------------------------------------------


def test_cache_counters_monotonic_across_cache_clear(metrics_client):
    """Prefix-cache ``_total`` counters never decrease across a cache clear.

    Prometheus contract: counters MUST be monotonically non-decreasing for
    ``rate()`` to work. The raw cache stats reset to zero on
    ``cache.clear()`` (admin ``POST /cache/clear`` or recovery paths) —
    the accumulator must fold the reset into a baseline so the exposed
    counter resumes from the previous total rather than dropping to 0.

    Sequence simulated:
      1) cache reports hits=10        → exposed should be 10
      2) cache reports hits=15        → exposed should be 15
      3) cache CLEARED, reports hits=0 → exposed should remain ≥ 15
      4) cache reports hits=3         → exposed should be 18 (15 + 3)
    """
    stats: dict[str, Any] = {
        "num_waiting": 0,
        "num_running": 0,
        "num_requests_processed": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "steps_executed": 0,
        "uptime_seconds": 0,
        "prefix_cache": {
            "hits": 10,
            "misses": 5,
            "evictions": 2,
            "tokens_saved": 100,
        },
    }
    metrics_client.cfg.engine = _fake_engine(stats)

    def _hits_value(body: str) -> int:
        for line in body.splitlines():
            if line.startswith(
                "rapid_mlx_prefix_cache_hits_total "
            ) and not line.startswith("#"):
                return int(line.rsplit(" ", 1)[1])
        raise AssertionError("hits_total line not found")

    # Step 1: hits=10 → exposed 10
    body = metrics_client.client.get("/metrics").text
    assert _hits_value(body) == 10

    # Step 2: hits=15 → exposed 15
    stats["prefix_cache"]["hits"] = 15
    body = metrics_client.client.get("/metrics").text
    assert _hits_value(body) == 15

    # Step 3: simulate cache.clear() — raw drops to 0; exposed must stay ≥ 15
    stats["prefix_cache"]["hits"] = 0
    body = metrics_client.client.get("/metrics").text
    after_clear = _hits_value(body)
    assert after_clear >= 15, (
        f"counter regressed: was 15, now {after_clear} (cache.clear must not "
        f"decrement the Prometheus counter)"
    )

    # Step 4: hits=3 — fresh activity after clear; exposed should be 15+3=18
    stats["prefix_cache"]["hits"] = 3
    body = metrics_client.client.get("/metrics").text
    assert _hits_value(body) == 18, (
        "after a reset the accumulator should fold the previous total "
        "(15) into the baseline and add new raw (3) on top → 18"
    )


def test_sticky_accumulator_unit_behavior():
    """Direct unit test of the accumulator — covers branches the route test misses.

    The integration test above only exercises one key (hits). Verify the
    accumulator handles a multi-key state (each key independent) and a
    no-op same-value advance correctly.
    """
    from vllm_mlx.routes.metrics import _StickyCounterAccumulator

    acc = _StickyCounterAccumulator()

    # First advance establishes baseline=0, last_raw=raw.
    assert acc.advance("k1", 5) == 5
    assert acc.advance("k2", 100) == 100

    # Same value twice in a row — no-op.
    assert acc.advance("k1", 5) == 5

    # Monotonic increase.
    assert acc.advance("k1", 8) == 8

    # Reset — raw drops; exposed must NOT drop.
    assert acc.advance("k1", 0) == 8  # 8 (baseline) + 0 (raw)
    assert acc.advance("k1", 2) == 10  # 8 (baseline) + 2 (raw)

    # The other key is untouched by k1's reset.
    assert acc.advance("k2", 105) == 105

    # Defensive: negative raw is floored to 0.
    assert acc.advance("k1", -1) == 10  # 10 (baseline so far) + 0
    # After previous step state is (last_raw=0, baseline=10); next reset
    # check needs raw < 0 (impossible after flooring) so baseline stays.
    assert acc.advance("k1", 4) == 14


# ---------------------------------------------------------------------------
# Format-compliance smoke test (codex r1 LOW: tests should parse the body)
# ---------------------------------------------------------------------------


def test_metrics_output_parses_cleanly_via_prometheus_client(metrics_client):
    """The emitted body parses without errors via the official Prometheus parser.

    Catches whole classes of regression that the hand-picked substring
    assertions don't: malformed escapes, locale-dependent float formatting,
    bad HELP/TYPE ordering, etc. The dependency is dev-only (see
    ``pyproject.toml`` ``dev`` extras); the runtime route still hand-rolls
    the format.
    """
    from prometheus_client.parser import text_string_to_metric_families

    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    body = metrics_client.client.get("/metrics").text

    families = list(text_string_to_metric_families(body))
    assert len(families) > 0, "parser found zero metric families"

    # Every emitted family should have at least one sample.
    for fam in families:
        assert len(fam.samples) > 0, f"family {fam.name} has no samples"

    # Verify a few specific families round-trip the values we set.
    by_name = {fam.name: fam for fam in families}
    assert "rapid_mlx_requests_processed" in by_name  # parser strips _total
    assert by_name["rapid_mlx_requests_processed"].samples[0].value == 17

    # build_info gauge round-trips labels.
    assert "rapid_mlx_build_info" in by_name
    build_sample = by_name["rapid_mlx_build_info"].samples[0]
    assert build_sample.labels["model"] == "qwen3.5-4b"
    assert build_sample.value == 1.0


def test_metrics_output_parses_cleanly_with_escaped_label(metrics_client):
    """Output with escaped quote/backslash in a label still parses cleanly.

    Regression guard for ``_escape_label_value`` — a broken escape would
    let the body pass substring assertions but choke the real parser.
    """
    from prometheus_client.parser import text_string_to_metric_families

    metrics_client.cfg.model_name = 'foo"bar\\baz'
    metrics_client.cfg.engine = _fake_engine(_FULL_STATS)
    body = metrics_client.client.get("/metrics").text

    families = list(text_string_to_metric_families(body))
    by_name = {fam.name: fam for fam in families}
    assert "rapid_mlx_build_info" in by_name
    # The parser should reverse the escapes to recover the original string.
    assert by_name["rapid_mlx_build_info"].samples[0].labels["model"] == 'foo"bar\\baz'


def test_metrics_output_parses_cleanly_when_engine_missing(metrics_client):
    """Even the engine-missing fallback path parses cleanly.

    The fallback emits build_info plus the H-06 response_format
    counters (which live in process-local module state, not on the
    engine) so dashboards never flip to "no data" between restarts.
    Strict-mode counters always present even at zero — same pattern as
    PFlash counters in the loaded-engine path.
    """
    from prometheus_client.parser import text_string_to_metric_families

    metrics_client.cfg.engine = None
    body = metrics_client.client.get("/metrics").text

    families = list(text_string_to_metric_families(body))
    by_name = {fam.name: fam for fam in families}
    # build_info + H-06 response_format counters — three families.
    assert "rapid_mlx_build_info" in by_name
    assert "rapid_mlx_response_format_strict" in by_name
    assert "rapid_mlx_response_format_strict_violations" in by_name
