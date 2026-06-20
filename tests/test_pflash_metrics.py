# SPDX-License-Identifier: Apache-2.0
"""Observability tests for PFlash-bypass counters (M-02 reframe).

Background
----------
PFlash compression replaces the prompt with a position-shifted subsequence;
the resulting ids do not share KV-positional semantics with the original
prompt, so reusing prefix-cache entries computed for the uncompressed
prefix would inject the wrong state into later requests. The scheduler is
correct to bypass both the fetch (``scheduler.py`` ~2891) and the store
(``scheduler.py`` ~3554) for compressed requests. The catch is that the
existing ``rapid_mlx_prefix_cache_*`` series sees zero traffic on tiers
where PFlash mode is forced to ``"always"`` (verified-tier aliases such as
``qwen3.5-4b-4bit``) — they look frozen at ``hits=0/misses=1`` even though
PFlash is doing meaningful work.

This test suite locks in two new counters surfaced by
``Scheduler.get_stats()`` and rendered by ``routes/metrics.py``:

* ``rapid_mlx_pflash_bypass_total`` — N+= for every request that hit the
  PFlash bypass.
* ``rapid_mlx_pflash_compressed_tokens_total`` — cumulative
  ``len(original) - len(compressed)`` across all bypassed requests.

The scheduler counters are exercised by driving the real
``Scheduler.add_request`` path through the same ``_compressing_config``
fixture used by ``test_pflash_scheduler.py``. The metrics rendering is
covered by injecting a fake ``engine.get_stats()`` into the FastAPI test
client (same pattern as ``test_metrics_route.py``) — keeps the suite at
unit-test speed and avoids the 2.5 GB Qwen3.5-4B download.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.pflash import PFlashConfig
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig

# ---------------------------------------------------------------------------
# Scheduler-side: counters accumulate on the PFlash bypass path
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Identity tokenizer — scheduler tests don't run the model.

    Mirrors the helper used by ``test_pflash_scheduler.py`` so behaviour
    stays in lockstep with the existing PFlash test suite.
    """

    eos_token_id = None

    def encode(self, prompt):
        if isinstance(prompt, str):
            return [ord(char) for char in prompt]
        return list(prompt)

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def _make_scheduler(pflash_config: PFlashConfig) -> Scheduler:
    return Scheduler(
        model=object(),
        tokenizer=_DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            pflash_config=pflash_config,
        ),
    )


def _compressing_config() -> PFlashConfig:
    """Forces compression on any prompt of length >= 1.

    Same shape as the helper in ``test_pflash_scheduler.py`` so the
    scheduler counters are exercised by the same code path that backs the
    verified-tier ``"always"`` mode in production.
    """
    return PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.25,
        min_keep_tokens=16,
        sink_tokens=4,
        tail_tokens=4,
        block_size=4,
    )


def test_scheduler_counters_start_at_zero():
    """Fresh scheduler exposes both counters at zero through ``get_stats``.

    Operators dashboards rely on a stuck-at-zero counter being present
    rather than absent — a missing series flips Prometheus panels to "no
    data" and triggers spurious alerts after a deploy.
    """
    scheduler = _make_scheduler(PFlashConfig(mode="off"))
    stats = scheduler.get_stats()
    assert stats["pflash_bypass_count"] == 0
    assert stats["pflash_compressed_tokens_dropped"] == 0


def test_scheduler_bypass_counter_increments_per_compressed_request():
    """Three PFlash-bypass requests advance ``bypass_count`` by 3.

    Sanity check that the counter increments once per request that
    actually engaged compression, not e.g. once per token or once per
    batch step.
    """
    scheduler = _make_scheduler(_compressing_config())

    for index in range(3):
        request = Request(
            f"req-bypass-{index}",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )
        scheduler.add_request(request)
        assert request.pflash_metadata is not None
        assert request.pflash_metadata["compressed"] is True

    stats = scheduler.get_stats()
    assert stats["pflash_bypass_count"] == 3


def test_scheduler_compressed_tokens_counter_matches_dropped_tokens():
    """``compressed_tokens_dropped`` equals sum of (original - kept)."""
    scheduler = _make_scheduler(_compressing_config())

    dropped_expected = 0
    for index in range(3):
        original = list(range(128))
        request = Request(
            f"req-tokens-{index}",
            original,
            SamplingParams(max_tokens=4),
        )
        scheduler.add_request(request)
        # ``model_prompt_tokens`` is the post-compression length.
        # Recon: ``logical - kept`` is the operator-facing "dropped".
        dropped_expected += len(original) - request.model_prompt_tokens

    assert dropped_expected > 0  # sanity — the harness must be compressing
    stats = scheduler.get_stats()
    assert stats["pflash_compressed_tokens_dropped"] == dropped_expected


def test_scheduler_counters_untouched_when_pflash_skips():
    """PFlash skip paths (tools, integrity, threshold) leave counters at zero.

    The bypass counter must reflect actual bypass events; a skipped
    request still goes through ``compress_request_tokens`` but
    ``metadata["compressed"]`` is False and the prefix cache fetch + store
    paths run normally — counting it would mislead capacity planning.
    """
    scheduler = _make_scheduler(_compressing_config())

    # Has tools → skipped with reason="tools".
    scheduler.add_request(
        Request(
            "req-tools",
            list(range(128)),
            SamplingParams(max_tokens=4),
            has_tools=True,
        )
    )
    # Requires prompt integrity → skipped with reason="protected_prompt".
    scheduler.add_request(
        Request(
            "req-protected",
            list(range(128)),
            SamplingParams(max_tokens=4),
            requires_prompt_integrity=True,
        )
    )
    # Threshold gate in auto mode → skipped with reason="threshold".
    scheduler_auto = _make_scheduler(
        PFlashConfig(mode="auto", threshold=10_000, keep_ratio=0.10)
    )
    scheduler_auto.add_request(
        Request(
            "req-short",
            list(range(64)),
            SamplingParams(max_tokens=4),
        )
    )

    assert scheduler.get_stats()["pflash_bypass_count"] == 0
    assert scheduler.get_stats()["pflash_compressed_tokens_dropped"] == 0
    assert scheduler_auto.get_stats()["pflash_bypass_count"] == 0


def test_scheduler_counters_zero_when_pflash_disabled():
    """``mode="off"`` never triggers ``compress_request_tokens`` at all."""
    scheduler = _make_scheduler(PFlashConfig(mode="off"))
    scheduler.add_request(
        Request("req-off", list(range(128)), SamplingParams(max_tokens=4))
    )
    stats = scheduler.get_stats()
    assert stats["pflash_bypass_count"] == 0
    assert stats["pflash_compressed_tokens_dropped"] == 0


# ---------------------------------------------------------------------------
# Route-side: counters surface in the /metrics Prometheus body
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_client():
    """FastAPI TestClient mounting only the metrics router.

    Mirrors ``test_metrics_route.metrics_client`` so the PFlash counters
    are exercised through the same render path as every other series.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3.5-4b"
    _reset_accumulator_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()


def _fake_engine(stats: dict[str, Any]):
    return SimpleNamespace(get_stats=lambda: stats)


_PFLASH_STATS = {
    "num_waiting": 0,
    "num_running": 0,
    "num_requests_processed": 3,
    "total_prompt_tokens": 15000,
    "total_completion_tokens": 12,
    "steps_executed": 6,
    "uptime_seconds": 9.0,
    # Three identical 5K-token chats, ratio 0.25 → 1250 kept, 3750 dropped each.
    "pflash_bypass_count": 3,
    "pflash_compressed_tokens_dropped": 11_250,
}


def test_metrics_route_renders_pflash_bypass_counter(metrics_client):
    """``rapid_mlx_pflash_bypass_total`` HELP/TYPE/value all present."""
    metrics_client.cfg.engine = _fake_engine(_PFLASH_STATS)
    body = metrics_client.client.get("/metrics").text

    assert "# HELP rapid_mlx_pflash_bypass_total" in body
    assert "# TYPE rapid_mlx_pflash_bypass_total counter" in body
    assert "rapid_mlx_pflash_bypass_total 3" in body


def test_metrics_route_renders_pflash_compressed_tokens_counter(metrics_client):
    """``rapid_mlx_pflash_compressed_tokens_total`` HELP/TYPE/value all present."""
    metrics_client.cfg.engine = _fake_engine(_PFLASH_STATS)
    body = metrics_client.client.get("/metrics").text

    assert "# HELP rapid_mlx_pflash_compressed_tokens_total" in body
    assert "# TYPE rapid_mlx_pflash_compressed_tokens_total counter" in body
    assert "rapid_mlx_pflash_compressed_tokens_total 11250" in body


def test_metrics_route_renders_zero_when_pflash_keys_missing(metrics_client):
    """Engines without PFlash render flat-line 0, not a missing series.

    Older non-hybrid engines won't populate the keys at all. The route
    must default to zero so dashboards stay stable rather than flipping
    panels to "no data".
    """
    stats_without_pflash = {
        "num_requests_processed": 1,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 5,
        "num_running": 0,
        "num_waiting": 0,
    }
    metrics_client.cfg.engine = _fake_engine(stats_without_pflash)
    body = metrics_client.client.get("/metrics").text

    assert "rapid_mlx_pflash_bypass_total 0" in body
    assert "rapid_mlx_pflash_compressed_tokens_total 0" in body
