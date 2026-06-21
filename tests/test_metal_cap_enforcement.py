# SPDX-License-Identifier: Apache-2.0
"""D-METAL-CAP regression tests for admission-time Metal cap enforcement.

Background
----------
``mx.set_memory_limit`` is documented as a *guideline* — MLX will silently
grow the Metal active working set PAST the limit while system RAM remains
available. On a 256 GB M3 Ultra with ``--gpu-memory-utilization 0.45``
(soft cap ≈ 115 GB) the bug repro grew Metal active to 179 GB on a single
32k-prefill request, with no warning, no counter, and no backpressure.

These tests pin the admission-time enforcement that closes that gap:
- ``Scheduler.add_request`` consults a real Metal-active probe and
  raises ``BackpressureError`` when active ≥ cap.
- ``num_metal_cap_violations`` (surfaced as
  ``rapid_mlx_metal_cap_violations_total`` in /metrics) increments
  once per rejected admission.
- The first violation logs a single WARNING; subsequent violations rely
  on the Prometheus counter to keep logs readable on a sustained storm.

Implementation note: we use a stub for ``mx.get_active_memory`` so the
test is deterministic on any host (CI Linux + Apple Silicon dev boxes
alike) and doesn't depend on the actual GPU pressure at test time.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import BackpressureError, Scheduler, SchedulerConfig


def _make_request(rid: str = "req-1", tokens: int = 16) -> Request:
    """Tiny synthetic request — only ``request_id`` and
    ``prompt_token_ids`` matter for admission control."""
    req = Request(
        request_id=rid,
        prompt="x" * tokens,
        prompt_token_ids=list(range(tokens)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    req.num_prompt_tokens = tokens
    return req


def _make_scheduler(
    *,
    gpu_memory_utilization: float = 0.5,
    enable_prefix_cache: bool = False,
) -> Scheduler:
    """Build a Scheduler against a stub model+tokenizer so we can drive
    admission control in isolation. We disable the prefix cache by
    default so the test focuses on the admission gate."""
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=enable_prefix_cache,
        use_memory_aware_cache=False,
        use_paged_cache=False,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    model = MagicMock()
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


class TestMetalCapAdmissionEnforcement:
    """Direct unit tests for ``_enforce_metal_cap_at_admission``."""

    def test_cap_disabled_when_util_is_zero(self):
        """SchedulerConfig default ``gpu_memory_utilization=0.0`` means
        the admission gate is a no-op — back-compat for callers that
        never set the flag (most unit tests, doctor harness)."""
        sched = _make_scheduler(gpu_memory_utilization=0.0)
        # Even with a synthetic 1 PB Metal active, the gate must not fire.
        with (
            patch.object(sched, "_current_metal_active_bytes", return_value=10**15),
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=0),
        ):
            # Should NOT raise — cap is disabled
            sched._enforce_metal_cap_at_admission(_make_request())
        assert sched.num_metal_cap_violations == 0

    def test_admit_passes_when_active_below_cap(self):
        """Below-cap active memory → admission proceeds, counter stays
        at zero. The gate must not be flaky on the happy path."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=50 * 10**9),
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        assert sched.num_metal_cap_violations == 0

    def test_admit_rejected_when_active_at_cap(self):
        """When active == cap, the admission gate fires (>= boundary,
        not strict >). Documented behavior — the original D-METAL-CAP
        repro showed allocator growth past the cap occurred BEFORE the
        next admit, so any equal-or-greater observation must reject."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=100 * 10**9
            ),
            pytest.raises(BackpressureError, match="D-METAL-CAP"),
        ):
            sched._enforce_metal_cap_at_admission(_make_request("req-a"))
        assert sched.num_metal_cap_violations == 1

    def test_counter_increments_per_rejection(self):
        """One increment per rejected admit. Sustained-pressure
        scenarios must show one counter tick per attempted request,
        not one per (request × eviction loop)."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=120 * 10**9
            ),
        ):
            for i in range(5):
                with pytest.raises(BackpressureError):
                    sched._enforce_metal_cap_at_admission(_make_request(f"req-{i}"))
        assert sched.num_metal_cap_violations == 5

    def test_warning_logged_only_once_per_process(self, caplog):
        """First over-cap admission logs a single WARNING (operator-
        actionable on first observation). Subsequent rejections only
        bump the counter so a sustained over-cap storm doesn't drown
        the rest of the engine log at 10 Hz (D-METAL-CAP repro
        sustained thousands of attempts per minute)."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        # ``_metal_cap_warning_logged`` should start False.
        assert sched._metal_cap_warning_logged is False
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            caplog.at_level(logging.WARNING),
        ):
            for i in range(3):
                with pytest.raises(BackpressureError):
                    sched._enforce_metal_cap_at_admission(_make_request(f"req-{i}"))
        d_metal_warnings = [
            r for r in caplog.records if "D-METAL-CAP" in r.getMessage()
        ]
        assert len(d_metal_warnings) == 1, (
            f"expected exactly 1 D-METAL-CAP WARNING, got "
            f"{len(d_metal_warnings)}: {[r.getMessage() for r in d_metal_warnings]}"
        )
        assert sched._metal_cap_warning_logged is True
        assert sched.num_metal_cap_violations == 3


class TestMetalCapAddRequestIntegration:
    """End-to-end checks via the public ``add_request`` entry point —
    proves the cap check fires BEFORE tokenization and propagates the
    BackpressureError to existing route plumbing."""

    def test_add_request_rejects_over_cap(self):
        """Over-cap admission via ``add_request`` raises
        ``BackpressureError`` — the same exception type the existing
        concurrent-cap path raises, so route handlers translate both to
        HTTP 503 via the same except branch."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=150 * 10**9
            ),
            pytest.raises(BackpressureError),
        ):
            sched.add_request(_make_request())
        # Request must NOT have been registered.
        assert "req-1" not in sched.requests
        assert sched.num_metal_cap_violations == 1

    def test_add_request_admits_under_cap(self):
        """Under-cap admission via ``add_request`` succeeds and the
        request appears in the scheduler's tracking dict — the existing
        cache-hit/prefix-cache machinery still runs after the cap
        check."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=10 * 10**9),
        ):
            sched.add_request(_make_request("req-ok"))
        assert "req-ok" in sched.requests
        assert sched.num_metal_cap_violations == 0


class TestGetStatsExposesCounter:
    """The Prometheus exporter renders
    ``rapid_mlx_metal_cap_violations_total`` from
    ``scheduler.get_stats()`` — this contract test pins the dict
    key so the route side cannot drift away."""

    def test_get_stats_exposes_counter_key(self):
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        stats = sched.get_stats()
        assert "num_metal_cap_violations" in stats
        assert stats["num_metal_cap_violations"] == 0

    def test_get_stats_reflects_rejection_count(self):
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            for i in range(2):
                with pytest.raises(BackpressureError):
                    sched._enforce_metal_cap_at_admission(_make_request(f"req-{i}"))
        stats = sched.get_stats()
        assert stats["num_metal_cap_violations"] == 2


class TestMetricsRoute:
    """Pin the Prometheus exposition format — operator dashboards key
    off the exact series name."""

    def test_metric_series_in_render(self):
        """``rapid_mlx_metal_cap_violations_total`` must appear in
        /metrics with the value from get_stats."""
        import types

        from vllm_mlx.routes.metrics import _render_prometheus

        cfg = types.SimpleNamespace(
            model_name="test",
            engine=types.SimpleNamespace(
                get_stats=lambda: {
                    "num_metal_cap_violations": 42,
                    "num_prefix_cache_pressure_evictions": 0,
                },
            ),
        )
        body = _render_prometheus(cfg)
        assert "rapid_mlx_metal_cap_violations_total 42" in body
        assert "# TYPE rapid_mlx_metal_cap_violations_total counter" in body, (
            "metric type must be 'counter' for monotonic rate() to work"
        )
