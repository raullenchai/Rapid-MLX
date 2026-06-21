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


class TestProjectedKVAdmissionGate:
    """Codex round 3 BLOCKING #1 regression — when
    ``metal_cap_kv_bytes_per_token > 0`` the admission gate must
    reject a request whose ``active + projected_kv`` would push the
    Metal allocator past the cap, EVEN IF current active is still
    below cap. Without this, a single large 32k-prefill admitted at
    e.g. 60% of cap can grow active to 130% of cap before the
    next admission tick fires — the exact D-METAL-CAP failure mode."""

    def _make_scheduler_with_kv_estimate(
        self,
        gpu_memory_utilization: float = 0.5,
        kv_bytes_per_token: int = 1_000_000,
    ) -> Scheduler:
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=gpu_memory_utilization,
            metal_cap_kv_bytes_per_token=kv_bytes_per_token,
        )
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))
        return Scheduler(model=MagicMock(), tokenizer=tokenizer, config=config)

    def test_below_cap_admitted_when_projection_zero(self):
        """Back-compat: ``metal_cap_kv_bytes_per_token=0`` (default)
        means projection is 0 and the gate falls back to the old
        current-active-only check."""
        sched = self._make_scheduler_with_kv_estimate(kv_bytes_per_token=0)
        req = _make_request(tokens=32_000)
        req.sampling_params = SamplingParams(max_tokens=1024)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=60 * 10**9),
        ):
            # Should NOT raise — active < cap, projection is zero.
            sched._enforce_metal_cap_at_admission(req)
        assert sched.num_metal_cap_violations == 0

    def test_below_cap_rejected_when_projected_kv_pushes_over(self):
        """The core round 3 BLOCKING #1 fix: active < cap, but
        ``active + projected_kv >= cap`` → reject."""
        # 1 MB per token × (32_000 prompt + 1024 max_tokens) ≈ 33 GB
        sched = self._make_scheduler_with_kv_estimate(kv_bytes_per_token=1_000_000)
        req = _make_request(tokens=32_000)
        req.sampling_params = SamplingParams(max_tokens=1024)
        # 80 GB active + ~33 GB projected = 113 GB > 100 GB cap
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=80 * 10**9),
            pytest.raises(BackpressureError, match="D-METAL-CAP"),
        ):
            sched._enforce_metal_cap_at_admission(req)
        assert sched.num_metal_cap_violations == 1

    def test_below_cap_admitted_when_projection_fits(self):
        """A small request whose projection stays inside cap should
        still admit — projection-aware gate must not reject every
        request."""
        sched = self._make_scheduler_with_kv_estimate(kv_bytes_per_token=1_000_000)
        req = _make_request(tokens=100)
        req.sampling_params = SamplingParams(max_tokens=100)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=10 * 10**9),
        ):
            # 10 GB active + (100+100) × 1 MB = ~10.2 GB << 100 GB cap
            sched._enforce_metal_cap_at_admission(req)
        assert sched.num_metal_cap_violations == 0

    def test_estimate_uses_prompt_token_ids_when_num_prompt_tokens_zero(self):
        """The estimate must fall back to ``len(prompt_token_ids)``
        when ``num_prompt_tokens`` hasn't been populated yet (route
        layer hasn't tokenized — admission runs BEFORE tokenization).
        Otherwise the projection would always be ``max_tokens × per_tok``
        on freshly-arrived requests and miss long-prompt over-cap
        cases."""
        sched = self._make_scheduler_with_kv_estimate(kv_bytes_per_token=1_000_000)
        req = _make_request(tokens=32_000)
        req.num_prompt_tokens = 0  # simulate pre-tokenization
        req.sampling_params = SamplingParams(max_tokens=10)
        # 32k from len(prompt_token_ids) + 10 max_tokens → ~32 GB
        kv = sched._estimate_request_kv_bytes(req)
        assert 30 * 10**9 < kv < 35 * 10**9

    def test_auto_derived_kv_bytes_from_model_config(self):
        """Codex round 4 BLOCKING #1+#2 closure: when
        ``metal_cap_kv_bytes_per_token=0`` (default), the scheduler
        MUST auto-derive a conservative per-token KV size from the
        model.config so the projection branch protects the cap OUT
        OF THE BOX. Pre-fix, the default 0 made the projection
        branch dead code and the gate fell back to current-active-
        only — the exact "below cap, single prefill grows past cap"
        failure mode this PR claims to fix."""
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            # Operator did NOT set this — auto-derivation kicks in.
            metal_cap_kv_bytes_per_token=0,
        )
        # Synthesize a model.config that looks like a 35B-ish setup:
        # num_layers=80, num_kv_heads=8, head_dim=128 →
        # per_tok = 2 (K+V) × 80 × 8 × 128 × 2 (fp16) = 327_680
        model = MagicMock()
        model.config.num_hidden_layers = 80
        model.config.num_key_value_heads = 8
        model.config.head_dim = 128
        model.config.torch_dtype = "float16"
        sched = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        per_tok = sched._resolve_kv_bytes_per_token()
        assert per_tok == 2 * 80 * 8 * 128 * 2, (
            f"auto-derived per-token KV size should be {2 * 80 * 8 * 128 * 2}, "
            f"got {per_tok}"
        )

    def test_operator_override_wins_over_auto_derivation(self):
        """When the operator explicitly sets
        ``metal_cap_kv_bytes_per_token > 0``, that value MUST be
        used instead of the auto-derivation — operators on
        quantized-KV deployments want a tighter value than the
        fp16-assuming auto path computes."""
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            metal_cap_kv_bytes_per_token=42,  # explicit override
        )
        model = MagicMock()
        # Even with a plausible auto-derivation source available,
        # the explicit knob wins.
        model.config.num_hidden_layers = 80
        model.config.num_key_value_heads = 8
        model.config.head_dim = 128
        sched = Scheduler(model=model, tokenizer=MagicMock(), config=config)
        per_tok = sched._resolve_kv_bytes_per_token()
        assert per_tok == 42, (
            f"operator override should win, got {per_tok} instead of 42"
        )

    def test_auto_derivation_falls_back_to_zero_on_missing_model_config(self):
        """When the model has no ``config`` attribute (unit-test
        MagicMock without explicit setup), auto-derivation must
        return 0 (= projection branch disabled) rather than
        raise — keeps back-compat for the large body of existing
        unit tests built against stub models."""
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            metal_cap_kv_bytes_per_token=0,
        )
        # Bare model with no config — MagicMock will return a MagicMock
        # for .config but int(MagicMock()) would raise TypeError. The
        # auto-derivation must swallow that and return 0.
        sched = Scheduler(model=MagicMock(), tokenizer=MagicMock(), config=config)
        per_tok = sched._resolve_kv_bytes_per_token()
        assert per_tok == 0, (
            f"auto-derivation must return 0 on missing/malformed "
            f"model.config to preserve unit-test back-compat, got {per_tok}"
        )

    def test_warning_log_safe_on_nonstring_request_id(self, caplog):
        """Codex round 3 NIT #4 regression: ``request_id`` slicing
        must coerce via ``str(...)`` so a malformed test/request
        object with a non-string id doesn't turn the backpressure
        warning into an unrelated ``TypeError``."""
        import logging

        sched = self._make_scheduler_with_kv_estimate(kv_bytes_per_token=0)
        req = _make_request()
        req.request_id = 12345  # malformed — int instead of str
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            caplog.at_level(logging.WARNING),
            pytest.raises(BackpressureError),
        ):
            sched._enforce_metal_cap_at_admission(req)
        # The WARNING must have been emitted, not crashed with a
        # TypeError before logging.
        d_metal_warnings = [
            r for r in caplog.records if "D-METAL-CAP" in r.getMessage()
        ]
        assert len(d_metal_warnings) == 1, (
            "non-string request_id must not break the D-METAL-CAP "
            "WARNING path — codex round 3 NIT #4."
        )


class TestInFlightKVReservation:
    """Codex round 5 BLOCKING #1: the admission gate must include
    KV reservations of every already-admitted-but-not-finished
    request in its check. Without this, N concurrent admits each
    individually under cap will STACK and blow the cap collectively
    — ``mx.get_active_memory`` lags the allocator until the
    BatchGenerator picks up each request and runs its first step."""

    def _make_scheduler(self, kv_bytes_per_token: int = 1_000_000):
        config = SchedulerConfig(
            max_num_seqs=64,
            max_concurrent_requests=128,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            metal_cap_kv_bytes_per_token=kv_bytes_per_token,
        )
        return Scheduler(
            model=MagicMock(), tokenizer=MagicMock(), config=config
        )

    def test_in_flight_reservations_counted_in_cap_check(self):
        """3 already-admitted requests of ~25 GB each → admission
        of a 4th request that ALONE would fit must reject because
        the cumulative reservations push over cap."""
        sched = self._make_scheduler(kv_bytes_per_token=1_000_000)
        # Pre-seed 3 in-flight requests of ~25 GB each (25_000 tokens)
        for i in range(3):
            prev = _make_request(rid=f"prev-{i}", tokens=25_000)
            prev.sampling_params = SamplingParams(max_tokens=1)
            sched.requests[prev.request_id] = prev
        # The NEW request alone: 25 GB
        new_req = _make_request(rid="new", tokens=25_000)
        new_req.sampling_params = SamplingParams(max_tokens=1)
        # Cap is 100 GB, active is 10 GB. Alone, new_req would fit
        # (10 + 25 = 35 < 100). With 3 × 25 GB reserved, the cap
        # check sees 10 + 75 + 25 = 110 GB > 100 GB → reject.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=10 * 10**9),
            pytest.raises(BackpressureError, match="reserved KV"),
        ):
            sched._enforce_metal_cap_at_admission(new_req)
        assert sched.num_metal_cap_violations == 1

    def test_no_in_flight_reservations_does_not_block(self):
        """Empty in-flight set → reserved_kv == 0, gate behaves
        exactly like the round-4 single-request projection path."""
        sched = self._make_scheduler(kv_bytes_per_token=1_000_000)
        new_req = _make_request(rid="new", tokens=10_000)
        new_req.sampling_params = SamplingParams(max_tokens=10)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=10 * 10**9),
        ):
            # 10 GB active + 0 reserved + ~10 GB projected = 20 GB << 100 GB
            sched._enforce_metal_cap_at_admission(new_req)
        assert sched.num_metal_cap_violations == 0

    def test_sum_in_flight_kv_bytes_returns_zero_without_per_tok(self):
        """When per-token KV size cannot be resolved (per_tok == 0,
        e.g. unit-test stub model with no .config), the helper
        must short-circuit to 0 rather than wandering into
        ``_estimate_request_kv_bytes`` which also returns 0 — keeps
        the inner loop cheap on the back-compat path."""
        sched = self._make_scheduler(kv_bytes_per_token=0)
        # Pre-seed some requests; with per_tok=0, the sum must be 0.
        for i in range(5):
            req = _make_request(rid=f"req-{i}", tokens=10_000)
            sched.requests[req.request_id] = req
        # Force the auto-derivation to return 0 (no .config on model)
        with patch.object(sched, "_resolve_kv_bytes_per_token", return_value=0):
            total = sched._sum_in_flight_kv_bytes()
        assert total == 0


class TestDtypeInference:
    """Codex round 5 BLOCKING #2: auto-derived per-token KV bytes
    must reflect the model dtype rather than hardcoding ``2`` for
    fp16/bf16. Operators running fp32 KV cache need the gate to
    over-estimate, not under-estimate."""

    def _build_sched_with_dtype(self, dtype):
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            metal_cap_kv_bytes_per_token=0,  # exercise auto-derivation
        )
        model = MagicMock()
        model.config.num_hidden_layers = 4
        model.config.num_key_value_heads = 2
        model.config.head_dim = 64
        model.config.torch_dtype = dtype
        return Scheduler(
            model=model, tokenizer=MagicMock(), config=config
        )

    def test_fp16_uses_2_bytes(self):
        sched = self._build_sched_with_dtype("float16")
        per_tok = sched._resolve_kv_bytes_per_token()
        # 2 (K+V) × 4 layers × 2 kv_heads × 64 head_dim × 2 bytes = 2048
        assert per_tok == 2 * 4 * 2 * 64 * 2

    def test_bf16_uses_2_bytes(self):
        sched = self._build_sched_with_dtype("bfloat16")
        per_tok = sched._resolve_kv_bytes_per_token()
        assert per_tok == 2 * 4 * 2 * 64 * 2

    def test_fp32_uses_4_bytes(self):
        sched = self._build_sched_with_dtype("float32")
        per_tok = sched._resolve_kv_bytes_per_token()
        # Should DOUBLE the fp16 size — the round-5 fix.
        assert per_tok == 2 * 4 * 2 * 64 * 4

    def test_unknown_dtype_falls_back_to_4_bytes(self):
        """Unknown / missing dtype → assume fp32 (largest plausible)
        so admission gate OVER-estimates KV size. Under-estimating
        is the unsafe direction."""
        sched = self._build_sched_with_dtype("some-future-quant-format")
        per_tok = sched._resolve_kv_bytes_per_token()
        assert per_tok == 2 * 4 * 2 * 64 * 4


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
