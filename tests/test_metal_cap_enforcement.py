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

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import BackpressureError, Scheduler, SchedulerConfig

# Rotating-cache capacity granularity — mirrors kv_estimation's
# ``_ROTATING_CACHE_STEP`` / ``_ROTATING_CACHE_KEEP`` (mlx_lm's
# ``RotatingKVCache.step = 256``; ``KEEP = 4`` is the largest keep any shipped
# construction uses). A sliding layer's per-request slot count is
# ``ceil((min(T, window) + keep) / step) * step`` — sublinear in the token
# budget, capped at the full window.
_STEP = 256
_KEEP = 4


def _slots(window: int, tokens: int) -> int:
    """Reference re-implementation of ``kv_estimation.rotating_cache_slots``."""
    if window <= 0 or tokens <= 0:
        return 0
    effective = min(tokens, window) + _KEEP
    return ((effective + _STEP - 1) // _STEP) * _STEP


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

    def test_str_prompt_fallback_uses_utf8_byte_length(self):
        """Codex round 6 HIGH #1 regression: when neither
        ``num_prompt_tokens`` nor ``prompt_token_ids`` is set, the
        estimate falls back to the prompt text. ``len(str)`` counts
        Python code points, which UNDER-estimates byte-fallback BPE
        tokenization of multi-byte glyphs (e.g. emoji). The fallback
        must use UTF-8 byte length so the gate stays conservative
        on adversarial prompts."""
        sched = self._make_scheduler_with_kv_estimate(kv_bytes_per_token=1)
        req = Request(
            request_id="emoji-prompt",
            # 100 skulls, each 4 UTF-8 bytes = 400 bytes. ``len(str)``
            # alone would say 100, which is a 4× undercount on a
            # byte-fallback tokenizer.
            prompt="💀" * 100,
            prompt_token_ids=None,
            sampling_params=SamplingParams(max_tokens=0),
        )
        req.num_prompt_tokens = 0
        kv = sched._estimate_request_kv_bytes(req)
        assert kv >= 400, (
            f"expected ≥400 bytes from UTF-8 byte-length fallback on "
            f"100 × 💀 (4 bytes/glyph), got {kv} — under-estimate "
            f"path round 6 HIGH #1 regression."
        )

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
        return Scheduler(model=MagicMock(), tokenizer=MagicMock(), config=config)

    def test_waiting_reservations_counted_in_cap_check(self):
        """3 already-admitted-but-not-stepped requests of ~25 GB each
        → admission of a 4th request that ALONE would fit must reject
        because the cumulative WAITING reservations push over cap."""
        sched = self._make_scheduler(kv_bytes_per_token=1_000_000)
        # Pre-seed 3 WAITING requests of ~25 GB each (25_000 tokens)
        for i in range(3):
            prev = _make_request(rid=f"prev-{i}", tokens=25_000)
            prev.sampling_params = SamplingParams(max_tokens=1)
            sched.waiting.append(prev)
        # The NEW request alone: 25 GB
        new_req = _make_request(rid="new", tokens=25_000)
        new_req.sampling_params = SamplingParams(max_tokens=1)
        # Cap is 100 GB, active is 10 GB. Alone, new_req would fit
        # (10 + 25 = 35 < 100). With 3 × 25 GB waiting, the cap
        # check sees 10 + 75 + 25 = 110 GB > 100 GB → reject.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=10 * 10**9),
            pytest.raises(BackpressureError, match="reserved KV"),
        ):
            sched._enforce_metal_cap_at_admission(new_req)
        assert sched.num_metal_cap_violations == 1

    def test_running_reservations_excluded_to_avoid_double_count(self):
        """Codex round 6 BLOCKING #3: requests in ``self.running``
        have ALREADY allocated their KV (which shows up in
        ``mx.get_active_memory()``). Including them in
        ``_sum_in_flight_kv_bytes`` double-counts and rejects every
        new admit after the first big request. The fix is to skip
        ``self.running`` and only iterate ``self.waiting``."""
        sched = self._make_scheduler(kv_bytes_per_token=1_000_000)
        # Pre-seed 3 RUNNING requests of ~25 GB each — they would
        # double-count active 75 GB if the bug were still live.
        for i in range(3):
            prev = _make_request(rid=f"prev-{i}", tokens=25_000)
            prev.sampling_params = SamplingParams(max_tokens=1)
            sched.running[prev.request_id] = prev
        new_req = _make_request(rid="new", tokens=25_000)
        new_req.sampling_params = SamplingParams(max_tokens=1)
        # Active honestly reports the 75 GB of running KV. Adding
        # waiting reservation (0) + new projection (25 GB) =
        # 75 + 0 + 25 = 100, which is == cap, so it would still
        # reject the new request — but ONLY because real Metal
        # genuinely sits at cap, not because of double-count.
        # Drop the new request to fit comfortably to verify the
        # admit path on the no-double-count case:
        small_req = _make_request(rid="small", tokens=100)
        small_req.sampling_params = SamplingParams(max_tokens=1)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=75 * 10**9),
        ):
            # 75 active + 0 waiting + 0.1 projected = 75.1 GB << 100 GB
            sched._enforce_metal_cap_at_admission(small_req)
        assert sched.num_metal_cap_violations == 0, (
            "running requests must be excluded from in-flight sum — "
            "their KV is already in active. Double-counting was "
            "the round 6 BLOCKING #3 regression."
        )

    def test_no_waiting_reservations_does_not_block(self):
        """Empty waiting deque → reserved_kv == 0, gate behaves
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
        # Pre-seed some WAITING requests; with per_tok=0, the sum
        # must be 0.
        for i in range(5):
            req = _make_request(rid=f"req-{i}", tokens=10_000)
            sched.waiting.append(req)
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
        return Scheduler(model=model, tokenizer=MagicMock(), config=config)

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


class TestModernDtypeKeyInference:
    """gemma4-on-18GB false-rejection regression.

    Transformers renamed the config dtype key ``torch_dtype`` →
    ``dtype``. Newer configs (e.g. Gemma 4, bf16) carry ONLY ``dtype``,
    so ``_infer_kv_dtype_bytes`` reading only ``torch_dtype`` fell
    through to the fp32 (4-byte) fallback — DOUBLING the projected KV vs
    the real bf16 (2-byte) cache and rejecting at admission a request
    whose real usage fit under the cap (gemma-4-12b on an 18 GB MBP:
    real 10.7 GB < 11.6 GB cap, but projected 14.4 GB).

    Uses ``SimpleNamespace`` rather than ``MagicMock`` so an UNSET
    attribute resolves to ``None`` (a MagicMock auto-creates every
    attribute, which would mask whether the modern key is actually
    read)."""

    def _sched(self, config_obj):
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
        model.config = config_obj
        return Scheduler(model=model, tokenizer=MagicMock(), config=config)

    def test_modern_dtype_key_bf16_uses_2_bytes(self):
        # Gemma 4 shape: only the modern ``dtype`` key (bf16), no
        # legacy ``torch_dtype``.
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        assert sched._infer_kv_dtype_bytes(cfg) == 2
        # 2 (K+V) × 4 layers × 2 kv_heads × 64 head_dim × 2 bytes —
        # NOT the fp32 (×4) over-estimate that caused the false reject.
        assert sched._resolve_kv_bytes_per_token() == 2 * 4 * 2 * 64 * 2

    def test_modern_dtype_key_torchdtype_str_form(self):
        # ``str(torch.bfloat16)`` form on the modern key.
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
            dtype="torch.bfloat16",
        )
        sched = self._sched(cfg)
        assert sched._infer_kv_dtype_bytes(cfg) == 2

    def test_modern_dtype_key_preferred_when_both_present(self):
        # Both keys present with CONFLICTING byte sizes → modern
        # ``dtype`` (bf16, 2 B) must win over legacy ``torch_dtype``
        # (fp32, 4 B), matching transformers' own precedence. Using a
        # conflicting pair is what actually pins the ordering — a
        # bf16/fp16 pair would pass regardless since both map to 2.
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
            dtype="bfloat16",
            torch_dtype="float32",
        )
        sched = self._sched(cfg)
        assert sched._infer_kv_dtype_bytes(cfg) == 2

    def test_legacy_torch_dtype_still_read(self):
        # Back-compat: a config carrying ONLY the legacy key still works.
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
            torch_dtype="float32",
        )
        sched = self._sched(cfg)
        assert sched._infer_kv_dtype_bytes(cfg) == 4

    def test_nested_text_config_dtype(self):
        # Multimodal config: the LM dtype lives on the nested
        # ``text_config`` while the top level has none.
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
            text_config=SimpleNamespace(dtype="bfloat16"),
        )
        sched = self._sched(cfg)
        assert sched._infer_kv_dtype_bytes(cfg) == 2

    def test_no_dtype_anywhere_falls_back_to_4(self):
        # Neither key, no text_config → genuine-unknown fp32 fallback.
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=2,
            head_dim=64,
        )
        sched = self._sched(cfg)
        assert sched._infer_kv_dtype_bytes(cfg) == 4

    def test_gemma4_12b_on_18gb_admitted_after_fix(self):
        """End-to-end regression for the reported gemma4用不了 bug.

        gemma-4-12b real config: 48 layers, 8 KV heads, head_dim 256,
        bf16 KV exposed via the MODERN ``dtype`` key. On an 18 GB MBP
        the sidecar default ``--gpu-memory-utilization 0.90`` yields an
        ~11.6 GB cap; the 4-bit weights sit at ~6.7 GB Metal-active.

        Per-token KV with the fix (bf16, 2 bytes):
            2 (K+V) × 48 × 8 × 256 × 2 = 393_216 B  (384 KB)
        A ~10 K-token budget projects ≈ 3.94 GB, so
            6.7 GB active + 3.94 GB projected ≈ 10.6 GB < 11.6 GB cap
        → ADMIT. Pre-fix the modern ``dtype`` key was ignored, the
        fp32 fallback doubled per-token KV to 786_432 B → ≈ 7.87 GB
        projected → 14.6 GB > cap → the first chat was rejected with
        BackpressureError ("Internal error during streaming") even
        though the model genuinely fit."""
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.90,
            metal_cap_kv_bytes_per_token=0,  # auto-derive from config
        )
        cfg = SimpleNamespace(
            num_hidden_layers=48,
            num_key_value_heads=8,
            head_dim=256,
            dtype="bfloat16",  # modern key only (Gemma 4 shape)
        )
        model = MagicMock()
        model.config = cfg
        sched = Scheduler(model=model, tokenizer=MagicMock(), config=config)

        # Per-token KV is the bf16 value, not the fp32 over-estimate.
        assert sched._resolve_kv_bytes_per_token() == 2 * 48 * 8 * 256 * 2

        req = _make_request(tokens=12)
        req.sampling_params = SamplingParams(max_tokens=10_000)
        with (
            patch.object(
                sched, "_resolve_metal_cap_bytes", return_value=11_600_000_000
            ),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=6_700_000_000
            ),
        ):
            # Must NOT raise — the request genuinely fits under the cap.
            sched._enforce_metal_cap_at_admission(req)
        assert sched.num_metal_cap_violations == 0

        # And the fp32 mis-read WOULD have exceeded the cap — proving the
        # fix is what flips this from reject to admit.
        fp32_projected = (2 * 48 * 8 * 256 * 4) * (12 + 10_000)
        assert 6_700_000_000 + fp32_projected > 11_600_000_000


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


class TestArchitectureAwareKVEstimate:
    """Wire-through of the architecture-aware KV estimator into the D-METAL-CAP
    admission path (``_resolve_kv_bytes_per_token`` /
    ``_estimate_request_kv_bytes``). The uniform-fallback and operator-override
    contracts must stay byte-identical (covered by the classes above); these
    add the hybrid-reduction cases.

    Uses ``SimpleNamespace`` for ``model.config`` so unset fields resolve to
    ``None`` (a ``MagicMock`` auto-creates every attribute, which would falsely
    look like a hybrid layer structure).
    """

    def _sched(self, config_obj, *, kv_bytes_override: int = 0):
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            metal_cap_kv_bytes_per_token=kv_bytes_override,
        )
        model = MagicMock()
        model.config = config_obj
        return Scheduler(model=model, tokenizer=MagicMock(), config=config)

    def test_dense_config_stays_uniform_with_zero_baseline(self):
        # No layer_types / sliding / sharing → byte-identical uniform, no
        # fixed baseline.
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        assert sched._resolve_kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
        assert sched._resolve_kv_fixed_baseline_bytes() == 0

    def test_gpt_oss_hybrid_reduces_per_token_and_exposes_slot_term(self):
        # ~50% sliding-window layers → unbounded per-token growth halved vs
        # uniform, and the sliding layers are exposed as a request-dependent slot
        # term (per-slot bytes + shared window), NOT a fixed baseline (codex round
        # 11 BLOCKING #1: a flat whole-window baseline over-counts short requests).
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        uniform = 2 * n * kv * hd * 2
        assert sched._resolve_kv_bytes_per_token() == uniform // 2
        # Sliding footprint is request-dependent → not in the fixed baseline.
        assert sched._resolve_kv_fixed_baseline_bytes() == 0
        assert sched._kv_sliding_slot_bytes == 2 * (n // 2) * kv * hd * 2
        assert sched._kv_sliding_window == window

    def test_gemma4_sharing_reduces_per_token(self):
        n, n_shared, kv, hd = 35, 20, 1, 128
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["full_attention"] * n,
            num_kv_shared_layers=n_shared,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        assert sched._resolve_kv_bytes_per_token() == 2 * (n - n_shared) * kv * hd * 2
        assert sched._resolve_kv_fixed_baseline_bytes() == 0

    def test_multimodal_nested_text_config_engages_reduction(self):
        # A multimodal config nests the text-tower dims under ``text_config`` and
        # the OUTER config carries a DECOY ``num_hidden_layers`` (the vision-tower
        # count) that differs from the nested text layer count. The scheduler must
        # resolve the base dims from the config whose OWN layer_types is valid —
        # the nested text config, NOT the decoy outer count — so the hybrid
        # reduction engages end-to-end and the uniform base is not read off the
        # wrong tower (codex round 8 + round 9 BLOCKING #1).
        n, kv, hd, window = 24, 8, 64, 128
        vision_layers = 27  # decoy != n
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            # DECOY: the outer config exposes the vision-tower layer count with
            # NO valid text layer_types of its own.
            num_hidden_layers=vision_layers,
            vision_config=SimpleNamespace(num_hidden_layers=vision_layers),
            text_config=SimpleNamespace(
                num_hidden_layers=n,
                num_key_value_heads=kv,
                head_dim=hd,
                layer_types=layer_types,
                sliding_window=window,
            ),
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        uniform = 2 * n * kv * hd * 2  # text dims (n=24), NOT the decoy 27
        # Reduction engaged off the nested text dims: growth halved, sliding
        # exposed as the request-dependent slot term.
        assert sched._resolve_kv_bytes_per_token() == uniform // 2
        assert sched._resolve_kv_fixed_baseline_bytes() == 0
        assert sched._kv_sliding_slot_bytes == 2 * (n // 2) * kv * hd * 2
        assert sched._kv_sliding_window == window

    def test_nested_config_without_layer_types_stays_byte_identical(self):
        # BYTE-IDENTICAL guard: a multimodal config whose text_config carries NO
        # valid layer_types (a non-hybrid nested model) must NOT adopt the nested
        # dims — the outer config has no num_hidden_layers, so admission
        # accounting stays exactly as before (0, no phantom estimate). Only a
        # genuine nested hybrid (valid layer_types) flips to the nested dims.
        cfg = SimpleNamespace(
            vision_config=SimpleNamespace(num_hidden_layers=27),
            text_config=SimpleNamespace(
                num_hidden_layers=32,
                num_key_value_heads=8,
                head_dim=128,
                # no layer_types → not a recognized nested hybrid
            ),
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        assert sched._resolve_kv_bytes_per_token() == 0
        assert sched._resolve_kv_fixed_baseline_bytes() == 0

    def test_operator_override_wins_and_zeroes_baseline(self):
        # Explicit knob short-circuits the estimator; no baseline is charged.
        cfg = SimpleNamespace(
            num_hidden_layers=24,
            num_key_value_heads=8,
            head_dim=64,
            layer_types=["sliding_attention", "full_attention"] * 12,
            sliding_window=128,
        )
        sched = self._sched(cfg, kv_bytes_override=42)
        assert sched._resolve_kv_bytes_per_token() == 42
        # The override short-circuits ALL architecture-aware refinement — no
        # baseline and no sliding term.
        assert sched._resolve_kv_fixed_baseline_bytes() == 0
        assert sched._kv_sliding_slot_bytes == 0
        assert sched._kv_sliding_window == 0

    def test_estimator_failure_preserves_uniform_estimate(self):
        # If the architecture-aware estimator raises, the scheduler must keep
        # the already-computed uniform per-token value (zero baseline), NOT
        # reset admission accounting to 0 (codex round 3 BLOCKING #3).
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        with patch(
            "vllm_mlx.scheduler.estimate_kv_footprint",
            side_effect=RuntimeError("boom"),
        ):
            per_tok = sched._resolve_kv_bytes_per_token()
        assert per_tok == 2 * 32 * 8 * 128 * 2  # uniform, not 0
        assert sched._resolve_kv_fixed_baseline_bytes() == 0

    def test_magicmock_config_yields_zero_and_no_baseline(self):
        # Bare MagicMock model — int(MagicMock()) path must still yield 0 with
        # no phantom baseline (back-compat for the stub-model unit tests).
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            gpu_memory_utilization=0.5,
            metal_cap_kv_bytes_per_token=0,
        )
        sched = Scheduler(model=MagicMock(), tokenizer=MagicMock(), config=config)
        assert sched._resolve_kv_bytes_per_token() == 0
        assert sched._resolve_kv_fixed_baseline_bytes() == 0

    def test_estimate_request_charges_recurrent_baseline_plus_growth(self):
        # A GatedDeltaNet hybrid: the recurrent state is a real per-sequence
        # fixed baseline (token-independent), charged once on top of the
        # full-attention layers' per-token growth.
        n, kv, hd = 16, 8, 64
        layer_types = ["linear_attention"] * (n - 4) + ["full_attention"] * 4
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        req = _make_request(tokens=100)
        req.sampling_params = SamplingParams(max_tokens=50)
        growth = sched._resolve_kv_bytes_per_token()
        baseline = sched._resolve_kv_fixed_baseline_bytes()
        assert baseline > 0  # recurrent state reserved
        assert sched._estimate_request_kv_bytes(req) == baseline + growth * (100 + 50)

    def test_estimate_request_charges_per_request_sliding_term(self):
        # A GPT-OSS-style hybrid: the request projection is
        # ``growth * T + slot_bytes * rotating_cache_slots(window, T)`` — the
        # sliding term GROWS with the token budget up to the full window (codex
        # round 11 BLOCKING #1), not a flat whole-window baseline. It is charged
        # identically regardless of prompt/decode split for the same token total.
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        growth = sched._resolve_kv_bytes_per_token()
        slot_bytes = sched._kv_sliding_slot_bytes
        assert slot_bytes == 2 * (n // 2) * kv * hd * 2
        assert sched._resolve_kv_fixed_baseline_bytes() == 0
        per_layer = 2 * kv * hd * 2
        n_full = n // 2
        n_sliding = n // 2

        for prompt, budget in ((40, 20), (16, 8_000), (8_000, 50)):
            req = _make_request(tokens=prompt)
            req.sampling_params = SamplingParams(max_tokens=budget)
            tokens = prompt + budget
            projected = sched._estimate_request_kv_bytes(req)
            assert projected == growth * tokens + slot_bytes * _slots(window, tokens)
            # Never under-counts the true footprint of the counted layers: full
            # layers grow exactly per token; the sliding slot count is >= the
            # sliding layers' min(tokens, window) peak.
            true_footprint = (
                n_full * tokens * per_layer
                + n_sliding * min(tokens, window) * per_layer
            )
            assert projected >= true_footprint

    def test_short_request_not_over_charged_full_window(self):
        # The codex round 11 win end-to-end: a SHORT request (T=1) against a big
        # window is charged only ONE step block for the sliding layers, far below
        # a whole-window baseline — so the D-METAL-CAP gate stops spuriously
        # rejecting short requests a flat baseline would have rejected.
        n, kv, hd, window = 8, 4, 64, 4096
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["sliding_attention"] * n,
            sliding_window=window,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        assert sched._resolve_kv_bytes_per_token() == 0  # all sliding, no growth
        slot_bytes = sched._kv_sliding_slot_bytes
        req = _make_request(tokens=1)
        req.sampling_params = SamplingParams(max_tokens=0)
        short = sched._estimate_request_kv_bytes(req)
        assert short == slot_bytes * 256  # one step block, NOT the whole window
        assert short < slot_bytes * _slots(window, window)  # << full-window charge

    def test_dict_config_behaves_identically_to_attribute_config(self):
        # Finding 3 (codex round 11 NIT): ``estimate_kv_footprint`` supports dict
        # configs, so the scheduler wiring (struct selection, base-dim reads, AND
        # dtype inference) must resolve dict-backed configs IDENTICALLY to
        # attribute configs — bare ``getattr`` on a dict returns ``None`` for
        # every field and would collapse the estimate to 0 (or fall to the fp32
        # dtype default).
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        fields = dict(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
            dtype="bfloat16",
        )
        attr_sched = self._sched(SimpleNamespace(**fields))
        dict_sched = self._sched(dict(fields))
        assert (
            dict_sched._resolve_kv_bytes_per_token()
            == attr_sched._resolve_kv_bytes_per_token()
        )
        assert dict_sched._kv_sliding_slot_bytes == attr_sched._kv_sliding_slot_bytes
        assert dict_sched._kv_sliding_window == attr_sched._kv_sliding_window
        assert (
            dict_sched._resolve_kv_fixed_baseline_bytes()
            == attr_sched._resolve_kv_fixed_baseline_bytes()
        )
        # Identical per-request projection too (dtype resolved to bf16, not fp32).
        req = _make_request(tokens=100)
        req.sampling_params = SamplingParams(max_tokens=200)
        assert dict_sched._estimate_request_kv_bytes(
            req
        ) == attr_sched._estimate_request_kv_bytes(req)

    def test_orphan_borrower_charged_not_zeroed_end_to_end(self):
        # Finding 2 (codex round 11 BLOCKING #2) through the scheduler: a config
        # whose LAST-N borrowers include one whose attention type has no producer
        # below the split — the exact per-index map (mirrored from
        # ``models/gemma4_vendored/language.py``) zeroes the mappable borrower and
        # charges the orphan its own footprint. Producers [full, full]; borrowers
        # [full (maps → zeroed), sliding (orphan → charged as a slot term)].
        n, kv, hd, window = 4, 8, 64, 128
        layer_types = [
            "full_attention",
            "full_attention",
            "full_attention",  # borrower idx2 → maps to a full producer → zeroed
            "sliding_attention",  # borrower idx3 → orphan → charged
        ]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            num_kv_shared_layers=2,
            sliding_window=window,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        # idx0, idx1 full grow per token; idx2 zeroed; idx3 orphan sliding charged.
        assert sched._resolve_kv_bytes_per_token() == 2 * 2 * kv * hd * 2
        assert sched._kv_sliding_slot_bytes == 2 * 1 * kv * hd * 2
        assert sched._kv_sliding_window == window

    def test_hybrid_admitted_where_uniform_would_reject(self):
        # The headline win: a GPT-OSS-style hybrid whose uniform per-token
        # over-count would trip the projection gate is ADMITTED once the
        # sliding layers are excluded from per-token growth (their whole window
        # is a bounded per-sequence baseline, dwarfed by the halved growth over a
        # long generation).
        #
        # This test carries its OWN negative control (uniform scheduler, SAME cap,
        # SAME request → REJECTED) so it proves the admission gate genuinely
        # consumes the projected KV bytes — not merely that the hybrid helper
        # returns a smaller number. Without the negative control the positive
        # "does not raise" assertion would still pass even if
        # ``_enforce_metal_cap_at_admission`` stopped applying the projection
        # entirely (codex round 10 test-strength BLOCKING).
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
            dtype="bfloat16",
        )
        sched = self._sched(cfg)
        req = _make_request(tokens=16)
        req.sampling_params = SamplingParams(max_tokens=8_000)
        tokens = 16 + 8_000
        uniform_per_tok = 2 * n * kv * hd * 2
        arch_projected = sched._estimate_request_kv_bytes(req)
        uniform_projected = uniform_per_tok * tokens
        # Arch-aware projection is strictly smaller than the uniform one for a
        # large token budget: the full layers grow at half the uniform rate and
        # the sliding layers contribute a bounded per-request slot term (capped at
        # the full window) instead of unbounded per-token growth.
        assert arch_projected < uniform_projected
        # Pick a cap that sits STRICTLY BETWEEN the two projections, so the SAME
        # cap admits the hybrid and rejects the uniform over-count. Concrete
        # numbers for this config (dtype=bf16, n=24, kv=8, hd=64, window=128,
        # 16+8000 tokens): uniform_per_tok = 49_152 B; arch per_tok = 24_576 B
        # (half) + a 6_291_456 B sliding baseline. So arch_projected ≈ 203.3 MB,
        # uniform_projected ≈ 394.0 MB, and the cap delta above ``active`` is
        # their midpoint ≈ 298.6 MB — comfortably above arch, comfortably below
        # uniform.
        active = 1_000_000_000
        cap = active + (arch_projected + uniform_projected) // 2
        assert active + arch_projected < cap  # hybrid fits under the cap
        assert active + uniform_projected > cap  # uniform over-count exceeds it

        # POSITIVE: the architecture-aware hybrid scheduler ADMITS the request at
        # this cap (does not raise), because its reduced projection fits.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(sched, "_current_metal_active_bytes", return_value=active),
        ):
            sched._enforce_metal_cap_at_admission(req)  # must NOT raise
        assert sched.num_metal_cap_violations == 0

        # NEGATIVE CONTROL: a scheduler on the SAME base dims (same num layers /
        # kv heads / head_dim) but a DENSE config (no ``layer_types`` / no
        # ``sliding_window``) forces the UNIFORM estimate — no hybrid reduction,
        # zero baseline — so its projection for the SAME request is exactly
        # ``uniform_projected``. At the SAME cap that admitted the hybrid, this
        # uniform scheduler MUST raise ``BackpressureError``. The pair
        # (uniform → rejected, hybrid → admitted, same cap, same request) is the
        # real proof that the reduced estimate is what flips admission and that
        # the gate is actually applying the projected KV bytes.
        dense_cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            dtype="bfloat16",  # NO layer_types / sliding_window → uniform, base 0
        )
        dense_sched = self._sched(dense_cfg)
        dense_req = _make_request(tokens=16)
        dense_req.sampling_params = SamplingParams(max_tokens=8_000)
        # The dense scheduler really is on the uniform estimate with no baseline,
        # so its projection is precisely the uniform over-count that exceeds cap.
        assert dense_sched._resolve_kv_bytes_per_token() == uniform_per_tok
        assert dense_sched._resolve_kv_fixed_baseline_bytes() == 0
        assert dense_sched._estimate_request_kv_bytes(dense_req) == uniform_projected
        with (
            patch.object(dense_sched, "_resolve_metal_cap_bytes", return_value=cap),
            patch.object(
                dense_sched, "_current_metal_active_bytes", return_value=active
            ),
            pytest.raises(BackpressureError, match="D-METAL-CAP"),
        ):
            dense_sched._enforce_metal_cap_at_admission(dense_req)
        assert dense_sched.num_metal_cap_violations == 1
