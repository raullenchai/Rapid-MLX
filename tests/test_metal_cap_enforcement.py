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

from rapid_mlx.errors import MetalMemoryBackpressureError
from rapid_mlx.request import Request, RequestStatus, SamplingParams
from rapid_mlx.scheduler import (
    BackpressureError,
    BatchGenerator,
    Scheduler,
    SchedulerConfig,
)

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
    allow_context_overcommit: bool = False,
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
        allow_context_overcommit=allow_context_overcommit,
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

    def test_explicit_context_can_exceed_automatic_memory_projection(self):
        sched = _make_scheduler(allow_context_overcommit=True)
        # The explicit logical window takes priority over the automatic
        # estimate; the model or allocator may still run out of real memory.
        assert sched._resolve_metal_cap_bytes() == 0
        with patch.object(sched, "_current_metal_active_bytes", return_value=10**15):
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

    def test_waiting_reservations_queue_instead_of_rejecting(self):
        """Temporary waiting pressure belongs to promotion, not the 503 gate."""
        sched = self._make_scheduler(kv_bytes_per_token=1_000_000)
        for i in range(3):
            sched.waiting.append(_make_request(rid=f"prev-{i}", tokens=25_000))
        new_req = _make_request(rid="new", tokens=25_000)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=10 * 10**9),
        ):
            sched._enforce_metal_cap_at_admission(new_req)
        assert sched.num_metal_cap_violations == 0

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

        from rapid_mlx.routes.metrics import _render_prometheus

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

    def test_nested_config_without_layer_types_uses_geometry_fallback(self):
        # When the outer config lacks geometry, the live-dims fallback can
        # still resolve valid integer text-tower dims without hybrid metadata.
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
        assert sched._resolve_kv_bytes_per_token() == 2 * 32 * 8 * 128 * 4
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
            "rapid_mlx.scheduler.estimate_kv_footprint",
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


class TestAdmissionFallbackDebugLogging:
    """Silent estimator fallbacks must be visible at DEBUG, once per key."""

    def test_metal_probe_failure_logs_debug_once_with_traceback(self, caplog):
        sched = _make_scheduler()
        with (
            patch(
                "rapid_mlx.scheduler.mx.get_active_memory",
                side_effect=RuntimeError("no metal"),
            ),
            caplog.at_level(logging.DEBUG, logger="rapid_mlx.scheduler"),
        ):
            assert sched._current_metal_active_bytes() == 0
            assert sched._current_metal_active_bytes() == 0
        records = [
            r
            for r in caplog.records
            if "Metal active-memory probe failed" in r.getMessage()
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.DEBUG
        assert records[0].exc_info is not None

    def test_kv_dtype_inference_failure_logs_and_keeps_fp32_fallback(self, caplog):
        class _ExplodingConfig:
            @property
            def dtype(self):
                raise RuntimeError("broken config")

        sched = _make_scheduler()
        with caplog.at_level(logging.DEBUG, logger="rapid_mlx.scheduler"):
            assert sched._infer_kv_dtype_bytes(_ExplodingConfig()) == 4
        assert any(
            "KV dtype inference" in r.getMessage() and r.exc_info is not None
            for r in caplog.records
        )

    def test_phys_footprint_failure_falls_back_to_rss_and_logs(self, caplog):
        import psutil

        sched = _make_scheduler()
        rss = SimpleNamespace(rss=123_456)
        with (
            patch(
                "rapid_mlx.runtime.process_memory.get_phys_footprint",
                side_effect=OSError("no footprint"),
            ),
            patch.object(
                psutil, "Process", return_value=SimpleNamespace(memory_info=lambda: rss)
            ),
            caplog.at_level(logging.DEBUG, logger="rapid_mlx.scheduler"),
        ):
            assert sched._current_process_resident_bytes() == 123_456
        assert any(
            "phys_footprint probe failed" in r.getMessage() and r.exc_info
            for r in caplog.records
        )

    def test_rss_failure_after_footprint_failure_returns_zero_and_logs(self, caplog):
        import psutil

        sched = _make_scheduler()
        with (
            patch(
                "rapid_mlx.runtime.process_memory.get_phys_footprint",
                side_effect=OSError("no footprint"),
            ),
            patch.object(psutil, "Process", side_effect=RuntimeError("no psutil")),
            caplog.at_level(logging.DEBUG, logger="rapid_mlx.scheduler"),
        ):
            assert sched._current_process_resident_bytes() == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any("phys_footprint probe failed" in m for m in messages)
        assert any("RSS probe failed" in m for m in messages)


class TestOverCapGarbageReclaim:
    """#4108: an idle server whose Metal memory was pinned by cyclic garbage
    (prefix cache empty, nothing running) rejected every request until
    restart. Over the cap, admission now collects garbage first and
    re-measures; a server that truly holds the memory says retrying cannot
    help instead of claiming requests are in flight."""

    CAP = 100 * 10**9

    @pytest.fixture(autouse=True)
    def _fresh_reclaim_limiter(self):
        # The limiter is process-wide; give every test a clean slate.
        Scheduler._last_metal_gc_reclaim_at = float("-inf")
        yield
        Scheduler._last_metal_gc_reclaim_at = float("-inf")

    def test_projection_rejection_under_load_does_not_collect(self):
        """A full collection pauses the step loop. Requests in flight with
        active memory below the cap is ordinary load, not garbage: reject
        without collecting."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        sched.requests["other"] = _make_request(rid="other")
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(sched, "_current_metal_active_bytes", return_value=60 * 10**9),
            patch.object(sched, "_estimate_request_kv_bytes", return_value=50 * 10**9),
            patch("rapid_mlx.scheduler.gc.collect") as collect,
            pytest.raises(MetalMemoryBackpressureError),
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        collect.assert_not_called()

    def test_idle_projection_rejection_collects(self):
        """Idle, a collection costs nobody anything and garbage may be what
        pushes the projection over the cap."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        readings = iter([60 * 10**9, 60 * 10**9, 20 * 10**9, 20 * 10**9])
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(
                sched, "_current_metal_active_bytes", side_effect=lambda: next(readings)
            ),
            patch.object(sched, "_estimate_request_kv_bytes", return_value=50 * 10**9),
            patch("rapid_mlx.scheduler.gc.collect", return_value=1) as collect,
            patch("rapid_mlx.scheduler.mx.clear_cache"),
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        collect.assert_called_once()
        assert sched.num_metal_cap_violations == 0

    def test_limiter_is_shared_across_schedulers(self):
        first = _make_scheduler(gpu_memory_utilization=0.5)
        second = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(first, "_current_metal_active_bytes", return_value=1),
            patch.object(second, "_current_metal_active_bytes", return_value=1),
            patch("rapid_mlx.scheduler.gc.collect", return_value=0) as collect,
            patch("rapid_mlx.scheduler.mx.clear_cache"),
        ):
            first._reclaim_unreachable_metal()
            second._reclaim_unreachable_metal()
        assert collect.call_count == 1

    def test_admits_when_collection_frees_memory(self, caplog):
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        # Over cap before the collection, well under after it.
        readings = iter([120 * 10**9, 120 * 10**9, 40 * 10**9, 40 * 10**9])
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(
                sched, "_current_metal_active_bytes", side_effect=lambda: next(readings)
            ),
            patch("rapid_mlx.scheduler.gc.collect", return_value=7) as collect,
            patch("rapid_mlx.scheduler.mx.clear_cache") as clear,
            caplog.at_level(logging.WARNING, logger="rapid_mlx.scheduler"),
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        collect.assert_called_once()
        clear.assert_called_once()
        assert sched.num_metal_cap_violations == 0
        assert "[D-METAL-CAP-gc-reclaim] released 80.0 GB" in caplog.text

    def test_collection_runs_before_prefix_cache_eviction(self):
        """Collecting is non-destructive; the warm prefix cache must not be
        evicted when garbage alone accounts for the overage."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        sched.memory_aware_cache = MagicMock()
        readings = iter([120 * 10**9, 120 * 10**9, 40 * 10**9, 40 * 10**9])
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(
                sched, "_current_metal_active_bytes", side_effect=lambda: next(readings)
            ),
            patch.object(sched, "_evict_one_prefix_cache_entry") as evict,
            patch("rapid_mlx.scheduler.gc.collect", return_value=0),
            patch("rapid_mlx.scheduler.mx.clear_cache"),
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        evict.assert_not_called()

    def test_reclaim_is_rate_limited(self):
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        with (
            patch.object(sched, "_current_metal_active_bytes", return_value=self.CAP),
            patch("rapid_mlx.scheduler.gc.collect", return_value=0) as collect,
            patch("rapid_mlx.scheduler.mx.clear_cache"),
            patch("rapid_mlx.scheduler.time.monotonic", side_effect=[10.0, 10.5, 11.0]),
        ):
            assert sched._reclaim_unreachable_metal() is False
            assert sched._reclaim_unreachable_metal() is False  # within 1 s
            assert sched._reclaim_unreachable_metal() is False
        assert collect.call_count == 2

    def test_reclaim_tolerates_clear_cache_failure(self):
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        readings = iter([50 * 10**9, 30 * 10**9])
        with (
            patch.object(
                sched, "_current_metal_active_bytes", side_effect=lambda: next(readings)
            ),
            patch("rapid_mlx.scheduler.gc.collect", return_value=3),
            patch("rapid_mlx.scheduler.mx.clear_cache", side_effect=RuntimeError("x")),
        ):
            assert sched._reclaim_unreachable_metal() is True

    def test_idle_server_over_cap_says_retry_cannot_help(self):
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        sched._metal_cap_effective_utilization = 0.5
        assert not sched.requests
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=self.CAP + 1
            ),
            patch("rapid_mlx.scheduler.gc.collect", return_value=0),
            pytest.raises(MetalMemoryBackpressureError) as info,
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        message = str(info.value)
        assert "with no request running" in message
        assert "Retrying will not help" in message
        assert "Retry after in-flight requests drain" not in message
        assert "needs approximately" not in message
        assert "--gpu-memory-utilization" in message
        assert "D-METAL-CAP" in message and "reserved KV" in message

    def test_idle_message_drops_utilization_advice_at_max(self):
        sched = _make_scheduler(gpu_memory_utilization=1.0)
        sched._metal_cap_effective_utilization = 1.0
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(sched, "_current_metal_active_bytes", return_value=self.CAP),
            patch("rapid_mlx.scheduler.gc.collect", return_value=0),
            pytest.raises(MetalMemoryBackpressureError) as info,
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        message = str(info.value)
        assert "Restart the server to release it." in message
        assert "--gpu-memory-utilization" not in message

    def test_busy_server_keeps_drain_advice(self):
        """With requests in flight, waiting for them to drain IS the remedy."""
        sched = _make_scheduler(gpu_memory_utilization=0.5)
        sched.requests["other"] = _make_request(rid="other")
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=self.CAP),
            patch.object(sched, "_current_metal_active_bytes", return_value=self.CAP),
            patch("rapid_mlx.scheduler.gc.collect", return_value=0),
            pytest.raises(MetalMemoryBackpressureError) as info,
        ):
            sched._enforce_metal_cap_at_admission(_make_request())
        message = str(info.value)
        assert "needs approximately" in message
        assert "Retry after in-flight requests drain" in message


# Cohort admission uses fake allocator bytes and a recording generator; no GPU
# allocations are needed to exercise the production waiting->running path.
def _cohort_scheduler():
    sched = _make_scheduler()
    sched._resolve_metal_cap_bytes = lambda: 1000
    sched._current_metal_active_bytes = lambda: 400
    sched._resolve_kv_bytes_per_token = lambda: 0
    sched._resolve_kv_fixed_baseline_bytes = lambda: 0
    sched._kv_sliding_slot_bytes = 0
    sched._kv_sliding_window = 0
    sched._estimate_request_kv_bytes = lambda r, **kwargs: 50
    sched._prefill_workspace_high_water = 1.5
    sched._prefill_workspace_calibrated = True
    sched.config.prefill_step_size = 10
    queue = []

    def insert(tokens, **kwargs):
        uid = len(sched.running) + 100
        queue.append((uid, [tokens[0]], 1, None, []))
        return [uid]

    sched.batch_generator = SimpleNamespace(
        prefill_step_size=10,
        _prompt_batch=SimpleNamespace(uids=[]),
        _currently_processing=[],
        _unprocessed_sequences=queue,
        insert=insert,
    )
    sched._ensure_batch_generator = lambda params: True
    return sched


def test_prefill_cohorts_queue_eight_finish_in_order_and_retry(caplog):
    sched = _cohort_scheduler()
    requests = [_make_request(str(i), 10) for i in range(8)]
    for r in requests:
        sched._enforce_metal_cap_at_admission(r)
        sched.waiting.append(r)
        sched.requests[r.request_id] = r
    with caplog.at_level(logging.INFO):
        first = sched._schedule_waiting()
        assert first == requests[:3]
        assert list(sched.waiting) == requests[3:]
        assert sched._schedule_waiting() == []
    assert sum("[prefill-cohort]" in r.message for r in caplog.records) == 1
    finished = []
    while sched.running or sched.waiting:
        finished.extend(sched.running.values())
        for r in sched.running.values():
            r.output_token_ids = [1]
            r.status = RequestStatus.FINISHED_STOPPED
        with patch("rapid_mlx.scheduler.mx.clear_cache"):
            sched._cleanup_finished(set(sched.running))
        sched.batch_generator._unprocessed_sequences.clear()
        sched._schedule_waiting()
    assert finished == requests
    assert sched.finished_req_ids == {r.request_id for r in requests}
    assert all(r.status == RequestStatus.FINISHED_STOPPED for r in requests)
    assert sched.num_metal_cap_violations == 0


def test_prefill_existing_processed_and_hidden_insert_rows_counted():
    sched = _cohort_scheduler()
    rows = [_make_request(str(i), 10) for i in range(4)]
    sched.waiting.extend(rows[:2])
    assert sched._schedule_waiting() == rows[:2]
    bg = sched.batch_generator
    uid = rows[0].batch_uid
    bg._unprocessed_sequences.pop(0)
    bg._prompt_batch.uids = [uid]
    bg._currently_processing = [[[[1, 2]], 8, 10]]
    assert sched._prefill_memory_rows() == [(rows[0], 8), (rows[1], 0)]
    sched.waiting.extend(rows[2:])
    reclaims = []
    original_reclaim = sched._reclaim_prefix_cache_for_prefill
    sched._reclaim_prefix_cache_for_prefill = lambda *a, **k: (
        reclaims.append(a) or original_reclaim(*a, **k)
    )
    assert sched._schedule_waiting() == []  # stepped cohorts stay sealed
    assert reclaims == []  # a sealed cohort never evicts prefixes to defer
    bg._currently_processing[0][1] = 0
    assert sched._schedule_waiting() == rows[2:3]
    assert list(sched.waiting) == rows[3:]


def test_prefill_prefix_and_processed_kv_are_not_double_charged():
    sched = _cohort_scheduler()
    r = _make_request(tokens=10)
    sched._resolve_kv_bytes_per_token = lambda: 10
    sched._resolve_kv_fixed_baseline_bytes = lambda: 20
    sched._estimate_request_kv_bytes = lambda r, **kwargs: 130
    assert sched._prefill_unmaterialized_bytes(r, 0) == 130
    r.cached_tokens = 6
    assert sched._prefill_unmaterialized_bytes(r, 2) == 30
    r.batch_uid = 100
    sched.running[r.request_id] = r
    r.output_token_ids = [7]
    assert sched._prefill_unmaterialized_bytes(r, 0) == 0
    assert sched._prefill_workspace_units(r) == 40  # full context, suffix chunk


@pytest.mark.parametrize("disabled_cap", [0, -1])
def test_prefill_singleton_progress_and_disabled_cap(disabled_cap):
    sched = _cohort_scheduler()
    sched._current_metal_active_bytes = lambda: 9999
    sched.waiting.extend([_make_request("a", 10), _make_request("b", 10)])
    assert [r.request_id for r in sched._schedule_waiting()] == ["a"]
    assert [r.request_id for r in sched.waiting] == ["b"]
    sched._resolve_metal_cap_bytes = lambda: disabled_cap
    assert sched._begin_prefill_memory_sample() is None
    assert [r.request_id for r in sched._schedule_waiting()] == ["b"]


def test_prefill_memory_sample_skips_without_generator():
    sched = _cohort_scheduler()
    sched._resolve_metal_cap_bytes = lambda: 1000
    sched.batch_generator = None
    assert sched._begin_prefill_memory_sample() is None


def test_issue5021_cold_short_cohort_admits_four_without_calibration():
    sched = _cohort_scheduler()
    sched._prefill_workspace_calibrated = False
    requests = [_make_request(str(i), 3) for i in range(4)]
    sched.waiting.extend(requests)

    assert sched._schedule_waiting() == requests
    assert len(sched.running) == 4
    assert not sched._prefill_workspace_calibrated
    assert sched._begin_prefill_memory_sample() is None


@pytest.mark.parametrize("step_size", [10, 512])
@pytest.mark.parametrize(
    "case", ["short", "long_row", "cached_context", "boundary", "progressed", "cap"]
)
def test_issue5021_cold_short_exemption_preserves_seals_and_cap(step_size, case):
    sched = _cohort_scheduler()
    sched.config.prefill_step_size = step_size
    sched._prefill_workspace_calibrated = False
    minimum = min(256, step_size)
    row = _make_request("existing", minimum + 1 if case == "long_row" else 3)
    candidate = _make_request(
        "candidate",
        minimum + 1
        if case == "cached_context"
        else minimum
        if case == "boundary"
        else 3,
    )
    if case == "cached_context":
        candidate.cached_tokens = minimum  # one uncached token, long attention context
    sched.running[row.request_id] = row
    sched._prefill_memory_rows = lambda: [(row, 1 if case == "progressed" else 0)]
    sched._resolve_metal_cap_bytes = lambda: 1 if case == "cap" else 10**9
    sched._reclaim_prefix_cache_for_prefill = MagicMock()

    assert sched._prefill_cohort_fits(candidate) == (case == "short")
    assert not sched._prefill_workspace_calibrated
    if case in {"long_row", "cached_context", "boundary", "progressed"}:
        sched._reclaim_prefix_cache_for_prefill.assert_not_called()
    else:
        sched._reclaim_prefix_cache_for_prefill.assert_called_once()


def test_prefill_uncalibrated_cohort_is_narrow_and_high_water_only_rises():
    sched = _cohort_scheduler()
    del sched._prefill_workspace_high_water
    del sched._prefill_workspace_calibrated
    sched.waiting.extend(_make_request(str(i), 10) for i in range(8))
    assert len(sched._schedule_waiting()) == 1
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory") as reset:
        before = sched._begin_prefill_memory_sample()
        reset.assert_called_once()
    assert before is not None
    assert before[:2] == (400, [(next(iter(sched.running.values())), 10, 10)])
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=500):
        sched._finish_prefill_memory_sample(before)
    assert sched._prefill_workspace_high_water == 1
    assert sched._schedule_waiting() == []  # one chunk cannot widen the cohort
    req = next(iter(sched.running.values()))
    sched._process_batch_responses(
        [
            SimpleNamespace(
                uid=req.batch_uid,
                token=1,
                finish_reason="stop",
                logprobs=None,
                prompt_cache=None,
                all_tokens=req.prompt_token_ids,
            )
        ]
    )
    assert sched._prefill_workspace_calibrated
    assert not req._prefill_memory_sampled
    sched.running.clear()
    sched.batch_generator._unprocessed_sequences.clear()
    assert len(sched._schedule_waiting()) == 4
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=900):
        sched._finish_prefill_memory_sample(before)
    assert sched._prefill_workspace_high_water == 5
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=450):
        sched._finish_prefill_memory_sample(before)
    sched._finish_prefill_memory_sample(None)
    assert sched._prefill_workspace_high_water == 5
    assert sched._schedule_waiting() == []
    sched.running.clear()
    sched.batch_generator._unprocessed_sequences.clear()
    assert sched._begin_prefill_memory_sample() is None
    assert len(sched._schedule_waiting()) == 1


def test_prefill_decode_growth_reserved_while_new_rows_wait():
    sched = _cohort_scheduler()
    r = _make_request("decode", 10)
    r.output_token_ids = [2]
    sched.running[r.request_id] = r
    sched._current_metal_active_bytes = lambda: 760
    assert not sched._prefill_cohort_fits(_make_request("new", 10))


def test_args_geometry_reuses_hybrid_estimator_and_override_wins():
    sched = _make_scheduler()
    args = SimpleNamespace(
        num_hidden_layers=4,
        num_key_value_heads=2,
        head_dim=64,
        layer_types=["full_attention", "linear_attention"] * 2,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
    )
    sched.model = SimpleNamespace(args=args)
    expected = sched._footprint_from_model_dims()
    assert expected.per_token_growth_bytes > 0
    assert expected.fixed_baseline_bytes > 0
    assert sched._resolve_kv_bytes_per_token() == expected.per_token_growth_bytes
    assert sched._resolve_kv_fixed_baseline_bytes() == expected.fixed_baseline_bytes
    sched._kv_bytes_per_token_resolved = False
    sched.config.metal_cap_kv_bytes_per_token = 777
    assert sched._resolve_kv_bytes_per_token() == 777
    assert sched._resolve_kv_fixed_baseline_bytes() == 0


def test_args_missing_dims_preserves_zero_geometry():
    sched = _make_scheduler()
    sched.model = SimpleNamespace(args=SimpleNamespace())
    assert sched._resolve_kv_bytes_per_token() == 0


def test_prefill_cohort_reclaims_prefixes_before_deferring_without_pressure_discount():
    sched = _cohort_scheduler()
    active = [900]
    sched._current_metal_active_bytes = lambda: active[0]
    sched.memory_aware_cache = SimpleNamespace(_entries={1: object()})
    sched._prefill_reclaim_footprint = lambda: None  # dense also reclaims for cohorts

    def evict():
        active[0] -= 300
        return True

    sched._evict_one_prefix_cache_entry = evict
    with patch("rapid_mlx.scheduler.mx.clear_cache"):
        assert sched._prefill_cohort_fits(_make_request(tokens=10))
    assert active[0] == 600
    assert sched.num_prefix_cache_pressure_evictions == 1
    # Fit below the cap but above the 90% cache-pressure watermark.
    active[0] = 750
    assert sched._prefill_cohort_fits(_make_request(tokens=10))
    assert sched.num_prefix_cache_pressure_evictions == 1


@pytest.mark.parametrize("prompt", ["raw long prompt", [1, 2, 3]])
def test_prefill_admission_projects_actual_tokens_before_rejecting(prompt):
    sched = _cohort_scheduler()
    sched._encode_prompt_string = lambda text: [1, 2]
    req = Request(
        request_id="untokenized",
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=1),
    )
    sched._enforce_metal_cap_at_admission(req)
    assert req.num_prompt_tokens == (2 if isinstance(prompt, str) else 3)


def test_prefill_no_generator_and_tiny_warmup_dont_calibrate():
    sched = _cohort_scheduler()
    bg = sched.batch_generator
    sched.batch_generator = None
    assert sched._prefill_memory_rows() == []
    sched.batch_generator = bg
    req = _make_request(tokens=1)
    sched.waiting.append(req)
    sched._schedule_waiting()
    assert sched._begin_prefill_memory_sample() is None


def test_prefill_held_request_can_be_cancelled():
    sched = _cohort_scheduler()
    requests = [_make_request(str(i), 10) for i in range(8)]
    sched.waiting.extend(requests)
    sched.requests.update((r.request_id, r) for r in requests)
    assert len(sched._schedule_waiting()) == 3
    assert sched.abort_request("4")
    sched._process_pending_aborts()
    assert requests[4] not in sched.waiting
    assert [r.request_id for r in sched.waiting] == ["3", "5", "6", "7"]


def test_prefill_calibration_uses_dispatched_chunk_and_bounds_shorter_context():
    sched = _cohort_scheduler()
    sched.config.prefill_step_size = 2048
    sched.batch_generator.prefill_step_size = 512
    req = _make_request(tokens=8192)
    sched.waiting.append(req)
    sched._schedule_waiting()
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory"):
        before = sched._begin_prefill_memory_sample()
    assert before is not None
    assert before[:2] == (400, [(req, 512, 512)])
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=1424):
        sched._finish_prefill_memory_sample(before)
    # A shorter context still reserves the activation high-water per chunk.
    shorter = _make_request("shorter", 2048)
    assert sched._prefill_workspace_bytes(shorter) >= 4096
    # Growing final attention context never lowers the bound.
    assert sched._prefill_workspace_bytes(req) >= sched._prefill_workspace_bytes(
        shorter
    )


def test_prefill_heterogeneous_rows_reserve_padded_workspace():
    sched = _cohort_scheduler()
    long = _make_request("long", 100)
    long.cached_tokens = 99
    short = _make_request("short", 10)
    # Long prefix-hit row needs only a one-token chunk; adding the ten-token
    # short row pads the forward. Both must price ten tokens at context 100.
    sched.waiting.append(long)
    assert sched._schedule_waiting() == [long]
    assert not sched._prefill_cohort_fits(short)


def test_prefill_invalid_prefix_is_priced_as_cold_before_inserting():
    sched = _cohort_scheduler()
    sched._current_metal_active_bytes = lambda: 500
    rows = [_make_request(str(i), 10) for i in range(3)]
    sched.waiting.extend(rows[:2])
    assert sched._schedule_waiting() == rows[:2]
    stale = rows[2]
    stale.prompt_cache = []
    stale.cached_tokens = 9
    stale.remaining_tokens = [9]
    sched.waiting.append(stale)
    assert sched._schedule_waiting() == []
    assert list(sched.waiting) == [stale]
    assert stale.cached_tokens == 0
    assert stale.remaining_tokens == stale.prompt_token_ids
    assert len(sched.batch_generator._unprocessed_sequences) == 2


def test_prefill_failed_cache_insert_rechecks_cold_retry_budget():
    sched = _cohort_scheduler()
    sched._current_metal_active_bytes = lambda: 450
    sched._resolve_kv_bytes_per_token = lambda: 10
    rows = [_make_request(str(i), 10) for i in range(3)]
    sched.waiting.extend(rows[:2])
    assert sched._schedule_waiting() == rows[:2]
    stale = rows[2]
    stale.prompt_cache = [object()]
    stale.cached_tokens = 9
    stale.remaining_tokens = [9]
    sched._validate_cache = lambda cache: True
    sched.block_aware_cache = MagicMock()
    stale.block_table = object()
    stale.shared_prefix_blocks = 2
    original = sched.batch_generator.insert

    def insert(tokens, **kwargs):
        if kwargs.get("caches"):
            raise ValueError("incompatible cached shapes")
        return original(tokens, **kwargs)

    sched.batch_generator.insert = insert
    sched.waiting.append(stale)
    assert sched._schedule_waiting() == []
    assert list(sched.waiting) == [stale]
    assert stale.cached_tokens == 0
    assert stale.block_table is None
    assert stale.shared_prefix_blocks == 0
    sched.block_aware_cache.release_cache.assert_called_once_with(stale.request_id)
    assert len(sched.batch_generator._unprocessed_sequences) == 2


def test_prefill_deferral_restores_position_after_shared_prefix_follower_bypass():
    sched = _cohort_scheduler()
    sched.waiting.extend(_make_request(str(i), 10) for i in range(3))
    assert len(sched._schedule_waiting()) == 3
    head, candidate = _make_request("follower", 10), _make_request("candidate", 10)
    sched.waiting.extend([head, candidate])

    def select_unblocked():
        sched.waiting.remove(candidate)
        return candidate

    sched._pop_waiting_for_admission = select_unblocked
    assert sched._schedule_waiting() == []
    assert list(sched.waiting) == [head, candidate]


def test_prefill_sample_preserves_process_cumulative_peak(monkeypatch):
    from rapid_mlx import memory_peak

    sched = _cohort_scheduler()
    sched.waiting.append(_make_request("sample", 10))
    sched._schedule_waiting()
    peak = [9_000_000_000]
    monkeypatch.setattr(memory_peak, "_peak_carry", 0)
    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: peak[0])
    monkeypatch.setattr(
        memory_peak.mx, "reset_peak_memory", lambda: peak.__setitem__(0, 400)
    )
    assert memory_peak.cumulative_peak_bytes() == 9_000_000_000
    before = sched._begin_prefill_memory_sample()
    peak[0] = 500
    sched._finish_prefill_memory_sample(before)
    assert sched._prefill_workspace_high_water == 1.5
    assert memory_peak.cumulative_peak_bytes() == 9_000_000_000
    assert sched.get_stats()["metal_peak_memory_gb"] == 9.0
    peak[0] = 12_000_000_000
    assert memory_peak.cumulative_peak_bytes() == 12_000_000_000
    sched._begin_prefill_memory_sample()
    assert memory_peak.cumulative_peak_bytes() == 12_000_000_000


@pytest.mark.parametrize(
    "generator",
    [
        object(),
        BatchGenerator.__new__(BatchGenerator),
        SimpleNamespace(
            _prompt_batch=SimpleNamespace(uids=[101]), _currently_processing=[(101, 3)]
        ),
    ],
)
def test_prefill_rows_tolerate_generators_without_private_queues(generator):
    sched = _cohort_scheduler()
    request = _make_request("live", 10)
    request.batch_uid = 101
    sched.running[request.request_id] = request
    sched.batch_generator = generator
    expected = [(request, 3)] if hasattr(generator, "_prompt_batch") else []
    assert sched._prefill_memory_rows() == expected


def test_optimization_status_keeps_cumulative_peak_after_sample(monkeypatch):
    from rapid_mlx import memory_peak, optimizations

    peak = [9_000_000_000]
    monkeypatch.setattr(memory_peak, "_peak_carry", 0)
    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: peak[0])
    monkeypatch.setattr(
        memory_peak.mx, "reset_peak_memory", lambda: peak.__setitem__(0, 1_000_000_000)
    )
    memory_peak.reset_peak_for_sample()
    assert (
        optimizations.get_optimization_status()["mlx_memory"]["peak_bytes"]
        == 9_000_000_000
    )


def test_periodic_memory_log_keeps_cumulative_peak_after_sample(monkeypatch, caplog):
    from rapid_mlx import memory_peak

    sched = _make_scheduler()
    sched._memory_log_interval = 1
    peak = [9_000_000_000]
    monkeypatch.setattr(memory_peak, "_peak_carry", 0)
    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: peak[0])
    monkeypatch.setattr(
        memory_peak.mx, "reset_peak_memory", lambda: peak.__setitem__(0, 1_000_000_000)
    )
    monkeypatch.setattr(memory_peak.mx.metal, "is_available", lambda: True)
    memory_peak.reset_peak_for_sample()
    with caplog.at_level(logging.INFO):
        sched.step()
    assert any(
        "[Metal memory]" in r.message and "peak=9.0GB" in r.message
        for r in caplog.records
    )


def test_mllm_stats_keep_cumulative_peak_after_prefill_sample(monkeypatch):
    from rapid_mlx import memory_peak
    from rapid_mlx.mllm_scheduler import MLLMScheduler

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.running = {}
    scheduler.num_requests_processed = 0
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.performance = SimpleNamespace(snapshot=lambda: SimpleNamespace())
    scheduler.waiting = []
    scheduler._aborted_queue_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.num_requests_cancelled = 0
    scheduler.num_requests_cancelled_via_disconnect = 0
    scheduler.batch_generator = None
    scheduler.vision_cache = None
    peak = [9_000_000_000]
    monkeypatch.setattr(memory_peak, "_peak_carry", 0)
    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: peak[0])
    monkeypatch.setattr(
        memory_peak.mx, "reset_peak_memory", lambda: peak.__setitem__(0, 1_000_000_000)
    )
    monkeypatch.setattr(memory_peak.mx.metal, "is_available", lambda: True)
    memory_peak.reset_peak_for_sample()
    assert scheduler.get_stats()["metal_peak_memory_gb"] == 9.0


@pytest.mark.parametrize("failure", ["nontrimmable", "inspection", "trim"])
def test_review_exact_hit_cold_fallback_defers_and_releases_refs(failure):
    sched = _cohort_scheduler()
    decode = _make_request("decode", 1)
    decode.output_token_ids = [1]
    sched.running[decode.request_id] = decode
    req = _make_request("exact", 100)
    req.prompt_cache = [object()]
    req.cached_tokens = 100
    req.remaining_tokens = []
    req.block_table = object()
    req.shared_prefix_blocks = 3
    sched._validate_cache = lambda cache: True
    sched.block_aware_cache = MagicMock()
    sched.waiting.append(req)
    with (
        patch(
            "mlx_lm.models.cache.can_trim_prompt_cache",
            return_value=failure != "nontrimmable",
            side_effect=ValueError("inspect") if failure == "inspection" else None,
        ),
        patch(
            "mlx_lm.models.cache.trim_prompt_cache",
            side_effect=ValueError("trim") if failure == "trim" else None,
        ),
    ):
        assert sched._schedule_waiting() == []
    assert list(sched.waiting) == [req]
    assert req.prompt_cache is None
    assert req.cached_tokens == 0
    assert req.block_table is None
    assert req.shared_prefix_blocks == 0
    sched.block_aware_cache.release_cache.assert_called_once_with("exact")
    assert not sched.batch_generator._unprocessed_sequences
    assert not sched._hybrid_checkpoints
    # Cold retry proceeds normally once the decoding row drains.
    sched.running.clear()
    assert sched._schedule_waiting() == [req]


@pytest.mark.parametrize("interruption", ["cancel", "abort", "error"])
def test_review_interrupted_sample_does_not_qualify_calibration(interruption):
    sched = _cohort_scheduler()
    del sched._prefill_workspace_calibrated
    sched.waiting.append(_make_request("sample", 10))
    req = sched._schedule_waiting()[0]
    sched.requests[req.request_id] = req
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory"):
        before = sched._begin_prefill_memory_sample()
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=500):
        sched._finish_prefill_memory_sample(before)
    if interruption == "error":
        with patch("rapid_mlx.scheduler.mx.clear_cache"):
            sched._recover_from_generation_error()
        # Simulate the newly created compatible generator.
        sched.batch_generator = _cohort_scheduler().batch_generator
    else:
        sched.batch_generator.remove = lambda uids: None
        if interruption == "cancel":
            sched.abort_request(req.request_id)
            sched._process_pending_aborts()
        else:
            sched._do_abort_request(req.request_id)
        sched.batch_generator._unprocessed_sequences.clear()
    sched.waiting.extend(_make_request(str(i), 10) for i in range(3))
    assert len(sched._schedule_waiting()) == 1
    assert not getattr(sched, "_prefill_workspace_calibrated", False)
    assert not getattr(req, "_prefill_memory_sampled", False)


def test_review_pflash_physical_kv_reservation_preserves_usage_and_add_gate():
    sched = _cohort_scheduler()
    del sched._estimate_request_kv_bytes
    sched._resolve_kv_bytes_per_token = lambda: 10
    req = _make_request("compressed", 100)
    req.prompt_token_ids = req.prompt_token_ids[:10]
    req.model_prompt_tokens = 10
    assert sched._estimate_request_kv_bytes(req) == 1010  # pre-transform add gate
    sched.waiting.append(req)
    assert sched._sum_in_flight_kv_bytes() == 1010
    sched.waiting.clear()
    assert sched._prefill_unmaterialized_bytes(req, 0) == 110
    req.cached_tokens = 3
    assert sched._prefill_unmaterialized_bytes(req, 2) == 60
    req.cached_tokens = 0
    req.output_token_ids = [1]
    sched.running[req.request_id] = req
    assert sched._prefill_unmaterialized_bytes(req, 0) == 0
    assert sched._prefill_cohort_fits(_make_request("candidate", 1))
    assert req.num_prompt_tokens == 100


def test_review_abort_response_cannot_qualify_sample():
    sched = _cohort_scheduler()
    del sched._prefill_workspace_calibrated
    req = _make_request("aborted-response", 10)
    sched.waiting.append(req)
    sched._schedule_waiting()
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory"):
        before = sched._begin_prefill_memory_sample()
    sched._finish_prefill_memory_sample(before)
    sched._process_batch_responses(
        [
            SimpleNamespace(
                uid=req.batch_uid,
                token=1,
                finish_reason="abort",
                logprobs=None,
                prompt_cache=None,
                all_tokens=req.prompt_token_ids,
            )
        ]
    )
    assert not getattr(sched, "_prefill_workspace_calibrated", False)
    assert not req._prefill_memory_sampled
    assert req.status == RequestStatus.FINISHED_ABORTED


@pytest.fixture
def review2_generator(monkeypatch):
    import importlib

    import mlx.core as mx

    # Earlier tests in the same worker may leave MLX's default stream and
    # mlx-lm's generation stream bound to another thread's stream. Rebind both
    # to a stream owned by this thread, restoring the default afterwards.
    previous_default = mx.default_stream(mx.default_device())
    current_stream = mx.new_stream(mx.default_device())
    mx.set_default_stream(current_stream)
    monkeypatch.setattr(
        importlib.import_module("mlx_lm.generate"),
        "generation_stream",
        current_stream,
    )

    class Model:
        def __call__(self, tokens, cache):
            return mx.zeros((*tokens.shape, 4))

    try:
        with patch("mlx.core.metal.is_available", return_value=False):
            generator = BatchGenerator(Model(), prefill_step_size=2048)
        yield generator
        generator.close()
    finally:
        mx.set_default_stream(previous_default)


@pytest.mark.parametrize(
    "lengths, qualifies", [([1, 254, 1], False), ([1, 256, 1], True)]
)
def test_review2_segmented_sample_requires_representative_dispatch(
    review2_generator, lengths, qualifies
):
    sched = _cohort_scheduler()
    del sched._prefill_workspace_calibrated
    sched.config.prefill_step_size = 2048
    sched.batch_generator = bg = review2_generator
    del sched._prefill_workspace_high_water
    req = _make_request("segmented", sum(lengths))
    segments = [[0] * n for n in lengths]
    req.batch_uid = bg.insert_segments([segments], caches=[[]], max_tokens=[1])[0]
    sched.running[req.request_id] = req
    sched.uid_to_request_id[req.batch_uid] = req.request_id
    # The initial one-token segment must not reset peaks or qualify a sample.
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory") as reset:
        before = sched._begin_prefill_memory_sample()
    assert before is None
    reset.assert_not_called()
    bg.next()
    # Subsequent sampling must see the active segment, rather than total suffix.
    before = sched._begin_prefill_memory_sample()
    assert (before is not None) == qualifies
    bg.next()
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=1424):
        sched._finish_prefill_memory_sample(before)
    for _ in range(2):
        _, responses = bg.next()
        sched._process_batch_responses(responses)
    assert bool(getattr(sched, "_prefill_workspace_calibrated", False)) == qualifies
    if qualifies:
        assert sched._prefill_workspace_per_chunk == 4  # 1024 / 256
        assert sched._prefill_workspace_high_water == 1024 / (256 * sum(lengths))


def test_review2_generation_batch_prompt_kv_already_resident(review2_generator):
    sched = _cohort_scheduler()
    del sched._estimate_request_kv_bytes
    sched._resolve_kv_bytes_per_token = lambda: 10
    sched.batch_generator = bg = review2_generator
    req = _make_request("promoted", 40)
    req.batch_uid = bg.insert([req.prompt_token_ids], caches=[[]], max_tokens=[1])[0]
    sched.running[req.request_id] = req
    bg.next()  # prefill 39 tokens
    _, responses = bg.next()  # final prompt token; first output arrives next step
    assert not responses and not req.num_output_tokens
    assert req.batch_uid in bg._generation_batch.uids
    assert sched._prefill_memory_rows() == []
    assert sched._prefill_unmaterialized_bytes(req, 0) == 10
    # Resident KV must not force cache eviction or defer a fitting candidate.
    sched._prefill_workspace_high_water = 0
    sched._resolve_metal_cap_bytes = lambda: 500
    sched._reclaim_prefix_cache_for_prefill = MagicMock()
    assert sched._prefill_cohort_fits(_make_request("candidate", 1))
    assert sched._reclaim_prefix_cache_for_prefill.call_args.kwargs["cohort_need"] < 100


@pytest.mark.parametrize("dispatch", ["tiny", "large", "both", "none"])
def test_review2_sample_counts_only_dispatched_rows_and_marks_large_rows(dispatch):
    sched = _cohort_scheduler()
    del sched._prefill_workspace_high_water
    del sched._prefill_workspace_calibrated
    sched.config.prefill_step_size = 2048
    bg = sched.batch_generator
    bg.prefill_step_size = 2048
    tiny = _make_request("tiny", 400)
    large = _make_request("large", 800)
    tiny.batch_uid, large.batch_uid = 101, 102
    sched.running.update((r.request_id, r) for r in (tiny, large))
    # One active segment plus an inserted row which may remain queued.
    bg._prompt_batch.uids = [101]
    bg._currently_processing = [[[[0], [0] * 399], 0, 400]]
    bg._unprocessed_sequences = [(102, [[0] * 512, [0] * 288])]
    before = sched._begin_prefill_memory_sample()
    assert before is not None
    assert before[:2] == (400, [(tiny, 1, 1), (large, 512, 512)])
    assert not getattr(large, "_prefill_memory_sampled", False)
    uids = {"tiny": [101], "large": [102], "both": [101, 102], "none": []}[dispatch]
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=1424):
        sched._finish_prefill_memory_sample(
            before, ([SimpleNamespace(uid=uid) for uid in uids], [])
        )
    assert not getattr(tiny, "_prefill_memory_sampled", False)
    if dispatch in ("tiny", "none"):
        assert not hasattr(sched, "_prefill_workspace_high_water")
        assert not getattr(large, "_prefill_memory_sampled", False)
    else:
        assert large._prefill_memory_sampled
        chunks = 513 if dispatch == "both" else 512
        units = 512 * 800 + (400 if dispatch == "both" else 0)
        assert sched._prefill_workspace_high_water == 1024 / units
        assert sched._prefill_workspace_per_chunk == 1024 / chunks


def test_review2_failed_sample_cannot_mark_a_trajectory():
    sched = _cohort_scheduler()
    sched.waiting.append(_make_request("sample", 10))
    req = sched._schedule_waiting()[0]
    before = sched._begin_prefill_memory_sample()
    with (
        patch("rapid_mlx.scheduler.mx.get_peak_memory", side_effect=RuntimeError),
        pytest.raises(RuntimeError),
    ):
        sched._finish_prefill_memory_sample(before)
    assert not getattr(req, "_prefill_memory_sampled", False)


@pytest.mark.parametrize("timing", [False, True])
def test_review2_step_samples_generator_dispatch_with_timing(review2_generator, timing):
    sched = _cohort_scheduler()
    sched._step_timing_enabled = timing
    sched.config.prefill_step_size = 2048
    sched.batch_generator = bg = review2_generator
    req = _make_request("step-sample", 512)
    req.batch_uid = bg.insert([req.prompt_token_ids], caches=[[]], max_tokens=[1])[0]
    sched.running[req.request_id] = req
    sched.requests[req.request_id] = req
    sched.uid_to_request_id[req.batch_uid] = req.request_id
    with patch.object(
        sched,
        "_finish_prefill_memory_sample",
        wraps=sched._finish_prefill_memory_sample,
    ) as finish:
        for _ in range(2):
            output = sched.step()
            assert not output.finished_request_ids
        assert finish.call_count == 2
        before, raw_next = finish.call_args_list[0].args
        assert before[1] == [(req, 511, 511)]
        assert raw_next[0][0].uid == req.batch_uid
    assert req._prefill_memory_sampled


@pytest.mark.parametrize("config", [SimpleNamespace(), {}])
def test_incomplete_config_falls_back_to_integer_args_geometry(config):
    sched = _make_scheduler()
    sched.model = SimpleNamespace(
        config=config,
        args=SimpleNamespace(num_hidden_layers=4, num_key_value_heads=2, head_dim=64),
    )
    assert sched._resolve_kv_bytes_per_token() == 2 * 4 * 2 * 64 * 4


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(config=SimpleNamespace()),
        SimpleNamespace(config=SimpleNamespace(), args=SimpleNamespace()),
        SimpleNamespace(config=SimpleNamespace(), args=MagicMock()),
        MagicMock(),
    ],
)
def test_incomplete_geometry_does_not_fabricate_kv_bytes(model):
    sched = _make_scheduler()
    sched.model = model
    assert sched._footprint_from_model_dims() is None
    assert sched._resolve_kv_bytes_per_token() == 0


def test_peak_window_token_is_invalidated_by_each_reset(monkeypatch):
    from rapid_mlx import memory_peak

    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: 400)
    monkeypatch.setattr(memory_peak.mx, "reset_peak_memory", lambda: None)
    first = memory_peak.reset_peak_for_sample()
    assert memory_peak.sample_window_intact(first)
    second = memory_peak.reset_peak_for_sample()
    assert second > first
    assert not memory_peak.sample_window_intact(first)
    assert memory_peak.sample_window_intact(second)
    third = memory_peak.reset_peak_for_sample()
    assert third > second
    assert not memory_peak.sample_window_intact(second)
    assert memory_peak.sample_window_intact(third)


@pytest.mark.parametrize("interruption", [None, "before_read", "during_read"])
def test_prefill_peak_window_controls_high_water_and_calibration(
    monkeypatch, interruption
):
    from rapid_mlx import memory_peak

    sched = _cohort_scheduler()
    sched._prefill_workspace_calibrated = False
    sched._prefill_workspace_per_chunk = 1.0
    sched.waiting.append(_make_request("window", 10))
    req = sched._schedule_waiting()[0]
    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: 400)
    monkeypatch.setattr(memory_peak.mx, "reset_peak_memory", lambda: None)
    before = sched._begin_prefill_memory_sample()
    assert before is not None
    if interruption == "before_read":
        memory_peak.reset_peak_for_sample()

    def read_peak():
        if interruption == "during_read":
            # A second scheduler resets while the first reads its final peak.
            monkeypatch.setattr(memory_peak.mx, "get_peak_memory", lambda: 400)
            memory_peak.reset_peak_for_sample()
        return 900

    monkeypatch.setattr(memory_peak.mx, "get_peak_memory", read_peak)
    sched._finish_prefill_memory_sample(before)
    intact = interruption is None
    assert sched._prefill_workspace_high_water == (5 if intact else 1.5)
    assert sched._prefill_workspace_per_chunk == (50 if intact else 1.0)
    assert bool(getattr(req, "_prefill_memory_sampled", False)) == intact
    assert not sched._prefill_workspace_calibrated
    sched._process_batch_responses(
        [
            SimpleNamespace(
                uid=req.batch_uid,
                token=1,
                finish_reason="stop",
                logprobs=None,
                prompt_cache=None,
                all_tokens=req.prompt_token_ids,
            )
        ]
    )
    assert sched._prefill_workspace_calibrated == intact
    assert not getattr(req, "_prefill_memory_sampled", False)


def _workspace_scheduler(heads=16, kv_bytes=20480):
    sched = _cohort_scheduler()
    sched.model = SimpleNamespace(
        config={
            "num_hidden_layers": 40,
            "layer_types": ["full_attention"] * 10 + ["linear_attention"] * 30,
            "num_attention_heads": heads,
            "num_key_value_heads": 2,
            "head_dim": 256,
        }
    )
    sched.config.prefill_step_size = sched.batch_generator.prefill_step_size = 2048
    sched._resolve_kv_bytes_per_token = lambda: kv_bytes
    sched._estimate_request_kv_bytes = lambda r, **kw: (
        kv_bytes * len(r.prompt_token_ids)
    )
    sched._resolve_metal_cap_bytes = lambda: 215_000_000_000
    sched._current_metal_active_bytes = lambda: 25_000_000_000
    sched._reclaim_prefix_cache_for_prefill = MagicMock()
    del sched._prefill_workspace_high_water
    return sched


def _workspace_sample(sched, length, chunk, delta, *, cached=0, processed=None):
    req = _make_request("sample", length)
    req.cached_tokens = cached
    req.batch_uid = 101
    if processed is None:
        processed = length - cached - chunk
    sched.running = {req.request_id: req}
    bg = sched.batch_generator
    bg._unprocessed_sequences = []
    bg._prompt_batch.uids = [101]
    bg._currently_processing = [[[[0] * chunk], processed, length - cached]]
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory"):
        before = sched._begin_prefill_memory_sample()
    assert before is not None
    assert before[1] == [(req, chunk, cached + processed + chunk)]
    with patch(
        "rapid_mlx.scheduler.mx.get_peak_memory",
        return_value=25_000_000_000 + delta,
    ):
        sched._finish_prefill_memory_sample(before, ([SimpleNamespace(uid=101)], []))
    return req


def test_issue4824_hybrid_long_prompts_can_share_a_cohort():
    sched = _workspace_scheduler()
    _workspace_sample(sched, 512, 512, 881_000_000)
    _workspace_sample(sched, 32768, 2048, 3_281_000_000)
    row = _make_request("first", 32768)
    units = 2048 * 32768
    # Reproduction: short-prompt delta / (chunk * final context), extrapolated
    # to 32k. These exact inputs yield 225.536 GB (the same ~150 GB failure
    # scale), despite the measured 32k workspace being only 3.281 GB.
    legacy = int(sched._prefill_workspace_high_water * units)
    assert legacy == 225_536_000_000
    workspace = sched._prefill_workspace_bytes(row)
    kv = sched._estimate_request_kv_bytes(row)
    assert workspace == int(2048 * (881_000_000 / 512) * 1.25)
    assert kv == 671_088_640
    assert 3_281_000_000 <= workspace <= 1.5 * (3_281_000_000 + kv)
    total = 25_000_000_000 + 2 * (workspace + kv)
    assert total == 35_152_177_280 < 215_000_000_000
    bg = sched.batch_generator
    bg._prompt_batch.uids = []
    bg._currently_processing = []
    row.batch_uid = 102
    bg._unprocessed_sequences = [(102, [row.prompt_token_ids])]
    sched.running = {row.request_id: row}
    assert sched._prefill_cohort_fits(_make_request("second", 32768))
    assert sched._reclaim_prefix_cache_for_prefill.call_args.kwargs["cohort_need"] == (
        total - 25_000_000_000
    )


def test_issue4824_live_49k_prefill_need_covers_admission_peak():
    sched = _workspace_scheduler()
    _workspace_sample(sched, 49152, 2048, 2048 * 2.472e6)
    row = _make_request(tokens=49359)
    workspace = sched._prefill_workspace_bytes(row)
    assert workspace == int(2048 * (2.472e6 + 16 * 4 * 8 * (49359 - 49152)) * 1.25)
    need = sched._prefill_unmaterialized_bytes(row, 0) + workspace
    assert need == 20480 * 49359 + workspace
    assert 6.29e9 <= need <= 1.5 * 6.29e9


def test_issue4824_context_extrapolation_and_padding():
    sched = _workspace_scheduler()
    row = _make_request(tokens=32768)
    prior = 3.5 * 20480 / 2048 * (2048 * 32768)
    assert sched._prefill_workspace_bytes(row) == prior  # no samples
    _workspace_sample(sched, 512, 512, 881_000_000)
    assert sched._prefill_workspace_max_context == 512
    rate = 881_000_000 / 512
    expected = int(2048 * (rate + 16 * 4 * 8 * (32768 - 512)) * 1.25)
    assert sched._prefill_workspace_bytes(row) == expected >= 2048 * rate
    shorter = _make_request("padded", 256)
    assert (
        sched._prefill_workspace_bytes(
            shorter, padded_units=2048 * 32768, padded_chunk=2048
        )
        == expected
    )
    _workspace_sample(sched, 32768, 2048, 3_281_000_000)
    assert sched._prefill_workspace_max_context == 32768
    assert sched._prefill_workspace_bytes(row) >= 3_281_000_000
    _workspace_sample(sched, 512, 512, 100)
    assert sched._prefill_workspace_max_context == 32768
    assert sched._prefill_workspace_bytes(shorter) == int(256 * rate * 1.25)


def test_issue4824_dense_attention_preserves_growth_prior_and_large_samples():
    sched = _workspace_scheduler(32, 147456)
    sched.model.config = {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    _workspace_sample(sched, 8192, 2048, 4_000_000_000)
    row = _make_request(tokens=32768)
    growth = int(2048 * (4_000_000_000 / 2048 + 32 * 4 * 8 * (32768 - 8192)) * 1.25)
    prior = 3.5 * 147456 / 2048 * (2048 * 32768)
    assert sched._prefill_workspace_bytes(row) == max(growth, prior)
    assert sched._prefill_workspace_bytes(row) >= growth
    assert sched._prefill_workspace_bytes(row) >= prior
    _workspace_sample(sched, 32768, 2048, 30_000_000_000)
    assert sched._prefill_workspace_bytes(row) >= 30_000_000_000


@pytest.mark.parametrize("heads", [None, 0, -1, 16.0, True])
def test_issue4824_unknown_heads_preserve_legacy_formula(heads):
    sched = _workspace_scheduler(heads)
    _workspace_sample(sched, 512, 512, 881_000_000)
    row = _make_request(tokens=32768)
    for padded_units, padded_chunk in [(0, 0), (4096 * 65536, 4096)]:
        units = max(padded_units, sched._prefill_workspace_units(row))
        expected = int(
            max(
                max(3.5 * 20480 / 2048, sched._prefill_workspace_high_water) * units,
                sched._prefill_workspace_per_chunk * max(2048, padded_chunk),
            )
        )
        assert (
            sched._prefill_workspace_bytes(
                row, padded_units=padded_units, padded_chunk=padded_chunk
            )
            == expected
        )


def test_issue4824_sample_context_uses_cached_processed_and_dispatched_chunk():
    sched = _workspace_scheduler()
    _workspace_sample(sched, 32768, 2048, 3_281_000_000, cached=4096, processed=1024)
    assert sched._prefill_workspace_max_context == 7168  # not final prompt length


@pytest.mark.parametrize("holder", ["config", "args"])
@pytest.mark.parametrize("structural", [False, True])
def test_issue4824_attention_heads_follow_text_tower_and_are_cached(holder, structural):
    sched = _workspace_scheduler()
    text = {"num_attention_heads": 16}
    if structural:
        text.update(num_hidden_layers=2, layer_types=["full_attention"] * 2)
    sched.model = SimpleNamespace(
        **{holder: {"num_attention_heads": 100, "text_config": text}}
    )
    assert sched._resolve_prefill_attention_beta() == 512
    text["num_attention_heads"] = 32
    assert sched._resolve_prefill_attention_beta() == 512


@pytest.mark.parametrize("dispatch_both", [False, True])
def test_issue4824_sample_anchor_is_minimum_dispatched_context(dispatch_both):
    sched = _workspace_scheduler()
    long = _make_request("long", 32768)
    short = _make_request("short", 8192)
    long.batch_uid, short.batch_uid = 101, 102
    long.cached_tokens = 4096
    sched.running = {r.request_id: r for r in (long, short)}
    bg = sched.batch_generator
    bg._prompt_batch.uids = [101, 102]
    bg._currently_processing = [
        [[long.prompt_token_ids], 1024, 32768 - 4096],
        [[short.prompt_token_ids], 256, 8192],
    ]
    bg._unprocessed_sequences = []
    with patch("rapid_mlx.scheduler.mx.reset_peak_memory"):
        before = sched._begin_prefill_memory_sample()
    assert before[1] == [(long, 2048, 7168), (short, 2048, 2304)]
    dispatched = [101, 102] if dispatch_both else [101]
    with patch("rapid_mlx.scheduler.mx.get_peak_memory", return_value=28_000_000_000):
        sched._finish_prefill_memory_sample(
            before, ([SimpleNamespace(uid=uid) for uid in dispatched], [])
        )
    assert sched._prefill_workspace_max_context == (2304 if dispatch_both else 7168)


def test_issue4824_outer_structural_config_wins_over_decoy_text_heads():
    sched = _workspace_scheduler()
    sched.model.config["text_config"] = {"num_attention_heads": 100}
    assert sched._resolve_prefill_attention_beta() == 512


def test_issue4824_non_fused_softcapped_attention_is_not_under_counted():
    # Measured on hardware: a softcapped (non-fused attention) model with 8
    # heads, chunk 256, calibrated on a 512-token prompt, peaked 544.3 MB per
    # 8k row. The extrapolation must cover it.
    sched = _workspace_scheduler(heads=8, kv_bytes=4096)
    sched.config.prefill_step_size = sched.batch_generator.prefill_step_size = 256
    _workspace_sample(sched, 512, 256, 85_000_000)
    row = _make_request(tokens=8192)
    assert sched._prefill_workspace_bytes(row) >= 544_300_000
