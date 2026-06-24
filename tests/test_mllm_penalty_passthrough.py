# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #512 — MLLM scheduler penalty passthrough.

Before the fix, ``MLLMScheduler.add_request`` constructed ``SamplingParams``
with only ``max_tokens`` / ``temperature`` / ``top_p``, so
``repetition_penalty`` / ``presence_penalty`` / ``frequency_penalty`` were
silently dropped from every OpenAI request routed to a vision model. The
LLM scheduler wires these via ``mlx_lm.sample_utils.make_logits_processors``
(see ``scheduler.py:4147``) — the MLLM path had no analogue.

This module pins behaviour at three layers:

  1. Logits-processor application — ``_maybe_apply_penalty_processors``
     mutates the row for non-neutral knobs and is a true no-op for the
     neutral defaults (allocation-free fast path preserved).
  2. Scheduler plumbing — ``MLLMScheduler.add_request`` reads the three
     penalty kwargs and stamps them onto ``SamplingParams`` (and from there
     onto ``MLLMBatchRequest`` at schedule time).
  3. Engine plumbing — ``BatchedEngine`` MLLM branch in ``generate()`` /
     ``stream_generate()`` forwards the three keys out of ``**kwargs`` into
     the scheduler (mirroring the LLM branch's ``_sp_kwargs`` block).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_mlx.mllm_batch_generator import (
    MLLMBatchRequest,
    _maybe_apply_penalty_processors,
)


# =============================================================================
# Layer 1 — logits-processor application
# =============================================================================


def _make_req(**overrides) -> MLLMBatchRequest:
    kwargs = dict(uid=0, request_id="r0", prompt="hi")
    kwargs.update(overrides)
    return MLLMBatchRequest(**kwargs)


def test_neutral_defaults_skip_processor_allocation():
    """All-neutral knobs must take the early-return fast path with zero
    allocation: no processor list cached on the request and the logits
    tensor returned unchanged. Pre-#512 every MLLM step paid this cost
    once allocation was added defensively; we keep the fast path."""
    req = _make_req()  # all penalties at neutral defaults
    row = mx.ones((1, 8))
    out = _maybe_apply_penalty_processors(req, row)
    assert out is row, "neutral knobs must return the input row unchanged"
    assert not hasattr(req, "_cached_penalty_processors"), (
        "neutral defaults must not allocate processor cache"
    )


def test_repetition_penalty_suppresses_already_seen_tokens():
    """``repetition_penalty=2.0`` divides logits of tokens that already
    appear in ``output_tokens`` by 2 (mlx-lm semantics for positive logits).
    Pre-fix this knob never reached the VLM sampler."""
    req = _make_req(repetition_penalty=2.0)
    req.output_tokens.extend([0, 1])  # tokens "already generated"
    row = mx.array([[10.0, 10.0, 10.0]])  # row of three positive logits
    out = _maybe_apply_penalty_processors(req, row)
    vals = out.tolist()[0]
    # Seen tokens (0, 1) penalised; unseen token (2) untouched.
    assert vals[0] == pytest.approx(5.0), "seen token 0 should be /2"
    assert vals[1] == pytest.approx(5.0), "seen token 1 should be /2"
    assert vals[2] == pytest.approx(10.0), "unseen token must be unchanged"


def test_presence_penalty_subtracts_constant_from_seen_tokens():
    """``presence_penalty=0.5`` subtracts 0.5 from every token that has
    appeared in the history. Confirms the OpenAI-spec wiring (additive,
    not multiplicative — the multiplicative one is repetition_penalty)."""
    req = _make_req(presence_penalty=0.5)
    req.output_tokens.append(2)
    row = mx.array([[1.0, 1.0, 1.0]])
    out = _maybe_apply_penalty_processors(req, row)
    vals = out.tolist()[0]
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(1.0)
    assert vals[2] == pytest.approx(0.5)


def test_frequency_penalty_scales_with_occurrence_count():
    """``frequency_penalty`` subtracts ``penalty`` for EACH occurrence
    of the token in the history (mlx-lm uses ``.at[].subtract`` which
    accumulates duplicates). Confirms the wiring picks the
    occurrence-counting processor, not the presence-counting one."""
    req = _make_req(frequency_penalty=0.25)
    req.output_tokens.extend([1, 1, 1])  # token 1 seen 3x
    row = mx.array([[2.0, 2.0]])
    out = _maybe_apply_penalty_processors(req, row)
    vals = out.tolist()[0]
    assert vals[0] == pytest.approx(2.0)
    assert vals[1] == pytest.approx(2.0 - 0.25 * 3)


def test_processor_cache_reused_across_steps():
    """The processor list must be memoised on the request so subsequent
    steps don't re-build it. Pre-fix would have re-allocated on every
    token; this keeps the per-token overhead at one dict-lookup."""
    req = _make_req(presence_penalty=0.5)
    req.output_tokens.append(0)
    row = mx.array([[1.0, 1.0]])
    _maybe_apply_penalty_processors(req, row)
    first_cache = req._cached_penalty_processors
    _maybe_apply_penalty_processors(req, row)
    assert req._cached_penalty_processors is first_cache, "cache must be reused"


def test_first_token_no_history_is_unchanged():
    """The first sampled token from the VLM prefill happens before any
    output_tokens accumulate; mlx-lm processors no-op on empty history.
    Confirms our gate doesn't accidentally penalise the prefill token."""
    req = _make_req(repetition_penalty=2.0, presence_penalty=0.5, frequency_penalty=0.5)
    # output_tokens is empty (first generated token, no history yet)
    row = mx.array([[3.0, 3.0]])
    out = _maybe_apply_penalty_processors(req, row)
    assert mx.array_equal(out, mx.array([[3.0, 3.0]]))


# =============================================================================
# Layer 2 — MLLMScheduler.add_request stamps SamplingParams + BatchRequest
# =============================================================================


def _stub_scheduler():
    """Construct a scheduler with all I/O dependencies stubbed out so we can
    drive ``add_request`` synchronously without booting Metal/VLM."""
    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig()
    scheduler.requests = {}
    scheduler.waiting = __import__("collections").deque()
    scheduler.running = {}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler._cancelled_request_ids = set()
    scheduler._disconnect_abort_ids = set()
    scheduler._pending_abort_ids = set()
    scheduler.num_requests_cancelled = 0
    scheduler.num_requests_cancelled_via_disconnect = 0

    import threading

    scheduler._cancel_counter_lock = threading.Lock()
    return scheduler


def test_scheduler_add_request_stamps_penalties_on_sampling_params():
    """``MLLMScheduler.add_request`` must thread the three penalty kwargs
    onto ``SamplingParams`` so ``_schedule_waiting`` can copy them onto
    ``MLLMBatchRequest``. Pre-fix the SamplingParams() constructor was
    called with only max_tokens/temperature/top_p — the three penalty
    fields stayed at their dataclass defaults regardless of caller input."""
    scheduler = _stub_scheduler()
    rid = scheduler.add_request(
        prompt="hi",
        max_tokens=8,
        repetition_penalty=2.5,
        presence_penalty=0.75,
        frequency_penalty=-0.5,
    )
    req = scheduler.requests[rid]
    assert req.sampling_params.repetition_penalty == 2.5
    assert req.sampling_params.presence_penalty == 0.75
    assert req.sampling_params.frequency_penalty == -0.5


def test_scheduler_add_request_defaults_neutral_when_omitted():
    """Callers that don't pass the penalty kwargs (everyone pre-#512) must
    keep getting the neutral defaults so ``_maybe_apply_penalty_processors``
    stays on its allocation-free fast path."""
    scheduler = _stub_scheduler()
    rid = scheduler.add_request(prompt="hi", max_tokens=8)
    req = scheduler.requests[rid]
    assert req.sampling_params.repetition_penalty == 1.0
    assert req.sampling_params.presence_penalty == 0.0
    assert req.sampling_params.frequency_penalty == 0.0


def test_scheduler_add_request_preserves_explicit_zero_values():
    """Codex r1 MAJOR #2 guard: an explicit ``repetition_penalty=0.0``
    is a legitimate (if extreme) request the API schema accepts and the
    LLM path preserves on ``SamplingParams``. The earlier
    ``kwargs.pop(...) or NEUTRAL`` pattern silently rewrote ``0.0`` to
    ``1.0`` for ``repetition_penalty`` because ``0.0 or 1.0`` is ``1.0``
    in Python. Only ``None`` (absent kwarg) may collapse to neutral."""
    scheduler = _stub_scheduler()
    rid = scheduler.add_request(
        prompt="hi",
        max_tokens=8,
        repetition_penalty=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    req = scheduler.requests[rid]
    assert req.sampling_params.repetition_penalty == 0.0, (
        "explicit repetition_penalty=0.0 must NOT be coerced to 1.0"
    )
    assert req.sampling_params.presence_penalty == 0.0
    assert req.sampling_params.frequency_penalty == 0.0


# =============================================================================
# Layer 3 — BatchedEngine MLLM branch forwards the kwargs
# =============================================================================


@pytest.mark.asyncio
async def test_engine_stream_generate_mllm_forwards_penalty_kwargs():
    """``BatchedEngine.stream_generate``'s MLLM branch must pop the three
    penalty keys from ``**kwargs`` and forward them to
    ``_mllm_scheduler.add_request_async``. Pre-fix they stayed in
    ``**kwargs`` and were silently discarded by the scheduler
    signature, so the route-layer cascade
    (``build_extended_sampling_kwargs`` → ``chat_kwargs`` →
    ``engine.stream_chat`` → ``engine.stream_generate``) bottomed out
    here for vision models."""
    from vllm_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = True
    engine._mllm_scheduler = MagicMock()
    captured: dict = {}

    async def _fake_add(**kw):
        captured.update(kw)
        return "rid"

    async def _empty_stream(*_a, **_kw):
        if False:
            yield None  # pragma: no cover

    engine._mllm_scheduler.add_request_async = _fake_add
    engine._mllm_scheduler.stream_outputs = _empty_stream

    async for _ in engine.stream_generate(
        prompt="hi",
        max_tokens=8,
        repetition_penalty=1.7,
        presence_penalty=0.3,
        frequency_penalty=0.4,
    ):
        pass

    assert captured["repetition_penalty"] == 1.7
    assert captured["presence_penalty"] == 0.3
    assert captured["frequency_penalty"] == 0.4
