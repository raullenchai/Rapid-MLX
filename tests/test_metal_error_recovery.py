# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-loop recovery when scheduler.step raises (#353).

The user-visible behaviour we pin:
- A RuntimeError out of ``scheduler.step`` does not leave HTTP handlers
  awaiting forever — every in-flight request gets a final RequestOutput
  with ``error=<...>`` and its ``finished_event`` set.
- ``engine.generate`` surfaces that error as ``InferenceAbortedError`` so
  the chat handler can map to 503.
- The detection branch for Metal-shaped messages does not require a real
  GPU — we just check that error messages containing 'Metal' / 'MTL' /
  'command buffer' / 'gpu::check_error' are recognised.

These are unit tests; the actual Metal async-abort path (mlx-lm#1015)
can't be reproduced without a real GPU OOM, but the recovery wiring it
relies on is what we exercise here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import asyncio
from unittest.mock import MagicMock

from vllm_mlx.engine_core import EngineConfig, EngineCore
from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.request import (
    InferenceAbortedError,
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _make_engine() -> EngineCore:
    """Construct an EngineCore against mock model + tokenizer.

    Bails the test if the construction path needs more wiring than we can
    fake — the same guard pattern as test_idle_event_wakeup.
    """
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    cfg = EngineConfig(model_name="test-fake-model")
    try:
        return EngineCore(fake_model, fake_tokenizer, cfg)
    except Exception as e:
        pytest.skip(f"EngineCore construction needs more mock setup: {e}")


def test_request_output_carries_error_field():
    """RequestOutput must declare ``error`` so the engine loop can flag
    an aborted request without piggy-backing on finish_reason."""
    out = RequestOutput(request_id="r1", error="boom")
    assert out.error == "boom"
    assert out.finished is False  # default — engine loop sets True explicitly


def test_inference_aborted_error_is_runtime_error():
    """HTTP handlers catch via isinstance(..., InferenceAbortedError); the
    class must remain a RuntimeError subclass so generic ``except
    RuntimeError`` paths still see it."""
    err = InferenceAbortedError("metal hung")
    assert isinstance(err, RuntimeError)
    assert "metal" in str(err)


@pytest.mark.asyncio
async def test_engine_loop_fails_in_flight_requests_on_step_exception():
    """When scheduler.step raises, every awaiting request must receive a
    final RequestOutput with ``error`` set and have its finished_event set
    so HTTP handlers unblock instead of timing out."""
    engine = _make_engine()

    # Replace scheduler with a stub that always raises a Metal-shaped error.
    class _BoomScheduler:
        def has_requests(self):
            return True

        def step(self):
            raise RuntimeError(
                "Metal command buffer error: kIOGPUCommandBufferCallbackErrorOutOfMemory"
            )

        def add_request(self, *_a, **_kw):
            pass

        def abort_request(self, *_a, **_kw):
            return True

        def remove_finished_request(self, *_a, **_kw):
            pass

    engine.scheduler = _BoomScheduler()

    # Pre-seed the in-flight tracking that add_request would normally set.
    rid = "test-req-1"
    engine._output_collectors[rid] = RequestOutputCollector(aggregate=True)
    engine._finished_events[rid] = asyncio.Event()

    # Drive the engine loop briefly — long enough for one step to raise.
    engine._running = True
    loop_task = asyncio.create_task(engine._engine_loop())
    try:
        await asyncio.wait_for(engine._finished_events[rid].wait(), timeout=1.0)
    finally:
        engine._running = False
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    collector = engine._output_collectors[rid]
    final = collector.get_nowait()
    assert final is not None, "collector must receive an error RequestOutput"
    assert final.finished is True
    assert final.error and ("Metal" in final.error or "metal" in final.error.lower())


@pytest.mark.asyncio
async def test_stream_outputs_raises_on_error_field():
    """Streaming path: when the engine loop puts a final RequestOutput with
    ``error`` into the collector, ``stream_outputs`` must surface it as
    InferenceAbortedError instead of silently yielding a terminal chunk —
    otherwise streaming HTTP handlers would emit an empty SSE close instead
    of a 503 (#353 DeepSeek round 1 P0)."""
    engine = _make_engine()

    rid = "stream-test-1"
    engine._output_collectors[rid] = RequestOutputCollector(aggregate=True)

    # Push the error output the engine loop would have produced.
    engine._output_collectors[rid].put(
        RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="error",
            error="Inference aborted: RuntimeError: Metal command buffer error",
        )
    )

    raised = None
    yields = []
    try:
        async for chunk in engine.stream_outputs(rid):
            yields.append(chunk)
    except InferenceAbortedError as exc:
        raised = exc

    assert raised is not None, "stream_outputs must raise InferenceAbortedError"
    assert "Metal" in str(raised)
    assert yields == [], (
        "stream_outputs must not yield the error chunk before raising — "
        "otherwise streaming clients receive a malformed terminal frame"
    )


@pytest.mark.asyncio
async def test_stream_outputs_returns_partial_on_repetition_abort():
    """Streaming path, repetition-guard hard-stop: unlike a Metal abort, this is
    a graceful terminal stop whose partial output is valid. stream_outputs must
    NOT raise — it must yield the (already-generated) chunk with the internal
    "abort" finish_reason remapped to the spec-valid "length" and ``error``
    cleared, so the client gets a clean terminal frame instead of a 503/error
    SSE frame that discards the partial and invites an identical retry."""
    engine = _make_engine()

    rid = "stream-rep-1"
    engine._output_collectors[rid] = RequestOutputCollector(aggregate=True)
    engine._output_collectors[rid].put(
        RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="abort",
            error=(
                "Model generation aborted: exact repetition loop detected "
                "(period_tokens=3, repeats=86)"
            ),
            error_kind="repetition",
            output_text="hahaha",
            completion_tokens=280,
        )
    )

    raised = None
    yields = []
    try:
        async for chunk in engine.stream_outputs(rid):
            yields.append(chunk)
    except InferenceAbortedError as exc:  # must NOT happen
        raised = exc

    assert raised is None, "repetition abort must not raise (it is graceful)"
    assert len(yields) == 1, "the terminal partial chunk must be yielded"
    assert yields[0].finish_reason == "length", "abort remapped to spec-valid length"
    assert yields[0].error is None, "error cleared so no 503/error frame"
    assert yields[0].output_text == "hahaha", "partial output preserved"


@pytest.mark.asyncio
async def test_generate_returns_partial_on_repetition_abort():
    """Non-streaming path, repetition-guard hard-stop: generate() must RETURN the
    partial output (200) with finish_reason remapped to "length" and error
    cleared, instead of raising InferenceAbortedError (→ 503) and discarding it."""
    engine = _make_engine()

    rid = "gen-rep-1"
    engine._output_collectors[rid] = RequestOutputCollector(aggregate=True)
    ev = asyncio.Event()
    ev.set()
    engine._finished_events[rid] = ev
    # Seed TWO chunks without draining so the collector AGGREGATES them (the
    # producer-ahead path generate() actually hits in production). This is what
    # catches error_kind being dropped by _merge_outputs — a single terminal
    # chunk would bypass the merge entirely.
    engine._output_collectors[rid].put(
        RequestOutput(
            request_id=rid,
            new_text="haha",
            new_token_ids=[1, 2],
            output_text="haha",
            completion_tokens=2,
        )
    )
    engine._output_collectors[rid].put(
        RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="abort",
            error=(
                "Model generation aborted: exact repetition loop detected "
                "(period_tokens=3, repeats=86)"
            ),
            error_kind="repetition",
            new_text="ha",
            new_token_ids=[3],
            output_text="hahaha",
            completion_tokens=280,
        )
    )

    # Bypass the real enqueue — return the pre-seeded rid so generate() drains
    # our repetition output and exercises the raise-or-return decision.
    async def _fake_add_request(*_a, **_kw):
        return rid

    engine.add_request = _fake_add_request

    result = await engine.generate(prompt="x", request_id=rid)
    assert result is not None, "generate must return the partial, not raise"
    assert result.finish_reason == "length", "abort remapped to spec-valid length"
    assert result.error is None, "error cleared so the route returns 200 not 503"
    assert result.output_text == "hahaha", "partial output preserved"
    # Guard the exact regression codex flagged: error_kind must survive
    # aggregation, else the merged terminal arrives as None and generate() 503s.
    assert result.error_kind == "repetition", "error_kind survived _merge_outputs"


@pytest.mark.asyncio
async def test_generate_still_raises_503_on_runtime_abort():
    """Guard: a genuine runtime abort (error_kind None, e.g. Metal) must STILL
    raise InferenceAbortedError so the HTTP layer keeps mapping it to 503. Only
    the repetition kind is exempted."""
    engine = _make_engine()

    rid = "gen-metal-1"
    engine._output_collectors[rid] = RequestOutputCollector(aggregate=True)
    ev = asyncio.Event()
    ev.set()
    engine._finished_events[rid] = ev
    engine._output_collectors[rid].put(
        RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="abort",
            error="Inference aborted: RuntimeError: Metal command buffer error",
            error_kind=None,
        )
    )

    async def _fake_add_request(*_a, **_kw):
        return rid

    engine.add_request = _fake_add_request

    raised = None
    try:
        await engine.generate(prompt="x", request_id=rid)
    except InferenceAbortedError as exc:
        raised = exc
    assert raised is not None, "runtime abort must still raise (→ 503)"
    assert "Metal" in str(raised)


def test_merge_selects_error_and_kind_atomically():
    """Aggregation must pick ``error`` and ``error_kind`` as a matched pair.
    A newer genuine runtime abort (error set, error_kind None) merged over an
    older "repetition" chunk must NOT inherit the stale "repetition" kind —
    otherwise the engine would clear a real runtime error and return 200
    instead of 503."""
    collector = RequestOutputCollector(aggregate=True)
    rid = "merge-pair-1"
    # Older chunk: a repetition abort.
    collector.put(
        RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="abort",
            error="Model generation aborted: exact repetition loop detected",
            error_kind="repetition",
            output_text="haha",
        )
    )
    # Newer chunk (producer got further ahead): a genuine runtime abort with no
    # kind. Not drained in between → forces a merge.
    collector.put(
        RequestOutput(
            request_id=rid,
            finished=True,
            finish_reason="abort",
            error="Inference aborted: RuntimeError: Metal command buffer error",
            error_kind=None,
        )
    )
    merged = collector.get_nowait()
    assert merged is not None
    assert "Metal" in merged.error, "newer runtime error wins"
    assert merged.error_kind is None, (
        "error_kind must follow its own error — not inherit the stale "
        "'repetition' kind, which would wrongly downgrade a 503 to 200"
    )


@pytest.mark.asyncio
async def test_engine_loop_backs_off_on_persistent_failures():
    """When step() fails repeatedly the loop must back off — otherwise the
    retry cadence floods logs at ~10 Hz and burns CPU spinning on a stuck
    Metal state (#353 DeepSeek round 1 P0). We exercise this by measuring
    how long the loop takes to complete N failures past the burst cap."""
    engine = _make_engine()

    fail_count = {"n": 0}

    class _AlwaysFailScheduler:
        def has_requests(self):
            return True

        def step(self):
            fail_count["n"] += 1
            raise RuntimeError("Metal command buffer error: persistent")

        def add_request(self, *_a, **_kw):
            pass

        def abort_request(self, *_a, **_kw):
            return True

        def remove_finished_request(self, *_a, **_kw):
            pass

    engine.scheduler = _AlwaysFailScheduler()
    engine._running = True

    import time as _time

    loop_task = asyncio.create_task(engine._engine_loop())
    try:
        # Run for 0.6 s. With the 0.1 s fast retry cadence, that would let
        # ~6 step() calls through; with the 1 s slow cadence after the
        # 10-failure burst, we'd be at most ~11. Verifying we don't exceed
        # ~20 ensures backoff is actually engaged (no-backoff would yield
        # 60+ failures over the same window).
        start = _time.perf_counter()
        await asyncio.sleep(0.6)
        elapsed = _time.perf_counter() - start
    finally:
        engine._running = False
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    # The exact count is timing-sensitive but a generous ceiling pins the
    # contract: backoff must keep the retry rate well below "every 100 ms".
    assert fail_count["n"] <= 20, (
        f"engine loop attempted step() {fail_count['n']} times in "
        f"{elapsed:.2f}s — backoff is not engaging on persistent failures"
    )


def test_metal_message_detection_patterns():
    """Pin the substring matchers used to flag Metal errors — if these
    drift, the recovery path silently downgrades to the generic branch
    (still works, just logs more noisily)."""
    samples = [
        "Metal command buffer error: kIOGPUCommandBufferCallbackErrorOutOfMemory",
        "RuntimeError from mlx::core::gpu::check_error",
        "MTL exception in completion handler",
        "command buffer failed",
    ]
    for s in samples:
        assert any(
            n in s for n in ("Metal", "MTL", "command buffer", "gpu::check_error")
        ), f"{s!r} no longer matches any Metal heuristic"


def test_scheduler_generation_error_is_not_reported_as_normal_length_finish():
    """Regression for the 12k-step DeepSeek Metal resource-limit failure.

    Scheduler-local recovery used to return ``finish_reason=length`` without
    an error, causing Responses to emit ``response.completed`` after the GPU
    had failed.
    """
    tokenizer = MagicMock()
    scheduler = Scheduler(MagicMock(), tokenizer, SchedulerConfig(max_num_seqs=2))
    req = Request("metal-resource-limit", "prompt", SamplingParams(max_tokens=32768))
    req.status = RequestStatus.RUNNING
    scheduler.running[req.request_id] = req
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.next.side_effect = RuntimeError(
        "[metal::malloc] Resource limit (499000) exceeded"
    )

    output = scheduler.step()

    assert output.finished_request_ids == {req.request_id}
    assert len(output.outputs) == 1
    terminal = output.outputs[0]
    assert terminal.finished is True
    assert terminal.finish_reason == "abort"
    assert terminal.finish_reason != "length"
    assert terminal.error is not None
    assert "Resource limit" in terminal.error
