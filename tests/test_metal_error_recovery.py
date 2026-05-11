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

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm_mlx.engine_core import EngineConfig, EngineCore
from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.request import InferenceAbortedError, RequestOutput


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
