"""Text step failures retire only the work owned by the failed operation."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

from tests.headless_mlx._headless_mlx import install_headless_mlx_import_stubs

# These lifecycle contracts never execute tensor operations.
if "mlx.core" not in sys.modules:
    original_find_spec = importlib.util.find_spec
    with patch(
        "importlib.util.find_spec",
        side_effect=lambda name: None if name == "mlx" else original_find_spec(name),
    ):
        install_headless_mlx_import_stubs()

from rapid_mlx.engine_core import EngineConfig, EngineCore
from rapid_mlx.output_collector import RequestOutputCollector
from rapid_mlx.request import (
    InferenceAbortedError,
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
    inference_aborted_error_payload,
)
from rapid_mlx.scheduler import SchedulerConfig, SchedulerOutput


def make_engine():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    engine = EngineCore(
        MagicMock(),
        tokenizer,
        EngineConfig(scheduler_config=SchedulerConfig(enable_prefix_cache=False)),
    )
    engine._await_admission_wave = MagicMock(side_effect=lambda: asyncio.sleep(0))
    return engine


def seed(engine):
    for rid in ("bad", "active", "queued"):
        request = Request(rid, [1, 2], SamplingParams())
        request.prompt_token_ids = [1, 2]
        request.num_prompt_tokens = 2
        engine.scheduler.requests[rid] = request
        engine._output_collectors[rid] = RequestOutputCollector(aggregate=True)
        engine._finished_events[rid] = asyncio.Event()
    scheduler = engine.scheduler
    scheduler.running["active"] = scheduler.requests["active"]
    scheduler.requests["active"].status = RequestStatus.RUNNING
    scheduler.waiting.extend([scheduler.requests["bad"], scheduler.requests["queued"]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code",
    [
        (
            RuntimeError("Metal command buffer execution failed: private/model/path"),
            "engine_aborted",
        ),
        (
            RuntimeError(
                "[metal::malloc] Resource limit (499000) exceeded private/model/path"
            ),
            "insufficient_memory",
        ),
        (
            RuntimeError(
                "Metal kIOGPUCommandBufferCallbackErrorOutOfMemory private/model/path"
            ),
            "insufficient_memory",
        ),
    ],
)
async def test_admission_failure_isolated_and_classified(monkeypatch, error, code):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    # Exercise the real admission loop up to request-local prefill validation.
    monkeypatch.setattr(scheduler, "_max_running_sequences", lambda: 2)
    monkeypatch.setattr(
        scheduler, "_pop_waiting_for_admission", lambda: scheduler.waiting.popleft()
    )
    monkeypatch.setattr(
        scheduler, "_prefill_cohort_fits", lambda request: (_ for _ in ()).throw(error)
    )
    original_step = scheduler.step
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_step()
        ids = set(scheduler.running) | {r.request_id for r in scheduler.waiting}
        scheduler.running.clear()
        scheduler.waiting.clear()
        return SchedulerOutput(
            outputs=[
                RequestOutput(request_id=rid, finished=True, output_text="ok")
                for rid in ids
            ],
            finished_request_ids=ids,
        )

    monkeypatch.setattr(scheduler, "step", step)
    engine._running = True
    task = asyncio.create_task(engine._engine_loop())
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(event.wait() for event in engine._finished_events.values())
            ),
            2,
        )
    finally:
        engine._running = False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    outputs = {
        rid: collector.get_nowait()
        for rid, collector in engine._output_collectors.items()
    }
    assert outputs["bad"].error_kind == code
    assert "private/model/path" not in outputs["bad"].error
    payload = inference_aborted_error_payload(
        InferenceAbortedError(
            outputs["bad"].error, error_kind=outputs["bad"].error_kind
        )
    )
    assert payload["code"] == code
    for rid in ("active", "queued"):
        assert outputs[rid].error is None
        assert outputs[rid].output_text == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("escaped", [True, False])
async def test_shared_step_failure_preserves_queued(monkeypatch, escaped):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    scheduler.running["bad"] = scheduler.waiting.popleft()
    scheduler.requests["bad"].status = RequestStatus.RUNNING
    generator = MagicMock()
    scheduler.batch_generator = generator
    scheduler.request_id_to_uid.update({"bad": 1, "active": 2})
    scheduler.uid_to_request_id.update({1: "bad", 2: "active"})
    cache = MagicMock()
    scheduler.block_aware_cache = cache
    error = RuntimeError("Metal command buffer execution failed: private/model/path")
    generator.next.side_effect = error
    original_step = scheduler.step
    monkeypatch.setattr(scheduler, "_schedule_waiting", lambda: [])
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        if calls == 1:
            if escaped:
                raise error
            return original_step()
        scheduler.waiting.clear()
        return SchedulerOutput(
            outputs=[
                RequestOutput(request_id="queued", finished=True, output_text="ok")
            ],
            finished_request_ids={"queued"},
        )

    monkeypatch.setattr(scheduler, "step", step)
    engine._running = True
    task = asyncio.create_task(engine._engine_loop())
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(event.wait() for event in engine._finished_events.values())
            ),
            2,
        )
    finally:
        engine._running = False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    outputs = {
        rid: collector.get_nowait()
        for rid, collector in engine._output_collectors.items()
    }
    assert outputs["queued"].error is None
    for rid in ("bad", "active"):
        assert outputs[rid].error_kind == "engine_aborted"
        assert "private/model/path" not in outputs[rid].error
    assert not scheduler.running
    assert not scheduler.request_id_to_uid
    generator.close.assert_called_once()
    assert {call.args[0] for call in cache.release_cache.call_args_list} == {
        "bad",
        "active",
    }


def test_repeated_failures_escalate_to_queued_work():
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    scheduler._consecutive_step_failures = 9
    output = scheduler.fail_step(RuntimeError("persistent step failure"))
    assert {out.request_id for out in output.outputs} == {"bad", "active", "queued"}
    assert not scheduler.running
    assert not scheduler.waiting
    assert all(out.error_kind == "engine_aborted" for out in output.outputs)


@pytest.mark.parametrize("clear_fails", [False, True])
@pytest.mark.parametrize(
    "message,kind",
    [
        ("[metal::malloc] Resource limit (499000) exceeded", "insufficient_memory"),
        ("Metal kIOGPUCommandBufferCallbackErrorOutOfMemory", "insufficient_memory"),
        ("Metal command buffer execution failed", "engine_aborted"),
    ],
)
def test_attributed_step_failure_clears_memory_cache(
    monkeypatch, message, kind, clear_fails
):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    scheduler._step_request_id = "bad"
    # Ordinary abort cleanup flushes encoders before failure recovery reclaims memory.
    clear_cache = MagicMock(
        side_effect=[None, RuntimeError("cache clear failed") if clear_fails else None]
    )
    monkeypatch.setattr("rapid_mlx.scheduler.mx.clear_cache", clear_cache)

    output = scheduler.fail_step(RuntimeError(message))

    assert clear_cache.call_count == (2 if kind == "insufficient_memory" else 1)
    assert output.finished_request_ids == {"bad"}
    assert output.outputs[0].error_kind == kind
    assert message not in output.outputs[0].error
    assert set(scheduler.running) == {"active"}
    assert [request.request_id for request in scheduler.waiting] == ["queued"]
