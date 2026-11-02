"""Text step failures retire only the work owned by the failed operation."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections import Counter
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
@pytest.mark.parametrize("escaped", [True, False])
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
async def test_admission_failure_isolated_and_classified(
    monkeypatch, error, code, escaped
):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    record_failure = MagicMock(wraps=scheduler.performance.record_failed_performance)
    monkeypatch.setattr(
        scheduler.performance, "record_failed_performance", record_failure
    )
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
            if escaped:
                scheduler._step_request_id = "bad"
                raise error
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
    assert Counter(
        call.args[0].request_id for call in record_failure.call_args_list
    ) == {"bad": 1}
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
    record_failure = MagicMock(wraps=scheduler.performance.record_failed_performance)
    monkeypatch.setattr(
        scheduler.performance, "record_failed_performance", record_failure
    )
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
    assert Counter(
        call.args[0].request_id for call in record_failure.call_args_list
    ) == {
        "bad": 1,
        "active": 1,
    }
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


@pytest.mark.parametrize("shared_failures", [0, 9])
def test_repeated_admission_failures_preserve_healthy_work(
    monkeypatch, shared_failures
):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    scheduler._consecutive_step_failures = shared_failures
    monkeypatch.setattr(scheduler, "_max_running_sequences", lambda: 2)
    monkeypatch.setattr(
        scheduler, "_pop_waiting_for_admission", lambda: scheduler.waiting.popleft()
    )

    def fail_admission(request):
        raise ValueError("invalid admission input")

    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", fail_admission)

    for index in range(11):
        rid = "bad" if index == 0 else f"bad-{index}"
        if index:
            request = Request(rid, [1, 2], SamplingParams())
            request.prompt_token_ids = [1, 2]
            request.num_prompt_tokens = 2
            scheduler.requests[rid] = request
            scheduler.waiting.appendleft(request)

        output = scheduler.step()

        assert output.finished_request_ids == {rid}
        assert [out.request_id for out in output.outputs] == [rid]
        assert output.outputs[0].finish_reason == "abort"
        assert scheduler.requests[rid].status == RequestStatus.FINISHED_ABORTED
        assert scheduler._step_request_id is None
        assert set(scheduler.running) == {"active"}
        assert scheduler.requests["active"].status == RequestStatus.RUNNING
        assert [request.request_id for request in scheduler.waiting] == ["queued"]
        assert not RequestStatus.is_finished(scheduler.requests["queued"].status)

    assert scheduler._consecutive_step_failures == shared_failures


@pytest.mark.parametrize("operation", ["insert", "insert_segments", "ensure"])
def test_shared_insert_failure_recovers_batch(monkeypatch, operation):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    generator = MagicMock()
    error = RuntimeError("shared insert failed")
    generator.insert.side_effect = error
    generator.insert_segments.side_effect = error
    scheduler.batch_generator = generator
    scheduler.request_id_to_uid["active"] = 2
    scheduler.uid_to_request_id[2] = "active"
    monkeypatch.setattr(scheduler, "_max_running_sequences", lambda: 2)
    ensure = MagicMock(return_value=True)
    if operation == "ensure":
        ensure.side_effect = error
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", ensure)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    monkeypatch.setattr(
        scheduler, "_pop_waiting_for_admission", lambda: scheduler.waiting.popleft()
    )
    monkeypatch.setattr(
        scheduler,
        "_shared_prefix_local_split",
        lambda *args: 1 if operation == "insert_segments" else None,
    )
    recovery = MagicMock(wraps=scheduler._recover_from_generation_error)
    monkeypatch.setattr(scheduler, "_recover_from_generation_error", recovery)

    output = scheduler.step()

    recovery.assert_called_once_with()
    if operation == "ensure":
        ensure.assert_called_once()
        generator.insert.assert_not_called()
        generator.insert_segments.assert_not_called()
    else:
        getattr(generator, operation).assert_called_once()
    assert output.finished_request_ids == {"active"}
    assert all(out.error_kind == "engine_aborted" for out in output.outputs)
    assert scheduler._step_request_id is None
    assert not scheduler.running
    assert [request.request_id for request in scheduler.waiting] == ["queued"]
    generator.close.assert_called_once()


def test_shared_admission_bookkeeping_failure_recovers_batch(monkeypatch):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    generator = MagicMock()
    generator.insert.return_value = [1]
    scheduler.batch_generator = generator
    scheduler.request_id_to_uid["active"] = 2
    scheduler.uid_to_request_id[2] = "active"
    monkeypatch.setattr(scheduler, "_ensure_batch_generator", lambda params: True)
    monkeypatch.setattr(scheduler, "_prefill_cohort_fits", lambda request: True)
    monkeypatch.setattr(
        scheduler, "_pop_waiting_for_admission", lambda: scheduler.waiting.popleft()
    )
    recovery = MagicMock(wraps=scheduler._recover_from_generation_error)
    monkeypatch.setattr(scheduler, "_recover_from_generation_error", recovery)
    capacity_checks = 0

    def check_capacity():
        nonlocal capacity_checks
        capacity_checks += 1
        if capacity_checks == 1:
            return 3
        # The next loop condition is shared work after successful admission.
        assert set(scheduler.running) == {"bad", "active"}
        assert scheduler.requests["bad"].batch_uid == 1
        raise RuntimeError("shared admission bookkeeping failed")

    monkeypatch.setattr(scheduler, "_max_running_sequences", check_capacity)

    output = scheduler.step()

    assert capacity_checks == 2
    generator.insert.assert_called_once()
    recovery.assert_called_once_with()
    assert output.finished_request_ids == {"bad", "active"}
    assert all(out.error_kind == "engine_aborted" for out in output.outputs)
    assert scheduler._step_request_id is None
    assert not scheduler.running
    assert [request.request_id for request in scheduler.waiting] == ["queued"]
    generator.close.assert_called_once()


def test_repeated_failures_escalate_to_queued_work(monkeypatch):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    record_failure = MagicMock(wraps=scheduler.performance.record_failed_performance)
    monkeypatch.setattr(
        scheduler.performance, "record_failed_performance", record_failure
    )
    scheduler._consecutive_step_failures = 9
    output = scheduler.fail_step(RuntimeError("persistent step failure"))
    assert {out.request_id for out in output.outputs} == {"bad", "active", "queued"}
    assert not scheduler.running
    assert not scheduler.waiting
    assert all(out.error_kind == "engine_aborted" for out in output.outputs)
    assert Counter(
        call.args[0].request_id for call in record_failure.call_args_list
    ) == {
        "bad": 1,
        "active": 1,
        "queued": 1,
    }


def test_failed_performance_is_idempotent():
    engine = make_engine()
    seed(engine)
    performance = engine.scheduler.performance
    request = engine.scheduler.requests["bad"]
    before = performance.snapshot().requests_failed

    assert performance.record_failed_performance(request) is True
    assert performance.record_failed_performance(request) is False
    assert performance.snapshot().requests_failed == before + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_fails", [False, True])
@pytest.mark.parametrize("escaped", [True, False])
@pytest.mark.parametrize("partial_recovery", [True, False])
async def test_recovery_failure_unblocks_requests_and_loop_survives(
    monkeypatch, escaped, partial_recovery, abort_fails
):
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    before = scheduler.performance.snapshot().requests_failed
    engine._output_collectors["done"] = RequestOutputCollector(aggregate=True)
    engine._finished_events["done"] = asyncio.Event()
    engine._finished_events["done"].set()
    error = RuntimeError("Metal command buffer execution failed: private/model/path")
    recovery_error = RuntimeError("recovery failed: private/recovery/path")

    def recover():
        if partial_recovery:
            request = scheduler.running.pop("active", None)
            if request is not None:
                scheduler.performance.record_failed_performance(request)
                request.set_finished(RequestStatus.FINISHED_ABORTED)
        raise recovery_error

    monkeypatch.setattr(scheduler, "_recover_from_generation_error", recover)
    fail_step = MagicMock(wraps=scheduler.fail_step)
    monkeypatch.setattr(scheduler, "fail_step", fail_step)
    monkeypatch.setattr(scheduler, "_schedule_waiting", MagicMock(side_effect=error))
    abort_request = MagicMock(wraps=scheduler.abort_request)
    if abort_fails:
        abort_request.side_effect = RuntimeError("abort enqueue failed")
    monkeypatch.setattr(scheduler, "abort_request", abort_request)
    original_step = scheduler.step
    calls = 0

    def step():
        nonlocal calls
        calls += 1
        if calls == 1:
            if escaped:
                raise error
            return original_step()
        scheduler.running.clear()
        scheduler.waiting.clear()
        if "later" in scheduler.requests:
            return SchedulerOutput(
                outputs=[
                    RequestOutput(request_id="later", finished=True, output_text="ok")
                ],
                finished_request_ids={"later"},
            )
        return SchedulerOutput()

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
        for rid in ("bad", "active", "queued"):
            output = engine._output_collectors[rid].get_nowait()
            assert output.request_id == rid
            assert output.finished
            assert output.error_kind == "engine_aborted"
            assert output.error
            assert "private/" not in output.error
        assert engine._output_collectors["done"].get_nowait() is None
        assert scheduler.performance.snapshot().requests_failed == before + 3
        assert fail_step.call_count == (1 if escaped else 2)
        assert Counter(call.args[0] for call in abort_request.call_args_list) == {
            "bad": 1,
            "active": 1,
            "queued": 1,
        }
        assert not task.done()

        request = Request("later", [1, 2], SamplingParams())
        scheduler.requests["later"] = request
        scheduler.waiting.append(request)
        engine._output_collectors["later"] = RequestOutputCollector(aggregate=True)
        engine._finished_events["later"] = asyncio.Event()
        engine._idle_event.set()
        await asyncio.wait_for(engine._finished_events["later"].wait(), 2)
        output = engine._output_collectors["later"].get_nowait()
        assert output.finished
        assert output.error is None
        assert output.output_text == "ok"
        assert not task.done()
    finally:
        engine._running = False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_compatibility_step_failure_records_each_request_once():
    engine = make_engine()
    seed(engine)
    scheduler = engine.scheduler
    record_failure = MagicMock(wraps=scheduler.performance.record_failed_performance)

    class CompatibilityScheduler:
        performance = MagicMock(record_failed_performance=record_failure)

        def has_requests(self):
            return True

        def get_request(self, rid):
            return scheduler.get_request(rid)

        def step(self):
            raise RuntimeError("step failed")

    engine.scheduler = CompatibilityScheduler()
    engine._finished_events["queued"].set()
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

    assert Counter(
        call.args[0].request_id for call in record_failure.call_args_list
    ) == {
        "bad": 1,
        "active": 1,
    }


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
