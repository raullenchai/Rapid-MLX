# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for scheduler-level MLLM step failures (#1367)."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx.mllm_batch_generator import MLLMBatchResponse  # noqa: E402
from vllm_mlx.mllm_scheduler import (  # noqa: E402
    MLLMRequest,
    MLLMScheduler,
    MLLMSchedulerConfig,
)
from vllm_mlx.request import (  # noqa: E402
    ClientRequestError,
    InferenceAbortedError,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)


def _repetition_scheduler() -> MLLMScheduler:
    tokenizer = MagicMock()
    tokenizer.decode = lambda tokens, **_kwargs: " ".join(map(str, tokens))
    tokenizer.eos_token_id = 0
    processor = MagicMock()
    processor.tokenizer = tokenizer
    scheduler = MLLMScheduler(
        MagicMock(),
        processor,
        MLLMSchedulerConfig(enable_vision_cache=False),
        model_name="headless-mllm-repetition-test",
    )
    scheduler.batch_generator = MagicMock()
    return scheduler


def _repeating_mllm_request(scheduler: MLLMScheduler) -> MLLMRequest:
    pattern = list(range(61))
    request = MLLMRequest(
        request_id="vision-repeat",
        prompt="extract the table",
        images=["statement.png"],
        sampling_params=SamplingParams(max_tokens=32_768),
    )
    request.status = RequestStatus.RUNNING
    request.output_tokens = pattern * 3
    request.num_output_tokens = len(request.output_tokens)
    scheduler.running[request.request_id] = request
    scheduler.uid_to_request_id[7] = request.request_id
    return request


def test_mllm_repetition_stop_retires_live_row_without_mlx() -> None:
    scheduler = _repetition_scheduler()
    request = _repeating_mllm_request(scheduler)
    response = MLLMBatchResponse(
        uid=7,
        request_id=request.request_id,
        token=0,
        logprobs=None,
        finish_reason=None,
    )

    outputs, finished = scheduler._process_batch_responses([response])

    assert finished == {request.request_id}
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert outputs[0].finish_reason == "abort"
    assert outputs[0].error_kind == "repetition"
    assert "period_tokens=61" in (outputs[0].error or "")
    scheduler.batch_generator.remove.assert_called_once_with([7])
    assert scheduler.num_repetition_loop_stops == 1


def test_mllm_repetition_stop_refuses_unowned_batch_row() -> None:
    scheduler = _repetition_scheduler()
    request = _repeating_mllm_request(scheduler)
    scheduler.batch_generator = None
    response = MLLMBatchResponse(
        uid=7,
        request_id=request.request_id,
        token=0,
        logprobs=None,
        finish_reason=None,
    )

    with pytest.raises(RuntimeError, match="without a batch generator"):
        scheduler._process_batch_responses([response])


@pytest.mark.asyncio
async def test_mllm_repetition_stop_streams_valid_partial_response() -> None:
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"vision-repeat": asyncio.Queue()}
    scheduler.abort_request = MagicMock()
    await scheduler.output_queues["vision-repeat"].put(
        RequestOutput(
            request_id="vision-repeat",
            output_text="partial valid answer",
            finished=True,
            finish_reason="abort",
            error="Model generation aborted: exact repetition loop detected",
            error_kind="repetition",
        )
    )

    outputs = [output async for output in scheduler.stream_outputs("vision-repeat")]

    assert len(outputs) == 1
    assert outputs[0].output_text == "partial valid answer"
    assert outputs[0].finish_reason == "length"
    assert outputs[0].error is None
    scheduler.abort_request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_kind", "expected_error"),
    [
        ("lifecycle", InferenceAbortedError),
        ("invalid_request", ClientRequestError),
        (None, ValueError),
    ],
)
async def test_mllm_non_repetition_errors_keep_existing_exception_contract(
    error_kind: str | None,
    expected_error: type[Exception],
) -> None:
    """The repetition exception must not soften unrelated error classes."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"failed": asyncio.Queue()}
    scheduler.abort_request = MagicMock()
    await scheduler.output_queues["failed"].put(
        RequestOutput(
            request_id="failed",
            finished=True,
            finish_reason="abort",
            error="terminal failure",
            error_kind=error_kind,
        )
    )

    with pytest.raises(expected_error, match="terminal failure"):
        _ = [output async for output in scheduler.stream_outputs("failed")]

    scheduler.abort_request.assert_not_called()


def test_batch_generator_uses_worker_default_stream(monkeypatch) -> None:
    """Construction binds each generator lifetime to its worker's stream."""
    from vllm_mlx import mllm_batch_generator as module

    stream = object()
    monkeypatch.setattr(module.mx, "default_device", lambda: "gpu")
    monkeypatch.setattr(module.mx, "default_stream", lambda device: stream)
    monkeypatch.setattr(module.mx.metal, "is_available", lambda: False)

    generator = module.MLLMBatchGenerator(
        model=SimpleNamespace(language_model=object()),
        processor=object(),
        enable_vision_cache=False,
    )

    assert generator._stream is stream


def test_batch_generator_close_tolerates_retired_worker_stream(
    monkeypatch, caplog
) -> None:
    """A stale thread-local stream cannot prevent wired-limit cleanup."""
    from vllm_mlx import mllm_batch_generator as module

    generator = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    generator._stream = object()
    generator._old_wired_limit = 123
    restored: list[int] = []
    caplog.set_level(logging.DEBUG, logger=module.__name__)
    monkeypatch.setattr(
        module.mx,
        "synchronize",
        lambda _stream: (_ for _ in ()).throw(RuntimeError("retired stream")),
    )
    monkeypatch.setattr(module.mx, "set_wired_limit", restored.append)

    generator.close()

    assert restored == [123]
    assert generator._old_wired_limit is None
    assert "retired stream" in caplog.text


def test_batch_generator_prefill_enters_owned_stream(monkeypatch) -> None:
    """Vision prefill executes under the generator-owned stream context."""
    from contextlib import contextmanager

    from mlx_lm.models import cache as cache_module

    from vllm_mlx import mllm_batch_generator as module

    owned_stream = object()
    entered: list[object] = []

    @contextmanager
    def stream_context(stream):
        entered.append(stream)
        yield

    generator = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    generator.language_model = object()
    generator._stream = owned_stream
    generator.vision_prefill_token_budget = 8192
    generator.allow_arrays_cache = False
    generator._stats = SimpleNamespace(prompt_tokens=0)
    generator._preprocess_request = lambda _request: None
    generator._run_vision_encoding = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("prefill reached")
    )
    request = SimpleNamespace(input_ids=SimpleNamespace(size=1))
    monkeypatch.setattr(
        cache_module, "make_prompt_cache", lambda _model: object(), raising=False
    )
    monkeypatch.setattr(module.mx, "stream", stream_context)
    monkeypatch.setattr(module, "_prefill_cap_violation", lambda *_args: None)

    with pytest.raises(RuntimeError, match="prefill reached"):
        generator._process_prompts([request])

    assert entered == [owned_stream]


def test_batch_generator_next_uses_owned_stream(monkeypatch) -> None:
    """Each decode step runs in the same worker-owned stream context."""
    from contextlib import contextmanager

    from vllm_mlx import mllm_batch_generator as module

    owned_stream = object()
    entered: list[object] = []

    @contextmanager
    def stream_context(stream):
        entered.append(stream)
        yield

    generator = module.MLLMBatchGenerator.__new__(module.MLLMBatchGenerator)
    generator._stream = owned_stream
    generator._next = lambda: ["token"]
    monkeypatch.setattr(module.mx, "stream", stream_context)

    assert generator.next() == ["token"]
    assert entered == [owned_stream]


def test_mllm_scheduler_rejects_duplicate_public_ids() -> None:
    """A duplicate cannot replace the MLLM request cancellation addresses."""
    public_id = "chatcmpl-" + "a" * 32
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._generation_paused = False
    scheduler._paused_admission_tokens = set()
    scheduler._paused_add_allowance = 0
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler._cancelled_request_ids = set()
    scheduler._disconnect_abort_ids = set()
    request = SimpleNamespace(request_id=public_id, lifecycle_admission_token=None)

    scheduler._commit_request(request)

    with pytest.raises(ValueError, match="already exists"):
        scheduler._commit_request(request)
    assert scheduler.requests[public_id] is request


@pytest.mark.asyncio
async def test_process_loop_failure_unblocks_every_inflight_request() -> None:
    """Unexpected mlx-vlm/model errors must not be logged and retried forever."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    waiting = MLLMRequest(request_id="waiting-request", prompt="hello")
    running = MLLMRequest(request_id="running-request", prompt="hello")
    uid_only = "uid-only-request"
    pending_only = "pending-only-request"
    scheduler.requests = {
        waiting.request_id: waiting,
        running.request_id: running,
    }
    scheduler.waiting = __import__("collections").deque([waiting])
    # The running map is keyed by request ID; generator UIDs live only in the
    # two adjacent translation maps.
    scheduler.running = {running.request_id: running}
    aborted = "already-aborted-request"
    full_running_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    full_running_queue.put_nowait(object())  # stale partial output
    scheduler.output_queues = {
        waiting.request_id: asyncio.Queue(),
        running.request_id: full_running_queue,
        uid_only: asyncio.Queue(),
        pending_only: asyncio.Queue(),
        aborted: asyncio.Queue(),
    }
    scheduler.request_id_to_uid = {running.request_id: 42, uid_only: 43}
    scheduler.uid_to_request_id = {42: running.request_id, 43: uid_only}
    scheduler._detokenizer_pool = {running.request_id: object()}
    scheduler._pending_abort_ids = {running.request_id, pending_only}
    scheduler._aborted_queue_ids = {aborted}
    scheduler._abort_error_kinds = {aborted: "lifecycle"}
    scheduler.finished_req_ids = set()
    scheduler._running = True
    scheduler._injected_step_executor = None
    scheduler._step_executor = None
    scheduler._owns_step_executor = True
    batch_generator = MagicMock()
    scheduler.batch_generator = batch_generator
    scheduler._step_no_queue = MagicMock(
        side_effect=TypeError("Model.__call__() missing required argument: mask")
    )

    task = asyncio.create_task(scheduler._process_loop())
    try:
        # Let the fatal distributor confront the already-full bounded queue
        # before any consumer frees a slot.
        async def _wait_until_failed() -> None:
            while scheduler.requests:
                await asyncio.sleep(0)

        await asyncio.wait_for(_wait_until_failed(), timeout=0.5)
        outputs = await asyncio.wait_for(
            asyncio.gather(
                scheduler.output_queues[waiting.request_id].get(),
                scheduler.output_queues[running.request_id].get(),
                scheduler.output_queues[uid_only].get(),
                scheduler.output_queues[pending_only].get(),
                scheduler.output_queues[aborted].get(),
            ),
            timeout=0.5,
        )
    finally:
        scheduler._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert {output.request_id for output in outputs[:-1]} == {
        waiting.request_id,
        running.request_id,
        uid_only,
        pending_only,
    }
    assert outputs[-1].request_id == aborted
    assert outputs[-1].finished is True
    assert outputs[-1].error_kind == "lifecycle"
    for output in outputs[:-1]:
        assert output.finished is True
        assert output.finish_reason == "length"
        assert output.error == (
            "MLLM inference was interrupted by a transient engine error; "
            "retry the request"
        )
        assert output.error_kind == "lifecycle"
        assert "mask" not in output.error
    assert scheduler._step_no_queue.call_count == 1
    batch_generator.close.assert_called_once_with()
    assert scheduler.batch_generator is None
    assert not scheduler.requests
    assert not scheduler.waiting
    assert not scheduler.running
    assert not scheduler.request_id_to_uid
    assert not scheduler.uid_to_request_id
    assert not scheduler._detokenizer_pool
    assert not scheduler._pending_abort_ids
    assert not scheduler._aborted_queue_ids


def test_scheduler_step_marks_internal_failure_retryable_without_leaking_details(
    caplog,
) -> None:
    """A runtime batch failure is an observable, retryable lifecycle error."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="runtime-failure", prompt="hello")
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = __import__("collections").deque()
    scheduler.running = {request.request_id: request}
    scheduler.request_id_to_uid = {request.request_id: 42}
    scheduler.uid_to_request_id = {42: request.request_id}
    scheduler.finished_req_ids = set()
    scheduler._detokenizer_pool = {}
    scheduler._pending_abort_ids = set()
    scheduler._aborted_queue_ids = set()
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.next.side_effect = RuntimeError(
        "private runtime detail: /Users/example/model"
    )
    scheduler._process_pending_aborts = MagicMock()
    scheduler._schedule_waiting = MagicMock(return_value=[])

    output = scheduler._step_no_queue()

    assert output.finished_request_ids == {request.request_id}
    assert len(output.outputs) == 1
    terminal = output.outputs[0]
    assert terminal.finished is True
    assert terminal.finish_reason == "length"
    assert terminal.error == (
        "MLLM inference was interrupted by a transient engine error; retry the request"
    )
    assert terminal.error_kind == "lifecycle"
    assert "/Users/example" not in terminal.error
    assert "RuntimeError" in caplog.text
    assert request.request_id in caplog.text
    assert "private runtime detail" in caplog.text
    scheduler.batch_generator.remove.assert_called_once_with([42])


@pytest.mark.asyncio
async def test_scheduler_internal_failure_streams_as_retryable_503_class() -> None:
    """The scheduler-to-route boundary preserves the retryable error type."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"failed": asyncio.Queue()}
    scheduler.abort_request = MagicMock()
    await scheduler.output_queues["failed"].put(
        RequestOutput(
            request_id="failed",
            finished=True,
            finish_reason="abort",
            error=(
                "MLLM inference was interrupted by a transient engine error; "
                "retry the request"
            ),
            error_kind="lifecycle",
        )
    )

    with pytest.raises(InferenceAbortedError, match="retry the request"):
        _ = [output async for output in scheduler.stream_outputs("failed")]

    scheduler.abort_request.assert_not_called()
