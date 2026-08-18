# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for scheduler-level MLLM step failures (#1367)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler


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
    assert outputs[-1] is None
    for output in outputs[:-1]:
        assert output.finished is True
        assert output.finish_reason == "length"
        assert output.error == "MLLM inference failed due to an internal engine error"
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


def test_scheduler_step_does_not_turn_internal_failure_into_fake_success() -> None:
    """A model/runtime error must terminate as an error without leaking details."""
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
    assert terminal.error == "MLLM inference failed due to an internal engine error"
    assert "/Users/example" not in terminal.error
    scheduler.batch_generator.remove.assert_called_once_with([42])
