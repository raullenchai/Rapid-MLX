# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLLM status and metrics statistics."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import threading
import time
from collections import deque
from types import SimpleNamespace

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler
from vllm_mlx.request import RequestStatus
from vllm_mlx.runtime.model_performance import ModelPerformanceLedger


class _FakeDetokenizer:
    """Small streaming-detokenizer stub for scheduler accounting tests."""

    last_segment = "x"
    text = "x"

    def reset(self) -> None:
        pass

    def add_token(self, _token: int) -> None:
        pass

    def finalize(self) -> None:
        pass


class _FakeTokenizer:
    def __init__(self) -> None:
        self.detokenizer = _FakeDetokenizer()


def _bare_scheduler() -> MLLMScheduler:
    """Build an MLLM scheduler without loading MLX model components."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.processor = SimpleNamespace(tokenizer=_FakeTokenizer())
    scheduler.uid_to_request_id = {}
    scheduler.running = {}
    scheduler._detokenizer_pool = {}
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0
    scheduler.performance = ModelPerformanceLedger("gemma-test")
    return scheduler


def test_batched_engine_promotes_mllm_stats_to_common_top_level():
    """Status and metrics consumers must see MLLM counters at top level."""

    class FakeScheduler:
        def get_stats(self) -> dict[str, object]:
            return {
                "num_running": 1,
                "num_waiting": 2,
                "num_requests_processed": 3,
                "total_prompt_tokens": 40,
                "total_completion_tokens": 7,
                "prefix_cache": {
                    "hits": 2,
                    "misses": 1,
                    "evictions": 0,
                    "tokens_saved": 30,
                },
                "model_performance": {
                    "model_name": "gemma-test",
                    "requests_succeeded": 3,
                },
            }

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model_name = "gemma-test"
    engine._is_mllm = True
    engine._loaded = True
    engine._stream_interval = 1
    engine._mllm_scheduler = FakeScheduler()
    engine._engine = None
    engine._start_time = time.monotonic() - 2.0
    engine._mllm_scheduler._step_count = 12

    stats = engine.get_stats()

    assert stats["mllm_scheduler"]["num_running"] == 1
    assert stats["num_running"] == 1
    assert stats["num_waiting"] == 2
    assert stats["num_requests_processed"] == 3
    assert stats["total_prompt_tokens"] == 40
    assert stats["total_completion_tokens"] == 7
    assert stats["prefix_cache"]["tokens_saved"] == 30
    assert stats["model_performance"]["model_name"] == "gemma-test"
    assert stats["steps_executed"] == 12
    assert stats["uptime_seconds"] >= 2.0


def test_batched_engine_mllm_stats_cannot_overwrite_engine_identity():
    """A scheduler snapshot must not replace BatchedEngine-owned fields."""

    class FakeScheduler:
        _step_count = 0

        def get_stats(self) -> dict[str, object]:
            return {
                "engine_type": "scheduler-controlled",
                "model_name": "wrong-model",
                "loaded": False,
                "total_prompt_tokens": 5,
            }

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._model_name = "real-model"
    engine._is_mllm = True
    engine._loaded = True
    engine._stream_interval = 1
    engine._mllm_scheduler = FakeScheduler()
    engine._engine = None

    stats = engine.get_stats()

    assert stats["engine_type"] == "batched"
    assert stats["model_name"] == "real-model"
    assert stats["loaded"] is True
    assert stats["total_prompt_tokens"] == 5
    assert stats["uptime_seconds"] == 0.0


def test_mllm_completed_request_adds_prompt_and_completion_tokens():
    """Completed MLLM requests must contribute both token counters."""
    scheduler = _bare_scheduler()
    request = MLLMRequest(request_id="req-1", prompt="hello")
    scheduler.uid_to_request_id[7] = request.request_id
    scheduler.running[request.request_id] = request

    response = SimpleNamespace(
        uid=7,
        token=11,
        prompt_tokens=19,
        finish_reason="stop",
        token_is_stop_token=False,
        logprobs=None,
    )

    outputs, finished_ids = scheduler._process_batch_responses([response])

    assert finished_ids == {request.request_id}
    assert outputs[0].finished is True
    assert scheduler.total_prompt_tokens == 19
    assert scheduler.total_completion_tokens == 1
    assert scheduler.num_requests_processed == 1
    performance = scheduler.performance.snapshot()
    assert performance.model_name == "gemma-test"
    assert performance.requests_succeeded == 1
    assert performance.prompt_tokens == 19
    assert performance.completion_tokens == 1
    assert performance.ttft_seconds_count == 1


def test_mllm_cancelled_request_adds_prompt_tokens_without_completion_tokens():
    """Cancelled requests still account for their already-prefilled prompt."""
    scheduler = _bare_scheduler()
    request = MLLMRequest(request_id="req-2", prompt="hello")
    request.status = RequestStatus.RUNNING
    request.num_prompt_tokens = 23
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = deque()
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.running = {request.request_id: request}
    scheduler.batch_generator = None
    scheduler.finished_req_ids = set()
    scheduler._cancel_counter_lock = threading.Lock()
    scheduler._aborted_queue_ids = set()

    scheduler._do_abort_request(request.request_id)

    assert scheduler.total_prompt_tokens == 23
    assert scheduler.total_completion_tokens == 0
    assert request.status is RequestStatus.FINISHED_CANCELLED
    performance = scheduler.performance.snapshot()
    assert performance.requests_cancelled == 1
    assert performance.prompt_tokens == 23


def test_mllm_failed_request_records_partial_work():
    scheduler = _bare_scheduler()
    request = MLLMRequest(
        request_id="req-failed",
        prompt="hello",
        arrival_time=time.time() - 0.2,
        first_token_time=time.time() - 0.1,
        num_prompt_tokens=17,
        num_output_tokens=2,
    )
    request.status = RequestStatus.RUNNING

    scheduler._record_terminal_performance(request, "failed")

    performance = scheduler.performance.snapshot()
    assert performance.requests_failed == 1
    assert performance.prompt_tokens == 17
    assert performance.completion_tokens == 2
    assert performance.ttft_seconds_count == 1
    assert performance.decode_observations == 1


def test_mllm_best_effort_failure_does_not_assume_request_fields(caplog):
    scheduler = _bare_scheduler()

    scheduler._record_terminal_performance(object(), "failed")

    assert scheduler.performance.snapshot().requests_failed == 0


def test_mllm_waiting_failure_records_no_unprocessed_prompt_tokens():
    scheduler = _bare_scheduler()
    request = MLLMRequest(
        request_id="waiting-failure",
        prompt="hello",
        num_prompt_tokens=17,
    )

    scheduler._record_terminal_performance(request, "failed")

    performance = scheduler.performance.snapshot()
    assert performance.requests_failed == 1
    assert performance.prompt_tokens == 0


def test_mllm_does_not_commit_success_when_later_response_fails():
    scheduler = _bare_scheduler()
    first = MLLMRequest(request_id="first", prompt="hello")
    second = MLLMRequest(request_id="second", prompt="hello")
    scheduler.running = {"first": first, "second": second}
    scheduler.uid_to_request_id = {1: "first", 2: "second"}
    scheduler._detokenizer_pool["second"] = SimpleNamespace(
        add_token=lambda _token: (_ for _ in ()).throw(RuntimeError("decode failed"))
    )
    first_response = SimpleNamespace(
        uid=1,
        token=11,
        prompt_tokens=3,
        finish_reason="stop",
        token_is_stop_token=False,
        logprobs=None,
    )
    second_response = SimpleNamespace(
        uid=2,
        token=12,
        prompt_tokens=4,
        finish_reason=None,
        token_is_stop_token=False,
        logprobs=None,
    )

    with pytest.raises(RuntimeError, match="decode failed"):
        scheduler._process_batch_responses([first_response, second_response])

    assert scheduler.performance.snapshot().requests_succeeded == 0
