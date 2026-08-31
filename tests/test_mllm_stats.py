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
