"""Regression coverage for text-model throughput in ``/v1/status``."""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import time
from unittest.mock import MagicMock

from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _scheduler() -> Scheduler:
    tokenizer = MagicMock()
    tokenizer.encode = lambda value: list(range(len(value.split())))
    return Scheduler(MagicMock(), tokenizer, SchedulerConfig(max_num_seqs=1))


def test_text_scheduler_exports_live_batch_generator_throughput():
    scheduler = _scheduler()
    request = Request(
        request_id="status-tps",
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=20),
    )
    request.status = RequestStatus.RUNNING
    request.num_prompt_tokens = 8
    request.first_token_time = time.time() - 2.0
    for token in range(10):
        request.append_output_token(token)
    scheduler.running[request.request_id] = request
    scheduler._last_prompt_tps = 40.0

    stats = scheduler.get_stats()
    throughput = stats["batch_generator"]

    assert throughput["prompt_tps"] == 40.0
    assert 4.0 < throughput["generation_tps"] < 6.0


def test_prompt_throughput_aggregates_requests_in_the_same_batch():
    scheduler = _scheduler()
    started = time.time() - 1.0
    responses = []

    for uid, prompt_tokens in enumerate((10, 20)):
        request = Request(
            request_id=f"batch-{uid}",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=20),
        )
        request.status = RequestStatus.RUNNING
        request.num_prompt_tokens = prompt_tokens
        request._prefill_started_at = started
        request._decoder = MagicMock()
        request._decoder.add_token.return_value = "x"
        scheduler.running[request.request_id] = request
        scheduler.uid_to_request_id[uid] = request.request_id

        response = MagicMock(uid=uid, token=uid, finish_reason=None, logprobs=None)
        del response.prompt_cache
        responses.append(response)

    scheduler._process_batch_responses(responses)

    # Both requests prefetched concurrently for ~1 second: 10 + 20 tok/s.
    assert 25.0 < scheduler._last_prompt_tps < 35.0
