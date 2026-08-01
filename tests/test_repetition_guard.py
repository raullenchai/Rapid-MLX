# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the agent repetition-loop safety stop."""

from unittest.mock import MagicMock

from vllm_mlx.repetition_guard import detect_repeated_token_suffix
from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _scheduler() -> Scheduler:
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: list(range(len(text.split())))
    tokenizer.decode = lambda tokens, **_kwargs: " ".join(map(str, tokens))
    return Scheduler(MagicMock(), tokenizer, SchedulerConfig(max_num_seqs=2))


def test_detects_long_exact_periodic_suffix():
    pattern = list(range(12))
    match = detect_repeated_token_suffix([999, 998] + pattern * 6)
    assert match is not None
    assert match.period_tokens == 12
    assert match.repeats == 6


def test_long_loop_period_stops_after_three_copies():
    pattern = list(range(61))
    assert detect_repeated_token_suffix(pattern * 2) is None
    match = detect_repeated_token_suffix(pattern * 3)
    assert match is not None
    assert match.period_tokens == 61
    assert match.repeats == 3


def test_ignores_short_stutter_and_near_repeat():
    assert detect_repeated_token_suffix([7] * 200) is None
    pattern = list(range(12))
    assert detect_repeated_token_suffix(pattern * 5) is None
    assert detect_repeated_token_suffix(pattern * 5 + pattern[:-1] + [999]) is None


def _drive_repeating_request(*, has_tools: bool):
    scheduler = _scheduler()
    pattern = list(range(12))
    req = Request("repeat", "prompt", SamplingParams(max_tokens=1000))
    req.status = RequestStatus.RUNNING
    req.has_tools = has_tools
    # The response below lands on both the sixth repetition and the scheduler's
    # every-eight-token check boundary (72 output tokens).
    req.output_token_ids = pattern * 5 + pattern[:-1]
    scheduler.running[req.request_id] = req
    scheduler.uid_to_request_id[1] = req.request_id

    response = MagicMock(uid=1, token=pattern[-1], finish_reason=None, logprobs=None)
    del response.prompt_cache
    outputs, finished = scheduler._process_batch_responses([response])
    return scheduler, req, outputs[0], finished


def test_scheduler_stops_exact_loop_for_tool_request():
    scheduler, req, output, finished = _drive_repeating_request(has_tools=True)
    assert finished == {req.request_id}
    assert req.status == RequestStatus.FINISHED_STOPPED
    assert output.finished is True
    assert output.finish_reason == "stop"
    assert scheduler.num_repetition_loop_stops == 1
    assert scheduler.get_stats()["num_repetition_loop_stops"] == 1


def test_scheduler_does_not_change_plain_chat_semantics():
    scheduler, req, output, finished = _drive_repeating_request(has_tools=False)
    assert finished == set()
    assert req.status == RequestStatus.RUNNING
    assert output.finished is False
    assert scheduler.num_repetition_loop_stops == 0
