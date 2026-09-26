# SPDX-License-Identifier: Apache-2.0
"""Scheduler integration for answer-scoped user stops on ``<think>`` models.

A stop string the model writes while reasoning must not end the request;
the same string in the answer must. Drives ``Scheduler._process_batch_
responses`` with a crafted decoded surface (no model) and the multimodal
scheduler's rolling-window matcher. Helper and route coverage lives in
``test_think_stop_answer_only.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from unittest.mock import MagicMock  # noqa: E402

from rapid_mlx.mllm_scheduler import MLLMScheduler  # noqa: E402
from rapid_mlx.reasoning.think_stop import ReasoningStopScope  # noqa: E402
from rapid_mlx.request import Request, RequestStatus, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

PROMPT_OPENED = ReasoningStopScope("<think>", "</think>", starts_in_reasoning=True)

# Qwen3.5-style: the template opens <think> in the prompt, the model counts
# in its reasoning, closes it, and answers.
COUNTING = "I will count 1 to 10 and stop at 10.\n</think>\n\n1\n2\n9\n10\n11"
COUNTING_ANSWER_BEFORE_STOP = (
    "I will count 1 to 10 and stop at 10.\n</think>\n\n1\n2\n9\n"
)


# ---------------------------------------------------------------------------
# Text scheduler
# ---------------------------------------------------------------------------


def _scheduler():
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s.split())))
    tokenizer.get_vocab = lambda: {"<think>": 1, "</think>": 2}
    tokenizer.name_or_path = "mlx-community/Qwen3.5-4B-4bit"
    scheduler = Scheduler(MagicMock(), tokenizer, SchedulerConfig(max_num_seqs=4))
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.remove.return_value = {}
    assert scheduler._is_harmony_family is False
    return scheduler


def _step(
    scheduler, *, decoded: str, stop: list[str], scope, finish_reason: str | None = None
):
    sp = SamplingParams(max_tokens=100, stop=stop, reasoning_stop_scope=scope)
    request = Request(request_id="r", prompt="ignored", sampling_params=sp)
    request.num_prompt_tokens = 4
    request.status = RequestStatus.RUNNING
    for token in (10, 11):
        request.append_output_token(token)
    decoder = MagicMock()
    decoder.get_full_text = lambda: decoded
    decoder.add_token = lambda _t: ""
    decoder.prev_text = decoded
    request._decoder = decoder

    scheduler.running["r"] = request
    scheduler.uid_to_request_id[0] = "r"
    scheduler._decode_tokens = lambda tokens: ""  # type: ignore[method-assign]
    response = MagicMock()
    response.uid = 0
    response.token = 42
    response.finish_reason = finish_reason
    response.logprobs = None
    del response.prompt_cache
    outputs, finished = scheduler._process_batch_responses([response])
    return outputs[0], finished


def test_scheduler_does_not_stop_on_a_stop_string_inside_reasoning():
    scheduler = _scheduler()
    output, finished = _step(
        scheduler,
        decoded="I will count 1 to 10 and stop at 10.",
        stop=["10"],
        scope=PROMPT_OPENED,
    )
    assert output.finish_reason is None
    assert finished == set()
    scheduler.batch_generator.remove.assert_not_called()


def test_scheduler_stops_in_the_answer_and_keeps_the_reasoning():
    scheduler = _scheduler()
    output, finished = _step(
        scheduler, decoded=COUNTING, stop=["10"], scope=PROMPT_OPENED
    )
    assert output.finish_reason == "stop"
    assert output.matched_stop == "10"
    assert "r" in finished
    assert output.output_text == COUNTING_ANSWER_BEFORE_STOP


def test_scheduler_without_scope_keeps_the_raw_stream_match():
    """Requests with no scope (no ``<think>`` parser, /v1/completions)
    behave exactly as before: the first raw occurrence stops."""
    scheduler = _scheduler()
    output, _ = _step(scheduler, decoded=COUNTING, stop=["10"], scope=None)
    assert output.finish_reason == "stop"
    assert output.output_text == "I will count 1 to "


def test_scheduler_resolves_a_partial_opener_on_terminal_step():
    scheduler = _scheduler()
    output, finished = _step(
        scheduler,
        decoded="<thi",
        stop=["<"],
        scope=ReasoningStopScope("<think>", "</think>", starts_in_reasoning=False),
        finish_reason="length",
    )
    assert output.finish_reason == "stop"
    assert output.matched_stop == "<"
    assert output.output_text == ""
    assert "r" in finished


# ---------------------------------------------------------------------------
# Multimodal scheduler rolling matcher
# ---------------------------------------------------------------------------


def _mllm_scheduler():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._is_harmony_family = False
    return scheduler


def test_mllm_matcher_ignores_reasoning_and_stops_in_answer():
    scheduler = _mllm_scheduler()
    reasoning = "I will count 1 to 10 and stop at 10."
    assert scheduler._match_user_stop(reasoning, 0, ["10"], PROMPT_OPENED) is None
    # Reasoning already seen; the close and answer arrive in one window.
    idx, stop_str = scheduler._match_user_stop(
        COUNTING, len(reasoning), ["10"], PROMPT_OPENED
    )
    assert stop_str == "10"
    assert COUNTING[:idx] == COUNTING_ANSWER_BEFORE_STOP
    # No scope: the raw rolling-window match is unchanged.
    idx, _ = scheduler._match_user_stop(reasoning, 0, ["10"])
    assert reasoning[:idx] == "I will count 1 to "


def test_mllm_matcher_resolves_a_partial_opener_at_terminal():
    scheduler = _mllm_scheduler()
    scope = ReasoningStopScope("<think>", "</think>", starts_in_reasoning=False)
    assert scheduler._match_user_stop("<thi", 0, ["<"], scope) is None
    assert scheduler._match_user_stop("<thi", 0, ["<"], scope, terminal=True) == (
        0,
        "<",
    )
