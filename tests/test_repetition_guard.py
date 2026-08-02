# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the agent repetition-loop safety stop."""

from unittest.mock import MagicMock

import mlx.core as mx

from vllm_mlx.repetition_guard import (
    AgentRepetitionLogitsProcessor,
    detect_repeated_token_suffix,
    predict_repeated_token_suffix,
)
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


def test_predicts_only_the_token_that_would_extend_loop():
    pattern = list(range(12))
    # The hard guard needs six copies; the preventative path arms after five.
    intervention = predict_repeated_token_suffix(pattern * 5)
    assert intervention is not None
    assert intervention.period_tokens == 12
    assert intervention.repeats == 5
    assert intervention.blocked_token_id == pattern[0]
    assert predict_repeated_token_suffix(pattern * 4) is None


def test_logits_processor_masks_predicted_token_once():
    pattern = list(range(12))
    processor = AgentRepetitionLogitsProcessor(pattern * 5)
    logits = mx.zeros((1, 32))
    processed = processor(mx.array([pattern[-1]]), logits)
    mx.eval(processed)
    assert float(processed[0, pattern[0]].item()) == float("-inf")
    assert float(processed[0, pattern[1]].item()) == 0.0
    assert processor.interventions == 1
    # Speculative verification can invoke processors for several positions
    # before the streamed history advances. Intervene only once per history.
    second = processor(mx.array([pattern[-1]]), logits)
    mx.eval(second)
    assert float(second[0, pattern[0]].item()) == 0.0
    assert processor.interventions == 1


def test_ignores_short_stutter_and_near_repeat():
    assert detect_repeated_token_suffix([7] * 200) is None
    pattern = list(range(12))
    assert detect_repeated_token_suffix(pattern * 5) is None
    assert detect_repeated_token_suffix(pattern * 5 + pattern[:-1] + [999]) is None


def test_stops_sustained_single_token_loop_before_resource_exhaustion():
    """DeepSeek can collapse into a one-token CJK loop (for example 商店).

    Keep the threshold high enough to tolerate intentional short stutters, but
    stop the production failure well before it reaches ten thousand tokens and
    exhausts Metal resource handles.
    """
    assert detect_repeated_token_suffix([7] * 255) is None
    match = detect_repeated_token_suffix([7] * 256)
    assert match is not None
    assert match.period_tokens == 1
    assert match.repeats == 256


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


def test_scheduler_fails_exact_loop_for_tool_request():
    scheduler, req, output, finished = _drive_repeating_request(has_tools=True)
    assert finished == {req.request_id}
    assert req.status == RequestStatus.FINISHED_ABORTED
    assert output.finished is True
    assert output.finish_reason == "abort"
    assert output.error is not None
    assert "repetition" in output.error.lower()
    assert scheduler.num_repetition_loop_stops == 1
    assert scheduler.get_stats()["num_repetition_loop_stops"] == 1


def test_scheduler_does_not_change_plain_chat_semantics():
    scheduler, req, output, finished = _drive_repeating_request(has_tools=False)
    assert finished == set()
    assert req.status == RequestStatus.RUNNING
    assert output.finished is False
    assert scheduler.num_repetition_loop_stops == 0


def test_scheduler_fails_single_token_loop_for_tool_request():
    scheduler = _scheduler()
    req = Request("repeat-short", "prompt", SamplingParams(max_tokens=20_000))
    req.status = RequestStatus.RUNNING
    req.has_tools = True
    req.output_token_ids = [42] * 255
    scheduler.running[req.request_id] = req
    scheduler.uid_to_request_id[1] = req.request_id

    response = MagicMock(uid=1, token=42, finish_reason=None, logprobs=None)
    del response.prompt_cache
    outputs, finished = scheduler._process_batch_responses([response])

    assert finished == {req.request_id}
    assert outputs[0].finished is True
    assert outputs[0].finish_reason == "abort"
    assert outputs[0].error is not None
    assert scheduler.num_repetition_loop_stops == 1


def test_long_diverse_stream_stress_has_no_false_positive():
    """Exercise more tokens than the production failure with bounded scans."""
    tokens: list[int] = []
    for i in range(20_000):
        # Deterministic, non-periodic-enough agent-like stream containing
        # punctuation/newline stutters but no exact repeated suffix.
        tokens.append((i * i + 31 * i + (i % 97) * 13) % 32_749)
        if len(tokens) % 8 == 0:
            assert detect_repeated_token_suffix(tokens) is None


def test_guard_only_reads_bounded_suffix_under_large_history():
    class _BoundedSequence:
        def __init__(self, size: int):
            self.size = size
            self.requested_slice = None

        def __len__(self):
            return self.size

        def __getitem__(self, key):
            self.requested_slice = key
            assert isinstance(key, slice)
            assert key.start == -768 and key.stop is None
            return list(range(768))

    history = _BoundedSequence(1_000_000)
    assert detect_repeated_token_suffix(history) is None
    assert history.requested_slice == slice(-768, None, None)
