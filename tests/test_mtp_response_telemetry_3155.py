"""#3155 request-scoped MTP telemetry response contract."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from rapid_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionResponse,
)
from rapid_mlx.api.responses_adapter import openai_to_responses
from rapid_mlx.api.responses_models import ResponsesRequest
from rapid_mlx.engine import GenerationOutput
from rapid_mlx.service.helpers import _build_response_metrics, _merge_response_metrics

_RAW_METRICS = {
    "verify_calls": 3,
    "correction_tokens": 1,
    "bonus_tokens": 2,
    "accepted_by_depth": [3, 2, 1],
    "drafted_by_depth": [3, 3, 2],
}


def _metrics():
    return _build_response_metrics(
        GenerationOutput(text="ok", spec_decode_metrics=_RAW_METRICS)
    )


def test_chat_response_emits_metrics_only_when_mtp_ran():
    choice = ChatCompletionChoice(message=AssistantMessage(content="ok"))
    plain = ChatCompletionResponse(model="model", choices=[choice])
    assert "metrics" not in plain.model_dump(exclude_none=True)

    measured = ChatCompletionResponse(
        model="model", choices=[choice], metrics=_metrics()
    )
    assert measured.model_dump(exclude_none=True)["metrics"] == {
        "speculative_decoding": _RAW_METRICS
    }

    assert _build_response_metrics(MagicMock()) is None


def test_terminal_chat_chunk_serializes_the_same_metrics_envelope():
    chunk = ChatCompletionChunk(
        model="model",
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(), finish_reason="stop"
            )
        ],
        metrics=_metrics(),
    )
    payload = json.loads(chunk.model_dump_json(exclude_none=True))
    assert payload["metrics"]["speculative_decoding"] == _RAW_METRICS


def test_responses_adapter_preserves_request_metrics():
    response = ChatCompletionResponse(
        model="model",
        choices=[ChatCompletionChoice(message=AssistantMessage(content="ok"))],
        metrics=_metrics(),
    )
    converted = openai_to_responses(
        response,
        model="model",
        request=ResponsesRequest(model="model", input="hello"),
        created_at=1,
    )
    payload = converted.model_dump(exclude_none=True)
    assert payload["metrics"]["speculative_decoding"] == _RAW_METRICS


def test_multi_prompt_completion_metrics_are_summed_by_depth():
    first = GenerationOutput(text="a", spec_decode_metrics=_RAW_METRICS)
    second = GenerationOutput(
        text="b",
        spec_decode_metrics={
            "verify_calls": 2,
            "correction_tokens": 2,
            "bonus_tokens": 0,
            "accepted_by_depth": [1],
            "drafted_by_depth": [2],
        },
    )

    merged = _merge_response_metrics([first, GenerationOutput(text="plain"), second])

    assert merged is not None
    assert merged.speculative_decoding is not None
    assert merged.speculative_decoding.model_dump() == {
        "verify_calls": 5,
        "correction_tokens": 3,
        "bonus_tokens": 2,
        "accepted_by_depth": [4, 2, 1],
        "drafted_by_depth": [5, 3, 2],
    }


_TIMING = {"time_to_first_token_ms": 250.0, "mean_itl_ms": 20.0}


def test_timing_response_metrics_coexist_with_speculative_counters():
    output = GenerationOutput(
        text="ok", timing_metrics=_TIMING, spec_decode_metrics=_RAW_METRICS
    )
    metrics = _build_response_metrics(output)
    assert metrics.model_dump(exclude_none=True) == {
        **_TIMING,
        "speculative_decoding": _RAW_METRICS,
    }


def test_single_generation_timing_survives_completion_merge():
    output = GenerationOutput(text="ok", timing_metrics=_TIMING)
    assert _merge_response_metrics([output]).model_dump(exclude_none=True) == _TIMING
    assert _merge_response_metrics([output, output]) is None


def test_repair_attempt_aggregation_does_not_invent_request_timing():
    from rapid_mlx.service.helpers import _aggregate_generation_attempts

    output = GenerationOutput(text="ok", timing_metrics=_TIMING)
    assert _aggregate_generation_attempts(output, output).timing_metrics is None
