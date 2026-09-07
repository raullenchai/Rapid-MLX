"""#3155 request-scoped MTP telemetry response contract."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionResponse,
)
from vllm_mlx.api.responses_adapter import openai_to_responses
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.engine import GenerationOutput
from vllm_mlx.service.helpers import _build_response_metrics, _merge_response_metrics

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
