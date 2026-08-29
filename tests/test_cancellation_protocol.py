# SPDX-License-Identifier: Apache-2.0
"""Cancellation terminal-state mapping across public serving protocols."""

import json

import pytest

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest
from vllm_mlx.api.responses_adapter import _convert_status
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput


class _CancelledChatEngine:
    preserve_native_tool_format = False
    tokenizer = None

    def __init__(self, *, before_first_token: bool):
        self.before_first_token = before_first_token

    async def stream_chat(self, messages, **kwargs):
        del messages
        if "request_admitted_event" in kwargs:
            kwargs["request_admitted_event"].set()
        if not self.before_first_token:
            yield GenerationOutput(
                text="partial",
                new_text="partial",
                finished=False,
                finish_reason=None,
                prompt_tokens=1,
                completion_tokens=1,
            )
        yield GenerationOutput(
            text="partial" if not self.before_first_token else "",
            new_text="" if self.before_first_token else "partial",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="cancelled",
        )


class _CancelledCompletionEngine:
    async def stream_generate(self, **kwargs):
        del kwargs
        yield GenerationOutput(text="partial", new_text="partial")
        yield GenerationOutput(
            text="partial",
            new_text="partial",
            finished=True,
            finish_reason="cancelled",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("before_first_token", [True, False])
async def test_chat_stream_cancellation_has_no_successful_finish_frame(
    before_first_token,
):
    from vllm_mlx.routes.chat import stream_chat_completion

    cfg = reset_config()
    cfg.model_name = "test-model"
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            _CancelledChatEngine(before_first_token=before_first_token),
            request.messages,
            request,
        )
    ]
    events = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]

    if not before_first_token:
        assert any(
            choice.get("delta", {}).get("content") == "partial"
            for event in events
            for choice in event.get("choices", [])
        )
    assert all(
        event.get("finish_reason") not in ("cancelled", "length") for event in events
    )


@pytest.mark.asyncio
async def test_completion_stream_cancellation_has_no_successful_finish_frame():
    from vllm_mlx.routes.completions import stream_completion

    cfg = reset_config()
    cfg.model_name = "test-model"
    request = CompletionRequest(model="test-model", prompt="hi", stream=True)

    chunks = [
        chunk
        async for chunk in stream_completion(
            _CancelledCompletionEngine(), "hi", request
        )
    ]
    events = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]

    assert any(event["choices"][0]["text"] == "partial" for event in events)
    assert all(
        event["choices"][0]["finish_reason"] not in ("cancelled", "length")
        for event in events
    )


def test_responses_status_does_not_treat_cancellation_as_truncation():
    assert _convert_status("cancelled") == "failed"
    assert _convert_status("length") == "incomplete"


@pytest.mark.asyncio
async def test_anthropic_stream_cancellation_uses_protocol_error():
    from vllm_mlx.routes.anthropic import _stream_anthropic_messages

    cfg = reset_config()
    cfg.model_name = "test-model"
    openai_request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    anthropic_request = AnthropicRequest(
        model="test-model",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )

    chunks = [
        chunk
        async for chunk in _stream_anthropic_messages(
            _CancelledChatEngine(before_first_token=False),
            openai_request,
            anthropic_request,
        )
    ]

    events = [
        json.loads(chunk.split("data: ", 1)[1])
        for chunk in chunks
        if chunk.startswith("event:") and "data: " in chunk
    ]
    assert any(event["type"] == "error" for event in events)
    assert all(event["type"] != "message_delta" for event in events)
