# SPDX-License-Identifier: Apache-2.0
"""Tests for chat streaming tool-continuation retry behavior."""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.domain.events import StreamEvent
from vllm_mlx.engine import GenerationOutput
from vllm_mlx.routes.chat import stream_chat_completion
from vllm_mlx.service.helpers import (
    _TOOL_CALL_REQUIRED_RETRY_PROMPT,
    _TOOL_CONTINUATION_RETRY_PROMPT,
)


class _EngineThatExhaustsThenCallsTool:
    def __init__(self):
        self.calls = 0
        self.messages_seen = []
        self.tokenizer = MagicMock()

    async def stream_chat(self, messages, **kwargs):
        self.calls += 1
        self.messages_seen.append(messages)
        if self.calls == 1:
            yield GenerationOutput(
                text="Thinking only.",
                new_text="Thinking only.",
                prompt_tokens=10,
                completion_tokens=2,
                finished=False,
                finish_reason=None,
            )
            return

        yield GenerationOutput(
            text="<tool_call>",
            new_text="<tool_call>",
            prompt_tokens=12,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )


class _FakeStreamingPostProcessor:
    instances = 0

    def __init__(self, *args, **kwargs):
        self.index = _FakeStreamingPostProcessor.instances
        _FakeStreamingPostProcessor.instances += 1

    def set_thinking_model(self, model_name):
        pass

    def reset(self):
        pass

    def process_chunk(self, output):
        if self.index == 0:
            return [StreamEvent(type="content", content="Thinking only.")]
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                tool_calls_detected=True,
            )
        ]

    def finalize(self):
        return []


class _FakeRepetitionPostProcessor(_FakeStreamingPostProcessor):
    def process_chunk(self, output):
        if self.index == 0:
            return [StreamEvent(type="content", content="point " * 40)]
        return [
            StreamEvent(
                type="tool_call",
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_after_retry",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
                tool_calls_detected=True,
            )
        ]


@pytest.mark.asyncio
async def test_tool_continuation_retries_when_stream_exhausts_without_finish(
    monkeypatch,
):
    """Retry when engine stream ends after narration without finish event."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeStreamingPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do work"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    engine = _EngineThatExhaustsThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "tool", "content": "done"}],
            request,
            tool_continuation_retry=True,
            max_tokens=16,
        )
    ]

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CONTINUATION_RETRY_PROMPT
    assert not any("Thinking only." in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_tool_request_retries_when_model_repeats_text_before_tool_call(
    monkeypatch,
):
    """Retry first tool turn when model enters repeated-text loop."""
    from vllm_mlx.service import postprocessor

    _FakeStreamingPostProcessor.instances = 0
    monkeypatch.setattr(
        postprocessor,
        "StreamingPostProcessor",
        _FakeRepetitionPostProcessor,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do work"}],
        stream=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    engine = _EngineThatExhaustsThenCallsTool()

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            engine,
            [{"role": "user", "content": "Build app"}],
            request,
            tool_continuation_retry=False,
            max_tokens=16,
        )
    ]

    assert engine.calls == 2
    assert engine.messages_seen[1][-1]["content"] == _TOOL_CALL_REQUIRED_RETRY_PROMPT
    assert not any("point point" in chunk for chunk in chunks)
    assert any('"tool_calls"' in chunk for chunk in chunks)
