# SPDX-License-Identifier: Apache-2.0
"""Tests for chat reasoning exposure defaults."""

from vllm_mlx.api.models import ChatCompletionRequest, ToolDefinition
from vllm_mlx.routes.chat import _should_emit_reasoning


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="local",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


def test_reasoning_is_hidden_by_default():
    assert _should_emit_reasoning(_request()) is False


def test_reasoning_is_emitted_when_explicitly_requested_without_tools():
    assert _should_emit_reasoning(_request(enable_thinking=True)) is True


def test_reasoning_stays_hidden_for_tool_requests():
    tool = ToolDefinition(
        function={
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {}},
        }
    )

    assert _should_emit_reasoning(_request(enable_thinking=True, tools=[tool])) is False
