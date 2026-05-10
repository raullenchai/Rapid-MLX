# SPDX-License-Identifier: Apache-2.0
"""Tests for chat reasoning exposure defaults."""

import pytest

from vllm_mlx.api.models import ChatCompletionRequest, ToolDefinition
from vllm_mlx.config import get_config
from vllm_mlx.routes.chat import (
    _looks_like_deferred_tool_use,
    _should_emit_reasoning,
    _tool_turn_max_tokens,
)


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="local",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


@pytest.fixture
def reset_no_thinking():
    cfg = get_config()
    original = cfg.no_thinking
    cfg.no_thinking = False
    try:
        yield cfg
    finally:
        cfg.no_thinking = original


def test_reasoning_is_emitted_by_default(reset_no_thinking):
    assert _should_emit_reasoning(_request()) is True


def test_reasoning_is_emitted_with_tools(reset_no_thinking):
    tool = ToolDefinition(
        function={
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {}},
        }
    )
    assert _should_emit_reasoning(_request(tools=[tool])) is True


def test_reasoning_suppressed_when_client_opts_out(reset_no_thinking):
    assert _should_emit_reasoning(_request(enable_thinking=False)) is False


def test_reasoning_suppressed_when_server_no_thinking(reset_no_thinking):
    reset_no_thinking.no_thinking = True
    assert _should_emit_reasoning(_request(enable_thinking=True)) is False


def test_calling_tool_equals_text_is_deferred_tool_use():
    assert _looks_like_deferred_tool_use("[Calling tool=read]") is True


def test_tool_turn_max_tokens_are_bounded_without_over_truncating():
    assert _tool_turn_max_tokens(None) == 1536
    assert _tool_turn_max_tokens(32768) == 1536
    assert _tool_turn_max_tokens(512) == 512
