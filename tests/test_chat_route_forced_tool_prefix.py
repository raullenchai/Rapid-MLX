# SPDX-License-Identifier: Apache-2.0
"""Tests for the forced-tool-choice assistant-prefix injection lever.

The OpenAI ``tool_choice`` spec guarantees that when a client forces a
specific function — either via ``{"type":"function","function":{"name":X}}``
or via ``"required"`` with a single tool — the model's response MUST carry
a tool call to that function. Local inference has no decoder-level
constraint, so we use the strongest non-FSM lever available: prepend the
parser's wire-envelope opener to the assistant turn so the model
continues INSIDE the tool call.

These tests pin the prefix builder + the route's wiring of it into
``chat_kwargs``. They are mock-based; the integration test with a live
model is in the smoke section of the PR body.
"""

from __future__ import annotations

from vllm_mlx.routes.chat import _forced_tool_call_prefix


def test_prefix_hermes_named_function():
    """Hermes is the largest parser family — wire opener is
    ``<tool_call>\\n{"name": "X", "arguments": ``."""
    out = _forced_tool_call_prefix("hermes", "get_weather")
    assert out is not None
    assert out.startswith("<tool_call>")
    assert '"name": "get_weather"' in out
    assert out.endswith('"arguments": ')


def test_prefix_qwen3coder():
    out = _forced_tool_call_prefix("qwen3_coder_xml", "lookup_user")
    assert out is not None
    assert "<tool_call>" in out
    assert '"name": "lookup_user"' in out


def test_prefix_llama_kimi_glm_minimax():
    for parser in ("llama", "kimi", "glm47", "minimax", "mistral"):
        out = _forced_tool_call_prefix(parser, "fn")
        assert out is not None, parser
        assert '"name": "fn"' in out


def test_prefix_channel_routed_returns_none():
    """Channel-routed parsers (harmony / gemma4) publish tool calls via
    the OutputRouter's tool-call channel — pre-pending the wire opener
    would confuse the channel state machine. Skip prefix injection."""
    assert _forced_tool_call_prefix("harmony", "fn") is None
    assert _forced_tool_call_prefix("gemma4", "fn") is None


def test_prefix_unknown_parser_returns_none():
    """Defensive: an unknown parser falls through to the post-parse
    synthesis fallback rather than guessing a wire shape."""
    assert _forced_tool_call_prefix(None, "fn") is None
    assert _forced_tool_call_prefix("future_parser_xyz", "fn") is None


def test_prefix_empty_name_returns_none():
    assert _forced_tool_call_prefix("hermes", "") is None
