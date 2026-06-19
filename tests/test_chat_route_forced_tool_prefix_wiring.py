# SPDX-License-Identifier: Apache-2.0
"""Route-level wiring test for the forced ``tool_choice`` assistant-prefix
injection lever.

The chat route MUST pass ``forced_assistant_prefix`` to the engine's
``chat()`` / ``stream_chat()`` whenever:

  * ``tool_choice == {"type":"function","function":{"name":X}}``, OR
  * ``tool_choice == "required"`` AND there is exactly one tool (the
    named-tool form's unambiguous sibling).

Conversely, the prefix MUST NOT be passed when:
  * ``tool_choice`` is "auto" / "none" / unset.
  * The parser is channel-routed (harmony / gemma4) — those publish tool
    calls via the OutputRouter and the prefix would break the channel
    state machine.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _RecordingEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.last_chat_kwargs: dict[str, Any] | None = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_chat_kwargs = kwargs
        # Return synthesized tool call so the post-parse enforcement
        # gate is satisfied (we only care about ``forced_assistant_prefix``).
        return GenerationOutput(
            text='<tool_call>\n{"name": "get_weather", "arguments": {}}</tool_call>',
            raw_text='<tool_call>\n{"name": "get_weather", "arguments": {}}</tool_call>',
            prompt_tokens=4,
            completion_tokens=8,
            finished=True,
            finish_reason="stop",
        )


def _make_client(engine, parser="hermes"):
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = parser
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
]


def test_forced_named_function_sets_assistant_prefix():
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs is not None
    prefix = engine.last_chat_kwargs.get("forced_assistant_prefix")
    assert prefix is not None
    assert prefix.startswith("<tool_call>")
    assert '"name": "get_weather"' in prefix


def test_forced_required_with_solo_tool_sets_prefix():
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_TOOLS[0]],
            "tool_choice": "required",
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    prefix = engine.last_chat_kwargs.get("forced_assistant_prefix")
    assert prefix is not None
    assert '"name": "get_weather"' in prefix


def test_required_with_multiple_tools_no_prefix():
    """``required`` with multiple tools is ambiguous — we don't pick a
    function for the model. Post-parse enforcement still fires."""
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": "required",
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs.get("forced_assistant_prefix") is None


def test_auto_choice_no_prefix():
    engine = _RecordingEngine()
    client = _make_client(engine, parser="hermes")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs.get("forced_assistant_prefix") is None


def test_channel_routed_parser_no_prefix():
    """``harmony`` / ``gemma4`` are channel-routed — no prefix even when
    a function is forced. Prefix would confuse the channel state machine."""
    engine = _RecordingEngine()
    client = _make_client(engine, parser="harmony")
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": _TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs.get("forced_assistant_prefix") is None
