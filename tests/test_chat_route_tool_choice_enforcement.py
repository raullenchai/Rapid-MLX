# SPDX-License-Identifier: Apache-2.0
"""Prompt-level ``tool_choice`` enforcement on the chat route (#445).

Pre-fix: ``tool_choice`` was forwarded only to the cloud router — local
inference accepted the field but never enforced it, so ``"none"`` and
specific-function modes silently behaved as ``"auto"``. On harmony/hybrid
models this meant a user explicitly disabling tools (``"none"``) still
got tool_calls back, and a user forcing function X got whatever the
model preferred instead.

Fix (``routes/chat.py``): before any downstream consumption of
``request.tools`` (suffix injection at line ~398, chat-template
``tools=`` kwarg at line ~474, build_prompt at line ~483), normalize
the request based on ``tool_choice``:

- ``"none"``: ``request.tools = None`` — the model never sees tools,
  no suffix injection fires, the chat template renders without tools.
- ``{"type":"function","function":{"name":X}}``: filter ``request.tools``
  to only the named function. Raises HTTP 400 if X is not in
  ``request.tools`` (matches OpenAI spec).
- ``"auto"`` / ``"required"`` / ``None``: tools untouched. ``"required"``
  enforcement is tracked separately under #442 (needs decoder-level
  constraints).

These tests fire HTTP requests through the live FastAPI router and
assert on what the engine receives in ``chat_kwargs``. A unit-level
inspection of the request object alone would miss regressions where
the normalization happens but ``chat_kwargs["tools"]`` is later set
from a stale local reference instead of ``request.tools``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _RecordingEngine:
    """Mock engine that captures the kwargs passed to ``chat()``.

    Returns a vanilla GenerationOutput so the route's post-processing
    runs without errors. We only care about what the engine sees on
    its way in.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.last_chat_kwargs: dict[str, Any] | None = None
        self.last_messages: Any = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )


def _make_client(engine: _RecordingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


_TOOLS_FIXTURE = [
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


def test_tool_choice_none_strips_tools_from_engine_kwargs():
    """``tool_choice: "none"`` must drop ``tools`` from the engine call
    entirely. Without this the model still sees the tools in the chat
    template and ignores the choice. Spec: "none" means do NOT call
    any tool — model must respond in natural language.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "what's the weather in Tokyo?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "none",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs is not None
    assert "tools" not in engine.last_chat_kwargs or not engine.last_chat_kwargs.get(
        "tools"
    ), (
        f"tool_choice='none' must strip tools from engine call; "
        f"got tools={engine.last_chat_kwargs.get('tools')!r}"
    )


def test_tool_choice_specific_function_filters_to_named_only():
    """``tool_choice: {type: function, function: {name: X}}`` must
    filter ``tools`` to a single entry — the named function. Without
    this the model sees both candidates and picks the one it prefers,
    silently ignoring the user's force choice.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "what time is it in Tokyo?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_time"},
            },
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    tools = engine.last_chat_kwargs.get("tools") if engine.last_chat_kwargs else None
    assert tools and len(tools) == 1, (
        f"specific-function tool_choice must filter to one tool; "
        f"got {len(tools) if tools else 0} tool(s)"
    )
    # The chat template converter may reshape the tool, but the
    # function name must survive intact.
    fn = tools[0].get("function") if isinstance(tools[0], dict) else None
    name = fn.get("name") if isinstance(fn, dict) else None
    assert name == "get_time", f"filtered tool must be 'get_time'; got {name!r}"


def test_tool_choice_specific_function_unknown_name_returns_400():
    """Specific-function ``tool_choice`` for a name NOT in ``tools``
    is an invalid request — return 400 per OpenAI spec rather than
    silently delivering an empty tool list to the model (which would
    look like an internal server error to clients).
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "do something"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {
                "type": "function",
                "function": {"name": "nonexistent_function"},
            },
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 400
    assert "nonexistent_function" in resp.text


def test_tool_choice_function_missing_name_returns_400():
    """``tool_choice: {type: function}`` with no ``function.name`` is
    malformed per OpenAI spec — return 400 explicitly rather than
    silently treating it as ``auto``.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "do something"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "function", "function": {}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("choice", ["auto", "required", None])
def test_tool_choice_auto_required_none_field_leave_tools_untouched(choice):
    """``"auto"``, ``"required"``, and unset ``tool_choice`` must NOT
    mutate ``request.tools``. ``"required"`` enforcement is tracked
    separately (#442 needs FSM constraints); for now it falls through
    to model behavior — same as ``"auto"``. This test pins that the
    normalization layer is precisely scoped to ``"none"`` and the
    specific-function form and doesn't accidentally rewrite the other
    paths.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    payload: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "what's the weather?"}],
        "tools": _TOOLS_FIXTURE,
        "max_tokens": 32,
    }
    if choice is not None:
        payload["tool_choice"] = choice
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
    tools = engine.last_chat_kwargs.get("tools") if engine.last_chat_kwargs else None
    assert tools and len(tools) == 2, (
        f"tool_choice={choice!r} must leave the full tools array intact; "
        f"got {len(tools) if tools else 0} tool(s)"
    )


def test_tool_choice_none_with_no_tools_is_noop():
    """``tool_choice: "none"`` on a request that has no tools to begin
    with is a no-op — the request must succeed cleanly without
    raising 400 or otherwise misbehaving. Pins the defensive branch
    that checks ``request.tools`` before mutation.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text


def test_tool_choice_specific_function_with_no_tools_returns_400():
    """``tool_choice: {type:function, function:{name:X}}`` with no
    ``tools`` array (or empty) is malformed — must return 400 instead
    of silently passing through. Codex round-1 review of v0.6.66
    caught the earlier guard ``if tc is not None and request.tools:``
    skipping validation entirely for the no-tools case. Pin the fix
    so future refactors don't reintroduce silent acceptance.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    # No ``tools`` key at all.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "do it"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 400
    assert "get_weather" in resp.text
    # Empty ``tools`` array — same outcome.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "do it"}],
            "tools": [],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 400
    assert "get_weather" in resp.text


def test_tool_choice_function_missing_name_with_no_tools_returns_400():
    """The ``function:{}`` malformed-payload check must also fire when
    there are no ``tools`` — the validation is unconditional on the
    ``tool_choice`` shape, not gated by ``tools`` truthiness.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "do it"}],
            "tool_choice": {"type": "function", "function": {}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 400
