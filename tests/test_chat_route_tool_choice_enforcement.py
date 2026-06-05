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
from vllm_mlx.routes.chat import _tool_call_name
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


def _make_client(
    engine: _RecordingEngine, tool_call_parser: str | None = "hermes"
) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    # Suffix injection at chat.py is gated on ``cfg.tool_call_parser`` being
    # set; default to ``"hermes"`` so the ``tool_choice`` enforcement tests
    # exercise the injection branch. Pass ``None`` explicitly to assert the
    # alternate path.
    cfg.tool_call_parser = tool_call_parser
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

    Uses ``_ToolCallingEngine`` (returns matching tool_call) so the
    post-parse #468 enforcement gate (added in the same change) does
    not fire — the engine HAS to return the right call for this test
    to isolate the tools-filter behavior; a separate test covers the
    enforcement gate failure mode.
    """
    engine = _ToolCallingEngine(fn_name="get_time")
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


@pytest.mark.parametrize("choice", ["auto", None])
def test_tool_choice_auto_and_unset_leave_tools_untouched(choice):
    """``"auto"`` and unset ``tool_choice`` must NOT mutate
    ``request.tools``. ``"required"`` has its own enforcement path
    (post-parse 422 / suffix injection — see #468 tests below) and is
    excluded from this parametrize.
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


# ──────────────────────────────────────────────────────────────────
# ``tool_choice="required"`` enforcement (#468)
# ──────────────────────────────────────────────────────────────────


class _ToolCallingEngine(_RecordingEngine):
    """Mock engine that injects an engine-surfaced structured tool_call
    so the route's parse path emits a real ``tool_calls`` field.

    The route consumes ``GenerationOutput.tool_calls`` via the PR #515
    structured passthrough (``_parse_tool_calls_with_parser`` honors
    ``structured_tool_calls=...``), so we can simulate a model that
    actually called a tool without standing up a full parser fixture.
    """

    def __init__(
        self, fn_name: str = "get_weather", arguments: str = '{"city":"Tokyo"}'
    ):
        super().__init__()
        self._fn_name = fn_name
        self._arguments = arguments

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text="",
            raw_text="",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="tool_calls",
            tool_calls=[{"name": self._fn_name, "arguments": self._arguments}],
        )


def test_tool_choice_required_empty_response_returns_422():
    """``tool_choice="required"`` with a model that returned text-only
    output (zero tool_calls) must surface as 422, not a silent
    content-bearing 200 (#468). The OpenAI spec guarantees a tool_call
    is present in the response when ``required`` is set.
    """
    engine = _RecordingEngine()  # default: text="ok", tool_calls=None
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 422, resp.text
    assert "required" in resp.text.lower()


def test_tool_choice_required_with_tool_call_returns_200():
    """When the model DID emit a tool_call, ``tool_choice="required"``
    must succeed cleanly and forward the call to the client. Pins the
    happy path so the 422 gate doesn't accidentally fire on success.
    """
    engine = _ToolCallingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tcs = body["choices"][0]["message"].get("tool_calls") or []
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "get_weather"


def test_tool_choice_required_injects_strict_system_suffix():
    """The ``required`` mode must inject the strict suffix
    (``_TOOL_USE_REQUIRED_SUFFIX``) into the system prompt — this is
    the strongest enforcement lever in the absence of FSM-constrained
    decoding (#132). Verify the suffix lands by inspecting what the
    engine receives in ``last_messages``.
    """
    engine = _ToolCallingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    sys_msgs = [
        m
        for m in engine.last_messages
        if (m.get("role") if isinstance(m, dict) else m.role) == "system"
    ]
    assert sys_msgs, "expected an auto-injected system message"
    content = (
        sys_msgs[0]["content"] if isinstance(sys_msgs[0], dict) else sys_msgs[0].content
    )
    assert "MUST call one of the provided tools" in content


def test_tool_choice_named_function_wrong_call_returns_422():
    """``tool_choice={"type":"function","function":{"name":X}}`` with a
    model that called a DIFFERENT tool must 422. The non-stream path
    filters ``request.tools`` to the named function pre-flight (so the
    model only sees that one), but a malicious / drifted parser could
    still surface a different call from the engine — the route must
    refuse to forward a contract-violating response.
    """
    # Engine returns get_time even though tool_choice pins get_weather.
    engine = _ToolCallingEngine(fn_name="get_time", arguments='{"city":"Tokyo"}')
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Pick a function"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 422, resp.text
    assert "get_weather" in resp.text


class _MultiCallEngine(_RecordingEngine):
    """Mock engine that emits TWO tool_calls: the target plus an
    extra. Used to verify the round-4 BLOCKING #2 ``all(...)`` gate.
    """

    def __init__(self, *, target_name: str, extra_name: str):
        super().__init__()
        self._target = target_name
        self._extra = extra_name

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text="",
            raw_text="",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="tool_calls",
            tool_calls=[
                {"name": self._target, "arguments": "{}"},
                {"name": self._extra, "arguments": "{}"},
            ],
        )


def test_tool_choice_named_function_extra_call_returns_422():
    """PR #518 round-4 codex BLOCKING #2: ``tool_choice`` with
    ``function.name = X`` allows ONLY X. A response carrying X PLUS
    an extra call (e.g. ``[X, Y]``) is a contract violation — the
    prior ``any(...)`` gate accepted it silently.
    """
    engine = _MultiCallEngine(target_name="get_weather", extra_name="get_time")
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 422, resp.text
    assert "get_time" in resp.text


def test_tool_choice_named_function_injects_named_suffix():
    """The named-function form must inject a suffix that names the
    target tool explicitly — without this, the model may still pick
    a different tool from the (now-filtered) singleton list.
    """
    engine = _ToolCallingEngine()  # returns get_weather
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    sys_msgs = [
        m
        for m in engine.last_messages
        if (m.get("role") if isinstance(m, dict) else m.role) == "system"
    ]
    assert sys_msgs
    content = (
        sys_msgs[0]["content"] if isinstance(sys_msgs[0], dict) else sys_msgs[0].content
    )
    assert "'get_weather'" in content
    assert "MUST call the tool named" in content


def test_tool_choice_auto_keeps_loose_suffix():
    """``tool_choice="auto"`` (and unset) must keep the original
    ``_TOOL_USE_SYSTEM_SUFFIX`` (loose) — NOT the strict required
    variant. This pins the gate so the strict suffix doesn't bleed
    into auto-mode requests where the model is allowed to opt out.
    """
    engine = _ToolCallingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    sys_msgs = [
        m
        for m in engine.last_messages
        if (m.get("role") if isinstance(m, dict) else m.role) == "system"
    ]
    assert sys_msgs
    content = (
        sys_msgs[0]["content"] if isinstance(sys_msgs[0], dict) else sys_msgs[0].content
    )
    # The loose suffix says "When the user's request can be answered ..."
    assert "When the user's request can be answered" in content
    # The strict suffix would say "MUST call one of the provided tools" — verify absent.
    assert "MUST call one of the provided tools" not in content


# ======================================================================
# _tool_call_name shape-agnostic extraction (PR #518 round-2 BLOCKING)
# ======================================================================


class _AttrFunction:
    def __init__(self, name):
        self.name = name


class _AttrToolCall:
    def __init__(self, name):
        self.function = _AttrFunction(name)


def test_tool_call_name_handles_attr_shape():
    """Pydantic ``ToolCall`` instances expose ``function.name`` as
    attribute access — text-parser path output.
    """
    assert _tool_call_name(_AttrToolCall("get_weather")) == "get_weather"


def test_tool_call_name_handles_dict_shape():
    """Raw dict from engine structured passthrough — both outer and
    inner are dicts. Pure-attr access used to return None here and
    silently 422 matching named-function calls.
    """
    tc = {
        "id": "call_a",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }
    assert _tool_call_name(tc) == "get_weather"


def test_tool_call_name_handles_mixed_shape():
    """Outer attr-object whose ``function`` is a dict (or vice versa)
    — the helper must walk both layers independently.
    """

    class _OuterAttr:
        function = {"name": "get_time"}

    assert _tool_call_name(_OuterAttr()) == "get_time"

    outer_dict = {"function": _AttrFunction("get_date")}
    assert _tool_call_name(outer_dict) == "get_date"


def test_tool_call_name_handles_flat_dict_shape():
    """PR #518 round-3 codex BLOCKING: raw engine
    ``GenerationOutput.tool_calls`` is ``[{"name": ..., "arguments":
    ...}]`` — no ``function`` wrapper. ``_ToolCallingEngine`` in this
    file emits exactly that shape, so the helper must extract the
    name directly from the top-level dict.
    """
    tc = {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}
    assert _tool_call_name(tc) == "get_weather"


def test_tool_call_name_handles_flat_attr_shape():
    """Symmetric attr-shape: outer object exposes ``.name`` directly
    (no ``.function``). Future passthrough surfaces may use this.
    """
    obj = _AttrFunction("get_weather")
    assert _tool_call_name(obj) == "get_weather"


def test_tool_call_name_returns_none_when_no_name_anywhere():
    """Defensive guard: a tool_call lacking ``function`` AND
    ``name`` returns None instead of raising mid-422-check.
    """
    assert _tool_call_name({}) is None
    assert _tool_call_name(object()) is None
