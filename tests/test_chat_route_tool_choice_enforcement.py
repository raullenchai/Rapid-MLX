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


@pytest.mark.parametrize("tools_field", [None, []])
def test_tool_choice_required_without_tools_returns_400(tools_field):
    """F-034: ``tool_choice="required"`` with ``tools`` absent or empty
    must surface as a 4xx error — the request is unsatisfiable, since
    "required" guarantees a tool_call but there is no tool to call.
    Pre-fix both cases silently 200'd as plain chat completions,
    masking a client bug. The Pydantic ``ValueError`` raised in
    ``ChatCompletionRequest._validate_tool_choice_against_tools``
    surfaces as HTTP 400 via ``vllm_mlx.middleware.exception_handlers``
    (Pydantic 422 → 400 OpenAI-shape envelope; same conversion the
    F-011 reasoning_max_tokens / nonfinite-sampling validators rely on).
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    payload: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required",
        "max_tokens": 32,
    }
    if tools_field is not None:
        payload["tools"] = tools_field
    resp = client.post("/v1/chat/completions", json=payload)
    # Direct Pydantic ValidationError surfaces as 422 from raw FastAPI;
    # the OpenAI-shape middleware in the real serve binary rewrites to
    # 400. This test harness mounts only the chat router (no middleware),
    # so accept either — the contract being pinned is "non-200 plus the
    # 'tool_choice=required requires non-empty tools' string".
    assert resp.status_code in (400, 422), resp.text
    assert "tool_choice='required'" in resp.text
    assert "non-empty 'tools'" in resp.text


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


def test_tool_choice_required_with_stream_passes_through_cloud_routing():
    """PR #518 round-8 codex BLOCKING #1: the streaming-required 422
    guard must NOT fire when the cloud router is configured to handle
    the request — cloud backends (e.g. GPT-4o) DO support
    ``required`` with streaming via decoder-side constraints. The
    guard now lives below the cloud routing block.

    Verifies the guard's PRE-cloud placement was wrong by simulating
    a cloud-routed request and asserting we don't 422 before cloud
    routing decides.
    """

    # Minimal cloud router stub that always claims it would route.
    # Sets ``stream_completion_called`` on first invocation so the
    # test can prove the cloud path actually executed (not just that
    # the local 422 didn't fire — round-9 codex NIT).
    cloud_called = {"stream": False}

    _CLOUD_SENTINEL_CONTENT = "CLOUD_SENTINEL_X"

    class _FakeCloudRouter:
        threshold = 0
        cloud_model = "gpt-4o"

        def should_route_to_cloud(self, _new_tokens):
            return True

        async def stream_completion(self, *args, **kwargs):
            cloud_called["stream"] = True
            yield (
                'data: {"choices":[{"delta":{"content":"'
                + _CLOUD_SENTINEL_CONTENT
                + '"}}]}\n\n'
            ).encode()
            yield b"data: [DONE]\n\n"

        async def completion(self, *args, **kwargs):
            return {"choices": [{"message": {"content": "x"}}]}

    engine = _RecordingEngine()
    engine.estimate_new_tokens = lambda prompt: (10, 5)  # noqa: E731
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.tool_call_parser = "hermes"
    cfg.cloud_router = _FakeCloudRouter()
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "stream": True,
            "max_tokens": 32,
        },
    )
    # Must reach the cloud path: 200 OK + cloud's sentinel content in
    # the body + the cloud router's stream_completion was invoked.
    assert resp.status_code == 200, resp.text
    assert _CLOUD_SENTINEL_CONTENT in resp.text, (
        f"cloud sentinel missing — local path was hit instead: {resp.text[:300]}"
    )
    assert cloud_called["stream"], (
        "cloud router's stream_completion was never called — local 422 fired "
        "or the request silently fell back to local inference"
    )


def test_tool_choice_required_with_stream_no_parser_returns_422():
    """PR #518 round-9 codex BLOCKING: the streaming-required 422 now
    fires ONLY when no streaming tool-call parser is configured.
    Without a parser the server has no path to emit tool_calls in SSE,
    so the OpenAI ``tool_call guaranteed`` contract can never be met.
    """
    engine = _RecordingEngine()
    # tool_call_parser=None explicitly — no streaming tool-call path.
    client = _make_client(engine, tool_call_parser=None)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "stream": True,
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 422, resp.text
    assert "tool-call parser" in resp.text
    # Verify the engine was never invoked — we rejected before the
    # streaming handler opened.
    assert engine.last_chat_kwargs is None


def test_tool_choice_required_with_stream_channel_routed_bypasses_422(monkeypatch):
    """PR #518 round-10 codex BLOCKING #1: when no text parser is set
    but the engine has channel-routed tool-call capability (harmony /
    Gemma 4), the streaming-required 422 must NOT fire — the
    OutputRouter's tool channel is the production path that emits
    structured tool_calls for those models. Monkeypatch the capability
    probe to True so we exercise the bypass deterministically without
    needing a real harmony tokenizer in the test fixture.
    """
    from vllm_mlx.routes import chat as chat_module

    engine = _RecordingEngine()
    monkeypatch.setattr(
        chat_module,
        "_engine_supports_channel_routed_tool_calls",
        lambda _e: True,
    )
    # No text parser — pre-round-10 this combination always 422'd.
    client = _make_client(engine, tool_call_parser=None)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "stream": True,
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text


def test_engine_supports_channel_routed_helper_returns_false_on_no_tokenizer():
    """The capability probe must return ``False`` (not raise) when the
    engine has no tokenizer attribute or it is ``None`` — the gate
    must fall back to the parser-only path safely.
    """
    from vllm_mlx.routes.chat import _engine_supports_channel_routed_tool_calls

    class _NoTokenizerEngine:
        tokenizer = None

    assert _engine_supports_channel_routed_tool_calls(_NoTokenizerEngine()) is False


def test_engine_supports_channel_routed_helper_returns_true_for_harmony_router(
    monkeypatch,
):
    """When ``OutputRouter.from_tokenizer_for_streaming`` returns a
    router whose ``format_tag`` is in the engine allowlist (harmony /
    gemma4), the helper returns True so the streaming-required gate
    lets the request through.
    """
    from types import SimpleNamespace

    from vllm_mlx.output_router import OutputRouter
    from vllm_mlx.routes.chat import _engine_supports_channel_routed_tool_calls

    fake_router = SimpleNamespace(map=SimpleNamespace(format_tag="harmony"))
    monkeypatch.setattr(
        OutputRouter,
        "from_tokenizer_for_streaming",
        classmethod(lambda cls, tokenizer, **kw: fake_router),
    )

    class _HarmonyEngine:
        tokenizer = object()

    assert _engine_supports_channel_routed_tool_calls(_HarmonyEngine()) is True


def test_engine_supports_channel_routed_helper_returns_false_for_unsupported_format(
    monkeypatch,
):
    """Format tags outside the engine allowlist (e.g. ``think_tag``)
    must NOT trip the bypass — those routers don't emit structured
    tool calls, so the streaming-required 422 still needs to fire.
    """
    from types import SimpleNamespace

    from vllm_mlx.output_router import OutputRouter
    from vllm_mlx.routes.chat import _engine_supports_channel_routed_tool_calls

    fake_router = SimpleNamespace(map=SimpleNamespace(format_tag="think_tag"))
    monkeypatch.setattr(
        OutputRouter,
        "from_tokenizer_for_streaming",
        classmethod(lambda cls, tokenizer, **kw: fake_router),
    )

    class _ThinkTagEngine:
        tokenizer = object()

    assert _engine_supports_channel_routed_tool_calls(_ThinkTagEngine()) is False


def test_tool_choice_required_with_stream_and_parser_passes():
    """PR #518 round-9 codex BLOCKING: when a streaming tool-call
    parser IS configured (e.g. hermes), the streaming path CAN emit
    tool_calls and the 422 must NOT fire. Local inference still
    can't decoder-enforce, but prompt injection + parser is the
    best lever we have and was the documented behavior pre-PR.
    """
    engine = _RecordingEngine()
    # ``_make_client`` defaults tool_call_parser="hermes".
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "stream": True,
            "max_tokens": 32,
        },
    )
    # The stream may emit text-only output (model didn't comply with
    # prompt injection) — but the server must NOT block the request
    # upfront; the parser path is the agreed enforcement mechanism.
    assert resp.status_code == 200, resp.text


def test_tool_choice_required_without_stream_still_works():
    """Symmetric pin: ``required`` + ``stream=false`` must still
    succeed (covered by the existing happy-path test, restated here
    so the 422 above doesn't accidentally regress the non-stream
    path during future refactors).
    """
    engine = _ToolCallingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "stream": False,
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text


def test_tool_choice_named_with_stream_passes_through():
    """The named-function form is still allowed under streaming —
    the filtered tools list makes the wrong-tool case much rarer
    (the model only sees one tool), even though we can't 422
    mid-stream if it still produces text. Documented limitation
    in the 422 message for ``required``.
    """
    engine = _ToolCallingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "stream": True,
            "max_tokens": 32,
        },
    )
    # Streaming response: 200 OK + SSE body. Test passes if we got
    # past the upfront-reject gate.
    assert resp.status_code == 200, resp.text


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


# ======================================================================
# Issue #571 — parser-path symmetry for forced ``tool_choice``
#
# Pre-#571: text-parser engines (hermes / qwen3_coder / minimax / glm47)
# 422'd on ``tool_choice="required"`` and on the named-function form
# whenever the model produced text the parser didn't recognise.
# Channel-routed engines (harmony / gemma4) succeeded on identical
# requests because the ``OutputRouter`` lifts structured tool_calls out
# of a dedicated channel. The asymmetry broke clients (Cline, Codex,
# OpenAI SDKs) that hard-code a forced ``tool_choice``: same wire
# format, parser-dependent outcome.
#
# Fix: synthesise a tool_call server-side when the target tool is
# unambiguous — named-function names it, or ``"required"`` paired with
# a single tool entry resolves it. ``"required"`` with multiple tools
# is genuinely ambiguous and still 422s with the pre-#571 message.
# ======================================================================


def test_required_with_single_tool_synthesizes_when_parser_empty():
    """#571: ``tool_choice="required"`` with EXACTLY ONE tool and a
    parser that returned no tool_calls must synthesise a call to that
    sole tool. Without this, hermes-class engines 422 on the same
    request that harmony returns 200 for, breaking OpenAI's
    parser-agnostic ``tool_choice`` contract.
    """
    engine = _RecordingEngine()  # parser path returns empty
    client = _make_client(engine)
    single_tool = [_TOOLS_FIXTURE[0]]  # just get_weather
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi."}],
            "tools": single_tool,
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tcs = body["choices"][0]["message"].get("tool_calls") or []
    assert len(tcs) == 1, f"expected 1 synthesized tool_call, got {len(tcs)}"
    assert tcs[0]["function"]["name"] == "get_weather"
    # Arguments default to ``"{}"`` — the contract guarantee is "a
    # tool_call is present", not "the arguments are correct".
    assert tcs[0]["function"]["arguments"] == "{}"


def test_required_with_multiple_tools_still_422_when_parser_empty():
    """#571: ``tool_choice="required"`` with MULTIPLE tools and an
    empty parser output is genuinely ambiguous — the route can't pick
    a winner, so the pre-#571 422 still fires. Pins the boundary so
    future refactors don't silently default to ``tools[0]``.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi."}],
            "tools": _TOOLS_FIXTURE,  # two tools
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 422, resp.text
    assert "required" in resp.text.lower()


def test_named_function_synthesizes_when_parser_empty():
    """#571: ``tool_choice={"type":"function","function":{"name":X}}``
    with an empty parser output must synthesise a call to X. The
    target tool is named in the request — there's no ambiguity. This
    is the precise case from issue #571's repro
    (``tool_choice={type:function,function:{name:web_search}}`` →
    422 on hermes, 200 on harmony).
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi."}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tcs = body["choices"][0]["message"].get("tool_calls") or []
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "get_time"
    assert tcs[0]["function"]["arguments"] == "{}"


def test_named_function_wrong_call_still_422_after_571():
    """#571 regression: synthesis fires ONLY when the parser returned
    NOTHING. When the model actively defied the choice and called a
    different tool, the route must still 422 — silently dropping the
    model's real output and substituting our synthesised stub would
    be a worse client experience than the explicit failure.
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
    assert "get_time" in resp.text


def test_harmony_path_unchanged_by_571(monkeypatch):
    """#571 regression: the harmony / channel-routed path already
    succeeded by emitting structured tool_calls — synthesis must NOT
    fire there (it would double the response). Verified by feeding
    the route an engine that surfaces ``GenerationOutput.tool_calls``
    directly (the harmony surface) and asserting the result carries
    exactly one call with the engine's name + arguments, not the
    synthesised ``"{}"`` stub.
    """
    # ``_ToolCallingEngine`` already surfaces structured tool_calls
    # via ``GenerationOutput.tool_calls`` — the same passthrough
    # ``HarmonyStreamingRouter`` uses (#515).
    engine = _ToolCallingEngine(fn_name="get_weather", arguments='{"city":"Tokyo"}')
    # No text parser configured — exact harmony shape on the route's
    # perspective: structured surface, parser path bypassed.
    client = _make_client(engine, tool_call_parser=None)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": _TOOLS_FIXTURE,
            "tool_choice": "required",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tcs = body["choices"][0]["message"].get("tool_calls") or []
    assert len(tcs) == 1
    # Engine-emitted, NOT synthesised — args must carry the engine's
    # payload, not the ``"{}"`` stub.
    assert tcs[0]["function"]["name"] == "get_weather"
    assert tcs[0]["function"]["arguments"] == '{"city":"Tokyo"}'


def test_synthesize_forced_tool_call_helper_shape():
    """Pin the synthesised wire shape: id prefix, type=function,
    function.name = target, function.arguments = string (JSON-encoded).
    Stops future refactors from accidentally emitting ``arguments`` as
    a dict (the OpenAI ToolCall spec uses a JSON string).
    """
    from vllm_mlx.routes.chat import _synthesize_forced_tool_call

    tc = _synthesize_forced_tool_call("get_weather")
    assert tc.id.startswith("call_")
    assert tc.type == "function"
    assert tc.function.name == "get_weather"
    assert isinstance(tc.function.arguments, str)
    assert tc.function.arguments == "{}"

    # Custom arguments are passed through verbatim — the helper does
    # not parse or re-serialise, matching the model-emitted shape.
    tc2 = _synthesize_forced_tool_call("calc", arguments='{"expr":"1+1"}')
    assert tc2.function.arguments == '{"expr":"1+1"}'


def test_named_function_outside_tools_returns_422(monkeypatch):
    """Codex R1 BLOCKING (#675): the named-function synthesis branch
    must never fabricate a call to a tool the client did not submit.

    The early prompt-level validation (~chat.py:488) already 400s when
    ``tool_choice`` names a function absent from ``request.tools``, but
    that gate sits hundreds of lines upstream from the post-parse
    synthesis branch — a future refactor could shift or bypass it, at
    which point the synthesis path would mint a call to
    ``ghost_function`` and ship a 200 with a fabricated tool_call to a
    tool the client never defined.

    This test exercises the defense-in-depth at the synthesis branch
    DIRECTLY by monkeypatching ``_parse_tool_calls_with_parser`` (which
    runs between the early gate and the synthesis branch, with the
    ``request`` object in scope) to clear ``request.tools`` and rewrite
    ``request.tool_choice`` to a ghost target — simulating a future
    bypass of the early 400 gate. The synthesis branch must then refuse
    rather than fabricating, surfacing as 422.
    """
    from vllm_mlx.routes import chat as chat_module

    synth_calls: list[str] = []
    real_synth = chat_module._synthesize_forced_tool_call

    def _spy(name, arguments="{}"):
        synth_calls.append(name)
        return real_synth(name, arguments)

    monkeypatch.setattr(chat_module, "_synthesize_forced_tool_call", _spy)

    # Simulate the BLOCKING scenario: by the time the synthesis branch
    # runs, ``request.tools`` no longer contains ``_target``. We
    # achieve this by hooking ``_parse_tool_calls_with_parser`` — it
    # runs after the early gate and has the ``request`` in scope, so
    # we can mutate ``request.tools`` and ``request.tool_choice`` to
    # the post-bypass state. Returns ``("", None)`` to match the empty
    # parser-output path that triggers synthesis.
    real_parse = chat_module._parse_tool_calls_with_parser

    def _bypass_then_parse(text, request, structured_tool_calls=None):
        # Rewrite ``tool_choice`` to a ghost target while leaving
        # ``request.tools`` intact (it still contains ``real_function``)
        # so the synthesis branch's outer guard (line ~1212:
        # ``request.tool_choice is not None and request.tools``) lets
        # us in — and the new ``_target_is_submitted`` check must then
        # fail closed (422), not synthesise.
        request.tool_choice = {
            "type": "function",
            "function": {"name": "ghost_function"},
        }
        return real_parse(text, request, structured_tool_calls=structured_tool_calls)

    monkeypatch.setattr(
        chat_module, "_parse_tool_calls_with_parser", _bypass_then_parse
    )

    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "do it"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "real_function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            # Early-gate-friendly: real_function IS in tools so the
            # upstream 400 doesn't fire; the parser hook then rewrites
            # state to the BLOCKING scenario.
            "tool_choice": {
                "type": "function",
                "function": {"name": "real_function"},
            },
            "max_tokens": 32,
        },
    )
    # Defense-in-depth must produce 422 — NEVER 200 with a fabricated
    # call to ``ghost_function``.
    assert resp.status_code == 422, resp.text
    assert "ghost_function" in resp.text
    # The critical assertion: the synthesis helper must NEVER be
    # invoked with the ghost name. Without the BLOCKING fix, this
    # would record ``["ghost_function"]`` and the response would be a
    # 200 with a fabricated call.
    assert "ghost_function" not in synth_calls, (
        f"BLOCKING REGRESSION: _synthesize_forced_tool_call was called "
        f"with ghost target; calls={synth_calls!r}"
    )
