# SPDX-License-Identifier: Apache-2.0
"""D-ANTHRO-TOOL-USAGE regressions — F3 + F5 from Sergei dogfood.

F3: ``tool_choice={"type":"any"}`` on /v1/messages was a silent no-op.
The adapter mapped it to OpenAI ``"required"`` but the Anthropic route
bypasses chat.py so the prompt-suffix + post-parse synth/422 enforcement
levers never fired. Tests in this module verify both the non-stream and
stream branches now mirror chat.py's behaviour.

F5: streaming ``message_start.usage.input_tokens`` was hard-coded to 0
because the prompt-token count wasn't computed until inside the engine
loop (i.e. AFTER message_start was already on the wire). The fix
pre-computes the count via the same ``build_prompt`` +
``count_prompt_tokens`` source of truth the non-stream adapter and the
context-length DoS gate use.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.routes.anthropic import (
    _enforce_required_tool_choice_present,
    _estimate_anthropic_prompt_tokens,
    _inject_tool_use_required_suffix,
    _is_required_tool_choice,
    _synthesize_anthropic_forced_tool_call,
    router,
)
from vllm_mlx.service.helpers import _TOOL_USE_REQUIRED_SUFFIX

# ──────────────────────────────────────────────────────────────────
# Engine doubles
# ──────────────────────────────────────────────────────────────────


class _Tokenizer:
    chat_template = "{% if add_generation_prompt %}<think>{% endif %}"

    def encode(self, text, add_special_tokens=True):
        # 1 token per word — deterministic and tokenizer-independent
        return [0] * (len(text.split()) + (1 if add_special_tokens else 0))


class _BaseEngine:
    preserve_native_tool_format = False
    is_mllm = False
    tokenizer = _Tokenizer()

    def build_prompt(self, messages, tools=None):
        # Concatenate role+content into a single string the tokenizer
        # can count. Tools are not rendered into the prompt body in this
        # double, only their NAMES contribute to the count — same shape
        # the production tokenizer would see after chat-template render.
        parts = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            )
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else "" for c in content
                )
            parts.append(f"{role}: {content}")
        if tools:
            for t in tools:
                fn = t.function if hasattr(t, "function") else t.get("function", {})
                name = fn.get("name") if isinstance(fn, dict) else fn
                if name:
                    parts.append(f"tool: {name}")
        return "\n".join(parts)


class _ToolStreamingEngine(_BaseEngine):
    """Streaming engine that lets us assert the route saw the
    pre-message_start ``prompt_tokens`` estimate AND that the
    enforcement levers (suffix injection, synth, error event) fire on
    the right ``tool_choice`` shapes."""

    def __init__(
        self,
        deltas: list[str],
        *,
        engine_prompt_tokens: int | None = None,
        engine_cached_tokens: int = 0,
        tool_calls_per_chunk: list[list] | None = None,
    ):
        self._deltas = deltas
        self._engine_prompt_tokens = engine_prompt_tokens
        self._engine_cached_tokens = engine_cached_tokens
        self._tool_calls_per_chunk = tool_calls_per_chunk or [[] for _ in deltas]
        self.stream_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        for i, text in enumerate(self._deltas):
            payload = {
                "new_text": text,
                "completion_tokens": i + 1,
                "cached_tokens": self._engine_cached_tokens,
                "tool_calls": self._tool_calls_per_chunk[i],
            }
            if self._engine_prompt_tokens is not None:
                payload["prompt_tokens"] = self._engine_prompt_tokens
            yield SimpleNamespace(**payload)

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        text = "".join(self._deltas)
        return SimpleNamespace(
            text=text,
            raw_text=text,
            reasoning_text="",
            completion_tokens=len(self._deltas),
            prompt_tokens=self._engine_prompt_tokens or 0,
            cached_tokens=self._engine_cached_tokens,
            tool_calls=self._tool_calls_per_chunk[-1] if self._deltas else [],
            finish_reason="stop",
            matched_stop=None,
        )


# ──────────────────────────────────────────────────────────────────
# Client factory + SSE helpers
# ──────────────────────────────────────────────────────────────────


def _make_client(engine: _BaseEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.reasoning_parser_name = None
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _parse_sse(response_text: str) -> list[dict]:
    events = []
    for raw_event in response_text.split("\n\n"):
        data_line = next(
            (line for line in raw_event.splitlines() if line.startswith("data: ")),
            None,
        )
        if not data_line:
            continue
        data = data_line.removeprefix("data: ")
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


# ──────────────────────────────────────────────────────────────────
# F3 — unit-level helpers
# ──────────────────────────────────────────────────────────────────


def test_required_tool_choice_predicate_matches_only_required_string():
    assert _is_required_tool_choice("required") is True
    assert _is_required_tool_choice("auto") is False
    assert _is_required_tool_choice("none") is False
    assert _is_required_tool_choice(None) is False
    assert (
        _is_required_tool_choice({"type": "function", "function": {"name": "x"}})
        is False
    )


def test_inject_suffix_for_required_appends_to_existing_system():
    tools = [SimpleNamespace(function={"name": "x"})]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    _inject_tool_use_required_suffix(messages, "required", tools=tools)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are helpful.")
    assert _TOOL_USE_REQUIRED_SUFFIX in messages[0]["content"]


def test_inject_suffix_for_required_prepends_system_when_absent():
    tools = [SimpleNamespace(function={"name": "x"})]
    messages = [{"role": "user", "content": "hi"}]
    _inject_tool_use_required_suffix(messages, "required", tools=tools)
    assert messages[0]["role"] == "system"
    assert _TOOL_USE_REQUIRED_SUFFIX.strip() in messages[0]["content"]


def test_inject_suffix_named_tool_uses_named_variant():
    tools = [SimpleNamespace(function={"name": "get_weather"})]
    messages = [{"role": "user", "content": "hi"}]
    _inject_tool_use_required_suffix(
        messages,
        {"type": "function", "function": {"name": "get_weather"}},
        tools=tools,
    )
    assert messages[0]["role"] == "system"
    assert "get_weather" in messages[0]["content"]


def test_inject_suffix_noop_when_no_tools():
    """Suffix is meaningless without tools to call — skip injection."""
    messages = [{"role": "user", "content": "hi"}]
    before = list(messages)
    _inject_tool_use_required_suffix(messages, "required", tools=None)
    assert messages == before
    _inject_tool_use_required_suffix(messages, "required", tools=[])
    assert messages == before


def test_inject_suffix_noop_when_tool_choice_is_auto():
    tools = [SimpleNamespace(function={"name": "x"})]
    messages = [{"role": "user", "content": "hi"}]
    before = list(messages)
    _inject_tool_use_required_suffix(messages, "auto", tools=tools)
    assert messages == before
    _inject_tool_use_required_suffix(messages, "none", tools=tools)
    assert messages == before
    _inject_tool_use_required_suffix(messages, None, tools=tools)
    assert messages == before


def test_enforce_required_synthesises_call_for_single_tool():
    """Single-tool unambiguous synthesis path: empty tool_calls →
    synthesised call with empty arguments + no error detail."""
    tools = [SimpleNamespace(function={"name": "get_weather"})]
    calls, err = _enforce_required_tool_choice_present([], "required", tools=tools)
    assert err is None
    assert len(calls) == 1
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == "{}"


def test_enforce_required_returns_error_for_multi_tool():
    tools = [
        SimpleNamespace(function={"name": "a"}),
        SimpleNamespace(function={"name": "b"}),
    ]
    calls, err = _enforce_required_tool_choice_present([], "required", tools=tools)
    assert calls == []
    assert err is not None
    assert "tool_choice" in err
    assert "no tool_calls" in err


def test_enforce_required_passes_through_existing_calls():
    tools = [SimpleNamespace(function={"name": "x"})]
    existing = [_synthesize_anthropic_forced_tool_call("x")]
    calls, err = _enforce_required_tool_choice_present(
        existing, "required", tools=tools
    )
    assert calls is existing
    assert err is None


def test_enforce_required_passes_through_when_choice_is_not_required():
    """auto / none / named-pin: enforcement is a no-op so the named-pin
    helper (which runs first) is responsible for its own contract."""
    tools = [SimpleNamespace(function={"name": "x"})]
    for tc in (
        "auto",
        "none",
        None,
        {"type": "function", "function": {"name": "x"}},
    ):
        calls, err = _enforce_required_tool_choice_present([], tc, tools=tools)
        assert calls == []
        assert err is None


# ──────────────────────────────────────────────────────────────────
# F3 — route-level (non-stream)
# ──────────────────────────────────────────────────────────────────


def _tool_dict(name: str, prop: str = "x"):
    return {
        "name": name,
        "description": f"Call {name}",
        "input_schema": {
            "type": "object",
            "properties": {prop: {"type": "string"}},
            "required": [prop],
        },
    }


def test_nonstream_tool_choice_any_synthesises_single_tool():
    """Non-stream + tool_choice=any + 1 tool + parser saw no calls →
    synthesise a tool_use for the sole tool. stop_reason=tool_use."""
    engine = _ToolStreamingEngine(
        ["I cannot help with that."],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": [_tool_dict("get_weather")],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Tell me a joke."}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stop_reason"] == "tool_use"
    tool_uses = [c for c in body["content"] if c["type"] == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "get_weather"


def test_nonstream_tool_choice_any_multi_tool_no_call_returns_422():
    """Non-stream + tool_choice=any + 2 tools + parser saw no calls →
    422 (ambiguous which tool to synthesise)."""
    engine = _ToolStreamingEngine(
        ["I cannot help."],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": [_tool_dict("get_weather"), _tool_dict("lookup_zip", "zip")],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Tell me a joke."}],
        },
    )
    assert r.status_code == 422, r.text
    body = r.json()
    # The test app mounts the router directly so the FastAPI default
    # ``{"detail": ...}`` envelope is what we see — production wraps
    # this into Anthropic's ``{"error": {"message": ...}}`` shape via
    # the server-level handler, which is unit-tested elsewhere.
    detail = body.get("detail") or body.get("error", {}).get("message", "")
    assert "tool_choice" in detail
    assert "no tool_calls" in detail or "any" in detail


def test_nonstream_tool_choice_any_injects_required_suffix():
    """The forced-call prompt suffix MUST reach the engine — same
    lever chat.py applies. This is the prompt-level half of the F3 fix."""
    engine = _ToolStreamingEngine(
        ["text"],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": [_tool_dict("get_weather")],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "anything"}],
        },
    )
    assert engine.chat_calls, "engine.chat was never invoked"
    seen = engine.chat_calls[0]["messages"]
    # System message is prepended (no prior system block); the suffix
    # text must appear somewhere in the rendered system content.
    assert any(
        (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
        == "system"
        and "MUST call" in (m.get("content") if isinstance(m, dict) else m.content)
        for m in seen
    ), seen


def test_nonstream_tool_choice_named_still_routes_to_specific_tool():
    """H-05 regression guard: ``tool_choice={"type":"tool","name":X}``
    must still pin THAT tool, not synthesise a different one. The new
    required-branch must not steal the named-pin enforcement."""
    engine = _ToolStreamingEngine(
        ["text"],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": [_tool_dict("get_weather"), _tool_dict("lookup_zip", "zip")],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "anything"}],
        },
    )
    # Named pin with parser-only model that returned text → 422
    # ("model returned a text response with no tool_calls"). Either
    # 422 or a 200 with a synth call to ``get_weather`` is contract-
    # compliant; what we explicitly do NOT want is a tool_use for
    # ``lookup_zip`` or a 200 ``end_turn`` text response.
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        body = r.json()
        tool_uses = [c for c in body["content"] if c["type"] == "tool_use"]
        for tu in tool_uses:
            assert tu["name"] == "get_weather", body
    else:
        body = r.json()
        detail = body.get("detail") or body.get("error", {}).get("message", "")
        assert "get_weather" in detail


def test_nonstream_tool_choice_auto_does_not_force_tool_call():
    """Regression guard: ``tool_choice={"type":"auto"}`` must NOT force
    a tool call. The model is allowed to return plain text and the
    response carries ``stop_reason=end_turn``."""
    engine = _ToolStreamingEngine(
        ["just text"],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": [_tool_dict("get_weather")],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "Tell me a joke."}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stop_reason"] == "end_turn"
    tool_uses = [c for c in body["content"] if c["type"] == "tool_use"]
    assert tool_uses == []


# ──────────────────────────────────────────────────────────────────
# F3 — route-level (stream)
# ──────────────────────────────────────────────────────────────────


def test_stream_tool_choice_any_single_tool_synthesises_tool_use():
    """Stream variant of the synthesise path: model emits text, route
    post-parses + synthesises the tool_use for the sole tool, and the
    terminal ``message_delta`` carries ``stop_reason=tool_use``."""
    engine = _ToolStreamingEngine(
        ["I ", "cannot ", "help."],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "tools": [_tool_dict("get_weather")],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Tell me a joke."}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    deltas = [e for e in events if e.get("type") == "message_delta"]
    assert deltas, events
    assert deltas[-1]["delta"]["stop_reason"] == "tool_use"
    tool_blocks = [
        e
        for e in events
        if e.get("type") == "content_block_start"
        and e.get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["content_block"]["name"] == "get_weather"


def test_stream_tool_choice_any_multi_tool_emits_error_event():
    """Stream variant of the multi-tool 422: SSE error event with
    ``invalid_request_error`` and terminal ``message_delta`` close-out."""
    engine = _ToolStreamingEngine(
        ["I ", "cannot ", "help."],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "tools": [_tool_dict("get_weather"), _tool_dict("lookup_zip", "zip")],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Tell me a joke."}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events, events
    assert error_events[0]["error"]["type"] == "invalid_request_error"
    assert "tool_choice" in error_events[0]["error"]["message"]
    # The forbidden text bytes must NOT have been streamed to the wire
    # — buffer-drop semantics same as the named-pin failure branch.
    text_deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
        and e.get("delta", {}).get("type") == "text_delta"
    ]
    assert "".join(text_deltas) == ""


# ──────────────────────────────────────────────────────────────────
# F5 — streaming usage.input_tokens
# ──────────────────────────────────────────────────────────────────


def test_estimate_prompt_tokens_uses_build_prompt():
    """Source of truth for F5: the helper goes through
    ``engine.build_prompt`` + tokenizer.encode and returns a > 0 count."""
    engine = _BaseEngine()
    messages = [{"role": "user", "content": "hello there world"}]
    n = _estimate_anthropic_prompt_tokens(engine, messages, tools=None)
    # The fake tokenizer returns 1 token per word + 1 BOS. Render is
    # "user: hello there world" → 5 tokens (4 words + BOS).
    assert n == 5, n


def test_estimate_prompt_tokens_zero_when_no_build_prompt():
    """MLLM engines (and minimal test stubs) have no ``build_prompt`` —
    the helper returns 0 and the route falls back to the engine-reported
    ``output.prompt_tokens`` as before. No 500."""

    class _NoBuildPromptEngine:
        is_mllm = False
        tokenizer = _Tokenizer()

    n = _estimate_anthropic_prompt_tokens(
        _NoBuildPromptEngine(), [{"role": "user", "content": "x"}], tools=None
    )
    assert n == 0


def test_estimate_prompt_tokens_zero_when_mllm():
    """MLLM engines: tokens are computed by the multimodal processor;
    skip text-only render to avoid a misleading estimate."""

    class _MLLMEngine(_BaseEngine):
        is_mllm = True

    n = _estimate_anthropic_prompt_tokens(
        _MLLMEngine(), [{"role": "user", "content": "x"}], tools=None
    )
    assert n == 0


def test_stream_message_start_carries_nonzero_input_tokens():
    """F5 primary repro: ``message_start.usage.input_tokens`` must
    reflect the pre-computed prompt-token count, NOT the hard-coded 0
    that under-reported the input share by 100%."""
    engine = _ToolStreamingEngine(
        ["Direct ", "answer"],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly please"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    starts = [e for e in events if e.get("type") == "message_start"]
    assert starts, events
    usage = starts[0]["message"]["usage"]
    assert usage["input_tokens"] > 0, usage
    # Must match what the pre-compute helper would have returned for
    # the same messages — i.e. byte-for-byte tied to the source of
    # truth (``build_prompt`` + tokenizer.encode).
    expected = _estimate_anthropic_prompt_tokens(
        engine,
        [{"role": "user", "content": "answer directly please"}],
        tools=None,
    )
    assert usage["input_tokens"] == expected


def test_stream_message_delta_accumulates_output_tokens():
    """``message_delta`` must report ``output_tokens > 0`` so cost
    accounting clients can rely on the SSE stream alone."""
    engine = _ToolStreamingEngine(
        ["one ", "two ", "three"],
        engine_prompt_tokens=5,
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "go"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    deltas = [e for e in events if e.get("type") == "message_delta"]
    assert deltas, events
    usage = deltas[-1]["usage"]
    assert usage["output_tokens"] == 3  # 3 chunks → 3 cumulative tokens
    # Final ``input_tokens`` is the (prompt - cached) non-cached share;
    # with the engine reporting cached=0 it equals the engine count.
    assert usage["input_tokens"] == 5


def test_stream_message_delta_input_tokens_floored_to_estimate_when_engine_silent():
    """When the engine never surfaces ``prompt_tokens`` on its chunks,
    the route uses the pre-computed estimate as a FLOOR so the
    terminal ``message_delta`` is no worse than ``message_start``."""
    engine = _ToolStreamingEngine(
        ["one ", "two"],
        engine_prompt_tokens=None,  # engine never sets prompt_tokens
    )
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "go now"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    starts = [e for e in events if e.get("type") == "message_start"]
    deltas = [e for e in events if e.get("type") == "message_delta"]
    start_input = starts[0]["message"]["usage"]["input_tokens"]
    delta_input = deltas[-1]["usage"]["input_tokens"]
    assert start_input > 0
    assert delta_input == start_input, (start_input, delta_input)


def test_stream_message_start_consistent_with_nonstream_input_tokens():
    """The streaming + non-streaming surfaces must agree on the prompt
    token count for the same prompt — this is the spec-level identity
    Sergei's evidence was checking. Engine reports a fixed prompt_tokens
    so the two surfaces should land on the same total non-cached share."""
    deltas = ["Direct ", "answer"]
    engine_stream = _ToolStreamingEngine(deltas, engine_prompt_tokens=11)
    client_s = _make_client(engine_stream)
    r_s = client_s.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "answer directly please"}],
        },
    )
    events = _parse_sse(r_s.text)
    delta_input = next(
        e for e in events if e.get("type") == "message_delta"
    )["usage"]["input_tokens"]

    engine_ns = _ToolStreamingEngine(deltas, engine_prompt_tokens=11)
    client_ns = _make_client(engine_ns)
    r_ns = client_ns.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "answer directly please"}],
        },
    )
    ns_body = r_ns.json()
    assert delta_input == ns_body["usage"]["input_tokens"]
