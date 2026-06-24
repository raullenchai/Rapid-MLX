# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #591 — Codex CLI integration follow-ups (PR #588).

Covers the three follow-ups from `gh issue view 591` that survived a
git-log audit (the other four were already fixed in subsequent PRs):

  * Item 2 (HIGH): channel-routed engines emitting on the ``tool_call``
    channel must NOT leak the JSON argument bytes into
    ``response.output_text.delta``. Earlier code routed the channel
    through ``_emit_text_delta`` which works only because every
    channel-emitting engine today populates ``output.tool_calls`` (and
    the structured-tool-calls branch ``continue``s before reaching the
    channel router). A future channel-only engine would leak.
  * Item 5 (P2): ``ResponsesRequest.model`` rejects the empty string at
    the Pydantic layer with a 422 instead of letting it slip past the
    ``gpt-*`` / ``claude-*`` startswith bypass and 400 from the route.
  * Item 6 (P2): ``cached_tokens`` clamp gets a floor of 0 in BOTH the
    stream path (``routes/responses.py``) and the non-stream path
    (``api/responses_adapter.py``). A buggy engine emitting a negative
    ``cached_tokens`` would otherwise propagate ``cached_tokens=-N`` to
    the SDK consumer.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Tokenizer:
    chat_template = ""

    def __init__(self):
        self.calls = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""


class _ToolCallChannelEngine:
    """Engine that emits on the ``tool_call`` channel WITHOUT populating
    ``output.tool_calls`` — the latent-engine path #591 item 2 covers.

    Today's channel-emitting engines (harmony / gemma4) always sidecar
    the structured calls so the ``tool_call`` channel bytes are masked
    upstream by the ``engine_tool_calls`` branch. This synthetic engine
    simulates the future regression where a router-stage engine surfaces
    tool args through the channel itself.
    """

    preserve_native_tool_format = False
    is_mllm = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.stream_calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="hi",
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        # Mid-stream: emit a legit content chunk, then a tool_call channel
        # chunk that carries JSON bytes WITHOUT structured tool_calls.
        # Pre-fix, the JSON bytes leaked to response.output_text.delta.
        yield _GenerationOutput(
            text="Hello",
            new_text="Hello",
            prompt_tokens=2,
            completion_tokens=1,
            finish_reason=None,
            channel="content",
        )
        yield _GenerationOutput(
            text='Hello{"name":"x","arguments":{}}',
            new_text='{"name":"x","arguments":{}}',
            prompt_tokens=0,
            completion_tokens=2,
            finish_reason=None,
            channel="tool_call",
        )
        yield _GenerationOutput(
            text='Hello{"name":"x","arguments":{}}',
            new_text="",
            prompt_tokens=0,
            completion_tokens=3,
            finish_reason="stop",
            channel="content",
        )


class _NegativeCachedTokensEngine:
    """Engine that emits a negative ``cached_tokens`` counter. Floor-clamp
    in #591 item 6 must convert that to 0 on the wire."""

    preserve_native_tool_format = False
    is_mllm = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.stream_calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="ok",
            prompt_tokens=5,
            completion_tokens=2,
            cached_tokens=-3,  # buggy engine
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        yield _GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=5,
            completion_tokens=2,
            cached_tokens=-3,  # buggy engine
            finish_reason="stop",
        )


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


_IMPORTED = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _make_client(monkeypatch, engine):
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)

    def _teardown():
        reset_config()
        rate_limiter.enabled = False
        rate_limiter.requests_per_minute = 60
        rate_limiter._requests.clear()

        for name, previous in previous_modules.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

        for (module_name, attr), previous in previous_attrs.items():
            module = sys.modules.get(module_name)
            if module is None:
                continue
            if previous is _MISSING:
                if hasattr(module, attr):
                    delattr(module, attr)
            else:
                setattr(module, attr, previous)

    return TestClient(app), _teardown


HEADERS = {"Authorization": "Bearer test-secret"}


def _parse_sse_events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev_name = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if ev_name is None or not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        events.append((ev_name, payload))
    return events


# =============================================================================
# Item 2 — HIGH: tool_call channel must NOT leak into output_text.delta
# =============================================================================


class TestToolCallChannelNoTextLeak:
    def test_tool_call_channel_bytes_do_not_leak_to_output_text_delta(
        self, monkeypatch
    ):
        """#591 item 2 regression. The engine emits a ``tool_call``
        channel chunk carrying JSON argument bytes; the streamed SSE
        events must contain those bytes in NO ``response.output_text.delta``
        payload (only the ``content``-channel ``Hello`` should land in
        the user-facing text)."""
        engine = _ToolCallChannelEngine()
        client, teardown = _make_client(monkeypatch, engine)
        try:
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "test-model", "input": "hi", "stream": True},
                headers=HEADERS,
            ) as resp:
                assert resp.status_code == 200, resp.read()
                body = resp.read().decode()
        finally:
            teardown()

        events = _parse_sse_events(body)
        # Collect every output_text.delta payload.
        text_deltas = [
            payload.get("delta", "")
            for ev_name, payload in events
            if ev_name == "response.output_text.delta"
        ]
        joined_text = "".join(text_deltas)

        # The legit content chunk must reach the wire.
        assert "Hello" in joined_text, (
            f"content-channel byte 'Hello' must be in output_text.delta; "
            f"got {joined_text!r}"
        )
        # The tool_call channel JSON must NOT reach the wire as text.
        assert '"name":"x"' not in joined_text, (
            "tool_call-channel JSON leaked into response.output_text.delta — "
            f"#591 item 2 regression. joined_text={joined_text!r}"
        )
        assert '"arguments"' not in joined_text, (
            "tool_call-channel JSON leaked into response.output_text.delta — "
            f"#591 item 2 regression. joined_text={joined_text!r}"
        )


# =============================================================================
# Item 5 — P2: empty-string model rejected at Pydantic layer
# =============================================================================


class TestEmptyStringModelRejected:
    def test_empty_model_rejected_with_422(self, monkeypatch):
        """#591 item 5. ``model=""`` must produce a Pydantic 422 from the
        ``min_length=1`` constraint on ``ResponsesRequest.model``, NOT
        slip into the route and get a downstream 400 from
        ``_validate_model_name`` (which works today but masks the schema
        bug)."""
        engine = _ToolCallChannelEngine()
        client, teardown = _make_client(monkeypatch, engine)
        try:
            resp = client.post(
                "/v1/responses",
                json={"model": "", "input": "hi"},
                headers=HEADERS,
            )
        finally:
            teardown()
        # Pydantic min_length violation → 400 via global handler
        # (vllm_mlx maps Pydantic ValidationError to 400 with OpenAI envelope).
        assert resp.status_code in (400, 422), resp.text
        # The chat/engine path was NOT hit (no leak past the schema).
        assert engine.calls == []
        assert engine.stream_calls == []

    def test_empty_model_pydantic_layer_rejects(self):
        """#591 item 5 — schema-level guard. Construct a
        ``ResponsesRequest`` with ``model=""`` directly and assert that
        the Pydantic validator raises (the route-level
        ``_validate_model_name`` is a defense-in-depth backstop, NOT the
        layer this fix targets). This pins the contract on the Pydantic
        model itself so a future refactor that moves the route gate
        can't silently regress."""
        from pydantic import ValidationError

        # Defer-import after no monkeypatch state to ensure we're
        # validating the production model class.
        from vllm_mlx.api.responses_models import ResponsesRequest

        with pytest.raises(ValidationError) as exc_info:
            ResponsesRequest(model="", input="hi")
        # Pydantic surfaces ``string_too_short`` (min_length=1).
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("model",) and err["type"] == "string_too_short"
            for err in errors
        ), f"expected string_too_short on 'model'; got {errors!r}"


# =============================================================================
# Item 6 — P2: negative cached_tokens floor-clamped to 0
# =============================================================================


class TestNegativeCachedTokensClamp:
    def test_non_stream_negative_cached_tokens_floor_to_zero(self, monkeypatch):
        """#591 item 6 (non-stream). A buggy engine surfacing a negative
        ``cached_tokens`` must NOT leak ``cached_tokens=-N`` onto the wire
        — the floor clamp converts it to 0 (semantically "no cache info")
        and the adapter then drops the empty ``input_tokens_details``
        block."""
        engine = _NegativeCachedTokensEngine()
        client, teardown = _make_client(monkeypatch, engine)
        try:
            resp = client.post(
                "/v1/responses",
                json={"model": "test-model", "input": "hi"},
                headers=HEADERS,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
        finally:
            teardown()

        usage = body["usage"]
        # No negative cached_tokens leak.
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        assert cached >= 0, (
            f"negative cached_tokens leaked through non-stream adapter — "
            f"#591 item 6 regression. usage={usage!r}"
        )

    def test_stream_negative_cached_tokens_floor_to_zero(self, monkeypatch):
        """#591 item 6 (stream). Same invariant on the streaming
        ``response.completed`` terminal event."""
        engine = _NegativeCachedTokensEngine()
        client, teardown = _make_client(monkeypatch, engine)
        try:
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "test-model", "input": "hi", "stream": True},
                headers=HEADERS,
            ) as resp:
                assert resp.status_code == 200, resp.read()
                body = resp.read().decode()
        finally:
            teardown()

        events = _parse_sse_events(body)
        completed = [p for ev_name, p in events if ev_name == "response.completed"]
        assert completed, f"no response.completed event; body={body!r}"
        usage = completed[-1]["response"]["usage"]
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        assert cached >= 0, (
            f"negative cached_tokens leaked through stream terminal usage — "
            f"#591 item 6 regression. usage={usage!r}"
        )


# =============================================================================
# Item 5 (adjacent) — model="gpt-..." still passes through (bypass works)
# =============================================================================


class TestGptBypassStillWorks:
    def test_non_empty_gpt_model_still_passes_through(self, monkeypatch):
        """#591 item 5 sibling — ensure the empty-string rejection does
        NOT regress the ``gpt-*`` / ``claude-*`` bypass (#557) that
        Codex CLI relies on. ``model=\"gpt-5-codex\"`` must reach the
        engine."""
        engine = _ToolCallChannelEngine()
        client, teardown = _make_client(monkeypatch, engine)
        try:
            resp = client.post(
                "/v1/responses",
                json={"model": "gpt-5-codex", "input": "hi"},
                headers=HEADERS,
            )
            assert resp.status_code == 200, resp.text
        finally:
            teardown()
        assert len(engine.calls) == 1
