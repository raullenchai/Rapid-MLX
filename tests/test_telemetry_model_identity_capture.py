# SPDX-License-Identifier: Apache-2.0
"""The served model's telemetry identity is captured at request START.

Codex P1 on PR #3600: the terminal telemetry emit called
``served_model_id(request.model)``, which re-resolves the *live* model
registry. A resident-model swap, an eviction, or a ``set_default`` while a
request (or a stream) was in flight therefore made the event name the model
that is current NOW, not the one that produced the tokens. The Anthropic
surface makes it concrete: ``claude-*`` / ``gpt-*`` names deliberately fall
through to the default engine, so a default switch repoints them.

The fix: resolve the identity once, next to ``get_engine``, keyed on the
ENGINE OBJECT, and carry that immutable string to the emit. Each behaviour
test below is written so that deleting the production hop it covers turns
it red (mutations M11-M21 in the PR body record each one); two of them
additionally ship an explicit CONTROL that drives the same path WITHOUT the
captured value and shows the event flipping to model B.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

_A_PATH = "mlx-community/Qwen3.5-9B-4bit"
_A_ID = "qwen3.5-9b-4bit"
_B_PATH = "mlx-community/Qwen3.5-4B-4bit"
_B_ID = "qwen3.5-4b-4bit"


# ------------------------------------------------------------- the registry


def _entry(engine, name, path, telemetry_id):
    from rapid_mlx.runtime.model_registry import ModelEntry

    return ModelEntry(
        engine=engine,
        model_name=name,
        model_path=path,
        telemetry_model_id=telemetry_id,
    )


@pytest.fixture
def _registry_config():
    """A server config carrying a real two-model registry, default = A."""
    from rapid_mlx.config.server_config import get_config, reset_config
    from rapid_mlx.runtime.model_registry import ModelRegistry

    reset_config()
    cfg = get_config()
    engine_a = SimpleNamespace(name="engine-a")
    engine_b = SimpleNamespace(name="engine-b")
    registry = ModelRegistry()
    registry.add(_entry(engine_a, "model-a", _A_PATH, _A_ID), is_default=True)
    cfg.model_registry = registry
    yield SimpleNamespace(
        cfg=cfg,
        registry=registry,
        engine_a=engine_a,
        engine_b=engine_b,
        promote_b=lambda: registry.add(
            _entry(engine_b, "model-b", _B_PATH, _B_ID), is_default=True
        ),
    )
    reset_config()


# ------------------------------------------------- engine_telemetry_id rules


def test_identity_is_keyed_on_the_engine_not_the_name(_registry_config):
    """A default swap cannot move the answer for an engine we already hold."""
    from rapid_mlx.telemetry.model_id import engine_telemetry_id, served_model_id

    ctx = _registry_config
    assert engine_telemetry_id(ctx.engine_a) == _A_ID
    ctx.promote_b()
    # The name-based resolver now answers B for the same client string...
    assert served_model_id("claude-sonnet-4") == _B_ID
    # ...but the engine that actually ran is still A.
    assert engine_telemetry_id(ctx.engine_a) == _A_ID
    assert engine_telemetry_id(ctx.engine_b) == _B_ID


def test_an_unloaded_engine_is_custom_not_a_guess(_registry_config):
    """An entry unloaded mid-request reports ``<custom>``; we never fall back
    to "whatever is default now", which is the bug this fixes."""
    from rapid_mlx.telemetry.model_id import engine_telemetry_id

    ctx = _registry_config
    ctx.registry.remove("model-a")
    ctx.promote_b()
    assert engine_telemetry_id(ctx.engine_a) == "<custom>"


def test_identity_falls_back_to_the_resolved_path_without_a_registry():
    from rapid_mlx.config.server_config import get_config, reset_config
    from rapid_mlx.telemetry.model_id import engine_telemetry_id

    reset_config()
    try:
        cfg = get_config()
        cfg.model_registry = None
        cfg.model_name = "acme-internal-support-bot"  # served name
        cfg.model_path = _A_PATH
        assert engine_telemetry_id(object()) == _A_ID
    finally:
        reset_config()


def test_identity_when_the_config_cannot_be_imported(monkeypatch):
    import sys

    from rapid_mlx.telemetry.model_id import engine_telemetry_id

    monkeypatch.setitem(sys.modules, "rapid_mlx.config.server_config", None)
    assert engine_telemetry_id(object()) == "<custom>"


def test_identity_swallows_a_broken_registry(_registry_config):
    from rapid_mlx.telemetry.model_id import engine_telemetry_id

    class _Exploding:
        def __bool__(self):
            raise RuntimeError("registry is broken")

    _registry_config.cfg.model_registry = _Exploding()
    assert engine_telemetry_id(object()) == "<custom>"


def test_identity_does_not_swallow_keyboard_interrupt(_registry_config):
    from rapid_mlx.telemetry.model_id import engine_telemetry_id

    class _Interrupting:
        def __bool__(self):
            raise KeyboardInterrupt

    _registry_config.cfg.model_registry = _Interrupting()
    with pytest.raises(KeyboardInterrupt):
        engine_telemetry_id(object())


# ---------------------------------------------------- the non-streaming route


class _RawRequest:
    headers: dict = {}

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


class _SwappingChatEngine:
    """Promotes model B to default *while generating*, like a resident-model
    handoff landing mid-request."""

    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "model-a"
    tokenizer = SimpleNamespace(encode=lambda _text: [1])

    def __init__(self, promote_b):
        self._promote_b = promote_b

    async def chat(self, messages, **kwargs):
        from rapid_mlx.engine.base import GenerationOutput

        self._promote_b()
        return GenerationOutput(
            text="hello there",
            finish_reason="stop",
            prompt_tokens=12,
            completion_tokens=8,
        )


async def _await_direct(coro, *_a, **_k):
    return await coro


def _patch_chat_route(monkeypatch, engine, emit_calls):
    from rapid_mlx.routes import chat
    from rapid_mlx.telemetry import emit

    monkeypatch.setattr(emit, "request", lambda **kw: emit_calls.append(kw))
    monkeypatch.setattr(emit, "is_enabled", lambda *a, **k: False)
    monkeypatch.setattr(chat, "_resolve_max_tokens", lambda *a, **k: 64)
    monkeypatch.setattr(chat, "get_engine", lambda *a, **k: engine)
    monkeypatch.setattr(chat, "ensure_engine_ready", _noop_async)
    monkeypatch.setattr(chat, "_validate_model_name", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_check_admission_or_503", lambda *a, **k: None)
    monkeypatch.setattr(
        chat, "_release_admission_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(
        chat, "_release_primary_request_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "_wait_with_disconnect", _await_direct)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "enforce_context_length_for_messages", lambda *a, **k: 1)


async def _noop_async(*_a, **_k):
    return None


def _chat_request(stream=False):
    """A request whose ``model`` is NOT a registry name.

    ``claude-*`` is the codex scenario verbatim: the Anthropic-compatible
    names deliberately fall through to whatever the *default* engine is, so
    a name-based resolution at emit time follows the default swap.
    """
    from rapid_mlx.api.models import ChatCompletionRequest

    return ChatCompletionRequest(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        stream=stream,
    )


@pytest.mark.asyncio
async def test_nonstreaming_chat_event_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    """THE P1: the default flips mid-request; the event must still say A."""
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingChatEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_chat_route(monkeypatch, engine, calls)

    await chat.create_chat_completion(_chat_request(), _RawRequest())

    assert len(calls) == 1
    assert calls[0]["model_alias"] == _A_ID
    assert _B_ID not in repr(calls[0])


@pytest.mark.asyncio
async def test_nonstreaming_chat_control_without_the_capture(
    monkeypatch, _registry_config
):
    """Control: the same swap, with the captured id NOT threaded, reports B.

    This is what the code did before the fix — it pins that the test above
    is actually exercising the capture and not passing for free.
    """
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingChatEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_chat_route(monkeypatch, engine, calls)

    await chat._create_chat_completion_impl(
        _chat_request(),
        _RawRequest(),
        engine,
        [False],
        [False],
    )

    assert calls[0]["model_alias"] == _B_ID


# -------------------------------------------------------- the streaming route


class _FakeStreamingOutput:
    def __init__(self, new_text: str, finished: bool):
        self.new_text = new_text
        self.text = new_text
        self.finished = finished
        self.finish_reason = "stop" if finished else None
        self.channel = None
        self.prompt_tokens = 11
        self.completion_tokens = 7
        self.cached_tokens = 0
        self.tokens = []
        self.logprobs = None
        self.tool_calls = None
        self.matched_stop = None
        self.raw_text = new_text


class _SwappingStreamEngine:
    """Promotes B to default after the first streamed token."""

    preserve_native_tool_format = False
    model_name = "model-a"

    def __init__(self, promote_b):
        self._promote_b = promote_b
        self.tokenizer = SimpleNamespace(encode=lambda _text: [1])
        self.is_mllm = False
        self.supports_tool_calls = False
        self.supports_guided_generation = False

    async def stream_chat(self, **kwargs):
        deltas = ["Hello", " there", "!"]
        for i, delta in enumerate(deltas):
            if i == 1:
                self._promote_b()
            yield _FakeStreamingOutput(delta, finished=(i == len(deltas) - 1))

    def build_prompt(self, *args, **kwargs):
        return "prompt"


def _drive_stream(monkeypatch, engine, emit_calls, **stream_kwargs):
    from rapid_mlx.config import server_config
    from rapid_mlx.routes import chat
    from rapid_mlx.telemetry import emit

    cfg = server_config.get_config()
    for attr, value in (
        ("tool_call_parser", None),
        ("reasoning_parser_name", None),
        ("reasoning_parser", None),
        ("enable_auto_tool_choice", False),
        ("gc_control", False),
    ):
        monkeypatch.setattr(cfg, attr, value, raising=False)
    monkeypatch.setattr(emit, "request", lambda **kw: emit_calls.append(kw))

    async def _run():
        gen = chat.stream_chat_completion(
            engine,
            [{"role": "user", "content": "hi"}],
            _chat_request(stream=True),
            **stream_kwargs,
        )
        async for _sse in gen:
            pass

    asyncio.run(_run())


def test_streaming_chat_event_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    """A mid-STREAM default swap must not repoint the terminal event."""
    ctx = _registry_config
    calls: list[dict] = []
    _drive_stream(
        monkeypatch,
        _SwappingStreamEngine(ctx.promote_b),
        calls,
        served_telemetry_id=_A_ID,
    )

    assert len(calls) == 1
    assert calls[0]["stream"] is True
    assert calls[0]["model_alias"] == _A_ID
    assert _B_ID not in repr(calls[0])


def test_streaming_chat_control_without_the_capture(monkeypatch, _registry_config):
    """Control for the stream: no captured id, and the event flips to B."""
    ctx = _registry_config
    calls: list[dict] = []
    _drive_stream(monkeypatch, _SwappingStreamEngine(ctx.promote_b), calls)

    assert calls[0]["model_alias"] == _B_ID


# --------------------------------------- the ROUTE-level streaming wiring
#
# Codex-round finding: driving ``stream_chat_completion`` directly and
# handing it ``served_telemetry_id=`` by hand proves only that the
# generator uses a keyword it was given. The production hop — the route
# passing the captured id INTO the generator — was untested, and deleting
# it from chat.py left every test green. These drive the real route and
# drain the SSE body, so each hop is mutation-killable.


async def _drain(response):
    """Consume a StreamingResponse body to completion."""
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_streaming_chat_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    """route → _create_chat_completion_impl → stream_chat_completion → emit."""
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingStreamEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_chat_route(monkeypatch, engine, calls)
    _patch_stream_cfg(monkeypatch)

    response = await chat.create_chat_completion(
        _chat_request(stream=True), _RawRequest()
    )
    await _drain(response)

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["stream"] is True
    assert calls[0]["model_alias"] == _A_ID
    assert _B_ID not in repr(calls[0])


def _patch_stream_cfg(monkeypatch):
    from rapid_mlx.config import server_config

    cfg = server_config.get_config()
    for attr, value in (
        ("tool_call_parser", None),
        ("reasoning_parser_name", None),
        ("reasoning_parser", None),
        ("enable_auto_tool_choice", False),
        ("gc_control", False),
    ):
        monkeypatch.setattr(cfg, attr, value, raising=False)


# ------------------------------------- the other two surfaces, end to end
#
# The finding also noted that /v1/completions and /v1/messages — the
# surface the P1 report actually named, because ``claude-*`` falls through
# to the default engine — had no route-level test at all.


class _SwappingCompletionEngine:
    """Non-streaming completion engine that swaps the default mid-request."""

    preserve_native_tool_format = False
    is_mllm = False
    model_name = "model-a"

    def __init__(self, promote_b):
        self._promote_b = promote_b
        self.tokenizer = SimpleNamespace(encode=lambda _text: [1])

    async def generate(self, prompt, **kwargs):
        from rapid_mlx.engine.base import GenerationOutput

        self._promote_b()
        return GenerationOutput(
            text="Paris",
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=1,
        )


def _patch_completions_route(monkeypatch, engine, emit_calls):
    from rapid_mlx.routes import completions
    from rapid_mlx.telemetry import emit

    monkeypatch.setattr(emit, "request", lambda **kw: emit_calls.append(kw))
    monkeypatch.setattr(emit, "is_enabled", lambda *a, **k: False)
    monkeypatch.setattr(completions, "get_engine", lambda *a, **k: engine)
    monkeypatch.setattr(completions, "ensure_engine_ready", _noop_async)
    monkeypatch.setattr(completions, "_validate_model_name", lambda *a, **k: None)
    monkeypatch.setattr(completions, "_check_admission_or_503", lambda *a, **k: None)
    monkeypatch.setattr(
        completions, "_release_admission_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(completions, "_release_route_ownership", lambda *a, **k: None)


@pytest.mark.asyncio
async def test_completions_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingCompletionEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_completions_route(monkeypatch, engine, calls)

    request = CompletionRequest(
        model="claude-sonnet-4", prompt="The capital of France is", max_tokens=8
    )
    await completions.create_completion(request, _RawRequest())

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["endpoint"] == "/v1/completions"
    assert calls[0]["model_alias"] == _A_ID
    assert _B_ID not in repr(calls[0])


class _AnthRawRequest:
    """Raw Starlette-ish request carrying an Anthropic Messages body."""

    def __init__(self, body):
        self._body = body
        self.headers: dict = {}

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


def _patch_anthropic_route(monkeypatch, engine, emit_calls):
    from rapid_mlx.routes import anthropic
    from rapid_mlx.telemetry import emit

    monkeypatch.setattr(emit, "request", lambda **kw: emit_calls.append(kw))
    monkeypatch.setattr(emit, "is_enabled", lambda *a, **k: False)
    monkeypatch.setattr(anthropic, "get_engine", lambda *a, **k: engine)
    monkeypatch.setattr(anthropic, "ensure_engine_ready", _noop_async)
    monkeypatch.setattr(anthropic, "_validate_model_name", lambda *a, **k: None)
    monkeypatch.setattr(anthropic, "_check_admission_or_503", lambda *a, **k: None)
    monkeypatch.setattr(
        anthropic, "_release_admission_unless_committed", lambda *a, **k: None
    )


@pytest.mark.asyncio
async def test_anthropic_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    """The surface the P1 report named: ``claude-*`` falls through to the
    DEFAULT engine, so a default swap mid-request is exactly the case that
    used to repoint the event."""
    from rapid_mlx.routes import anthropic

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingChatEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_anthropic_route(monkeypatch, engine, calls)

    await anthropic.create_anthropic_message(
        _AnthRawRequest(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "say hi"}],
            }
        )
    )

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["endpoint"] == "/v1/messages"
    assert calls[0]["model_alias"] == _A_ID
    assert _B_ID not in repr(calls[0])


# ------------------------------------------- the two chat stream wrappers
#
# Both wrappers ultimately delegate to ``stream_chat_completion``, which
# owns the terminal emit. If either drops the captured id on the way, the
# event silently falls back to the live-registry lookup.


_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def _drive_wrapper(monkeypatch, factory, emit_calls):
    from rapid_mlx.telemetry import emit

    _patch_stream_cfg(monkeypatch)
    monkeypatch.setattr(emit, "request", lambda **kw: emit_calls.append(kw))

    async def _run():
        async for _chunk in factory():
            pass

    asyncio.run(_run())


class _GuidedFailingEngine(_SwappingStreamEngine):
    """Guided generation is advertised but blows up, so both the route and
    the helper take the unconstrained fallback — the hops that forward the
    captured id.

    ``supports_guided_generation`` is set on the INSTANCE: the base class
    sets it in ``__init__``, so a class attribute here would be silently
    overwritten and the route would never enter the guided branch.
    """

    def __init__(self, promote_b):
        super().__init__(promote_b)
        self.supports_guided_generation = True

    async def generate_with_schema(self, *a, **k):
        raise RuntimeError("llguidance unavailable")


def test_guided_fallback_carries_the_captured_id(monkeypatch, _registry_config):
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    engine = _GuidedFailingEngine(ctx.promote_b)
    _drive_wrapper(
        monkeypatch,
        lambda: chat.stream_chat_completion_guided(
            engine,
            [{"role": "user", "content": "hi"}],
            _chat_request(stream=True),
            _SCHEMA,
            served_telemetry_id=_A_ID,
        ),
        calls,
    )

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["model_alias"] == _A_ID


def test_strict_postgen_carries_the_captured_id(monkeypatch, _registry_config):
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    _drive_wrapper(
        monkeypatch,
        lambda: chat.stream_chat_completion_strict_postgen(
            _SwappingStreamEngine(ctx.promote_b),
            [{"role": "user", "content": "hi"}],
            _chat_request(stream=True),
            _SCHEMA,
            served_telemetry_id=_A_ID,
        ),
        calls,
    )

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["model_alias"] == _A_ID


# ------------------------------------ the two remaining streaming ROUTES


class _SwappingStreamCompletionEngine(_SwappingCompletionEngine):
    """Legacy-completions streaming engine that swaps the default mid-stream."""

    async def stream_generate(self, *args, **kwargs):
        deltas = ["Par", "is", "."]
        for i, delta in enumerate(deltas):
            if i == 1:
                self._promote_b()
            yield _FakeStreamingOutput(delta, finished=(i == len(deltas) - 1))


@pytest.mark.asyncio
async def test_streaming_completions_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingStreamCompletionEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_completions_route(monkeypatch, engine, calls)
    _patch_stream_cfg(monkeypatch)

    response = await completions.create_completion(
        CompletionRequest(
            model="claude-sonnet-4", prompt="The capital of France is", stream=True
        ),
        _RawRequest(),
    )
    await _drain(response)

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["endpoint"] == "/v1/completions"
    assert calls[0]["stream"] is True
    assert calls[0]["model_alias"] == _A_ID


@pytest.mark.asyncio
async def test_streaming_anthropic_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    """The exact scenario the P1 report described: a ``claude-*`` stream in
    flight while the resident default is replaced."""
    from rapid_mlx.routes import anthropic

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingStreamEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_anthropic_route(monkeypatch, engine, calls)
    _patch_stream_cfg(monkeypatch)

    response = await anthropic.create_anthropic_message(
        _AnthRawRequest(
            {
                "model": "claude-sonnet-4",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "say hi"}],
            }
        )
    )
    await _drain(response)

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["endpoint"] == "/v1/messages"
    assert calls[0]["stream"] is True
    assert calls[0]["model_alias"] == _A_ID


# ------------------- the guided / strict-postgen branches of the ROUTE
#
# The last two hops: the route's json_schema streaming branches, which
# hand the captured id to the two wrapper generators.


def _json_schema_request(strict: bool):
    from rapid_mlx.api.models import ChatCompletionRequest

    return ChatCompletionRequest(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        stream=True,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": _SCHEMA, "strict": strict},
        },
    )


@pytest.mark.asyncio
async def test_guided_streaming_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    engine = _GuidedFailingEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_chat_route(monkeypatch, engine, calls)
    _patch_stream_cfg(monkeypatch)

    response = await chat.create_chat_completion(
        _json_schema_request(strict=False), _RawRequest()
    )
    await _drain(response)

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["model_alias"] == _A_ID


@pytest.mark.asyncio
async def test_strict_postgen_streaming_route_keeps_the_model_that_served(
    monkeypatch, _registry_config
):
    from rapid_mlx.routes import chat

    ctx = _registry_config
    calls: list[dict] = []
    engine = _SwappingStreamEngine(ctx.promote_b)
    ctx.registry.add(_entry(engine, "model-a", _A_PATH, _A_ID), is_default=True)
    _patch_chat_route(monkeypatch, engine, calls)
    _patch_stream_cfg(monkeypatch)

    response = await chat.create_chat_completion(
        _json_schema_request(strict=True), _RawRequest()
    )
    await _drain(response)

    assert len(calls) == 1, f"expected one request event, got {calls}"
    assert calls[0]["model_alias"] == _A_ID
