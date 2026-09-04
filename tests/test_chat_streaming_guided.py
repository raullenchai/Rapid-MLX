# SPDX-License-Identifier: Apache-2.0
"""Streaming + guided generation route contract.

Pins Gap #2 from the v0.6.60 onboarding sweep: ``stream=true`` requests
with ``response_format: json_schema`` must route through
``engine.generate_with_schema`` (constrained), NOT ``engine.stream_chat``
(unconstrained). Pre-fix, the stream branch of
``_create_chat_completion_impl`` ignored ``supports_guided_generation``
entirely and the model would emit unconstrained tokens (e.g. a
``\\`\\`\\`json ... \\`\\`\\`\\`` markdown fence) defeating the user's intent.

Two contract tests:

1. **Success path** — guided streaming is used, fallback engine.stream_chat
   is not called, the synthesized SSE stream carries the constrained text.

2. **Fallback path** — if ``generate_with_schema`` raises, the helper
   falls back to ``engine.stream_chat`` so request liveness is preserved
   (clients in strict-mode use cases should validate themselves; this
   matches the non-streaming fallback semantics).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api.errors import GuidedGenerationCancelledError
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.routes.chat import stream_chat_completion_guided
from vllm_mlx.routes.health import cancel_request


class _GuidedEngine:
    """Mock engine that supports guided generation.

    Records every call to ``generate_with_schema`` and ``stream_chat``
    so tests can assert which path the route dispatched to.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = True
    tokenizer = None

    def __init__(
        self, *, guided_text: str = '{"k": "v"}', raise_in_guided: bool = False
    ):
        self._guided_text = guided_text
        self._raise = raise_in_guided
        self.guided_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        # Stream branch validates the template eagerly; return a no-op
        # string so that pre-flight passes without exercising a real
        # chat-template engine.
        return "PROMPT"

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append(
            {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
        )
        if self._raise:
            raise RuntimeError("simulated guided-decode failure")
        return GenerationOutput(
            text=self._guided_text,
            new_text=self._guided_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        """Unconstrained fallback path: emit a single text delta."""
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        text = "FALLBACK"
        yield GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_client(engine: _GuidedEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _parse_sse_events(text: str) -> tuple[list[dict], bool]:
    """Return ``(parsed_events, saw_done)``.

    ``parsed_events`` excludes the ``[DONE]`` sentinel.
    """
    events: list[dict] = []
    saw_done = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events, saw_done


@pytest.mark.asyncio
async def test_guided_stream_publishes_cancellable_id_before_buffered_output():
    """The first SSE event addresses live guided work, not completed work."""

    class _CancellableGuidedEngine(_GuidedEngine):
        def __init__(self):
            super().__init__()
            self.cancelled = asyncio.Event()
            self.live_request_id: str | None = None

        async def generate_with_schema(self, *, messages, json_schema, **kwargs):
            self.guided_calls.append(
                {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
            )
            self.live_request_id = kwargs["request_id"]
            kwargs["request_id_holder"][0] = self.live_request_id
            kwargs["request_admitted_event"].set()
            await self.cancelled.wait()
            raise GuidedGenerationCancelledError()

        def abort_guided_request(self, request_id: str) -> bool:
            if request_id != self.live_request_id or self.cancelled.is_set():
                return False
            self.cancelled.set()
            return True

        async def abort_request(self, request_id: str) -> bool:
            return self.abort_guided_request(request_id)

    engine = _CancellableGuidedEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "emit json"}],
    )
    holder: list[str | None] = [None]
    stream = stream_chat_completion_guided(
        engine,
        request.messages,
        request,
        {"type": "object"},
        response_id="chatcmpl-" + "a" * 32,
        strict_mode=True,
        request_id_holder=holder,
    )

    first = json.loads((await anext(stream)).removeprefix("data: "))
    request_id = first["id"]
    assert request_id == "chatcmpl-" + "a" * 32
    assert holder == [request_id]
    assert engine.cancelled.is_set() is False

    response = await cancel_request(request_id)
    assert response == {
        "object": "request.cancel",
        "id": request_id,
        "cancelled": True,
    }

    terminal = json.loads((await anext(stream)).removeprefix("data: "))
    assert terminal["choices"][0]["finish_reason"] == "cancelled"
    assert terminal["choices"][0]["delta"] == {}
    assert await anext(stream) == "data: [DONE]\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert engine.stream_calls == [], "cancellation must never fall back unconstrained"


@pytest.mark.asyncio
async def test_guided_stream_shutdown_consumes_exact_lifecycle_owner():
    """Shutdown emits the model-replacement terminal and clears its ledger."""

    class _ShutdownGuidedEngine(_GuidedEngine):
        def __init__(self):
            super().__init__()
            self.lifecycle_owner = object()
            self.lifecycle_consumed = False

        async def generate_with_schema(self, *, messages, json_schema, **kwargs):
            kwargs["request_admitted_event"].set()
            raise GuidedGenerationCancelledError(lifecycle_task=self.lifecycle_owner)

        def consume_lifecycle_task_abort(self, task) -> bool:
            if task is not self.lifecycle_owner or self.lifecycle_consumed:
                return False
            self.lifecycle_consumed = True
            return True

    engine = _ShutdownGuidedEngine()
    reset_config().engine = engine
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "emit json"}],
    )
    stream = stream_chat_completion_guided(
        engine,
        request.messages,
        request,
        {"type": "object"},
        response_id="chatcmpl-" + "b" * 32,
        strict_mode=True,
    )

    events = [event async for event in stream]

    assert any('"code": "model_replacement"' in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"
    assert engine.lifecycle_consumed is True
    assert engine.stream_calls == []


@pytest.mark.asyncio
async def test_shutdown_during_retained_handoff_keeps_replacement_semantics():
    """A retained guided owner carries shutdown cause through handoff."""
    from types import SimpleNamespace

    class _ShutdownHandoffEngine(_GuidedEngine):
        def __init__(self):
            super().__init__(raise_in_guided=True)
            self.lifecycle_owner = object()
            self.lifecycle_consumed = False

        def finish_guided_handoff(self, _request_id: str):
            return SimpleNamespace(
                cancelled=True,
                lifecycle_task=self.lifecycle_owner,
            )

        def consume_lifecycle_task_abort(self, task) -> bool:
            if task is not self.lifecycle_owner or self.lifecycle_consumed:
                return False
            self.lifecycle_consumed = True
            return True

    engine = _ShutdownHandoffEngine()
    reset_config().engine = engine
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "emit json"}],
    )
    stream = stream_chat_completion_guided(
        engine,
        request.messages,
        request,
        {"type": "object"},
        response_id="chatcmpl-" + "c" * 32,
        strict_mode=True,
    )

    events = [event async for event in stream]

    assert any('"code": "model_replacement"' in event for event in events)
    assert all('"finish_reason":"cancelled"' not in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"
    assert engine.lifecycle_consumed is True
    assert engine.stream_calls == []


@pytest.mark.asyncio
async def test_cancel_during_guided_to_scheduler_handoff_never_leaks_fallback():
    """The public id stays owned until the fallback scheduler is admitted."""

    class _HandoffEngine(_GuidedEngine):
        def __init__(self):
            super().__init__(raise_in_guided=True)
            self.handoff_calls = 0
            self.scheduler_aborts: list[str] = []
            self.guided_owned = True
            self.cancelled = False
            self.fallback_started = asyncio.Event()
            self.release_fallback = asyncio.Event()

        def finish_guided_handoff(self, request_id: str) -> bool:
            assert request_id == "chatcmpl-handoff"
            self.handoff_calls += 1
            self.guided_owned = False
            return self.cancelled

        async def stream_chat(self, messages, **kwargs):
            self.stream_calls.append({"messages": messages, "kwargs": kwargs})
            self.fallback_started.set()
            await self.release_fallback.wait()
            kwargs["request_id_holder"][0] = kwargs["request_id"]
            kwargs["request_admitted_event"].set()
            yield GenerationOutput(
                text="FALLBACK-MUST-NOT-LEAK",
                new_text="FALLBACK-MUST-NOT-LEAK",
                finished=True,
                finish_reason="stop",
            )

        async def abort_request(self, request_id: str) -> bool:
            if self.guided_owned:
                self.cancelled = True
                return True
            self.scheduler_aborts.append(request_id)
            return True

    engine = _HandoffEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "emit json"}],
    )
    holder: list[str | None] = [None]
    stream = stream_chat_completion_guided(
        engine,
        request.messages,
        request,
        {"type": "object"},
        response_id="chatcmpl-handoff",
        request_id_holder=holder,
    )

    admission = json.loads((await anext(stream)).removeprefix("data: "))
    assert admission["id"] == "chatcmpl-handoff"
    fallback_result = asyncio.create_task(anext(stream))
    await engine.fallback_started.wait()
    assert fallback_result.done() is False

    response = await cancel_request("chatcmpl-handoff")
    assert response["cancelled"] is True
    engine.release_fallback.set()

    terminal = json.loads((await fallback_result).removeprefix("data: "))
    assert terminal["choices"][0]["finish_reason"] == "cancelled"
    assert terminal["choices"][0]["delta"] == {}
    assert "FALLBACK-MUST-NOT-LEAK" not in json.dumps(terminal)
    done = await anext(stream)
    assert done == "data: [DONE]\n\n"
    assert "FALLBACK-MUST-NOT-LEAK" not in done
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert engine.handoff_calls == 1
    assert engine.scheduler_aborts == ["chatcmpl-handoff"]
    assert len(engine.stream_calls) == 1
    assert engine.guided_calls[0]["kwargs"]["retain_guided_request_on_failure"] is True


@pytest.mark.asyncio
async def test_closing_after_admission_cancels_unfinished_guided_task():
    """Disconnect after ID publication cannot leave buffered work alive."""

    class _BlockedGuidedEngine(_GuidedEngine):
        def __init__(self):
            super().__init__()
            self.cancelled = asyncio.Event()

        async def generate_with_schema(self, *, messages, json_schema, **kwargs):
            kwargs["request_admitted_event"].set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

    engine = _BlockedGuidedEngine()
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "emit json"}],
    )
    stream = stream_chat_completion_guided(
        engine,
        request.messages,
        request,
        {"type": "object"},
        response_id="chatcmpl-close-after-admission",
    )

    await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(engine.cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_cancelled_fallback_await_finishes_retained_handoff():
    """A disconnect before fallback admission releases the guided identity."""

    class _BlockedFallbackEngine(_GuidedEngine):
        def __init__(self):
            super().__init__(raise_in_guided=True)
            self.fallback_started = asyncio.Event()
            self.handoffs = 0

        async def stream_chat(self, messages, **kwargs):
            self.fallback_started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover - establishes the async-generator shape

        def finish_guided_handoff(self, request_id: str):
            assert request_id == "chatcmpl-cancel-fallback"
            self.handoffs += 1
            return False

    engine = _BlockedFallbackEngine()
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "emit json"}],
    )
    stream = stream_chat_completion_guided(
        engine,
        request.messages,
        request,
        {"type": "object"},
        response_id="chatcmpl-cancel-fallback",
    )

    await anext(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(engine.fallback_started.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert engine.handoffs == 1


def test_nonstream_guided_user_cancel_is_not_model_replacement():
    """An explicit cancel propagates through the ordinary cancellation path."""
    import concurrent.futures

    class _CancelledEngine(_GuidedEngine):
        async def generate_with_schema(self, *, messages, json_schema, **kwargs):
            raise GuidedGenerationCancelledError()

    cfg = reset_config()
    cfg.engine = _CancelledEngine()
    cfg.model_name = "test-model"
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    with pytest.raises(concurrent.futures.CancelledError):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "emit json"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "schema": {"type": "object"},
                        "strict": False,
                    },
                },
            },
        )


_SCHEMA = {
    "type": "object",
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "qty": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["name", "qty"],
            "additionalProperties": False,
        }
    },
    "properties": {
        "label": {"type": "string", "enum": ["red", "green", "blue"]},
        "items": {
            "type": "array",
            "items": {"$ref": "#/$defs/Item"},
            "minItems": 1,
        },
    },
    "required": ["label", "items"],
    "additionalProperties": False,
}


_GUIDED_OUTPUT = json.dumps({"label": "red", "items": [{"name": "alpha", "qty": 2}]})


def test_streaming_json_schema_routes_through_guided_generation():
    """stream=true + json_schema must call generate_with_schema, NOT stream_chat.

    The bug class this gates: a refactor that re-wires the stream branch
    to ``engine.stream_chat`` without consulting ``supports_guided_generation``
    would silently downgrade strict-mode requests to unconstrained tokens
    — invisible in unit smoke (small schemas the model would emit anyway)
    but catastrophic for adversarial / complex schemas.
    """
    engine = _GuidedEngine(guided_text=_GUIDED_OUTPUT)
    client = _make_client(engine)

    payload = {
        "model": "test-model",
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "pick a color"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": True},
        },
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")

    # Constraint dispatch: guided path called exactly once, stream_chat
    # never invoked. This is the load-bearing assertion for Gap #2.
    assert len(engine.guided_calls) == 1
    assert engine.stream_calls == []

    # The route must hand the RAW schema dict to generate_with_schema —
    # not the strict outer ``response_format`` wrapper. Hand-off through
    # the wrapper would silently re-introduce the schema-projection bug
    # that PR #419 fixed.
    assert engine.guided_calls[0]["json_schema"] == _SCHEMA

    # ``raise_on_failure=True`` is load-bearing: it forces the engine
    # to raise instead of silently falling back to ``self.chat(...)``
    # (which would buffer a long unconstrained reply into a single
    # content chunk and defeat SSE). The streaming helper catches the
    # raise and delegates to the unconstrained streaming fallback
    # instead (codex Round 2 finding). A refactor that drops this
    # kwarg silently re-introduces the buffered-reply-pretending-to-be-
    # streaming bug.
    assert engine.guided_calls[0]["kwargs"].get("raise_on_failure") is True

    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done, "streaming response must terminate with [DONE]"

    # Reassemble the content from delta chunks; it must equal the
    # constrained text the engine returned.
    content_parts: list[str] = []
    saw_role = False
    saw_finish = False
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {}) or {}
            if delta.get("role") == "assistant":
                saw_role = True
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
            if choice.get("finish_reason"):
                saw_finish = True

    assert saw_role, "first SSE chunk must announce assistant role"
    assert saw_finish, "stream must emit a finish_reason chunk"
    assert "".join(content_parts) == _GUIDED_OUTPUT


def test_mllm_streaming_schema_stays_on_scheduler_with_request_processor(
    monkeypatch,
):
    """Vision-capable serving keeps its lane and constrains decode in place."""
    from vllm_mlx.api import guided

    marker = object()
    monkeypatch.setattr(
        guided,
        "build_json_schema_logits_processor",
        lambda _tokenizer, schema: marker if schema == _SCHEMA else None,
    )
    from vllm_mlx.routes import chat as chat_route

    monkeypatch.setattr(
        chat_route,
        "_build_reasoning_budget_processor",
        lambda *_args, **_kwargs: object(),
    )
    engine = _GuidedEngine(guided_text=_GUIDED_OUTPUT)
    engine.is_mllm = True
    engine.supports_guided_generation = False
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "pick a color"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Pick", "schema": _SCHEMA},
            },
        },
    )

    assert resp.status_code == 200, resp.text
    assert engine.guided_calls == []
    assert len(engine.stream_calls) == 1
    assert engine.stream_calls[0]["kwargs"]["grammar_logits_processor"] is marker
    assert "reasoning_budget_logits_processor" not in engine.stream_calls[0]["kwargs"]
    assert engine.stream_calls[0]["kwargs"]["enable_thinking"] is False
    assert "data: [DONE]" in resp.text


def test_streaming_guided_no_duplicate_usage_when_include_usage_true():
    """When ``stream_options.include_usage`` is True, usage must appear
    ONLY in the dedicated usage chunk, NOT in the finish chunk too —
    emitting it in both places would have aggregating clients
    double-count tokens. DeepSeek review caught this on first pass; a
    later refactor that re-introduces the duplication trips this gate.

    When ``include_usage`` is False / unset (D-SSE-USAGE, v0.8.2),
    usage MUST be absent from EVERY chunk including the finish chunk —
    per the OpenAI streaming spec. The two pin assertions below lock
    both branches.
    """
    engine = _GuidedEngine(guided_text=_GUIDED_OUTPUT)
    client = _make_client(engine)

    # include_usage=True branch: usage only in dedicated chunk.
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": True},
            },
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    finish_events = [
        e for e in events for c in e.get("choices", []) if c.get("finish_reason")
    ]
    usage_only_events = [e for e in events if not e.get("choices") and e.get("usage")]
    assert len(finish_events) == 1, "exactly one finish chunk expected"
    assert finish_events[0].get("usage") is None, (
        "finish chunk must NOT carry usage when include_usage=True — "
        "double-emission would have clients double-count tokens"
    )
    assert len(usage_only_events) == 1, (
        "expected exactly one dedicated usage chunk when include_usage=True"
    )

    # All chunks in one completion stream must share a single ``created``
    # timestamp per the OpenAI streaming spec. The new helper pre-computes
    # ``_sse_created`` and passes it explicitly to ChatCompletionChunk —
    # without that, ``ChatCompletionChunk.created`` would default-factory
    # to a fresh ``int(time.time())`` per instantiation and break the
    # invariant (DeepSeek pr_validate round 2 finding).
    created_values = {e["created"] for e in events if "created" in e}
    assert len(created_values) == 1, (
        f"all SSE chunks must share one created timestamp; saw {created_values}"
    )

    # include_usage default-False branch (D-SSE-USAGE, v0.8.2):
    # ``usage`` MUST be absent from EVERY chunk including the finish
    # chunk. Pre-fix the finish chunk carried a populated usage block
    # under the "legacy bare-client accommodation" — LangChain /
    # AI-SDK / vercel-ai-stream parsers double-counted as a result.
    engine2 = _GuidedEngine(guided_text=_GUIDED_OUTPUT)
    client2 = _make_client(engine2)
    resp2 = client2.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": True},
            },
        },
    )
    assert resp2.status_code == 200, resp2.text
    events2, _ = _parse_sse_events(resp2.text)
    finish_events2 = [
        e for e in events2 for c in e.get("choices", []) if c.get("finish_reason")
    ]
    usage_only_events2 = [e for e in events2 if not e.get("choices") and e.get("usage")]
    assert len(finish_events2) == 1
    assert finish_events2[0].get("usage") is None, (
        "finish chunk MUST NOT carry usage when include_usage is unset — "
        "OpenAI streaming spec requires opt-in via stream_options"
    )
    assert usage_only_events2 == [], (
        "no dedicated usage chunk when include_usage is unset"
    )
    any_usage_key2 = [e for e in events2 if "usage" in e]
    assert any_usage_key2 == [], (
        f"no SSE chunk may carry the usage KEY when include_usage is "
        f"unset; got {len(any_usage_key2)} chunk(s) with the key "
        f'(includes regressions to ``"usage": null``)'
    )


def test_streaming_guided_fallback_preserves_id_and_created():
    """Fallback to unconstrained streaming must share id/created with
    what the outer helper would emit on the success path. Without this,
    a client tracking the completion id across the guided→unconstrained
    handoff would see two different ids/timestamps for what is logically
    one request (DeepSeek pr_validate round 5 finding).

    The contract is enforced by passing ``response_id`` and ``created``
    kwargs to ``stream_chat_completion``. The mock fallback stream emits
    its standard chunks; this test reassembles them and asserts every
    chunk shares one id and one created value (the outer helper's).

    H-06 note: this test asserts the suggestion-only contract
    (``strict=False``). Under ``strict=True``, the H-06 fix
    refuses the unconstrained fallback entirely and emits a
    canonical SSE error envelope instead — covered by
    ``test_strict_true_streaming_guided_raises_emits_error_sse_no_fallback``
    in ``test_response_format_json_schema_strict.py``.
    """
    engine = _GuidedEngine(raise_in_guided=True)
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "pick a color"}],
            "response_format": {
                "type": "json_schema",
                # strict=False: suggestion-only, fallback IS legal
                # under this contract — that's what this test pins.
                "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": False},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done

    ids = {e["id"] for e in events if "id" in e}
    createds = {e["created"] for e in events if "created" in e}
    role_events = [
        e
        for e in events
        for choice in e.get("choices", [])
        if (choice.get("delta") or {}).get("role") == "assistant"
    ]
    assert len(role_events) == 1, "guided fallback must not duplicate the role frame"
    assert len(ids) == 1, (
        f"all chunks must share one id across the guided→unconstrained "
        f"fallback handoff; saw {ids}"
    )
    assert len(createds) == 1, (
        f"all chunks must share one created timestamp across the "
        f"guided→unconstrained fallback handoff; saw {createds}"
    )


def test_streaming_guided_falls_back_to_unconstrained_on_engine_failure():
    """If generate_with_schema raises, the helper must fall back to
    stream_chat so the request still returns a response.

    Fallback rationale: a failure in guided decoding (llguidance import
    error at runtime, grammar compilation error on a pathological schema,
    etc.) under
    ``strict=False`` (suggestion-only) should degrade to unconstrained
    generation rather than 500. Clients in suggestion-only use cases
    can validate the response themselves; defensive servers log the
    failure with full traceback (via logger.exception in
    GuidedGenerator.generate_json) so the regression surfaces in ops
    visibility — see knowledge/sop_gap_guided_schema_passthrough.md.

    H-06 note: ``strict=True`` is now an explicit contract — the
    fix refuses the fallback and surfaces the breach as either a
    502 (non-stream) or a canonical SSE error envelope (stream).
    See ``test_response_format_json_schema_strict.py`` for those
    contract-level pins.
    """
    engine = _GuidedEngine(raise_in_guided=True)
    client = _make_client(engine)

    payload = {
        "model": "test-model",
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "pick a color"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": False},
        },
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text

    # Guided path was attempted and raised; fallback unconstrained path
    # was exercised. Both calls must be recorded.
    assert len(engine.guided_calls) == 1
    assert len(engine.stream_calls) == 1

    _, saw_done = _parse_sse_events(resp.text)
    assert saw_done, "fallback streaming response must still terminate with [DONE]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
