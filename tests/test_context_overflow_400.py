# SPDX-License-Identifier: Apache-2.0
"""R6-H5: cross-route context-window enforcement (40K-token prompt at
``context_window=40960`` must return HTTP 400 ``context_length_exceeded``).

The 0.8.7 dogfood (Hiro R1) flagged that the four chat surfaces —
``/v1/chat/completions``, ``/v1/completions``, ``/v1/messages``, and
``/v1/responses`` — must each call the ``enforce_context_length*``
helpers BEFORE handing the request to the engine. Pre-fix the existing
``test_context_length_exceeded.py`` exercised only the helper in
isolation. These tests build a fake engine whose model exposes
``max_position_embeddings = 40960`` (matching qwen3-0.6b-8bit) and
verify the structured 400 lands on every route, so a route-level
refactor that silently drops the helper call cannot regress past
CI.

Each test uses a stub engine — no real model load, no Apple Silicon
GPU needed — and a deterministic 4-chars-per-token tokenizer so the
test can target the exact ``prompt + max_tokens > context_window``
boundary without flaky BPE drift.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config

_CONTEXT_WINDOW = 40960


class _StubArgs:
    max_position_embeddings = _CONTEXT_WINDOW


class _StubModel:
    args = _StubArgs()


class _StubTokenizer:
    """Deterministic tokenizer: 1 token per 4 chars. Used by the
    helper's ``count_prompt_tokens`` so the test can hit the exact
    boundary without depending on a real BPE."""

    model_max_length = _CONTEXT_WINDOW
    bos_token = None

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return [0] * max(1, len(text) // 4)

    # FastAPI/responses routes pass tools through ``convert_tools_for_template``;
    # the chat-template render of the user message is what we count.
    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
        **_kwargs,
    ):
        """Trivial template: join the user contents with newlines so
        the tokenizer's char/4 heuristic gets us the prompt-token
        count we want."""
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
            else:
                parts.append(content)
        return "\n".join(parts)


class _StubEngine:
    """Stub for the text-only AR engine. Exposes the minimal surface
    each route's pre-engine validation reads. ``build_prompt`` calls
    the tokenizer's chat-template path so the helper-side token count
    matches the route's own resolved prompt.
    """

    is_mllm = False
    preserve_native_tool_format = False
    supports_guided_generation = False

    def __init__(self):
        self._model = _StubModel()
        self.tokenizer = _StubTokenizer()
        self._tokenizer = self.tokenizer

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )

    async def chat(self, **kwargs):  # noqa: ARG002
        # Should never be reached on the 400 path; raising makes the
        # failure mode loud if the gate is bypassed.
        raise AssertionError("engine.chat must not be reached on the 400 path")

    async def stream_chat(self, *args, **kwargs):  # noqa: ARG002
        raise AssertionError("engine.stream_chat must not be reached on the 400 path")


def _make_app(
    routes: list[Any],
    *,
    max_prompt_tokens: int | None = None,
    context_length: int | None = None,
    engine: _StubEngine | None = None,
) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine or _StubEngine()
    cfg.model_name = "qwen3-0.6b-8bit"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    cfg.reasoning_parser_name = None
    cfg.default_max_tokens = 1024
    cfg.thinking_token_budget = 0
    cfg.max_prompt_tokens = max_prompt_tokens
    cfg.context_length = context_length

    app = FastAPI()
    for router in routes:
        app.include_router(router)
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers

    install_exception_handlers(app)
    return TestClient(app)


def _huge_text(approx_tokens: int) -> str:
    """Build a string the stub tokenizer maps to ``approx_tokens``
    tokens (4 chars / token)."""
    return "x" * (approx_tokens * 4)


def _extract_error(body: dict) -> dict:
    """Pull the OpenAI-style error envelope out of a FastAPI response.

    The structured 400 handler in ``rapid_mlx/server.py`` unwraps
    ``HTTPException(detail={"error": {...}})`` into a top-level
    ``{"error": {...}}`` body. When the route is mounted on a bare
    FastAPI app (these tests do that to avoid the full server
    bootstrap), FastAPI's default handler wraps the same dict under
    ``"detail"`` instead. Accept both shapes so the assertion targets
    the canonical fields.
    """
    if isinstance(body.get("error"), dict):
        return body["error"]
    if isinstance(body.get("detail"), dict):
        inner = body["detail"]
        if isinstance(inner.get("error"), dict):
            return inner["error"]
        return inner
    return body


def test_expanded_media_context_error_has_token_counts_and_remedy():
    from rapid_mlx.request import ClientRequestError
    from rapid_mlx.service.helpers import context_overflow_from_client_error

    error = context_overflow_from_client_error(
        ClientRequestError(
            "context_length_exceeded: prompt has 101 tokens after media "
            "expansion, exceeding --context-length 80"
        )
    )
    assert error is not None
    assert error.status_code == 400
    assert error.prompt_tokens == 101
    assert error.limit == 80
    assert error.detail["error"]["code"] == "context_length_exceeded"
    assert "--context-length" in error.detail["error"]["message"]
    assert context_overflow_from_client_error(ValueError("internal")) is None
    assert context_overflow_from_client_error(ClientRequestError("other")) is None


def test_generation_window_error_names_serve_override():
    from rapid_mlx.service.helpers import context_window_exhausted

    cfg = reset_config()
    cfg.context_length = 80
    engine = _StubEngine()
    assert context_window_exhausted(engine, 70, 9, "length") is None
    error = context_window_exhausted(engine, 70, 10, "length")
    assert error is not None
    assert error.prompt_tokens == 81
    assert "--context-length" in error.detail["error"]["message"]


def test_generic_exception_fallback_keeps_anthropic_overflow_wording():
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.service.helpers import ContextLengthExceeded

    app = FastAPI()
    install_exception_handlers(app)
    handler = app.exception_handlers[Exception]
    error = ContextLengthExceeded(prompt_tokens=101, limit=80, message="overflow")
    request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"))
    response = asyncio.run(handler(request, error))
    assert response.status_code == 400
    assert response.body == (
        b'{"type":"error","error":{"type":"invalid_request_error",'
        b'"message":"prompt is too long: 101 tokens > 80 maximum"}}'
    )


def test_unrelated_anthropic_400_keeps_origin_main_golden_shape():
    from fastapi import HTTPException

    from rapid_mlx.middleware.exception_handlers import install_exception_handlers

    app = FastAPI()

    @app.get("/v1/messages")
    def fail():
        raise HTTPException(status_code=400, detail="bad input")

    install_exception_handlers(app)
    response = TestClient(app).get("/v1/messages")
    assert response.status_code == 400
    assert response.content == (
        b'{"type":"error","error":{"message":"bad input",'
        b'"type":"invalid_request_error","code":null,"param":null}}'
    )


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
)
def test_late_stream_guard_preserves_context_protocol(path):
    from rapid_mlx.request import ClientRequestError
    from rapid_mlx.service.helpers import _disconnect_guard

    async def failing_stream():
        yield "data: first\n\n"
        raise ClientRequestError(
            "context_length_exceeded: prompt has 101 tokens after media "
            "expansion, exceeding --context-length 80"
        )

    async def is_disconnected():
        return False

    async def collect():
        request = SimpleNamespace(
            url=SimpleNamespace(path=path), is_disconnected=is_disconnected
        )
        return [
            chunk
            async for chunk in _disconnect_guard(
                failing_stream(), request, keepalive_seconds=0
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks[0] == "data: first\n\n"
    wire = "".join(chunks[1:])
    if path == "/v1/messages":
        assert "event: error" in wire
        assert "prompt is too long: 101 tokens > 80 maximum" in wire
    elif path == "/v1/responses":
        assert "event: response.failed" in wire
        assert "context_length_exceeded" in wire
    else:
        assert "context_length_exceeded" in wire
        assert "data: [DONE]" in wire


@pytest.mark.parametrize("failure", ["window_end", "expanded_prompt", "guided_budget"])
def test_guided_chat_stream_reports_context_error(failure):
    from rapid_mlx.api.errors import GuidedTokenLimitError
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.request import ClientRequestError
    from rapid_mlx.routes.chat import stream_chat_completion_guided

    class GuidedEngine(_StubEngine):
        async def generate_with_schema(self, **kwargs):
            kwargs["request_admitted_event"].set()
            if failure == "expanded_prompt":
                raise ClientRequestError(
                    "context_length_exceeded: prompt has 101 tokens after media "
                    "expansion, exceeding --context-length 80"
                )
            if failure == "guided_budget":
                raise GuidedTokenLimitError(70, 10)
            return GenerationOutput(
                text="{}",
                new_text="{}",
                prompt_tokens=70,
                completion_tokens=10,
                finished=True,
                finish_reason="length",
            )

    cfg = reset_config()
    cfg.context_length = 80
    request = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    async def collect():
        return [
            chunk
            async for chunk in stream_chat_completion_guided(
                GuidedEngine(),
                request.messages,
                request,
                {"type": "object"},
                strict_mode=True,
            )
        ]

    wire = "".join(asyncio.run(collect()))
    assert "context_length_exceeded" in wire
    assert "strict_schema_violation" not in wire
    assert wire.endswith("data: [DONE]\n\n")


def test_guided_budget_only_reports_context_when_window_is_full():
    from rapid_mlx.api.errors import GuidedTokenLimitError
    from rapid_mlx.service.helpers import context_overflow_from_guided_limit

    cfg = reset_config()
    cfg.context_length = 80
    engine = _StubEngine()
    assert (
        context_overflow_from_guided_limit(engine, GuidedTokenLimitError(70, 5)) is None
    )
    error = context_overflow_from_guided_limit(engine, GuidedTokenLimitError(70, 10))
    assert error is not None
    assert error.detail["error"]["code"] == "context_length_exceeded"


def test_guided_token_budget_signal_survives_wrappers(monkeypatch):
    from rapid_mlx.api import guided
    from rapid_mlx.api.errors import GuidedTokenLimitError
    from rapid_mlx.engine import batched

    signal = GuidedTokenLimitError(70, 10)

    class Matcher:
        @staticmethod
        def grammar_from_json_schema(*args, **kwargs):  # noqa: ARG004
            return "grammar"

    def exhausted(*args, **kwargs):  # noqa: ARG001
        raise signal

    monkeypatch.setattr(guided, "LLMatcher", Matcher)
    generator = guided.GuidedGenerator(object(), object())
    monkeypatch.setattr(generator, "_decode_constrained", exhausted)
    with pytest.raises(GuidedTokenLimitError) as caught:
        generator.generate_json("hi", {"type": "object"})
    assert caught.value is signal
    with pytest.raises(GuidedTokenLimitError):
        generator.generate_json_object("hi")

    monkeypatch.setattr(guided, "HAS_LLGUIDANCE", True)
    monkeypatch.setattr(guided, "GuidedGenerator", lambda *_args: generator)
    with pytest.raises(GuidedTokenLimitError):
        guided.generate_with_schema(object(), object(), "hi", {"type": "object"})

    monkeypatch.setattr(batched, "GuidedGenerator", lambda *_args: generator)
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._model = object()
    engine._tokenizer = object()
    engine._is_mllm = False
    with pytest.raises(GuidedTokenLimitError):
        engine._run_guided_generation("hi", {"type": "object"}, 10, 0.0)


@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_strict_guided_budget_exhaustion_uses_context_error(surface, stream):
    from rapid_mlx.api.errors import GuidedTokenLimitError
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.responses import router as responses_router

    class GuidedEngine(_StubEngine):
        supports_guided_generation = True

        async def generate_with_schema(self, **kwargs):  # noqa: ARG002
            raise GuidedTokenLimitError(70, 10)

    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    cases = {
        "chat": (
            chat_router,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Result",
                        "schema": schema,
                        "strict": True,
                    },
                },
            },
        ),
        "responses": (
            responses_router,
            "/v1/responses",
            {
                "input": "hi",
                "max_output_tokens": 16,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "Result",
                        "schema": schema,
                        "strict": True,
                    }
                },
            },
        ),
    }
    router, path, payload = cases[surface]
    response = _make_app([router], context_length=80, engine=GuidedEngine()).post(
        path, json={"model": "qwen3-0.6b-8bit", "stream": stream, **payload}
    )
    assert "context_length_exceeded" in response.text, response.text
    assert "strict_schema_violation" not in response.text


@pytest.mark.parametrize("upstream_kind", ["event", "exception"])
def test_strict_chat_wrapper_preserves_context_error(monkeypatch, upstream_kind):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.request import ClientRequestError
    from rapid_mlx.routes import chat as chat_module

    def raise_overflow():
        raise ClientRequestError(
            "context_length_exceeded: prompt has 101 tokens after media "
            "expansion, exceeding --context-length 80"
        )

    async def upstream(*args, **kwargs):  # noqa: ARG001
        if upstream_kind == "event":
            yield (
                "event: chat.completion.error\ndata: "
                '{"error":{"code":"context_length_exceeded"}}\n\n'
            )
        else:
            yield raise_overflow()

    monkeypatch.setattr(chat_module, "stream_chat_completion", upstream)
    request = ChatCompletionRequest(
        model="qwen3-0.6b-8bit",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    async def collect():
        return [
            chunk
            async for chunk in chat_module.stream_chat_completion_strict_postgen(
                _StubEngine(), request.messages, request, {"type": "object"}
            )
        ]

    wire = "".join(asyncio.run(collect()))
    assert "context_length_exceeded" in wire
    assert "strict_schema_violation" not in wire
    assert wire.count("event: chat.completion.error") == 1
    assert wire.endswith("data: [DONE]\n\n")


# ─── /v1/chat/completions ───────────────────────────────────────────


def test_chat_completions_rejects_over_context_window():
    """``/v1/chat/completions`` must surface the structured 400
    envelope when ``prompt + max_tokens > context_window``."""
    from rapid_mlx.routes.chat import router as chat_router

    client = _make_app([chat_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": _huge_text(41_000)}],
        "max_tokens": 16,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"
    assert err.get("type") == "invalid_request_error"
    assert str(_CONTEXT_WINDOW) in err.get("message", "")


def test_chat_completions_rejects_over_operational_prompt_cap_before_engine():
    """A serve-level prompt cap wins even when the model window has room."""
    from rapid_mlx.routes.chat import router as chat_router

    client = _make_app([chat_router], max_prompt_tokens=16_384)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "messages": [{"role": "user", "content": _huge_text(20_000)}],
            "max_tokens": 16,
        },
    )

    assert resp.status_code == 400, resp.text
    err = _extract_error(resp.json())
    assert err.get("code") == "context_length_exceeded"
    assert "16384" in err.get("message", "")


def test_context_length_override_applies_across_generation_routes():
    """A user window below the native 40K limit applies to every API lane."""
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.completions import router as completions_router
    from rapid_mlx.routes.responses import router as responses_router

    cases = [
        (
            chat_router,
            "/v1/chat/completions",
            {
                "model": "qwen3-0.6b-8bit",
                "messages": [{"role": "user", "content": _huge_text(100)}],
                "max_tokens": 16,
            },
        ),
        (
            completions_router,
            "/v1/completions",
            {
                "model": "qwen3-0.6b-8bit",
                "prompt": _huge_text(100),
                "max_tokens": 16,
            },
        ),
        (
            anthropic_router,
            "/v1/messages",
            {
                "model": "qwen3-0.6b-8bit",
                "messages": [{"role": "user", "content": _huge_text(100)}],
                "max_tokens": 16,
            },
        ),
        (
            responses_router,
            "/v1/responses",
            {
                "model": "qwen3-0.6b-8bit",
                "input": _huge_text(100),
                "max_output_tokens": 16,
            },
        ),
    ]
    for router, path, payload in cases:
        client = _make_app([router], context_length=80)
        resp = client.post(path, json=payload)
        assert resp.status_code == 400, (path, resp.text)
        err = _extract_error(resp.json())
        if path == "/v1/messages":
            assert err == {
                "type": "invalid_request_error",
                "message": "prompt is too long: 100 tokens > 80 maximum",
            }
        else:
            assert err.get("code") == "context_length_exceeded"
            assert "80" in err.get("message", "")


def test_context_length_override_keeps_native_model_window_visible():
    from types import SimpleNamespace

    from rapid_mlx.routes.models import (
        _resolve_context_window,
        _resolve_max_model_len,
    )

    cfg = reset_config()
    cfg.engine = _StubEngine()
    cfg.engine.scheduler = SimpleNamespace(
        projected_memory_max_context=lambda native: 4_096
    )
    cfg.model_name = "qwen3-0.6b-8bit"
    cfg.model_registry = None
    assert _resolve_max_model_len(cfg.model_name, _CONTEXT_WINDOW) == 4_096
    cfg.context_length = 32_768
    assert _resolve_context_window(cfg.model_name) == _CONTEXT_WINDOW
    assert _resolve_max_model_len(cfg.model_name, _CONTEXT_WINDOW) == 32_768


def test_context_length_above_model_declared_window_is_rejected():
    from fastapi import HTTPException

    from rapid_mlx.service.helpers import get_model_max_context

    cfg = reset_config()
    cfg.context_length = _CONTEXT_WINDOW + 1
    with pytest.raises(HTTPException) as excinfo:
        get_model_max_context(_StubEngine())
    assert excinfo.value.status_code == 400
    assert "declared" in excinfo.value.detail["error"]["message"]


# ─── /v1/completions ────────────────────────────────────────────────


def test_completions_rejects_over_context_window():
    """``/v1/completions`` (raw-prompt API) must enforce the same
    cap. The helper here is ``enforce_context_length_for_prompt``
    — no chat template applied."""
    from rapid_mlx.routes.completions import router as completions_router

    client = _make_app([completions_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "prompt": _huge_text(41_000),
        "max_tokens": 16,
    }
    resp = client.post("/v1/completions", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"


# ─── /v1/messages (Anthropic) ───────────────────────────────────────


def test_anthropic_messages_rejects_over_context_window():
    """``/v1/messages`` (Anthropic shape) must enforce the same cap.
    Anthropic SDKs branch on ``error.type`` so we pin the envelope
    matches the chat lane."""
    from rapid_mlx.routes.anthropic import router as anthropic_router

    client = _make_app([anthropic_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": _huge_text(41_000)}],
        "max_tokens": 16,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"].startswith("prompt is too long: ")
    assert body["error"]["message"].endswith(" tokens > 40960 maximum")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["chat", "responses", "messages"])
def test_agent_routes_surface_protocol_correct_context_error(route, stream):
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.responses import router as responses_router

    if route == "chat":
        router, path = chat_router, "/v1/chat/completions"
        payload = {"messages": [{"role": "user", "content": _huge_text(100)}]}
    elif route == "responses":
        router, path = responses_router, "/v1/responses"
        payload = {"input": _huge_text(100)}
    else:
        router, path = anthropic_router, "/v1/messages"
        payload = {"messages": [{"role": "user", "content": _huge_text(100)}]}
    payload.update({"model": "qwen3-0.6b-8bit", "stream": stream})
    if route == "messages":
        payload["max_tokens"] = 16
    elif route == "responses":
        payload["max_output_tokens"] = 16
    else:
        payload["max_tokens"] = 16

    resp = _make_app([router], context_length=80).post(path, json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    if route == "messages":
        assert body == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 100 tokens > 80 maximum",
            },
        }
    else:
        error = body["error"]
        assert error["code"] == "context_length_exceeded"
        assert "100 tokens" in error["message"]
        assert "80 tokens" in error["message"]
        assert "Start a new session or compact" in error["message"]
        assert "--context-length" in error["message"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("route", ["chat", "responses", "messages"])
def test_agent_routes_surface_operational_prompt_cap(route, stream):
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.responses import router as responses_router

    cases = {
        "chat": (
            chat_router,
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": _huge_text(100)}],
                "max_tokens": 16,
            },
        ),
        "responses": (
            responses_router,
            "/v1/responses",
            {"input": _huge_text(100), "max_output_tokens": 16},
        ),
        "messages": (
            anthropic_router,
            "/v1/messages",
            {
                "messages": [{"role": "user", "content": _huge_text(100)}],
                "max_tokens": 16,
            },
        ),
    }
    router, path, payload = cases[route]
    response = _make_app([router], max_prompt_tokens=80).post(
        path, json={"model": "qwen3-0.6b-8bit", "stream": stream, **payload}
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    if route == "messages":
        assert error == {
            "type": "invalid_request_error",
            "message": "prompt is too long: 100 tokens > 80 maximum",
        }
    else:
        assert error["code"] == "context_length_exceeded"
        assert "100 tokens" in error["message"]
        assert "80 tokens" in error["message"]
        assert "--max-prompt-tokens" in error["message"]


# ─── /v1/responses ──────────────────────────────────────────────────


def test_responses_rejects_over_context_window():
    """``/v1/responses`` (OpenAI Responses API) must enforce the
    same cap. The route re-extracts multimodal content before the
    gate, so this also pins that the gate fires on the re-extracted
    text-only shape."""
    from rapid_mlx.routes.responses import router as responses_router

    client = _make_app([responses_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "input": _huge_text(41_000),
        "max_output_tokens": 16,
    }
    resp = client.post("/v1/responses", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"


# ─── Under-cap requests still pass through the gate ────────────────


def test_chat_completions_passes_when_within_context_window():
    """Sanity check: a prompt that fits inside ``prompt + max_tokens
    <= context_window`` must NOT be rejected. Locks in that the
    enforcement is bounded — over-strict gates would block legitimate
    long-context requests."""
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes.chat import router as chat_router

    # Build the app, then swap in a chat impl that returns a real
    # response (the default stub raises on .chat to catch silent
    # bypass on the 400 path).
    cfg = reset_config()
    engine = _StubEngine()

    async def _chat(**kwargs):  # noqa: ARG001
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=10,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    engine.chat = _chat  # type: ignore[assignment]

    cfg.engine = engine
    cfg.model_name = "qwen3-0.6b-8bit"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    cfg.reasoning_parser_name = None
    cfg.default_max_tokens = 1024
    cfg.thinking_token_budget = 0

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    # 30K tokens + max_tokens 16 = 30016 < 40960. Must pass.
    payload = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": _huge_text(30_000)}],
        "max_tokens": 16,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text


class _ClampEngine(_StubEngine):
    def __init__(self):
        super().__init__()
        self.captured_max_tokens: list[int | None] = []

    def _output(self, max_tokens):
        from rapid_mlx.engine.base import GenerationOutput

        self.captured_max_tokens.append(max_tokens)
        return GenerationOutput(
            text="x" * int(max_tokens or 0),
            new_text="x" * int(max_tokens or 0),
            prompt_tokens=_CONTEXT_WINDOW - 10,
            completion_tokens=int(max_tokens or 0),
            finished=True,
            finish_reason="length",
            channel=None,
        )

    async def chat(self, *args, **kwargs):  # noqa: ARG002
        return self._output(kwargs.get("max_tokens"))

    async def generate(self, **kwargs):
        return self._output(kwargs.get("max_tokens"))

    async def stream_chat(self, *args, **kwargs):  # noqa: ARG002
        yield self._output(kwargs.get("max_tokens"))


class _LateOverflowEngine(_StubEngine):
    @staticmethod
    def _overflow():
        from rapid_mlx.request import ClientRequestError

        raise ClientRequestError(
            "context_length_exceeded: prompt has 101 tokens after media "
            "expansion, exceeding --context-length 80"
        )

    async def chat(self, **kwargs):  # noqa: ARG002
        return self._overflow()

    async def stream_chat(self, **kwargs):  # noqa: ARG002
        yield self._overflow()


def test_chat_mllm_preflight_returns_400_before_stream_commit():
    from rapid_mlx.routes.chat import router as chat_router

    class MllmLateOverflowEngine(_LateOverflowEngine):
        is_mllm = True

    response = _make_app(
        [chat_router], context_length=80, engine=MllmLateOverflowEngine()
    ).post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "context_length_exceeded"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_expanded_media_late_overflow_uses_route_protocol(surface, stream):
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.responses import router as responses_router

    cases = {
        "chat": (
            chat_router,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        ),
        "responses": (
            responses_router,
            "/v1/responses",
            {"input": "hi", "max_output_tokens": 16},
        ),
        "messages": (
            anthropic_router,
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
        ),
    }
    router, path, payload = cases[surface]
    response = _make_app(
        [router], context_length=80, engine=_LateOverflowEngine()
    ).post(path, json={"model": "qwen3-0.6b-8bit", "stream": stream, **payload})
    expected_status = 200 if stream else 400
    assert response.status_code == expected_status, response.text
    if response.status_code == 400:
        body = response.json()
        if surface == "messages":
            assert body["error"]["message"] == (
                "prompt is too long: 101 tokens > 80 maximum"
            )
        else:
            assert body["error"]["code"] == "context_length_exceeded"
    else:
        if surface == "responses":
            assert "event: response.failed" in response.text
            assert "context_length_exceeded" in response.text
        elif surface == "messages":
            assert "event: error" in response.text
            assert "prompt is too long: 101 tokens > 80 maximum" in response.text
        else:
            assert "context_length_exceeded" in response.text


@pytest.mark.parametrize("surface", ["chat", "completions", "responses", "messages"])
def test_routes_clamp_large_completion_budget_and_report_window_end(surface):
    """Every compatibility surface sends only the remaining budget downstream."""
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.completions import router as completions_router
    from rapid_mlx.routes.responses import router as responses_router

    prompt = _huge_text(_CONTEXT_WINDOW - 10)
    cases = {
        "chat": (
            chat_router,
            "/v1/chat/completions",
            {
                "model": "qwen3-0.6b-8bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
            },
        ),
        "completions": (
            completions_router,
            "/v1/completions",
            {
                "model": "qwen3-0.6b-8bit",
                "prompt": prompt,
                "max_tokens": 1000,
            },
        ),
        "responses": (
            responses_router,
            "/v1/responses",
            {
                "model": "qwen3-0.6b-8bit",
                "input": prompt,
                "max_output_tokens": 1000,
            },
        ),
        "messages": (
            anthropic_router,
            "/v1/messages",
            {
                "model": "qwen3-0.6b-8bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
            },
        ),
    }
    engine = _ClampEngine()
    router, path, payload = cases[surface]
    response = _make_app([router], engine=engine).post(path, json=payload)

    assert engine.captured_max_tokens == [10]
    body = response.json()
    if surface == "completions":
        assert response.status_code == 200, response.text
        assert body["choices"][0]["finish_reason"] == "length"
    elif surface == "messages":
        assert response.status_code == 400, response.text
        assert body["error"]["message"] == (
            "prompt is too long: 40961 tokens > 40960 maximum"
        )
    else:
        assert response.status_code == 400, response.text
        assert body["error"]["code"] == "context_length_exceeded"
        assert "40950 tokens" in body["error"]["message"]


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_stream_reports_generation_reaching_context_window(surface):
    from rapid_mlx.routes.anthropic import router as anthropic_router
    from rapid_mlx.routes.chat import router as chat_router
    from rapid_mlx.routes.responses import router as responses_router

    prompt = _huge_text(_CONTEXT_WINDOW - 10)
    cases = {
        "chat": (
            chat_router,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1000},
        ),
        "responses": (
            responses_router,
            "/v1/responses",
            {"input": prompt, "max_output_tokens": 1000},
        ),
        "messages": (
            anthropic_router,
            "/v1/messages",
            {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1000},
        ),
    }
    engine = _ClampEngine()
    router, path, payload = cases[surface]
    response = _make_app([router], engine=engine).post(
        path, json={"model": "qwen3-0.6b-8bit", "stream": True, **payload}
    )
    assert response.status_code == 200, response.text
    if surface == "chat":
        assert "event: chat.completion.error" in response.text
        assert '"code":"context_length_exceeded"' in response.text
    elif surface == "responses":
        assert "event: response.failed" in response.text
        assert '"code": "context_length_exceeded"' in response.text
    else:
        assert "event: error" in response.text
        assert "prompt is too long: 40961 tokens > 40960 maximum" in response.text


def test_responses_stream_context_error_precedes_required_tool_failure():
    import json

    from rapid_mlx.routes.responses import router

    response = _make_app([router], engine=_ClampEngine()).post(
        "/v1/responses",
        json={
            "model": "qwen3-0.6b-8bit",
            "stream": True,
            "input": _huge_text(_CONTEXT_WINDOW - 10),
            "max_output_tokens": 1000,
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Find a value",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    assert '"code": "context_length_exceeded"' in response.text
    assert "tool_choice_unfulfilled" not in response.text
    failed = next(
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and '"response.failed"' in line
    )
    assert failed["response"]["usage"]["input_tokens"] == _CONTEXT_WINDOW - 10
    assert failed["response"]["usage"]["output_tokens"] == 10


def test_empty_completion_prompt_keeps_requested_budget(monkeypatch):
    """An empty legacy prompt has no context cost and bypasses tokenization."""
    from rapid_mlx.service import helpers

    monkeypatch.setattr(
        helpers,
        "count_prompt_tokens",
        lambda *_args, **_kwargs: pytest.fail("empty prompt must not be tokenized"),
    )

    assert (
        helpers.enforce_context_length_for_prompt(_StubEngine(), "", max_tokens=321)
        == 321
    )
