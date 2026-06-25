# SPDX-License-Identifier: Apache-2.0
"""rapid-mlx#254 — `/v1/responses` must accept the canonical OpenAI
input shape `{role, content}` without an explicit `type: "message"`.

Background
----------
OpenAI's Responses-API spec lets clients omit ``type`` on plain message
items. The official ``openai-python`` SDK always normalizes the item to
``type="message"`` before putting it on the wire, so SDK-mediated traffic
never hit the bug. Raw REST consumers (curl, fetch, openai-go,
openai-js's raw HTTP path) and anyone copy-pasting from the OpenAI docs
sent ``{"role":"user","content":"hi"}`` and got::

    400  input.0.type: Field required

because rapid-mlx's ``ResponsesInputItem`` declared ``type: str`` with no
default. Fix: ``mode="before"`` model_validator on
``ResponsesInputItem`` defaults ``type`` to ``"message"`` when absent
and ``role`` is present (the message-shape marker). Other variants
(function_call / function_call_output / reasoning) still require an
explicit ``type`` — the closed-set discriminator the downstream adapter
relies on stays load-bearing for those.

Contract pinned by this file
----------------------------
1. ``{role,content}`` (no ``type``)              → 200, parses as message
2. ``{type:"message",role,content}`` (explicit) → 200, parses as message
3. ``{type:"function_call", call_id, name, ...}`` (other variant)
                                                 → 200, parses as function_call
4. ``{role}`` only, no ``content``               → 400 (content gate
   downstream — we loosen the *type* default, not the *content*
   requirement).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Lightweight engine shim — mirrors tests/test_responses_param_validation.py
# so the test suite does not need to spin a real MLX engine.
# ---------------------------------------------------------------------------


class _Tokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:
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
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""


class _Engine:
    preserve_native_tool_format = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        yield _GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )


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


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


@pytest.fixture
def responses_client(monkeypatch):
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
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

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


HEADERS = {"Authorization": "Bearer test-secret"}


# =============================================================================
# Pydantic-layer unit tests — no route, just the schema contract
# =============================================================================


class TestResponsesInputItemTypeDefault:
    """Direct ``ResponsesInputItem`` tests — proves the model_validator
    runs at the right layer regardless of whether the route is wired up.
    Independent of FastAPI / TestClient so a route-level regression
    cannot mask a Pydantic-level regression."""

    def test_message_shape_without_type_defaults_to_message(self):
        from vllm_mlx.api.responses_models import ResponsesRequest

        req = ResponsesRequest(
            model="test-model",
            input=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(req.input, list)
        assert req.input[0].type == "message"
        assert req.input[0].role == "user"
        assert req.input[0].content == "hi"

    def test_explicit_type_message_unchanged(self):
        from vllm_mlx.api.responses_models import ResponsesRequest

        req = ResponsesRequest(
            model="test-model",
            input=[{"type": "message", "role": "user", "content": "hi"}],
        )
        assert req.input[0].type == "message"

    def test_function_call_variant_still_requires_explicit_type(self):
        """Other discriminator branches must keep their explicit-tag
        contract — only the message shape is loosened."""
        from vllm_mlx.api.responses_models import ResponsesRequest

        req = ResponsesRequest(
            model="test-model",
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": "{}",
                }
            ],
        )
        assert req.input[0].type == "function_call"
        assert req.input[0].call_id == "call_abc"

    def test_item_with_no_type_and_no_role_still_rejected(self):
        """An empty ``{}`` item has no role marker — we do NOT silently
        treat it as a message; the discriminator gate still fires."""
        from pydantic import ValidationError

        from vllm_mlx.api.responses_models import ResponsesRequest

        with pytest.raises(ValidationError):
            ResponsesRequest(model="test-model", input=[{}])


# =============================================================================
# Route-level integration — proves the wire contract end-to-end
# =============================================================================


class TestResponsesRouteInputTypeDefault:
    def test_canonical_openai_shape_without_type_returns_200(
        self, responses_client
    ):
        """The bug report payload: copy-pasted-from-OpenAI-docs curl
        with ``{role, content}`` and no ``type``. Pre-fix → 400, post-fix
        → 200."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [{"role": "user", "content": "hi"}],
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        engine = responses_client.engine
        assert len(engine.calls) == 1, "engine should have been invoked once"
        # Adapter should have produced a normal user message. The engine
        # receives the messages as dicts (the OpenAI chat-completions wire
        # shape) — assert on dict keys rather than object attributes so a
        # future refactor that swaps the dataclass shape can't mask a
        # contract regression.
        messages = engine.calls[0].messages
        assert messages[-1]["role"] == "user"

    def test_explicit_message_type_still_returns_200(self, responses_client):
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "hi"}
                ],
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    def test_function_call_variant_still_routes_via_discriminator(
        self, responses_client
    ):
        """A real function_call item (with explicit ``type``) must still
        flow through ``_function_call_to_chat`` — proves the loosening
        is scoped to the message-shape branch only."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"role": "user", "content": "use the tool please"},
                    {
                        "type": "function_call",
                        "call_id": "call_abc",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_abc",
                        "output": "sunny",
                    },
                ],
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        engine = responses_client.engine
        messages = engine.calls[0].messages
        # The discriminator must still treat the function_call /
        # function_call_output items distinctly from the message item —
        # we don't pin the exact downstream wire shape (which may be
        # either a raw ``tool_calls`` array or a synthesised text
        # transcript depending on the chat_template), only that the
        # request reached the engine at all (i.e. no Pydantic 400) and
        # the leading user turn round-tripped its content. That is the
        # load-bearing parity guarantee: function_call items are still
        # parsed through the function_call branch — not silently coerced
        # into the message branch by our type-defaulting validator.
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assert any("use the tool please" in (m.get("content") or "") for m in user_msgs), (
            f"leading user message lost during conversion: {messages}"
        )
        # And the function_call_output payload must have shown up
        # somewhere in the message stream — that proves the
        # function_call_output branch ran (it would NOT run if our
        # validator had folded the explicit-type item into the message
        # branch).
        assert any("sunny" in (m.get("content") or "") for m in messages), (
            f"function_call_output payload not propagated: {messages}"
        )

    def test_role_only_no_content_still_400(self, responses_client):
        """We loosen the ``type`` default, NOT the content requirement.
        ``{"role":"user"}`` (no content) must still 400 — the content
        gate fires at the adapter layer with a clear ``message content
        is required`` error."""
        client = responses_client.client
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [{"role": "user"}],
            },
            headers=HEADERS,
        )
        assert resp.status_code == 400, resp.text
