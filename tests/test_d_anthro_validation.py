# SPDX-License-Identifier: Apache-2.0
"""D-ANTHRO-VALIDATION — four tightly-related Anthropic-spec gaps
surfaced by Sergei dogfood on 0.8.3.

Bug summaries:

* **F1** — ``/v1/messages`` error envelopes were missing the canonical
  Anthropic top-level ``{"type":"error","error":{...}}`` wrapper and
  the message body leaked the ``<field>`` placeholder text for
  schema-owned field names (the H-17 round-2 sanitizer collapsed every
  string ``loc`` component including safe schema-owned ones).
* **F4** — Malformed content blocks (``{type:"text"}`` with no
  ``text``, unknown block types, ``tool_use`` without ``id`` /
  ``name`` / ``input``) were accepted with HTTP 200 and the model ran
  on empty / unknown content.
* **F10** — Cross-role block-type violations slipped past the schema:
  a user-role message could carry a ``thinking`` block; an
  assistant-role message could carry a ``tool_result`` block. Both
  resulted in HTTP 200 with undefined semantics.
* **F11** — ``messages=[]`` on ``/v1/messages`` returned HTTP 500
  ``Internal server error`` instead of the documented 400
  ``invalid_request_error``.

All four are fixed at the Pydantic model layer (no per-route try/
except band-aids):

* The exception handlers in ``vllm_mlx/middleware/exception_handlers``
  detect Anthropic-prefixed paths and rewrap the OpenAI-shaped
  envelope into the Anthropic shape.
* ``_sanitize_loc`` now applies a closed allowlist of schema-owned
  field names so the safe names (``temperature``, ``messages``,
  ``max_tokens``, …) are echoed verbatim while attacker-controlled
  bytes still collapse to ``<field>``.
* ``AnthropicContentBlock._validate_block_shape`` rejects unknown
  ``type`` values and missing per-type required fields.
* ``AnthropicMessage._validate_role_block_compat`` enforces the
  role-block compatibility matrix.
* ``AnthropicRequest.messages`` gains ``min_length=1``;
  ``ChatCompletionRequest.messages`` does too for OpenAI-side parity.
* ``ResponsesRequest`` gains an after-validator that rejects empty
  ``input``.

The test app reuses the lightweight fixture from
``test_no_pydantic_error_leak`` so the four bugs share one fixture
surface.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Lightweight engine stubs (shape-compatible with the H-17 fixture) ─


class _Tokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str = ""
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


class _Engine:
    preserve_native_tool_format = False

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):  # noqa: ARG002
        return _GenerationOutput(text="hello", prompt_tokens=3, completion_tokens=1)


_IMPORTED_UNDER_LIGHTWEIGHT_ENGINE = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.anthropic",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "anthropic"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _install_lightweight_engine_modules(monkeypatch) -> None:
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


def _build_app(monkeypatch):
    previous_modules = {
        name: sys.modules.get(name, _MISSING)
        for name in _IMPORTED_UNDER_LIGHTWEIGHT_ENGINE
    }
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.anthropic import router as anthropic_router
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.api_key = None
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(anthropic_router)
    app.include_router(responses_router)
    # NOTE: chat_router intentionally NOT included here. The chat
    # route's module-level ``from ..engine import GenerationOutput``
    # binds at import time, and the lightweight engine stub installed
    # by ``_install_lightweight_engine_modules`` would cache stub
    # references that confuse later tests in the same session that
    # exercise the chat-route admission flow. The chat-route parity
    # checks live on the separate ``chat_client`` fixture below which
    # skips the lightweight stub.

    def teardown():
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

    return app, cfg, teardown


@pytest.fixture
def client(monkeypatch):
    app, cfg, teardown = _build_app(monkeypatch)
    try:
        yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    finally:
        teardown()


@pytest.fixture
def chat_client():
    """Separate fixture for chat-route parity checks (F1 + F11
    OpenAI surface). Does NOT use the lightweight engine stub — the
    chat route's module-level ``from ..engine import GenerationOutput``
    captures real names at import time; swapping them via a stub on
    the session-shared sys.modules leaks stub references into later
    admission-flow tests. Keeping the chat-router tests on a
    real-engine-imports fixture sidesteps that interaction entirely.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.chat import router as chat_router
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.api_key = None
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(responses_router)

    try:
        yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    finally:
        reset_config()
        rate_limiter._requests.clear()


# ============================================================
# F1 — Anthropic envelope wrapper + <field> placeholder leak
# ============================================================


class TestF1AnthropicErrorEnvelopeWrapper:
    """Every error path on /v1/messages must return the Anthropic
    canonical envelope ``{"type":"error","error":{...}}`` and must
    never leak the ``<field>`` placeholder for schema-owned field
    names."""

    @pytest.mark.parametrize(
        ("body", "expected_field"),
        [
            # Pre-fix Sergei F1 evidence — temperature type error.
            (
                {
                    "model": "test-model",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": "hot",
                },
                "temperature",
            ),
            # Missing required field — messages.
            ({"model": "test-model", "max_tokens": 10}, "messages"),
            # Wrong type for messages.
            (
                {"model": "test-model", "max_tokens": 10, "messages": "not-a-list"},
                "messages",
            ),
            # Wrong type for max_tokens.
            (
                {
                    "model": "test-model",
                    "max_tokens": "not-an-int",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "max_tokens",
            ),
        ],
    )
    def test_validation_error_returns_anthropic_envelope_with_named_field(
        self, client, body, expected_field
    ):
        response = client.client.post("/v1/messages", json=body)
        assert response.status_code == 400, response.text
        envelope = response.json()
        # Canonical Anthropic shape: ``{"type":"error","error":{...}}``.
        assert envelope.get("type") == "error", envelope
        assert isinstance(envelope.get("error"), dict), envelope
        err = envelope["error"]
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "invalid_request"
        # The named field must appear in the message — no <field> leak.
        assert expected_field in err["message"], err
        assert "<field>" not in err["message"], err

    def test_http_exception_paths_also_get_wrapper(self, client):
        """Non-validation 4xx paths (HTTPException) must also surface
        the wrapped envelope — Anthropic SDKs route on the outer
        ``type`` field regardless of which gate fired."""
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "banana"},  # M-03 gate
            },
        )
        assert response.status_code == 400, response.text
        envelope = response.json()
        assert envelope.get("type") == "error"
        assert isinstance(envelope.get("error"), dict)

    def test_malformed_json_body_returns_wrapper(self, client):
        """JSON decode error path must also wrap."""
        response = client.client.post(
            "/v1/messages",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400, response.text
        envelope = response.json()
        assert envelope.get("type") == "error"
        assert envelope["error"]["type"] == "invalid_request_error"

    def test_no_field_placeholder_in_user_visible_message(self, client):
        """Across the full bad-body matrix, the <field> placeholder
        must never appear in the user-visible message when the failing
        field is a schema-owned name."""
        for body in [
            {
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": "hot",
            },
            {
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": "nope",
            },
            {
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "top_k": "not-an-int",
            },
        ]:
            response = client.client.post("/v1/messages", json=body)
            assert response.status_code == 400, response.text
            assert "<field>" not in response.text, response.text

    def test_openai_chat_route_keeps_openai_envelope(self, chat_client):
        """The Anthropic wrapper must NOT apply to /v1/chat/completions
        — the OpenAI surface still returns the bare ``{"error":{...}}``
        shape."""
        response = chat_client.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": "hot",
            },
        )
        assert response.status_code == 400, response.text
        body = response.json()
        # OpenAI shape: top-level type is NOT "error".
        assert body.get("type") != "error", body
        # Error envelope intact at top level.
        assert isinstance(body.get("error"), dict)
        assert "temperature" in body["error"]["message"]
        assert "<field>" not in body["error"]["message"]


# ============================================================
# F4 — Content block strict-typing
# ============================================================


class TestF4ContentBlockStrictTyping:
    """Malformed content blocks must surface as 400 with the canonical
    envelope — never 200 with garbage output."""

    def test_text_block_without_text_rejected(self, client):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": [{"type": "text"}]}],
            },
        )
        assert response.status_code == 400, response.text
        envelope = response.json()
        assert envelope["type"] == "error"
        assert "is missing required field(s): text" in envelope["error"]["message"]

    def test_tool_use_block_without_required_fields_rejected(self, client):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use"}],
                    }
                ],
            },
        )
        assert response.status_code == 400, response.text
        msg = response.json()["error"]["message"]
        assert "tool_use" in msg
        # All three of id/name/input must be flagged missing.
        for field_name in ("id", "name", "input"):
            assert field_name in msg, msg

    def test_tool_result_block_without_required_fields_rejected(self, client):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result"}],
                    }
                ],
            },
        )
        assert response.status_code == 400, response.text
        msg = response.json()["error"]["message"]
        assert "tool_result" in msg
        for field_name in ("tool_use_id", "content"):
            assert field_name in msg, msg

    def test_image_block_without_source_rejected(self, client):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}],
                    }
                ],
            },
        )
        assert response.status_code == 400, response.text
        msg = response.json()["error"]["message"]
        assert "image" in msg
        assert "source" in msg

    def test_unknown_block_type_rejected(self, client):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "weirdblock", "data": 1}],
                    }
                ],
            },
        )
        assert response.status_code == 400, response.text
        msg = response.json()["error"]["message"]
        assert "weirdblock" in msg
        assert "not a recognized Anthropic content block type" in msg
        # Allowed types are listed so the client knows what to use.
        assert "text" in msg
        assert "tool_use" in msg

    def test_well_formed_block_still_accepted(self, client):
        """Sanity — well-formed block should not be rejected."""
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            },
        )
        # Engine stub returns "hello" — pass through to 200.
        assert response.status_code == 200, response.text


# ============================================================
# F10 — Role-block compatibility
# ============================================================


class TestF10RoleBlockCompatibility:
    """Cross-role block-type violations must surface as 400."""

    @pytest.mark.parametrize(
        ("role", "block"),
        [
            # user-role disallowed blocks (assistant-only)
            ("user", {"type": "thinking", "thinking": "hidden"}),
            ("user", {"type": "tool_use", "id": "x", "name": "f", "input": {}}),
            # assistant-role disallowed blocks (user-only)
            (
                "assistant",
                {"type": "tool_result", "tool_use_id": "x", "content": "hi"},
            ),
            (
                "assistant",
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "x",
                    },
                },
            ),
        ],
    )
    def test_cross_role_block_violations_rejected(self, client, role, block):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [{"role": role, "content": [block]}],
            },
        )
        assert response.status_code == 400, response.text
        envelope = response.json()
        assert envelope["type"] == "error"
        msg = envelope["error"]["message"]
        assert role in msg
        assert block["type"] in msg
        assert "not allowed" in msg

    def test_unknown_role_rejected(self, client):
        """Roles outside ``user``/``assistant``/``system`` must 400."""
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "wizard",
                        "content": [{"type": "text", "text": "hi"}],
                    }
                ],
            },
        )
        assert response.status_code == 400, response.text
        msg = response.json()["error"]["message"]
        assert "wizard" in msg
        assert "not recognized" in msg

    def test_unknown_role_rejected_with_string_content(self, client):
        """Codex round-1 BLOCKING fix: the unknown-role gate must fire
        regardless of whether ``content`` is a string or a block array.
        Pre-fix the validator returned early on string content and
        ``{"role":"wizard","content":"hi"}`` slipped through."""
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [{"role": "wizard", "content": "hi"}],
            },
        )
        assert response.status_code == 400, response.text
        msg = response.json()["error"]["message"]
        assert "wizard" in msg
        assert "not recognized" in msg

    def test_valid_user_text_block_accepted(self, client):
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi"}],
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text

    def test_valid_assistant_thinking_block_accepted(self, client):
        """Assistant CAN carry a thinking block (echoing a prior turn)
        — only user-role thinking is rejected."""
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "let me think"},
                            {"type": "text", "text": "ok"},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "go on"}],
                    },
                ],
            },
        )
        assert response.status_code == 200, response.text

    def test_string_content_unaffected(self, client):
        """``content`` as a bare string bypasses the role-block check
        (there are no blocks to validate)."""
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text


# ============================================================
# F11 — Empty messages array
# ============================================================


class TestF11EmptyMessages:
    """``messages=[]`` must 400 with a clear envelope — never 500."""

    def test_anthropic_empty_messages_returns_400(self, client):
        response = client.client.post(
            "/v1/messages",
            json={"model": "test-model", "max_tokens": 10, "messages": []},
        )
        assert response.status_code == 400, response.text
        envelope = response.json()
        assert envelope["type"] == "error"
        err = envelope["error"]
        assert err["type"] == "invalid_request_error"
        assert "messages" in err["message"]
        # ``min_length=1`` Pydantic message.
        assert "at least 1 item" in err["message"]

    def test_openai_chat_empty_messages_returns_400(self, chat_client):
        """OpenAI parity — same gap on /v1/chat/completions."""
        response = chat_client.client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": []},
        )
        assert response.status_code == 400, response.text
        body = response.json()
        # OpenAI envelope (no anthropic wrapper).
        assert body.get("type") != "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "messages" in body["error"]["message"]

    def test_responses_empty_input_string_returns_400(self, client):
        response = client.client.post(
            "/v1/responses",
            json={"model": "test-model", "input": ""},
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "input" in body["error"]["message"]

    def test_responses_empty_input_list_returns_400(self, client):
        response = client.client.post(
            "/v1/responses",
            json={"model": "test-model", "input": []},
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "input" in body["error"]["message"]


# ============================================================
# Cross-cutting: locked-in invariants
# ============================================================


class TestEnvelopeInvariants:
    """End-to-end invariants spanning all four bugs."""

    def test_every_anthropic_4xx_path_carries_wrapper(self, client):
        """Sweep every 4xx code we can trip on /v1/messages and
        assert the Anthropic wrapper is present on each."""
        # 400 — validation
        # 400 — empty messages
        # 400 — bad tool_choice
        # 400 — malformed JSON
        # 405 — wrong HTTP verb (GET on POST-only route)
        cases = [
            (
                "post",
                "/v1/messages",
                {
                    "json": {
                        "model": "test-model",
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "hi"}],
                        "temperature": "hot",
                    }
                },
            ),
            (
                "post",
                "/v1/messages",
                {
                    "json": {
                        "model": "test-model",
                        "max_tokens": 10,
                        "messages": [],
                    }
                },
            ),
            (
                "post",
                "/v1/messages",
                {
                    "content": b"{not json",
                    "headers": {"Content-Type": "application/json"},
                },
            ),
            ("get", "/v1/messages", {}),
        ]
        for method, path, kwargs in cases:
            response = getattr(client.client, method)(path, **kwargs)
            assert 400 <= response.status_code < 500, (
                f"{method.upper()} {path} {kwargs!r}: status={response.status_code}"
            )
            try:
                envelope = response.json()
            except json.JSONDecodeError:
                pytest.fail(f"{method.upper()} {path}: non-JSON body {response.text!r}")
            assert envelope.get("type") == "error", (
                f"{method.upper()} {path}: missing Anthropic wrapper, got {envelope!r}"
            )
            assert isinstance(envelope.get("error"), dict)

    def test_anthropic_wrapper_idempotent_on_already_wrapped(self, client):
        """If a route already emitted the Anthropic shape (via
        ``HTTPException(detail={"type":"error",...})``), the wrapper
        must not double-wrap. Use a tool_choice gate that does this
        explicitly."""
        # The model-name 404 path returns an Anthropic-shaped detail
        # (see routes/anthropic.py).
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "unknown-claude-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        # Whether 200 or 4xx depends on model alias resolution; if
        # 4xx, the envelope must be canonical (and no double-wrap).
        if response.status_code >= 400:
            envelope = response.json()
            assert envelope.get("type") == "error"
            # No nested wrapper.
            err = envelope["error"]
            assert err.get("type") != "error", "double-wrap detected"

    def test_path_match_does_not_classify_lookalike_paths_as_anthropic(self):
        """Codex round-1 NIT: ``startswith("/v1/messages")`` would also
        match ``/v1/messages-foo`` / ``/v1/messagesevil`` and wrap their
        404/405 responses with the Anthropic envelope. Use a fresh app
        that exposes only the lookalike paths so the classification
        function is exercised directly (no fixture interaction with
        the Anthropic router)."""
        from fastapi import HTTPException

        from vllm_mlx.middleware.exception_handlers import (
            _is_anthropic_path,
            install_exception_handlers,
        )

        app = FastAPI()
        install_exception_handlers(app)

        @app.post("/v1/messages-foo")
        async def _lookalike():
            raise HTTPException(status_code=400, detail="bad")

        @app.post("/v1/messages/sub")
        async def _real_sub():
            raise HTTPException(status_code=400, detail="bad")

        c = TestClient(app)

        # Lookalike path — must NOT get the Anthropic wrapper.
        resp = c.post("/v1/messages-foo")
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("type") != "error", body
        assert isinstance(body.get("error"), dict)

        # Real sub-path — MUST get the Anthropic wrapper.
        resp = c.post("/v1/messages/sub")
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("type") == "error", body

        # Direct classifier checks.
        class _Req:
            class _URL:
                pass

            def __init__(self, p):
                self.url = self._URL()
                self.url.path = p

        assert _is_anthropic_path(_Req("/v1/messages")) is True
        assert _is_anthropic_path(_Req("/v1/messages/count_tokens")) is True
        assert _is_anthropic_path(_Req("/v1/messages-foo")) is False
        assert _is_anthropic_path(_Req("/v1/messagesevil")) is False
        assert _is_anthropic_path(_Req("/v1/messages_evil")) is False

    def test_h17_attacker_keys_still_collapsed_under_allowlist(self, client):
        """The F1 allowlist must NOT re-open the H-17 round-2 leak.
        An attacker-controlled extra-field name must still collapse to
        ``<field>``."""
        sentinel = "AWS_SECRET_ACCESS_KEY_pwned"
        response = client.client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                sentinel: "bouncing-secret",
            },
        )
        # Either the body is silently ignored (current Pydantic
        # default with extra="allow") OR validates and 4xx fires for
        # another reason. Either way the sentinel must NOT appear.
        assert sentinel not in response.text, response.text
