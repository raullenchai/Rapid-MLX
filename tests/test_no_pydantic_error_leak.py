# SPDX-License-Identifier: Apache-2.0
"""H-17 — Pydantic ``ValidationError`` text must not leak from
``/v1/messages`` or ``/v1/responses``.

Pre-fix repro (Rhea r0.8.1 audit):

    POST /v1/responses
    {"model":"x","messages":[],"FOO_BAR_INJECTED":"<attacker_pwned>"}

    HTTP/1.1 400
    {"error":{"message":"1 validation error for ResponsesRequest\\ninput\\n"
              "  Field required [type=missing, input_value={..., 'FOO_BAR_"
              "INJECTED': '<attacker_pwned>'}, input_type=dict]\\n    For "
              "further information visit https://errors.pydantic.dev/2.13/"
              "v/missing","type":"invalid_request_error","code":null,
              "param":null}}

The body leaks **three things at once**:

1. The pinned Pydantic version (``errors.pydantic.dev/2.13/...``) — a
   dependency-fingerprint vector that helps an attacker target
   known-CVE Pydantic releases against this binary.
2. The internal request-model class name (``ResponsesRequest``,
   ``AnthropicRequest``, ``ChatCompletionRequest``) — code structure
   recon.
3. The attacker-controlled ``input_value`` echoed verbatim inside
   ``[input_value=..., input_type=dict]`` — a stored-XSS-style
   reflection vector (chat clients render error messages) and a
   side-channel for any secret the attacker probes by including it
   in a body field that fails validation.

The systemic fix routes raw :class:`pydantic.ValidationError` through
the same sanitized envelope ``/v1/chat/completions`` uses, via a new
``@app.exception_handler(PydanticValidationError)`` in
``middleware/exception_handlers.py``. The per-route
``try: M(**body) except ValidationError: raise HTTPException(detail=str(e))``
patches in ``routes/anthropic.py`` and ``routes/responses.py`` were
deleted so the exception bubbles to the global handler.

This module exercises the public surface end-to-end (TestClient
against both routers + the installed exception handlers) and asserts
both directions:

* The 400 envelope MUST NOT contain ``pydantic``, ``ValidationError``,
  the pinned version, the request-model class name, the
  ``input_value=...`` echo, ``errors.pydantic.dev``, or any
  attacker-supplied sentinel.
* The 400 envelope MUST contain the canonical OpenAI error shape with
  ``error.type == "invalid_request_error"`` and
  ``error.code == "invalid_request"`` so legitimate clients can still
  programmatically distinguish bad-input from rate-limit or auth
  errors.
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

# ── Lightweight engine stubs (same shape as test_exception_handlers) ─


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


def _build_app(monkeypatch, *, with_handlers: bool = True):
    """Lightweight FastAPI app mounting /v1/messages + /v1/responses.

    Mirrors :func:`tests.test_exception_handlers._build_app` — kept
    local rather than imported so this regression is fully isolated
    from the F-160/F-161/F-162 fixtures (the H-17 fix MUST stand on
    its own even if the older fixture file is renamed).
    """
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
    if with_handlers:
        from vllm_mlx.middleware.exception_handlers import (
            install_exception_handlers,
        )

        install_exception_handlers(app)
    app.include_router(anthropic_router)
    app.include_router(responses_router)

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
    app, cfg, teardown = _build_app(monkeypatch, with_handlers=True)
    try:
        yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    finally:
        teardown()


# ── Bad-payload matrix ──────────────────────────────────────────────
#
# Cases chosen to hit each Pydantic v2 error class that would normally
# emit the leaky default ``str(e)`` shape:
#
# * ``malformed-body``  — body parses to a non-dict (Pydantic's
#   ``model_type`` error). The route's ``M(**body)`` itself raises
#   ``TypeError`` for non-dict; we test the *list* case which goes
#   through ``M(*body)``-style unpacking only when it's a dict, so we
#   actually pass a dict that fails ``model_validator``.
# * ``type-violation``  — ``messages: "not-a-list"`` triggers
#   Pydantic's ``list_type`` error. Pre-fix leaked
#   ``input_value='not-a-list'``.
# * ``missing-required`` — drop the required ``model`` / ``input`` /
#   ``max_tokens`` field. Pre-fix leaked ``input_value={'messages':
#   ...}`` which echoes the whole body.
# * ``extra-strict-injection`` — include an attacker-controlled extra
#   field. Pre-fix the body echo in ``input_value=...`` carried the
#   sentinel verbatim — clearest demonstration of the secret-bounce
#   vector.

_ATTACKER_SENTINEL = "<H17_ATTACKER_SENTINEL_pwned>"

# Each entry: (case_id, /v1/messages body, /v1/responses body).
# We keep separate per-endpoint bodies because the two request models
# disagree on which fields are required (Anthropic needs ``max_tokens``;
# Responses needs ``input`` instead of ``messages``). Reusing one body
# would silently make one endpoint pass for the wrong reason.
_BAD_BODIES = [
    (
        "type-violation",
        {"model": "test-model", "messages": "not-a-list", "max_tokens": 10},
        {"model": "test-model", "input": 12345, "max_output_tokens": 10},
    ),
    (
        "missing-required-field",
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "hi"}]},
    ),
    (
        "extra-strict-injection",
        {
            "messages": [{"role": "user", "content": "x"}],
            # leave out required ``model`` + ``max_tokens`` so the
            # attacker sentinel ends up in Pydantic's body echo on the
            # legacy path.
            "FOO_BAR_INJECTED": _ATTACKER_SENTINEL,
        },
        {
            "messages": [{"role": "user", "content": "x"}],
            "FOO_BAR_INJECTED": _ATTACKER_SENTINEL,
        },
    ),
    (
        "nested-type-violation",
        {
            "model": "test-model",
            "max_tokens": 10,
            # Each message must be a dict; passing an int per element
            # trips the inner validator and the original ``str(e)``
            # would have echoed the int back inside ``input_value=12345``.
            "messages": [12345],
        },
        {
            "model": "test-model",
            # ``input`` is required for ResponsesRequest. Passing an
            # int for the field hits the union-discriminator validator
            # and the legacy path echoed the int in ``input_value=...``.
            "input": 12345,
            "max_output_tokens": 10,
        },
    ),
]


_LEAK_NEEDLES = (
    # 1. Dependency-fingerprint vector — Pydantic help URLs include the
    #    pinned major.minor version.
    "errors.pydantic.dev",
    # 2. Pinned Pydantic version (rapid-mlx v0.8.1 ships 2.13.x). Wider
    #    than ``errors.pydantic.dev/2.13`` so a future upgrade to 2.14
    #    doesn't silently make this test pass on a still-leaking 2.13
    #    binary.
    "pydantic",
    # 3. The standard Pydantic v2 ``str(ValidationError)`` preamble.
    "validation error for",
    # 4. The ``[type=..., input_value=..., input_type=...]`` annotation
    #    Pydantic appends per error — carries the attacker echo.
    "input_value",
    # 5. Internal request-model class names — code-structure recon.
    "AnthropicRequest",
    "ResponsesRequest",
    "ChatCompletionRequest",
)


@pytest.mark.parametrize("case_id, msgs_body, resp_body", _BAD_BODIES)
def test_messages_no_pydantic_leak(client, case_id, msgs_body, resp_body):
    """H-17 — /v1/messages must sanitize Pydantic ValidationError."""
    response = client.client.post("/v1/messages", json=msgs_body)
    body_text = response.text.lower()
    err = response.json().get("error")
    assert response.status_code == 400, (
        f"[{case_id}] /v1/messages expected 400; got {response.status_code} "
        f"body={response.text!r}"
    )
    assert err is not None, f"[{case_id}] missing top-level 'error' object"
    # Canonical envelope shape — clients programmatically rely on these.
    assert err["type"] == "invalid_request_error", (
        f"[{case_id}] error.type={err['type']!r}"
    )
    assert err["code"] == "invalid_request", f"[{case_id}] error.code={err['code']!r}"
    assert "message" in err, f"[{case_id}] envelope missing message"
    assert err["message"].startswith("Invalid request body:"), (
        f"[{case_id}] message must start with canonical prefix; got {err['message']!r}"
    )

    # Defence-in-depth: no leak strings anywhere in the body (case-
    # insensitive).
    for needle in _LEAK_NEEDLES:
        assert needle.lower() not in body_text, (
            f"[{case_id}] /v1/messages leaked {needle!r} in 400 envelope; "
            f"body={response.text!r}"
        )
    # Attacker-supplied sentinel must not bounce.
    assert _ATTACKER_SENTINEL not in response.text, (
        f"[{case_id}] /v1/messages echoed attacker sentinel; body={response.text!r}"
    )


@pytest.mark.parametrize("case_id, msgs_body, resp_body", _BAD_BODIES)
def test_responses_no_pydantic_leak(client, case_id, msgs_body, resp_body):
    """H-17 — /v1/responses must sanitize Pydantic ValidationError."""
    response = client.client.post("/v1/responses", json=resp_body)
    body_text = response.text.lower()
    err = response.json().get("error")
    assert response.status_code == 400, (
        f"[{case_id}] /v1/responses expected 400; got {response.status_code} "
        f"body={response.text!r}"
    )
    assert err is not None, f"[{case_id}] missing top-level 'error' object"
    assert err["type"] == "invalid_request_error", (
        f"[{case_id}] error.type={err['type']!r}"
    )
    assert err["code"] == "invalid_request", f"[{case_id}] error.code={err['code']!r}"
    assert "message" in err, f"[{case_id}] envelope missing message"
    assert err["message"].startswith("Invalid request body:"), (
        f"[{case_id}] message must start with canonical prefix; got {err['message']!r}"
    )

    for needle in _LEAK_NEEDLES:
        assert needle.lower() not in body_text, (
            f"[{case_id}] /v1/responses leaked {needle!r} in 400 envelope; "
            f"body={response.text!r}"
        )
    assert _ATTACKER_SENTINEL not in response.text, (
        f"[{case_id}] /v1/responses echoed attacker sentinel; body={response.text!r}"
    )


# ── H-17 round-2 (codex): attacker sentinel inside a KEY name ────────


def test_attacker_key_in_loc_is_collapsed(monkeypatch):
    """Codex H-17 round-2 finding: the original sanitizer joined the
    full ``loc`` tuple into the 400 message verbatim. Pydantic v2
    puts attacker-controlled dict keys / JSON-pointer escapes / extra-
    forbid field names directly into ``loc``, so a body like
    ``{"tags": {"<attacker_pwned>": "x"}}`` against a schema that
    types ``tags`` as ``dict[str, int]`` would reflect the sentinel
    in the 400 envelope even though the value was harmless.

    Wires a throwaway endpoint with a ``dict[str, int]`` field, posts
    an attacker-controlled key, and asserts the sentinel does NOT
    appear anywhere in the envelope while the canonical shape still
    surfaces. Uses :func:`_build_app` so the global handler install is
    exercised end-to-end.
    """
    from fastapi import Request as _FastAPIRequest
    from pydantic import BaseModel

    app, _cfg, teardown = _build_app(monkeypatch, with_handlers=True)
    try:

        class _KeyEcho(BaseModel):
            # ``dict[str, int]`` triggers per-key validation, so a bad
            # value bubbles the key into ``loc``.
            tags: dict[str, int]

        @app.post("/__h17_key_probe__")
        async def _probe(req: _FastAPIRequest):
            body = await req.json()
            return _KeyEcho(**body)

        client = TestClient(app)
        evil_key = "<H17_KEY_SENTINEL_pwned>"
        response = client.post(
            "/__h17_key_probe__",
            json={"tags": {evil_key: "not-an-int"}},
        )
        assert response.status_code == 400, response.text
        assert evil_key not in response.text, (
            f"loc echoed attacker-controlled key: {response.text!r}"
        )
        # Envelope shape still canonical.
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "invalid_request"
        assert err["message"].startswith("Invalid request body:")
        # Sanitized placeholder shows up in place of the dangerous key.
        assert "<key>" in err["message"], (
            f"expected sanitized <key> placeholder; got {err['message']!r}"
        )
    finally:
        teardown()


def test_attacker_extra_field_name_is_collapsed(monkeypatch):
    """Sibling of the dict-key test, but for ``ConfigDict(extra=
    "forbid")`` schemas: Pydantic puts the rejected extra-field NAME
    into ``loc[0]``. An attacker who can pick the field name
    (``POST {"<sentinel>": 1, ...}``) would otherwise see it echoed in
    the 400 envelope on any forbid-extras model wired to this handler.
    """
    from fastapi import Request as _FastAPIRequest
    from pydantic import BaseModel, ConfigDict

    app, _cfg, teardown = _build_app(monkeypatch, with_handlers=True)
    try:

        class _Strict(BaseModel):
            model_config = ConfigDict(extra="forbid")
            x: int

        @app.post("/__h17_extra_probe__")
        async def _probe(req: _FastAPIRequest):
            body = await req.json()
            return _Strict(**body)

        client = TestClient(app)
        evil_field = "<H17_EXTRA_FIELD_pwned>"
        response = client.post(
            "/__h17_extra_probe__", json={"x": 1, evil_field: "anything"}
        )
        assert response.status_code == 400, response.text
        assert evil_field not in response.text, (
            f"extra-field name echoed: {response.text!r}"
        )
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "invalid_request"
    finally:
        teardown()


# ── Defence-in-depth: the global handler covers the raw Pydantic ─────
# ``ValidationError`` even when no route is involved (a future endpoint
# that builds its model manually inherits the fix automatically).


def test_global_handler_routes_raw_pydantic_validation_error(monkeypatch):
    """A throwaway endpoint that raises raw ``pydantic.ValidationError``
    must get the same sanitized 400 envelope as the route fix targets.

    This makes the regression structural: the fix isn't ``/v1/messages``
    + ``/v1/responses`` -specific, it's the global handler.
    """
    from fastapi import Request as _FastAPIRequest
    from pydantic import BaseModel

    app, _cfg, teardown = _build_app(monkeypatch, with_handlers=True)
    try:

        class _Inner(BaseModel):
            field_x: int

        @app.post("/__h17_probe__")
        async def _probe(req: _FastAPIRequest):
            body = await req.json()
            # Manual construction — the same anti-pattern messages /
            # responses use. Raises pydantic.ValidationError directly.
            return _Inner(**body)

        client = TestClient(app)
        response = client.post("/__h17_probe__", json={"field_x": "not-an-int"})
        assert response.status_code == 400, response.text
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "invalid_request"
        assert err["message"].startswith("Invalid request body:")
        # Same sanitization on the probe path.
        body_text = response.text.lower()
        for needle in ("pydantic", "validation error for", "input_value"):
            assert needle.lower() not in body_text, (
                f"global handler leaked {needle!r}; body={response.text!r}"
            )
    finally:
        teardown()


# ── Pre-fix-state guard: assert the legacy ``str(e)`` shape NEVER ─────
# returns by hitting a known-leak input with the production handler
# installed. The negation is what the H-17 fix bought us; lock it in.


def test_pre_fix_leak_no_longer_reproduces(client):
    """Negative control — the exact byte sequence Rhea's repro lifted
    from r0.8.1 must not appear in any 400 envelope shape."""
    rhea_repro_body = {
        "messages": [{"role": "user", "content": "x"}],
        "FOO_BAR_INJECTED": _ATTACKER_SENTINEL,
    }
    for path in ("/v1/messages", "/v1/responses"):
        response = client.client.post(path, json=rhea_repro_body)
        # The pre-fix body literally contained:
        #   "1 validation error for ResponsesRequest"
        #   "[type=missing, input_value={'messages': ..."
        #   "https://errors.pydantic.dev/2.13/v/missing"
        # Lock each substring out.
        assert "1 validation error for" not in response.text, response.text
        assert "errors.pydantic.dev" not in response.text, response.text
        assert "[type=missing" not in response.text, response.text
        assert _ATTACKER_SENTINEL not in response.text, response.text
        # And the canonical envelope is still there.
        assert response.json()["error"]["type"] == "invalid_request_error"
        # JSON-encoding sanity: response must be valid JSON.
        json.loads(response.text)
