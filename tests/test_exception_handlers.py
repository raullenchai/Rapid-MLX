# SPDX-License-Identifier: Apache-2.0
"""Unified exception-handler regressions for F-160/F-161/F-162/F-165.

These cover the bundled wave-6 fix:

* F-161 / F-162 — malformed JSON bodies on ``/v1/messages``,
  ``/v1/messages/count_tokens``, and ``/v1/responses`` were returning
  HTTP 500 ``Internal server error`` because ``await request.json()``
  raises :class:`json.JSONDecodeError` and the only catch-all was the
  global ``Exception`` handler. A dedicated 400 handler is now wired
  in ``vllm_mlx.server`` and is also installed on per-route test apps
  via :func:`install_exception_handlers`.

* F-160 / F-167 — ``/v1/messages/count_tokens`` silently returned
  ``{"input_tokens": 0}`` for empty/missing ``messages`` and used the
  loaded engine's tokenizer for unknown models. Both now return
  structured 400 / 404 envelopes.

* F-165 — ``/v1/audio/transcriptions`` collapsed every failure mode
  (unknown alias, missing audio file, mlx-audio import error) into a
  generic 500 ``could not open/decode file``. The route now resolves
  the model alias BEFORE draining the upload and returns 404
  ``model_not_found_error`` for unknown names. The catch-all 500 no
  longer echoes the raw exception string.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
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
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None


class _Engine:
    preserve_native_tool_format = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="hello",
            prompt_tokens=3,
            completion_tokens=1,
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


def _build_app(monkeypatch, *, with_handlers: bool = True):
    """Construct a FastAPI app mounting the anthropic + responses routers
    and (optionally) the rapid-mlx exception handlers.

    Tests use ``with_handlers=False`` to assert the *current* default
    FastAPI behaviour (500 / 422 echo) is what gets repaired by the
    handler install, rather than papering over with a brittle "before"
    string.
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
    cfg.api_key = None  # turn off auth so the test focuses on validation
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


# ── F-161 + F-162: malformed JSON → 400 with clean envelope ─────────


@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses",
    ],
)
def test_malformed_json_returns_400_with_clean_envelope(client, path):
    """F-161 / F-162: malformed JSON body must produce a 400 with the
    OpenAI-shaped error envelope, not a 500 ``Internal server error``."""
    response = client.client.post(
        path,
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400, (
        f"{path}: expected 400, got {response.status_code} body={response.text!r}"
    )
    body = response.json()
    assert "error" in body
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_json"
    assert "Invalid JSON" in err["message"]
    # Defence in depth — make sure we don't leak the python module name
    # or filesystem path in the envelope.
    assert "json.decoder" not in err["message"]
    assert "Traceback" not in err["message"]


def test_malformed_json_does_not_leak_python_internals(client):
    """The 400 envelope must NOT carry a pydantic.dev help URL (F-163
    cousin) or python-module paths from ``json.decoder``."""
    response = client.client.post(
        "/v1/messages",
        content=b"<<not json>>",
        headers={"Content-Type": "application/json"},
    )
    payload_text = response.text
    assert "errors.pydantic.dev" not in payload_text
    assert "/site-packages/" not in payload_text


# ── F-161 fallback wiring: the install helper covers all routes ─────


def test_install_exception_handlers_wires_json_decode_handler(monkeypatch):
    """Without ``install_exception_handlers`` the same malformed body
    would surface as 500. The helper is the seam that downstream test
    apps and the production server both rely on."""
    app, _, teardown = _build_app(monkeypatch, with_handlers=False)
    try:
        # With NO handlers wired, FastAPI's default behaviour is to let
        # the JSONDecodeError bubble up as an unhandled 500. We assert
        # the broken state explicitly so the test fails loudly if some
        # future refactor accidentally re-introduces a different
        # default behaviour.
        client = TestClient(app, raise_server_exceptions=False)
        bad = client.post(
            "/v1/messages",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert bad.status_code == 500
    finally:
        teardown()

    app, _, teardown = _build_app(monkeypatch, with_handlers=True)
    try:
        client = TestClient(app)
        fixed = client.post(
            "/v1/messages",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert fixed.status_code == 400
        assert fixed.json()["error"]["code"] == "invalid_json"
    finally:
        teardown()


# ── F-160: count_tokens validates empty / missing messages ──────────


@pytest.mark.parametrize(
    "body",
    [
        {},  # empty body
        {"messages": []},  # explicit empty array
        {"model": "test-model"},  # only model, no messages
    ],
)
def test_count_tokens_rejects_empty_messages(client, body):
    """F-160: an empty/missing ``messages`` is a cost-estimation
    footgun — return 400 instead of a silent ``{"input_tokens": 0}``."""
    response = client.client.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 400, response.text
    err = response.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "messages"
    assert "non-empty array" in err["message"]


def test_count_tokens_rejects_non_array_messages(client):
    """Defensive: ``messages`` must be an array (even if non-empty)."""
    response = client.client.post(
        "/v1/messages/count_tokens",
        json={"model": "test-model", "messages": "hi"},
    )
    assert response.status_code == 400, response.text
    err = response.json()["error"]
    assert err["param"] == "messages"


def test_count_tokens_normal_request_still_returns_tokens(client):
    """Sanity: a valid request still returns ``input_tokens > 0``."""
    response = client.client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["input_tokens"] > 0


def test_count_tokens_accepts_claude_alias(client):
    """Claude-Code style ``claude-*`` model names pass through to the
    loaded engine — matches the ``/v1/messages`` contract."""
    response = client.client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "claude-opus-4-5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text


# ── F-167: count_tokens validates unknown model ─────────────────────


def test_count_tokens_rejects_unknown_model(client):
    """F-167: unknown model names must 404, not silently fall back to
    the loaded engine's tokenizer."""
    response = client.client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "definitely-not-loaded",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404, response.text
    body = response.json()
    assert "error" in body
    err = body["error"]
    assert err["type"] == "not_found_error"
    assert "definitely-not-loaded" in err["message"]


# ── F-165: audio transcriptions model validation ────────────────────


def test_audio_resolve_stt_model_known_alias():
    from vllm_mlx.routes.audio import _resolve_stt_model

    assert _resolve_stt_model("whisper-small") == "mlx-community/whisper-small-mlx"


def test_audio_resolve_stt_model_passthrough_repo_path():
    from vllm_mlx.routes.audio import _resolve_stt_model

    # Anything containing a "/" is treated as a HuggingFace repo id —
    # the STT engine attempts to load it directly.
    assert _resolve_stt_model("org/private-stt") == "org/private-stt"


def test_audio_resolve_stt_model_rejects_bogus_name():
    """F-165: bogus aliases (no slash, not in the curated map) must
    raise a structured 404 BEFORE any STT engine load is attempted."""
    from fastapi import HTTPException

    from vllm_mlx.routes.audio import _resolve_stt_model

    with pytest.raises(HTTPException) as exc_info:
        _resolve_stt_model("definitely-not-a-whisper")
    assert exc_info.value.status_code == 404
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["error"]["type"] == "model_not_found_error"
    assert detail["error"]["code"] == "model_not_found"
    assert "definitely-not-a-whisper" in detail["error"]["message"]


def test_audio_resolve_stt_model_rejects_empty_string():
    from fastapi import HTTPException

    from vllm_mlx.routes.audio import _resolve_stt_model

    with pytest.raises(HTTPException) as exc_info:
        _resolve_stt_model("")
    assert exc_info.value.status_code == 400


# ── Direct server-side handler shape ────────────────────────────────


def test_decode_error_response_shape():
    """The shared ``_decode_error_response`` builder yields the OpenAI
    envelope. Exercised via the public app + via the helper, but the
    direct call protects the shape from drift."""
    from vllm_mlx.middleware.exception_handlers import _decode_error_response

    exc = None
    try:
        json.loads("not json")
    except json.JSONDecodeError as e:
        exc = e
    assert exc is not None
    response = _decode_error_response(exc)
    payload = json.loads(response.body)
    assert response.status_code == 400
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "invalid_json"
    assert "Expecting" in payload["error"]["message"]
