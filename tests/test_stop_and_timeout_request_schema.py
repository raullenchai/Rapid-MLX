# SPDX-License-Identifier: Apache-2.0
"""HTTP-level contract tests for the #2359 request-schema fix.

Covers one behaviour on both ``/v1/chat/completions`` and ``/v1/completions``:

  * omitted, null, and numeric-zero ``timeout`` follow the server-default
    path; negative, boolean, and non-finite values are rejected with the
    unified ``invalid_request_error`` 400 naming the field.

The schema-level normalization / rejection logic itself is unit-tested in
``tests/test_api_models.py`` (scalar-``stop`` acceptance + ``timeout``
validation); this file pins the wire behaviour through the real routes.
The 400-not-504 claim is the HTTP layer's responsibility and only provable
here.

The engine is stubbed only to satisfy route setup; the bad-``timeout``
cases are rejected at request parsing (schema layer), so the route handler
never runs. These tests are deterministic: a bad ``timeout`` ALWAYS yields
400 before any engine interaction.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def patched_config():
    """Patch the global config singleton and restore on teardown."""
    from rapid_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


def _stub_engine_cfg(patch_cfg):
    engine = MagicMock()
    engine.is_mllm = False
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )
    return engine


@pytest.fixture
def chat_client(patched_config, monkeypatch):
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes import chat as chat_route

    engine = _stub_engine_cfg(patched_config)
    monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

    app = FastAPI()
    app.include_router(chat_route.router)
    install_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def completion_client(patched_config, monkeypatch):
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes import completions as comp_route

    engine = _stub_engine_cfg(patched_config)
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)

    app = FastAPI()
    app.include_router(comp_route.router)
    install_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _chat_body(**kw) -> dict:
    body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return body


def _completion_body(**kw) -> dict:
    body = {"model": "stub-model", "prompt": "Once upon a time"}
    body.update(kw)
    return body


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [
        ("chat_client", "/v1/chat/completions", _chat_body),
        ("completion_client", "/v1/completions", _completion_body),
    ],
)
@pytest.mark.parametrize("bad", [-1, -0.5, -1.0, False, True])
def test_negative_or_boolean_timeout_400(request, client_name, url, body_fn, bad):
    """A malformed timeout must 400 with the unified envelope naming the field."""
    client = request.getfixturevalue(client_name)
    r = client.post(url, json=body_fn(timeout=bad))
    assert r.status_code == 400, (
        f"expected 400 for timeout={bad}; got {r.status_code} body={r.text[:200]}"
    )
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "timeout" in body["error"]["message"]


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [
        ("chat_client", "/v1/chat/completions", _chat_body),
        ("completion_client", "/v1/completions", _completion_body),
    ],
)
@pytest.mark.parametrize("value", [0, 0.0])
def test_zero_timeout_uses_server_default(request, client_name, url, body_fn, value):
    """Zero is a compatibility sentinel, never a malformed request."""
    client = request.getfixturevalue(client_name)
    response = client.post(url, json=body_fn(timeout=value))

    assert response.status_code != 400
    assert "timeout" not in response.text.lower()


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [
        ("chat_client", "/v1/chat/completions", _chat_body),
        ("completion_client", "/v1/completions", _completion_body),
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timeout_400(request, client_name, url, body_fn, bad):
    """NaN / ±inf travel as raw JSON tokens (allow_nan), so we post the
    raw payload rather than httpx's json= channel. Must 400 naming the
    field, not 500 (non-finite in a serialized error body is a 500 trap)
    and not 504."""
    client = request.getfixturevalue(client_name)
    payload = json.dumps(body_fn(timeout=bad))  # allow_nan=True
    r = client.post(url, content=payload, headers={"Content-Type": "application/json"})
    assert r.status_code == 400, (
        f"expected 400 for timeout={bad!r}; got {r.status_code} body={r.text[:200]}"
    )
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "timeout" in body["error"]["message"]
