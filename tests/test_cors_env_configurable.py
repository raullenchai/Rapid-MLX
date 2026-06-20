# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-090 + F-091.

F-090 (HIGH): the server registered ``CORSMiddleware(allow_origins=["*"])``
by default, which let any browser-side attacker make authenticated
cross-origin requests against ``/v1/chat/completions``.

F-091 (MED): the preflight ``OPTIONS`` returned
``Access-Control-Allow-Methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST,
PUT`` — over-broad for a server that only routes POST/GET/OPTIONS.

The fix moves CORS to an env-var-driven opt-in. These tests pin both the
default-deny stance and the new env-var family
(``RAPID_MLX_CORS_ALLOW_ORIGINS`` / ``_METHODS`` / ``_HEADERS`` /
``_MAX_AGE`` / ``_ALLOW_CREDENTIALS``).

The tests mount the CORS resolver against a fresh ``FastAPI()`` so they
don't touch the production module-level ``app`` singleton (and don't
require the engine stack to be loaded).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def fresh_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """Yield a fresh ``FastAPI`` app with ``vllm_mlx.server.app`` monkey-
    patched to point at it, so ``configure_cors`` /
    ``configure_cors_from_env`` register middleware on the test app
    rather than the production singleton.

    Each test also gets a clean env (no leaked ``RAPID_MLX_CORS_*`` from
    other tests).
    """
    import vllm_mlx.server as server_mod

    # Reload to drop any state from previous tests in the same worker.
    importlib.reload(server_mod)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat() -> dict[str, str]:
        return {"ok": "true"}

    @app.get("/healthz")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    monkeypatch.setattr(server_mod, "app", app)

    for var in (
        "RAPID_MLX_CORS_ALLOW_ORIGINS",
        "RAPID_MLX_CORS_ALLOW_METHODS",
        "RAPID_MLX_CORS_ALLOW_HEADERS",
        "RAPID_MLX_CORS_MAX_AGE",
        "RAPID_MLX_CORS_ALLOW_CREDENTIALS",
    ):
        monkeypatch.delenv(var, raising=False)

    yield app


def _server_mod():
    import vllm_mlx.server as server_mod

    return server_mod


# ──────────────────────────────────────────────────────────────────────
# F-090: default-deny when neither CLI flag nor env var is set
# ──────────────────────────────────────────────────────────────────────


def test_default_no_cors_middleware_registered(fresh_app: FastAPI) -> None:
    """No env, no CLI flag → no CORSMiddleware. Cross-origin POST must
    return 200 (auth not enforced in this fixture) but WITHOUT any
    ``Access-Control-Allow-Origin`` header — i.e. browsers will block
    the response when they enforce same-origin."""
    origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == []

    client = TestClient(fresh_app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://evil.com"},
    )
    assert r.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_default_preflight_returns_405(fresh_app: FastAPI) -> None:
    """Without CORS middleware, ``OPTIONS /v1/chat/completions`` falls
    through to Starlette's default router and returns 405 (route is
    POST-only). Critically, no ``Access-Control-*`` headers leak."""
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 405
    leaked = [k for k in r.headers if k.lower().startswith("access-control-")]
    assert leaked == [], f"CORS headers leaked on default-deny preflight: {leaked}"


# ──────────────────────────────────────────────────────────────────────
# F-090: explicit allowlist via env var
# ──────────────────────────────────────────────────────────────────────


def test_env_explicit_origin_matching(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RAPID_MLX_CORS_ALLOW_ORIGINS=https://chat.openai.com`` → matching
    origin gets that origin echoed back; non-matching origin gets no
    ACAO header."""
    monkeypatch.setenv(
        "RAPID_MLX_CORS_ALLOW_ORIGINS",
        "https://chat.openai.com,https://claude.ai",
    )
    origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == ["https://chat.openai.com", "https://claude.ai"]

    client = TestClient(fresh_app)
    ok = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://chat.openai.com"},
    )
    assert ok.status_code == 200
    assert ok.headers.get("access-control-allow-origin") == "https://chat.openai.com"

    bad = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://evil.com"},
    )
    # Starlette's CORSMiddleware lets the request through but omits the
    # ACAO header on a non-matching origin (so the browser blocks the
    # response).
    assert bad.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in bad.headers}


# ──────────────────────────────────────────────────────────────────────
# F-091: default methods are POST/GET/OPTIONS (not DELETE/PATCH/PUT)
# ──────────────────────────────────────────────────────────────────────


def test_default_methods_do_not_include_destructive_verbs(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When CORS is enabled, the default preflight ACAM must only list
    methods the server actually serves: POST + GET + OPTIONS. Pre-fix
    the response listed DELETE/PATCH/PUT too — over-broad surface that
    invited a future routing mistake."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    methods = r.headers.get("access-control-allow-methods", "")
    method_set = {m.strip().upper() for m in methods.split(",") if m.strip()}
    assert method_set == {"POST", "GET", "OPTIONS"}, (
        f"Expected POST/GET/OPTIONS only; got {method_set!r}"
    )
    for forbidden in ("DELETE", "PATCH", "PUT", "HEAD"):
        assert forbidden not in method_set, (
            f"{forbidden} leaked into the default Access-Control-Allow-Methods"
        )


def test_env_methods_override(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RAPID_MLX_CORS_ALLOW_METHODS=POST,OPTIONS`` narrows the default
    allowlist further. The operator can lock down to POST + OPTIONS for
    a webhook-style deployment."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_METHODS", "POST,OPTIONS")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    methods = r.headers.get("access-control-allow-methods", "")
    method_set = {m.strip().upper() for m in methods.split(",") if m.strip()}
    assert method_set == {"POST", "OPTIONS"}


# ──────────────────────────────────────────────────────────────────────
# Wildcard back-compat: works but logs a WARNING
# ──────────────────────────────────────────────────────────────────────


def test_wildcard_logs_warning_and_works(
    fresh_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``RAPID_MLX_CORS_ALLOW_ORIGINS=*`` matches the old default behavior
    (any origin echoed back) BUT emits a WARNING at startup so an
    operator who set it intentionally gets a sanity check, and an
    operator who copy-pasted from a stale doc notices."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "*")
    with caplog.at_level("WARNING", logger="vllm_mlx.server"):
        origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == ["*"]
    assert any("wildcard" in rec.message.lower() for rec in caplog.records), (
        f"Expected a wildcard-CORS warning; got {[r.message for r in caplog.records]!r}"
    )

    client = TestClient(fresh_app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://evil.com"},
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"
    # Fetch spec: wildcard + credentials must NOT combine; the credentials
    # header must be absent.
    assert "access-control-allow-credentials" not in {k.lower() for k in r.headers}


# ──────────────────────────────────────────────────────────────────────
# CLI flag overrides env var (priority sanity check)
# ──────────────────────────────────────────────────────────────────────


def test_cli_origins_override_env(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``--cors-origins`` is passed, the env var is ignored. This
    matches the precedent set by ``--max-request-bytes`` vs
    ``RAPID_MLX_MAX_REQUEST_BYTES``."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://from-env.example")
    origins = _server_mod().configure_cors_from_env(
        cli_origins=["https://from-cli.example"]
    )
    assert origins == ["https://from-cli.example"]

    client = TestClient(fresh_app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://from-cli.example"},
    )
    assert r.headers.get("access-control-allow-origin") == "https://from-cli.example"


# ──────────────────────────────────────────────────────────────────────
# Env-var hardening: malformed values fall back to defaults
# ──────────────────────────────────────────────────────────────────────


def test_malformed_max_age_falls_back_to_default(
    fresh_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bad ``RAPID_MLX_CORS_MAX_AGE`` logs a warning and uses 3600 s. We
    don't crash startup on a typo — same shape as the ``--max-request-bytes``
    fallback added in PR #732."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    monkeypatch.setenv("RAPID_MLX_CORS_MAX_AGE", "not-a-number")
    with caplog.at_level("WARNING", logger="vllm_mlx.server"):
        _server_mod().configure_cors_from_env(cli_origins=None)
    assert any("RAPID_MLX_CORS_MAX_AGE" in rec.message for rec in caplog.records), (
        f"Expected a malformed-max-age warning; got {[r.message for r in caplog.records]!r}"
    )

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Default max-age is 3600.
    assert r.headers.get("access-control-max-age") == "3600"


def test_empty_csv_value_treated_as_unset(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RAPID_MLX_CORS_ALLOW_ORIGINS=" , ,, "`` (whitespace + empty
    fragments) parses to an empty list and falls back to default-deny.
    Defends against the easy-to-miss config bug where a deploy script
    expands an empty array variable."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", " , ,, ")
    origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == []

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # No CORS middleware → preflight returns 405 with no ACAO leak.
    assert r.status_code == 405
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
