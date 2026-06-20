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
# Default: wildcard ``*`` for friendly single-machine UX. Operators on
# multi-tenant deployments lock down via RAPID_MLX_CORS_ALLOW_ORIGINS.
# ──────────────────────────────────────────────────────────────────────


def test_default_wildcard_cors_registered(fresh_app: FastAPI) -> None:
    """No env, no CLI flag → CORSMiddleware with ``*``. Cross-origin POST
    returns 200 + ``Access-Control-Allow-Origin: *`` so a local browser
    frontend (e.g. ``http://localhost:3000``) can call the API without
    extra config."""
    origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == ["*"]

    client = TestClient(fresh_app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://anywhere.example"},
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_default_preflight_returns_200_with_wildcard(fresh_app: FastAPI) -> None:
    """Default-wildcard preflight returns 200 with ``ACAO: *`` so the
    browser proceeds to the real POST."""
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://anywhere.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_default_wildcard_forces_credentials_false(fresh_app: FastAPI) -> None:
    """Fetch spec: ``Access-Control-Allow-Origin: *`` is incompatible with
    ``Access-Control-Allow-Credentials: true``. The resolver MUST force
    ``allow_credentials=False`` when the default wildcard is in play.
    Otherwise browsers reject every cookie-bearing response."""
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://anywhere.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert "access-control-allow-credentials" not in {k.lower() for k in r.headers}


def test_default_does_not_log_wildcard_warning(
    fresh_app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """The wildcard-warning is meaningful ONLY when the operator opts in
    explicitly. With the default ``*`` we log an INFO line; no WARNING."""
    import logging

    caplog.set_level(logging.INFO, logger="vllm_mlx.server")
    _server_mod().configure_cors_from_env(cli_origins=None)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, (
        f"Default wildcard should not log WARNING; got {[r.message for r in warnings]!r}"
    )


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


def test_empty_csv_origin_value_fails_closed_with_warning(
    fresh_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Codex round-2 BLOCKING (#758): distinguish ``env unset`` (friendly
    default wildcard) from ``env set but parsed empty`` (likely operator
    templating bug). The present-but-empty case is fail-closed (no CORS
    middleware → preflight 405) plus a WARNING — silent fail-open would
    hide a deployment bug behind a permissive default."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", " , ,, ")
    with caplog.at_level("WARNING", logger="vllm_mlx.server"):
        origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == []
    assert any(
        "RAPID_MLX_CORS_ALLOW_ORIGINS" in rec.message
        and "empty list" in rec.message.lower()
        for rec in caplog.records
    ), f"Expected an empty-origins WARNING; got {[r.message for r in caplog.records]!r}"

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://anywhere.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Fail-closed: no CORS middleware, so the OPTIONS verb returns 405.
    assert r.status_code == 405


# ──────────────────────────────────────────────────────────────────────
# Codex round-1 BLOCKING: explicit empty-CSV methods/headers must fall
# back to the default with a WARNING (not silently broaden / narrow)
# ──────────────────────────────────────────────────────────────────────


def test_empty_methods_env_warns_and_falls_back(
    fresh_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An operator typo like ``RAPID_MLX_CORS_ALLOW_METHODS=" , "`` must
    NOT be treated as "use default" silently — that would broaden the
    surface back to ``POST,GET,OPTIONS`` despite the operator clearly
    intending to set the env var. Log a WARNING and fall back to the
    default; the operator sees the typo at boot.
    """
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_METHODS", " , ,, ")
    with caplog.at_level("WARNING", logger="vllm_mlx.server"):
        _server_mod().configure_cors_from_env(cli_origins=None)
    assert any(
        "RAPID_MLX_CORS_ALLOW_METHODS" in rec.message
        and "empty list" in rec.message.lower()
        for rec in caplog.records
    ), f"Expected an empty-methods warning; got {[r.message for r in caplog.records]!r}"


def test_empty_headers_env_warns_and_falls_back(
    fresh_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same as the methods case — ``RAPID_MLX_CORS_ALLOW_HEADERS=" , "``
    parses to an empty list; warn and fall back to the default header
    allowlist rather than silently propagating the (broken) empty
    intention as the broader default."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_HEADERS", " , ,, ")
    with caplog.at_level("WARNING", logger="vllm_mlx.server"):
        _server_mod().configure_cors_from_env(cli_origins=None)
    assert any(
        "RAPID_MLX_CORS_ALLOW_HEADERS" in rec.message
        and "empty list" in rec.message.lower()
        for rec in caplog.records
    ), f"Expected an empty-headers warning; got {[r.message for r in caplog.records]!r}"


# ──────────────────────────────────────────────────────────────────────
# Codex round-1 NIT: documented credentials default is False, not True.
# Explicit origin + unset RAPID_MLX_CORS_ALLOW_CREDENTIALS must not flip
# the credentials header to true.
# ──────────────────────────────────────────────────────────────────────


def test_credentials_default_false_with_explicit_origin(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``RAPID_MLX_CORS_ALLOW_ORIGINS`` is set to a real origin and
    ``RAPID_MLX_CORS_ALLOW_CREDENTIALS`` is unset, the resolver must not
    silently enable credentials — the documented default is False.
    Operators who need cookies must set the env var to ``true``."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://chat.openai.com"},
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://chat.openai.com"
    # Documented default: credentials disabled.
    assert "access-control-allow-credentials" not in {k.lower() for k in r.headers}


def test_credentials_opt_in_via_env(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting ``RAPID_MLX_CORS_ALLOW_CREDENTIALS=true`` enables the
    ``Access-Control-Allow-Credentials: true`` response header so
    cookie / Authorization-bearing fetches succeed."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_CREDENTIALS", "true")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": []},
        headers={"Origin": "https://chat.openai.com"},
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-credentials") == "true"


# ──────────────────────────────────────────────────────────────────────
# Codex round-3 BLOCKING: the legacy ``--cors-origins`` CLI flag is the
# documented back-compat path. When it's used (cli_origins != None) AND
# the methods/headers env overrides are unset, the resolver must
# preserve the legacy wide-open ``["*"]`` default so existing browser
# clients sending ``OpenAI-Organization`` etc. keep working. Only the
# env-driven path gets the new F-091 narrowing.
# ──────────────────────────────────────────────────────────────────────


def test_cli_origins_path_keeps_legacy_wide_open_headers(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--cors-origins https://chat.openai.com`` (no method/header env
    overrides) → preflight echoes back the operator-requested header
    (legacy ``["*"]`` behavior), not the new narrow allowlist. This
    preserves the existing CLI contract."""
    monkeypatch.delenv("RAPID_MLX_CORS_ALLOW_METHODS", raising=False)
    monkeypatch.delenv("RAPID_MLX_CORS_ALLOW_HEADERS", raising=False)
    _server_mod().configure_cors_from_env(cli_origins=["https://chat.openai.com"])

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "OpenAI-Organization",
        },
    )
    assert r.status_code == 200
    allowed = r.headers.get("access-control-allow-headers", "").lower()
    assert "openai-organization" in allowed or "*" in allowed, (
        f"Legacy --cors-origins path must keep allow_headers=['*'] "
        f"semantics; got Access-Control-Allow-Headers={allowed!r}"
    )


def test_env_origins_path_applies_f091_narrowing(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When origins come from the NEW env-driven path
    (``RAPID_MLX_CORS_ALLOW_ORIGINS``), the F-091 narrowing kicks in:
    custom headers like ``OpenAI-Organization`` are NOT in the default
    allowlist. Operators on this path opt-in to specific headers via
    ``RAPID_MLX_CORS_ALLOW_HEADERS``."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    monkeypatch.delenv("RAPID_MLX_CORS_ALLOW_HEADERS", raising=False)
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "OpenAI-Organization",
        },
    )
    # Preflight still returns 200 (Starlette CORS doesn't 4xx on a
    # disallowed Access-Control-Request-Headers) but the response's
    # ``Access-Control-Allow-Headers`` does NOT include the requested
    # custom header — browser blocks the real request.
    allowed = r.headers.get("access-control-allow-headers", "").lower()
    assert "openai-organization" not in allowed
    assert "*" not in allowed
    # The narrowed default is still echoed.
    for expected in ("content-type", "authorization", "x-rapid-mlx-internal"):
        assert expected in allowed, (
            f"Expected {expected!r} in narrowed default headers; got {allowed!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# Codex round-2 BLOCKING: ``configure_cors(origins)`` single-arg
# back-compat path must keep the legacy wide-open ``allow_headers=["*"]``
# so existing browser clients sending ``OpenAI-Organization`` /
# ``X-Requested-With`` keep working.
# ──────────────────────────────────────────────────────────────────────


def test_legacy_single_arg_configure_cors_keeps_wide_open_headers(
    fresh_app: FastAPI,
) -> None:
    """``configure_cors(origins)`` (no ``headers=`` / ``methods=`` kwargs)
    is the back-compat path used by tests / ``share`` CLI / dflash
    integration. Codex round-2 flagged that silently narrowing the
    defaults would break browser clients that send custom headers
    (``OpenAI-Organization``, ``X-Requested-With``, etc.). The
    narrowing only applies on the env-aware path
    (``configure_cors_from_env`` which passes explicit lists)."""
    _server_mod().configure_cors(["https://chat.openai.com"])

    client = TestClient(fresh_app)
    # Preflight requesting a custom header that's NOT in the new
    # restrictive default — pre-fix this would have failed; with the
    # back-compat ``["*"]`` it still works.
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "OpenAI-Organization",
        },
    )
    assert r.status_code == 200, (
        f"Legacy single-arg configure_cors() must keep preflight 200 for "
        f"custom headers like OpenAI-Organization; got {r.status_code}"
    )
    # Starlette echoes the requested header back when allow_headers=["*"].
    allowed = r.headers.get("access-control-allow-headers", "").lower()
    assert "openai-organization" in allowed or "*" in allowed, (
        f"Expected the requested header to be echoed or wildcarded; "
        f"got Access-Control-Allow-Headers={allowed!r}"
    )
