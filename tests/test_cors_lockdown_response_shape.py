# SPDX-License-Identifier: Apache-2.0
"""Regression tests for L-02 — env-locked CORS rejection envelope shape.

Pre-fix: when ``RAPID_MLX_CORS_ALLOW_ORIGINS`` was set to an explicit
allowlist and the preflight ``OPTIONS`` arrived with an ``Origin`` not on
that allowlist, Starlette's stock ``CORSMiddleware`` returned
``400 Bad Request`` with body ``"Disallowed CORS origin"``. Spec-correct
(browsers still block because ACAO is absent) but noisy in devtools and
confusing for reverse-proxy operators who read 4xx as "the upstream is
unhealthy".

Post-fix (``_SpecAlignedCORSMiddleware``): the preflight rejection is
returned as ``200 OK`` with no ``Access-Control-Allow-Origin`` header and
``Vary: Origin`` so caches don't bleed across origins. The browser still
blocks the real request (ACAO is absent — that's the only browser-
observable signal that matters); devtools shows the missing-header
signal instead of a cryptic 400.

This file ONLY covers the rejection-branch shape. The fail-closed empty-
CSV path (``3da8230``) registers no middleware at all — its preflight
still 405s (no allowed verb on the route) and is covered separately by
``tests/test_cors_env_configurable.py::test_empty_csv_origin_value_fails_closed_with_warning``.
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
    patched to point at it, so the CORS resolver mounts middleware on
    the test app rather than the production singleton."""
    import vllm_mlx.server as server_mod

    importlib.reload(server_mod)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat() -> dict[str, str]:
        return {"ok": "true"}

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
# L-02 — preflight from disallowed origin: 200 + no ACAO + Vary: Origin
# (was: 400 ``Disallowed CORS origin``)
# ──────────────────────────────────────────────────────────────────────


def test_disallowed_origin_preflight_is_200_not_400(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env-locked allowlist, preflight from a non-listed origin → 200
    (not 400). This is the headline L-02 change."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200, (
        f"Expected 200 (spec-aligned, browser blocks via missing ACAO) "
        f"but got {r.status_code} with body {r.text!r}"
    )


def test_disallowed_origin_preflight_omits_acao(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 200 response MUST NOT carry ``Access-Control-Allow-Origin`` —
    that header's absence is what makes the browser block the real
    request. If we accidentally echoed the requested origin we'd be
    silently failing open."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    header_keys = {k.lower() for k in r.headers}
    assert "access-control-allow-origin" not in header_keys, (
        f"ACAO must be absent on origin-mismatch (browser-block signal). "
        f"Got headers: {dict(r.headers)!r}"
    )


def test_disallowed_origin_preflight_sets_vary_origin(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Vary: Origin`` MUST be present on the rejection response so a
    shared HTTP cache (CDN, reverse proxy) doesn't reuse this 200 across
    different origins. Without it, a CDN could serve the same "no ACAO"
    body to a request from an allowed origin and silently break the
    allowed cross-origin path."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # ``Vary`` must include ``Origin`` token — single-row, normalized.
    vary = r.headers.get("vary", "")
    vary_tokens = {t.strip().lower() for t in vary.split(",") if t.strip()}
    assert "origin" in vary_tokens, (
        f"Expected ``Vary: Origin`` on the spec-aligned rejection; "
        f"got Vary={vary!r}"
    )
    # Sanity: no duplicate-row ``Vary: Origin, Origin`` (could trip
    # downstream cache normalizers).
    assert vary.lower().count("origin") == 1, (
        f"Vary must not list Origin twice; got Vary={vary!r}"
    )


def test_disallowed_method_preflight_is_200_not_400(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Access-Control-Request-Method: DELETE`` against the default
    ``POST,GET,OPTIONS`` allowlist also goes 200-not-400. Browsers still
    block because the response doesn't echo ``DELETE`` in
    ``Access-Control-Allow-Methods``. Same shape as the origin case —
    upstream Starlette lumped them in the same 400 envelope."""
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", "https://chat.openai.com")
    _server_mod().configure_cors_from_env(cli_origins=None)

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://chat.openai.com",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert r.status_code == 200
    methods = r.headers.get("access-control-allow-methods", "")
    method_set = {m.strip().upper() for m in methods.split(",") if m.strip()}
    assert "DELETE" not in method_set, (
        f"DELETE must not appear in ACAM on a 200-rejection; got {method_set!r}"
    )


def test_allowed_origin_preflight_still_returns_acao(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity check the happy path is untouched — the subclass only
    overrides the failure envelope. A matching origin still gets a 200
    with ``Access-Control-Allow-Origin`` set to that origin."""
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
    assert r.headers.get("access-control-allow-origin") == "https://chat.openai.com"


def test_failclosed_empty_csv_path_unaffected(
    fresh_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L-02 must NOT touch the fail-closed empty-CSV path (``3da8230``).

    When ``RAPID_MLX_CORS_ALLOW_ORIGINS`` is set to whitespace-only,
    no middleware is registered at all — preflight ``OPTIONS`` on a
    ``POST``-only route still returns 405 because nothing wires the
    verb. This is the operator-visible signal that the env var has a
    templating bug, and the L-02 spec-aligned 200 must not weaken it.
    """
    monkeypatch.setenv("RAPID_MLX_CORS_ALLOW_ORIGINS", " , ,, ")
    origins = _server_mod().configure_cors_from_env(cli_origins=None)
    assert origins == [], "fail-closed branch should return empty list"

    client = TestClient(fresh_app)
    r = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://anywhere.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    # 405 — no CORS middleware → OPTIONS hits the route handler which
    # only declares POST. NOT 200 (would be the L-02 rejection envelope
    # if middleware were registered) and NOT 400 (would be the upstream
    # Starlette envelope L-02 is replacing).
    assert r.status_code == 405, (
        f"Fail-closed branch must still 405 (no CORS middleware "
        f"registered). L-02 must not weaken the operator-visible "
        f"signal from #758 ``3da8230``. Got {r.status_code}."
    )
