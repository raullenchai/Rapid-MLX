# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for F-150 / F-151 control-plane auth.

Pre-fix, ``POST /v1/cache/clear`` and ``POST /v1/requests/{id}/cancel`` were
reachable from any LAN client when ``--api-key`` was not configured (the
default deployment): ``verify_api_key`` returns True the moment ``cfg.api_key``
is unset, so the route group ran wide-open. An attacker could wipe the prefix
cache every few seconds (DoS amplifier — every subsequent prompt is a cache
miss) or fan out abort calls into the engine without ever proving they were
the owner of the request.

The fix moves all destructive control-plane routes onto a dedicated
``admin_router`` whose dependency is ``verify_internal_admin`` — that gate
ALWAYS requires ``X-Rapid-MLX-Internal: true`` regardless of ``--api-key``
state, and additionally requires a valid API key when one is configured.

This file pins the gate's behaviour route-by-route so a future refactor that
slips a destructive route back onto the unauthenticated ``router`` is caught
in CI rather than at the next external pen test.

Pattern mirrors trio's ``/internal/cookie-status`` (``X-Trio-Internal: true``)
referenced in the F-150 / F-151 bug report.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Routes that MUST require ``X-Rapid-MLX-Internal: true``. Parametrize over all
# of them so a future addition to ``admin_router`` only needs an entry here to
# get full unauth/auth/leak coverage.
_DESTRUCTIVE_ROUTES = [
    ("POST", "/v1/cache/clear"),
    ("POST", "/v1/requests/some-request-id/cancel"),
    ("DELETE", "/v1/requests/some-request-id"),
    ("DELETE", "/v1/cache"),
]


@pytest.fixture
def client_factory():
    """Yield ``(client, cfg)`` with a mock engine wired in.

    The mock engine's ``abort_request`` returns True so the cancel route can
    reach a 200 envelope (we check the F-151 leak shape from there). Cache
    routes don't touch the engine for the happy path; the mock satisfies the
    503-check guard in the route handlers.
    """
    from vllm_mlx.config import get_config
    from vllm_mlx.routes.health import admin_router, router

    cfg = get_config()
    prev = {
        "engine": cfg.engine,
        "model_name": cfg.model_name,
        "api_key": cfg.api_key,
    }

    engine = MagicMock()
    engine.abort_request = AsyncMock(return_value=True)
    # ``clear_cache`` looks at ``engine._model._prompt_cache``; a spec=[]
    # MagicMock makes ``hasattr(model, "_prompt_cache")`` return False so the
    # handler takes the "no prompt cache to clear" branch and returns 200 with
    # no engine mutation — exactly what we want for an auth-only test.
    engine._model = MagicMock(spec=[])
    cfg.engine = engine
    # Use a repo-id-shaped name so the F-151 leak assertion can grep for the
    # canonical pattern (org/model) rather than a generic word that might
    # already appear in error envelopes.
    cfg.model_name = "mlx-community/secret-org-model-12b-8bit"

    def build(api_key: str | None = None) -> TestClient:
        cfg.api_key = api_key
        app = FastAPI()
        app.include_router(router)
        app.include_router(admin_router)
        return TestClient(app)

    try:
        yield build, cfg
    finally:
        cfg.engine = prev["engine"]
        cfg.model_name = prev["model_name"]
        cfg.api_key = prev["api_key"]


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_rejects_missing_header(client_factory, method, path):
    """F-150: no ``X-Rapid-MLX-Internal`` header → 403 even when ``--api-key``
    is not configured.

    Pre-fix this returned 200 (cache wipe / abort-on-arbitrary-id). The 403 is
    the ASGI-level signal monitoring rules should alert on.
    """
    build, _ = client_factory
    client = build(api_key=None)

    r = client.request(method, path)

    assert r.status_code == 403, (
        f"{method} {path} should require X-Rapid-MLX-Internal: true even "
        f"when --api-key is unset, got {r.status_code}: {r.text}"
    )
    assert "X-Rapid-MLX-Internal" in r.json()["detail"]


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_rejects_wrong_header_value(
    client_factory, method, path
):
    """The header must literally equal ``true`` (case-insensitive). A typo'd
    or shell-default value like ``1`` / ``yes`` / empty MUST 403 — accepting
    them would turn the gate into a no-op for any client that ships a default
    truthy header injector."""
    build, _ = client_factory
    client = build(api_key=None)

    for bad in ("1", "yes", "false", "", "TRUE_"):
        r = client.request(method, path, headers={"X-Rapid-MLX-Internal": bad})
        assert r.status_code == 403, (
            f"{method} {path} accepted header value {bad!r}: {r.text}"
        )


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_accepts_internal_header_when_no_api_key(
    client_factory, method, path
):
    """F-150 happy path: ``X-Rapid-MLX-Internal: true`` alone is sufficient
    when ``--api-key`` is not configured.

    We only assert that the auth gate passes (status != 401/403). Specific
    response shapes are validated in ``test_routes.py`` /
    ``test_request_cancellation.py``.
    """
    build, _ = client_factory
    client = build(api_key=None)

    r = client.request(method, path, headers={"X-Rapid-MLX-Internal": "true"})

    assert r.status_code not in (401, 403), (
        f"{method} {path} with valid internal header should pass auth, "
        f"got {r.status_code}: {r.text}"
    )


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_case_insensitive_header_value(
    client_factory, method, path
):
    """``True`` / ``TRUE`` are accepted (shells frequently uppercase by reflex).
    Pin this so a future tightening doesn't break documented client code."""
    build, _ = client_factory
    client = build(api_key=None)

    for variant in ("true", "True", "TRUE", " true "):
        r = client.request(
            method, path, headers={"X-Rapid-MLX-Internal": variant}
        )
        assert r.status_code not in (401, 403), (
            f"{method} {path} rejected variant {variant!r}: {r.text}"
        )


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_requires_api_key_when_configured(
    client_factory, method, path
):
    """When ``--api-key`` IS set, the internal header alone is NOT enough —
    a valid Bearer token (or x-api-key) must accompany it. Without this, a
    LAN attacker could bypass the operator's API-key gate just by adding the
    internal header.
    """
    build, _ = client_factory
    client = build(api_key="operator-secret")

    # Internal header but no Bearer → 401 (API key required).
    r = client.request(method, path, headers={"X-Rapid-MLX-Internal": "true"})
    assert r.status_code == 401, (
        f"{method} {path}: internal header should NOT bypass --api-key, "
        f"got {r.status_code}: {r.text}"
    )

    # Both internal header AND valid Bearer → auth passes.
    r = client.request(
        method,
        path,
        headers={
            "X-Rapid-MLX-Internal": "true",
            "Authorization": "Bearer operator-secret",
        },
    )
    assert r.status_code not in (401, 403), (
        f"{method} {path}: internal header + valid bearer should pass, "
        f"got {r.status_code}: {r.text}"
    )

    # Internal header AND valid x-api-key → also passes (Anthropic auth shape).
    r = client.request(
        method,
        path,
        headers={
            "X-Rapid-MLX-Internal": "true",
            "x-api-key": "operator-secret",
        },
    )
    assert r.status_code not in (401, 403), (
        f"{method} {path}: internal header + valid x-api-key should pass, "
        f"got {r.status_code}: {r.text}"
    )


# ---------------------------------------------------------------------------
# F-151 specific: cancel must not leak model_name and must 404 on unknown IDs
# ---------------------------------------------------------------------------


def test_cancel_success_envelope_does_not_leak_model_name(client_factory):
    """F-151 part 2: cancel response MUST NOT include ``model`` (or any other
    server-side fingerprint of the loaded weights).

    The fixture configures ``cfg.model_name`` to a repo-id-shaped string —
    if the route ever re-introduces an envelope field that echoes it, this
    assertion catches the regression before merge."""
    build, cfg = client_factory
    client = build(api_key=None)

    r = client.post(
        "/v1/requests/chatcmpl-real-id/cancel",
        headers={"X-Rapid-MLX-Internal": "true"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "object": "request.cancel",
        "id": "chatcmpl-real-id",
        "cancelled": True,
    }
    # Belt + braces: literal grep over the raw response text.
    assert cfg.model_name not in r.text
    assert "mlx-community" not in r.text


def test_cancel_unknown_id_returns_404_without_leak(client_factory):
    """F-151 part 1: an unknown request ID returns 404, NOT 200 with
    ``cancelled: true``. The 404 detail also must not echo model_name.

    The engine mock simulates the post-fix scheduler returning False for
    unknown IDs; the scheduler-level guarantee is pinned in
    ``test_batching.py::test_abort_nonexistent_request``."""
    build, cfg = client_factory
    client = build(api_key=None)

    # Tell the mock to behave as a post-fix scheduler for unknown IDs.
    cfg.engine.abort_request = AsyncMock(return_value=False)

    r = client.post(
        "/v1/requests/some-bogus-id/cancel",
        headers={"X-Rapid-MLX-Internal": "true"},
    )
    assert r.status_code == 404, r.text
    assert cfg.model_name not in r.text
    assert "mlx-community" not in r.text


def test_cancel_500_error_path_does_not_leak_exception_detail(client_factory):
    """F-151 part 3: when the engine raises during abort, the 500 envelope
    MUST NOT echo the exception message — engine exceptions sometimes carry
    the HF snapshot path / repo id."""
    build, cfg = client_factory
    client = build(api_key=None)

    cfg.engine.abort_request = AsyncMock(
        side_effect=RuntimeError(
            "loaded from /Users/op/.cache/huggingface/hub/secret-snapshot"
        )
    )

    r = client.post(
        "/v1/requests/some-id/cancel",
        headers={"X-Rapid-MLX-Internal": "true"},
    )
    assert r.status_code == 500
    # The leaked path components must not surface in the JSON envelope.
    assert "secret-snapshot" not in r.text
    assert "huggingface" not in r.text
    assert ".cache" not in r.text
