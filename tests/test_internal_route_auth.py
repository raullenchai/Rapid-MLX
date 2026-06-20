# SPDX-License-Identifier: Apache-2.0
"""Wire-level coverage for the destructive control-plane routes.

History: the F-150 / F-151 fixes (#728) gated ``POST /v1/cache/clear``,
``POST /v1/requests/{id}/cancel``, the ``DELETE`` aliases, and
``DELETE /v1/cache`` behind ``verify_internal_admin`` — an ``X-Rapid-MLX-
Internal: true`` header plus loopback-or-api-key check. Per operator
intent (single-machine UX, no API key gate), #728's auth bundle was
reverted; these routes now run on plain ``verify_api_key`` (no-op when
``--api-key`` is unset).

What stays from #728 and is pinned here:

* Cancel envelope sanitization — F-151 part 2. The success body must NOT
  echo ``cfg.model_name``; the 404 path must NOT echo the request id;
  the 500 path must NOT echo the engine exception text.
* Cache export/import 501 envelope sanitization (added in the #756
  partial revert) — resolved sandbox paths + manifest contents stay in
  server logs only.

The auth-matrix tests from #728 (no-header → 403, wrong header → 403,
LAN without api-key → 403) are gone because the auth gate is gone.
The ``X-Rapid-MLX-Internal: true`` header is now harmless extra
metadata; tests can pass it or not without changing behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Destructive routes that still need leak-shape coverage. The cache router
# is mounted separately in cache-specific tests; ``health.admin_router``
# carries the cancel + cache-clear routes covered here.
_DESTRUCTIVE_ROUTES = [
    ("POST", "/v1/cache/clear"),
    ("POST", "/v1/requests/some-request-id/cancel"),
    ("DELETE", "/v1/requests/some-request-id"),
    ("DELETE", "/v1/cache"),
]


@pytest.fixture
def client_factory():
    """Yield ``(build, cfg)`` with a mock engine wired in.

    The mock engine's ``abort_request`` returns True so the cancel route
    can reach a 200 envelope (we check the F-151 leak shape from there).
    Cache routes don't touch the engine for the happy path; the mock
    satisfies the 503-check guard in the route handlers.
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
    # ``clear_cache`` looks at ``engine._model._prompt_cache``; spec=[]
    # makes ``hasattr(model, "_prompt_cache")`` return False so the
    # handler takes the "no prompt cache to clear" branch and returns
    # 200 with no engine mutation — exactly what we want for a wire test.
    engine._model = MagicMock(spec=[])
    cfg.engine = engine
    # Repo-id-shaped name so the F-151 leak assertion can grep for the
    # canonical pattern (``org/model``) rather than a generic word that
    # might already appear in error envelopes.
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


# ---------------------------------------------------------------------------
# verify_api_key still gates when --api-key is set (sanity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_requires_credential_when_api_key_configured(
    client_factory, method, path
):
    """When the operator sets ``--api-key``, the destructive routes still
    require a matching credential — ``verify_api_key_or_x_api_key``
    accepts EITHER ``Authorization: Bearer ...`` OR ``x-api-key`` (the
    dual shape that the reverted ``verify_internal_admin`` also accepted,
    so Anthropic-style clients hitting these routes don't break). Codex
    r1 on PR #760: switching to plain ``verify_api_key`` here would have
    silently dropped ``x-api-key`` callers."""
    build, _ = client_factory
    client = build(api_key="operator-secret")

    no_creds = client.request(method, path)
    assert no_creds.status_code == 401, (
        f"{method} {path}: --api-key set but no credential → expected 401, "
        f"got {no_creds.status_code}: {no_creds.text}"
    )

    with_bearer = client.request(
        method,
        path,
        headers={"Authorization": "Bearer operator-secret"},
    )
    assert with_bearer.status_code not in (401, 403), (
        f"{method} {path}: valid bearer should pass, "
        f"got {with_bearer.status_code}: {with_bearer.text}"
    )

    with_x_api_key = client.request(
        method,
        path,
        headers={"x-api-key": "operator-secret"},
    )
    assert with_x_api_key.status_code not in (401, 403), (
        f"{method} {path}: valid x-api-key should pass (Anthropic shape), "
        f"got {with_x_api_key.status_code}: {with_x_api_key.text}"
    )


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_open_when_no_api_key(client_factory, method, path):
    """When ``--api-key`` is unset (single-machine default), the routes
    run wide open per the #728 revert. Pin this so a future tightening is
    a conscious decision, not an accidental regression."""
    build, _ = client_factory
    client = build(api_key=None)

    r = client.request(method, path)
    assert r.status_code not in (401, 403), (
        f"{method} {path}: no --api-key should NOT 401/403 after revert, "
        f"got {r.status_code}: {r.text}"
    )


# ---------------------------------------------------------------------------
# F-151: cancel must not leak model_name and must 404 on unknown IDs
# (kept from #728 — these are real bugs unrelated to the auth gate)
# ---------------------------------------------------------------------------


def test_cancel_success_envelope_does_not_leak_model_name(client_factory):
    """F-151 part 2: cancel response MUST NOT include ``model`` (or any
    other server-side fingerprint of the loaded weights).

    The fixture configures ``cfg.model_name`` to a repo-id-shaped string
    — if the route ever re-introduces an envelope field that echoes it,
    this assertion catches the regression before merge."""
    build, cfg = client_factory
    client = build(api_key=None)

    r = client.post("/v1/requests/chatcmpl-real-id/cancel")
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

    cfg.engine.abort_request = AsyncMock(return_value=False)

    r = client.post("/v1/requests/some-bogus-id/cancel")
    assert r.status_code == 404, r.text
    assert cfg.model_name not in r.text
    assert "mlx-community" not in r.text


def test_cancel_500_error_path_does_not_leak_exception_detail(client_factory):
    """F-151 part 3: when the engine raises during abort, the 500 envelope
    MUST NOT echo the exception message — engine exceptions sometimes
    carry the HF snapshot path / repo id."""
    build, cfg = client_factory
    client = build(api_key=None)

    cfg.engine.abort_request = AsyncMock(
        side_effect=RuntimeError(
            "loaded from /Users/op/.cache/huggingface/hub/secret-snapshot"
        )
    )

    r = client.post("/v1/requests/some-id/cancel")
    assert r.status_code == 500
    assert "secret-snapshot" not in r.text
    assert "huggingface" not in r.text
    assert ".cache" not in r.text


# ---------------------------------------------------------------------------
# Cache export/import 501 envelope sanitization (kept from #756)
# ---------------------------------------------------------------------------


_EXPECTED_NOT_IMPLEMENTED_ENVELOPE = {
    "error": {
        "message": "engine integration pending",
        "type": "not_implemented_error",
        "code": None,
    }
}


def test_cache_export_501_envelope_does_not_leak_operator_path(client_factory):
    """``POST /v1/cache/export`` 501 stub must not echo the resolved sandbox
    destination — that expands to ``/Users/<USERNAME>/.cache/rapid-mlx/
    cache_exports`` and leaks operator home dir / username to any
    bearer-token holder. After the #728 revert the route runs on plain
    ``verify_api_key``, so this leak shape matters even more when
    ``--api-key`` is unset."""
    from vllm_mlx.routes.cache import router as cache_router

    build, _ = client_factory
    client = build(api_key=None)
    client.app.include_router(cache_router)

    r = client.post("/v1/cache/export", json={})
    assert r.status_code == 501, r.text
    body = r.json()
    for needle in ("/Users/", ".cache", "rapid-mlx", "cache_exports"):
        assert needle not in r.text, f"{needle!r} leaked into 501 body: {r.text!r}"
    for needle in ("github.com", "issues/"):
        assert needle not in r.text, f"{needle!r} leaked into 501 body: {r.text!r}"
    assert body.get("detail") == _EXPECTED_NOT_IMPLEMENTED_ENVELOPE, body


_EXPECTED_SANDBOX_ESCAPE_ENVELOPE = {
    "error": {
        "message": "destination must resolve under the cache-export sandbox",
        "type": "invalid_request_error",
        "code": "sandbox_escape",
    }
}


@pytest.mark.parametrize(
    "destination",
    [
        "/tmp/foo",
        "/",
        "../../etc/passwd",
        "/Users/x",
        "/etc/passwd",
    ],
)
def test_cache_export_403_sandbox_escape_does_not_leak_operator_path(
    client_factory, destination
):
    """H-02: ``POST /v1/cache/export`` with an out-of-sandbox destination
    returns 403, and the body must NOT echo the resolved sandbox root.

    Pre-fix the 403 detail expanded ``InvalidExportPathError`` via
    ``str(exc)``, which embeds ``/Users/<USERNAME>/.cache/rapid-mlx/
    cache_exports`` — username + home-dir disclosure to any LAN caller
    after the #756 auth-gate revert. Mirrors the strictness of the
    sibling ``test_cache_export_501_envelope_does_not_leak_operator_path``
    so a future regression at the same site is caught by the same
    needles.
    """
    from pathlib import Path

    from vllm_mlx.routes.cache import router as cache_router

    build, _ = client_factory
    client = build(api_key=None)
    client.app.include_router(cache_router)

    r = client.post("/v1/cache/export", json={"destination": destination})
    assert r.status_code == 403, r.text
    body = r.json()
    # Belt + braces leak sweep — these are the substrings that the
    # pre-fix envelope leaked verbatim. Same shape as the 501 test.
    home = str(Path.home())
    for needle in (home, "/Users/", ".cache", "rapid-mlx", "cache_exports"):
        assert needle not in r.text, (
            f"{needle!r} leaked into 403 sandbox-escape body for "
            f"destination={destination!r}: {r.text!r}"
        )
    assert body.get("detail") == _EXPECTED_SANDBOX_ESCAPE_ENVELOPE, body


def test_cache_import_501_envelope_does_not_leak_operator_path(
    client_factory, tmp_path, monkeypatch
):
    """Same shape check for ``POST /v1/cache/import``. Point the sandbox
    at a tmp dir, hand-craft a valid manifest so the route gets past
    validation into the 501 stub, then assert the body is path-free +
    manifest-free."""
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(tmp_path))

    import json

    from vllm_mlx.cache.protocol import PROTOCOL_VERSION
    from vllm_mlx.routes.cache import router as cache_router

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "secret-org-model-12b-8bit",
        "entries": 0,
        "total_bytes": 0,
        "created_at": "2026-06-20T00:00:00Z",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    build, _ = client_factory
    client = build(api_key=None)
    client.app.include_router(cache_router)

    r = client.post("/v1/cache/import", json={"source": str(tmp_path)})
    assert r.status_code == 501, r.text
    body = r.json()
    # The resolved source path is the most direct leak the 501 stub
    # could surface — exact-string check before the substring sweep.
    assert str(tmp_path) not in r.text, (
        f"resolved source path {str(tmp_path)!r} leaked into 501 body: {r.text!r}"
    )
    for needle in ("/Users/", ".cache", "cache_exports"):
        assert needle not in r.text, f"{needle!r} leaked into 501 body: {r.text!r}"
    assert "secret-org-model" not in r.text
    assert body.get("detail") == _EXPECTED_NOT_IMPLEMENTED_ENVELOPE, body


# ---------------------------------------------------------------------------
# H-12: /v1/cache/info envelope must not leak the resolved sandbox path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "needle",
    [
        "/Users/",
        ".cache",
        "cache_exports",
        "rapid-mlx",
    ],
)
def test_cache_info_does_not_leak_operator_path(
    client_factory, tmp_path, monkeypatch, needle
):
    """H-12: ``GET /v1/cache/info`` returned ``{"path": str(root), ...}``
    where ``root`` is the fully resolved sandbox subdirectory
    (``/Users/<USERNAME>/.cache/rapid-mlx/cache_exports/<sub>`` on macOS).
    Same shape as H-02 on a different cache endpoint — leaks the
    operator's home dir + username to any LAN caller after the #756
    auth-gate revert removed ``verify_internal_admin`` from this route.

    Mimic the user-home-shaped sandbox so the substrings actually appear
    in the resolved path even though ``tmp_path`` lives under ``/private
    /var/folders/...``. Drop a valid manifest, then sweep the 200 body
    for each leak needle (parametrized so a failure pinpoints which
    substring leaked). Codex r1 NIT: the needle list mirrors the H-02
    sibling sweep — substrings that actually appear in the constructed
    sandbox path, not unexpanded shell variables like ``$HOME``.
    """
    # Build a tmp sandbox whose absolute path contains every leak needle
    # we care about — so a route that echoes ``str(root)`` would fail
    # every parametrized case, not just the ones whose substring happens
    # to land in the OS-provided ``tmp_path``.
    sandbox = tmp_path / "Users" / "yuki" / ".cache" / "rapid-mlx" / "cache_exports"
    sandbox.mkdir(parents=True)
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(sandbox))

    import json

    from vllm_mlx.cache.protocol import PROTOCOL_VERSION
    from vllm_mlx.routes.cache import router as cache_router

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "secret-org-model-12b-8bit",
        "entries": 0,
        "total_bytes": 0,
        "created_at": "2026-06-20T00:00:00Z",
    }
    (sandbox / "manifest.json").write_text(json.dumps(manifest))

    build, _ = client_factory
    client = build(api_key=None)
    client.app.include_router(cache_router)

    r = client.get("/v1/cache/info")
    assert r.status_code == 200, r.text
    assert needle not in r.text, (
        f"{needle!r} leaked into /v1/cache/info 200 body: {r.text!r}"
    )


def test_cache_info_returns_canonical_shape_without_path_field(
    client_factory, tmp_path, monkeypatch
):
    """H-12: positive contract pin. Post-fix the 200 envelope carries
    ``protocol_version`` + ``manifest`` but NOT a top-level ``"path"``
    field — the resolved sandbox root stays in the server log only.

    The exact-string check on ``str(sandbox)`` catches a hypothetical
    regression that serializes the path into a renamed field (e.g.
    ``resolved_path``, ``root``, ``location``) which would slip past
    the substring sweep when the tmp dir happens to live outside
    ``/Users/``.
    """
    sandbox = tmp_path / "h12-canonical-shape"
    sandbox.mkdir(parents=True)
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(sandbox))

    import json

    from vllm_mlx.cache.protocol import PROTOCOL_VERSION
    from vllm_mlx.routes.cache import router as cache_router

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "secret-org-model-12b-8bit",
        "entries": 7,
        "total_bytes": 0,
        "created_at": "2026-06-20T00:00:00Z",
    }
    (sandbox / "manifest.json").write_text(json.dumps(manifest))

    build, _ = client_factory
    client = build(api_key=None)
    client.app.include_router(cache_router)

    r = client.get("/v1/cache/info")
    assert r.status_code == 200, r.text
    body = r.json()
    # H-12 wire-shape pin: top-level keys are exactly these two. A new
    # field that re-introduces the path leak under a different name
    # (``resolved_path``, ``root``, ``location``) trips this set check.
    assert set(body.keys()) == {"protocol_version", "manifest"}, body
    assert body["protocol_version"] == PROTOCOL_VERSION
    # Manifest payload is intentionally preserved — the leak was the
    # resolved sandbox root, NOT the manifest contents which the caller
    # supplied / would receive on a full import.
    assert body["manifest"]["model_id"] == "secret-org-model-12b-8bit"
    assert body["manifest"]["entries"] == 7
    # Belt + braces: the sandbox path is unique per test run and would
    # not otherwise appear in the response. Catches a regression where
    # the path field is renamed but still echoed.
    assert str(sandbox) not in r.text, (
        f"resolved root {str(sandbox)!r} leaked into /v1/cache/info "
        f"200 body: {r.text!r}"
    )
