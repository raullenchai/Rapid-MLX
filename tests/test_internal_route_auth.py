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
#
# F-180: ``/v1/cache/{export,import,info}`` were originally on the
# unauth-when-no-api-key ``cache`` router. They are now gated by the same
# ``verify_internal_admin`` dep — pin that here so a future refactor splitting
# the cache router doesn't silently drop the gate again.
_DESTRUCTIVE_ROUTES = [
    ("POST", "/v1/cache/clear"),
    ("POST", "/v1/requests/some-request-id/cancel"),
    ("DELETE", "/v1/requests/some-request-id"),
    ("DELETE", "/v1/cache"),
    ("POST", "/v1/cache/export"),
    ("POST", "/v1/cache/import"),
    ("GET", "/v1/cache/info"),
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
    from vllm_mlx.routes.cache import router as cache_router
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

    def build(api_key: str | None = None, client_host: str = "127.0.0.1") -> TestClient:
        """Build a TestClient with a configurable origin host.

        ``client_host`` defaults to ``127.0.0.1`` so the loopback branch
        of ``verify_internal_admin`` resolves true — this lets tests
        focus on the header gate without also having to thread an
        operator api-key through every case. Tests that want to assert
        the LAN-no-api-key 403 path pass ``client_host="10.0.0.5"`` or
        similar (codex r1 BLOCKING fix).
        """
        cfg.api_key = api_key
        app = FastAPI()
        app.include_router(router)
        app.include_router(admin_router)
        # F-180: ``cache_router`` carries the three sibling cache routes
        # (export/import/info) that PR #728 missed. Mount it here so the
        # _DESTRUCTIVE_ROUTES parametrize matrix can pin their auth shape.
        app.include_router(cache_router)
        # ``TestClient(app, client=(host, port))`` injects the supplied
        # tuple into ``scope["client"]`` so ``request.client.host`` reads
        # back as ``host``. The default ``("testclient", 50000)`` is NOT
        # loopback, so without overriding it the new loopback-required
        # branch would reject every test — masking real bugs.
        return TestClient(app, client=(client_host, 50000))

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
def test_destructive_route_rejects_wrong_header_value(client_factory, method, path):
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
    """F-150 happy path: ``X-Rapid-MLX-Internal: true`` + loopback caller is
    sufficient when ``--api-key`` is not configured.

    Per the codex r1 BLOCKING fix, the header alone is NOT enough — a LAN
    caller still needs an api-key. Loopback is the dev-only escape hatch.
    The fixture defaults ``client_host="127.0.0.1"`` so this test exercises
    that branch. We only assert that the auth gate passes (status != 401/403).
    Specific response shapes are validated in ``test_routes.py`` /
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
def test_destructive_route_case_insensitive_header_value(client_factory, method, path):
    """``True`` / ``TRUE`` are accepted (shells frequently uppercase by reflex).
    Pin this so a future tightening doesn't break documented client code."""
    build, _ = client_factory
    client = build(api_key=None)

    for variant in ("true", "True", "TRUE", " true "):
        r = client.request(method, path, headers={"X-Rapid-MLX-Internal": variant})
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


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_rejects_lan_caller_without_api_key(
    client_factory, method, path
):
    """Codex PR #728 round-1 BLOCKING: when ``--api-key`` is unset, a non-
    loopback caller with the internal header MUST still 403.

    Pre-codex-fix the header alone was sufficient. That left any LAN client
    who read the open-source code able to send the header and wipe cache —
    the header is not a secret, it's a "you meant this" signal. The
    production posture is: operator sets ``--api-key`` → bearer required;
    dev posture is: ``--api-key`` unset → loopback-only. There is no
    "anonymous remote callers with the header" posture anymore.
    """
    build, _ = client_factory
    # 10.x is the canonical "definitely not loopback" LAN range used by
    # cloud / k8s test fixtures.
    client = build(api_key=None, client_host="10.0.0.5")

    r = client.request(method, path, headers={"X-Rapid-MLX-Internal": "true"})
    assert r.status_code == 403, (
        f"{method} {path}: LAN caller (10.0.0.5) with header but no --api-key "
        f"should 403, got {r.status_code}: {r.text}"
    )


@pytest.mark.parametrize(("method", "path"), _DESTRUCTIVE_ROUTES)
def test_destructive_route_accepts_lan_caller_with_valid_api_key(
    client_factory, method, path
):
    """Cross-check the codex fix: a remote caller with the header AND a
    valid bearer key (the production posture) still works. Otherwise the
    fix would have made the routes effectively localhost-only, breaking
    operators who run ``rapid-mlx`` behind a private VPN with ``--api-key``
    set and want to hit the admin routes from their laptop.
    """
    build, _ = client_factory
    client = build(api_key="operator-secret", client_host="10.0.0.5")

    r = client.request(
        method,
        path,
        headers={
            "X-Rapid-MLX-Internal": "true",
            "Authorization": "Bearer operator-secret",
        },
    )
    assert r.status_code not in (401, 403), (
        f"{method} {path}: LAN caller with valid api-key should pass, "
        f"got {r.status_code}: {r.text}"
    )


def test_loopback_ipv6_is_recognised():
    """``::1`` (canonical IPv6 loopback) is treated the same as 127.0.0.1.
    Defends against a future refactor that switches to a string-equality
    check and accidentally locks out callers binding on ``::1``."""
    from vllm_mlx.middleware.auth import _is_loopback_client

    # Build a minimal request-like object exposing ``.client.host`` and
    # ``.headers.get(...)`` — both are consumed by ``_is_loopback_client``
    # (codex r3 added the proxy-trail header check).
    class _C:
        def __init__(self, host: str) -> None:
            self.host = host
            self.port = 50000

    class _H:
        def __init__(self, headers: dict | None = None) -> None:
            self._h = headers or {}

        def get(self, name: str, default=None):
            # Normalise to lower-case to match Starlette's case-insensitive
            # header lookup — the production code passes already-lowered
            # keys but a future caller might not.
            return self._h.get(name.lower(), default)

    class _R:
        def __init__(self, host: str, headers: dict | None = None) -> None:
            self.client = _C(host)
            self.headers = _H(headers)

    assert _is_loopback_client(_R("127.0.0.1")) is True
    assert _is_loopback_client(_R("::1")) is True
    assert _is_loopback_client(_R("localhost")) is True
    # Bare IPv4-mapped-IPv6 form of loopback also resolves via is_loopback.
    assert _is_loopback_client(_R("::ffff:127.0.0.1")) is True
    # Anything in the 127.0.0.0/8 block.
    assert _is_loopback_client(_R("127.0.0.42")) is True
    # And anything outside is NOT loopback.
    assert _is_loopback_client(_R("10.0.0.5")) is False
    assert _is_loopback_client(_R("8.8.8.8")) is False
    assert _is_loopback_client(_R("not-an-ip")) is False
    # ``request.client`` may be None on some ASGI servers / test rigs.
    assert _is_loopback_client(_R(None)) is False  # type: ignore[arg-type]


def test_loopback_rejects_proxied_caller(client_factory):
    """Codex PR #728 round-3 BLOCKING: a same-host reverse proxy (nginx
    ``proxy_pass http://127.0.0.1:8000``) makes every external client look
    like ``127.0.0.1``. The fix REJECTS any request carrying a proxy-trail
    header, so a misconfigured deploy can't expose the admin routes via
    its load balancer.

    Each parametrised header is tested independently to pin which signals
    we trust. The set is documented in ``_is_loopback_client``'s docstring.
    """
    build, _ = client_factory
    client = build(api_key=None, client_host="127.0.0.1")

    proxy_signals = [
        {"X-Forwarded-For": "8.8.8.8"},
        {"X-Forwarded-Host": "example.com"},
        {"X-Forwarded-Proto": "https"},
        {"Forwarded": 'for="8.8.8.8";proto=https'},
        {"Via": "1.1 nginx-proxy"},
        {"CF-Connecting-IP": "8.8.8.8"},
        {"True-Client-IP": "8.8.8.8"},
    ]
    for extra in proxy_signals:
        r = client.post(
            "/v1/cache/clear",
            headers={"X-Rapid-MLX-Internal": "true", **extra},
        )
        assert r.status_code == 403, (
            f"Request with proxy signal {list(extra)[0]!r} should 403 "
            f"even from 127.0.0.1, got {r.status_code}: {r.text}"
        )


def test_loopback_accepts_direct_caller_with_no_proxy_headers(client_factory):
    """Cross-check: a direct loopback caller without any proxy-trail
    headers still gets through. Otherwise the round-3 fix would have
    accidentally locked out the dev-on-localhost path entirely.
    """
    build, _ = client_factory
    client = build(api_key=None, client_host="127.0.0.1")

    r = client.post(
        "/v1/cache/clear",
        headers={"X-Rapid-MLX-Internal": "true"},
    )
    assert r.status_code not in (401, 403), (
        f"Direct loopback caller should pass, got {r.status_code}: {r.text}"
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


# ---------------------------------------------------------------------------
# F-180 specific: cache export/import 501 envelope must not leak operator-
# controlled disk paths or GitHub tracking URLs.
# ---------------------------------------------------------------------------


def test_cache_export_501_envelope_does_not_leak_operator_path(client_factory):
    """F-180: pre-fix, ``POST /v1/cache/export`` 501-stubbed back the resolved
    sandbox destination — which expands to ``/Users/<USERNAME>/.cache/rapid-mlx/
    cache_exports`` and leaks the operator's home dir / username. The fix
    sanitizes the envelope to a minimal ``{"error": {...}}`` shape; the
    resolved path stays in server logs only.
    """
    build, _ = client_factory
    client = build(api_key=None)

    r = client.post(
        "/v1/cache/export",
        headers={"X-Rapid-MLX-Internal": "true"},
        json={},
    )
    assert r.status_code == 501, r.text
    body = r.json()
    # No resolved-path-shaped strings in the body.
    assert "/Users/" not in r.text
    assert ".cache" not in r.text
    assert "rapid-mlx" not in r.text
    assert "cache_exports" not in r.text
    # No tracking-URL leak either (per test-loop8 finding).
    assert "github.com" not in r.text
    assert "issues/" not in r.text
    # Minimal envelope shape.
    detail = body.get("detail")
    assert isinstance(detail, dict) and "error" in detail, body


def test_cache_import_501_envelope_does_not_leak_operator_path(
    client_factory, tmp_path, monkeypatch
):
    """F-180: same shape check for ``POST /v1/cache/import``. We point the
    sandbox env at a tmp dir so we can hand-craft a manifest the route accepts,
    then assert the resulting 501 body is path-free.
    """
    # Point the export sandbox at tmp_path so the resolved source / manifest
    # path is under the test rig, not the operator's home dir.
    monkeypatch.setenv("RAPID_MLX_CACHE_EXPORT_DIR", str(tmp_path))

    # Hand-craft a valid manifest.json so import gets past the validation
    # branches and into the 501 stub.
    import json

    from vllm_mlx.cache.protocol import PROTOCOL_VERSION

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "model_id": "test-model",
        "entries": 0,
        "total_bytes": 0,
        "created_at": "2026-06-20T00:00:00Z",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    build, _ = client_factory
    client = build(api_key=None)

    r = client.post(
        "/v1/cache/import",
        headers={"X-Rapid-MLX-Internal": "true"},
        json={"source": str(tmp_path)},
    )
    assert r.status_code == 501, r.text
    body = r.json()
    # No resolved-path-shaped strings.
    assert str(tmp_path) not in r.text
    assert "/Users/" not in r.text
    assert "cache_exports" not in r.text
    # No tracking-URL leak.
    assert "github.com" not in r.text
    assert "issues/" not in r.text
    # Minimal envelope shape.
    detail = body.get("detail")
    assert isinstance(detail, dict) and "error" in detail, body
