# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #574 — bind→auth ordering invariant.

The reported risk: between uvicorn binding the loopback socket and the
FastAPI auth dependency being registered + ready to reject anonymous
requests, a co-located process polling ``127.0.0.1:<port>`` could land
an unauthenticated request on a serve that was meant to be auth-gated.
On a multi-tenant box this is effectively free GPU inference.

Why this is structurally not a window in our app:

* Auth is wired via FastAPI route ``dependencies=[Depends(verify_api_key)]``
  at app construction time (module load of ``vllm_mlx/server.py``).
* By the time ``uvicorn.run(app, ...)`` is invoked, every protected
  router has its dependency chain attached.
* uvicorn binds inside ``Server.serve()`` — strictly AFTER the app is
  fully built. There is no moment where the socket is accepting and
  the dependency is "not yet attached".

These tests pin that invariant so a future refactor (e.g. switching to
``Starlette`` middleware-based auth that registers later, or moving the
``include_router`` calls into the ``lifespan`` hook) cannot silently
reopen the window without a failing test.

Strategy: use FastAPI ``TestClient``, which runs the full ASGI stack
(routing + dependencies). The FIRST request — no warmup, no priming —
must already see auth. We assert against ``GET /v1/models`` because
that's the simplest endpoint a misconfigured supervisor would poll to
detect liveness.

We also pin the inverse contract: with no ``--api-key`` configured,
the development path stays anonymous-OK so plain ``rapid-mlx serve
<alias>`` still works without auth headers.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_models_app() -> FastAPI:
    """Construct the same auth-bearing app routers production serves.

    Mirrors ``vllm_mlx/server.py`` router wiring for the surface most
    likely to be polled by a supervisor: ``/v1/models``.
    """
    from vllm_mlx.routes.models import router as models_router

    app = FastAPI()
    app.include_router(models_router)
    return app


def _patch_cfg(**kwargs):
    """Patch ``get_config()`` fields and return originals for restore."""
    from vllm_mlx.config import get_config

    cfg = get_config()
    originals = {}
    for k, v in kwargs.items():
        originals[k] = getattr(cfg, k)
        setattr(cfg, k, v)
    return originals


def _restore_cfg(originals: dict) -> None:
    from vllm_mlx.config import get_config

    cfg = get_config()
    for k, v in originals.items():
        setattr(cfg, k, v)


# ---------------------------------------------------------------------------
# Leg 1 — bind→auth ordering regression
# ---------------------------------------------------------------------------


def test_first_request_with_api_key_set_is_rejected_no_window():
    """With ``cfg.api_key`` set, the FIRST request to ``/v1/models`` with
    NO ``Authorization`` header gets 401. No 200 window.

    ``TestClient`` runs the full ASGI stack including FastAPI's
    dependency injection — equivalent evidence to a real uvicorn-bound
    socket for the question "is auth attached before the app accepts a
    request?". A 200 here would mean the auth dependency was either
    (a) skipped on this route, or (b) attached lazily and not in place
    for the first request — either way, a regression of #574.
    """
    orig = _patch_cfg(
        api_key="test-secret",
        model_registry=None,
        model_name="test-model",
        model_alias=None,
    )
    try:
        client = TestClient(_make_models_app())
        # First request, no warmup. No Authorization header.
        r = client.get("/v1/models")
        assert r.status_code == 401, (
            f"bind→auth ordering regression: first GET /v1/models with "
            f"no Authorization returned {r.status_code} {r.text!r}; "
            f"expected 401. Issue #574."
        )
        assert r.json()["detail"] == "API key required"
    finally:
        _restore_cfg(orig)


def test_first_request_with_api_key_set_rejects_wrong_key():
    """A WRONG bearer token must still get 401. Pins that the dependency
    runs ``_verify_api_key_values`` (constant-time compare), not just a
    presence check on the header.
    """
    orig = _patch_cfg(
        api_key="test-secret",
        model_registry=None,
        model_name="test-model",
        model_alias=None,
    )
    try:
        client = TestClient(_make_models_app())
        r = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 401
        assert r.json()["detail"] == "Invalid API key"
    finally:
        _restore_cfg(orig)


def test_first_request_with_valid_key_passes():
    """Sanity check on the assertion above: with the correct key, the
    first request returns 200. If this fails, the test harness is
    broken, not the auth ordering.
    """
    orig = _patch_cfg(
        api_key="test-secret",
        model_registry=None,
        model_name="test-model",
        model_alias=None,
    )
    try:
        client = TestClient(_make_models_app())
        r = client.get("/v1/models", headers={"Authorization": "Bearer test-secret"})
        assert r.status_code == 200
    finally:
        _restore_cfg(orig)


def test_no_api_key_keeps_dev_path_anonymous():
    """The no-auth (development) path MUST stay anonymous-OK.

    ``rapid-mlx serve <alias>`` without ``--api-key`` /
    ``RAPID_MLX_API_KEY`` is the on-laptop developer ergonomic — a
    well-meaning fix to #574 must not break it. Pin the contract here so
    a future "always require auth" reflex regression fails loudly.
    """
    orig = _patch_cfg(
        api_key=None,
        model_registry=None,
        model_name="test-model",
        model_alias=None,
    )
    try:
        client = TestClient(_make_models_app())
        r = client.get("/v1/models")
        assert r.status_code == 200, (
            f"dev path regression: GET /v1/models with no api_key "
            f"configured returned {r.status_code} {r.text!r}; expected "
            f"200. The no-auth on-laptop ergonomic must remain."
        )
    finally:
        _restore_cfg(orig)


def test_models_router_has_auth_dependency_declared_statically():
    """Static check: ``verify_api_key`` is attached as a route dependency
    at IMPORT TIME, not lazily during startup.

    This is the structural reason there is no bind→auth window. We do
    this with a route inspection rather than a runtime probe so a
    refactor that moves the dependency into a lifespan hook fails
    explicitly here, not just by accident at the runtime test above.
    """
    from vllm_mlx.middleware.auth import verify_api_key
    from vllm_mlx.routes.models import router

    models_routes = [r for r in router.routes if getattr(r, "path", "") == "/v1/models"]
    assert models_routes, "expected GET /v1/models on models router"
    route = models_routes[0]
    # FastAPI flattens dependencies into ``route.dependant.dependencies``.
    dep_calls = [d.call for d in route.dependant.dependencies]
    assert verify_api_key in dep_calls, (
        "regression: GET /v1/models no longer declares verify_api_key as "
        "a route dependency. The bind→auth ordering guarantee depends on "
        "auth being attached at app-construction time, not lazily."
    )
