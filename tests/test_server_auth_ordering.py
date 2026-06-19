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


# ---------------------------------------------------------------------------
# Static checks across every protected router
# ---------------------------------------------------------------------------

# Every router that ``vllm_mlx/server.py`` mounts with the OpenAI-shape
# contract (chat, embeddings, audio, etc.) AND must reject anonymous
# requests when ``cfg.api_key`` is set. Health-probe endpoints
# (``/healthz``, ``/readyz``) are deliberately NOT in this list — they
# answer the liveness contract anonymously by design.
#
# Codex round-5 PR #696 broadened the auth-ordering test from
# ``models`` alone to every protected router so a regression that
# accidentally drops ``Depends(verify_api_key)`` from any single router
# (e.g. a future ``audio`` rewrite) fails the gate explicitly. The
# membership of this list IS the public contract — add a new entry
# whenever a new protected router lands, remove one only if it's
# moved to anonymous-by-design (and document why).
PROTECTED_ROUTER_MODULES = (
    "vllm_mlx.routes.anthropic",
    "vllm_mlx.routes.audio",
    "vllm_mlx.routes.cache",
    "vllm_mlx.routes.chat",
    "vllm_mlx.routes.completions",
    "vllm_mlx.routes.embeddings",
    "vllm_mlx.routes.mcp_routes",
    "vllm_mlx.routes.models",
    "vllm_mlx.routes.responses",
)


def _route_paths_with_auth(router):
    """Yield ``(path, has_auth_dep)`` for every concrete route on
    ``router`` that has a ``dependant`` (i.e. excludes ``Mount`` /
    ``WebSocketRoute`` plumbing without dependency graphs).

    A route counts as auth-gated if it declares ANY dependency from the
    ``verify_api_key*`` family — currently:

    * ``verify_api_key`` — OpenAI-style bearer-token gate used by
      chat/completions/embeddings/audio/cache/health/mcp/models/responses.
    * ``verify_api_key_or_x_api_key`` — same gate, additionally
      accepting Anthropic-native ``x-api-key`` header. Used by the
      ``/v1/messages*`` routes that mirror Anthropic's API shape.

    Both gates run BEFORE the route handler executes; either is
    structurally equivalent for the bind→auth ordering invariant.
    """
    from vllm_mlx.middleware import auth as auth_mod

    auth_funcs = {
        getattr(auth_mod, name)
        for name in dir(auth_mod)
        if name.startswith("verify_api_key")
    }
    for r in router.routes:
        dep = getattr(r, "dependant", None)
        if dep is None:
            continue
        dep_calls = {d.call for d in dep.dependencies}
        yield (
            getattr(r, "path", "<unknown>"),
            bool(dep_calls & auth_funcs),
        )


import pytest  # noqa: E402  (kept near the parametrize'd test for locality)


@pytest.mark.parametrize("module_name", PROTECTED_ROUTER_MODULES)
def test_every_protected_router_declares_verify_api_key_statically(module_name):
    """Each protected router MUST declare ``verify_api_key`` as a static
    route dependency at IMPORT TIME — not lazily during startup.

    This is the structural reason there is no bind→auth window: by the
    time ``uvicorn.run(app, ...)`` is called, every protected route's
    dependant graph is already wired. We parameterize across the public
    contract so a regression that drops auth from ANY single router
    (e.g. a refactor of ``audio`` that forgets the ``dependencies=``
    kwarg) fails this gate by name, not just by accident.

    Codex round-5 PR #696: the prior single-router version of this
    invariant could pass while auth was accidentally dropped from
    chat/embeddings/audio/etc. — this loop closes that gap.
    """
    import importlib

    router = importlib.import_module(module_name).router

    routes_seen = list(_route_paths_with_auth(router))
    assert routes_seen, (
        f"{module_name}.router exposes no inspectable routes — has the "
        f"module's surface been refactored into Mounts? Update this "
        f"test to follow the new structure."
    )

    missing = [path for path, has_auth in routes_seen if not has_auth]
    assert not missing, (
        f"{module_name} router has route(s) missing verify_api_key as a "
        f"static dependency: {missing}. The bind→auth ordering guarantee "
        f"depends on auth being attached at app-construction time, not "
        f"lazily. Either re-add ``Depends(verify_api_key)`` to those "
        f"endpoints or, if the endpoint is intentionally anonymous, "
        f"split it into its own router and remove that router from "
        f"PROTECTED_ROUTER_MODULES with a comment explaining why."
    )
