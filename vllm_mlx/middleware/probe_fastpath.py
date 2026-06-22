# SPDX-License-Identifier: Apache-2.0
"""ASGI fast-path for k8s liveness probes.

R8-H6 (dogfood-089 Talia r1/r2): ``GET /healthz`` and ``GET /livez``
p99 climbed to 67-113 ms under 8-way streaming concurrency, well past
the 50 ms k8s probe budget. R7-D-2 already made the route handler
constant-time (no ``engine.get_stats()``, no scheduler lock — see
``routes/health.py``). The remaining tail came from the per-request
overhead of going through Starlette's routing + FastAPI's dependency
resolution + Pydantic-aware response serialization on every probe hit,
all of which compete for the event loop with the streaming SSE
generators.

This middleware short-circuits ``GET /healthz`` and ``GET /livez``
**before** any of that machinery runs. The path comparison, three
attribute reads off the singleton ``ServerConfig`` dataclass, the
``json.dumps`` of a 4-key dict, and two ``send`` calls are the entire
hot path — no router traversal, no dependency graph walk, no response
class construction, no exception-handler chain wrapping. Under
streaming load this drops the probe past p99 from "queued behind
several SSE chunk emissions on the loop" to "one extra coroutine
yield".

Constraints + invariants:

* Installed via :func:`install_probe_fastpath_middleware` AFTER all
  other ``add_middleware`` / install calls so it ends up OUTERMOST in
  the Starlette stack (Starlette stacks middleware in reverse install
  order). Outermost is critical: the fast-path must run before any
  body-size / body-depth / CORS work, because those iterate scope
  headers and add per-request overhead.
* GET-only and path-equal-only. ``HEAD`` falls through to the router so
  Starlette's HEAD-on-GET auto-derivation keeps working. Any other
  method (POST/PUT/DELETE) on these paths is a misuse and stays on the
  normal handler path — the cost there is bounded by the existing 405
  / 404 envelope.
* Response shape MUST match the existing ``routes/health.py``
  handlers byte-for-byte where the dynamic fields agree, so dashboards
  / probe specs reading either path interchangeably do not see drift.
  The ``/healthz`` shape (``status``, ``ready``, ``model_loaded``,
  ``model_name``) and ``/livez`` shape (``status``) are pinned by the
  existing route tests in ``test_routes.py``; this middleware ships
  the same JSON.
* Does NOT regress R7-D-1 (SIGHUP stay-alive — the signal handler
  chain lives in ``_signal_observability.py`` and runs orthogonally)
  or R7-D-2 (SIGTERM graceful drain — the lifespan shutdown order is
  untouched).
* CORS / preflight pass-through: if the request carries an ``Origin``
  header AND CORS middleware is registered, we fall through to the
  normal stack so CORSMiddleware can attach the
  ``Access-Control-Allow-Origin`` response header. K8s / Docker /
  systemd probes do not send ``Origin``; browser-driven dashboards
  that hit ``/healthz`` from a different origin still get the
  spec-compliant CORS handling. This keeps the fast-path safe for the
  vast majority of probe traffic (no Origin) without breaking the
  browser cross-origin case.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import get_config

logger = logging.getLogger(__name__)

# Paths the fast-path handles. Path equality (not prefix) — these are
# leaf routes with no path parameters.
_FAST_PATHS: frozenset[bytes] = frozenset(
    {
        b"/healthz",
        b"/livez",
    }
)

# Response headers — must be byte-shape compatible with what
# FastAPI's default JSONResponse emits from ``routes/health.py``,
# so callers / dashboards / probe specs that compare the documented
# probe surface byte-for-byte don't see drift. FastAPI's default
# JSONResponse emits only ``content-type: application/json`` (plus
# ``content-length``, which we set per-response below). Codex r2
# BLOCKING #1: an earlier round added ``cache-control: no-store``
# on speculative grounds, but the route handler does NOT emit it,
# so the fast-path was diverging from the route's response shape.
# Stay minimal here; an operator who wants a no-store hint should
# add it on both code paths via FastAPI's response middleware so
# the route + fast-path stay aligned.
_BASE_HEADERS: list[tuple[bytes, bytes]] = [
    (b"content-type", b"application/json"),
]


def _build_healthz_payload() -> bytes:
    """Build the ``/healthz`` JSON response body.

    Reads three constant-time fields off the ``ServerConfig`` singleton.
    No engine call, no MCP iteration, no scheduler lock. Identical
    semantics to ``routes/health.py::healthz`` — that handler is kept
    as the fall-through path for HEAD requests, requests with
    ``Origin`` headers (CORS), and as the reference shape for tests.
    """
    cfg = get_config()
    payload = {
        "status": "healthy",
        "ready": bool(cfg.ready),
        "model_loaded": cfg.engine is not None,
        "model_name": cfg.model_name,
    }
    # ``separators`` strips whitespace — shaves a handful of bytes and
    # microseconds off the json.dumps call. The output is still valid
    # JSON; the trailing newline is intentionally omitted (responders
    # like nginx-ingress don't expect one).
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# Pre-encoded ``/livez`` payload — completely static, never touches the
# config singleton. ``status:alive`` mirrors the existing handler;
# pre-encoding here means the only per-request work is the path match
# and two ASGI ``send`` calls.
_LIVEZ_BODY: bytes = b'{"status":"alive"}'


def _has_origin(scope: dict[str, Any]) -> bool:
    """Return True if the request carries an ``Origin`` header.

    K8s probes, supervisord, systemd, Docker healthchecks do not send
    Origin — the fast-path serves them. A browser dashboard hitting
    ``/healthz`` cross-origin DOES send Origin; in that case we fall
    through to the normal stack so CORSMiddleware can attach the
    ``Access-Control-Allow-Origin`` header. Trading a few microseconds
    of fast-path for spec-compliant CORS on the cross-origin slice is
    the right call.

    Codex r3 BLOCKING #1: the ASGI spec (PEP 3333 / ASGI 2.x) requires
    headers to be lowercased before they reach app code, but not every
    ASGI server or test harness enforces this — uvicorn does, h11 with
    a custom scope adapter may not, and TestClient's manual scope
    construction is operator-dependent. Lowercasing defensively here
    avoids a silent CORS-bypass failure mode where the fast-path
    serves a 200 to a cross-origin browser request that should have
    received ACAO from the outer CORS middleware.
    """
    for name, _value in scope.get("headers", ()):
        if name.lower() == b"origin":
            return True
    return False


class ProbeFastPathMiddleware:
    """ASGI middleware that short-circuits ``GET /healthz`` + ``GET /livez``.

    The fall-through cases are listed in :func:`__call__`. Anything not
    matched takes the normal Starlette/FastAPI router path with full
    middleware semantics — this middleware is a pure perf optimization
    for the hot probe path, not a security / correctness boundary.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        # Fast-path only HTTP requests. WebSocket / lifespan scopes
        # fall through unconditionally.
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        # GET-only. HEAD falls through so Starlette's HEAD-on-GET
        # auto-derivation keeps working; POST/PUT/DELETE on these
        # paths take the normal 405 path.
        method = scope.get("method")
        if method != "GET":
            return await self.app(scope, receive, send)

        # ``raw_path`` is the byte-string path before percent-decoding;
        # falling back to ``path`` (str) handles the rare ASGI server
        # that doesn't populate raw_path. Path equality (not prefix) —
        # ``/healthz/foo`` falls through to the 404 handler.
        raw_path = scope.get("raw_path")
        if raw_path is None:
            path_str = scope.get("path") or ""
            raw_path = path_str.encode("ascii", "replace")
        # Strip any trailing query string from raw_path. ASGI spec says
        # raw_path doesn't include the query string, but be defensive
        # — a probe URL of ``/healthz?ts=1234`` (cache-busting) should
        # still hit the fast-path.
        qmark = raw_path.find(b"?")
        if qmark != -1:
            raw_path = raw_path[:qmark]
        if raw_path not in _FAST_PATHS:
            return await self.app(scope, receive, send)

        # Cross-origin browser hits fall through so CORSMiddleware can
        # attach the ACAO header. K8s / supervisord / Docker probes
        # don't send Origin, so they stay on the fast-path — which is
        # the slice that needs the p99 win.
        if _has_origin(scope):
            return await self.app(scope, receive, send)

        # Build the response body. ``/livez`` is completely static;
        # ``/healthz`` reads three constant-time fields off the config
        # singleton.
        if raw_path == b"/livez":
            body = _LIVEZ_BODY
        else:
            try:
                body = _build_healthz_payload()
            except Exception:
                # Defensive: if the config singleton is in a half-init
                # state (lifespan crash, embedded harness mid-tear-
                # down), fall through to the normal handler rather
                # than 500'ing the probe. The route-level handler has
                # the same try/except shape via the dataclass defaults.
                logger.debug("[probe_fastpath] payload build raised; falling through")
                return await self.app(scope, receive, send)

        headers = _BASE_HEADERS + [
            (b"content-length", str(len(body)).encode("ascii")),
        ]

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


def install_probe_fastpath_middleware(app: Any) -> None:
    """Register :class:`ProbeFastPathMiddleware` on ``app``.

    MUST be called AFTER every other ``add_middleware`` / install hook
    so this middleware ends up OUTERMOST in the Starlette stack
    (Starlette stacks middleware in reverse install order — last
    install runs first per request). Outermost is required: the
    fast-path must run before any body-size / body-depth / CORS work
    so probe hits never pay for header iteration on those layers.
    """
    app.add_middleware(ProbeFastPathMiddleware)
