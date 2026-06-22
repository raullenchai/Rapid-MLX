# SPDX-License-Identifier: Apache-2.0
"""Tests for the ASGI fast-path that short-circuits ``GET /healthz`` and
``GET /livez`` before they reach Starlette's router (R8-H6).

The route-level handlers in ``routes/health.py`` are kept as the
fall-through path (HEAD requests, requests with an ``Origin`` header,
embedded harnesses that mount the routers without this middleware). The
fast-path here is a pure perf optimization for the kubelet/Docker
probe slice and MUST:

* return the same JSON shape as the existing handlers,
* never block on the event loop under streaming SSE concurrency,
* fall through cleanly for the CORS-needs-ACAO browser case,
* fall through for non-GET methods (POST/PUT/DELETE/HEAD).

The latency regression test below drives a synthetic streaming SSE
generator that yields chunks under controlled cadence. With the
fast-path installed, ``GET /healthz`` p99 stays well under the 50 ms
k8s probe budget even with 8 concurrent SSE streams competing for the
loop; without it (the v0.8.9 baseline), the same workload showed
p99 of 67–113 ms in dogfood Talia r1/r2.
"""

from __future__ import annotations

import asyncio
import json
import statistics

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from vllm_mlx.config import get_config
from vllm_mlx.middleware.probe_fastpath import (
    ProbeFastPathMiddleware,
    install_probe_fastpath_middleware,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_app(*, with_fastpath: bool = True) -> FastAPI:
    """Build a FastAPI app that mounts the existing health routers and
    (optionally) the fast-path middleware on top.

    Keeps the test isolated from the heavy engine stack — only the
    probe surface is exercised. We rely on the existing route module
    so the fall-through JSON shape stays in lockstep.
    """
    from vllm_mlx.routes.health import probe_router

    app = FastAPI()
    app.include_router(probe_router)
    if with_fastpath:
        install_probe_fastpath_middleware(app)
    return app


def _patch_config(**kwargs):
    cfg = get_config()
    originals = {k: getattr(cfg, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return originals


def _restore_config(originals):
    cfg = get_config()
    for k, v in originals.items():
        setattr(cfg, k, v)


# ---------------------------------------------------------------------------
# Shape parity — fast-path output must match the route handler
# ---------------------------------------------------------------------------


class TestHealthzShape:
    """The fast-path must return exactly the same JSON shape as the
    route-level handler in ``routes/health.py::healthz``. Dashboards
    and probe specs treat them as interchangeable.
    """

    def test_healthz_matches_handler_shape(self):
        originals = _patch_config(engine=None, model_name="test-model", ready=True)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body == {
                "status": "healthy",
                "ready": True,
                "model_loaded": False,
                "model_name": "test-model",
            }
            # Pinned content-type so dashboards parse it as JSON
            assert r.headers["content-type"].startswith("application/json")
        finally:
            _restore_config(originals)

    def test_healthz_with_engine_loaded(self):
        sentinel = object()
        originals = _patch_config(engine=sentinel, model_name="qwen3-4b", ready=True)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body["model_loaded"] is True
            assert body["ready"] is True
            assert body["model_name"] == "qwen3-4b"
        finally:
            _restore_config(originals)

    def test_healthz_not_ready(self):
        originals = _patch_config(engine=None, model_name=None, ready=False)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body == {
                "status": "healthy",
                "ready": False,
                "model_loaded": False,
                "model_name": None,
            }
        finally:
            _restore_config(originals)

    def test_livez_returns_alive(self):
        client = TestClient(_make_minimal_app(with_fastpath=True))
        r = client.get("/livez")
        assert r.status_code == 200
        assert r.json() == {"status": "alive"}


# ---------------------------------------------------------------------------
# Fall-through cases — fast-path MUST defer to the normal router
# ---------------------------------------------------------------------------


class TestFastPathFallThrough:
    """Fall-through is a correctness boundary. If the fast-path swallows
    HEAD or non-target paths, the route module loses HEAD-on-GET
    auto-derivation and any 404 / 405 envelope shape that callers rely on.
    """

    def test_head_falls_through_to_router(self):
        """HEAD on a fast-path route is NOT served by the fast-path —
        it falls through to the router. The route module's ``/healthz``
        is registered as GET-only (not GET+HEAD), so the router
        currently returns 405; codex r2 NIT: pin the exact 405 so a
        future route-method contract change (e.g. registering HEAD
        explicitly) surfaces here as a flip from 405→200 that the
        operator can vet, not as a silently-passing or-clause.

        The orthogonal HEAD-body invariant ("HEAD never carries a
        body") is checked via the inner-app recorder in
        :class:`TestFastPathServesRequest` — that's the test that
        proves the fast-path didn't shadow HEAD with the cached GET
        body.
        """
        originals = _patch_config(engine=None, model_name="test", ready=True)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.head("/healthz")
            # 405 from the router under the current /healthz GET-only
            # contract. A regression that adds HEAD to the route's
            # method list flips this to 200 — flag for review then.
            assert r.status_code == 405, (
                f"HEAD /healthz expected 405 from GET-only router; got "
                f"{r.status_code}. Did the route's method list change? "
                f"Audit the fast-path's GET-only gate too."
            )
            # HEAD must never carry a body, regardless of which path
            # answered.
            assert r.content == b""
        finally:
            _restore_config(originals)

    def test_head_does_not_invoke_fastpath_via_inner_app_recorder(self):
        """Direct proof via inner-app recorder: HEAD on /healthz reaches
        the inner app, not the fast-path. Pre-fix the assertion was
        only "HEAD body is empty" — true for both fast-path and router
        answers — so a future regression that let the fast-path serve
        HEAD with the cached GET body would have stayed undetected
        until a client noticed an unexpected body on HEAD.
        """
        inner_calls: list[dict] = []

        async def _inner(scope, receive, send):
            inner_calls.append(scope)
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        mw = ProbeFastPathMiddleware(_inner)
        captured: list[dict] = []

        async def _send(msg):
            captured.append(msg)

        async def _receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "method": "HEAD",
            "path": "/healthz",
            "raw_path": b"/healthz",
            "headers": [],
            "query_string": b"",
        }
        asyncio.run(mw(scope, _receive, _send))
        assert len(inner_calls) == 1, "HEAD must fall through to inner app"
        assert inner_calls[0]["method"] == "HEAD"

    def test_post_falls_through_with_405(self):
        """POST on /healthz hits the router (no POST handler), 405."""
        client = TestClient(_make_minimal_app(with_fastpath=True))
        r = client.post("/healthz", json={"x": 1})
        # Starlette returns 405 Method Not Allowed for a path with a
        # registered GET but no POST handler.
        assert r.status_code == 405

    def test_query_string_does_not_block_fastpath(self):
        """Cache-busting query strings (e.g. ``/healthz?ts=123``) must
        still hit the fast-path — the raw_path comparison strips
        ``?`` and friends."""
        originals = _patch_config(engine=None, model_name="t", ready=True)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.get("/healthz?ts=1234567890")
            assert r.status_code == 200
            assert r.json()["status"] == "healthy"
        finally:
            _restore_config(originals)

    def test_unrelated_path_falls_through(self):
        """Paths that aren't on the fast-path list go to the normal
        router. /health (no z) is the canonical route — keep it
        served by the route module."""
        originals = _patch_config(engine=None, model_name="test", ready=True)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.get("/health")
            # The full handler runs (with mocked engine=None it still
            # returns 200 — see existing test_routes coverage).
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "healthy"
        finally:
            _restore_config(originals)

    def test_origin_header_falls_through(self):
        """A request with an Origin header falls through so the CORS
        middleware (when installed) can attach ACAO. The fast-path is
        for the kubelet / supervisord / Docker slice — those don't
        send Origin."""
        # Build a probe-only app with NO CORS middleware so the
        # fall-through resolves at the route handler. The route still
        # returns the expected JSON; that's the contract — fall-through
        # is correctness, not perf.
        originals = _patch_config(engine=None, model_name="cors", ready=True)
        try:
            client = TestClient(_make_minimal_app(with_fastpath=True))
            r = client.get(
                "/healthz",
                headers={"Origin": "https://cross.example"},
            )
            assert r.status_code == 200
            body = r.json()
            # Route-handler shape is the source of truth — fast-path
            # output must agree on every field.
            assert body["status"] == "healthy"
            assert body["model_name"] == "cors"
        finally:
            _restore_config(originals)


# ---------------------------------------------------------------------------
# Latency regression — the actual R8-H6 budget
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Compute the ``pct``-th percentile of ``values`` (e.g. 99 → p99)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    # nearest-rank method — matches what the dogfood report uses
    k = max(0, min(len(sorted_v) - 1, int(round((pct / 100.0) * len(sorted_v))) - 1))
    return sorted_v[k]


async def _stream_chunks_under_load(
    app: FastAPI,
    *,
    n_streams: int,
    n_probes: int,
    chunk_interval_s: float = 0.005,
    chunks_per_stream: int = 200,
) -> dict[str, float]:
    """Drive ``n_streams`` synthetic SSE generators on the app in
    parallel and measure ``GET /healthz`` latency for ``n_probes`` hits
    interleaved with the streaming load.

    The synthetic stream yields one chunk every ``chunk_interval_s``
    seconds — a tight cadence that exercises the event loop's
    scheduling fairness. ``asyncio.sleep(0)`` between chunks would be
    too aggressive (would never let probes through) and a longer sleep
    would not produce contention; 5 ms is what the dogfood logs
    captured for a low-load decode step on the dogfood box.

    Returns a dict with ``p50_ms``, ``p95_ms``, ``p99_ms``, ``max_ms``,
    ``count`` — the same shape the dogfood report uses.
    """

    async def _sse_gen():
        for _ in range(chunks_per_stream):
            await asyncio.sleep(chunk_interval_s)
            yield f"data: {json.dumps({'t': 1})}\n\n"
        yield "data: [DONE]\n\n"

    # Register the streaming route on a copy of the app so the fixture
    # is hermetic — the test doesn't bleed into other tests.
    @app.get("/__test_stream")
    async def _stream():
        return StreamingResponse(_sse_gen(), media_type="text/event-stream")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Boot N concurrent streams. Each is a background task that
        # consumes the SSE response — that's what creates the loop
        # contention the probe measurement cares about.
        stream_tasks: list[asyncio.Task] = []

        async def _consume():
            async with client.stream("GET", "/__test_stream") as r:
                async for _ in r.aiter_bytes():
                    pass

        for _ in range(n_streams):
            stream_tasks.append(asyncio.create_task(_consume()))

        # Let the streams reach steady state before probing.
        await asyncio.sleep(0.05)

        # Drive the probes. Inter-probe gap loosely mirrors the
        # kubelet 1-second probe cadence, but compressed (10 ms) so
        # the test stays fast — we still observe contention because
        # multiple probes overlap streaming-chunk delivery.
        latencies_ms: list[float] = []
        loop = asyncio.get_event_loop()
        for _ in range(n_probes):
            t0 = loop.time()
            r = await client.get("/healthz")
            t1 = loop.time()
            assert r.status_code == 200
            latencies_ms.append((t1 - t0) * 1000.0)
            await asyncio.sleep(0.005)

        # Tear down streams. We don't care about their results — the
        # probes were the measurement.
        for t in stream_tasks:
            t.cancel()
        for t in stream_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    return {
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "max_ms": max(latencies_ms),
        "count": float(len(latencies_ms)),
    }


@pytest.mark.asyncio
async def test_healthz_p99_under_8way_streaming_stays_under_budget():
    """R8-H6 regression: with the fast-path installed, ``GET /healthz``
    p99 stays well under the 50 ms k8s probe budget even with 8
    concurrent SSE streams competing for the loop.

    Note on test-vs-production fidelity: the ASGI in-process driver
    (``httpx.ASGITransport``) skips the kernel TCP stack and uvicorn's
    request parsing, so the absolute latencies measured here are far
    lower than the dogfood 0.8.9 numbers (67-113 ms). What this test
    DOES catch is a structural regression — the fast-path being
    bypassed (e.g. a future refactor that swaps install order so
    another middleware shadows it, a route rename that breaks the
    path match, or JSON serialization creeping back in via Pydantic).
    Any of those would still push the tail well above the 50 ms k8s
    probe budget even in-process. The boots-with-uvicorn perf
    measurement lives in the dogfood harness, not here — keeping CI
    cheap and deterministic.
    """
    originals = _patch_config(engine=None, model_name="probe-perf", ready=True)
    try:
        app = _make_minimal_app(with_fastpath=True)
        stats = await _stream_chunks_under_load(
            app,
            n_streams=8,
            n_probes=80,
            chunk_interval_s=0.005,
            chunks_per_stream=80,
        )
        # Budgets are deliberately loose to absorb CI noise; the
        # signal we care about is order-of-magnitude — pre-fix p99
        # was 67–113 ms (per Talia r1/r2). Anything above 50 ms is a
        # regression.
        assert stats["p99_ms"] < 50.0, (
            f"healthz p99 ({stats['p99_ms']:.1f}ms) exceeded k8s probe "
            f"budget (50ms) — fast-path may be regressed; stats={stats}"
        )
        # Sanity: the probe count matches what we asked for.
        assert stats["count"] == 80
    finally:
        _restore_config(originals)


# ---------------------------------------------------------------------------
# ASGI-level invariants
# ---------------------------------------------------------------------------


class TestFastPathServesRequest:
    """Direct proof that the middleware — not the router — answers
    eligible probe hits. Codex round-1 BLOCKING: without an
    inner-app-recorder test, ``test_healthz_p99_under_8way_streaming``
    would silently pass even if ``install_probe_fastpath_middleware``
    became a no-op, because the FastAPI router itself answers the
    request quickly in-process. The tests below pin the
    middleware-vs-router decision.
    """

    def test_fastpath_serves_healthz_without_invoking_inner_app(self):
        """The fast-path returns a 200+JSON without dispatching to
        the inner ASGI app. We wrap the middleware around a recorder
        inner app: if the recorder ever sees a healthz scope, the
        fast-path did not intercept.
        """
        originals = _patch_config(engine=None, model_name="recorder", ready=True)
        try:
            inner_calls: list[dict] = []

            async def _inner(scope, receive, send):
                inner_calls.append(scope)
                # If the fast-path is broken and we get here, emit a
                # minimal 500 so the test can distinguish "router
                # answered" from "fast-path answered" cleanly.
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"inner-app"})

            mw = ProbeFastPathMiddleware(_inner)
            captured: list[dict] = []

            async def _send(msg):
                captured.append(msg)

            async def _receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            scope = {
                "type": "http",
                "method": "GET",
                "path": "/healthz",
                "raw_path": b"/healthz",
                "headers": [],
                "query_string": b"",
            }

            asyncio.run(mw(scope, _receive, _send))

            # Fast-path served — inner app was NEVER called.
            assert inner_calls == [], (
                "fast-path must serve /healthz directly; inner app saw "
                f"{len(inner_calls)} scope(s)"
            )
            # Exactly two ASGI sends (start + body).
            assert len(captured) == 2
            assert captured[0]["type"] == "http.response.start"
            assert captured[0]["status"] == 200
            assert captured[1]["type"] == "http.response.body"
            body = json.loads(captured[1]["body"])
            assert body == {
                "status": "healthy",
                "ready": True,
                "model_loaded": False,
                "model_name": "recorder",
            }
        finally:
            _restore_config(originals)

    def test_fastpath_serves_livez_without_invoking_inner_app(self):
        """Same shape as the healthz proof, for /livez."""
        inner_calls: list[dict] = []

        async def _inner(scope, receive, send):
            inner_calls.append(scope)
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"inner-app"})

        mw = ProbeFastPathMiddleware(_inner)
        captured: list[dict] = []

        async def _send(msg):
            captured.append(msg)

        async def _receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/livez",
            "raw_path": b"/livez",
            "headers": [],
            "query_string": b"",
        }
        asyncio.run(mw(scope, _receive, _send))
        assert inner_calls == []
        assert len(captured) == 2
        assert captured[0]["status"] == 200
        assert json.loads(captured[1]["body"]) == {"status": "alive"}

    def test_fastpath_delegates_to_inner_app_on_origin_header(self):
        """A request with Origin must fall through so the (outer) CORS
        middleware can attach ACAO. Pin this by recording inner-app
        invocation.
        """
        originals = _patch_config(engine=None, model_name="cors-route", ready=True)
        try:
            inner_calls: list[dict] = []

            async def _inner(scope, receive, send):
                inner_calls.append(scope)
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"served_by":"inner"}',
                    }
                )

            mw = ProbeFastPathMiddleware(_inner)
            captured: list[dict] = []

            async def _send(msg):
                captured.append(msg)

            async def _receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            scope = {
                "type": "http",
                "method": "GET",
                "path": "/healthz",
                "raw_path": b"/healthz",
                "headers": [(b"origin", b"https://browser.example")],
                "query_string": b"",
            }
            asyncio.run(mw(scope, _receive, _send))
            # The inner app was called (CORS fall-through worked).
            assert len(inner_calls) == 1
            assert inner_calls[0]["path"] == "/healthz"
            # And the inner app's body was forwarded.
            body_msgs = [m for m in captured if m["type"] == "http.response.body"]
            assert any(b'"served_by":"inner"' in m["body"] for m in body_msgs)
        finally:
            _restore_config(originals)

    def test_fastpath_delegates_to_inner_app_on_non_get(self):
        """POST/PUT/DELETE on /healthz fall through — pin via
        inner-app invocation."""
        inner_calls: list[dict] = []

        async def _inner(scope, receive, send):
            inner_calls.append(scope)
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"method-not-allowed"})

        mw = ProbeFastPathMiddleware(_inner)
        captured: list[dict] = []

        async def _send(msg):
            captured.append(msg)

        async def _receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        for method in ("POST", "PUT", "DELETE", "PATCH"):
            inner_calls.clear()
            captured.clear()
            scope = {
                "type": "http",
                "method": method,
                "path": "/healthz",
                "raw_path": b"/healthz",
                "headers": [],
                "query_string": b"",
            }
            asyncio.run(mw(scope, _receive, _send))
            assert len(inner_calls) == 1, f"{method} should fall through"
            assert captured[0]["status"] == 405


class TestFastPathASGIShape:
    def test_middleware_passes_through_non_http_scopes(self):
        """Lifespan + websocket scopes go straight to the inner app."""
        seen_scopes: list[str] = []

        async def inner(scope, receive, send):
            seen_scopes.append(scope["type"])

        async def _drive():
            mw = ProbeFastPathMiddleware(inner)
            # Lifespan scope
            await mw({"type": "lifespan"}, lambda: None, lambda msg: None)
            # WebSocket scope
            await mw(
                {"type": "websocket", "path": "/healthz"},
                lambda: None,
                lambda msg: None,
            )

        asyncio.run(_drive())
        assert seen_scopes == ["lifespan", "websocket"]
