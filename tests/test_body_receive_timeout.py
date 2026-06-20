# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the body-receive idle timeout (F-072 slow-DoS gate).

Pre-fix: the server held a POST connection open indefinitely when the
client shipped only the HTTP headers (with an honest, in-range
``Content-Length``) and then sent zero body bytes. A repro hit ≥30 s
of "no response" before the test gave up. Multiplied across a fleet of
slow-body sockets this pins the worker pool indefinitely — a classic
slowloris pattern against the body channel instead of the header
channel.

Post-fix: ``RequestBodyLimitMiddleware`` wraps each ``receive()``
ASGI call in ``asyncio.wait_for`` until the body is fully on the wire.
When no body bytes arrive within
``ServerConfig.body_receive_timeout_seconds`` (default 15 s, env
``RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS``, 0 disables), the
middleware raises ``_BodyReceiveTimeoutError``, intercepts FastAPI's
generic body-parse 400 in ``guarded_send``, and rewrites the response
to a clean 408 with an OpenAI-shaped JSON envelope.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_config():
    """Each test starts from a clean ServerConfig singleton so a
    previous test that monkey-patched the timeout doesn't leak its
    value into the next case."""
    from vllm_mlx.config.server_config import reset_config

    reset_config()
    yield
    reset_config()


def _build_app() -> FastAPI:
    """Mirror the production app's middleware wiring: a minimal
    FastAPI app + the body-size middleware + a tiny POST handler at
    a guarded path. Keeps the slow-body gate the only moving piece."""
    from vllm_mlx.middleware.body_size import install_request_body_limit_middleware

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _echo(payload: dict):
        return {"received_bytes": len(json.dumps(payload).encode("utf-8"))}

    install_request_body_limit_middleware(app)
    return app


def test_default_serverconfig_carries_body_receive_timeout():
    """Catch a regression that removes the dataclass field. The
    middleware reads ``ServerConfig.body_receive_timeout_seconds``
    per request — without the field the gate silently no-ops."""
    from vllm_mlx.config.server_config import ServerConfig, get_config

    # Default value is the documented 15 s.
    assert ServerConfig().body_receive_timeout_seconds == 15.0
    # Singleton reflects the same default.
    assert get_config().body_receive_timeout_seconds == 15.0


def test_resolve_body_receive_timeout_clamps_and_falls_back():
    """Pin the two defensive branches in
    :func:`_resolve_body_receive_timeout`. Codex r1 NIT on PR #732
    spotted that the previous docstring conflated two distinct paths:

    * negative numeric → ``max(0.0, …)`` clamps it to 0 (gate
      disabled). This is the legitimate operator escape hatch — set
      ``body_receive_timeout_seconds = -1`` in a fixture to disable.
    * non-numeric / coercion failure → falls back to the documented
      15 s default (NOT 0). Silently disabling the gate on a typo
      would mask the real cause — the resolver mirrors the
      :func:`_resolve_limit` "sane default beats unlimited" choice.
    """
    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import _resolve_body_receive_timeout

    # Negative numeric: clamp to 0 (gate disabled).
    get_config().body_receive_timeout_seconds = -7.0
    assert _resolve_body_receive_timeout() == 0.0

    # Positive numeric: pass through unchanged.
    get_config().body_receive_timeout_seconds = 25.0
    assert _resolve_body_receive_timeout() == 25.0

    # Non-numeric (str — what an unmonkey-patched buggy fixture might
    # inject): coerce-fail falls back to the documented 15 s default,
    # NOT silently disables the gate. We coerce-through ``object()``
    # to make sure the test fails clearly if a future refactor
    # changes the defensive branch to ``except (ValueError, TypeError):
    # return 0`` — a silent-disable regression would skip the gate
    # under a fixture typo, exactly the F-072 surface this test pins.
    get_config().body_receive_timeout_seconds = "not-a-number"  # type: ignore[assignment]
    assert _resolve_body_receive_timeout() == 15.0


def test_normal_post_under_timeout_passes_through():
    """Positive control: a normal POST whose body arrives in one
    receive frame (TestClient's behaviour) MUST reach the handler
    and return its response. A regression that fired the timeout
    against well-behaved clients would 408 every request."""
    from vllm_mlx.config.server_config import get_config

    get_config().body_receive_timeout_seconds = 10.0

    app = _build_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["received_bytes"] > 0


def test_slow_body_receive_emits_408():
    """F-072 fix: when ``receive()`` stalls past the configured
    timeout, the middleware MUST short-circuit with a clean 408 (not
    a 400 "error parsing the body", which is FastAPI's default when
    the body-reader sees an exception). The 408 carries the
    documented OpenAI-shaped envelope so SDKs handle it predictably.

    We exercise the middleware against a hand-rolled ASGI stub
    because TestClient buffers the body in-memory before invoking
    the app — there's no socket-level "slow ship" hook. Driving the
    middleware directly lets us pin the timeout firing path without
    a real network."""
    import asyncio

    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().body_receive_timeout_seconds = 0.1

    async def _inner_app(scope, receive, send):
        # Realistic handler: try to drain the body. The wrap should
        # raise our timeout sentinel before any byte arrives.
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"unreachable"})

    middleware = RequestBodyLimitMiddleware(_inner_app)

    async def receive():
        # Honest slow-DoS shape: pretend the client opened the socket,
        # sent headers, and then went quiet. Sleep longer than the
        # configured timeout so the wait_for boundary fires.
        await asyncio.sleep(5.0)
        return {"type": "http.request", "body": b"x", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", b"1000000")],
    }

    asyncio.run(middleware(scope, receive, send))

    # First send frame is the 408 start; second is the body.
    assert len(sent) == 2, sent
    start, body = sent
    assert start["type"] == "http.response.start"
    assert start["status"] == 408
    header_map = {k: v for (k, v) in start["headers"]}
    assert header_map.get(b"content-type") == b"application/json"
    # Per RFC 9110 §15.5.9: 408 responses include Connection: close.
    assert header_map.get(b"connection") == b"close"

    assert body["type"] == "http.response.body"
    payload = json.loads(body["body"])
    err = payload["error"]
    assert err["code"] == "request_timeout"
    assert err["type"] == "invalid_request_error"
    assert "no body bytes received" in err["message"]


def test_timeout_disabled_when_zero():
    """``RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS=0`` (the documented
    escape hatch) must disable the gate. Operators with their own
    upstream slow-DoS defenses (e.g. an nginx ``client_body_timeout``)
    rely on this — otherwise the gate would double-fire and confuse
    log analysis."""
    import asyncio

    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().body_receive_timeout_seconds = 0.0
    # Cap also off so we don't even hit the bounded-receive wrapper
    # for the wrong reason — we want to assert the no-overhead fast
    # path is taken.
    get_config().max_request_bytes = 0

    seen = []

    async def _inner_app(scope, receive, send):
        msg = await receive()
        seen.append(msg)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestBodyLimitMiddleware(_inner_app)

    body = b'{"messages":[]}'

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", str(len(body)).encode())],
    }

    asyncio.run(middleware(scope, receive, send))
    # Handler reached, 200 returned.
    assert any(m["type"] == "http.response.start" and m["status"] == 200 for m in sent)


def test_timeout_does_not_truncate_long_running_response():
    """Once the request body is fully delivered (``more_body=False``),
    the per-message timeout MUST switch off. Otherwise legitimate
    long-running inference (a 30 s generation that ships nothing
    upstream for many seconds) would be torn down by the slow-DoS
    timer the moment the engine pauses for a beat between chunks."""
    import asyncio

    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().body_receive_timeout_seconds = 0.05

    async def _inner_app(scope, receive, send):
        # Drain the body first (one receive call, body_complete flips).
        await receive()
        # Then "pause" longer than the receive timeout — modelling a
        # slow generation step. The middleware MUST NOT inject a
        # timeout exception during this wait.
        await asyncio.sleep(0.2)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"done"})

    middleware = RequestBodyLimitMiddleware(_inner_app)

    body = b'{"hello":"world"}'

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", str(len(body)).encode())],
    }

    asyncio.run(middleware(scope, receive, send))
    # We MUST see the inner app's 200 + "done", not a synthesised 408.
    assert any(
        m.get("status") == 200 for m in sent if m["type"] == "http.response.start"
    )
    bodies = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert b"done" in bodies


def test_timeout_only_guards_listed_path_prefixes():
    """The middleware scopes its receive wrapper to ``/v1/*`` /
    ``/internal/*`` / ``/anthropic/*``. A health-check POST to
    ``/healthz`` (outside the guarded prefixes) MUST flow through
    even when the body sits on receive() longer than the configured
    timeout — no slow-DoS gate, no overhead.

    Codex r1 BLOCKING on PR #732 — the original version of this
    test had ``receive()`` return immediately, so the test would
    have passed even if a regression dragged ``/healthz`` into the
    timeout-guarded path. Fixed by stalling receive() *past* the
    configured timeout: a wrongly guarded /healthz would now 408 and
    fail the assertions below."""
    import asyncio

    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

    receive_timeout = 0.05
    receive_stall = receive_timeout * 6  # comfortably past the gate
    get_config().body_receive_timeout_seconds = receive_timeout

    handler_called = {"value": False}

    async def _inner_app(scope, receive, send):
        # Stall longer than the configured timeout. If the middleware
        # were wrapping ``receive`` for this path, the stall would
        # trip ``asyncio.wait_for`` and synthesise a 408 here — the
        # handler would never run.
        await receive()
        handler_called["value"] = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestBodyLimitMiddleware(_inner_app)

    async def receive():
        # Stall ``receive_stall`` seconds — longer than the gate's
        # ``receive_timeout``. The unguarded path short-circuits BEFORE
        # ``bounded_receive`` wraps us, so this sleep is harmless.
        # A regression that wrongly guarded ``/healthz`` would see
        # this stall fire the slow-DoS gate and 408 the request, and
        # the assertions below would catch it.
        await asyncio.sleep(receive_stall)
        return {"type": "http.request", "body": b"{}", "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/healthz",
        "headers": [],
    }
    asyncio.run(middleware(scope, receive, send))
    # Inner handler reached — the gate did NOT bounce /healthz.
    assert handler_called["value"] is True
    # ONLY the inner handler's 200 was sent — no 408 from a
    # regressively guarded path.
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 200
    # And NOT a 408 — a regression that guards /healthz would emit
    # the slow-body timeout response instead.
    assert not any(m.get("status") == 408 for m in sent)


def test_timeout_path_does_not_double_send_when_body_size_also_trips():
    """Belt-and-braces: a request that has BOTH an over-cap streamed
    body AND a slow ship MUST emit EXACTLY ONE terminal response
    (not a 413 + 408 sandwich). The middleware uses
    ``downstream_completed_response`` to gate the rewrite path so two
    boundary catches can't both flush a response.

    We deliberately keep the advertised ``Content-Length`` under the
    cap so the fast-path 413 doesn't fire, leaving the slow-ship
    timeout as the load-bearing gate. Then we ship 8 KiB of body
    that would have tripped the streaming 413 path had the timeout
    not won the race — asserting the boundary catches don't both
    fire."""
    import asyncio

    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().body_receive_timeout_seconds = 0.05
    get_config().max_request_bytes = 1024

    async def _inner_app(scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"x"})

    middleware = RequestBodyLimitMiddleware(_inner_app)

    async def receive():
        # Slow ship — timeout fires before the body lands. If it
        # finally landed, the 8 KiB body would also exceed the 1 KiB
        # cap, so the streaming-413 path is armed too.
        await asyncio.sleep(1.0)
        return {"type": "http.request", "body": b"x" * 8192, "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    # Advertised CL UNDER the cap so the fast-path 413 doesn't
    # short-circuit — we want both boundary catches to be reachable.
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-length", b"500")],
    }

    asyncio.run(middleware(scope, receive, send))

    # Exactly one response.start frame, exactly one terminal body.
    starts = [m for m in sent if m["type"] == "http.response.start"]
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert len(starts) == 1, sent
    assert len(bodies) == 1, sent
    # And it's the 408 — the slow body trips first because the
    # wait_for boundary fires before any bytes (and thus before any
    # ``total["bytes"] > limit`` check).
    assert starts[0]["status"] == 408
