# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the global request-body size cap.

Defends against the F-007 DoS pattern (rapid-desktop#273 / #463):
a 10–100 MB JSON body silently ran ~60–90 s of useless prefill on a
27B alias before the client gave up. The cap MUST reject oversized
bodies at the ASGI layer with HTTP 413 *before* FastAPI JSON parsing
runs.

The audio variant has its own multipart-aware cap with a higher
budget — these tests assert the JSON-route cap leaves that path alone.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

# NIT from codex round 1: ``RequestBodyLimitMiddleware`` is pure ASGI
# logic with no mlx-lm import path — ``rapid_mlx.middleware.body_size``
# only pulls in ``rapid_mlx.config.server_config`` (a dataclass) and
# stdlib. Empirically verified: the module loads under an import hook
# that blocks ``mlx*``. So we no longer ``importorskip("mlx.core")``
# here — the security gate gets exercised on every runner, including
# the minimal Linux pr-validate CI matrix.


@pytest.fixture(autouse=True)
def _isolate_config():
    """Each test starts from a clean ServerConfig singleton so the
    8 MiB default isn't carried over from a previous test that
    monkey-patched ``max_request_bytes``."""
    from rapid_mlx.config.server_config import get_config, reset_config

    reset_config()
    original_timeout = get_config().body_receive_timeout_seconds
    yield
    # Restore default for the next module.
    get_config().max_request_bytes = 8 * 1024 * 1024
    get_config().body_receive_timeout_seconds = original_timeout


def _build_app() -> FastAPI:
    """A minimal FastAPI app wired exactly like ``rapid_mlx.server`` —
    the request-body middleware plus a tiny POST handler under one of
    the guarded path prefixes. Keeps the body-size guard the only
    moving piece under test."""
    from rapid_mlx.middleware.body_size import install_request_body_limit_middleware

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _echo(payload: dict):
        return {"received_bytes": len(json.dumps(payload).encode("utf-8"))}

    @app.post("/v1/embeddings")
    async def _emb(payload: dict):  # noqa: ARG001
        return {"ok": True}

    @app.post("/healthz")  # outside the guarded prefix
    async def _health(payload: dict):  # noqa: ARG001
        return {"ok": True}

    install_request_body_limit_middleware(app)
    return app


def test_honest_content_length_over_cap_returns_413():
    """A request advertising Content-Length > cap must be rejected
    with HTTP 413 in microseconds. No JSON parsing, no Pydantic
    construction.

    This is the F-007 happy-path: a malicious client advertises a
    huge body and the server bounces it immediately. The legacy
    behavior (60-90 s prefill) only happens once the body is read,
    so the proof-by-receive-tracer in
    ``test_body_never_reaches_handler_on_413`` below covers the
    "did we actually read the bytes" property.
    """
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 1024  # 1 KiB cap

    app = _build_app()
    client = TestClient(app)

    # 4 KiB body well over the 1 KiB cap.
    big_payload = {"data": "x" * 4096}
    resp = client.post("/v1/chat/completions", json=big_payload)

    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "request_too_large"
    assert "1024" in body["error"]["message"]


def test_body_just_under_cap_passes_through():
    """Positive control: a body under the cap must reach the handler
    and return its normal response. Catches regressions where the
    middleware is overly aggressive (e.g. off-by-one on the boundary,
    or rejection of every POST regardless of size)."""
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 16 * 1024  # 16 KiB cap

    app = _build_app()
    client = TestClient(app)

    small_payload = {"messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=small_payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["received_bytes"] > 0


def test_disabled_when_cap_is_zero():
    """``--max-request-bytes 0`` (the documented escape hatch)
    must disable the cap entirely. Operators with their own DoS
    controls upstream rely on this."""
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 0

    app = _build_app()
    client = TestClient(app)

    # 256 KiB payload — small enough that the test stays fast, but
    # well above any reasonable accidental default we might land on.
    big_payload = {"data": "y" * (256 * 1024)}
    resp = client.post("/v1/chat/completions", json=big_payload)
    assert resp.status_code == 200, resp.text


def test_unguarded_path_is_not_capped():
    """The cap only applies to ``/v1/*`` / ``/internal/*`` /
    ``/anthropic/*`` paths. Probes like ``/healthz``, ``/metrics``,
    or anything else outside those prefixes must pass through even
    if their body would exceed the cap. Without this scoping, a
    health-check tool that POSTs JSON would 413 spuriously."""
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 512  # tiny cap

    app = _build_app()
    client = TestClient(app)

    big_payload = {"data": "z" * 4096}
    resp = client.post("/healthz", json=big_payload)
    assert resp.status_code == 200, resp.text


def test_audio_path_is_excluded_from_generic_cap():
    """The audio transcription endpoint has its own multipart-aware
    25 MB cap (see ``routes/audio.py``). The generic JSON cap MUST
    NOT short-circuit it — a legitimate 5 MB voice upload should
    flow through to the audio middleware, not get bounced by the
    8 MiB JSON cap at a wire-level honest-Content-Length check.

    We assert by ensuring our generic middleware does NOT respond
    to ``/v1/audio/transcriptions`` even when the body advertises
    a length that would otherwise exceed our cap."""
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 1024

    app = _build_app()

    # Add the route AFTER the middleware so the middleware definitely
    # sees the path; assert the request reaches the handler instead
    # of being 413'd at our layer.
    @app.post("/v1/audio/transcriptions")
    async def _aud(payload: dict):  # noqa: ARG001
        return {"ok": True}

    client = TestClient(app)
    big_payload = {"data": "a" * 4096}
    resp = client.post("/v1/audio/transcriptions", json=big_payload)
    # Reached the handler — our cap did NOT apply.
    assert resp.status_code == 200, resp.text


def test_body_never_reaches_handler_on_413():
    """The load-bearing assertion: when the cap fires on an honest
    Content-Length, NO ``http.request`` message is ever drained
    from the ``receive`` channel. This is the empirical proof that
    we don't repeat the F-007 pattern where the body is read +
    JSON-parsed + tokenized before we bounce.

    A receive-tracer counts how often the inner ASGI app sees a
    body message. If the middleware does its job, that count is 0.
    """
    from rapid_mlx.config.server_config import get_config
    from rapid_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().max_request_bytes = 1024

    receive_calls = []

    async def _inner_app(scope, receive, send):
        # If the middleware does its job, this never runs.
        while True:
            msg = await receive()
            receive_calls.append(msg.get("type"))
            if not msg.get("more_body"):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestBodyLimitMiddleware(_inner_app)

    body = b"X" * 16384  # 16 KiB, far over the 1 KiB cap

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    asyncio.run(middleware(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413, sent
    # The inner app was never reached — no body parsing happened.
    assert receive_calls == [], (
        f"middleware let body parsing begin (receive saw {receive_calls}) — "
        "guard regressed to a handler-level check"
    )


def test_chunked_streaming_body_aborts_mid_stream():
    """The chunked / no-Content-Length / lying-Content-Length attack
    path. The client omits Content-Length and streams chunks past
    the cap. Downstream reads MUST stop once the running total exceeds
    the cap; the middleware discards the remaining chunks before its 413.

    Without this guard, a Transfer-Encoding: chunked client could
    stream gigabytes before any byte-count gate fired."""
    from rapid_mlx.config.server_config import get_config
    from rapid_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().max_request_bytes = 1024  # 1 KiB cap

    downstream_chunks = []

    async def _draining_inner(scope, receive, send):
        while True:
            msg = await receive()
            downstream_chunks.append(msg)
            if msg.get("type") == "http.disconnect":
                return
            if not msg.get("more_body"):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestBodyLimitMiddleware(_draining_inner)

    total_chunks = 16
    chunk_size = 256
    received_count = {"n": 0}

    async def receive():
        i = received_count["n"]
        received_count["n"] += 1
        if i >= total_chunks:
            return {"type": "http.request", "body": b"", "more_body": False}
        more = i < total_chunks - 1
        return {"type": "http.request", "body": b"X" * chunk_size, "more_body": more}

    sent = []

    async def send(msg):
        sent.append(msg)

    # No Content-Length — chunked transfer encoding scope.
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"transfer-encoding", b"chunked"),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    asyncio.run(middleware(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413, sent

    # Only the four in-range chunks reach downstream; chunk five trips
    # the guard, and the remaining chunks are discarded by the middleware.
    assert len(downstream_chunks) == 4
    assert received_count["n"] == total_chunks
    assert (b"connection", b"close") not in start["headers"]


def test_no_double_response_when_handler_already_sent_headers():
    """Codex round-1 BLOCKING #2 regression: if a downstream handler
    has already emitted ``http.response.start`` BEFORE the cap trips
    (e.g. a streaming handler that reads body lazily after first
    flushing 200 OK headers), the middleware must NOT inject a 413
    on top of the in-flight response — that would corrupt the wire.

    Test design: build an inner app that emits ``http.response.start``
    BEFORE reading the body, then reads chunks. The cap trips inside
    the read loop; the middleware catches ``_BodyTooLargeError`` and must
    silently let the response complete (logged warning, no double 413).
    """
    from rapid_mlx.config.server_config import get_config
    from rapid_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().max_request_bytes = 1024

    async def _start_then_read_inner(scope, receive, send):
        # Emit response start BEFORE reading body — this is the
        # codex-flagged window where a 413 from the middleware would
        # collide with an in-flight 200.
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        # Now read body — middleware will raise _BodyTooLargeError here.
        while True:
            msg = await receive()
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestBodyLimitMiddleware(_start_then_read_inner)

    body = b"X" * 4096

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    sent = []

    async def send(msg):
        sent.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"transfer-encoding", b"chunked"),
        ],
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
    }

    asyncio.run(middleware(scope, receive, send))

    # Exactly one http.response.start frame on the wire — no 413
    # injected after the 200 went out. This is the load-bearing
    # assertion that the codex round-1 BLOCKING #2 was about.
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1, sent
    assert starts[0]["status"] == 200, (
        "middleware injected a 413 on top of an in-flight 200 — "
        "double-response regression"
    )
    # codex round-2 BLOCKING #2: the test must also assert the
    # response stream was properly terminated. Without the close-frame
    # logic added in round 2, the middleware would silently return
    # leaving the client hanging on a Content-Length / chunked
    # trailer that never arrived.
    terminal_bodies = [
        m
        for m in sent
        if m["type"] == "http.response.body" and not m.get("more_body", False)
    ]
    assert len(terminal_bodies) == 1, (
        f"response stream not terminated after cap trip — got {sent}"
    )


def test_get_request_is_not_capped():
    """The middleware skips GET (and other body-less methods) entirely
    so it adds no latency to ``/v1/models``, ``/v1/health/...``,
    etc. A spuriously large GET shouldn't be possible in HTTP/1.1
    anyway, but we still want zero per-request overhead on read
    paths."""
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 1  # absurdly tiny — every body would fail

    app = _build_app()

    @app.get("/v1/models")
    async def _list():
        return {"data": []}

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200


def test_oversized_body_returns_413_before_auth_check():
    """Codex r3 F3: pin the deliberate ordering choice — the body-size
    cap runs at the ASGI layer (``add_middleware`` is the outermost
    Starlette middleware stack), while bearer-token auth is a FastAPI
    ``Depends(verify_api_key)`` at the route level. So an oversized
    unauthenticated POST is rejected with 413, NOT 401 — denying
    unauthenticated clients the cap-probing reconnaissance channel.

    This is the deliberate design (reject DoS payloads cheapest, before
    any per-request work including auth). The test exists to make the
    ordering load-bearing — anyone moving the body cap *behind* the
    auth dependency will fail it and have to think about it.
    """
    from rapid_mlx.config.server_config import get_config
    from rapid_mlx.middleware.body_size import install_request_body_limit_middleware

    get_config().max_request_bytes = 1024  # 1 KiB cap

    app = FastAPI()

    auth_calls = {"n": 0}

    async def _fake_auth():
        # Simulates a bearer check that would 401 if reached. The
        # 413 must trip before we ever land here.
        auth_calls["n"] += 1
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Missing bearer")

    from fastapi import Depends

    @app.post("/v1/chat/completions", dependencies=[Depends(_fake_auth)])
    async def _gated(payload: dict):  # noqa: ARG001
        return {"ok": True}

    install_request_body_limit_middleware(app)
    client = TestClient(app)

    oversized = {"messages": [{"role": "user", "content": "x" * 4096}]}
    resp = client.post(
        "/v1/chat/completions",
        json=oversized,
        # No Authorization header — auth would normally 401.
    )
    assert resp.status_code == 413, (
        f"expected 413 (body cap before auth), got {resp.status_code}"
    )
    assert auth_calls["n"] == 0, (
        "auth dependency ran before the body cap — ordering regressed"
    )


def test_cli_flag_overrides_config_default():
    """The ``--max-request-bytes`` flag is wired through
    ``rapid_mlx.server._max_request_bytes`` and ``_sync_config`` into
    ``ServerConfig.max_request_bytes``. We don't exercise the full
    CLI here (too heavy — needs model load) but we do assert the
    wiring: writing the module global and calling ``_sync_config``
    propagates the value to the config singleton the middleware
    reads."""
    import rapid_mlx.server as server_mod
    from rapid_mlx.config.server_config import get_config

    original = server_mod._max_request_bytes
    try:
        server_mod._max_request_bytes = 12345
        server_mod._sync_config()
        assert get_config().max_request_bytes == 12345
    finally:
        server_mod._max_request_bytes = original
        server_mod._sync_config()


@pytest.mark.parametrize("chunked", [False, True], ids=["content-length", "chunked"])
def test_oversized_upload_receives_413_over_socket(chunked):
    """A client still uploading must finish sendall and read the JSON rejection."""
    import uvicorn

    from rapid_mlx.config.server_config import get_config
    from rapid_mlx.middleware.body_size import install_request_body_limit_middleware

    get_config().max_request_bytes = 64 * 1024
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _consume(request: Request):
        return {"received_bytes": len(await request.body())}

    install_request_body_limit_middleware(app)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, http="h11", log_level="warning")
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    address = listener.getsockname()
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    try:
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive(), "server failed to start"
            assert time.monotonic() < deadline, "server startup timed out"
            time.sleep(0.01)
        body = b"x" * (4 * 1024 * 1024)
        framing = (
            "Transfer-Encoding: chunked" if chunked else f"Content-Length: {len(body)}"
        )
        with socket.create_connection(address, timeout=30) as client:
            client.sendall(
                (
                    "POST /v1/chat/completions HTTP/1.1\r\n"
                    "Host: localhost\r\n"
                    "Connection: close\r\n"
                    f"{framing}\r\n\r\n"
                ).encode("ascii")
            )
            if chunked:
                for offset in range(0, len(body), 16 * 1024):
                    chunk = body[offset : offset + 16 * 1024]
                    client.sendall(f"{len(chunk):x}\r\n".encode("ascii"))
                    client.sendall(chunk)
                    client.sendall(b"\r\n")
                client.sendall(b"0\r\n\r\n")
            else:
                client.sendall(body)
            response = bytearray()
            while data := client.recv(64 * 1024):
                response.extend(data)
        headers, payload = bytes(response).split(b"\r\n\r\n", 1)
        assert headers.split(b"\r\n", 1)[0].startswith(b"HTTP/1.1 413 ")
        error = json.loads(payload)["error"]
        assert error["code"] == "request_too_large"
        assert "65536" in error["message"]
        assert "--max-request-bytes" in error["message"]
    finally:
        server.should_exit = True
        if thread.ident is not None:
            thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "server thread did not shut down"


def _run_rejection(receive, *, headers=(), app=None):
    from rapid_mlx.config.server_config import get_config
    from rapid_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().max_request_bytes = 1024

    async def unexpected_app(scope, receive, send):
        pytest.fail("fast rejection reached downstream")

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(
        RequestBodyLimitMiddleware(app or unexpected_app)(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": headers,
            },
            receive,
            send,
        )
    )
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "request_too_large"
    return sent[0]["headers"]


def test_expect_continue_rejects_without_receiving():
    async def receive():
        pytest.fail("receive would invite the oversized upload")

    headers = _run_rejection(
        receive,
        headers=[(b"Content-Length", b"4096"), (b"ExPeCt", b"100-CoNtInUe")],
    )
    assert (b"connection", b"close") in headers


def test_drain_idle_timeout_closes_connection():
    from rapid_mlx.config.server_config import get_config

    get_config().body_receive_timeout_seconds = 0.001

    async def receive():
        await asyncio.Event().wait()

    headers = _run_rejection(receive, headers=[(b"content-length", b"4096")])
    assert (b"connection", b"close") in headers


@pytest.mark.parametrize("busy", [False, True], ids=["waiting", "continuous-chunks"])
def test_drain_total_budget_closes_connection(monkeypatch, busy):
    import rapid_mlx.middleware.body_size as body_size

    monkeypatch.setattr(body_size, "_DRAIN_TOTAL_SECONDS", 0.001)
    calls = []

    async def receive():
        calls.append(True)
        if busy:
            # Complete a receive after the deadline without yielding: the
            # next loop iteration must enforce the wall-clock budget too.
            time.sleep(0.002)
        else:
            await asyncio.sleep(1)
        return {"type": "http.request", "body": b"x", "more_body": True}

    headers = _run_rejection(receive, headers=[(b"content-length", b"4096")])
    assert calls
    assert (b"connection", b"close") in headers


@pytest.mark.parametrize("slow_path", [False, True])
def test_disconnect_during_drain_closes_connection(slow_path):
    calls = []

    async def receive():
        calls.append(True)
        if slow_path and len(calls) == 1:
            return {"type": "http.request", "body": b"x" * 4096, "more_body": True}
        return {"type": "http.disconnect"}

    async def app(scope, receive, send):
        await receive()
        pytest.fail("oversized chunk reached downstream")

    headers = _run_rejection(
        receive,
        headers=[] if slow_path else [(b"content-length", b"4096")],
        app=app if slow_path else None,
    )
    assert len(calls) == (2 if slow_path else 1)
    assert (b"connection", b"close") in headers


def test_final_oversized_chunk_needs_no_further_receive():
    calls = []

    async def receive():
        calls.append(True)
        assert len(calls) == 1, "receive called after the body completed"
        return {"type": "http.request", "body": b"x" * 4096, "more_body": False}

    async def app(scope, receive, send):
        await receive()
        pytest.fail("oversized chunk reached downstream")

    headers = _run_rejection(receive, app=app)
    assert len(calls) == 1
    assert (b"connection", b"close") not in headers


@pytest.mark.parametrize(
    ("receive_timeout", "expected_idle"),
    [(0, 2.0), (-1, 2.0), (0.5, 0.5), (2.0, 2.0), (15.0, 2.0)],
)
def test_drain_uses_short_grace_period(monkeypatch, receive_timeout, expected_idle):
    import rapid_mlx.middleware.body_size as body_size
    from rapid_mlx.config.server_config import get_config

    get_config().body_receive_timeout_seconds = receive_timeout
    drain_calls = []

    async def drain(receive, *, idle_timeout, total_budget):
        drain_calls.append((idle_timeout, total_budget))
        return False

    monkeypatch.setattr(body_size, "_drain_request_body", drain)

    async def receive():
        pytest.fail("drain should be replaced")

    headers = _run_rejection(receive, headers=[(b"content-length", b"4096")])
    assert drain_calls == [(expected_idle, 10.0)]
    assert (b"connection", b"close") in headers


def test_disabled_receive_timeout_uses_bounded_drain_max(monkeypatch):
    import rapid_mlx.middleware.body_size as body_size
    from rapid_mlx.config.server_config import get_config

    get_config().body_receive_timeout_seconds = 0
    monkeypatch.setattr(body_size, "_DRAIN_IDLE_SECONDS_MAX", 0.001)

    async def receive():
        await asyncio.Event().wait()

    headers = _run_rejection(receive, headers=[(b"content-length", b"4096")])
    assert (b"connection", b"close") in headers


@pytest.mark.parametrize("slow_path", [False, True])
def test_full_drain_capacity_rejects_without_receiving(monkeypatch, slow_path):
    import rapid_mlx.middleware.body_size as body_size

    monkeypatch.setattr(body_size, "_active_drains", body_size._MAX_CONCURRENT_DRAINS)
    calls = []

    async def receive():
        calls.append(True)
        assert slow_path and len(calls) == 1, "rejection must skip draining"
        return {"type": "http.request", "body": b"x" * 4096, "more_body": True}

    async def app(scope, receive, send):
        await receive()
        pytest.fail("oversized chunk reached downstream")

    headers = _run_rejection(
        receive,
        headers=[] if slow_path else [(b"content-length", b"4096")],
        app=app if slow_path else None,
    )
    assert len(calls) == int(slow_path)
    assert (b"connection", b"close") in headers
    assert body_size._active_drains == body_size._MAX_CONCURRENT_DRAINS


@pytest.mark.parametrize("outcome", ["complete", "exception", "timeout"])
def test_drain_capacity_is_released(outcome):
    import rapid_mlx.middleware.body_size as body_size

    async def receive():
        assert body_size._active_drains == 1
        if outcome == "exception":
            raise RuntimeError("receive failed")
        if outcome == "timeout":
            await asyncio.Event().wait()
        return {"type": "http.request", "body": b"done", "more_body": False}

    async def run():
        assert body_size._active_drains == 0
        if outcome == "exception":
            with pytest.raises(RuntimeError, match="receive failed"):
                await body_size._drain_if_capacity(
                    receive, idle_timeout=0.001, total_budget=1
                )
        else:
            drained = await body_size._drain_if_capacity(
                receive, idle_timeout=0.001, total_budget=1
            )
            assert drained is (outcome == "complete")
        assert body_size._active_drains == 0

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True], ids=["complete", "cancel"])
def test_concurrent_drains_are_bounded(cancel):
    import rapid_mlx.middleware.body_size as body_size
    from rapid_mlx.config.server_config import get_config

    get_config().max_request_bytes = 1024

    async def run():
        release = asyncio.Event()
        all_receiving = asyncio.Event()
        receive_calls = 0

        async def stalled_receive():
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == body_size._MAX_CONCURRENT_DRAINS:
                all_receiving.set()
            await release.wait()
            return {"type": "http.request", "body": b"done", "more_body": False}

        async def app(scope, receive, send):
            pytest.fail("fast rejection reached downstream")

        async def reject(receive):
            sent = []

            async def send(message):
                sent.append(message)

            await body_size.RequestBodyLimitMiddleware(app)(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "headers": [(b"content-length", b"4096")],
                },
                receive,
                send,
            )
            assert sent[0]["status"] == 413
            return sent[0]["headers"]

        assert body_size._active_drains == 0
        tasks = [
            asyncio.create_task(reject(stalled_receive))
            for _ in range(body_size._MAX_CONCURRENT_DRAINS)
        ]
        extra_receive_calls = 0

        async def extra_receive():
            nonlocal extra_receive_calls
            extra_receive_calls += 1
            await release.wait()

        try:
            await asyncio.wait_for(all_receiving.wait(), timeout=1)
            assert body_size._active_drains == body_size._MAX_CONCURRENT_DRAINS
            assert all(not task.done() for task in tasks)
            headers = await asyncio.wait_for(reject(extra_receive), timeout=0.1)
            assert (b"connection", b"close") in headers
            assert extra_receive_calls == 0
        finally:
            if cancel:
                for task in tasks:
                    task.cancel()
            else:
                release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
        assert body_size._active_drains == 0
        assert receive_calls == body_size._MAX_CONCURRENT_DRAINS
        if cancel:
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
        else:
            assert all(isinstance(result, list) for result in results)
            assert all((b"connection", b"close") not in result for result in results)

    asyncio.run(run())
