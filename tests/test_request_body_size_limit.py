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

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Same skip rationale as test_audio_upload_size_limit.py — the audio
# import is light (no model load) but the middleware module itself
# pulls in ``vllm_mlx.config`` which transitively imports mlx-lm.
pytest.importorskip(
    "mlx.core",
    reason="middleware imports transitively pull in mlx; "
    "runs on Apple Silicon / dev, not minimal Linux CI runners",
)
pytest.importorskip(
    "mlx_lm",
    reason="middleware imports transitively pull in mlx_lm; "
    "runs on Apple Silicon / dev, not minimal Linux CI runners",
)


@pytest.fixture(autouse=True)
def _isolate_config():
    """Each test starts from a clean ServerConfig singleton so the
    8 MiB default isn't carried over from a previous test that
    monkey-patched ``max_request_bytes``."""
    from vllm_mlx.config.server_config import get_config, reset_config

    reset_config()
    yield
    # Restore default for the next module.
    get_config().max_request_bytes = 8 * 1024 * 1024


def _build_app() -> FastAPI:
    """A minimal FastAPI app wired exactly like ``vllm_mlx.server`` —
    the request-body middleware plus a tiny POST handler under one of
    the guarded path prefixes. Keeps the body-size guard the only
    moving piece under test."""
    from vllm_mlx.middleware.body_size import install_request_body_limit_middleware

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
    from vllm_mlx.config.server_config import get_config

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
    from vllm_mlx.config.server_config import get_config

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
    from vllm_mlx.config.server_config import get_config

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
    from vllm_mlx.config.server_config import get_config

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
    from vllm_mlx.config.server_config import get_config

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
    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

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
    the cap. The middleware MUST stop calling ``receive`` once the
    running total exceeds the cap and emit a 413.

    Without this guard, a Transfer-Encoding: chunked client could
    stream gigabytes before any byte-count gate fired."""
    from vllm_mlx.config.server_config import get_config
    from vllm_mlx.middleware.body_size import RequestBodyLimitMiddleware

    get_config().max_request_bytes = 1024  # 1 KiB cap

    async def _draining_inner(scope, receive, send):
        while True:
            msg = await receive()
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

    # The cap is 1024 B at 256 B/chunk → trip on chunk 5 (1280 > 1024).
    # We must NOT have drained all 16 chunks.
    assert received_count["n"] < total_chunks, (
        f"middleware read {received_count['n']}/{total_chunks} chunks — "
        "streaming abort regressed"
    )
    assert received_count["n"] <= 6, (
        f"middleware over-read: {received_count['n']} chunks of "
        f"{chunk_size} B vs limit 1024"
    )


def test_get_request_is_not_capped():
    """The middleware skips GET (and other body-less methods) entirely
    so it adds no latency to ``/v1/models``, ``/v1/health/...``,
    etc. A spuriously large GET shouldn't be possible in HTTP/1.1
    anyway, but we still want zero per-request overhead on read
    paths."""
    from vllm_mlx.config.server_config import get_config

    get_config().max_request_bytes = 1  # absurdly tiny — every body would fail

    app = _build_app()

    @app.get("/v1/models")
    async def _list():
        return {"data": []}

    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200


def test_cli_flag_overrides_config_default():
    """The ``--max-request-bytes`` flag is wired through
    ``vllm_mlx.server._max_request_bytes`` and ``_sync_config`` into
    ``ServerConfig.max_request_bytes``. We don't exercise the full
    CLI here (too heavy — needs model load) but we do assert the
    wiring: writing the module global and calling ``_sync_config``
    propagates the value to the config singleton the middleware
    reads."""
    import vllm_mlx.server as server_mod
    from vllm_mlx.config.server_config import get_config

    original = server_mod._max_request_bytes
    try:
        server_mod._max_request_bytes = 12345
        server_mod._sync_config()
        assert get_config().max_request_bytes == 12345
    finally:
        server_mod._max_request_bytes = original
        server_mod._sync_config()
