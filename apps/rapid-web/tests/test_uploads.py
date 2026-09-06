# SPDX-License-Identifier: Apache-2.0
"""Upload limits must fire before multipart parsing buffers the body."""

from __future__ import annotations

import asyncio
import json

import pytest

from rmlx_web.uploads import UploadBodyLimitMiddleware


class _DrainBody:
    def __init__(self) -> None:
        self.received = 0

    async def __call__(self, scope, receive, send) -> None:
        while True:
            message = await receive()
            self.received += len(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _scope(*, content_length: int | None = None) -> dict:
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {
        "type": "http",
        "method": "POST",
        "path": "/api/audio/transcriptions",
        "headers": headers,
    }


@pytest.mark.asyncio
async def test_advertised_oversize_is_rejected_without_reading() -> None:
    app = _DrainBody()
    middleware = UploadBodyLimitMiddleware(app, limits={"/api/audio/transcriptions": 8})
    receive_calls = 0
    sent = []

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"never"}

    async def send(message):
        sent.append(message)

    await middleware(_scope(content_length=9), receive, send)

    assert receive_calls == 0
    assert app.received == 0
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["type"] == "payload_too_large"


@pytest.mark.asyncio
async def test_chunked_oversize_stops_before_the_rest_of_the_body() -> None:
    app = _DrainBody()
    middleware = UploadBodyLimitMiddleware(app, limits={"/api/audio/transcriptions": 8})
    chunks = [b"1234", b"5678", b"over", b"unread"]
    receive_calls = 0
    sent = []

    async def receive():
        nonlocal receive_calls
        body = chunks[receive_calls]
        receive_calls += 1
        return {"type": "http.request", "body": body, "more_body": True}

    async def send(message):
        sent.append(message)

    await middleware(_scope(), receive, send)

    assert receive_calls == 3
    assert app.received == 8
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_stalled_upload_times_out() -> None:
    app = _DrainBody()
    middleware = UploadBodyLimitMiddleware(
        app,
        limits={"/api/audio/transcriptions": 8},
        receive_timeout=0.01,
    )
    sent = []

    async def receive():
        await asyncio.Future()

    async def send(message):
        sent.append(message)

    await middleware(_scope(), receive, send)

    assert sent[0]["status"] == 408
    assert json.loads(sent[1]["body"])["error"]["type"] == "request_timeout"
    assert app.received == 0
