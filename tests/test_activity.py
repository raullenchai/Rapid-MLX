import asyncio

import pytest

from rapid_mlx.middleware.activity import ActivityMiddleware, ActivityTracker


@pytest.mark.asyncio
async def test_activity_tracks_stream_and_disconnect():
    tracker = ActivityTracker()
    started = asyncio.Event()
    release = asyncio.Event()

    async def app(scope, receive, send):
        started.set()
        await release.wait()
        if scope.get("disconnect"):
            raise ConnectionError("disconnected")

    middleware = ActivityMiddleware(app, tracker)
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
    task = asyncio.create_task(middleware(scope, None, None))
    await started.wait()
    assert tracker.snapshot() == {"in_flight": 1, "idle_seconds": 0.0}
    release.set()
    await task
    await asyncio.sleep(0.01)
    assert tracker.snapshot()["idle_seconds"] > 0

    scope["disconnect"] = True
    with pytest.raises(ConnectionError):
        await middleware(scope, None, None)
    assert tracker.snapshot()["in_flight"] == 0


@pytest.mark.asyncio
async def test_read_only_and_other_paths_do_not_count():
    tracker = ActivityTracker()

    async def app(scope, receive, send):
        assert tracker.snapshot()["in_flight"] == 0

    middleware = ActivityMiddleware(app, tracker)
    for method, path in [
        ("GET", "/v1/models"),
        ("GET", "/v1/models/residency"),
        ("POST", "/health"),
    ]:
        await middleware({"type": "http", "method": method, "path": path}, None, None)


@pytest.mark.asyncio
async def test_streaming_response_remains_in_flight_until_body_finishes():
    from starlette.responses import StreamingResponse

    tracker = ActivityTracker()
    first_chunk = asyncio.Event()
    release = asyncio.Event()

    async def body():
        yield b"first"
        first_chunk.set()
        await release.wait()
        yield b"last"

    response = StreamingResponse(body())
    middleware = ActivityMiddleware(response, tracker)

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    task = asyncio.create_task(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/completions",
                "asgi": {"spec_version": "2.4"},
            },
            receive,
            send,
        )
    )
    await first_chunk.wait()
    assert tracker.snapshot()["in_flight"] == 1
    release.set()
    await task
    assert tracker.snapshot()["in_flight"] == 0
