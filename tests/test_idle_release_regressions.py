"""Idle-release safety across dedicated apps, background work and uploads."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from rapid_mlx.middleware import activity


@pytest.fixture(autouse=True)
def _isolate_server_config(monkeypatch):
    """Dedicated app construction must not leak shared launch configuration."""
    from rapid_mlx.config import get_config
    from rapid_mlx.middleware import auth

    cfg = get_config()
    for name, value in vars(cfg).items():
        monkeypatch.setattr(cfg, name, value)
    monkeypatch.setattr(auth, "rate_limiter", auth.RateLimiter(enabled=False))


def scope(path, *, headers=()):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1"), *headers],
        "server": ("test", 80),
        "client": ("test", 1),
    }


async def empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


@pytest.mark.asyncio
async def test_dedicated_residency_tracks_open_stream(monkeypatch):
    from rapid_mlx.speculative.dflash.server import _build_app

    app = _build_app(
        model=None,
        processor=SimpleNamespace(),
        runtime=SimpleNamespace(),
        served_model_name="test-model",
        default_max_tokens=8,
        cors_origins=[],
        api_key="secret",
    )
    opened, release = asyncio.Event(), asyncio.Event()

    @app.post("/v1/test-stream")
    async def stream():
        async def body():
            yield b"first"
            opened.set()
            await release.wait()
            yield b"last"

        return StreamingResponse(body())

    async def send(message):
        pass

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (await client.get("/v1/models/residency")).status_code == 401
        client.headers["Authorization"] = "Bearer secret"
        response = await client.get("/v1/models/residency")
        assert response.status_code == 200, response.text
        snapshot = response.json()
        assert snapshot["activity"]["in_flight"] == 0
        assert snapshot["activity"]["idle_seconds"] >= 0
        for field in (
            "memory_limit_bytes",
            "memory_used_bytes",
            "idle_ttl_seconds",
            "loads_total",
            "evictions_total",
        ):
            assert isinstance(snapshot[field], (int, float))
        model = snapshot["models"][0]
        assert model["id"] == "test-model"
        assert model["state"] == "resident"
        for field in (
            "model_path",
            "aliases",
            "modality",
            "pinned",
            "primary",
            "active_requests",
            "estimated_bytes",
            "idle_seconds",
        ):
            assert field in model
        task = asyncio.create_task(app(scope("/v1/test-stream"), empty_receive, send))
        try:
            await asyncio.wait_for(opened.wait(), 2)
            busy = (await client.get("/v1/models/residency")).json()
            assert busy["activity"] == {"in_flight": 1, "idle_seconds": 0.0}
            assert busy["models"][0]["active_requests"] == 1
        finally:
            release.set()
            await task
        idle = (await client.get("/v1/models/residency")).json()
        assert idle["activity"]["in_flight"] == 0
        assert idle["activity"]["idle_seconds"] < 2


@pytest.mark.asyncio
async def test_residency_counts_queued_and_active_local_indexing(monkeypatch, tmp_path):
    from rapid_mlx.local_search.service import LocalSearch
    from rapid_mlx.middleware.auth import verify_api_key
    from rapid_mlx.routes import residency

    clock = [100.0]
    monkeypatch.setattr(activity, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    tracker = activity.ActivityTracker()
    monkeypatch.setattr(activity, "activity_tracker", tracker)
    service = LocalSearch(tmp_path, resolve_model=lambda _: None)
    monkeypatch.setattr(service.store, "call", AsyncMock(return_value={"id": "root"}))
    opened, release = asyncio.Event(), asyncio.Event()

    async def index(folder):
        opened.set()
        await release.wait()

    monkeypatch.setattr(service, "_index", index)
    monkeypatch.setattr(
        residency, "_manager", lambda: SimpleNamespace(snapshot=lambda: {"models": []})
    )
    app = FastAPI()
    app.state.local_search = service
    app.include_router(residency.router)
    app.dependency_overrides[verify_api_key] = lambda: None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:

        async def snapshot():
            response = await client.get("/v1/models/residency")
            assert response.status_code == 200
            return response.json()["activity"]

        try:
            assert (await snapshot())["in_flight"] == 0
            async with service._coordination:
                await service.queue("root")
                assert (await snapshot())["in_flight"] > 0
            await asyncio.wait_for(opened.wait(), 2)
            assert not service.pending
            assert await snapshot() == {"in_flight": 1, "idle_seconds": 0.0}
            clock[0] += 600
            release.set()
            await service.task
            assert await snapshot() == {"in_flight": 0, "idle_seconds": 0.0}
            clock[0] += 10
            assert (await snapshot())["idle_seconds"] == 10
        finally:
            release.set()
            await service.close()


@pytest.mark.asyncio
async def test_main_stack_counts_partial_json_upload(monkeypatch):
    from rapid_mlx import server

    tracker = activity.ActivityTracker()
    monkeypatch.setattr(activity, "activity_tracker", tracker)
    app = FastAPI()
    # Reuse the actual registered middleware and ordering without running lifespan.
    app.user_middleware = list(server.app.user_middleware)
    for middleware in app.user_middleware:
        if middleware.cls is activity.ActivityMiddleware:
            monkeypatch.setitem(middleware.kwargs, "tracker", tracker)
    waiting, release = asyncio.Event(), asyncio.Event()
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b'{"prompt":', "more_body": True}
        waiting.set()
        await release.wait()
        return {"type": "http.request", "body": b'"hello"}', "more_body": False}

    async def send(message):
        pass

    task = asyncio.create_task(
        app(
            scope("/v1/completions", headers=[(b"content-type", b"application/json")]),
            receive,
            send,
        )
    )
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        assert tracker.snapshot() == {"in_flight": 1, "idle_seconds": 0.0}
    finally:
        release.set()
        await task
    assert tracker.snapshot()["in_flight"] == 0


def test_stt_inherited_fd_readiness_banner(monkeypatch, capsys):
    from rapid_mlx import server
    from rapid_mlx.config import get_config
    from rapid_mlx.runtime import optional_runtime

    cfg = get_config()
    monkeypatch.setattr(cfg, "model_alias", "whisper-large-v3-turbo")
    monkeypatch.setattr(cfg, "model_name", "whisper-large-v3-turbo")
    monkeypatch.setattr(cfg, "bind_listen_fd", 3)
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: True)
    monkeypatch.setattr(server, "_served_model_name_set", False)
    server.print_ready_banner()
    assert "Ready: inherited fd 3" in capsys.readouterr().out
