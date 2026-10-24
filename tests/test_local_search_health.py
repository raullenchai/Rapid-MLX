# SPDX-License-Identifier: Apache-2.0
"""CPU-only health and task ownership regression for optional local search."""

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from rapid_mlx import local_search_runtime as runtime
from rapid_mlx.config import ServerConfig
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes import health


@pytest.mark.parametrize("mode", ["disabled", "healthy", "invalid"])
def test_local_search_health_lifecycle(tmp_path, monkeypatch, caplog, mode):
    directory = tmp_path / "index"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    if mode == "invalid":
        (directory / "unrelated.txt").write_text("preserve me")
    cfg = ServerConfig(ready=True, model_name="test-model")
    monkeypatch.setattr(health, "get_config", lambda: cfg)
    monkeypatch.setattr(health, "primary_switching", lambda: False)
    monkeypatch.setattr(runtime, "_enabled", mode != "disabled")
    monkeypatch.setattr(runtime, "_directory", str(directory))
    caplog.set_level(logging.ERROR)

    async def scenario():
        app = FastAPI()
        app.include_router(health.probe_router)
        install_exception_handlers(app)
        loop_errors = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: loop_errors.append(context))
        try:
            await runtime.startup(app)
            service = getattr(app.state, "local_search", None)
            if service is not None:
                assert service.health["status"] == "starting"
                await asyncio.wait_for(service.wait_idle(), 5)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                for _ in range(2):
                    for path in ("/health", "/health/ready", "/readyz"):
                        response = await client.get(path)
                        assert response.status_code == 200
                        data = response.json()
                        assert data["ready"] is True
                        component = data["local_search"]
                        if mode == "disabled":
                            assert component is None
                        else:
                            assert component == service.health
                            assert component["enabled"] is True
                            assert component["status"] == (
                                "degraded" if mode == "invalid" else "healthy"
                            )
                            if mode == "invalid":
                                assert "index directory" in component["reason"]
                                assert str(tmp_path) not in response.text
                            else:
                                assert component["reason"] is None
                        if path == "/health":
                            assert data["status"] == (
                                "degraded" if mode == "invalid" else "healthy"
                            )
                if mode == "invalid":
                    response = await client.get("/v1/local-search/status")
                    assert response.status_code == 503
                    assert response.json()["error"]["code"] == "store_open_failed"
                    await asyncio.sleep(0)
                    assert service.refresh_task.cancelled()
                    assert (directory / "unrelated.txt").read_text() == "preserve me"
        finally:
            await runtime.shutdown(app)
            loop.set_exception_handler(previous)
        assert not loop_errors
        assert not hasattr(app.state, "local_search")
        # A subsequent disabled lifespan has no stale failure status.
        monkeypatch.setattr(runtime, "_enabled", False)
        await runtime.startup(app)
        assert health._local_search_health(SimpleNamespace(app=app)) is None
        assert health._local_search_health(None) is None

    asyncio.run(scenario())
    failures = [
        r for r in caplog.records if r.message == "Local-search initialization failed"
    ]
    assert len(failures) == (1 if mode == "invalid" else 0)
    if failures:
        assert isinstance(failures[0].exc_info[1], PermissionError)


def test_resume_cancellation_is_not_a_failure(tmp_path, caplog):
    from rapid_mlx.local_search.service import LocalSearch

    async def scenario():
        service = LocalSearch(tmp_path / "index")
        entered = asyncio.Event()

        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()

        service.store.call = blocked
        service.start()
        await entered.wait()
        task = service.task
        await service.close()
        assert task.cancelled()
        assert service.health["status"] != "degraded"

    asyncio.run(scenario())
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
