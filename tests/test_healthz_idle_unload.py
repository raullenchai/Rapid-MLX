# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for health probes across primary idle unload."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import ServerConfig, reset_config
from rapid_mlx.middleware.probe_fastpath import install_probe_fastpath_middleware
from rapid_mlx.runtime.primary_lifecycle import PrimaryModelLifecycle


class FakeEngine:
    def __init__(self, *, loaded: bool) -> None:
        self._loaded = loaded

    async def start(self) -> None:
        self._loaded = True

    async def stop(self) -> None:
        self._loaded = False

    async def pause_generation(self, mode: str, *, timeout: float | None = None):
        assert mode == "wait"
        assert timeout == 0
        return {"active_requests": 0}

    async def resume_generation(self):
        return {"active_requests": 0}

    def lifecycle_status(self) -> dict[str, int]:
        return {"active_requests": 0}

    def get_stats(self) -> dict[str, object]:
        return {}


def _make_app() -> FastAPI:
    from rapid_mlx.routes.health import probe_router
    from rapid_mlx.routes.metrics import router as metrics_router

    app = FastAPI()
    app.include_router(probe_router)
    app.include_router(metrics_router)
    install_probe_fastpath_middleware(app)
    return app


def _model_loaded_metric(client: TestClient) -> int:
    response = client.get("/metrics")
    assert response.status_code == 200
    line = next(
        line
        for line in response.text.splitlines()
        if line.startswith("rapid_mlx_model_loaded ")
    )
    return int(line.rsplit(" ", 1)[1])


def _assert_healthz(
    client: TestClient, *, loaded: bool, origin: str | None = None
) -> None:
    headers = {"Origin": origin} if origin is not None else None
    response = client.get("/healthz", headers=headers)
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "ready": True,
        "model_loaded": loaded,
        "model_name": "public-model",
    }


def test_server_config_model_loaded_prefers_lifecycle_and_keeps_fallback():
    cfg = ServerConfig(engine=object())
    assert cfg.model_loaded is True

    cfg.engine = None
    assert cfg.model_loaded is False

    cfg.primary_model_lifecycle = SimpleNamespace(model_loaded=False)
    cfg.engine = object()
    assert cfg.model_loaded is False

    cfg.primary_model_lifecycle.model_loaded = True
    cfg.engine = None
    assert cfg.model_loaded is True


@pytest.mark.asyncio
async def test_healthz_and_metrics_track_idle_unload_and_reload():
    now = [100.0]
    engine = FakeEngine(loaded=True)
    lifecycle = PrimaryModelLifecycle(
        engine,
        model_name="public-model",
        idle_unload_seconds=10,
        clock=lambda: now[0],
    )
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "public-model"
    cfg.ready = True
    cfg.primary_model_lifecycle = lifecycle

    try:
        client = TestClient(_make_app())

        _assert_healthz(client, loaded=True)
        _assert_healthz(client, loaded=True, origin="https://dashboard.example")
        assert _model_loaded_metric(client) == 1

        now[0] += 11
        assert await lifecycle.evict_if_idle() is True
        _assert_healthz(client, loaded=False)
        _assert_healthz(client, loaded=False, origin="https://dashboard.example")
        assert _model_loaded_metric(client) == 0

        await lifecycle.ensure_loaded()
        _assert_healthz(client, loaded=True)
        _assert_healthz(client, loaded=True, origin="https://dashboard.example")
        assert _model_loaded_metric(client) == 1
    finally:
        reset_config()
