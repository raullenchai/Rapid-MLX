"""CUA-only transport and lifecycle safety at its public host boundary."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rapid_mlx.config import reset_config
from rapid_mlx.cua import server


def test_lifespan_sets_readiness_and_drains_even_after_error(monkeypatch):
    cfg = reset_config()
    closed = []

    async def close():
        closed.append(True)

    monkeypatch.setattr(server, "close_cua_service", close)

    async def exercise():
        with pytest.raises(RuntimeError, match="request failed"):
            async with server.lifespan(FastAPI()):
                assert cfg.ready is True
                assert cfg.draining is False
                raise RuntimeError("request failed")

    asyncio.run(exercise())
    assert cfg.ready is False
    assert cfg.draining is True
    assert closed == [True]


def test_transport_config_reads_environment_once_and_enforces_hosts(monkeypatch):
    isolated_app = FastAPI()
    monkeypatch.setattr(server, "app", isolated_app)
    monkeypatch.setattr(server, "_configured", False)
    monkeypatch.setenv(
        "RAPID_MLX_CORS_ALLOW_ORIGINS", " https://desk.example, ,http://127.0.0.1 "
    )
    origins = server.configure_cua_server(
        cors_origins=None, trusted_hosts=["localhost"]
    )
    assert origins == ["https://desk.example", "http://127.0.0.1"]
    assert any(m.cls is CORSMiddleware for m in isolated_app.user_middleware)
    assert any(m.cls is TrustedHostMiddleware for m in isolated_app.user_middleware)
    with pytest.raises(RuntimeError, match="already configured"):
        server.configure_cua_server(cors_origins=[], trusted_hosts=None)
