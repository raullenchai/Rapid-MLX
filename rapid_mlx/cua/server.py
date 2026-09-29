"""Lightweight FastAPI host for model-free Computer Use serving."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rapid_mlx.config import get_config
from rapid_mlx.cua.service import close_cua_service
from rapid_mlx.middleware.body_size import install_request_body_limit_middleware
from rapid_mlx.routes.cua import router as cua_router
from rapid_mlx.routes.health import probe_router
from rapid_mlx.routes.health import router as health_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Publish readiness without constructing or warming a model engine."""

    cfg = get_config()
    cfg.draining = False
    cfg.ready = True
    try:
        yield
    finally:
        cfg.ready = False
        cfg.draining = True
        await close_cua_service()


app = FastAPI(
    title="Rapid-MLX Computer Use API",
    description="Authenticated model-free Computer Use control plane",
    version="1",
    lifespan=lifespan,
)
install_request_body_limit_middleware(app)
app.include_router(probe_router)
app.include_router(health_router)
app.include_router(cua_router)

_configured = False


def configure_cua_server(
    *,
    cors_origins: list[str] | None,
    trusted_hosts: list[str] | None,
) -> list[str]:
    """Apply transport middleware once and return the resolved CORS origins."""

    global _configured
    if _configured:
        raise RuntimeError("CUA-only server transport is already configured")

    resolved_origins = cors_origins
    if resolved_origins is None:
        raw = os.environ.get("RAPID_MLX_CORS_ALLOW_ORIGINS", "").strip()
        resolved_origins = [item.strip() for item in raw.split(",") if item.strip()]
    if resolved_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
            allow_credentials="*" not in resolved_origins,
            max_age=3600,
        )
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    _configured = True
    return list(resolved_origins or [])
