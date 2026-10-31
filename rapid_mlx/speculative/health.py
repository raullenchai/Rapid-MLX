# SPDX-License-Identifier: Apache-2.0
"""Common public probes for dedicated speculative servers."""

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def install_health_probes(
    app: FastAPI, model_name: str, healthz: Callable[[], Awaitable[dict[str, Any]]]
) -> None:
    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok"}

    @app.head("/", include_in_schema=False)
    async def root_head() -> None:
        return None

    @app.get("/health")
    async def health() -> dict[str, Any]:
        details = await healthz()
        ready = (
            bool(details["ready"])
            if "ready" in details
            else details.get("status") == "ok"
        )
        return {
            **details,
            "status": "healthy",
            "ready": ready,
            "state": "ready" if ready else (details.get("status") or "loading"),
            "model_loaded": ready,
            "model_name": model_name,
            "model_type": "llm",
            "engine_type": details.get("engine", "unknown"),
        }

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    @app.get("/health/ready")
    async def health_ready() -> Any:
        details = await healthz()
        ready = (
            bool(details["ready"])
            if "ready" in details
            else details.get("status") == "ok"
        )
        if ready:
            return {
                "status": "ok",
                "ready": True,
                "state": "ready",
                "model_loaded": True,
                "model": model_name,
            }
        state = details.get("status") or "loading"
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "model failed to load"
                    if state == "error"
                    else "model loading",
                    "type": "service_unavailable",
                    "code": "model_not_ready",
                },
                "ready": False,
                "state": state,
                "model_loaded": False,
            },
        )
