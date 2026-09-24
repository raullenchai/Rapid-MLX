# SPDX-License-Identifier: Apache-2.0
"""Standalone TypeSafe-compatible System One API service."""

from __future__ import annotations

import asyncio
import hmac
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .backends import DecisionBackend
from .schema import RankRequest, SystemOneRequest


def create_app(backend: DecisionBackend, api_key: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Rapid-MLX System One API",
        description="TypeSafe-compatible typed decisions on Apple Silicon",
        version="1.0.0",
    )
    app.state.backend = backend
    # Apply the same pre-parse JSON size and nesting limits as the generative
    # server. Typed state is intentionally flexible, so route-level schemas
    # alone cannot bound parser work.
    from rapid_mlx.middleware.body_depth import (
        install_request_body_depth_middleware,
    )
    from rapid_mlx.middleware.body_size import install_request_body_limit_middleware

    install_request_body_depth_middleware(app)
    install_request_body_limit_middleware(app)

    def verify(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        expected = f"Bearer {api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "models": [item["name"] for item in backend.models()]}

    @app.get("/v1/models", dependencies=[Depends(verify)])
    def models() -> dict:
        return {"models": backend.models()}

    @app.post("/v1/systemone", dependencies=[Depends(verify)])
    async def system_one(request: SystemOneRequest):
        model = request.model or backend.default_model
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                backend.answer,
                request.state,
                request.questions,
                model,
                request.temperature,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"invalid request: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        result["latency_ms"] = latency_ms
        return JSONResponse(
            result,
            headers={"X-Rapid-MLX-Latency-Ms": f"{latency_ms:.1f}"},
        )

    @app.post("/v1/rank", dependencies=[Depends(verify)])
    async def rank(request: RankRequest):
        model = request.model or backend.default_model
        started = time.perf_counter()
        try:
            ranked = await asyncio.to_thread(
                backend.rank,
                request.context,
                request.question,
                request.answers,
                model,
                request.temperature,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"invalid request: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return JSONResponse(
            {"model": model, "ranked": ranked, "latency_ms": latency_ms},
            headers={"X-Rapid-MLX-Latency-Ms": f"{latency_ms:.1f}"},
        )

    return app
