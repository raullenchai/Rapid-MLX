# SPDX-License-Identifier: Apache-2.0
"""Standalone TypeSafe-compatible System One API service."""

from __future__ import annotations

import asyncio
import hmac
import time
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .backends import DecisionBackend
from .schema import RankRequest, SystemOneRequest


def create_app(
    backend: DecisionBackend,
    api_key: str | None = None,
    *,
    max_concurrent_requests: int = 8,
) -> FastAPI:
    if max_concurrent_requests < 1:
        raise ValueError("max_concurrent_requests must be positive")
    try:
        api_key_bytes = api_key.encode("ascii") if api_key is not None else None
    except UnicodeEncodeError as exc:
        raise ValueError(
            "System One API keys must contain ASCII characters only"
        ) from exc
    app = FastAPI(
        title="Rapid-MLX System One API",
        description="TypeSafe-compatible typed decisions on Apple Silicon",
        version="1.0.0",
    )
    app.state.backend = backend
    # Apply the same pre-parse JSON protections as the generative server.
    # RequestBodyLimitMiddleware owns BOTH the size cap and the per-chunk
    # ServerConfig.body_receive_timeout_seconds slow-body timeout.
    from rapid_mlx.middleware.body_depth import (
        install_request_body_depth_middleware,
    )
    from rapid_mlx.middleware.body_size import install_request_body_limit_middleware

    install_request_body_depth_middleware(app)
    install_request_body_limit_middleware(app)
    in_flight = 0

    async def run_backend(call: Callable[..., Any], *args: Any) -> Any:
        nonlocal in_flight
        # No await occurs between the check and increment, so reservation is
        # atomic with respect to every other task on this event loop.
        if in_flight >= max_concurrent_requests:
            raise HTTPException(
                status_code=503,
                detail="System One request capacity is full",
                headers={"Retry-After": "1"},
            )
        in_flight += 1
        try:
            worker = asyncio.create_task(asyncio.to_thread(call, *args))
        except BaseException:
            in_flight -= 1
            raise

        def release_permit(_worker: asyncio.Task) -> None:
            nonlocal in_flight
            try:
                if not _worker.cancelled():
                    _worker.exception()
            finally:
                in_flight -= 1

        # A client disconnect cancels the request coroutine, but cannot stop a
        # Python worker thread. Keep its permit until the worker really exits.
        worker.add_done_callback(release_permit)
        return await asyncio.shield(worker)

    def verify(authorization: str | None = Header(default=None)) -> None:
        if api_key_bytes is None:
            return
        scheme, separator, token = (authorization or "").partition(" ")
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError:
            token_bytes = b""
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(token_bytes, api_key_bytes)
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/v1/models", dependencies=[Depends(verify)])
    def models() -> dict:
        return {"models": backend.models()}

    @app.post("/v1/systemone", dependencies=[Depends(verify)])
    async def system_one(request: SystemOneRequest):
        model = request.model or backend.default_model
        started = time.perf_counter()
        try:
            result = await run_backend(
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
            ranked = await run_backend(
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
