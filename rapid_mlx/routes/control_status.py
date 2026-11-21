# SPDX-License-Identifier: Apache-2.0
"""The sole P1 control forwarder, inside the existing browser/LAN boundary."""

import asyncio
import http.client
import ipaddress
import os
from collections import deque

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from rapid_mlx.control import RUNNER_ENV, SECRET_ENV, SOCKET_ENV
from rapid_mlx.control.protocol import MAX_BODY, TIMEOUT
from rapid_mlx.control.protocol import request as control_request
from rapid_mlx.middleware.auth import verify_api_key


async def require_loopback(request: Request):
    peer = request.scope.get("_rapid_mlx_transport_peer", request.scope.get("client"))
    try:
        address = ipaddress.ip_address(peer[0]) if peer else None
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if address is not None and address.is_loopback:
            return
    except ValueError:
        pass
    raise HTTPException(403, "Control API requires a loopback peer")


def agent_jobs():
    """Adapt public run summaries for the unified jobs view."""
    from rapid_mlx.routes.agents import list_agent_runs

    return [
        dict(
            id=run["id"],
            kind="agent",
            status=run["status"],
            created=run["created"],
            updated=run["updated"],
            summary="".join(
                c for c in " ".join(run["goal"].split()) if c.isprintable()
            )[:120],
        )
        for run in list_agent_runs()
    ]


def mount_status(app):
    if os.environ.get(RUNNER_ENV) != "1":
        return
    path, secret = os.environ[SOCKET_ENV], os.environ[SECRET_ENV]
    from rapid_mlx.control.call_log import CallLog

    calls: deque[dict] = deque(maxlen=1000)
    app.add_middleware(CallLog, calls=calls)
    router = APIRouter()

    @router.get(
        "/rapid/v1/log",
        dependencies=[Depends(require_loopback), Depends(verify_api_key)],
    )
    async def log(limit: int = Query(50, ge=1, le=1000)):
        return {"calls": list(calls)[-limit:]}

    @router.get(
        "/rapid/v1/jobs",
        dependencies=[Depends(require_loopback), Depends(verify_api_key)],
    )
    async def jobs():
        return {"jobs": agent_jobs()}

    @router.post(
        "/rapid/v1/jobs/{job_id}/cancel",
        dependencies=[Depends(require_loopback), Depends(verify_api_key)],
    )
    async def cancel_job(job_id: str):
        from rapid_mlx.routes.agents import cancel_agent_run

        return await cancel_agent_run(job_id)

    @router.get(
        "/rapid/v1/status",
        dependencies=[Depends(require_loopback), Depends(verify_api_key)],
    )
    async def status(request: Request):
        # Never buffer an arbitrary GET body. Even chunked clients have a bound.
        size = 0
        try:
            async with asyncio.timeout(TIMEOUT):
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BODY:
                        raise HTTPException(413, "Control request too large")
                # No client headers are copied. The private secret always wins.
                return await asyncio.to_thread(control_request, path, secret)
        except (TimeoutError, OSError, ValueError, http.client.HTTPException):
            raise HTTPException(503, "Control plane unavailable") from None

    app.include_router(router)
