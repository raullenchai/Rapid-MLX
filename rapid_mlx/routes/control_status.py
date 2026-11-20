# SPDX-License-Identifier: Apache-2.0
"""Explicit control routes, inside the existing browser/LAN boundary."""

import asyncio
import http.client
import ipaddress
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

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


def mount_status(app):
    if os.environ.get(RUNNER_ENV) != "1":
        return
    path, secret = os.environ[SOCKET_ENV], os.environ[SECRET_ENV]
    router = APIRouter()

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

    dependencies = [Depends(require_loopback), Depends(verify_api_key)]

    @router.api_route(
        "/rapid/v1/log", methods=["GET", "DELETE"], dependencies=dependencies
    )
    @router.api_route(
        "/rapid/v1/log/settings", methods=["GET", "PUT"], dependencies=dependencies
    )
    async def logs(request: Request):
        size = 0
        chunks = []
        try:
            async with asyncio.timeout(TIMEOUT):
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BODY:
                        raise HTTPException(413, "Control request too large")
                    chunks.append(chunk)
                import json

                payload = json.loads(b"".join(chunks)) if size else None
                target = request.url.path
                if request.url.query:
                    target += "?" + request.url.query
                return await asyncio.to_thread(
                    control_request,
                    path,
                    secret,
                    request.method,
                    payload,
                    target=target,
                )
        except (TimeoutError, OSError, ValueError, http.client.HTTPException):
            raise HTTPException(503, "Control plane unavailable") from None

    @router.get("/rapid/v1/events", dependencies=dependencies)
    async def events(request: Request):
        try:
            cursor = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            raise HTTPException(400, "Invalid Last-Event-ID") from None
        return BoundedEventResponse(
            event_stream(path, secret, cursor), media_type="text/event-stream"
        )

    app.include_router(router)


class BoundedEventResponse(StreamingResponse):
    async def stream_response(self, send):
        async def bounded_send(message):
            async with asyncio.timeout(5):
                await send(message)

        await super().stream_response(bounded_send)


async def event_stream(path, secret, cursor):
    """Pull from the fixed parent ring; no per-client accumulating queue.

    A client falling behind the retained ring must reconnect. Each response
    holds at most one bounded snapshot and sends have a five-second deadline.
    """
    import json

    while True:
        try:
            batch = await asyncio.to_thread(
                control_request, path, secret, target=f"/rapid/v1/events?since={cursor}"
            )
        except (OSError, ValueError, http.client.HTTPException):
            return
        if batch and cursor and batch[0]["id"] > cursor + 1:
            return
        for event in batch:
            cursor = event["id"]
            yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
        yield ": heartbeat\n\n"
        await asyncio.sleep(1)
