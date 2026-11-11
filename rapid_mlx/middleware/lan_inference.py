# SPDX-License-Identifier: Apache-2.0
"""Opt-in Desktop LAN boundary. Unlisted routes remain local, including new APIs.

RAPID_MLX_LAN_INFERENCE_ONLY=1 enables the boundary. The peer address must be
provided by the listener, not a forwarded header (Desktop disables proxy trust).
This is deliberately separate from normal CLI authentication and CORS policy.
"""

import ipaddress
import os

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from ..config import get_config
from .auth import verify_api_key_for_scope

# Match methods as well as paths: model activation/deletion is administration.
_INFERENCE_ROUTES = {
    "GET": frozenset(
        {
            "/v1/models",
            "/v1/audio/voices",
            "/v1/videos",
            "/v1/videos/capabilities",
            "/health",
            "/health/ready",
            "/healthz",
            "/readyz",
            "/livez",
        }
    ),
    "POST": frozenset(
        {
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/responses",
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/rerank",
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/audio/speech",
            "/v1/audio/music",
            "/v1/images/generations",
            "/v1/images/edits",
            "/v1/videos",
            "/v1/videos/extend",
        }
    ),
}


def is_loopback_peer(client) -> bool:
    """Classify the ASGI socket peer, never a Host or forwarded header."""
    try:
        address = ipaddress.ip_address(client[0])
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_loopback
    except (ValueError, TypeError, IndexError):
        return False


_LOCAL_ONLY_DETAIL = "This route is available only on this Mac"
_PENDING_ROUTE_CHECK = "rapid_mlx.lan_pending_route_check"
_DYNAMIC_GET_ROUTES = frozenset(
    {
        "/v1/models/{model_id:path}",
        "/v1/videos/{video_id}",
        "/v1/videos/{video_id}/content",
    }
)


async def _authenticate(scope):
    # Even probes and routes with no auth dependency require a key.
    if not get_config().api_key:
        raise HTTPException(401, "LAN access requires an API key")
    await verify_api_key_for_scope(scope)


class _LANRouteError(Exception):
    """Return boundary errors through the middleware, preserving its JSON body."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


async def verify_lan_inference_route(request: Request):
    """App-level dependency: authorize dynamic GETs after routing, before handlers."""
    if not request.scope.get(_PENDING_ROUTE_CHECK):
        return
    request.scope[_PENDING_ROUTE_CHECK] = False
    # Model IDs may contain slashes, but /v1/models/residency is management.
    # Check the route actually selected, so a more specific admin route
    # (including future additions) cannot inherit model-lookup permission.
    if getattr(request.scope.get("route"), "path", None) not in _DYNAMIC_GET_ROUTES:
        raise _LANRouteError(403, _LOCAL_ONLY_DETAIL)
    try:
        await _authenticate(request.scope)
    except HTTPException as exc:
        raise _LANRouteError(exc.status_code, exc.detail) from exc


class LANInferenceMiddleware:
    def __init__(self, app, *, enabled: bool | None = None):
        self.app = app
        self.enabled = (
            os.environ.get("RAPID_MLX_LAN_INFERENCE_ONLY") == "1"
            if enabled is None
            else enabled
        )

    async def __call__(self, scope, receive, send):
        if (
            not self.enabled
            or scope["type"] not in {"http", "websocket"}
            or is_loopback_peer(scope.get("client"))
        ):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if (
            scope["method"] == "GET"
            and scope["path"].startswith(("/v1/models/", "/v1/videos/"))
            and scope["path"] not in _INFERENCE_ROUTES["GET"]
        ):
            scope[_PENDING_ROUTE_CHECK] = True
            refused = False

            async def checked_send(message):
                nonlocal refused
                if refused:
                    return
                if message["type"] == "http.response.start" and scope.get(
                    _PENDING_ROUTE_CHECK
                ):
                    # No dependency ran (e.g. an unmatched URL or redirect).
                    # Keep unlisted paths local rather than leaking router responses.
                    refused = True
                    response = JSONResponse(
                        {"detail": _LOCAL_ONLY_DETAIL}, status_code=403
                    )
                    await response(scope, receive, send)
                    return
                await send(message)

            try:
                return await self.app(scope, receive, checked_send)
            except _LANRouteError as exc:
                response = JSONResponse(
                    {"detail": exc.detail}, status_code=exc.status_code
                )
                return await response(scope, receive, send)
        if scope["path"] not in _INFERENCE_ROUTES.get(scope["method"], ()):
            response = JSONResponse({"detail": _LOCAL_ONLY_DETAIL}, status_code=403)
        else:
            try:
                await _authenticate(scope)
            except HTTPException as exc:
                response = JSONResponse(
                    {"detail": exc.detail}, status_code=exc.status_code
                )
            else:
                return await self.app(scope, receive, send)
        await response(scope, receive, send)
