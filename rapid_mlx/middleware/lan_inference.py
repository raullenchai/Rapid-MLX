# SPDX-License-Identifier: Apache-2.0
"""Opt-in Desktop LAN boundary. Unlisted routes remain local, including new APIs.

RAPID_MLX_LAN_INFERENCE_ONLY=1 enables the boundary. The peer address must be
provided by the listener, not a forwarded header (Desktop disables proxy trust).
This is deliberately separate from normal CLI authentication and CORS policy.
"""

import ipaddress
import os

from fastapi import HTTPException
from starlette.responses import JSONResponse
from starlette.routing import Match

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


def _allowed(scope, router) -> bool:
    method, path = scope["method"], scope["path"]
    if path in _INFERENCE_ROUTES.get(method, ()):
        return True
    # Model IDs may contain slashes, but /v1/models/residency is management.
    # Check the first route the router would select, so a more specific admin
    # route (including future additions) cannot inherit model-lookup permission.
    if (
        method == "GET"
        and path.startswith(("/v1/models/", "/v1/videos/"))
        and router is not None
    ):
        for route in router.routes:
            match, _ = route.matches(scope)
            if match == Match.FULL:
                return getattr(route, "path", None) in {
                    "/v1/models/{model_id:path}",
                    "/v1/videos/{video_id}",
                    "/v1/videos/{video_id}/content",
                }
    return False


class LANInferenceMiddleware:
    def __init__(self, app, *, enabled: bool | None = None, router=None):
        self.app = app
        self.router = router
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
        if not _allowed(scope, self.router):
            response = JSONResponse(
                {"detail": "This route is available only on this Mac"}, status_code=403
            )
        else:
            try:
                # Even probes and routes with no auth dependency require a key.
                if not get_config().api_key:
                    raise HTTPException(401, "LAN access requires an API key")
                await verify_api_key_for_scope(scope)
            except HTTPException as exc:
                response = JSONResponse(
                    {"detail": exc.detail}, status_code=exc.status_code
                )
            else:
                return await self.app(scope, receive, send)
        await response(scope, receive, send)
