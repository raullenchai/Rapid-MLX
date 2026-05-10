# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware: tracks inference request inflight + ensures model loaded.

Lazy-reload behavior: if the IdleManager has unloaded the model after an idle
window, the first inference request awaits `ensure_loaded()` (which calls the
reload function) before the route runs. Other paths (health, models, mcp,
metrics) bypass tracking entirely so they don't keep the model warm.
"""

from __future__ import annotations

import logging

from ..runtime.idle_manager import get_idle_manager

logger = logging.getLogger(__name__)

_TRACKED_PREFIXES = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/embeddings",
    "/v1/audio/",
)


def _is_tracked(path: str) -> bool:
    return any(path.startswith(p) for p in _TRACKED_PREFIXES)


class IdleMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not _is_tracked(path):
            await self.app(scope, receive, send)
            return

        idle = get_idle_manager()
        if not idle.enabled:
            await self.app(scope, receive, send)
            return

        try:
            await idle.ensure_loaded()
        except Exception as exc:
            logger.error(f"[idle] reload failed for {path}: {exc}", exc_info=True)
            from starlette.responses import JSONResponse

            response = JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"Model reload failed: {exc}",
                        "type": "model_reload_error",
                    }
                },
            )
            await response(scope, receive, send)
            return

        idle.request_start()
        try:
            await self.app(scope, receive, send)
        finally:
            idle.request_end()
