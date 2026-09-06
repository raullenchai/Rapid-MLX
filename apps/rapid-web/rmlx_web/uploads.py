# SPDX-License-Identifier: Apache-2.0
"""Bound upload request bodies before Starlette parses multipart forms."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from fastapi.responses import JSONResponse


class _UploadTooLargeError(Exception):
    pass


class _UploadTimedOutError(Exception):
    pass


class UploadBodyLimitMiddleware:
    """Reject oversized or stalled uploads at the ASGI receive boundary."""

    def __init__(
        self,
        app: Any,
        *,
        limits: Mapping[str, int],
        receive_timeout: float = 15.0,
    ) -> None:
        self.app = app
        self.limits = dict(limits)
        self.receive_timeout = receive_timeout

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        limit = self.limits.get(scope.get("path", ""))
        if limit is None:
            await self.app(scope, receive, send)
            return

        advertised = _content_length(scope)
        if advertised is not None and advertised > limit:
            await _error_response(413, "upload request is too large", send, scope)
            return

        total = 0

        async def bounded_receive():
            nonlocal total
            try:
                if self.receive_timeout > 0:
                    message = await asyncio.wait_for(
                        receive(), timeout=self.receive_timeout
                    )
                else:
                    message = await receive()
            except asyncio.TimeoutError:
                raise _UploadTimedOutError from None

            if message.get("type") == "http.request":
                total += len(message.get("body", b"") or b"")
                if total > limit:
                    raise _UploadTooLargeError
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except _UploadTooLargeError:
            await _error_response(413, "upload request is too large", send, scope)
        except _UploadTimedOutError:
            await _error_response(408, "upload body timed out", send, scope)


def _content_length(scope) -> int | None:
    for raw_name, raw_value in scope.get("headers", ()):
        if raw_name.lower() != b"content-length":
            continue
        try:
            value = int(raw_value.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if value >= 0 else None
    return None


async def _error_response(status: int, message: str, send, scope) -> None:
    error_type = "request_timeout" if status == 408 else "payload_too_large"
    response = JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": error_type}},
    )

    async def disconnected():
        return {"type": "http.disconnect"}

    await response(scope, disconnected, send)
