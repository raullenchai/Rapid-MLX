# SPDX-License-Identifier: Apache-2.0
"""Request logging middleware — logs method, path, status code, and duration.

Activated at DEBUG level on the ``vllm_mlx.middleware.request_logging``
logger (which inherits from the ``vllm_mlx`` root configured by
``--log-level``). At INFO and above the middleware short-circuits on
every request (zero-cost guard).

Log line format (single line, fields unambiguous)::

    DEBUG:vllm_mlx.middleware.request_logging:
      POST "/v1/chat/completions" 200 0.847s

The path is quoted so spaces in decoded URLs do not shift field positions.

The duration is wall-clock time measured with ``time.perf_counter()``
(sub-microsecond resolution on macOS/Linux).

Health-probe endpoints (``/health``, ``/health/ready``) are excluded
to avoid flooding the log on k8s deployments with 10-second liveness
intervals.
"""

from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger(__name__)

# Paths excluded from request logging (high-frequency probes).
_EXCLUDED_PATHS = frozenset({"/health", "/health/ready"})

# Strip control characters AND Unicode line/paragraph separators from
# attacker-controlled path strings to prevent log injection.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]+")


def _sanitize(value: str) -> str:
    """Replace control characters with ``?`` and escape quotes/backslashes."""
    cleaned = _CONTROL_RE.sub("?", value)
    return cleaned.replace("\\", "\\\\").replace('"', '\\"')


class RequestLoggingMiddleware:
    """Pure-ASGI middleware that logs every HTTP request at DEBUG level.

    Zero-cost when the logger is above DEBUG: the ``isEnabledFor`` check
    is a single integer comparison that short-circuits before any
    string formatting or timer work.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not logger.isEnabledFor(logging.DEBUG):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _EXCLUDED_PATHS:
            await self.app(scope, receive, send)
            return

        method = _sanitize(scope.get("method", "?"))
        safe_path = _sanitize(path)
        status_code = 0

        def _capture_status(original_send):
            async def _send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message.get("status", 0)
                await original_send(message)

            return _send

        start = time.perf_counter()
        try:
            await self.app(scope, receive, _capture_status(send))
        except Exception:
            # Starlette's ServerErrorMiddleware will return 500 to the
            # client — log that instead of a misleading 0.
            if status_code == 0:
                status_code = 500
            raise
        finally:
            elapsed = time.perf_counter() - start
            logger.debug('%s "%s" %d %.3fs', method, safe_path, status_code, elapsed)


def install_request_logging_middleware(app) -> None:
    """Attach :class:`RequestLoggingMiddleware` to ``app``."""
    app.add_middleware(RequestLoggingMiddleware)
