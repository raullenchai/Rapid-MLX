# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware: caps concurrent inference requests via asyncio.Semaphore.

Configured by ``--max-concurrent N``. When N > 0, only N inference requests
run in parallel; additional requests await a free slot (FIFO via the loop's
semaphore wait queue). Non-inference paths bypass the cap.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import get_config

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


class ConcurrencyMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        cfg = get_config()
        if cfg.max_concurrent <= 0 or not _is_tracked(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        sem = cfg.concurrency_semaphore
        if sem is None:
            sem = asyncio.Semaphore(cfg.max_concurrent)
            cfg.concurrency_semaphore = sem

        async with sem:
            await self.app(scope, receive, send)
