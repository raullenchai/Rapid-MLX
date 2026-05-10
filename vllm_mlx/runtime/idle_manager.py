# SPDX-License-Identifier: Apache-2.0
"""Idle unload manager.

Tracks request activity. When the model has been idle for `timeout` seconds
with zero in-flight requests, calls `unload_fn` to free GPU memory. The next
request triggers `reload_fn` via `ensure_loaded()` before being dispatched.

Intended usage:
    idle = get_idle_manager()
    idle.configure(timeout=60, reload_fn=..., unload_fn=..., is_loaded_fn=...)
    idle.start()  # background asyncio task
    # In ASGI middleware:
    await idle.ensure_loaded()
    idle.request_start()
    try: ...
    finally: idle.request_end()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class IdleManager:
    def __init__(self) -> None:
        self.timeout: float = 0.0
        self.enabled: bool = False
        self._last_activity: float = time.monotonic()
        self._inflight: int = 0
        self._lock: asyncio.Lock | None = None
        self._loop_task: asyncio.Task | None = None
        self._reload_fn: Callable[[], Awaitable[None]] | None = None
        self._unload_fn: Callable[[], Awaitable[None]] | None = None
        self._is_loaded_fn: Callable[[], bool] | None = None

    def configure(
        self,
        timeout: float,
        reload_fn: Callable[[], Awaitable[None]],
        unload_fn: Callable[[], Awaitable[None]],
        is_loaded_fn: Callable[[], bool],
    ) -> None:
        self.timeout = float(timeout)
        self.enabled = self.timeout > 0
        self._reload_fn = reload_fn
        self._unload_fn = unload_fn
        self._is_loaded_fn = is_loaded_fn
        # Reset clock so the idle window starts after configure() (typically
        # post-warmup), not at module import.
        self._last_activity = time.monotonic()

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    def request_start(self) -> None:
        self._inflight += 1
        self.touch()

    def request_end(self) -> None:
        if self._inflight > 0:
            self._inflight -= 1
        self.touch()

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def ensure_loaded(self) -> None:
        if not self.enabled or self._is_loaded_fn is None or self._reload_fn is None:
            return
        if self._is_loaded_fn():
            return
        async with self._ensure_lock():
            if self._is_loaded_fn():
                return
            logger.info("[idle] Reloading model after idle unload...")
            t0 = time.monotonic()
            await self._reload_fn()
            self.touch()
            logger.info(f"[idle] Model reloaded in {time.monotonic() - t0:.1f}s")

    async def _maybe_unload(self) -> None:
        if (
            not self.enabled
            or self._is_loaded_fn is None
            or self._unload_fn is None
        ):
            return
        if not self._is_loaded_fn():
            return
        if self._inflight > 0:
            return
        if self.idle_seconds < self.timeout:
            return
        async with self._ensure_lock():
            if self._inflight > 0:
                return
            if not self._is_loaded_fn():
                return
            if self.idle_seconds < self.timeout:
                return
            idle_for = self.idle_seconds
            logger.info(
                f"[idle] Idle {idle_for:.0f}s ≥ {self.timeout:.0f}s — "
                "unloading model to free memory"
            )
            t0 = time.monotonic()
            try:
                await self._unload_fn()
            except Exception as e:
                logger.error(f"[idle] unload failed: {e}", exc_info=True)
                return
            logger.info(f"[idle] Model unloaded in {time.monotonic() - t0:.1f}s")

    async def _loop(self) -> None:
        try:
            interval = max(1.0, min(5.0, self.timeout / 4))
            while True:
                await asyncio.sleep(interval)
                try:
                    await self._maybe_unload()
                except Exception as e:
                    logger.warning(f"[idle] check error: {e}", exc_info=True)
        except asyncio.CancelledError:
            return

    def start(self) -> None:
        if not self.enabled or self._loop_task is not None:
            return
        try:
            self._loop_task = asyncio.create_task(self._loop())
            logger.info(
                f"[idle] manager started (timeout={self.timeout:.0f}s)"
            )
        except RuntimeError:
            logger.warning("[idle] no running loop, manager not started")

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
            self._loop_task = None


_idle_manager = IdleManager()


def get_idle_manager() -> IdleManager:
    return _idle_manager
