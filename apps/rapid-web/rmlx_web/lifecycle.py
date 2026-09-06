# SPDX-License-Identifier: Apache-2.0
"""Atomic admission for engine work and engine transitions."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum


class TransitionStart(str, Enum):
    STARTED = "started"
    ALREADY_PENDING = "already_pending"
    BUSY_TRANSITION = "busy_transition"
    BUSY_ACTIVITY = "busy_activity"


@dataclass(frozen=True)
class PendingTransition:
    kind: str
    model: str


class ActivityLease:
    """One engine user. Release is idempotent for stream cleanup paths."""

    def __init__(self, lifecycle: EngineLifecycle) -> None:
        self._lifecycle = lifecycle
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lifecycle._release_activity()

    def __enter__(self) -> ActivityLease:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class EngineLifecycle:
    """Single-flight transition ownership on one asyncio event loop.

    Admission methods contain no ``await``. A request therefore checks and
    mutates the gate without another request observing an intermediate state.
    """

    def __init__(self) -> None:
        self._active = 0
        self._pending: PendingTransition | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def active(self) -> int:
        return self._active

    @property
    def pending(self) -> PendingTransition | None:
        return self._pending

    def acquire_activity(self) -> ActivityLease | None:
        if self._pending is not None:
            return None
        self._active += 1
        return ActivityLease(self)

    def _release_activity(self) -> None:
        if self._active <= 0:
            raise RuntimeError("engine activity released without admission")
        self._active -= 1

    def start_transition(
        self,
        *,
        kind: str,
        model: str,
        work: Callable[[], Awaitable[None]],
    ) -> TransitionStart:
        operation = PendingTransition(kind=kind, model=model)
        if self._pending is not None:
            if self._pending == operation:
                return TransitionStart.ALREADY_PENDING
            return TransitionStart.BUSY_TRANSITION
        if self._active:
            return TransitionStart.BUSY_ACTIVITY

        self._pending = operation
        self._task = asyncio.create_task(self._run(work))
        return TransitionStart.STARTED

    async def _run(self, work: Callable[[], Awaitable[None]]) -> None:
        task = asyncio.current_task()
        try:
            await work()
        finally:
            if self._task is task:
                self._task = None
                self._pending = None

    async def shutdown(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._task is task:
            self._task = None
            self._pending = None
