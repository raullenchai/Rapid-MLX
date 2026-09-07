"""Demand-driven lifecycle for the configured primary model.

The resident-model manager owns routing and memory policy for additional
models.  This small coordinator handles the one model whose fully configured
``BatchedEngine`` is constructed by ``rapid-mlx serve``: it may start in
standby, load on the first request, and return to standby after an idle TTL.

The engine object and registry entry deliberately survive standby.  That keeps
the configured model identity, tokenizer/parser settings, and every serve-time
engine option intact without accepting arbitrary request-triggered downloads.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

LifecycleHook = Callable[[], object | Awaitable[object]]
StateHook = Callable[[str], object]


async def _run_hook(hook: LifecycleHook | None) -> None:
    if hook is None:
        return
    result = hook()
    if inspect.isawaitable(result):
        await result


class PrimaryModelLifecycle:
    """Serialize primary load/unload transitions and monitor idle time."""

    def __init__(
        self,
        engine: object,
        *,
        lazy_load: bool = False,
        idle_unload_seconds: float = 0,
        on_loaded: LifecycleHook | None = None,
        before_unload: LifecycleHook | None = None,
        release_allocator_cache: LifecycleHook | None = None,
        on_state_change: StateHook | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self.lazy_load = bool(lazy_load)
        self.idle_unload_seconds = max(0.0, float(idle_unload_seconds))
        self._on_loaded = on_loaded
        self._before_unload = before_unload
        self._release_allocator_cache = release_allocator_cache
        self._on_state_change = on_state_change
        self._clock = clock
        self._transition_lock = asyncio.Lock()
        self._load_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._last_activity = self._clock()
        self._last_error: str | None = None
        self._detached = False
        self.state = "ready" if self._is_loaded() else "standby"

    def _set_state(self, state: str) -> None:
        self.state = state
        if not self._detached and self._on_state_change is not None:
            self._on_state_change(state)

    @property
    def enabled(self) -> bool:
        return self.lazy_load or self.idle_unload_seconds > 0

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def model_loaded(self) -> bool:
        return self._is_loaded()

    @property
    def transitioning(self) -> bool:
        task = self._load_task
        return (
            self._transition_lock.locked()
            or self.state in {"loading", "unloading"}
            or (task is not None and not task.done())
        )

    def _is_loaded(self) -> bool:
        loaded = getattr(self.engine, "_loaded", None)
        return bool(loaded) if loaded is not None else True

    def touch(self) -> None:
        self._last_activity = self._clock()

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "model_loaded": self.model_loaded,
            "idle_seconds": max(0.0, self._clock() - self._last_activity),
            "idle_unload_seconds": self.idle_unload_seconds,
            "lazy_load": self.lazy_load,
            "error": self._last_error,
        }

    async def start(self) -> None:
        if (
            self._detached
            or self.idle_unload_seconds <= 0
            or self._monitor_task is not None
        ):
            return
        self._monitor_task = asyncio.create_task(
            self._monitor_idle(), name="primary-model-idle-unload"
        )

    async def shutdown(self) -> None:
        task = self._monitor_task
        self._monitor_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def detach(self) -> None:
        """Synchronously retire this coordinator during a primary handoff."""

        if self.transitioning:
            raise RuntimeError("primary model lifecycle transition is in progress")
        self._detached = True
        task = self._monitor_task
        self._monitor_task = None
        if task is not None:
            task.cancel()
        self.idle_unload_seconds = 0

    async def ensure_loaded(self) -> None:
        """Load once and let concurrent/cancelled callers share the attempt."""

        if self._detached:
            raise RuntimeError("primary model lifecycle is detached")
        self.touch()
        if self._is_loaded() and self.state == "ready":
            self._set_state("ready")
            self._last_error = None
            return

        async with self._transition_lock:
            if self._is_loaded() and self.state == "ready":
                self._set_state("ready")
                self._last_error = None
                return
            if self._is_loaded() and self.state == "error":
                raise RuntimeError(
                    "configured engine is still resident after a failed "
                    "lifecycle transition"
                )
            task = self._load_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._load(), name="primary-model-demand-load"
                )
                self._load_task = task

        # One disconnected client must not cancel a load shared by another
        # request.  The task remains the single transition authority.
        await asyncio.shield(task)

    async def _load(self) -> None:
        async with self._transition_lock:
            if self._is_loaded():
                self._set_state("ready")
                self._last_error = None
                return
            self._set_state("loading")
            self._last_error = None
            try:
                start = getattr(self.engine, "start", None)
                if not callable(start):
                    raise RuntimeError("configured engine does not support start()")
                result = start()
                if inspect.isawaitable(result):
                    await result
                await _run_hook(self._on_loaded)
            except BaseException as exc:
                self._set_state("error")
                self._last_error = type(exc).__name__
                raise
            self._set_state("ready")
            self.touch()

    async def evict_if_idle(self) -> bool:
        """Move an idle loaded primary to standby without removing its route."""

        if self._detached or self.idle_unload_seconds <= 0 or self.state != "ready":
            return False
        if self._clock() - self._last_activity < self.idle_unload_seconds:
            return False

        async with self._transition_lock:
            if not self._is_loaded() or self.state != "ready":
                return False
            if self._clock() - self._last_activity < self.idle_unload_seconds:
                return False

            status = getattr(self.engine, "lifecycle_status", None)
            if callable(status):
                activity = status()
                if int(activity.get("active_requests", 0) or 0) > 0:
                    # Start the idle window after the most recent observation
                    # of real work, not after request admission.
                    self.touch()
                    return False

            pause = getattr(self.engine, "pause_generation", None)
            paused = False
            if callable(pause):
                try:
                    result = pause("wait", timeout=0)
                    if inspect.isawaitable(result):
                        await result
                    paused = True
                except TimeoutError:
                    self.touch()
                    resume = getattr(self.engine, "resume_generation", None)
                    if callable(resume):
                        result = resume()
                        if inspect.isawaitable(result):
                            await result
                    return False
                except BaseException:
                    # Cancellation can arrive after the engine closed
                    # admission but before pause_generation returned. Reopen
                    # defensively even though our local `paused` flag was not
                    # assigned yet.
                    self.touch()
                    resume = getattr(self.engine, "resume_generation", None)
                    if callable(resume):
                        result = resume()
                        if inspect.isawaitable(result):
                            await result
                    raise

            self._set_state("unloading")
            try:
                try:
                    await _run_hook(self._before_unload)
                except asyncio.CancelledError:
                    self._set_state("ready")
                    self.touch()
                    raise
                except Exception as exc:
                    # Cache persistence is best-effort. A failure before stop
                    # leaves a perfectly usable resident engine and must not
                    # turn the stable endpoint into a permanent 503.
                    self._last_error = type(exc).__name__
                    self._set_state("ready")
                    self.touch()
                    logger.exception(
                        "Primary model pre-unload hook failed; keeping it resident"
                    )
                    return False

                stop = getattr(self.engine, "stop", None)
                if not callable(stop):
                    self._set_state("ready")
                    self.touch()
                    raise RuntimeError("configured engine does not support stop()")
                try:
                    result = stop()
                    if inspect.isawaitable(result):
                        await result
                except BaseException as exc:
                    self._last_error = type(exc).__name__
                    if self._is_loaded():
                        self._set_state("ready")
                        self.touch()
                    else:
                        self._set_state("error")
                    raise

                try:
                    await _run_hook(self._release_allocator_cache)
                except Exception:
                    # The engine and its model references are already gone.
                    # Allocator cleanup can reduce RSS but is not required for
                    # correctness, so remain reloadable in standby.
                    logger.exception("Primary allocator cache release failed")
            finally:
                if paused:
                    resume = getattr(self.engine, "resume_generation", None)
                    if callable(resume):
                        try:
                            result = resume()
                            if inspect.isawaitable(result):
                                await result
                        except BaseException:
                            logger.exception(
                                "Failed to reopen primary admission after idle unload"
                            )

            self._set_state("standby")
            self._last_error = None
            return True

    async def _monitor_idle(self) -> None:
        interval = min(60.0, max(1.0, self.idle_unload_seconds / 4.0))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.evict_if_idle()
            except asyncio.CancelledError:
                raise
            except BaseException:
                logger.exception("Primary model idle unload failed")
