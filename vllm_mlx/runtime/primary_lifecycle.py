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
from contextvars import ContextVar

logger = logging.getLogger(__name__)

LifecycleHook = Callable[[], object | Awaitable[object]]
StateHook = Callable[[str], object]
_request_token_context: ContextVar[tuple[int, object] | None] = ContextVar(
    "primary_model_request_token", default=None
)


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
        self._load_total = 0
        self._load_failures_total = 0
        self._last_load_duration_seconds: float | None = None
        self._unload_total_by_reason: dict[str, int] = {}
        self._last_unload_reason: str | None = None
        self._detached = False
        self._closed = False
        self._resume_required = False
        self._request_tokens: dict[object, bool] = {}
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
            or bool(self._request_tokens)
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
            "active_request_owners": len(self._request_tokens),
            "error": self._last_error,
            "load_total": self._load_total,
            "load_failures_total": self._load_failures_total,
            "last_load_duration_seconds": self._last_load_duration_seconds,
            "unload_total": sum(self._unload_total_by_reason.values()),
            "unload_total_by_reason": dict(self._unload_total_by_reason),
            "last_unload_reason": self._last_unload_reason,
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
        self._closed = True
        task = self._monitor_task
        self._monitor_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # A request waiter is cancellation-isolated from the shared model
        # load. Drain that task before lifespan teardown stops/frees the same
        # engine; cancelling an executor-backed MLX load cannot stop its worker
        # safely and would create a start/stop race.
        load_task = self._load_task
        if load_task is not None and not load_task.done():
            try:
                await asyncio.shield(load_task)
            except asyncio.CancelledError:
                # Propagate process-teardown cancellation. The caller must not
                # proceed to stop the engine concurrently with the shielded
                # load; lifespan cancellation terminates that teardown path.
                raise
            except Exception:
                logger.exception("Primary model load failed during shutdown drain")

    def detach(self) -> None:
        """Synchronously retire this coordinator during a primary handoff."""

        if self.transitioning:
            raise RuntimeError("primary model lifecycle transition is in progress")
        self._detached = True
        self._closed = True
        task = self._monitor_task
        self._monitor_task = None
        if task is not None:
            task.cancel()
        self.idle_unload_seconds = 0

    async def ensure_loaded(self) -> None:
        """Load once and let concurrent/cancelled callers share the attempt."""

        if self._closed:
            raise RuntimeError("primary model lifecycle is closed")
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
                # start() can fail after publishing part of the engine. Retry
                # the cleanup on every later request so a transient stop
                # failure cannot wedge the endpoint forever.
                await self._reset_partial_load()
            task = self._load_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._load(), name="primary-model-demand-load"
                )
                self._load_task = task

        # One disconnected client must not cancel a load shared by another
        # request.  The task remains the single transition authority.
        await asyncio.shield(task)

    def acquire_request(self) -> None:
        """Protect one accepted route from idle unload through completion."""

        if self._closed:
            raise RuntimeError("primary model lifecycle is closed")
        current = _request_token_context.get()
        if current is not None and current[0] == id(self):
            return
        token = object()
        self._request_tokens[token] = False
        _request_token_context.set((id(self), token))
        task = asyncio.current_task()
        if task is not None:

            def release_when_done(_task: asyncio.Task) -> None:
                self._release_abandoned_request(token)

            task.add_done_callback(release_when_done)
        self.touch()

    def release_request(self) -> None:
        current = _request_token_context.get()
        if current is None or current[0] != id(self):
            return
        self._release_request_token(current[1])
        _request_token_context.set(None)

    def transfer_request_to_stream(self) -> None:
        """Keep the route token alive after its handler returns an SSE body."""

        current = _request_token_context.get()
        if current is not None and current[0] == id(self):
            token = current[1]
            if token in self._request_tokens:
                self._request_tokens[token] = True

    def _release_abandoned_request(self, token: object) -> None:
        if self._request_tokens.get(token) is False:
            self._release_request_token(token)

    def _release_request_token(self, token: object) -> None:
        if self._request_tokens.pop(token, None) is not None:
            self.touch()

    async def _reset_partial_load(self) -> None:
        stop = getattr(self.engine, "stop", None)
        if not callable(stop):
            raise RuntimeError("partially loaded engine cannot be reset")
        result = stop()
        if inspect.isawaitable(result):
            await result
        await _run_hook(self._release_allocator_cache)
        if self._is_loaded():
            raise RuntimeError("partially loaded engine did not stop cleanly")

    async def _resume_admission(self) -> None:
        resume = getattr(self.engine, "resume_generation", None)
        if not callable(resume):
            raise RuntimeError("configured engine does not support resume_generation()")
        result = resume()
        if inspect.isawaitable(result):
            await result
        self._resume_required = False

    async def _load(self) -> None:
        async with self._transition_lock:
            if self._is_loaded():
                self._set_state("ready")
                self._last_error = None
                return
            self._set_state("loading")
            self._last_error = None
            self._load_total += 1
            started_at = self._clock()
            try:
                start = getattr(self.engine, "start", None)
                if not callable(start):
                    raise RuntimeError("configured engine does not support start()")
                result = start()
                if inspect.isawaitable(result):
                    await result
                if self._resume_required:
                    await self._resume_admission()
                await _run_hook(self._on_loaded)
            except BaseException as exc:
                self._load_failures_total += 1
                self._set_state("error")
                self._last_error = type(exc).__name__
                if self._is_loaded():
                    try:
                        await self._reset_partial_load()
                    except BaseException:
                        logger.exception(
                            "Failed to reset partially loaded primary engine"
                        )
                self._last_load_duration_seconds = max(0.0, self._clock() - started_at)
                raise
            self._last_load_duration_seconds = max(0.0, self._clock() - started_at)
            self._set_state("ready")
            self.touch()

    async def evict_if_idle(self) -> bool:
        """Move an idle loaded primary to standby without removing its route."""

        if (
            self._closed
            or bool(self._request_tokens)
            or self.idle_unload_seconds <= 0
            or self.state != "ready"
        ):
            return False
        if self._clock() - self._last_activity < self.idle_unload_seconds:
            return False

        async with self._transition_lock:
            if (
                bool(self._request_tokens)
                or not self._is_loaded()
                or self.state != "ready"
            ):
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
            # Close ensure_loaded()'s ready fast path before the first await.
            # A request arriving while pause_generation is in flight will take
            # a lifecycle lease and wait on this transition lock.
            self._set_state("unloading")
            if callable(pause):
                try:
                    result = pause("wait", timeout=0)
                    if inspect.isawaitable(result):
                        await result
                    paused = True
                    self._resume_required = True
                except TimeoutError:
                    self.touch()
                    self._resume_required = True
                    try:
                        await self._resume_admission()
                    except BaseException:
                        self._set_state("error")
                        self._last_error = "AdmissionResumeError"
                        raise
                    self._set_state("ready")
                    return False
                except BaseException:
                    # Cancellation can arrive after the engine closed
                    # admission but before pause_generation returned. Reopen
                    # defensively even though our local `paused` flag was not
                    # assigned yet.
                    self.touch()
                    self._resume_required = True
                    try:
                        await self._resume_admission()
                    except BaseException:
                        self._set_state("error")
                        self._last_error = "AdmissionResumeError"
                        raise
                    self._set_state("ready")
                    raise

            if self._request_tokens:
                try:
                    await self._resume_admission()
                except BaseException:
                    self._set_state("error")
                    self._last_error = "AdmissionResumeError"
                    raise
                paused = False
                self._set_state("ready")
                self.touch()
                return False

            try:
                try:
                    await _run_hook(self._before_unload)
                except asyncio.CancelledError:
                    self._set_state("ready")
                    self.touch()
                    raise
                except Exception as exc:
                    # Cache persistence is best-effort: losing this cache is
                    # preferable to silently defeating the operator's memory
                    # policy and retaining the full model indefinitely.
                    logger.exception(
                        "Primary model pre-unload hook failed; continuing unload"
                    )

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
                    try:
                        await self._resume_admission()
                    except BaseException:
                        self._set_state("error")
                        self._last_error = "AdmissionResumeError"
                        logger.exception(
                            "Failed to reopen primary admission after idle unload"
                        )
                        raise

            self._set_state("standby")
            self._last_error = None
            self._last_unload_reason = "idle"
            self._unload_total_by_reason["idle"] = (
                self._unload_total_by_reason.get("idle", 0) + 1
            )
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
