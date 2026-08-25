"""Server-owned MLX worker dispatch and lifecycle state for audio lanes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

_T = TypeVar("_T")


class ModelWorker(Protocol):
    """Minimal execution surface exported by an inference engine."""

    async def execute_on_model_worker(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T: ...

    def execute_on_model_worker_sync(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T: ...


class AudioWorkerBusyError(RuntimeError):
    """The owning model worker cannot change while audio work is active."""


class AudioWorkerHandoff:
    """Exclusive lease for changing the model worker without split ownership."""

    def __init__(
        self,
        dispatcher: AudioWorkerDispatcher,
        token: object,
        original_worker: ModelWorker | None,
    ) -> None:
        self._dispatcher = dispatcher
        self._token = token
        self._original_worker = original_worker
        self._finished = False

    def commit(self, worker: ModelWorker | None) -> None:
        """Publish ``worker`` and release the handoff lease."""

        if self._finished:
            raise RuntimeError("audio worker handoff is already complete")
        self._dispatcher._finish_handoff(self._token, worker)
        self._finished = True

    def rollback(self) -> None:
        """Keep the original worker and release the handoff lease."""

        if self._finished:
            return
        self._dispatcher._finish_handoff(self._token, self._original_worker)
        self._finished = True


@dataclass
class AudioLaneState:
    model: str | None = None
    state: str = "registered"
    active_requests: int = 0
    loaded_at: float | None = None
    last_used_at: float | None = None
    last_error: str | None = None


class AudioWorkerDispatcher:
    """Route audio MLX work through the server's model-owning worker."""

    def __init__(self) -> None:
        self._worker: ModelWorker | None = None
        self._fallback: concurrent.futures.ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._lanes: dict[str, AudioLaneState] = {}
        self._handoff_token: object | None = None

    @staticmethod
    def _validate_worker(worker: ModelWorker | None) -> None:
        if worker is not None and (
            not callable(getattr(worker, "execute_on_model_worker", None))
            or not callable(getattr(worker, "execute_on_model_worker_sync", None))
        ):
            raise TypeError("engine does not expose the model-worker contract")

    def bind(self, worker: ModelWorker | None) -> None:
        self._validate_worker(worker)
        fallback = None
        with self._lock:
            if self._handoff_token is not None:
                raise AudioWorkerBusyError("model worker handoff is in progress")
            if worker is not self._worker and any(
                state.active_requests > 0 for state in self._lanes.values()
            ):
                raise AudioWorkerBusyError(
                    "cannot replace the model worker while audio work is active"
                )
            self._worker = worker
            fallback = self._fallback
            self._fallback = None
        if fallback is not None:
            fallback.shutdown(wait=True)

    def begin_handoff(self) -> AudioWorkerHandoff:
        """Reserve worker ownership while the primary lifecycle changes."""

        with self._lock:
            if self._handoff_token is not None:
                raise AudioWorkerBusyError("model worker handoff is in progress")
            if any(state.active_requests > 0 for state in self._lanes.values()):
                raise AudioWorkerBusyError(
                    "cannot replace the model worker while audio work is active"
                )
            token = object()
            self._handoff_token = token
            return AudioWorkerHandoff(self, token, self._worker)

    def _finish_handoff(self, token: object, worker: ModelWorker | None) -> None:
        self._validate_worker(worker)
        fallback = None
        with self._lock:
            if token is not self._handoff_token:
                raise RuntimeError("audio worker handoff lease is not active")
            self._worker = worker
            self._handoff_token = None
            fallback = self._fallback
            self._fallback = None
        if fallback is not None:
            fallback.shutdown(wait=True)

    def _bound_worker(self) -> ModelWorker | None:
        with self._lock:
            return self._worker

    def _fallback_worker(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the dedicated worker used by audio-only/non-batched servers."""

        with self._lock:
            if self._fallback is None:
                from ..engine_core import _init_mlx_step_thread

                self._fallback = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="mlx-audio",
                    initializer=_init_mlx_step_thread,
                )
            return self._fallback

    def _begin(self, lane: str, model: str, operation: str) -> None:
        now = time.monotonic()
        with self._lock:
            if self._handoff_token is not None:
                raise AudioWorkerBusyError("model worker handoff is in progress")
            state = self._lanes.setdefault(lane, AudioLaneState())
            state.model = model
            state.active_requests += 1
            state.state = "loading" if operation == "load" else "busy"
            state.last_used_at = now
            state.last_error = None

    def _finish(
        self, lane: str, model: str, operation: str, error: BaseException | None
    ) -> None:
        now = time.monotonic()
        with self._lock:
            state = self._lanes.setdefault(lane, AudioLaneState())
            state.model = model
            state.active_requests = max(0, state.active_requests - 1)
            state.last_used_at = now
            if error is None:
                state.state = "registered" if operation == "unload" else "resident"
                state.last_error = None
                if operation == "load":
                    state.loaded_at = now
                elif operation == "unload":
                    state.model = None
                    state.loaded_at = None
            else:
                state.state = "failed"
                state.last_error = type(error).__name__

    def snapshot(self) -> list[dict[str, object]]:
        """Return stable, secret-free lane state for the residency API."""

        now = time.monotonic()
        with self._lock:
            return [
                {
                    "lane": lane,
                    "model": state.model,
                    "state": state.state,
                    "active_requests": state.active_requests,
                    "loaded_at": state.loaded_at,
                    "idle_seconds": (
                        max(0.0, now - state.last_used_at)
                        if state.last_used_at is not None and state.active_requests == 0
                        else 0.0
                    ),
                    "last_error": state.last_error,
                }
                for lane, state in sorted(self._lanes.items())
            ]

    async def execute(
        self,
        lane: str,
        model: str,
        operation: str,
        func: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        async def invoke() -> _T:
            self._begin(lane, model, operation)
            error: BaseException | None = None
            try:
                worker = self._bound_worker()
                if worker is not None:
                    return await worker.execute_on_model_worker(func, *args, **kwargs)
                loop = asyncio.get_running_loop()
                executor = self._fallback_worker()
                return await loop.run_in_executor(
                    executor, lambda: func(*args, **kwargs)
                )
            except BaseException as exc:
                error = exc
                raise
            finally:
                self._finish(lane, model, operation, error)

        task = asyncio.create_task(invoke())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The callable may still be reading a request tempfile or touching
            # cached weights. Do not let the route release its lane lock and
            # cleanup resources until the worker reaches a terminal state.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            # Retrieve a terminal exception to avoid an unobserved-task
            # warning; cancellation remains the caller-visible outcome.
            try:
                task.result()
            except BaseException:
                pass
            raise

    def execute_sync(
        self,
        lane: str,
        model: str,
        operation: str,
        func: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        self._begin(lane, model, operation)
        error: BaseException | None = None
        try:
            worker = self._bound_worker()
            if worker is not None:
                return worker.execute_on_model_worker_sync(func, *args, **kwargs)
            return self._fallback_worker().submit(func, *args, **kwargs).result()
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(lane, model, operation, error)


audio_worker = AudioWorkerDispatcher()


def bind_audio_worker(worker: ModelWorker | None) -> None:
    audio_worker.bind(worker)


async def run_audio_mlx(
    lane: str,
    model: str,
    operation: str,
    func: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    return await audio_worker.execute(lane, model, operation, func, *args, **kwargs)


def run_audio_mlx_sync(
    lane: str,
    model: str,
    operation: str,
    func: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    return audio_worker.execute_sync(lane, model, operation, func, *args, **kwargs)
