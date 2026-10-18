# SPDX-License-Identifier: Apache-2.0
"""Shared worker lane for retrieval-model MLX operations."""

import asyncio
import concurrent.futures
import threading

from ..errors import BackpressureError

MAX_PENDING_RETRIEVAL_JOBS = 64
_retrieval_executor: concurrent.futures.ThreadPoolExecutor | None = None
_draining_executor: concurrent.futures.ThreadPoolExecutor | None = None
_pending_jobs = 0
_worker_lock = threading.Lock()


class RetrievalWorkerBusyError(BackpressureError):
    """The retrieval lane is full or draining during shutdown."""


def _executor():
    """Called with the worker lock held."""
    global _retrieval_executor
    if _draining_executor is not None:
        raise RetrievalWorkerBusyError("Retrieval worker is shutting down.")
    if _retrieval_executor is None:
        _retrieval_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-retrieval"
        )
    return _retrieval_executor


async def run_on_mlx_worker(func, /, *args, **kwargs):
    """Admit bounded retrieval work and retain its slot until execution ends."""
    global _pending_jobs
    with _worker_lock:
        if _pending_jobs >= MAX_PENDING_RETRIEVAL_JOBS:
            raise RetrievalWorkerBusyError("Retrieval worker queue is full.")
        executor = _executor()
        future = executor.submit(func, *args, **kwargs)
        _pending_jobs += 1

    def completed(_future):
        global _pending_jobs
        with _worker_lock:
            # Only the current or draining lifecycle owns pending slots.
            if _retrieval_executor is executor or _draining_executor is executor:
                _pending_jobs -= 1

    future.add_done_callback(completed)
    # Client cancellation must not cancel queued executor work or release its slot.
    return await asyncio.shield(asyncio.wrap_future(future))


def run_on_mlx_worker_sync(func, /, *args, **kwargs):
    """Blocking model loads bypass request admission on the same owning lane."""
    with _worker_lock:
        future = _executor().submit(func, *args, **kwargs)
    return future.result()


def shutdown_mlx_worker():
    """Drain the owning lane before allowing a fresh executor lifecycle."""
    global _retrieval_executor, _draining_executor, _pending_jobs
    with _worker_lock:
        executor = _retrieval_executor
        if executor is not None:
            _draining_executor = executor
            _retrieval_executor = None
    if executor is not None:
        # Completion callbacks acquire the worker lock, so wait outside it.
        executor.shutdown(wait=True)
        with _worker_lock:
            _draining_executor = None
            _pending_jobs = 0
