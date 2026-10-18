# SPDX-License-Identifier: Apache-2.0
"""Shared worker lane for retrieval-model MLX operations."""

import asyncio
import concurrent.futures
import threading

from ..errors import BackpressureError

MAX_PENDING_RETRIEVAL_JOBS = 64
_retrieval_executor: concurrent.futures.ThreadPoolExecutor | None = None
_pending_jobs = 0
_worker_lock = threading.Lock()


class RetrievalWorkerBusyError(BackpressureError):
    """The retrieval lane has reached its pending request limit."""


def _executor():
    """Called with the worker lock held."""
    global _retrieval_executor
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
            # A detached executor must not decrement a new lifecycle's count.
            if _retrieval_executor is executor:
                _pending_jobs -= 1

    future.add_done_callback(completed)
    # Client cancellation must not cancel queued executor work or release its slot.
    return await asyncio.shield(asyncio.wrap_future(future))


def run_on_mlx_worker_sync(func, /, *args, **kwargs):
    """Blocking model loads bypass request admission on the same owning lane."""
    with _worker_lock:
        future = _executor().submit(func, *args, **kwargs)
    return future.result()


def shutdown_mlx_worker(wait: bool = True):
    """Detach the lifecycle's executor; subsequent use creates a fresh lane."""
    global _retrieval_executor, _pending_jobs
    with _worker_lock:
        executor = _retrieval_executor
        _retrieval_executor = None
        _pending_jobs = 0
    if executor is not None:
        executor.shutdown(wait=wait)
