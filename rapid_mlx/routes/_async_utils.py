# SPDX-License-Identifier: Apache-2.0
"""Shared async helpers for the generation route lanes (audio / video).

These lanes all have the same shape: an ``async def`` handler that must
hand seconds-to-minutes of blocking engine work to a worker thread, while
holding a lock and owning a temp file for exactly as long as that work
runs. Getting the cancellation semantics right is subtle enough that the
audio and video routes share one implementation rather than each carrying
a copy that can drift.
"""

import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


async def run_to_completion(func, /, *args):
    """``asyncio.to_thread(func, *args)`` that survives cancellation.

    A plain ``await asyncio.to_thread(...)`` is NOT cancellable — the
    worker thread keeps running — but the await returns immediately when
    the surrounding task is cancelled. That combination is dangerous for
    the generation lanes, where a client disconnect cancels the handler
    mid-render:

    * an ``async with`` lock around the await would unwind and admit
      another request while the abandoned thread is still using the
      cached engine (or running a multi-GB subprocess), destroying the
      one-at-a-time memory guarantee the lock exists to provide, and
    * a ``finally`` block would delete the temp file or directory the
      abandoned worker is still writing into.

    So on cancellation we wait for the worker to actually finish before
    propagating, keeping "lock held" and "output path alive" true for
    exactly as long as the thread is running. The client is already gone,
    so the extra wait costs nothing user-visible, and it is bounded by
    the engine's own timeout.

    The drain is a SHIELDED LOOP, not a bare ``await task``. A bare await is
    itself cancellable, so a second ``Task.cancel()`` — a shutdown signal
    racing the client disconnect, or a supervisor giving up on a hung
    request — would interrupt the drain and unwind the lock / cleanup while
    the thread still runs, the exact failure this helper exists to prevent,
    one cancel later. Re-awaiting a fresh ``asyncio.shield(task)`` and
    swallowing cancels keeps us parked on the worker until it truly
    finishes: ``shield`` protects ``task`` from each cancel, so a repeated
    cancel costs one extra loop turn instead of ending the drain.

    Worker completion is tracked by a thread-level event rather than by the
    cancellable asyncio wrapper. This also covers shutdown code that directly
    cancels every asyncio task: cancelling ``to_thread`` does not stop its
    underlying thread, so the outer task must keep draining until the worker
    itself signals completion.
    """
    completed = threading.Event()

    def invoke():
        try:
            return func(*args)
        finally:
            completed.set()

    task = asyncio.ensure_future(asyncio.to_thread(invoke))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Drain the worker before letting the cancellation unwind our lock +
        # cleanup. See the shielded-loop rationale in the docstring.
        while not completed.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                # Repeated/direct cancellation cannot stop the worker.
                logger.debug("Ignoring cancellation while draining an abandoned worker")
        # The worker sets ``completed`` in its finally block just before the
        # executor schedules the asyncio Future's completion callback. Drain
        # that small handoff window before inspecting ``task.exception()``.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                logger.debug(
                    "Ignoring cancellation while finalizing an abandoned worker"
                )
            except Exception:
                break
        # Retrieve the outcome so asyncio doesn't warn "exception never
        # retrieved" at GC, and we keep the reason the abandoned work failed.
        worker_error = None if task.cancelled() else task.exception()
        if worker_error is not None:
            logger.debug(
                "Abandoned worker finished with an error",
                exc_info=(
                    type(worker_error),
                    worker_error,
                    worker_error.__traceback__,
                ),
            )
        raise
