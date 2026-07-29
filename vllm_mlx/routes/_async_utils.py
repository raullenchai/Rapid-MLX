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
    """
    task = asyncio.ensure_future(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Drain the worker (ignoring its outcome — nobody is listening)
        # before letting the cancellation unwind our lock + cleanup.
        try:
            await task
        except Exception:
            logger.debug("Abandoned worker finished with an error", exc_info=True)
        raise
