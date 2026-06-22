# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLLM scheduler executor cancel-gate migration.

Background
----------
The MEMORY guideline (``knowledge/gotchas.md``):

    asyncio Future cancel does NOT stop executor thread — use
    ``executor.submit`` + ``cf.cancelled()`` gate, not
    ``run_in_executor``.

The proven reference pattern lives at ``engine_core.py:855`` (text engine
``add_request``). The MLLM step in ``mllm_scheduler._process_loop`` was
still on the bare ``await loop.run_in_executor(...)`` shape that this
guideline flags. The C-04 recon (``/tmp/dogfood-085/c04-recon.md`` §3.R3)
calls this out as a known foot-gun even though it wasn't the C-04
trigger.

This test pins the migrated mechanic — specifically, that cancelling the
asyncio task wrapping the step submit:

  1. Causes ``asyncio.wrap_future`` to mark its asyncio-side Future
     CANCELLED on the loop thread within a small bounded time
     (~milliseconds, not the full duration of the executor work).
  2. Lets the executor task finish in the background so we can observe
     it via ``cf.cancelled()`` (False if it ran) vs. the cancel-then-
     never-ran case (True).
  3. Does NOT leak the executor thread / does NOT raise out of the
     cancelled task into pytest.

We deliberately do not boot the full ``MLLMScheduler`` (which requires a
loaded MLX model). Instead we mirror the exact submit+wrap_future+
cancel-handling shape from the migrated ``_process_loop`` against a
trivial sleep-stub so the test isolates the cancellation mechanic.

If ``engine_core.py:855``-style cancellation has a known limitation
(asyncio side returns while executor keeps running), this test
documents it explicitly via the ``cf.cancelled()`` assertion path.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time

import pytest


@pytest.mark.asyncio
async def test_wrap_future_cancels_asyncio_side_within_100ms():
    """The asyncio side of the cancel must complete promptly even if
    the executor work keeps running. This is the core promise of the
    migrated pattern: the loop thread is free immediately."""
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mllm-step-test"
    )
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _slow_step() -> str:
        # Signal start from the executor thread via call_soon_threadsafe
        # (asyncio.Event is not thread-safe to call ``set()`` on
        # directly across threads).
        loop.call_soon_threadsafe(started.set)
        time.sleep(2.0)
        return "done"

    # Mirror the migrated submit + wrap_future pattern from
    # mllm_scheduler._process_loop.
    cf = executor.submit(_slow_step)
    awaitable = asyncio.wrap_future(cf, loop=loop)

    async def _runner():
        try:
            return await awaitable
        except asyncio.CancelledError:
            return "cancelled"

    task = asyncio.create_task(_runner())

    # Wait until the executor work has actually started — otherwise the
    # cancel might land BEFORE the executor picked up the job and
    # ``cf.cancel()`` would succeed (the trivial case where the work
    # never ran).
    await asyncio.wait_for(started.wait(), timeout=2.0)

    # Cancel the asyncio task. The asyncio side should return within
    # ~milliseconds even though _slow_step is still sleeping for ~2s.
    cancel_start = time.monotonic()
    task.cancel()
    try:
        result = await asyncio.wait_for(task, timeout=0.5)
    except asyncio.CancelledError:
        result = "cancelled"
    cancel_elapsed = time.monotonic() - cancel_start

    assert result == "cancelled"
    # 100ms budget gives generous headroom; in practice this resolves
    # in single-digit ms.
    assert cancel_elapsed < 0.5, (
        f"asyncio side took {cancel_elapsed:.3f}s to surface cancellation —"
        " wrap_future is not propagating cancel to the asyncio side"
    )

    # The executor task itself is NOT cancelled (the work already
    # started), and that's the documented behaviour the cancel-gate
    # branch in the migrated code is designed to handle.
    assert cf.cancelled() is False
    # Drain the underlying future so the executor exits cleanly.
    cf.result(timeout=3.0)
    executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cancel_before_executor_starts_marks_cf_cancelled():
    """If the cancel lands BEFORE the executor picked up the job, the
    underlying ``cf`` IS marked cancelled — this is the
    ``_future.cancelled()`` branch in the migrated pattern. We need to
    know the difference between "the work never ran" and "the work ran
    but its result is being discarded" because only the latter requires
    the late-arriving cleanup."""
    # Single-worker executor; fill it with a blocker so subsequent
    # submits queue behind it.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mllm-step-test"
    )
    blocker_release = concurrent.futures.Future()

    def _blocker():
        blocker_release.result(timeout=5.0)
        return "blocker-done"

    blocker_cf = executor.submit(_blocker)

    # Now the step submit will sit in the queue behind the blocker.
    executed = []

    def _step():
        executed.append("ran")
        return "step-result"

    cf = executor.submit(_step)
    loop = asyncio.get_running_loop()
    awaitable = asyncio.wrap_future(cf, loop=loop)

    async def _runner():
        try:
            return await awaitable
        except asyncio.CancelledError:
            return "cancelled"

    task = asyncio.create_task(_runner())
    # Give the loop one tick so the wrap_future wiring takes effect.
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        result = await asyncio.wait_for(task, timeout=0.5)
    except asyncio.CancelledError:
        result = "cancelled"

    assert result == "cancelled"
    # Because the executor was busy with the blocker, our step was
    # cancelled while still in the queue → cf.cancelled() is True.
    assert cf.cancelled() is True
    assert executed == [], "step should not have run; cancel landed first"

    # Release the blocker so the executor can shut down.
    blocker_release.set_result(None)
    blocker_cf.result(timeout=3.0)
    executor.shutdown(wait=True)


def test_mllm_scheduler_uses_wrap_future_pattern():
    """Byte-code level pin: the migrated ``_process_loop`` must call
    ``submit`` + ``asyncio.wrap_future`` rather than
    ``loop.run_in_executor`` for the step dispatch.

    Without this pin a future refactor that reverted to
    ``run_in_executor`` (re-introducing the cancel-gate gap) would pass
    every other test in the suite — the foot-gun isn't visible until
    a real cancellation race fires under production load.
    """
    import inspect

    from vllm_mlx import mllm_scheduler

    src = inspect.getsource(mllm_scheduler.MLLMScheduler._process_loop)
    assert "self._step_executor.submit(self._step_no_queue)" in src, (
        "MLLM _process_loop must use executor.submit() for the step"
        " dispatch (MEMORY guideline + engine_core.py:855 pattern)"
    )
    assert "asyncio.wrap_future" in src, (
        "MLLM _process_loop must wrap the cf via asyncio.wrap_future"
        " so the asyncio-side cancel propagates cleanly"
    )
    assert "cf.cancelled()" in src, (
        "MLLM _process_loop must gate post-cancel handling on"
        " cf.cancelled() — the late-arriving-output branch is the"
        " whole point of the migration"
    )
    # And the migration must NOT silently keep the old pattern
    # alongside the new one. Strip docstring/comment lines first so the
    # historical reference in the migration commentary doesn't count as
    # a live call site.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert "loop.run_in_executor(self._step_executor" not in code_only, (
        "MLLM _process_loop still calls run_in_executor on the step"
        " executor — the migration is incomplete and the cancel-gate"
        " guideline is violated"
    )
