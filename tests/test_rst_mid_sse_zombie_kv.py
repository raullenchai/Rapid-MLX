# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-012: RST mid-SSE under storm leaves zombie
requests in the scheduler that consume KV cache until Metal OOM
(upstream cause of F-010 dense path and F-030 tool path).

Root cause: ``AsyncEngineCore.add_request`` awaits
``loop.run_in_executor(..., scheduler.add_request, request)`` to push
the request to the MLX worker thread. ``run_in_executor`` does NOT
propagate ``CancelledError`` into the executor task — if the awaiter is
cancelled mid-flight, the executor keeps running and ``scheduler.add_request``
may complete AFTER the route layer has already unwound. The result is a
request alive in the scheduler with no consumer; ``stream_outputs.finally``
never runs (it was never entered), so the request runs to its full
``max_tokens`` budget pinning KV slots. Under a 30-RST storm this
produces 0-30 orphans per storm, leading to unbounded Metal growth.

The fix has two layers:

* ``engine_core.add_request`` shields the executor await so cancellation
  always lands AFTER the scheduler has the request, then deferred-aborts
  + ``_cleanup_request`` on the cancellation path.

* ``batched.stream_generate`` wraps ``add_request → stream_outputs`` in
  a ``try/finally`` that defensively aborts the request when it
  unwinds, covering the narrow race between ``add_request`` returning
  and ``stream_outputs.try`` actually starting.

These tests pin both behaviours so a future refactor cannot silently
restore the leak. They drive the ``BatchedEngine`` / ``AsyncEngineCore``
abort path directly via mocks — no actual MLX inference required.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest


def _build_engine_core_mock():
    """Build a minimal ``EngineCore``-shaped mock with the fields
    ``add_request`` reads: ``scheduler``, ``_mlx_executor``,
    ``_idle_event``, the collectors / events / stream-state dicts, and
    a ``config`` carrying ``stream_interval``."""
    from vllm_mlx.engine_core import EngineCore

    eng = EngineCore.__new__(EngineCore)
    # Per-request state (allocated inside add_request)
    eng._output_collectors = {}
    eng._stream_states = {}
    eng._stream_buffers = {}
    eng._finished_events = {}
    eng._idle_event = asyncio.Event()
    # Throttle is off — set just enough to short-circuit
    eng._hybrid_throttle = False
    eng._hybrid_lock = None
    eng._last_request_time = 0.0
    # Config for RequestStreamState
    eng.config = MagicMock()
    eng.config.stream_interval = 1
    # Scheduler mock — track add_request + abort_request calls
    eng.scheduler = MagicMock()
    eng.scheduler.add_request = MagicMock()
    eng.scheduler.abort_request = MagicMock(return_value=True)
    eng.scheduler.remove_finished_request = MagicMock()
    return eng


@pytest.mark.asyncio
async def test_add_request_cancellation_aborts_after_executor_completes():
    """When ``add_request`` is cancelled while the executor is queuing
    the request, the cleanup ordering MUST be: executor finishes
    (request landed in scheduler) → THEN ``scheduler.abort_request``
    fires. If the production code aborts BEFORE the executor catches
    up, the request still lands in the scheduler later and orphans —
    the exact F-012 timing window.

    Codex r1 P1 #2: this test pins the ORDERING, not just the call.
    We record a monotonic timestamp when ``slow_add_request`` returns
    (= the moment the request is in the scheduler) and when
    ``scheduler.abort_request`` is invoked, and require the abort to
    fire AFTER the executor completes — proving the drain in the fix
    runs to completion before cleanup.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    from vllm_mlx.request import SamplingParams

    eng = _build_engine_core_mock()

    # Slow executor: simulate the MLX worker thread taking ~100ms to
    # actually run scheduler.add_request, so we can cancel mid-flight.
    started = threading.Event()
    executor_returned_at: list[float] = []
    abort_called_at: list[float] = []

    def slow_add_request(request):
        started.set()
        threading.Event().wait(0.1)
        # Record the moment the scheduler "sees" the request — the
        # ordering invariant is that abort fires AFTER this point.
        executor_returned_at.append(_time.monotonic())
        return None

    def record_abort(request_id):
        abort_called_at.append(_time.monotonic())
        return True

    eng.scheduler.add_request = MagicMock(side_effect=slow_add_request)
    eng.scheduler.abort_request = MagicMock(side_effect=record_abort)

    with ThreadPoolExecutor(max_workers=1) as pool:
        eng._mlx_executor = pool

        async def driver():
            return await eng.add_request(
                prompt="hello",
                sampling_params=SamplingParams(max_tokens=64),
            )

        task = asyncio.create_task(driver())

        # Wait for the executor to actually start, then cancel.
        await asyncio.get_event_loop().run_in_executor(None, started.wait, 1.0)
        assert started.is_set(), "executor did not start running"
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # CancelledError unwinds before the executor finishes (the
        # underlying ``concurrent.futures.Future`` is independent of
        # asyncio cancellation). Give the executor a deadline to
        # actually complete + the done-callback to fire its cleanup.
        # Without this the test would race ahead of the production
        # cleanup that runs from the done-callback on the executor
        # thread.
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline:
            if (
                executor_returned_at
                and abort_called_at
                and not eng._output_collectors
                and not eng._finished_events
                and not eng._stream_states
            ):
                break
            await asyncio.sleep(0.01)

        # Executor MUST have completed BEFORE the abort fires — the
        # core ordering invariant of the F-012 fix.
        assert executor_returned_at, (
            "executor never recorded its completion — the cleanup"
            " path in engine_core.add_request did not wait for it"
        )
        assert abort_called_at, (
            "scheduler.abort_request was never called on the"
            " cancellation path (F-012 leak path)"
        )
        assert abort_called_at[0] >= executor_returned_at[0], (
            "scheduler.abort_request fired BEFORE the executor"
            " completed (executor_returned_at="
            f"{executor_returned_at[0]:.6f},"
            f" abort_called_at={abort_called_at[0]:.6f}). "
            "The abort then races with scheduler.add_request and"
            " can fail to remove the zombie request — the exact"
            " F-012 race window."
        )

        # Per-request state (collectors, finished events, stream state)
        # MUST be released. Otherwise the dicts grow unbounded under
        # a RST storm.
        assert not eng._output_collectors, (
            "output collector left behind after cancellation cleanup"
        )
        assert not eng._finished_events, (
            "finished event left behind after cancellation cleanup"
        )
        assert not eng._stream_states, (
            "stream state left behind after cancellation cleanup"
        )


@pytest.mark.asyncio
async def test_add_request_success_path_does_not_abort():
    """Happy path: when ``add_request`` completes normally, the
    scheduler MUST NOT be aborted. Without this guard the fix could
    regress into aborting every request.
    """
    from concurrent.futures import ThreadPoolExecutor

    from vllm_mlx.request import SamplingParams

    eng = _build_engine_core_mock()

    with ThreadPoolExecutor(max_workers=1) as pool:
        eng._mlx_executor = pool

        request_id = await eng.add_request(
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=64),
        )

        assert isinstance(request_id, str) and request_id
        assert eng.scheduler.add_request.called
        assert not eng.scheduler.abort_request.called, (
            "happy path must not call abort_request — only the"
            " cancellation/error branch does"
        )
        # Collectors must remain — stream_outputs will read them.
        assert request_id in eng._output_collectors
        assert request_id in eng._finished_events


@pytest.mark.asyncio
async def test_stream_generate_finally_is_double_safety_net():
    """``stream_generate``'s belt-and-suspenders ``try/finally``
    around ``add_request → stream_outputs`` MUST call
    ``scheduler.abort_request`` + ``_cleanup_request`` even when the
    inner ``stream_outputs.finally`` did not run.

    Codex r3 P1 #2 honest naming: in real code, once
    ``stream_outputs`` has yielded its first chunk it is INSIDE its
    ``try`` block, so its ``finally`` would always abort on real-
    production aclose. The first-chunk pull below proves this
    invariant lines up with reality. This test pins the OUTER finally
    as a double-safety net — it MUST fire as well, idempotently — so
    a future refactor that breaks ``stream_outputs.finally`` (e.g.
    moves cleanup into an ``async with`` that swallows
    ``GeneratorExit``) does not silently re-open the F-012 leak.
    Idempotency is then exercised separately by the regression suite
    on the scheduler itself (``_do_abort_request`` handles
    double-abort).
    """
    from vllm_mlx.engine.batched import BatchedEngine

    eng = BatchedEngine("fake-model")
    eng._loaded = True
    eng._is_mllm = False
    eng._mllm_scheduler = None
    eng._apply_chat_template = lambda *args, **kwargs: "prompt"
    eng._compute_prefix_boundary = lambda *args, **kwargs: 0
    eng._is_hybrid_model = lambda: False
    eng._create_output_router = lambda: None

    # Fake AsyncEngineCore: add_request returns immediately;
    # stream_outputs yields ONE chunk and then awaits forever so we
    # can close the generator after the first yield.
    fake_engine = MagicMock()
    fake_engine.add_request = MagicMock(return_value=_completed_future("req-xyz"))

    async def stream_outputs(request_id):
        # Single chunk so the consumer enters the loop body
        from vllm_mlx.request import RequestOutput

        try:
            yield RequestOutput(
                request_id=request_id,
                new_token_ids=[1],
                new_text="x",
                output_token_ids=[1],
                output_text="x",
                finished=False,
                finish_reason=None,
                prompt_tokens=1,
                completion_tokens=1,
            )
            # Block on a sleep so the consumer can aclose us mid-stream
            await asyncio.sleep(10)
        finally:
            # Simulate the (intentional) failure of stream_outputs.finally
            # to abort — so we exercise stream_generate.finally as the
            # ONLY abort path. This is the worst-case state a future
            # refactor could land in.
            pass

    fake_engine.stream_outputs = stream_outputs
    fake_engine.scheduler = MagicMock()
    fake_engine.scheduler.abort_request = MagicMock(return_value=True)
    fake_engine._cleanup_request = MagicMock()
    eng._engine = fake_engine

    gen = eng.stream_generate(prompt="hi", max_tokens=16)
    aiter = gen.__aiter__()
    # Pull the first chunk so the inner async-for is engaged
    first = await aiter.__anext__()
    assert first.new_text == "x"

    # Close the generator from the outside — mimics disconnect_guard
    # calling aclose() on a TCP-RST'd SSE stream.
    await gen.aclose()

    # The belt-and-suspenders finally MUST have hit scheduler.abort_request
    # and _cleanup_request even though stream_outputs.finally
    # (faked-empty above) did not.
    assert fake_engine.scheduler.abort_request.called, (
        "stream_generate.finally must defensively abort the request"
        " on close — without this, the F-012 race window leaves a"
        " zombie in the scheduler"
    )
    assert fake_engine._cleanup_request.called, (
        "stream_generate.finally must also release per-request state"
        " when stream_outputs.finally didn't run"
    )


@pytest.mark.asyncio
async def test_add_request_pure_cancellation_before_executor_runs():
    """If the asyncio Future wrapper around the executor job IS
    cancelled before the executor thread picks it up (the
    ``cf.cancel()`` succeeds path), ``scheduler.add_request`` never
    runs and we MUST NOT call ``scheduler.abort_request`` for a
    request that was never admitted. Per-request collectors / events
    should still be released so the dicts don't grow unbounded.

    Codex r3 P1 #1: ``asyncio.wrap_future`` propagates cancellation
    to the underlying ``concurrent.futures.Future``; if the
    executor's worker has not started yet, ``cf.cancel()`` succeeds
    and the done-callback runs with ``_future.cancelled() is True``.
    We branch on that to avoid a spurious abort for an unadmitted
    request.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    from vllm_mlx.request import SamplingParams

    eng = _build_engine_core_mock()

    # No-op slow blocker on the SOLE executor thread so the next
    # submit cannot start until we release. This guarantees the
    # cancellation path hits a ``cf`` that has not yet run.
    block = threading.Event()

    def blocker():
        block.wait(2.0)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(blocker)  # parks the only worker

        eng._mlx_executor = pool

        async def driver():
            return await eng.add_request(
                prompt="hello",
                sampling_params=SamplingParams(max_tokens=64),
            )

        task = asyncio.create_task(driver())
        # Give the driver time to submit + start awaiting.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The executor thread is still blocked on ``blocker``; release
        # it so the submitted ``scheduler.add_request`` cf transitions
        # to its final state (cancelled, because nothing ran it).
        block.set()

        # Give the done-callback a beat to fire on the executor.
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline:
            if not eng._output_collectors:
                break
            await asyncio.sleep(0.01)

        # ``scheduler.add_request`` was never invoked — the cf was
        # cancelled before it ran.
        assert not eng.scheduler.add_request.called, (
            "executor job ran despite cancellation — test setup is wrong"
        )
        # And we MUST NOT have asked the scheduler to abort a request
        # that was never admitted.
        assert not eng.scheduler.abort_request.called, (
            "abort_request fired for an un-admitted request — the"
            " codex r3 P1 #1 spurious-abort path. _on_executor_done"
            " must branch on _future.cancelled()."
        )
        # Per-request state MUST still be released.
        assert not eng._output_collectors, (
            "output collector leaked on cancelled-before-run path"
        )
        assert not eng._finished_events, (
            "finished event leaked on cancelled-before-run path"
        )
    finally:
        block.set()
        pool.shutdown(wait=False, cancel_futures=True)


def _completed_future(value):
    """Helper: return a completed asyncio Future carrying ``value`` so
    ``await fake_engine.add_request(...)`` resolves synchronously."""
    fut = asyncio.get_event_loop().create_future()
    fut.set_result(value)
    return fut
