"""C-01: ``_disconnect_guard`` must force-call ``scheduler.abort_request``
on client disconnect.

Astrid r3 (pydantic-ai + Qwen3.5-4B-MLX-4bit) ran
``agent.run_stream("Tell me a joke")``; the model failed to emit EOS
and ran past 6144 tokens. The httpx side raised
``RemoteProtocolError: peer closed connection`` after ~35s but the
server's ``_disconnect_guard`` polled 70+ times without intervening,
because Starlette's ``request.is_disconnected()`` never returned
True for this combination. Even when the disconnect signal DID fire
in other situations, the abort relied solely on the
``generator.aclose()`` cascade unwinding through
``stream_generate.finally`` to reach ``scheduler.abort_request`` —
which is fine for a graceful shutdown but doesn't bound how long
the upstream batch step keeps consuming GPU between the cancel and
the next yield boundary.

This module nails the C-01 contract: when a disconnect is detected
AND the engine has published its admitted ``request_id`` into the
``request_id_holder``, the guard MUST call
``engine.scheduler.abort_request(rid)`` synchronously. The cascade
remains as belt-and-suspenders but is no longer the primary signal.
"""

from __future__ import annotations

import asyncio
import time

import pytest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeScheduler:
    """Captures every ``abort_request`` call for inspection."""

    def __init__(self):
        self.aborts: list[str] = []

    def abort_request(self, request_id: str) -> bool:
        self.aborts.append(request_id)
        return True


class _FakeEngine:
    """Engine stub exposing ``scheduler.abort_request`` and the
    admission-release hook the disconnect_guard's pre-C-01 contract
    already relied on.
    """

    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.admission_released = False

    def release_admission_reservation(self) -> None:
        self.admission_released = True


class _StoppingRequest:
    """ASGI Request stub whose ``is_disconnected()`` flips True after
    ``after`` seconds, simulating uvicorn delivering ``http.disconnect``.
    """

    def __init__(self, after: float):
        self._t0 = time.monotonic()
        self._after = after

    async def is_disconnected(self) -> bool:
        return time.monotonic() - self._t0 > self._after


class _NeverDisconnectsRequest:
    """ASGI Request stub that NEVER reports disconnect — exactly
    Astrid r3's failure mode where ``is_disconnected()`` returns
    False for the entire ~35s runaway generation. Used for the
    GeneratorExit-path tests where the consumer (StreamingResponse)
    tears down via ``aclose()`` without the disconnect channel
    firing.
    """

    async def is_disconnected(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_fires_force_abort_via_scheduler():
    """When a disconnect is detected mid-stream AND the engine has
    published its scheduler request id into the holder, the guard
    MUST synchronously call ``scheduler.abort_request(rid)``.

    Astrid r3 fingerprint: pre-C-01, the abort relied on
    ``generator.aclose()`` cascading through ``stream_generate.finally``
    to reach the scheduler. This test pins the contract that the
    guard calls into the scheduler DIRECTLY the moment disconnect
    fires — so the very next ``step()`` drops the request.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _FakeEngine()
    holder: list[str | None] = ["req-runaway-abc"]

    async def _never_ending_stream():
        # Yield one chunk, then block forever — exercises the
        # disconnect branch (not StopAsyncIteration).
        yield "data: hello\n\n"
        await asyncio.sleep(60.0)
        yield "data: never\n\n"

    chunks = []
    t0 = time.monotonic()
    async for chunk in _disconnect_guard(
        _never_ending_stream(),
        _StoppingRequest(after=0.15),
        poll_interval=0.05,
        engine=engine,
        keepalive_seconds=0.0,  # disable keepalive so the test isn't noisy
        request_id_holder=holder,
    ):
        chunks.append(chunk)
    elapsed = time.monotonic() - t0

    # Disconnect must short-circuit promptly, not wait for the
    # 60-second sleep.
    assert elapsed < 2.0, f"guard hung for {elapsed:.2f}s after disconnect"
    # The first chunk made it through; the second one is behind the
    # 60s sleep and must NEVER appear.
    assert chunks == ["data: hello\n\n"], chunks
    # The contract: scheduler.abort_request was called with the
    # holder's request id. The guard fires force-abort both on the
    # disconnect branch AND in the ``finally`` (belt-and-suspenders
    # for non-disconnect exits) — Scheduler.abort_request is
    # idempotent against duplicate enqueues per its docstring.
    assert len(engine.scheduler.aborts) >= 1, engine.scheduler.aborts
    assert all(rid == "req-runaway-abc" for rid in engine.scheduler.aborts), (
        engine.scheduler.aborts
    )
    # Pre-C-01 admission release path still runs.
    assert engine.admission_released is True


@pytest.mark.asyncio
async def test_disconnect_with_unknown_request_id_is_noop():
    """Holder is provided but the engine never published a request
    id into it (e.g. client disconnected before ``add_request``
    returned). The guard MUST NOT crash and MUST NOT make up a
    request id — abort is simply skipped.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _FakeEngine()
    holder: list[str | None] = [None]  # never populated

    async def _gen():
        yield "data: hi\n\n"
        await asyncio.sleep(60.0)

    chunks = []
    async for chunk in _disconnect_guard(
        _gen(),
        _StoppingRequest(after=0.1),
        poll_interval=0.05,
        engine=engine,
        keepalive_seconds=0.0,
        request_id_holder=holder,
    ):
        chunks.append(chunk)

    assert chunks == ["data: hi\n\n"]
    # No abort was called because the holder was empty — the
    # cascade through aclose() is the only remaining defense and
    # that's fine for the "never admitted" case.
    assert engine.scheduler.aborts == []
    assert engine.admission_released is True


@pytest.mark.asyncio
async def test_no_holder_preserves_pre_c01_contract():
    """When the caller passes ``request_id_holder=None`` (the
    pre-C-01 contract), the guard MUST behave exactly as before:
    no force-abort, only the admission-release safety net runs.

    Pinning this prevents the C-01 fix from accidentally requiring
    every existing caller to update its signature.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _FakeEngine()

    async def _gen():
        yield "data: hi\n\n"
        await asyncio.sleep(60.0)

    chunks = []
    async for chunk in _disconnect_guard(
        _gen(),
        _StoppingRequest(after=0.1),
        poll_interval=0.05,
        engine=engine,
        keepalive_seconds=0.0,
    ):
        chunks.append(chunk)

    assert chunks == ["data: hi\n\n"]
    assert engine.scheduler.aborts == []
    assert engine.admission_released is True


@pytest.mark.asyncio
async def test_generator_exit_branch_force_aborts_before_finally(monkeypatch):
    """C-01 codex r1 BLOCKING #1: pin the ``except GeneratorExit``
    branch specifically (not just "abort happened somewhere").

    The disconnect_guard now triggers force-abort from THREE places:
    the explicit disconnect branch, the ``except GeneratorExit``
    block, AND the ``finally`` belt-and-suspenders. A test that only
    checks ``scheduler.aborts == [rid]`` would still pass if the
    ``except GeneratorExit`` block were deleted, because ``finally``
    would catch the case. This test discriminates by replacing
    ``_force_abort_request`` with a wrapper that captures the calling
    frame's local variable namespace — only the ``except GeneratorExit``
    block (NOT ``finally``) accumulates ``chunk_count`` AND has yet to
    cancel ``disconnect_task``. Asserting that the FIRST abort call
    came from a frame WITHOUT ``finished_normally`` resolved/with
    ``disconnect_task`` still live pins the branch specifically.

    The Astrid r3 fingerprint: Starlette's StreamingResponse tears
    down via ``aclose()`` (GeneratorExit into our wrapper) when uvicorn
    detects a write failure to a dead socket — even though
    ``is_disconnected()`` returned False the entire time. Catching the
    branch here closes the second half of the runaway window.
    """
    import inspect

    from vllm_mlx.service import helpers as _helpers

    engine = _FakeEngine()
    holder: list[str | None] = ["req-runaway-xyz"]

    # Record at which suite the abort fired. The except-GeneratorExit
    # branch runs BEFORE the finally clause; we identify the calling
    # frame by its source line via ``inspect``.
    call_sites: list[str] = []
    original = _helpers._force_abort_request

    def _capturing_force_abort(eng, hld):
        # Walk back to the disconnect_guard frame and capture which
        # block we're in. The ``except GeneratorExit`` branch and the
        # ``finally`` clause sit at distinct source lines in helpers.py.
        frame = inspect.currentframe().f_back
        if frame is not None:
            call_sites.append(f"{frame.f_code.co_name}:line-{frame.f_lineno}")
        return original(eng, hld)

    monkeypatch.setattr(_helpers, "_force_abort_request", _capturing_force_abort)

    async def _gen():
        yield "data: chunk1\n\n"
        yield "data: chunk2\n\n"
        # Sleep so the consumer has time to aclose() us before we
        # try to yield more.
        await asyncio.sleep(30.0)
        yield "data: never\n\n"

    agen = _helpers._disconnect_guard(
        _gen(),
        _NeverDisconnectsRequest(),
        poll_interval=0.05,
        engine=engine,
        keepalive_seconds=0.0,
        request_id_holder=holder,
    )

    # Pull two chunks then close the wrapper from outside —
    # simulates StreamingResponse tearing down on a dead socket.
    first = await agen.__anext__()
    second = await agen.__anext__()
    await agen.aclose()

    assert first == "data: chunk1\n\n"
    assert second == "data: chunk2\n\n"
    assert len(engine.scheduler.aborts) >= 1, engine.scheduler.aborts
    assert all(rid == "req-runaway-xyz" for rid in engine.scheduler.aborts), (
        engine.scheduler.aborts
    )
    assert engine.admission_released is True

    # The discriminating assertion: at least TWO distinct source
    # lines fired _force_abort_request — the except-GeneratorExit
    # block AND the finally belt-and-suspenders. If a future refactor
    # deletes the GeneratorExit branch, the call_sites would collapse
    # to ONE distinct line (the finally only) and this test would
    # flag it. (The two source lines live within the same function
    # ``_disconnect_guard``, so co_name is identical; the
    # discrimination comes from the f_lineno.)
    distinct_lines = {site for site in call_sites if "_disconnect_guard" in site}
    assert len(distinct_lines) >= 2, (
        "expected force-abort to fire from BOTH the except-GeneratorExit "
        "branch AND the finally; if only one distinct call site appears, "
        "one of the branches is missing. call_sites="
        f"{call_sites} distinct_lines={distinct_lines}"
    )


@pytest.mark.asyncio
async def test_finally_does_not_force_abort_on_normal_stream_exhaustion(
    monkeypatch,
):
    """C-01 codex r1 NIT #3: on a normal stream exhaustion
    (``StopAsyncIteration`` — generator yielded everything and
    returned cleanly), the ``finally`` block MUST NOT force-abort.

    Pre-NIT-fix the ``finally`` enqueued an abort on every exit path
    including successful streams, making the abort logs/metrics
    indistinguishable from real disconnect cleanup AND uselessly
    polluting the scheduler's ``_pending_abort_ids`` set with
    already-finished ids. The fix is to track a
    ``finished_normally`` flag and skip the belt-and-suspenders
    when set.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _FakeEngine()
    holder: list[str | None] = ["req-normal-exit"]

    async def _short_stream():
        yield "data: chunk1\n\n"
        yield "data: chunk2\n\n"
        # Generator returns cleanly — StopAsyncIteration.

    chunks = []
    async for chunk in _disconnect_guard(
        _short_stream(),
        _NeverDisconnectsRequest(),
        poll_interval=0.05,
        engine=engine,
        keepalive_seconds=0.0,
        request_id_holder=holder,
    ):
        chunks.append(chunk)

    assert chunks == ["data: chunk1\n\n", "data: chunk2\n\n"]
    # The contract: NO force-abort fired — the scheduler already
    # marked the request as finished_normally inside the cascade,
    # and a defensive abort here would just pollute logs.
    assert engine.scheduler.aborts == [], engine.scheduler.aborts
    # Admission release still happens on every exit path.
    assert engine.admission_released is True


@pytest.mark.asyncio
async def test_force_abort_is_idempotent_against_double_call():
    """The pre-C-01 cascade through ``stream_generate.finally``
    ALSO calls ``scheduler.abort_request``. With the new explicit
    force-abort firing first, the same request id may land in
    ``_pending_abort_ids`` twice. The contract on
    ``Scheduler.abort_request`` is that this is idempotent
    (``set.add`` of an already-present id is a no-op), so the
    guard MUST NOT crash on double-abort.

    Pinning this so a future refactor of the abort signal can't
    accidentally reintroduce a duplicate-key error.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _DoubleAbortScheduler:
        def __init__(self):
            self.calls = 0

        def abort_request(self, rid: str) -> bool:
            self.calls += 1
            return True

    class _Engine:
        def __init__(self):
            self.scheduler = _DoubleAbortScheduler()

    engine = _Engine()
    holder = ["req-id-1"]

    assert _force_abort_request(engine, holder) is True
    assert _force_abort_request(engine, holder) is True
    assert engine.scheduler.calls == 2


@pytest.mark.asyncio
async def test_force_abort_resolves_sync_scheduler_via_inner_engine():
    """C-01 codex r1 BLOCKING #2: the helper MUST reach the SYNC
    scheduler entry point for engines that hide it behind a private
    inner backend (e.g. ``BatchedEngine`` over ``AsyncEngineCore`` —
    where ``engine.scheduler`` doesn't exist but
    ``engine._engine.scheduler.abort_request`` does and is sync).

    Pre-BLOCKING-#2 fix, the helper would have fallen straight to
    the public async ``engine.abort_request`` and fired-and-forget
    the coroutine; the abort would NOT have landed in the scheduler
    by the time disconnect handling returned. The walk through
    ``engine._engine.scheduler.abort_request`` closes that gap.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncScheduler:
        def __init__(self):
            self.aborts: list[str] = []

        def abort_request(self, rid: str) -> bool:
            self.aborts.append(rid)
            return True

    class _AsyncEngineCoreLike:
        def __init__(self):
            self.scheduler = _SyncScheduler()

        async def abort_request(self, rid: str) -> bool:  # noqa: ARG002
            # Async — pre-fix this is what the helper would have used,
            # missing the sync path on ``self.scheduler``.
            raise AssertionError(
                "_force_abort_request should NOT reach the async public "
                "abort path when a sync inner scheduler is available"
            )

    class _BatchedEngineLike:
        def __init__(self):
            self._engine = _AsyncEngineCoreLike()

        async def abort_request(self, rid: str) -> bool:  # noqa: ARG002
            raise AssertionError(
                "_force_abort_request should NOT reach the async public "
                "abort path when a sync inner scheduler is available"
            )

    engine = _BatchedEngineLike()
    holder = ["req-via-inner-sched"]

    assert _force_abort_request(engine, holder) is True
    # Sync path: the inner scheduler MUST have recorded the abort
    # synchronously by the time _force_abort_request returns —
    # without yielding to the event loop.
    assert engine._engine.scheduler.aborts == ["req-via-inner-sched"]


@pytest.mark.asyncio
async def test_force_abort_resolves_sync_mllm_scheduler():
    """C-01 codex r1 BLOCKING #2: the helper MUST also reach the
    sync ``_mllm_scheduler.abort_request`` on engines where the
    MLLM backend is the active path (``BatchedEngine`` with
    ``_is_mllm=True``). Symmetric to the text-path resolution
    above.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncMLLMScheduler:
        def __init__(self):
            self.aborts: list[str] = []

        def abort_request(self, rid: str) -> bool:
            self.aborts.append(rid)
            return True

    class _BatchedEngineLikeMLLM:
        def __init__(self):
            self._mllm_scheduler = _SyncMLLMScheduler()

        async def abort_request(self, rid: str) -> bool:  # noqa: ARG002
            raise AssertionError(
                "should not reach async public abort when sync MLLM path is available"
            )

    engine = _BatchedEngineLikeMLLM()
    holder = ["req-via-mllm-sched"]

    assert _force_abort_request(engine, holder) is True
    assert engine._mllm_scheduler.aborts == ["req-via-mllm-sched"]


@pytest.mark.asyncio
async def test_force_abort_async_only_fallback_returns_false():
    """C-01 codex r1 BLOCKING #2: when the engine only exposes an
    async ``abort_request`` and NO sync scheduler is reachable, the
    helper MUST signal that fact by returning ``False`` (not
    ``True``) — even though it schedules the coroutine
    fire-and-forget. The route's force-abort contract is "the
    abort is in flight in the scheduler by the time we return";
    a coroutine that hasn't run yet doesn't satisfy that, and
    pretending otherwise misleads downstream tests / metrics.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    abort_calls: list[str] = []

    class _AsyncOnlyEngine:
        # No ``scheduler``, no ``_engine.scheduler``, no
        # ``_mllm_scheduler`` — only the public async entry point.
        async def abort_request(self, rid: str) -> bool:
            abort_calls.append(rid)
            return True

    engine = _AsyncOnlyEngine()
    holder = ["req-via-async-only"]

    # Returns False because the abort did NOT land synchronously.
    assert _force_abort_request(engine, holder) is False
    # Yield to the event loop so the fire-and-forget coro runs.
    await asyncio.sleep(0)
    # The coro DID get scheduled (best-effort), but the contract
    # signal is "False = abort not guaranteed in flight at return".
    assert abort_calls == ["req-via-async-only"]


@pytest.mark.asyncio
async def test_force_abort_swallows_scheduler_exception():
    """If ``scheduler.abort_request`` raises (engine in a broken
    state, scheduler shut down mid-stream), the guard MUST log and
    continue — disconnect handling cannot itself derail. The
    cascade through aclose() in ``finally`` is the remaining
    safety net.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _BrokenScheduler:
        def abort_request(self, rid: str) -> bool:
            raise RuntimeError("scheduler dead")

    class _Engine:
        def __init__(self):
            self.scheduler = _BrokenScheduler()

    engine = _Engine()
    holder = ["req-abc"]

    # MUST NOT raise.
    result = _force_abort_request(engine, holder)
    # Returns False because the exception was swallowed; caller
    # treats that as "I tried, cascade is the remaining defense".
    assert result is False


# ---------------------------------------------------------------------------
# End-to-end: engine publishes its request_id into the holder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batched_engine_publishes_request_id_into_holder():
    """End-to-end pin: when the route passes a ``request_id_holder``
    kwarg into ``BatchedEngine.stream_generate``, the engine MUST
    populate ``holder[0]`` with the scheduler-issued request id the
    moment ``add_request`` returns.

    Without this, the disconnect_guard has nothing to abort — the
    force-abort path is a no-op. Pinning the contract end-to-end so
    a future refactor of the engine's add_request return shape
    can't silently break C-01.
    """
    from unittest.mock import MagicMock

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.request import RequestOutput

    # Build a BatchedEngine instance just enough for stream_generate
    # to reach add_request → publish → stream_outputs.
    eng = BatchedEngine.__new__(BatchedEngine)
    eng._loaded = True
    eng._is_mllm = False
    eng._mllm_scheduler = None
    eng._stream_interval = 1
    eng._is_hybrid_model = lambda: False
    eng._create_output_router = lambda: None

    # Fake AsyncEngineCore so add_request returns a concrete id and
    # stream_outputs yields one finished chunk.
    fake_engine = MagicMock()
    fut: asyncio.Future = asyncio.Future()
    fut.set_result("req-engine-issued-7777")
    fake_engine.add_request = MagicMock(return_value=fut)

    async def stream_outputs(request_id):
        yield RequestOutput(
            request_id=request_id,
            new_token_ids=[1],
            new_text="x",
            output_token_ids=[1],
            output_text="x",
            finished=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )

    fake_engine.stream_outputs = stream_outputs
    fake_engine.scheduler = MagicMock()
    fake_engine.scheduler.abort_request = MagicMock(return_value=True)
    fake_engine._cleanup_request = MagicMock()
    eng._engine = fake_engine

    holder: list[str | None] = [None]
    async for _ in eng.stream_generate(
        prompt="hi",
        max_tokens=8,
        request_id_holder=holder,
    ):
        # Once we've consumed one chunk the engine must already
        # have populated the holder — add_request returned before
        # stream_outputs started yielding.
        assert holder[0] == "req-engine-issued-7777", holder
        break

    assert holder[0] == "req-engine-issued-7777"
