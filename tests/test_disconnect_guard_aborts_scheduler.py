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
async def test_generator_exit_path_also_force_aborts():
    """When the consumer aborts the iteration via ``aclose()`` (the
    ``GeneratorExit`` path — Starlette's StreamingResponse uses this
    when it detects a write failure), the guard MUST also
    force-abort even though the disconnect_task never fired.

    Astrid r3 fingerprint: ``is_disconnected()`` may NEVER return
    True even after the client RSTs, but uvicorn eventually
    surfaces the dead socket via a write failure that propagates
    back as ``GeneratorExit`` into the streaming generator. Catching
    this path closes the second half of the runaway window.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _FakeEngine()
    holder: list[str | None] = ["req-runaway-xyz"]

    async def _gen():
        yield "data: chunk1\n\n"
        yield "data: chunk2\n\n"
        # Sleep so the consumer has time to aclose() us before we
        # try to yield more.
        await asyncio.sleep(30.0)
        yield "data: never\n\n"

    agen = _disconnect_guard(
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
    # The contract: even on the GeneratorExit path (no disconnect
    # signal ever fired), the guard force-aborts. ``Scheduler.abort_request``
    # is idempotent, so the guard fires it both from ``except
    # GeneratorExit`` and the ``finally`` belt-and-suspenders.
    assert len(engine.scheduler.aborts) >= 1, engine.scheduler.aborts
    assert all(rid == "req-runaway-xyz" for rid in engine.scheduler.aborts), (
        engine.scheduler.aborts
    )
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
async def test_force_abort_falls_back_to_engine_abort_request():
    """When the engine doesn't expose a ``scheduler`` attribute
    (the public ``abort_request`` is the only entry point — e.g.
    BatchedEngine in some wiring), the guard MUST fall back to
    ``engine.abort_request(rid)``. Covers the async case where the
    fallback is a coroutine (BatchedEngine over AsyncEngineCore) —
    we fire-and-forget without awaiting so the disconnect path
    stays synchronous.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    abort_calls: list[str] = []

    class _NoSchedulerEngine:
        async def abort_request(self, rid: str) -> bool:
            abort_calls.append(rid)
            return True

    engine = _NoSchedulerEngine()
    holder = ["req-via-engine"]

    assert _force_abort_request(engine, holder) is True
    # Yield to the event loop so the fire-and-forget coro runs.
    await asyncio.sleep(0)
    assert abort_calls == ["req-via-engine"]


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
