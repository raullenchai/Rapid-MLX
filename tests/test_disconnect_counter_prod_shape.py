# SPDX-License-Identifier: Apache-2.0
"""D-M01-DEAD + D-M01-2X regression: pin counter behaviour on the
PRODUCTION ``BatchedEngine`` over ``AsyncEngineCore`` shape.

Background
----------
0.8.2 dogfood (3 independent personas) reported on PyPI 0.8.2:

* **D-M01-DEAD**: ``rapid_mlx_requests_cancelled_via_disconnect_total``
  NEVER increments under real client disconnect on the production
  ``rapid-mlx serve`` engine shape, even though
  ``[disconnect_guard] force-abort`` warnings fire every time.
* **D-M01-2X**: ``rapid_mlx_requests_cancelled_total`` over-counts 2x
  on every BatchedEngine disconnect — 20 ticks for 10 actual aborts.

PR #783's 9 rounds of codex review covered SYNTHETIC engine shapes
(``engine.scheduler`` directly, or ``engine._engine.scheduler`` one
hop deep). The real production shape is ``BatchedEngine._engine`` →
``AsyncEngineCore.engine`` → ``EngineCore.scheduler`` — TWO hops past
``_engine``, where ``AsyncEngineCore`` does NOT expose ``.scheduler``
directly. The previous tests under
``tests/test_disconnect_guard_aborts_scheduler.py`` and
``tests/test_cancelled_requests_metric.py`` did NOT reproduce this
shape, so the resolvers + dedupe-ledger race went uncaught.

The two root causes
-------------------
1. ``_resolve_sync_scheduler_for_abort`` only walked one hop past
   ``engine._engine`` and missed the production deep path. Result:
   ``_force_abort_request`` falls into the async fallback every time,
   logs the "force-abort fell back to async" warning, and never
   reaches the sync scheduler entry directly.

2. ``Scheduler.remove_finished_request`` discarded the lifetime
   ledgers (``_cancelled_request_ids`` and ``_disconnect_abort_ids``)
   when called by ``EngineCore._cleanup_request``. The deferred
   ``_await_and_record`` coroutine that runs after the async fallback
   then sees a wiped ledger and:
     * RE-INCREMENTS the total counter on the second
       ``scheduler.abort_request`` call (the id is still in
       ``_pending_abort_ids`` so the membership check passes, but
       ``already_counted`` reads False from the cleared ledger),
       producing the 2x over-count.
     * The disconnect sub-counter then NO-OPS because
       ``record_disconnect_abort`` runs AFTER ``_cleanup_request`` has
       wiped ``_cancelled_request_ids``, so the lifetime-ledger gate
       at the top of ``record_disconnect_abort`` silently returns.

The tests below build the EXACT production engine shape with stubbed
mlx-step internals and pin both invariants.
"""

from __future__ import annotations

import asyncio

import pytest

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Identity tokenizer good enough for Scheduler.add_request."""

    eos_token_id = None

    def encode(self, prompt):
        if isinstance(prompt, str):
            return [ord(c) for c in prompt]
        return list(prompt)

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)


def _make_scheduler() -> Scheduler:
    return Scheduler(
        model=object(),
        tokenizer=_DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
        ),
    )


def _admit(scheduler: Scheduler, request_id: str) -> None:
    """Stage a request so ``abort_request`` finds it in ``self.requests``."""
    request = Request(
        request_id=request_id,
        prompt=[42],
        sampling_params=SamplingParams(max_tokens=128),
    )
    scheduler.add_request(request)


class _EngineCoreLike:
    """Mimics ``EngineCore`` for the production shape: exposes
    ``.scheduler`` and re-implements ``abort_request`` + ``_cleanup_request``
    EXACTLY as the production version does.

    The production sequence — ``self.scheduler.abort_request(rid)``
    followed by ``self._cleanup_request(rid)`` (which calls
    ``self.scheduler.remove_finished_request(rid)``) — is what races
    the deferred attribution helper. We mirror it byte-for-byte so the
    pin is faithful.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    async def abort_request(self, request_id: str) -> bool:
        result = self.scheduler.abort_request(request_id)
        self._cleanup_request(request_id)
        return result

    def _cleanup_request(self, request_id: str) -> None:
        self.scheduler.remove_finished_request(request_id)


class _AsyncEngineCoreLike:
    """Mimics ``AsyncEngineCore``: wraps an ``EngineCore``-like as
    ``self.engine`` and forwards ``abort_request``. Does NOT expose
    ``.scheduler`` directly — that's the production gap PR #783 missed.
    """

    def __init__(self, engine_core: _EngineCoreLike):
        self.engine = engine_core

    async def abort_request(self, request_id: str) -> bool:
        return await self.engine.abort_request(request_id)


class _BatchedEngineLike:
    """Mimics ``BatchedEngine`` over ``AsyncEngineCore``: exposes
    ``_engine`` (the AsyncEngineCore) and the ``_is_mllm`` flag.
    Does NOT expose ``.scheduler`` directly.

    The public ``abort_request`` mirrors ``BatchedEngine.abort_request``
    — async, awaitable, routes through ``self._engine.abort_request``.
    """

    def __init__(self, async_engine_core: _AsyncEngineCoreLike):
        self._engine = async_engine_core
        self._mllm_scheduler = None
        self._is_mllm = False

    async def abort_request(self, request_id: str) -> bool:
        return await self._engine.abort_request(request_id)


# ---------------------------------------------------------------------------
# D-M01-DEAD: via_disconnect sub-counter advances on prod shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_subcounter_advances_on_prod_engine_shape():
    """``rapid_mlx_requests_cancelled_via_disconnect_total`` MUST
    increment by exactly 1 per disconnect-driven abort on the
    production ``BatchedEngine`` over ``AsyncEngineCore`` shape.

    Pre-fix repro: PR #783's resolvers couldn't reach the deep
    scheduler from the sync abort path, so ``_force_abort_request``
    fell into the async fallback. The deferred coroutine ran
    ``EngineCore.abort_request`` which called ``scheduler.abort_request``
    AND ``_cleanup_request`` (wiping the lifetime ledger). The
    attribution helper that runs AFTER the coroutine then sees an
    empty ledger and the gate at the top of
    ``record_disconnect_abort`` silently returns. The dogfood
    fingerprint: ``via_disconnect_total`` stays flat-zero through
    every real client disconnect.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    request_id = "req-prod-shape-dead"
    _admit(scheduler, request_id)

    holder = [request_id]
    # Fire the force-abort exactly the way disconnect_guard does.
    _force_abort_request(engine, holder)
    # Allow the deferred async coroutine (if any) to run.
    for _ in range(5):
        await asyncio.sleep(0)

    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled_via_disconnect"] == 1, (
        "D-M01-DEAD: via_disconnect sub-counter did not advance on "
        f"production engine shape — got {stats}"
    )
    # Invariant: via_disconnect <= total always holds.
    assert (
        stats["num_requests_cancelled_via_disconnect"]
        <= stats["num_requests_cancelled"]
    )


# ---------------------------------------------------------------------------
# D-M01-2X: total counter increments exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_counter_no_2x_overcount_on_prod_shape():
    """``rapid_mlx_requests_cancelled_total`` MUST advance by exactly
    1 per abort on the production engine shape — not 2 (or more).

    Pre-fix repro: the disconnect_guard fires ``_force_abort_request``
    from up to three branches. Under the async fallback each fire
    schedules a ``_await_and_record`` coroutine. The coroutine calls
    ``scheduler.abort_request`` AND ``_cleanup_request``. The cleanup
    wipes ``_cancelled_request_ids``, so the NEXT
    ``scheduler.abort_request`` (from another fire branch, or from
    ``stream_outputs.finally``) sees an empty ledger and re-counts.
    The dogfood fingerprint: 10 aborts → 20 ticks.

    The pin: ONE id, multiple force-abort fires (mirroring the
    three-branch helper), total counter advances by exactly 1.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    request_id = "req-prod-shape-2x"
    _admit(scheduler, request_id)

    holder = [request_id]
    # disconnect_guard fires _force_abort_request from BOTH the
    # disconnect branch AND the finally belt-and-suspenders. Replay
    # the same two-fire pattern here.
    _force_abort_request(engine, holder)
    _force_abort_request(engine, holder)
    # Drain whatever ensure_future coroutines were scheduled.
    for _ in range(10):
        await asyncio.sleep(0)

    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 1, (
        f"D-M01-2X: total counter over-counted on production engine shape — got {stats}"
    )
    assert stats["num_requests_cancelled_via_disconnect"] == 1
    # Invariant: via_disconnect <= total always holds.
    assert (
        stats["num_requests_cancelled_via_disconnect"]
        <= stats["num_requests_cancelled"]
    )


# ---------------------------------------------------------------------------
# Combined: 10 aborts on the prod shape → 10 / 10, not 20 / 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ten_disconnects_on_prod_shape_yield_ten_ten():
    """End-to-end repro of the dogfood numbers: 10 distinct client
    disconnects on the production engine shape yield
    ``cancelled_total=10`` AND ``cancelled_via_disconnect_total=10``
    — not the 0.8.2 ``20 / 0`` shape three personas observed.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    for i in range(10):
        rid = f"req-disc-{i}"
        _admit(scheduler, rid)
        holder = [rid]
        # Two fires per disconnect to match disconnect_guard's branches
        # (disconnect + finally).
        _force_abort_request(engine, holder)
        _force_abort_request(engine, holder)

    # Drain the deferred coroutines.
    for _ in range(30):
        await asyncio.sleep(0)

    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 10, f"expected 10 aborts, got {stats}"
    assert stats["num_requests_cancelled_via_disconnect"] == 10, (
        f"expected 10 disconnect-attributed aborts, got {stats}"
    )
    assert (
        stats["num_requests_cancelled_via_disconnect"]
        <= stats["num_requests_cancelled"]
    )


# ---------------------------------------------------------------------------
# Resolver coverage — production engine shape is recognised
# ---------------------------------------------------------------------------


def test_sync_scheduler_resolver_finds_deep_prod_path():
    """The sync-abort resolver MUST find the deep production
    scheduler at ``engine._engine.engine.scheduler``. Without this,
    every disconnect falls into the async fallback and pays the
    ``_await_and_record`` race tax.
    """
    from vllm_mlx.service.helpers import _resolve_sync_scheduler_for_abort

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    abort = _resolve_sync_scheduler_for_abort(engine)
    assert abort is not None, (
        "sync abort resolver returned None for production engine shape — "
        "every disconnect will fall into the async fallback"
    )
    # Must be the actual scheduler.abort_request, not the engine's
    # async public method.
    assert abort == scheduler.abort_request


def test_disconnect_recorder_resolver_finds_deep_prod_path():
    """The attribution resolver MUST find the deep production
    scheduler at ``engine._engine.engine.scheduler`` and return its
    bound ``record_disconnect_abort`` method.
    """
    from vllm_mlx.service.helpers import _resolve_disconnect_abort_recorder

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    recorder = _resolve_disconnect_abort_recorder(engine)
    assert recorder is not None, (
        "disconnect-abort recorder resolver returned None for production "
        "engine shape — sub-counter will stay flat-zero through every "
        "real disconnect"
    )
    assert recorder == scheduler.record_disconnect_abort


# ---------------------------------------------------------------------------
# Logging guard: future engine-shape changes do not silently regress
# ---------------------------------------------------------------------------


def test_unresolved_engine_shape_logs_explicit_warning(caplog):
    """When neither resolver finds a recorder, the helper layer MUST
    emit an explicit WARNING so the next engine-shape change cannot
    silently regress the sub-counter again — which is precisely how
    D-M01-DEAD escaped PR #783's 9 codex rounds.

    Codex r10 NIT: assert the stable, actionable fragments of the
    warning (the leading tag, the "no recorder found" diagnostic,
    AND the engine type name) rather than a loose substring — so a
    future refactor that accidentally degrades the warning to a
    generic message still fails this test loudly.
    """
    import logging

    from vllm_mlx.service.helpers import _record_disconnect_abort_on_scheduler

    class _NakedEngineXYZ:
        # No .scheduler, no _engine, no _mllm_scheduler — the future
        # shape that PR #783's resolvers would silently no-op on.
        _is_mllm = False

    # Capture WARNING from whatever logger the helpers module ends up
    # bound to (rapid-mlx aliases ``vllm_mlx`` → ``rapid_mlx`` on the
    # logging tree, see runtime/__init__.py).
    caplog.set_level(logging.WARNING)
    _record_disconnect_abort_on_scheduler(_NakedEngineXYZ(), "req-naked")

    warning_records = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert warning_records, (
        "expected at least one WARNING-level record when the engine "
        f"shape exposes no recorder; got: {[r.getMessage() for r in caplog.records]}"
    )
    # The combined warning text MUST carry the stable diagnostic
    # fragments operators / dashboards key on.
    combined = " ".join(rec.getMessage() for rec in warning_records)
    assert "[disconnect_guard]" in combined, combined
    assert "no record_disconnect_abort recorder" in combined, combined
    # The engine type name is part of the actionable signal — it's
    # how an operator knows WHICH backend shape the resolvers don't
    # yet recognise.
    assert "_NakedEngineXYZ" in combined, combined
    # The "via_disconnect" diagnostic identifies WHICH metric will
    # under-count, so dashboards can flag a sticky warning.
    assert "via_disconnect" in combined, combined
