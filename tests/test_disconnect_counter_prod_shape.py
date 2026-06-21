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

    Pre-fix repro fingerprint: the disconnect_guard
    ``_force_abort_request`` reaches the scheduler via one path,
    and ``EngineCore.stream_outputs.finally`` reaches it via a
    SECOND path (``scheduler.abort_request`` + ``_cleanup_request``
    →  ``remove_finished_request``). Pre-fix the cleanup wiped the
    lifetime ledger, so the SECOND public abort entry observed an
    empty ledger and double-counted the same lifetime. The dogfood
    fingerprint: 10 aborts → 20 ticks.

    The pin replays the EXACT production sequence: one
    ``_force_abort_request`` (the helper) followed by the
    ``EngineCore.abort_request`` path (sync ``scheduler.abort_request``
    + ``_cleanup_request``). Codex r10 BLOCKING: each fire path
    MUST be modelled distinctly — calling ``_force_abort_request``
    twice with the same holder doesn't exercise the
    "two independent abort entries" race the dogfood data captures.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    request_id = "req-prod-shape-2x"
    _admit(scheduler, request_id)

    holder = [request_id]
    # Path 1: disconnect_guard fires _force_abort_request. With the
    # D-M01-DEAD resolver fix this lands SYNC on the deep prod
    # scheduler — scheduler.abort_request returns True, counter ticks
    # to 1, ledger now carries the id.
    fired = _force_abort_request(engine, holder)
    assert fired is True, (
        "_force_abort_request must resolve to the sync deep prod "
        "scheduler — falling back to async fallback is the pre-fix shape"
    )
    # Drain any deferred coroutines scheduled by the helper.
    for _ in range(5):
        await asyncio.sleep(0)
    assert scheduler.get_stats()["num_requests_cancelled"] == 1

    # Path 2: EngineCore.stream_outputs.finally explicitly calls
    # ``scheduler.abort_request`` then ``_cleanup_request`` — this is
    # the OTHER abort entry that races the helper in production. Pre-
    # fix the cleanup wiped the ledger here, so this second entry's
    # ``already_counted`` check returned False and the counter ticked
    # to 2 (the dogfood "20/0" shape). Post-fix the ledger is
    # lifetime-persistent and the second entry is correctly dedup'd.
    second_result = scheduler.abort_request(request_id)
    # ``stream_outputs.finally`` then calls ``_cleanup_request`` →
    # ``remove_finished_request`` (we drive it explicitly here so the
    # ledger-clear semantic is what's actually under test, not
    # whether _cleanup_request fired).
    scheduler.remove_finished_request(request_id)
    # Drain any further deferred coroutines.
    for _ in range(5):
        await asyncio.sleep(0)

    stats = scheduler.get_stats()
    # The contract: even though TWO independent public-API abort
    # paths each ran sync against the scheduler (one from the
    # disconnect_guard helper, one from EngineCore's cleanup), the
    # lifetime-persistent ledger dedupes them and the counter is
    # exactly 1.
    assert stats["num_requests_cancelled"] == 1, (
        f"D-M01-2X: two distinct public-API abort paths must "
        f"dedupe to a single counter tick; got {stats} "
        f"(second_result={second_result})"
    )
    # via_disconnect was attributed by the helper path; the
    # stream_outputs.finally path is NOT a disconnect path so the
    # sub-counter stays at 1, not 2.
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

    Each iteration models BOTH abort paths the disconnect
    actually traverses in production: (1) the disconnect_guard
    helper's force-abort, AND (2) ``EngineCore.stream_outputs.finally``
    invoking ``scheduler.abort_request`` + ``_cleanup_request``.
    These run on different async surfaces in production; here we
    drive them sequentially so the lifetime-ledger dedupe contract
    is the only thing keeping the counter from ticking to 20.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    scheduler = _make_scheduler()
    engine_core = _EngineCoreLike(scheduler)
    async_core = _AsyncEngineCoreLike(engine_core)
    engine = _BatchedEngineLike(async_core)

    for i in range(10):
        rid = f"req-disc-{i}"
        _admit(scheduler, rid)
        # Path 1: helper force-abort (sync to deep prod scheduler).
        assert _force_abort_request(engine, [rid]) is True
        # Path 2: stream_outputs.finally cleanup sequence — direct
        # scheduler.abort_request followed by remove_finished_request
        # (the EngineCore._cleanup_request equivalent). Pre-fix this
        # second path is what drove the counter to 2N because the
        # cleanup wiped the dedupe ledger.
        scheduler.abort_request(rid)
        scheduler.remove_finished_request(rid)

    # Drain any deferred coroutines.
    for _ in range(30):
        await asyncio.sleep(0)

    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 10, (
        f"expected 10 aborts (lifetime-persistent ledger dedupes the "
        f"two abort paths per disconnect); got {stats}"
    )
    assert stats["num_requests_cancelled_via_disconnect"] == 10, (
        f"expected 10 disconnect-attributed aborts, got {stats}"
    )
    assert (
        stats["num_requests_cancelled_via_disconnect"]
        <= stats["num_requests_cancelled"]
    )


# ---------------------------------------------------------------------------
# MLLM twin: verify the same lifetime-persistent ledger contract holds
# ---------------------------------------------------------------------------


def test_mllm_scheduler_admit_below_cap_does_not_attributeerror_on_lock():
    """Codex r12 BLOCKING follow-up: ``MLLMScheduler.add_request``
    now acquires ``self._cancel_counter_lock`` to do the
    lifetime-ledger clear + commit under one critical section. The
    constructor MUST initialise this lock or the first multimodal
    request raises ``AttributeError`` and breaks every MLLM admit.

    Drive a below-cap admit through ``add_request`` against a stub
    that mimics the bare scheduler state (no real model needed)
    and assert the lock + commit path completes without
    AttributeError. A future refactor that drops the constructor's
    ``_cancel_counter_lock = threading.Lock()`` line fails here.
    """
    import threading

    from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

    sched = MLLMScheduler.__new__(MLLMScheduler)
    # Mirror the constructor's ``_cancel_counter_lock`` /
    # ``_cancelled_request_ids`` / ``_disconnect_abort_ids`` /
    # ``_pending_abort_ids`` / ``requests`` / ``waiting`` init,
    # exactly as the real ``__init__`` does. The minimal stub is
    # sufficient to drive ``add_request`` past the ledger clear +
    # commit critical section. Anything missing here would raise
    # AttributeError BEFORE the cap-check vs after; pinning that
    # add_request reaches the commit line is the contract.
    sched._cancel_counter_lock = threading.Lock()
    sched._cancelled_request_ids = set()
    sched._disconnect_abort_ids = set()
    sched._pending_abort_ids = set()
    sched.config = MLLMSchedulerConfig(max_concurrent_requests=4)
    sched.requests = {}
    sched.waiting = []

    # Call into the real add_request with a stub prompt — minimal
    # prompt is enough to reach the commit line; we don't care
    # whether the downstream processing succeeds, only that the
    # admission gate + ledger clear + commit ran without
    # AttributeError on the lock.
    request_id = sched.add_request(prompt="hi", request_id="req-mllm-smoke")
    assert request_id == "req-mllm-smoke"
    assert "req-mllm-smoke" in sched.requests
    assert sched.requests["req-mllm-smoke"] is not None
    # The lifetime ledger MUST be empty at this point (the clear
    # ran, no abort has fired yet).
    assert "req-mllm-smoke" not in sched._cancelled_request_ids
    assert "req-mllm-smoke" not in sched._disconnect_abort_ids


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

    from vllm_mlx.service import helpers as _helpers
    from vllm_mlx.service.helpers import _record_disconnect_abort_on_scheduler

    # Use a name unique to this test so the once-per-engine-type
    # dedupe in the helper doesn't suppress us due to a previous
    # test's stub class.
    class _NakedEngineForWarningTest:
        # No .scheduler, no _engine, no _mllm_scheduler — the future
        # shape that PR #783's resolvers would silently no-op on.
        _is_mllm = False

    # Reset the once-per-engine-type dedupe so the warning is
    # guaranteed to fire even under repeated test runs. Uses the
    # same (module, qualname) key the helper consults.
    dedupe_key = _helpers._unresolved_engine_dedupe_key(_NakedEngineForWarningTest)
    with _helpers._unresolved_engine_lock:
        _helpers._unresolved_engine_logged.discard(dedupe_key)

    # Capture WARNING from whatever logger the helpers module ends up
    # bound to (rapid-mlx aliases ``vllm_mlx`` → ``rapid_mlx`` on the
    # logging tree, see runtime/__init__.py).
    caplog.set_level(logging.WARNING)
    _record_disconnect_abort_on_scheduler(_NakedEngineForWarningTest(), "req-naked")

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
    assert "_NakedEngineForWarningTest" in combined, combined
    # The "via_disconnect" diagnostic identifies WHICH metric will
    # under-count, so dashboards can flag a sticky warning.
    assert "via_disconnect" in combined, combined


def test_unresolved_engine_warning_keyed_by_module_qualname(caplog):
    """Codex r11 NIT: the dedupe ledger MUST key on
    ``(module, qualname)``, not the leaf class name, so two
    different unresolved engine classes that happen to share a leaf
    name (e.g. nested vs. top-level, or two modules each defining
    ``BatchedEngine``) each get their own actionable warning.

    Build two distinct classes with the same leaf name by
    monkey-patching ``__qualname__`` so they share ``__name__`` but
    differ in qualified identity. Both MUST warn.
    """
    import logging

    from vllm_mlx.service import helpers as _helpers
    from vllm_mlx.service.helpers import _record_disconnect_abort_on_scheduler

    class _SameLeafA:
        _is_mllm = False

    class _SameLeafB:
        _is_mllm = False

    # Force a leaf-name collision so a leaf-keyed dedupe would
    # incorrectly suppress one of the warnings.
    _SameLeafA.__name__ = "BatchedEngineCollision"
    _SameLeafB.__name__ = "BatchedEngineCollision"

    # Reset both dedupe slots.
    key_a = _helpers._unresolved_engine_dedupe_key(_SameLeafA)
    key_b = _helpers._unresolved_engine_dedupe_key(_SameLeafB)
    with _helpers._unresolved_engine_lock:
        _helpers._unresolved_engine_logged.discard(key_a)
        _helpers._unresolved_engine_logged.discard(key_b)

    assert key_a != key_b, (
        f"dedupe key collision: {key_a} == {key_b} — qualnames "
        "should differ even when __name__ matches"
    )

    caplog.set_level(logging.WARNING)
    _record_disconnect_abort_on_scheduler(_SameLeafA(), "req-a")
    _record_disconnect_abort_on_scheduler(_SameLeafB(), "req-b")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2, (
        "Two unresolved engine classes with the same leaf name MUST "
        "each get their own actionable warning, not be silenced by a "
        f"leaf-keyed dedupe. Got {len(warnings)}: "
        f"{[r.getMessage() for r in warnings]}"
    )


def test_unresolved_engine_warning_dedupes_per_engine_type(caplog):
    """Codex r10 NIT: the unresolved-engine warning is rate-limited
    to once-per-engine-type so it does not drown the log under
    sustained cancel traffic. Repeat calls for the SAME engine type
    emit DEBUG, not WARNING.
    """
    import logging

    from vllm_mlx.service import helpers as _helpers
    from vllm_mlx.service.helpers import _record_disconnect_abort_on_scheduler

    class _NakedEngineForDedupeTest:
        _is_mllm = False

    # Reset for a clean slate.
    dedupe_key = _helpers._unresolved_engine_dedupe_key(_NakedEngineForDedupeTest)
    with _helpers._unresolved_engine_lock:
        _helpers._unresolved_engine_logged.discard(dedupe_key)

    caplog.set_level(logging.DEBUG)
    # First call: fires the WARNING.
    _record_disconnect_abort_on_scheduler(_NakedEngineForDedupeTest(), "req-1")
    # Subsequent calls: suppressed to DEBUG.
    for i in range(5):
        _record_disconnect_abort_on_scheduler(
            _NakedEngineForDedupeTest(), f"req-{i + 2}"
        )

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    # Exactly one warning, despite 6 calls. The dedupe contract is
    # what stops a sustained disconnect storm from drowning logs.
    assert len(warnings) == 1, (
        f"expected exactly 1 warning across 6 unresolved-engine calls "
        f"for the same engine type; got {len(warnings)}: "
        f"{[r.getMessage() for r in warnings]}"
    )
