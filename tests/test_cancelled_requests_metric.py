# SPDX-License-Identifier: Apache-2.0
"""Observability tests for the M-01 cancelled-requests counter.

Background
----------
Mei r0.8.1 and Yana r3 both flagged that ``/metrics`` reported
``rapid_mlx_requests_processed_total = 0`` after fifty client-cancelled
streaming requests. ``num_requests_processed`` deliberately excludes
aborted requests (a request that never produced an EOS-bounded response
shouldn't be billed as "completed"), so operators staring at the
counter-of-record can't distinguish "model is idle" from "every caller
is bailing out before EOS". That's the M-01 gap.

The fix exposes two new scheduler counters via ``Scheduler.get_stats()``
and ``Scheduler`` MLLM-twin, rendered as Prometheus counters in
``routes/metrics.py``:

* ``rapid_mlx_requests_cancelled_total`` — +1 per public-API abort the
  scheduler accepted (client disconnect, explicit cancel route,
  timeout, or internal abort), once per ``request_id`` irrespective of
  how many idempotent re-enqueues fire.
* ``rapid_mlx_requests_cancelled_via_disconnect_total`` — sub-counter
  attributing the subset triggered through the disconnect_guard
  ``_force_abort_request`` path, so the (total - disconnect) gap
  surfaces explicit-cancel + timeout traffic for capacity planning.

These are observability-only counters. ``Scheduler.abort_request`` and
``_force_abort_request`` semantics are unchanged — the counters bolt on
to existing returns. Defaults to zero on engines that never see an
abort so dashboards never flip to "no data" after a deploy (mirrors the
M-02 PFlash flat-line treatment).

The tests below cover four behavioural surfaces:

1. **Scheduler-side total counter** — increments once per accepted
   abort, never on rejected unknown-id aborts, idempotent against
   double-enqueue of the same id, and surfaced through ``get_stats``.
2. **Scheduler-side disconnect sub-counter** — bumps only when
   ``record_disconnect_abort`` is invoked, deduplicated by
   ``_disconnect_abort_ids`` so the three-branch helper-layer fire
   (disconnect + GeneratorExit + finally) attributes once per request.
3. **End-to-end disconnect_guard** — `_force_abort_request` calls
   `scheduler.abort_request` AND `scheduler.record_disconnect_abort`,
   driving both counters to +N for N aborted requests. Three streaming
   requests aborted via raw httpx aclose → counter = 3; two requests
   that complete normally → counter unchanged.
4. **Route-side Prometheus render** — both counters surface with
   correct HELP/TYPE lines, zero-default when the keys are absent
   (older engines), and stable monotonic values.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig

# ---------------------------------------------------------------------------
# Helpers — real scheduler driven without the live model
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    eos_token_id = None

    def encode(self, prompt):
        if isinstance(prompt, str):
            return [ord(c) for c in prompt]
        return list(prompt)

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)


def _make_scheduler() -> Scheduler:
    """Build a real ``Scheduler`` against an identity tokenizer.

    Mirrors the helper in ``test_pflash_metrics.py`` so the M-01
    counters are exercised by the same scheduler entry points that
    serve production traffic.
    """
    return Scheduler(
        model=object(),
        tokenizer=_DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=False,
            use_memory_aware_cache=False,
        ),
    )


def _admit(scheduler: Scheduler, request_id: str) -> None:
    """Push a request into the scheduler so ``abort_request`` will
    consider its ``request_id`` "known" via ``self.requests``.

    The scheduler's abort gate returns False for an unknown id, so
    every "abort that should count" must first appear in this dict.
    """
    request = Request(request_id, list(range(8)), SamplingParams(max_tokens=4))
    scheduler.add_request(request)


# ---------------------------------------------------------------------------
# Scheduler-side: total counter
# ---------------------------------------------------------------------------


def test_total_counter_starts_at_zero():
    """Fresh scheduler exposes both counters at zero through ``get_stats``.

    Dashboard panels treat an absent series as "no data" and trigger
    spurious alerts after a deploy; we always want a flat-line zero
    instead.
    """
    scheduler = _make_scheduler()
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 0
    assert stats["num_requests_cancelled_via_disconnect"] == 0


def test_total_counter_increments_once_per_accepted_abort():
    """Three known request_ids aborted → counter = 3.

    Sanity check that the counter advances per public-API abort, not
    per token or per batch step.
    """
    scheduler = _make_scheduler()
    for index in range(3):
        request_id = f"req-{index}"
        _admit(scheduler, request_id)
        assert scheduler.abort_request(request_id) is True
    assert scheduler.get_stats()["num_requests_cancelled"] == 3


def test_total_counter_not_bumped_for_unknown_request_id():
    """``abort_request`` returns False for unknown ids and the
    counter does NOT advance.

    F-151 hardening: an attacker who pokes random ids must not be
    able to inflate the counter (it would be a free signal that
    /metrics is observable; more importantly, it would corrupt the
    series and trigger false alarms).
    """
    scheduler = _make_scheduler()
    assert scheduler.abort_request("nonexistent-request-id") is False
    assert scheduler.get_stats()["num_requests_cancelled"] == 0


def test_total_counter_is_idempotent_against_double_enqueue():
    """Two ``abort_request`` calls for the same id → counter = 1.

    The scheduler is intentionally idempotent against double-enqueue
    (``Scheduler.abort_request`` docstring) because the disconnect
    guard fires from multiple branches per request. Without the de-dup
    gate the counter would double-count every disconnect-triggered
    abort.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-double")
    assert scheduler.abort_request("req-double") is True
    assert scheduler.abort_request("req-double") is True  # idempotent
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


def test_total_counter_counts_reused_request_id_after_full_abort_cycle():
    """Codex r3 BLOCKING #1: ``_cancelled_request_ids`` is an
    ACTIVE-LIFETIME guard, not a process-lifetime ledger. Reusing the
    same ``request_id`` for a NEW distinct request (after the prior
    lifetime fully aborted AND was popped from ``self.requests``) MUST
    increment the counter.

    Pre-r3-fix the lifetime ledger persisted until ``reset()``, so
    integrations that hash request_ids deterministically (or clients
    that retry the same operation with a stable id) would silently
    see their second cancel omitted from ``num_requests_cancelled``.
    The r3 fix discards the id from BOTH dedupe ledgers inside
    ``_do_abort_request`` so a future request with the same id
    starts from a clean ledger.

    Lifecycle: ``_do_abort_request`` clears the deduper but the text
    scheduler keeps the ``Request`` object in ``self.requests`` until
    a separate ``remove_finished_request`` call removes it (the
    engine_core cleanup path). We drive that explicitly here to
    mirror the production order — only AFTER the request is fully
    out of the scheduler will the abort_request predicate take
    the ``not in self.requests`` branch and require a fresh admit.
    """
    scheduler = _make_scheduler()

    # First lifetime: admit, cancel, drain, cleanup.
    _admit(scheduler, "req-reused")
    assert scheduler.abort_request("req-reused") is True
    assert scheduler.get_stats()["num_requests_cancelled"] == 1
    scheduler._process_pending_aborts()
    # Codex r4 fix: the dedupe ledger MUST stay populated until the
    # request actually leaves ``self.requests``. Otherwise a
    # redundant ``abort_request`` arriving while the request is
    # mid-cleanup would observe ``in self.requests`` AND an empty
    # ledger and double-count the same lifetime.
    assert "req-reused" in scheduler._cancelled_request_ids
    # Engine-core cleanup: remove the finished request from the
    # scheduler's request map; the discard of the dedupe ledger
    # happens HERE per codex r4.
    scheduler.remove_finished_request("req-reused")
    assert "req-reused" not in scheduler.requests
    assert "req-reused" not in scheduler._cancelled_request_ids

    # Second lifetime: distinct request, same id.
    _admit(scheduler, "req-reused")
    assert scheduler.abort_request("req-reused") is True
    # MUST count again — the new request is a distinct cancel.
    assert scheduler.get_stats()["num_requests_cancelled"] == 2


def test_abort_request_revalidates_membership_under_lock():
    """Codex r6 BLOCKING #1: ``abort_request`` MUST re-check
    request membership INSIDE ``_cancel_counter_lock`` so that
    ``remove_finished_request`` racing in (and popping
    ``self.requests`` + clearing the ledger under the same lock)
    can't leave ``abort_request`` to spuriously increment the
    counter for an already-removed request.

    Pre-r6-fix the predicate ran lock-free, so an abort that
    observed the request alive could acquire the lock AFTER the
    cleanup path ran and re-add the id to ``_pending_abort_ids``
    + increment ``num_requests_cancelled``. Post-r6 the
    membership check is inside the critical section.

    Test approach: monkey the lock's ``__enter__`` to simulate
    the racing ``remove_finished_request`` call THE MOMENT
    ``abort_request`` acquires the lock — the request is removed
    from ``self.requests`` BEFORE the inside-lock predicate
    evaluates. Pre-r6 the abort had already committed to True
    (predicate ran outside lock); post-r6 the inside-lock
    predicate flips the result to False and the counter stays
    at 0.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-stale-race")

    # Wrap the lock with a class whose ``__enter__`` simulates the
    # concurrent ``remove_finished_request`` racing in the moment
    # ``abort_request`` acquires the lock. Pre-r6 fix the
    # membership predicate ran outside the lock — it observed the
    # request alive, then acquired the lock after this mutation,
    # then re-added the id to ``_pending_abort_ids`` and bumped
    # the counter (spurious). Post-r6 the predicate runs INSIDE
    # the critical section and sees the cleared state.
    real_lock = scheduler._cancel_counter_lock

    class _RaceInjectingLock:
        def __init__(self, inner):
            self._inner = inner
            self.injected = False

        def __enter__(self):
            self._inner.acquire()
            # Inject ONCE per test, mirroring a single racing
            # ``remove_finished_request`` running between this
            # ``abort_request``'s predicate and lock acquire.
            if not self.injected:
                self.injected = True
                scheduler.requests.pop("req-stale-race", None)
                scheduler._cancelled_request_ids.discard("req-stale-race")
                scheduler._disconnect_abort_ids.discard("req-stale-race")
            return self

        def __exit__(self, *exc_info):
            self._inner.release()
            return False

        def locked(self):
            return self._inner.locked()

        def acquire(self, *args, **kwargs):
            return self._inner.acquire(*args, **kwargs)

        def release(self, *args, **kwargs):
            return self._inner.release(*args, **kwargs)

    scheduler._cancel_counter_lock = _RaceInjectingLock(real_lock)

    # With the codex r6 fix the inside-lock predicate observes
    # the cleared state and returns False; the counter does NOT
    # advance.
    result = scheduler.abort_request("req-stale-race")
    assert result is False, (
        "Pre-r6 fix would have returned True; the inside-lock "
        "membership check must reject the stale id"
    )
    assert scheduler.get_stats()["num_requests_cancelled"] == 0


def test_remove_finished_request_pops_and_discards_atomically():
    """Codex r5 BLOCKING: ``remove_finished_request`` must perform
    the ``self.requests.pop`` AND the dedupe-ledger discards
    inside a single critical section so a concurrent
    ``abort_request`` cannot observe ``request_id in self.requests``
    AND an empty ledger.

    Pre-r5-fix the discards happened under the lock but the pop
    was outside, opening a window where a racing thread saw:
      * ``request_id in self.requests``  (pop hadn't run yet)
      * ``request_id not in _cancelled_request_ids`` (discard
        already cleared it)
    → double-count of the same request lifetime.

    Test approach: monkeypatch ``self.requests.pop`` to verify the
    lock is HELD across both ``pop`` and ``discard``. We can't
    reliably exercise the race via real threads (the window is
    microseconds), so we assert the ordering contract by tracking
    when each operation runs relative to the lock state. A
    regression that moves either operation out of the critical
    section flips one of the assertions.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-atomic-cleanup")
    scheduler.abort_request("req-atomic-cleanup")
    scheduler._process_pending_aborts()
    # Pre-state: ledger populated, request still resident.
    assert "req-atomic-cleanup" in scheduler._cancelled_request_ids
    assert "req-atomic-cleanup" in scheduler.requests

    # Wrap ``self.requests`` and ``_cancelled_request_ids`` with
    # observers that snapshot the lock state at each mutation.
    # Can't monkeypatch ``dict.pop`` / ``set.discard`` directly
    # (they're slot-defined read-only), so we replace the whole
    # container with a subclass.
    lock = scheduler._cancel_counter_lock

    class _DictObserver(dict):
        def __init__(self, src):
            super().__init__(src)
            self.pop_lock_states: list[bool] = []

        def pop(self, *args, **kwargs):
            self.pop_lock_states.append(lock.locked())
            return super().pop(*args, **kwargs)

    class _SetObserver(set):
        def __init__(self, src):
            super().__init__(src)
            self.discard_lock_states: list[bool] = []

        def discard(self, *args, **kwargs):
            self.discard_lock_states.append(lock.locked())
            return super().discard(*args, **kwargs)

    scheduler.requests = _DictObserver(scheduler.requests)
    scheduler._cancelled_request_ids = _SetObserver(scheduler._cancelled_request_ids)
    scheduler._disconnect_abort_ids = _SetObserver(scheduler._disconnect_abort_ids)

    scheduler.remove_finished_request("req-atomic-cleanup")

    # Both operations must have observed the lock held — that's
    # the codex r5 contract.
    assert scheduler.requests.pop_lock_states == [True], (
        scheduler.requests.pop_lock_states
    )
    assert scheduler._cancelled_request_ids.discard_lock_states == [True], (
        scheduler._cancelled_request_ids.discard_lock_states
    )
    # Sanity: the actual cleanup happened.
    assert "req-atomic-cleanup" not in scheduler.requests
    assert "req-atomic-cleanup" not in scheduler._cancelled_request_ids


def test_total_counter_does_not_double_count_redundant_abort_pre_cleanup():
    """Codex r4 BLOCKING #1: between ``_do_abort_request`` and
    ``remove_finished_request`` the request stays resident in
    ``self.requests``. A redundant ``abort_request`` for the same id
    in that window MUST NOT double-count.

    Pre-r4-fix (codex r3): the dedupe ledger was cleared inside
    ``_do_abort_request``, so the redundant call observed the
    request still in ``self.requests`` AND an empty ledger and
    incremented the counter a second time for the same lifetime.

    Post-r4: the ledger is only cleared inside
    ``remove_finished_request``, by which point the request has
    truly left every admit predicate and a fresh ``abort_request``
    can only land via a new admit (a distinct lifetime).
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-mid-cleanup")
    scheduler.abort_request("req-mid-cleanup")
    assert scheduler.get_stats()["num_requests_cancelled"] == 1
    # Simulate the executor running _do_abort_request — but NOT
    # yet the engine_core remove_finished_request call. The
    # request is mid-cleanup.
    scheduler._process_pending_aborts()
    assert "req-mid-cleanup" in scheduler.requests  # still resident

    # Redundant abort in the cleanup window. MUST NOT increment.
    assert scheduler.abort_request("req-mid-cleanup") is True
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


def test_disconnect_subcounter_counts_reused_request_id_after_full_abort_cycle():
    """Codex r3 BLOCKING #2 symmetry: the via_disconnect sub-counter
    must also re-count a reused ``request_id`` after the prior
    lifetime completed.

    Same rationale as the total counter — keeping the id in
    ``_disconnect_abort_ids`` past the request's lifetime would
    silently swallow attribution for the next distinct request.
    """
    scheduler = _make_scheduler()

    # First lifetime: full cancel cycle with disconnect attribution.
    _admit(scheduler, "req-disc-reused")
    scheduler.abort_request("req-disc-reused")
    scheduler.record_disconnect_abort("req-disc-reused")
    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 1
    scheduler._process_pending_aborts()
    # Codex r4: still in ledger until remove_finished_request.
    assert "req-disc-reused" in scheduler._disconnect_abort_ids
    scheduler.remove_finished_request("req-disc-reused")
    assert "req-disc-reused" not in scheduler._disconnect_abort_ids

    # Second lifetime: distinct request reusing the id.
    _admit(scheduler, "req-disc-reused")
    scheduler.abort_request("req-disc-reused")
    scheduler.record_disconnect_abort("req-disc-reused")
    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 2


def test_total_counter_dedupes_against_lifetime_ledger_not_pending_set():
    """Codex r2 BLOCKING #1: the dedupe ledger must be a lifetime
    set, not the drainable ``_pending_abort_ids``.

    Pre-fix the dedupe used ``request_id in self._pending_abort_ids``.
    ``_pending_abort_ids`` is drained every step via
    ``_process_pending_aborts``; once drained, a later
    ``abort_request(rid)`` for a still-resident request id (e.g.
    while the request still lives in ``self.requests`` between the
    first abort enqueue and the executor draining the pending set,
    OR request_id reuse across distinct lifetimes) would see
    ``already_pending=False`` again and double-count.

    The new ledger ``_cancelled_request_ids`` is wiped only on
    ``reset()`` (matching the prior drain treatment) and survives
    individual abort drains.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-drain-race")

    # First abort enqueues into _pending_abort_ids AND increments
    # the counter via the new lifetime ledger.
    assert scheduler.abort_request("req-drain-race") is True
    assert scheduler.get_stats()["num_requests_cancelled"] == 1

    # Simulate the executor thread draining the pending set
    # (``_process_pending_aborts`` pops every id). The lifetime
    # ledger must NOT be touched.
    scheduler._pending_abort_ids.clear()
    # The request itself stays in ``self.requests`` until
    # ``_do_abort_request`` runs — we don't simulate that here,
    # because the failure mode codex flagged is the SECOND
    # ``abort_request`` call landing while the request is still
    # known (e.g. another abort source races with the drain).
    assert "req-drain-race" in scheduler.requests

    # Second abort_request for the same still-known id — passes the
    # ``request_id in self.requests`` predicate. Pre-fix this would
    # have bumped the counter again because the pending set was
    # empty. Post-fix the lifetime ledger catches it.
    assert scheduler.abort_request("req-drain-race") is True
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


def test_reset_clears_ledgers_after_abort_loop():
    """Codex r8 BLOCKING #1: ``reset()`` MUST clear the
    cancellation lifetime ledgers AFTER aborting every live
    request, not before. Clearing before opens a window during
    the abort loop where a concurrent ``record_disconnect_abort``
    could either no-op (id discarded ahead of its lifetime
    ending) or re-add the id after ``_do_abort_request`` runs
    (the discard inside ``_do_abort_request`` would conflict).

    Test approach: track the order of operations by patching
    ``_do_abort_request`` to record when ``_cancelled_request_ids``
    is empty. Pre-r8 the set is empty for every iteration of the
    abort loop (cleared upfront); post-r8 the set still contains
    the in-flight id (cleared after the loop).
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-reset-1")
    _admit(scheduler, "req-reset-2")
    scheduler.abort_request("req-reset-1")
    scheduler.abort_request("req-reset-2")
    # Pre-reset: both ids in the lifetime ledger.
    assert "req-reset-1" in scheduler._cancelled_request_ids
    assert "req-reset-2" in scheduler._cancelled_request_ids

    # Track ledger state at the START of each _do_abort_request
    # call during reset's loop.
    ledger_snapshots: list[set[str]] = []
    original_do_abort = scheduler._do_abort_request

    def tracked_do_abort(rid):
        ledger_snapshots.append(set(scheduler._cancelled_request_ids))
        return original_do_abort(rid)

    scheduler._do_abort_request = tracked_do_abort  # type: ignore[method-assign]

    scheduler.reset()

    # The abort loop must have observed a NON-EMPTY ledger on at
    # least one iteration (pre-r8 fix it would be empty on every
    # iteration because reset cleared it upfront).
    assert any(snap for snap in ledger_snapshots), (
        "reset() cleared the lifetime ledger BEFORE the abort loop ran; "
        "concurrent record_disconnect_abort would have seen "
        "inconsistent state. Snapshots: " + repr(ledger_snapshots)
    )
    # Post-reset: ledger is empty.
    assert scheduler._cancelled_request_ids == set()


def test_total_counter_survives_reset_unchanged():
    """``reset()`` clears in-flight aborts but NOT the lifetime counter.

    Prometheus counters MUST be monotonic — a step backward would be
    interpreted as a process restart and corrupt rate() / increase()
    calculations on the scraper side.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-pre-reset")
    scheduler.abort_request("req-pre-reset")
    assert scheduler.get_stats()["num_requests_cancelled"] == 1

    scheduler.reset()
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


# ---------------------------------------------------------------------------
# Scheduler-side: disconnect sub-counter
# ---------------------------------------------------------------------------


def test_disconnect_sub_counter_bumps_on_record_call():
    """``record_disconnect_abort`` advances the sub-counter."""
    scheduler = _make_scheduler()
    _admit(scheduler, "req-disc")
    scheduler.abort_request("req-disc")
    scheduler.record_disconnect_abort("req-disc")
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 1
    assert stats["num_requests_cancelled_via_disconnect"] == 1


def test_disconnect_sub_counter_dedupes_per_request_id():
    """Three ``record_disconnect_abort`` calls for the same id → sub = 1.

    The disconnect_guard fires ``_force_abort_request`` from the
    ``if disconnect_task in done`` branch, the ``except GeneratorExit``
    branch, AND the ``finally`` belt-and-suspenders. Without the
    per-id de-dup the sub-counter would over-count by up to 3x per
    disconnect event.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-multi")
    scheduler.abort_request("req-multi")
    scheduler.record_disconnect_abort("req-multi")
    scheduler.record_disconnect_abort("req-multi")
    scheduler.record_disconnect_abort("req-multi")
    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 1


def test_disconnect_sub_counter_silent_on_id_never_accepted_as_cancel():
    """Codex r7 NIT #3: ``record_disconnect_abort`` must reject ids
    that were never accepted by ``abort_request``, so the dashboard
    invariant ``via_disconnect_total <= cancelled_total`` holds by
    construction even on programmer error.

    Pre-r7 the method incremented the sub-counter for any non-empty
    id, so a buggy caller could push ``via_disconnect`` above
    ``cancelled_total``. The lifetime ledger gate catches it.
    """
    scheduler = _make_scheduler()
    # No ``abort_request("req-bogus")`` call — the id was never
    # admitted into the lifetime ledger ``_cancelled_request_ids``.
    scheduler.record_disconnect_abort("req-bogus")
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 0
    assert stats["num_requests_cancelled_via_disconnect"] == 0


def test_disconnect_sub_counter_silent_on_empty_request_id():
    """Empty string / None ``request_id`` is a no-op, not a crash.

    The disconnect_guard sometimes invokes ``_force_abort_request``
    with an empty holder (engine cancelled before ``add_request``
    returned). The helper must swallow that path silently rather than
    propagate the no-op into the live disconnect flow.
    """
    scheduler = _make_scheduler()
    scheduler.record_disconnect_abort("")
    scheduler.record_disconnect_abort(None)  # type: ignore[arg-type]
    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 0


def test_disconnect_sub_counter_decoupled_from_total():
    """An explicit cancel (no disconnect attribution) leaves the
    sub-counter at zero.

    The total counter ticks on every accepted abort; the sub-counter
    only ticks when ``_force_abort_request`` (disconnect path) was
    the trigger. The (total - via_disconnect) gap is the operator's
    signal for explicit-cancel + timeout traffic — if the sub-counter
    were entangled with the total this gap would always be zero.
    """
    scheduler = _make_scheduler()
    _admit(scheduler, "req-explicit")
    scheduler.abort_request("req-explicit")  # explicit cancel route
    # No ``record_disconnect_abort`` call.
    stats = scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 1
    assert stats["num_requests_cancelled_via_disconnect"] == 0


# ---------------------------------------------------------------------------
# Helper-layer: _force_abort_request feeds both counters end-to-end
# ---------------------------------------------------------------------------


class _CountingScheduler:
    """Real-shaped scheduler stand-in that records both the public-API
    abort and the disconnect attribution call.

    Lets the helper-layer test pin the exact contract between
    ``_force_abort_request`` and the scheduler-side counters without
    spinning up a full scheduler + tokenizer.
    """

    def __init__(self):
        self.aborts: list[str] = []
        self.disconnect_records: list[str] = []

    def abort_request(self, request_id: str) -> bool:
        self.aborts.append(request_id)
        return True

    def record_disconnect_abort(self, request_id: str) -> None:
        self.disconnect_records.append(request_id)


class _Engine:
    """Engine stub that exposes ``scheduler`` directly (the simple
    pre-BatchedEngine shape ``_resolve_sync_scheduler_for_abort``
    matches first).
    """

    def __init__(self):
        self.scheduler = _CountingScheduler()


def test_force_abort_bumps_disconnect_subcounter():
    """``_force_abort_request`` calls ``record_disconnect_abort``
    once after a successful sync abort.

    This is the contract that ties the helper-layer disconnect path
    to the scheduler-side sub-counter; deleting the
    ``_record_disconnect_abort_on_scheduler`` call would silently
    drop the attribution and the operator's "via disconnect" series
    would stay flat through real disconnects.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    engine = _Engine()
    holder = ["req-disconnect"]

    fired = _force_abort_request(engine, holder)

    assert fired is True
    assert engine.scheduler.aborts == ["req-disconnect"]
    assert engine.scheduler.disconnect_records == ["req-disconnect"]


def test_force_abort_does_not_record_when_sync_abort_rejected():
    """If ``abort_request`` returned False (unknown id), the helper
    must NOT call ``record_disconnect_abort``.

    Pre-fix candidate bug: a helper that records unconditionally
    would inflate the sub-counter every time a stale holder fires
    through ``_force_abort_request`` (e.g. when the request finished
    before the disconnect_guard tore down). The total counter is
    already protected against this by ``Scheduler.abort_request``
    returning False for unknown ids; the helper must propagate that
    gate to the sub-counter or the two series will drift.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _RejectingScheduler:
        def __init__(self):
            self.disconnect_records: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            return False  # unknown id — F-151 path

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _RejectingEngine:
        def __init__(self):
            self.scheduler = _RejectingScheduler()

    engine = _RejectingEngine()
    holder = ["req-unknown"]

    fired = _force_abort_request(engine, holder)

    # The sync abort entry returned False but the helper still
    # returns True per its docstring (it DID dispatch). The
    # sub-counter, however, must NOT advance.
    assert fired is True
    assert engine.scheduler.disconnect_records == []


def test_force_abort_does_not_crash_when_record_method_absent():
    """Older schedulers without ``record_disconnect_abort`` continue
    to work — the helper swallows ``AttributeError`` silently and
    the total counter alone keeps surfacing the abort.

    Critical because the helper layer sits across a public API
    surface that downstream forks / external schedulers depend on.
    A regression here would break every non-rapid_mlx scheduler.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _LegacyScheduler:
        def __init__(self):
            self.aborts: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            self.aborts.append(request_id)
            return True

        # NB: no record_disconnect_abort.

    class _LegacyEngine:
        def __init__(self):
            self.scheduler = _LegacyScheduler()

    engine = _LegacyEngine()
    holder = ["req-legacy"]

    fired = _force_abort_request(engine, holder)

    assert fired is True
    assert engine.scheduler.aborts == ["req-legacy"]


@pytest.mark.asyncio
async def test_force_abort_async_fallback_does_not_attribute_on_false_result():
    """Codex r1 BLOCKING #1: async-fallback path must await the abort
    coroutine and skip the sub-counter when the eventual result is
    False (unknown id rejected by the scheduler).

    Pre-fix the helper called ``_record_disconnect_abort_on_scheduler``
    immediately after ``asyncio.ensure_future(coro)`` without ever
    inspecting ``coro``'s eventual result, so a stale holder with
    a request_id that's already finished would leave the total
    counter unchanged (correct — scheduler returned False) but tick
    the via_disconnect sub-counter (wrong — would inflate the gap
    operators rely on). The fix chains attribution on the awaited
    coroutine result.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    records: list[str] = []

    class _SyncScheduler:
        def record_disconnect_abort(self, rid: str) -> None:
            records.append(rid)

    class _EngineCoreLike:
        def __init__(self):
            self.scheduler = _SyncScheduler()

    class _AsyncEngineCoreLike:
        def __init__(self):
            self.engine = _EngineCoreLike()

        async def abort_request(self, rid: str) -> bool:
            # Scheduler-side rejection: the public abort returned
            # False (unknown id / already finished). The sub-counter
            # MUST NOT advance.
            return False

    class _BatchedEngineLike:
        def __init__(self):
            self._engine = _AsyncEngineCoreLike()
            self._is_mllm = False

        async def abort_request(self, rid: str) -> bool:
            return await self._engine.abort_request(rid)

    engine = _BatchedEngineLike()
    holder = ["req-stale"]

    fired = _force_abort_request(engine, holder)
    # Let the chained awaiter run to completion.
    for _ in range(4):
        await asyncio.sleep(0)

    # The helper returned False (async fallback path) AND no
    # attribution landed because the coroutine returned False.
    assert fired is False
    assert records == []


@pytest.mark.asyncio
async def test_force_abort_async_fallback_attributes_on_true_result():
    """Symmetric to the False case: async-fallback attribution DOES
    fire when the awaited coroutine returns True.

    This is the production text-path happy case (BatchedEngine over
    AsyncEngineCore) — the abort is accepted by the underlying
    scheduler, the total counter ticks, and the sub-counter
    attributes the cause.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    records: list[str] = []

    class _SyncScheduler:
        def record_disconnect_abort(self, rid: str) -> None:
            records.append(rid)

    class _EngineCoreLike:
        def __init__(self):
            self.scheduler = _SyncScheduler()

    class _AsyncEngineCoreLike:
        def __init__(self):
            self.engine = _EngineCoreLike()

        async def abort_request(self, rid: str) -> bool:
            return True  # scheduler accepted

    class _BatchedEngineLike:
        def __init__(self):
            self._engine = _AsyncEngineCoreLike()
            self._is_mllm = False

        async def abort_request(self, rid: str) -> bool:
            return await self._engine.abort_request(rid)

    engine = _BatchedEngineLike()
    holder = ["req-accepted"]

    fired = _force_abort_request(engine, holder)
    for _ in range(4):
        await asyncio.sleep(0)

    assert fired is False  # async-fallback returns False per contract
    assert records == ["req-accepted"]


def test_total_counter_atomic_against_concurrent_aborts():
    """Codex r1 BLOCKING #2: concurrent ``abort_request`` calls for
    the same id must not double-count.

    Pre-fix the check-add-increment was not atomic: two threads
    could both observe ``request_id not in _pending_abort_ids`` and
    both bump the counter. With the lock the second thread sees the
    first thread's add and skips the increment.

    Uses a ``threading.Barrier`` to force all N threads into a hot
    race rather than relying on luck — pre-fix this test would
    catch the double-count in a small number of runs; with the lock
    it's deterministic.
    """
    import threading as _threading

    scheduler = _make_scheduler()
    _admit(scheduler, "req-race")

    n_threads = 32
    barrier = _threading.Barrier(n_threads)
    results: list[bool] = []
    results_lock = _threading.Lock()

    def race():
        barrier.wait()
        accepted = scheduler.abort_request("req-race")
        with results_lock:
            results.append(accepted)

    threads = [_threading.Thread(target=race) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All N threads return True (idempotent), but the counter must
    # be exactly 1.
    assert all(results)
    assert scheduler.get_stats()["num_requests_cancelled"] == 1


def test_disconnect_subcounter_atomic_against_concurrent_records():
    """Codex r1 BLOCKING #3: concurrent ``record_disconnect_abort``
    calls for the same id must not double-count.

    Pinning the helper-layer fire pattern: the disconnect_guard
    invokes ``_force_abort_request`` from up to three branches
    (disconnect / GeneratorExit / finally) which can be running on
    different async tasks, each one potentially in a different
    executor thread. Without the lock the set-membership / add /
    increment sequence races and the sub-counter inflates.
    """
    import threading as _threading

    scheduler = _make_scheduler()
    _admit(scheduler, "req-race-disc")
    scheduler.abort_request("req-race-disc")  # total = 1

    n_threads = 32
    barrier = _threading.Barrier(n_threads)

    def race():
        barrier.wait()
        scheduler.record_disconnect_abort("req-race-disc")

    threads = [_threading.Thread(target=race) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert scheduler.get_stats()["num_requests_cancelled_via_disconnect"] == 1


@pytest.mark.asyncio
async def test_force_abort_attribution_walks_production_batched_engine_shape():
    """Production ``BatchedEngine`` over ``AsyncEngineCore`` hides the
    scheduler at ``engine._engine.engine.scheduler`` (one hop deeper
    than the abort resolver covers, because ``AsyncEngineCore`` wraps
    ``EngineCore`` via ``self.engine``).

    The attribution resolver MUST dig the extra hop or the
    via_disconnect sub-counter stays flat through every real
    production disconnect — which is the exact symptom Mei + Yana
    flagged the cancel-rate counter for in the first place. A static
    introspection / unit-test wins; the alternative is to ship the
    fix and discover via dashboards two weeks later that the gap is
    always 100% of total.

    The production sync abort path falls through to the async
    fallback (also covered by the helper now), but the attribution
    must still land on the right scheduler so the (total -
    via_disconnect) gap reflects reality.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncScheduler:
        def __init__(self):
            self.disconnect_records: list[str] = []
            self._async_abort_calls: list[str] = []

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _EngineCoreLike:
        """Mirrors the production ``EngineCore`` shape — owns the
        scheduler directly."""

        def __init__(self):
            self.scheduler = _SyncScheduler()

    class _AsyncEngineCoreLike:
        """Production-shaped ``AsyncEngineCore`` — has ``.engine``
        (the inner ``EngineCore``) but NO direct ``.scheduler``,
        and exposes an ``async abort_request`` shim that lands on
        ``self.engine.engine.abort_request`` in production. Crucially
        the helper's sync-abort resolver returns ``None`` here, so
        the async fallback fires (matching the production text path).
        """

        def __init__(self):
            self.engine = _EngineCoreLike()

        async def abort_request(self, request_id: str) -> bool:
            # Production-fidelity: the async shim ultimately reaches
            # ``self.engine.scheduler.abort_request`` (sync).
            self.engine.scheduler._async_abort_calls.append(request_id)
            return True

    class _BatchedEngineLike:
        def __init__(self):
            self._engine = _AsyncEngineCoreLike()
            self._is_mllm = False

        async def abort_request(self, request_id: str) -> bool:
            return await self._engine.abort_request(request_id)

    engine = _BatchedEngineLike()
    holder = ["req-prod-shape"]

    fired = _force_abort_request(engine, holder)
    # Drain the fire-and-forget task the helper queued via
    # ``asyncio.ensure_future`` so the test doesn't trip the
    # "Task was destroyed but it is pending" runtime warning. The
    # production code intentionally fires-and-forgets (operators see
    # the WARNING log) — the test owns the loop, so we drain.
    await asyncio.sleep(0)

    # The sync resolver returned None (no sync abort path), so the
    # helper hits the async fallback branch and returns False per
    # the docstring contract.
    assert fired is False
    # But the attribution MUST still have landed — that's the bug
    # we're guarding against.
    assert engine._engine.engine.scheduler.disconnect_records == ["req-prod-shape"]


def test_attribution_resolver_honors_is_mllm_before_direct_scheduler():
    """Codex r7 BLOCKING #2: a dual-shaped engine exposing BOTH
    ``engine.scheduler`` (text-path leftover) AND ``_mllm_scheduler``
    (MLLM-path active) with ``_is_mllm=True`` must attribute the
    disconnect to the MLLM scheduler — NOT the direct
    ``engine.scheduler``.

    Pre-r7-fix the resolver checked ``engine.scheduler`` FIRST and
    short-circuited, mis-attributing every MLLM disconnect to the
    text scheduler. The fix honors ``_is_mllm`` first; direct
    ``engine.scheduler`` is only consulted as a fallback when the
    flag is absent entirely.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncSched:
        def __init__(self, name):
            self.name = name
            self.disconnect_records: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            return True

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _DualShapedEngine:
        """Both ``.scheduler`` (text leftover, e.g. a stale attribute
        from a prior backend init) AND ``_mllm_scheduler`` populated;
        the active backend is MLLM per ``_is_mllm=True``.
        """

        def __init__(self):
            self.scheduler = _SyncSched("text-leftover")
            self._mllm_scheduler = _SyncSched("mllm-active")
            self._is_mllm = True

        async def abort_request(self, request_id: str) -> bool:
            return True

    engine = _DualShapedEngine()
    _force_abort_request(engine, ["req-mllm-active"])

    # The MLLM scheduler MUST own the attribution.
    assert engine._mllm_scheduler.disconnect_records == ["req-mllm-active"]
    # The text-leftover scheduler MUST NOT be touched.
    assert engine.scheduler.disconnect_records == []


def test_force_abort_attribution_walks_active_backend():
    """Cancel-attribution resolver respects ``_is_mllm`` like the abort
    resolver — text-active engines record on ``_engine.scheduler``,
    MLLM-active engines on ``_mllm_scheduler``.

    The codex r2 BLOCKING #1 finding on PR #777 pinned this for the
    abort resolver; the attribution resolver must follow the same
    rule or it would bump the sub-counter on the wrong scheduler.
    """
    from vllm_mlx.service.helpers import _force_abort_request

    class _SyncSched:
        def __init__(self, name):
            self.name = name
            self.aborts: list[str] = []
            self.disconnect_records: list[str] = []

        def abort_request(self, request_id: str) -> bool:
            self.aborts.append(request_id)
            return True

        def record_disconnect_abort(self, request_id: str) -> None:
            self.disconnect_records.append(request_id)

    class _Inner:
        def __init__(self):
            self.scheduler = _SyncSched("text")

    class _DualEngine:
        def __init__(self, is_mllm: bool):
            self._is_mllm = is_mllm
            self._engine = _Inner()
            self._mllm_scheduler = _SyncSched("mllm")

    # Text path
    text_engine = _DualEngine(is_mllm=False)
    _force_abort_request(text_engine, ["req-text"])
    assert text_engine._engine.scheduler.disconnect_records == ["req-text"]
    assert text_engine._mllm_scheduler.disconnect_records == []

    # MLLM path
    mllm_engine = _DualEngine(is_mllm=True)
    _force_abort_request(mllm_engine, ["req-mllm"])
    assert mllm_engine._mllm_scheduler.disconnect_records == ["req-mllm"]
    assert mllm_engine._engine.scheduler.disconnect_records == []


# ---------------------------------------------------------------------------
# End-to-end: streaming-route abort drives both counters via TestClient
# ---------------------------------------------------------------------------


class _RealSchedulerEngine:
    """Engine stub that exposes a REAL ``Scheduler`` instance so the
    end-to-end disconnect_guard tests pin the actual
    ``get_stats()["num_requests_cancelled"]`` counter — not a fake
    counter on a stub.

    Codex r4 BLOCKING #3 fix: the earlier ``_CountingScheduler`` stub
    only recorded method calls, so the test would still pass if
    ``Scheduler.num_requests_cancelled`` never advanced. Driving the
    real scheduler closes that gap — a regression that breaks the
    counter wiring will now fail the assertion on
    ``get_stats()``.
    """

    def __init__(self):
        self.scheduler = _make_scheduler()


@pytest.mark.asyncio
async def test_three_aborted_streaming_requests_advance_counters_by_three():
    """Fire 3 streaming requests through ``_disconnect_guard`` and
    abort each via ``GeneratorExit``. The total counter and the
    disconnect sub-counter both increment by 3 on a REAL
    ``Scheduler``.

    This is the headline behaviour Mei r0.8.1 / Yana r3 asked for:
    aborted streams MUST be visible in /metrics so operators see the
    cancel-rate. Pre-fix the counters didn't exist; post-fix the test
    pins the +N contract against the real scheduler's
    ``get_stats()``.

    Codex r4 BLOCKING #3: this test previously used a fake
    ``_CountingScheduler`` and would have passed even if the real
    scheduler counters never advanced. Replaced with
    ``_RealSchedulerEngine`` so the assertion now pins the actual
    Prometheus-facing counter.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _RealSchedulerEngine()

    # Admit three requests directly into the real scheduler so
    # ``abort_request(rid)`` will pass the predicate.
    for i in range(3):
        _admit(engine.scheduler, f"req-stream-{i}")

    for i in range(3):
        holder = [f"req-stream-{i}"]

        async def upstream():
            yield 'data: {"chunk":"hello"}\n\n'
            await asyncio.sleep(60)
            yield "data: [DONE]\n\n"

        class _NeverDisconnects:
            async def is_disconnected(self) -> bool:
                return False

        guard = _disconnect_guard(
            upstream(),
            _NeverDisconnects(),
            poll_interval=0.05,
            engine=engine,
            request_id_holder=holder,
            keepalive_seconds=0,
        )

        # Pull one chunk then close — simulates Starlette tearing
        # down the StreamingResponse mid-stream (Astrid r3
        # fingerprint).
        agen = guard.__aiter__()
        await agen.__anext__()
        await agen.aclose()

    # The real scheduler's counters MUST have advanced by 3 each.
    # If the wiring breaks (e.g. someone removes the
    # ``_record_disconnect_abort_on_scheduler`` call, or the
    # scheduler stops counting on accepted aborts), THIS is what
    # catches it — not the side-channel attribute checks the
    # stub-based version did.
    stats = engine.scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 3, stats
    assert stats["num_requests_cancelled_via_disconnect"] == 3, stats


@pytest.mark.asyncio
async def test_two_completed_streaming_requests_leave_counter_unchanged():
    """Streams that exhaust normally do NOT advance the cancellation
    counters on a REAL ``Scheduler``.

    ``_disconnect_guard.finally`` has a ``finished_normally`` flag
    (codex r1 NIT #3 on PR #777) that skips the belt-and-suspenders
    force-abort when the upstream drained cleanly via
    ``StopAsyncIteration``. Without that gate the cancellation
    counter would tick once per completed request and the operator's
    "cancel rate" panel would look like "100% of traffic" — useless.

    Codex r4 BLOCKING #3 same fix as the abort test above: drive a
    real ``Scheduler`` so the counter assertion has bite.
    """
    from vllm_mlx.service.helpers import _disconnect_guard

    engine = _RealSchedulerEngine()
    for i in range(2):
        _admit(engine.scheduler, f"req-clean-{i}")

    for i in range(2):
        holder = [f"req-clean-{i}"]

        async def upstream():
            yield 'data: {"chunk":"a"}\n\n'
            yield 'data: {"chunk":"b"}\n\n'
            yield "data: [DONE]\n\n"

        class _NeverDisconnects:
            async def is_disconnected(self) -> bool:
                return False

        guard = _disconnect_guard(
            upstream(),
            _NeverDisconnects(),
            poll_interval=0.05,
            engine=engine,
            request_id_holder=holder,
            keepalive_seconds=0,
        )

        chunks = []
        async for chunk in guard:
            chunks.append(chunk)
        # Sanity: the consumer pulled the full stream.
        assert "[DONE]" in chunks[-1]

    # Neither counter should have moved on the REAL scheduler.
    stats = engine.scheduler.get_stats()
    assert stats["num_requests_cancelled"] == 0, stats
    assert stats["num_requests_cancelled_via_disconnect"] == 0, stats


# ---------------------------------------------------------------------------
# Route-side: counters surface in the /metrics Prometheus body
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics_client():
    """FastAPI TestClient mounting only the metrics router.

    Mirrors ``test_metrics_route.metrics_client`` and ``test_pflash_
    metrics.metrics_client`` so the M-01 counters are exercised
    through the same render path as every other series.
    """
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.metrics import _reset_accumulator_for_tests, router

    cfg = reset_config()
    cfg.model_name = "qwen3-0.6b"
    _reset_accumulator_for_tests()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), cfg=cfg)
    reset_config()
    _reset_accumulator_for_tests()


def _fake_engine(stats: dict[str, Any]):
    return SimpleNamespace(get_stats=lambda: stats)


def _assert_prom_counter(
    body: str, metric_name: str, expected_value: int, help_substr: str | None = None
) -> None:
    """Codex r8 NIT #3: exact line-by-line assertion of a Prometheus
    counter render (HELP, TYPE, sample), avoiding substring checks
    that could match malformed duplicate lines.

    The Prometheus text exposition format pins the line structure:
      # HELP <name> <help_text>
      # TYPE <name> <metric_type>
      <name>[{labels}] <value>

    We parse the body, look for those EXACT lines for the given
    metric name, and assert (a) HELP appears exactly once, (b) TYPE
    is ``counter`` exactly once, (c) the sample line is exactly
    ``<name> <value>`` with the right value, and (d) the sample
    appears exactly once. A malformed regression (duplicate series,
    wrong type, missing newline) flips at least one of those
    counts.
    """
    lines = body.splitlines()
    help_lines = [line for line in lines if line.startswith(f"# HELP {metric_name} ")]
    type_lines = [line for line in lines if line.startswith(f"# TYPE {metric_name} ")]
    # Sample lines: the metric name MUST be followed by a space and
    # the value (we don't render labels for these counters, so the
    # ``{...}`` form is also a regression). Match by exact split,
    # not substring.
    sample_lines = [line for line in lines if line.split(" ", 1)[:1] == [metric_name]]

    assert len(help_lines) == 1, (
        f"Expected exactly one HELP line for {metric_name}, got {help_lines}"
    )
    assert len(type_lines) == 1, (
        f"Expected exactly one TYPE line for {metric_name}, got {type_lines}"
    )
    assert type_lines[0].endswith(" counter"), (
        f"Expected TYPE line to end with ' counter', got {type_lines[0]!r}"
    )
    assert len(sample_lines) == 1, (
        f"Expected exactly one sample line for {metric_name}, got {sample_lines}"
    )
    assert sample_lines[0] == f"{metric_name} {expected_value}", (
        f"Expected sample line {metric_name} {expected_value!r}, "
        f"got {sample_lines[0]!r}"
    )
    if help_substr is not None:
        assert help_substr in help_lines[0], (
            f"Expected HELP line to contain {help_substr!r}, got {help_lines[0]!r}"
        )


_CANCEL_STATS = {
    "num_waiting": 0,
    "num_running": 0,
    "num_requests_processed": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "num_requests_cancelled": 5,
    "num_requests_cancelled_via_disconnect": 3,
}


def test_metrics_route_renders_total_cancelled_counter(metrics_client):
    """``rapid_mlx_requests_cancelled_total`` HELP / TYPE / value all
    present and the value matches the scheduler stat.

    Codex r8 NIT #3: assertion uses exact line-by-line parsing of
    the Prometheus body so a malformed duplicate sample line
    couldn't mask a regression.
    """
    metrics_client.cfg.engine = _fake_engine(_CANCEL_STATS)
    body = metrics_client.client.get("/metrics").text

    _assert_prom_counter(
        body,
        "rapid_mlx_requests_cancelled_total",
        5,
        help_substr="aborted via the scheduler abort path",
    )


def test_metrics_route_renders_disconnect_subcounter(metrics_client):
    """``rapid_mlx_requests_cancelled_via_disconnect_total`` HELP /
    TYPE / value all present and the value matches the scheduler stat.
    """
    metrics_client.cfg.engine = _fake_engine(_CANCEL_STATS)
    body = metrics_client.client.get("/metrics").text

    _assert_prom_counter(
        body,
        "rapid_mlx_requests_cancelled_via_disconnect_total",
        3,
        help_substr="attributed to client disconnect",
    )


def test_metrics_route_renders_zero_when_cancel_keys_missing(metrics_client):
    """Engines without the M-01 keys render flat-line zero rather than
    omit the series.

    Older non-rapid_mlx schedulers (downstream forks, custom backends)
    may not populate the keys at all. Dashboards configured against
    these series must not flip to "no data".
    """
    stats_without_cancel = {
        "num_requests_processed": 1,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 5,
        "num_running": 0,
        "num_waiting": 0,
    }
    metrics_client.cfg.engine = _fake_engine(stats_without_cancel)
    body = metrics_client.client.get("/metrics").text

    _assert_prom_counter(body, "rapid_mlx_requests_cancelled_total", 0)
    _assert_prom_counter(body, "rapid_mlx_requests_cancelled_via_disconnect_total", 0)


def test_metrics_route_renders_zero_when_both_counters_zero(metrics_client):
    """Quiet engine — both counters render at zero (not absent)."""
    quiet_stats = {
        "num_waiting": 0,
        "num_running": 0,
        "num_requests_processed": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "num_requests_cancelled": 0,
        "num_requests_cancelled_via_disconnect": 0,
    }
    metrics_client.cfg.engine = _fake_engine(quiet_stats)
    body = metrics_client.client.get("/metrics").text

    _assert_prom_counter(body, "rapid_mlx_requests_cancelled_total", 0)
    _assert_prom_counter(body, "rapid_mlx_requests_cancelled_via_disconnect_total", 0)
