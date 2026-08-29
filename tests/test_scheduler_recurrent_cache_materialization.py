# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Metal handle exhaustion in hybrid text batches."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from types import SimpleNamespace

from vllm_mlx import scheduler as scheduler_module
from vllm_mlx.scheduler import Scheduler


class _Node:
    def __init__(self, parent=None):
        self.parent = parent


class _Layer:
    def __init__(self, *, trimmable: bool):
        self._trimmable = trimmable
        self.head = _Node()

    @property
    def state(self):
        return [self.head]

    def is_trimmable(self):
        return self._trimmable

    def advance(self):
        self.head = _Node(self.head)


class _UnclassifiableLayer(_Layer):
    def is_trimmable(self):
        raise RuntimeError("unknown cache classification")


class _UnclassifiedLayer:
    def __init__(self):
        self.head = _Node()

    @property
    def state(self):
        return [self.head]


class _Batch:
    def __init__(self, cache):
        self.prompt_cache = cache


class _Generator:
    def __init__(self, cache, *, modern=True):
        if modern:
            self._generation_batch = _Batch(cache)
        else:
            self.active_batch = _Batch(cache)


def _scheduler_with(cache, *, modern=True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = _Generator(cache, modern=modern)
    return scheduler


def _chain_length(node):
    length = 0
    while node is not None:
        length += 1
        node = node.parent
    return length


def test_recurrent_cache_barrier_bounds_lazy_decode_chain(monkeypatch):
    """Fixed live cache count must not hide a per-step Metal handle chain.

    ``_Node.parent`` models MLX's functional recurrent-state update: the live
    head retains every prior update until ``mx.eval`` realizes and detaches it.
    One cache entry and 1,000 decode steps therefore reproduce the issue's
    count-vs-bytes distinction without loading 35B weights or waiting 90 min.
    """
    recurrent = _Layer(trimmable=False)
    dense = _Layer(trimmable=True)
    scheduler = _scheduler_with([dense, recurrent])
    evaluations = []

    def evaluate(states):
        evaluations.append(states)
        for state in states:
            for node in state:
                node.parent = None

    monkeypatch.setattr(scheduler_module.mx, "eval", evaluate)

    for _ in range(1_000):
        recurrent.advance()
        dense.advance()
        assert scheduler._materialize_active_recurrent_cache() == 1

    assert len(evaluations) == 1_000
    assert evaluations[-1] == [[recurrent.head]]
    assert _chain_length(recurrent.head) == 1


def test_dense_batch_does_not_pay_per_token_eval(monkeypatch):
    dense = _Layer(trimmable=True)
    scheduler = _scheduler_with([dense])
    evaluations = []
    monkeypatch.setattr(
        scheduler_module.mx, "eval", lambda value: evaluations.append(value)
    )

    assert scheduler._materialize_active_recurrent_cache() == 0
    assert evaluations == []


def test_unknown_cache_classification_fails_safe_to_materialization(monkeypatch):
    unknown = _UnclassifiableLayer(trimmable=True)
    scheduler = _scheduler_with([unknown])
    evaluations = []
    monkeypatch.setattr(
        scheduler_module.mx, "eval", lambda value: evaluations.append(value)
    )

    assert scheduler._materialize_active_recurrent_cache() == 1
    assert evaluations == [[[unknown.head]]]


def test_missing_cache_classification_fails_safe_to_materialization(monkeypatch):
    unknown = _UnclassifiedLayer()
    scheduler = _scheduler_with([unknown])
    evaluations = []
    monkeypatch.setattr(
        scheduler_module.mx, "eval", lambda value: evaluations.append(value)
    )

    assert scheduler._materialize_active_recurrent_cache() == 1
    assert evaluations == [[[unknown.head]]]


def test_legacy_active_batch_surface_is_covered(monkeypatch):
    recurrent = _Layer(trimmable=False)
    scheduler = _scheduler_with([recurrent], modern=False)
    evaluations = []
    monkeypatch.setattr(
        scheduler_module.mx, "eval", lambda value: evaluations.append(value)
    )

    assert scheduler._materialize_active_recurrent_cache() == 1
    assert evaluations == [[[recurrent.head]]]


def test_recurrent_materialize_interval_is_batch_adaptive():
    """The barrier widens at low batch and holds the #1834 every-8 floor
    under concurrency."""
    s = Scheduler.__new__(Scheduler)
    assert s._recurrent_materialize_interval(1) == 64
    assert s._recurrent_materialize_interval(2) == 32
    assert s._recurrent_materialize_interval(4) == 16
    assert s._recurrent_materialize_interval(8) == 8
    assert s._recurrent_materialize_interval(16) == 8
    assert s._recurrent_materialize_interval(0) == 64  # guarded lower bound


def test_recurrent_materialize_interval_never_raises_steadystate_handles():
    """Steady-state safety: at a FIXED batch depth, ``active × interval``
    never exceeds what the flat every-8 barrier already tolerated
    (``active × 8`` under concurrency, the 64-unit budget below it). Batch
    TRANSITIONS are covered by the step-driven test below — a constant
    depth cannot exercise the drift codex #1895 flagged."""
    s = Scheduler.__new__(Scheduler)
    for active in range(1, 65):
        interval = s._recurrent_materialize_interval(active)
        flat_every_8 = active * scheduler_module._RECURRENT_CACHE_MATERIALIZE_INTERVAL
        budget = scheduler_module._RECURRENT_MATERIALIZE_HANDLE_BUDGET
        assert active * interval <= max(budget, flat_every_8)


def _make_step_scheduler(monkeypatch, running, events):
    """A ``__new__`` scheduler wired so ``step()`` runs only the barrier /
    response bookkeeping, appending 'next' / 'barrier' / 'responses' markers
    to ``events``. ``_clear_cache_interval`` is parked high so the unrelated
    cache-clear path never fires within these short runs."""
    scheduler = Scheduler.__new__(Scheduler)

    class _StepGenerator:
        def next(self):
            events.append("next")
            return [object()]

    scheduler.batch_generator = _StepGenerator()
    scheduler.running = running
    scheduler.finished_req_ids = set()
    scheduler._stateful_tombstones = set()
    scheduler._clear_cache_interval = 100_000
    scheduler._step_count = 0
    scheduler._recurrent_chain_depth = 0
    # Default: already-active scheduler, so the idle->active arm does NOT
    # fire on step 1 — cadence tests exercise the depth counter in isolation.
    # The arm test overrides this to 0.
    scheduler._recurrent_prev_running = 1
    scheduler._memory_log_interval = 100_000
    scheduler.config = SimpleNamespace()

    monkeypatch.setattr(scheduler, "_process_pending_aborts", lambda: None)
    monkeypatch.setattr(scheduler, "_reconcile_orphaned_running_requests", lambda: [])
    monkeypatch.setattr(scheduler, "_schedule_waiting", lambda: [])
    monkeypatch.setattr(scheduler, "_realign_guard_armed", lambda: False)
    monkeypatch.setattr(scheduler, "_apply_adaptive_prefill_size", lambda: None)
    monkeypatch.setattr(
        scheduler,
        "_materialize_active_recurrent_cache",
        lambda: events.append("barrier"),
    )

    def process(responses):
        events.append("responses")
        return [], set()

    monkeypatch.setattr(scheduler, "_process_batch_responses", process)
    monkeypatch.setattr(scheduler, "_cleanup_finished", lambda finished: None)
    return scheduler


def test_step_barrier_fires_off_chain_depth_at_b1(monkeypatch):
    """At B=1 the barrier widens to its 64-step max interval: 64 steps span
    exactly one barrier, landing on the step the chain depth reaches the
    interval, with event order next -> barrier -> responses preserved."""
    events = []
    scheduler = _make_step_scheduler(monkeypatch, {"r0": object()}, events)

    for _ in range(64):
        scheduler.step()

    assert events.count("barrier") == 1
    assert events[-3:] == ["next", "barrier", "responses"]
    assert events.count("next") == events.count("responses") == 64


def test_step_barrier_fires_immediately_when_batch_grows(monkeypatch):
    """codex #1895: the barrier keys off live chain DEPTH, not the global
    step counter, so a deep B=1 chain materializes the moment concurrency
    rises past its (now tighter) interval — it never drifts to the next
    global-step multiple and overshoots the handle budget."""
    events = []
    running = {"r0": object()}
    scheduler = _make_step_scheduler(monkeypatch, running, events)

    # 45 steps at B=1 (interval 64): chain depth grows to 45, no barrier.
    # 45 is deliberately NOT a multiple of the every-8 floor, so the old
    # ``_step_count % interval`` logic would drift to step 48 here — this
    # is the exact case codex #1895 flagged.
    for _ in range(45):
        scheduler.step()
    assert "barrier" not in events

    # Batch jumps to 8 -> interval drops to 8. Depth (45) already exceeds
    # it, so the barrier MUST fire on the very next step, not drift to a
    # later global-step multiple.
    for i in range(1, 8):
        running[f"r{i}"] = object()
    before = len(events)
    scheduler.step()
    assert "barrier" in events[before:], (
        "deep B=1 chain did not materialize when the batch grew to 8"
    )

    # Depth reset -> it now cadences at the #1834 every-8 floor.
    tail = len(events)
    for _ in range(8):
        scheduler.step()
    assert events[tail:].count("barrier") == 1


def test_barrier_transient_bounded_at_depth63_growth_boundary(monkeypatch):
    """codex #1895 r4 — the tightest boundary: a B=1 chain at depth 63 that
    grows to B=8. ``next()`` advances all 8 rows before the barrier check, so
    the transient peaks at 64 + 7 = 71 handle-units for ONE step before the
    barrier fires (depth 64 >= interval(8)=8) and clears it.

    The invariant this proves is the real safety property — the transient is
    bounded to a SINGLE step and cleared immediately, never unbounded growth.
    Its absolute size (~71 units, i.e. ~3.4k Metal handles on a 48-layer
    hybrid) sits ~150x below the 499000-handle ceiling #1827 guards, so the
    bounded spike is not a resource risk; only unbounded chains are."""
    events = []
    running = {"r0": object()}
    scheduler = _make_step_scheduler(monkeypatch, running, events)

    for _ in range(63):
        scheduler.step()
    assert "barrier" not in events
    assert scheduler._recurrent_chain_depth == 63  # deepest a B=1 chain reaches

    for i in range(1, 8):
        running[f"r{i}"] = object()
    before = len(events)
    scheduler.step()  # depth 64 with 8 rows live -> barrier fires THIS step
    assert "barrier" in events[before:], "depth-63 growth boundary did not barrier"
    assert scheduler._recurrent_chain_depth == 0  # transient cleared in one step


def test_step_arms_barrier_on_idle_to_active_edge(monkeypatch):
    """codex #1895 r2+r3: a sequence entering an EMPTY batch (fresh scheduler
    or one that went idle) may carry a prefill-inherited recurrent graph, so
    the idle->active edge arms the barrier on that first decode step — the
    #1834 step-zero barrier generalized to every activation, not just
    construction."""
    events = []
    scheduler = _make_step_scheduler(monkeypatch, {"r0": object()}, events)
    scheduler._recurrent_prev_running = 0  # scheduler was idle

    scheduler.step()
    assert events[:3] == ["next", "barrier", "responses"]  # armed on activation

    # After the arm the depth resets and cadences at interval(1)=64.
    tail = len(events)
    for _ in range(63):
        scheduler.step()
    assert "barrier" not in events[tail:]  # depth 63 < 64, no second barrier yet
    scheduler.step()  # depth reaches 64
    assert events[-3:] == ["next", "barrier", "responses"]
