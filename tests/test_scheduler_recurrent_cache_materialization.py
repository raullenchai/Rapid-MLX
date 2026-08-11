# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Metal handle exhaustion in hybrid text batches."""

from __future__ import annotations

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


def test_step_bounds_barrier_interval_and_preserves_event_order(monkeypatch):
    events = []
    response = object()

    class _StepGenerator:
        def next(self):
            events.append("next")
            return [response]

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = _StepGenerator()
    scheduler.running = {"request": object()}
    scheduler.finished_req_ids = set()
    scheduler._stateful_tombstones = set()
    scheduler._clear_cache_interval = 32
    scheduler._step_count = 0
    scheduler._memory_log_interval = 256
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
        assert responses == [response]
        events.append("responses")
        return [], set()

    monkeypatch.setattr(scheduler, "_process_batch_responses", process)
    monkeypatch.setattr(scheduler, "_cleanup_finished", lambda finished: None)

    for _ in range(9):
        scheduler.step()

    assert events[:3] == ["next", "barrier", "responses"]
    assert events[-3:] == ["next", "barrier", "responses"]
    assert events.count("barrier") == 2
    assert events.count("next") == events.count("responses") == 9
