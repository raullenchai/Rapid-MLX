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


class _OutputNode:
    """Models a lazily-scheduled decode OUTPUT tensor (issue #2834).

    ``parent`` models the MLX lazy-graph reference an ``async_eval`` output
    retains to the prior step: on a recurrent/hybrid lane ``gb._next_tokens`` /
    ``_next_logprobs`` / ``token_context`` are re-scheduled by the forward each
    step and are NOT detached unless someone evaluates them. The #1834 barrier
    only realizes ``layer.state``, so this separate output chain can grow
    unboundedly and exhaust Metal's 499000-handle ceiling even though the cache
    state stays bounded.
    """

    def __init__(self, parent=None):
        self.parent = parent


class _TokenBuffer:
    """Models mlx_lm's ``TokenBuffer`` — the logits-processor
    ``token_context`` accumulator whose lazily grown buffer feeds ``_step``'s
    ``async_eval``."""

    def __init__(self):
        self._buf = _OutputNode()

    @property
    def tokens(self):
        return self._buf


class _OutputBatch:
    """A recurrent generation batch exposing the per-step output chain the
    scheduler's barrier must realize."""

    def __init__(self, cache):
        self.prompt_cache = cache
        self._next_tokens = _OutputNode()
        self._next_logprobs = [_OutputNode(), _OutputNode()]
        self._token_context = [_TokenBuffer(), _TokenBuffer()]

    def advance_outputs(self):
        # One recurrent decode step: every output re-references the prior step.
        self._next_tokens = _OutputNode(self._next_tokens)
        self._next_logprobs = [_OutputNode(p) for p in self._next_logprobs]
        for tc in self._token_context:
            tc._buf = _OutputNode(tc._buf)


class _OutputGenerator:
    def __init__(self, batch):
        self._generation_batch = batch


class _OutputRecurrentLayer:
    def __init__(self):
        self.head = _OutputNode()

    def is_trimmable(self):
        return False

    @property
    def state(self):
        return [self.head]


def _detaching_eval(values, seen=None):
    """A monkeypatched ``mx.eval`` that detaches every graph it is handed —
    exactly like a real realization pass — and records what it received."""
    if seen is None:
        seen = []
    if not isinstance(values, (list, tuple)):
        values = [values]
    for v in values:
        seen.append(v)
        if isinstance(v, (list, tuple)):
            _detaching_eval(v, seen=seen)
            continue
        node = v
        while hasattr(node, "parent") and node.parent is not None:
            node.parent = None
            node = node.parent
    return len(values)


def test_recurrent_barrier_realizes_per_step_output_chain(monkeypatch):
    """#2834 regression: the hybrid barrier must realize the per-step decode
    OUTPUT chain (``_next_tokens`` / ``_next_logprobs`` / ``token_context``),
    not just ``layer.state``. Before the fix this chain was never evaluated, so
    it grew unboundedly; after the fix the barrier detaches it on the SAME
    cadence as the cache state."""
    recurrent = _OutputRecurrentLayer()
    batch = _OutputBatch([recurrent])
    generator = _OutputGenerator(batch)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = generator

    seen = []
    monkeypatch.setattr(
        scheduler_module.mx,
        "eval",
        lambda *a, **k: _detaching_eval(*a, seen=seen),
    )

    # Build an unbounded output chain exactly the way a running recurrent
    # model would: many decode steps with no intervening realization.
    for _ in range(1_000):
        batch.advance_outputs()

    # The chain is deep before the barrier (the bug state).
    assert _chain_length(batch._next_tokens) > 900

    scheduler._materialize_active_recurrent_cache()

    # The barrier realized the cache state AND detached the output chain.
    assert _chain_length(batch._next_tokens) == 1
    for lp in batch._next_logprobs:
        assert _chain_length(lp) == 1
    for tc in batch._token_context:
        assert _chain_length(tc._buf) == 1

    # Every output surface was handed to mx.eval at least once.
    seen_kinds = {id(v) for v in seen}
    assert id(batch._next_tokens) in seen_kinds
    for lp in batch._next_logprobs:
        assert id(lp) in seen_kinds
    for tc in batch._token_context:
        assert id(tc._buf) in seen_kinds


def test_dense_batch_barrier_does_not_touch_output_chain(monkeypatch):
    """The dense no-per-token-eval guarantee must hold for the OUTPUT chain
    too: on a dense (all-trimmable) lane the barrier returns before reaching
    the output-chain realization, so it influences none of the batch's
    ``_next_*`` / ``token_context`` tensors."""
    dense = _Layer(trimmable=True)
    batch = _OutputBatch([dense])
    generator = _OutputGenerator(batch)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = generator

    for _ in range(200):
        batch.advance_outputs()
    deep = _chain_length(batch._next_tokens)
    assert deep > 100

    evals = []
    monkeypatch.setattr(scheduler_module.mx, "eval", lambda value: evals.append(value))

    assert scheduler._materialize_active_recurrent_cache() == 0
    assert evals == []
    # The output chain is untouched — dense lanes never realize it per step.
    assert _chain_length(batch._next_tokens) == deep


def test_barrier_tolerates_missing_output_surface(monkeypatch):
    """Hot-path safety: the output-chain passthrough must not raise when the
    batch lacks the per-step output attributes (older mlx-lm surfaces) or holds
    None placeholders — and it must never suppress the cache-state eval."""
    layer = _Layer(trimmable=False)
    batch = _NoOutputBatch([layer])
    generator = _GeneratorNoOut(batch)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = generator

    evals = []
    monkeypatch.setattr(scheduler_module.mx, "eval", lambda value: evals.append(value))

    assert scheduler._materialize_active_recurrent_cache() == 1
    assert evals == [[[layer.head]]]  # cache-state barrier still fired


class _NoOutputBatch:
    def __init__(self, cache):
        self.prompt_cache = cache
        # Deliberately NO _next_tokens / _next_logprobs / _token_context.


class _GeneratorNoOut:
    def __init__(self, batch):
        self._generation_batch = batch


def test_output_chain_failure_escalates_after_limit(monkeypatch):
    """#2834 (codex r1+r2 converge): a persistent OUTPUT-chain realize failure
    must escalate, not silently concede to an unbounded output graph — while a
    cache-state failure must propagate IMMEDIATELY (never suppressed).

    The barrier realizes cache state first, unguarded, then the output chain in
    a guarded block. So: (a) a failing output chain while cache state succeeds
    retries and escalates after ``_RECURRENT_OUTPUT_CHAIN_FAILURE_LIMIT``; (b) a
    failing cache state propagates on the very first barrier. Modelling the two
    independently is what round-2 codex demanded — under the earlier combined
    single-eval design this test could not express either correctly.
    """
    recurrent = _OutputRecurrentLayer()
    batch = _OutputBatch([recurrent])
    generator = _OutputGenerator(batch)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = generator
    scheduler._recurrent_output_chain_failures = 0
    limit = scheduler_module._RECURRENT_OUTPUT_CHAIN_FAILURE_LIMIT

    cache_head = recurrent.head

    # --- (a) cache-state succeeds, output-chain fails persistently ---
    def fail_only_outputs(value):
        # The cache-state eval realizes ``states`` = [[recurrent.head]]
        # (layer.state returns a one-element list). Every other realizable
        # tensor is part of the per-step output chain. Succeed (and detach)
        # the cache-head eval; fail any output-chain realize.
        values = value if isinstance(value, list) else [value]
        if values == [[cache_head]]:
            _detaching_eval(value)
            return
        raise RuntimeError("simulated persistent output-chain eval failure")

    monkeypatch.setattr(scheduler_module.mx, "eval", fail_only_outputs)

    for _ in range(limit - 1):
        assert scheduler._materialize_active_recurrent_cache() is not None
    assert scheduler._recurrent_output_chain_failures == limit - 1

    with pytest.raises(
        scheduler_module._RecurrentOutputChainError, match="Metal handle"
    ):
        scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == limit

    # Recover: a later successful realize resets the counter.
    monkeypatch.setattr(
        scheduler_module.mx, "eval", lambda value: _detaching_eval(value)
    )
    scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == 0

    # --- (b) cache-state failure propagates immediately (unguarded) ---
    def fail_cache(value):
        raise RuntimeError("simulated cache-state eval failure")

    monkeypatch.setattr(scheduler_module.mx, "eval", fail_cache)
    with pytest.raises(RuntimeError, match="simulated cache-state"):
        scheduler._materialize_active_recurrent_cache()


def test_output_chain_collection_failure_escalates_after_limit(monkeypatch):
    """#2834 (codex r4): a persistently RAISING output surface — as opposed to
    a merely ABSENT one — must feed the same escalation counter. Before r4,
    ``_collect_recurrent_outputs`` swallowed a raising ``_next_*`` /
    ``TokenBuffer.tokens`` access as if the surface were absent, the barrier
    saw ``collection_failed=False`` and no outputs, cleared the counter every
    step, and a persistently-uncollectable output chain never reached the
    escalation limit — the exact unbounded chain this barrier exists to bound.
    """
    recurrent = _OutputRecurrentLayer()
    batch = _RaisingOutputBatch([recurrent])
    generator = _OutputGenerator(batch)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = generator
    scheduler._recurrent_output_chain_failures = 0
    limit = scheduler_module._RECURRENT_OUTPUT_CHAIN_FAILURE_LIMIT

    # Cache-state realize succeeds; the ``_next_tokens`` surface is
    # uncollectable while ``_next_logprobs`` / ``token_context`` stay valid.
    # Record every value handed to mx.eval so we can assert the cache-state
    # barrier (`[[head]]`) still fires on each collection-failed step (codex
    # r5/r6) — it must not be skipped for up to `limit` intervals while
    # collection escalates — and that the valid partial outputs ARE realized
    # every step (r6#2: never discard a collectible output chain).
    evals = []

    def eval_and_record(value):
        evals.append(value)
        return _detaching_eval(value)

    monkeypatch.setattr(scheduler_module.mx, "eval", eval_and_record)

    cache_head = recurrent.head
    for _ in range(limit - 1):
        assert scheduler._materialize_active_recurrent_cache() is not None
    assert scheduler._recurrent_output_chain_failures == limit - 1
    # The cache-state eval (states = [[head]]) fired on EVERY step.
    assert evals.count([[cache_head]]) == limit - 1
    # The valid partial outputs (any non-cache-state realize) were realized
    # every step too — not discarded because one surface raised (r6#2).
    output_evals = [v for v in evals if v != [[cache_head]]]
    assert len(output_evals) == limit - 1
    assert all(isinstance(v, list) and v for v in output_evals)

    with pytest.raises(
        scheduler_module._RecurrentOutputChainError, match="Metal handle"
    ):
        scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == limit


def test_failure_counter_resets_when_recurrent_lane_changes(monkeypatch):
    """#2834 (codex r7 r8#1/#2): the output-chain failure counter is scoped to
    the ACTIVE generation batch. A streak accumulated on one recurrent request
    must NOT cascade into a later one when the batch IDENTITY changes — via a
    dense interval OR a DIRECT recurrent->recurrent replacement — otherwise a
    new request could hit the escalation limit on its FIRST failure, failing a
    lane that was never actually broken. One persistent scheduler is reused
    and its generation batch is replaced in place, so the test exercises the
    identity-tracking reset rather than depending on separate instances."""
    limit = scheduler_module._RECURRENT_OUTPUT_CHAIN_FAILURE_LIMIT

    monkeypatch.setattr(scheduler_module.mx, "eval", _detaching_eval)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._recurrent_output_chain_failures = 0
    scheduler._recurrent_output_chain_batch = None

    def _recurrent(layer):
        # Replace the active batch IN PLACE on the SAME scheduler: a direct
        # batch-identity change without an intervening idle/dense step.
        scheduler.batch_generator = _OutputGenerator(_RaisingOutputBatch([layer]))

    def _dense():
        # All-trimmable lane on the SAME scheduler.
        scheduler.batch_generator = _Generator([_Layer(trimmable=True)])

    # --- recurrent lane with a persistently-raising output surface ---
    _recurrent(_OutputRecurrentLayer())
    for _ in range(limit - 1):
        scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == limit - 1

    # --- DIRECT recurrent->recurrent replacement must reset the streak ---
    _recurrent(_OutputRecurrentLayer())  # new batch identity on the same lane
    scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == 1  # fresh streak

    # --- a NONZERO streak, then go dense: dense period must reset ---
    # (Not all the way to limit-1 again — a fresh counter of 1 plus (limit-1)
    # more would overflow into escalation. Two failures is enough to prove the
    # dense reset clears a nonzero streak.)
    scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == 2
    _dense()
    assert scheduler._materialize_active_recurrent_cache() == 0
    assert scheduler._recurrent_output_chain_failures == 0

    # --- back to a NEW recurrent batch: first failure is fresh (never limit) ---
    _recurrent(_OutputRecurrentLayer())
    scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == 1


class _RaisingOutputBatch(_OutputBatch):
    """An output batch whose ``_next_tokens`` accessor raises on every read —
    modelling a persistent mlx-lm patch-level incompatibility on that surface
    while ``layer.state`` (the cache-state barrier) stays fully intact.

    The base ``__init__`` assigns ``self._next_tokens`` directly, which the
    read-only property forbids, so this subclass wires its own surfaces (and
    the ``_token_context`` buffers ``advance_outputs`` mutates) without ever
    assigning ``_next_tokens``."""

    @property
    def _next_tokens(self):
        raise RuntimeError("simulated persistent output-surface access failure")

    def __init__(self, cache):
        self.prompt_cache = cache
        self._next_logprobs = [_OutputNode(), _OutputNode()]
        self._token_context = [_TokenBuffer(), _TokenBuffer()]


class _OnlyRaisingNextTokensBatch(_OutputBatch):
    """An output batch where EVERY surface is missing except ``_next_tokens``,
    which raises on access. The collector gathers NO outputs while recording a
    collection_error — exercising the 'nothing collectible AND a surface
    raised' escalation branch (codex r4/r5: the cache-state barrier already
    fired in the caller, so escalation is safe)."""

    @property
    def _next_tokens(self):
        raise RuntimeError("simulated only-raising output surface")

    def __init__(self, cache):
        self.prompt_cache = cache
        self._next_logprobs = None
        self._token_context = None


def test_output_chain_nothing_collectible_and_surface_raised_escalates(monkeypatch):
    """#2834 (coverage): when the collector gathers ZERO outputs AND a surface
    raised on access, ``_retry_materialize_output_chain`` must escalate
    (nothing was realized by mB, so the only thing keeping the graph bounded is
    escalating) — not silently clear the counter as if it were a clean no-op."""
    recurrent = _OutputRecurrentLayer()
    batch = _OnlyRaisingNextTokensBatch([recurrent])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = _OutputGenerator(batch)
    scheduler._recurrent_output_chain_failures = 0
    limit = scheduler_module._RECURRENT_OUTPUT_CHAIN_FAILURE_LIMIT

    monkeypatch.setattr(scheduler_module.mx, "eval", _detaching_eval)

    for _ in range(limit - 1):
        assert scheduler._materialize_active_recurrent_cache() is not None
    assert scheduler._recurrent_output_chain_failures == limit - 1

    with pytest.raises(
        scheduler_module._RecurrentOutputChainError, match="Metal handle"
    ):
        scheduler._materialize_active_recurrent_cache()


class _ScalarLogprobsRaisingContextBatch(_OutputBatch):
    """Covers the two remaining collector branches: (1) a scalar (non-list)
    ``_next_logprobs`` value is appended directly; (2) a ``TokenBuffer`` whose
    ``tokens`` accessor raises is recorded as a collection error rather than an
    absent surface — while the valid buffer and scalar logprobs still reach
    ``mx.eval`` (never discard a collectible output chain, r6#2)."""

    def __init__(self, cache):
        self.prompt_cache = cache
        self._next_tokens = _OutputNode()
        self._next_logprobs = _OutputNode()  # scalar, not a list
        self._token_context = [_TokenBufferWithRaisingTokens(), _TokenBuffer()]


class _TokenBufferWithRaisingTokens(_TokenBuffer):
    @property
    def tokens(self):
        raise RuntimeError("simulated TokenBuffer.tokens access failure")


def test_collector_scalar_logprobs_and_raising_token_buffer(monkeypatch):
    """#2834 (coverage): the collector must (a) append a scalar ``_next_logprobs``
    value and (b) treat a raising ``TokenBuffer.tokens`` as a collection error
    (not absence) — while still realizing the valid scalar logprobs and the
    valid token buffer. The raising surface routes into the escalation counter;
    the valid outputs are NOT discarded."""
    recurrent = _OutputRecurrentLayer()
    batch = _ScalarLogprobsRaisingContextBatch([recurrent])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.batch_generator = _OutputGenerator(batch)
    scheduler._recurrent_output_chain_failures = 0

    evals = []

    def eval_and_record(value):
        evals.append(value)
        return _detaching_eval(value)

    monkeypatch.setattr(scheduler_module.mx, "eval", eval_and_record)

    # One step: valid outputs (scalar logprobs + valid buffer tokens) are
    # realized, and the raising surface feeds the escalation counter (to 1,
    # below the limit so no raise).
    result = scheduler._materialize_active_recurrent_cache()
    assert scheduler._recurrent_output_chain_failures == 1

    # The cache-state barrier fired ([[head]]), and the collected outputs were
    # realized even though one surface raised (r6#2).
    realized = [v for v in evals if v != [[recurrent.head]]]
    assert realized, "expected the collected output chain to be realized"
    flattened = []
    for v in realized:
        flattened.extend(v if isinstance(v, list) else [v])
    assert id(batch._next_logprobs) in {id(x) for x in flattened}
    assert id(batch._token_context[1]._buf) in {id(x) for x in flattened}


def test_materialize_with_no_batch_generator_resets_counter(monkeypatch):
    """#2834 (coverage): with NO active lane, the barrier clears any failure
    counter left by a prior recurrent request and returns 0 — so a NEW request
    never inherits a stale streak (codex r7)."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._recurrent_output_chain_failures = 5
    scheduler._recurrent_output_chain_batch = object()
    scheduler.batch_generator = None

    assert scheduler._materialize_active_recurrent_cache() == 0
    assert scheduler._recurrent_output_chain_failures == 0
    assert scheduler._recurrent_output_chain_batch is None


def test_materialize_with_empty_cache_resets_counter(monkeypatch):
    """#2834 (coverage): a generation batch with NO ``prompt_cache`` (falsy)
    is nothing to materialize — the barrier clears a stale failure streak and
    returns without touching the output chain."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._recurrent_output_chain_failures = 3
    scheduler._recurrent_output_chain_batch = object()
    scheduler.batch_generator = _OutputGenerator(_OutputBatch([]))  # empty cache

    assert scheduler._materialize_active_recurrent_cache() == 0
    # The empty cache means nothing was materialized: the stale streak is
    # cleared so a NEW request never inherits it, and the batch-identity
    # tracking now points at the new (empty-cache) batch.
    assert scheduler._recurrent_output_chain_failures == 0
    assert (
        scheduler._recurrent_output_chain_batch
        is scheduler.batch_generator._generation_batch
    )
