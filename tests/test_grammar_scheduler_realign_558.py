# SPDX-License-Identifier: Apache-2.0
"""Scheduler grammar/penalty processor↔uid realignment (#558 PR-3, codex).

mlx-lm's ``GenerationBatch`` stores ``logits_processors`` as a positional
per-uid list. When a NO-processor request finishes while a grammar request is
mid-flight, a stale entry can survive the filter and DESYNC that list from
``uids`` — mlx-lm's ``_step`` then iterates ``range(len(uids))`` and applies
the wrong (or no) processor, silently leaving an explicit ``required``/named
call unconstrained (observed live on qwen3.5-4b: the first tool request after
the boot warmup fell back to free-form).

``Scheduler._realign_grammar_logits_processors`` rebuilds the positional list
from AUTHORITATIVE per-uid state: ``uid_to_request_processors`` holds the FULL
processor list (grammar + penalties) for EVERY live uid that carries any
processor — grammar AND penalty-only — so a bystander's penalties survive a
length-desync rebuild (codex #3). A ``_known_stateful_processors`` identity set
(with tombstones for finished-but-still-leaked grammars) scrubs leaked grammars
from untracked slots. These tests drive that method against a stand-in
generation batch reproducing the desync shapes — no model needed.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx


class _FakeBatchGen:
    """Minimal stand-in for mlx-lm's BatchGenerator + its _generation_batch."""

    def __init__(self, uids, logits_processors):
        self._generation_batch = SimpleNamespace(
            uids=list(uids),
            logits_processors=[list(x) for x in logits_processors],
        )


def _make_scheduler_stub():
    """Bind the realign + forget methods onto a bare state object (no model)."""
    from vllm_mlx.scheduler import Scheduler

    stub = SimpleNamespace(
        uid_to_request_processors={},
        _uids_with_grammar=set(),
        _uids_with_reasoning_budget={},
        _uids_with_suppressed_tokens={},
        _known_stateful_processors=set(),
        _stateful_processor_objs={},
        _stateful_tombstones=set(),
        batch_generator=None,
    )
    stub._realign_grammar_logits_processors = (
        Scheduler._realign_grammar_logits_processors.__get__(stub, type(stub))
    )
    stub._flush_stateful_tombstones = Scheduler._flush_stateful_tombstones.__get__(
        stub, type(stub)
    )
    stub._forget_uid_grammar = Scheduler._forget_uid_grammar.__get__(stub, type(stub))
    stub._realign_guard_armed = Scheduler._realign_guard_armed.__get__(stub, type(stub))
    stub._register_uid_processors = Scheduler._register_uid_processors.__get__(
        stub, type(stub)
    )
    return stub


def _register(stub, uid, processors, grammar=None, budget=None, suppression=None):
    """Register a uid through the PRODUCTION bookkeeping helper.

    Delegates to ``Scheduler._register_uid_processors`` (the exact method the
    admission path calls) rather than hand-reproducing its writes, so these
    tests break if the real registration drops a grammar/budget arm or a penalty
    list (codex NIT). ``grammar=None`` registers a penalty-only uid (tracked for
    desync repair but not a grammar). A non-None ``grammar`` also registers its
    identity + arms the guard. A non-None ``budget`` arms the guard for a
    generation-time reasoning-budget processor AND registers its identity in the
    same stateful set/tombstone machinery as grammar — a leaked force-close
    processor must be scrubbed from a foreign slot exactly like a grammar (codex).
    """
    request = SimpleNamespace(
        reasoning_budget_logits_processor=budget,
        suppressed_tokens_logits_processor=suppression,
    )
    stub._register_uid_processors(
        uid, request, list(processors) if processors else None, grammar
    )


def test_realign_repairs_stale_leading_entry():
    # The live-repro shape: a finished no-processor uid left a stale empty
    # entry at index 0; the grammar uid sits at index 1 but is the ONLY live
    # uid. Realign must place the grammar processor at the grammar uid's real
    # position.
    glp = object()
    stub = _make_scheduler_stub()
    _register(stub, 42, [glp], glp)
    stub.batch_generator = _FakeBatchGen(uids=[42], logits_processors=[[], [glp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert len(lp) == 1, "logits_processors must be realigned to len(uids)"
    assert lp[0] == [glp], "the grammar processor must land at its uid's index"


def test_realign_two_uids_grammar_not_leaked():
    # A grammar request (uid 7) batched alongside a plain request (uid 9).
    # uid 7 -> [glp]; uid 9 -> [] (no grammar leaked onto it) even when the
    # desynced list wrongly had glp on both slots.
    glp = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp], glp)
    stub.batch_generator = _FakeBatchGen(uids=[7, 9], logits_processors=[[glp], [glp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [glp], "uid 7 must keep its grammar processor"
    assert lp[1] == [], "uid 9 must NOT carry another uid's grammar processor"


def test_realign_scrubs_finished_token_suppression_from_plain_request():
    suppression = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [suppression], suppression=suppression)
    assert stub._realign_guard_armed()

    stub._forget_uid_grammar(7)
    assert id(suppression) in stub._stateful_tombstones
    stub.batch_generator = _FakeBatchGen(uids=[9], logits_processors=[[suppression]])

    stub._realign_grammar_logits_processors()

    assert stub.batch_generator._generation_batch.logits_processors == [[]]
    assert id(suppression) not in stub._known_stateful_processors


def test_realign_preserves_penalties_on_grammar_uid():
    # A grammar uid's penalty processors must be PRESERVED — the authoritative
    # per-uid list carries grammar + penalties in insert order.
    glp = object()
    penalty = object()
    stub = _make_scheduler_stub()
    _register(stub, 5, [penalty, glp], glp)
    # Desynced batch (length mismatch) — a naive rebuild would drop penalty.
    stub.batch_generator = _FakeBatchGen(
        uids=[5], logits_processors=[[], [penalty, glp]]
    )

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [penalty, glp], "grammar uid must keep penalty + grammar in order"


def test_realign_preserves_untracked_penalty_when_aligned():
    # A penalty-only uid keeps its penalties when the list is length-aligned.
    glp = object()
    penalty = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp], glp)  # a grammar uid must exist to trigger realign
    _register(stub, 9, [penalty])  # penalty-only uid, now TRACKED
    stub.batch_generator = _FakeBatchGen(
        uids=[7, 9], logits_processors=[[glp], [penalty]]
    )

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [glp]
    assert lp[1] == [penalty], "tracked penalty-only uid's penalty must be preserved"


def test_realign_preserves_penalty_only_uid_on_desync():
    # codex #3 (the load-bearing regression): when the positional list is
    # LENGTH-DESYNCED, a penalty-only bystander must NOT lose its penalties.
    # Before the fix, an untracked uid got [] on a desync, permanently deleting
    # its repetition/frequency/presence processors (running requests are never
    # re-inserted). Now every processor-carrying uid is tracked, so its slot is
    # reconstructed verbatim from uid-keyed state.
    glp = object()
    penalty = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp], glp)
    _register(stub, 9, [penalty])  # penalty-only, tracked
    # Desync: a stale entry makes the positional list longer than uids.
    stub.batch_generator = _FakeBatchGen(
        uids=[7, 9], logits_processors=[[], [glp], [penalty]]
    )

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert len(lp) == 2, "realigned to len(uids)"
    assert lp[0] == [glp]
    assert lp[1] == [penalty], (
        "penalty-only uid must keep its penalties across a length-desync — "
        "not be zeroed (codex #3)"
    )


def test_realign_untracked_uid_with_no_processors_gets_empty_slot():
    # A truly processor-free uid (never registered) gets an empty slot, and any
    # grammar that leaked into it is scrubbed.
    glp = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp], glp)
    # uid 9 was never registered (no processors) but a desync leaked glp in.
    stub.batch_generator = _FakeBatchGen(uids=[7, 9], logits_processors=[[glp], [glp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [glp]
    assert lp[1] == [], "processor-free uid must get an empty, grammar-scrubbed slot"


def test_realign_scrubs_finished_grammar_via_tombstone():
    # codex #1: a grammar whose owning uid already FINISHED can still linger in
    # another slot. _forget_uid_grammar TOMBSTONES it (keeps it in
    # _known_stateful_processors) so realign still scrubs it; only once it's
    # absent from every live slot is it fully forgotten.
    glp_live = object()
    glp_dead = object()  # finished owner, but leaked into uid 9's slot
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp_live], glp_live)
    _register(stub, 8, [glp_dead], glp_dead)
    # uid 8 finishes: forget tombstones glp_dead (still known + scrubbable).
    stub._forget_uid_grammar(8)
    assert id(glp_dead) in stub._known_stateful_processors, (
        "must be tombstoned, not gone"
    )
    assert id(glp_dead) in stub._stateful_tombstones
    # A leaked slot still carries glp_dead at uid 9's (untracked) position.
    stub.batch_generator = _FakeBatchGen(
        uids=[7, 9], logits_processors=[[glp_live], [glp_dead]]
    )

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [glp_live]
    assert lp[1] == [], "a finished (tombstoned) grammar must be scrubbed"
    # Now that it's absent from every slot, the tombstone is flushed.
    assert id(glp_dead) not in stub._known_stateful_processors
    assert id(glp_dead) not in stub._stateful_tombstones
    assert id(glp_dead) not in stub._stateful_processor_objs


def test_tombstone_survives_until_absent_then_flushes():
    # A tombstoned grammar that is STILL present in a leaked slot is NOT
    # forgotten on this tick — it stays known so the NEXT tick can still scrub
    # it. Only when absent from every slot does the flush drop it. This closes
    # the cleanup-ordering gap where the last grammar finishing would disarm the
    # guard before its leaked slot was cleaned.
    glp_dead = object()
    stub = _make_scheduler_stub()
    _register(stub, 8, [glp_dead], glp_dead)
    stub._forget_uid_grammar(8)  # tombstoned; uid_to_request_processors now empty
    # A stale slot still holds glp_dead at an untracked position. The guard is
    # armed by the tombstone even though uid_to_request_processors is empty.
    assert stub._stateful_tombstones, "tombstone must arm the realign guard"
    stub.batch_generator = _FakeBatchGen(uids=[9], logits_processors=[[glp_dead]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [], "the tombstoned grammar is scrubbed from the leaked slot"
    # Scrubbed this tick -> now absent -> flushed.
    assert id(glp_dead) not in stub._known_stateful_processors


def test_realign_noop_when_already_correct():
    glp = object()
    stub = _make_scheduler_stub()
    _register(stub, 1, [glp], glp)
    stub.batch_generator = _FakeBatchGen(uids=[1], logits_processors=[[glp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp == [[glp]]


def test_penalty_only_uid_does_not_arm_the_guard():
    # codex #558-PR3 (perf): a penalty-only request is tracked in
    # uid_to_request_processors for desync repair, but MUST NOT be in
    # _uids_with_grammar — the realign guard arms on _uids_with_grammar (not the
    # broader processor map), so plain penalty traffic never triggers an
    # O(batch) rebuild every token.
    penalty = object()
    stub = _make_scheduler_stub()
    _register(stub, 9, [penalty])  # penalty-only, grammar=None

    assert 9 in stub.uid_to_request_processors, "tracked for desync repair"
    assert 9 not in stub._uids_with_grammar, "penalty-only must NOT arm the guard"
    assert not stub._known_stateful_processors
    assert not stub._stateful_tombstones


def test_forget_penalty_only_uid_creates_no_tombstone():
    penalty = object()
    stub = _make_scheduler_stub()
    _register(stub, 9, [penalty])  # penalty-only

    stub._forget_uid_grammar(9)

    assert 9 not in stub.uid_to_request_processors
    assert not stub._stateful_tombstones, "a penalty-only uid must not tombstone"
    assert not stub._known_stateful_processors


def test_forget_uid_stateful_tombstones_then_flush_drops_state():
    glp = object()
    stub = _make_scheduler_stub()
    _register(stub, 1, [glp], glp)
    assert id(glp) in stub._known_stateful_processors

    stub._forget_uid_grammar(1)

    # uid-keyed state is gone immediately; the grammar identity is TOMBSTONED
    # (still known so a leaked slot can be scrubbed) until a flush confirms it's
    # absent from every slot.
    assert 1 not in stub.uid_to_request_processors
    assert 1 not in stub._uids_with_grammar
    assert id(glp) in stub._known_stateful_processors, "tombstoned, not yet dropped"
    assert id(glp) in stub._stateful_tombstones

    # No slot holds it -> flush drops it entirely.
    stub._flush_stateful_tombstones(present=set())
    assert id(glp) not in stub._known_stateful_processors
    assert id(glp) not in stub._stateful_processor_objs
    assert id(glp) not in stub._stateful_tombstones


def test_realign_handles_missing_generation_batch():
    stub = _make_scheduler_stub()
    _register(stub, 1, [object()], object())
    stub.batch_generator = SimpleNamespace(_generation_batch=None)
    stub._realign_grammar_logits_processors()  # no raise


def test_realign_handles_empty_uids_flushes_tombstones():
    glp_dead = object()
    stub = _make_scheduler_stub()
    _register(stub, 1, [glp_dead], glp_dead)
    stub._forget_uid_grammar(1)  # tombstoned
    stub.batch_generator = _FakeBatchGen(uids=[], logits_processors=[])
    stub._realign_grammar_logits_processors()  # no raise
    # No live slots -> tombstone is trivially absent everywhere -> flushed.
    assert id(glp_dead) not in stub._known_stateful_processors


def test_realign_empty_uids_scrubs_stale_processor_list():
    # codex #558-PR3 blocking: the LAST grammar request finished and the batch
    # emptied its ``uids`` WITHOUT the parallel positional ``logits_processors``
    # list being cleared (mlx-lm keys the two separately). If realign only
    # flushed the tombstone and returned, the leaked ``[glp_dead]`` at index 0
    # would be inherited by the NEXT admitted plain request and silently
    # constrain it. The no-live-slots branch must scrub the stale list to ``[]``
    # while the grammar identity is still known.
    glp_dead = object()
    stub = _make_scheduler_stub()
    _register(stub, 1, [glp_dead], glp_dead)
    stub._forget_uid_grammar(1)  # tombstoned (uid gone, processor may linger)
    # uids empty, but a stale processor slot survived the filter.
    stub.batch_generator = _FakeBatchGen(uids=[], logits_processors=[[glp_dead]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp == [], (
        "a batch with zero uids must carry zero processors — the leaked grammar "
        "slot must be scrubbed so the next plain request cannot inherit it"
    )
    # And the tombstone is flushed now that the grammar is absent everywhere.
    assert id(glp_dead) not in stub._known_stateful_processors
    assert id(glp_dead) not in stub._stateful_tombstones


# ── generation-time reasoning-budget processors ───────────────────────
#
# A budget processor (force-close </think>) is a per-request logits processor
# with the same desync exposure as a grammar: mlx-lm's positional list drops it
# when a no-processor request finishes while the budget request is mid-flight.
# Observed live on qwen3-0.6b: a plain greedy request preceding a budget request
# left the budget inert (the </think> was never forced, reasoning ran unbounded)
# because the budget uid did NOT arm the realign guard. So a live budget uid now
# arms it too — its slot is rebuilt from ``uid_to_request_processors`` every tick.


def test_realign_repairs_budget_processor_stale_leading_entry():
    # The live-repro shape: a finished no-processor uid left a stale empty entry
    # at index 0; the budget uid sits at index 1 but is the ONLY live uid.
    # Realign must place the budget processor at the budget uid's real position.
    rblp = object()
    stub = _make_scheduler_stub()
    _register(stub, 42, [rblp], budget=rblp)
    stub.batch_generator = _FakeBatchGen(uids=[42], logits_processors=[[], [rblp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert len(lp) == 1, "logits_processors must be realigned to len(uids)"
    assert lp[0] == [rblp], "the budget processor must land at its uid's index"


def test_realign_preserves_budget_processor_on_length_desync():
    # A budget request (uid 7) batched with a plain request (uid 9). A stale
    # entry length-desyncs the positional list — the budget processor must NOT
    # be zeroed off its uid (the exact bug: reasoning ran unbounded).
    rblp = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [rblp], budget=rblp)
    stub.batch_generator = _FakeBatchGen(
        uids=[7, 9], logits_processors=[[], [rblp], []]
    )

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert len(lp) == 2, "realigned to len(uids)"
    assert lp[0] == [rblp], "budget uid must keep its force-close processor"
    assert lp[1] == [], "plain uid carries no processor"


def test_realign_preserves_budget_plus_penalties_in_order():
    # A budget uid may also carry penalties; the authoritative per-uid list keeps
    # both in insert order (penalties first, budget appended LAST so its
    # force-close mask has final say).
    penalty = object()
    rblp = object()
    stub = _make_scheduler_stub()
    _register(stub, 5, [penalty, rblp], budget=rblp)
    stub.batch_generator = _FakeBatchGen(
        uids=[5], logits_processors=[[], [penalty, rblp]]
    )

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [penalty, rblp], "budget uid keeps penalty + budget in order"


def test_realign_grammar_and_budget_coexist_on_distinct_uids():
    # A grammar request and a budget request in the same batch each keep their
    # own processor — neither leaks onto the other.
    glp = object()
    rblp = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp], grammar=glp)
    _register(stub, 9, [rblp], budget=rblp)
    stub.batch_generator = _FakeBatchGen(uids=[7, 9], logits_processors=[[glp], [rblp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp[0] == [glp]
    assert lp[1] == [rblp]


def test_realign_guard_armed_by_budget_uid_alone():
    # Exercises the PRODUCTION arming predicate the scheduler step loop calls, so
    # deleting the budget arm fails here (not only on live hardware). A live
    # budget uid — with NO grammar and NO tombstone — must arm the realign.
    rblp = object()
    stub = _make_scheduler_stub()
    assert stub._realign_guard_armed() is False, "empty batch: no realign"
    _register(stub, 9, [rblp], budget=rblp)
    assert stub._realign_guard_armed() is True, "a budget uid must arm the guard"
    # It arms via the live-budget set, NOT as a grammar uid...
    assert 9 in stub._uids_with_reasoning_budget
    assert 9 not in stub._uids_with_grammar
    # ...but its identity IS tracked in the stateful set so a leaked force-close
    # processor can be tombstoned + scrubbed like a grammar (codex #1).
    assert id(rblp) in stub._known_stateful_processors
    assert not stub._stateful_tombstones, "not tombstoned while still live"


def test_realign_guard_disarms_when_budget_processor_ended():
    # codex R10 #4: once a budget processor has forced </think> (``_ended``) it is
    # inert — it must STOP arming the guard so the answer-token decode loop does
    # not pay an O(batch) slot rebuild per token. A live (not-ended) budget uid
    # still arms; flipping ``_ended`` disarms it (no grammar / no tombstone here).
    class _Budget:
        def __init__(self):
            self._ended = False

    rblp = _Budget()
    stub = _make_scheduler_stub()
    _register(stub, 9, [rblp], budget=rblp)
    assert stub._realign_guard_armed() is True, "active budget uid arms the guard"
    rblp._ended = True
    assert stub._realign_guard_armed() is False, (
        "an ended (inert) budget processor must no longer arm the guard"
    )
    # Still tracked (uid live) so a later finish/leak is tombstoned + scrubbed.
    assert 9 in stub._uids_with_reasoning_budget


def test_realign_guard_not_armed_by_penalty_only_uid():
    # A penalty-only uid is tracked for desync repair but MUST NOT arm the guard
    # (perf: no O(batch) rebuild on the plain decode hot path).
    penalty = object()
    stub = _make_scheduler_stub()
    _register(stub, 9, [penalty])  # penalty-only, no grammar, no budget
    assert 9 in stub.uid_to_request_processors
    assert stub._realign_guard_armed() is False


def test_realign_guard_armed_by_grammar_uid():
    glp = object()
    stub = _make_scheduler_stub()
    _register(stub, 7, [glp], grammar=glp)
    assert stub._realign_guard_armed() is True


def test_forget_budget_uid_tombstones_until_scrubbed():
    # A finished budget uid must TOMBSTONE its force-close processor (not forget
    # it outright): mlx-lm can leave the stateful processor in a leaked
    # positional slot for a tick, and the tombstone keeps the realign guard armed
    # so that slot is scrubbed before the processor is dropped (codex #1).
    rblp = object()
    stub = _make_scheduler_stub()
    _register(stub, 9, [rblp], budget=rblp)

    stub._forget_uid_grammar(9)

    assert 9 not in stub._uids_with_reasoning_budget, "live arming cleared"
    assert 9 not in stub.uid_to_request_processors, "per-uid state dropped"
    # Tombstoned, NOT forgotten — the guard stays armed for the scrub tick.
    assert id(rblp) in stub._stateful_tombstones
    assert id(rblp) in stub._known_stateful_processors
    assert stub._realign_guard_armed(), "tombstone keeps the guard armed"


def test_leaked_budget_processor_scrubbed_from_foreign_slot_then_flushed():
    # codex #1 end-to-end: budget uid 5 finishes, but mlx-lm leaves its
    # force-close processor leaked in a stale slot alongside a later no-processor
    # uid 6. The realign must scrub the leak (uid 6 -> []) and, once the leaked
    # processor is absent from every slot, flush the tombstone + disarm — without
    # this, the stateful </think> latch would force-close uid 6's output.
    rblp = object()
    stub = _make_scheduler_stub()
    _register(stub, 5, [rblp], budget=rblp)
    stub._forget_uid_grammar(5)  # uid 5 done → rblp tombstoned
    # mlx-lm's lagging filter: rblp leaked onto uid 6's positional slot.
    stub.batch_generator = _FakeBatchGen(uids=[6], logits_processors=[[rblp]])

    stub._realign_grammar_logits_processors()

    lp = stub.batch_generator._generation_batch.logits_processors
    assert lp == [[]], "leaked force-close processor scrubbed from the foreign uid"
    assert not stub._stateful_tombstones, "tombstone flushed once absent from slots"
    assert not stub._realign_guard_armed(), "guard disarms after the scrub"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
