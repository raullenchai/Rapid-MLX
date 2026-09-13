# SPDX-License-Identifier: Apache-2.0
"""Round N+1's drafter must start before round N's tokens leave the generator.

Why
---
``mtp_generate_step`` is a generator: everything the caller does per token --
detokenize, stop-string scan, SSE framing, socket write -- runs *inside* our
``yield``. The drafter chain is built lazily, so it does no device work until
some consumer materializes it, and its only consumer is the NEXT round's
single ``mx.eval``. Drafting last therefore issued the drafter forward after
the caller had finished draining the whole round, with the GPU idle through
all of it. Drafting first, then ``mx.async_eval``-ing the chain, puts that
forward into the gap. Measured on an M4 Pro over six prompts, interleaved
A/B/A/B: 24.01 -> 24.66 tok/s (+2.7%) with tok/round bit-identical at 2.4361
in every run, i.e. purely a scheduling win with no change to what is decoded.
It is the placement Ollama's ``x/mlxrunner/mtp.go`` uses for its
``mlx.AsyncEval`` after propose.

What this costs, and what these tests pin
-----------------------------------------
Moving the draft ahead of the yields means the verify path can no longer
``yield`` at each emission point and ``return`` on ``max_tokens``: the drafter
has to run between the last commit and the first delivery. So the round now
collects its emissions and hands them over afterwards. Two invariants get
subtle and are pinned here:

* **Guard state advances with DELIVERY, not with the collection pass.** Each
  emission carries the transactional-processor snapshot its position captured
  during verification, and delivery replays them in order -- even though the
  drafter has already run at the round's final state. A caller that stops
  pulling mid-round must leave the processors exactly at the last DELIVERED
  prefix, not at the round's end.
* **A round capped by ``max_tokens`` still pays for no draft.** ``round_done``
  stands in for the ``return`` each emission point used to take.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


class _ScriptedModel:
    """Deterministic backbone + MTP head that logs its forwards to ``events``.

    ``backbone`` is consumed one entry per (call, position) and becomes that
    position's argmax; ``mtp`` likewise per drafted position. A 50.0 logit on
    the scripted id makes greedy sampling pick it, so accept/reject at each
    verify position is decided purely by whether ``backbone`` repeats the
    ``mtp`` entry the draft proposed.
    """

    def __init__(self, backbone, mtp, events, vocab=32, hidden_size=8):
        self._backbone = list(backbone)
        self._mtp = list(mtp)
        self._b = 0
        self._m = 0
        self.events = events
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.layers = []
        self.mtp_calls = 0

    def _logits(self, ids, batch):
        rows = [
            mx.where(
                mx.arange(self.vocab)[None, :] == tid,
                mx.array(50.0),
                mx.array(0.0),
            )
            + mx.zeros((batch, self.vocab))
            for tid in ids
        ]
        return mx.stack(rows, axis=1)

    def _take(self, seq, cursor_name, n):
        cur = getattr(self, cursor_name)
        out = []
        for _ in range(n):
            out.append(seq[cur] if cur < len(seq) else 0)
            cur += 1
        setattr(self, cursor_name, cur)
        return out

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        batch, span = inputs.shape
        logits = self._logits(self._take(self._backbone, "_b", span), batch)
        if return_hidden:
            return logits, mx.zeros((batch, span, self.hidden_size))
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        span = next_token_ids.shape[1]
        self.mtp_calls += 1
        self.events.append(("draft", span))
        return self._logits(self._take(self._mtp, "_m", span), next_token_ids.shape[0])

    def make_mtp_cache(self):
        return []


class _RecordingProcessor:
    """Transactional processor double whose state is its committed prefix.

    ``mtp_apply`` runs once per verified position and moves the temporary
    state forward, so the snapshot the generator captures after each position
    is distinct. Every restore is logged, which is what lets these tests read
    the guard's walk directly instead of inferring it from emitted tokens.
    """

    def __init__(self, events):
        self.events = events
        self.state = 0
        self._served = 0

    def mtp_snapshot_state(self):
        return self.state

    def mtp_restore_state(self, state):
        self.state = state
        self.events.append(("restore", state))

    def mtp_apply(self, tokens, tentative, logits):
        self.events.append(("apply", self.state))
        self._served += 1
        self.state = self._served
        return logits

    def __call__(self, tokens, logits):  # ordinary (non-MTP) call path
        return logits


def _run(
    model,
    *,
    max_tokens,
    max_k=2,
    events=None,
    processors=None,
    stop_after=None,
    counter=None,
    **kwargs,
):
    """Drive the generator, logging each delivered token into ``events``."""
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    emitted = []
    stream = mtp_generate_step(
        mx.array([1], dtype=mx.uint32),
        model,
        max_tokens=max_tokens,
        max_k=max_k,
        disable_auto_k=True,
        logits_processors=processors,
        accept_counter=counter,
        **kwargs,
    )
    try:
        for tok, _lp, from_draft in stream:
            emitted.append((tok, from_draft))
            if events is not None:
                events.append(("emit", tok))
            if stop_after is not None and len(emitted) >= stop_after:
                break
    finally:
        if stop_after is not None:
            stream.close()
    return emitted


# Round 0 (K=0 bootstrap) emits 7 and drafts [11, 12]; round 1 verifies both
# against 11, 12 and so accepts each, then emits bonus 13 and drafts again.
_ALL_ACCEPT_BACKBONE = [7, 11, 12, 13, 21, 22, 23, 31, 32, 33]
_ALL_ACCEPT_MTP = [11, 12, 21, 22, 31, 32, 41, 42]


def test_the_next_round_s_drafter_runs_before_this_round_s_tokens_are_delivered():
    """The whole point of the change: draft events precede the round's emits.

    Round 1 accepts both drafts and emits three tokens (11, 12, bonus 13).
    Its successor's two draft forwards must appear in the log BEFORE 11 goes
    out, because the caller's per-token work happens inside those yields and
    that is the window the drafter is meant to run in.
    """
    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)

    emitted = _run(model, max_tokens=4 + 1, events=events)

    assert emitted[:4] == [(7, False), (11, True), (12, True), (13, False)], emitted
    order = [e for e in events if e[0] in ("draft", "emit")]
    first_verify_emit = order.index(("emit", 11))
    drafts_before = sum(1 for e in order[:first_verify_emit] if e[0] == "draft")
    # Two for round 0's chain, two for round 2's -- the latter is the overlap.
    assert drafts_before == 4, (
        "round 2's drafter must be launched before round 1's first token is "
        f"handed to the caller; event order was {order}"
    )


@pytest.mark.parametrize(
    ("max_tokens", "expected_mtp_calls"),
    [
        # Cut lands on the first accepted draft, mid-emission-list.
        (2, 2),
        # Cut lands on the second accepted draft.
        (3, 2),
        # Cut lands exactly on the bonus token, the round's last emission.
        (4, 2),
        # No cut: the round drafts for a successor that does run.
        (5, 4),
    ],
)
def test_a_round_capped_by_max_tokens_never_pays_for_a_drafter_forward(
    max_tokens, expected_mtp_calls
):
    """``round_done`` has to suppress drafting exactly where ``return`` did.

    Only round 0's two draft forwards may have happened when the cap lands
    inside round 1, at any of its three emission points.
    """
    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)

    emitted = _run(model, max_tokens=max_tokens, events=events)

    assert len(emitted) == max_tokens, emitted
    assert model.mtp_calls == expected_mtp_calls, (
        f"max_tokens={max_tokens} produced {model.mtp_calls} drafter forwards; "
        "a capped round must not draft for a successor that never runs"
    )


def test_a_partially_accepted_round_delivers_the_residual_after_drafting():
    """Reject path: same ordering guarantee, and the stream is unchanged.

    Round 1 proposes [11, 12] but the target's second position disagrees, so
    the round emits the accepted 11 and then the target's residual 29.
    """
    events: list = []
    backbone = [7, 11, 29, 13, 21, 22, 23]
    mtp = [11, 12, 21, 22, 31, 32]
    model = _ScriptedModel(backbone, mtp, events)

    emitted = _run(model, max_tokens=4, events=events)

    assert emitted[:3] == [(7, False), (11, True), (29, False)], emitted
    order = [e for e in events if e[0] in ("draft", "emit")]
    assert sum(1 for e in order[: order.index(("emit", 11))] if e[0] == "draft") == 4, (
        f"drafter must precede delivery on the reject path too: {order}"
    )


def test_every_delivered_token_is_preceded_by_its_own_processor_restore():
    """Delivery replays each position's snapshot, in emission order.

    The drafter has already run at the round's *final* state by the time the
    first token goes out, so the guard would be left ahead of the stream if
    delivery did not restore. Restores are absolute, which is what makes
    replaying them at delivery reproduce the old per-yield walk.
    """
    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)
    proc = _RecordingProcessor(events)

    emitted = _run(model, max_tokens=4, events=events, processors=[proc])

    assert [t for t, _ in emitted] == [7, 11, 12, 13], emitted
    # Every verify-path emission must be immediately preceded by a restore.
    for i, ev in enumerate(events):
        if ev == ("emit", 7):
            continue  # bootstrap round has no per-position snapshots
        if ev[0] == "emit":
            assert events[i - 1][0] == "restore", (
                f"token {ev[1]} was delivered without restoring its position's "
                f"guard state; log was {events}"
            )
    # And the three verify positions restore strictly increasing snapshots.
    verify_restores = [
        events[i - 1][1]
        for i, ev in enumerate(events)
        if ev[0] == "emit" and ev[1] != 7
    ]
    assert verify_restores == sorted(set(verify_restores)), verify_restores


def test_a_caller_that_stops_mid_round_leaves_the_guard_at_the_last_delivered_token():
    """Abandoning the stream must not leak the round's later guard state.

    A client that disconnects, or a stop string that hits on the first
    accepted draft, stops pulling at that yield. The processors have to sit at
    that token's prefix -- not at the bonus token's, and not at the state the
    drafter ran under.
    """
    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)
    proc = _RecordingProcessor(events)

    emitted = _run(model, max_tokens=8, events=events, processors=[proc], stop_after=2)

    assert [t for t, _ in emitted] == [7, 11], emitted
    state_at_first_verify_emit = events[events.index(("emit", 11)) - 1][1]
    assert proc.state == state_at_first_verify_emit, (
        "the guard advanced past the last delivered token after the caller "
        f"stopped pulling: state={proc.state}, expected "
        f"{state_at_first_verify_emit}, log={events}"
    )


def test_the_drafter_chain_is_handed_to_async_eval(monkeypatch):
    """``_launch_drafts`` is what actually puts the forward on the device.

    Without the ``mx.async_eval`` the reordering alone changes nothing: the
    chain stays lazy and still first executes under the next round's
    ``mx.eval``, back on the critical path.
    """
    import mlx.core as mlx_core

    calls: list[int] = []
    real_async_eval = mlx_core.async_eval

    def _spy(*arrays):
        calls.append(len(arrays))
        return real_async_eval(*arrays)

    monkeypatch.setattr(mlx_core, "async_eval", _spy)

    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)
    _run(model, max_tokens=5, events=events)

    assert calls, "the drafted chain was never handed to mx.async_eval"
    assert all(n > 0 for n in calls), calls


def test_a_caller_that_stops_mid_round_does_not_bank_undelivered_accepts():
    """`accept_counter` is process-global, so it must count deliveries.

    Round 1 accepts two drafts and would emit three tokens. A caller that
    takes only the first of them -- a stop string matching, a client
    disconnecting -- must leave the counter at one accept. Banking the whole
    round up front would let every abandoned request inflate the acceptance
    rate the server publishes, and the drafter now runs before delivery, so
    "record it while collecting" is exactly the tempting shortcut.
    """
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    events: list = []
    counter = MTPAcceptCounter()
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)

    emitted = _run(model, max_tokens=8, events=events, counter=counter, stop_after=2)

    assert [t for t, _ in emitted] == [7, 11], emitted
    snap = counter.snapshot()
    assert snap.accepts == 1, (
        f"one accepted draft was delivered but {snap.accepts} were counted"
    )
    assert snap.tokens_saved == 1, snap.tokens_saved


def test_the_round_s_counter_calls_ride_delivery_not_the_collection_pass():
    """Each accounting call must land with its own token, in delivery order.

    ``record_reject`` happens to leave the public counters untouched today
    (the rejection is derivable as ``attempts - accepts``), so asserting on
    snapshot fields alone would not notice it drifting away from delivery.
    Spy on the calls instead: on the reject path a caller that stops on the
    accepted draft must trigger no rejection accounting at all, and a caller
    that drains the round must trigger exactly one, after its residual's
    guard restore.
    """
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter

    class _SpyCounter(MTPAcceptCounter):
        def __init__(self, events):
            super().__init__()
            self.events = events

        def record_accept(self, tokens_saved: int = 1) -> None:
            self.events.append(("count", "accept"))
            super().record_accept(tokens_saved)

        def record_reject(self) -> None:
            self.events.append(("count", "reject"))
            super().record_reject()

    backbone = [7, 11, 29, 13, 21, 22, 23]
    mtp = [11, 12, 21, 22, 31, 32]

    stopped_events: list = []
    _run(
        _ScriptedModel(backbone, mtp, stopped_events),
        max_tokens=8,
        events=stopped_events,
        counter=_SpyCounter(stopped_events),
        stop_after=2,
    )
    assert [e for e in stopped_events if e[0] == "count"] == [("count", "accept")], (
        f"a caller that stopped on the accepted draft still counted the "
        f"round's undelivered rejection: {stopped_events}"
    )

    drained_events: list = []
    emitted = _run(
        _ScriptedModel(backbone, mtp, drained_events),
        max_tokens=3,
        events=drained_events,
        counter=_SpyCounter(drained_events),
    )
    assert [t for t, _ in emitted] == [7, 11, 29], emitted
    counted = [e for e in drained_events if e[0] == "count"]
    assert counted == [("count", "accept"), ("count", "reject")], drained_events
    # ... and each one sits with the token it belongs to.
    assert (
        drained_events.index(("count", "accept"))
        < drained_events.index(("emit", 11))
        < drained_events.index(("count", "reject"))
        < drained_events.index(("emit", 29))
    )


def test_drafting_before_the_yield_never_leaks_guard_state_to_the_caller():
    """The bootstrap round drafts before its yield; the guards must not move.

    ``_draft_chain_timed`` snapshots the transactional processors and restores
    them in a ``finally`` -- draft-side interventions shape q(d), but only
    target-verified, delivered positions may advance processor state. That
    contract is what makes it safe to run the drafter ahead of delivery
    without the per-emission rewind the verify branch needs, so it is now
    load-bearing rather than merely tidy: a caller taking only the bootstrap
    token must be left exactly where that token committed the guards.
    """
    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)
    proc = _RecordingProcessor(events)

    emitted = _run(model, max_tokens=8, events=events, processors=[proc], stop_after=1)

    assert [t for t, _ in emitted] == [7], emitted
    # The bootstrap forward applied the processor once (one position), so the
    # committed state is 1; the two draft positions that ran after it advanced
    # it to 3 and the drafter's own rewind is what brings it back.
    assert proc.state == 1, (
        f"the caller was handed guard state {proc.state} after one bootstrap "
        f"token; drafting leaked into it. Log: {events}"
    )
    assert [e[1] for e in events if e[0] == "apply"] == [0, 1, 2], events


def test_a_round_ending_on_an_accepted_eos_draft_pays_for_no_draft():
    """EOS inside the accepted run ends the round before the drafter runs.

    Positions past a stop token are never reached in real decode, so the round
    truncates at EOS: no bonus token, no residual, and -- now that drafting
    moved to the front of the delivery phase -- no drafter chain either. The
    old code reached its ``return`` before the draft; ``round_done`` has to
    stand in for that, or every terminating request would pay one wasted
    drafter forward whose output is discarded with the request.
    """
    events: list = []
    model = _ScriptedModel(_ALL_ACCEPT_BACKBONE, _ALL_ACCEPT_MTP, events)

    emitted = _run(model, max_tokens=10, events=events, stop_tokens={11})

    # 7 bootstraps, 11 is accepted and is the stop token: nothing after it.
    assert emitted == [(7, False), (11, True)], emitted
    # Two forwards for round 0's chain and none for the terminated round.
    assert model.mtp_calls == 2, events


def test_prompt_lookup_drafts_replace_the_drafter_chain_on_a_parked_round():
    """A parked round with a prompt hit drafts by copy, not by MTP forward.

    ``max_k=0`` parks the drafter, so every round takes the bootstrap path;
    once the generated suffix matches an indexed prompt n-gram the proposal
    comes from the prompt itself. This is the one draft-setup branch that must
    NOT reach ``_draft_chain_timed`` -- there is no chain to launch and no
    draft wall time to charge the next round -- so ``mtp_forward`` staying at
    zero is the assertion that matters.
    """
    from vllm_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    events: list = []
    # Round 0 emits 7, round 1 emits 20; the suffix (7, 20) then matches the
    # history below, whose continuation is (25, 26) -- and the backbone goes
    # on to confirm both, so they are delivered as accepted drafts. Every id
    # stays under ``_ScriptedModel``'s 32-wide vocab, or the one-hot logit
    # lands out of range and greedy silently falls back to 0.
    model = _ScriptedModel([7, 20, 25, 26, 27, 27, 27], [], events)
    model.mtp_prompt_lookup_supported = True

    emitted = _run(
        model,
        max_tokens=4,
        max_k=0,
        events=events,
        prompt_lookup_enabled=True,
        prompt_lookup_history=[7, 20, 25, 26],
        prompt_lookup_policy=PromptLookupPolicy(
            enabled_by_default=True, min_ngram=2, max_ngram=2, max_tokens=4
        ),
    )

    assert [tok for tok, _ in emitted][:4] == [7, 20, 25, 26], emitted
    assert [from_draft for _, from_draft in emitted][:4] == [
        False,
        False,
        True,
        True,
    ], emitted
    assert model.mtp_calls == 0, events
