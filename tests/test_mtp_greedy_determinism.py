# SPDX-License-Identifier: Apache-2.0
"""Greedy MTP output must be a function of the request, not of the run.

A real target's verify forward does not reproduce single-step logits bit for
bit: a block of ``S`` rows runs different matmul / attention / recurrent
kernels than a one-row step, and in bf16 the top two logits are often equal
or one ulp apart. So which width verified a position can flip a greedy
argmax. On ``origin/main`` before this test, the widths came from two
adaptive pieces that read the wall clock and process-wide state -- the depth
controller (cost EWMA, trained by every earlier request) and ``CopyDraftGate``
(tokens per millisecond) -- so the same greedy request returned different
text from run to run, on a fresh server and on a warm one (measured on
Qwen3.5-4B-4bit: 2-8 distinct outputs per prose prompt over 10 runs).

The doubles below model exactly that: a target whose argmax at one "tie"
token depends on how many rows the forward verified, and a clock the test
controls. Greedy output has to come out the same whatever the controller has
learned and however long the rounds took. Only the host itself may move the
depth -- once per process (``reproducible_depth``), never per round.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.requires_mlx

from tests.test_mtp_spec_decode import _CountingKVCache  # noqa: E402

VOCAB = 256
TIE = 130
# The tie token's two candidates. Which one wins depends on the verify width,
# like a bf16 near-tie whose last bit depends on the kernel that produced it.
NARROW_WINNER = 200
WIDE_WINNER = 220
# Long enough that the rollback guard admits a full-depth block from the
# first drafting round on.
PROMPT = list(range(5, 15))


@pytest.fixture(autouse=True)
def _reset_mtp_module_state():
    """Same isolation as ``test_mtp_lossless.py``: module singletons and the
    mlx-lm generation stream are process-global."""
    from rapid_mlx.spec_decode.mtp.accept_counter import (
        reset_global_counter_for_tests,
    )
    from rapid_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests
    from rapid_mlx.spec_decode.mtp.draft_k_controller_v2 import reset_controllers
    from rapid_mlx.spec_decode.mtp.reproducible_depth import reset_for_tests

    def _reset():
        _unpatch_for_tests()
        reset_global_counter_for_tests()
        reset_controllers()
        reset_for_tests()
        import mlx_lm.generate  # noqa: F401 -- ensure the module is loaded

        sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
            mx.default_device()
        )

    _reset()
    yield
    _reset()


class _VirtualClock:
    """``time.perf_counter`` stand-in the doubles advance per forward."""

    def __init__(self, target_cost, draft_cost: float = 0.1):
        self.now = 0.0
        self.target_cost = target_cost
        self.draft_cost = draft_cost

    def perf_counter(self) -> float:
        return self.now


class _WidthSensitiveTarget:
    """Counting language model (``t -> t + 1``) with one width-dependent tie.

    Every row predicts its input token plus one, except the row whose input
    is ``TIE``: there the winner depends on how many rows the forward
    verified, as decided by ``tie_winner(rows)``. The MTP head always drafts
    ``t + 1``, so it is right everywhere except at the tie.
    """

    mtp_prompt_lookup_supported = True

    def __init__(self, tie_winner, clock: _VirtualClock | None = None):
        self.tie_winner = tie_winner
        self.clock = clock
        self.layers = [object()]
        self.widths: list[int] = []
        self.confirmed: list[tuple[int, int]] = []

    def _rows(self, inputs: mx.array, winner) -> mx.array:
        rows = []
        for token in inputs.reshape(-1).tolist():
            target = winner if token == TIE else (token + 1) % VOCAB
            rows.append(mx.where(mx.arange(VOCAB) == target, 50.0, 0.0))
        return mx.stack(rows)[None, :, :]

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        rows = int(inputs.shape[1])
        self.confirmed.append((rows, n_confirmed))
        if cache and isinstance(cache[0], _CountingKVCache):
            # The request's own forwards; the once-per-process depth timing
            # runs on scratch caches and is not part of the schedule.
            self.widths.append(rows)
        if cache is not None:
            for entry in cache:
                entry.offset += rows
        if self.clock is not None:
            self.clock.now += self.clock.target_cost(rows)
        logits = self._rows(inputs, self.tie_winner(rows))
        hidden = mx.zeros((1, rows, 8))
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        for entry in mtp_cache:
            entry.offset += int(next_token_ids.shape[1])
        if self.clock is not None:
            self.clock.now += self.clock.draft_cost
        rows = [
            mx.where(mx.arange(VOCAB) == (token + 1) % VOCAB, 50.0, 0.0)
            for token in next_token_ids.reshape(-1).tolist()
        ]
        return mx.stack(rows)[None, :, :]

    def make_mtp_cache(self):
        return []


def _install_clock(monkeypatch, clock: _VirtualClock) -> None:
    import rapid_mlx.spec_decode.mtp.generator as generator_mod
    import rapid_mlx.spec_decode.mtp.reproducible_depth as depth_mod

    for module in (generator_mod, depth_mod):
        monkeypatch.setattr(
            module, "time", SimpleNamespace(perf_counter=clock.perf_counter)
        )


def _host_schedule(model, max_k: int = 2):
    """What the scheduler passes as ``greedy_schedule`` on ``model``'s host."""
    from rapid_mlx.spec_decode.mtp.reproducible_depth import greedy_schedule

    return greedy_schedule(model, "determinism-test", max_k)


def _flat_host() -> _VirtualClock:
    """Every forward costs the same: drafting at ``max_k`` clearly pays."""
    return _VirtualClock(lambda rows: 1.0)


def _generate(model, prompt: list[int], max_tokens: int, **kwargs) -> list[int]:
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    kwargs.setdefault("prompt_lookup_enabled", False)
    return [
        token
        for token, _lp, _drafted in mtp_generate_step(
            mx.array(prompt, dtype=mx.uint32),
            model,
            max_tokens=max_tokens,
            prompt_cache=[_CountingKVCache(), _CountingKVCache()],
            accept_counter=MTPAcceptCounter(),
            model_id="determinism-test",
            **kwargs,
        )
    ]


def test_greedy_output_does_not_depend_on_what_the_depth_controller_learned(
    monkeypatch,
):
    """Same request, two controller states: parked (K=0) and drafting (K=2).

    A process-wide controller trained by earlier requests -- or by a fresh
    server's first noisy timings -- can start the same request at either
    depth. Before the fix that decided whether the tie was verified in a
    one-row step or a three-row block, and so which token came out.
    """
    from rapid_mlx.spec_decode.mtp import draft_k_controller_v2

    def tie_winner(rows: int) -> int:
        return NARROW_WINNER if rows == 1 else WIDE_WINNER

    picks: list[int] = []
    outputs = []
    clock = _flat_host()
    _install_clock(monkeypatch, clock)
    for learned_depth in (0, 2):
        monkeypatch.setattr(
            draft_k_controller_v2.DepthController,
            "pick_k",
            lambda self, depth=learned_depth: picks.append(depth) or depth,
        )
        outputs.append(
            _generate(
                _WidthSensitiveTarget(tie_winner, clock),
                [TIE - 12],
                max_tokens=24,
                max_k=2,
                temp=0.0,
            )
        )

    assert TIE in outputs[0], "the request never reached the tie it tests"
    assert outputs[0] == outputs[1], (
        "greedy output changed with the depth controller's state: "
        f"{outputs[0]} vs {outputs[1]}"
    )
    # Not merely equal by luck: a greedy request does not ask the controller.
    assert picks == []


def test_sampled_requests_still_adapt_depth(monkeypatch):
    """The pin is for reproducible requests only; sampling keeps auto-K."""
    from rapid_mlx.spec_decode.mtp import draft_k_controller_v2

    picks: list[int] = []
    monkeypatch.setattr(
        draft_k_controller_v2.DepthController,
        "pick_k",
        lambda self: picks.append(1) or 1,
    )
    _generate(
        _WidthSensitiveTarget(lambda rows: WIDE_WINNER),
        [5],
        max_tokens=8,
        max_k=2,
        temp=0.7,
    )
    assert picks, "a sampled request without a seed no longer consults auto-K"


def test_greedy_copy_draft_schedule_does_not_depend_on_the_clock(monkeypatch):
    """Same request on a host where wide verifies are cheap, and on one where
    they are dear: same copy-draft decisions, same verify widths, same text.

    ``CopyDraftGate`` compares tokens per millisecond of copy-draft rounds
    against the speculative rounds they displace. On the dear host it starts
    declining copies, so the tie lands in a three-row MTP block instead of a
    wide copy block -- and before the fix the output followed the clock.
    """
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    def tie_winner(rows: int) -> int:
        return NARROW_WINNER if rows <= 3 else WIDE_WINNER

    # Generation starts at 90 and counts up. 90-99 are not in the prompt, so
    # the first rounds are ordinary MTP rounds (the gate's baseline); from 100
    # on, every bigram is a prompt bigram and copy-drafts take over.
    prompt = list(range(100, 160)) + [89]
    policy = PromptLookupPolicy(
        enabled_by_default=True, min_ngram=2, max_ngram=2, max_tokens=31
    )
    # Both hosts price a step and a three-row MTP verify alike, so the
    # once-per-process depth agrees; they differ only on the wide blocks the
    # copy gate times.
    hosts = {
        "wide verifies cheap": lambda rows: 1.0,
        "wide verifies dear": lambda rows: 1.0 if rows <= 3 else 50.0 * rows,
    }
    runs = {}
    for host, cost in hosts.items():
        clock = _VirtualClock(cost)
        _install_clock(monkeypatch, clock)
        model = _WidthSensitiveTarget(tie_winner, clock)
        tokens = _generate(
            model,
            prompt,
            max_tokens=48,
            max_k=2,
            temp=0.0,
            prompt_lookup_enabled=True,
            prompt_lookup_history=prompt,
            prompt_lookup_policy=policy,
        )
        runs[host] = (tokens, model.widths)

    (cheap_tokens, cheap_widths), (dear_tokens, dear_widths) = runs.values()
    assert TIE in cheap_tokens, "the request never reached the tie it tests"
    assert max(cheap_widths[1:]) > 3, "no copy-draft block was ever verified"
    assert cheap_widths == dear_widths, (
        f"greedy verify schedule followed the clock: {cheap_widths} vs {dear_widths}"
    )
    assert cheap_tokens == dear_tokens


def test_verify_cost_estimate_follows_the_measured_verify_curve():
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        COPY_DRAFT_TILE_ROWS,
        verify_cost_estimate,
    )

    assert verify_cost_estimate(1) == 1.0
    # Grows through the vector path, flat inside the GEMM tile...
    assert verify_cost_estimate(2) < verify_cost_estimate(3) < verify_cost_estimate(13)
    assert (
        verify_cost_estimate(13)
        < verify_cost_estimate(32)
        < 1.05 * (verify_cost_estimate(13))
    )
    # ...and a whole tile for the first row past its edge.
    assert verify_cost_estimate(COPY_DRAFT_TILE_ROWS + 1) == pytest.approx(
        2 * verify_cost_estimate(COPY_DRAFT_TILE_ROWS)
    )
    with pytest.raises(ValueError):
        verify_cost_estimate(0)
    # A steep host pays most of a step per row from the second row on.
    assert verify_cost_estimate(3, steep=True) == pytest.approx(2.38)
    assert verify_cost_estimate(3, steep=True) > verify_cost_estimate(3)
    assert verify_cost_estimate(64, steep=True) == pytest.approx(
        2 * verify_cost_estimate(32, steep=True)
    )


def test_greedy_requests_park_on_a_host_where_drafting_cannot_pay(monkeypatch):
    """The host still bounds the depth -- once, not per round.

    Where every verified row and every draft costs a whole step, even a fully
    accepted round of any depth cannot beat plain steps, so greedy requests
    run plain decode there -- every time.
    """
    clock = _VirtualClock(lambda rows: float(rows), draft_cost=1.0)
    _install_clock(monkeypatch, clock)
    runs = []
    for _ in range(2):
        model = _WidthSensitiveTarget(lambda rows: WIDE_WINNER, clock)
        schedule = _host_schedule(model)
        tokens = _generate(
            model, PROMPT, max_tokens=12, max_k=2, temp=0.0, greedy_schedule=schedule
        )
        runs.append((tokens, model.widths))
    assert runs[0] == runs[1]
    assert schedule.depth == 0
    # Prefill, then one-row steps only.
    assert set(runs[0][1][1:]) == {1}, runs[0][1]

    # The same request on a host where verifying is nearly free, with drafts
    # that always land, settles on full-depth blocks.
    from rapid_mlx.spec_decode.mtp.reproducible_depth import reset_for_tests

    reset_for_tests()
    clock = _flat_host()
    _install_clock(monkeypatch, clock)
    model = _WidthSensitiveTarget(lambda rows: WIDE_WINNER, clock)
    schedule = _host_schedule(model)
    assert schedule.depth == 2 and schedule.steep_verify is False
    _generate(
        model, PROMPT, max_tokens=200, max_k=2, temp=0.0, greedy_schedule=schedule
    )
    # Once its own acceptance is trusted (the controller's ramp-up).
    settled = model.widths[len(model.widths) // 2 :]
    assert settled.count(3) > len(settled) // 2, model.widths


class _WrongDrafter(_WidthSensitiveTarget):
    """Same target; a drafter that is never right (prose-like acceptance 0)."""

    def mtp_forward(self, hidden, next_token_ids, mtp_cache):
        logits = super().mtp_forward(hidden, next_token_ids, mtp_cache)
        return mx.roll(logits, 7, axis=-1)


@pytest.mark.parametrize("drafter", [_WidthSensitiveTarget, _WrongDrafter])
def test_steep_host_drafts_only_where_this_request_drafts_well(monkeypatch, drafter):
    """Where every verified row costs most of a step, the depth follows this
    request's own acceptance on a fixed cost curve -- the same every run."""
    # Each verified row costs a whole step: steep.
    clock = _VirtualClock(lambda rows: float(rows), draft_cost=0.15)
    _install_clock(monkeypatch, clock)
    runs = []
    for _ in range(2):
        model = drafter(lambda rows: WIDE_WINNER, clock)
        schedule = _host_schedule(model)
        tokens = _generate(
            model, PROMPT, max_tokens=200, max_k=2, temp=0.0, greedy_schedule=schedule
        )
        runs.append((tokens, model.widths))
    assert schedule.steep_verify is True
    assert runs[0] == runs[1], "the per-round depth must not vary between runs"
    # Once its own acceptance is trusted (the controller's ramp-up).
    settled = runs[0][1][len(runs[0][1]) // 2 :]
    if drafter is _WidthSensitiveTarget:
        # Every draft lands: drafting rounds dominate.
        assert sum(w > 1 for w in settled) > len(settled) * 3 // 4, runs[0][1]
    else:
        # Nothing lands: one-row steps dominate.
        assert settled.count(1) > len(settled) * 3 // 4, runs[0][1]


def test_requests_decide_on_the_class_curve_not_the_timings():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import (
        GreedySchedule,
        class_round_costs,
        reproducible_schedule,
        request_round_cost,
        round_costs,
    )

    # Two boots of an M4 Pro-like host whose timings differ by a few percent
    # read the same class, so they decide on the same curve.
    boots = [(21.8, (26.0, 30.4), 3.0), (23.1, (27.9, 31.0), 3.3)]
    schedules = [
        reproducible_schedule(f"boot-{i}", 2, lambda k, t=t: t)
        for i, t in enumerate(boots)
    ]
    assert schedules[0] == schedules[1]
    assert schedules[0] == GreedySchedule(
        depth=2, steep_verify=False, round_costs=class_round_costs(False, 2)
    )
    # The class curves: dearer per depth on a steep host.
    cheap, steep = class_round_costs(False, 2), class_round_costs(True, 2)
    assert cheap[0] == steep[0] == 1.0
    assert cheap[1] < steep[1] and cheap[2] < steep[2]
    # The default curve is unchanged; on the steep curve a parked round is
    # pipelined like plain decode, so only drafting rounds carry the
    # generator's host overhead.
    assert cheap == (1.0, 1.272, 1.544)
    assert steep == (1.0, 1.87, 2.49)
    # Without pipelined parks (logits processors) the steep curve charges the
    # overhead to every round, as before; the default curve is the same.
    assert class_round_costs(True, 2, parks_pipelined=False) == (1.0, 1.496, 1.992)
    assert class_round_costs(False, 2, parks_pipelined=False) == cheap
    # Depths past the curve read its deepest point.
    assert request_round_cost(cheap, 5) == cheap[-1]
    # Measured costs (for the park verdict) include drafts and overhead.
    assert round_costs(10.0, (12.0,), 1.0) == (1.0, (1.2 + 0.1 + 0.25) / 1.25)
    with pytest.raises(ValueError):
        round_costs(0.0, (26.0,), 3.0)
    with pytest.raises(ValueError):
        round_costs(21.8, (), 3.0)


def test_operator_fixed_depth_is_not_second_guessed(monkeypatch):
    """``disable_auto_k`` keeps the configured depth even on a dear host."""
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    model = _WidthSensitiveTarget(lambda rows: WIDE_WINNER)
    _generate(
        model,
        PROMPT,
        max_tokens=12,
        max_k=2,
        temp=0.0,
        disable_auto_k=True,
        greedy_schedule=GreedySchedule(depth=0),
    )
    # Prefill, then the bootstrap step, then max_k blocks.
    assert set(model.widths[2:]) == {3}, model.widths


def test_depth_that_pays():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import depth_that_pays

    # A fully accepted K=2 round commits 3 tokens for 2.34 plain rounds.
    assert depth_that_pays((1.0, 1.7, 2.34), 2) == 2
    # Nothing commits more than it costs: park.
    assert depth_that_pays((1.0, 2.2, 3.4), 2) == 0
    assert depth_that_pays((1.0, 1.7, 2.34), 0) == 0
    assert depth_that_pays((1.0,), 2) == 0


def test_verify_is_steep():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import verify_is_steep

    # M2 Pro-like: three rows at 2.4 steps -> 0.7 of a step per extra row.
    assert verify_is_steep(17.5, (29.5, 41.5), 2) is True
    # M4 Pro-like: three rows at 1.4 steps -> 0.2 per extra row.
    # (M2 Pro timed round-robin: 1.93 steps -> 0.46, still steep.)
    assert verify_is_steep(24.0, (34.0, 46.4), 2) is True
    assert verify_is_steep(21.8, (26.0, 30.4), 2) is False
    assert verify_is_steep(17.5, (29.5, 41.5), 0) is False
    assert verify_is_steep(17.5, (), 2) is False


def test_reproducible_schedule_is_decided_once_and_survives_a_failed_measurement():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import (
        GreedySchedule,
        class_round_costs,
        reproducible_schedule,
    )

    calls: list[int] = []

    def dear(max_k: int):
        calls.append(max_k)
        return 10.0, (20.0, 30.0), 5.0

    expected = GreedySchedule(
        depth=0, steep_verify=True, round_costs=class_round_costs(True, 2)
    )
    assert reproducible_schedule("host-a", 2, dear) == expected
    assert reproducible_schedule("host-a", 2, dear) == expected
    assert calls == [2], "the schedule must be measured once, then reused"
    assert reproducible_schedule("no-drafts", 0, dear) == GreedySchedule(depth=0)
    assert calls == [2]

    def broken(max_k: int):
        raise RuntimeError("no scratch cache for this model")

    # A model the timing cannot run on keeps the configured depth.
    assert reproducible_schedule("host-b", 3, broken) == GreedySchedule(depth=3)


@pytest.mark.parametrize(
    ("temperature", "expected"), [(0.0, "host schedule"), (0.7, None)]
)
def test_scheduler_hands_greedy_requests_the_host_depth(
    monkeypatch, temperature, expected
):
    """The served path decides a greedy request's depth once per process
    (``reproducible_depth.greedy_schedule``) and leaves sampled ones to auto-K."""
    from types import SimpleNamespace

    from rapid_mlx.scheduler import _install_mtp_vendored
    from rapid_mlx.spec_decode.mtp import generator as generator_mod
    from rapid_mlx.spec_decode.mtp import reproducible_depth as depth_mod
    from tests.test_mtp_cli_wiring import _make_batch_gen_with_gb, _StubModel

    seen: dict[str, object] = {}
    asked: list[tuple[object, str, int]] = []

    class _FakeGen:
        def __iter__(self):
            return self

        def __next__(self):
            return (201, mx.array([0.0]), False)

        def close(self):
            pass

    def _recording_generator(*args, **kwargs):
        seen.update(kwargs)
        return _FakeGen()

    def _host_schedule_for(model, key, max_k):
        asked.append((model, key, max_k))
        return "host schedule"

    monkeypatch.setattr(generator_mod, "mtp_generate_step", _recording_generator)
    monkeypatch.setattr(depth_mod, "greedy_schedule", _host_schedule_for)

    batch_gen, gb = _make_batch_gen_with_gb()
    gb.uids = [41]
    gb._next_tokens = mx.array([500], dtype=mx.uint32)
    gb._next_logprobs = [mx.array([0.0])]
    model = _StubModel()
    request = SimpleNamespace(sampling_params=SimpleNamespace(temperature=temperature))
    assert _install_mtp_vendored(
        batch_gen,
        model=model,
        requests={"req-41": request},
        uid_to_request_id={41: "req-41"},
        max_k=2,
    )
    gb._step()

    assert seen["greedy_schedule"] == expected
    if expected is None:
        assert asked == []
    else:
        assert [(m, k) for m, _key, k in asked] == [(model, 2)]
        assert asked[0][1] == seen["model_id"]


@pytest.mark.parametrize("fused_declines", [False, True])
def test_host_timing_uses_the_draft_a_greedy_round_runs(monkeypatch, fused_declines):
    """The fused greedy head is what greedy rounds draft with when a family
    has one; it may decline a call, and then the full head is timed."""
    from rapid_mlx.spec_decode.mtp.reproducible_depth import (
        TIMING_ROUNDS,
        TIMING_WARMUP_ROUNDS,
        time_target_forwards,
    )

    clock = _VirtualClock(lambda rows: 1.0, draft_cost=0.5)
    _install_clock(monkeypatch, clock)
    fused_calls: list[int] = []

    class _FusedHead(_WidthSensitiveTarget):
        def mtp_greedy(self, hidden, next_token_ids, mtp_cache):
            fused_calls.append(1)
            if fused_declines:
                return None
            clock.now += 0.25
            return next_token_ids + 1, hidden

    model = _FusedHead(lambda rows: WIDE_WINNER, clock)
    step_ms, verify_ms, draft_ms = time_target_forwards(model, 2)
    # Each width is timed the way the generator verifies it.
    assert set(model.confirmed) == {(1, 0), (2, 1), (3, 2)}
    assert (step_ms, verify_ms) == (1000.0, (1000.0, 1000.0))
    assert len(fused_calls) == TIMING_WARMUP_ROUNDS + TIMING_ROUNDS
    # Timed head: the full one (0.5 s) when the fused one declines.
    assert draft_ms == (500.0 if fused_declines else 250.0)


def test_steep_host_prices_copy_drafts_on_the_steep_curve(monkeypatch):
    """The copy gate's clock-free cost follows the host's measured class."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import (
        CopyDraftGate,
        PromptLookupPolicy,
        verify_cost_estimate,
    )
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    charged: list[tuple[bool, float]] = []
    real_observe = CopyDraftGate.observe

    def recording_observe(self, *, is_copy_draft, committed, round_ms, **kw):
        charged.append((is_copy_draft, round_ms))
        return real_observe(
            self,
            is_copy_draft=is_copy_draft,
            committed=committed,
            round_ms=round_ms,
            **kw,
        )

    monkeypatch.setattr(CopyDraftGate, "observe", recording_observe)
    prompt = list(range(100, 160)) + [89]
    model = _WidthSensitiveTarget(lambda rows: WIDE_WINNER)
    _generate(
        model,
        prompt,
        max_tokens=40,
        max_k=2,
        temp=0.0,
        prompt_lookup_enabled=True,
        prompt_lookup_history=prompt,
        prompt_lookup_policy=PromptLookupPolicy(
            enabled_by_default=True, min_ngram=2, max_ngram=2, max_tokens=31
        ),
        greedy_schedule=GreedySchedule(depth=2, steep_verify=True),
    )
    steep_costs = {verify_cost_estimate(r, steep=True) for r in range(1, 33)}
    assert any(is_copy for is_copy, _ in charged), "no copy-draft was verified"
    assert all(cost in steep_costs for _, cost in charged), charged


def test_one_models_timing_does_not_block_another_model():
    """Measuring holds a per-key lock: a second model's first greedy request
    is not queued behind the first model's timing forwards."""
    import threading

    from rapid_mlx.spec_decode.mtp.reproducible_depth import (
        reproducible_schedule,
    )

    entered = threading.Event()
    release = threading.Event()

    def slow(max_k: int):
        entered.set()
        assert release.wait(5.0), "never released"
        return 10.0, (11.0, 12.0), 1.0

    worker = threading.Thread(target=reproducible_schedule, args=("slow", 2, slow))
    worker.start()
    try:
        assert entered.wait(5.0)
        # Completes while "slow" is still measuring.
        fast = reproducible_schedule("fast", 2, lambda k: (10.0, (11.0, 12.0), 1.0))
        assert (fast.depth, fast.steep_verify) == (2, False)
    finally:
        release.set()
        worker.join(5.0)
    assert reproducible_schedule("slow", 2, slow) == fast


@pytest.mark.parametrize(
    ("acceptance", "pipelined", "drafts"),
    [
        (0.95, True, True),  # nearly every draft lands: drafting pays
        (0.5, True, False),  # prose-like: plain steps are faster
        (0.8, True, False),  # 2.44 tokens for 2.49 steps at K=2 does not pay
        (0.8, False, True),  # ...but does while parked rounds pay overhead
    ],
)
def test_steep_request_curve_drafts_only_where_it_pays(acceptance, pipelined, drafts):
    """The steep curve's EV decision on a request's own acceptance."""
    from rapid_mlx.spec_decode.mtp.reproducible_depth import (
        GreedySchedule,
        class_round_costs,
        request_depth_controller,
        request_round_cost,
        request_round_costs,
    )

    schedule = GreedySchedule(
        depth=2, steep_verify=True, round_costs=class_round_costs(True, 2)
    )
    costs = request_round_costs(schedule, parks_pipelined=pipelined)
    controller = request_depth_controller(costs, 2)
    # Feed sustained acceptance at depth 2 (the deterministic pattern stands
    # in for a long run at that rate).
    pattern = [True] * round(acceptance * 20) + [False] * (20 - round(acceptance * 20))
    for round_index in range(400):
        first = pattern[round_index % 20]
        second = pattern[(round_index * 7 + 3) % 20]
        accepts = [first, second] if first else [False]
        controller.record(2, request_round_cost(costs, 2), accepts)
    picks = [controller.pick_k() for _ in range(64)]
    for depth in picks:
        controller.record(depth, request_round_cost(costs, depth), [])
    drafted = sum(1 for depth in picks if depth > 0) > len(picks) // 2
    assert drafted is drafts, (acceptance, pipelined, picks)


def test_request_curve_follows_whether_parked_rounds_pipeline():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import (
        GreedySchedule,
        class_round_costs,
        request_round_costs,
    )

    steep = GreedySchedule(
        depth=2, steep_verify=True, round_costs=class_round_costs(True, 2)
    )
    cheap = GreedySchedule(
        depth=2, steep_verify=False, round_costs=class_round_costs(False, 2)
    )
    assert request_round_costs(steep, parks_pipelined=True) == steep.round_costs
    assert request_round_costs(steep, parks_pipelined=False) == (1.0, 1.496, 1.992)
    assert request_round_costs(cheap, parks_pipelined=False) == cheap.round_costs
    assert request_round_costs(GreedySchedule(depth=2), parks_pipelined=False) == ()
