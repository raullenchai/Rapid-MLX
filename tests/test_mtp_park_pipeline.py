# SPDX-License-Identifier: Apache-2.0
"""A parked MTP round must cost what a plain decode step costs.

When the depth controller parks (depth 0: no drafts), each MTP round is a
plain one-row target step. Plain decode (mlx-lm's ``GenerationBatch._step``)
launches step n+1 before it reads token n back, so graph construction and the
caller's per-token work overlap the GPU. The parked MTP round used to build
its step only when the caller asked for the next token, which put all of that
host time on the critical path: on an M2 Pro with Qwen3.5-4B-4bit, parked MTP
decoded ~46 tok/s against ~62 for ``--no-spec-decode`` with identical tokens.

These doubles count target forwards at each token handed to the caller: a
parked round must already have launched the next round's step, and handing
over to a prompt copy or an MTP chain after a launched step must stay
lossless.
"""

from __future__ import annotations

import sys

import pytest

mx = pytest.importorskip("mlx.core")

pytestmark = pytest.mark.requires_mlx

from tests.test_mtp_greedy_determinism import (  # noqa: E402
    PROMPT,
    TIE,
    WIDE_WINNER,
    _WidthSensitiveTarget,
)
from tests.test_mtp_spec_decode import _CountingKVCache  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mtp_module_state():
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


def _parked():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    return GreedySchedule(depth=0, steep_verify=True)


def _run(model, prompt, max_tokens, **kwargs):
    """Tokens, plus how many target forwards were ahead of each token."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    kwargs.setdefault("prompt_lookup_enabled", False)
    kwargs.setdefault("temp", 0.0)
    kwargs.setdefault("max_k", 2)
    tokens: list[int] = []
    ahead: list[int] = []
    for token, _lp, _drafted in mtp_generate_step(
        mx.array(prompt, dtype=mx.uint32),
        model,
        max_tokens=max_tokens,
        prompt_cache=[_CountingKVCache(), _CountingKVCache()],
        accept_counter=MTPAcceptCounter(),
        model_id="park-pipeline-test",
        **kwargs,
    ):
        tokens.append(int(token))
        # widths[0] is the prompt prefill; every other forward produces (or
        # verifies) at least one token.
        ahead.append(sum(1 for rows in model.widths[1:] if rows == 1) - len(tokens))
    return tokens, ahead


def _no_tie(rows: int) -> int:
    """The doubles' tie token counts on like every other token."""
    return TIE + 1


def _counting(start: int, n: int) -> list[int]:
    return [(start + i) % 256 for i in range(n)]


def test_parked_rounds_launch_the_next_step_before_handing_over_a_token():
    """Like plain decode: while parked, step n+1 is in flight when token n
    reaches the caller, and no step is launched past ``max_tokens``."""
    model = _WidthSensitiveTarget(_no_tie)
    tokens, ahead = _run(model, PROMPT, 12, greedy_schedule=_parked())

    assert tokens == _counting(PROMPT[-1] + 1, 12)
    # The last round has no next round to launch.
    assert ahead == [1] * 11 + [0], ahead
    assert model.widths[1:] == [1] * 12, model.widths


def test_parked_rounds_without_a_schedule_pipeline_too(monkeypatch):
    """A sampled request whose controller parks pipelines the same way."""
    from rapid_mlx.spec_decode.mtp import generator as generator_mod

    class _AlwaysPark:
        def __init__(self, *args, **kwargs):
            self.cost = self

        def pick_k(self):
            return 0

        def record(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        generator_mod,
        "get_or_create_controller",
        lambda *args, **kwargs: _AlwaysPark(),
    )
    model = _WidthSensitiveTarget(_no_tie)
    tokens, ahead = _run(model, PROMPT, 8, temp=0.5)

    assert len(tokens) == 8
    assert ahead[1:-1] == [1] * 6, ahead


def test_logits_processors_keep_parked_rounds_unpipelined():
    """A processor's state must advance with delivered tokens, not one
    launched step ahead of them."""
    model = _WidthSensitiveTarget(_no_tie)
    tokens, ahead = _run(
        model,
        PROMPT,
        8,
        greedy_schedule=_parked(),
        logits_processors=[lambda history, logits: logits],
        initial_tokens=PROMPT,
    )

    assert tokens == _counting(PROMPT[-1] + 1, 8)
    assert ahead == [0] * 8, ahead


def test_parked_prose_that_starts_quoting_the_prompt_copies_without_delay():
    """Parked prose that starts quoting the prompt: no step is bet on where a
    copy can follow, so the copy is verified at the first token it matches,
    and every token is still the target's own."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    # Generation counts up from 90; 90-99 are not in the prompt (parked
    # rounds), and from 100 on every bigram is a prompt bigram.
    prompt = list(range(100, 160)) + [89]
    policy = PromptLookupPolicy(
        enabled_by_default=True, min_ngram=2, max_ngram=2, max_tokens=31
    )
    model = _WidthSensitiveTarget(_no_tie)
    tokens, _ahead = _run(
        model,
        prompt,
        48,
        greedy_schedule=_parked(),
        prompt_lookup_enabled=True,
        prompt_lookup_history=prompt,
        prompt_lookup_policy=policy,
    )

    assert tokens == _counting(90, 48)
    assert max(model.widths[1:]) > 1, "no copy-draft block was ever verified"
    # Exactly one forward per round: a launched step is never thrown away.
    rows = sum(model.widths[1:])
    rounds = len(model.widths) - 1
    assert len(tokens) <= rows and rounds < len(tokens)


def test_drafting_resumes_after_a_launched_step(monkeypatch):
    """The controller leaves depth 0 while a parked step is in flight: that
    step becomes the next round and the chain is drafted from its state, so
    drafts are verified against committed positions and accepted as before.
    """
    from rapid_mlx.spec_decode.mtp import generator as generator_mod
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    script = iter(([0] * 4 + [2] * 3) * 10)
    picks: list[int] = []

    class _Scripted:
        def pick_k(self):
            depth = next(script)
            picks.append(depth)
            return depth

        def record(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        generator_mod, "request_depth_controller", lambda *args: _Scripted()
    )
    model = _WidthSensitiveTarget(_no_tie)
    tokens, ahead = _run(
        model,
        PROMPT,
        40,
        greedy_schedule=GreedySchedule(
            depth=2, steep_verify=False, round_costs=(1.0, 1.2, 1.4)
        ),
    )

    assert tokens == _counting(PROMPT[-1] + 1, 40)
    assert 3 in model.widths[1:], model.widths
    assert 1 in ahead, "parked rounds never pipelined"
    # Every step the parked rounds launched was used: no forward beyond the
    # rows the delivered tokens needed.
    assert sum(model.widths[1:]) <= len(tokens) + 2 * model.widths.count(3)


def test_greedy_output_is_reproducible_with_pipelined_parks(monkeypatch):
    """Two runs of a request that parks, copies and drafts: same text, same
    verify widths."""
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    prompt = list(range(100, 130)) + list(range(200, 210)) + [89]
    policy = PromptLookupPolicy(
        enabled_by_default=True, min_ngram=2, max_ngram=2, max_tokens=31
    )
    runs = []
    for _ in range(2):
        model = _WidthSensitiveTarget(lambda rows: 77 if rows > 1 else WIDE_WINNER)
        tokens, _ = _run(
            model,
            prompt,
            60,
            greedy_schedule=GreedySchedule(
                depth=2, steep_verify=True, round_costs=(1.0, 1.4, 1.9)
            ),
            prompt_lookup_enabled=True,
            prompt_lookup_history=prompt,
            prompt_lookup_policy=policy,
        )
        runs.append((tokens, list(model.widths)))
    assert runs[0] == runs[1]


def test_a_park_after_a_verify_round_is_launched_before_its_tokens_go_out(
    monkeypatch,
):
    """Leaving a drafting round for a park: the parked step starts before the
    verified tokens reach the caller."""
    from rapid_mlx.spec_decode.mtp import generator as generator_mod
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    # The first pick is the pre-loop bootstrap call.
    script = iter([0, 2] + [0] * 50)

    class _Scripted:
        def pick_k(self):
            return next(script)

        def record(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        generator_mod, "request_depth_controller", lambda *args: _Scripted()
    )
    model = _WidthSensitiveTarget(_no_tie)
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    tokens: list[int] = []
    forwards_at_yield: list[list[int]] = []
    for token, _lp, _drafted in mtp_generate_step(
        mx.array(PROMPT, dtype=mx.uint32),
        model,
        max_tokens=10,
        prompt_cache=[_CountingKVCache(), _CountingKVCache()],
        accept_counter=MTPAcceptCounter(),
        model_id="park-pipeline-test",
        temp=0.0,
        max_k=2,
        prompt_lookup_enabled=False,
        greedy_schedule=GreedySchedule(
            depth=2, steep_verify=False, round_costs=(1.0, 1.2, 1.4)
        ),
    ):
        tokens.append(int(token))
        forwards_at_yield.append(list(model.widths[1:]))

    assert tokens == _counting(PROMPT[-1] + 1, 10)
    # Bootstrap step, one three-row verify, then one-row steps only.
    assert model.widths[1:3] == [1, 3], model.widths
    assert set(model.widths[3:]) == {1}, model.widths
    # The verify round's three tokens go out with the next step in flight.
    for delivered in forwards_at_yield[1:4]:
        assert delivered[:3] == [1, 3, 1], forwards_at_yield


def test_prompt_index_says_when_no_next_token_can_complete_a_match():
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupIndex

    index = PromptLookupIndex([1, 2, 3, 4, 5, 6], min_ngram=3, max_ngram=4)
    # Too short to end in a two-token head.
    assert index.may_match_after_next([2]) is False
    # (2, 3) heads the indexed (2, 3, 4); (3, 2) heads nothing.
    assert index.may_match_after_next([9, 2, 3]) is True
    assert index.may_match_after_next([9, 3, 2]) is False
    # (5, 6) would need a continuation past the end of the prompt.
    assert index.may_match_after_next([5, 6]) is False
    # And it agrees with ``propose`` for every possible next token.
    for generated in ([9, 2, 3], [9, 3, 2], [5, 6], [1, 2]):
        could = any(
            index.propose([*generated, nxt]) is not None for nxt in range(1, 10)
        )
        assert could <= index.may_match_after_next(generated), generated


def _scripted_depths(monkeypatch, depths):
    from rapid_mlx.spec_decode.mtp import generator as generator_mod

    script = iter(depths)

    class _Scripted:
        def pick_k(self):
            return next(script)

        def record(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        generator_mod, "request_depth_controller", lambda *args: _Scripted()
    )


def test_holding_the_run_ahead_changes_no_schedule(monkeypatch):
    """``may_run_ahead`` (another request waiting) only holds the launch:
    the request verifies the same widths and returns the same tokens."""
    from rapid_mlx.spec_decode.mtp.reproducible_depth import GreedySchedule

    depths = ([0] * 4 + [2] * 3) * 10
    runs = []
    for allow in (True, False):
        _scripted_depths(monkeypatch, iter(depths))
        model = _WidthSensitiveTarget(lambda rows: 77 if rows > 1 else 200)
        tokens, ahead = _run(
            model,
            PROMPT,
            40,
            greedy_schedule=GreedySchedule(
                depth=2, steep_verify=True, round_costs=(1.0, 1.4, 1.9)
            ),
            may_run_ahead=lambda allow=allow: allow,
        )
        runs.append((tokens, list(model.widths), ahead))
    (ran_tokens, ran_widths, ran_ahead), (held_tokens, held_widths, held_ahead) = runs
    assert ran_tokens == held_tokens
    assert ran_widths == held_widths
    assert 1 in ran_ahead
    assert max(held_ahead) <= 0, held_ahead


def test_a_held_run_ahead_leaves_the_cache_at_the_delivered_tokens():
    """Once held, the next yielded token has no target row beyond it: the
    cache holds the prompt plus every delivered token but the last."""
    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    waiting = {"now": False}
    caches = [_CountingKVCache(), _CountingKVCache()]
    model = _WidthSensitiveTarget(_no_tie)
    offsets_at_yield = []
    for index, (_token, _lp, _drafted) in enumerate(
        mtp_generate_step(
            mx.array(PROMPT, dtype=mx.uint32),
            model,
            max_tokens=12,
            prompt_cache=caches,
            accept_counter=MTPAcceptCounter(),
            model_id="park-pipeline-test",
            temp=0.0,
            max_k=2,
            prompt_lookup_enabled=False,
            greedy_schedule=_parked(),
            may_run_ahead=lambda: not waiting["now"],
        )
    ):
        offsets_at_yield.append(caches[0].offset)
        if index == 4:
            waiting["now"] = True
    delivered_boundary = [len(PROMPT) + i for i in range(12)]
    # Running ahead: one row past the boundary.
    assert offsets_at_yield[:5] == [b + 1 for b in delivered_boundary[:5]]
    # The step already in flight is consumed; from then on, at the boundary.
    assert offsets_at_yield[6:] == delivered_boundary[6:], offsets_at_yield


def test_parked_round_cost_excludes_the_callers_time(monkeypatch):
    """The adaptive controller compares depths on what a round costs on the
    generator's side of the yield. A step launched ahead of delivery runs
    while the caller works, but the caller's time (a slow client, a long
    detokenize) is not the step's cost and must not be charged to it."""
    from rapid_mlx.spec_decode.mtp import generator as generator_mod
    from tests.test_mtp_greedy_determinism import _install_clock, _VirtualClock

    charged: list[tuple[int, float]] = []

    class _ParkedController:
        def pick_k(self):
            return 0

        def record(self, depth, cost_ms, accepts):
            charged.append((depth, cost_ms))

    monkeypatch.setattr(
        generator_mod,
        "get_or_create_controller",
        lambda *args, **kwargs: _ParkedController(),
    )
    clock = _VirtualClock(lambda rows: 0.001)  # a forward's build: 1 ms
    _install_clock(monkeypatch, clock)
    model = _WidthSensitiveTarget(_no_tie, clock)

    from rapid_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    delivered = 0
    for _token in mtp_generate_step(
        mx.array(PROMPT, dtype=mx.uint32),
        model,
        max_tokens=10,
        prompt_cache=[_CountingKVCache(), _CountingKVCache()],
        accept_counter=MTPAcceptCounter(),
        model_id="park-pipeline-test",
        temp=0.5,
        max_k=2,
        prompt_lookup_enabled=False,
    ):
        delivered += 1
        clock.now += 0.1  # the caller takes 100 ms per token

    assert delivered == 10
    assert [depth for depth, _ in charged] == [0] * 10
    # Each round pays for building at most two one-row steps (its own, and
    # the next one it launched); never for the caller's 100 ms.
    assert all(cost <= 2.0 + 1e-6 for _, cost in charged), charged
    assert sum(cost for _, cost in charged) == pytest.approx(10.0)
