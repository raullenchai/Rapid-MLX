# SPDX-License-Identifier: Apache-2.0
"""Frozen MTP cost profiles and EV decisions, independent of Metal."""

import sys
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rapid_mlx.spec_decode.mtp.reproducible_depth import (
    GreedySchedule,
    Timings,
    advance_greedy_round,
    bounded_round_costs,
    class_round_costs,
    reproducible_schedule,
    request_depth_controller,
    request_round_costs,
    reset_for_tests,
    round_costs,
    time_target_forwards,
)


@pytest.fixture(autouse=True)
def isolated_profiles():
    reset_for_tests()
    yield
    reset_for_tests()


def test_measured_profile_is_frozen_and_shared_between_requests():
    calls = []

    def measure(k):
        calls.append(k)
        return 50.0, (50.5, 51.0, 51.5), 0.4

    schedule = reproducible_schedule("model", 3, measure)
    assert schedule.round_costs == pytest.approx(
        round_costs(50.0, (50.5, 51.0, 51.5), 0.4)
    )
    assert schedule.round_costs != class_round_costs(False, 3)
    assert (
        reproducible_schedule("model", 3, lambda k: pytest.fail("remeasured"))
        is schedule
    )
    assert calls == [3]
    with pytest.raises(FrozenInstanceError):
        schedule.round_costs = ()
    first = request_depth_controller(schedule.round_costs, 3)
    second = request_depth_controller(schedule.round_costs, 3)
    for _ in range(50):
        a, b = first.pick_k(), second.pick_k()
        assert a == b
        first.record(a, schedule.round_costs[a], [True] * a)
        second.record(b, schedule.round_costs[b], [True] * b)
    # Acceptance remains request-local.
    first.record(3, schedule.round_costs[3], [False])
    assert first.acc._seen != second.acc._seen


@pytest.mark.parametrize("steep", [False, True])
def test_measured_curve_is_normalized_monotone_and_bounded(steep):
    fallback = class_round_costs(steep, 3)
    costs = bounded_round_costs((20.0, 20.01, 19.99, 60.0), steep, 3)
    assert costs[0] == 1.0
    assert costs[1] == pytest.approx(1.005)
    assert costs[2] == pytest.approx(1.01)
    for k in range(1, 4):
        delta = costs[k] - costs[k - 1]
        assert delta >= 0.005 - 1e-12
        assert delta <= fallback[k] - fallback[k - 1] + 1e-12
        assert costs[k] <= fallback[k] + 1e-12


@pytest.mark.parametrize(
    "costs",
    [
        (),
        (1.0, 1.1),
        (1.0, float("nan"), 1.2),
        (1.0, float("inf"), 1.2),
        (0.0, 1.0, 2.0),
        (1.0, 1.5, 1.1),
    ],
)
def test_invalid_or_badly_inverted_curve_uses_class_fallback(costs):
    assert bounded_round_costs(costs, True, 2) == class_round_costs(True, 2)


@pytest.mark.parametrize(
    "timings",
    [
        (0.0, (1.0, 2.0), 0.1),
        (1.0, (), 0.1),
        (1.0, (1.1,), 0.1),
        (1.0, (1.1, 1.2), -0.1),
        (float("nan"), (1.1, 1.2), 0.1),
        (1.0, (1.1, float("inf")), 0.1),
    ],
)
def test_invalid_measurement_falls_back(timings):
    schedule = reproducible_schedule("invalid", 2, lambda k: timings)
    assert schedule == GreedySchedule(2, round_costs=class_round_costs(False, 2))


def test_failed_measurement_fallback_is_also_frozen():
    def broken(k):
        raise RuntimeError("timing unavailable or unstable")

    schedule = reproducible_schedule("broken", 3, broken)
    assert schedule.round_costs == class_round_costs(False, 3)
    assert (
        reproducible_schedule("broken", 3, lambda k: pytest.fail("retry")) is schedule
    )


def test_complete_round_measurement_takes_precedence_over_isolated_draft():
    timings = Timings(50.0, (50.1, 50.2, 50.3), 10.0, (50.0, 50.8, 51.6, 52.4))
    schedule = reproducible_schedule("complete", 3, lambda k: timings)
    assert schedule.round_costs == pytest.approx((1.0, 1.016, 1.032, 1.048))
    assert schedule.round_costs[2] < 1.04


@pytest.mark.parametrize("step_ms", [1.0, 1000.0])
@pytest.mark.parametrize("row_cost,depth", [(0.46, 2), (1.0, 0)])
def test_steep_real_rounds_gate_on_raw_profitability_before_bounding(
    step_ms, row_cost, depth
):
    verify = tuple(step_ms * (1 + k * row_cost) for k in (1, 2))
    rounds = (step_ms,) + tuple(v + k * 0.15 * step_ms for k, v in enumerate(verify, 1))
    schedule = reproducible_schedule(
        "steep-real", 2, lambda k: Timings(step_ms, verify, 0.15 * step_ms, rounds)
    )
    assert schedule.steep_verify
    assert schedule.depth == depth
    assert schedule.round_costs == pytest.approx(
        bounded_round_costs(tuple(c / step_ms for c in rounds), True, 2)
    )
    if depth == 0:
        # Bounding alone makes this curve appear profitable; the raw gate
        # must still park rather than inventing a speedup from the ceiling.
        assert schedule.round_costs[2] < 3


def test_processor_adjustment_preserves_measured_increments():
    schedule = GreedySchedule(3, True, (1.0, 1.50, 1.55, 1.61))
    adjusted = request_round_costs(schedule, parks_pipelined=False)
    assert adjusted == pytest.approx((1.0, 1.2, 1.24, 1.288))
    assert adjusted != class_round_costs(True, 3, parks_pipelined=False)


def test_processor_adjustment_cannot_make_shallow_drafts_free():
    # A host can be steep at its widest shape but cheap at K=1.
    schedule = GreedySchedule(3, True, (1.0, 1.02, 1.10, 1.60))
    assert request_round_costs(schedule, parks_pipelined=False) == pytest.approx(
        (1.0, 1.005, 1.01, 1.28)
    )


def test_measured_costs_make_second_draft_pay_at_moderate_acceptance():
    measured = request_depth_controller((1.0, 1.015, 1.03, 1.045), 3)
    old = request_depth_controller(class_round_costs(False, 3), 3)
    for controller in (measured, old):
        # Exact conditional rates, already trusted at every candidate depth.
        controller.acc._rate = [0.0, 0.70, 0.45, 0.0]
        controller.acc._seen = [0, 100, 100, 100]
    assert measured._selected() >= 2
    assert old._selected() == 1


def test_unprofitable_measurement_still_parks_before_cost_clamping():
    schedule = reproducible_schedule("dear", 2, lambda k: (10.0, (20.0, 30.0), 5.0))
    assert schedule.depth == 0
    assert schedule.round_costs == class_round_costs(True, 2)


def test_round_drain_does_not_enter_next_round():
    stats = {}
    entered = []

    def rounds():
        for count in (3, 1, 2):
            entered.append(count)
            stats["round_tokens"] = count
            yield from range(count)

    gen = rounds()
    assert advance_greedy_round(gen, stats) == 3
    assert entered == [3]
    assert advance_greedy_round(gen, stats) == 1
    assert entered == [3, 1]


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("budget", [1.5, 5.0])
def test_startup_times_real_rounds_round_robin_without_inner_sync(
    monkeypatch, noisy, budget
):
    import rapid_mlx.spec_decode.mtp.reproducible_depth as depth_module

    monkeypatch.setattr(depth_module, "TIMING_BUDGET_SECONDS", budget)
    clock = SimpleNamespace(now=0.0)
    events = []
    mlx = ModuleType("mlx")
    mx = ModuleType("mlx.core")
    mx.zeros, mx.uint32 = np.zeros, np.uint32
    mx.eval = lambda *args: None
    mx.synchronize = lambda: events.append("drain")
    mx.stream = lambda _: nullcontext()
    mx.default_stream = lambda _: None
    mx.default_device = lambda: None
    mlx.core = mx
    models = ModuleType("mlx_lm.models")
    models.cache = SimpleNamespace(make_prompt_cache=lambda model: [])
    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.models = models
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", mx),
        ("mlx_lm", mlx_lm),
        ("mlx_lm.models", models),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        depth_module, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )

    class Model:
        def __call__(self, tokens, cache, return_hidden, n_confirmed):
            rows = tokens.shape[1]
            clock.now += 0.050 + (rows - 1) * 0.001
            return np.zeros((1, rows, 4)), np.ones((1, rows, 8))

        def make_mtp_cache(self):
            return []

        def mtp_forward(self, hidden, tokens, cache):
            clock.now += 0.001
            return np.zeros((1, 1, 4))

    closed = []

    def real_rounds(model, prompt, depth, stats):
        try:
            count = 0
            while True:
                count += 1
                events.append((depth, count))
                # Re-entry warmup is odd after bootstrap; timed visits even.
                multiplier = (0.5, 1, 1.5)[(count // 2) % 3] if noisy else 1
                clock.now += (0.050 + 0.001 * depth) * multiplier
                stats.update(round_tokens=depth + 1, round_depth=depth)
                yield from range(depth + 1)
        finally:
            closed.append(depth)

    monkeypatch.setattr(depth_module, "calibration_generator", real_rounds)
    if budget < 5 or noisy:
        with pytest.raises(depth_module.RoundCalibrationError) as failed:
            time_target_forwards(Model(), 3)
        # The widths (+1 ms per row on 50 ms) classify a cheap host.
        assert failed.value.steep is False
        if budget < 5:
            assert isinstance(failed.value.__cause__, TimeoutError)
            assert "insufficient startup round samples" in str(failed.value.__cause__)
        else:
            assert isinstance(failed.value.__cause__, ValueError)
            assert "unstable startup round timings" in str(failed.value.__cause__)
    else:
        timings = time_target_forwards(Model(), 3)
        assert timings.step_ms == pytest.approx(50)
        assert timings.verify_ms == pytest.approx((51, 52, 53))
        assert timings.draft_ms == pytest.approx(1)
        assert timings.round_ms == pytest.approx((50, 51, 52, 53))
        # Every sample has a drain followed by two intact rounds of one K;
        # the timed second round has no inserted sync at its boundary.
        assert events[:4] == [(0, 1), (1, 1), (2, 1), (3, 1)]
        assert events[4:10] == ["drain", (0, 2), (0, 3), "drain", (1, 2), (1, 3)]
    assert sorted(closed) == [0, 1, 2, 3]


def test_incomplete_real_round_profile_falls_back():
    schedule = reproducible_schedule(
        "incomplete", 3, lambda k: Timings(50, (51, 52, 53), 1, (51, 52, 53))
    )
    assert schedule.round_costs == class_round_costs(False, 3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_real_round_timings_keep_fallback_depth(bad):
    schedule = reproducible_schedule(
        "invalid-real", 3, lambda k: Timings(50, (51, 52, 53), 1, (50, 51, bad, 53))
    )
    assert schedule.depth == 3
    assert schedule.round_costs == class_round_costs(False, 3)


def test_failed_round_calibration_keeps_measured_host_class():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import RoundCalibrationError

    def measure(k):
        raise RoundCalibrationError(True)

    schedule = reproducible_schedule("round-fail-steep", 3, measure)
    assert schedule.depth == 3
    assert schedule.steep_verify is True
    assert schedule.round_costs == class_round_costs(True, 3)


def test_inverted_real_round_timings_fall_back_depth_and_costs_together():
    # A rejected curve must not leave its own (here: zero) depth behind.
    schedule = reproducible_schedule(
        "inverted-real", 2, lambda k: Timings(50, (51, 52), 1, (1, 4, 3.8))
    )
    assert schedule.depth == 2
    assert schedule.round_costs == class_round_costs(False, 2)
