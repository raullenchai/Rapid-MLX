"""Calibration must not amplify small forward-time noise into a class cliff."""

import pytest

from rapid_mlx.spec_decode.mtp.reproducible_depth import (
    reproducible_schedule,
    request_depth_controller,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def reset_profiles():
    reset_for_tests()
    yield
    reset_for_tests()


def test_verify_threshold_does_not_flip_the_request_cost_curve():
    schedules = [
        reproducible_schedule(str(verify), 1, lambda k, v=verify: (10.0, (v,), 1.0))
        for verify in (13.29, 13.31)
    ]
    assert abs(schedules[1].round_costs[1] - schedules[0].round_costs[1]) < 0.01
    from rapid_mlx.spec_decode.mtp.reproducible_depth import request_round_costs

    # Logits processors disable run-ahead, but must not restore the old class
    # cliff on the very same measured profile.
    for pipelined in (False, True):
        prices = [request_round_costs(s, parks_pipelined=pipelined) for s in schedules]
        assert abs(prices[1][1] - prices[0][1]) < 0.01
    choices = []
    for schedule in schedules:
        controller = request_depth_controller(schedule.round_costs, 1)
        # Identical observed quality on either side of the old class threshold.
        for i in range(100):
            controller.acc.observe(1, int(i % 10 < 7))
        choices.append(controller.pick_k())
    assert choices == [1, 1]


def test_calibration_tracks_common_drift_and_rejects_isolated_stalls():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import summarize_forward_samples

    steps = [10.0, 20.0, 11.0, 40.0, 12.0, 30.0, 13.0]
    verify = [v * 1.35 for v in steps]
    draft = [v * 0.1 for v in steps]
    verify[3] *= 20  # A verify-only stall must not choose a permanently dear curve.
    step, widths, head = summarize_forward_samples([steps, verify, draft])
    assert widths[0] / step == pytest.approx(1.35)
    assert head / step == pytest.approx(0.1)


@pytest.mark.parametrize(
    "samples",
    [
        [[10.0], [13.0]],
        [[], [13.0], [1.0]],
        [[10.0, 10.0], [13.0], [1.0]],
        [[10.0], [float("nan")], [1.0]],
        [[10.0], [13.0], [0.0]],
    ],
)
def test_incomplete_or_invalid_samples_are_rejected(samples):
    from rapid_mlx.spec_decode.mtp.reproducible_depth import summarize_forward_samples

    with pytest.raises(ValueError):
        summarize_forward_samples(samples)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_measurement_keeps_configured_depth(invalid):
    schedule = reproducible_schedule("bad", 1, lambda k: (invalid, (13.0,), 1.0))
    assert schedule.depth == 1
    assert schedule.round_costs == ()
    assert schedule.verify_costs == ()


def test_partial_calibration_is_not_a_partial_policy():
    schedule = reproducible_schedule("partial", 2, lambda k: (10.0, (13.0,), 1.0))
    assert schedule.depth == 2 and not schedule.round_costs


def test_measured_copy_cost_has_no_class_cliff_and_charges_wide_tiles():
    from rapid_mlx.spec_decode.mtp.reproducible_depth import measured_verify_cost

    profiles = [
        reproducible_schedule(str(v), 1, lambda k, v=v: (10.0, (v,), 1.0))
        for v in (13.29, 13.31)
    ]
    for rows in (1, 2, 3, 9, 32, 64):
        costs = [measured_verify_cost(profile, rows) for profile in profiles]
        assert abs(costs[0] - costs[1]) / costs[0] < 0.002
    assert measured_verify_cost(profiles[0], 2) == pytest.approx(1.329)
    assert measured_verify_cost(profiles[0], 64) == 2 * measured_verify_cost(
        profiles[0], 32
    )
    with pytest.raises(ValueError):
        measured_verify_cost(profiles[0], 0)


def test_profile_is_frozen_even_if_later_measurement_would_park():
    first = reproducible_schedule("same", 1, lambda k: (10.0, (13.0,), 1.0))
    second = reproducible_schedule("same", 1, lambda k: (10.0, (30.0,), 10.0))
    assert first is second
    assert first.depth == 1


def test_unprofitable_target_parks_without_a_forced_class():
    schedule = reproducible_schedule(
        "expensive", 2, lambda k: (10.0, (30.0, 40.0), 5.0)
    )
    assert schedule.depth == 0


@pytest.mark.parametrize("max_k", [1, 2, 8])
def test_measured_profile_respects_the_configured_ceiling(max_k):
    widths = tuple(10.0 + 2.0 * k for k in range(1, max_k + 1))
    schedule = reproducible_schedule("depth", max_k, lambda k: (10.0, widths, 1.0))
    controller = request_depth_controller(schedule.round_costs, schedule.depth)
    for _ in range(20):
        controller.acc.observe(max_k, max_k)
    assert controller.pick_k() <= max_k
    assert len(schedule.round_costs) == len(schedule.verify_costs) == max_k + 1


def test_good_and_bad_drafters_follow_their_own_quality_on_the_same_profile():
    schedule = reproducible_schedule("quality", 1, lambda k: (10.0, (13.5,), 1.0))
    choices = []
    for accepted in (0, 1):
        controller = request_depth_controller(schedule.round_costs, schedule.depth)
        for _ in range(30):
            controller.acc.observe(1, accepted)
        choices.append(controller.pick_k())
    assert choices == [0, 1]
