"""Measured tuning cells freeze greedy MTP costs across process restarts."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx import tuning
from rapid_mlx.spec_decode.mtp import reproducible_depth as rd
from rapid_mlx.tuning import defaults
from rapid_mlx.tuning import mtp as policy

NAME = "mtp_forward_costs_ms"
COSTS = (6.0, 8.0, 9.5, 11.0, 12.5, 0.8)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(defaults, "overlay_path", lambda: tmp_path / "overlay.json")
    defaults.overlay_profile.cache_clear()
    rd.reset_for_tests()
    yield
    defaults.overlay_profile.cache_clear()
    rd.reset_for_tests()


def context():
    return tuning.TuningContext(
        tuning.HardwareProfile(gpu_generation=15, tier="Ultra"),
        "0.32.4",
        "26.5.2",
        mode="mtp",
        has_mtp_head=True,
    )


def model():
    return SimpleNamespace(
        mtp=object(), mtp_forward=lambda: None, make_mtp_cache=lambda: []
    )


def entry(value=COSTS, name="test-costs"):
    return dict(
        id=name,
        hardware=[dict(gpu_generation=15, tier="Ultra")],
        knobs={NAME: value},
        provenance={"evidence": "synthetic test measurement"},
    )


def install_profile(monkeypatch, entries=None):
    monkeypatch.setattr(
        tuning,
        "builtin_profile",
        lambda: defaults.Profile("test.1", tuple(entries or [entry()])),
    )
    monkeypatch.setattr(policy, "_model_context", lambda m: context())


@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        1,
        "1,2,3",
        (),
        (1.0, 2.0),
        (1.0,) * 7,
        (True, 2.0, 0.1),
        (1.0, "2", 0.1),
        (0.0, 2.0, 0.1),
        (1.0, -2.0, 0.1),
        (1.0, 2.0, -0.1),
        (1.0, float("nan"), 0.1),
        (1.0, 2.0, float("inf")),
        (10**1000, 2, 0),
    ],
)
def test_invalid_curves(value):
    assert not tuning.KNOBS[NAME].valid(value)


@pytest.mark.parametrize("value", [(1, 2, 0), COSTS, list(COSTS)])
def test_valid_curves(value):
    assert tuning.KNOBS[NAME].valid(value)


def test_profile_load_normalizes_and_rejects_invalid_siblings(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=1,
                profile_version="test.1",
                generated_at="2026-11-10T00:00:00Z",
                entries=[entry(), entry([1, 0, 1], "invalid")],
            )
        )
    )
    profile = defaults.load_profile(path, tuning.KNOBS)
    assert len(profile.entries) == 1
    assert profile.entries[0]["knobs"][NAME] == COSTS


@pytest.mark.parametrize(
    "changes",
    [dict(mode="plain"), dict(batch=2), dict(width=2), dict(has_mtp_head=False)],
)
def test_curve_requires_single_request_head_capability(changes):
    decision = tuning.resolve(NAME, replace(context(), **changes), environ={})
    assert decision.value == () and decision.source == "capability"


def test_unmeasured_cell_has_empty_default():
    assert tuning.resolve(NAME, context(), environ={}) == tuning.Decision((), "default")


@pytest.mark.parametrize("cap", range(1, 5))
def test_curve_is_trimmed_to_requested_depth(monkeypatch, cap, caplog):
    install_profile(monkeypatch)
    with caplog.at_level("INFO"):
        assert policy.forward_costs(model(), cap) == (
            COSTS[0],
            *COSTS[1 : cap + 1],
            COSTS[-1],
        )
    assert "builtin:test-costs" in caplog.text and "test.1" in caplog.text


def test_no_head_and_zero_cap_avoid_hardware_probe(monkeypatch):
    import mlx.core as mx

    probe = Mock(side_effect=AssertionError("unexpected hardware probe"))
    monkeypatch.setattr(mx, "device_info", probe, raising=False)
    assert policy.forward_costs(object(), 1) == ()
    assert policy.forward_costs(model(), 0) == ()
    probe.assert_not_called()


@pytest.mark.parametrize(
    "entries", [[entry((6.0, 8.0, 0.8))], [entry(), entry((6, 8, 1), "conflict")]]
)
def test_partial_or_conflicting_curve_uses_live_fallback(monkeypatch, entries):
    install_profile(monkeypatch, entries)
    measure = Mock(return_value=(10.0, (12.0, 14.0), 1.0))
    monkeypatch.setattr(rd, "time_target_forwards", measure)
    result = rd.greedy_schedule(model(), "partial", 2)
    assert result == rd.schedule_from_timings(10.0, (12.0, 14.0), 1.0, 2)
    assert rd.greedy_schedule(model(), "partial", 2) is result
    measure.assert_called_once()


def test_profile_precedes_cached_live_costs_and_never_times(monkeypatch):
    rd.reproducible_schedule("same-model", 3, lambda k: (1, (4, 8, 12), 2))
    install_profile(monkeypatch)
    measure = Mock(side_effect=AssertionError("profile must bypass live timing"))
    monkeypatch.setattr(rd, "time_target_forwards", measure)
    schedules = []
    sequences = []
    for _ in range(5):
        schedule = rd.greedy_schedule(model(), "same-model", 3)
        schedules.append(schedule)
        controller = rd.request_depth_controller(schedule.round_costs, 3)
        choices = []
        for i in range(150):
            k = controller.pick_k(refresh_park_costs=False)
            choices.append(k)
            controller.record(
                k,
                rd.request_round_cost(schedule.round_costs, k),
                [j < i % 4 for j in range(k)],
            )
        sequences.append(choices)
        rd.reset_for_tests()
    assert all(s == schedules[0] for s in schedules)
    assert all(s == sequences[0] for s in sequences)
    assert schedules[0].round_costs == rd.round_costs(COSTS[0], COSTS[1:4], COSTS[-1])
    measure.assert_not_called()


def test_overlay_cost_curve_and_unknown_cell(monkeypatch):
    install_profile(monkeypatch)
    monkeypatch.setattr(
        tuning,
        "overlay_profile",
        lambda: defaults.Profile("overlay", (entry((1, 2, 0), "user"),)),
    )
    assert policy.forward_costs(model(), 1) == (1, 2, 0)
    assert policy.forward_costs(model(), 2) == ()
    monkeypatch.setattr(
        policy,
        "_model_context",
        lambda m: replace(context(), hardware=tuning.HardwareProfile()),
    )
    assert policy.forward_costs(model(), 1) == ()


def test_real_adapter_qualifies_installed_experts(monkeypatch):
    import mlx.core as mx

    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"device_name": "Apple M3 Ultra", "memory_size": 256 * 2**30},
        raising=False,
    )
    fake = model()
    fake.args = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        moe_intermediate_size=512,
        num_experts=256,
        num_experts_per_tok=8,
    )
    fake.named_modules = lambda: [
        (
            "model.layers.0.mlp.switch_mlp.gate_proj",
            SimpleNamespace(weight=object(), bits=4, group_size=64, mode="affine"),
        )
    ]
    seen = []

    def capture_context(info, **kwargs):
        ctx = tuning.runtime_context(
            info, library_version="0.32.4", macos="26.5.2", **kwargs
        )
        seen.append(ctx)
        return ctx

    monkeypatch.setattr(policy, "runtime_context", capture_context)
    result = policy.forward_costs(fake, 4)
    assert len(result) == 6
    assert result == tuning.resolve(NAME, seen[0], environ={}).value
    assert seen[0].model.geometry == tuning.QWEN35_AFFINE4
    assert seen[0].model.quant_mode == "affine"
    fake.named_modules = lambda: []
    assert policy.forward_costs(fake, 4) == ()
    assert seen[1].model.bits is None


@pytest.mark.parametrize(
    "generation,tier,cores,memory,mlx,macos,depth",
    [
        (15, "Ultra", 60, 256, "0.32.4", "26.5.2", 3),
        (17, "Max", 40, 36, "0.32.3", "27.0.1", 4),
    ],
)
def test_builtin_measured_cells_and_explicit_depth_priority(
    monkeypatch, generation, tier, cores, memory, mlx, macos, depth
):
    ctx = replace(
        context(),
        hardware=tuning.HardwareProfile(
            gpu_generation=generation,
            tier=tier,
            gpu_cores=cores,
            memory_size=memory * 2**30,
        ),
        mlx=mlx,
        macos=macos,
        model=tuning.ModelProfile(
            "qwen3_5_moe_text", *tuning.QWEN35_AFFINE4, quant_mode="affine"
        ),
    )
    decision = tuning.resolve(NAME, ctx, environ={})
    assert decision.source.startswith("builtin:mtp-costs-")
    assert len(decision.value) >= depth + 2
    assert tuning.resolve("mtp_max_depth", ctx, environ={}).value == depth
    monkeypatch.setattr(policy, "_model_context", lambda m: ctx)
    assert policy.resolve_max_depth(model(), 2, explicit=True) == 2
    monkeypatch.setenv("RAPID_MLX_MTP_MAX_DEPTH", "1")
    assert policy.resolve_max_depth(model(), depth, explicit=False) == 1
    assert len(policy.forward_costs(model(), 1)) == 3
    # The explicit cell includes unknown cores but pins memory/software/model.
    assert (
        tuning.resolve(
            NAME,
            replace(ctx, hardware=replace(ctx.hardware, gpu_cores=None)),
            environ={},
        ).value
        == decision.value
    )
    for unknown in (
        replace(ctx, hardware=replace(ctx.hardware, gpu_generation=99)),
        replace(ctx, hardware=replace(ctx.hardware, tier="Pro")),
        replace(ctx, hardware=replace(ctx.hardware, memory_size=None)),
        replace(ctx, mlx="99.0"),
        replace(ctx, macos="99.0"),
        replace(ctx, model=tuning.ModelProfile("qwen3_5_text", hidden_size=2560)),
        replace(ctx, model=replace(ctx.model, quant_mode="mxfp4")),
    ):
        assert tuning.resolve(NAME, unknown, environ={}).value == ()


def test_small_calibration_noise_changes_the_depth_program():
    sequences = []
    for price in (1.34, 1.35):
        costs = (1.0, price, 1.7, 2.0)
        controller = rd.request_depth_controller(costs, 3)
        choices = []
        for i in range(150):
            k = controller.pick_k(refresh_park_costs=False)
            choices.append(k)
            controller.record(k, costs[k], [j < i % 4 for j in range(k)])
        sequences.append(choices)
    assert sequences[0] != sequences[1]
