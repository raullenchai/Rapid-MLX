import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from rapid_mlx.byom import preflight
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.recommendations import fitting_pick, load_recommendation_tiers

pytestmark = pytest.mark.catalog_memory_preflight

GIB = 1 << 30
_REAL_PROBE = preflight._mlx_recommended_working_set_bytes


@pytest.fixture(autouse=True)
def offline_preflight(monkeypatch):
    monkeypatch.setattr(preflight, "_mlx_recommended_working_set_bytes", lambda: None)
    monkeypatch.setattr("rapid_mlx._download_gate.is_repo_cached", lambda _: False)
    monkeypatch.setattr(
        "huggingface_hub.model_info", lambda *a, **k: pytest.fail("network lookup")
    )
    monkeypatch.setattr(
        "rapid_mlx._download_gate.estimate_repo_size_bytes",
        lambda *a, **k: pytest.fail("network size lookup"),
    )
    with patch.object(preflight, "_emit_rejection") as emit:
        yield emit


def run(alias, ram, monkeypatch, **overrides):
    profile = resolve_profile(alias)
    assert profile is not None
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: ram * GIB)
    args = dict(
        command="serve",
        model=profile.hf_path,
        _original_alias=alias,
        no_preflight=False,
        disk_stream=False,
    )
    args.update(overrides)
    preflight.run_cli_preflight(
        SimpleNamespace(**args), spinner_factory=lambda _: nullcontext()
    )


@pytest.mark.parametrize(
    ("alias", "ram"),
    [
        ("lfm2.5-24b-a2b-4bit", 16),
        ("qwen3.8-27b-4bit", 24),
        ("qwen3.8-27b-mixed-3.5bpw", 24),
    ],
)
def test_refuses_before_download(alias, ram, monkeypatch, capsys, offline_preflight):
    with pytest.raises(SystemExit) as exc:
        run(alias, ram, monkeypatch)
    assert exc.value.code == 1
    output = capsys.readouterr().err
    assert "Nothing was downloaded" in output
    pick = output.split("rapid-mlx serve ", 1)[1].split()[0]
    assert pick.casefold() != alias.casefold()
    assert (
        offline_preflight.call_args.args[1].failure_class
        == preflight.INSUFFICIENT_MEMORY
    )


@pytest.mark.parametrize(
    ("alias", "ram"), [("bonsai-27b-2bit", 24), ("qwen3.5-4b-4bit", 16)]
)
def test_fitting_models_pass(alias, ram, monkeypatch):
    run(alias, ram, monkeypatch)


def test_refuses_when_fitting_pick_fails(monkeypatch, capsys):
    with (
        patch("rapid_mlx.recommendations.fitting_pick", side_effect=RuntimeError),
        pytest.raises(SystemExit) as exc,
    ):
        run("qwen3.8-27b-4bit", 24, monkeypatch)
    assert exc.value.code == 1
    assert "rapid-mlx models" in capsys.readouterr().err


def test_tier_picks_pass_at_floor(monkeypatch):
    for tier in load_recommendation_tiers():
        for pick in tier.picks:
            run(pick.alias, tier.floor_gb, monkeypatch)


@pytest.mark.parametrize("mode", ["disk_stream", "no_preflight", "pull", "cached"])
def test_bypasses(mode, monkeypatch):
    overrides = {}
    if mode == "pull":
        overrides["command"] = "pull"
    elif mode == "cached":
        monkeypatch.setattr("rapid_mlx._download_gate.is_repo_cached", lambda _: True)
    else:
        overrides[mode] = True
    run("qwen3.8-27b-4bit", 24, monkeypatch, **overrides)


def test_metal_budget(monkeypatch):
    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: 20 * GIB
    )
    assert preflight.metal_working_set_bytes(16 * GIB) == 16 * GIB
    assert preflight.metal_working_set_bytes(64 * GIB) == 20 * GIB
    monkeypatch.setattr(preflight, "_mlx_recommended_working_set_bytes", lambda: None)
    assert preflight.metal_working_set_bytes(16 * GIB) == int(16 * GIB * 2 / 3)
    assert preflight.metal_working_set_bytes(64 * GIB) == 48 * GIB
    assert preflight.metal_working_set_bytes(None) is None


@pytest.mark.parametrize("value, expected", [(123, 123), (0, None), (None, None)])
def test_real_mlx_working_set_probe(monkeypatch, value, expected):
    fake_pkg = ModuleType("mlx")
    fake_core = ModuleType("mlx.core")

    def device_info():
        if value is None:
            raise RuntimeError("device unavailable")
        return {"max_recommended_working_set_size": value}

    fake_core.device_info = device_info
    fake_pkg.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    assert _REAL_PROBE() == expected


def test_catalog_check_passes_when_ram_unknown(monkeypatch):
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: None)
    args = SimpleNamespace(command="serve", model="qwen3.8-27b-4bit")
    preflight._check_catalog_memory_fit(args)


def test_met_floor_passes_when_weights_unknown(monkeypatch):
    monkeypatch.setattr("rapid_mlx.model_sizes.size_bytes", lambda _: None)
    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: 12 * GIB
    )
    run("qwen3-coder-30b-4bit", 24, monkeypatch)


def test_unmeasured_model_passes_when_weights_unknown(monkeypatch):
    monkeypatch.setattr("rapid_mlx.model_sizes.size_bytes", lambda _: None)
    run("qwen3.8-27b-mixed-3.5bpw", 24, monkeypatch)


def test_catalog_check_swallows_resolver_exception(monkeypatch, capsys):
    args = SimpleNamespace(command="serve", model="qwen3.8-27b-4bit")

    def fail(_):
        raise RuntimeError("resolution failed")

    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_profile", fail)
    preflight._check_catalog_memory_fit(args)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("recommended", [20 * GIB, None])
def test_met_floor_passes_with_sufficient_or_unknown_metal_limit(
    monkeypatch, recommended
):
    probe = patch.object(
        preflight, "_mlx_recommended_working_set_bytes", return_value=recommended
    )
    with probe as mock_probe:
        run("qwen3-coder-30b-4bit", 24, monkeypatch)
    mock_probe.assert_called_once()


def test_met_floor_refuses_weights_above_metal_limit(monkeypatch, capsys):
    from rapid_mlx.model_sizes import size_bytes

    alias = "qwen3-coder-30b-4bit"
    profile = resolve_profile(alias)
    assert profile is not None
    weights = size_bytes(profile.hf_path)
    assert weights is not None and weights > 12 * GIB
    with (
        patch.object(
            preflight, "_mlx_recommended_working_set_bytes", return_value=12 * GIB
        ) as probe,
        pytest.raises(SystemExit),
    ):
        run(alias, 24, monkeypatch)
    probe.assert_called_once()
    assert f"needs ~{preflight._gb(weights)} of GPU memory" in capsys.readouterr().err


def test_fitting_pick():
    tier = next(t for t in load_recommendation_tiers() if t.floor_gb == 16)
    assert fitting_pick(16, 100) == tier.picks[0].alias
    assert (
        fitting_pick(16, 100, exclude=tier.picks[0].alias.upper())
        == tier.picks[1].alias
    )
    assert fitting_pick(16, 0) is None


def test_fitting_pick_searches_lower_tiers():
    tiers = load_recommendation_tiers()
    lower, current = tiers[1], tiers[2]
    budget = 3.0
    assert all(pick.footprint_gb > budget for pick in current.picks)
    expected = next(pick.alias for pick in lower.picks if pick.footprint_gb <= budget)
    assert fitting_pick(current.floor_gb, budget) == expected
    assert (
        fitting_pick(current.floor_gb, budget, exclude=expected.upper())
        == tiers[0].picks[0].alias
    )
