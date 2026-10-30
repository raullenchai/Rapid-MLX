from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rapid_mlx.byom import preflight
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.recommendations import fitting_pick, load_recommendation_tiers

GIB = 1 << 30


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


def test_fitting_pick():
    tier = next(t for t in load_recommendation_tiers() if t.floor_gb == 16)
    assert fitting_pick(16, 100) == tier.picks[0].alias
    assert (
        fitting_pick(16, 100, exclude=tier.picks[0].alias.upper())
        == tier.picks[1].alias
    )
    assert fitting_pick(16, 0) is None
