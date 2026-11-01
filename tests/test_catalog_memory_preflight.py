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
def offline_preflight(monkeypatch, request):
    monkeypatch.setattr(preflight, "_mlx_recommended_working_set_bytes", lambda: None)
    if request.node.name != "test_cached_mflux_snapshot_keeps_floor_warning":
        monkeypatch.setattr("rapid_mlx._download_gate.is_repo_cached", lambda _: False)
    calls = []

    def reject(name):
        def lookup(*args, **kwargs):
            calls.append((name, args, kwargs))
            raise RuntimeError("network lookup")

        return lookup

    monkeypatch.setattr("huggingface_hub.model_info", reject("model_info"))
    monkeypatch.setattr(
        "rapid_mlx._download_gate.estimate_repo_size_bytes",
        reject("estimate_repo_size_bytes"),
    )
    with patch.object(preflight, "_emit_rejection") as emit:
        emit.network_calls = calls
        yield emit
    assert not calls, f"Unexpected network lookups: {calls!r}"


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


def test_deepseek_v41_resident_memory_admission(monkeypatch, capsys):
    alias = "deepseek-v41-flash-1017"
    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: int(222.72 * GIB)
    )
    run(alias, 256, monkeypatch)
    fit = preflight.catalog_memory_fit(
        alias, ram_bytes=256 * GIB, recommended_working_set_bytes=int(222.72 * GIB)
    )
    assert (fit.estimate_bytes, fit.estimate_source, fit.fits) == (
        204 * GIB,
        "curated_footprint",
        True,
    )

    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: 192 * GIB
    )
    with pytest.raises(SystemExit):
        run(alias, 256, monkeypatch)
    assert f"needs ~{preflight._gb(204 * GIB)} of GPU memory" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        run(alias, 128, monkeypatch)
    assert "at least 224 GB" in capsys.readouterr().err


def test_bonsai_alias_and_repo_are_refused_identically(monkeypatch):
    alias = "bonsai-27b-2bit"
    repo = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
    fits = []
    for name in (alias, repo):
        fits.append(
            preflight.catalog_memory_fit(
                name, ram_bytes=16 * GIB, recommended_working_set_bytes=None
            )
        )
        with pytest.raises(SystemExit) as exc:
            run(name, 16, monkeypatch)
        assert exc.value.code == 1
    assert fits[0] == fits[1]
    assert fits[0].fits is False
    assert fits[0].estimate_bytes == 13 * GIB
    assert fits[0].estimate_source == "curated_footprint"


def test_refuses_when_fitting_pick_fails(monkeypatch, capsys):
    with (
        patch("rapid_mlx.recommendations.fitting_pick", side_effect=RuntimeError),
        pytest.raises(SystemExit) as exc,
    ):
        run("qwen3.8-27b-4bit", 24, monkeypatch)
    assert exc.value.code == 1
    assert "rapid-mlx models" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("alias", "modality", "operations", "expected_pick"),
    [
        ("qwen-image-edit", "image-gen", {"image_to_image"}, None),
        (
            "qwen-image-2.1-bf16",
            "image-gen",
            {"text_to_image", "image_to_image"},
            None,
        ),
        (
            "diffusion-gemma-26b-8bit",
            "text-diffusion",
            set(),
            "diffusion-gemma-26b-4bit",
        ),
        ("gemma-4-31b-8bit", "text", set(), None),
    ],
)
def test_refusal_suggests_fitting_same_task_model(
    alias, modality, operations, expected_pick, monkeypatch, capsys
):
    from rapid_mlx.catalog.legacy import _main_capabilities

    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: 25 * GIB
    )
    with pytest.raises(SystemExit) as exc:
        run(alias, 32, monkeypatch)
    assert exc.value.code == 1
    output = capsys.readouterr().err
    pick = output.split("rapid-mlx serve ", 1)[1].split()[0]
    profile = resolve_profile(pick)
    assert profile is not None
    assert profile.modality == modality
    assert operations.issubset(_main_capabilities(profile)["operation_modes"])
    if expected_pick is not None:
        assert pick == expected_pick
    run(pick, 32, monkeypatch)


def test_text_diffusion_falls_back_to_text_pick():
    pick, label = preflight._same_task_fitting_pick(
        "diffusion-gemma-26b-8bit",
        ram_bytes=32 * GIB,
        recommended_working_set_bytes=12 * GIB,
    )
    assert (pick, label) == (
        fitting_pick(32, 12, exclude="diffusion-gemma-26b-8bit"),
        "text",
    )
    assert pick is not None
    assert resolve_profile(pick).modality == "text"


def test_same_task_pick_unknown_alias():
    assert preflight._same_task_fitting_pick(
        "not-a-catalog-model",
        ram_bytes=32 * GIB,
        recommended_working_set_bytes=25 * GIB,
    ) == (None, None)


def test_refusal_without_fitting_image_model(monkeypatch, capsys):
    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: 3 * GIB
    )
    with pytest.raises(SystemExit) as exc:
        run("flux2-klein-4b", 32, monkeypatch)
    assert exc.value.code == 1
    output = capsys.readouterr().err
    assert "No image generation model in the catalog fits this Mac." in output
    assert "rapid-mlx serve " not in output
    assert "See every model and its size: rapid-mlx models" in output


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


def test_cli_piped_alias_precedes_memory_refusal():
    import os
    import subprocess

    code = """
import sys
from rapid_mlx.byom import preflight
import rapid_mlx._download_gate as gate
preflight.physical_ram_bytes = lambda: 16 << 30
preflight._mlx_recommended_working_set_bytes = lambda: 11 << 30
gate.is_repo_cached = lambda _: False
sys.argv = ['rapid-mlx', 'serve', 'gemma-4-31b-8bit']
from rapid_mlx.cli import main
main()
"""
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout
    assert result.stdout.index("Alias:") < result.stdout.index("needs ~")


def test_met_floor_refuses_when_recommendation_lookup_fails(monkeypatch, capsys):
    with (
        patch.object(
            preflight, "_mlx_recommended_working_set_bytes", return_value=12 * GIB
        ),
        patch(
            "rapid_mlx.recommendations.recommendation_footprint_gb",
            side_effect=RuntimeError,
        ),
        pytest.raises(SystemExit),
    ):
        run("qwen3-coder-30b-4bit", 24, monkeypatch)
    assert "Nothing was downloaded" in capsys.readouterr().err


def test_catalog_fit_uses_measured_footprint_when_weights_lookup_fails():
    with (
        patch("rapid_mlx.model_sizes.size_bytes", side_effect=RuntimeError),
        patch("rapid_mlx.recommendations.recommendation_footprint_gb", return_value=20),
    ):
        result = preflight.catalog_memory_fit(
            "qwen3-coder-30b-4bit",
            ram_bytes=8 * GIB,
            recommended_working_set_bytes=6 * GIB,
        )
    assert result.fits is False
    assert result.fit_basis == "gpu_budget"
    assert result.estimate_source == "curated_footprint"


def test_image_fit_uses_admission_charge():
    result = preflight.catalog_memory_fit(
        "flux-schnell", ram_bytes=32 * GIB, recommended_working_set_bytes=24 * GIB
    )
    assert result.estimate_bytes == int(21.4 * GIB)
    assert result.estimate_source == "measured_peak"


@pytest.mark.parametrize("alias", ["qwen-image", "qwen-image-2.1"])
def test_image_fit_uses_admission_charge_instead_of_weights(alias):
    kwargs = dict(ram_bytes=16 * GIB, recommended_working_set_bytes=int(10.67 * GIB))
    result = preflight.catalog_memory_fit(alias, **kwargs)
    assert result.fits is True
    assert result.estimate_bytes == 6 * GIB
    assert result.estimate_source == "measured_peak"
    assert preflight.catalog_memory_refusal(alias, cached=False, **kwargs) is None


def test_image_bf16_uses_admission_charge():
    result = preflight.catalog_memory_fit(
        "qwen-image-2.1-bf16", ram_bytes=64 * GIB, recommended_working_set_bytes=None
    )
    assert result.estimate_bytes == 28 * GIB
    assert result.estimate_source == "measured_peak"


def test_image_at_floor_refuses_when_weights_exceed_gpu_budget(monkeypatch):
    alias = "flux-schnell"
    weights = 20 * GIB
    monkeypatch.setattr("rapid_mlx.model_sizes.size_bytes", lambda _: weights)
    result = preflight.catalog_memory_fit(
        alias, ram_bytes=32 * GIB, recommended_working_set_bytes=8 * GIB
    )
    assert (result.fits, result.fit_basis) == (False, "memory_floor")
    refusal = preflight.catalog_memory_refusal(
        alias, ram_bytes=32 * GIB, recommended_working_set_bytes=8 * GIB, cached=False
    )
    assert refusal.kind == "gpu_budget"
    assert refusal.footprint_bytes == result.estimate_bytes


def test_image_at_floor_refuses_when_admission_charge_exceeds_gpu_budget(monkeypatch):
    alias = "flux-schnell"
    estimate = 20 * GIB
    monkeypatch.setattr("rapid_mlx.model_sizes.size_bytes", lambda _: 4 * GIB)
    monkeypatch.setattr(
        "rapid_mlx.runtime.resident_models.measured_image_estimate_bytes",
        lambda _: estimate,
    )
    monkeypatch.setattr(
        "rapid_mlx.recommendations.recommendation_footprint_gb", lambda _: None
    )
    result = preflight.catalog_memory_fit(
        alias, ram_bytes=32 * GIB, recommended_working_set_bytes=12 * GIB
    )
    assert result.estimate_bytes == estimate
    assert result.fits is False
    refusal = preflight.catalog_memory_refusal(
        alias, ram_bytes=32 * GIB, recommended_working_set_bytes=12 * GIB, cached=False
    )
    assert refusal.kind == "gpu_budget"
    assert refusal.footprint_bytes == estimate


def test_image_below_floor_refusal():
    refusal = preflight.catalog_memory_refusal(
        "qwen-image-edit",
        ram_bytes=32 * GIB,
        recommended_working_set_bytes=24 * GIB,
        cached=False,
    )
    assert refusal.kind == "memory_floor"


@pytest.mark.parametrize(
    "estimate_bytes, budget_bytes",
    [(20 * GIB, None), (None, 12 * GIB)],
    ids=["missing-budget", "missing-footprint"],
)
def test_catalog_refusal_passes_when_gpu_fit_details_missing(
    estimate_bytes, budget_bytes
):
    result = preflight.CatalogMemoryFit(
        estimate_bytes=estimate_bytes,
        estimate_source="unknown",
        budget_bytes=budget_bytes,
        ram_bytes=16 * GIB,
        fits=False,
        fit_basis="gpu_budget",
        floor_gib=None,
    )
    with patch.object(preflight, "catalog_memory_fit", return_value=result) as fit:
        refusal = preflight.catalog_memory_refusal(
            "qwen3.8-27b-4bit",
            ram_bytes=16 * GIB,
            recommended_working_set_bytes=12 * GIB,
            cached=False,
        )
    assert refusal is None
    fit.assert_called_once_with(
        "qwen3.8-27b-4bit",
        ram_bytes=16 * GIB,
        recommended_working_set_bytes=12 * GIB,
    )


def test_catalog_refusal_swallows_fit_exception():
    with patch.object(
        preflight, "catalog_memory_fit", side_effect=RuntimeError("fit failed")
    ) as fit:
        refusal = preflight.catalog_memory_refusal(
            "qwen3.8-27b-4bit",
            ram_bytes=16 * GIB,
            recommended_working_set_bytes=12 * GIB,
            cached=False,
        )
    assert refusal is None
    fit.assert_called_once_with(
        "qwen3.8-27b-4bit",
        ram_bytes=16 * GIB,
        recommended_working_set_bytes=12 * GIB,
    )


@pytest.mark.parametrize("alias", ["diffusion-gemma-26b-8bit", "qwen-image-edit"])
def test_non_text_refuses_before_download(alias, monkeypatch, capsys):
    monkeypatch.setattr(
        preflight, "_mlx_recommended_working_set_bytes", lambda: 24 * GIB
    )
    with pytest.raises(SystemExit):
        run(alias, 32, monkeypatch)
    output = capsys.readouterr().err
    assert "Nothing was downloaded" in output
    assert "proceeding anyway" not in output


def test_cached_mflux_snapshot_keeps_floor_warning(tmp_path, monkeypatch):
    import huggingface_hub.file_download as file_download

    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS

    profile = resolve_profile("qwen-image-edit")
    repo = profile.hf_path
    repo_root = tmp_path / f"models--{repo.replace('/', '--')}"
    revision = IMAGE_MODEL_REVISIONS[repo]
    snap = repo_root / "snapshots" / revision
    (snap / "tokenizer").mkdir(parents=True)
    (snap / "tokenizer" / "tokenizer.json").write_text("{}")
    for component in ("transformer", "text_encoder", "vae"):
        directory = snap / component
        directory.mkdir()
        (directory / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}}'
        )
        for shard in ("0.safetensors", "1.safetensors"):
            (directory / shard).write_bytes(b"weights")
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text(revision)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(file_download, "HF_HUB_CACHE", str(tmp_path), raising=False)
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(
        preflight, "mlx_recommended_working_set_bytes", lambda: 48 * GIB
    )
    preflight._check_catalog_memory_fit(
        SimpleNamespace(command="serve", model="qwen-image-edit")
    )


def test_cached_image_keeps_floor_warning(monkeypatch, capsys):
    from rapid_mlx.cli import _check_alias_min_memory

    monkeypatch.setattr("rapid_mlx.cli._cache_runnability", lambda _: True)
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(total=32 * GIB)
    )
    run("qwen-image-edit", 32, monkeypatch)
    _check_alias_min_memory("qwen-image-edit")
    assert "proceeding anyway" in capsys.readouterr().out


def test_cached_alias_below_enforced_floor_exits(monkeypatch):
    monkeypatch.setattr("rapid_mlx._download_gate.is_repo_cached", lambda _: True)
    monkeypatch.setattr(preflight, "physical_ram_bytes", lambda: 36 * GIB)
    args = SimpleNamespace(command="serve", model="qwen3.8-27b-tensorfold")
    with pytest.raises(SystemExit):
        preflight._check_catalog_memory_fit(args)


@pytest.mark.parametrize("ram, fits", [(36, False), (48, True), (64, True)])
def test_catalog_fit_enforced_floor(ram, fits):
    result = preflight.catalog_memory_fit(
        "qwen3.8-27b-tensorfold",
        ram_bytes=ram * GIB,
        recommended_working_set_bytes=None,
    )
    assert (result.fits, result.fit_basis) == (fits, "memory_floor")


@pytest.mark.parametrize("raises", [False, True])
def test_catalog_fit_unknown_profile(monkeypatch, raises):
    if raises:

        def fail(_alias):
            raise RuntimeError("profile lookup failed")

        monkeypatch.setattr("rapid_mlx.model_aliases.resolve_profile", fail)

    result = preflight.catalog_memory_fit(
        "unknown-alias", ram_bytes=8 * GIB, recommended_working_set_bytes=6 * GIB
    )
    assert result.estimate_bytes is None
    assert result.fits is None
    assert result.fit_basis == "unknown"


def test_offline_guard_records_swallowed_lookup(offline_preflight):
    from huggingface_hub import model_info

    try:
        model_info("example", revision="main")
    except Exception:
        pass
    assert offline_preflight.network_calls == [
        ("model_info", ("example",), {"revision": "main"})
    ]
    offline_preflight.network_calls.clear()
