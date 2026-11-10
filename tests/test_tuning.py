"""Policy precedence, combination measurements, qualification and diagnostics."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from rapid_mlx import tuning

PAIR = ("qwen35_affine4_bank_views", "qwen35_residual_norm")


def context(chip="Apple M3 Ultra", **kwargs):
    return tuning.runtime_context(
        {"device_name": chip},
        model=tuning.ModelProfile("qwen35_moe", *tuning.QWEN35_AFFINE4),
        library_version=kwargs.pop("library_version", "0.32.4"),
        macos="26.5.2",
        **kwargs,
    )


@pytest.mark.parametrize(
    "name,key",
    [
        ("Apple M3 Ultra", "M3 Ultra"),
        ("Apple M4 Pro", "M4 Pro"),
        ("Apple M5 Max", "M5 Max"),
        ("Apple M1", "M1"),
        (" Apple M12 Ultra ", "M12 Ultra"),
        ("Apple M5 Extreme", "unknown"),
        ("Intel", "unknown"),
        (None, "unknown"),
    ],
)
def test_device_key(name, key):
    assert tuning.device_key({"device_name": name}) == key


@pytest.mark.parametrize("name", PAIR)
def test_precedence_and_fallback(monkeypatch, name):
    switch = tuning.KNOBS[name].env
    monkeypatch.delenv(switch, raising=False)
    assert tuning.resolve(name, context()).value
    assert not tuning.resolve(name, context("Apple M99 Max")).value
    assert not tuning.resolve(name, context(library_version="unmeasured")).value
    monkeypatch.setattr(tuning, "local_calibration", lambda c: {name: False})
    assert tuning.resolve(name, context()).source == "local-calibration"
    assert not tuning.resolve(name, context()).value
    monkeypatch.setenv(switch, "yes")
    assert tuning.resolve(name, context("unknown")).value
    monkeypatch.setenv(switch, "0")
    assert not tuning.resolve(name, context()).value
    monkeypatch.setenv(switch, "garbage")
    assert not tuning.resolve(name, context()).value


@pytest.mark.parametrize("name", PAIR)
@pytest.mark.parametrize(
    "change",
    [{"model": tuning.ModelProfile()}, {"width": 2}, {"mode": "other"}, {"batch": 2}],
)
def test_override_cannot_expand_scope(name, change):
    assert not tuning.resolve(
        name, replace(context(), **change), environ={tuning.KNOBS[name].env: "1"}
    ).value


def test_mtp_scope():
    ctx = context(mode="mtp")
    assert tuning.resolve(PAIR[0], ctx, environ={}).value
    assert not tuning.resolve(
        PAIR[1], ctx, environ={tuning.KNOBS[PAIR[1]].env: "1"}
    ).value


def test_combination_gate_and_nearest_fallback(monkeypatch):
    for name in PAIR:
        assert tuning.resolve(name, context("Apple M5 Max"), environ={}).value
        assert not tuning.resolve(name, context("Apple M4 Pro"), environ={}).value
    entry = next(e for e in tuning.MEASURED_DEFAULTS if e["id"] == "singleton-17-max")
    assert entry["repetitions"]["plain"] >= 3
    assert entry["plain_ranges_tps"][0][1] < entry["plain_ranges_tps"][1][0]
    assert entry["plain_tps"][1] / entry["plain_tps"][0] >= 1.02
    ctx = replace(
        context(),
        hardware=replace(context().hardware, gpu_cores=72, memory_size=192 * 2**30),
    )
    assert tuning.resolve(PAIR[0], ctx, environ={}).source == "table:singleton-15-ultra"
    closer = dict(
        tuning.MEASURED_DEFAULTS[0],
        id="closer",
        gpu_cores=72,
        memory_size=192 * 2**30,
        knobs={PAIR[0]: False},
    )
    monkeypatch.setattr(
        tuning, "MEASURED_DEFAULTS", [tuning.MEASURED_DEFAULTS[0], closer]
    )
    assert not tuning.resolve(PAIR[0], ctx, environ={}).value
    # Equidistant conflicts fall back conservatively, independent of order.
    monkeypatch.setattr(
        tuning, "MEASURED_DEFAULTS", [closer, dict(closer, knobs={PAIR[0]: True})]
    )
    assert tuning.resolve(PAIR[0], ctx, environ={}).source == "default"


@pytest.mark.parametrize(
    "name,arch,width",
    [("qsa_stage1", "applegpu_g14s", 64), ("qsa_indexed_splitk", "applegpu_g15d", 3)],
)
def test_capabilities_before_environment(name, arch, width):
    ctx = tuning.runtime_context(
        {"architecture": arch}, library_version="0.32.2", width=width
    )
    env = {tuning.KNOBS[name].env: "1"}
    assert tuning.resolve(name, ctx, environ=env).value
    assert not tuning.resolve(name, replace(ctx, mlx="0.32.4"), environ=env).value
    assert not tuning.resolve(
        name,
        replace(ctx, hardware=replace(ctx.hardware, architecture="applegpu_g17s")),
        environ=env,
    ).value


def test_preserved_qualified_choices():
    ctx = context("Apple M4 Pro", library_version="future", width=9)
    assert tuning.resolve("qwen35_affine4_rows", ctx).value == (2, 4)
    assert tuning.resolve("qwen35_affine4_rows", context()).value == (1, 1)
    for arch, units, probe in [
        ("applegpu_g14s", False, True),
        ("applegpu_g15d", False, False),
        ("applegpu_g17s", True, False),
        ("applegpu_g18s", True, False),
        ("unknown", False, True),
    ]:
        ctx = tuning.runtime_context({"architecture": arch})
        assert tuning.resolve("lane_tensor_units", ctx).value is units
        assert tuning.resolve("lane_simd_probe", ctx).value is probe
    ctx = replace(context(library_version="0.32.3"), macos="26.5.2")
    assert tuning.resolve("affine8_attention", ctx, environ={}).value
    assert not tuning.resolve(
        "affine8_attention",
        replace(ctx, macos="27.0"),
        environ={"RAPID_MLX_AFFINE8_ATTENTION": "1"},
    ).value


def test_loaded_profile_immutable_and_dump_private():
    config = {
        "_name_or_path": "/Users/private/model",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "layer_types": ["linear_attention", "full_attention"],
        },
        "quantization": {"bits": 4, "group_size": 64},
    }
    ctx = tuning.runtime_context(
        {
            "device_name": "Apple M3 Ultra",
            "architecture": "applegpu_g15d",
            "gpu_cores": 80,
            "memory_size": 256 * 2**30,
            "serial": "private-id",
        },
        config=config,
    )
    assert ctx.model.geometry == tuning.QWEN35_AFFINE4
    assert tuning.resolve(PAIR[0], replace(ctx, mlx="0.32.4"), environ={}).value
    assert ctx.model.attention_kinds == ("full_attention", "linear_attention")
    assert ctx.model.quant_mode == "affine"
    config["text_config"]["hidden_size"] = 9
    assert ctx.model.hidden_size == 2048
    with pytest.raises(FrozenInstanceError):
        ctx.width = 2
    payload = tuning.dump(ctx)
    assert isinstance(payload, dict)
    assert all(
        s not in json.dumps(payload)
        for s in ("/Users", "private-id", "Apple M3 Ultra", "_name_or_path")
    )


def test_import_does_not_load_mlx():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import rapid_mlx.tuning; assert 'mlx.core' not in sys.modules",
        ],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        check=True,
    )


def test_both_gates_consult_registry(monkeypatch):
    pytest.importorskip("mlx.core")
    from rapid_mlx import qwen35_residual_norm as residual
    from rapid_mlx.patches import qwen35_affine4 as affine

    calls = []

    def decline(name, info, **scope):
        calls.append((name, scope))
        return False

    monkeypatch.delenv("RAPID_MLX_QWEN35_RESIDUAL_NORM", raising=False)
    monkeypatch.setattr(affine, "enabled", decline)
    monkeypatch.setattr(residual, "enabled", decline)
    affine._bank_views_policy.cache_clear()
    assert not affine._singleton_bank_views_enabled()
    affine._bank_views_policy.cache_clear()
    assert residual.install_qwen35_residual_norm(object()) == 0
    assert [name for name, _ in calls] == list(PAIR)
    assert all(scope["width"] == 1 for _, scope in calls)


def test_missing_mlx_metadata_is_conservative(monkeypatch):
    from rapid_mlx.tuning import profiles

    def missing(name):
        raise profiles.PackageNotFoundError(name)

    profiles.mlx_version.cache_clear()
    monkeypatch.setattr(profiles, "version", missing)
    try:
        assert profiles.mlx_version() == "unknown"
        assert not tuning.resolve(
            PAIR[0], context(library_version="unknown"), environ={}
        ).value
    finally:
        profiles.mlx_version.cache_clear()


def test_runtime_facts_are_read_once(monkeypatch):
    from rapid_mlx.tuning import profiles

    calls = []
    profiles.macos_version.cache_clear()
    monkeypatch.setattr(
        profiles.platform, "mac_ver", lambda: (calls.append(1) or "26.5.2", (), "")
    )
    try:
        contexts = [
            tuning.runtime_context({"device_name": "Apple M3 Ultra"}) for _ in range(40)
        ]
        assert calls == [1]
        assert all(c is contexts[0] for c in contexts)
    finally:
        profiles.macos_version.cache_clear()
