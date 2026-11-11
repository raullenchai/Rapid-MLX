"""Measured MTP depth cannot escape its model, hardware or software scope."""

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice.mtp import resolve_max_depth

ENTRY = next(
    e
    for e in lattice.builtin_profile().cells
    if e["id"] == "mtp-depth-17-max-36g-qwen35-moe"
)
INFO = dict(
    device_name="Apple M5 Max", architecture="applegpu_g17s", memory_size=36 * 2**30
)


def context():
    return lattice.runtime_context(
        INFO,
        model=lattice.ModelProfile(
            "qwen3_5_moe_text", *lattice.QWEN35_AFFINE4, quant_mode="affine"
        ),
        library_version="0.32.3",
        macos="27.0.1",
        mode="mtp",
        has_mtp_head=True,
    )


def expert(bits=4, group_size=64, mode="affine"):
    return SimpleNamespace(weight=object(), bits=bits, group_size=group_size, mode=mode)


def loaded_model(modules):
    return SimpleNamespace(
        mtp=object(),
        mtp_forward=lambda: None,
        make_mtp_cache=lambda: [],
        args=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            hidden_size=2048,
            moe_intermediate_size=512,
            num_experts=256,
            num_experts_per_tok=8,
            quantization=dict(bits=4, group_size=64),
        ),
        named_modules=lambda: modules,
    )


def test_qualified_context_and_overrides():
    ctx = context()
    assert lattice.resolve("mtp_max_depth", ctx, environ={}) == lattice.Decision(
        4, "builtin:" + ENTRY["id"], profile_version=lattice.builtin_profile().version
    )
    assert lattice.resolve(
        "mtp_max_depth", ctx, environ={"RAPID_MLX_MTP_MAX_DEPTH": "1"}
    ) == lattice.Decision(1, "env")
    assert (
        lattice.resolve(
            "mtp_max_depth", ctx, environ={"RAPID_MLX_MTP_MAX_DEPTH": "bad"}
        ).value
        == 4
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(family="qwen3_5"),
        dict(hidden_size=4096),
        dict(intermediate_size=1024),
        dict(num_experts=128),
        dict(top_k=4),
        dict(bits=8),
        dict(group_size=32),
        dict(quant_mode="mxfp4"),
    ],
)
def test_other_models_remain_conservative(changes):
    ctx = context()
    assert (
        lattice.resolve(
            "mtp_max_depth",
            replace(ctx, model=replace(ctx.model, **changes)),
            environ={},
        ).value
        == 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(gpu_generation=15),
        dict(tier="Pro"),
        dict(memory_size=None),
        dict(memory_size=32 * 2**30),
        dict(memory_size=64 * 2**30),
    ],
)
def test_other_or_unknown_hardware_remains_conservative(changes):
    ctx = context()
    assert (
        lattice.resolve(
            "mtp_max_depth",
            replace(ctx, hardware=replace(ctx.hardware, **changes)),
            environ={},
        ).value
        == 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(mlx="0.32.4"),
        dict(macos="26.5.2"),
        dict(mode="plain"),
        dict(batch=2),
        dict(has_mtp_head=False),
    ],
)
def test_other_workloads_and_software_remain_conservative(changes):
    assert (
        lattice.resolve(
            "mtp_max_depth", replace(context(), **changes), environ={}
        ).value
        == 1
    )


def test_exact_features_include_pinned_cores_if_provided(monkeypatch):
    from rapid_mlx.lattice.defaults import Profile

    hardware = [dict(ENTRY["hardware"][0], gpu_cores=[40, 40])]
    monkeypatch.setattr(
        lattice,
        "builtin_profile",
        lambda: Profile("test", (dict(ENTRY, hardware=hardware),)),
    )
    ctx = context()
    assert lattice.resolve("mtp_max_depth", ctx, environ={}).value == 1
    assert (
        lattice.resolve(
            "mtp_max_depth",
            replace(ctx, hardware=replace(ctx.hardware, gpu_cores=40)),
            environ={},
        ).value
        == 4
    )


def test_actual_experts_supply_format_and_ignore_router(monkeypatch):
    import mlx.core as mx

    from rapid_mlx.lattice import mtp

    monkeypatch.setattr(mx, "device_info", lambda: INFO, raising=False)
    monkeypatch.setattr(
        mtp, "runtime_context", lambda info, **kw: replace(context(), model=kw["model"])
    )
    monkeypatch.delenv("RAPID_MLX_MTP_MAX_DEPTH", raising=False)
    model = loaded_model(
        [
            ("model.layers.0.mlp.switch_mlp.gate_up_proj", expert()),
            ("model.layers.0.mlp.switch_mlp.down_proj", expert()),
            ("model.layers.0.mlp.switch_mlp.activation", SimpleNamespace()),
            ("mtp.layers.0.mlp.switch_mlp.gate_proj", expert()),
            ("model.layers.0.mlp.shared_expert_gate", expert(bits=8)),
        ]
    )
    assert resolve_max_depth(model, 1, explicit=False) == 4
    assert resolve_max_depth(model, 2, explicit=True) == 2


@pytest.mark.parametrize(
    "modules",
    [
        [],
        [("other", expert())],
        [("model.layers.0.mlp.switch_mlp.up_proj", expert(bits=8))],
        [("model.layers.0.mlp.switch_mlp.up_proj", expert(group_size=32))],
        [("model.layers.0.mlp.switch_mlp.up_proj", expert(bits=None))],
        [("model.layers.0.mlp.switch_mlp.up_proj", expert(group_size=None))],
        [("model.layers.0.mlp.switch_mlp.up_proj", expert(mode="mxfp4"))],
        [
            ("model.layers.0.mlp.switch_mlp.up_proj", expert()),
            ("model.layers.0.mlp.switch_mlp.down_proj", expert(bits=8)),
        ],
    ],
)
def test_declared_quantization_cannot_qualify_unknown_or_mixed_weights(
    monkeypatch, modules
):
    import mlx.core as mx

    from rapid_mlx.lattice import mtp

    monkeypatch.setattr(mx, "device_info", lambda: INFO, raising=False)
    monkeypatch.setattr(
        mtp, "runtime_context", lambda info, **kw: replace(context(), model=kw["model"])
    )
    monkeypatch.delenv("RAPID_MLX_MTP_MAX_DEPTH", raising=False)
    assert resolve_max_depth(loaded_model(modules), 1, explicit=False) == 1


@pytest.mark.parametrize("broken", [False, True])
def test_missing_module_metadata_is_conservative(monkeypatch, broken):
    import mlx.core as mx

    from rapid_mlx.lattice import mtp

    monkeypatch.setattr(mx, "device_info", lambda: INFO, raising=False)
    monkeypatch.setattr(
        mtp, "runtime_context", lambda info, **kw: replace(context(), model=kw["model"])
    )
    monkeypatch.delenv("RAPID_MLX_MTP_MAX_DEPTH", raising=False)
    model = loaded_model([])
    if broken:

        def unavailable():
            raise RuntimeError("metadata unavailable")

        model.named_modules = unavailable
    else:
        model.named_modules = None
    assert resolve_max_depth(model, 1, explicit=False) == 1


@pytest.mark.parametrize(
    "cell",
    [e for e in lattice.builtin_profile().cells if e["id"].startswith("mtp-depth-")],
)
def test_cell_has_speed_and_paired_quality_evidence(cell):
    assert Path(cell["provenance"]["evidence"]).is_file()
    before, after = cell["provenance"]["mtp_tps"]
    assert after / before >= 1.02
    assert (
        cell["provenance"]["mtp_ranges_tps"][0][1]
        < cell["provenance"]["mtp_ranges_tps"][1][0]
    )
    assert cell["provenance"]["repetitions"]["mtp"] >= 3
    q = cell["provenance"]["quality"]
    n = q["n"]
    assert n >= 100 and len(q["questions_sha256"]) == 64
    for reference, pair in q["paired"].items():
        wins, losses = pair["wins"], pair["losses"]
        d = wins + losses
        p = (
            min(
                1.0,
                2 * sum(math.comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d,
            )
            if d
            else 1.0
        )
        assert pair["mcnemar_exact_two_sided_p"] == p >= 0.05
        assert pair["delta_pp"] == 100 * (wins - losses) / n
        assert abs(pair["delta_pp"]) <= 3
        assert (
            q["correct"][f"cap{cell['knobs']['mtp_max_depth']}"]
            - q["correct"][reference]
            == wins - losses
        )


def m3_context():
    return lattice.runtime_context(
        {
            "device_name": "Apple M3 Ultra",
            "architecture": "applegpu_g15d",
            "memory_size": 256 * 2**30,
        },
        model=context().model,
        library_version="0.32.4",
        macos="26.5.2",
        mode="mtp",
        has_mtp_head=True,
    )


def test_m3_measured_context_and_override():
    ctx = m3_context()
    cell = next(
        e
        for e in lattice.builtin_profile().cells
        if e["id"] == "mtp-depth-15-ultra-256g-qwen35-moe"
    )
    assert lattice.resolve("mtp_max_depth", ctx, environ={}) == lattice.Decision(
        cell["knobs"]["mtp_max_depth"],
        "builtin:" + cell["id"],
        profile_version=lattice.builtin_profile().version,
    )
    assert lattice.resolve(
        "mtp_max_depth", ctx, environ={"RAPID_MLX_MTP_MAX_DEPTH": "1"}
    ) == lattice.Decision(1, "env")


@pytest.mark.parametrize("memory", [None, 192 * 2**30, 512 * 2**30])
def test_m3_unmeasured_memory_does_not_borrow(memory):
    ctx = m3_context()
    assert (
        lattice.resolve(
            "mtp_max_depth",
            replace(ctx, hardware=replace(ctx.hardware, memory_size=memory)),
            environ={},
        ).value
        == 1
    )


@pytest.mark.parametrize(
    "profile",
    [
        lattice.ModelProfile(
            "qwen3_5_text", hidden_size=2560, bits=4, group_size=64, quant_mode="affine"
        ),
        lattice.ModelProfile(
            "qwen3_5_text", hidden_size=4096, bits=4, group_size=64, quant_mode="affine"
        ),
        lattice.ModelProfile(
            "qwen3_5_moe_text", 2048, 512, 256, 8, 8, 64, quant_mode="affine"
        ),
    ],
)
def test_m3_dense_controls_and_other_quantization_stay_at_one(profile):
    assert (
        lattice.resolve(
            "mtp_max_depth", replace(m3_context(), model=profile), environ={}
        ).value
        == 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(mlx="0.32.3"),
        dict(macos="27.0.1"),
        dict(mode="plain"),
        dict(batch=2),
        dict(has_mtp_head=False),
    ],
)
def test_m3_unmeasured_runtime_or_workload_stays_conservative(changes):
    assert (
        lattice.resolve(
            "mtp_max_depth", replace(m3_context(), **changes), environ={}
        ).value
        == 1
    )
