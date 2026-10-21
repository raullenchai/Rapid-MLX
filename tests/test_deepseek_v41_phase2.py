from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deepseek_v41_rd_build import build, plan
from scripts.repack_deepseek_v41_native import CheckpointIndex


@pytest.fixture
def raw_source(tmp_path):
    pytest.importorskip("mlx.core")
    from tests.test_deepseek_v41_quant_error import tiny_source

    root = tmp_path / "source"
    c = tiny_source(root)
    c["text_config"]["num_experts_per_tok"] = 1
    (root / "config.json").write_text(json.dumps(c))
    return root, c


def allocation():
    return {
        "layers": {
            "0": {
                "kept_expert_ids": [1],
                "projection_bits": {"gate_proj": 2, "up_proj": 2, "down_proj": 3},
            }
        }
    }


@pytest.mark.requires_mlx
def test_dry_run_no_bodies_or_writes_and_policy(raw_source, tmp_path, monkeypatch):
    from scripts import deepseek_v41_expert_quant_error as error

    source, c = raw_source
    monkeypatch.setattr(
        error, "read_tensor", lambda *_: pytest.fail("dry run read tensor body")
    )
    result = build(source, tmp_path / "build", tmp_path / "shared", allocation(), True)
    assert not (tmp_path / "build").exists() and not (tmp_path / "shared").exists()
    entries, modules, _ = plan(CheckpointIndex(source), c, allocation())
    assert modules["layers.0.ffn.experts.down_proj"]["bits"] == 3
    assert "head" not in modules
    assert modules["layers.0.ffn.shared_experts.w1"]["bits"] == 8
    assert result["components"]["router"] == 136
    assert result["components"]["engram_ssd"] == 2 * 256 * (0.5 + 4 / 32)
    assert all(e.dtype == "BF16" for e in entries if e.name == "norm.weight")


@pytest.mark.requires_mlx
def test_streamed_conversion_actual_bytes_router_and_shared_inodes(
    raw_source, tmp_path
):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    source, _ = raw_source
    reports = []
    for name in ("a", "b"):
        reports.append(
            build(source, tmp_path / name, tmp_path / "shared", allocation())
        )
        assert (
            sum(f.stat().st_size for f in (tmp_path / name).glob("*.safetensors"))
            == reports[-1]["total_safetensors_file_bytes"]
        )
    engram_a = tmp_path / "a" / "model-engram-0-weight.safetensors"
    engram_b = tmp_path / "b" / "model-engram-0-weight.safetensors"
    assert engram_a.stat().st_ino == engram_b.stat().st_ino
    assert not engram_a.is_symlink()
    from rapid_mlx.models.deepseek_v41_native.load import resolve_indexed_shard

    assert resolve_indexed_shard(str(tmp_path / "a"), engram_a.name) == str(engram_a)
    weights = mx.load(str(tmp_path / "a" / "model-layer-0.safetensors"))
    original = mx.load(str(source / "tiny.safetensors"))
    for suffix in ("weight", "bias", "bias_vl"):
        name = "layers.0.ffn.gate." + suffix
        assert np.array_equal(
            np.array(weights[name].astype(mx.float32)),
            np.array(original[name][[1]].astype(mx.float32)),
        )
    restored = mx.dequantize(
        weights["layers.0.ffn.experts.down_proj.weight"],
        weights["layers.0.ffn.experts.down_proj.scales"],
        weights["layers.0.ffn.experts.down_proj.biases"],
        bits=3,
        group_size=64,
    )
    assert restored.shape == (1, 64, 64)
    shared_file = next((tmp_path / "shared").rglob("*.safetensors"))
    with shared_file.open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="size mismatch"):
        build(source, tmp_path / "c", tmp_path / "shared", allocation())


@pytest.mark.requires_mlx
def test_builder_rejects_per_expert_widths_bad_ids_and_overbudget(raw_source, tmp_path):
    source, c = raw_source
    for ids in ([1, 1], [2], [], [1, 0]):
        a = allocation()
        a["layers"]["0"]["kept_expert_ids"] = ids
        with pytest.raises(ValueError):
            plan(CheckpointIndex(source), c, a)
    a = allocation()
    a["layers"]["0"]["projection_bits"]["down_proj"] = [2]
    with pytest.raises(ValueError):
        plan(CheckpointIndex(source), c, a)
    a = allocation()
    a["budget_bytes"] = 1
    with pytest.raises(ValueError, match="exceeds"):
        build(source, tmp_path / "bad", tmp_path / "shared", a, True)


@pytest.mark.requires_mlx
def test_strict_native_reload_per_layer_counts_and_projection_bits(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.load import load, quant_predicate
    from rapid_mlx.models.deepseek_v41_native.model import Model

    mx.set_default_device(mx.cpu)
    config = {
        "hidden_size": 64,
        "vocab_size": 32,
        "num_hidden_layers": 2,
        "moe_intermediate_size": 64,
        "n_routed_experts": 3,
        "experts_per_layer": [3, 2],
        "num_experts_per_tok": 1,
        "num_attention_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 64,
        "o_lora_rank": 64,
        "o_groups": 1,
        "hc_mult": 1,
        "compress_ratios": [0, 0],
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    modules = {
        f"layers.{i}.ffn.experts.{p}": {"bits": b, "group_size": 64}
        for i, bs in enumerate(((2, 2, 3), (3, 4, 2)))
        for p, b in zip(("gate_proj", "up_proj", "down_proj"), bs)
    }
    nn.quantize(
        model,
        bits=8,
        group_size=64,
        class_predicate=quant_predicate(64, 8, None, modules),
    )
    config["quantization"] = {"bits": 8, "group_size": 64, "modules": modules}
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    restored, _ = load(str(tmp_path), lazy=True, strict=True)
    assert [layer.ffn.gate.weight.shape[0] for layer in restored.layers] == [3, 2]
    for i in range(2):
        experts = restored.layers[i].ffn.experts
        original = model.layers[i].ffn.experts
        for n in (1, 70):  # decode plus sorted prefill gather/scatter
            x = mx.ones((n, 64))
            ids = mx.zeros((n, 1), dtype=mx.int32)
            assert mx.allclose(experts(x, ids), original(x, ids)).item()


@pytest.mark.requires_mlx
def test_qmv_guard_preserves_generic_projection_and_prefill_paths(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from rapid_mlx.models.deepseek_v41_native import serving
    from rapid_mlx.models.deepseek_v41_native.moe import ClampedSwiGLU

    mx.set_default_device(mx.cpu)
    experts = []
    for bits in (2, 3, 4):
        module = SwitchGLU(64, 64, 2, activation=ClampedSwiGLU(10))
        module.set_dtype(mx.bfloat16)
        nn.quantize(module, group_size=64, bits=bits)
        experts.append(module)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(experts=e)) for e in experts]
    )
    assert serving.install_target_qmv(model) == 1
    assert experts[1].__class__ is SwitchGLU and experts[2].__class__ is SwitchGLU
    calls = []

    def fake_qmv(module, x, indices):
        calls.append(module.bits)
        return module(x[:, None, None, :], indices).squeeze((-2, -3))

    monkeypatch.setattr(serving, "affine2_route_down_qmv", fake_qmv)
    x = mx.ones((1, 64))
    ids = mx.zeros((1, 1), dtype=mx.int32)
    actual = experts[0](x, ids)
    expected = SwitchGLU.__call__(experts[0], x, ids)
    assert mx.allclose(actual, expected).item() and calls == [2]
    # Replacing down precision after installation must still safely fall back.
    experts[0].down_proj = copy.deepcopy(experts[1].down_proj)
    assert mx.allclose(
        experts[0](x, ids), SwitchGLU.__call__(experts[0], x, ids)
    ).item()
    assert calls == [2]
    for e in experts:
        output = e(mx.ones((70, 64)), mx.zeros((70, 1), dtype=mx.int32))
        assert output.shape == (70, 1, 64)
    assert calls == [2]


@pytest.mark.requires_mlx
def test_invalid_layer_counts_rejected():
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs

    for counts in ((4,), (4, 0), (4, 5)):
        with pytest.raises(ValueError, match="experts_per_layer"):
            ModelArgs(
                n_layers=2,
                n_routed_experts=4,
                n_activated_experts=1,
                experts_per_layer=counts,
            )


def test_layer_mask_schema_for_phase2_and_bad_ids():
    from scripts.evaluate_deepseek_v41_rd import validate_allocation

    a = allocation()
    validate_allocation(a, 1, 2, 1)
    a["layers"]["0"]["kept_expert_ids"] = [1, 1]
    with pytest.raises(ValueError):
        validate_allocation(a, 1, 2, 1)


def test_corpus_manifest_covers_heldout_and_rejects_incomplete_exports():
    from scripts.deepseek_v41_rd_corpus_prepare import prepare

    manifest = json.loads(
        (
            Path(__file__).parents[1] / "scripts/deepseek_v41_rd_corpus_manifest.json"
        ).read_text()
    )
    specs = {s["name"]: s for s in manifest["slices"]}
    assert specs["heldout_cybersecurity"]["calibration_tokens"] == 0
    with pytest.raises(ValueError, match="cover exactly"):
        prepare(manifest, {"slices": {}}, {"slices": {}})


@pytest.mark.requires_mlx
def test_projection_error_probes_are_measured_not_joint_divided_by_three():
    import mlx.core as mx

    from scripts.deepseek_v41_expert_quant_error import projection_quant_error

    mx.set_default_device(mx.cpu)
    weights = [mx.random.normal((64, 64)) for _ in range(3)]
    result = projection_quant_error(weights, mx.random.normal((4, 64)))
    assert result.shape == (3, 3) and np.isfinite(result).all() and np.all(result >= 0)
    assert not np.array_equal(result[0], result[2])


@pytest.mark.requires_mlx
def test_engram_independent_group_size_replayed_by_loader(tmp_path, monkeypatch):
    import rapid_mlx.models.deepseek_v41_native.load as native_load
    from tests.test_deepseek_v41_native_load import (
        _patch_minimal_offload_loader,
        _write_offload_fixture,
    )

    _write_offload_fixture(tmp_path)
    _patch_minimal_offload_loader(monkeypatch)
    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization"] = {
        "engram_group_size": 32,
        "group_size": 64,
        "engram_bits": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model, _ = native_load.load(str(tmp_path), lazy=True, engram_ssd_offload=True)
    assert model.layers[0].engram.embed.group_size == 32


@pytest.mark.requires_mlx
def test_dspark_dense_io_contract_rejects_partial_affine_triples(tmp_path, monkeypatch):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native import dspark
    from tests.test_deepseek_v41_dspark import _config

    (tmp_path / "config.json").write_text(json.dumps(_config()))
    name = "mtp.2.markov_head.head"
    arrays = {
        name + ".weight": mx.ones((2, 2), mx.bfloat16),
        name + ".scales": mx.ones((2, 1), mx.bfloat16),
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model.safetensors" for n in arrays}})
    )
    with pytest.raises(ValueError, match="incomplete affine triple"):
        dspark.DSparkWeights(tmp_path)
    # Isolate dense I/O contract with the existing strict loader, no fake tensors.
    arrays.pop(name + ".scales")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "model.safetensors" for n in arrays}})
    )
    monkeypatch.setattr(dspark, "_required_tensors", lambda _: {name + ".weight"})
    weights = dspark.DSparkWeights(tmp_path)
    assert weights.linear(name, mx.ones((1, 2), mx.bfloat16)).shape == (1, 2)


@pytest.mark.requires_mlx
def test_raw_fp4_build_strict_reload_full_tiny_native_model(tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.load import load
    from rapid_mlx.models.deepseek_v41_native.model import Model

    mx.set_default_device(mx.cpu)
    config = {
        "model_type": "deepseek_v41",
        "hidden_size": 64,
        "vocab_size": 32,
        "num_hidden_layers": 1,
        "moe_intermediate_size": 64,
        "n_routed_experts": 3,
        "num_experts_per_tok": 1,
        "num_attention_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 64,
        "o_lora_rank": 64,
        "o_groups": 1,
        "hc_mult": 1,
        "compress_ratios": [0],
        "quantization_config": {"expert_dtype": "fp4"},
    }
    original = Model(ModelArgs.from_dict(config))
    raw = {}
    reverse = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for name, value in tree_flatten(original.parameters()):
        if ".ffn.experts." in name:
            proj = reverse[name.split(".")[-2]]
            for expert in range(3):
                base = f"layers.0.ffn.experts.{expert}.{proj}"
                raw[base + ".weight"] = mx.full((64, 32), expert + 1, mx.uint8)
                raw[base + ".scale"] = mx.full((64, 2), 124, mx.uint8)
        else:
            if name.endswith(".wo_a.weight"):
                value = value.reshape(-1, value.shape[-1])
            raw[name] = value.astype(mx.bfloat16)
    # Exercise real FP8 block dequantization on a fixed shared projection.
    raw["layers.0.ffn.shared_experts.w1.weight"] = mx.full((64, 64), 56, mx.uint8)
    raw["layers.0.ffn.shared_experts.w1.scale"] = mx.full((2, 2), 124, mx.uint8)
    source = tmp_path / "raw"
    source.mkdir()
    mx.save_safetensors(str(source / "source.safetensors"), raw)
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "source.safetensors" for n in raw}})
    )
    a = allocation()
    a["layers"]["0"]["kept_expert_ids"] = [0, 2]
    report = build(source, tmp_path / "packed", tmp_path / "shared", a)
    model, _ = load(str(tmp_path / "packed"), lazy=True, strict=True)
    assert model.layers[0].ffn.gate.weight.shape == (2, 64)
    assert model.layers[0].ffn.experts.down_proj.bits == 3
    assert model.layers[0].ffn.shared_experts.w1.bits == 8
    y = model(mx.array([[1, 2]]), model.make_cache(max_seq_len=4))
    mx.eval(y)
    assert y.shape == (1, 2, 32) and mx.all(mx.isfinite(y)).item()
    assert (
        sum(f.stat().st_size for f in (tmp_path / "packed").glob("*.safetensors"))
        == report["total_safetensors_file_bytes"]
    )
    # A reference plan must retain every expert and disable fixed quantization.
    a["reference"] = True
    with pytest.raises(ValueError, match="reference must keep"):
        plan(CheckpointIndex(source), config, a)
    a["layers"]["0"] = {"kept_expert_ids": [0, 1, 2], "bits": 16}
    entries, modules, _ = plan(CheckpointIndex(source), config, a)
    assert not modules
    assert all(e.bits is None for e in entries)
