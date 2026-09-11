from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "repack_deepseek_v41_native.py"
    spec = importlib.util.spec_from_file_location("repack_deepseek_v41_native", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repack = _load_module()


def _fixture(tmp_path: Path, experts: int = 4) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    tensors: dict[str, np.ndarray] = {
        "layers.0.ffn.gate.weight": np.arange(experts * 3, dtype=np.float32).reshape(
            experts, 3
        ),
        "layers.0.ffn.gate.bias": np.arange(experts, dtype=np.float32),
        "layers.0.ffn.gate.bias_vl": np.arange(experts, dtype=np.float32) + 10,
        "layers.0.attn.wq_a.weight": np.arange(8, dtype=np.uint32).reshape(2, 4),
        "layers.0.attn.wq_a.scales": np.ones((2, 1), dtype=np.float32),
        "layers.0.attn.wq_a.biases": np.zeros((2, 1), dtype=np.float32),
        "mtp.0.norm.weight": np.ones((2,), dtype=np.float32),
        "vision.block.weight": np.ones((2,), dtype=np.float32),
    }
    for expert in range(experts):
        for projection in ("w1", "w2", "w3"):
            prefix = f"layers.0.ffn.experts.{expert}.{projection}"
            value = 100 * expert + 10 * int(projection[-1])
            tensors[f"{prefix}.weight"] = np.full((2, 2), value, dtype=np.uint32)
            tensors[f"{prefix}.scales"] = np.full((2, 1), value + 1, dtype=np.float32)
            tensors[f"{prefix}.biases"] = np.full((2, 1), value + 2, dtype=np.float32)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard for name in tensors},
            }
        )
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "text_config": {
                    "num_hidden_layers": 1,
                    "n_routed_experts": experts,
                },
                "quantization": {
                    "bits": 2,
                    "group_size": 64,
                    "mode": "affine",
                },
            }
        )
    )
    return root


def test_repack_stacks_experts_byte_exactly(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    checkpoint = repack.CheckpointIndex(source)
    selection = {0: tuple(range(4))}
    groups = repack.build_plans(
        checkpoint,
        layer_count=1,
        expert_count=4,
        selection=selection,
    )
    output = tmp_path / "layer.safetensors"
    repack.write_safetensors(output, groups["layer-00"])
    tensors = load_file(output)

    packed = tensors["layers.0.ffn.experts.gate_proj.weight"]
    assert packed.shape == (4, 2, 2)
    for expert in range(4):
        np.testing.assert_array_equal(packed[expert], 100 * expert + 10)
    assert "mtp.0.norm.weight" not in tensors
    assert "vision.block.weight" not in tensors


def test_repack_pruning_subsets_experts_and_router_together(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    checkpoint = repack.CheckpointIndex(source)
    selection = {0: (1, 3)}
    groups = repack.build_plans(
        checkpoint,
        layer_count=1,
        expert_count=4,
        selection=selection,
    )
    output = tmp_path / "pruned.safetensors"
    repack.write_safetensors(output, groups["layer-00"])
    tensors = load_file(output)

    np.testing.assert_array_equal(
        tensors["layers.0.ffn.experts.up_proj.weight"][:, 0, 0],
        np.array([130, 330], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        tensors["layers.0.ffn.gate.weight"],
        np.array([[3, 4, 5], [9, 10, 11]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tensors["layers.0.ffn.gate.bias_vl"],
        np.array([11, 13], dtype=np.float32),
    )


def test_module_map_tracks_quantized_native_names(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    groups = repack.build_plans(
        repack.CheckpointIndex(source),
        layer_count=1,
        expert_count=4,
        selection={0: tuple(range(4))},
    )
    modules = repack._module_map(groups, bits=2, group_size=64)

    assert modules["layers.0.ffn.experts.down_proj"] == {
        "bits": 2,
        "group_size": 64,
    }
    assert modules["layers.0.attn.wq_a"] == {"bits": 2, "group_size": 64}


def test_pruning_requires_calibrated_saliency() -> None:
    try:
        repack._selection_from_saliency(None, layers=1, experts=4, keep=2)
    except ValueError as error:
        assert "--saliency is required" in str(error)
    else:
        raise AssertionError("uncalibrated pruning must be rejected")


def test_selection_prefers_cross_half_robust_saliency(tmp_path: Path) -> None:
    path = tmp_path / "saliency.npz"
    np.savez(
        path,
        saliency=np.array([[100.0, 90.0, 80.0, 70.0]]),
        robust_saliency=np.array([[0.1, 0.2, 0.9, 0.8]]),
    )

    selected = repack._selection_from_saliency(path, layers=1, experts=4, keep=2)

    assert selected == {0: (2, 3)}


def test_generated_model_card_refuses_unqualified_publication(tmp_path: Path) -> None:
    repack._write_model_card(
        tmp_path,
        {"original_experts": 384, "kept_experts": 336},
        tensor_bytes=212_930_051_680,
    )

    card = (tmp_path / "README.md").read_text()
    assert "not product-qualified" in card
    assert "must not be published or cataloged" in card
    assert "336 of 384" in card
