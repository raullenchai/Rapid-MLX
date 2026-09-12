from __future__ import annotations

import json

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from vllm_mlx.models.deepseek_v41_native.dspark import (
    DSpark,
    DSparkWeights,
    Weights,
    _safe_shard,
    _validate_config,
    verify_greedy,
)


def test_runtime_provider_exposes_weights_contract() -> None:
    assert Weights is DSparkWeights


def _config():
    return {
        "dspark_block_size": 5,
        "dspark_n_routed_experts": 128,
        "dspark_noise_token_id": 3,
        "dspark_num_experts_per_tok": 3,
        "dspark_target_layer_ids": [7, 8, 9],
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 512,
        "hidden_size": 5120,
        "model_type": "deepseek_v4",
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "o_groups": 8,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-20,
        "rope_theta": 10000,
        "routed_scaling_factor": 1.5,
        "sliding_window": 8,
        "swiglu_limit": 10.0,
    }


def test_config_rejects_invalid_expert_topk() -> None:
    config = _config()
    config["dspark_num_experts_per_tok"] = 129
    with pytest.raises(ValueError, match="top-k"):
        _validate_config(config)


@pytest.mark.parametrize(
    "model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v41_text"]
)
def test_config_accepts_release_model_type_aliases(model_type) -> None:
    config = _config()
    config["model_type"] = model_type
    _validate_config(config)


def test_safe_shard_rejects_traversal_and_symlink(tmp_path) -> None:
    with pytest.raises(ValueError, match="basenames"):
        _safe_shard(tmp_path, "../model.safetensors")
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    link = tmp_path / "model.safetensors"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        _safe_shard(tmp_path, link.name)


def test_loader_rejects_non_string_index_shard(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.0.weight": 7}})
    )
    with pytest.raises(ValueError, match="shard names must be strings"):
        DSparkWeights(tmp_path)


def test_loader_rejects_index_shard_mismatch(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.0.weight": "model.safetensors"}})
    )
    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(mx, "load", lambda path: {"mtp.0.other": mx.zeros((1,))})
    with pytest.raises(ValueError, match="index/shard mismatch"):
        DSparkWeights(tmp_path)


def test_release_tensor_drops_both_backing_and_resident_references() -> None:
    weights = DSparkWeights.__new__(DSparkWeights)
    value = mx.zeros((8,), dtype=mx.float32)
    weights._arrays = {"mtp.0.weight": value}
    weights.entries = {"mtp.0.weight": {"shape": value.shape}}
    weights.resident = {"mtp.0.weight": value}
    weights.resident_bytes = value.nbytes

    weights.release_tensor("mtp.0.weight")

    assert weights._arrays == {}
    assert weights.entries == {}
    assert weights.resident == {}
    assert weights.resident_bytes == 0


def test_pin_mtp_materializes_each_weight_once(monkeypatch) -> None:
    value = mx.zeros((8,), dtype=mx.float32)
    weights = DSparkWeights.__new__(DSparkWeights)
    weights._arrays = {"mtp.0.weight": value, "embed.weight": value}
    weights.resident = {}
    weights.resident_bytes = 0
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: type("Memory", (), {"available": 10**12})()
    )
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 10**12},
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)

    assert weights.pin_mtp(reserve_gb=0) == value.nbytes
    assert weights.pin_mtp(reserve_gb=0) == value.nbytes
    assert weights.resident_bytes == value.nbytes


def test_dspark_initializes_one_window_per_contiguous_stage(monkeypatch) -> None:
    weights = type(
        "Weights",
        (),
        {
            "entries": {"mtp.0.weight": {}, "mtp.1.weight": {}},
            "pin_mtp": lambda self: None,
        },
    )()
    target = type("Target", (), {"w": weights, "c": {"dspark_block_size": 5}})()

    draft = DSpark(target)

    assert draft.position == -1
    assert draft.windows == {0: [], 1: []}


def test_verify_greedy_never_observes_rejected_token() -> None:
    observations = []
    advances = []

    def advance(token):
        advances.append(token)
        logits = mx.array([[0.0, 0.0, 1.0]])
        return logits, mx.array([token])

    output, _, stats = verify_greedy(
        [1, 9],
        mx.array([[0.0, 1.0, 0.0]]),
        advance,
        observations.append,
        eos=99,
        remaining=3,
    )

    assert output == [1, 2]
    assert advances == [1, 2]
    assert 9 not in advances
    assert len(observations) == 2
    assert stats == {"accepted_draft_tokens": 0, "rejected_blocks": 1}
