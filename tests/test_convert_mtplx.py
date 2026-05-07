import json

import mlx.core as mx

from vllm_mlx.convert_mtplx import MTPLX_SUFFIX, convert_mtplx, default_output_path


def _write_fake_qwen36_mtp_model(path):
    path.mkdir()
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "mtp_num_hidden_layers": 1,
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tensors = {
        "mtp.fc.weight": mx.ones((2, 4)),
        "mtp.layers.0.input_layernorm.weight": mx.ones((2,)),
        "mtp.layers.0.mlp.experts.down_proj": mx.ones((3, 2, 1)),
        "mtp.layers.0.mlp.experts.gate_up_proj": mx.arange(24).reshape(3, 4, 2),
        "mtp.layers.0.mlp.gate.weight": mx.ones((3, 2)),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": mx.ones((2, 1)),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": mx.ones((1, 2)),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": mx.ones((1, 2)),
        "mtp.layers.0.mlp.shared_expert_gate.weight": mx.ones((1, 2)),
        "mtp.layers.0.post_attention_layernorm.weight": mx.ones((2,)),
        "mtp.layers.0.self_attn.k_norm.weight": mx.ones((1,)),
        "mtp.layers.0.self_attn.k_proj.weight": mx.ones((1, 2)),
        "mtp.layers.0.self_attn.o_proj.weight": mx.ones((2, 2)),
        "mtp.layers.0.self_attn.q_norm.weight": mx.ones((1,)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.ones((2, 2)),
        "mtp.layers.0.self_attn.v_proj.weight": mx.ones((1, 2)),
        "mtp.norm.weight": mx.ones((2,)),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((2,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((2,)),
    }
    shard = path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), tensors)
    index = {
        "metadata": {},
        "weight_map": {key: shard.name for key in tensors},
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _write_fake_base_model_without_mtp(path):
    path.mkdir()
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shard = path / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {"model.embed_tokens.weight": mx.ones((2, 2))})
    index = {
        "metadata": {},
        "weight_map": {"model.embed_tokens.weight": shard.name},
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def test_default_output_path_adds_mtplx_suffix(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B"
    assert default_output_path(model).name == f"{model.name}{MTPLX_SUFFIX}"


def test_convert_mtplx_writes_expected_sidecar_layout(tmp_path):
    model = tmp_path / "Qwen3.6-35B-A3B"
    _write_fake_qwen36_mtp_model(model)

    result = convert_mtplx(model)

    assert result.output.name == "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
    assert result.tensor_count == 20
    assert result.mtp_file == result.output / "mtp.safetensors"
    assert result.runtime_contract_file == result.output / "mtplx_runtime.json"

    sidecar = mx.load(str(result.mtp_file))
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in sidecar
    assert "mtp.layers.0.mlp.down_proj.weight" in sidecar
    assert "mtp.layers.0.mlp.gate.weight" in sidecar
    assert "mtp.layers.0.mlp.gate_proj.weight" in sidecar
    assert "mtp.layers.0.mlp.shared_expert.down_proj.weight" in sidecar
    assert "mtp.layers.0.mlp.shared_expert.gate_proj.weight" in sidecar
    assert "mtp.layers.0.mlp.shared_expert.up_proj.weight" in sidecar
    assert "mtp.layers.0.mlp.shared_expert_gate.weight" in sidecar
    assert "mtp.layers.0.mlp.up_proj.weight" in sidecar
    assert sidecar["mtp.layers.0.mlp.gate_proj.weight"].shape == (3, 2, 2)
    assert sidecar["mtp.layers.0.mlp.up_proj.weight"].shape == (3, 2, 2)
    assert mx.all(sidecar["mtp.norm.weight"] == 2.0).item()
    assert mx.all(sidecar["mtp.layers.0.input_layernorm.weight"] == 2.0).item()
    assert mx.all(sidecar["mtp.pre_fc_norm_embedding.weight"] == 2.0).item()
    assert mx.all(sidecar["mtp.pre_fc_norm_hidden.weight"] == 2.0).item()

    config = json.loads((result.output / "config.json").read_text())
    assert config["mlx_lm_extra_tensors"]["mtp_file"] == "mtp.safetensors"
    assert config["num_nextn_predict_layers"] == 1

    source_config = json.loads((model / "config.json").read_text())
    assert "mlx_lm_extra_tensors" not in source_config
    assert "num_nextn_predict_layers" not in source_config

    contract = json.loads(result.runtime_contract_file.read_text())
    assert contract["arch_id"] == "qwen3-next-mtp"
    assert contract["mtp_depth_max"] == 1
    assert contract["recommended_profile"] == "performance-cold"
    assert contract["exactness_baseline"]["context"] == 2048
    assert contract["exactness_baseline"]["max_abs_diff"] == 0.0


def test_convert_mtplx_can_read_mtp_from_separate_source(tmp_path):
    base = tmp_path / "Qwen3.6-35B-A3B-4bit"
    mtp_source = tmp_path / "Qwen3.6-35B-A3B"
    _write_fake_base_model_without_mtp(base)
    _write_fake_qwen36_mtp_model(mtp_source)

    result = convert_mtplx(base, mtp_source=mtp_source)

    assert result.source == base
    assert result.mtp_source == mtp_source
    assert result.output.name == "Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed"
    assert result.tensor_count == 20
    assert (result.output / "model-00001-of-00001.safetensors").exists()

    sidecar = mx.load(str(result.mtp_file))
    assert "mtp.layers.0.mlp.gate_proj.weight" in sidecar

    config = json.loads((result.output / "config.json").read_text())
    assert config["mlx_lm_extra_tensors"]["mtp_file"] == "mtp.safetensors"
    assert config["num_nextn_predict_layers"] == 1
