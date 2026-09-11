from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "attach_deepseek_v41_dspark.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("attach_deepseek_v41_dspark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dspark_config_flattens_target_and_preserves_separate_experts():
    module = _load_script()
    target = {
        "model_type": "deepseek_v41",
        "n_routed_experts": 336,
        "text_config": {
            "hidden_size": 7168,
            "n_routed_experts": 336,
            "num_experts_per_tok": 6,
        },
        "rapid_quantization": {"mtp": False, "kept_experts": 336},
    }
    source = {
        "text_config": {
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128,
            "dspark_noise_token_id": 128799,
            "dspark_num_experts_per_tok": 3,
            "dspark_target_layer_ids": [37, 38, 39],
        }
    }

    config = module._dspark_config(target, source)

    assert config["model_type"] == "deepseek_v4"
    assert config["hidden_size"] == 7168
    assert config["n_routed_experts"] == 336
    assert config["num_experts_per_tok"] == 6
    assert config["dspark_n_routed_experts"] == 128
    assert config["dspark_num_experts_per_tok"] == 3
    assert config["rapid_quantization"]["mtp"] is True


def test_inference_config_carries_detection_geometry():
    module = _load_script()
    config = {
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_noise_token_id": 128799,
        "dspark_num_experts_per_tok": 3,
        "dspark_target_layer_ids": [37, 38, 39],
    }

    inference = module._inference_config(config)

    assert inference == {
        "n_mtp_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
    }


def test_script_refuses_to_overwrite_destination(tmp_path):
    module = _load_script()
    destination = tmp_path / "exists"
    destination.mkdir()

    try:
        module.build_overlay(tmp_path / "source", tmp_path / "target", destination)
    except FileExistsError as exc:
        assert str(destination) in str(exc)
    else:
        raise AssertionError("existing destination was overwritten")


def test_script_has_no_cache_redirect_flags():
    module = _load_script()
    source = Path(module.__file__).read_text()
    assert "cache_dir" not in source
    assert "local_dir" not in source


def _write_overlay_inputs(tmp_path, shard_name):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "config.json").write_text("{}")
    (target / "config.json").write_text("{}")
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"head.weight": shard_name}})
    )
    return source, target


def test_overlay_rejects_traversing_target_shard(tmp_path):
    module = _load_script()
    source, target = _write_overlay_inputs(tmp_path, "../outside.safetensors")

    with pytest.raises(ValueError, match="must be a basename"):
        module.build_overlay(source, target, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_overlay_rejects_reserved_mtp_shard_collision(tmp_path):
    module = _load_script()
    source, target = _write_overlay_inputs(tmp_path, "model-mtp.safetensors")

    with pytest.raises(ValueError, match="reserved shard"):
        module.build_overlay(source, target, tmp_path / "output")

    assert not (tmp_path / "output").exists()
