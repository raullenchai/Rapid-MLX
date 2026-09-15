from __future__ import annotations

import json

from rapid_mlx.spec_decode.dspark.detect import detect_dspark_metadata


def _write_checkpoint(tmp_path, *, complete: bool = True):
    (tmp_path / "inference").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
    (tmp_path / "inference" / "config.json").write_text(
        json.dumps(
            {
                "n_mtp_layers": 3,
                "dspark_block_size": 5,
                "dspark_noise_token_id": 128799,
                "dspark_target_layer_ids": [40, 41, 42],
                "dspark_markov_rank": 256,
            }
        )
    )
    keys = {
        "mtp.0.main_proj.weight": "model-1.safetensors",
        "mtp.2.markov_head.markov_w1.weight": "model-1.safetensors",
        "mtp.2.markov_head.markov_w2.weight": "model-1.safetensors",
    }
    if not complete:
        keys.pop("mtp.2.markov_head.markov_w2.weight")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": keys})
    )


def test_detects_complete_deepseek_v4_dspark_checkpoint(tmp_path):
    _write_checkpoint(tmp_path)
    metadata = detect_dspark_metadata(tmp_path)
    assert metadata is not None
    assert metadata.block_size == 5
    assert metadata.target_layer_ids == (40, 41, 42)


def test_rejects_stripped_dspark_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, complete=False)
    assert detect_dspark_metadata(tmp_path) is None
