import json
from pathlib import Path

from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
    _find_mtp_weights_file,
    _load_mtplx_runtime_contract,
)


def test_finds_mtplx_root_sidecar_before_target_shards(tmp_path: Path):
    sidecar = tmp_path / "mtp.safetensors"
    target = tmp_path / "model.safetensors"
    sidecar.touch()
    target.touch()

    assert _find_mtp_weights_file(tmp_path) == sidecar


def test_finds_nested_mtp_sidecar_in_self_contained_repo(tmp_path: Path):
    sidecar = tmp_path / "mtp" / "model.safetensors"
    sidecar.parent.mkdir()
    sidecar.touch()

    assert _find_mtp_weights_file(tmp_path) == sidecar


def test_explicit_root_sidecar_wins_over_nested_layout(tmp_path: Path):
    explicit = tmp_path / "model-mtp.safetensors"
    nested = tmp_path / "mtp" / "model.safetensors"
    nested.parent.mkdir()
    explicit.touch()
    nested.touch()

    assert _find_mtp_weights_file(tmp_path) == explicit


def test_nested_sidecar_wins_over_target_root_weights(tmp_path: Path):
    target = tmp_path / "model.safetensors"
    sidecar = tmp_path / "mtp" / "model.safetensors"
    sidecar.parent.mkdir()
    target.touch()
    sidecar.touch()

    assert _find_mtp_weights_file(tmp_path) == sidecar


def test_reads_closed_mtplx_runtime_contract(tmp_path: Path):
    weights = tmp_path / "mtp.safetensors"
    weights.touch()
    (tmp_path / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "mtp_contract": {
                    "base_hidden_variant": "post_norm",
                    "hidden_variant": "post_norm",
                    "concat_order": "embedding_hidden",
                    "mtp_position_mode": "local",
                }
            }
        ),
        encoding="utf-8",
    )

    assert _load_mtplx_runtime_contract(weights) == {
        "base_hidden_variant": "post_norm",
        "hidden_variant": "post_norm",
        "concat_order": "embedding_hidden",
        "mtp_position_mode": "local",
    }


def test_rejects_unknown_mtplx_contract_values(tmp_path: Path):
    weights = tmp_path / "mtp.safetensors"
    weights.touch()
    (tmp_path / "mtplx_runtime.json").write_text(
        json.dumps({"mtp_contract": {"base_hidden_variant": "mystery"}}),
        encoding="utf-8",
    )

    assert _load_mtplx_runtime_contract(weights) == {}
