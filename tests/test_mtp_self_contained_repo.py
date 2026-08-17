from pathlib import Path

from vllm_mlx.spec_decode.mtp.qwen3_5_inject import _find_mtp_weights_file


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
