import json
from pathlib import Path

from vllm_mlx.spec_decode.mtp.qwen3_5_inject import (
    BASE_HIDDEN_VARIANT_DEFAULT,
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


def test_manifest_without_a_hidden_variant_takes_the_trained_contract(tmp_path: Path):
    """An MTPLX manifest may pin only the fields it disagrees with.

    Whatever it leaves out has to land on the same default a sidecar with
    no manifest at all gets, or the two paths drift and the same head is
    driven two different ways depending on whether an unrelated field was
    worth writing down.
    """
    weights = tmp_path / "mtp.safetensors"
    weights.touch()
    (tmp_path / "mtplx_runtime.json").write_text(
        json.dumps({"mtp_contract": {"mtp_position_mode": "local"}}),
        encoding="utf-8",
    )

    assert _load_mtplx_runtime_contract(weights) == {
        "base_hidden_variant": BASE_HIDDEN_VARIANT_DEFAULT,
        "hidden_variant": BASE_HIDDEN_VARIANT_DEFAULT,
        "concat_order": "embedding_hidden",
        "mtp_position_mode": "local",
    }


def test_the_default_base_hidden_is_the_tensor_the_output_head_scores():
    """Qwen3-Next MTP heads are trained on the backbone's final hidden.

    ``mlx_lm.models.qwen3_5.TextModel.__call__`` ends in
    ``self.norm(hidden_states)``, and vLLM's ``qwen3_next_mtp.py`` hands
    ``pre_fc_norm_hidden`` the post-norm tensor its ``compute_logits``
    scores. Pinning the constant keeps a future edit to it deliberate.
    """
    assert BASE_HIDDEN_VARIANT_DEFAULT == "post_norm"
