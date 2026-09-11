from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx


def _load_script():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "extract_deepseek_v41_dspark_precision.py"
    )
    spec = importlib.util.spec_from_file_location("dspark_precision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "mtp.0.attn.wq_a.weight",
        "mtp.0.main_proj.weight",
        "mtp.1.ffn.shared_experts.w2.weight",
        "mtp.2.ffn.experts.127.w3.weight",
        "mtp.2.confidence_head.proj.weight",
        "mtp.2.markov_head.embed.weight",
    ],
)
def test_quantized_base_covers_every_dspark_linear(name):
    module = _load_script()
    assert module._quantized_base(name, mx.zeros((64, 64)), 64) == name[:-7]


@pytest.mark.parametrize(
    "name",
    [
        "mtp.0.attn_norm.weight",
        "mtp.0.ffn.gate.weight",
        "mtp.0.hc_attn_fn",
        "layers.0.attn.wq_a.weight",
    ],
)
def test_quantized_base_keeps_non_linear_contract_tensors_unquantized(name):
    module = _load_script()
    assert module._quantized_base(name, mx.zeros((64, 64)), 64) is None


def test_script_cannot_redirect_or_download_hub_cache():
    module = _load_script()
    source = Path(module.__file__).read_text()
    assert "hf_hub_download" not in source
    assert "snapshot_download" not in source
    assert "cache_dir" not in source
    assert "local_dir" not in source


def test_weight_map_rejects_missing_or_malformed_map(tmp_path):
    module = _load_script()
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    missing.write_text("{}")
    malformed.write_text('{"weight_map":{"tensor":3}}')

    with pytest.raises(ValueError, match="has no weight_map"):
        module._weight_map(missing)
    with pytest.raises(ValueError, match="map strings to strings"):
        module._weight_map(malformed)


def test_finalize_rejects_state_output_outside_destination(tmp_path):
    module = _load_script()
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "bad.conversion.json").write_text(
        '{"output_file":"../foreign.safetensors"}'
    )

    with pytest.raises(ValueError, match="unsafe output filename"):
        module._finalize(
            destination,
            tmp_path / "config.json",
            tmp_path / "index.json",
            "revision",
        )


def test_mixed_policy_changes_only_routed_experts():
    module = _load_script()
    assert module._use_low_precision("mtp.0.ffn.experts.127.w2.weight")
    assert not module._use_low_precision("mtp.0.ffn.shared_experts.w2.weight")
    assert not module._use_low_precision("mtp.0.attn.wq_a.weight")


def test_mixed_quantization_preserves_dense_four_bit_and_experts_two_bit():
    module = _load_script()
    high = {
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "modules": {
                "mtp.0.attn.wq_a": {"bits": 4},
                "mtp.0.ffn.experts.0.w1": {"bits": 4},
            },
        }
    }
    low = {"quantization": {"bits": 2, "group_size": 64, "mode": "affine"}}

    result = module._mixed_quantization(high, low)

    assert result["modules"]["mtp.0.attn.wq_a"] == {"bits": 4}
    assert result["modules"]["mtp.0.ffn.experts.0.w1"] == {
        "bits": 2,
        "group_size": 64,
        "mode": "affine",
    }
    assert result["modules"]["embed"]["bits"] == 2
    assert result["modules"]["head"]["bits"] == 2


def test_mixed_quantization_rejects_an_unqualified_precision_pair():
    module = _load_script()
    high = {
        "quantization": {
            "bits": 6,
            "group_size": 64,
            "modules": {"mtp.0.attn.wq_a": {"bits": 6}},
        }
    }
    low = {"quantization": {"bits": 3, "group_size": 64}}

    with pytest.raises(ValueError, match="requires a 2-bit low input and 4-bit high"):
        module._mixed_quantization(high, low)
