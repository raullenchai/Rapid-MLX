from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vllm_mlx.quantization.glm53_rmq import TensorDescriptor

mx = pytest.importorskip("mlx.core")


def _converter_module():
    path = Path(__file__).parents[1] / "scripts" / "glm53_rmq_convert.py"
    spec = importlib.util.spec_from_file_location("glm53_rmq_convert", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERTER = _converter_module()
FP8_CONFIG = {
    "model_type": "glm5_next",
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    },
}


def test_official_block_fp8_is_a_supported_source_format():
    assert CONVERTER._is_supported_block_fp8(FP8_CONFIG)
    assert not CONVERTER._is_supported_block_fp8(
        {"quantization_config": {"quant_method": "fp8"}}
    )


def test_fp8_descriptors_pair_weight_and_scale_grid():
    weight = TensorDescriptor("layers.0.proj.weight", (129, 257), "F8_E4M3")
    scale = TensorDescriptor("layers.0.proj.weight_scale_inv", (2, 3), "F32")
    logical, scales = CONVERTER._logical_source_descriptors([weight, scale], FP8_CONFIG)
    assert logical == [TensorDescriptor(weight.name, weight.shape, "BF16")]
    assert scales == {weight.name: scale.name}


def test_fp8_descriptor_rejects_a_wrong_scale_grid():
    weight = TensorDescriptor("layers.0.proj.weight", (129, 257), "F8_E4M3")
    scale = TensorDescriptor("layers.0.proj.weight_scale_inv", (1, 3), "F32")
    with pytest.raises(RuntimeError, match="invalid FP8 scale grid"):
        CONVERTER._logical_source_descriptors([weight, scale], FP8_CONFIG)


def test_fp8_descriptor_rejects_an_integer_scale_grid():
    weight = TensorDescriptor("layers.0.proj.weight", (128, 128), "F8_E4M3")
    scale = TensorDescriptor("layers.0.proj.weight_scale_inv", (1, 1), "U8")
    with pytest.raises(RuntimeError, match="floating dtype"):
        CONVERTER._logical_source_descriptors([weight, scale], FP8_CONFIG)


def test_block_fp8_restore_applies_each_128_square_inverse_scale():
    source = mx.ones((128, 256), dtype=mx.bfloat16)
    encoded = mx.to_fp8(source)
    scales = mx.array([[2.0, 3.0]], dtype=mx.float32)
    restored = CONVERTER._restore_block_fp8(encoded, scales)
    mx.eval(restored)
    assert restored.shape == source.shape
    assert float(restored[0, 0]) == 2.0
    assert float(restored[0, 200]) == 3.0
