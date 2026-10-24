"""Tiny hardware qualification probes; no checkpoint or large model required."""

import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import (
    affine_route_qmv,
    decode_attention,
    rope_quant,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="qualification needs Metal"
)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_attention_real_probe(dtype):
    assert decode_attention._probe(dtype, mx.float32, 8, 128, 64, 8)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
@pytest.mark.parametrize("sequence", [1, 7, 4096])
def test_rope_real_probe(dtype, fmt, sequence):
    assert rope_quant._probe((sequence, 2, 128), dtype, 64, fmt, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_route_real_probe(dtype, bits):
    assert affine_route_qmv._probe(dtype, bits, 6, 2, 64, 8)
