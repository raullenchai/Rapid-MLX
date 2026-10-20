from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import fakequant, rope_quant  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.layers import rope_tail  # noqa: E402

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal kernel requires a GPU"
)


def _graph(x, rd, cos, sin, fmt, out_dtype=None):
    block, _, quant = rope_quant.FORMATS[fmt]
    return quant(rope_tail(x, rd, cos, sin), block).astype(out_dtype or x.dtype)


def _inputs(shape, dtype, seed):
    mx.random.seed(seed)
    s = shape[1]
    x = mx.random.normal(shape) * mx.exp(mx.random.normal(shape) * 3)
    # whole blocks of zeros, tiny values and values past every scale ceiling
    x[..., :32] = 0.0
    x[..., 32:64] = mx.random.normal(x[..., 32:64].shape) * 1e-30
    x[..., 64:96] = mx.random.normal(x[..., 64:96].shape) * 1e6
    angles = mx.random.uniform(0, 6.3, (s, 32))
    return x.astype(dtype), mx.cos(angles), mx.sin(angles)


@needs_metal
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("shape", [(2, 3, 4, 128), (1, 1, 32, 128), (2, 5, 512)])
def test_kernel_is_bit_identical_to_the_graph(fmt, dtype, shape):
    x, cos, sin = _inputs(shape, dtype, seed=len(shape) + shape[1])
    if dtype == mx.float16:
        x = mx.clip(x, -3e4, 3e4)  # keep the rotated values finite
    got = rope_quant.rope_fake_quant(x, 64, cos, sin, fmt)
    want = _graph(x, 64, cos, sin, fmt)
    mx.eval(got, want)
    assert got.dtype == want.dtype == dtype
    assert np.array_equal(
        np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32))
    )


@needs_metal
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
def test_rounding_ties_and_grid_points(fmt):
    # every e2m1/e4m3 grid point and midpoint times a range of block scales
    grid = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6])
    grid = np.concatenate([grid, grid * 7.5, grid * 74.6667])
    vals = np.concatenate([grid, -grid])[None, :] * (2.0 ** np.arange(-8, 8))[:, None]
    vals = vals.reshape(-1)[: (vals.size // 32) * 32].reshape(1, -1, 32)
    x = mx.array(np.concatenate([vals] * 4, axis=-1), dtype=mx.float32)
    cos = mx.ones((x.shape[1], 32))
    sin = mx.zeros((x.shape[1], 32))
    got = rope_quant.rope_fake_quant(x, 64, cos, sin, fmt)
    want = _graph(x, 64, cos, sin, fmt)
    assert np.array_equal(np.array(got), np.array(want))


@needs_metal
def test_output_dtype_rounds_through_the_input_dtype():
    x, cos, sin = _inputs((1, 2, 128), mx.bfloat16, seed=7)
    got = rope_quant.rope_fake_quant(x, 64, cos, sin, "fp4_e4m3", mx.float32)
    want = _graph(x, 64, cos, sin, "fp4_e4m3", mx.float32)
    assert got.dtype == mx.float32
    assert np.array_equal(np.array(got), np.array(want))


def test_supports_and_graph_fallback(monkeypatch):
    monkeypatch.setattr(rope_quant.mx, "default_device", lambda: mx.gpu)
    x = mx.zeros((1, 1, 4, 128), dtype=mx.bfloat16)
    assert rope_quant.supports(x, 64)
    assert rope_quant.supports(mx.zeros((1, 2, 512)), 64)
    assert not rope_quant.supports(mx.zeros((1, 2, 4, 120)), 64)
    assert not rope_quant.supports(mx.zeros((2, 128)), 64)
    assert not rope_quant.supports(x.astype(mx.int32), 64)
    assert not rope_quant.supports(x, 63)
    assert not rope_quant.supports(x, 0)
    monkeypatch.setattr(fakequant, "DISABLE", True)
    assert not rope_quant.supports(x, 64)
    monkeypatch.setattr(fakequant, "DISABLE", False)
    monkeypatch.setattr(rope_quant, "ENABLED", False)
    assert not rope_quant.supports(x, 64)
    # the graph path when the kernel declines
    mx.random.seed(0)
    y = mx.random.normal((1, 2, 4, 128))
    cos, sin = mx.ones((2, 32)), mx.zeros((2, 32))
    got = rope_quant.rope_fake_quant(y, 64, cos, sin, "fp4_ue8m0", mx.bfloat16)
    assert got.dtype == mx.bfloat16
    assert np.array_equal(
        np.array(got.astype(mx.float32)),
        np.array(_graph(y, 64, cos, sin, "fp4_ue8m0", mx.bfloat16).astype(mx.float32)),
    )


def test_dispatch_geometry(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return [mx.zeros(kw["output_shapes"][0], dtype=kw["output_dtypes"][0])]

    monkeypatch.setattr(rope_quant, "supports", lambda x, rd: True)
    monkeypatch.setattr(rope_quant, "_kernel", lambda: fake)
    x = mx.zeros((2, 3, 4, 128), dtype=mx.bfloat16)
    out = rope_quant.rope_fake_quant(
        x, 64, mx.ones((3, 32)), mx.zeros((3, 32)), "fp4_e4m3", mx.float32
    )
    assert out.shape == x.shape and out.dtype == mx.float32
    (call,) = calls
    assert call["grid"] == (x.size, 1, 1) and call["threadgroup"] == (256, 1, 1)
    template = dict(call["template"])
    assert (template["H"], template["S"], template["BLK"], template["FMT"]) == (
        4,
        3,
        16,
        2,
    )
    assert np.array(call["inputs"][3]).item() == pytest.approx(1 / 6)
    calls.clear()
    rope_quant.rope_fake_quant(
        mx.zeros((1, 2, 512)), 64, mx.ones((2, 32)), mx.zeros((2, 32)), "fp8_ue8m0"
    )
    template = dict(calls[0]["template"])
    assert (template["H"], template["BLK"], template["FMT"]) == (1, 32, 0)
    assert np.array(calls[0]["inputs"][3]).item() == pytest.approx(1 / 448)


def test_kernel_builder(monkeypatch):
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(kw) or kw)
    rope_quant._kernel.cache_clear()
    try:
        rope_quant._kernel()
    finally:
        rope_quant._kernel.cache_clear()
    assert built[0]["name"] == "rapid_v41_rope_fake_quant"
