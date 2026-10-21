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


def _assert_same(got, want):
    """Equal values, NaN in the same places, and the same sign bits (-0, NaN)."""
    a = np.array(got.astype(mx.float32))
    b = np.array(want.astype(mx.float32))
    assert np.array_equal(a, b, equal_nan=True)
    assert np.array_equal(np.signbit(a), np.signbit(b))


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
    _assert_same(got, want)


@needs_metal
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
def test_rounding_ties_and_grid_points(fmt):
    # e2m1 grid points and midpoints (and the same scaled onto the e4m3 grid)
    # times a range of block scales
    grid = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6])
    grid = np.concatenate([grid, grid * 7.5, grid * 74.6667])
    vals = np.concatenate([grid, -grid])[None, :] * (2.0 ** np.arange(-8, 8))[:, None]
    vals = vals.reshape(-1)[: (vals.size // 32) * 32].reshape(1, -1, 32)
    x = mx.array(np.concatenate([vals] * 4, axis=-1), dtype=mx.float32)
    cos = mx.ones((x.shape[1], 32))
    sin = mx.zeros((x.shape[1], 32))
    got = rope_quant.rope_fake_quant(x, 64, cos, sin, fmt)
    want = _graph(x, 64, cos, sin, fmt)
    _assert_same(got, want)


@needs_metal
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("special", [np.nan, -np.nan, np.inf, -np.inf, -0.0])
def test_non_finite_and_signed_zero_match_the_graph(fmt, dtype, special):
    # the graph propagates NaN through the block max (poisoning the whole
    # block's scale), saturates to_fp8 (NaN -> +-448) and keeps -0 through FP8
    rs = np.random.RandomState(3)
    x = rs.randn(2, 3, 2, 128).astype(np.float32)
    x[0, 0, 0, 5] = special  # pass-through channel
    x[0, 1, 1, 100] = special  # rope tail channel
    x[1, 2, 0, 64:80] = special  # a whole FP4 block
    x[1, 0, 1, :] = -0.0  # an all -0 row
    angles = rs.uniform(0, 6.3, (3, 32)).astype(np.float32)
    cos, sin = mx.array(np.cos(angles)), mx.array(np.sin(angles))
    x = mx.array(x).astype(dtype)
    got = rope_quant.rope_fake_quant(x, 64, cos, sin, fmt)
    want = _graph(x, 64, cos, sin, fmt)
    _assert_same(got, want)


@needs_metal
def test_every_e4m3_value_midpoint_and_neighbor():
    # each block of 32 opens with 448 * 2**k so the ue8m0 scale is exactly 2**k
    # and the other 31 lanes land on, between and next to e4m3 codes
    codes = np.arange(0x00, 0x7F, dtype=np.uint8)  # +0 .. 448
    vals = np.array(mx.from_fp8(mx.array(codes), mx.float32))
    mids = (vals[:-1] + vals[1:]) / 2
    pts = np.concatenate([vals, mids, np.nextafter(mids, 0), np.nextafter(mids, 1e9)])
    pts = np.concatenate([pts, -pts]).astype(np.float32)
    pts = np.resize(pts, (-(-pts.size // 124)) * 124).reshape(-1, 31)
    blocks = []
    for k in (-6, 0, 5):
        lead = np.full((pts.shape[0], 1), 448.0 * 2.0**k, dtype=np.float32)
        blocks.append(np.concatenate([lead, pts * 2.0**k], axis=1))
    rows = np.concatenate(blocks).reshape(1, -1, 128)  # 4 blocks per row
    x = mx.array(rows)
    s = x.shape[1]
    cos, sin = mx.ones((s, 32)), mx.zeros((s, 32))
    _assert_same(
        rope_quant.rope_fake_quant(x, 64, cos, sin, "fp8_ue8m0"),
        _graph(x, 64, cos, sin, "fp8_ue8m0"),
    )


@needs_metal
def test_e4m3_scale_midpoints_and_neighbors():
    # fp4_e4m3 blocks of 16 whose amax / 6 lands on, between and next to every
    # e4m3 scale code, so the scale rounding itself is exercised
    codes = np.arange(0x00, 0x7F, dtype=np.uint8)
    vals = np.array(mx.from_fp8(mx.array(codes), mx.float32))
    mids = (vals[:-1] + vals[1:]) / 2
    amax = 6 * np.concatenate(
        [vals, mids, np.nextafter(mids, 0), np.nextafter(mids, 1e9)]
    )
    amax = np.resize(amax, (-(-amax.size // 8)) * 8).astype(np.float32)
    rs = np.random.RandomState(9)
    blocks = rs.uniform(-1, 1, (amax.size, 16)).astype(np.float32) * amax[:, None]
    blocks[:, 0] = amax * np.where(rs.rand(amax.size) < 0.5, -1, 1)
    x = mx.array(blocks.reshape(1, -1, 128))
    s = x.shape[1]
    cos, sin = mx.ones((s, 32)), mx.zeros((s, 32))
    _assert_same(
        rope_quant.rope_fake_quant(x, 64, cos, sin, "fp4_e4m3"),
        _graph(x, 64, cos, sin, "fp4_e4m3"),
    )


@needs_metal
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
def test_rope_products_are_rounded_separately(fmt):
    # re*c - im*s with re*c == im*s up to rounding: a fused multiply-add keeps
    # the product's rounding error and returns a tiny nonzero instead of 0
    rs = np.random.RandomState(11)
    re = rs.uniform(0.5, 2, (1, 4, 32)).astype(np.float32)
    ang = rs.uniform(0.1, 1.4, (4, 32)).astype(np.float32)
    c, sn = np.cos(ang), np.sin(ang)
    im = (re * c / sn).astype(np.float32)
    x = np.zeros((1, 4, 128), dtype=np.float32)
    x[..., 64::2], x[..., 65::2] = re, im
    xa, ca, sa = mx.array(x), mx.array(c), mx.array(sn)
    tail = np.array(rope_tail(xa, 64, ca, sa))[..., 64::2]
    assert np.any(tail != 0) and np.any(tail == 0)  # cancellation is in play
    _assert_same(
        rope_quant.rope_fake_quant(xa, 64, ca, sa, fmt),
        _graph(xa, 64, ca, sa, fmt),
    )


@needs_metal
@pytest.mark.parametrize("fmt", sorted(rope_quant.FORMATS))
def test_strided_inputs_gathered_tables_and_partial_threadgroups(fmt):
    rs = np.random.RandomState(5)
    # [b, h, s, d] transposed to a non-contiguous [b, s, h, d]
    x = mx.array(rs.randn(2, 3, 7, 96).astype(np.float32)).transpose(0, 2, 1, 3)
    table = rs.uniform(0, 6.3, (50, 32)).astype(np.float32)
    cos_t, sin_t = mx.array(np.cos(table)), mx.array(np.sin(table))
    pos = (3 + mx.arange(7)) * 4  # int32 group positions, as the callers gather
    cos, sin = cos_t[pos], sin_t[pos]
    assert x.size % 256  # the last threadgroup is partial
    _assert_same(
        rope_quant.rope_fake_quant(x.astype(mx.bfloat16), 64, cos, sin, fmt),
        _graph(x.astype(mx.bfloat16), 64, cos, sin, fmt),
    )
    _assert_same(
        rope_quant.rope_fake_quant(x, 64, cos, sin, fmt),
        _graph(x, 64, cos, sin, fmt),
    )


@needs_metal
def test_output_dtype_rounds_through_the_input_dtype():
    x, cos, sin = _inputs((1, 2, 128), mx.bfloat16, seed=7)
    got = rope_quant.rope_fake_quant(x, 64, cos, sin, "fp4_e4m3", mx.float32)
    want = _graph(x, 64, cos, sin, "fp4_e4m3", mx.float32)
    assert got.dtype == mx.float32
    assert np.array_equal(np.array(got), np.array(want))


def test_supports_and_graph_fallback(monkeypatch):
    monkeypatch.setattr(rope_quant.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(rope_quant.mx.metal, "is_available", lambda: True)
    x = mx.zeros((1, 1, 4, 128), dtype=mx.bfloat16)
    t1, t2 = mx.zeros((1, 32)), mx.zeros((2, 32))
    assert rope_quant.supports(x, 64, t1, t1)
    assert rope_quant.supports(mx.zeros((1, 2, 512)), 64, t2, t2)
    assert not rope_quant.supports(mx.zeros((1, 2, 4, 120)), 64, t2, t2)
    assert not rope_quant.supports(mx.zeros((2, 128)), 64, t2, t2)
    assert not rope_quant.supports(x.astype(mx.int32), 64, t1, t1)
    assert not rope_quant.supports(x, 63, t1, t1)
    assert not rope_quant.supports(x, 0, t1, t1)
    # tables that are not one [rd // 2] row per position
    assert not rope_quant.supports(x, 64, t2, t2)
    assert not rope_quant.supports(x, 64, mx.zeros((1, 64)), mx.zeros((1, 64)))
    assert not rope_quant.supports(x, 64, t1, mx.zeros((1, 16)))
    assert not rope_quant.supports(x, 64, mx.zeros((32,)), mx.zeros((32,)))
    # tables the graph would promote, and empty inputs
    c64 = mx.zeros((1, 32), dtype=mx.complex64)
    assert not rope_quant.supports(x, 64, c64, c64)
    assert not rope_quant.supports(x, 64, t1.astype(mx.bfloat16), t1)
    assert not rope_quant.supports(mx.zeros((0, 1, 128)), 64, t1, t1)
    assert not rope_quant.supports(mx.zeros((1, 0, 128)), 64, t1[:0], t1[:0])
    # a GPU default device without usable Metal stays on the graph
    monkeypatch.setattr(rope_quant.mx.metal, "is_available", lambda: False)
    assert not rope_quant.supports(x, 64, t1, t1)
    monkeypatch.setattr(rope_quant.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(fakequant, "DISABLE", True)
    assert not rope_quant.supports(x, 64, t1, t1)
    monkeypatch.setattr(fakequant, "DISABLE", False)
    monkeypatch.setattr(rope_quant, "ENABLED", False)
    assert not rope_quant.supports(x, 64, t1, t1)
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

    monkeypatch.setattr(rope_quant, "supports", lambda x, rd, cos, sin: True)
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
