"""Rope tail plus activation fake-quantization as one Metal dispatch.

``rope_tail`` followed by one of the :mod:`fakequant` round trips is ~30 small
graph nodes (the concatenation, the pair split, four products, the block max,
the scale bit tricks, the e2m1 threshold ladder and its table gather, the casts).
Every op is elementwise or a block max, so the kernel reproduces each rounding
point exactly and its output is bit-identical to the graph:

* rope: fp32 products rounded separately, ``re*c - im*s`` / ``re*s + im*c``,
  rounded to the input dtype (channels before the tail pass through);
* block absolute max over the rounded values, clamped to the format's floor;
* scale: ue8m0 ``2**ceil(log2(amax * (1/max)))`` from the IEEE bits, or the
  e4m3-rounded ``amax / 6``;
* value: ``x / scale`` (exact, or the format's own rounding), clipped, rounded to
  the e4m3 or e2m1 grid (ties to even), times the scale, rounded to the input
  dtype and then to the output dtype.

One thread per element; a quantization block (16 or 32 elements) lives inside
one simdgroup, so its max is a shuffle reduction.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod

import mlx.core as mx

from . import fakequant
from .fakequant import FP4_MAX, FP8_MAX
from .layers import rope_tail
from .kernels import custom_kernels_enabled
from .self_check import SelfCheck

_CHECK = SelfCheck("V4.1 rope + fake-quant")

# Module switch: tests and A/B runs turn the kernel off to compare.
ENABLED = True

# format -> (block, kernel constant, graph round trip)
FORMATS = {
    "fp8_ue8m0": (32, 0, fakequant.fake_quant_fp8_ue8m0),
    "fp4_ue8m0": (32, 1, fakequant.fake_quant_fp4_ue8m0),
    "fp4_e4m3": (16, 2, fakequant.fake_quant_fp4_e4m3),
}

_HEADER = r"""
// E4M3 round-to-nearest-even of a value already clamped to [-448, 448]; the
// sign is kept, including that of zero (to_fp8 maps -0 to -0).
inline float round_e4m3(float v) {
    const float a = abs(v);
    const int step_e = max(int((as_type<uint>(a) >> 23) & 0xFFu) - 127 - 3, -9);
    const float step = as_type<float>(uint(step_e + 127) << 23);
    const float q = min(rint(a / step) * step, 448.0f);
    return copysign(q, v);
}
// to_fp8 saturates: NaN becomes 448 with the NaN's sign, +-inf becomes +-448.
inline float fp8_round_trip(float v) {
    return isnan(v) ? copysign(448.0f, v) : round_e4m3(clamp(v, -448.0f, 448.0f));
}
// mx.max / mx.maximum propagate NaN; Metal's max returns the other operand.
inline float nan_max(float a, float b) {
    return (isnan(a) || isnan(b)) ? NAN : max(a, b);
}
// 2^ceil(log2(x)) for positive normal x, from the IEEE bits (sign ignored, so
// non-finite x builds the same bit pattern as the graph's _ue8m0_scale).
inline float pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    const int e = int((bits >> 23) & 0xFFu) - 127 + int((bits & 0x7FFFFFu) != 0u);
    return as_type<float>(uint(e + 127) << 23);
}
// Nearest e2m1 value of v clipped to [-6, 6], ties to the even code, as
// sign(v) * lut[idx]: NaN and -0 give +0, a negative value rounding to zero -0.
inline float round_e2m1(float v) {
    if (isnan(v)) return 0.0f;
    const float m = min(abs(v), 6.0f);
    float q;
    if (m < 0.75f) q = m > 0.25f ? 0.5f : 0.0f;
    else if (m < 1.75f) q = m > 1.25f ? 1.5f : 1.0f;
    else if (m < 3.5f) q = m > 2.5f ? 3.0f : 2.0f;
    else q = m > 5.0f ? 6.0f : 4.0f;
    return v > 0.0f ? q : v < 0.0f ? -q : 0.0f;
}
"""

_SOURCE = r"""
    #pragma clang fp contract(off)
    const uint e = thread_position_in_grid.x;
    const uint d = e % D;
    const uint s = (e / (D * H)) % S;
    float v = float(x[e]);
    // rope on the tail: adjacent (re, im) pairs, products rounded separately
    const float partner = simd_shuffle_xor(v, 1u);
    if (d >= uint(D - RD)) {
        const uint p = (d - uint(D - RD)) / 2;
        const float c = cosv[s * (RD / 2) + p], sn = sinv[s * (RD / 2) + p];
        const bool odd = (d & 1u) != 0u;
        const float re = odd ? partner : v, im = odd ? v : partner;
        float o;
        if (odd) { const float a = re * sn; const float bb = im * c; o = a + bb; }
        else { const float a = re * c; const float bb = im * sn; o = a - bb; }
        v = float(T(o));
    }
    // NaN-propagating block max, like the graph's mx.max then mx.maximum
    float amax = abs(v);
    for (uint off = BLK / 2; off > 0; off /= 2) amax = nan_max(amax, simd_shuffle_xor(amax, off));
    float q;
    if (FMT == 0) {
        // a NaN amax gives the graph's bit-built scale (-0) here too
        const float sc = pow2_ceil(nan_max(amax, 1e-4f) * max_inv[0]);
        q = fp8_round_trip(v / sc) * sc;
    } else if (FMT == 1) {
        const float sc = pow2_ceil(nan_max(amax, 6.0f * 1.17549435e-38f) * max_inv[0]);
        q = round_e2m1(v / sc) * sc;
    } else {
        const float sc = fp8_round_trip(nan_max(amax, 6.0f / 512.0f) / 6.0f);
        q = round_e2m1(v / sc) * sc;
    }
    out[e] = OT(T(q));
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="rapid_v41_rope_fake_quant",
        input_names=["x", "cosv", "sinv", "max_inv"],
        output_names=["out"],
        source=_SOURCE,
        header=_HEADER,
    )


@lru_cache(maxsize=None)
def _scalar(value: float) -> mx.array:
    a = mx.array([value], dtype=mx.float32)
    mx.eval(a)
    return a


def supports(x: mx.array, rd: int, cos: mx.array, sin: mx.array) -> bool:
    """Whether the kernel covers ``x`` [b, s, (h,) d] with an ``rd`` rope tail.

    The kernel reads one float32 ``cos``/``sin`` row per sequence position, so
    the tables must be exactly [s, rd // 2] float32; anything the graph would
    broadcast or promote differently, and empty inputs, take the graph.
    """
    if not (
        ENABLED
        and custom_kernels_enabled()
        and not fakequant.DISABLE
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        return False
    return (
        x.ndim in (3, 4)
        and x.size > 0
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and x.shape[-1] % 32 == 0  # whole simdgroups, whole blocks
        and 0 < rd <= x.shape[-1]
        and rd % 2 == 0
        and tuple(cos.shape) == tuple(sin.shape) == (x.shape[1], rd // 2)
        and cos.dtype == sin.dtype == mx.float32
    )


def rope_fake_quant(
    x: mx.array, rd: int, cos: mx.array, sin: mx.array, fmt: str, out_dtype=None
) -> mx.array:
    """``fake_quant_<fmt>(rope_tail(x, rd, cos, sin)).astype(out_dtype)``.

    x [b, s, (h,) d]; cos/sin [s, rd // 2], one row per sequence position.
    """
    block, code, graph = FORMATS[fmt]
    out_dtype = out_dtype or x.dtype
    if not supports(x, rd, cos, sin) or not _CHECK.qualified(
        (x.dtype, out_dtype, x.shape[1:], rd, fmt),
        _probe,
        x.shape[1:],
        x.dtype,
        rd,
        fmt,
        out_dtype,
    ):
        return graph(rope_tail(x, rd, cos, sin), block).astype(out_dtype)
    return _fused(x, rd, cos, sin, fmt, out_dtype)


def _fused(x, rd, cos, sin, fmt, out_dtype, *, sequence=None):
    block, code, _ = FORMATS[fmt]
    max_inv = 1.0 / (FP8_MAX if code == 0 else FP4_MAX)
    return _kernel()(
        inputs=[x, cos, sin, _scalar(max_inv)],
        template=[
            ("T", x.dtype),
            ("OT", out_dtype),
            ("D", x.shape[-1]),
            ("H", x.shape[2] if x.ndim == 4 else 1),
            ("S", x.shape[1] if sequence is None else sequence),
            ("RD", rd),
            ("BLK", block),
            ("FMT", code),
        ],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[out_dtype],
    )[0]


def _probe(shape, dtype, rd, fmt, out_dtype):
    # Concrete arrays, independent of request values and compiled placeholders.
    # The launch grid is x.size, NOT S * H * D. With one batch and at most
    # two rows, e < rows*H*D and s=(e/(D*H))%S < rows; all x/out/cos/sin
    # accesses remain in bounds even when the compiled S is much larger.
    sequence = shape[0]
    shape = (min(sequence, 2), *shape[1:])
    x = mx.sin(mx.arange(prod(shape), dtype=mx.float32))
    x = (x.reshape(1, *shape) * 7.0).astype(dtype)
    angle = (
        mx.arange(shape[0] * rd // 2, dtype=mx.float32).reshape(shape[0], rd // 2)
        * 0.17
    )
    cos, sin = mx.cos(angle), mx.sin(angle)
    block, _, graph = FORMATS[fmt]
    expected = graph(rope_tail(x, rd, cos, sin), block).astype(out_dtype)
    actual = _fused(x, rd, cos, sin, fmt, out_dtype, sequence=sequence)
    return mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()
