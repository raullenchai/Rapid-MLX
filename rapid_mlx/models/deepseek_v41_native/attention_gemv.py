"""Exact affine attention projections with shared weight loads for 1..8 rows.

Each lane retains stock single-row QMV's K blocks, dot expression and SIMD
reduction. Up to ``MAX_ROWS`` rows share packed uint/uint2 weight loads and run
the same per-row arithmetic: each row is bit-identical to its own single-row
stock projection. Speculative verify and batched lanes therefore get the same
per-row arithmetic as sequential decode. Larger batches stay on the stock path.
Stock multi-row calls can dispatch a different quantized-matmul algorithm and
are not the bit reference.

The first call for each weight, dtype and shape checks seeded probes covering
the full row count against concatenated single-row stock projections, retaining
grouped broadcasting. Any bit difference or compile/launch failure keeps that
shape on stock with a warning; replaced weights are validated again.
The reused MLX helpers carry their MIT notice in moe_decode.py.
"""

from __future__ import annotations

import logging
import math
import weakref
from functools import lru_cache

import mlx.core as mx

from .kernels import custom_kernels_enabled
from .moe_decode import _HEADER, _qmv_vpt

logger = logging.getLogger(__name__)
ENABLED = True
MAX_ROWS = 8
_CHECK_ATTR = "_rapid_attention_gemv_check"

# qdot_w's 4/8-bit expressions, with register rather than device operands.
_DOT = r"""
template <int BITS, int VPT>
inline float attention_dot(uint2 packed, const thread float* x,
                           float scale, float bias, float sum) {
    float dot = 0;
    if (BITS == 4) {
        for (int i = 0; i < VPT / 4; i++) {
            const uint word = i < 2 ? packed.x : packed.y;
            const uint w = (word >> ((i % 2) * 16)) & 0xffff;
            dot += (x[4*i] * (w & 0x000f) + x[4*i+1] * (w & 0x00f0)
                  + x[4*i+2] * (w & 0x0f00) + x[4*i+3] * (w & 0xf000));
        }
    } else {
        for (int i = 0; i < VPT; i++) {
            const uint word = i < 4 ? packed.x : packed.y;
            dot += x[i] * ((word >> ((i % 4) * 8)) & 0xff);
        }
    }
    return scale * dot + sum * bias;
}
"""
_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int group = int(threadgroup_position_in_grid.z);
    const int col = int(threadgroup_position_in_grid.y) * 16
                  + int(simdgroup_index_in_threadgroup) * 4;
    float acc[ROWS][4] = {};
    for (int k = 0; k < K; k += 32 * VPT) {
        const int pos = k + int(lane) * VPT;
        if (pos >= K) break;
        float xt[ROWS][VPT];
        float sums[ROWS];
        for (int t = 0; t < ROWS; t++) {
            sums[t] = load_x<IT, BITS, VPT>(
                input + (t * GROUPS + group) * K + pos, xt[t]);
        }
        for (int r = 0; r < 4; r++) {
            if (col + r >= N) continue;
            const size_t row = size_t(group) * N + col + r;
            const device uint* w = weight + row * (K * BITS / 32)
                                           + pos * BITS / 32;
            uint2 packed;
            if constexpr (VPT * BITS == 64) {
                packed = *reinterpret_cast<const device uint2*>(w);
            } else {
                packed = uint2(*w, 0);
            }
            const float s = scales[row * (K / 64) + pos / 64];
            const float b = biases[row * (K / 64) + pos / 64];
            for (int t = 0; t < ROWS; t++) {
                acc[t][r] += attention_dot<BITS, VPT>(packed, xt[t], s, b, sums[t]);
            }
        }
    }
    for (int t = 0; t < ROWS; t++) {
        for (int r = 0; r < 4; r++) {
            const float value = simd_sum(acc[t][r]);
            if (lane == 0 && col + r < N) {
                output[(t * GROUPS + group) * N + col + r] = T(value);
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="rapid_v41_attention_affine_rows",
        input_names=["input", "weight", "scales", "biases"],
        output_names=["output"],
        source=_SOURCE,
        header=_HEADER + _DOT,
        ensure_row_contiguous=True,
    )


def _layout(module, x: mx.array) -> tuple[int, int, int, int] | None:
    """(rows, groups, N, K); grouped inputs retain [..., groups, 1, K]."""
    w = getattr(module, "weight", None)
    s = getattr(module, "scales", None)
    b = getattr(module, "biases", None)
    if w is None or s is None or b is None or "bias" in module:
        return None
    bits = getattr(module, "bits", None)
    if (
        bits not in (4, 8)
        or getattr(module, "group_size", None) != 64
        or getattr(module, "mode", "affine") != "affine"
        or w.ndim not in (2, 3)
        or s.ndim != w.ndim
        or tuple(b.shape) != tuple(s.shape)
        or tuple(w.shape[:-1]) != tuple(s.shape[:-1])
        or w.dtype != mx.uint32
        or s.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or b.dtype != s.dtype
        or x.dtype not in (mx.float32, mx.bfloat16, mx.float16)
        or x.ndim < 2
    ):
        return None
    k, n = int(s.shape[-1]) * 64, int(s.shape[-2])
    groups = int(w.shape[0]) if w.ndim == 3 else 1
    if w.ndim == 3 and (x.ndim < 4 or tuple(x.shape[-3:-1]) != (groups, 1)):
        return None
    rows = 1
    for size in x.shape[:-3] if w.ndim == 3 else x.shape[:-1]:
        rows *= int(size)
    # Whole affine groups guarantee complete lane packs, including the K tail.
    # Short reductions use the stock quad kernel, a different lane topology.
    if k <= 128 or n < 1 or x.shape[-1] != k or w.shape[-1] != k * bits // 32:
        return None
    return (rows, groups, n, k) if 1 <= rows <= MAX_ROWS else None


def supports(module, x: mx.array) -> bool:
    return bool(
        ENABLED
        and custom_kernels_enabled()
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and _layout(module, x) is not None
    )


def _launch(module, x: mx.array) -> mx.array:
    layout = _layout(module, x)
    if layout is None:
        raise ValueError("unsupported attention affine GEMV layout")
    rows, groups, n, k = layout
    bits = int(module.bits)
    dtype = mx.result_type(x.dtype, module.scales.dtype)
    (out,) = _kernel()(
        inputs=[x, module.weight, module.scales, module.biases],
        template=[
            ("IT", x.dtype),
            ("T", dtype),
            ("Q", module.scales.dtype),
            ("ROWS", rows),
            ("GROUPS", groups),
            ("N", n),
            ("K", k),
            ("BITS", bits),
            ("VPT", _qmv_vpt(bits, k, n)),
        ],
        grid=(32, ((n + 15) // 16) * 4, groups),
        threadgroup=(32, 4, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[dtype],
    )
    result: mx.array = out
    return result


def _qualified(module, x: mx.array, label: str) -> bool:
    arrays = (module.weight, module.scales, module.biases)
    settings = (module.bits, module.group_size, getattr(module, "mode", "affine"))
    cached = getattr(module, _CHECK_ATTR, None)
    if (
        cached is None
        or cached[1] != settings
        or any(ref() is not a for ref, a in zip(cached[0], arrays, strict=True))
    ):
        cached = (tuple(weakref.ref(a) for a in arrays), settings, {})
        object.__setattr__(module, _CHECK_ATTR, cached)
    verdicts: dict = cached[2]
    key = (x.dtype, tuple(x.shape))
    if key not in verdicts:
        ok = True
        try:
            # Both signs and more than one magnitude; no global RNG mutation.
            for seed, scale in ((41, 1.0), (73, 0.03125)):
                probe = (
                    mx.random.normal(x.shape, key=mx.random.key(seed)) * scale
                ).astype(x.dtype)
                # Flatten only the logical row axes, retaining each grouped
                # row's [groups, 1, K] shape for stock broadcasting.
                row_shape = (
                    probe.shape[-3:] if module.weight.ndim == 3 else probe.shape[-1:]
                )
                row_probe = probe.reshape(-1, *row_shape)
                expected = mx.concatenate(
                    [module(row_probe[r : r + 1]) for r in range(row_probe.shape[0])],
                    axis=0,
                ).reshape(*probe.shape[:-1], module.scales.shape[-2])
                actual = _launch(module, probe)
                ok = (
                    bool(
                        mx.array_equal(
                            actual.view(mx.uint8), expected.view(mx.uint8)
                        ).item()
                    )
                    and ok
                )
        except Exception:
            logger.warning("V4.1 attention GEMV unavailable: %s", label, exc_info=True)
            ok = False
        verdicts[key] = ok
        if not ok:
            logger.warning(
                "V4.1 attention GEMV stock fallback: %s shape=%s dtype=%s",
                label,
                x.shape,
                x.dtype,
            )
        checked = [
            (
                math.prod(
                    int(v) for v in shape[: -3 if module.weight.ndim == 3 else -1]
                ),
                passed,
            )
            for (dtype, shape), passed in verdicts.items()
            if dtype == x.dtype
        ]
        logger.info(
            "V4.1 attention GEMV coverage: %s bits=%s dtype=%s shape=%s exact=%s qualified_rows=%s fallback_rows=%s (unchecked shapes stay lazy)",
            label,
            module.bits,
            x.dtype,
            x.shape,
            ok,
            sorted({rows for rows, passed in checked if passed}),
            sorted({rows for rows, passed in checked if not passed}),
        )
    return bool(verdicts[key])


def project(module, x: mx.array, label: str = "attention") -> mx.array:
    """Use only a per-weight/dtype/shape qualified kernel; otherwise call stock."""
    if not supports(module, x) or not _qualified(module, x, label):
        result: mx.array = module(x)
        return result
    try:
        return _launch(module, x)
    except Exception:
        getattr(module, _CHECK_ATTR)[2][(x.dtype, tuple(x.shape))] = False
        logger.warning("V4.1 attention GEMV launch fallback: %s", label, exc_info=True)
        fallback: mx.array = module(x)
        return fallback
