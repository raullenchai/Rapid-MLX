# Copyright © 2025-2026 Apple Inc.
# SPDX-License-Identifier: MIT
# Dot grouping and lane ownership adapted from MLX's fp_quantized.h (v0.32.5).
"""Single-token routed NVFP4 projections matching the stock FP32 reduction."""

from functools import cache

import mlx.core as mx

_HEADER = """
// Stock's precompiled kernels use -fno-fast-math. Safe runtime math alone
// still permits contracting sum += scale * dot into a differently rounded FMA.
#pragma clang fp contract(off)
inline float fp4(uint c) {
    int v = (0xc8643210u >> (4 * (c & 7))) & 15;
    float f = float(v) * 0.5f;
    return (c & 8) ? -f : f;
}
inline float scale8(uint s) {
    if ((s & 127) == 127) return NAN;
    uint e = (s >> 3) & 15;
    float v = e == 0 ? float(s & 7) * (1.f / 512.f)
        : as_type<float>(((s & 127) << 20) + (120u << 23));
    return (s & 128) ? -v : v;
}
"""

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    uint k = thread_position_in_grid.z;
    if (row >= M) return;
    uint expert = idx[k];
    uint wb = (expert * M + row) * (N / 8);
    uint sb = (expert * M + row) * (N / 16);
    uint xb = SHARED ? 0 : k * N;
    float sum = 0;
    // gather_qmv_fast owns 16 values/lane in 512-value blocks; gather_qmv
    // owns 8 in 256-value blocks. Here N is K and M is the output width;
    // only the former requires K % 512 == 0 and output width % 8 == 0.
    constexpr uint words = (N % 512 == 0 && M % 8 == 0) ? 2 : 1;
    for (uint q = lane * words; q < N / 8; q += 32 * words) {
        float dot = 0;
        const device ushort* ws = (const device ushort*)(w + wb + q);
        // qdot adds left-associated quads, then scales the partial dot.
        for (uint i = 0; i < 2 * words; ++i) {
            uint a = ws[i];
            uint xi = xb + q * 8 + 4 * i;
            dot += (float(x[xi]) * fp4(a & 15)
                + float(x[xi + 1]) * fp4((a >> 4) & 15)
                + float(x[xi + 2]) * fp4((a >> 8) & 15)
                + float(x[xi + 3]) * fp4((a >> 12) & 15));
        }
        // For K % 16 == 0, every active tail lane owns all eight values.
        sum += scale8(scales[sb + q / 2]) * dot;
    }
    sum = simd_sum(sum);
    // Tensor scaling stays in FP32 until the sole output rounding.
    if (HAS_SCALE) sum *= float(global_scale[expert]) / 2688.f;
    if (lane == 0) out[k * M + row] = bfloat16_t(sum);
"""


@cache
def _get_kernel():
    try:
        return mx.fast.metal_kernel(
            name="inkling2_nvfp4_decode",
            input_names=["x", "w", "scales", "idx", "global_scale"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    except Exception:
        # Cache failure too, so unsupported runtimes keep the stock path.
        return None


def project(x, w, scales, indices, global_scale):
    kernel = _get_kernel()
    if kernel is None:
        return None
    rows, n, p = w.shape[1], x.shape[-1], indices.size
    shape = (1,) * max(x.ndim - 3, indices.ndim - 1) + (p, 1, rows)
    scale = global_scale if global_scale is not None else scales
    return kernel(
        inputs=[x, w, scales, indices.astype(mx.uint32), scale],
        template=[
            ("N", n),
            ("M", rows),
            ("SHARED", x.shape[-3] == 1),
            ("HAS_SCALE", global_scale is not None),
        ],
        grid=(32, rows, p),
        threadgroup=(32, 16, 1),
        output_shapes=[shape],
        output_dtypes=[mx.bfloat16],
    )[0]
