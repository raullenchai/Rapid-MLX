"""Experimental exact down-projection affine 2/3/4-bit QMV for V4.1.

The verification batch has only 24--36 routed rows spread over hundreds of
experts.  A block GEMM therefore pads most expert groups to a much larger row
tile.  This kernel instead assigns one QMV grid to each real route.  It is
restricted to the mixed-dtype FP32 activation / affine-weight down projection,
where it is bit-exact with the stock path.  Gate/up deliberately stay stock.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .moe_decode import _HEADER, _packed_width, _qmv_vpt

# The shared helpers adapt the stock kernels under the MIT notice in moe_decode.
_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int route = int(threadgroup_position_in_grid.z);
    const int col = int(threadgroup_position_in_grid.y) * 8
        + int(simdgroup_index_in_threadgroup) * 4;
    const size_t base = size_t(indices[route]) * OUT + col;
    float acc[4] = {};
    rows_partial<T, Q, BITS, VPT, K, 4, false>(
        input + (route / TOPK) * K, weight, scales, biases,
        weight, scales, biases, base, lane, acc, acc);
    for (int r = 0; r < 4; r++) {
        const float value = simd_sum(acc[r]);
        if (lane == 0) {
            output[route * OUT + col + r] = T(value);
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="rapid_v41_affine2_route_qmv",
        input_names=["input", "weight", "scales", "biases", "indices"],
        output_names=["output"],
        source=_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=True,
    )


def _shape(module, inputs: mx.array, indices: mx.array) -> tuple[int, int, int, int]:
    if inputs.ndim != 2 or indices.ndim != 2:
        raise ValueError("inputs and indices must both be rank two")
    tokens, input_dims = map(int, inputs.shape)
    if int(indices.shape[0]) != tokens:
        raise ValueError("indices must have one row per input token")
    topk = int(indices.shape[1])
    routes = tokens * topk
    required = ("weight", "scales", "biases", "bits", "group_size")
    if any(not hasattr(module, name) for name in required):
        raise ValueError("unsupported exact affine down-route QMV layout")
    output_dims = int(module.scales.shape[1]) if module.scales.ndim == 3 else 0
    if not (
        module.weight.ndim == module.scales.ndim == module.biases.ndim == 3
        and int(module.bits) in (2, 3, 4)
        and int(module.group_size) == 64
        and getattr(module, "mode", "affine") == "affine"
        and inputs.dtype == mx.float32
        and module.scales.dtype in (mx.bfloat16, mx.float16)
        and module.biases.dtype == module.scales.dtype
        and input_dims % 64 == 0
        and output_dims % 8 == 0
        and tuple(module.scales.shape) == tuple(module.biases.shape)
        and tuple(module.weight.shape[:2]) == tuple(module.scales.shape[:2])
        and int(module.scales.shape[2]) == input_dims // 64
        and int(module.weight.shape[2]) == _packed_width(input_dims, module.bits)
        and 1 <= routes <= 36
    ):
        raise ValueError("unsupported exact affine down-route QMV layout")
    return routes, topk, input_dims, output_dims


def affine2_route_down_qmv(module, inputs: mx.array, indices: mx.array) -> mx.array:
    routes, topk, input_dims, output_dims = _shape(module, inputs, indices)
    flat_indices = mx.contiguous(indices.astype(mx.uint32).reshape(-1))
    (output,) = _kernel()(
        inputs=[inputs, module.weight, module.scales, module.biases, flat_indices],
        template=[
            ("T", inputs.dtype),
            ("Q", module.scales.dtype),
            ("ROUTES", routes),
            ("TOPK", topk),
            ("K", input_dims),
            ("OUT", output_dims),
            ("BITS", int(module.bits)),
            ("VPT", _qmv_vpt(int(module.bits), input_dims, output_dims)),
        ],
        grid=(32, output_dims // 4, routes),
        threadgroup=(32, 2, 1),
        output_shapes=[(routes, output_dims)],
        output_dtypes=[inputs.dtype],
    )
    result: mx.array = output.reshape((int(inputs.shape[0]), topk, output_dims))
    return result
