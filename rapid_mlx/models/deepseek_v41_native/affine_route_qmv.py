"""Experimental exact down-projection affine 2-bit QMV for V4.1.

The verification batch has only 24--36 routed rows spread over hundreds of
experts.  A block GEMM therefore pads most expert groups to a much larger row
tile.  This kernel instead assigns one QMV grid to each real route.  It is
restricted to the mixed-dtype FP32 activation / affine-weight down projection,
where it is bit-exact with the stock path.  Gate/up deliberately stay stock.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_SOURCE = r"""
    // Match the stock affine 2-bit QMV lane geometry and operation order:
    // one uint32 pack (16 values) per lane and a 512-wide K block.
    constexpr int VALUES_PER_THREAD = 16;
    constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constexpr int RESULTS_PER_SIMDGROUP = 4;
    constexpr int OUTPUTS_PER_THREADGROUP = 8;

    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int route = int(threadgroup_position_in_grid.z);
    const int output_base =
        int(threadgroup_position_in_grid.y) * OUTPUTS_PER_THREADGROUP
        + int(simd_group) * RESULTS_PER_SIMDGROUP;
    if (route >= ROUTES || output_base >= OUT) {
        return;
    }

    const int token = route / TOPK;
    const uint expert = indices[route];
    const device T* input_ptr = input + token * K + int(lane) * VALUES_PER_THREAD;
    device T* output_ptr = output + route * OUT + output_base;

    constexpr int PACKED_K_BYTES = K / 4;
    constexpr int GROUPS = K / 64;
    float accum[RESULTS_PER_SIMDGROUP] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        float input_values[VALUES_PER_THREAD];
        float input_sum = 0.0f;
        const bool active_lane = k + int(lane) * VALUES_PER_THREAD < K;
        if (active_lane) {
            for (int idx = 0; idx < VALUES_PER_THREAD; idx += 4) {
                const float x0 = float(input_ptr[idx]);
                const float x1 = float(input_ptr[idx + 1]);
                const float x2 = float(input_ptr[idx + 2]);
                const float x3 = float(input_ptr[idx + 3]);
                input_sum += x0 + x1 + x2 + x3;
                input_values[idx] = x0;
                input_values[idx + 1] = x1 / 4.0f;
                input_values[idx + 2] = x2 / 16.0f;
                input_values[idx + 3] = x3 / 64.0f;
            }
        }
        const int group = k / 64 + int(lane) / 4;
        const int packed_col = k / 4 + int(lane) * 4;
        for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
            const int out_col = output_base + result;
            if (out_col >= OUT) {
                continue;
            }
            if (!active_lane) {
                continue;
            }
            const device uint8_t* row_weight =
                reinterpret_cast<const device uint8_t*>(weight)
                + (size_t(expert) * OUT + out_col) * PACKED_K_BYTES
                + packed_col;
            const Q scale = scales[(size_t(expert) * OUT + out_col) * GROUPS + group];
            const Q bias = biases[(size_t(expert) * OUT + out_col) * GROUPS + group];
            float dot = 0.0f;
            for (int pack = 0; pack < 4; ++pack) {
                const uint value = row_weight[pack];
                const int base = 4 * pack;
                dot +=
                    input_values[base] * float(value & 0x03) +
                    input_values[base + 1] * float(value & 0x0c) +
                    input_values[base + 2] * float(value & 0x30) +
                    input_values[base + 3] * float(value & 0xc0);
            }
            accum[result] += float(scale) * dot + float(bias) * input_sum;
        }
        input_ptr += BLOCK_SIZE;
    }

    for (int result = 0; result < RESULTS_PER_SIMDGROUP; ++result) {
        const float value = simd_sum(accum[result]);
        if (lane == 0 && output_base + result < OUT) {
            output_ptr[result] = T(value);
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
        and int(module.bits) == 2
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
        and int(module.weight.shape[2]) * 16 == input_dims
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
        ],
        grid=(32, output_dims // 4, routes),
        threadgroup=(32, 2, 1),
        output_shapes=[(routes, output_dims)],
        output_dtypes=[inputs.dtype],
    )
    return output.reshape((int(inputs.shape[0]), topk, output_dims))
