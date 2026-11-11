# SPDX-License-Identifier: Apache-2.0
# The GEMV, softmax and sigmoid arithmetic adapts MLX Metal kernels.
# MIT License
#
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Model-local single-token Metal routing and weighted expert reduction.

Install after loading weights. Larger batches, prefill, sharded blocks and
unsupported gate layouts retain the original block call. No weights are copied.
Set RAPID_MLX_MOE_DECODE_FUSION=0 to disable installation and execution.
"""

from __future__ import annotations

from functools import cache, lru_cache

from .._env import env_falsey

_LOGITS = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint first = threadgroup_position_in_grid.x * 4;
    float acc[4] = {0};
    for (uint k = (sg * 32 + lane) * 4; k < H; k += BN * 128) {
        for (uint r = 0; r < 4; ++r) {
            for (uint j = 0; j < 4; ++j) {
                acc[r] += float(w[(first + r) * H + k + j]) * float(x[k + j]);
            }
        }
    }
    for (uint r = 0; r < 4; ++r) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) {
            acc[r] += simd_shuffle_down(acc[r], delta);
        }
    }
    threadgroup float partial[BN][4];
    if (lane == 0) {
        for (uint r = 0; r < 4; ++r) partial[sg][r] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (uint r = 0; r < 4; ++r) {
            for (uint q = 1; q < BN; ++q) acc[r] += partial[q][r];
            logits[first + r] = float(G(acc[r]));
        }
    }
"""

_ROUTE = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint PER = E / 32;
    threadgroup float probs[E];
    threadgroup float red[2];
    float values[4];
    float maximum = -INFINITY;
    for (uint j = 0; j < 4; ++j) {
        uint index = tid * 4 + j;
        values[j] = index < E ? float(logits[index]) : -INFINITY;
        maximum = max(maximum, values[j]);
    }
    maximum = simd_max(maximum);
    if (lane == 0) red[sg] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (E > 128) maximum = max(red[0], red[1]);
    float total = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        values[j] = metal::fast::exp(values[j] - maximum);
        total += values[j];
    }
    total = simd_sum(total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (E > 128) total = red[0] + red[1];
    float reciprocal = 1.0f / total;
    for (uint j = 0; j < 4; ++j) {
        uint index = tid * 4 + j;
        if (index < E) {
            float probability = float(values[j] * reciprocal);
            if (BIAS) probability += float(bias[index]);
            probs[index] = probability;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float p[PER];
    for (uint j = 0; j < PER; ++j) p[j] = probs[lane * PER + j];
    float picked[K];
    uint indices[K];
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint index = 0;
        for (uint j = 0; j < PER; ++j) {
            if (p[j] >= best) { best = p[j]; index = lane * PER + j; }
        }
        float winner = simd_max(best);
        uint wi = simd_max(best == winner ? index : 0u);
        if (wi / PER == lane) p[wi % PER] = -INFINITY;
        picked[K - 1 - k] = winner;
        indices[K - 1 - k] = wi;
    }
    float denom = 0.0f;
    for (uint j = 0; j < K; ++j) denom += picked[j];
    float divisor = denom + 1e-20f;
    if (lane == 0) {
        for (uint j = 0; j < K; ++j) {
            inds[j] = indices[j];
            scores[j] = NORMALIZE ? picked[j] / divisor : picked[j];
        }
    }
"""

_COMBINE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= H) return;
    T acc = T(0);
    for (uint j = 0; j < K; ++j) {
        acc += T(float(y[j * H + i]) * float(scores[j]));
    }
    out[i] = T(acc);
"""


@lru_cache(maxsize=3)
def _kernel(kind: str):
    import mlx.core as mx

    source, inputs, outputs = {
        "logits": (_LOGITS, ["x", "w"], ["logits"]),
        "route": (_ROUTE, ["logits", "bias"], ["inds", "scores"]),
        "combine": (_COMBINE, ["y", "scores"], ["out"]),
    }[kind]
    return mx.fast.metal_kernel(
        name=f"rapid_moe_decode_{kind}",
        input_names=inputs,
        output_names=outputs,
        source=source,
        ensure_row_contiguous=True,
    )


def route(block, x):
    """Parallel gate GEMV, followed by fused softmax/selection/normalization."""
    import mlx.core as mx

    e, h = block.num_experts, x.shape[-1]
    k = block.top_k
    gate_dtype = mx.result_type(x, block.gate.weight)
    # Match the dependency's GEMV lane split and accumulation order.
    bn = 8 if h >= 16 * e else 1
    (logits,) = _kernel("logits")(
        inputs=[x, block.gate.weight],
        template=[("H", h), ("G", gate_dtype), ("BN", bn)],
        grid=(e // 4 * bn * 32, 1, 1),
        threadgroup=(bn * 32, 1, 1),
        output_shapes=[(e,)],
        output_dtypes=[mx.float32],
    )
    biased = block.use_expert_bias
    # Unused by the no-bias specialization; reuse an existing buffer.
    bias = block.expert_bias if biased else logits
    indices, scores = _kernel("route")(
        inputs=[logits, bias],
        template=[
            ("E", e),
            ("K", k),
            ("BIAS", biased),
            ("NORMALIZE", getattr(block, "norm_topk_prob", True)),
        ],
        grid=(((e + 127) // 128) * 32, 1, 1),
        threadgroup=(((e + 127) // 128) * 32, 1, 1),
        output_shapes=[(*x.shape[:-1], k)] * 2,
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, scores.astype(x.dtype)


def combine(y, scores):
    """Round products and each addition in the promoted dtype, as upstream."""
    import mlx.core as mx

    h, k = y.shape[-1], y.shape[-2]
    dtype = mx.result_type(y, scores)
    template = [("H", h), ("K", k), ("T", dtype)]
    inputs = [y, scores]
    (output,) = _kernel("combine")(
        inputs=inputs,
        template=template,
        grid=(((h + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*y.shape[:-2], h)],
        output_dtypes=[dtype],
    )
    return output


def eligible(block, x) -> bool:
    """Restrict fusion to the supported single-row unsharded inference path."""
    if env_falsey("RAPID_MLX_MOE_DECODE_FUSION"):
        return False
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        return False

    gate = getattr(block, "gate", None)
    if gate is None or type(gate) is not nn.Linear:
        return False
    w = gate.weight
    return (
        x.ndim == 3
        and x.shape[0] * x.shape[1] == 1
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and "bias" not in block.gate
        and w.ndim == 2
        and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and w.dtype in (mx.float32, mx.bfloat16)
        and w.shape == (block.num_experts, x.shape[-1])
        and x.shape[-1] % 128 == 0
        and 32 <= block.num_experts <= 256
        and block.num_experts % 32 == 0
        and 1 <= block.top_k <= min(8, block.num_experts)
        and getattr(block, "sharding_group", None) is None
        and (
            not getattr(block, "use_expert_bias", False)
            or (
                block.expert_bias.shape == (block.num_experts,)
                and block.expert_bias.dtype == mx.float32
            )
        )
    )


@cache
def _decode_type(base, original):

    def call(self, x):
        if not eligible(self, x):
            return original(self, x)
        indices, scores = route(self, x)
        y = self.switch_mlp(x, indices)
        return combine(y, scores)

    return type(f"RapidDecode{base.__name__}", (base,), {"__call__": call})


def _block_types():
    from mlx_lm.models.lfm2_moe import Lfm2MoeSparseMoeBlock

    return {
        Lfm2MoeSparseMoeBlock: _decode_type(
            Lfm2MoeSparseMoeBlock, Lfm2MoeSparseMoeBlock.__call__
        ),
    }


def install_moe_decode_fusion(model) -> int:
    """Specialize supported blocks on this loaded model only, keeping weights."""
    if env_falsey("RAPID_MLX_MOE_DECODE_FUSION"):
        return 0
    try:
        import mlx.core as mx
    except ImportError:
        return 0

    if not mx.metal.is_available():
        return 0
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return 0
    types = _block_types()
    count = 0
    for _, block in named_modules():
        replacement = types.get(type(block))
        if replacement is not None:
            block.__class__ = replacement
            count += 1
    return count
