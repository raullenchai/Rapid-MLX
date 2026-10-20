"""Exact FP32-activation / BF16-weight vocabulary-head GEMV for V4.1.

The head keeps the reference's FP32 logits: ``head(h.astype(float32))`` with a
BF16 weight.  The stock path promotes the whole 129280 x 5120 weight to FP32 on
every call, so one decode step reads ~1.3 GB, writes ~2.6 GB and reads it back
before the GEMV even starts.  This kernel converts each weight element in
registers instead and reproduces the stock FP32 GEMV lane geometry for this
shape (32 lanes across K, four contiguous values per lane, one simdgroup per
output, shuffle-down tree reduction), so a single row is bit-exact with the
stock path.  Up to ``MAX_ROWS`` rows (batched lanes, speculative verify) run
the same per-row arithmetic, i.e. each row equals its own single-row result.
Larger batches (prefill) stay on the stock path.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

# Module switch: False routes every call through the stock projection.
ENABLED = True
MAX_ROWS = 8
_LANES = 32
_VALUES_PER_LANE = 4
_BLOCK_K = _LANES * _VALUES_PER_LANE
_OUTPUTS_PER_THREADGROUP = 4

_SOURCE = r"""
    // Stock FP32 GEMV order for K % 128 == 0: lane-strided blocks of four,
    // fused multiply-add in element order, then a shuffle-down tree.
    constexpr int BLOCK_K = 128;
    const uint lane = thread_index_in_simdgroup;
    const int out_col = int(threadgroup_position_in_grid.x) * OUTPUTS
        + int(simdgroup_index_in_threadgroup);
    const int row = int(threadgroup_position_in_grid.y);
    if (out_col >= N) {
        return;
    }
    const device bfloat16_t* weight_row = weight + size_t(out_col) * K;
    const device float* input_row = input + size_t(row) * K;
    float result = 0.0f;
    int offset = int(lane) * 4;
    for (int block = 0; block < K / BLOCK_K; ++block) {
        float values[4];
        float weights[4];
        for (int index = 0; index < 4; ++index) {
            values[index] = input_row[offset + index];
        }
        for (int index = 0; index < 4; ++index) {
            weights[index] = float(weight_row[offset + index]);
        }
        for (int index = 0; index < 4; ++index) {
            result += weights[index] * values[index];
        }
        offset += BLOCK_K;
    }
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result += simd_shuffle_down(result, delta);
    }
    if (lane == 0) {
        output[size_t(row) * N + out_col] = result;
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="rapid_v41_fp32_bf16_head_gemv",
        input_names=["input", "weight"],
        output_names=["output"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def _gpu_default() -> bool:
    return bool(mx.default_device() == mx.gpu)


def supports(head, x: mx.array) -> bool:
    """True when ``head(x)`` can take the exact GEMV path."""
    if not ENABLED or not _gpu_default():
        return False
    weight = getattr(head, "weight", None)
    if (
        type(head) is not nn.Linear
        or "bias" in head
        or weight is None
        or weight.ndim != 2
        or weight.dtype != mx.bfloat16
        or x.dtype != mx.float32
        or x.ndim < 1
        or x.shape[-1] != weight.shape[1]
        or weight.shape[1] % _BLOCK_K
    ):
        return False
    rows = x.size // x.shape[-1] if x.shape[-1] else 0
    return 1 <= rows <= MAX_ROWS


def head_logits(head, x: mx.array) -> mx.array:
    """FP32 logits for ``head(x)``; exact GEMV when supported, stock otherwise."""
    if not supports(head, x):
        return head(x)
    weight = head.weight
    outputs, inputs = int(weight.shape[0]), int(weight.shape[1])
    rows = x.size // inputs
    (logits,) = _kernel()(
        inputs=[x.reshape(rows, inputs), weight],
        template=[("K", inputs), ("N", outputs), ("OUTPUTS", _OUTPUTS_PER_THREADGROUP)],
        grid=(
            -(-outputs // _OUTPUTS_PER_THREADGROUP) * _LANES * _OUTPUTS_PER_THREADGROUP,
            rows,
            1,
        ),
        threadgroup=(_LANES * _OUTPUTS_PER_THREADGROUP, 1, 1),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.float32],
    )
    return logits.reshape(*x.shape[:-1], outputs)
