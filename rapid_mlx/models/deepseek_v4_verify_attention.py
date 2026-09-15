# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant short-row attention for embedded DeepSeek DSpark.

DeepSeek-V4 uses 64 query heads and one KV head.  MLX collapses those heads
into GEMM's M dimension for ordinary one-token decode, but a DSpark verify
batch with one physical KV view per row takes the batched GEMV route instead.
The two routes are mathematically equivalent but not bitwise equivalent.

These kernels keep the per-head reduction fixed for both decode and verify.
Multiple verify rows share one dispatch while retaining independent KV views.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import mlx.core as mx


_ROWWISE_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int output_index = int(threadgroup_position_in_grid.y);
    const int row = output_index / (M * N);
    const int rem = output_index - row * M * N;
    const int m = rem / N;
    const int n = rem - m * N;
    float accum = 0.0f;
    for (int k = int(lane); k < K; k += 32) {
        const int lhs_idx = (row * M + m) * K + k;
        const int rhs_idx = TRANSPOSE_RHS
            ? (row * N + n) * K + k
            : (row * K + k) * N + n;
        accum += float(lhs[lhs_idx]) * float(rhs[rhs_idx]);
    }
    accum = simd_sum(accum);
    if (lane == 0) {
        output[output_index] = T(accum);
    }
"""


@lru_cache(maxsize=1)
def _rowwise_kernel():
    return mx.fast.metal_kernel(
        name="rapid_dspark_rowwise_gemm",
        input_names=["lhs", "rhs"],
        output_names=["output"],
        source=_ROWWISE_SOURCE,
        ensure_row_contiguous=True,
    )


def _metal_rowwise_gemm(
    lhs: mx.array, rhs: mx.array, transpose_rhs: bool
) -> mx.array:
    rows, m, k = map(int, lhs.shape)
    n = int(rhs.shape[1] if transpose_rhs else rhs.shape[2])
    (output,) = _rowwise_kernel()(
        inputs=[lhs, rhs],
        template=[
            ("T", lhs.dtype),
            ("M", m),
            ("N", n),
            ("K", k),
            ("TRANSPOSE_RHS", transpose_rhs),
        ],
        grid=(32, rows * m * n, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, m, n)],
        output_dtypes=[lhs.dtype],
    )
    return output


def rowwise_gemm(lhs: mx.array, rhs: mx.array, transpose_rhs: bool) -> mx.array:
    """Dispatch independent M=64 Steel GEMMs in one grid when available."""
    try:
        architecture = str(mx.device_info().get("architecture", ""))
        if architecture.endswith(("d", "g", "p")):
            from omlx.custom_kernels.glm_moe_dsa import fast

            if fast.has_symbol("dspark_rowwise_gemm"):
                return fast.dspark_rowwise_gemm(
                    mx.contiguous(lhs),
                    mx.contiguous(rhs),
                    transpose_rhs,
                )
    except Exception:
        pass
    return _metal_rowwise_gemm(lhs, rhs, transpose_rhs)


def exact_attention(
    queries: mx.array,
    key_rows: Sequence[mx.array],
    scale: float,
    sinks: mx.array | None,
) -> mx.array:
    """Attend M query rows to M same-length KV snapshots.

    ``queries`` is ``[1, H, M, D]`` and every key row is ``[1, 1, S, D]``.
    The return layout is ``[1, H, M, D]``.  Different key lengths are kept
    correct with small same-length groups rather than padding the softmax.
    """
    if len(key_rows) != int(queries.shape[2]):
        raise ValueError("one DeepSeek KV view is required for each query row")

    scaled_rows = queries * mx.array(scale, dtype=queries.dtype)
    outputs = []
    start = 0
    while start < len(key_rows):
        stop = start + 1
        group_length = int(key_rows[start].shape[2])
        while (
            stop < len(key_rows)
            and int(key_rows[stop].shape[2]) == group_length
        ):
            stop += 1

        query_batch = scaled_rows[0, :, start:stop].transpose(1, 0, 2)
        keys = mx.concatenate(key_rows[start:stop], axis=0)[:, 0]
        scores = rowwise_gemm(query_batch, keys, True)

        if sinks is None:
            weights = mx.softmax(scores, axis=-1, precise=True)
        else:
            sink_scores = mx.broadcast_to(
                sinks.astype(scores.dtype)[None, :, None],
                (*scores.shape[:-1], 1),
            )
            weights = mx.softmax(
                mx.concatenate([sink_scores, scores], axis=-1),
                axis=-1,
                precise=True,
            )[..., 1:]
        group_output = rowwise_gemm(weights, keys, False)
        outputs.append(group_output.transpose(1, 0, 2)[None])
        start = stop

    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def exact_local_scores(
    queries: mx.array,
    key_rows: Sequence[mx.array],
    scale: float,
) -> mx.array:
    """Return batch-invariant local scores for sparse split attention."""
    if len(key_rows) != int(queries.shape[2]):
        raise ValueError("one DeepSeek KV view is required for each query row")
    if any(int(row.shape[2]) != int(key_rows[0].shape[2]) for row in key_rows):
        raise ValueError("sparse local KV views must have equal lengths")
    query_batch = (
        queries * mx.array(scale, dtype=queries.dtype)
    )[0].transpose(1, 0, 2)
    keys = mx.concatenate(key_rows, axis=0)[:, 0]
    return rowwise_gemm(query_batch, keys, True).transpose(1, 0, 2)[None]


def exact_local_values(weights: mx.array, key_rows: Sequence[mx.array]) -> mx.array:
    """Apply sparse local weights with the same decode/verify reduction."""
    weight_rows = weights[0].transpose(1, 0, 2)
    keys = mx.concatenate(key_rows, axis=0)[:, 0]
    return rowwise_gemm(weight_rows, keys, False).transpose(1, 0, 2)[None]


__all__ = [
    "exact_attention",
    "exact_local_scores",
    "exact_local_values",
    "rowwise_gemm",
]
