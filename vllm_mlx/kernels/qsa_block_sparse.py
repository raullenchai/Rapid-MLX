# SPDX-License-Identifier: Apache-2.0
"""Opt-in block-sparse QSA prefill attention for Apple Silicon."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

ENABLE_ENV = "RAPID_MLX_QSA_BLOCK_SPARSE"
MIN_QUERY_LENGTH = 64
MIN_PHYSICAL_KV_LENGTH = 16_384
MAX_GQA_HEADS = 32
MAX_THREADGROUP_MEMORY_BYTES = 32 * 1024

_SOURCE = r"""
    constexpr int VALUES_PER_LANE = HEAD_DIM / 32;
    constexpr int THREADS = GQA_HEADS * 32;

    // Prompt geometry is a runtime uniform so one compiled pipeline serves
    // every prefill chunk width instead of retaining one library per shape.
    const int query_length = dims[0];
    const int key_length = dims[1];

    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const int query_index = int(threadgroup_position_in_grid.x);
    const int batch_kv = int(threadgroup_position_in_grid.y);
    const int batch = batch_kv / KV_HEADS;
    const int kv_head = batch_kv - batch * KV_HEADS;
    const int query_head = kv_head * GQA_HEADS + int(simdgroup);

    threadgroup T shared_keys[BLOCK_SIZE * HEAD_DIM];
    threadgroup T shared_values[BLOCK_SIZE * HEAD_DIM];

    float query_values[VALUES_PER_LANE];
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float output_values[VALUES_PER_LANE];

    for (int value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        const int dim = int(lane) + value_index * 32;
        const int offset =
            ((batch * QUERY_HEADS + query_head) * query_length + query_index)
            * HEAD_DIM + dim;
        query_values[value_index] = float(queries[offset]);
        output_values[value_index] = 0.0f;
    }

    const int query_offset = batch * query_length + query_index;
    const int raw_block_count = block_counts[query_offset];
    const int block_count = raw_block_count < 0
        ? 0
        : (raw_block_count > BLOCK_TOPK ? BLOCK_TOPK : raw_block_count);
    const int block_base = query_offset * BLOCK_TOPK;
    for (int block_index = 0; block_index < block_count; ++block_index) {
        const int physical_start = block_starts[block_base + block_index];
        if (physical_start < 0 || physical_start > key_length - BLOCK_SIZE) {
            continue;
        }
        for (int element = int(tid); element < BLOCK_SIZE * HEAD_DIM;
             element += THREADS) {
            const int token = element / HEAD_DIM;
            const int dim = element - token * HEAD_DIM;
            const int key_index = physical_start + token;
            const int offset =
                ((batch * KV_HEADS + kv_head) * key_length + key_index)
                * HEAD_DIM + dim;
            shared_keys[element] = keys[offset];
            shared_values[element] = values[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int token = 0; token < BLOCK_SIZE; ++token) {
            float partial = 0.0f;
            for (int value_index = 0; value_index < VALUES_PER_LANE;
                 ++value_index) {
                const int dim = int(lane) + value_index * 32;
                partial += query_values[value_index]
                    * float(shared_keys[token * HEAD_DIM + dim]);
            }
            const float score = simd_sum(partial)
                * metal::rsqrt(float(HEAD_DIM));
            const float next_max = metal::max(running_max, score);
            const float old_weight = metal::exp(running_max - next_max);
            const float new_weight = metal::exp(score - next_max);
            running_sum = running_sum * old_weight + new_weight;
            for (int value_index = 0; value_index < VALUES_PER_LANE;
                 ++value_index) {
                const int dim = int(lane) + value_index * 32;
                output_values[value_index] =
                    output_values[value_index] * old_weight
                    + new_weight * float(shared_values[token * HEAD_DIM + dim]);
            }
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int raw_tail_count = tail_counts[query_offset];
    const int tail_count = raw_tail_count < 0
        ? 0
        : (raw_tail_count > BLOCK_SIZE ? BLOCK_SIZE : raw_tail_count);
    const int tail_base = query_offset * BLOCK_SIZE;
    for (int tail_index = 0; tail_index < tail_count; ++tail_index) {
        const int key_index = tail_indices[tail_base + tail_index];
        if (key_index < 0 || key_index >= key_length) {
            continue;
        }
        for (int dim = int(tid); dim < HEAD_DIM; dim += THREADS) {
            const int offset =
                ((batch * KV_HEADS + kv_head) * key_length + key_index)
                * HEAD_DIM + dim;
            shared_keys[dim] = keys[offset];
            shared_values[dim] = values[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float partial = 0.0f;
        for (int value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
            const int dim = int(lane) + value_index * 32;
            partial += query_values[value_index] * float(shared_keys[dim]);
        }
        const float score = simd_sum(partial) * metal::rsqrt(float(HEAD_DIM));
        const float next_max = metal::max(running_max, score);
        const float old_weight = metal::exp(running_max - next_max);
        const float new_weight = metal::exp(score - next_max);
        running_sum = running_sum * old_weight + new_weight;
        for (int value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
            const int dim = int(lane) + value_index * 32;
            output_values[value_index] = output_values[value_index] * old_weight
                + new_weight * float(shared_values[dim]);
        }
        running_max = next_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        const int dim = int(lane) + value_index * 32;
        const int output_index =
            ((batch * QUERY_HEADS + query_head) * query_length + query_index)
            * HEAD_DIM + dim;
        output[output_index] = running_sum == 0.0f
            ? T(0.0f)
            : T(output_values[value_index] / running_sum);
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="rapid_qsa_block_sparse",
        input_names=[
            "queries",
            "keys",
            "values",
            "block_starts",
            "block_counts",
            "tail_indices",
            "tail_counts",
            "dims",
        ],
        output_names=["output"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def block_sparse_decline_reason(
    query_length: int,
    physical_kv_length: int,
    *,
    training: bool = False,
) -> str | None:
    """Return ``None`` when the opt-in route is eligible, else why it declined."""
    enabled = os.environ.get(ENABLE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return "disabled"
    if training:
        return "training"
    if query_length < MIN_QUERY_LENGTH:
        return "query below crossover"
    if physical_kv_length < MIN_PHYSICAL_KV_LENGTH:
        return "physical KV below crossover"
    if not mx.metal.is_available():
        return "Metal runtime unavailable"
    return None


def block_sparse_layout_supported(
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: mx.Dtype,
) -> bool:
    """Return whether the layout fits the kernel and Metal memory limits."""
    if kv_heads <= 0 or block_size <= 0 or head_dim <= 0:
        return False
    if head_dim % 32 or query_heads % kv_heads:
        return False
    gqa_heads = query_heads // kv_heads
    if not 0 < gqa_heads <= MAX_GQA_HEADS:
        return False
    if dtype not in {mx.float16, mx.bfloat16, mx.float32}:
        return False
    threadgroup_bytes = 2 * block_size * head_dim * int(dtype.size)
    return threadgroup_bytes <= MAX_THREADGROUP_MEMORY_BYTES


@lru_cache(maxsize=1)
def _log_activation() -> None:
    logger.info(
        "QSA block-sparse prefill enabled (query>=%d, physical_kv>=%d)",
        MIN_QUERY_LENGTH,
        MIN_PHYSICAL_KV_LENGTH,
    )


def block_sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_starts: mx.array,
    block_counts: mx.array,
    tail_indices: mx.array,
    tail_counts: mx.array,
    *,
    block_size: int,
) -> mx.array:
    """Attend to sorted selected blocks plus the current incomplete tail."""
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("QSA query/K/V arrays must be rank four")
    if block_size <= 0:
        raise ValueError("QSA block size must be positive")
    batch, query_heads, query_length, head_dim = map(int, queries.shape)
    key_batch, kv_heads, key_length, key_dim = map(int, keys.shape)
    expected_rows = (batch, query_length)
    if block_starts.ndim != 3 or tuple(block_starts.shape[:2]) != expected_rows:
        raise ValueError("QSA block starts must have shape [batch, query, topk]")
    if tuple(block_counts.shape) != expected_rows:
        raise ValueError("QSA block counts must have shape [batch, query]")
    if tuple(tail_indices.shape) != (*expected_rows, block_size):
        raise ValueError("QSA tail indices must have shape [batch, query, block_size]")
    if tuple(tail_counts.shape) != expected_rows:
        raise ValueError("QSA tail counts must have shape [batch, query]")
    compact_arrays = {
        "block starts": block_starts,
        "block counts": block_counts,
        "tail indices": tail_indices,
        "tail counts": tail_counts,
    }
    invalid_dtypes = [
        name for name, array in compact_arrays.items() if array.dtype != mx.int32
    ]
    if invalid_dtypes:
        raise ValueError(
            "QSA compact indices and counts must use int32: "
            + ", ".join(invalid_dtypes)
        )
    block_topk = int(block_starts.shape[-1])
    if key_batch != batch or key_dim != head_dim or values.shape != keys.shape:
        raise ValueError("QSA query/KV shapes are inconsistent")
    if queries.dtype != keys.dtype or queries.dtype != values.dtype:
        raise ValueError("QSA query/K/V arrays must have the same dtype")
    if head_dim % 32:
        raise ValueError("QSA head dimension must be divisible by 32")
    if kv_heads <= 0:
        raise ValueError("QSA requires at least one KV head")
    if query_heads % kv_heads:
        raise ValueError("QSA query heads must be divisible by KV heads")
    gqa_heads = query_heads // kv_heads
    if gqa_heads > MAX_GQA_HEADS:
        raise ValueError(
            f"QSA supports at most {MAX_GQA_HEADS} query heads per KV head"
        )
    if not block_sparse_layout_supported(
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=queries.dtype,
    ):
        raise ValueError("QSA query/KV layout is unsupported by the sparse kernel")
    _log_activation()
    (output,) = _kernel()(
        inputs=[
            queries,
            keys,
            values,
            block_starts,
            block_counts,
            tail_indices,
            tail_counts,
            mx.array([query_length, key_length], dtype=mx.int32),
        ],
        template=[
            ("T", queries.dtype),
            ("QUERY_HEADS", query_heads),
            ("KV_HEADS", kv_heads),
            ("GQA_HEADS", gqa_heads),
            ("HEAD_DIM", head_dim),
            ("BLOCK_SIZE", block_size),
            ("BLOCK_TOPK", block_topk),
        ],
        grid=(gqa_heads * 32 * query_length, batch * kv_heads, 1),
        threadgroup=(gqa_heads * 32, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return output
