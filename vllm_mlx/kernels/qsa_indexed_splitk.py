# SPDX-License-Identifier: Apache-2.0
"""Opt-in indexed split-K attention for narrow QSA decode/verify calls.

The compact QSA selection is consumed directly: K/V stay in their physical
cache and each split walks logical selected-token ordinals.  The two-pass
reduction deliberately mirrors MLX's ``sdpa_vector_2pass`` accumulator types,
``metal::fast::exp`` calls, bf16/fp16 partials, and second-pass reduction order.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import mlx.core as mx

logger = logging.getLogger(__name__)

ENABLE_ENV = "RAPID_MLX_QSA_INDEXED_SPLITK"
QUALIFIED_MLX_VERSIONS = frozenset({"0.32.2"})
QUALIFIED_METAL_ARCHITECTURES = frozenset({"applegpu_g15d"})
MAX_QUERY_LENGTH = 3
QUALIFIED_QUERY_LENGTHS = frozenset({1, 3})
MIN_KV_LENGTH_VERIFY = 16_384
MIN_KV_LENGTH_DECODE = 65_536
QUALIFIED_QUERY_HEADS = 24
QUALIFIED_KV_HEADS = 2
QUALIFIED_HEAD_DIM = 256
QUALIFIED_DTYPE = mx.bfloat16
QUALIFIED_BLOCK_SIZE = 4
QUALIFIED_BLOCK_TOPK = 512
MAX_GQA_HEADS = 32
SIMD_WIDTH = 32


@lru_cache(maxsize=1)
def _mlx_version() -> str:
    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def _metal_architecture() -> str:
    return str(mx.device_info().get("architecture", "unknown"))


def _enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def indexed_splitk_decline_reason(
    query_length: int,
    physical_kv_length: int,
    *,
    batch_size: int,
    training: bool = False,
    mlx_version: str | None = None,
    metal_architecture: str | None = None,
) -> str | None:
    """Return ``None`` only for the narrow, qualified opt-in route."""
    if not _enabled():
        return "disabled"
    if training:
        return "training"
    if batch_size != 1:
        return "batch size is not qualified"
    if query_length not in QUALIFIED_QUERY_LENGTHS:
        return "query length is not qualified"
    crossover = MIN_KV_LENGTH_DECODE if query_length == 1 else MIN_KV_LENGTH_VERIFY
    if physical_kv_length < crossover:
        return "physical KV below crossover"
    installed_version = _mlx_version() if mlx_version is None else mlx_version
    if installed_version not in QUALIFIED_MLX_VERSIONS:
        return f"unqualified MLX version {installed_version}"
    if not mx.metal.is_available():
        return "Metal runtime unavailable"
    architecture = (
        _metal_architecture() if metal_architecture is None else metal_architecture
    )
    if architecture not in QUALIFIED_METAL_ARCHITECTURES:
        return f"unqualified Metal architecture {architecture}"
    return None


def indexed_splitk_layout_supported(
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    block_topk: int,
    dtype: mx.Dtype,
) -> bool:
    """Return whether this is the exact production-qualified Qwen3.8 layout."""
    return (
        query_heads == QUALIFIED_QUERY_HEADS
        and kv_heads == QUALIFIED_KV_HEADS
        and head_dim == QUALIFIED_HEAD_DIM
        and block_size == QUALIFIED_BLOCK_SIZE
        and block_topk == QUALIFIED_BLOCK_TOPK
        and dtype == QUALIFIED_DTYPE
    )


def _kernel_layout_supported(
    *, query_heads: int, kv_heads: int, head_dim: int, dtype: mx.Dtype
) -> bool:
    """Validate the broader set of layouts that the Metal kernel can represent."""
    if kv_heads <= 0 or head_dim <= 0 or query_heads % kv_heads:
        return False
    gqa_heads = query_heads // kv_heads
    if not 0 < gqa_heads <= MAX_GQA_HEADS:
        return False
    if gqa_heads * SIMD_WIDTH > 1024 or head_dim % SIMD_WIDTH:
        return False
    return dtype in {mx.float16, mx.bfloat16, mx.float32}


def _split_count(physical_kv_length: int, query_length: int) -> int:
    """Match MLX's M3-Ultra schedule for the gathered 2K QSA reference."""
    del physical_kv_length, query_length
    return 128


_PASS1_SOURCE = r"""
    constexpr int VALUES_PER_LANE = HEAD_DIM / 32;
    constexpr int THREADS = GQA_HEADS * 32;

    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const int query_index = int(threadgroup_position_in_grid.x);
    const int batch_kv = int(threadgroup_position_in_grid.y);
    const int split = int(threadgroup_position_in_grid.z);
    const int batch = batch_kv / KV_HEADS;
    const int kv_head = batch_kv - batch * KV_HEADS;
    const int query_head = kv_head * GQA_HEADS + int(simdgroup);
    const int query_length = dims[0];
    const int key_length = dims[1];
    threadgroup T shared_keys[HEAD_DIM];
    threadgroup T shared_values[HEAD_DIM];
    float q[VALUES_PER_LANE];
    float o[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        // Match MLX sdpa_vector_2pass: a lane owns one contiguous D / 32
        // chunk.  Pass 2 relies on this exact partials layout when it
        // transposes dimensions and split blocks through threadgroup memory.
        const int dim = int(lane) * VALUES_PER_LANE + i;
        const size_t q_offset = batch * queries_strides[0]
            + query_head * queries_strides[1]
            + query_index * queries_strides[2]
            + dim * queries_strides[3];
        q[i] = scale[0] * float(queries[q_offset]);
        o[i] = 0.0f;
    }

    const size_t count_offset = batch * block_counts_strides[0]
        + query_index * block_counts_strides[1];
    const int raw_blocks = block_counts[count_offset];
    const int block_count = metal::max(0, metal::min(raw_blocks, BLOCK_TOPK));
    const size_t tail_count_offset = batch * tail_counts_strides[0]
        + query_index * tail_counts_strides[1];
    const int raw_tail = tail_counts[tail_count_offset];
    const int tail_count = metal::max(0, metal::min(raw_tail, BLOCK_SIZE));
    const int block_tokens = block_count * BLOCK_SIZE;
    const int selected_tokens = block_tokens + tail_count;
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (int ordinal = split; ordinal < selected_tokens; ordinal += SPLITS) {
        int key_index;
        bool valid_key;
        if (ordinal < block_tokens) {
            const int selected_block = ordinal / BLOCK_SIZE;
            const int within_block = ordinal - selected_block * BLOCK_SIZE;
            const size_t block_offset = batch * block_starts_strides[0]
                + query_index * block_starts_strides[1]
                + selected_block * block_starts_strides[2];
            const int block_start = block_starts[block_offset];
            valid_key = block_start >= 0
                && block_start <= key_length - BLOCK_SIZE;
            key_index = valid_key ? block_start + within_block : 0;
        } else {
            const size_t tail_offset = batch * tail_indices_strides[0]
                + query_index * tail_indices_strides[1]
                + (ordinal - block_tokens) * tail_indices_strides[2];
            key_index = tail_indices[tail_offset];
            valid_key = key_index >= 0 && key_index < key_length;
        }
        for (int dim = int(tid); dim < HEAD_DIM; dim += THREADS) {
            if (valid_key) {
                const size_t k_offset = batch * keys_strides[0]
                    + kv_head * keys_strides[1]
                    + key_index * keys_strides[2]
                    + dim * keys_strides[3];
                const size_t v_offset = batch * values_strides[0]
                    + kv_head * values_strides[1]
                    + key_index * values_strides[2]
                    + dim * values_strides[3];
                shared_keys[dim] = keys[k_offset];
                shared_values[dim] = values[v_offset];
            } else {
                shared_keys[dim] = T(0.0f);
                shared_values[dim] = T(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_key) {
            float score = 0.0f;
            for (int i = 0; i < VALUES_PER_LANE; ++i) {
                const int dim = int(lane) * VALUES_PER_LANE + i;
                score += q[i] * float(shared_keys[dim]);
            }
            score = simd_sum(score);
            const float new_max = metal::max(max_score, score);
            const float old_factor = metal::fast::exp(max_score - new_max);
            const float exp_score = metal::fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * old_factor + exp_score;
            for (int i = 0; i < VALUES_PER_LANE; ++i) {
                const int dim = int(lane) * VALUES_PER_LANE + i;
                o[i] = o[i] * old_factor
                    + exp_score * float(shared_values[dim]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int partial_base =
        (((batch * QUERY_HEADS + query_head) * query_length + query_index)
        * SPLITS + split) * HEAD_DIM;
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        partials[partial_base + int(lane) * VALUES_PER_LANE + i] = T(o[i]);
    }
    if (lane == 0) {
        const int stat_offset =
            ((batch * QUERY_HEADS + query_head) * query_length + query_index)
            * SPLITS + split;
        sums[stat_offset] = sum_exp_score;
        maxs[stat_offset] = max_score;
    }
"""


_PASS2_SOURCE = r"""
    constexpr int BN = 32;
    constexpr int VALUES_PER_LANE = HEAD_DIM / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const int batch_head = int(threadgroup_position_in_grid.x);
    const int query_index = int(threadgroup_position_in_grid.y);
    const int query_length = dims[0];
    const int q_offset = batch_head * query_length + query_index;
    const int stats_base = q_offset * SPLITS;
    const int partial_base = stats_base * HEAD_DIM;

    float max_score = -INFINITY;
    for (int b = 0; b < SPLITS / BN; ++b) {
        max_score = metal::max(max_score, maxs[stats_base + int(lane) + BN * b]);
    }
    max_score = simd_max(max_score);

    float sum_exp_score = 0.0f;
    for (int b = 0; b < SPLITS / BN; ++b) {
        const int split = int(lane) + BN * b;
        const float factor = sums[stats_base + split] > 0.0f
            ? metal::fast::exp(maxs[stats_base + split] - max_score)
            : 0.0f;
        sum_exp_score += factor * sums[stats_base + split];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    float o[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        o[i] = 0.0f;
    }
    for (int b = 0; b < SPLITS / BN; ++b) {
        const int split = int(simdgroup) + BN * b;
        const float factor = sums[stats_base + split] > 0.0f
            ? metal::fast::exp(maxs[stats_base + split] - max_score)
            : 0.0f;
        const int offset = partial_base + split * HEAD_DIM
            + int(lane) * VALUES_PER_LANE;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            o[i] += factor * float(partials[offset + i]);
        }
    }

    threadgroup float transpose[BN * BN];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        transpose[int(lane) * BN + int(simdgroup)] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(transpose[int(simdgroup) * BN + int(lane)]);
        o[i] = sum_exp_score == 0.0f ? o[i] : o[i] / sum_exp_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const int output_base = q_offset * HEAD_DIM
            + int(simdgroup) * VALUES_PER_LANE;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            output[output_base + i] = T(o[i]);
        }
    }
"""


@lru_cache(maxsize=1)
def _pass1_kernel():
    return mx.fast.metal_kernel(
        name="rapid_qsa_indexed_splitk_pass1",
        input_names=[
            "queries",
            "keys",
            "values",
            "block_starts",
            "block_counts",
            "tail_indices",
            "tail_counts",
            "dims",
            "scale",
        ],
        output_names=["partials", "sums", "maxs"],
        source=_PASS1_SOURCE,
        # KVCache returns a view over capacity. Copying it on every decode
        # layer defeats indexed reads, so pass 1 consumes MLX-provided strides.
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def _pass2_kernel():
    return mx.fast.metal_kernel(
        name="rapid_qsa_indexed_splitk_pass2",
        input_names=["partials", "sums", "maxs", "dims"],
        output_names=["output"],
        source=_PASS2_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _log_activation() -> None:
    logger.info("QSA indexed split-K attention enabled for narrow decode/verify")


def indexed_splitk_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_starts: mx.array,
    block_counts: mx.array,
    tail_indices: mx.array,
    tail_counts: mx.array,
    *,
    block_size: int,
    scale: float,
    splits: int | None = None,
) -> mx.array:
    """Attend directly through compact, sorted QSA block/tail indices."""
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("QSA query/K/V arrays must be rank four")
    batch, query_heads, query_length, head_dim = map(int, queries.shape)
    key_batch, kv_heads, key_length, key_dim = map(int, keys.shape)
    expected_rows = (batch, query_length)
    if batch != 1:
        raise ValueError("indexed split-K QSA currently requires batch size one")
    if not 1 <= query_length <= MAX_QUERY_LENGTH:
        raise ValueError("indexed split-K QSA requires one to three query tokens")
    if key_batch != batch or key_dim != head_dim or values.shape != keys.shape:
        raise ValueError("QSA query/KV shapes are inconsistent")
    if queries.dtype != keys.dtype or queries.dtype != values.dtype:
        raise ValueError("QSA query/K/V arrays must have the same dtype")
    if block_size <= 0 or tuple(block_counts.shape) != expected_rows:
        raise ValueError("QSA block geometry is inconsistent")
    if block_starts.ndim != 3 or tuple(block_starts.shape[:2]) != expected_rows:
        raise ValueError("QSA block starts must have shape [batch, query, topk]")
    if tuple(tail_indices.shape) != (*expected_rows, block_size):
        raise ValueError("QSA tail indices must have shape [batch, query, block_size]")
    if tuple(tail_counts.shape) != expected_rows:
        raise ValueError("QSA tail counts must have shape [batch, query]")
    compact = (block_starts, block_counts, tail_indices, tail_counts)
    if any(array.dtype != mx.int32 for array in compact):
        raise ValueError("QSA compact indices and counts must use int32")
    if not _kernel_layout_supported(
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=queries.dtype,
    ):
        raise ValueError("QSA query/K/V layout is unsupported by indexed split-K")
    split_count = _split_count(key_length, query_length) if splits is None else splits
    if split_count <= 0 or split_count % SIMD_WIDTH:
        raise ValueError(
            "indexed split-K split count must be a positive multiple of 32"
        )

    block_topk = int(block_starts.shape[-1])
    gqa_heads = query_heads // kv_heads
    dims = mx.array([query_length, key_length], dtype=mx.int32)
    scale_array = mx.array([scale], dtype=mx.float32)
    partial_shape = (batch, query_heads, query_length, split_count, head_dim)
    stat_shape = partial_shape[:-1]
    _log_activation()
    partials, sums, maxs = _pass1_kernel()(
        inputs=[
            queries,
            keys,
            values,
            block_starts,
            block_counts,
            tail_indices,
            tail_counts,
            dims,
            scale_array,
        ],
        template=[
            ("T", queries.dtype),
            ("QUERY_HEADS", query_heads),
            ("KV_HEADS", kv_heads),
            ("GQA_HEADS", gqa_heads),
            ("HEAD_DIM", head_dim),
            ("BLOCK_SIZE", block_size),
            ("BLOCK_TOPK", block_topk),
            ("SPLITS", split_count),
        ],
        grid=(gqa_heads * SIMD_WIDTH * query_length, batch * kv_heads, split_count),
        threadgroup=(gqa_heads * SIMD_WIDTH, 1, 1),
        output_shapes=[partial_shape, stat_shape, stat_shape],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
    )
    (output,) = _pass2_kernel()(
        inputs=[partials, sums, maxs, dims],
        template=[
            ("T", queries.dtype),
            ("HEAD_DIM", head_dim),
            ("SPLITS", split_count),
        ],
        grid=(1024 * batch * query_heads, query_length, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return output
