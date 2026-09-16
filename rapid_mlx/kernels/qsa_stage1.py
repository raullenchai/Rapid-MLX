# SPDX-License-Identifier: MIT
#
# Adapted from pierre427/mlx-lm-unified at commit
# 685dff6c1ee602f1bae2e51c758a3b07f593a0c2. See LICENSE-QSA-STAGE1.
# The radix-selection design is adapted from MTPLX PR #397 (Apache-2.0).

"""Native exact radix selector for Qwen4 QSA stage one."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import mlx.core as mx

_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_MAX_TOPK = 512
ENABLE_ENV = "RAPID_MLX_QSA_STAGE1"
MIN_QUERY_LENGTH = 64
MIN_PHYSICAL_KV_LENGTH = 65_024
QUALIFIED_MLX_VERSIONS = frozenset({"0.32.2"})
QUALIFIED_METAL_ARCHITECTURES = frozenset({"applegpu_g14s", "applegpu_g15d"})


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


def qsa_stage1_decline_reason(
    query_length: int,
    physical_kv_length: int,
    *,
    batch_size: int,
    training: bool = False,
    mlx_version: str | None = None,
    metal_architecture: str | None = None,
) -> str | None:
    """Return ``None`` only for the measured, opt-in M2/M3 envelope."""
    if not _enabled():
        return "disabled"
    if training:
        return "training"
    if batch_size != 1:
        return "batch size is not qualified"
    if query_length < MIN_QUERY_LENGTH:
        return "query below crossover"
    if physical_kv_length < MIN_PHYSICAL_KV_LENGTH:
        return "physical KV below crossover"
    installed = _mlx_version() if mlx_version is None else mlx_version
    if installed not in QUALIFIED_MLX_VERSIONS:
        return f"unqualified MLX version {installed}"
    if not mx.metal.is_available():
        return "Metal runtime unavailable"
    architecture = (
        _metal_architecture() if metal_architecture is None else metal_architecture
    )
    if architecture not in QUALIFIED_METAL_ARCHITECTURES:
        return f"unqualified Metal architecture {architecture}"
    return None


def qsa_stage1_kernel_available() -> bool:
    """Return whether a Metal custom kernel can run in this process."""

    return bool(mx.metal.is_available() and mx.default_device() == mx.gpu)


def qsa_stage1_supported(
    q: mx.array,
    pooled: mx.array,
    q_positions: mx.array,
    *,
    block_topk: int,
    compress_ratio: int,
) -> bool:
    """Check static geometry without evaluating device arrays."""

    if not qsa_stage1_kernel_available():
        return False
    if q.ndim != 4 or pooled.ndim != 3 or q_positions.ndim != 2:
        return False
    if q.shape[:2] != q_positions.shape:
        return False
    if q.shape[0] != pooled.shape[0] or q.shape[-1] != pooled.shape[-1]:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or pooled.dtype not in _SUPPORTED_DTYPES:
        return False
    if q_positions.dtype not in (mx.int32, mx.int64):
        return False
    if int(q.shape[1]) <= 0 or int(pooled.shape[1]) <= int(block_topk):
        return False
    return 1 <= int(block_topk) <= _MAX_TOPK and int(compress_ratio) > 0


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline uint qsa_float_order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong qsa_composite_key(float score, uint block_id) {
    return (ulong(qsa_float_order_key(score)) << 32) | ulong(block_id);
}

inline bool qsa_id_before(
    uint a_index, bool a_valid, uint b_index, bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    return a_index < b_index;
}
"""


@lru_cache(maxsize=32)
def _stage1_kernel(
    heads: int,
    head_dim: int,
    topk: int,
    ratio: int,
    q_dtype: mx.Dtype,
    pooled_dtype: mx.Dtype,
):
    width = 1 << (max(256, topk) - 1).bit_length()
    if width > 1024:
        raise ValueError(f"QSA stage-one threadgroup width {width} is unsupported")

    header = (
        _HEADER
        + f"""
constant constexpr uint HEADS = {heads};
constant constexpr uint HEAD_DIM = {head_dim};
constant constexpr uint TOP_K = {topk};
constant constexpr uint RATIO = {ratio};
constant constexpr uint WIDTH = {width};
constant constexpr uint RADIX_BINS = 256;
constant constexpr float SQRT_HEAD_DIM = {math.sqrt(head_dim)!r}f;
"""
    )
    source = r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint blocks = uint(dims[0]);
        const uint query_rows = uint(dims[1]);
        const uint batch = row / query_rows;
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t scratch_base = size_t(row) * blocks;

        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;

        for (uint block = lane; block < valid_count; block += WIDTH) {
            float score_sum = 0.0f;
            for (uint head = 0; head < HEADS; ++head) {
                float dot = 0.0f;
                const size_t q_base =
                    (size_t(row) * HEADS + head) * HEAD_DIM;
                const size_t pooled_base =
                    (size_t(batch) * blocks + block) * HEAD_DIM;
                for (uint dim = 0; dim < HEAD_DIM; ++dim) {
                    dot += float(q[q_base + dim]) *
                           float(pooled[pooled_base + dim]);
                }
                score_sum += metal::max(dot, 0.0f);
            }
            score_scratch[scratch_base + block] = score_sum / SQRT_HEAD_DIM;
        }
        threadgroup_barrier(
            mem_flags::mem_threadgroup | mem_flags::mem_device);

        if (lane == 0) {
            radix_prefix = 0ul;
            radix_rank = selected_valid > 0 ? selected_valid - 1 : 0;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0) {
            for (uint pass = 0; pass < 8; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 56u - pass * 8u;
                const ulong prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const float score = score_scratch[scratch_base + block];
                    const ulong key = qsa_composite_key(score, block);
                    const bool prefix_matches = pass == 0 ||
                        (key >> (shift + 8u)) == prefix;
                    if (prefix_matches) {
                        atomic_fetch_add_explicit(
                            &radix_histogram[uint((key >> shift) & 0xfful)],
                            1u,
                            memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0) {
                    uint rank = radix_rank;
                    uint chosen = 0;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8) | ulong(chosen);
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0;
        if (lane == 0) {
            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (selected_valid > 0) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const float score = score_scratch[scratch_base + block];
                if (qsa_composite_key(score, block) >= threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0;
        for (uint sequence = 2; sequence <= WIDTH; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1 : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;
                const bool lower_wants_before = (lane & sequence) == 0;
                const bool b_before_a = qsa_id_before(
                    b_index, b_valid, a_index, a_valid);
                const bool a_before_b = qsa_id_before(
                    a_index, a_valid, b_index, b_valid);
                const bool swap = lower_wants_before ? b_before_a : a_before_b;
                if (swap) {
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }

        if (lane < TOP_K) {
            uint output_id = my_index;
            if (lane >= selected_valid) {
                const uint invalid_count = TOP_K - selected_valid;
                output_id = blocks - invalid_count + (lane - selected_valid);
            }
            block_ids[size_t(row) * TOP_K + lane] = output_id;
        }
    """
    return mx.fast.metal_kernel(
        name=(
            f"mlx_lm_qwen4_qsa_stage1_h{heads}_d{head_dim}_k{topk}_r{ratio}_"
            f"{str(q_dtype).replace('.', '_')}_{str(pooled_dtype).replace('.', '_')}"
        ),
        input_names=["q", "pooled", "q_positions", "dims"],
        output_names=["block_ids", "score_scratch"],
        header=header,
        source=source,
    )


def _select_scores(
    scores: mx.array,
    q_positions: mx.array,
    *,
    topk: int,
    compress_ratio: int,
) -> mx.array:
    """Select score-column IDs with the exact radix kernel."""

    rows, blocks = map(int, scores.shape)
    score_q = mx.ones((rows, 1, 1, 1), dtype=mx.float32)
    score_keys = scores.reshape(rows, blocks, 1)
    score_positions = q_positions.reshape(rows, 1)
    kernel = _stage1_kernel(
        1,
        1,
        int(topk),
        int(compress_ratio),
        mx.float32,
        mx.float32,
    )
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    outputs = kernel(
        inputs=[
            score_q,
            score_keys,
            score_positions.astype(mx.int32),
            mx.array([blocks, 1], dtype=mx.int32),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(rows, 1, int(topk)), (rows, blocks)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0].reshape(rows, int(topk))


def qsa_stage1_select(
    q: mx.array,
    pooled: mx.array,
    q_positions: mx.array,
    *,
    block_topk: int,
    compress_ratio: int,
) -> mx.array:
    """Return deterministic selected block IDs with shape ``[B,L,K]``."""

    if not qsa_stage1_supported(
        q,
        pooled,
        q_positions,
        block_topk=block_topk,
        compress_ratio=compress_ratio,
    ):
        raise ValueError("unsupported QSA stage-one kernel geometry")

    batch, length, _, _ = map(int, q.shape)
    blocks = int(pooled.shape[1])
    topk = int(block_topk)
    rows = batch * length
    positions = q_positions.reshape(rows)
    # Preserve Rapid's existing score producer operation for operation. In
    # particular, q/k matmul rounds to the activation dtype before the FP32
    # ReLU and head reduction. Casting the operands to FP32 here can change a
    # near-boundary top-k set even when ordinary random probes happen to pass.
    scores = mx.matmul(q.transpose(0, 2, 1, 3), pooled.swapaxes(-1, -2)[:, None])
    scores = mx.sum(mx.maximum(scores.astype(mx.float32), 0), axis=1) / math.sqrt(
        q.shape[-1]
    )
    selected = _select_scores(
        scores.reshape(rows, blocks),
        positions,
        topk=topk,
        compress_ratio=compress_ratio,
    )
    return selected.reshape(batch, length, topk)


def qsa_stage1_kernel_cache_info():
    """Return the bounded template-cache receipt."""

    selector = _stage1_kernel.cache_info()
    return SimpleNamespace(
        currsize=selector.currsize,
        maxsize=selector.maxsize,
    )
