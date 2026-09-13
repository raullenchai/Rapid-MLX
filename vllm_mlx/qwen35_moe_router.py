# SPDX-License-Identifier: Apache-2.0
"""Fuse short-row Qwen3.5-family MoE top-k routing on Metal.

The stock routing tail uses separate ``argpartition``, gather, sum, and divide
operations after the router softmax.  Decode repeats that tiny chain in every
MoE layer.  This module replaces only that tail with one Metal launch while
leaving the model's router projection and precise softmax unchanged.

Per-model enrollment and an install-time parity probe ensure that the
process-global class patch cannot accidentally enroll a later, incompatible
model.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx

logger = logging.getLogger(__name__)

_KERNEL = None
_MAX_ROWS = 8
_MAX_EXPERTS = 512
_MAX_TOP_K = 16
_TAG = "_rapid_qwen35_fused_router"

_SOURCE = r"""
    constexpr uint PER = uint(NE) / 32;
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    const device T* g = probs + row * uint(NE);

    float vals[PER];
    bool taken[PER];
    for (uint i = 0; i < PER; ++i) {
        vals[i] = float(g[lane * PER + i]);
        taken[i] = false;
    }

    float sel_p[K];
    uint sel_i[K];
    for (uint j = 0; j < K; ++j) {
        float best = -INFINITY;
        uint best_i = uint(NE);
        for (uint i = 0; i < PER; ++i) {
            if (!taken[i] && vals[i] >= best) {
                best = vals[i];
                best_i = lane * PER + i;
            }
        }
        const float group_best = simd_max(best);
        const uint candidate = (best == group_best) ? best_i : 0u;
        const uint group_best_i = simd_max(candidate);
        if (best == group_best && best_i == group_best_i) {
            taken[best_i - lane * PER] = true;
        }
        sel_p[j] = group_best;
        sel_i[j] = group_best_i;
    }

    if (lane == 0) {
        T total = T(0);
        // Match the stock top-k tail's ascending order for the reduction.
        for (int j = int(K) - 1; j >= 0; --j) {
            total += T(sel_p[uint(j)]);
        }
        for (uint j = 0; j < K; ++j) {
            // mlx.argpartition returns this top-k tail from the smallest
            // selected probability to the largest for the eligible shape.
            // Reverse the descending selection so the expert reduction keeps
            // the stock accumulation order, not merely the same expert set.
            const uint out = uint(K) - 1u - j;
            indices[row * uint(K) + out] = sel_i[j];
            scores[row * uint(K) + out] = T(sel_p[j]) / total;
        }
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="rapid_qwen35_moe_router_topk",
            input_names=["probs"],
            output_names=["indices", "scores"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_router_topk(probs: mx.array, top_k: int) -> tuple[mx.array, mx.array]:
    """Return stock-ordered top-k indices and normalized scores."""
    shape = probs.shape
    if probs.ndim < 2 or probs.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("fused Qwen router requires rank >= 2 BF16/FP16 probabilities")
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    num_experts = int(shape[-1])
    if not (
        rows <= _MAX_ROWS
        and 32 <= num_experts <= _MAX_EXPERTS
        and num_experts % 32 == 0
        and 1 <= top_k <= _MAX_TOP_K
    ):
        raise ValueError("unsupported fused Qwen router shape")
    outputs = _kernel()(
        inputs=[probs],
        template=[("T", probs.dtype), ("NE", num_experts), ("K", top_k)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*shape[:-1], top_k), (*shape[:-1], top_k)],
        output_dtypes=[mx.uint32, probs.dtype],
    )
    return outputs[0], outputs[1]


def _eligible(x: mx.array, block: Any) -> bool:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    num_experts = int(getattr(block, "num_experts", 0))
    top_k = int(getattr(block, "top_k", 0))
    return (
        getattr(block, _TAG, False)
        and x.ndim == 3
        and getattr(block, "sharding_group", None) is None
        and bool(getattr(block, "norm_topk_prob", True))
        and rows <= _MAX_ROWS
        and num_experts >= 32
        and num_experts <= _MAX_EXPERTS
        and num_experts % 32 == 0
        and 1 <= top_k <= _MAX_TOP_K
        and x.dtype in (mx.bfloat16, mx.float16)
    )


def _composed(probs: mx.array, top_k: int) -> tuple[mx.array, mx.array]:
    indices = mx.argpartition(probs, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(probs, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


def _probe(num_experts: int, top_k: int, dtype: mx.Dtype) -> bool:
    # No RNG: model loading must not perturb the caller's sampling state.
    base = mx.arange(num_experts, dtype=mx.float32)
    logits = mx.stack([mx.sin(base * 0.731 + row) for row in range(_MAX_ROWS)])
    inputs = [
        mx.softmax(logits.astype(dtype), axis=-1, precise=True),
        mx.ones((_MAX_ROWS, num_experts), dtype=dtype),
    ]
    for probs in inputs:
        expected_i, expected_s = _composed(probs, top_k)
        actual_i, actual_s = fused_router_topk(probs, top_k)
        mx.eval(expected_i, expected_s, actual_i, actual_s)
        if not (
            bool(mx.array_equal(expected_i, actual_i))
            and bool(mx.array_equal(expected_s, actual_s))
        ):
            return False
    return True


def _patch_class(block_class: type) -> None:
    if getattr(block_class, "_rapid_qwen35_router_patched", False):
        return
    original = cast(Callable[[Any, mx.array], mx.array], block_class.__call__)

    def patched(self, x: mx.array) -> mx.array:
        if not _eligible(x, self):
            return original(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices, scores = fused_router_topk(gates, self.top_k)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return cast(mx.array, routed + shared)

    dynamic_class = cast(Any, block_class)
    dynamic_class.__call__ = patched
    dynamic_class._rapid_qwen35_router_patched = True
    dynamic_class._rapid_qwen35_router_original_call = original


def install_qwen35_moe_router(model: Any) -> int:
    """Enroll compatible blocks from one loaded model; return their count."""
    if os.environ.get("RAPID_MLX_QWEN35_MOE_ROUTER", "1") == "0":
        return 0
    if not mx.metal.is_available():
        return 0
    block_classes = []
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
    except ImportError:
        pass
    else:
        block_classes.append(Qwen3NextSparseMoeBlock)
    # mlx-vlm keeps a separate Qwen implementation rather than re-exporting
    # mlx-lm's class.  Import it only when mlx-vlm is already resident so a
    # text-only install does not turn vision into a hard dependency.
    vlm_qwen = sys.modules.get("mlx_vlm.models.qwen3_5_moe.language")
    if vlm_qwen is not None:
        vlm_block = getattr(vlm_qwen, "Qwen3_5MoeSparseMoeBlock", None)
        if vlm_block is not None:
            block_classes.append(vlm_block)
    named_modules = getattr(model, "named_modules", None)
    if not block_classes or not callable(named_modules):
        return 0

    blocks = [module for _, module in named_modules() if type(module) in block_classes]
    if not blocks:
        return 0
    signatures = {
        (int(block.num_experts), int(block.top_k))
        for block in blocks
        if getattr(block, "norm_topk_prob", True)
        and 32 <= int(getattr(block, "num_experts", 0)) <= _MAX_EXPERTS
        and int(getattr(block, "num_experts", 0)) % 32 == 0
        and 1 <= int(getattr(block, "top_k", 0)) <= _MAX_TOP_K
    }
    try:
        if not signatures or any(
            not _probe(num_experts, top_k, dtype)
            for num_experts, top_k in signatures
            for dtype in (mx.bfloat16, mx.float16)
        ):
            logger.warning(
                "Qwen MoE fused router parity probe failed; using stock path"
            )
            return 0
    except Exception:
        logger.warning(
            "Qwen MoE fused router probe failed; using stock path", exc_info=True
        )
        return 0

    for block_class in block_classes:
        if any(type(block) is block_class for block in blocks):
            _patch_class(block_class)
    enrolled = 0
    for block in blocks:
        num_experts = int(getattr(block, "num_experts", 0))
        top_k = int(getattr(block, "top_k", 0))
        if (
            getattr(block, "norm_topk_prob", True)
            and num_experts >= 32
            and num_experts <= _MAX_EXPERTS
            and num_experts % 32 == 0
            and 1 <= top_k <= _MAX_TOP_K
        ):
            setattr(block, _TAG, True)
            enrolled += 1
    if enrolled:
        logger.info("[qwen35_router] fused top-k enrolled: %d MoE layers", enrolled)
    return enrolled
