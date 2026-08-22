# SPDX-License-Identifier: Apache-2.0
"""Disk-streaming forward for Qwen3-Next's shared+routed MoE block."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx.expert_cache import ExpertCache


def qwen3_next_streaming_forward(
    block, x: mx.array, layer_idx: int, cache: ExpertCache
) -> mx.array:
    """Reproduce ``Qwen3NextSparseMoeBlock.__call__`` expert by expert.

    Router and shared-expert weights remain resident. Only the selected
    routed experts' quantized gate/up/down projections are fetched through
    ``cache`` from the checkpoint's stacked ``switch_mlp`` tensors.
    """
    sharding_group = getattr(block, "sharding_group", None)
    if sharding_group is not None:
        from mlx.nn.layers.distributed import sum_gradients

        x = sum_gradients(sharding_group)(x)

    gates = block.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)

    gate_proj = block.switch_mlp.gate_proj
    up_proj = block.switch_mlp.up_proj
    down_proj = block.switch_mlp.down_proj

    batch_size, sequence_length, hidden_size = x.shape
    inds_list = inds.tolist()
    rows_b = []
    for batch_idx in range(batch_size):
        rows_l = []
        for position in range(sequence_length):
            token_x = x[batch_idx, position][None, :]
            output = mx.zeros((hidden_size,), dtype=x.dtype)
            for selected_idx, expert_id in enumerate(inds_list[batch_idx][position]):
                bundle = cache.get(layer_idx, expert_id)
                gate = bundle["gate_proj"]
                up = bundle["up_proj"]
                down = bundle["down_proj"]

                gate_output = mx.quantized_matmul(
                    token_x,
                    gate["weight"],
                    scales=gate["scales"],
                    biases=gate["biases"],
                    transpose=True,
                    group_size=gate_proj.group_size,
                    bits=gate_proj.bits,
                    mode=gate_proj.mode,
                )
                up_output = mx.quantized_matmul(
                    token_x,
                    up["weight"],
                    scales=up["scales"],
                    biases=up["biases"],
                    transpose=True,
                    group_size=up_proj.group_size,
                    bits=up_proj.bits,
                    mode=up_proj.mode,
                )
                hidden = nn.silu(gate_output) * up_output
                down_output = mx.quantized_matmul(
                    hidden,
                    down["weight"],
                    scales=down["scales"],
                    biases=down["biases"],
                    transpose=True,
                    group_size=down_proj.group_size,
                    bits=down_proj.bits,
                    mode=down_proj.mode,
                )
                output = (
                    output + down_output[0] * scores[batch_idx, position, selected_idx]
                )
            rows_l.append(output)
        rows_b.append(mx.stack(rows_l, axis=0))
    routed_output = mx.stack(rows_b, axis=0)

    shared_output = block.shared_expert(x)
    shared_output = mx.sigmoid(block.shared_expert_gate(x)) * shared_output
    output = routed_output + shared_output
    if sharding_group is not None:
        output = mx.distributed.all_sum(output, group=sharding_group)
    return output
