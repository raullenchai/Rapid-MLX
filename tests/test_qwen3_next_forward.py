"""Focused math test for Qwen3-Next's disk-streaming MoE forward."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from vllm_mlx.expert_cache import ExpertCache
from vllm_mlx.qwen3_next_forward import qwen3_next_streaming_forward


def test_qwen3_next_streaming_forward_matches_real_quantized_block():
    """Selected-expert matmuls match mlx-lm's real QuantizedSwitchLinear."""
    mx.random.seed(42)
    args = SimpleNamespace(
        hidden_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        num_experts=4,
        num_experts_per_tok=2,
    )
    block = Qwen3NextSparseMoeBlock(args)
    nn.quantize(block, group_size=32, bits=4)
    block.eval()

    def fetch_expert(_layer_idx: int, expert_id: int):
        return {
            projection: {
                component: getattr(getattr(block.switch_mlp, projection), component)[
                    expert_id
                ]
                for component in ("weight", "scales", "biases")
            }
            for projection in ("gate_proj", "up_proj", "down_proj")
        }

    cache = ExpertCache(fetch_fn=fetch_expert, budget_bytes=10_000_000)
    x = mx.random.normal((2, 3, args.hidden_size)).astype(mx.float16)

    expected = block(x)
    actual = qwen3_next_streaming_forward(block, x, layer_idx=0, cache=cache)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert cache.misses > 0
