# SPDX-License-Identifier: Apache-2.0
"""
Tests for the vendored DeepSeek-V4 architecture.

mlx-lm 0.31.x doesn't ship `deepseek_v4` yet (see ml-explore/mlx-lm#1192).
We vendor the module so users can serve mlx-community/DeepSeek-V4-Flash-*
day-0. These tests pin the contract that:

1. The vendored module is importable on its own.
2. `_register_vendored_archs()` exposes it to mlx-lm's importlib lookup.
3. A tiny synthetic config can construct + run the model end-to-end
   (proves Metal kernels compile and the forward path produces logits).
"""

import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _clear_vendored_register():
    """Registration is sys.modules-level state — reset before each test."""
    sys.modules.pop("mlx_lm.models.deepseek_v4", None)
    yield
    sys.modules.pop("mlx_lm.models.deepseek_v4", None)


def test_module_imports():
    from vllm_mlx.models import deepseek_v4

    assert hasattr(deepseek_v4, "Model")
    assert hasattr(deepseek_v4, "ModelArgs")
    assert deepseek_v4.ModelArgs.__dataclass_fields__["model_type"].default == (
        "deepseek_v4"
    )


def test_register_vendored_archs_makes_mlx_lm_loader_find_it():
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    assert "mlx_lm.models.deepseek_v4" not in sys.modules
    _register_vendored_archs()
    assert "mlx_lm.models.deepseek_v4" in sys.modules

    # mlx-lm's _get_classes() does exactly this lookup.
    mod = importlib.import_module("mlx_lm.models.deepseek_v4")
    assert mod is sys.modules["mlx_lm.models.deepseek_v4"]
    assert mod.__name__ == "vllm_mlx.models.deepseek_v4"
    assert hasattr(mod, "Model")


def test_register_vendored_archs_is_idempotent():
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    first = sys.modules["mlx_lm.models.deepseek_v4"]
    _register_vendored_archs()
    second = sys.modules["mlx_lm.models.deepseek_v4"]
    assert first is second


def test_tiny_model_forward_pass():
    """Smoke test the full forward path on a CPU-sized synthetic config.

    This is the same shape as upstream PR #1192's test_deepseek_v4 — it
    exercises HCA attention + sinkhorn + MoE routing without needing any
    real weights. If a Metal kernel breaks, this catches it.
    """
    import mlx.core as mx

    from vllm_mlx.models import deepseek_v4

    args = deepseek_v4.ModelArgs(
        model_type="deepseek_v4",
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=4,
        sliding_window=16,
        compress_ratios=[0, 0, 4, 0],
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    model = deepseek_v4.Model(args)
    inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
    cache = model.make_cache()
    logits = model(inputs, cache=cache)
    mx.eval(logits, [c.state for c in cache])

    assert logits.shape == (1, 8, args.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
