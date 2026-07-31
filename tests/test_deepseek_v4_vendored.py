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


def test_deepseek_v4_rope_honors_explicit_yarn_attention_factor():
    """HF's explicit YaRN attention factor scales only rotary channels."""
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4 import DeepseekV4RoPE

    scaling = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 128,
    }
    x = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 10
    baseline = DeepseekV4RoPE(dims=4, base=10000.0, scaling_config=scaling)
    explicit_one = DeepseekV4RoPE(
        dims=4,
        base=10000.0,
        scaling_config={**scaling, "attention_factor": 1.0},
    )
    explicit_null = DeepseekV4RoPE(
        dims=4,
        base=10000.0,
        scaling_config={**scaling, "attention_factor": None},
    )
    explicit_half = DeepseekV4RoPE(
        dims=4,
        base=10000.0,
        scaling_config={**scaling, "attention_factor": 0.5},
    )

    original_input = mx.array(x)
    expected_input = mx.concatenate([x[..., :4], 0.5 * x[..., 4:]], axis=-1)
    baseline_output = baseline(x)
    one_output = explicit_one(x)
    null_output = explicit_null(x)
    half_output = explicit_half(x)
    expected_half_output = baseline(expected_input)
    mx.eval(
        original_input,
        x,
        baseline_output,
        one_output,
        null_output,
        half_output,
        expected_half_output,
    )

    assert mx.array_equal(x, original_input).item()
    assert mx.allclose(baseline_output, one_output, atol=1e-6).item()
    assert mx.allclose(baseline_output, null_output, atol=1e-6).item()
    assert mx.allclose(half_output, expected_half_output, atol=1e-6).item()


def test_deepseek_v4_rope_applies_per_row_integer_offsets():
    """Continuous batches may contain caches at different positions."""
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4 import DeepseekV4RoPE

    rope = DeepseekV4RoPE(dims=4, base=10000.0)
    x = mx.arange(48, dtype=mx.float32).reshape(2, 2, 3, 4) / 10
    offsets = mx.array([3, 11], dtype=mx.int32)

    actual = rope(x, offsets)
    expected = mx.concatenate([rope(x[:1], 3), rope(x[1:], 11)], axis=0)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=1e-6).item()


def test_deepseek_v4_rope_rejects_offset_batch_mismatch():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4 import DeepseekV4RoPE

    rope = DeepseekV4RoPE(dims=4, base=10000.0)
    x = mx.zeros((2, 1, 3, 4))

    with pytest.raises(ValueError, match="one offset per batch row"):
        rope(x, mx.array([3], dtype=mx.int32))


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


def test_tiny_model_decodes_merged_caches_with_different_offsets():
    """Regression for mixed-length continuous batching on compressed layers."""
    mx = pytest.importorskip("mlx.core")

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
    short_cache = model.make_cache()
    long_cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4]]), cache=short_cache)
    model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=long_cache)
    mx.eval([cache.state for cache in short_cache + long_cache])

    merged_cache = [
        type(short).merge([short, long]) for short, long in zip(short_cache, long_cache)
    ]
    logits = model(mx.array([[9], [10]]), cache=merged_cache)
    mx.eval(logits, [cache.state for cache in merged_cache])

    assert logits.shape == (2, 1, args.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
