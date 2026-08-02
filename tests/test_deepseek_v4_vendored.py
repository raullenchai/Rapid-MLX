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


def test_deepseek_v4_rope_attention_factor_scales_only_rotary_channels():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4 import DeepseekV4RoPE

    scaling = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 4096,
        "attention_factor": 0.5,
    }
    baseline = DeepseekV4RoPE(
        dims=4,
        base=10000.0,
        scaling_config={**scaling, "attention_factor": 1.0},
    )
    scaled = DeepseekV4RoPE(dims=4, base=10000.0, scaling_config=scaling)
    x = mx.arange(16, dtype=mx.float32).reshape(1, 1, 2, 8) / 10
    expected_input = mx.concatenate([x[..., :4], 0.5 * x[..., 4:]], axis=-1)

    actual = scaled(x, offset=7)
    expected = baseline(expected_input, offset=7)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=1e-6).item()


def test_batch_pooling_cache_restores_neutral_lengths_and_accepts_zero_padding():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4_cache import BatchPoolingCache

    cache = BatchPoolingCache(ratio=4, left_padding=[0, 0])
    cache.prepare(lengths=[3, 3], left_padding=[0, 0])
    kv = mx.zeros((2, 3, 4))
    gate = mx.zeros((2, 3, 2))
    cache.accumulate_windows(kv, gate, 0)

    restored = BatchPoolingCache.from_state(cache.state, cache.meta_state)
    assert restored._lengths == [2**31, 2**31]
    restored.prepare(lengths=[1, 1], left_padding=[0, 0])
    restored.accumulate_windows(mx.zeros((2, 1, 4)), mx.zeros((2, 1, 2)), 3)


def test_batch_pooling_cache_rejects_nonzero_left_padding():
    pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4_cache import BatchPoolingCache

    cache = BatchPoolingCache(ratio=4, left_padding=[0, 0])
    with pytest.raises(RuntimeError, match="does not support left padding"):
        cache.prepare(left_padding=[0, 1])


def test_extend_mask_preserves_pooled_mask_without_local_mask():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4 import _extend_mask

    pooled = mx.array([[[True, False]]])
    actual = _extend_mask(None, pooled, N=5)
    mx.eval(actual)

    assert actual.shape == (1, 1, 1, 5)
    assert actual.tolist() == [[[[True, True, True, True, False]]]]


def test_extend_mask_converts_boolean_pool_mask_to_additive_semantics():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4 import _extend_mask

    local = mx.zeros((1, 1, 1, 3), dtype=mx.float32)
    pooled = mx.array([[[True, False]]])
    actual = _extend_mask(local, pooled, N=5)
    mx.eval(actual)

    assert actual.shape == (1, 1, 1, 5)
    assert actual[0, 0, 0, 3].item() == 0
    assert actual[0, 0, 0, 4].item() < -1e30


def test_hyper_connection_uses_ops_for_non_four_way_multiplicity(monkeypatch):
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models import deepseek_v4_hyper_connection as hc

    class Config:
        hc_mult = 2
        hc_sinkhorn_iters = 1
        hc_eps = 1e-6
        rms_norm_eps = 1e-6
        hidden_size = 4

    layer = hc.HyperConnection(Config())
    called = {"ops": False}
    original = hc._hc_ops

    def tracked_ops(*args, **kwargs):
        called["ops"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(hc, "_hc_ops", tracked_ops)
    monkeypatch.setattr(hc.mx, "default_device", lambda: hc.mx.gpu)
    monkeypatch.setattr(hc.mx.metal, "is_available", lambda: True)
    layer(mx.zeros((1, 1, 2, 4)))

    assert called["ops"]


def test_batch_pooling_cache_empty_batch_is_a_noop():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4_cache import BatchPoolingCache

    cache = BatchPoolingCache(ratio=4, left_padding=[])
    kv = mx.zeros((0, 2, 3), dtype=mx.float16)
    gate = mx.zeros((0, 2, 2), dtype=mx.float32)
    out_kv, out_gate, base = cache.accumulate_windows(kv, gate, 0)
    mx.eval(out_kv, out_gate, base)

    assert out_kv.shape == kv.shape
    assert out_gate.shape == gate.shape
    assert base.shape == (0,)


def test_batch_pooling_cache_merge_preserves_projection_dtypes():
    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.models.deepseek_v4_cache import (
        BatchPoolingCache,
        PoolingCache,
    )

    cache = PoolingCache(ratio=4)
    cache.buf_kv = mx.ones((1, 4, 3), dtype=mx.float16)
    cache.buf_gate = mx.ones((1, 4, 2), dtype=mx.float32)
    cache.remainder = 2
    merged = BatchPoolingCache.merge([cache])

    assert merged.buf_kv.dtype == mx.float16
    assert merged.buf_gate.dtype == mx.float32


def test_scheduler_reconstructs_vendored_cache_list_at_prefill_boundary():
    mx = pytest.importorskip("mlx.core")

    from mlx_lm.models.cache import CacheList, RotatingKVCache

    from vllm_mlx.models.deepseek_v4_cache import DeepseekV4PoolingCache
    from vllm_mlx.scheduler import Scheduler

    rotating = RotatingKVCache(max_size=128)
    keys = mx.zeros((1, 1, 3, 4))
    rotating.update_and_fetch(keys, keys)
    pooling = DeepseekV4PoolingCache(ratio=4)
    pooling.accumulate_windows(mx.zeros((1, 3, 4)), mx.zeros((1, 3, 2)), 0)
    original = [CacheList(rotating, pooling)]
    scheduler = Scheduler.__new__(Scheduler)

    states = scheduler._extract_cache_states(original)
    restored = scheduler._reconstruct_cache_from_states(states)

    assert restored is not None
    assert isinstance(restored[0], CacheList)
    assert isinstance(restored[0].caches[1], DeepseekV4PoolingCache)
    assert restored[0].caches[0].offset == rotating.offset
    assert restored[0].caches[1].remainder == pooling.remainder


def test_scheduler_rejects_mismatched_vendored_cache_list_metadata():
    pytest.importorskip("mlx.core")

    from mlx_lm.models.cache import CacheList

    from vllm_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    states = [
        {
            "class_name": "CacheList",
            "class_ref": CacheList,
            "state": [([], None)],
            "meta_state": (["KVCache", "DeepseekV4PoolingCache"], [None]),
        }
    ]

    assert scheduler._reconstruct_cache_from_states(states) is None


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


def test_csa_cached_decode_matches_single_prefill():
    """CSA overlap state must survive an incremental forward boundary."""
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
        sliding_window=64,
        compress_ratios=[0, 4, 128, 0],
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
    tokens = mx.array([[i % args.vocab_size for i in range(20)]], dtype=mx.int32)

    full_cache = model.make_cache()
    full = model(tokens, cache=full_cache)

    split_cache = model.make_cache()
    prefix = model(tokens[:, :9], cache=split_cache)
    mx.eval(prefix, [cache.state for cache in split_cache])
    suffix = model(tokens[:, 9:], cache=split_cache)
    mx.eval(full, suffix, [cache.state for cache in split_cache])

    assert mx.allclose(full[:, 9:], suffix, rtol=1e-4, atol=1e-4).item()


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


def test_deepseek_v4_dspark_drafts_checkpoint_block():
    import mlx.core as mx

    from vllm_mlx.models.deepseek_v4 import Model, ModelArgs

    args = ModelArgs(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        n_routed_experts=8,
        q_lora_rank=16,
        qk_rope_head_dim=4,
        num_experts_per_tok=2,
        head_dim=8,
        compress_ratios=[0, 0, 0],
        hc_mult=4,
        num_hash_layers=0,
        sliding_window=16,
        o_groups=2,
        o_lora_rank=8,
        dspark_num_layers=3,
        dspark_block_size=5,
        dspark_noise_token_id=127,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=8,
    )
    model = Model(args)
    target_cache = model.make_cache()
    draft_cache = model.make_dspark_cache()
    model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=target_cache)
    assert (
        model.dspark_forward(
            mx.array([[3]], dtype=mx.int32), model._last_dspark_hidden, draft_cache
        )
        is None
    )
    model(mx.array([[4]], dtype=mx.int32), cache=target_cache)
    proposal = model.dspark_forward(
        mx.array([[4]], dtype=mx.int32), model._last_dspark_hidden, draft_cache
    )
    assert proposal is not None
    output_ids, logits = proposal
    mx.eval(output_ids, logits)
    assert output_ids.shape == (1, 6)
    assert logits.shape == (1, 5, 128)

    short_proposal = model.dspark_forward(
        mx.array([[4]], dtype=mx.int32),
        model._last_dspark_hidden,
        draft_cache,
        max_draft_tokens=2,
    )
    assert short_proposal is not None
    short_ids, short_logits = short_proposal
    mx.eval(short_ids, short_logits)
    assert short_ids.shape == (1, 3)
    assert short_logits.shape == (1, 2, 128)
