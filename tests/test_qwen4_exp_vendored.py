import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache

import vllm_mlx.models.qwen4_exp as qwen4_exp
from scripts import qwen38_streaming_convert as converter
from scripts.qwen38_streaming_convert import quantized_tensor_names
from vllm_mlx.models.qwen4_exp import (
    GatedDeltaNet,
    GatedResidual,
    Model,
    ModelArgs,
    NGramEmbedding,
    PLELayer,
    QSAAttention,
    QSAIndexer,
    Qwen4ExpStateCache,
    ShardedEmbedding,
    SparseMoeBlock,
    TextModelArgs,
    ZeroCenteredRMSNorm,
    apply_qwen4_exp_rope,
    build_layer_multipliers,
)
from vllm_mlx.models.qwen4_exp_cache import QSAIndexCache


def _args(**overrides):
    values = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "hc_count": 4,
        "hc_lowrank": 3,
        "layer_types": ["linear_attention", "full_attention"],
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "ple_layer_ids": [],
        "eos_token_id": 31,
    }
    values.update(overrides)
    return TextModelArgs(**values)


def _repair_fixture(tmp_path, shard_tensors, weight_map):
    output = tmp_path / "converted"
    output.mkdir()
    for shard_name, tensors in shard_tensors.items():
        mx.save_safetensors(str(output / shard_name), tensors)
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 8}, "weight_map": weight_map})
    )
    converter._write_sha256sums(output)
    return output


def test_config_normalizes_checkpoint_full_attention_to_qsa():
    args = _args()
    assert args.layer_types == ["linear_attention", "qwen_sparse_attention"]
    assert args.linear_num_value_heads // args.linear_num_key_heads == 3


def test_config_rejects_partial_indexer_contract():
    with pytest.raises(ValueError, match="complete indexer contract"):
        _args(indexer_budget=None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_hidden_layers": 1}, "one entry per decoder layer"),
        ({"layer_types": ["linear_attention", "unknown"]}, "Unsupported"),
        ({"hc_count": 1}, "four-stream HC dimensions"),
        (
            {"linear_num_key_heads": 3, "linear_num_value_heads": 2},
            "divisible by key heads",
        ),
        ({"output_gate_type": "silu"}, "sigmoid GDN gate"),
        ({"indexer_budget": 0}, "indexer values must be positive"),
        ({"indexer_kv_heads": 2}, "one indexer KV head"),
        ({"indexer_budget": 7}, "divisible by indexer_compress_ratio"),
        (
            {"rope_parameters": {"partial_rotary_factor": 2.0}},
            "rotary dimensions must fit",
        ),
        ({"num_experts_per_tok": 5}, "within num_experts"),
        ({"ple_layer_ids": [1], "ple_embed_dim": 15}, "divide evenly"),
        ({"ple_layer_ids": [3], "ple_embed_dim": 16}, "one-indexed"),
        (
            {"ple_layer_ids": [2], "ple_embed_dim": 16},
            "only valid on linear-attention",
        ),
        (
            {"ple_layer_ids": [1], "ple_embed_dim": 16, "eos_token_id": None},
            "requires eos_token_id",
        ),
        (
            {"ple_layer_ids": [1], "ple_embed_dim": 16, "eos_token_id": []},
            "must not be empty",
        ),
    ],
)
def test_config_rejects_invalid_architecture_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _args(**overrides)


def test_config_synthesizes_layer_schedule_and_accepts_flat_model_args():
    args = _args(layer_types=None, full_attention_interval=2)
    assert args.layer_types == ["linear_attention", "qwen_sparse_attention"]
    flat = asdict(args)
    flat["model_type"] = "qwen4_exp"
    parsed = ModelArgs.from_dict(flat)
    assert parsed.text_config["hidden_size"] == args.hidden_size
    nested = ModelArgs.from_dict(
        {"model_type": "qwen4_exp", "text_config": asdict(_args())}
    )
    assert nested.text_config["hidden_size"] == args.hidden_size


def test_qwen4_rope_uses_reference_half_split_pairing():
    values = np.array([[[[0.25, -0.5, 0.75, 1.0]]]], dtype=np.float32)
    output = apply_qwen4_exp_rope(
        mx.array(values),
        mx.array([[39]], dtype=mx.int64),
        rotary_dim=4,
        base=10_000_000,
    )
    frequencies = 1.0 / (10_000_000 ** (np.arange(0, 4, 2, dtype=np.float32) / 4))
    angles = 39 * frequencies
    first = values[..., :2]
    second = values[..., 2:]
    expected = np.concatenate(
        [
            first * np.cos(angles) - second * np.sin(angles),
            second * np.cos(angles) + first * np.sin(angles),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-5, atol=1e-5)


def test_rotary_fallback_guards_and_one_dimensional_positions():
    x = mx.ones((1, 2, 1, 4))
    assert (
        qwen4_exp.apply_rotary_positions(x, mx.array([0, 1]), rotary_dim=0, base=10_000)
        is x
    )
    with pytest.raises(ValueError, match="must be even"):
        qwen4_exp.apply_rotary_positions(x, mx.array([0, 1]), rotary_dim=3, base=10_000)
    output = qwen4_exp.apply_rotary_positions(
        x, mx.array([0, 1]), rotary_dim=4, base=10_000
    )
    assert output.shape == x.shape


def test_qwen4_rope_kernel_contract_and_fallback(monkeypatch):
    with pytest.raises(ValueError, match="requires 2-D position IDs"):
        qwen4_exp._qwen4_exp_rope_kernel.__wrapped__(4, 1)

    monkeypatch.setattr(qwen4_exp.mx.metal, "is_available", lambda: False)
    assert qwen4_exp._qwen4_exp_rope_kernel.__wrapped__(4, 2) is None

    monkeypatch.setattr(qwen4_exp, "_qwen4_exp_rope_kernel", lambda *_args: None)
    output = apply_qwen4_exp_rope(
        mx.ones((1, 1, 2, 4)),
        mx.array([[0, 1]], dtype=mx.int64),
        rotary_dim=4,
        base=10_000,
    )
    assert output.shape == (1, 1, 2, 4)


def test_zero_centered_grouped_rms_norm_matches_numpy():
    norm = ZeroCenteredRMSNorm(8, group_size=4, eps=1e-6)
    norm.weight = mx.array(np.linspace(-0.2, 0.2, 8, dtype=np.float32))
    x_np = np.arange(1, 17, dtype=np.float32).reshape(1, 2, 8) / 10
    out = norm(mx.array(x_np))
    grouped = x_np.reshape(1, 2, 2, 4)
    expected = grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)
    expected *= (1 + np.linspace(-0.2, 0.2, 8, dtype=np.float32)).reshape(2, 4)
    np.testing.assert_allclose(
        np.array(out), expected.reshape(x_np.shape), rtol=2e-5, atol=2e-5
    )


def test_zero_centered_rms_norm_rejects_partial_group():
    with pytest.raises(ValueError, match="divide the feature width"):
        ZeroCenteredRMSNorm(7, group_size=4)


def test_gated_residual_matches_reference_equations():
    args = _args()
    layer = GatedResidual(args)
    rng = np.random.default_rng(7)
    layer.hc_norm.weight = mx.array(rng.normal(0, 0.1, (32,)).astype(np.float32))
    layer.input_mix_weight_down.weight = mx.array(
        rng.normal(0, 0.1, (3, 32)).astype(np.float32)
    )
    layer.input_mix_weight_up.weight = mx.array(
        rng.normal(0, 0.1, (32, 3)).astype(np.float32)
    )
    layer.block_inject_weight.weight = mx.array(
        rng.normal(0, 0.1, (4, 32)).astype(np.float32)
    )
    x_np = rng.normal(0, 0.2, (1, 2, 32)).astype(np.float32)
    mixed, residual, injection = layer(mx.array(x_np))

    grouped = x_np.reshape(1, 2, 4, 8)
    weight = np.array(layer.hc_norm.weight).reshape(4, 8)
    normalized = grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)
    normalized *= 1 + weight
    flat = normalized.reshape(1, 2, 32)
    down = flat @ np.array(layer.input_mix_weight_down.weight).T / 4
    silu = down / (1 + np.exp(-down))
    mix = 1 / (1 + np.exp(-(silu @ np.array(layer.input_mix_weight_up.weight).T)))
    expected_mixed = np.mean(mix.reshape(1, 2, 4, 8) * normalized, axis=-2)
    expected_injection = 2 / (
        1 + np.exp(-(flat @ np.array(layer.block_inject_weight.weight).T / 4))
    )
    np.testing.assert_allclose(np.array(mixed), expected_mixed, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(np.array(residual), x_np)
    np.testing.assert_allclose(
        np.array(injection), expected_injection, rtol=3e-5, atol=3e-5
    )


def test_qwen4_moe_verify_rows_match_fused_projection_path():
    from vllm_mlx.moe_fusion import fuse_gate_up

    block = SparseMoeBlock(_args())
    inputs = mx.array(
        np.random.default_rng(19).normal(size=(1, 2, 8)).astype(np.float32)
    )
    expected = block(inputs)
    assert fuse_gate_up(block) == 1
    actual = block(inputs, target_verify=True)
    mx.eval(expected, actual)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("masked", [False, True])
def test_qwen4_gdn_verify_kernel_matches_reference_and_boundaries(masked):
    from vllm_mlx.kernels.qwen4_gdn_verify import (
        gated_delta_verify_with_states,
    )

    rng = np.random.default_rng(23)
    batch, steps, key_heads, value_heads = 1, 3, 1, 2
    key_dim, value_dim = 32, 4

    def array(shape, scale=0.1):
        return mx.array((rng.normal(size=shape) * scale).astype(np.float32))

    query = array((batch, steps, key_heads, key_dim))
    key = array((batch, steps, key_heads, key_dim))
    value = array((batch, steps, value_heads, value_dim))
    alpha = array((batch, steps, value_heads))
    beta = array((batch, steps, value_heads))
    a_log = array((value_heads,), scale=0.01)
    dt_bias = array((value_heads,), scale=0.01)
    state = array((batch, value_heads, value_dim, key_dim))
    mask = mx.array([[True, False, True]]) if masked else None

    expected = gated_delta_verify_with_states(
        query,
        key,
        value,
        alpha,
        beta,
        a_log,
        dt_bias,
        state,
        mask,
        use_kernel=False,
    )
    actual = gated_delta_verify_with_states(
        query,
        key,
        value,
        alpha,
        beta,
        a_log,
        dt_bias,
        state,
        mask,
        use_kernel=True,
    )
    mx.eval(*expected, *actual)
    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_allclose(
            np.asarray(actual_array),
            np.asarray(expected_array),
            rtol=1e-5,
            atol=1e-6,
        )


def test_gated_residual_shape_guard_and_read_only_variant():
    args = _args()
    layer = GatedResidual(args, use_combine=False)
    mixed = layer(mx.zeros((1, 2, args.hc_count * args.hidden_size)))
    assert mixed.shape == (1, 2, args.hidden_size)
    with pytest.raises(ValueError, match="HC expected"):
        layer(mx.zeros((1, 2, args.hidden_size)))


def test_gdn_ratio_three_state_shapes_and_cached_decode():
    args = _args()
    layer = GatedDeltaNet(args)
    cache = ArraysCache(size=2)
    prompt = mx.zeros((1, 3, args.hidden_size), dtype=mx.float32)
    prompt_out = layer(prompt, cache=cache)
    mx.eval(prompt_out, cache.state)
    assert prompt_out.shape == (1, 3, args.hidden_size)
    assert cache[0].shape == (1, args.linear_conv_kernel_dim - 1, 20)
    assert cache[1].shape == (1, 3, 4, 4)

    token_out = layer(mx.zeros((1, 1, args.hidden_size)), cache=cache)
    mx.eval(token_out, cache.state)
    assert token_out.shape == (1, 1, args.hidden_size)
    assert cache[0].shape == (1, 2, 20)
    assert cache[1].shape == (1, 3, 4, 4)


def test_gdn_honors_mask_and_per_row_valid_lengths():
    layer = GatedDeltaNet(_args())
    cache = ArraysCache(size=2)
    cache.prepare(lengths=[1])
    output = layer(
        mx.zeros((1, 2, 8)),
        mask=mx.array([[True, False]]),
        cache=cache,
    )
    mx.eval(output, cache.state)
    assert output.shape == (1, 2, 8)


def test_sharded_embedding_preserves_global_row_identity():
    embedding = ShardedEmbedding(num_embeddings=16, dims=4, parts=4)
    for shard_id, shard in enumerate(embedding.shards):
        rows = np.arange(shard_id * 4, shard_id * 4 + 4, dtype=np.float32)
        shard.weight = mx.array(np.repeat(rows[:, None], 4, axis=1))
    ids = mx.array([[0, 3, 4, 9, 15]])
    output = embedding(ids)
    expected = np.repeat(
        np.array([[0, 3, 4, 9, 15]], dtype=np.float32)[..., None], 4, axis=-1
    )
    np.testing.assert_array_equal(np.array(output), expected)


def test_sharded_embedding_guards_and_empty_input():
    with pytest.raises(ValueError, match="divide evenly"):
        ShardedEmbedding(num_embeddings=15, dims=4, parts=4)
    embedding = ShardedEmbedding(num_embeddings=16, dims=4, parts=4)
    output = embedding(mx.array([], dtype=mx.int32).reshape(1, 0))
    assert output.shape == (1, 0, 4)
    assert qwen4_exp._is_prime(1) is False
    assert qwen4_exp._is_prime(2) is True


def test_ngram_multipliers_match_released_reference_constants():
    assert build_layer_multipliers(248320, 3, 0, 1234) == [
        23703573157769,
        20109073645365,
        8052911324071,
    ]


def _ple_args(**overrides):
    values = {
        "layer_types": ["linear_attention", "qwen_sparse_attention"],
        "ple_layer_ids": [1],
        "ple_embed_dim": 16,
        "ngram_vocab_size_base": 17,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 4,
        "rope_parameters": {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.5,
        },
    }
    values.update(overrides)
    return _args(**values)


def test_ngram_cache_matches_one_shot_across_request_chunks():
    args = _ple_args()
    embedding = NGramEmbedding(args, embedding_dim=16, ple_layer_index=0)
    one_shot = embedding.compute_ids(mx.array([[1, 2, 3, 4]]))

    cache = ArraysCache(size=4)
    first = embedding.compute_ids(mx.array([[1, 2]]), cache)
    second = embedding.compute_ids(mx.array([[3, 4]]), cache)
    mx.eval(one_shot, first, second, cache.state)
    np.testing.assert_array_equal(np.array(first), np.array(one_shot[:, :2]))
    np.testing.assert_array_equal(np.array(second), np.array(one_shot[:, 2:]))
    np.testing.assert_array_equal(np.array(cache[3]), np.array([[3, 4]]))


def test_ngram_context_resets_at_eos_boundary():
    args = _ple_args()
    embedding = NGramEmbedding(args, embedding_dim=16, ple_layer_index=0)
    with_history = embedding.compute_ids(mx.array([[7, 8, 31, 9]]))
    fresh_segment = embedding.compute_ids(mx.array([[9]]))
    mx.eval(with_history, fresh_segment)
    np.testing.assert_array_equal(
        np.array(with_history[:, -1]), np.array(fresh_segment[:, -1])
    )


def test_ngram_context_resets_at_every_declared_eos_boundary():
    args = _ple_args(eos_token_id=[31, 32])
    embedding = NGramEmbedding(args, embedding_dim=16, ple_layer_index=0)
    with_history = embedding.compute_ids(mx.array([[7, 8, 32, 9]]))
    fresh_segment = embedding.compute_ids(mx.array([[9]]))
    mx.eval(with_history, fresh_segment)
    np.testing.assert_array_equal(
        np.array(with_history[:, -1]), np.array(fresh_segment[:, -1])
    )


def test_ple_state_shapes_survive_cached_decode():
    args = _ple_args()
    ple = PLELayer(args, ple_layer_index=0)
    cache = ArraysCache(size=4)
    hidden = mx.zeros((1, 3, args.hc_count * args.hidden_size))
    output = ple(hidden, mx.array([[1, 2, 3]]), cache)
    mx.eval(output, cache.state)
    assert output.shape == hidden.shape
    assert cache[2].shape == (1, 9, args.hc_count * args.hidden_size)
    assert cache[3].shape == (1, 2)

    token_output = ple(
        mx.zeros((1, 1, args.hc_count * args.hidden_size)),
        mx.array([[4]]),
        cache,
    )
    mx.eval(token_output, cache.state)
    assert token_output.shape == (1, 1, args.hc_count * args.hidden_size)
    assert cache[2].shape == (1, 9, args.hc_count * args.hidden_size)
    np.testing.assert_array_equal(np.array(cache[3]), np.array([[3, 4]]))


def test_ple_right_padded_batch_caches_only_each_rows_valid_history():
    args = _ple_args()
    ple = PLELayer(args, ple_layer_index=0)
    batch_cache = ArraysCache(size=4)
    batch_cache.prepare(lengths=[3, 1])
    batch_ids = mx.array([[1, 2, 3], [7, 0, 0]])
    mask = mx.array([[True, True, True], [True, False, False]])
    batch_output = ple(
        mx.zeros((2, 3, args.hc_count * args.hidden_size)),
        batch_ids,
        batch_cache,
        mask,
    )

    single_cache = ArraysCache(size=4)
    single_output = ple(
        mx.zeros((1, 1, args.hc_count * args.hidden_size)),
        mx.array([[7]]),
        single_cache,
    )
    mx.eval(batch_output, single_output, batch_cache.state, single_cache.state)
    np.testing.assert_array_equal(np.array(batch_cache[3][1]), np.array([31, 7]))
    np.testing.assert_allclose(
        np.array(batch_cache[2][1]),
        np.array(single_cache[2][0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_qsa_cache_keeps_only_raw_ring_and_persistent_compressed_keys():
    cache = QSAIndexCache(compress_ratio=2)

    def transform(group, start):
        return group + start

    compressed = cache.update(mx.array([[[1.0], [3.0], [5.0], [7.0]]]), transform)
    mx.eval(compressed, cache.state)
    np.testing.assert_array_equal(np.array(compressed), np.array([[[2.0], [8.0]]]))
    assert cache.offset == 4
    assert cache.raw_ring.shape == (1, 2, 1)

    unchanged = cache.update(mx.array([[[9.0]]]), transform)
    mx.eval(unchanged, cache.state)
    np.testing.assert_array_equal(np.array(unchanged), np.array([[[2.0], [8.0]]]))
    np.testing.assert_array_equal(np.array(cache.raw_ring), np.array([[[9.0], [7.0]]]))
    assert cache.offset == 5
    assert cache.is_trimmable()

    restored = QSAIndexCache.from_state(cache.state, cache.meta_state)
    assert restored.offset == 5
    assert restored.compress_ratio == 2
    np.testing.assert_array_equal(
        np.array(restored.compressed_keys), np.array([[[2.0], [8.0]]])
    )


@pytest.mark.parametrize("length", [1, 4, 7, 8, 9])
def test_qsa_cache_vectorized_aligned_prefill_matches_scalar_update(length):
    values = mx.arange(length, dtype=mx.float32).reshape(1, length, 1)

    def transform(group, start):
        return group + start

    def transform_many(groups, starts):
        return groups + starts[None, :, None]

    scalar = QSAIndexCache(compress_ratio=4)
    vectorized = QSAIndexCache(compress_ratio=4)
    scalar.update(values, transform)
    vectorized.update(values, transform, transform_groups=transform_many)
    mx.eval(scalar.state, vectorized.state)

    assert scalar.meta_state == vectorized.meta_state
    np.testing.assert_array_equal(
        np.array(scalar.raw_ring), np.array(vectorized.raw_ring)
    )
    if scalar.compressed_keys is None:
        assert vectorized.compressed_keys is None
    else:
        np.testing.assert_array_equal(
            np.array(scalar.compressed_keys), np.array(vectorized.compressed_keys)
        )


def test_qsa_cache_vectorized_chunks_and_scalar_tail_keep_identical_state():
    def transform(group, start):
        return group + start

    def transform_many(groups, starts):
        return groups + starts[None, :, None]

    scalar = QSAIndexCache(compress_ratio=4)
    vectorized = QSAIndexCache(compress_ratio=4)
    # A small allocation step exercises initial allocation, reuse within the
    # existing capacity, and growth without constructing a synthetic 1K token
    # input solely for the production step=256 boundary.
    scalar.step = 4
    vectorized.step = 4
    for start, length in ((0, 8), (8, 8), (16, 8), (24, 7), (31, 1), (32, 5)):
        values = mx.arange(start, start + length, dtype=mx.float32).reshape(
            1, length, 1
        )
        scalar.update(values, transform)
        vectorized.update(values, transform, transform_groups=transform_many)
        mx.eval(scalar.state, vectorized.state)
        assert scalar.meta_state == vectorized.meta_state
        np.testing.assert_array_equal(
            np.array(scalar.raw_ring), np.array(vectorized.raw_ring)
        )
        np.testing.assert_array_equal(
            np.array(scalar.compressed_keys), np.array(vectorized.compressed_keys)
        )


def test_qsa_vectorized_cache_matches_architecture_reference():
    args = _args(
        indexer_budget=4,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    prompt = mx.arange(9 * args.hidden_size, dtype=mx.float32).reshape(
        1, 9, args.hidden_size
    )
    projected = attention.indexer.index_qk_proj(prompt)
    query_width = args.indexer_n_heads * args.indexer_head_dim
    _, raw_keys = mx.split(projected, [query_width], axis=-1)
    raw_keys = raw_keys.reshape(1, 9, 1, args.indexer_head_dim).squeeze(2)
    pooled = mx.mean(
        raw_keys[:, :8, :].reshape(1, 4, 2, args.indexer_head_dim).astype(mx.float32),
        axis=2,
    ).astype(raw_keys.dtype)
    normalized = attention.indexer.k_layernorm(pooled)
    expected = apply_qwen4_exp_rope(
        normalized[:, None, :, :],
        mx.arange(0, 8, 2, dtype=mx.int64)[None, :],
        rotary_dim=attention.indexer.rotary_dim,
        base=attention.indexer.rope_theta,
    )[:, 0, :, :]
    mx.eval(expected, raw_keys)

    cache = QSAIndexCache(compress_ratio=2)
    attention.indexer(prompt, cache, physical_kv_length=9)
    mx.eval(expected, cache.state)

    assert cache._compressed_count == 4
    np.testing.assert_array_equal(
        np.array(cache.compressed_keys[:, :4, :]), np.array(expected)
    )
    np.testing.assert_array_equal(
        np.array(cache.raw_ring),
        np.array(mx.concatenate([raw_keys[:, 8:9, :], raw_keys[:, 7:8, :]], axis=1)),
    )


@pytest.mark.parametrize("length", [8, 9])
def test_qsa_cache_rewinds_recoverable_group_and_recomputes_divergence(length):
    def transform(group, start):
        return group + start

    original = QSAIndexCache(compress_ratio=4)
    values = np.arange(1, length + 1, dtype=np.float32)
    original.update(mx.array(values.reshape(1, -1, 1)), transform)
    assert original.trim(1) == 1
    original.update(mx.array([[[99.0]]]), transform)

    cold = QSAIndexCache(compress_ratio=4)
    expected_values = np.concatenate([values[:-1], np.array([99.0])])
    cold.update(mx.array(expected_values.reshape(1, -1, 1)), transform)
    mx.eval(original.state, cold.state)
    assert original.offset == cold.offset == length
    np.testing.assert_array_equal(np.array(original.state[1]), np.array(cold.state[1]))
    np.testing.assert_array_equal(np.array(original.raw_ring), np.array(cold.raw_ring))


def test_qsa_cache_refuses_rewind_beyond_retained_raw_group():
    cache = QSAIndexCache(compress_ratio=4)
    cache.update(
        mx.arange(9, dtype=mx.float32).reshape(1, 9, 1),
        lambda group, start: group + start,
    )
    assert cache.trim(2) == 0
    assert cache.offset == 9


def test_scheduler_rollback_preflights_qsa_cachelist_for_full_rejection():
    """A multi-token rejection cannot trim KV before QSA refuses it."""
    from vllm_mlx.cache_rollback import can_trim, trim_all

    kv = KVCache()
    keys = mx.arange(9, dtype=mx.float32).reshape(1, 1, 9, 1)
    kv.update_and_fetch(keys, -keys)
    qsa = QSAIndexCache(compress_ratio=4)
    qsa.update(
        mx.arange(9, dtype=mx.float32).reshape(1, 9, 1),
        lambda group, start: group + start,
    )
    cache = CacheList(kv, qsa)

    assert cache.is_trimmable()
    assert not can_trim(cache, 2)
    assert not trim_all([cache], 2)
    assert kv.offset == qsa.offset == 9

    assert can_trim(cache, 1)
    assert trim_all([cache], 1)
    assert kv.offset == qsa.offset == 8


def test_suffix_scheduler_falls_through_before_qsa_multitoken_verify():
    from vllm_mlx.scheduler import _install_suffix_decoding

    kv = KVCache()
    keys = mx.arange(9, dtype=mx.float32).reshape(1, 1, 9, 1)
    kv.update_and_fetch(keys, -keys)
    qsa = QSAIndexCache(compress_ratio=4)
    qsa.update(
        mx.arange(9, dtype=mx.float32).reshape(1, 9, 1),
        lambda group, start: group + start,
    )

    class GenerationBatch:
        _next_tokens = mx.array([7], dtype=mx.int32)
        uids = [11]
        logits_processors = []
        prompt_cache = [CacheList(kv, qsa)]
        tokens = [[]]
        model = None

        def __init__(self):
            self.original_calls = 0

        def _step(self):
            self.original_calls += 1
            return [7], []

        def next(self):
            return []

    class BatchGenerator:
        def __init__(self):
            self._generation_batch = GenerationBatch()

        def remove(self, _uids, return_prompt_caches=False):
            return {} if return_prompt_caches else None

    class TwoTokenDrafter:
        max_draft_tokens = 2

        def add_generated_token(self, _token):
            return None

        def get_draft(self):
            return [8, 9]

    batch_generator = BatchGenerator()
    _install_suffix_decoding(
        batch_generator,
        model=None,
        profile=None,
        max_draft=2,
        max_suffix_len=2,
        min_confidence=0.0,
        requests={},
        uid_to_request_id={},
    )
    generation_batch = batch_generator._generation_batch
    generation_batch._suffix_drafters[11] = TwoTokenDrafter()

    assert generation_batch._step() == ([7], [])
    assert generation_batch.original_calls == 1
    assert generation_batch._suffix_stats["ft_non_trimmable_cache"] == 1
    assert kv.offset == qsa.offset == 9


def test_qsa_attention_prefill_and_decode_keep_both_cache_owners_aligned():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
    prompt = mx.zeros((1, 5, args.hidden_size))
    prompt_output = attention(prompt, cache)
    mx.eval(prompt_output, cache.state)
    assert prompt_output.shape == (1, 5, args.hidden_size)
    assert cache[0].offset == 5
    assert cache[1].offset == 5
    assert cache[1]._compressed_count == 2

    token_output = attention(mx.zeros((1, 1, args.hidden_size)), cache)
    mx.eval(token_output, cache.state)
    assert token_output.shape == (1, 1, args.hidden_size)
    assert cache[0].offset == 6
    assert cache[1].offset == 6
    assert cache[1]._compressed_count == 3


def test_qsa_attention_uses_reference_dense_path_below_sparse_budget(monkeypatch):
    args = _args(
        indexer_budget=8,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
    observed = []

    def fake_attention(queries, keys, values, *, cache, scale, mask):
        observed.append(mask)
        return mx.zeros_like(queries)

    monkeypatch.setattr(
        "vllm_mlx.models.qwen4_exp.scaled_dot_product_attention",
        fake_attention,
    )
    output = attention(mx.zeros((1, 5, args.hidden_size)), cache)
    mx.eval(output, cache.state)

    assert observed == ["causal"]
    assert cache[0].offset == cache[1].offset == 5


def test_qsa_vectorized_mask_preserves_budget_tail_and_causality():
    args = _args(indexer_budget=8, indexer_compress_ratio=2)
    selected = QSAIndexer(args)(
        mx.zeros((1, 65, args.hidden_size)),
        QSAIndexCache(compress_ratio=2),
        physical_kv_length=65,
    )
    assert selected is not None
    mx.eval(selected)
    mask = np.array(selected[0, 0])

    expected_counts = []
    for position in range(65):
        complete = (position + 1) // 2
        tail = position + 1 - complete * 2
        expected_counts.append(min(complete, 4) * 2 + tail)
    np.testing.assert_array_equal(mask.sum(axis=-1), expected_counts)
    assert mask[0, 0]
    assert mask[-1, -1]
    assert not np.any(np.triu(mask, k=1))


def test_qsa_batch_prefill_builds_mask_before_kv_update(monkeypatch):
    args = _args(indexer_budget=8, indexer_compress_ratio=2)
    attention = QSAAttention(args)
    qsa = QSAIndexCache(compress_ratio=2)
    qsa.left_padding = mx.array([0])
    cache = CacheList(BatchKVCache([0]), qsa)
    observed = []

    def fake_attention(queries, keys, values, *, cache, scale, mask):
        observed.append((keys.shape[-2], mask.shape[-1]))
        return mx.zeros_like(queries)

    monkeypatch.setattr(
        "vllm_mlx.models.qwen4_exp.scaled_dot_product_attention",
        fake_attention,
    )
    output = attention(mx.zeros((1, 5, args.hidden_size)), cache)
    mx.eval(output, cache.state)
    assert observed == [(5, 5)]


def test_scheduler_mid_prefill_restores_qsa_cachelist():
    """The live restore path recognizes the same vendored QSA side-cache."""
    from vllm_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    kv = KVCache()
    values = mx.arange(20, dtype=mx.float32).reshape(1, 1, 5, 4)
    kv.update_and_fetch(values, -values)
    qsa = QSAIndexCache(compress_ratio=2)
    qsa.update(
        mx.arange(20, dtype=mx.float32).reshape(1, 5, 4),
        lambda group, start: group + start,
    )
    original = CacheList(kv, qsa)

    restored = scheduler._reconstruct_cache_from_states(
        [
            {
                "state": original.state,
                "meta_state": original.meta_state,
                "class_ref": CacheList,
            }
        ]
    )

    assert restored is not None
    restored_qsa = restored[0].caches[1]
    assert isinstance(restored_qsa, QSAIndexCache)
    assert restored_qsa.offset == qsa.offset
    np.testing.assert_array_equal(
        np.array(restored_qsa.state[1]), np.array(qsa.state[1])
    )

    malformed = scheduler._reconstruct_cache_from_states(
        [
            {
                "state": original.state,
                "meta_state": ("missing-nested-metadata",),
                "class_ref": CacheList,
            }
        ]
    )
    assert malformed is None


def test_qsa_sparse_scores_use_one_reference_batched_matmul(monkeypatch):
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    indexer = QSAIndexer(args)
    cache = QSAIndexCache(compress_ratio=2)
    original = qwen4_exp.mx.matmul
    shapes = []

    def record_matmul(left, right):
        shapes.append((left.shape, right.shape))
        return original(left, right)

    monkeypatch.setattr(qwen4_exp.mx, "matmul", record_matmul)
    selected = indexer(
        mx.zeros((1, 6, args.hidden_size), dtype=mx.bfloat16),
        cache,
        physical_kv_length=6,
    )
    mx.eval(selected)
    assert shapes == [
        (
            (args.indexer_n_heads, 6, args.indexer_head_dim),
            (args.indexer_head_dim, 3),
        )
    ]


def test_qsa_indexer_fail_closed_internal_invariants():
    args = _args(indexer_budget=8, indexer_compress_ratio=2)
    indexer = QSAIndexer(args)
    indexer.num_kv_heads = 2
    indexer.index_qk_proj = qwen4_exp.nn.Linear(
        args.hidden_size,
        indexer.num_heads * indexer.head_dim + 2 * indexer.head_dim,
        bias=False,
    )
    with pytest.raises(ValueError, match="one indexer KV head"):
        indexer(
            mx.zeros((1, 2, args.hidden_size)),
            QSAIndexCache(compress_ratio=2),
            physical_kv_length=2,
        )

    inconsistent = QSAIndexer(args)
    inconsistent.block_topk = 1
    inconsistent_cache = QSAIndexCache(compress_ratio=2)
    inconsistent_cache._offsets = [4]
    inconsistent_cache._compressed_counts = [1]
    inconsistent_cache.raw_ring = mx.zeros((1, 2, args.indexer_head_dim))
    with pytest.raises(RuntimeError, match="selection was not materialized"):
        inconsistent(
            mx.zeros((1, 1, args.hidden_size)),
            inconsistent_cache,
            physical_kv_length=5,
        )


def test_qsa_cache_uses_standard_batch_lifecycle_without_rebuilding_history():
    def transform(group, start):
        return group + start

    first = QSAIndexCache(compress_ratio=2)
    second = QSAIndexCache(compress_ratio=2)
    first.update(mx.array([[[1.0], [3.0], [5.0]]]), transform)
    second.update(mx.array([[[2.0], [4.0], [6.0], [8.0], [10.0]]]), transform)

    batch = QSAIndexCache.merge([first, second])
    assert isinstance(batch, ArraysCache)
    assert batch._offsets == [3, 5]
    assert batch._compressed_counts == [1, 2]
    np.testing.assert_array_equal(np.array(batch.left_padding), np.array([2, 0]))

    batch.update(mx.array([[[7.0]], [[12.0]]]), transform)
    mx.eval(batch.state)
    assert batch._offsets == [4, 6]
    assert batch._compressed_counts == [2, 3]

    extracted = batch.extract(0)
    assert extracted.offset == 4
    np.testing.assert_array_equal(
        np.array(extracted.compressed_keys[:, :2]), np.array([[[2.0], [8.0]]])
    )

    batch.filter(mx.array([1]))
    assert batch._offsets == [6]
    np.testing.assert_array_equal(np.array(batch.left_padding), np.array([0]))
    batch.extend(QSAIndexCache.merge([extracted]))
    assert batch._offsets == [6, 4]
    np.testing.assert_array_equal(np.array(batch.left_padding), np.array([0, 2]))


def test_qsa_cache_skips_right_padding_and_preserves_physical_alignment():
    cache = QSAIndexCache(compress_ratio=2)
    # This is the same adoption seam used by mlx-lm's _make_cache for an
    # ArraysCache subclass.
    cache.left_padding = mx.array([0, 0])
    cache.prepare(lengths=[3, 1], right_padding=[0, 2])
    cache.update(
        mx.array(
            [
                [[1.0], [3.0], [5.0]],
                [[7.0], [99.0], [99.0]],
            ]
        ),
        lambda group, start: group + start,
    )
    assert cache._offsets == [3, 1]
    assert cache._compressed_counts == [1, 0]
    cache.finalize()
    np.testing.assert_array_equal(np.array(cache.left_padding), np.array([0, 2]))


def test_qsa_cache_skips_fresh_batch_left_padding():
    cache = QSAIndexCache(compress_ratio=2)
    cache.left_padding = mx.array([2, 0])
    cache.update(
        mx.array(
            [
                [[99.0], [99.0], [1.0], [3.0], [5.0]],
                [[2.0], [4.0], [6.0], [8.0], [10.0]],
            ]
        ),
        lambda group, start: group + start,
    )
    mx.eval(cache.state)
    assert cache._offsets == [3, 5]
    assert cache._compressed_counts == [1, 2]
    np.testing.assert_array_equal(
        np.array(cache.compressed_keys[0, :1]), np.array([[2.0]])
    )


def test_qsa_attention_continuous_batch_decode_matches_cache_lengths():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    caches = []
    for length in (3, 5):
        cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
        output = attention(mx.zeros((1, length, args.hidden_size)), cache)
        mx.eval(output, cache.state)
        caches.append(cache)

    batch_cache = CacheList.merge(caches)
    output = attention(mx.zeros((2, 1, args.hidden_size)), batch_cache)
    mx.eval(output, batch_cache.state)
    assert output.shape == (2, 1, args.hidden_size)
    np.testing.assert_array_equal(np.array(batch_cache[0].offset), np.array([4, 6]))
    np.testing.assert_array_equal(np.array(batch_cache[1].offset), np.array([4, 6]))


def test_qsa_attention_filter_preserves_shorter_row_padding_during_decode():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    caches = []
    for length in (3, 5):
        cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
        output = attention(mx.zeros((1, length, args.hidden_size)), cache)
        mx.eval(output, cache.state)
        caches.append(cache)

    filtered = CacheList.merge(caches)
    compact = filtered.extract(0)
    filtered.filter(mx.array([0]))
    np.testing.assert_array_equal(np.array(filtered[1].left_padding), np.array([2]))

    compact_output = attention(mx.zeros((1, 1, args.hidden_size)), compact)
    filtered_output = attention(mx.zeros((1, 1, args.hidden_size)), filtered)
    mx.eval(compact_output, filtered_output, compact.state, filtered.state)

    np.testing.assert_allclose(
        np.array(filtered_output), np.array(compact_output), rtol=0, atol=0
    )
    np.testing.assert_array_equal(np.array(filtered[0].offset), np.array([4]))
    np.testing.assert_array_equal(np.array(filtered[1].offset), np.array([4]))


def test_qsa_attention_fresh_left_padded_batch_aligns_with_main_kv():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    qsa = QSAIndexCache(compress_ratio=2)
    # mlx-lm adopts an ArraysCache subclass by assigning this metadata.
    qsa.left_padding = mx.array([2, 0])
    cache = CacheList(BatchKVCache([2, 0]), qsa)
    output = attention(mx.zeros((2, 5, args.hidden_size)), cache)
    mx.eval(output, cache.state)
    assert output.shape == (2, 5, args.hidden_size)
    np.testing.assert_array_equal(np.array(cache[0].offset), np.array([3, 5]))
    np.testing.assert_array_equal(np.array(cache[1].offset), np.array([3, 5]))


def test_complete_synthetic_text_model_prefill_and_decode():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    cache = model.make_cache()
    prompt = mx.array([[1, 2, 3]])
    logits = model(prompt, cache=cache)
    mx.eval(logits, [layer.state for layer in cache])
    assert logits.shape == (1, 3, args.vocab_size)
    assert cache[0][0].shape == (1, 2, 20)
    assert cache[0][2].shape == (1, 9, args.hc_count * args.hidden_size)
    assert cache[1][0].offset == 3
    assert cache[1][1].offset == 3

    next_logits = model(mx.array([[4]]), cache=cache)
    mx.eval(next_logits, [layer.state for layer in cache])
    assert next_logits.shape == (1, 1, args.vocab_size)
    assert cache[1][0].offset == 4
    assert cache[1][1].offset == 4


def test_return_hidden_exposes_pre_final_mixer_multistream():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    logits, hidden = model(mx.array([[1, 2, 3]]), return_hidden=True)
    mx.eval(logits, hidden)

    assert logits.shape == (1, 3, args.vocab_size)
    assert hidden.shape == (1, 3, args.hc_count * args.hidden_size)


def test_speculative_reject_restores_gdn_ple_and_qsa_state():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    baseline_cache = model.make_cache()
    verify_cache = model.make_cache()
    prompt = mx.array([[1, 2, 3]])
    mx.eval(model(prompt, cache=baseline_cache), model(prompt, cache=verify_cache))

    mx.eval(model(mx.array([[4]]), cache=baseline_cache))
    baseline_logits = model(mx.array([[6]]), cache=baseline_cache)

    verify_logits, verify_hidden = model(
        mx.array([[4, 5]]),
        cache=verify_cache,
        return_hidden=True,
        n_confirmed=1,
    )
    mx.eval(verify_logits, verify_hidden)
    for layer_cache in verify_cache:
        if isinstance(layer_cache, Qwen4ExpStateCache):
            layer_cache.restore_rollback(1, 2)
        elif layer_cache.is_trimmable():
            assert layer_cache.trim(1) == 1
    restored_logits = model(mx.array([[6]]), cache=verify_cache)
    mx.eval(
        baseline_logits,
        restored_logits,
        [cache.state for cache in baseline_cache],
        [cache.state for cache in verify_cache],
    )

    np.testing.assert_array_equal(
        np.array(mx.argmax(restored_logits, axis=-1)),
        np.array(mx.argmax(baseline_logits, axis=-1)),
    )
    np.testing.assert_allclose(
        np.array(restored_logits), np.array(baseline_logits), rtol=1e-5, atol=1e-6
    )
    for baseline, restored in zip(baseline_cache, verify_cache):
        if isinstance(baseline, Qwen4ExpStateCache):
            for expected, actual in zip(baseline.state, restored.state):
                np.testing.assert_allclose(
                    np.array(actual), np.array(expected), rtol=1e-5, atol=1e-6
                )
        else:
            assert baseline[0].offset == restored[0].offset
            assert baseline[1].offset == restored[1].offset


def test_qwen4_state_cache_restores_atomic_slot_boundary():
    cache = Qwen4ExpStateCache(size=2)
    cache.cache = [mx.array([1]), mx.array([2])]
    cache.record_slot_snapshots(0, [mx.array([1])])
    cache.record_slot_snapshots(1, [mx.array([2])], finalize=True)
    cache.cache = [mx.array([3]), mx.array([4])]
    cache.restore_rollback(1, 2)
    np.testing.assert_array_equal(np.array(cache.cache[0]), np.array([1]))
    np.testing.assert_array_equal(np.array(cache.cache[1]), np.array([2]))


def test_qwen4_state_cache_rejects_incomplete_or_invalid_boundaries():
    cache = Qwen4ExpStateCache(size=2)
    cache.cache = [mx.array([1]), mx.array([2])]

    cache.record_slot_snapshots(0, [])
    cache.record_slot_snapshots(0, [mx.array([1])])
    with pytest.raises(AssertionError, match="every state slot"):
        cache.record_slot_snapshots(0, [mx.array([1])], finalize=True)

    cache._rollback_slots = None
    cache.record_slot_snapshots(0, [mx.array([1])])
    with pytest.raises(AssertionError, match="lengths diverged"):
        cache.record_slot_snapshots(
            1,
            [mx.array([2]), mx.array([3])],
            finalize=True,
        )

    cache.rollback_state = None
    with pytest.raises(AssertionError, match="no saved boundary"):
        cache.restore_rollback(1, 2)

    cache.rollback_state = [[mx.array([1]), mx.array([2])]]
    with pytest.raises(AssertionError, match="invalid Qwen4 rollback boundary"):
        cache.restore_rollback(0, 2)


def test_qwen4_verify_block_matches_tokenwise_forward():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    block_cache = model.make_cache()
    step_cache = model.make_cache()
    prompt = mx.array([[1, 2, 3]])
    mx.eval(model(prompt, cache=block_cache), model(prompt, cache=step_cache))

    verify = mx.array([[4, 5, 6]])
    block_logits = model(verify, cache=block_cache)
    step_logits = mx.concatenate(
        [model(verify[:, i : i + 1], cache=step_cache) for i in range(3)],
        axis=1,
    )
    mx.eval(block_logits, step_logits)
    np.testing.assert_array_equal(
        np.array(mx.argmax(block_logits, axis=-1)),
        np.array(mx.argmax(step_logits, axis=-1)),
    )
    np.testing.assert_allclose(
        np.array(block_logits), np.array(step_logits), rtol=1e-5, atol=2e-7
    )


def test_qwen4_mtp_target_verify_matches_synthetic_greedy(monkeypatch):
    import importlib

    import mlx.nn as nn
    from mlx_lm.generate import generate_step

    from vllm_mlx.spec_decode.mtp import MTPAcceptCounter, dispatch_mtp_inject
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    # Earlier executor tests may leave MLX's process-global default tagged to
    # a worker-owned stream. Rebind this real-device test to the current
    # thread before its random model parameters are allocated.
    current_stream = mx.new_stream(mx.default_device())
    mx.set_default_stream(current_stream)
    importlib.import_module("mlx_lm.generate").generation_stream = current_stream
    args = _ple_args()
    args.mtp_num_hidden_layers = 1
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    prompt = mx.array([1, 2, 3], dtype=mx.uint32)
    baseline = [int(token) for token, _ in generate_step(prompt, model, max_tokens=12)]

    monkeypatch.setattr(nn, "quantize", lambda *_args, **_kwargs: None)
    assert dispatch_mtp_inject(model, "qwen4_exp", allow_random_init=True) is True
    counter = MTPAcceptCounter()
    speculative = [
        int(token)
        for token, _logprobs, _draft in mtp_generate_step(
            prompt,
            model.language_model,
            max_tokens=12,
            max_k=1,
            disable_auto_k=True,
            accept_counter=counter,
        )
    ]

    assert speculative == baseline
    assert counter.snapshot().attempts > 0


def test_qwen4_native_mtp_dispatch_attaches_synthetic_head(monkeypatch):
    import mlx.nn as nn

    from vllm_mlx.spec_decode.mtp import dispatch_mtp_inject, dispatch_mtp_validate
    from vllm_mlx.spec_decode.mtp.dispatch import (
        _MTP_INJECT_DISPATCH,
        _MTP_VALIDATE_DISPATCH,
    )

    args = _ple_args(tie_word_embeddings=True)
    args.mtp_num_hidden_layers = 1
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    # The tiny fixture's hidden width is below a production q4 group. Weight
    # quantization is separately covered by the real checkpoint experiment;
    # this unit test exercises dispatch and protocol attachment only.
    monkeypatch.setattr(nn, "quantize", lambda *_args, **_kwargs: None)

    assert _MTP_INJECT_DISPATCH["qwen4_exp"] == (
        "vllm_mlx.spec_decode.mtp.qwen4_exp_inject",
        "inject_qwen4_exp_mtp_support",
    )
    assert _MTP_VALIDATE_DISPATCH["qwen4_exp"] == (
        "vllm_mlx.spec_decode.mtp.qwen4_exp_inject",
        "validate_qwen4_exp_mtp_support",
    )
    assert dispatch_mtp_inject(model, "qwen4_exp", allow_random_init=True) is True
    assert dispatch_mtp_validate(model, "qwen4_exp") is True
    assert model.mtp_max_speculative_tokens == 1

    inner = model.language_model
    cache = inner.make_mtp_cache()
    logits, hidden = inner(
        mx.array([[1]]), cache=inner.make_cache(), return_hidden=True
    )
    mtp_logits = inner.mtp_forward(hidden, mx.array([[2]]), cache)
    mx.eval(logits, mtp_logits)
    assert mtp_logits.shape == (1, 1, args.vocab_size)


def test_qwen4_mtp_checkpoint_file_discovery_and_weight_sanitize(tmp_path):
    from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

    direct = tmp_path / "direct.safetensors"
    direct.touch()
    assert inject._mtp_weight_files(direct) == [direct]

    with pytest.raises(FileNotFoundError, match="does not exist"):
        inject._mtp_weight_files(tmp_path / "missing")

    indexed = tmp_path / "indexed"
    indexed.mkdir()
    (indexed / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.a": "b.safetensors",
                    "mtp.b": "a.safetensors",
                    "model.c": "ignored.safetensors",
                }
            }
        )
    )
    assert inject._mtp_weight_files(indexed) == [
        indexed / "a.safetensors",
        indexed / "b.safetensors",
    ]

    single = tmp_path / "single"
    single.mkdir()
    (single / "model.safetensors").touch()
    assert inject._mtp_weight_files(single) == [single / "model.safetensors"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"No safetensors containing mtp\.\*"):
        inject._mtp_weight_files(empty)

    gate_up = mx.arange(16).reshape(2, 4, 2)
    down = mx.ones((2, 2, 2))
    sanitized = inject._sanitize_mtp_weights(
        {
            "model.ignored": mx.zeros((1,)),
            "mtp.layers.0.mlp.experts.gate_up_proj": gate_up,
            "mtp.layers.0.mlp.experts.gate_up_proj.unknown": gate_up,
            "mtp.layers.0.mlp.experts.down_proj": down,
            "mtp.layers.0.keep": mx.ones((1,)),
        }
    )
    assert set(sanitized) == {
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.experts.gate_up_proj.unknown",
        "layers.0.mlp.switch_mlp.down_proj.weight",
        "layers.0.keep",
    }
    np.testing.assert_array_equal(
        np.asarray(sanitized["layers.0.mlp.switch_mlp.gate_proj.weight"]),
        np.asarray(gate_up[..., :2, :]),
    )


def test_qwen4_mtp_quantization_predicate_delegates_only_quantizable_modules(
    monkeypatch,
):
    import mlx.nn as nn

    from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

    args = _ple_args()
    args.mtp_num_hidden_layers = 1
    inner = Model(
        ModelArgs(model_type="qwen4_exp", text_config=asdict(args))
    ).language_model
    observed = []

    class Quantizable:
        to_quantized = object()

    def fake_quantize(_model, *, class_predicate, **_kwargs):
        observed.append(class_predicate("plain", object()))
        observed.append(class_predicate("quantized", Quantizable()))

    monkeypatch.setattr(nn, "quantize", fake_quantize)
    inject._build_mtp(inner)
    assert observed == [False, True]


def test_qwen4_mtp_inject_loads_complete_local_tensor_contract(tmp_path, monkeypatch):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

    args = _ple_args()
    args.mtp_num_hidden_layers = 1
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    monkeypatch.setattr(nn, "quantize", lambda *_args, **_kwargs: None)
    expected_mtp = inject._build_mtp(model.language_model)
    checkpoint = tmp_path / "mtp.safetensors"
    mx.save_safetensors(
        str(checkpoint),
        {f"mtp.{key}": value for key, value in tree_flatten(expected_mtp.parameters())},
    )

    assert inject.inject_qwen4_exp_mtp_support(model, mtp_sidecar=checkpoint) is True
    assert inject.validate_qwen4_exp_mtp_support(model) is True


def test_qwen4_mtp_inject_fails_closed_on_guards_tensor_mismatch_and_exception(
    tmp_path,
    monkeypatch,
    caplog,
):
    import mlx.nn as nn

    from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

    assert inject._resolve_inner(object()) is None
    assert inject.validate_qwen4_exp_mtp_support(object()) is False
    direct = SimpleNamespace(model_type="qwen4_exp_text")
    assert inject._resolve_inner(direct) is direct
    assert (
        inject.inject_qwen4_exp_mtp_support(object(), allow_random_init=True) is False
    )

    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    assert inject.inject_qwen4_exp_mtp_support(model, allow_random_init=True) is False

    args.mtp_num_hidden_layers = 1
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    assert inject.inject_qwen4_exp_mtp_support(model) is False

    monkeypatch.setattr(nn, "quantize", lambda *_args, **_kwargs: None)
    bad_checkpoint = tmp_path / "bad.safetensors"
    mx.save_safetensors(str(bad_checkpoint), {"mtp.unexpected": mx.ones((2,))})
    assert (
        inject.inject_qwen4_exp_mtp_support(model, mtp_sidecar=bad_checkpoint) is False
    )
    assert "tensor contract mismatch" in caplog.text

    monkeypatch.setattr(
        inject,
        "_build_mtp",
        lambda _inner: (_ for _ in ()).throw(RuntimeError("injected build failure")),
    )
    assert inject.inject_qwen4_exp_mtp_support(model, allow_random_init=True) is False
    assert "native MTP attachment failed" in caplog.text


def test_qwen4_mtp_untied_logits_and_validation_signature_failure(monkeypatch):
    import mlx.nn as nn

    from vllm_mlx.spec_decode.mtp import qwen4_exp_inject as inject

    args = _ple_args(tie_word_embeddings=False)
    args.mtp_num_hidden_layers = 1
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    monkeypatch.setattr(nn, "quantize", lambda *_args, **_kwargs: None)
    assert inject.inject_qwen4_exp_mtp_support(model, allow_random_init=True) is True

    inner = model.language_model
    _, hidden = inner(mx.array([[1]]), cache=inner.make_cache(), return_hidden=True)
    logits = inner.mtp_forward(hidden, mx.array([[2]]), inner.make_mtp_cache())
    mx.eval(logits)
    assert logits.shape == (1, 1, args.vocab_size)

    monkeypatch.setattr(
        inject.inspect,
        "signature",
        lambda _call: (_ for _ in ()).throw(ValueError("injected signature failure")),
    )
    assert inject.validate_qwen4_exp_mtp_support(model) is False


def test_sanitize_preserves_ple_shards_and_maps_experts_without_concat():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros((4, 8, 8)),
        "model.language_model.layers.0.mlp.experts.down_proj": mx.zeros((4, 8, 4)),
        "model.language_model.layers.0.mlp.experts.gate_up_proj.scales": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.mlp.experts.gate_up_proj.biases": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.mlp.experts.down_proj.scales": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.mlp.experts.down_proj.biases": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.zeros((20, 1, 3)),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_3.weight": mx.zeros(
            (32, 1)
        ),
        "model.visual.blocks.0.weight": mx.zeros((1,)),
        "mtp.layers.0.weight": mx.zeros((1,)),
    }
    sanitized = model.sanitize(weights)
    assert "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight" in sanitized
    assert "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight" in sanitized
    assert "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight" in sanitized
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for auxiliary in ("scales", "biases"):
            assert (
                f"language_model.model.layers.0.mlp.switch_mlp."
                f"{projection}.{auxiliary}" in sanitized
            )
    conv = sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"]
    assert conv.shape == (20, 3, 1)
    assert (
        "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.3.weight"
        in sanitized
    )
    assert all("visual" not in key and not key.startswith("mtp") for key in sanitized)


def test_converter_quantized_keys_match_loader_sanitizer_contract():
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(_ple_args())))
    ordinary = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert quantized_tensor_names(ordinary) == (
        ordinary,
        "model.language_model.layers.0.self_attn.q_proj.scales",
        "model.language_model.layers.0.self_attn.q_proj.biases",
    )

    emitted = {}
    for name, shape in (
        (ordinary, (8, 8)),
        ("model.language_model.layers.0.mlp.experts.gate_up_proj", (4, 8, 8)),
        ("model.language_model.layers.0.mlp.experts.down_proj", (4, 8, 4)),
    ):
        weight, scales, biases = quantized_tensor_names(name)
        emitted[weight] = mx.zeros(shape)
        aux_shape = (*shape[:-2], shape[-2], 1)
        emitted[scales] = mx.zeros(aux_shape)
        emitted[biases] = mx.zeros(aux_shape)

    sanitized = model.sanitize(emitted)
    expected = {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.scales",
        "language_model.model.layers.0.self_attn.q_proj.biases",
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        expected.update(
            {
                f"language_model.model.layers.0.mlp.switch_mlp.{projection}.weight",
                f"language_model.model.layers.0.mlp.switch_mlp.{projection}.scales",
                f"language_model.model.layers.0.mlp.switch_mlp.{projection}.biases",
            }
        )
    assert set(sanitized) == expected


def test_quantized_aux_repair_preflights_cross_shard_collision(tmp_path):
    output = _repair_fixture(
        tmp_path,
        {
            "model-00001-of-00002.safetensors": {
                "layer.weight.scales": mx.array([1.0])
            },
            "model-00002-of-00002.safetensors": {"layer.scales": mx.array([2.0])},
        },
        {
            "layer.weight.scales": "model-00001-of-00002.safetensors",
            "layer.scales": "model-00002-of-00002.safetensors",
        },
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(RuntimeError, match="output index rename collision"):
        converter.repair_quantized_aux_names(output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_quantized_aux_repair_commits_validated_plan(tmp_path):
    shard_name = "model-00001-of-00001.safetensors"
    output = _repair_fixture(
        tmp_path,
        {shard_name: {"layer.weight.scales": mx.array([1.0])}},
        {"layer.weight.scales": shard_name},
    )

    result = converter.repair_quantized_aux_names(output)

    assert result["changed_keys"] == result["changed_shards"] == 1
    tensors = mx.load(str(output / shard_name))
    assert set(tensors) == {"layer.scales"}
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"layer.scales": shard_name}


def test_quantized_aux_repair_rolls_back_commit_failure(tmp_path, monkeypatch):
    output = _repair_fixture(
        tmp_path,
        {"model-00001-of-00001.safetensors": {"layer.weight.scales": mx.array([1.0])}},
        {"layer.weight.scales": "model-00001-of-00001.safetensors"},
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(
        converter,
        "_write_sha256sums",
        lambda _output: (_ for _ in ()).throw(OSError("injected checksum failure")),
    )

    with pytest.raises(OSError, match="injected checksum failure"):
        converter.repair_quantized_aux_names(output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_converter_rejects_plain_source_symlink_escape(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not model data")
    (source / "model.safetensors").symlink_to(outside)

    with pytest.raises(RuntimeError, match="escapes model cache root"):
        converter._safe_source_shard(source, Path("model.safetensors"))


def test_converter_allows_standard_hf_snapshot_blob_symlink(tmp_path):
    model_root = tmp_path / "models--org--repo"
    snapshot = model_root / "snapshots" / "revision"
    blobs = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "abc123"
    blob.write_bytes(b"model data")
    (snapshot / "model.safetensors").symlink_to(blob)

    assert converter._safe_source_shard(snapshot, Path("model.safetensors")) == blob


def test_converter_failure_never_publishes_partial_output(tmp_path, monkeypatch):
    final = tmp_path / "published"

    def fail_after_partial_write(_source, staging, **_kwargs):
        (staging / "partial.safetensors").write_bytes(b"partial")
        raise OSError("injected conversion failure")

    monkeypatch.setattr(converter, "_convert_into", fail_after_partial_write)
    with pytest.raises(OSError, match="injected conversion failure"):
        converter.convert(
            tmp_path / "source",
            final,
            max_shard_bytes=1024,
            min_free_bytes=0,
        )

    assert not final.exists()
    assert list(tmp_path.glob(".published.staging-*")) == []


def test_quantization_contract_uses_shape_exact_ple_groups_and_q8_routing():
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=_ple_args().__dict__))
    predicate = model.quant_predicate

    assert predicate(
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.3",
        object(),
    ) == {"group_size": 32, "bits": 4}
    assert predicate("language_model.model.layers.1.mlp.gate", object()) == {
        "group_size": 64,
        "bits": 8,
    }
    assert predicate(
        "language_model.model.layers.1.mlp.shared_expert_gate", object()
    ) == {"group_size": 64, "bits": 8}
    assert predicate("language_model.model.layers.1.self_attn.q_proj", object()) is True


def test_qsa_cache_fail_closed_guards_and_diagnostics():
    with pytest.raises(ValueError, match="compression ratio must be positive"):
        QSAIndexCache(compress_ratio=0)

    cache = QSAIndexCache(compress_ratio=2)
    cache.update(mx.ones((1, 1, 3)), lambda pooled, _position: pooled)
    with pytest.raises(ValueError, match="shape changed"):
        cache.update(mx.ones((1, 1, 4)), lambda pooled, _position: pooled)
    with pytest.raises(ValueError, match="out of range"):
        cache.keys_for_blocks(0, 1)
    assert cache.keys_for_blocks(0, 0).shape == (0, 0)
    assert cache.valid_lengths(2) == [2]
    assert cache.can_trim(-1) is False
    assert cache.can_trim(0) is True
    checkpoint = cache.trim_checkpoint()
    cache.restore_trim_checkpoint(checkpoint)
    assert cache.size() == 1
    assert cache.empty() is False
    assert cache.nbytes > 0

    batched = QSAIndexCache.merge([cache, cache.extract(0)])
    with pytest.raises(AttributeError, match="per-row compressed counts"):
        _ = batched._compressed_count
    batched.prepare(lengths=[1, 1])
    batched.filter(mx.array([1], dtype=mx.int32))
    assert batched.size() == 1

    inconsistent = QSAIndexCache(compress_ratio=2)
    inconsistent._compressed_counts = [1]
    with pytest.raises(ValueError, match="compressed cache is empty"):
        inconsistent.keys_for_blocks(0, 1)


def test_qsa_cache_rejects_invalid_batch_and_merge_contracts():
    cache = QSAIndexCache(compress_ratio=2, left_padding=[0])
    with pytest.raises(ValueError, match="batch metadata"):
        cache._ensure_batch(2)

    committed = QSAIndexCache(compress_ratio=2)
    committed.update(mx.ones((1, 1, 2)), lambda pooled, _position: pooled)
    with pytest.raises(ValueError, match="outside cache lifecycle"):
        committed._ensure_batch(2)

    with pytest.raises(ValueError, match="empty QSA cache list"):
        QSAIndexCache.merge([])
    with pytest.raises(ValueError, match="share compression ratio"):
        QSAIndexCache.merge(
            [QSAIndexCache(compress_ratio=2), QSAIndexCache(compress_ratio=4)]
        )


def test_model_wrapper_properties_sanitize_and_tied_logits():
    args = _ple_args(tie_word_embeddings=True)
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    assert model.model is model.language_model.model
    assert model.layers is model.language_model.layers
    assert model.cast_predicate("layers.0.A_log") is False
    assert model.cast_predicate("layers.0.weight") is True

    mapped = model.sanitize(
        {
            "model.visual.weight": mx.ones((1,)),
            "vision_tower.weight": mx.ones((1,)),
            "mtp.weight": mx.ones((1,)),
            "model.language_model.embed_tokens.weight": mx.ones((2, 2)),
            "model.norm.weight": mx.ones((2,)),
            "language_model.model.keep.weight": mx.ones((2,)),
            "language_model.model.layers.0.mlp.experts.gate_up_proj.unknown": mx.ones(
                (1, 2, 2)
            ),
        }
    )
    assert all("visual" not in key and "mtp" not in key for key in mapped)
    assert any(key.endswith("gate_up_proj.unknown") for key in mapped)

    embeddings = mx.zeros((1, 2, args.hidden_size))
    logits = model(mx.array([[1, 2]]), input_embeddings=embeddings)
    assert logits.shape == (1, 2, args.vocab_size)


def test_experimental_capability_uses_live_residency_truth(monkeypatch):
    from vllm_mlx.routes import models as models_route

    entry = SimpleNamespace(
        experimental=True,
        matches=lambda model_id: model_id == "served-flash",
    )
    registry = SimpleNamespace(get_entry=lambda _model_id: entry)
    monkeypatch.setattr(
        models_route,
        "get_config",
        lambda: SimpleNamespace(
            model_registry=registry,
            model_name=None,
            model_alias=None,
            engine=None,
        ),
    )
    assert models_route._served_experimental("served-flash") is True
    assert models_route._served_experimental("other") is False

    registry.get_entry = lambda _model_id: (_ for _ in ()).throw(KeyError("gone"))
    assert models_route._served_experimental("served-flash") is False

    monkeypatch.setattr(
        models_route,
        "get_config",
        lambda: SimpleNamespace(
            model_registry=None,
            model_name="served-flash",
            model_alias=None,
            engine=SimpleNamespace(experimental=True),
            embedding_model_locked=False,
        ),
    )
    assert models_route._served_experimental("served-flash") is True
    assert models_route._served_experimental("other") is False
    assert (
        models_route._detect_capabilities(
            "served-flash",
            profile_modality="text",
            is_text_only=True,
            profile_tool_parser=None,
            experimental=True,
        )[-1]
        == "experimental"
    )


def test_qwen4_registration_probes_native_and_fails_closed(monkeypatch, caplog):
    import builtins
    import importlib.util
    import sys

    import vllm_mlx.models as model_package
    from vllm_mlx.utils import tokenizer

    module_name = "mlx_lm.models.qwen4_exp"
    monkeypatch.setattr(
        tokenizer, "_VENDORED_MODEL_TYPES", set(tokenizer._VENDORED_MODEL_TYPES)
    )
    tokenizer._register_vendored_archs()

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == module_name else original_find_spec(name),
    )
    tokenizer._register_vendored_archs()
    assert "qwen4_exp" in tokenizer._VENDORED_MODEL_TYPES

    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def rejecting_probe(name):
        if name == module_name:
            raise ValueError("injected missing parent")
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", rejecting_probe)
    tokenizer._register_vendored_archs()
    assert module_name in sys.modules

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "vllm_mlx.models.qwen4_exp", raising=False)
    monkeypatch.delattr(model_package, "qwen4_exp", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    original_import = builtins.__import__

    def rejecting_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 2 and "qwen4_exp" in fromlist:
            raise ImportError("injected vendored import failure")
        return original_import(name, globals, locals, fromlist, level)

    tokenizer._VENDORED_MODEL_TYPES.discard("qwen4_exp")
    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    tokenizer._register_vendored_archs()
    assert "qwen4_exp" not in tokenizer._VENDORED_MODEL_TYPES
    assert "failed to register" in caplog.text


def test_qwen4_gdn_verify_single_step_initializes_state_and_empty_boundaries():
    """Exercise the one-token reference edge after other GPU tests.

    MLX's shapeless compile cache can recycle a thread-local stream after a
    one-step shape, so keep this edge last in the module while still asserting
    the production fallback used outside multi-token verification.
    """
    from vllm_mlx.kernels.qwen4_gdn_verify import gated_delta_verify_with_states

    output, state, boundaries = gated_delta_verify_with_states(
        mx.ones((1, 1, 1, 4)),
        mx.ones((1, 1, 1, 4)),
        mx.ones((1, 1, 2, 3)),
        mx.zeros((1, 1, 2)),
        mx.zeros((1, 1, 2)),
        mx.zeros((2,)),
        mx.zeros((2,)),
        use_kernel=False,
    )
    mx.eval(output, state, boundaries)

    assert output.shape == (1, 1, 2, 3)
    assert state.shape == (1, 2, 3, 4)
    assert boundaries.shape == (1, 0, 2, 3, 4)
