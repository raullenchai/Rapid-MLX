import importlib
import sys

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
)

from rapid_mlx.models.lfm3 import Model, ModelArgs, RotaryEmbedding


def config(**kw):
    args = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        conv_L_cache=3,
        layer_types=["conv", "full_attention", "sliding_attention"],
        sliding_window=8,
    )
    args.update(kw)
    return ModelArgs(**args)


def test_shapes_and_caches():
    model = Model(config())
    assert model(mx.array([[1, 2, 3]])).shape == (1, 3, 32)
    cache = model.make_cache()
    assert [type(c) for c in cache] == [ArraysCache, KVCache, RotatingKVCache]


def test_incremental_and_prefix():
    from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(42)
        model = Model(config(sliding_window=256))
        model.set_dtype(mx.float32)
        ids = [i % 32 for i in range(300)]
        tokens = mx.array([ids])
        full = model(tokens)
        for chunk_size in (1, 37):
            cache = model.make_cache()
            chunks = [
                model(tokens[:, i : i + chunk_size], cache=cache)
                for i in range(0, len(ids), chunk_size)
            ]
            incremental = mx.concatenate(chunks, axis=1)
            mx.eval(full, incremental)
            assert mx.max(mx.abs(full - incremental)).item() <= 1e-3
            assert mx.array_equal(
                mx.argmax(full, axis=-1), mx.argmax(incremental, axis=-1)
            ).item()

        prefix = model.make_cache()
        mx.eval(model(tokens[:, :260], cache=prefix))
        store = MemoryAwarePrefixCache(
            model, MemoryCacheConfig(max_memory_mb=64, hybrid_reuse_max_entries=2)
        )
        assert store.store(ids[:260], prefix)
        cached, rest = store.fetch(ids)
        assert cached is not None and rest == ids[260:]
        warm = model(mx.array([rest]), cache=cached)
        cold_cache = model.make_cache()
        mx.eval(model(tokens[:, :260], cache=cold_cache))
        cold = model(tokens[:, 260:], cache=cold_cache)
        mx.eval(warm, cold)
        assert mx.max(mx.abs(warm - cold)).item() <= 1e-3
        assert mx.array_equal(mx.argmax(warm, axis=-1), mx.argmax(cold, axis=-1)).item()
        assert not can_trim_prompt_cache(cached)
    finally:
        mx.set_default_device(mx.gpu)


def test_rope_and_norm_and_tying():
    model = Model(config())
    full = model.layers[1].self_attn.rope
    sliding = model.layers[2].self_attn.rope
    assert (full.base, full.angles) == (1e6, 2)
    assert (sliding.base, sliding.angles) == (1e4, 4)
    x = mx.random.normal((1, 2, 3, 8))
    assert mx.array_equal(full(x, position_ids=mx.array([[0, 1, 2]])), full(x)).item()
    assert "lm_head" not in model
    assert "post_operator_norm" in model.layers[0]
    untied = Model(config(use_sandwich_norm=False, tie_word_embeddings=False))
    assert "post_operator_norm" not in untied.layers[0]
    assert untied(mx.array([[1]])).shape == (1, 1, 32)
    with pytest.raises(ValueError):
        RotaryEmbedding(8, {"rope_type": "unsupported"})


def test_loader_registration_without_vlm(monkeypatch):
    from rapid_mlx.utils.tokenizer import (
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    _register_vendored_archs()
    assert importlib.import_module("mlx_lm.models.lfm3").Model is Model
    assert "lfm3" in _VENDORED_MODEL_TYPES


def test_sanitize_and_validation():
    model = Model(config())
    conv = mx.zeros((16, 1, 3))
    weights = model.sanitize(
        {
            "language_model.model.layers.0.conv.conv.weight": conv,
            "lm_head.weight": mx.zeros((32, 16)),
        }
    )
    assert weights["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "lm_head.weight" not in weights
    with pytest.raises(ValueError):
        config(layer_types=None)
    with pytest.raises(ValueError):
        config(layer_types=["bad"] * 3)
    with pytest.raises(ValueError):
        config(sliding_window=0)


def test_additional_inputs_and_rope_validation():
    model = Model(config())
    ids = mx.array([[1, 2]])
    embeds = model.model.embed_tokens(ids)
    assert mx.allclose(model(input_ids=ids), model(input_embeddings=embeds)).item()
    assert model.head_dim == 8 and model.n_kv_heads == 1
    rope = RotaryEmbedding(8, {})
    assert rope(mx.ones((2, 1, 2, 8)), offset=mx.array([0, 2])).shape == (2, 1, 2, 8)
    with pytest.raises(ValueError):
        RotaryEmbedding(8, {"partial_rotary_factor": 0})


def test_real_prefix_cache_reuse():
    from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    mx.set_default_device(mx.cpu)
    try:
        model = Model(config(sliding_window=8))
        ids = list(range(20))
        tokens = mx.array([ids])
        prefix = model.make_cache()
        mx.eval(model(tokens[:, :12], cache=prefix))
        store = MemoryAwarePrefixCache(
            model, MemoryCacheConfig(max_memory_mb=64, hybrid_reuse_max_entries=2)
        )
        assert store.store(ids[:12], prefix)
        cached, rest = store.fetch(ids)
        assert cached is not None and rest == ids[12:]
        warm = model(mx.array([rest]), cache=cached)
        cold_cache = model.make_cache()
        mx.eval(model(tokens[:, :12], cache=cold_cache))
        cold = model(tokens[:, 12:], cache=cold_cache)
        assert mx.array_equal(warm, cold).item()
        assert not can_trim_prompt_cache(cached)
    finally:
        mx.set_default_device(mx.gpu)
