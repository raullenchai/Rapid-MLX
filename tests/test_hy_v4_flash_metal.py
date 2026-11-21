"""Native MLX checks; CPU graph tests do not substitute for these."""

import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.models.hy_v4_flash import Model
from tests.test_hy_v4_flash import tiny


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_native_forward_chunked_decode_and_trim(dtype):
    if not mx.metal.is_available():
        pytest.skip("Metal is unavailable")
    mx.random.seed(17)
    model = Model(tiny())
    from mlx.utils import tree_flatten

    weights = {
        k: v.astype(dtype)
        if model.cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating)
        else v
        for k, v in tree_flatten(model.parameters())
    }
    model.load_weights(list(model.sanitize(weights).items()))
    tolerance = 3e-5 if dtype == mx.float32 else 2e-2
    tokens = mx.array([[1, 2, 3, 4, 5]])
    whole = model(tokens)
    for split in range(1, tokens.shape[1]):
        cache = model.make_cache()
        first = model(tokens[:, :split], cache)
        second = model(tokens[:, split:], cache)
        mx.eval(whole, first, second)
        assert mx.allclose(
            mx.concatenate((first, second), axis=1), whole, atol=tolerance
        ).item()
    for layer in cache:
        layer.trim(2)
    replay = model(tokens[:, -2:], cache)
    mx.eval(replay)
    assert mx.allclose(replay, whole[:, -2:], atol=tolerance).item()


def test_native_affine_checkpoint_roundtrip(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import asdict

    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model, quantize_model

    from rapid_mlx.models import hy_v4_flash

    args = tiny(
        hidden_size=64,
        vocab_size=128,
        intermediate_size=64,
        q_lora_rank=64,
        kv_lora_rank=64,
        routed_expert_hidden_size=64,
        moe_intermediate_size=64,
        engram_n_embed_per_ngram=192,
    )
    model, config = quantize_model(Model(args), asdict(args), 32, 4)
    assert model.layers[1].engram_layer.multi_head_embedding.embedding.group_size == 32
    assert model.layers[0].self_attn.q_a_proj.group_size == 64
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setitem(sys.modules, "mlx_lm.models.hy_v4_flash", hy_v4_flash)
    loaded, _ = load_model(tmp_path, lazy=True)
    tokens = mx.array([[1, 2, 3]])
    expected, actual = model(tokens), loaded(tokens)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual, atol=1e-5).item()


def test_native_cache_merge_preserves_independent_request_histories():
    from mlx_lm.models.cache import CacheList

    model = Model(tiny())
    first_cache, second_cache = model.make_cache(), model.make_cache()
    model(mx.array([[1, 2, 3, 4]]), first_cache)
    model(mx.array([[3, 4]]), second_cache)
    merged = [
        CacheList.merge([first, second])
        for first, second in zip(first_cache, second_cache)
    ]
    batched = model(mx.array([[5], [5]]), merged)
    first = model(mx.array([[5]]), first_cache)
    second = model(mx.array([[5]]), second_cache)
    mx.eval(batched, first, second)
    assert mx.allclose(
        batched, mx.concatenate((first, second), axis=0), atol=3e-5
    ).item()
