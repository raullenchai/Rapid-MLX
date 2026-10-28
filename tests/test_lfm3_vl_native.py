import builtins
import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from PIL import Image

from rapid_mlx.models.lfm3_vl import Model, ModelArgs, grid_positions
from rapid_mlx.models.lfm3_vl_processor import (
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
    LFM3VLProcessor,
    preprocess_images,
    smart_resize,
)


def tiny_args():
    return ModelArgs(
        vision_config=dict(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            patch_size=2,
            layer_types=["conv", "full_attention"],
        ),
        text_config=dict(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            layer_types=["conv"],
        ),
        projector_hidden_size=16,
        image_token_id=30,
        video_token_id=29,
    )


@pytest.fixture
def model():
    return Model(tiny_args())


def test_vision_shapes_and_image_isolation(model):
    mx.random.seed(7)
    first = mx.random.normal((4, 24))
    second = mx.random.normal((8, 24))
    together = model.vision_tower(
        mx.concatenate([first, second]), [[1, 2, 2], [1, 2, 4]]
    )
    separate = mx.concatenate(
        [
            model.vision_tower(first, [[1, 2, 2]]),
            model.vision_tower(second, [[1, 2, 4]]),
        ]
    )
    np.testing.assert_allclose(np.asarray(together), np.asarray(separate), atol=1e-6)
    assert model.encode_images(
        mx.concatenate([first, second]), [[1, 2, 2], [1, 2, 4]]
    ).shape == (3, 16)


def test_rope_coordinates():
    assert np.asarray(grid_positions(2, 4, 2)).tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ]


def test_projector_groups_consecutive_tokens(model):
    projector = model.multi_modal_projector
    x = mx.random.normal((8, 16))
    expected = mx.concatenate([projector(x[:4]), projector(x[4:])])
    np.testing.assert_allclose(
        np.asarray(projector(x)), np.asarray(expected), atol=1e-6
    )


def test_splice_order_and_count(model):
    patches = mx.random.normal((12, 24))
    grid = [[1, 2, 2], [1, 2, 4]]
    ids = mx.array([[1, 30, 2, 30, 30, 3]])
    embeddings = model.get_input_embeddings(ids, patches, grid)
    features = model.encode_images(patches, grid)
    np.testing.assert_allclose(
        np.asarray(embeddings[0, [1, 3, 4]]), np.asarray(features), atol=1e-6
    )
    with pytest.raises(ValueError, match="spans"):
        model.get_input_embeddings(mx.array([[1, 30, 30, 2, 30, 3]]), patches, grid)
    with pytest.raises(ValueError, match="spans"):
        model.get_input_embeddings(mx.array([[1, 30, 2]]), patches, grid)
    with pytest.raises(ValueError, match="Video"):
        model.get_input_embeddings(mx.array([[29]]))


def test_processor_limits_and_video():
    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            return prompt

    proc = LFM3VLProcessor(Tokenizer())
    img = Image.new("RGB", (32, 32), "white")
    result = proc.prepare_inputs(
        [1, IMAGE_TOKEN_ID, 2], [img], min_pixels=1024, max_pixels=1024
    )
    assert result["input_ids"].tolist() == [[1, IMAGE_TOKEN_ID, 2]]
    assert result["pixel_values"].shape == (4, 1536)
    assert result["image_grid_thw"].tolist() == [[1, 2, 2]]
    with pytest.raises(ValueError, match="Video"):
        proc.prepare_inputs([VIDEO_TOKEN_ID])
    with pytest.raises(ValueError):
        smart_resize(0, 32)
    with pytest.raises(ValueError):
        preprocess_images([Image.new("RGB", (1, 201))])


def test_native_load_without_mlx_vlm(tmp_path, monkeypatch, model):
    import mlx_lm.utils

    from rapid_mlx.models.mllm import MLXMultimodalLM

    args = tiny_args()
    from dataclasses import asdict

    (tmp_path / "config.json").write_text(json.dumps(asdict(args)))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: object())
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise AssertionError("mlx_vlm imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    wrapper = MLXMultimodalLM(str(tmp_path), enable_cache=False)
    wrapper.load()
    assert wrapper.model.model_type == "lfm3_vl"


def test_strict_sanitizer_rejects_invalid_weights(model):
    weights = dict(tree_flatten(model.parameters()))
    assert set(model.sanitize(weights)) == set(weights)
    missing = dict(weights)
    missing.pop("vision_tower.patchifier.weight")
    with pytest.raises(ValueError, match="missing"):
        model.sanitize(missing)
    with pytest.raises(ValueError, match="Unexpected"):
        model.sanitize({**weights, "invalid.weight": mx.ones((1,))})
    wrong = dict(weights)
    wrong["vision_tower.patchifier.weight"] = mx.ones((1, 1))
    with pytest.raises(ValueError, match="shape"):
        model.sanitize(wrong)


def test_forward_output_shape(model):
    logits = model(
        mx.array([[1, 30, 2]]),
        pixel_values=mx.ones((4, 24)),
        image_grid_thw=[[1, 2, 2]],
    )
    assert logits.shape == (1, 3, 32)
    mx.eval(logits)


def test_native_lane_uses_a_registered_reason(tmp_path):
    from rapid_mlx.api.utils import resolve_serving_lane_decision
    from rapid_mlx.byom.preflight import RAPID_MODEL_TYPES

    (tmp_path / "config.json").write_text('{"model_type":"lfm3_vl"}')
    decision = resolve_serving_lane_decision(str(tmp_path))
    assert decision.is_mllm
    assert decision.reason == "vision_supported"
    assert "lfm3_vl" in RAPID_MODEL_TYPES


def test_native_cache_probes_do_not_import_optional_runtime(monkeypatch):
    import sys

    from mlx_lm.models.cache import ArraysCache, KVCache

    from rapid_mlx import mllm_batch_generator as generator

    monkeypatch.delitem(sys.modules, "mlx_vlm.models.cache", raising=False)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise AssertionError("optional vision runtime imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert generator._singleton_regular_cache_leaves([ArraysCache(2), KVCache()], True)
    generator._batched_leaf_types()


def test_image_prefill_cached_decode_matches_full_forward(model):
    mx.random.seed(11)
    tokens = mx.array([[1, 30, 2, 3, 4, 5]])
    pixels = mx.random.normal((4, 24))
    grid = [[1, 2, 2]]
    full = model(tokens, pixel_values=pixels, image_grid_thw=grid)
    cache = model.make_cache()
    prefill = model(tokens[:, :3], pixel_values=pixels, image_grid_thw=grid, cache=cache)
    mx.eval(prefill)
    chunks = [prefill]
    for i in range(3, tokens.shape[1]):
        chunk = model(tokens[:, i : i + 1], cache=cache)
        mx.eval(chunk)
        chunks.append(chunk)
    np.testing.assert_allclose(
        np.asarray(mx.concatenate(chunks, axis=1)), np.asarray(full), rtol=1e-4, atol=1e-5
    )
