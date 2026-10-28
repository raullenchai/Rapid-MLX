import builtins
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from rapid_mlx.models.lfm3_vl import Model, ModelArgs, grid_positions  # noqa: E402
from rapid_mlx.models.lfm3_vl_processor import (  # noqa: E402
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
    prefill = model(
        tokens[:, :3], pixel_values=pixels, image_grid_thw=grid, cache=cache
    )
    mx.eval(prefill)
    chunks = [prefill]
    for i in range(3, tokens.shape[1]):
        chunk = model(tokens[:, i : i + 1], cache=cache)
        mx.eval(chunk)
        chunks.append(chunk)
    np.testing.assert_allclose(
        np.asarray(mx.concatenate(chunks, axis=1)),
        np.asarray(full),
        rtol=1e-4,
        atol=1e-5,
    )


def test_invalid_configs_and_embeddings(model, monkeypatch):
    from rapid_mlx.models.lfm3_vl import VisionConfig

    for kwargs, message in [
        ({}, "layer types"),
        ({"layer_types": ["conv"] * 14, "hidden_size": 17}, "dimension"),
        (
            {"layer_types": ["conv"] * 14, "rope_parameters": {"rope_type": "other"}},
            "axial",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            VisionConfig(**kwargs)
    args = tiny_args()
    with pytest.raises(ValueError, match="projector"):
        ModelArgs(
            text_config=args.text_config,
            vision_config=args.vision_config,
            downsample_factor=3,
        )
    pixels = mx.ones((4, 24))
    ids = mx.array([[30]])
    for px, grid, message in [
        (None, None, "pixels"),
        (pixels, None, "required"),
        (pixels, [[1, 2]], "ordered"),
        (pixels, [[1, 3, 2]], "Invalid image grid"),
    ]:
        with pytest.raises(ValueError, match=message):
            model.get_input_embeddings(ids, px, grid)
    with pytest.raises(ValueError, match="patch count"):
        model.vision_tower(mx.ones((5, 24)), [[1, 2, 2]])
    monkeypatch.setattr(model, "encode_images", lambda *a: mx.ones((2, 16)))
    with pytest.raises(ValueError, match="count"):
        model.get_input_embeddings(ids, pixels, [[1, 2, 2]])
    assert model.layers is model.language_model.layers


def test_sanitize_quantization_and_dtype(model):
    weights = dict(tree_flatten(model.parameters()))
    conv = next(k for k in weights if k.endswith("spatial_conv.weight"))
    transposed = dict(weights, **{conv: weights[conv].transpose(0, 3, 1, 2)})
    assert model.sanitize(transposed)[conv].shape == weights[conv].shape
    key = next(
        k
        for k, v in weights.items()
        if k.startswith("language_model.") and k.endswith(".weight") and v.ndim == 2
    )
    prefix = key.removesuffix(".weight")

    def quant(weight, scales, biases=None):
        w = dict(weights, **{key: weight, prefix + ".scales": scales})
        if biases is not None:
            w[prefix + ".biases"] = biases
        return w

    with pytest.raises(ValueError, match="Incomplete"):
        model.sanitize(quant(weights[key], mx.ones((1, 1))))
    with pytest.raises(ValueError, match="Invalid quantized weight"):
        model.sanitize(quant(weights[key], mx.ones((1, 1)), mx.ones((1, 1))))
    out, inp = weights[key].shape
    packed = mx.zeros((out, inp // 4), dtype=mx.uint32)
    with pytest.raises(ValueError, match="metadata dtype"):
        model.sanitize(
            quant(
                packed,
                mx.zeros((out, inp // 64), dtype=mx.uint32),
                mx.ones((out, inp // 64)),
            )
        )
    with pytest.raises(ValueError, match="quantized shape"):
        model.sanitize(quant(packed, mx.ones((1, 1)), mx.ones((1, 1))))
    assert (
        model.sanitize(
            quant(packed, mx.ones((out, inp // 64)), mx.ones((out, inp // 64)))
        )[key].dtype
        == mx.uint32
    )
    with pytest.raises(ValueError, match="Invalid dtype"):
        model.sanitize(
            dict(weights, **{key: mx.zeros(weights[key].shape, dtype=mx.int32)})
        )


def test_native_dispatch_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx import model_aliases, model_downloads, model_metadata
    from rapid_mlx.models import media_backend, mllm
    from rapid_mlx.utils import tokenizer

    monkeypatch.setattr(model_aliases, "resolve_hub_model", lambda name: "owner/model")
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: None)
    monkeypatch.setattr(model_aliases, "resolve_subfolder", lambda name: None)
    monkeypatch.setattr(
        model_metadata,
        "read_model_metadata",
        lambda name: (
            SimpleNamespace(config={"model_type": "lfm3_vl"})
            if name == "owner/model"
            else None
        ),
    )
    assert media_backend.native_media_model_type("alias") == "lfm3_vl"
    mllm._require_mlx_vlm("alias")
    calls = []
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda name, **kw: calls.append((name, kw)) or str(tmp_path),
    )
    assert media_backend.resolve_native_checkpoint("alias") == tmp_path
    assert "*.safetensors" in calls[0][1]["allow_patterns"]
    monkeypatch.setattr(model_aliases, "resolve_subfolder", lambda name: "sub")
    assert media_backend.resolve_native_checkpoint("alias") == tmp_path / "sub"
    assert "sub/*.json" in calls[-1][1]["allow_patterns"]
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: "revision")
    monkeypatch.setattr(
        tokenizer, "_resolve_subfolder_checkpoint", lambda name: str(tmp_path / "rev")
    )
    assert media_backend.resolve_native_checkpoint("alias") == tmp_path / "rev"
    with pytest.raises(ValueError, match="Unsupported"):
        media_backend.load_native_media(tmp_path, {})


def test_native_processor_and_batch_integration(tmp_path, monkeypatch, model):
    from rapid_mlx.mllm_batch_generator import (
        ClientRequestError,
        MLLMBatchGenerator,
        MLLMBatchRequest,
        _supports_dynamic_vision_bounds,
    )
    from rapid_mlx.models import mllm

    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            return [1, IMAGE_TOKEN_ID, 2]

        def apply_chat_template(self, *args, **kwargs):
            return "rendered"

    proc = LFM3VLProcessor(Tokenizer())
    assert proc.apply_chat_template([]) == "rendered"
    assert _supports_dynamic_vision_bounds(proc)
    assert (
        proc.prepare_inputs(
            "x", [Image.new("RGB", (32, 32))], min_pixels=1024, max_pixels=None
        )["pixel_values"]
        is not None
    )
    gen = MLLMBatchGenerator(
        model,
        proc,
        enable_prefix_cache=True,
        enable_vision_cache=True,
        media_prefix_cache="auto",
    )
    request = MLLMBatchRequest(
        uid=0, request_id="r", prompt="x", images=[], videos=["video"], max_tokens=8
    )
    with pytest.raises(ClientRequestError, match="Video"):
        gen._preprocess_request(request)
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(path)
    monkeypatch.setattr(mllm, "process_image_input", lambda img: str(path))
    request.videos = []
    request.images = ["image"]
    gen._preprocess_request(request)
    assert request.pixel_values is not None
    assert request.image_grid_thw is not None
    path.write_bytes(b"not an image")
    with pytest.raises(ClientRequestError, match="could not be decoded"):
        gen._preprocess_request(request)
