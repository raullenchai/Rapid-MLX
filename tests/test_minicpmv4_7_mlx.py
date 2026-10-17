# SPDX-License-Identifier: Apache-2.0
"""Pinned-loader, config, processor and synthetic forward tests; no weights."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_mlx
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from rapid_mlx.models.minicpmv4_7 import install_minicpmv4_7
from rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7 import (
    Model,
    ModelConfig,
    TextConfig,
)

FIXTURES = Path(__file__).parent / "fixtures/minicpmv4_7"


@pytest.mark.parametrize(
    "size,hidden,layers,heads", [("1B", 1024, 24, 8), ("35B-A3B", 2048, 40, 16)]
)
def test_real_configs(size, hidden, layers, heads):
    from mlx_vlm.utils import update_module_configs

    install_minicpmv4_7()
    from mlx_vlm.models import minicpmv4_7

    raw = json.loads((FIXTURES / f"{size}-config.json").read_text())
    config = ModelConfig.from_dict(raw)
    # The pinned loader reparses nested configs through module exports.
    config = update_module_configs(config, minicpmv4_7, raw, ["text", "vision"])
    assert config.text_config.hidden_size == hidden
    assert config.text_config.num_hidden_layers == layers
    assert config.text_config.num_attention_heads == heads
    assert config.text_config.rope_parameters["mrope_interleaved"]
    assert config.text_config.rope_parameters["mrope_section"] == [11, 11, 10]
    assert config.vision_config.hidden_size == 1152
    assert config.vision_config.image_size == 980
    assert config.insert_layer_id == 6
    assert config.downsample_mode == "16x"
    assert config.mrope_mode == "canvas"
    assert config.image_token_id == 248056
    if size == "35B-A3B":
        from mlx_vlm.models.qwen3_5_moe.config import TextConfig as MoETextConfig

        assert isinstance(config.text_config, MoETextConfig)
        assert config.text_config.num_experts == 256
        assert config.text_config.num_experts_per_tok == 8
    else:
        assert isinstance(config.text_config, TextConfig)
        assert config.text_config.intermediate_size == 3584


def test_loader_binding_is_idempotent():
    from mlx_vlm.utils import get_model_and_args

    install_minicpmv4_7()
    arch, kind = get_model_and_args({"model_type": "minicpmv4_7"})
    install_minicpmv4_7()
    again, _ = get_model_and_args({"model_type": "minicpmv4_7"})
    assert arch is again
    assert arch.Model is Model
    assert kind == "minicpmv4_7"
    assert arch.MiniCPMV4_7Processor.model_type == kind


def tiny_config(size):
    raw = json.loads((FIXTURES / f"{size}-config.json").read_text())
    raw.update(
        image_token_id=20,
        video_token_id=21,
        image_start_id=22,
        image_end_id=23,
        slice_start_id=24,
        slice_end_id=25,
        newline_id=26,
        patch_size=2,
        insert_layer_id=0,
    )
    raw["text_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 1, 2],
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    )
    raw["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=8,
        patch_size=2,
    )
    return ModelConfig.from_dict(raw)


@pytest.mark.parametrize("size", ["1B", "35B-A3B"])
@pytest.mark.parametrize("mode,count", [("16x", 1), ("4x", 4)])
def test_synthetic_image_cached_features_and_decode(size, mode, count):
    model = Model(tiny_config(size))
    ids = mx.array([[1, 22, *([20] * count), 23, 2]])
    pixels = [[mx.ones((3, 8, 8), dtype=mx.float32)]]
    grids = [mx.array([[4, 4]])]
    kwargs = {"tgt_sizes": grids, "downsample_mode": mode}
    features = model.get_vision_embedding(pixels, grids, downsample_mode=mode)
    assert features[0].shape == (count, 16)
    fresh = model.get_input_embeddings(ids, pixels, **kwargs)
    cached = model.get_input_embeddings(ids, cached_image_features=features, **kwargs)
    assert mx.allclose(fresh.inputs_embeds, cached.inputs_embeds).item()
    assert mx.array_equal(fresh.position_ids, cached.position_ids).item()
    cache = model.language_model.make_cache()
    prefill = model(ids, pixels, cache=cache, **kwargs)
    decode = model(mx.array([[3]]), cache=cache)
    mx.eval(prefill.logits, decode.logits)
    assert decode.logits.shape == (1, 1, 64)
    assert mx.all(mx.isfinite(decode.logits)).item()
    with pytest.raises(ValueError, match="crop grids"):
        model.get_input_embeddings(ids, pixels)


def test_quantized_text_embeddings_do_not_select_vision_dtype():
    import mlx.nn as nn

    config = tiny_config("1B")
    config.text_config.hidden_size = 64
    model = Model(config)
    model.language_model.model.embed_tokens = nn.QuantizedEmbedding.from_embedding(
        model.language_model.model.embed_tokens, group_size=32, bits=8
    )
    features = model.get_vision_embedding([[mx.ones((3, 8, 8))]], [mx.array([[4, 4]])])
    assert model.language_model.model.embed_tokens.weight.dtype == mx.uint32
    assert (
        features[0].dtype == model.vision_tower.embeddings.patch_embedding.weight.dtype
    )
    assert mx.all(mx.isfinite(features[0])).item()


def test_processor_loads_locally_with_checkpoint_template(tmp_path):
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_processor
    from tokenizers import Tokenizer, models
    from transformers import AutoProcessor, Qwen2TokenizerFast

    from rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7 import (
        MiniCPMV4_7Processor,
    )

    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"<unk>": 0, "<|image_pad|>": 1, "<|video_pad|>": 2, "hello": 3},
                unk_token="<unk>",
            )
        )
    )
    tokenizer.save_pretrained(tmp_path)
    shutil.copy(FIXTURES / "1B-config.json", tmp_path / "config.json")
    shutil.copy(FIXTURES / "1B-chat_template.jinja", tmp_path / "chat_template.jinja")
    install_minicpmv4_7()
    processor = AutoProcessor.from_pretrained(
        tmp_path, local_files_only=True, trust_remote_code=False
    )
    assert isinstance(processor, MiniCPMV4_7Processor)
    # ProcessorMixin deep-copies instance state for serialization and repr.
    assert processor.to_dict()["processor_class"] == "MiniCPMV4_7Processor"
    assert "MiniCPMV4_7Processor" in repr(processor)
    assert (
        processor.tokenizer.chat_template
        == (tmp_path / "chat_template.jinja").read_text()
    )
    # Direct rendering leaves enable_thinking undefined in the checkpoint template.
    assert processor.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    ).endswith("<|im_start|>assistant\n<think>\n")
    output = processor(text="hello")
    assert output["downsample_mode"] == "16x"
    assert not np.any(output["mm_token_type_ids"])
    loaded = load_processor(tmp_path, trust_remote_code=False)
    prompt = apply_chat_template(
        loaded, {"model_type": "minicpmv4_7"}, "Describe", num_images=1
    )
    assert "<|image_pad|>" in prompt
    # Pinned mlx_vlm defaults enable_thinking to False for supported templates.
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # Native serving hands ordered media parts directly to the processor.
    prompt = loaded.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "Then"},
                    {"type": "image"},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|video_pad|>\nThen\n<|image_pad|>" in prompt


@pytest.mark.parametrize("mode,divisor", [("16x", 16), ("4x", 4)])
def test_synthetic_video_preprocessing(mode, divisor):
    from mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6 import (
        MiniCPMVImageProcessor,
        MiniCPMVVideoProcessor,
    )
    from tokenizers import Tokenizer, models
    from transformers import Qwen2TokenizerFast

    from rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7 import (
        MiniCPMV4_7Processor,
    )

    tokens = [
        "<unk>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<image>",
        "</image>",
        "<slice>",
        "</slice>",
        "<image_id>",
        "</image_id>",
    ]
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {token: index for index, token in enumerate(tokens)}, unk_token="<unk>"
            )
        )
    )
    options = dict(
        patch_size=2, scale_resolution=8, slice_mode=False, downsample_mode="16x"
    )
    processor = MiniCPMV4_7Processor(
        image_processor=MiniCPMVImageProcessor(**options),
        video_processor=MiniCPMVVideoProcessor(**options),
        tokenizer=tokenizer,
    )
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (0, 127, 255)]
    result = processor(text="<|video_pad|>", videos=frames, downsample_mode=mode)
    grids = result["tgt_sizes"][0]
    assert grids.tolist() == [[4, 4]] * len(frames)
    assert result["num_frames_per_video"] == [[len(frames)]]
    assert result["num_patches_per_frame"] == [[1] * len(frames)]
    assert result["grids_videos"] == [[(0, 0)] * len(frames)]
    assert len(result["pixel_values"][0]) == len(grids) == len(frames)
    expected_counts = np.prod(grids, axis=1) // divisor
    assert (
        np.count_nonzero(result["input_ids"] == processor.video_token_id)
        == expected_counts.sum()
    )
    bounds = result["image_bound"][0]
    assert (bounds[:, 1] - bounds[:, 0]).tolist() == expected_counts.tolist()
    assert np.count_nonzero(result["mm_token_type_ids"] == 2) == expected_counts.sum()
    assert not np.any(result["mm_token_type_ids"] == 1)
    assert (
        processor.image_processor.downsample_mode
        == processor.video_processor.downsample_mode
        == "16x"
    )
    assert processor.to_dict()["processor_class"] == "MiniCPMV4_7Processor"
    assert "MiniCPMV4_7Processor" in repr(processor)


@pytest.mark.parametrize("size", ["1B", "35B-A3B"])
def test_pinned_loader_roundtrip_and_mtp_filter(size, tmp_path):
    from dataclasses import asdict

    from mlx.utils import tree_flatten
    from mlx_vlm.utils import load_model

    install_minicpmv4_7()
    model = Model(tiny_config(size))
    expected = dict(tree_flatten(model.parameters()))
    sanitized = model.sanitize({**expected, "model.mtp.unused.weight": mx.zeros((1,))})
    assert sanitized.keys() == expected.keys()
    for key in expected:
        assert mx.array_equal(sanitized[key], expected[key]).item(), key
    (tmp_path / "config.json").write_text(json.dumps(asdict(model.config)))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), expected)
    loaded = load_model(tmp_path)
    assert isinstance(loaded, Model)
    for key, value in tree_flatten(loaded.parameters()):
        assert mx.array_equal(value, expected[key]).item(), key
