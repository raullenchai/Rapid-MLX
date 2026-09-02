# SPDX-License-Identifier: Apache-2.0
"""Fail-closed mapping tests for the pinned Wan 2.1 Desktop layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from vllm_mlx.video.wan import WanBackendError, WanVideoEngine
from vllm_mlx.video.wan_diffusers import (
    _read_index,
    _t5_key,
    _transformer_key,
    _vae_decoder_key,
    desktop_wan21_config,
    is_diffusers_wan21_layout,
    patched_diffusers_runtime,
)


def _layout(root: Path) -> Path:
    for relative in (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "text_encoder/model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    (root / "tokenizer").mkdir()
    return root


def test_official_layout_synthesizes_bounded_wan21_config(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    assert is_diffusers_wan21_layout(root)
    engine = WanVideoEngine(root)
    assert engine.config == desktop_wan21_config()
    assert engine.model_type == "t2v"
    assert engine.native_fps == 16
    assert engine.max_area == 704 * 1280


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("patch_embedding.weight", "patch_embedding_proj.weight"),
        ("blocks.7.attn1.to_out.0.weight", "blocks.7.self_attn.o.weight"),
        ("blocks.7.attn2.to_q.bias", "blocks.7.cross_attn.q.bias"),
        ("blocks.7.norm2.weight", "blocks.7.norm3.weight"),
        ("blocks.7.ffn.net.0.proj.weight", "blocks.7.ffn.fc1.weight"),
        ("blocks.7.scale_shift_table", "blocks.7.modulation"),
    ],
)
def test_transformer_mapping(source: str, target: str) -> None:
    assert _transformer_key(source) == target


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("shared.weight", "token_embedding.weight"),
        ("encoder.final_layer_norm.weight", "norm.weight"),
        ("encoder.block.3.layer.0.SelfAttention.q.weight", "blocks.3.attn.q.weight"),
        (
            "encoder.block.3.layer.1.DenseReluDense.wi_0.weight",
            "blocks.3.ffn.gate_proj.weight",
        ),
    ],
)
def test_t5_mapping(source: str, target: str) -> None:
    assert _t5_key(source) == target


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("post_quant_conv.weight", "conv2.weight"),
        ("decoder.mid_block.resnets.1.conv2.bias", "decoder.middle.2.residual.6.bias"),
        (
            "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
            "decoder.upsamples.4.shortcut.weight",
        ),
        (
            "decoder.up_blocks.2.upsamplers.0.time_conv.weight",
            "decoder.upsamples.11.time_conv.weight",
        ),
    ],
)
def test_vae_decoder_mapping(source: str, target: str) -> None:
    assert _vae_decoder_key(source) == target


def test_indexes_fail_closed_on_drift_duplicates_and_unsafe_shards(
    tmp_path: Path,
) -> None:
    index = tmp_path / "weights.index.json"
    index.write_text(
        json.dumps({"weight_map": {"shared.weight": "../escape.safetensors"}})
    )
    with pytest.raises(WanBackendError, match="unsafe"):
        _read_index(tmp_path, index.name, 1, _t5_key)

    index.write_text(json.dumps({"weight_map": {"shared.weight": "one.safetensors"}}))
    with pytest.raises(WanBackendError, match="unexpected tensor set"):
        _read_index(tmp_path, index.name, 2, _t5_key)

    with pytest.raises(WanBackendError, match="unsupported"):
        _transformer_key("new_component.weight")


def test_runtime_patch_is_scoped_and_tokenizer_is_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _layout(tmp_path)
    parent = ModuleType("mlx_video")
    parent.__path__ = []
    generator = ModuleType("mlx_video.generate_wan")
    originals = (object(), object(), object())
    (
        generator.load_wan_model,
        generator.load_t5_encoder,
        generator.load_vae_decoder,
    ) = originals
    parent.generate_wan = generator
    tokenizer_calls = []

    class OriginalTokenizer:
        @classmethod
        def from_pretrained(cls, model, *args, **kwargs):
            tokenizer_calls.append((model, args, kwargs))
            return "tokenizer"

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = OriginalTokenizer
    monkeypatch.setitem(sys.modules, "mlx_video", parent)
    monkeypatch.setitem(sys.modules, "mlx_video.generate_wan", generator)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with patched_diffusers_runtime(root) as view:
        assert json.loads((view / "config.json").read_text()) == desktop_wan21_config()
        assert generator.load_wan_model is not originals[0]
        assert transformers.AutoTokenizer.from_pretrained("ignored") == "tokenizer"
        assert tokenizer_calls == [(root / "tokenizer", (), {"local_files_only": True})]

    assert (
        generator.load_wan_model,
        generator.load_t5_encoder,
        generator.load_vae_decoder,
    ) == originals
    assert transformers.AutoTokenizer is OriginalTokenizer
    assert not view.exists()
