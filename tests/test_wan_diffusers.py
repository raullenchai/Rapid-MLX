# SPDX-License-Identifier: Apache-2.0
"""Fail-closed mapping tests for the pinned Wan 2.1 Desktop layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace

import numpy as np
import pytest

import vllm_mlx.video.wan_diffusers as wan_diffusers
from vllm_mlx.video.wan import WanBackendError, WanVideoEngine
from vllm_mlx.video.wan_diffusers import (
    _load_sharded,
    _load_t5,
    _load_transformer,
    _load_vae,
    _read_index,
    _scoped_generate_function,
    _t5_key,
    _transformer_key,
    _vae_decoder_key,
    _validate_target_parameters,
    desktop_wan21_config,
    diffusers_runtime,
    generate_with_runtime,
    is_diffusers_wan21_layout,
)

_WAN21_COMPONENT_CONFIGS = {
    "model_index.json": {
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    },
    "transformer/config.json": {
        "_class_name": "WanTransformer3DModel",
        "patch_size": [1, 2, 2],
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "num_layers": 30,
        "ffn_dim": 8960,
        "text_dim": 4096,
    },
    "text_encoder/config.json": {
        "model_type": "umt5",
        "vocab_size": 256384,
        "d_model": 4096,
        "d_ff": 10240,
        "num_heads": 64,
        "num_layers": 24,
        "relative_attention_num_buckets": 32,
    },
    "vae/config.json": {
        "_class_name": "AutoencoderKLWan",
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
    },
}


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
    for relative, payload in _WAN21_COMPONENT_CONFIGS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return root


def test_official_layout_synthesizes_bounded_wan21_config(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    assert is_diffusers_wan21_layout(root)
    engine = WanVideoEngine(root)
    assert engine.config == desktop_wan21_config()
    assert engine.model_type == "t2v"
    assert engine.native_fps == 16
    assert engine.max_area == 704 * 1280


def test_official_layout_rejects_mismatched_architecture(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    config = root / "transformer/config.json"
    payload = json.loads(config.read_text())
    payload["num_layers"] = 40
    config.write_text(json.dumps(payload))

    assert not is_diffusers_wan21_layout(root)


def test_official_layout_rejects_unreadable_component_config(tmp_path: Path) -> None:
    root = _layout(tmp_path)
    (root / "vae/config.json").write_text("{")

    assert not is_diffusers_wan21_layout(root)


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
        (
            "decoder.mid_block.attentions.0.norm.gamma",
            "decoder.middle.1.norm.gamma",
        ),
        (
            "decoder.mid_block.attentions.0.to_qkv.weight",
            "decoder.middle.1.to_qkv.weight",
        ),
        (
            "decoder.mid_block.attentions.0.to_qkv.bias",
            "decoder.middle.1.to_qkv.bias",
        ),
        (
            "decoder.mid_block.attentions.0.proj.weight",
            "decoder.middle.1.proj.weight",
        ),
        (
            "decoder.mid_block.attentions.0.proj.bias",
            "decoder.middle.1.proj.bias",
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
    with pytest.raises(WanBackendError, match="unsupported"):
        _transformer_key("blocks.0.new_component.weight")
    with pytest.raises(WanBackendError, match="unsupported"):
        _t5_key("new_component.weight")
    with pytest.raises(WanBackendError, match="unsupported"):
        _t5_key("encoder.block.0.layer.0.new_component.weight")
    assert _vae_decoder_key("encoder.conv_in.weight") is None
    with pytest.raises(WanBackendError, match="unsupported"):
        _vae_decoder_key("decoder.new_component.weight")

    index.write_text("{")
    with pytest.raises(WanBackendError, match="unreadable"):
        _read_index(tmp_path, index.name, 1, _t5_key)

    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "shared.weight": "one.safetensors",
                    "encoder.final_layer_norm.weight": "one.safetensors",
                }
            }
        )
    )
    with pytest.raises(WanBackendError, match="duplicate"):
        _read_index(tmp_path, index.name, 2, lambda _key: "same.target")


def test_sharded_loader_applies_pinned_patch_reshape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "transformer" / "weights.index.json"
    index.parent.mkdir()
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "patch_embedding.weight": "one.safetensors",
                    "patch_embedding.bias": "one.safetensors",
                }
            }
        )
    )
    (index.parent / "one.safetensors").write_bytes(b"pinned-shard")
    source = {
        "patch_embedding.weight": np.zeros((2, 3, 1, 2, 2), dtype=np.float16),
        "patch_embedding.bias": np.zeros((2,), dtype=np.float16),
    }
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_utils = ModuleType("mlx.utils")
    mlx_core.load = lambda *_args, **_kwargs: source
    mlx_utils.tree_flatten = lambda parameters: list(parameters.items())
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)

    class FakeModel:
        loaded = None

        def parameters(self):
            return {
                "patch_embedding_proj.weight": object(),
                "patch_embedding_proj.bias": object(),
            }

        def load_weights(self, weights, *, strict):
            self.loaded = (weights, strict)

    model = FakeModel()
    _load_sharded(
        model,
        tmp_path,
        "transformer/weights.index.json",
        2,
        _transformer_key,
        dtype=np.float32,
        reshape_patch=True,
    )

    assert model.loaded is not None
    weights, strict = model.loaded
    assert strict is False
    assert [name for name, _ in weights] == [
        "patch_embedding_proj.weight",
        "patch_embedding_proj.bias",
    ]
    assert weights[0][1].shape == (2, 12)
    assert all(value.dtype == np.float32 for _, value in weights)


def test_sharded_loader_rejects_missing_file_and_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "transformer" / "weights.index.json"
    index.parent.mkdir()
    index.write_text(
        json.dumps({"weight_map": {"patch_embedding.bias": "missing.safetensors"}})
    )
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_utils = ModuleType("mlx.utils")
    mlx_core.load = lambda *_args, **_kwargs: {}
    mlx_utils.tree_flatten = lambda parameters: list(parameters.items())
    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)

    class FakeModel:
        def parameters(self):
            return {"patch_embedding_proj.bias": object()}

    with pytest.raises(WanBackendError, match="missing 'missing.safetensors'"):
        _load_sharded(
            FakeModel(),
            index.parent.parent,
            "transformer/weights.index.json",
            1,
            _transformer_key,
        )

    (index.parent / "missing.safetensors").write_bytes(b"shard")
    with pytest.raises(WanBackendError, match="missing tensor 'patch_embedding.bias'"):
        _load_sharded(
            FakeModel(),
            index.parent.parent,
            "transformer/weights.index.json",
            1,
            _transformer_key,
        )


@pytest.mark.parametrize(
    "options", [{"quantization": {"bits": 4}}, {"loras": ["adapter"]}]
)
def test_transformer_loader_rejects_unsupported_overlays(
    tmp_path: Path, options: dict
) -> None:
    with pytest.raises(WanBackendError, match="do not support"):
        _load_transformer(tmp_path, object(), **options)


def test_pinned_runtime_loaders_bind_exact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluated = []
    source = {
        "encoder.ignored": np.zeros((1,), dtype=np.float16),
        "post_quant_conv.weight": np.zeros((2, 3, 1, 2, 2), dtype=np.float16),
        "decoder.mid_block.attentions.0.proj.weight": np.zeros(
            (2, 3, 1, 1), dtype=np.float16
        ),
        "post_quant_conv.bias": np.zeros((2,), dtype=np.float16),
    }
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_utils = ModuleType("mlx.utils")
    mlx_core.float32 = np.float32
    mlx_core.load = lambda *_args, **_kwargs: source
    mlx_core.transpose = np.transpose
    mlx_core.eval = lambda parameters: evaluated.append(parameters)
    mlx_utils.tree_flatten = lambda parameters: list(parameters.items())
    mlx.core = mlx_core

    mlx_video = ModuleType("mlx_video")
    models = ModuleType("mlx_video.models")
    wan = ModuleType("mlx_video.models.wan")
    model_module = ModuleType("mlx_video.models.wan.model")
    text_module = ModuleType("mlx_video.models.wan.text_encoder")
    vae_module = ModuleType("mlx_video.models.wan.vae")

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def parameters(self):
            return {"model": object()}

    class FakeT5:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def parameters(self):
            return {"t5": object()}

    class FakeVAE:
        def __init__(self, *, z_dim):
            self.z_dim = z_dim
            self.loaded = None

        def parameters(self):
            return {
                "conv2.weight": object(),
                "decoder.middle.1.proj.weight": object(),
                "conv2.bias": object(),
                "mean": object(),
                "std": object(),
                "inv_std": object(),
            }

        def load_weights(self, weights, *, strict):
            self.loaded = (weights, strict)

    model_module.WanModel = FakeModel
    text_module.T5Encoder = FakeT5
    vae_module.WanVAE = FakeVAE
    for name, module in {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx.utils": mlx_utils,
        "mlx_video": mlx_video,
        "mlx_video.models": models,
        "mlx_video.models.wan": wan,
        "mlx_video.models.wan.model": model_module,
        "mlx_video.models.wan.text_encoder": text_module,
        "mlx_video.models.wan.vae": vae_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sharded_calls = []
    monkeypatch.setattr(
        wan_diffusers,
        "_load_sharded",
        lambda *args, **kwargs: sharded_calls.append((args, kwargs)),
    )
    config = SimpleNamespace(
        t5_vocab_size=1,
        t5_dim=2,
        t5_dim_attn=3,
        t5_dim_ffn=4,
        t5_num_heads=5,
        t5_num_layers=6,
        t5_num_buckets=7,
    )
    transformer = _load_transformer(tmp_path, config)
    encoder = _load_t5(tmp_path, config)

    assert transformer.config is config
    assert encoder.kwargs == {
        "vocab_size": 1,
        "dim": 2,
        "dim_attn": 3,
        "dim_ffn": 4,
        "num_heads": 5,
        "num_layers": 6,
        "num_buckets": 7,
        "shared_pos": False,
    }
    assert len(sharded_calls) == 2
    assert sharded_calls[0][1] == {
        "reshape_patch": True,
        "ignored_model_parameters": frozenset({"freqs"}),
    }
    assert sharded_calls[1][1] == {"dtype": np.float32}

    monkeypatch.setattr(wan_diffusers, "_EXPECTED_VAE_DECODER_KEYS", 4)
    with pytest.raises(WanBackendError, match="unexpected decoder tensor set"):
        _load_vae(tmp_path)

    monkeypatch.setattr(wan_diffusers, "_EXPECTED_VAE_DECODER_KEYS", 3)
    vae = _load_vae(tmp_path)
    assert vae.z_dim == 16
    assert vae.loaded is not None
    weights, strict = vae.loaded
    assert strict is False
    assert [value.shape for _, value in weights] == [
        (2, 1, 2, 2, 3),
        (2, 1, 1, 3),
        (2,),
    ]
    assert len(evaluated) == 3


def test_runtime_patch_is_scoped_and_tokenizer_is_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _layout(tmp_path)
    generator = ModuleType("mlx_video.generate_wan")
    originals = (object(), object(), object())
    (
        generator.load_wan_model,
        generator.load_t5_encoder,
        generator.load_vae_decoder,
    ) = originals
    tokenizer_calls = []

    class OriginalTokenizer:
        @classmethod
        def from_pretrained(cls, model, *args, **kwargs):
            tokenizer_calls.append((model, args, kwargs))
            return "tokenizer"

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = OriginalTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def generate_template():
        import json

        from transformers import AutoTokenizer

        assert json.loads("null") is None
        return AutoTokenizer.from_pretrained("ignored")

    generator.generate_video = FunctionType(
        generate_template.__code__, generator.__dict__, "generate_video"
    )

    with diffusers_runtime(root, generator) as (view, scoped_generate):
        assert json.loads((view / "config.json").read_text()) == desktop_wan21_config()
        assert scoped_generate.__globals__["load_wan_model"] is not originals[0]
        assert scoped_generate() == "tokenizer"
        assert tokenizer_calls == [(root / "tokenizer", (), {"local_files_only": True})]

    assert (
        generator.load_wan_model,
        generator.load_t5_encoder,
        generator.load_vae_decoder,
    ) == originals
    assert transformers.AutoTokenizer is OriginalTokenizer
    assert not view.exists()


def test_scoped_runtime_rejects_incomplete_layout(tmp_path: Path) -> None:
    with pytest.raises(WanBackendError, match="layout is incomplete"):
        _scoped_generate_function(tmp_path, ModuleType("mlx_video.generate_wan"))


def test_generate_runtime_preserves_preconverted_fallback(tmp_path: Path) -> None:
    calls = []
    generator = SimpleNamespace(generate_video=lambda **kwargs: calls.append(kwargs))

    generate_with_runtime(tmp_path, generator, {"model_dir": "converted"})

    assert calls == [{"model_dir": "converted"}]


def test_generate_runtime_routes_all_pinned_loader_seams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _layout(tmp_path)
    generator = ModuleType("mlx_video.generate_wan")
    calls = []

    def generate_template(model_dir, marker):
        config = object()
        calls.append(("model", globals()["load_wan_model"](Path(model_dir), config)))
        calls.append(("t5", globals()["load_t5_encoder"](Path(model_dir), config)))
        calls.append(("vae", globals()["load_vae_decoder"](Path(model_dir), config)))
        calls.append(("marker", marker))

    generator.generate_video = FunctionType(
        generate_template.__code__,
        generate_template.__globals__,
        "generate_video",
        generate_template.__defaults__,
        generate_template.__closure__,
    )
    monkeypatch.setattr(
        wan_diffusers, "_load_transformer", lambda path, *_args: ("model", path)
    )
    monkeypatch.setattr(wan_diffusers, "_load_t5", lambda path, *_args: ("t5", path))
    monkeypatch.setattr(wan_diffusers, "_load_vae", lambda path, *_args: ("vae", path))

    generation_kwargs = {"model_dir": "ignored", "marker": "request"}
    generate_with_runtime(root, generator, generation_kwargs)

    assert [kind for kind, _ in calls] == ["model", "t5", "vae", "marker"]
    assert calls[-1] == ("marker", "request")
    assert all(value[1] == root for _, value in calls[:3])
    assert generation_kwargs["model_dir"] != "ignored"
    assert not Path(generation_kwargs["model_dir"]).exists()


def test_loader_target_validation_rejects_missing_and_unknown_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlx = ModuleType("mlx")
    mlx_utils = ModuleType("mlx.utils")

    def tree_flatten(parameters, prefix=""):
        flattened = []
        for name, value in parameters.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                flattened.extend(tree_flatten(value, path))
            else:
                flattened.append((path, value))
        return flattened

    mlx_utils.tree_flatten = tree_flatten
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)

    class FakeModel:
        def parameters(self):
            return {"layer": {"weight": object(), "bias": object()}}

    _validate_target_parameters(FakeModel(), {"layer.weight", "layer.bias"})
    with pytest.raises(WanBackendError, match="missing=.*layer.bias"):
        _validate_target_parameters(FakeModel(), {"layer.weight"})
    with pytest.raises(WanBackendError, match="unexpected=.*other.weight"):
        _validate_target_parameters(
            FakeModel(), {"layer.weight", "layer.bias", "other.weight"}
        )
