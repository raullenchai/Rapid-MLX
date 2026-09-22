# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 catalog, routing and repeated-request contracts."""

from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from rapid_mlx import _download_gate
from rapid_mlx.catalog import build_catalog_bundle
from rapid_mlx.image.engine import (
    ImageGenerationEngine,
    ImageRuntimeError,
    _detect_family,
    default_steps_for_model,
)
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.model_sizes import size_bytes
from rapid_mlx.runtime.resident_models import estimate_model_bytes

REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


@pytest.mark.parametrize(
    "name",
    [
        REPO,
        "qwen-image-2.1",
        "acme/qwen_image_2_1",
        "acme/qwen-image-21",
        "acme/qwen-image-v2.1",
        "z-image-lab/Qwen-Image-2.1",
    ],
)
def test_family_routes_before_broad_matches(name):
    assert _detect_family(name) == "qwen-image-2.1"
    assert default_steps_for_model(name) == 40


@pytest.mark.parametrize(
    "name",
    [
        "Qwen/Qwen-Image-2.2",
        "Qwen/Qwen-Image-Edit-2.1",
        "Qwen/Qwen-Image-2.1-Edit",
        "acme/qwen-image-v2.10",
    ],
)
def test_unsupported_2x_does_not_fall_into_1x(name):
    with pytest.raises(ImageRuntimeError, match="Only Qwen-Image 2.1"):
        ImageGenerationEngine(name)


def test_catalog_pin_and_capabilities():
    profile = resolve_profile("qwen-image-2.1")
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == 33_131_596_301
    assert estimate_model_bytes(REPO) == int(28.0 * 1024**3)
    assert estimate_model_bytes("acme/qwen_image_21") == int(28.0 * 1024**3)
    assert estimate_model_bytes("acme/qwen-image-v2.1-mflux-q8") == int(28.0 * 1024**3)
    engine = ImageGenerationEngine(REPO)
    assert engine.family == "qwen-image-2.1"
    assert engine.default_steps == engine.default_edit_steps == 40
    assert engine.default_edit_guidance == 1.0
    assert engine.supports_generation and engine.supports_editing
    assert engine._quantize == 8  # noqa: SLF001
    assert ImageGenerationEngine(REPO, quantize=4)._quantize == 4  # noqa: SLF001
    assert ImageGenerationEngine(REPO, quantize=None)._quantize is None  # noqa: SLF001
    operations = next(
        x["capabilities"]["operation_modes"]
        for x in build_catalog_bundle()["snapshot"]["aliases"]
        if x["alias"] == "qwen-image-2.1"
    )
    assert operations == ["text_to_image", "image_to_image"]


def test_rejects_foreign_quantized_pack():
    with pytest.raises(ImageRuntimeError, match="mflux format"):
        ImageGenerationEngine("mlx-community/Qwen-Image-2.1-MLX-4bit")
    with pytest.raises(ImageRuntimeError, match="mflux format"):
        ImageGenerationEngine("mflux-fan/Qwen-Image-2.1-MLX-4bit")


def test_cold_load_uses_pinned_snapshot(monkeypatch):
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(_download_gate, "mflux_local_snapshot", lambda _repo: None)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, **kwargs: calls.append((repo_id, kwargs)) or "/pinned/model",
    )
    assert engine._model_path_for_mflux() == "/pinned/model"  # noqa: SLF001
    assert calls == [(REPO, {"revision": REVISION})]


@pytest.mark.parametrize("for_edit", [False, True])
def test_load_constructs_qwen21_and_registers_memory_policy(monkeypatch, for_edit):
    class FakeCallbacks:
        def __init__(self):
            self.registered = []

        def register(self, callback):
            self.registered.append(callback)

    class FakeQwenImage21:
        def __init__(self, **kwargs):
            self.constructor_kwargs = kwargs
            self.callbacks = FakeCallbacks()

    class FakeMemorySaver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTilingConfig:
        pass

    class FakeModelConfig:
        @staticmethod
        def qwen_image_21():
            return "qwen21-config"

    def install_module(name, **members):
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            prefix = ".".join(parts[:count])
            if prefix not in sys.modules:
                package = types.ModuleType(prefix)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, prefix, package)
        for key, value in members.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)

    install_module(
        "mflux.models.common.config.model_config", ModelConfig=FakeModelConfig
    )
    install_module(
        "mflux.models.qwen21.variants.txt2img.qwen_image_21",
        QwenImage21=FakeQwenImage21,
    )
    install_module(
        "mflux.callbacks.instances.memory_saver", MemorySaver=FakeMemorySaver
    )
    install_module(
        "mflux.models.common.vae.tiling_config", TilingConfig=FakeTilingConfig
    )

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/pinned/model")
    monkeypatch.setattr(engine, "_ensure_runtime_assets", lambda: None)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    model = engine._ensure_loaded(for_edit=for_edit)  # noqa: SLF001

    assert isinstance(model, FakeQwenImage21)
    assert model.constructor_kwargs == {
        "quantize": 8,
        "model_path": "/pinned/model",
        "model_config": "qwen21-config",
    }
    assert isinstance(model.tiling_config, FakeTilingConfig)
    assert model.callbacks.registered[0] is engine._reporter  # noqa: SLF001
    memory_saver = model.callbacks.registered[1]
    assert isinstance(memory_saver, FakeMemorySaver)
    assert memory_saver.kwargs == {
        "model": model,
        "keep_transformer": True,
        "cache_limit_bytes": None,
        "num_seeds": 1,
    }


def test_img2img_passes_one_path_and_strength(monkeypatch, tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (640, 320)).save(path)
    model = types.SimpleNamespace(calls=[])

    def generate_image(**kwargs):
        model.calls.append(kwargs)
        return types.SimpleNamespace(image=Image.new("RGB", (8, 8)))

    model.generate_image = generate_image
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_: model)
    engine.generate(
        prompt="change sky",
        image_paths=[str(path)],
        width=None,
        height=None,
        num_inference_steps=40,
    )
    call = model.calls[0]
    assert call["image_path"] == str(path)
    assert call["image_strength"] == 0.4
    assert call["width"] == 1456 and call["height"] == 720
    assert "image_paths" not in call
    with pytest.raises(ImageRuntimeError, match="one input image"):
        engine.generate(prompt="x", image_paths=[str(path), str(path)])


def test_new_prompt_reloads_evicted_encoder(monkeypatch):
    engine = ImageGenerationEngine(REPO)
    builds = []

    def ensure_loaded(**_):
        if engine._model is None:  # noqa: SLF001
            engine._model = types.SimpleNamespace(
                text_encoder=object(), prompt_cache={}
            )  # noqa: SLF001
            builds.append(engine._model)  # noqa: SLF001
        model = engine._model  # noqa: SLF001

        def generate_image(**kwargs):
            model.prompt_cache[kwargs["prompt"]] = object()
            model.text_encoder = None  # mflux MemorySaver after prompt encoding
            return types.SimpleNamespace(image=Image.new("RGB", (8, 8)))

        model.generate_image = generate_image
        return model

    monkeypatch.setattr(engine, "_ensure_loaded", ensure_loaded)
    monkeypatch.setattr("rapid_mlx.image.engine._release_allocator_cache", lambda: None)
    for prompt in ("first", "first", "second"):
        engine.generate(prompt=prompt)
    assert len(builds) == 2


def test_text_encoder_guard_uses_qwen3_vl_key_and_width(monkeypatch, tmp_path):
    import json
    import struct

    encoder = tmp_path / "text_encoder"
    encoder.mkdir()
    key = "model.language_model.embed_tokens.weight"
    shard = "model-00001-of-00004.safetensors"
    (encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard}})
    )
    monkeypatch.setattr(
        _download_gate, "mflux_local_snapshot", lambda _repo: str(tmp_path)
    )
    engine = ImageGenerationEngine(REPO)

    def write_header(width):
        header = json.dumps(
            {key: {"dtype": "BF16", "shape": [151936, width], "data_offsets": [0, 0]}}
        ).encode()
        (encoder / shard).write_bytes(struct.pack("<Q", len(header)) + header)

    write_header(4096)
    engine._verify_text_encoder_not_quantized()  # noqa: SLF001
    write_header(3584)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()  # noqa: SLF001


def test_official_snapshot_layout_is_complete_and_detects_missing_shard(
    monkeypatch, tmp_path
):
    import json

    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen-Image-2.1" / "snapshots" / REVISION
    files = {
        "processor/tokenizer.json": "{}",
        "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "transformer.block.0": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "transformer.block.1": "diffusion_pytorch_model-00002-of-00002.safetensors",
                }
            }
        ),
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "one",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "two",
        "text_encoder/model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                    "model.language_model.layers.35.weight": "model-00004-of-00004.safetensors",
                }
            }
        ),
        "text_encoder/model-00001-of-00004.safetensors": "one",
        "text_encoder/model-00004-of-00004.safetensors": "four",
        "vae/diffusion_pytorch_model.safetensors": "vae",
    }
    for relative, data in files.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    assert _download_gate.mflux_missing_weights(REPO) == []
    assert _download_gate.mflux_local_snapshot(REPO) == str(snapshot)
    (
        snapshot / "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
    ).unlink()
    assert _download_gate.mflux_missing_weights(REPO) == [
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
    ]
