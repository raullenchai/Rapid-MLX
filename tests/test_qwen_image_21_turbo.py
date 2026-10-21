# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 Turbo catalog, request and load contracts."""

from __future__ import annotations

import json
import struct
import sys
import types

import pytest

from rapid_mlx import _download_gate
from rapid_mlx.catalog import build_catalog_bundle
from rapid_mlx.catalog.legacy import _image_operations
from rapid_mlx.image import engine as engine_module
from rapid_mlx.image.engine import (
    ImageGenerationEngine,
    ImageRequestError,
    ImageRuntimeError,
    _detect_family,
    default_steps_for_model,
)
from rapid_mlx.image.precision import (
    QWEN_IMAGE_21_BF16_REPO,
    QWEN_IMAGE_21_TURBO_ALIAS,
    QWEN_IMAGE_21_TURBO_REPO,
    is_qwen_image_21_turbo,
)
from rapid_mlx.image.qwen21_turbo import (
    SCHEDULER_PATH,
    TURBO_SIGMAS,
    TURBO_STEPS,
    turbo_sigmas,
)
from rapid_mlx.model_aliases import resolve_profile
from rapid_mlx.model_sizes import size_bytes
from rapid_mlx.runtime.resident_models import estimate_model_bytes

REPO = QWEN_IMAGE_21_TURBO_REPO
REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"


def test_alias_pin_size_and_memory_floor():
    profile = resolve_profile(QWEN_IMAGE_21_TURBO_ALIAS)
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == 32_451_593_975
    # Same bf16-encoder + q8-on-load transformer residency as the base path.
    assert estimate_model_bytes(REPO) == int(28.0 * 1024**3)
    assert estimate_model_bytes(QWEN_IMAGE_21_TURBO_ALIAS) == int(28.0 * 1024**3)


@pytest.mark.parametrize(
    "name",
    [REPO, QWEN_IMAGE_21_TURBO_ALIAS, "QWEN/qwen-image-2.1-turbo"],
)
def test_turbo_identity_is_exact(name):
    assert is_qwen_image_21_turbo(name)
    assert _detect_family(name) == "qwen-image-2.1"
    assert default_steps_for_model(name) == TURBO_STEPS == 8


@pytest.mark.parametrize(
    "name",
    [
        QWEN_IMAGE_21_BF16_REPO,
        "qwen-image-2.1",
        "acme/Qwen-Image-2.1-viggle-turbo",
        "acme/Qwen-Image-2.1-Turbo-mflux-q8",
        "",
        None,
    ],
)
def test_other_qwen21_checkpoints_are_not_turbo(name):
    assert not is_qwen_image_21_turbo(name)


def test_engine_defaults_and_text_to_image_only():
    engine = ImageGenerationEngine(REPO)
    assert engine.family == "qwen-image-2.1"
    assert engine._qwen21_turbo is True  # noqa: SLF001
    assert engine.default_steps == 8
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    # Same on-load q8 transformer path as the bf16 base alias.
    assert engine._quantize == 8  # noqa: SLF001
    assert engine._prequantized is False  # noqa: SLF001
    base = ImageGenerationEngine(QWEN_IMAGE_21_BF16_REPO)
    assert base._qwen21_turbo is False  # noqa: SLF001
    assert base.default_steps == 40
    assert base.supports_editing is True


def test_catalog_operations_and_cli_kind():
    assert _image_operations(REPO) == ["text_to_image"]
    assert _image_operations(QWEN_IMAGE_21_BF16_REPO) == [
        "text_to_image",
        "image_to_image",
    ]
    operations = next(
        x["capabilities"]["operation_modes"]
        for x in build_catalog_bundle()["snapshot"]["aliases"]
        if x["alias"] == QWEN_IMAGE_21_TURBO_ALIAS
    )
    assert operations == ["text_to_image"]


def test_cli_models_lists_turbo_as_generation_only(capsys):
    from rapid_mlx.cli import _available_models_json_payload, models_command

    models_command(types.SimpleNamespace(json=False, cached=False))
    line = next(
        row
        for row in capsys.readouterr().out.splitlines()
        if row.strip().startswith(QWEN_IMAGE_21_TURBO_ALIAS + " ")
    )
    assert "[image:gen]" in line
    assert REPO in line
    entry = next(
        x
        for x in _available_models_json_payload()["image"]
        if x["alias"] == QWEN_IMAGE_21_TURBO_ALIAS
    )
    assert entry["default_steps"] == 8
    assert entry["min_memory_gb"] == 32


def test_request_validation():
    engine = ImageGenerationEngine(REPO)
    engine.validate_request(prompt="sign", num_inference_steps=8)
    engine.validate_request(prompt="sign", num_inference_steps=8, guidance=1.0)
    # The checkpoint documents only its saved eight-node schedule.
    for steps in (4, 40):
        with pytest.raises(ImageRequestError, match="8-step schedule only"):
            engine.validate_request(prompt="sign", num_inference_steps=steps)
    with pytest.raises(ImageRequestError, match="classifier-free guidance"):
        engine.validate_request(prompt="sign", num_inference_steps=8, guidance=4.0)
    for paths in (["a.png"], ["a.png", "b.png"]):
        with pytest.raises(ImageRequestError, match="text-to-image only"):
            engine.validate_request(prompt="edit", image_paths=paths)


def test_gen_kwargs_select_turbo_scheduler():
    engine = ImageGenerationEngine(REPO)
    kwargs = engine._gen_kwargs(3, "sign", 8, None, "blurry")  # noqa: SLF001
    assert kwargs["scheduler"] == SCHEDULER_PATH
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance"] == 1.0
    base = ImageGenerationEngine(QWEN_IMAGE_21_BF16_REPO)
    assert "scheduler" not in base._gen_kwargs(3, "sign", 40, None, None)  # noqa: SLF001


def test_turbo_sigmas_are_the_published_nodes_only():
    assert turbo_sigmas(8) == [*TURBO_SIGMAS, 0.0]
    for steps in (0, 4, 16):
        with pytest.raises(ValueError, match="8-step schedule only"):
            turbo_sigmas(steps)


def _install_module(monkeypatch, name, **members):
    parts = name.split(".")
    for count in range(1, len(parts) + 1):
        prefix = ".".join(parts[:count])
        if prefix not in sys.modules:
            package = types.ModuleType(prefix)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, prefix, package)
    for key, value in members.items():
        monkeypatch.setattr(sys.modules[name], key, value, raising=False)


def test_load_globs_unsharded_text_encoder_and_restores_definition(monkeypatch):
    class Component:
        def __init__(self, name, loading_mode):
            self.name = name
            self.loading_mode = loading_mode

    class FakeDefinition:
        @staticmethod
        def get_components():
            return [
                Component("transformer", "multi_glob"),
                Component("text_encoder", "multi_json"),
            ]

    original = FakeDefinition.get_components

    class FakeCallbacks:
        def __init__(self):
            self.registered = []

        def register(self, callback):
            self.registered.append(callback)

    class FakeQwenImage21:
        def __init__(self, **kwargs):
            # Construction happens while the shared definition lock is held.
            assert engine_module._QWEN21_DEFINITION_LOCK.locked()  # noqa: SLF001
            modes = {c.name: c.loading_mode for c in FakeDefinition.get_components()}
            assert modes == {"transformer": "multi_glob", "text_encoder": "multi_glob"}
            self.constructor_kwargs = kwargs
            self.callbacks = FakeCallbacks()

    class FakeModelConfig:
        @staticmethod
        def qwen_image_21():
            return "qwen21-config"

    _install_module(
        monkeypatch,
        "mflux.models.common.config.model_config",
        ModelConfig=FakeModelConfig,
    )
    _install_module(
        monkeypatch,
        "mflux.models.qwen21.variants.txt2img.qwen_image_21",
        QwenImage21=FakeQwenImage21,
    )
    _install_module(
        monkeypatch,
        "mflux.models.qwen21.weights.qwen21_weight_definition",
        Qwen21WeightDefinition=FakeDefinition,
    )
    _install_module(
        monkeypatch,
        "mflux.callbacks.instances.memory_saver",
        MemorySaver=lambda **kwargs: kwargs,
    )
    _install_module(
        monkeypatch,
        "mflux.models.common.vae.tiling_config",
        TilingConfig=lambda: "tiling",
    )

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/pinned/turbo")
    monkeypatch.setattr(engine, "_ensure_runtime_assets", lambda: None)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    model = engine._ensure_loaded(for_edit=False)  # noqa: SLF001

    assert isinstance(model, FakeQwenImage21)
    assert model.constructor_kwargs == {
        "quantize": 8,
        "model_path": "/pinned/turbo",
        "model_config": "qwen21-config",
    }
    # The process-global mflux definition is restored after the build.
    assert FakeDefinition.get_components is original
    restored = {c.name: c.loading_mode for c in FakeDefinition.get_components()}
    assert restored["text_encoder"] == "multi_json"
    assert not engine_module._QWEN21_DEFINITION_LOCK.locked()  # noqa: SLF001


def _write_safetensors(path, tensors):
    header = json.dumps(
        {"__metadata__": {"format": "pt"}, **tensors}, separators=(",", ":")
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)


def _cache_snapshot(tmp_path, monkeypatch, *, text_encoder_tensors):
    hub = tmp_path / "hub"
    snapshot = hub / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION
    (snapshot / "processor").mkdir(parents=True)
    (snapshot / "processor" / "tokenizer.json").write_text("{}")
    (snapshot / "transformer").mkdir()
    (
        snapshot / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors"
    ).write_bytes(b"x")
    (
        snapshot / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    ).write_text(
        json.dumps(
            {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00001.safetensors"}}
        )
    )
    (snapshot / "vae").mkdir()
    (snapshot / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    (snapshot / "text_encoder").mkdir()
    if text_encoder_tensors is not None:
        _write_safetensors(
            snapshot / "text_encoder" / "model.safetensors", text_encoder_tensors
        )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    return snapshot


def test_completeness_contract_accepts_unsharded_encoder(tmp_path, monkeypatch):
    _cache_snapshot(
        tmp_path,
        monkeypatch,
        text_encoder_tensors={
            "model.language_model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [151936, 4096],
                "data_offsets": [0, 16],
            }
        },
    )
    assert _download_gate.mflux_missing_weights(REPO) == []
    # The bf16 encoder guard reads the single file's header in place of an index.
    ImageGenerationEngine(REPO)._verify_text_encoder_not_quantized()  # noqa: SLF001


def test_completeness_contract_names_missing_encoder(tmp_path, monkeypatch):
    _cache_snapshot(tmp_path, monkeypatch, text_encoder_tensors=None)
    assert _download_gate.mflux_missing_weights(REPO) == [
        "text_encoder/model.safetensors"
    ]


def test_encoder_guard_rejects_quantized_unsharded_encoder(tmp_path, monkeypatch):
    _cache_snapshot(
        tmp_path,
        monkeypatch,
        text_encoder_tensors={
            "model.language_model.embed_tokens.weight": {
                "dtype": "U32",
                "shape": [151936, 512],
                "data_offsets": [0, 8],
            },
            "model.language_model.embed_tokens.scales": {
                "dtype": "BF16",
                "shape": [151936, 64],
                "data_offsets": [8, 16],
            },
        },
    )
    with pytest.raises(ImageRuntimeError):
        ImageGenerationEngine(REPO)._verify_text_encoder_not_quantized()  # noqa: SLF001


def test_encoder_guard_skips_unreadable_unsharded_encoder(tmp_path):
    text_encoder = tmp_path / "text_encoder"
    text_encoder.mkdir()
    assert ImageGenerationEngine._read_text_encoder_weight_map(text_encoder) is None  # noqa: SLF001
    (text_encoder / "model.safetensors").write_bytes(b"\0" * 4)
    assert ImageGenerationEngine._read_text_encoder_weight_map(text_encoder) is None  # noqa: SLF001
