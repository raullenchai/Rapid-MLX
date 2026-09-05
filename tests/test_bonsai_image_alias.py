# SPDX-License-Identifier: Apache-2.0
"""Contracts for the pinned Bonsai Image MLX backend and product alias."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_mlx import _download_gate
from vllm_mlx.catalog import build_catalog_bundle
from vllm_mlx.image.engine import (
    ImageGenerationEngine,
    ImageRuntimeError,
    _detect_family,
)
from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.model_sizes import size_bytes
from vllm_mlx.runtime.resident_models import estimate_model_bytes

REPO = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
REVISION = "2c24c81b934a658ba5590cf39088ba929985b4a8"
SIZE = 3_888_262_196


def test_alias_routes_to_generation_only_image_backend() -> None:
    profile = resolve_profile("bonsai-image-4b-2bit")
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 12

    engine = ImageGenerationEngine(REPO)
    assert engine.family == "bonsai-image"
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.supports_negative_prompt is False
    assert engine.default_steps == 4
    assert engine._prequantized is True  # noqa: SLF001


def test_alias_pins_size_revision_and_resident_budget() -> None:
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == SIZE
    for name in ("bonsai-image-4b-2bit", REPO, "/models/bonsai_image"):
        assert estimate_model_bytes(name) == int(7.0 * 1024**3)


def test_atomic_catalog_names_the_bonsai_adapter() -> None:
    bundle = build_catalog_bundle()
    record = next(
        alias
        for alias in bundle["snapshot"]["aliases"]
        if alias["alias"] == "bonsai-image-4b-2bit"
    )
    assert record["capabilities"]["runtime_adapter"] == "rapid_mlx/bonsai_image"
    assert record["capabilities"]["operation_modes"] == ["text_to_image"]


@pytest.mark.parametrize(
    "name",
    [REPO, "bonsai-image-4b-2bit", "/models/bonsai_image"],
)
def test_family_detection(name: str) -> None:
    assert _detect_family(name) == "bonsai-image"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_inference_steps": 5}, "4-step"),
        ({"width": 257}, "multiples of 32"),
        ({"height": 224}, "between 256 and 2048"),
        ({"guidance": 1.1}, "guidance 1.0"),
        ({"negative_prompt": "blur"}, "negative_prompt"),
        ({"image_paths": ["input.png"]}, "text-to-image only"),
        ({"prompt": "x" * 4097}, "4096 characters"),
    ],
)
def test_request_contract_rejects_before_model_load(kwargs: dict, message: str) -> None:
    engine = ImageGenerationEngine(REPO)
    request = {
        "prompt": "a bonsai tree",
        "width": 512,
        "height": 512,
        "num_inference_steps": 4,
        **kwargs,
    }
    engine._ensure_loaded = lambda **_kwargs: pytest.fail(  # type: ignore[method-assign]  # noqa: SLF001
        "invalid request reached the model loader"
    )
    with pytest.raises(ImageRuntimeError, match=message):
        engine.generate(**request)


def test_generation_dispatch_uses_fixed_published_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class TinyModel:
        def generate_image(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (8, 8)))

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_kwargs: TinyModel())
    png = engine.generate(
        prompt="a bonsai tree", width=512, height=512, seed=17, guidance=1.0
    )
    assert png.startswith(b"\x89PNG")
    assert calls == [
        {
            "height": 512,
            "width": 512,
            "seed": 17,
            "prompt": "a bonsai tree",
            "num_inference_steps": 4,
            "guidance": 1.0,
        }
    ]


def _seed_snapshot(root: Path, *, omit: str | None = None) -> None:
    for relative in _download_gate.BONSAI_IMAGE_DATA_FILES:
        if relative == omit:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


@pytest.mark.parametrize(
    "missing_file",
    [
        "transformer-packed-mflux/diffusion_pytorch_model.safetensors",
        "text_encoder-mlx-4bit/model.safetensors.index.json",
        "tokenizer/chat_template.jinja",
    ],
)
def test_complete_snapshot_requires_every_reviewed_data_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_file: str
) -> None:
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path
        / "models--prism-ml--bonsai-image-ternary-4B-mlx-2bit"
        / "snapshots"
        / REVISION
    )
    _seed_snapshot(snapshot)
    assert _download_gate.mflux_missing_weights(REPO) == []
    (snapshot / missing_file).unlink()
    assert _download_gate.mflux_missing_weights(REPO) == [missing_file]


def test_cold_snapshot_is_pinned_allowlisted_and_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda _name: None
    )
    verified = []
    monkeypatch.setattr(
        engine, "_verify_weights_complete", lambda: verified.append(True)
    )
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda model, **kwargs: calls.append((model, kwargs)) or "/pinned/snapshot",
    )

    assert engine._model_path_for_mflux() == "/pinned/snapshot"
    assert calls == [
        (
            REPO,
            {
                "revision": REVISION,
                "allow_patterns": list(_download_gate.BONSAI_IMAGE_DATA_FILES),
            },
        )
    ]
    assert verified == [True]


@pytest.mark.parametrize("requested", [REPO, "bonsai-image-4b-2bit"])
def test_pull_uses_exact_revision_and_data_allowlist(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    from vllm_mlx import cli
    from vllm_mlx.audio import registry

    calls = []
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **kwargs: calls.append((args.model, kwargs)),
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    monkeypatch.setattr(registry, "runtime_assets_for", lambda _repo: ())
    monkeypatch.setattr(registry, "runtime_requirements_for", lambda _repo: ())

    cli.pull_command(SimpleNamespace(model=requested))

    assert calls == [
        (
            REPO,
            {
                "allow_patterns_override": list(_download_gate.BONSAI_IMAGE_DATA_FILES),
                "revision_override": REVISION,
            },
        )
    ]


@pytest.mark.requires_mlx
def test_runtime_checkpoint_validation_is_fail_closed(tmp_path: Path) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image.bonsai_runtime.runtime import (
        BONSAI_IMAGE_REPO,
        BONSAI_IMAGE_REVISION,
        BonsaiCheckpointError,
        _validate_checkpoint,
    )

    assert BONSAI_IMAGE_REPO == REPO
    assert BONSAI_IMAGE_REVISION == REVISION
    with pytest.raises(BonsaiCheckpointError, match="incomplete"):
        _validate_checkpoint(tmp_path)

    files = {
        "model_index.json": '{"_class_name":"Flux2KleinPipeline"}',
        "scheduler/scheduler_config.json": "{}",
        "tokenizer/tokenizer.json": "{}",
        "text_encoder-mlx-4bit/config.json": (
            '{"quantization":{"bits":4,"group_size":64}}'
        ),
        "text_encoder-mlx-4bit/model.safetensors": "weights",
        "transformer-packed-mflux/config.json": "{}",
        "transformer-packed-mflux/quantization_config.json": (
            '{"bits":2,"group_size":128}'
        ),
        "transformer-packed-mflux/diffusion_pytorch_model.safetensors": "weights",
        "vae/config.json": "{}",
        "vae/diffusion_pytorch_model.safetensors": "weights",
    }
    for relative, body in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    _validate_checkpoint(tmp_path)

    (tmp_path / "transformer-packed-mflux/quantization_config.json").write_text(
        '{"bits":4,"group_size":128}'
    )
    with pytest.raises(BonsaiCheckpointError, match="2-bit/group-128"):
        _validate_checkpoint(tmp_path)


@pytest.mark.requires_mlx
def test_engine_builds_only_the_fixed_bonsai_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image import bonsai_runtime

    built = []

    class TinyBonsai:
        def __init__(self, path):
            built.append(path)

    monkeypatch.setattr(bonsai_runtime, "BonsaiImage", TinyBonsai)
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/snapshot")
    assert isinstance(engine._build_model(), TinyBonsai)
    assert built == ["/snapshot"]

    unsupported = ImageGenerationEngine("/models/bonsai_image")
    with pytest.raises(ImageRuntimeError, match="pinned official checkpoint"):
        unsupported._build_model()

    missing = ImageGenerationEngine(REPO)
    monkeypatch.setattr(missing, "_model_path_for_mflux", lambda: None)
    with pytest.raises(ImageRuntimeError, match="local model snapshot"):
        missing._build_model()


@pytest.mark.requires_mlx
def test_runtime_definitions_and_json_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image.bonsai_runtime import runtime

    components = runtime._VAEWeightDefinition.get_components()
    assert [component.name for component in components] == ["vae"]
    assert runtime._VAEWeightDefinition.get_download_patterns() == [
        "vae/*.safetensors",
        "vae/*.json",
    ]
    assert runtime._VAEWeightDefinition.quantization_predicate(
        "linear", SimpleNamespace(to_quantized=True)
    )
    assert not runtime._VAEWeightDefinition.quantization_predicate("norm", object())
    tokenizer = runtime._tokenizer_definition()
    assert tokenizer.hf_subdir == "tokenizer"
    assert tokenizer.max_length == 512

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    with pytest.raises(runtime.BonsaiCheckpointError, match="Could not read"):
        runtime._read_json_object(malformed)
    malformed.write_text("[]")
    with pytest.raises(runtime.BonsaiCheckpointError, match="JSON object"):
        runtime._read_json_object(malformed)

    monkeypatch.setattr(runtime.mx, "load", lambda _path: "not named tensors")
    with pytest.raises(runtime.BonsaiCheckpointError, match="named tensors"):
        runtime._load_safetensors(tmp_path / "weights.safetensors")


@pytest.mark.requires_mlx
def test_checkpoint_rejects_wrong_pipeline_and_text_quantization(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image.bonsai_runtime import runtime

    files = {
        "model_index.json": '{"_class_name":"WrongPipeline"}',
        "scheduler/scheduler_config.json": "{}",
        "tokenizer/tokenizer.json": "{}",
        "text_encoder-mlx-4bit/config.json": "{}",
        "text_encoder-mlx-4bit/model.safetensors": "weights",
        "transformer-packed-mflux/config.json": "{}",
        "transformer-packed-mflux/quantization_config.json": (
            '{"bits":2,"group_size":128}'
        ),
        "transformer-packed-mflux/diffusion_pytorch_model.safetensors": "weights",
        "vae/config.json": "{}",
        "vae/diffusion_pytorch_model.safetensors": "weights",
    }
    for relative, body in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    with pytest.raises(runtime.BonsaiCheckpointError, match="not a FLUX.2"):
        runtime._validate_checkpoint(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name":"Flux2KleinPipeline"}')
    with pytest.raises(runtime.BonsaiCheckpointError, match="no quantization"):
        runtime._validate_checkpoint(tmp_path)
    (tmp_path / "text_encoder-mlx-4bit/config.json").write_text(
        '{"quantization":{"bits":8,"group_size":64}}'
    )
    with pytest.raises(runtime.BonsaiCheckpointError, match="4-bit/group-64"):
        runtime._validate_checkpoint(tmp_path)


@pytest.mark.requires_mlx
def test_runtime_load_helpers_apply_the_fixed_quantization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image.bonsai_runtime import runtime

    class Weight:
        def __init__(self, name: str):
            self.name = name

        def astype(self, dtype):
            return (self.name, dtype)

    class Encoder:
        def __init__(self, **overrides):
            self.overrides = overrides
            self.updated = None

        def update(self, weights):
            self.updated = weights

    quantized = []
    monkeypatch.setattr(
        runtime.mx,
        "load",
        lambda path: {"model.layers.0.weight": Weight(path)},
    )
    monkeypatch.setattr(runtime, "tree_unflatten", lambda items: {"nested": items})
    monkeypatch.setattr(runtime, "Qwen3TextEncoder", Encoder)
    monkeypatch.setattr(
        runtime.nn,
        "quantize",
        lambda model, **kwargs: quantized.append((model, kwargs)),
    )
    encoder = runtime._load_text_encoder(tmp_path, {"hidden_size": 4})
    assert encoder.overrides == {"hidden_size": 4}
    assert encoder.updated["nested"][0][0] == "layers.0.weight"
    assert quantized[0][1]["bits"] == 4
    assert quantized[0][1]["group_size"] == 64
    assert quantized[0][1]["class_predicate"](
        "linear", SimpleNamespace(to_quantized=True)
    )

    monkeypatch.setattr(runtime.mx, "load", lambda _path: {"metadata": Weight("x")})
    with pytest.raises(runtime.BonsaiCheckpointError, match="no model weights"):
        runtime._load_text_encoder(tmp_path, {})

    spec = SimpleNamespace(
        in_channels=3,
        num_double_blocks=1,
        num_single_blocks=2,
        head_dim=4,
        num_heads=5,
        context_dim=6,
        mlp_ratio=7,
        axes_dims_rope=(1, 1, 2),
        rope_theta=10_000,
        layer_norm_eps=1e-6,
        rms_norm_eps=1e-6,
    )
    transformer = SimpleNamespace(
        time_guidance_embed=SimpleNamespace(
            linear_1=SimpleNamespace(weight=None),
            linear_2=SimpleNamespace(weight=None),
        )
    )
    calls = []
    monkeypatch.setattr(runtime, "Flux2KleinMegakernelSpec", lambda: spec)
    monkeypatch.setattr(
        runtime,
        "load_klein_fast_packed_weights_from_disk",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "packed",
    )
    monkeypatch.setattr(
        runtime,
        "Flux2KleinFastTransformer",
        lambda **kwargs: calls.append(kwargs) or transformer,
    )
    monkeypatch.setattr(
        runtime.mx,
        "load",
        lambda _path: {
            "time_guidance_embed.timestep_embedder.linear_1.weight": Weight("one"),
            "time_guidance_embed.timestep_embedder.linear_2.weight": Weight("two"),
        },
    )
    assert runtime._load_transformer(tmp_path) is transformer
    assert calls[0][0][1] is spec
    assert calls[1]["precision"] == "2bit"
    assert transformer.time_guidance_embed.linear_1.weight[0] == "one"

    monkeypatch.setattr(runtime.mx, "load", lambda _path: {})
    with pytest.raises(runtime.BonsaiCheckpointError, match="timestep weights"):
        runtime._load_transformer(tmp_path)


@pytest.mark.requires_mlx
def test_runtime_constructor_and_prompt_cache_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image.bonsai_runtime import runtime

    model_config = SimpleNamespace(text_encoder_overrides={"hidden_size": 4})
    callbacks = object()
    encoder = object()
    vae_weights = object()
    transformer = object()
    applied = []
    monkeypatch.setattr(
        runtime, "_validate_checkpoint", lambda root: applied.append(root)
    )
    monkeypatch.setattr(runtime.ModelConfig, "flux2_klein_4b", lambda: model_config)
    monkeypatch.setattr(runtime, "CallbackRegistry", lambda: callbacks)
    monkeypatch.setattr(
        runtime.TokenizerLoader,
        "load_all",
        lambda **kwargs: applied.append(kwargs) or {"qwen3": object()},
    )
    monkeypatch.setattr(runtime, "_load_text_encoder", lambda *_args: encoder)

    class FakeVAE:
        _bonsai_tiling = "tiling"

        def __new__(cls):
            return "vae"

    monkeypatch.setattr(runtime, "_TiledFlux2VAE", FakeVAE)
    monkeypatch.setattr(runtime.WeightLoader, "load", lambda **_kwargs: vae_weights)
    monkeypatch.setattr(
        runtime.WeightApplier,
        "apply_and_quantize",
        lambda **kwargs: applied.append(kwargs),
    )
    monkeypatch.setattr(runtime, "_load_transformer", lambda _root: transformer)

    model = runtime.BonsaiImage(tmp_path)
    assert model.callbacks is callbacks
    assert model.text_encoder is encoder
    assert model.vae == "vae"
    assert model.transformer is transformer
    assert model.bits == 2
    assert applied[0] == tmp_path.resolve()

    model.prompt_cache = {}
    model.text_encoder = None
    reloaded = object()
    monkeypatch.setattr(runtime, "_load_text_encoder", lambda *_args: reloaded)
    encoded = []
    monkeypatch.setattr(
        runtime.Flux2Klein,
        "_encode_prompt_pair",
        lambda self, **kwargs: (
            encoded.append((self.text_encoder, kwargs))
            or ("embeds", "ids", "negative", "negative_ids")
        ),
    )
    monkeypatch.setattr(runtime.mx, "eval", lambda *args: applied.append(args))
    monkeypatch.setattr(runtime.mx, "clear_cache", lambda: applied.append("clear"))
    monkeypatch.setattr(runtime.gc, "collect", lambda: applied.append("gc") or 0)

    assert model._encode_prompt_pair(
        prompt="tree", negative_prompt="ignored", guidance=99
    ) == ("embeds", "ids", None, None)
    assert encoded == [
        (
            reloaded,
            {"prompt": "tree", "negative_prompt": None, "guidance": 1.0},
        )
    ]
    assert model.text_encoder is None
    assert model._encode_prompt_pair(
        prompt="tree", negative_prompt=None, guidance=1
    ) == ("embeds", "ids", None, None)
    assert len(encoded) == 1


@pytest.mark.requires_mlx
def test_tiled_vae_uses_bonsai_default_and_honors_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mflux")
    from vllm_mlx.image.bonsai_runtime import runtime

    calls = []
    monkeypatch.setattr(
        runtime.Flux2VAE,
        "decode_packed_latents",
        lambda self, packed, *, tiling_config: (
            calls.append((packed, tiling_config)) or "decoded"
        ),
    )
    vae = runtime._TiledFlux2VAE()
    assert vae.decode_packed_latents("latents") == "decoded"
    custom = object()
    assert vae.decode_packed_latents("other", custom) == "decoded"
    assert calls == [
        ("latents", runtime._TiledFlux2VAE._bonsai_tiling),
        ("other", custom),
    ]
