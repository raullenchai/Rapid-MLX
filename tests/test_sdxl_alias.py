# SPDX-License-Identifier: Apache-2.0
"""Contracts for the pinned, vendored SDXL image backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_mlx import _download_gate
from vllm_mlx.image.engine import (
    ImageGenerationCancelled,
    ImageGenerationEngine,
    ImageRuntimeError,
    _detect_family,
)
from vllm_mlx.model_aliases import resolve_profile

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"


def test_sdxl_alias_is_first_class_image_generation_model() -> None:
    profile = resolve_profile("sdxl-base")
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 16

    engine = ImageGenerationEngine(REPO)
    assert engine.family == "sdxl-base"
    assert engine.default_steps == 30
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.supports_negative_prompt is True


@pytest.mark.parametrize(
    "name",
    ["sdxl-base", REPO, "/models/my_sdxl_checkpoint"],
)
def test_sdxl_family_detection(name: str) -> None:
    assert _detect_family(name) == "sdxl-base"


def test_sdxl_alias_has_size_and_native_catalog_adapter() -> None:
    sizes = json.loads(
        (Path(__file__).parents[1] / "vllm_mlx/model_sizes.json").read_text()
    )["sizes"]
    assert sizes[REPO] == 6_941_201_645

    from vllm_mlx.catalog.legacy import _main_capabilities

    capabilities = _main_capabilities(resolve_profile("sdxl-base"))
    assert capabilities["runtime_adapter"] == "rapid_mlx/sdxl"
    assert capabilities["operation_modes"] == ["text_to_image"]


def test_size_generator_counts_only_the_pinned_sdxl_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    from scripts import gen_model_sizes

    siblings = [
        SimpleNamespace(rfilename=path, size=index + 1)
        for index, path in enumerate(_download_gate.SDXL_DATA_FILES)
    ]
    siblings.append(SimpleNamespace(rfilename="unused-fp32.safetensors", size=10**12))
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or SimpleNamespace(siblings=siblings)
        ),
    )

    assert gen_model_sizes._size_one(REPO) == (
        REPO,
        sum(range(1, len(_download_gate.SDXL_DATA_FILES) + 1)),
    )
    assert calls == [
        (
            (REPO,),
            {
                "revision": REVISION,
                "files_metadata": True,
                "timeout": 5,
            },
        )
    ]


@pytest.mark.requires_mlx
def test_vendored_runtime_is_torch_free_and_reports_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    from vllm_mlx.image.sdxl_runtime import runtime

    calls = []

    class TinyPipeline:
        def __call__(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            kwargs["on_step"](1, 2)
            kwargs["on_step"](2, 2)
            return mx.zeros((1, 8, 8, 3), dtype=mx.float32)

    monkeypatch.setattr(
        runtime.StableDiffusionXLPipeline,
        "from_diffusers",
        classmethod(lambda _cls, path, **kwargs: TinyPipeline()),
    )
    progress = []
    model = runtime.SDXL("/snapshot", on_step=lambda *args: progress.append(args))
    result = model.generate_image(
        prompt="fox",
        negative_prompt="blurry",
        height=512,
        width=768,
        num_inference_steps=2,
        seed=7,
    )

    assert result.image.size == (8, 8)
    assert progress == [(1, 2), (2, 2)]
    assert calls[0][1] == {
        "negative_prompt": "blurry",
        "height": 512,
        "width": 768,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
        "seed": 7,
        "cache_interval": 2,
        "tile_vae": True,
        "progress": False,
        "on_step": model._on_step,
    }

    source = Path(runtime.__file__).parent
    for path in source.rglob("*.py"):
        assert "import torch" not in path.read_text()


@pytest.mark.requires_mlx
def test_engine_build_dispatch_progress_cancel_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.image import sdxl_runtime

    built = []

    class TinySDXL:
        def __init__(self, path, *, on_step):
            built.append((path, on_step))
            self.calls = []

        def generate_image(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (8, 8)))

    monkeypatch.setattr(sdxl_runtime, "SDXL", TinySDXL)
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/snapshot")
    model = engine._build_model()
    assert built == [("/snapshot", engine._report_native_step)]

    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: None)
    with pytest.raises(ImageRuntimeError, match="requires a local model snapshot"):
        engine._build_model()

    engine._report_native_step(4, 30)
    assert engine.progress_snapshot()["step"] == 4
    monkeypatch.setattr(engine, "_is_cancelled", lambda: True)
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        engine._report_native_step(5, 30)

    monkeypatch.setattr(engine, "_is_cancelled", lambda: False)
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_kwargs: model)
    png = engine.generate(
        prompt="fox",
        negative_prompt="blurry",
        width=512,
        height=768,
        num_inference_steps=30,
        seed=11,
    )
    assert png.startswith(b"\x89PNG")
    assert model.calls == [
        {
            "height": 768,
            "width": 512,
            "seed": 11,
            "prompt": "fox",
            "num_inference_steps": 30,
            "negative_prompt": "blurry",
        }
    ]

    with pytest.raises(ImageRuntimeError, match="multiples of 8"):
        engine.generate(prompt="fox", width=513, height=512, num_inference_steps=30)
    with pytest.raises(ImageRuntimeError, match="between 256 and 2048"):
        engine.generate(prompt="fox", width=248, height=512, num_inference_steps=30)
    with pytest.raises(ImageRuntimeError, match="text-to-image only"):
        engine.generate(
            prompt="fox",
            width=512,
            height=512,
            num_inference_steps=30,
            image_paths=["source.png"],
        )


def _seed_snapshot(root: Path, *, omit: str | None = None) -> None:
    for relative in _download_gate.SDXL_DATA_FILES:
        if relative == omit:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


def test_sdxl_snapshot_is_pinned_and_requires_every_runtime_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path
        / "models--stabilityai--stable-diffusion-xl-base-1.0"
        / "snapshots"
        / REVISION
    )
    _seed_snapshot(snapshot)
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert _download_gate.mflux_missing_weights(REPO) == []
    assert _download_gate.mflux_local_snapshot(REPO) == str(snapshot)

    missing = "unet/diffusion_pytorch_model.fp16.safetensors"
    (snapshot / missing).unlink()
    assert _download_gate.mflux_missing_weights(REPO) == [missing]
    assert _download_gate.mflux_local_snapshot(REPO) is None


def test_cold_engine_download_uses_exact_revision_and_data_only_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    calls = []
    monkeypatch.setattr(_download_gate, "mflux_local_snapshot", lambda _repo: None)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/snapshot",
    )
    engine = ImageGenerationEngine(REPO)
    verified = []
    monkeypatch.setattr(
        engine, "_verify_weights_complete", lambda: verified.append(True)
    )

    assert engine._model_path_for_mflux() == "/snapshot"
    assert calls == [
        (
            REPO,
            {
                "revision": REVISION,
                "allow_patterns": list(_download_gate.SDXL_DATA_FILES),
            },
        )
    ]
    assert verified == [True]


@pytest.mark.parametrize("requested", [REPO, "sdxl-base"])
def test_pull_uses_exact_revision_and_data_allowlist(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **kwargs: calls.append((args.model, kwargs)),
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    cli.pull_command(SimpleNamespace(model=requested))

    assert calls == [
        (
            REPO,
            {
                "allow_patterns_override": list(_download_gate.SDXL_DATA_FILES),
                "revision_override": REVISION,
            },
        )
    ]


def test_sdxl_preflight_uses_vendored_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.runtime import image_lane

    probes = []
    monkeypatch.setattr(
        image_lane.importlib.util,
        "find_spec",
        lambda module: probes.append(module) or object(),
    )
    image_lane.require_image_runtime_or_exit(REPO)
    assert probes == ["PIL"]


def test_memory_preflight_sizes_only_the_pinned_sdxl_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub
    import psutil

    from vllm_mlx import cli

    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        lambda *_args, **_kwargs: pytest.fail("must not query the 72 GB whole repo"),
    )
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=32 * 1024**3, available=32 * 1024**3),
    )
    cli._check_memory_capacity(REPO, alias="sdxl-base")


def test_vendored_runtime_provenance_travels_with_source() -> None:
    root = Path(__file__).parents[1] / "vllm_mlx/image/sdxl_runtime"
    notice = (root / "NOTICE").read_text()
    license_text = (root / "LICENSE").read_text()
    assert "a26b42aee4e31999dbb4429226b66d896d49e1d8" in notice
    assert "CC0 1.0 Universal" in license_text
