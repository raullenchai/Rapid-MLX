# SPDX-License-Identifier: Apache-2.0
"""Contracts for the pinned, vendored SD3.5 Large image backend."""

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

REPO = _download_gate.SD35_REPO
REVISION = _download_gate.SD35_REVISION


def test_sd35_alias_is_first_class_image_generation_model() -> None:
    profile = resolve_profile("sd35-large-4bit")
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32

    engine = ImageGenerationEngine(REPO)
    assert engine.family == "sd35-large"
    assert engine.default_steps == 28
    assert engine._prequantized is True  # noqa: SLF001
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.supports_negative_prompt is True


@pytest.mark.parametrize(
    "name",
    ["sd35-large-4bit", REPO, "/models/my_sd3.5_checkpoint"],
)
def test_sd35_family_detection(name: str) -> None:
    assert _detect_family(name) == "sd35-large"


def test_sd35_alias_has_composite_size_and_native_catalog_adapter() -> None:
    sizes = json.loads(
        (Path(__file__).parents[1] / "vllm_mlx/model_sizes.json").read_text()
    )["sizes"]
    assert sizes[REPO] == 16_378_940_179

    from vllm_mlx.catalog.legacy import _main_capabilities

    capabilities = _main_capabilities(resolve_profile("sd35-large-4bit"))
    assert capabilities["runtime_adapter"] == "rapid_mlx/sd35"
    assert capabilities["operation_modes"] == ["text_to_image"]


def test_size_generator_sums_primary_and_pinned_runtime_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    from scripts import gen_model_sizes

    payloads = (
        (REPO, REVISION, _download_gate.SD35_DATA_FILES),
        (
            _download_gate.SD35_SHARED_REPO,
            _download_gate.SD35_SHARED_REVISION,
            _download_gate.SD35_SHARED_DATA_FILES,
        ),
        (
            _download_gate.SD35_T5_TOKENIZER_REPO,
            _download_gate.SD35_T5_TOKENIZER_REVISION,
            _download_gate.SD35_T5_TOKENIZER_DATA_FILES,
        ),
    )
    metadata = {
        repo: SimpleNamespace(
            siblings=[
                SimpleNamespace(rfilename=path, size=index + 1)
                for index, path in enumerate(files)
            ]
        )
        for repo, _revision, files in payloads
    }
    calls = []

    def model_info(repo, **kwargs):
        calls.append((repo, kwargs))
        return metadata[repo]

    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    expected = sum(sum(range(1, len(files) + 1)) for _, _, files in payloads)
    assert gen_model_sizes._size_one(REPO) == (REPO, expected)
    assert calls == [
        (
            repo,
            {"revision": revision, "files_metadata": True, "timeout": 5},
        )
        for repo, revision, _files in payloads
    ]


def _seed_snapshot(root: Path, files: tuple[str, ...], *, omit: str | None = None):
    for relative in files:
        if relative == omit:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


def _snapshot_root(cache: Path, repo: str, revision: str) -> Path:
    return cache / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def test_sd35_snapshot_requires_primary_and_every_auxiliary_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    payloads = (
        (REPO, REVISION, _download_gate.SD35_DATA_FILES),
        (
            _download_gate.SD35_SHARED_REPO,
            _download_gate.SD35_SHARED_REVISION,
            _download_gate.SD35_SHARED_DATA_FILES,
        ),
        (
            _download_gate.SD35_T5_TOKENIZER_REPO,
            _download_gate.SD35_T5_TOKENIZER_REVISION,
            _download_gate.SD35_T5_TOKENIZER_DATA_FILES,
        ),
    )
    roots = []
    for repo, revision, files in payloads:
        root = _snapshot_root(tmp_path, repo, revision)
        roots.append(root)
        _seed_snapshot(root, files)

    assert _download_gate.mflux_missing_weights(REPO) == []
    assert _download_gate.mflux_local_snapshot(REPO) == str(roots[0])

    (roots[1] / _download_gate.SD35_SHARED_DATA_FILES[-1]).unlink()
    assert _download_gate.mflux_missing_weights(REPO) == [
        f"{_download_gate.SD35_SHARED_REPO}@{_download_gate.SD35_SHARED_REVISION}"
    ]
    assert _download_gate.mflux_local_snapshot(REPO) is None


def test_pinned_snapshot_fails_closed_for_unavailable_or_unsafe_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import builtins
    import os

    import huggingface_hub.constants

    assert _download_gate.pinned_image_snapshot("publisher/not-registered") is None

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "huggingface_hub.constants":
            raise ImportError("hub constants unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert _download_gate.pinned_image_snapshot(REPO) is None
    monkeypatch.setattr(builtins, "__import__", real_import)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    assert _download_gate.pinned_image_snapshot(REPO) is None

    snapshot = _snapshot_root(tmp_path, REPO, REVISION)
    _seed_snapshot(snapshot, _download_gate.SD35_DATA_FILES)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"x")
    first = snapshot / _download_gate.SD35_DATA_FILES[0]
    first.unlink()
    first.symlink_to(outside)
    assert _download_gate.pinned_image_snapshot(REPO) is None

    first.unlink()
    first.write_bytes(b"x")
    monkeypatch.setattr(
        os.path, "getsize", lambda _path: (_ for _ in ()).throw(OSError())
    )
    assert _download_gate.pinned_image_snapshot(REPO) is None


@pytest.mark.parametrize("requested", [REPO, "sd35-large-4bit"])
def test_pull_fetches_all_three_exact_allowlisted_snapshots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], requested: str
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

    assert "Stability AI Community License" in capsys.readouterr().out

    assert calls == [
        (
            REPO,
            {
                "allow_patterns_override": list(_download_gate.SD35_DATA_FILES),
                "revision_override": REVISION,
            },
        ),
        (
            _download_gate.SD35_SHARED_REPO,
            {
                "allow_patterns_override": list(_download_gate.SD35_SHARED_DATA_FILES),
                "revision_override": _download_gate.SD35_SHARED_REVISION,
            },
        ),
        (
            _download_gate.SD35_T5_TOKENIZER_REPO,
            {
                "allow_patterns_override": list(
                    _download_gate.SD35_T5_TOKENIZER_DATA_FILES
                ),
                "revision_override": _download_gate.SD35_T5_TOKENIZER_REVISION,
            },
        ),
    ]


def test_engine_ensures_only_missing_pinned_runtime_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    engine = ImageGenerationEngine(REPO)
    calls = []
    monkeypatch.setattr(
        _download_gate,
        "pinned_image_snapshot",
        lambda repo: "/cached" if repo == _download_gate.SD35_SHARED_REPO else None,
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/downloaded",
    )
    engine._ensure_runtime_assets()  # noqa: SLF001
    assert calls == [
        (
            _download_gate.SD35_T5_TOKENIZER_REPO,
            {
                "revision": _download_gate.SD35_T5_TOKENIZER_REVISION,
                "allow_patterns": list(_download_gate.SD35_T5_TOKENIZER_DATA_FILES),
            },
        )
    ]


def test_engine_runtime_asset_download_is_local_noop_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_model = tmp_path / "sd3.5-local"
    local_model.mkdir()
    ImageGenerationEngine(str(local_model))._ensure_runtime_assets()  # noqa: SLF001

    import huggingface_hub

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(_download_gate, "pinned_image_snapshot", lambda _repo: None)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    with pytest.raises(ImageRuntimeError, match="Could not download.*offline"):
        engine._ensure_runtime_assets()  # noqa: SLF001


@pytest.mark.parametrize(
    "repo",
    [_download_gate.SD35_SHARED_REPO, _download_gate.SD35_T5_TOKENIZER_REPO],
)
def test_direct_pull_of_an_auxiliary_repo_keeps_generic_semantics(
    monkeypatch: pytest.MonkeyPatch, repo: str
) -> None:
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli, "_pull_repository", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    args = SimpleNamespace(model=repo)
    cli.pull_command(args)

    assert calls == [((args,), {})]


@pytest.mark.requires_mlx
def test_runtime_accepts_only_complete_local_assets_and_forwards_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_mlx.image.sd35_runtime import runtime

    model = tmp_path / "model"
    shared = tmp_path / "shared"
    t5 = tmp_path / "t5"
    _seed_snapshot(model, runtime.MODEL_FILES)
    _seed_snapshot(shared, runtime.SHARED_FILES)
    _seed_snapshot(t5, runtime.T5_TOKENIZER_FILES)
    (model / "config.json").write_text(
        json.dumps({"name": "stable-diffusion-3.5-large-4bit-quantized"})
    )
    constructed = []
    generated = []

    class TinyPipeline:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def generate_image(self, **kwargs):
            generated.append(kwargs)
            return Image.new("RGB", (8, 8)), {"peak_memory": 1.0}

    monkeypatch.setattr(runtime, "DiffusionPipeline", TinyPipeline)
    progress = lambda *_args: None
    adapter = runtime.SD35Large(model, shared, t5, on_step=progress)
    result = adapter.generate_image(
        prompt="fox",
        height=512,
        width=768,
        num_inference_steps=28,
        seed=7,
        guidance=None,
        negative_prompt="blurry",
    )
    assert result.image.size == (8, 8)
    assert constructed == [
        {
            "w16": True,
            "a16": True,
            "shift": 3.0,
            "use_t5": True,
            "model_version": runtime.MODEL_REPO,
            "low_memory_mode": True,
            "local_ckpt": model / runtime.MODEL_FILENAME,
            "on_step": progress,
        }
    ]
    assert generated == [
        {
            "text": "fox",
            "num_steps": 28,
            "cfg_weight": 3.5,
            "negative_text": "blurry",
            "latent_size": (64, 96),
            "seed": 7,
            "verbose": False,
        }
    ]

    with pytest.raises(ValueError, match="directory is missing"):
        runtime._require_files(tmp_path / "absent", ("weight.safetensors",))

    (model / "config.json").write_text(json.dumps({"name": "wrong-checkpoint"}))
    with pytest.raises(ValueError, match="Unsupported SD3.5 checkpoint"):
        runtime.SD35Large(model, shared, t5)
    (model / "config.json").write_text(
        json.dumps({"name": "stable-diffusion-3.5-large-4bit-quantized"})
    )

    (shared / runtime.SHARED_FILES[0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        runtime.SD35Large(model, shared, t5)


@pytest.mark.requires_mlx
def test_runtime_accepts_hf_blob_symlinks_but_rejects_external_ones(
    tmp_path: Path,
) -> None:
    import mlx.core as mx

    from vllm_mlx.image.sd35_runtime import runtime
    from vllm_mlx.image.sd35_runtime._vendor import model_io

    repo_root = tmp_path / "models--publisher--model"
    snapshot = repo_root / "snapshots" / "revision"
    blob = repo_root / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    staged_blob = blob.with_suffix(".safetensors")
    mx.save_safetensors(str(staged_blob), {"weight": mx.array([1.0])})
    staged_blob.replace(blob)
    snapshot.mkdir(parents=True)
    (snapshot / "weight.safetensors").symlink_to(blob)
    runtime._require_files(snapshot, ("weight.safetensors",))
    model_io.configure_asset_roots({"repo/model": snapshot}, t5_tokenizer_root=snapshot)
    asset_path = model_io._asset_path("repo/model", "weight.safetensors")
    assert asset_path == str(blob)
    assert set(model_io._load_safetensors(asset_path)) == {"weight"}

    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    (snapshot / "external.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="missing"):
        runtime._require_files(snapshot, ("external.safetensors",))
    with pytest.raises(FileNotFoundError, match="missing"):
        model_io._asset_path("repo/model", "external.safetensors")


@pytest.mark.requires_mlx
@pytest.mark.parametrize("fail", [False, True])
def test_t5_low_memory_scope_restores_the_callers_allocator_limit(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from vllm_mlx.image.sd35_runtime._vendor.t5 import TransformerEncoder

    class Layer:
        def __call__(self, value, *, mask):
            if fail:
                raise RuntimeError("encoder failed")
            return value

    class Identity:
        def __call__(self, value, *args):
            return value if not args else None

    encoder = TransformerEncoder.__new__(TransformerEncoder)
    nn.Module.__init__(encoder)
    encoder.low_memory_mode = True
    encoder.layers = [Layer()]
    encoder.ln = Identity()
    encoder.relative_attention_bias = Identity()
    calls = []

    def set_memory_limit(limit):
        calls.append(limit)
        return 9 * 1024**3

    monkeypatch.setattr(mx, "set_memory_limit", set_memory_limit)
    if fail:
        with pytest.raises(RuntimeError, match="encoder failed"):
            encoder(mx.ones((1, 2, 3)))
    else:
        assert encoder(mx.ones((1, 2, 3))).shape == (1, 2, 3)
    assert calls == [4 * 1024**3, 9 * 1024**3]


@pytest.mark.requires_mlx
def test_sd35_cancellation_restores_cached_modulation_weights() -> None:
    import mlx.core as mx

    from vllm_mlx.image.sd35_runtime._vendor.pipeline import sample_euler

    class Sampler:
        @staticmethod
        def timestep(sigmas):
            return sigmas

    class Denoiser:
        def __init__(self):
            self.model = SimpleNamespace(sampler=Sampler(), activation_dtype=mx.float32)
            self.cleared = 0

        def cache_modulation_params(self, pooled, timesteps):
            assert pooled == "pooled"

        def clear_cache(self):
            self.cleared += 1

        def __call__(self, value, *_args, **_kwargs):
            return mx.zeros_like(value)

    denoiser = Denoiser()

    def cancel(*_args):
        raise ImageGenerationCancelled("cancelled")

    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        sample_euler(
            denoiser,
            mx.ones((1, 2)),
            mx.array([1.0, 0.0]),
            extra_args={"pooled_conditioning": "pooled"},
            on_step=cancel,
        )
    assert denoiser.cleared == 1


@pytest.mark.requires_mlx
def test_engine_build_dispatch_progress_cancel_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.image import sd35_runtime

    built = []

    class TinySD35:
        def __init__(self, model, shared, t5, *, on_step):
            built.append((model, shared, t5, on_step))
            self.calls = []

        def generate_image(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (8, 8)))

    monkeypatch.setattr(sd35_runtime, "SD35Large", TinySD35)
    monkeypatch.setattr(
        _download_gate,
        "pinned_image_snapshot",
        lambda repo: {
            _download_gate.SD35_SHARED_REPO: "/shared",
            _download_gate.SD35_T5_TOKENIZER_REPO: "/t5",
        }.get(repo),
    )
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/model")
    model = engine._build_model()  # noqa: SLF001
    assert built == [
        ("/model", "/shared", "/t5", engine._report_native_step)  # noqa: SLF001
    ]

    monkeypatch.setattr(_download_gate, "pinned_image_snapshot", lambda _repo: None)
    with pytest.raises(ImageRuntimeError, match="requires its pinned model"):
        engine._build_model()  # noqa: SLF001

    engine._report_native_step(4, 28)  # noqa: SLF001
    assert engine.progress_snapshot()["step"] == 4
    monkeypatch.setattr(engine, "_is_cancelled", lambda: True)
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        engine._report_native_step(5, 28)  # noqa: SLF001

    monkeypatch.setattr(engine, "_is_cancelled", lambda: False)
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_kwargs: model)
    png = engine.generate(
        prompt="red panda astronaut",
        negative_prompt="blurry",
        width=512,
        height=768,
        num_inference_steps=28,
        seed=11,
        guidance=3.5,
    )
    assert png.startswith(b"\x89PNG")
    assert model.calls == [
        {
            "height": 768,
            "width": 512,
            "seed": 11,
            "prompt": "red panda astronaut",
            "num_inference_steps": 28,
            "guidance": 3.5,
            "negative_prompt": "blurry",
        }
    ]

    with pytest.raises(ImageRuntimeError, match="28-step"):
        engine.generate(prompt="fox", width=512, height=512, num_inference_steps=4)
    with pytest.raises(ImageRuntimeError, match="4096 characters"):
        engine.generate(
            prompt="x" * 4097, width=512, height=512, num_inference_steps=28
        )
    with pytest.raises(ImageRuntimeError, match="multiples of 16"):
        engine.generate(prompt="fox", width=520, height=512, num_inference_steps=28)
    with pytest.raises(ImageRuntimeError, match="between 256 and 1536"):
        engine.generate(prompt="fox", width=2048, height=512, num_inference_steps=28)
    with pytest.raises(ImageRuntimeError, match="guidance"):
        engine.generate(
            prompt="fox", width=512, height=512, num_inference_steps=28, guidance=21
        )


def test_sd35_preflight_uses_bundled_runtime_dependencies(
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


def test_vendored_runtime_provenance_and_model_license_boundary() -> None:
    root = Path(__file__).parents[1] / "vllm_mlx/image/sd35_runtime"
    assert "498e5dba5fb48b0f01cb6b5c2292a6dbea67a317" in (root / "NOTICE").read_text()
    assert "MIT License" in (root / "LICENSE").read_text()
    assert not list(root.rglob("*.safetensors"))
    assert (
        "no model weights are distributed"
        in (Path(__file__).parents[1] / "NOTICE").read_text().casefold()
    )
