"""Contracts for the explicit FLUX.2 Klein q4/bf16 selector (#3058)."""

import sys
from types import SimpleNamespace

import pytest

from vllm_mlx import model_sizes
from vllm_mlx._download_gate import IMAGE_MODEL_REVISIONS
from vllm_mlx.cli import build_parser
from vllm_mlx.image.engine import ImageGenerationEngine
from vllm_mlx.image.precision import (
    FLUX2_KLEIN_BF16_ALIAS,
    FLUX2_KLEIN_BF16_REPO,
    FLUX2_KLEIN_Q4_ALIAS,
    resolve_image_weight_precision,
)
from vllm_mlx.model_aliases import resolve_model, resolve_profile
from vllm_mlx.runtime.resident_models import estimate_model_bytes


@pytest.mark.parametrize(
    "source",
    [
        FLUX2_KLEIN_Q4_ALIAS,
        "Runpod/FLUX.2-klein-4B-mflux-4bit",
        FLUX2_KLEIN_BF16_ALIAS,
        FLUX2_KLEIN_BF16_REPO,
        "black-forest-labs/FLUX.2-klein-4B",
    ],
)
def test_explicit_precision_selects_curated_klein_alias(source):
    assert resolve_image_weight_precision(source, "q4") == FLUX2_KLEIN_Q4_ALIAS
    assert resolve_image_weight_precision(source, "bf16") == FLUX2_KLEIN_BF16_ALIAS


@pytest.mark.parametrize(
    "source",
    [
        "z-image-turbo",
        "filipstrand/Z-Image-Turbo-mflux-4bit",
        "diffusion-gemma-26b-4bit",
        "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    ],
)
def test_explicit_precision_rejects_unqualified_diffusion_families(source):
    with pytest.raises(ValueError, match="FLUX.2 Klein only"):
        resolve_image_weight_precision(source, "bf16")


def test_cli_precision_is_explicit_and_defaults_to_no_override():
    parser = build_parser()
    default = parser.parse_args(["serve", FLUX2_KLEIN_Q4_ALIAS])
    bf16 = parser.parse_args(
        ["serve", FLUX2_KLEIN_Q4_ALIAS, "--image-weight-precision", "bf16"]
    )

    assert default.image_weight_precision is None
    assert bf16.image_weight_precision == "bf16"


def test_real_cli_selects_bf16_before_download_and_load(monkeypatch):
    """Drive main far enough to prove the flag changes the actual source."""

    from vllm_mlx import cli, server

    captured = {}

    def _load_model(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["alias"] = server._model_alias

    def _ensure_model_downloaded(model_name, **_kwargs):
        captured.setdefault("download_models", []).append(model_name)

    monkeypatch.setattr(server, "load_model", _load_model)
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", _ensure_model_downloaded)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_resolve_audio_model_for_serve", lambda _n: None)
    monkeypatch.setattr(
        "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "vllm_mlx._version_check.print_staleness_warning_if_any",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            FLUX2_KLEIN_Q4_ALIAS,
            "--image-weight-precision",
            "bf16",
            "--port",
            "0",
        ],
    )

    cli.main()

    assert captured == {
        "model_name": FLUX2_KLEIN_BF16_REPO,
        "alias": FLUX2_KLEIN_BF16_ALIAS,
        "download_models": [FLUX2_KLEIN_BF16_REPO],
    }


def test_bf16_alias_is_image_generation_and_32gb_gated():
    profile = resolve_profile(FLUX2_KLEIN_BF16_ALIAS)
    assert profile is not None
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32
    assert resolve_model(FLUX2_KLEIN_BF16_ALIAS) == FLUX2_KLEIN_BF16_REPO
    assert model_sizes.size_bytes(FLUX2_KLEIN_BF16_REPO) == 15_975_684_703


def test_packaged_bf16_uses_model_path_without_onload_quantization(monkeypatch):
    engine = ImageGenerationEngine(FLUX2_KLEIN_BF16_REPO)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot",
        lambda repo: "/cache/snapshots/bf16",
    )

    assert engine.family == "flux2-klein"
    assert engine._prequantized is False
    assert engine._packaged_checkpoint is True
    assert engine._quantize is None
    assert engine._model_path_for_mflux() == "/cache/snapshots/bf16"


def test_bf16_residency_charge_does_not_fall_through_to_q4():
    gib = 1024**3
    assert estimate_model_bytes(FLUX2_KLEIN_BF16_ALIAS) == 18 * gib
    assert estimate_model_bytes(FLUX2_KLEIN_BF16_REPO) == 18 * gib
    assert estimate_model_bytes(FLUX2_KLEIN_Q4_ALIAS) < 18 * gib


def test_cold_prefetch_keeps_pinned_bf16_revision_through_every_layer(monkeypatch):
    """Do not download moving main and then the pinned 16 GB snapshot again."""

    from vllm_mlx import cli

    revision = IMAGE_MODEL_REVISIONS[FLUX2_KLEIN_BF16_REPO]
    observed = {}

    monkeypatch.setattr(cli.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _model: False)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)

    def _mirror(model_name, **kwargs):
        observed["mirror"] = (model_name, kwargs.get("revision"))
        return False

    def _model_info(model_name, **kwargs):
        observed["metadata"] = (model_name, kwargs.get("revision"))
        return SimpleNamespace(sha=revision, siblings=[])

    def _snapshot_download(model_name, **kwargs):
        observed["download"] = (model_name, kwargs.get("revision"))
        return "/cache/snapshot"

    monkeypatch.setattr(cli, "_try_mirror_prefetch", _mirror)
    monkeypatch.setattr("huggingface_hub.model_info", _model_info)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.pin_main_ref",
        lambda model_name, pinned: observed.setdefault("ref", (model_name, pinned)),
    )

    cli._ensure_model_downloaded(FLUX2_KLEIN_BF16_REPO)

    expected = (FLUX2_KLEIN_BF16_REPO, revision)
    assert observed == {
        "mirror": expected,
        "metadata": expected,
        "download": expected,
        "ref": expected,
    }
