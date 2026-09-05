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


def test_explicit_precision_rejects_unknown_precision():
    with pytest.raises(ValueError, match="must be one of: q4, bf16"):
        resolve_image_weight_precision(FLUX2_KLEIN_Q4_ALIAS, "fp8")


def test_explicit_precision_rejects_local_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="not local checkpoints"):
        resolve_image_weight_precision(str(tmp_path), "bf16")


def test_explicit_precision_preserves_local_path_precedence(monkeypatch, tmp_path):
    (tmp_path / FLUX2_KLEIN_Q4_ALIAS).mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not local checkpoints"):
        resolve_image_weight_precision(FLUX2_KLEIN_Q4_ALIAS, "bf16")


def test_cli_precision_is_explicit_and_defaults_to_no_override():
    parser = build_parser()
    default = parser.parse_args(["serve", FLUX2_KLEIN_Q4_ALIAS])
    bf16 = parser.parse_args(
        ["serve", FLUX2_KLEIN_Q4_ALIAS, "--image-weight-precision", "bf16"]
    )

    assert default.image_weight_precision is None
    assert bf16.image_weight_precision == "bf16"


def test_real_cli_selects_bf16_before_serve_dispatch(monkeypatch):
    """Drive main through selection and alias resolution without MLX imports."""

    from vllm_mlx import cli

    captured = {}

    def _serve_command(args):
        captured["model_name"] = args.model
        captured["alias"] = args._original_alias

    monkeypatch.setattr(cli, "serve_command", _serve_command)
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
    }


def test_real_cli_reports_unsupported_precision_family(monkeypatch, capsys):
    from vllm_mlx import cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "z-image-turbo",
            "--image-weight-precision",
            "bf16",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "FLUX.2 Klein only" in capsys.readouterr().err


@pytest.mark.parametrize(
    "model_name",
    [FLUX2_KLEIN_BF16_ALIAS, FLUX2_KLEIN_BF16_REPO],
)
def test_bf16_alias_and_repo_are_image_generation_and_32gb_gated(model_name):
    profile = resolve_profile(model_name)
    assert profile is not None
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32
    assert resolve_model(FLUX2_KLEIN_BF16_ALIAS) == FLUX2_KLEIN_BF16_REPO
    assert model_sizes.size_bytes(FLUX2_KLEIN_BF16_REPO) == 15_975_684_703


def test_bf16_repo_directly_triggers_32gb_admission_warning(monkeypatch, capsys):
    from vllm_mlx import cli

    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: SimpleNamespace(total=24 * 1024**3),
    )

    cli._check_alias_min_memory(FLUX2_KLEIN_BF16_REPO)

    warning = capsys.readouterr().out
    assert FLUX2_KLEIN_BF16_REPO in warning
    assert "32 GB unified-memory floor" in warning


@pytest.mark.requires_mlx
def test_packaged_bf16_uses_model_path_without_onload_quantization(monkeypatch):
    engine = ImageGenerationEngine(FLUX2_KLEIN_BF16_REPO)
    constructor_kwargs = {}
    built_model = object()

    def _build_flux2_klein(**kwargs):
        constructor_kwargs.update(kwargs)
        return built_model

    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot",
        lambda repo: "/cache/snapshots/bf16",
    )
    monkeypatch.setattr(
        "mflux.models.flux2.variants.txt2img.flux2_klein.Flux2Klein",
        _build_flux2_klein,
    )

    assert engine.family == "flux2-klein"
    assert engine._prequantized is False
    assert engine._packaged_checkpoint is True
    assert engine._quantize is None
    assert engine._model_path_for_mflux() == "/cache/snapshots/bf16"
    assert engine._build_model() is built_model
    assert constructor_kwargs["model_path"] == "/cache/snapshots/bf16"
    assert constructor_kwargs["quantize"] is None


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

    monkeypatch.setattr(
        "vllm_mlx._mirror.download_with_mirror_fallback",
        _mirror,
    )
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


def test_pinned_image_cache_does_not_accept_complete_moving_main(monkeypatch, tmp_path):
    """A cached main snapshot cannot hide a missing pinned image revision."""

    import json

    import huggingface_hub.constants

    from vllm_mlx import _download_gate as download_gate
    from vllm_mlx import cli

    main_revision = "b" * 40
    assert main_revision != IMAGE_MODEL_REVISIONS[FLUX2_KLEIN_BF16_REPO]
    repo_root = tmp_path / f"models--{FLUX2_KLEIN_BF16_REPO.replace('/', '--')}"
    main_snapshot = repo_root / "snapshots" / main_revision
    main_snapshot.mkdir(parents=True)
    (main_snapshot / "model.safetensors").write_bytes(b"complete main weights")
    tokenizer = main_snapshot / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    for component in ("transformer", "text_encoder", "vae"):
        component_dir = main_snapshot / component
        component_dir.mkdir()
        shard = "model-00001-of-00001.safetensors"
        (component_dir / shard).write_bytes(b"complete component weights")
        (component_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"tensor": shard}})
        )
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text(main_revision)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    assert download_gate.is_repo_cached(FLUX2_KLEIN_BF16_REPO) is True
    assert (
        download_gate._snapshot_is_complete_mflux_model(FLUX2_KLEIN_BF16_REPO) is False
    )
    assert cli._cache_runnability(FLUX2_KLEIN_BF16_REPO) is False
