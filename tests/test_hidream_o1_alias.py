# SPDX-License-Identifier: Apache-2.0
"""Contracts for the HiDream-O1 Dev MLX image backend and product alias."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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

REPO = "mlx-community/HiDream-O1-Image-Dev-mlx-bf16"
REVISION = "33c7a00bce8e3410304f83ec408a15a1eb6782df"
SIZE = 17_649_873_024


def test_alias_routes_to_generation_only_image_backend() -> None:
    profile = resolve_profile("hidream-o1-dev")
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32
    engine = ImageGenerationEngine(REPO)
    assert engine.family == "hidream-o1-dev"
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.default_steps == 28
    assert engine._prequantized is True  # noqa: SLF001


def test_alias_pins_download_size_revision_and_resident_budget() -> None:
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == SIZE
    for name in ("hidream-o1-dev", REPO):
        assert estimate_model_bytes(name) == int(18.0 * 1024**3)


def test_atomic_catalog_names_the_new_backend() -> None:
    bundle = build_catalog_bundle()
    record = next(
        alias
        for alias in bundle["snapshot"]["aliases"]
        if alias["alias"] == "hidream-o1-dev"
    )
    capabilities = record["capabilities"]
    assert capabilities["runtime_adapter"] == "rapid_mlx/hidream_o1"
    assert capabilities["operation_modes"] == ["text_to_image"]


@pytest.mark.parametrize(
    ("name", "family"),
    [
        (REPO, "hidream-o1-dev"),
        ("/models/hidream_o1_dev", "hidream-o1-dev"),
    ],
)
def test_family_detection(name: str, family: str) -> None:
    assert _detect_family(name) == family


def test_patch_round_trip_and_published_schedule() -> None:
    pytest.importorskip("mlx")
    from vllm_mlx.image.hidream_runtime.runtime import (
        DEFAULT_TIMESTEPS,
        FlashFlowMatchScheduler,
        _build_sample,
        _patchify,
        _unpatchify,
    )

    image = np.arange(3 * 64 * 96, dtype=np.float32).reshape(3, 64, 96)
    patches = _patchify(image)
    assert patches.shape == (6, 3 * 32 * 32)
    np.testing.assert_array_equal(_unpatchify(patches, 64, 96), image)
    scheduler = FlashFlowMatchScheduler()
    assert tuple(int(value) for value in scheduler.timesteps) == DEFAULT_TIMESTEPS
    assert len(scheduler.sigmas) == 29
    assert float(scheduler.sigmas[-1]) == 0.0

    class OversizedProcessor:
        tokenizer = None
        boi_token = "<boi>"
        tms_token = "<tms>"

        def __init__(self) -> None:
            self.tokenizer = self

        def apply_chat_template(self, *_args, **_kwargs) -> str:
            return "caption"

        def encode(self, *_args, **_kwargs) -> list[int]:
            return list(range(1025))

    config = SimpleNamespace(
        image_token_id=1, video_token_id=2, vision_start_token_id=3
    )
    with pytest.raises(ValueError, match="1024 tokens"):
        _build_sample("prompt", 1024, 1024, OversizedProcessor(), config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_inference_steps": 27}, "28-step"),
        ({"prompt": "x" * 4097}, "4096 characters"),
        ({"width": 0}, "between 256 and 2048"),
        ({"height": 4096}, "between 256 and 2048"),
        ({"width": 1008}, "multiples of 32"),
        ({"guidance": 4.0}, "omit the guidance"),
        ({"negative_prompt": "bad"}, "does not support negative_prompt"),
        ({"image_paths": ["source.png"]}, "text-to-image only"),
    ],
)
def test_unsupported_requests_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, message: str
) -> None:
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        engine,
        "_ensure_loaded",
        lambda **_kwargs: pytest.fail("invalid request reached the 17 GB loader"),
    )
    request = {
        "prompt": "test",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "seed": 1,
        "guidance": None,
        "negative_prompt": None,
        "image_paths": None,
    }
    request.update(kwargs)
    with pytest.raises(ImageRuntimeError, match=message):
        engine.generate(**request)


def _seed_snapshot(root: Path, *, omit: str | None = None) -> None:
    for relative in _download_gate.HIDREAM_O1_DATA_FILES:
        if relative == omit:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


@pytest.mark.parametrize(
    "missing_file",
    ["extras/custom_heads.safetensors", "generation_config.json"],
)
def test_complete_hidream_snapshot_requires_every_runtime_data_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_file: str
) -> None:
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path
        / "models--mlx-community--HiDream-O1-Image-Dev-mlx-bf16"
        / "snapshots"
        / REVISION
    )
    _seed_snapshot(snapshot)
    assert _download_gate.mflux_missing_weights(REPO) == []
    (snapshot / missing_file).unlink()
    assert _download_gate.mflux_missing_weights(REPO) == [missing_file]


def test_custom_heads_reject_missing_extra_and_shape_mismatch() -> None:
    pytest.importorskip("mlx")
    from vllm_mlx.image.hidream_runtime.runtime import (
        _validate_custom_head_weights,
    )

    expected = {
        "t_embedder1.fc1.weight": SimpleNamespace(shape=(4, 2)),
        "x_embedder.proj1.weight": SimpleNamespace(shape=(8, 4)),
    }
    with pytest.raises(ValueError, match="missing=.*x_embedder"):
        _validate_custom_head_weights(
            expected,
            {"t_embedder1.fc1.weight": SimpleNamespace(shape=(4, 2))},
        )
    with pytest.raises(ValueError, match="unexpected=.*rogue"):
        _validate_custom_head_weights(
            expected,
            {
                **expected,
                "rogue.weight": SimpleNamespace(shape=(1,)),
            },
        )
    with pytest.raises(ValueError, match="shape_mismatch=.*t_embedder1"):
        _validate_custom_head_weights(
            expected,
            {
                **expected,
                "t_embedder1.fc1.weight": SimpleNamespace(shape=(4, 3)),
            },
        )


def test_pull_uses_exact_revision_and_data_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **kwargs: calls.append((args.model, kwargs)),
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    args = SimpleNamespace(model=REPO)

    cli.pull_command(args)

    assert calls == [
        (
            REPO,
            {
                "allow_patterns_override": list(_download_gate.HIDREAM_O1_DATA_FILES),
                "revision_override": REVISION,
            },
        )
    ]


def test_pinned_pull_bypasses_mirror_and_pins_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lower-level pull path must enforce, not merely receive, the pin."""
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_try_mirror_prefetch",
        lambda *_args, **_kwargs: pytest.fail("pinned pulls cannot use mutable main"),
    )
    monkeypatch.setattr(cli, "_blob_identifier", lambda _root: ())
    monkeypatch.setattr(cli, "_print_pull_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _download_gate, "reap_orphan_incomplete_blobs", lambda _repo: (0, 0)
    )

    def fake_snapshot_download(repo_id: str, **kwargs) -> str:
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    cli._pull_repository(
        SimpleNamespace(model=REPO, bits=None, format=None, json=True),
        allow_patterns_override=list(_download_gate.HIDREAM_O1_DATA_FILES),
        revision_override=REVISION,
    )

    assert calls == [
        (
            REPO,
            {
                "allow_patterns": list(_download_gate.HIDREAM_O1_DATA_FILES),
                "revision": REVISION,
            },
        )
    ]
