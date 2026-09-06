# SPDX-License-Identifier: Apache-2.0
"""Product contracts for the revision-pinned Qwen Image Edit alias."""

from __future__ import annotations

from vllm_mlx import _download_gate
from vllm_mlx.catalog import build_catalog_bundle
from vllm_mlx.image.engine import ImageGenerationEngine
from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.model_sizes import size_bytes
from vllm_mlx.routes.models import _detect_capabilities
from vllm_mlx.runtime.resident_models import estimate_model_bytes

ALIAS = "qwen-image-edit"
REPO = "OsaurusAI/Qwen-Image-Edit-mflux-q8"
REVISION = "a458969f2a612433cf036bfc3d8d818ceba29fab"
SIZE = 37_472_689_129
_GIB = 1024**3


def test_alias_is_edit_only_and_uses_the_quality_quantization() -> None:
    profile = resolve_profile(ALIAS)
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 96

    engine = ImageGenerationEngine(profile.hf_path)
    assert engine.family == "qwen-image-edit"
    assert engine.supports_generation is False
    assert engine.supports_editing is True
    assert engine.default_edit_steps == 20
    assert engine.default_edit_guidance == 4.0
    assert engine._prequantized is True  # noqa: SLF001
    assert engine._quantize is None  # noqa: SLF001


def test_alias_pins_download_size_revision_and_conservative_budget() -> None:
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == SIZE
    for name in (ALIAS, REPO, "/models/qwen_image_edit"):
        assert estimate_model_bytes(name) == int(68.0 * _GIB)


def test_atomic_catalog_exposes_only_image_to_image() -> None:
    bundle = build_catalog_bundle()
    record = next(
        alias for alias in bundle["snapshot"]["aliases"] if alias["alias"] == ALIAS
    )
    capabilities = record["capabilities"]
    assert capabilities["runtime_adapter"] == "mflux"
    assert capabilities["operation_modes"] == ["image_to_image"]


def test_openai_model_card_advertises_editing_not_generation() -> None:
    for model_id in (ALIAS, REPO):
        capabilities = _detect_capabilities(
            model_id,
            profile_modality="image-gen",
        )
        assert capabilities == ["image.editing"]

    assert _detect_capabilities(
        "acme/qwen-image-edit-plus",
        profile_modality="image-gen",
    ) == ["image.generation"]


def test_cold_load_fetches_the_pinned_revision(monkeypatch) -> None:
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda _repo: None
    )
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, **kwargs: calls.append((repo_id, kwargs)) or "/pinned/model",
    )

    assert engine._model_path_for_mflux() == "/pinned/model"  # noqa: SLF001
    assert calls == [(REPO, {"revision": REVISION})]
