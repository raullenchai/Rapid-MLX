from unittest.mock import AsyncMock

import pytest

from rapid_mlx.model_aliases import catalog_alias_for, catalog_identity, list_profiles
from rapid_mlx.recommendations import recommendation_footprint_gb
from rapid_mlx.runtime.model_registry import ModelRegistry
from rapid_mlx.runtime.resident_models import (
    ResidentModelCapacityError,
    ResidentModelManager,
    estimate_model_bytes,
    measured_image_estimate_bytes,
)

GIB = 1024**3


def test_catalog_identity_returns_none_when_catalog_lookup_raises(monkeypatch):
    def fail(name):
        raise RuntimeError("catalog lookup failed")

    monkeypatch.setattr("rapid_mlx.model_aliases.catalog_alias_for", fail)

    assert catalog_identity("anything") is None


@pytest.mark.parametrize("model_name", ["qwen-image", "unknown/image-model"])
def test_measured_image_estimate_uses_hf_path_or_raw_spelling(monkeypatch, model_name):
    alias = "qwen-image"
    hf_path = list_profiles()[alias].hf_path
    sentinel_bytes = 123456789
    estimates = {
        alias: None,
        hf_path: sentinel_bytes,
        "unknown/image-model": sentinel_bytes,
    }
    monkeypatch.setattr(
        "rapid_mlx.runtime.resident_models._measured_image_estimate_for_spelling",
        estimates.__getitem__,
    )

    assert measured_image_estimate_bytes(model_name) == sentinel_bytes


@pytest.mark.parametrize(
    "alias,profile",
    [
        (alias, profile)
        for alias, profile in list_profiles().items()
        if catalog_alias_for(alias) == alias
    ],
)
def test_catalog_spellings_have_equal_estimates(alias, profile):
    org, name = profile.hf_path.split("/", 1)
    canonical = catalog_alias_for(profile.hf_path)
    snapshot = f"/x/hub/models--{org}--{name}/snapshots/{'a' * 40}"
    spellings = (
        profile.hf_path,
        profile.hf_path.upper(),
        snapshot,
        f"{snapshot}/4bit",
        canonical,
    )
    assert len({estimate_model_bytes(s) for s in spellings}) == 1
    assert len({recommendation_footprint_gb(s) for s in spellings}) == 1
    if canonical == alias:
        assert estimate_model_bytes(alias) == estimate_model_bytes(canonical)
        assert recommendation_footprint_gb(alias) == recommendation_footprint_gb(
            canonical
        )


def test_user_alias_uses_catalog_estimates_and_unknown_keeps_heuristic(monkeypatch):
    monkeypatch.setattr(
        "rapid_mlx.user_aliases.load_user_aliases",
        lambda: {"my-bonsai": "bonsai-27b-2bit"},
    )
    assert estimate_model_bytes("my-bonsai") == estimate_model_bytes("bonsai-27b-2bit")
    assert recommendation_footprint_gb("my-bonsai") == 13
    repo = "unknown/opaque-7b-4bit"
    path = f"/x/hub/models--unknown--opaque-7b-4bit/snapshots/{'a' * 40}"
    assert estimate_model_bytes(repo) == estimate_model_bytes(path) == int(7.55 * GIB)
    assert recommendation_footprint_gb(repo) is None


@pytest.mark.asyncio
async def test_flux_schnell_repo_and_alias_both_rejected_by_admission():
    loader = AsyncMock()
    manager = ResidentModelManager(
        ModelRegistry(), loader, memory_limit_bytes=16 * GIB, memory_reader=lambda: 0
    )
    for name in ("flux-schnell", "mflux-community/flux-1-schnell-mflux-q4"):
        with pytest.raises(ResidentModelCapacityError):
            await manager.load(name, estimated_bytes=None)
    loader.assert_not_awaited()
