from unittest.mock import AsyncMock

import pytest

from rapid_mlx.model_aliases import catalog_alias_for, list_profiles
from rapid_mlx.recommendations import recommendation_footprint_gb
from rapid_mlx.runtime.model_registry import ModelRegistry
from rapid_mlx.runtime.resident_models import (
    ResidentModelCapacityError,
    ResidentModelManager,
    estimate_model_bytes,
)

GIB = 1024**3


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
