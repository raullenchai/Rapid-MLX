# SPDX-License-Identifier: Apache-2.0
"""Curated Hub repo ids must retain the same serving profile as aliases."""

from __future__ import annotations

import pytest

from rapid_mlx import model_aliases
from rapid_mlx.audio.registry import list_audio_aliases, resolve_audio_alias
from rapid_mlx.model_aliases import list_builtin_aliases, resolve_model, resolve_profile
from rapid_mlx.telemetry.model_events import model_type


def _non_llm_catalog_cases() -> list[tuple[str, str, str]]:
    cases: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for alias, hf_path in list_builtin_aliases().items():
        key = hf_path.lower()
        if key in seen:
            continue
        seen.add(key)
        lane = model_type(alias)
        if lane in {
            "image-gen",
            "video-gen",
            "audio",
            "vlm",
            "embedding",
            "text-diffusion",
        }:
            cases.append((alias, hf_path, lane))
    return cases


@pytest.mark.parametrize(
    ("alias", "hf_path", "lane"),
    _non_llm_catalog_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_curated_repo_id_routes_like_first_catalog_alias(alias, hf_path, lane):
    """The reverse index's documented first-alias winner owns a raw repo id."""
    mixed_case = hf_path.swapcase()
    alias_profile = resolve_profile(alias)
    repo_profile = resolve_profile(mixed_case)

    assert alias_profile is not None
    assert repo_profile == alias_profile
    assert model_type(mixed_case) == lane
    assert resolve_model(mixed_case) == hf_path


def test_curated_repo_id_resolves_case_insensitively_from_cold_registry(monkeypatch):
    hf_path = "mlx-community/Qwen3.5-9B-4bit"

    monkeypatch.setattr(model_aliases, "_aliases", None)
    monkeypatch.setattr(model_aliases, "_hf_to_alias", None)

    assert resolve_model(hf_path.swapcase()) == hf_path


def test_curated_repo_id_rebuilds_cleared_reverse_index(monkeypatch):
    hf_path = "mlx-community/Qwen3.5-9B-4bit"
    profiles = model_aliases._load()

    monkeypatch.setattr(model_aliases, "_aliases", profiles)
    monkeypatch.setattr(model_aliases, "_hf_to_alias", None)

    assert resolve_model(hf_path.swapcase()) == hf_path


@pytest.mark.parametrize("entry", list_audio_aliases(), ids=lambda entry: entry.alias)
def test_audio_repo_id_routes_like_alias(entry):
    resolved = resolve_audio_alias(entry.hf_id.swapcase())

    assert resolved is not None
    assert resolved.type == entry.type
    assert resolved.hf_id == entry.hf_id
    assert model_type(entry.hf_id.swapcase()) == model_type(entry.alias) == "audio"


def test_shared_repo_uses_first_alias_in_catalog_order():
    repo = "mlx-community/Qwen3.6-35B-A3B-4bit"

    assert list_builtin_aliases()["qwen3.6-35b-4bit"] == repo
    assert resolve_profile(repo) == resolve_profile("qwen3.6-35b-4bit")
    assert resolve_profile(repo) != resolve_profile("qwen3.6-35b")
