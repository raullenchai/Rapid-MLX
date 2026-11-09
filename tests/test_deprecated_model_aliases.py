# SPDX-License-Identifier: Apache-2.0
"""Repo execution profiles survive the removal of curated short names."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pytest

from rapid_mlx import model_aliases as registry

ROOT = Path(__file__).resolve().parents[1]
ROWS = json.loads((ROOT / "rapid_mlx/repo_profiles.json").read_text())
ROWS.pop("_comment")


@pytest.fixture(autouse=True)
def isolated_user_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(tmp_path / "aliases.json"))
    monkeypatch.setattr(registry, "external_model_root_paths", lambda: ())


@pytest.mark.parametrize("name,row", ROWS.items())
def test_deprecated_name_warns_once_and_keeps_repo_profile(name, row, caplog):
    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        assert registry.resolve_model(name) == row["hf_path"]
    assert [r.getMessage() for r in caplog.records] == [
        f"Model alias '{name}' is deprecated and will be removed in the next release; "
        f"use 'rapid-mlx serve {row['hf_path']}' instead."
    ]
    caplog.clear()
    assert registry.resolve_model(row["hf_path"]) == row["hf_path"]
    assert not caplog.records
    assert asdict(registry.resolve_profile(row["hf_path"])) == asdict(
        registry._coerce(name, row)
    )
    assert registry.resolve_profile(name) == registry.resolve_profile(row["hf_path"])
    assert registry.resolve_revision(row["hf_path"]) == row.get("revision")


def test_names_are_not_public():
    from rapid_mlx.catalog.legacy import build_legacy_catalog_snapshot

    snapshot = build_legacy_catalog_snapshot()
    assert not ROWS.keys() & {row["alias"] for row in snapshot["aliases"]}
    from rapid_mlx._completion import _load_alias_names

    assert not ROWS.keys() & set(_load_alias_names())
    assert len(ROWS) == 44
    assert {
        k: v["hf_path"] for k, v in ROWS.items()
    } == registry._DEPRECATED_MODEL_ALIASES
    for listing in (
        registry.list_aliases(),
        registry.list_builtin_aliases(),
        registry.list_profiles(),
    ):
        assert not ROWS.keys() & listing.keys()
    # Reservations prevent user mappings from reintroducing these names.
    assert ROWS.keys() <= registry.user_alias_reserved_names()
    for name in ROWS:
        assert registry.catalog_alias_for(name) is None
        assert name not in registry.suggest_similar(name, n=1000, cutoff=0)
    assert not ROWS.keys() & set(registry.POPULAR_ALIASES)


@pytest.mark.parametrize("name,row", ROWS.items())
def test_raw_repo_telemetry_identity_is_stable(name, row):
    from rapid_mlx.telemetry.model_id import telemetry_model_id

    assert telemetry_model_id(row["hf_path"]) == name
    assert registry.catalog_alias_for(row["hf_path"]) is None


def test_curated_repo_profile_wins(monkeypatch):
    row = next(iter(ROWS.values()))
    curated = registry.AliasProfile(hf_path=row["hf_path"], tool_call_parser="llama")
    monkeypatch.setattr(registry, "_aliases", {"retained": curated})
    monkeypatch.setattr(registry, "_hf_to_alias", None)
    assert registry.resolve_profile(row["hf_path"].upper()) == curated
    assert registry.resolve_revision(row["hf_path"]) is None
    assert registry.resolve_model(row["hf_path"].upper()) == curated.hf_path


def test_repo_profile_schema_and_key_separation():
    assert not ROWS.keys() & registry._load().keys()
    parsed = {key: registry._coerce(key, value) for key, value in ROWS.items()}
    registry._assert_subfolder_is_unambiguous(parsed)
    registry._assert_revision_is_unambiguous(parsed)
    assert parsed == registry._load_repo_profiles()


@pytest.mark.parametrize("name,row", ROWS.items())
def test_revision_fallback_keeps_repo_pin(name, row, monkeypatch):
    def unavailable(_):
        raise RuntimeError("optional profile metadata unavailable")

    monkeypatch.setattr(registry, "resolve_profile", unavailable)
    assert registry.resolve_revision(row["hf_path"]) == row.get("revision")


def test_user_alias_to_repo_inherits_execution_metadata(tmp_path, monkeypatch):
    from rapid_mlx import user_aliases

    name, row = next(iter(ROWS.items()))
    user_aliases.set_user_alias(
        "my-checkpoint", row["hf_path"], registry.list_aliases()
    )
    expected = registry._coerce(name, row)
    assert registry.resolve_profile("my-checkpoint") == expected
    assert registry.list_profiles()["my-checkpoint"] == expected


def test_deprecated_local_directory_keeps_path_precedence(
    tmp_path, monkeypatch, caplog
):
    name = next(iter(ROWS))
    (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)
    assert registry.resolve_model(name) == name
    assert not caplog.records


@pytest.mark.parametrize("filename", ["aliases.json", "repo_profiles.json"])
def test_registry_keys_are_unique(filename):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"duplicate key {key} in {filename}"
            result[key] = value
        return result

    json.loads((ROOT / "rapid_mlx" / filename).read_text(), object_pairs_hook=unique)


def test_decode_projection_keeps_evidence_under_repo_ids():
    from scripts.refresh_decode_expectations import rekey_uncurated_models

    snapshot = json.loads((ROOT / "rapid_mlx/decode_expectations.json").read_text())
    assert not ROWS.keys() & snapshot["models"].keys()
    for row in ROWS.values():
        if row.get("modality", "text") in ("text", "vision"):
            assert row["hf_path"] in snapshot["models"]
    name, row = next(iter(ROWS.items()))
    evidence = {"repo_id": row["hf_path"], "cohorts": [{"median": 12.5}]}
    original = {"refreshed_at": "2026-11-07", "models": {name: evidence}}
    assert rekey_uncurated_models(original) == {
        "refreshed_at": "2026-11-07",
        "models": {row["hf_path"]: evidence},
    }
    assert rekey_uncurated_models(snapshot) == snapshot


@pytest.mark.parametrize("row", ROWS.values())
def test_lane_diagnostics_keep_raw_repo_execution_profile(row):
    from rapid_mlx.api.utils import _catalog_profile

    assert _catalog_profile(row["hf_path"]) == registry.resolve_profile(row["hf_path"])


@pytest.mark.parametrize("name,row", ROWS.items())
def test_existing_user_alias_target_does_not_break_raw_resolution(name, row):
    from rapid_mlx import user_aliases

    user_aliases.set_user_alias("my-legacy-model", name, registry.list_aliases())
    before = user_aliases.config_path().read_bytes()
    assert registry.resolve_model(row["hf_path"]) == row["hf_path"]
    assert registry.resolve_model("my-legacy-model") == row["hf_path"]
    assert registry.resolve_profile("my-legacy-model") == registry._coerce(name, row)
    assert registry.list_aliases()["my-legacy-model"] == row["hf_path"]
    assert user_aliases.config_path().read_bytes() == before


def _pinned_repo_only_row():
    curated = {p.hf_path.casefold() for p in registry.list_profiles().values()}
    return next(
        row
        for row in ROWS.values()
        if row.get("revision") and row["hf_path"].casefold() not in curated
    )


def test_repo_profile_keys_must_not_overlap_curated_aliases(monkeypatch):
    name = next(iter(ROWS))
    monkeypatch.setattr(registry, "_repo_profiles", None)
    monkeypatch.setattr(registry, "_load", lambda: {name: object()})
    with pytest.raises(ValueError, match="provenance keys overlap"):
        registry._load_repo_profiles()


def test_cached_repository_launch_path_uses_repo_profile_pin(monkeypatch):
    import huggingface_hub

    row = _pinned_repo_only_row()
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, filename, revision: calls.append((repo, filename, revision)),
    )
    assert (
        registry.cached_repository_launch_path(row["hf_path"], external=False) is None
    )
    prefix = f"{row['subfolder']}/" if row.get("subfolder") else ""
    assert calls == [(row["hf_path"], f"{prefix}config.json", row["revision"])]


def test_served_catalog_name_maps_retired_name_to_repo():
    from rapid_mlx.api.utils import served_model_catalog_name
    from rapid_mlx.config import reset_config

    name, row = next(iter(ROWS.items()))
    cfg = reset_config()
    try:
        engine = object()
        cfg.engine = engine
        cfg.model_alias = name
        assert served_model_catalog_name(engine) == row["hf_path"]
    finally:
        reset_config()


def test_dflash_registry_gate_consults_repo_profile_and_fails_closed():
    from rapid_mlx.speculative.dflash.eligibility import is_registry_verified_pair

    row = _pinned_repo_only_row()
    assert not is_registry_verified_pair(
        row["hf_path"], row["revision"], "org/drafter", "0" * 40, "dflash"
    )
