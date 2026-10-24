# SPDX-License-Identifier: Apache-2.0
"""Shipped identity locks preserve explicit snapshots without downloading weights."""

import json
from pathlib import Path

import pytest

from rapid_mlx.model_catalog import locked_repositories, resolve_repository_identity
from scripts.catalog_trust import catalog_changes, is_trust_control

ROOT = Path(__file__).resolve().parents[1]


def test_all_shipped_primary_and_nested_repositories_have_matching_pins():
    locked = locked_repositories()
    for relative, field in [
        ("aliases.json", "hf_path"),
        ("audio/aliases.json", "hf_id"),
    ]:
        catalog = json.loads((ROOT / "rapid_mlx" / relative).read_text())
        for name, entry in catalog.items():
            if name.startswith("_"):
                continue
            assert entry["revision"] == locked[entry[field]]["revision"], name
            assert resolve_repository_identity(name) == (
                entry[field],
                entry["revision"],
            )
        for assets in catalog.get("_runtime_assets", {}).values():
            for asset in assets:
                assert asset["revision"] == locked[asset["repo_id"]]["revision"]


def test_auxiliary_default_and_existing_image_pins_are_preserved():
    from rapid_mlx._download_gate import IMAGE_MODEL_REVISIONS
    from rapid_mlx.video.wan import WAN_REVISIONS

    locked = locked_repositories()
    for repo, revision in {**IMAGE_MODEL_REVISIONS, **WAN_REVISIONS}.items():
        assert locked[repo]["revision"] == revision
    for repo in ["t5-base", "lucasnewman/vocos-mel-24khz", "openai/whisper-large-v3"]:
        assert resolve_repository_identity(repo) == (repo, locked[repo]["revision"])


def test_metadata_evidence_records_executable_declarations_without_approval():
    path = ROOT / "rapid_mlx/model_catalog_lock.json"
    manifest = json.loads(path.read_text())
    assert not manifest["errors"]
    assert "not model-quality" in manifest["scope"]
    kimi = manifest["repositories"]["mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8"]
    assert kimi["configs"]["tokenizer_config.json"]["auto_map"]
    assert len(kimi["configs"]["config.json"]["sha256"]) == 64
    assert kimi["lfs_files"]
    assert len(kimi["git_files"]["config.json"]["blob_id"]) == 40


def test_local_directory_wins_over_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kokoro").mkdir()
    assert resolve_repository_identity("kokoro") == ("kokoro", None)


@pytest.mark.parametrize("revision", [None, "main", "short"])
def test_new_audio_and_runtime_asset_sources_require_full_pin(revision):
    entry = {"hf_id": "org/model", "revision": revision}
    assert catalog_changes({}, {"audio": entry}, "rapid_mlx/audio/aliases.json")[1]
    assets = {
        "_runtime_assets": {"family": [{"repo_id": "org/asset", "revision": revision}]}
    }
    assert catalog_changes({}, assets, "rapid_mlx/audio/aliases.json")[1]


def test_pinned_audio_and_runtime_asset_sources_still_require_review():
    pin = "a" * 40
    after = {
        "audio": {"hf_id": "org/model", "revision": pin},
        "_runtime_assets": {"family": [{"repo_id": "org/asset", "revision": pin}]},
    }
    changed, errors = catalog_changes({}, after, "rapid_mlx/audio/aliases.json")
    assert set(changed) == {"audio", "_runtime_assets"}
    assert not errors


@pytest.mark.parametrize(
    "path",
    [
        "rapid_mlx/model_catalog.py",
        "rapid_mlx/model_catalog_lock.json",
        "scripts/lock_model_catalog.py",
        "rapid_mlx/model_integrity.py",
        "rapid_mlx/server_exposure.py",
        "rapid_mlx/audio/f5_security.py",
        "rapid_mlx/audio/voxcpm_security.py",
    ],
)
def test_new_security_controls_require_review(path):
    assert is_trust_control(path)


def test_stray_directory_does_not_shadow_registered_text_alias(tmp_path, monkeypatch):
    from rapid_mlx.model_aliases import resolve_hub_model, resolve_revision

    name = "lfm2.5-2.6b-4bit"
    expected = (resolve_hub_model(name), resolve_revision(name))
    monkeypatch.chdir(tmp_path)
    (tmp_path / name).mkdir()
    assert resolve_repository_identity(name) == expected


@pytest.mark.parametrize(
    "lock,error",
    [
        ({"schema_version": 2, "repositories": {}}, "incomplete"),
        (
            {"schema_version": 1, "errors": ["unresolved repo"], "repositories": {}},
            "incomplete",
        ),
        (
            {"schema_version": 1, "repositories": {"org/model": {"revision": "main"}}},
            "Invalid locked",
        ),
    ],
)
def test_invalid_identity_lock_fails_closed(monkeypatch, lock, error):
    monkeypatch.setattr("rapid_mlx.model_catalog.json.load", lambda handle: lock)
    # Exercise parsing independently of the process's already loaded valid lock.
    with pytest.raises(ValueError, match=error):
        locked_repositories.__wrapped__()


def test_audio_identity_disagreement_is_rejected(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "rapid_mlx.audio.registry.resolve_audio_alias",
        lambda name: SimpleNamespace(hf_id="org/audio", revision="a" * 40),
    )
    monkeypatch.setattr(
        "rapid_mlx.model_catalog.locked_repositories",
        lambda: {"org/audio": {"revision": "b" * 40}},
    )
    with pytest.raises(ValueError, match="Audio catalog and identity lock disagree"):
        resolve_repository_identity("org/audio")
