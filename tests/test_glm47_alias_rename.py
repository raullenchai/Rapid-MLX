# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7 Flash has one honest catalog name and a compatible legacy name."""

import logging
import sys

import pytest

from rapid_mlx import model_aliases as registry

OLD = "glm4.7-9b-4bit"
NEW = "glm4.7-flash-4bit"
REPO = "mlx-community/GLM-4.7-Flash-4bit"


@pytest.fixture(autouse=True)
def isolated_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(tmp_path / "aliases.json"))
    monkeypatch.setattr(registry, "external_model_root_paths", lambda: ())


def test_rename_notice_and_profile(caplog):
    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        assert registry.resolve_model(OLD) == REPO
    assert [r.getMessage() for r in caplog.records] == [
        f"Model alias '{OLD}' was renamed to '{NEW}'; use the new name."
    ]
    caplog.clear()
    assert registry.resolve_model(NEW) == REPO
    assert not caplog.records
    profile = registry.resolve_profile(NEW)
    assert profile is not None and profile.is_moe is True
    assert registry.resolve_profile(OLD) == profile
    assert registry.resolve_profile(REPO) == profile
    assert registry.resolve_revision(OLD) == profile.revision
    assert OLD in registry.user_alias_reserved_names()


def test_catalog_lists_checkpoint_once(monkeypatch):
    from rapid_mlx import cli

    monkeypatch.setattr("rapid_mlx.decode_expectations.detected_chip", lambda: None)
    payload = cli._available_models_json_payload()
    rows = [row for row in payload["text"] if row["hf_path"] == REPO]
    assert len(rows) == 1
    assert rows[0]["alias"] == NEW
    assert rows[0]["is_moe"] is True
    assert rows[0]["size_bytes"] == 16872844852
    for listing in (registry.list_aliases(), registry.list_profiles()):
        assert OLD not in listing
    assert OLD not in registry.suggest_similar(OLD, n=1000, cutoff=0)


@pytest.mark.parametrize("name", [OLD, NEW, REPO])
def test_served_metadata(name, monkeypatch):
    from rapid_mlx.routes.models import _build_model_info

    monkeypatch.setitem(sys.modules, "rapid_mlx.server", None)
    assert _build_model_info(name).is_moe is True


@pytest.mark.parametrize("name", [OLD, NEW])
def test_served_models_endpoint(name, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import get_config
    from rapid_mlx.routes.models import router

    monkeypatch.setitem(sys.modules, "rapid_mlx.server", None)
    cfg = get_config()
    monkeypatch.setattr(cfg, "model_name", REPO)
    monkeypatch.setattr(cfg, "model_alias", name)
    monkeypatch.setattr(cfg, "model_registry", None)
    monkeypatch.setattr(cfg, "api_key", None)
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/v1/models")
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()["data"]}
    assert rows[name]["is_moe"] is True
    assert rows[REPO]["is_moe"] is True


def test_existing_user_alias_target_remains_valid():
    from rapid_mlx import user_aliases

    user_aliases.set_user_alias("my-glm", OLD, registry.list_aliases())
    assert registry.resolve_model("my-glm") == REPO
    assert registry.resolve_profile("my-glm") == registry.resolve_profile(NEW)


def test_existing_cache_is_shared(tmp_path, monkeypatch):
    import huggingface_hub

    config = tmp_path / "snapshot" / "config.json"
    config.parent.mkdir()
    config.write_text("{}")
    (config.parent / "model.safetensors").write_bytes(b"weights")
    (config.parent / "tokenizer.json").write_text("{}")
    calls = []

    def cached(repo, filename, revision):
        calls.append((repo, filename, revision))
        return str(config)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    for name in (OLD, NEW):
        repo = registry.resolve_model(name, include_external_roots=False)
        assert registry.cached_repository_launch_path(repo, external=False) == str(
            config.parent
        )
    assert calls == [(REPO, "config.json", registry.resolve_revision(NEW))] * 2


def test_decode_projection_preserves_renamed_evidence():
    from scripts.refresh_decode_expectations import rekey_uncurated_models

    evidence = {"repo_id": REPO, "cohorts": [{"median": 12.5}]}
    snapshot = {"models": {OLD: evidence}, "refreshed_at": "2026-11-07"}
    assert rekey_uncurated_models(snapshot) == {**snapshot, "models": {NEW: evidence}}
