# SPDX-License-Identifier: Apache-2.0
"""Repeated setup must preserve user catalogs through the actual adapter."""

import json

import pytest

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.adapter import _merge_json, setup_agent_config


@pytest.fixture(autouse=True)
def isolated_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "GOOSE_PATH_ROOT",
        "XDG_CONFIG_HOME",
        "CRUSH_GLOBAL_CONFIG",
        "RAPID_MLX_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.track_agent_configured", lambda *_: None
    )
    monkeypatch.setattr(
        "rapid_mlx.agents.adapter.track_agent_configure_failed", lambda *_: None
    )


@pytest.mark.parametrize("keyed", [False, True])
@pytest.mark.parametrize("selected", ["selected", "new-model"])
def test_goose_repeated_setup_preserves_user_catalog(
    tmp_path, monkeypatch, keyed, selected
):
    profile = get_profile("goose")
    setup_agent_config(
        profile, "http://old.invalid/v1", "selected", context_length=1024
    )
    path = tmp_path / ".config/goose/custom_providers/custom_rapid_mlx.json"
    existing = json.loads(path.read_text())
    existing["models"][0]["user_note"] = "keep selected metadata"
    second = {"name": "user-model", "context_limit": 8192, "user_note": "keep second"}
    existing["models"].append(second)
    existing["headers"] = {"X-User-Header": "fake-header-credential"}
    existing["credentials"] = {"custom_token": "fake-provider-credential"}
    path.write_text(json.dumps(existing))
    custom_provider = path.with_name("custom_user.json")
    custom_bytes = b'{"name":"custom_user","api_key_env":"USER_API_KEY"}\n'
    custom_provider.write_bytes(custom_bytes)
    credentials = path.parent.parent / "config.yaml"
    credentials_bytes = (
        b"USER_API_KEY: fake-user-credential\nGOOSE_PROVIDER: custom_user\n"
    )
    credentials.write_bytes(credentials_bytes)
    if keyed:
        monkeypatch.setenv("RAPID_MLX_API_KEY", "fake-rapid-secret")

    before = path.read_bytes()
    summary = setup_agent_config(
        profile, "http://new.invalid/v1", selected, context_length=65536, dry_run=True
    )
    assert "nothing written" in summary
    assert path.read_bytes() == before
    summary = setup_agent_config(
        profile, "http://new.invalid/v1", selected, context_length=65536
    )
    assert not summary.startswith("Cannot"), summary
    data = json.loads(path.read_text())
    expected_names = ["selected", "user-model"]
    if selected == "new-model":
        expected_names.append(selected)
    assert [model["name"] for model in data["models"]] == expected_names
    assert data["models"][1] == second
    assert data["models"][0]["user_note"] == "keep selected metadata"
    chosen = next(model for model in data["models"] if model["name"] == selected)
    assert chosen["context_limit"] == 65536
    if selected == "new-model":
        assert data["models"][0]["context_limit"] == 1024
    assert data["base_url"] == "http://new.invalid/v1/chat/completions"
    assert data["requires_auth"] is keyed
    assert data["api_key_env"] == ("RAPID_MLX_API_KEY" if keyed else "")
    assert data["headers"] == existing["headers"]
    assert data["credentials"] == existing["credentials"]
    assert "fake-rapid-secret" not in path.read_text()
    assert custom_provider.read_bytes() == custom_bytes
    assert credentials.read_bytes() == credentials_bytes

    after = path.read_bytes()
    assert "Already configured" in setup_agent_config(
        profile, "http://new.invalid/v1", selected, context_length=65536
    )
    assert path.read_bytes() == after


def test_crush_setup_still_merges_models_by_id(tmp_path):
    profile = get_profile("crush")
    setup_agent_config(
        profile, "http://old.invalid/v1", "selected", context_length=1024
    )
    path = tmp_path / ".config/crush/crush.json"
    existing = json.loads(path.read_text())
    provider = existing["providers"]["rapid-mlx"]
    provider["models"][0]["user_note"] = "keep"
    second = {
        "id": "user-model",
        "name": "Selected display name",
        "context_window": 8192,
    }
    provider["models"].append(second)
    provider["headers"] = {"X-User-Header": "fake-header"}
    existing["providers"]["custom"] = {"api_key": "fake-user-key"}
    path.write_text(json.dumps(existing))

    setup_agent_config(
        profile, "http://new.invalid/v1", "selected", context_length=65536
    )
    data = json.loads(path.read_text())
    updated = data["providers"]["rapid-mlx"]
    assert [model["id"] for model in updated["models"]] == ["selected", "user-model"]
    assert updated["models"][0]["context_window"] == 65536
    assert updated["models"][0]["user_note"] == "keep"
    assert updated["models"][1] == second
    assert updated["headers"] == provider["headers"]
    assert data["providers"]["custom"] == existing["providers"]["custom"]


def test_other_agents_name_keyed_lists_still_replace():
    existing = {"models": [{"name": "user"}], "tags": ["user"]}
    incoming = {"models": [{"name": "selected"}], "tags": ["configured"]}
    assert (
        json.loads(_merge_json(json.dumps(existing), json.dumps(incoming))) == incoming
    )
