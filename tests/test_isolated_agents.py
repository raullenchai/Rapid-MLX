"""Day-two configurations stay private and never rewrite existing agent state."""

import json
import stat
from pathlib import Path

import pytest
import tomllib

from rapid_mlx.agents import isolated


@pytest.fixture
def private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_stable_key_is_reused_and_owner_only(private_home):
    first = isolated.stable_key()
    assert len(first) == 64
    assert isolated.stable_key() == first
    assert stat.S_IMODE(isolated.key_path().stat().st_mode) == 0o600
    assert stat.S_IMODE(isolated.connection_dir().stat().st_mode) == 0o700


def test_profiles_are_isolated_idempotent_and_removable(private_home):
    existing = private_home / ".codex" / "config.toml"
    existing.parent.mkdir()
    existing.write_text('model = "my-remote-model"\n')
    for name in sorted(isolated.SUPPORTED):
        command = isolated.configure(name, "qwen3.5-9b-4bit")
        assert name.split("-")[0] in command
        root = isolated.connection_dir() / "agents" / name
        before = {
            p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
        assert isolated.configure(name, "qwen3.5-9b-4bit") == command
        assert before == {
            p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
        assert all(
            stat.S_IMODE(p.stat().st_mode) == (0o700 if p.name == "run" else 0o600)
            for p in root.rglob("*")
            if p.is_file()
        )
        assert isolated.remove(name)
        assert not isolated.remove(name)
    assert existing.read_text() == 'model = "my-remote-model"\n'


def test_codex_profile_and_claude_settings(private_home):
    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    root = isolated.connection_dir() / "agents"
    codex = tomllib.loads((root / "codex" / "config.toml").read_text())
    assert codex["profiles"]["rapid"]["model_provider"] == "rapid-mlx"
    assert (
        codex["model_providers"]["rapid-mlx"]["base_url"] == "http://127.0.0.1:7660/v1"
    )
    isolated.configure("claude-code", "qwen3.5-9b-4bit", port=7660)
    claude = json.loads((root / "claude-code" / "settings.json").read_text())
    assert claude["env"]["ANTHROPIC_API_KEY"] == isolated.stable_key()
    assert claude["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:7660"


def test_saved_port_is_reused_by_later_profiles(private_home):
    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    assert isolated.stable_port() == 7660
    command = isolated.configure("qwen-code", "qwen3.5-9b-4bit")
    assert command.endswith("/qwen-code/run")
    assert (
        "127.0.0.1:7660/v1"
        in (isolated.connection_dir() / "agents" / "qwen-code" / "run").read_text()
    )
    hermes = isolated.configure("hermes", "qwen3.5-9b-4bit")
    assert hermes.endswith("/hermes/run")
    assert (
        "-p rapid"
        in (isolated.connection_dir() / "agents" / "hermes" / "run").read_text()
    )
    assert (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).is_file()
