"""Day-two configurations stay private and never rewrite existing agent state."""

import json
import stat
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import patch

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
    codex = tomllib.loads((root / "codex" / "rapid.config.toml").read_text())
    assert codex["model_provider"] == "rapid-mlx"
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


def test_every_runner_refreshes_private_endpoint_after_port_change(private_home):
    for name in sorted(isolated.SUPPORTED):
        isolated.configure(name, "qwen3.5-9b-4bit", port=7660)
        runner = (isolated.connection_dir() / "agents" / name / "run").read_text()
        assert f"launch {name} --isolated --check" in runner
        assert f"launch {name} --isolated --model qwen3.5-9b-4bit" in runner


def test_old_opencode_shape_and_hermes_auth(private_home):
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="1.0.0"
    ):
        isolated.configure("opencode", "qwen3.5-9b-4bit")
    config = json.loads(
        (
            isolated.connection_dir() / "agents" / "opencode" / "opencode.json"
        ).read_text()
    )
    assert "providers" not in config
    isolated.configure("hermes", "qwen3.5-9b-4bit")
    hermes = (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).read_text()
    assert f'api_key: "{isolated.stable_key()}"' in hermes
    goose = isolated.configure("goose", "qwen3.5-9b-4bit")
    assert "goose session --provider" in Path(goose).read_text()


def test_isolated_dry_run_preserves_saved_profile(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    isolated.configure("codex", "qwen3.5-9b-4bit")
    args = Namespace(
        client="codex", isolated=False, remove=True, all=False, dry_run=True, run=False
    )
    launch_command(args)
    assert (
        isolated.connection_dir() / "agents" / "codex" / "rapid.config.toml"
    ).exists()
    assert "[dry-run]" in capsys.readouterr().out


def test_connection_requires_exact_advertised_model(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"qwen3.5-9b-4bit-WRONG"}]}'

    with patch("urllib.request.urlopen", return_value=Response()) as request:
        assert (
            isolated.connection_status(7659, "a" * 64, "qwen3.5-9b-4bit")
            == "model-mismatch"
        )
        assert (
            request.call_args.args[0].get_header("Authorization")
            == "Bearer " + "a" * 64
        )


def test_launch_waits_through_startup_http_error(private_home):
    from rapid_mlx.launch.cli import launch_command, register

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    args = parser.parse_args(
        [
            "launch",
            "codex",
            "--isolated",
            "--model",
            "qwen3.5-9b-4bit",
            "--run",
            "--start-server",
        ]
    )
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "server-error", "ready"],
        ),
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=0),
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.launch.cli.subprocess.call", return_value=0) as agent,
        patch("time.sleep"),
        pytest.raises(SystemExit) as exit_info,
    ):
        launch_command(args)
    assert exit_info.value.code == 0
    start.assert_called_once()
    agent.assert_called_once()
