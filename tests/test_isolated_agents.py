"""Day-two configurations stay private and never rewrite existing agent state."""

import json
import os
import shutil
import signal
import stat
import subprocess
import urllib.error
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

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
    qwen_run = (isolated.connection_dir() / "agents" / "qwen-code" / "run").read_text()
    assert f"$(cat {isolated.port_path()})" in qwen_run
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


def test_malformed_model_list_returns_one_server_error(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":null}'

    with patch("urllib.request.urlopen", return_value=Response()):
        assert isolated.connection_status(7659, "a" * 64, "model") == "server-error"


def test_connection_status_shares_local_refusal_classification(private_home):
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
        assert isolated.connection_status(7659, "a" * 64, "model") == "stopped"
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        assert isolated.connection_status(7659, "a" * 64, "model") == "server-error"


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


def test_private_storage_rejects_links_and_unsafe_files(private_home):
    target = private_home / "other"
    target.mkdir()
    directory = isolated.connection_dir()
    directory.parent.mkdir()
    directory.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated._private_dir(directory)
    with pytest.raises(ValueError, match="symlink"):
        isolated.stable_key()
    directory.unlink()
    directory.mkdir()
    (directory / "api-key").symlink_to(target / "key")
    with pytest.raises(ValueError, match="symlink"):
        isolated.stable_key()
    (directory / "api-key").unlink()
    key = directory / "api-key"
    key.write_text("bad\n")
    key.chmod(0o600)
    with pytest.raises(ValueError, match="invalid"):
        isolated.stable_key()
    key.write_text("a" * 64)
    key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        isolated.stable_key()
    key.unlink()
    assert isolated.stable_key(create=False) is None
    (directory / "port").write_text("0\n")
    with pytest.raises(ValueError, match="1..65535"):
        isolated.stable_port()
    with pytest.raises(ValueError, match="1..65535"):
        isolated.save_port(0)
    (directory / "port").unlink()
    (directory / "port").symlink_to(target / "missing-port")
    with pytest.raises(ValueError, match="symlink"):
        isolated.stable_port()


def test_private_write_rejects_file_link_and_closes_on_failure(private_home):
    destination = isolated.connection_dir() / "test"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(private_home / "other")
    with pytest.raises(ValueError, match="symlinked config file"):
        isolated._private_write(destination, "secret")
    destination.unlink()
    with (
        patch("rapid_mlx.agents.isolated.os.fchmod", side_effect=OSError("denied")),
        pytest.raises(OSError, match="denied"),
    ):
        isolated._private_write(destination, "secret")


def test_key_creation_race_reuses_winner(private_home):
    original_open = os.open

    def raced_open(path, flags, mode):
        if path == isolated.key_path():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("a" * 64 + "\n")
            path.chmod(0o600)
            raise FileExistsError(path)
        return original_open(path, flags, mode)

    with patch("rapid_mlx.agents.isolated.os.open", side_effect=raced_open):
        assert isolated.stable_key() == "a" * 64


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.HTTPError("url", 401, "no", {}, None), "key-mismatch"),
        (urllib.error.HTTPError("url", 503, "loading", {}, None), "server-error"),
        (urllib.error.URLError(ConnectionRefusedError()), "stopped"),
        (ValueError("bad JSON"), "server-error"),
    ],
)
def test_connection_status_categorizes_failures(private_home, error, expected):
    with patch("urllib.request.urlopen", side_effect=error):
        assert isolated.connection_status(7659, "a" * 64, "model") == expected


def test_configure_and_remove_refuse_unowned_paths(private_home, monkeypatch):
    with pytest.raises(ValueError, match="unsupported"):
        isolated.configure("not-an-agent", "model")
    with pytest.raises(ValueError, match="unsupported"):
        isolated.remove("not-an-agent")
    root = isolated.connection_dir() / "agents" / "codex"
    root.parent.mkdir(parents=True)
    root.symlink_to(private_home)
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated.remove("codex")
    root.unlink()
    root.mkdir()
    (root / "config.toml").write_text("old")
    isolated.configure("codex", "qwen3.5-9b-4bit")
    assert not (root / "config.toml").exists()
    (root / "config.toml").symlink_to(private_home / "other")
    with pytest.raises(ValueError, match="symlinked config file"):
        isolated.configure("codex", "qwen3.5-9b-4bit")
    monkeypatch.setenv("RAPID_MLX_API_KEY", "original")
    (root / "config.toml").unlink()
    isolated.configure("codex", "qwen3.5-9b-4bit")
    assert os.environ["RAPID_MLX_API_KEY"] == "original"


def test_runner_command_rejects_unknown_tool_and_uses_current_binary(
    private_home, monkeypatch
):
    with pytest.raises(ValueError, match="unsupported"):
        isolated._command("unknown", private_home, "model", 7659)
    binary = private_home / "rapid-mlx"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr("sys.argv", [str(binary)])
    assert isolated._cli_invocation() == str(binary)


def _launch_args(*argv):
    from rapid_mlx.launch.cli import register

    parser = ArgumentParser()
    register(parser.add_subparsers())
    return parser.parse_args(["launch", *argv])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["cline", "--run"], "require --isolated"),
        (["--isolated"], "name one agent"),
        (["codex", "--isolated", "--check"], "requires --model"),
        (["codex", "--isolated", "--check", "--model", "m"], "key is missing"),
        (["unknown", "--isolated", "--model", "m"], "unknown agent"),
    ],
)
def test_isolated_cli_rejects_incomplete_requests(private_home, capsys, argv, message):
    from rapid_mlx.launch.cli import launch_command

    with pytest.raises(SystemExit):
        launch_command(_launch_args(*argv))
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ready", None),
        ("stopped", "Start it with"),
        ("model-mismatch", "does not match"),
    ],
)
def test_isolated_cli_checks_saved_endpoint(private_home, capsys, status, expected):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value=status),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
    ):
        if expected:
            with pytest.raises(SystemExit, match="1"):
                launch_command(
                    _launch_args("codex", "--isolated", "--check", "--model", "m")
                )
            assert expected in capsys.readouterr().err
        else:
            launch_command(
                _launch_args("codex", "--isolated", "--check", "--model", "m")
            )


def test_isolated_cli_saves_warns_and_removes(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=18),
        patch("rapid_mlx.run.cli._fits_host", return_value=False),
    ):
        launch_command(
            _launch_args(
                "codex", "--isolated", "--model", "gemma-4-12b-it", "--port", "7659"
            )
        )
    output = capsys.readouterr()
    assert "Saved an isolated codex profile" in output.out
    assert "not recommended for this Mac's memory" in output.err
    assert "known to stall" in output.err
    launch_command(_launch_args("codex", "--remove"))
    assert "Removed codex" in capsys.readouterr().out
    launch_command(_launch_args("codex", "--remove"))
    assert "No saved profile" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("stopped", "Local server is stopped"),
        ("model-mismatch", "does not match"),
        ("server-error", "does not match"),
    ],
)
def test_isolated_run_reports_unavailable_server(private_home, capsys, status, message):
    from rapid_mlx.launch.cli import launch_command

    args = ["codex", "--isolated", "--run", "--model", "qwen3.5-9b-4bit"]
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value=status),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args(*args))
    output = capsys.readouterr()
    assert message in output.out + output.err


def test_isolated_run_executes_when_ready(private_home):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("rapid_mlx.launch.cli.subprocess.call", return_value=0) as call,
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(
            _launch_args("codex", "--isolated", "--run", "--model", "qwen3.5-9b-4bit")
        )
    assert call.call_args.args[0][:2] == ["/bin/zsh", "-lc"]


def test_short_launch_reuses_saved_model_and_starts_stopped_server(
    private_home, capsys
):
    from rapid_mlx.launch.cli import launch_command

    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "ready"],
        ),
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.launch.cli.subprocess.call", return_value=0) as call,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(_launch_args("codex"))
    start.assert_called_once_with(
        "qwen3.5-9b-4bit", 7660, api_key=isolated.stable_key()
    )
    assert call.call_args.args[0][:2] == ["/bin/zsh", "-lc"]
    assert "launch codex" in capsys.readouterr().out


def test_short_launch_requires_download_consent_before_spawning(private_home):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="stopped"),
        patch("rapid_mlx.run.cli._confirm_download", return_value=False) as consent,
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("codex", "--model", "qwen3.5-9b-4bit"))
    consent.assert_called_once_with("qwen3.5-9b-4bit", no_download=False, yes=False)
    start.assert_not_called()


def test_cancelled_model_start_terminates_only_spawned_server(
    private_home, monkeypatch
):
    from rapid_mlx.launch import cli as launch_cli

    pid_file = private_home / "launch.pid"
    pid_file.write_text("12345\n")
    monkeypatch.setattr(launch_cli, "PID_FILE", pid_file)
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", KeyboardInterrupt],
        ),
        patch("rapid_mlx.run.cli._confirm_download", return_value=True),
        patch("rapid_mlx.launch.cli._start_server_background", return_value=12345),
        patch("rapid_mlx.launch.cli.os.kill") as kill,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(KeyboardInterrupt),
    ):
        launch_cli.launch_command(_launch_args("codex", "--model", "qwen3.5-9b-4bit"))
    kill.assert_called_once_with(12345, signal.SIGTERM)
    assert not pid_file.exists()


def test_failed_start_cleanup_tolerates_exited_server_and_missing_pid_file(
    private_home, monkeypatch
):
    from rapid_mlx.launch import cli as launch_cli

    monkeypatch.setattr(launch_cli, "PID_FILE", private_home / "missing.pid")
    with patch("rapid_mlx.launch.cli.os.kill", side_effect=ProcessLookupError) as kill:
        launch_cli._stop_failed_start(12345)
    kill.assert_called_once_with(12345, signal.SIGTERM)


def test_started_server_with_wrong_key_is_stopped(private_home, monkeypatch, capsys):
    from rapid_mlx.launch import cli as launch_cli

    pid_file = private_home / "launch.pid"
    pid_file.write_text("12345\n")
    monkeypatch.setattr(launch_cli, "PID_FILE", pid_file)
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "key-mismatch"],
        ),
        patch("rapid_mlx.run.cli._confirm_download", return_value=True),
        patch("rapid_mlx.launch.cli._start_server_background", return_value=12345),
        patch("rapid_mlx.launch.cli.os.kill") as kill,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_cli.launch_command(_launch_args("codex", "--model", "qwen3.5-9b-4bit"))
    kill.assert_called_once_with(12345, signal.SIGTERM)
    assert not pid_file.exists()
    assert "does not match this profile" in capsys.readouterr().err


def test_isolated_cli_reports_no_suitable_model(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.run.cli._select_model", return_value=None),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("codex", "--isolated"))
    assert "no model fits this Mac" in capsys.readouterr().err


def test_isolated_run_times_out_during_startup(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", *["server-error"] * 600],
        ),
        patch("rapid_mlx.launch.cli._start_server_background"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("time.sleep"),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args(
                "codex",
                "--isolated",
                "--run",
                "--start-server",
                "--model",
                "qwen3.5-9b-4bit",
            )
        )
    assert "did not become ready" in capsys.readouterr().out


def test_no_setup_server_reuses_saved_key_without_overriding_explicit_env(
    private_home, monkeypatch
):
    from rapid_mlx.run.cli import _spawn_foreground_serve

    key = isolated.stable_key()
    captured = []
    args = Namespace(host="127.0.0.1", port=7659, no_setup=True, no_download=False)
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)

    def spawn(_command, **kwargs):
        captured.append(kwargs["env"].get("RAPID_MLX_API_KEY"))
        return object()

    with patch("rapid_mlx.run.cli.subprocess.Popen", side_effect=spawn):
        _spawn_foreground_serve("qwen3.5-9b-4bit", args)
        monkeypatch.setenv("RAPID_MLX_API_KEY", "explicit")
        _spawn_foreground_serve("qwen3.5-9b-4bit", args)
        monkeypatch.delenv("RAPID_MLX_API_KEY")
        args.no_setup = False
        _spawn_foreground_serve("qwen3.5-9b-4bit", args)
    assert captured == [key, "explicit", None]


@pytest.mark.parametrize("ancestor", ["connect", "agents"])
def test_profile_operations_reject_symlinked_ancestor(private_home, ancestor):
    external = private_home / "external"
    external.mkdir()
    marker = external / "do-not-delete"
    marker.write_text("keep")
    connect = isolated.connection_dir()
    agents = connect / "agents"
    if ancestor == "connect":
        connect.parent.mkdir()
        connect.symlink_to(external)
    else:
        connect.mkdir(parents=True)
        agents.symlink_to(external)
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated.configure("codex", "qwen3.5-9b-4bit")
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated.remove("codex")
    assert marker.read_text() == "keep"
    assert not (external / "codex").exists()


def test_qwen_runner_reads_current_port_after_saved_port_moves(
    private_home, monkeypatch
):
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    qwen = commands / "qwen"
    qwen.write_text('#!/bin/sh\nprintf "%s" "$OPENAI_BASE_URL" > "$TEST_CAPTURE"\n')
    qwen.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    capture = private_home / "qwen-endpoint"
    monkeypatch.setenv("TEST_CAPTURE", str(capture))
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("qwen-code", "qwen3.5-9b-4bit", port=7660)
    isolated.save_port(7661)
    runner = isolated.connection_dir() / "agents" / "qwen-code" / "run"
    subprocess.run([shutil.which("zsh") or "/bin/bash", str(runner)], check=True)
    assert capture.read_text() == "http://127.0.0.1:7661/v1"
