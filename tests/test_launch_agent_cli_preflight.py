"""`rapid-mlx launch <agent>` refuses early when the agent CLI is missing."""

import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import pytest

from rapid_mlx.agents import cli_preflight, get_profile, isolated
from rapid_mlx.agents.base import AgentProfile, AgentTestingSpec


@pytest.fixture
def private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def _launch_args(*argv):
    from rapid_mlx.launch.cli import register

    parser = ArgumentParser()
    register(parser.add_subparsers())
    return parser.parse_args(["launch", *argv])


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("name", sorted(isolated.SUPPORTED))
def test_every_launchable_agent_declares_cli_and_install_hint(name):
    profile = get_profile(name)
    assert profile is not None
    assert profile.testing.binary
    assert profile.testing.install_cmd


@pytest.mark.parametrize("name", sorted(isolated.SUPPORTED))
def test_missing_cli_stops_launch_before_model_work(private_home, capsys, name):
    from rapid_mlx.launch.cli import launch_command

    profile = get_profile(name)
    assert profile is not None
    with (
        patch.object(cli_preflight, "resolve_agent_binary", return_value=None),
        patch("rapid_mlx.run.cli._select_model") as select,
        patch("rapid_mlx.cli._ensure_model_downloaded") as download,
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.launch.cli.subprocess.call") as call,
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args(name))
    select.assert_not_called()
    download.assert_not_called()
    start.assert_not_called()
    call.assert_not_called()
    assert isolated.saved_model(name) is None
    err = capsys.readouterr().err
    assert f"({profile.testing.binary}) is not installed" in err
    assert profile.testing.install_cmd in err


def test_missing_cli_with_explicit_model_also_stops(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch.object(cli_preflight, "resolve_agent_binary", return_value=None),
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args(
                "claude-code", "--isolated", "--run", "--model", "qwen3.5-9b-4bit"
            )
        )
    start.assert_not_called()
    assert "npm install -g" in capsys.readouterr().err


def test_saving_profile_without_run_only_warns(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch.object(cli_preflight, "resolve_agent_binary", return_value=None),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
    ):
        launch_command(
            _launch_args("codex", "--isolated", "--model", "qwen3.5-9b-4bit")
        )
    output = capsys.readouterr()
    assert "Saved an isolated codex profile" in output.out
    assert "Warning:" in output.err
    assert "(codex) is not installed" in output.err
    assert isolated.saved_model("codex") == "qwen3.5-9b-4bit"


def test_installed_cli_launches_without_warning(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch.object(
            cli_preflight, "resolve_agent_binary", return_value="/usr/local/bin/goose"
        ),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("rapid_mlx.launch.cli.subprocess.call", return_value=0) as call,
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(
            _launch_args("goose", "--isolated", "--run", "--model", "qwen3.5-9b-4bit")
        )
    call.assert_called_once()
    assert "not installed" not in capsys.readouterr().err


MARK = "__rapid_mlx_cli_feedface__="


@pytest.fixture
def fixed_marker(monkeypatch):
    monkeypatch.setattr(cli_preflight.secrets, "token_hex", lambda _n: "feedface")


def test_login_shell_answer_is_authoritative(tmp_path, monkeypatch, fixed_marker):
    tool = _executable(tmp_path / "fake-agent")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    reply = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            b"welcome \xff\xfe\n__rapid_mlx_cli__=/bogus\n\n"
            + f"{MARK}{tool}\nlogout banner\n".encode()
        ),
        stderr="",
    )
    with patch.object(cli_preflight.subprocess, "run", return_value=reply) as run:
        assert cli_preflight.resolve_agent_binary("fake-agent") == str(tool)
    argv = run.call_args.args[0]
    assert argv[:2] == ["/bin/zsh", "-lc"]
    assert "command -v -- fake-agent" in argv[2]
    assert MARK in argv[2]


def test_current_path_hit_is_refused_when_login_shell_misses(
    tmp_path, monkeypatch, fixed_marker
):
    _executable(tmp_path / "fake-agent")
    monkeypatch.setenv("PATH", str(tmp_path))
    reply = subprocess.CompletedProcess(
        [], 0, stdout=f"\n{MARK}\n".encode(), stderr=b""
    )
    with patch.object(cli_preflight.subprocess, "run", return_value=reply):
        assert cli_preflight.resolve_agent_binary("fake-agent") is None


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "/opt/homebrew/bin/fake-agent\n",
        f"{MARK}alias fake-agent='something'\n",
        f"{MARK}fake-agent\n",
        "__rapid_mlx_cli_other__=/bin/sh\n",
    ],
)
def test_login_shell_miss_or_non_executable_is_not_installed(
    tmp_path, monkeypatch, fixed_marker, stdout
):
    monkeypatch.setenv("PATH", str(tmp_path))
    reply = subprocess.CompletedProcess([], 0, stdout=stdout.encode(), stderr=b"")
    with patch.object(cli_preflight.subprocess, "run", return_value=reply):
        assert cli_preflight.resolve_agent_binary("fake-agent") is None


def test_login_shell_answer_must_be_executable_file(
    tmp_path, monkeypatch, fixed_marker
):
    reply = subprocess.CompletedProcess(
        [], 0, stdout=f"{MARK}{tmp_path}\n".encode(), stderr=b""
    )
    with patch.object(cli_preflight.subprocess, "run", return_value=reply):
        assert cli_preflight.resolve_agent_binary("fake-agent") is None


@pytest.mark.parametrize(
    "error",
    [
        OSError("no shell"),
        subprocess.TimeoutExpired("zsh", 30),
        subprocess.SubprocessError("probe failed"),
    ],
)
def test_unprobeable_login_shell_falls_back_to_path_and_homebrew(
    tmp_path, monkeypatch, error
):
    brew = tmp_path / "brew-bin"
    brew.mkdir()
    _executable(brew / "brew-agent")
    current = tmp_path / "current-bin"
    current.mkdir()
    _executable(current / "path-agent")
    monkeypatch.setenv("PATH", str(current))
    monkeypatch.setattr(cli_preflight, "_HOMEBREW_PREFIXES", (str(brew),))
    with patch.object(cli_preflight.subprocess, "run", side_effect=error):
        assert cli_preflight.resolve_agent_binary("brew-agent") == str(
            brew / "brew-agent"
        )
        assert cli_preflight.resolve_agent_binary("path-agent") == str(
            current / "path-agent"
        )
        assert cli_preflight.resolve_agent_binary("other-agent") is None


def test_path_style_binary_requires_executable_file(tmp_path):
    tool = _executable(tmp_path / "tool")
    plain = tmp_path / "plain"
    plain.write_text("")
    assert cli_preflight.resolve_agent_binary(str(tool)) == str(tool)
    assert cli_preflight.resolve_agent_binary(str(plain)) is None
    assert cli_preflight.resolve_agent_binary(str(tmp_path)) is None
    assert cli_preflight.resolve_agent_binary(str(tmp_path / "absent")) is None


def test_message_handles_profiles_without_cli_metadata():
    unchecked = AgentProfile(name="x", display_name="X")
    assert cli_preflight.missing_agent_cli(unchecked) is None
    no_hint = AgentProfile(
        name="x",
        display_name="X Agent",
        testing=AgentTestingSpec(binary="x-agent-not-installed"),
    )
    with patch.object(cli_preflight, "resolve_agent_binary", return_value=None):
        message = cli_preflight.missing_agent_cli(no_hint)
    assert (
        message
        == "X Agent CLI (x-agent-not-installed) is not installed or not on PATH."
    )


@pytest.mark.skipif(not os.access("/bin/zsh", os.X_OK), reason="needs /bin/zsh")
def test_real_login_shell_sees_profile_path(tmp_path, monkeypatch):
    """A CLI added to PATH only by the login profile is not refused."""
    bin_dir = tmp_path / "login-bin"
    bin_dir.mkdir()
    _executable(bin_dir / "rapid-preflight-fake-agent")
    zdot = tmp_path / "zdot"
    zdot.mkdir()
    (zdot / ".zprofile").write_text(f'export PATH="{bin_dir}:$PATH"\n')
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert shutil.which("rapid-preflight-fake-agent") is None
    assert cli_preflight.resolve_agent_binary("rapid-preflight-fake-agent") == str(
        bin_dir / "rapid-preflight-fake-agent"
    )


@pytest.mark.skipif(not os.access("/bin/zsh", os.X_OK), reason="needs /bin/zsh")
def test_real_login_shell_output_around_answer_is_ignored(tmp_path, monkeypatch):
    bin_dir = tmp_path / "login-bin"
    bin_dir.mkdir()
    _executable(bin_dir / "rapid-preflight-chatty-agent")
    zdot = tmp_path / "zdot"
    zdot.mkdir()
    (zdot / ".zprofile").write_text(
        'echo "__rapid_mlx_cli__=/bin/sh"\n'
        "printf 'non-utf8 \\377\\376\\n'\n"
        f'echo "hello from profile"\nexport PATH="{bin_dir}:$PATH"\n'
    )
    (zdot / ".zlogout").write_text("echo bye; printf '\\377\\376\\n'\n")
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert cli_preflight.resolve_agent_binary("rapid-preflight-chatty-agent") == str(
        bin_dir / "rapid-preflight-chatty-agent"
    )
    assert cli_preflight.resolve_agent_binary("rapid-preflight-absent-agent") is None


def test_user_profile_shadowing_builtin_keeps_builtin_cli_check():
    shadow = AgentProfile(name="claude-code", display_name="My Claude")
    with patch.object(
        cli_preflight, "resolve_agent_binary", return_value=None
    ) as resolve:
        message = cli_preflight.missing_agent_cli(shadow)
    resolve.assert_called_once_with("claude")
    assert message is not None
    assert "npm install -g @anthropic-ai/claude-code" in message


@pytest.mark.skipif(not os.access("/bin/zsh", os.X_OK), reason="needs /bin/zsh")
def test_real_login_shell_that_drops_path_entry_is_refused(tmp_path, monkeypatch):
    """The runner would not find a CLI that the login profile removes."""
    bin_dir = tmp_path / "parent-bin"
    bin_dir.mkdir()
    _executable(bin_dir / "rapid-preflight-dropped-agent")
    zdot = tmp_path / "zdot"
    zdot.mkdir()
    (zdot / ".zprofile").write_text('export PATH="/usr/bin:/bin"\n')
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    assert shutil.which("rapid-preflight-dropped-agent") is not None
    assert cli_preflight.resolve_agent_binary("rapid-preflight-dropped-agent") is None
