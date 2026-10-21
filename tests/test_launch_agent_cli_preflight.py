"""`rapid-mlx launch <agent>` refuses early when the agent CLI is missing."""

import os
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
        patch.object(cli_preflight, "agent_cli_found", return_value=False),
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
        patch.object(cli_preflight, "agent_cli_found", return_value=False),
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
        patch.object(cli_preflight, "agent_cli_found", return_value=False),
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
        patch.object(cli_preflight, "agent_cli_found", return_value=True),
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


def _reply(stdout: bytes) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=b"")


def test_login_shell_exit_status_is_the_answer(fixed_marker):
    reply = _reply(
        b"welcome \xff\xfe\n__rapid_mlx_cli__=0\n\n" + f"{MARK}0\nbye\n".encode()
    )
    with patch.object(cli_preflight.subprocess, "run", return_value=reply) as run:
        assert cli_preflight.agent_cli_found("fake-agent") is True
    argv = run.call_args.args[0]
    assert argv[:2] == ["/bin/zsh", "-lc"]
    assert "builtin whence -p -- fake-agent" in argv[2]
    assert MARK in argv[2]


def test_path_style_binary_is_expanded_for_the_shell(fixed_marker):
    with patch.object(
        cli_preflight.subprocess, "run", return_value=_reply(f"{MARK}0\n".encode())
    ) as run:
        assert cli_preflight.agent_cli_found("~/bin/tool") is True
    assert (
        f"whence -p -- {os.path.expanduser('~/bin/tool')}" in run.call_args.args[0][2]
    )


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (f"{MARK}1\n", False),
        (f"{MARK}0\n{MARK}1\n", False),
        ("__rapid_mlx_cli_other__=0\n", None),
        ("", None),
    ],
)
def test_login_shell_miss_or_unmarked_output(fixed_marker, stdout, expected):
    with patch.object(
        cli_preflight.subprocess, "run", return_value=_reply(stdout.encode())
    ):
        assert cli_preflight.agent_cli_found("fake-agent") is expected


@pytest.mark.parametrize(
    "error",
    [
        OSError("no shell"),
        subprocess.TimeoutExpired("zsh", 30),
        subprocess.SubprocessError("probe failed"),
    ],
)
def test_unavailable_login_shell_is_unconfirmed(error):
    with patch.object(cli_preflight.subprocess, "run", side_effect=error):
        assert cli_preflight.agent_cli_found("fake-agent") is None


def test_unconfirmed_cli_refuses_with_its_own_message():
    profile = get_profile("codex")
    assert profile is not None
    with patch.object(cli_preflight, "agent_cli_found", return_value=None):
        message = cli_preflight.missing_agent_cli(profile)
    assert message is not None
    assert message.startswith("Could not confirm that the")
    assert "(codex)" in message
    assert profile.testing.install_cmd in message


def test_message_handles_profiles_without_cli_metadata():
    unchecked = AgentProfile(name="x", display_name="X")
    assert cli_preflight.missing_agent_cli(unchecked) is None
    no_hint = AgentProfile(
        name="x",
        display_name="X Agent",
        testing=AgentTestingSpec(binary="x-agent-not-installed"),
    )
    with patch.object(cli_preflight, "agent_cli_found", return_value=False):
        message = cli_preflight.missing_agent_cli(no_hint)
    assert (
        message
        == "X Agent CLI (x-agent-not-installed) is not installed or not on PATH."
    )


def test_user_profile_shadowing_builtin_keeps_builtin_cli_check():
    shadow = AgentProfile(name="claude-code", display_name="My Claude")
    with patch.object(cli_preflight, "agent_cli_found", return_value=False) as found:
        message = cli_preflight.missing_agent_cli(shadow)
    found.assert_called_once_with("claude")
    assert message is not None
    assert "npm install -g @anthropic-ai/claude-code" in message


needs_zsh = pytest.mark.skipif(
    not os.access("/bin/zsh", os.X_OK), reason="needs /bin/zsh"
)


@pytest.fixture
def login_profile(tmp_path, monkeypatch):
    """A private zsh login profile; returns a writer for its .zprofile."""
    zdot = tmp_path / "zdot"
    zdot.mkdir()
    monkeypatch.setenv("ZDOTDIR", str(zdot))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    def write(text: str) -> None:
        (zdot / ".zprofile").write_text(text)

    write("")
    return write


@needs_zsh
def test_real_login_shell_sees_profile_path(tmp_path, login_profile):
    """A CLI added to PATH only by the login profile is not refused."""
    bin_dir = tmp_path / "login-bin"
    bin_dir.mkdir()
    _executable(bin_dir / "rapid-preflight-fake-agent")
    login_profile(
        'echo "__rapid_mlx_cli__=1"\n'
        "printf 'non-utf8 \\377\\376\\n'\n"
        f'export PATH="{bin_dir}:$PATH"\n'
    )
    assert cli_preflight.agent_cli_found("rapid-preflight-fake-agent") is True
    assert cli_preflight.agent_cli_found("rapid-preflight-absent-agent") is False


@needs_zsh
def test_real_login_shell_that_drops_path_entry_is_refused(
    tmp_path, monkeypatch, login_profile
):
    """The runner would not find a CLI that the login profile removes."""
    bin_dir = tmp_path / "parent-bin"
    bin_dir.mkdir()
    _executable(bin_dir / "rapid-preflight-dropped-agent")
    login_profile('export PATH="/usr/bin:/bin"\n')
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    assert cli_preflight.agent_cli_found("rapid-preflight-dropped-agent") is False


@needs_zsh
def test_real_login_shell_checks_paths_and_ignores_profile_functions(
    tmp_path, login_profile
):
    """The runner script does not inherit profile functions or aliases."""
    tool = _executable(tmp_path / "tool")
    plain = tmp_path / "plain"
    plain.write_text("")
    login_profile(
        "rapid-preflight-fn() { :; }\nalias rapid-preflight-alias=ls\n"
        "whence() { return 0; }\nprintf() { echo spoofed; }\n"
    )
    assert cli_preflight.agent_cli_found(str(tool)) is True
    assert cli_preflight.agent_cli_found(str(plain)) is False
    assert cli_preflight.agent_cli_found(str(tmp_path)) is False
    assert cli_preflight.agent_cli_found("rapid-preflight-fn") is False
    assert cli_preflight.agent_cli_found("rapid-preflight-alias") is False
    assert cli_preflight.agent_cli_found("rapid-preflight-absent-agent") is False
