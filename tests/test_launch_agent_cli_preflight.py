"""`rapid-mlx launch <agent>` refuses early when the agent CLI is missing."""

import os
import shlex
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rapid_mlx.agents import cli_preflight, get_profile, isolated
from rapid_mlx.agents.base import AgentProfile, AgentTestingSpec


def _launch_child(returncode=0):
    child = MagicMock()
    child.__enter__.return_value = child
    child.wait.return_value = returncode
    return child


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
        patch("rapid_mlx.launch.cli.Popen") as call,
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


@pytest.mark.parametrize("exit_code", [0, 1])
def test_installed_cli_launches_without_warning(private_home, capsys, exit_code):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch.object(cli_preflight, "agent_cli_found", return_value=True),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch(
            "rapid_mlx.launch.cli.Popen",
            return_value=_launch_child(exit_code),
        ) as call,
        patch("rapid_mlx.onboarding.track_first_integration") as configured,
        pytest.raises(SystemExit, match=str(exit_code)),
    ):
        launch_command(
            _launch_args("goose", "--isolated", "--run", "--model", "qwen3.5-9b-4bit")
        )
    call.assert_called_once()
    expected = [(("goose",),)] if exit_code == 0 else []
    assert configured.call_args_list == expected
    assert "not installed" not in capsys.readouterr().err


def _run_goose_launch(child, tracked, code):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch.object(cli_preflight, "agent_cli_found", return_value=True),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("rapid_mlx.launch.cli.Popen", return_value=child),
        patch(
            "rapid_mlx.onboarding.track_first_integration",
            side_effect=tracked.append,
        ),
        pytest.raises(code),
    ):
        launch_command(
            _launch_args("goose", "--isolated", "--run", "--model", "qwen3.5-9b-4bit")
        )


def test_long_session_is_tracked_before_the_child_exits(private_home):
    tracked = []
    child = _launch_child()

    def wait(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("agent", timeout)
        assert tracked == ["goose"]
        return 130

    child.wait.side_effect = wait
    _run_goose_launch(child, tracked, SystemExit)
    assert tracked == ["goose"]


def test_interrupt_kills_the_agent_and_reraises(private_home):
    tracked = []
    child = _launch_child()
    child.wait.side_effect = KeyboardInterrupt
    _run_goose_launch(child, tracked, KeyboardInterrupt)
    child.kill.assert_called_once()
    assert tracked == []


@pytest.fixture(autouse=True)
def no_real_group_kill(request, monkeypatch):
    """Mocked probes have fake pids; never signal a real process group."""
    if request.node.get_closest_marker("skipif"):
        return
    monkeypatch.setattr(cli_preflight.os, "killpg", lambda *_: None)


def _reply(status: bytes | None):
    """A ``subprocess.Popen`` stand-in that writes the probe's answer file."""

    class Probe:
        pid = 424242

        def __init__(self, argv, **kwargs):
            assert kwargs["stdout"] is subprocess.DEVNULL
            assert kwargs["stderr"] is subprocess.DEVNULL
            assert kwargs["start_new_session"] is True
            if status is not None:
                redirect = argv[2].rsplit(">|", 1)[1].split(";", 1)[0]
                answer = shlex.split(redirect)[0]
                Path(answer).write_bytes(status)

        def wait(self, timeout=None):
            return 0

    return Probe


def test_login_shell_exit_status_is_the_answer():
    with patch.object(
        cli_preflight.subprocess, "Popen", side_effect=_reply(b"0\n")
    ) as run:
        assert cli_preflight.agent_cli_found("fake-agent") is True
    argv = run.call_args.args[0]
    assert argv[:2] == ["/bin/zsh", "-lc"]
    assert "builtin whence -p -- fake-agent" in argv[2]
    assert argv[2].endswith("; builtin kill -KILL 0")


def test_binary_is_passed_to_the_shell_verbatim():
    with patch.object(
        cli_preflight.subprocess, "Popen", side_effect=_reply(b"0\n")
    ) as popen:
        assert cli_preflight.agent_cli_found("my agent") is True
    assert "whence -p -- 'my agent' " in popen.call_args.args[0][2]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(b"1\n", False), (b"0\n", True), (b"", None), (None, None)],
)
def test_login_shell_answer_file(status, expected):
    with patch.object(cli_preflight.subprocess, "Popen", side_effect=_reply(status)):
        assert cli_preflight.agent_cli_found("fake-agent") is expected


def test_timed_out_probe_stops_its_group_before_reaping(monkeypatch):
    events = []

    class HungProbe:
        pid = 424242

        def __init__(self, argv, **kwargs):
            pass

        def kill(self):
            events.append(("kill", self.pid))

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired("zsh", timeout)
            return -9

    monkeypatch.setattr(
        cli_preflight.os, "killpg", lambda pid, sig: events.append(("killpg", pid))
    )
    with patch.object(cli_preflight.subprocess, "Popen", HungProbe):
        assert cli_preflight.agent_cli_found("fake-agent") is None
    assert events == [
        ("wait", cli_preflight._PROBE_TIMEOUT_S),
        ("killpg", 424242),
        ("kill", 424242),
        ("wait", None),
    ]


def test_unavailable_login_shell_is_unconfirmed():
    with patch.object(
        cli_preflight.subprocess, "Popen", side_effect=OSError("no shell")
    ):
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


def _assert_exits(pid: int) -> None:
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    pytest.fail("a process started by the login profile outlived the probe")


@needs_zsh
def test_real_login_shell_background_child_does_not_block_probe(
    tmp_path, login_profile
):
    """A profile-started process holding stdout neither stalls nor outlives it."""
    import time

    _executable(tmp_path / "rapid-preflight-bg-agent")
    pid_file = tmp_path / "child.pid"
    login_profile(
        f'export PATH="{tmp_path}:$PATH"\nsleep 20 &\nprint $! > "{pid_file}"\n'
    )
    started = time.monotonic()
    assert cli_preflight.agent_cli_found("rapid-preflight-bg-agent") is True
    assert time.monotonic() - started < 10
    _assert_exits(int(pid_file.read_text()))


@needs_zsh
def test_real_login_shell_timeout_stops_profile_children(
    tmp_path, monkeypatch, login_profile
):
    """A hung profile is unconfirmed, and nothing it started is left behind."""
    pid_file = tmp_path / "child.pid"
    login_profile(f'sleep 30 &\nprint $! > "{pid_file}"\nwait\n')
    monkeypatch.setattr(cli_preflight, "_PROBE_TIMEOUT_S", 2)
    assert cli_preflight.agent_cli_found("rapid-preflight-any-agent") is None
    _assert_exits(int(pid_file.read_text()))
