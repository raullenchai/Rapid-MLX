# SPDX-License-Identifier: Apache-2.0
"""Pin ``rapid-mlx feedback`` — the project's voice channel.

The command has one job (hand the user the community Discord invite)
and a long list of things it must NOT do: never open a browser when
nobody is watching the terminal, never fail because a browser could not
be launched, never drag the telemetry consent prompt in front of
somebody who is on their way to tell us something.

Everything here runs with telemetry untouched; the command reads no
telemetry state at all, which is the property the last test pins.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser

import pytest

from rapid_mlx.cli import FEEDBACK_URL, feedback_command


@pytest.fixture
def opened(monkeypatch):
    """Record every ``webbrowser.open`` call instead of launching one."""
    calls: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **kw: calls.append(url))
    return calls


def _args(**overrides) -> argparse.Namespace:
    base = {"command": "feedback", "no_open": False}
    base.update(overrides)
    return argparse.Namespace(**base)


def _tty(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: value, raising=False)


@pytest.fixture(autouse=True)
def _no_ssh_or_ci(monkeypatch):
    """The suite itself usually runs under CI markers — clear them so the
    interactive path is reachable, and set them back explicitly in the
    tests that are about those markers."""
    from rapid_mlx.cli import _FEEDBACK_NO_BROWSER_ENV

    for name in _FEEDBACK_NO_BROWSER_ENV:
        monkeypatch.delenv(name, raising=False)


def test_prints_the_invite_url(capsys, opened, monkeypatch):
    _tty(monkeypatch, False)
    feedback_command(_args())
    out = capsys.readouterr().out
    assert FEEDBACK_URL in out
    assert "Tell us what you want" in out


def test_url_is_the_invite_the_repo_already_publishes():
    """One invite, everywhere. A second link is a second community."""
    from pathlib import Path

    readme = Path(__file__).resolve().parent.parent / "README.md"
    assert FEEDBACK_URL in readme.read_text(encoding="utf-8")


def test_no_open_never_launches_a_browser(capsys, opened, monkeypatch):
    """Even at a real TTY, ``--no-open`` only prints."""
    _tty(monkeypatch, True)
    feedback_command(_args(no_open=True))
    assert opened == []
    assert FEEDBACK_URL in capsys.readouterr().out


def test_non_tty_never_launches_a_browser(capsys, opened, monkeypatch):
    """``rapid-mlx feedback > url.txt`` must not pop a window."""
    _tty(monkeypatch, False)
    feedback_command(_args())
    assert opened == []


def test_interactive_tty_opens_the_browser(capsys, opened, monkeypatch):
    """The guard above must not be a blanket 'never open' — otherwise the
    two tests before this one would pass with the feature deleted."""
    _tty(monkeypatch, True)
    feedback_command(_args())
    assert opened == [FEEDBACK_URL]


@pytest.mark.parametrize(
    "marker", ["SSH_CONNECTION", "SSH_TTY", "CI", "GITHUB_ACTIONS"]
)
def test_ssh_and_ci_never_launch_a_browser(marker, capsys, opened, monkeypatch):
    _tty(monkeypatch, True)
    monkeypatch.setenv(marker, "1")
    feedback_command(_args())
    assert opened == []
    assert FEEDBACK_URL in capsys.readouterr().out


def test_browser_failure_is_swallowed(capsys, monkeypatch):
    """A machine with no browser still gets the link and exit code 0."""

    def _boom(url, *a, **kw):
        raise OSError("no browser here")

    monkeypatch.setattr(webbrowser, "open", _boom)
    _tty(monkeypatch, True)
    feedback_command(_args())  # must not raise
    assert FEEDBACK_URL in capsys.readouterr().out


def test_unusable_stdout_is_treated_as_non_interactive(capsys, opened, monkeypatch):
    """A detached or already-closed stdout raises from ``isatty()`` rather
    than answering it. That is emphatically not an invitation to open a
    browser, and it must not propagate out of a command whose whole job
    is to print a URL."""

    def _raise():
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys.stdout, "isatty", _raise, raising=False)
    feedback_command(_args())
    assert opened == []
    assert FEEDBACK_URL in capsys.readouterr().out


def test_dispatch_reaches_the_handler_in_process(tmp_path, monkeypatch, capsys):
    """``main()`` itself, not a subprocess — the dispatch branch has to be
    exercised in this interpreter for anyone (coverage included) to see
    that the subcommand is actually wired up."""
    import rapid_mlx.cli as cli

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "feedback", "--no-open"])
    cli.main()
    assert FEEDBACK_URL in capsys.readouterr().out


def test_end_to_end_exit_code_and_output(tmp_path):
    """Through argparse + dispatch, as a user runs it. ``--no-open`` keeps
    the test from spawning a browser on the developer's machine."""
    import os

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "rapid_mlx.cli", "feedback", "--no-open"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert FEEDBACK_URL in r.stdout


def test_feedback_never_prompts_for_telemetry_consent():
    """A user on their way to tell us something is the worst possible
    moment for a consent disclosure."""
    from rapid_mlx.telemetry.consent import _NON_INTERACTIVE_SUBCOMMANDS

    assert "feedback" in _NON_INTERACTIVE_SUBCOMMANDS


def test_feedback_prompt_is_skipped_in_practice(tmp_path, monkeypatch, capsys):
    """Not just set membership — drive the real prompt entry point on a
    machine that has never been asked."""
    import importlib

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    import rapid_mlx.telemetry.state as state

    importlib.reload(state)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    from rapid_mlx.telemetry.consent import maybe_prompt_for_consent

    assert maybe_prompt_for_consent("feedback") is False
    assert capsys.readouterr().out == ""
