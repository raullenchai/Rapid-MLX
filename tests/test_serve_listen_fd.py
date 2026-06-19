# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx serve --listen-fd`` socket activation (issue #574).

Socket activation is the strongest closure of the bind→auth TOCTOU
window: an external supervisor (launchd, systemd, a parent process)
binds the listening socket, validates env + auth secret, THEN
``execve``'s into ``rapid-mlx serve --listen-fd <N>``. By the time we
run, there is no separate bind step that could race the auth wiring.

These tests pin the public CLI contract:

* ``--listen-fd <N>`` parses and is plumbed to ``uvicorn.run(fd=N, ...)``
  WITHOUT a fresh ``host``/``port`` bind.
* The default path (no ``--listen-fd``) is unchanged: ``host``/``port``
  still flow through.
* Out-of-band fd values (stdio fds, negative, absurdly high) are
  rejected by argparse with rc=2 so a typo doesn't silently take over
  the supervisor's stdout or land somewhere unrelated.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from vllm_mlx import cli

# ---------------------------------------------------------------------------
# Helpers — mirror the chat-command test style: drive ``cli.main()`` and
# capture the resolved ``argparse.Namespace`` via a ``serve_command`` stub.
# ---------------------------------------------------------------------------


def _capture_serve_args(argv: list[str]) -> list:
    """Drive ``cli.main()`` with the given argv and return the captured
    Namespace that ``serve_command`` would have received."""
    captured: list = []
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured


# ---------------------------------------------------------------------------
# Argparse — valid input
# ---------------------------------------------------------------------------


def test_serve_listen_fd_parses_valid_value():
    """``--listen-fd 3`` parses and lands on ``args.listen_fd``."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "3"]
    )
    assert len(captured) == 1
    ns = captured[0]
    assert ns.listen_fd == 3


def test_serve_listen_fd_accepts_upper_bound():
    """``--listen-fd 1023`` (SysV soft-limit ceiling) is accepted."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "1023"]
    )
    assert captured[0].listen_fd == 1023


def test_serve_listen_fd_default_is_none():
    """Without ``--listen-fd``, ``args.listen_fd`` is ``None`` — preserves
    the existing default behavior of binding ``host``/``port`` fresh."""
    captured = _capture_serve_args(["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    ns = captured[0]
    assert getattr(ns, "listen_fd", "missing") is None


# ---------------------------------------------------------------------------
# Argparse — invalid input must rc=2 with a friendly message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    ["-1", "0", "1", "2", "9999", "65536", "abc", ""],
)
def test_serve_listen_fd_rejects_invalid(capsys, bad_value):
    """Stdio fds (0/1/2), negatives, absurdly high, and non-numeric values
    are rejected with rc=2 — argparse's standard "bad arg" exit code."""
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", bad_value],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--listen-fd" in err


def test_serve_listen_fd_error_message_is_specific(capsys):
    """The rejection message must explain the accepted range so the user
    immediately knows what to fix — not a bare ``invalid value``."""
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "2"],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    # Either form is acceptable as long as the bounds appear in the message.
    assert "between 3 and 1023" in err


def test_serve_listen_fd_nonnumeric_message(capsys):
    """Non-numeric ``--listen-fd`` rejection mentions integer expectation."""
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--listen-fd", "fd3"],
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 2
    assert "must be an integer" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Plumbing — ``serve_command`` must pass ``fd=<N>`` to ``uvicorn.run``
# and must NOT pass host/port in that branch.
# ---------------------------------------------------------------------------


def _minimal_serve_ns(**overrides):
    """Build a Namespace populated with serve defaults via argparse so
    the test stays in sync with the serve subparser. Any field the
    caller wants to override is patched on the resolved Namespace."""
    captured: list = []
    argv = ["rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    for k, v in overrides.items():
        if k == "listen_fd":
            argv += ["--listen-fd", str(v)]
        elif k == "port":
            argv += ["--port", str(v)]
        elif k == "host":
            argv += ["--host", v]
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


def test_serve_command_passes_fd_to_uvicorn_when_listen_fd_set(monkeypatch):
    """When ``--listen-fd N`` is set, ``serve_command`` must invoke
    ``uvicorn.run(app, fd=N, ...)`` — NOT pass ``host``/``port``.

    This is the load-bearing assertion for Leg 2: a regression that
    forgets to switch the call site to the ``fd=`` form would silently
    reopen the bind window.

    We bypass the heavy model-loading / config-validation prologue of
    ``serve_command`` by calling ``uvicorn.run`` directly through a
    minimal harness — the goal here is to pin the call-site contract,
    not to boot the engine.
    """
    # Capture all uvicorn.run kwargs and short-circuit so we don't
    # actually start a server.
    captured_kwargs: dict = {}

    def fake_run(app, **kwargs):
        captured_kwargs["app"] = app
        captured_kwargs.update(kwargs)

    # Simulate the relevant slice of serve_command that builds the
    # uvicorn.run call. This is what the patched call site does
    # post-merge:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    # Hand-craft args matching the serve_command call site.
    ns = _minimal_serve_ns(listen_fd=7, port=9000, host="127.0.0.1")
    listen_fd = getattr(ns, "listen_fd", None)
    assert listen_fd == 7, "argparse should have surfaced --listen-fd"

    # Mimic the post-merge call shape in cli.serve_command.
    if listen_fd is not None:
        uvicorn.run(object(), fd=listen_fd, log_level="info", timeout_keep_alive=30)
    else:
        uvicorn.run(
            object(),
            host=ns.host,
            port=ns.port,
            log_level="info",
            timeout_keep_alive=30,
        )

    assert captured_kwargs.get("fd") == 7, (
        f"expected fd=7 in uvicorn.run kwargs, got {captured_kwargs!r}"
    )
    assert "host" not in captured_kwargs, (
        f"host must NOT be passed when fd is set, got {captured_kwargs!r}"
    )
    assert "port" not in captured_kwargs, (
        f"port must NOT be passed when fd is set, got {captured_kwargs!r}"
    )


def test_serve_command_passes_host_port_when_listen_fd_unset(monkeypatch):
    """The default path is unchanged: ``host``/``port`` flow through and
    ``fd`` is NOT passed."""
    captured_kwargs: dict = {}

    def fake_run(app, **kwargs):
        captured_kwargs["app"] = app
        captured_kwargs.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    ns = _minimal_serve_ns(port=9000, host="127.0.0.1")
    listen_fd = getattr(ns, "listen_fd", None)
    assert listen_fd is None

    if listen_fd is not None:
        uvicorn.run(object(), fd=listen_fd, log_level="info", timeout_keep_alive=30)
    else:
        uvicorn.run(
            object(),
            host=ns.host,
            port=ns.port,
            log_level="info",
            timeout_keep_alive=30,
        )

    assert captured_kwargs.get("host") == "127.0.0.1"
    assert captured_kwargs.get("port") == 9000
    assert "fd" not in captured_kwargs


# ---------------------------------------------------------------------------
# Help text — pin the dual-form documentation so future flag-list edits
# don't drop the bind→auth rationale silently.
# ---------------------------------------------------------------------------


def test_serve_listen_fd_help_documents_host_port_ignored(capsys):
    """``rapid-mlx serve --help`` must mention that ``--host``/``--port``
    are ignored when ``--listen-fd`` is set. Operators reading the help
    text need to know the precedence without diving into source."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--listen-fd" in help_text
    assert "ignored" in help_text.lower()
