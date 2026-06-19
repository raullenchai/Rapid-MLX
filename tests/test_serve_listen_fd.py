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


def test_run_uvicorn_passes_fd_when_listen_fd_set(monkeypatch):
    """When ``args.listen_fd`` is an int, ``_run_uvicorn`` MUST invoke
    ``uvicorn.run(app, fd=N, ...)`` and MUST NOT pass ``host``/``port``.

    This is the load-bearing assertion for Leg 2: a regression that
    silently re-introduces ``host=``/``port=`` in the fd branch reopens
    the bind→auth window. The companion bytecode test below pins that
    ``serve_command`` actually invokes this helper so the contract
    can't drift from the call site.

    Round-1 PR #696 codex review: the prior version reimplemented the
    branch inside the test, so it passed even if the production call
    site were deleted. This version exercises the real
    ``cli._run_uvicorn`` and patches ``uvicorn.run`` to capture kwargs.
    """
    captured_kwargs: dict = {}

    def fake_run(app, **kwargs):
        captured_kwargs["app"] = app
        captured_kwargs.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    ns = _minimal_serve_ns(listen_fd=7, port=9000, host="127.0.0.1")
    sentinel_app = object()
    cli._run_uvicorn(sentinel_app, ns, "info")

    assert captured_kwargs.get("app") is sentinel_app
    assert captured_kwargs.get("fd") == 7, (
        f"expected fd=7 in uvicorn.run kwargs, got {captured_kwargs!r}"
    )
    assert "host" not in captured_kwargs, (
        f"host must NOT be passed when fd is set, got {captured_kwargs!r}"
    )
    assert "port" not in captured_kwargs, (
        f"port must NOT be passed when fd is set, got {captured_kwargs!r}"
    )
    assert captured_kwargs.get("log_level") == "info"
    assert captured_kwargs.get("timeout_keep_alive") == 30


def test_run_uvicorn_passes_host_port_when_listen_fd_unset(monkeypatch):
    """The default path is unchanged: ``host``/``port`` flow through and
    ``fd`` is NOT passed. Same anti-regression shape as the listen-fd
    case — exercises the real ``cli._run_uvicorn`` rather than
    reimplementing the branch in the test body.
    """
    captured_kwargs: dict = {}

    def fake_run(app, **kwargs):
        captured_kwargs["app"] = app
        captured_kwargs.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    ns = _minimal_serve_ns(port=9000, host="127.0.0.1")
    assert getattr(ns, "listen_fd", None) is None
    sentinel_app = object()
    cli._run_uvicorn(sentinel_app, ns, "info")

    assert captured_kwargs.get("app") is sentinel_app
    assert captured_kwargs.get("host") == "127.0.0.1"
    assert captured_kwargs.get("port") == 9000
    assert "fd" not in captured_kwargs
    assert captured_kwargs.get("log_level") == "info"
    assert captured_kwargs.get("timeout_keep_alive") == 30


def test_serve_command_wires_run_uvicorn_helper():
    """``serve_command`` must actually invoke ``_run_uvicorn`` — without
    this, the unit tests above are tautological (a regression that
    deletes the helper call from ``serve_command`` would not be
    caught).

    Bytecode inspection mirrors the established pattern in
    ``tests/test_memory_capacity_check.py::
    test_check_is_wired_into_serve_and_bench``. Comments / docstrings
    are stripped at compile time, so only real references survive.
    """
    import dis

    refs = {
        ins.argval
        for ins in dis.get_instructions(cli.serve_command)
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF")
    }
    assert "_run_uvicorn" in refs, (
        "serve_command must reference _run_uvicorn — a regression that "
        "inlines uvicorn.run back into the function body would silently "
        "decouple the dispatch from the unit-tested helper"
    )


def test_serve_command_stashes_bind_listen_fd_for_ready_banner():
    """The lifespan "Ready:" banner switches on whether
    ``_cfg.bind_host``/``bind_port`` OR ``_cfg.bind_listen_fd`` is set
    (see ``vllm_mlx/server.py``). Without ``bind_listen_fd`` stamped in
    the fd branch, the banner falls through to silent — round-3 codex
    PR #696 flagged this as a missing source of truth. Bytecode
    inspection asserts ``serve_command`` references
    ``bind_listen_fd`` so a refactor that drops the stash silently is
    caught by the unit suite.
    """
    import dis

    refs = {
        ins.argval
        for ins in dis.get_instructions(cli.serve_command)
        if ins.opname in ("LOAD_ATTR", "STORE_ATTR", "LOAD_GLOBAL", "LOAD_NAME")
    }
    assert "bind_listen_fd" in refs, (
        "serve_command must stash listen_fd into ServerConfig.bind_listen_fd "
        "so the lifespan Ready banner has a source of truth in the "
        "socket-activation branch"
    )


def test_serve_command_does_not_inline_uvicorn_run():
    """Companion to the wiring test: assert ``serve_command`` does NOT
    call ``uvicorn.run`` directly. Belt-and-braces against a refactor
    that adds back an inline call while still keeping the helper
    invocation around — the inline path would skip the unit-tested
    dispatch logic entirely.
    """
    import dis

    seq = list(dis.get_instructions(cli.serve_command))
    # Look for ``uvicorn`` global followed by a ``run`` attribute lookup.
    for i, ins in enumerate(seq):
        if ins.opname == "LOAD_GLOBAL" and ins.argval == "uvicorn":
            attr = next(
                (
                    s
                    for s in seq[i + 1 : i + 4]
                    if s.opname in ("LOAD_ATTR", "LOAD_METHOD")
                ),
                None,
            )
            assert attr is None or attr.argval != "run", (
                "serve_command must dispatch through _run_uvicorn — found "
                "inline uvicorn.run reference"
            )


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
