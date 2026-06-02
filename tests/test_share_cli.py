# SPDX-License-Identifier: Apache-2.0
"""Integration smoke test for the share CLI orchestration.

We mock both subprocess spawns (serve + frpc) and the session HTTP call
so the test runs in <1s and doesn't touch the network. The assertions
focus on the pieces of the contract that are easy to silently break:
the security banner content, frpc config shape, and ordered shutdown.
"""

from __future__ import annotations

import argparse
import signal
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.share import cli as share_cli
from vllm_mlx.share.session import Session


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path, monkeypatch):
    """Redirect ``_state_dir()`` to a per-test tmp_path so we don't
    mkdir/chmod the user's real ``~/.cache/rapid-mlx/share``. Sandboxed
    CI runs (codex review, Docker builds) lack permission to chmod a
    cache dir owned by another uid, which surfaces as PermissionError
    rather than a code bug.
    """
    d = tmp_path / "share-state"
    d.mkdir()
    monkeypatch.setattr(share_cli, "_state_dir", lambda: d)
    return d


def _make_args(**overrides):
    defaults = dict(
        model="qwen3.5-4b",
        port=18765,  # explicit so the env-var fallback path isn't exercised
        no_thinking=True,
        cors_origins="*",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def fake_session() -> Session:
    return Session(
        subdomain="testabc",
        token="tk_abc",
        frps_host="tunnel.rapidmlx.com",
        frps_port=7000,
        public_url="https://testabc.rapidmlx.com",
        expires_at="2026-06-02T12:00:00Z",
    )


def test_share_command_happy_path(fake_session, capsys):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    # Raise Ctrl-C on the blocking wait, return cleanly on the cleanup wait
    # in the finally block. MagicMock applies side_effect to every call,
    # so we use a list-based side_effect to scope the raise.
    serve_proc.wait.side_effect = [KeyboardInterrupt, None]
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    with patch.object(share_cli, "_spawn_serve", return_value=serve_proc), patch.object(
        share_cli, "_wait_for_healthz", return_value=True
    ), patch.object(
        share_cli.session, "request", return_value=fake_session
    ), patch.object(
        share_cli.frpc_manager, "spawn", return_value=frpc_proc
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch(
        "time.sleep"
    ):
        share_cli.share_command(_make_args())

    out = capsys.readouterr().out
    # Banner pieces that protect users from leaking the key:
    assert "PUBLIC INTERNET" in out
    assert "Do NOT screenshot" in out
    # URL + curl example:
    assert "https://testabc.rapidmlx.com" in out
    # Ctrl-C path must terminate both children (frpc first, then serve).
    serve_proc.terminate.assert_called_once()
    frpc_proc.terminate.assert_called_once()


def test_share_command_aborts_when_serve_exits_before_ready(fake_session):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = 1  # already exited; cleanup is a no-op

    with patch.object(share_cli, "_spawn_serve", return_value=serve_proc), patch.object(
        share_cli, "_wait_for_healthz", return_value=False
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch(
        "time.sleep"
    ), pytest.raises(SystemExit) as exc_info:
            share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    # Process is already dead; we shouldn't bother re-terminating it.
    serve_proc.terminate.assert_not_called()


def test_share_command_aborts_when_frpc_dies_immediately(fake_session):
    """Codex review P2: dead frpc must not produce a "share ready" banner."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = 2  # frpc exited during 3s settle window

    with patch.object(share_cli, "_spawn_serve", return_value=serve_proc), patch.object(
        share_cli, "_wait_for_healthz", return_value=True
    ), patch.object(
        share_cli.session, "request", return_value=fake_session
    ), patch.object(
        share_cli.frpc_manager, "spawn", return_value=frpc_proc
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch(
        "time.sleep"
    ), pytest.raises(SystemExit) as exc_info:
            share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    # Serve was alive (poll=None) so cleanup terminates it.
    serve_proc.terminate.assert_called_once()


def test_share_command_surfaces_relay_failure(fake_session):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    with patch.object(share_cli, "_spawn_serve", return_value=serve_proc), patch.object(
        share_cli, "_wait_for_healthz", return_value=True
    ), patch.object(
        share_cli.session,
        "request",
        side_effect=RuntimeError("relay unreachable"),
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch(
        "time.sleep"
    ), pytest.raises(SystemExit) as exc_info:
            share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    serve_proc.terminate.assert_called_once()


def test_register_adds_share_to_subparsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    share_cli.register(subparsers)
    assert "share" in subparsers.choices
    # Sanity: the subparser actually parses the canonical invocation.
    args = parser.parse_args(["share", "qwen3.5-4b"])
    assert args.command == "share"
    assert args.model == "qwen3.5-4b"
    assert args.cors_origins == "*"
    # ``--port`` defaults to None so the env-var lookup happens lazily
    # inside share_command. If we eager-int(env) at parser build time,
    # a malformed RAPID_MLX_SHARE_PORT would crash unrelated subcommands.
    assert args.port is None


def test_share_command_rejects_garbage_port_env(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_SHARE_PORT", "not-a-number")
    args = _make_args(port=None)
    with pytest.raises(SystemExit) as exc_info:
        share_cli.share_command(args)
    assert exc_info.value.code == 2


def test_share_command_rejects_out_of_range_port():
    args = _make_args(port=70000)
    with pytest.raises(SystemExit) as exc_info:
        share_cli.share_command(args)
    assert exc_info.value.code == 2


def test_share_command_rejects_explicit_port_zero():
    """``--port 0`` is a user error, not a silent fallback to 8765."""
    args = _make_args(port=0)
    with pytest.raises(SystemExit) as exc_info:
        share_cli.share_command(args)
    assert exc_info.value.code == 2


def test_spawn_serve_passes_loopback_host():
    """Codex review P2: serve must bind 127.0.0.1, not 0.0.0.0."""
    with patch("subprocess.Popen") as mock_popen, patch(
        "pathlib.Path.open"
    ) as mock_open:
        mock_open.return_value = MagicMock()
        share_cli._spawn_serve(
            alias="qwen3.5-4b",
            port=18765,
            api_key="key",
            log_path=share_cli._state_dir() / "serve.log",
            extra_args=[],
        )
    cmd = mock_popen.call_args[0][0]
    # The pair must appear adjacently — argparse rejects "--host" without a value.
    host_idx = cmd.index("--host")
    assert cmd[host_idx + 1] == "127.0.0.1"


def test_wait_for_healthz_returns_false_if_serve_exits():
    """Codex review P2: bounded poll on serve_proc.poll(), not a wall clock."""
    serve_proc = MagicMock()
    # Simulate serve exiting on the third poll without /healthz ever responding.
    serve_proc.poll.side_effect = [None, None, 1]
    with patch("urllib.request.urlopen", side_effect=ConnectionError), patch(
        "time.sleep"
    ):
        result = share_cli._wait_for_healthz(18765, serve_proc)
    assert result is False


def test_share_command_sigterm_runs_cleanup(fake_session):
    """SIGTERM (systemd / supervisor kill) must trigger the same
    cleanup as Ctrl-C, otherwise the serve + frpc children orphan and
    keep a public tunnel open after the parent dies.
    """
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    # Use a side_effect *function* so the SIGTERM is delivered exactly
    # when share_command blocks on serve_proc.wait(), not at mock setup
    # time. The SIGTERM handler installed by share_command raises
    # KeyboardInterrupt, which the existing finally-block catches.
    call_count = {"n": 0}

    def wait_then_sigterm(*_, **__):
        call_count["n"] += 1
        if call_count["n"] == 1:
            import os as _os

            _os.kill(_os.getpid(), signal.SIGTERM)
        return None

    serve_proc.wait.side_effect = wait_then_sigterm
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None

    with patch.object(share_cli, "_spawn_serve", return_value=serve_proc), patch.object(
        share_cli, "_wait_for_healthz", return_value=True
    ), patch.object(
        share_cli.session, "request", return_value=fake_session
    ), patch.object(
        share_cli.frpc_manager, "spawn", return_value=frpc_proc
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch.object(
        share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b"
    ), patch(
        "time.sleep"
    ):
        share_cli.share_command(_make_args())

    serve_proc.terminate.assert_called_once()
    frpc_proc.terminate.assert_called_once()
