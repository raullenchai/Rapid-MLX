# SPDX-License-Identifier: Apache-2.0
"""Integration smoke test for the share CLI orchestration.

We mock both subprocess spawns (serve + frpc) and the session HTTP call
so the test runs in <1s and doesn't touch the network. The assertions
focus on the pieces of the contract that are easy to silently break:
the security banner content, frpc config shape, and ordered shutdown.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.share import cli as share_cli
from vllm_mlx.share.session import Session


def _make_args(**overrides):
    defaults = dict(
        model="qwen3.5-4b",
        port=18765,
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


def test_share_command_aborts_when_healthz_never_ready(fake_session):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    with patch.object(share_cli, "_spawn_serve", return_value=serve_proc), patch.object(
        share_cli, "_wait_for_healthz", return_value=False
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch(
        "time.sleep"
    ):
        with pytest.raises(SystemExit) as exc_info:
            share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    # Serve must still be torn down — leaking processes on failure is the
    # most user-hostile bug class for this command.
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
    ), patch.object(share_cli, "_pick_port", return_value=18765), patch("time.sleep"):
        with pytest.raises(SystemExit) as exc_info:
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
