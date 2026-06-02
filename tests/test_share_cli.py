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
import urllib.request
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
        thinking=False,  # default: forward --no-thinking to serve
        cors_origins=["*"],  # nargs='+' returns a list
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


def _ctrl_c_in_monitor_loop():
    """Side-effect helper: raise KeyboardInterrupt on the SECOND call to
    ``time.sleep`` and return None thereafter. Codex round-4 moved the
    blocking wait to a ``poll() + time.sleep(1)`` loop AFTER the banner,
    so the Ctrl-C signal now lands during that sleep. The pre-banner
    ``time.sleep(1)`` frpc-settle check is the FIRST call; the monitor
    loop is the second, which is where a real user would press Ctrl-C
    (the banner is up by then). Returns None on subsequent calls so
    cleanup paths that also sleep don't re-raise.
    """
    state = {"calls": 0}

    def _sleep(*_args, **_kwargs):
        state["calls"] += 1
        if state["calls"] == 2:
            raise KeyboardInterrupt
        return None

    return _sleep


# Back-compat alias for any test that referred to the original name.
_ctrl_c_on_first_sleep = _ctrl_c_in_monitor_loop


def test_share_command_happy_path(fake_session, capsys):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep", side_effect=_ctrl_c_on_first_sleep()),
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

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=False),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
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

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    # Serve was alive (poll=None) so cleanup terminates it.
    serve_proc.terminate.assert_called_once()


def test_share_command_surfaces_relay_failure(fake_session):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(
            share_cli.session,
            "request",
            side_effect=RuntimeError("relay unreachable"),
        ),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
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
    assert args.cors_origins == ["*"]
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
    with (
        patch("subprocess.Popen") as mock_popen,
        patch("pathlib.Path.open") as mock_open,
    ):
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


def test_spawn_serve_passes_api_key_via_env_not_argv():
    """DeepSeek BLOCKING round 3: ``--api-key <KEY>`` in argv leaks the
    secret to every local user via ``ps``. The bearer must travel as
    ``RAPID_MLX_API_KEY`` in the subprocess environment instead.
    """
    with (
        patch("subprocess.Popen") as mock_popen,
        patch("pathlib.Path.open") as mock_open,
    ):
        mock_open.return_value = MagicMock()
        share_cli._spawn_serve(
            alias="qwen3.5-4b",
            port=18765,
            api_key="TOPSECRETKEY",
            log_path=share_cli._state_dir() / "serve.log",
            extra_args=[],
        )
    cmd = mock_popen.call_args[0][0]
    env = mock_popen.call_args[1]["env"]
    # 1) Argv must NOT contain --api-key or the secret value anywhere.
    assert "--api-key" not in cmd, "key would be visible to `ps`"
    assert "TOPSECRETKEY" not in cmd, "secret leaked into argv"
    # 2) The env var IS set so serve can pick it up.
    assert env.get("RAPID_MLX_API_KEY") == "TOPSECRETKEY"


def test_wait_for_healthz_returns_false_if_serve_exits():
    """Codex review P2: bounded poll on serve_proc.poll(), not a wall clock."""
    serve_proc = MagicMock()
    # Simulate serve exiting on the third poll without /healthz ever responding.
    serve_proc.poll.side_effect = [None, None, 1]
    with (
        patch("urllib.request.urlopen", side_effect=ConnectionError),
        patch("time.sleep"),
    ):
        result = share_cli._wait_for_healthz(18765, serve_proc)
    assert result is False


def test_wait_for_public_url_sends_user_agent_header():
    """Cloudflare WAF returns HTTP 403 for the default ``Python-urllib/3.x``
    User-Agent. Without an explicit UA the probe loops on 403 until the
    15s deadline expires and share kills a perfectly-healthy tunnel.
    """
    captured: list[urllib.request.Request] = []

    def _capture(req, timeout):  # noqa: ARG001
        captured.append(req)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *_: None
        return resp

    with patch("urllib.request.urlopen", side_effect=_capture):
        result = share_cli._wait_for_public_url("https://abc.rapidmlx.com")

    assert result is True
    assert len(captured) == 1
    ua = captured[0].get_header("User-agent")
    assert ua, (
        "_wait_for_public_url must send a User-Agent (Cloudflare 403s the default)"
    )
    assert "python-urllib" not in ua.lower(), (
        f"User-Agent {ua!r} would be 403'd by Cloudflare WAF"
    )


def test_share_command_sigterm_runs_cleanup(fake_session):
    """SIGTERM (systemd / supervisor kill) must trigger the same
    cleanup as Ctrl-C, otherwise the serve + frpc children orphan and
    keep a public tunnel open after the parent dies.
    """
    # share_command installs its own SIGTERM handler (converting it to
    # KeyboardInterrupt for cleanup). Without restoring the original,
    # subsequent tests in the same process pick up that handler and any
    # incidental SIGTERM (CI runner timeout, fixture teardown) raises an
    # unrelated KeyboardInterrupt — flaky cross-test failure. Snapshot
    # + restore here. (Codex / DeepSeek BLOCKER #3 on PR #504.)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = 0

    # Round-4 update: the blocking wait is now a ``poll() + time.sleep(1)``
    # loop, so the SIGTERM must be delivered during a sleep call
    # (mirrors what happens in production — supervisor sends SIGTERM,
    # the installed handler raises KeyboardInterrupt). First sleep is
    # the pre-banner frpc settle, second is the monitor loop where a
    # real SIGTERM would land.
    call_count = {"n": 0}

    def sleep_then_sigterm(*_, **__):
        call_count["n"] += 1
        if call_count["n"] == 2:
            import os as _os

            _os.kill(_os.getpid(), signal.SIGTERM)
        return None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b"
        ),
        patch("time.sleep", side_effect=sleep_then_sigterm),
    ):
        try:
            share_cli.share_command(_make_args())
        finally:
            signal.signal(signal.SIGTERM, original_sigterm)

    serve_proc.terminate.assert_called_once()
    frpc_proc.terminate.assert_called_once()


def test_share_command_aborts_if_auth_gate_fails(fake_session):
    """Codex BLOCKER: if /v1/models doesn't 200 with our bearer key (e.g.
    a different process raced us to the port), we must NOT open a tunnel
    to it — that would relay random local traffic over a public URL.
    """
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        # /healthz passed but /v1/models did not — port-race scenario.
        patch.object(share_cli, "_verify_auth_gate", return_value=False),
        # session.request must NEVER be called if the auth gate failed.
        patch.object(share_cli.session, "request") as mock_session,
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    mock_session.assert_not_called()
    serve_proc.terminate.assert_called_once()


def test_share_command_surfaces_frpc_spawn_failure(fake_session):
    """DeepSeek round-4 BLOCKER #2: ``frpc_manager.spawn`` chains into
    ``ensure()`` which can raise (sha256 mismatch, download failure,
    unsupported platform). Without a try/except the user sees a raw
    traceback instead of an actionable error."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        # ensure() failure surfaces through spawn.
        patch.object(
            share_cli.frpc_manager,
            "spawn",
            side_effect=RuntimeError("frpc sha256 mismatch"),
        ),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    serve_proc.terminate.assert_called_once()


def test_share_command_aborts_if_public_url_unreachable(fake_session):
    """Codex CONCERN: frpc can stay alive (TCP connected to frps) while
    proxy registration silently fails. Don't print a banner with a URL
    that 502s.
    """
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None  # frpc still alive

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_wait_for_public_url", return_value=False),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    # BOTH children must be terminated, banner must NOT have printed.
    serve_proc.terminate.assert_called_once()
    frpc_proc.terminate.assert_called_once()


def test_banner_does_not_inline_key_in_curl_command():
    """Codex BLOCKER: the banner's copy-paste curl line must NOT embed the
    bearer key in argv (= shell history). Use ``$RAPID_MLX_SHARE_KEY``."""
    from vllm_mlx.share import warning

    out = warning.render(
        "https://abc.rapidmlx.com",
        "REAL_KEY_42",
        "mlx-community/X",
        "abc",
    )
    # The key still appears (so the user can see and export it), but the
    # curl line itself reads from the env var, not the literal key.
    assert "$RAPID_MLX_SHARE_KEY" in out
    # Specifically: the curl example must not put the literal key in
    # ``-H "Authorization: Bearer <KEY>"`` — that lands in shell history.
    assert "Bearer REAL_KEY_42" not in out


def test_banner_includes_one_click_chat_link():
    """The banner must surface a clickable chat.rapidmlx.com link with
    the subdomain + key baked into the fragment so a friend can paste
    once. Fragment-encoded so the combined token never reaches a
    server log."""
    from vllm_mlx.share import warning

    out = warning.render(
        "https://abc.rapidmlx.com",
        "REAL_KEY_42",
        "mlx-community/X",
        "abc",
    )
    expected_link = "https://chat.rapidmlx.com/#k=abc.REAL_KEY_42"
    assert expected_link in out, (
        f"banner must include the one-click chat link {expected_link!r}"
    )


def test_banner_chat_link_uses_subdomain_param_not_url_parse():
    """The subdomain is a separate parameter, not parsed out of ``url``.
    Verifies the call site passes the relay-provided subdomain rather
    than the warning module string-splitting the host. Different
    subdomain → different chat link, even if URL looks similar."""
    from vllm_mlx.share import warning

    out = warning.render(
        "https://something-other.rapidmlx.com",
        "K",
        "M",
        "xyz123",
    )
    assert "https://chat.rapidmlx.com/#k=xyz123.K" in out


def test_banner_chat_link_handles_hyphenated_subdomain():
    """Codex round-1 BLOCKING regression: the relay's subdomain charset
    permits hyphens (``^[a-z0-9][a-z0-9-]{0,62}$``), so a ``-`` delimiter
    between subdomain and key would split ambiguously on the splash
    side. The delimiter must be a character forbidden in both the
    subdomain set and the hex API key — ``.`` qualifies.
    """
    from vllm_mlx.share import warning

    out = warning.render(
        "https://foo-bar-baz.rapidmlx.com",
        "abcdef1234",
        "mlx-community/X",
        "foo-bar-baz",
    )
    assert "https://chat.rapidmlx.com/#k=foo-bar-baz.abcdef1234" in out
    # Negative assertion: the old hyphen-delimited shape must NOT appear
    # (would prove the regression came back).
    assert "#k=foo-bar-baz-abcdef1234" not in out


def test_share_command_surfaces_pick_port_failure():
    """DeepSeek round-5 BLOCKING #2: _pick_port can raise RuntimeError
    on a maxed-out ephemeral pool. Must surface as exit-1 + readable
    message, not a bare traceback."""
    with (
        patch.object(
            share_cli,
            "_pick_port",
            side_effect=RuntimeError("no free port available for share"),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


def test_resolve_served_model_name_sends_bearer():
    """Codex round-3 P2: serve is launched with --api-key, so /v1/models
    is auth-protected. Without the Authorization header the probe 401s
    and the banner silently falls back to the typed alias.
    """
    captured = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["headers"] = dict(req.header_items())
        captured["url"] = req.full_url

        class _R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"data":[{"id":"mlx-community/X"}]}'

        return _R()

    with (
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch("json.load", return_value={"data": [{"id": "mlx-community/X"}]}),
    ):
        out = share_cli._resolve_served_model_name(18765, "shhhh")

    assert out == "mlx-community/X"
    # Header keys are title-cased by urllib's Request.
    assert captured["headers"].get("Authorization") == "Bearer shhhh"


def test_share_command_exits_nonzero_when_serve_crashes(fake_session):
    """Codex round-1 BLOCKING: when serve exits with a non-zero status
    after the tunnel is up (OOM / crash), the parent must surface that
    as a non-zero exit code. Otherwise systemd / docker / supervisor
    wrappers see exit-0 and treat the failed share as a success.

    Updated for the round-4 poll-loop monitor: ``serve_proc.poll()``
    returns None first (post-spawn settle check) then 137 (in the
    main monitor loop) so the loop exits with ``serve_exit_code=137``.
    """
    serve_proc = MagicMock()
    # poll returns None on the first call (the post-spawn frpc settle
    # check), then 137 from the monitor loop onward.
    serve_proc.poll.side_effect = [None, 137, 137]
    serve_proc.wait.return_value = 137
    frpc_proc = MagicMock()
    # frpc stays alive throughout — we want the loop to break on serve,
    # not on frpc.
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1


def test_share_command_exits_when_frpc_tunnel_dies_post_banner(fake_session, capsys):
    """Codex round-4 BLOCKING: after the banner prints, blocking only on
    ``serve_proc.wait()`` meant an frpc crash / frps disconnect was
    never noticed — share kept running with a dead public URL while the
    local model stayed exposed. The monitor must watch BOTH children
    after the banner and exit (terminating serve) the moment either
    dies.
    """
    serve_proc = MagicMock()
    # serve stays alive throughout — we want frpc to be the trigger.
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    # poll returns None on the first call (post-spawn settle), then
    # 1 (crashed) on the monitor-loop call.
    frpc_proc.poll.side_effect = [None, 1, 1]
    frpc_proc.wait.return_value = 1

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "frpc tunnel exited" in err
    # serve must be cleaned up — we don't want to leave the local
    # endpoint running with no tunnel covering it.
    serve_proc.terminate.assert_called_once()


def test_share_command_ctrl_c_keeps_exit_zero(fake_session):
    """Companion to the crash-exit test: pressing Ctrl-C is a user-driven
    shutdown and must NOT be conflated with a serve crash. Exit code
    stays 0 in that path."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch("time.sleep", side_effect=_ctrl_c_on_first_sleep()),
    ):
        # Should NOT raise SystemExit — Ctrl-C is a clean stop.
        share_cli.share_command(_make_args())


def test_share_command_runs_download_gate_for_uncached_hf_repo():
    """Codex round-1 BLOCKING: ``rapid-mlx share <uncached HF repo>``
    must run the same B2 confirmation gate that chat/run/serve/pull/bench
    do. Without it, a non-interactive child silently kicks off a
    multi-GB download.

    Verifies the share-side helper calls ``confirm_or_abort`` via the
    download-gate module when the repo isn't cached. We patch
    ``is_repo_cached`` to False and assert ``confirm_or_abort`` is
    invoked exactly once with the repo id."""
    fake_confirm = MagicMock()
    with (
        patch("sys.stdin") as fake_stdin,
        patch.dict("os.environ", {}, clear=False),
        patch("vllm_mlx._download_gate.is_repo_cached", return_value=False),
        patch("vllm_mlx._download_gate.estimate_repo_size_bytes", return_value=42),
        patch("vllm_mlx._download_gate.confirm_or_abort", fake_confirm),
    ):
        fake_stdin.isatty.return_value = True
        # Remove the auto-pull env override if it's set in the runtime
        # so the prompt path is the one under test.
        import os as _os

        _os.environ.pop("RAPID_MLX_AUTO_PULL", None)
        _os.environ.pop("RAPID_MLX_CHAT_SPAWN", None)
        share_cli._maybe_confirm_download("mlx-community/Qwen3.5-4B-MLX-4bit")
    fake_confirm.assert_called_once()
    assert fake_confirm.call_args.args[0] == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_share_command_skips_download_gate_for_local_alias():
    """Local short aliases (``qwen3.5-4b``) don't have a ``/`` so they
    can never trigger an HF API round-trip. Verify the helper
    short-circuits without importing ``_download_gate``."""
    with patch(
        "vllm_mlx._download_gate.is_repo_cached",
        side_effect=AssertionError("should not be called"),
    ):
        share_cli._maybe_confirm_download("qwen3.5-4b")


def test_share_command_skips_download_gate_when_env_override_set():
    """``RAPID_MLX_AUTO_PULL=1`` is the documented opt-out for CI / cron
    / docker. The share gate must honor it the same way the top-level
    one does — no HF API call, no prompt."""
    with (
        patch.dict("os.environ", {"RAPID_MLX_AUTO_PULL": "1"}, clear=False),
        patch(
            "vllm_mlx._download_gate.is_repo_cached",
            side_effect=AssertionError("should not be called"),
        ),
    ):
        share_cli._maybe_confirm_download("mlx-community/Qwen3.5-4B-MLX-4bit")


def test_register_share_cors_origins_accepts_multiple_values():
    """Codex round-4 P3: ``rapid-mlx serve --cors-origins`` accepts
    multiple values via ``nargs='+'``. The share wrapper must accept
    the same shape so configurations carry over verbatim.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    share_cli.register(subparsers)
    args = parser.parse_args(
        ["share", "qwen3.5-4b", "--cors-origins", "http://a", "http://b"]
    )
    assert args.cors_origins == ["http://a", "http://b"]


def test_share_command_forwards_multiple_cors_origins_to_child(fake_session, capsys):
    """End-to-end: a multi-value ``--cors-origins`` lands in the child
    ``serve`` argv as ``--cors-origins http://a http://b`` (separate
    argv elements, not a single concatenated string)."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    captured = {}

    def fake_spawn_serve(**kwargs):
        captured.update(kwargs)
        return serve_proc

    with (
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn_serve),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch("time.sleep", side_effect=_ctrl_c_on_first_sleep()),
    ):
        share_cli.share_command(
            _make_args(cors_origins=["http://a", "http://b"]),
        )

    extra = captured["extra_args"]
    # Both values must appear as separate elements after --cors-origins.
    idx = extra.index("--cors-origins")
    assert extra[idx + 1 : idx + 3] == ["http://a", "http://b"]


def test_share_command_forwards_original_alias_to_child(fake_session, capsys):
    """Codex round-3 BLOCKING: ``main()`` rewrites ``args.model`` to the
    HF repo before dispatching to share, stashing the user-typed alias
    on ``args._original_alias``. share_command must forward the
    ORIGINAL alias to the child ``serve`` subprocess — otherwise the
    child runs without ``_model_alias`` and the public ``/v1/models``
    advertises the HF id instead of the short name the user typed,
    breaking clients configured with that alias.
    """
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    args = _make_args()
    # Simulate what cli.main() does before dispatch: rewrite to HF repo
    # and stash the typed alias.
    args.model = "mlx-community/Qwen3.5-4B-MLX-4bit"
    args._original_alias = "qwen3.5-4b"

    captured = {}

    def fake_spawn_serve(**kwargs):
        captured.update(kwargs)
        return serve_proc

    with (
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn_serve),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(args)

    # Child gets the SHORT alias, not the HF id — so its own main()
    # rerun lands on the same _model_alias path as ``rapid-mlx serve``.
    assert captured["alias"] == "qwen3.5-4b"


def test_share_command_falls_back_to_args_model_when_no_original_alias(
    fake_session,
):
    """When ``rapid-mlx share <raw HF id>`` is typed directly (no alias),
    ``_original_alias`` is never set. The share command must fall back
    to ``args.model`` rather than crashing on the missing attribute or
    forwarding ``None`` to the child."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    frpc_proc = MagicMock()
    frpc_proc.poll.return_value = None
    frpc_proc.wait.return_value = None

    args = _make_args(model="mlx-community/SomeRepo-MLX-4bit")
    # No _original_alias — this is the raw-HF-id path.
    captured = {}

    def fake_spawn_serve(**kwargs):
        captured.update(kwargs)
        return serve_proc

    with (
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn_serve),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_wait_for_public_url", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(share_cli.frpc_manager, "spawn", return_value=frpc_proc),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch("time.sleep", side_effect=_ctrl_c_on_first_sleep()),
    ):
        share_cli.share_command(args)

    assert captured["alias"] == "mlx-community/SomeRepo-MLX-4bit"


def test_share_command_surfaces_frpc_oserror(fake_session):
    """Codex round-3: a local filesystem / exec error during frpc spawn
    (disk full while extracting, PermissionError on the cache dir,
    FileNotFoundError on the binary) raises OSError, which the prior
    handler didn't catch — escaping as a raw traceback AFTER the serve
    child was already started. The handler must cover OSError too."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.session, "request", return_value=fake_session),
        patch.object(
            share_cli.frpc_manager,
            "spawn",
            side_effect=PermissionError("read-only cache dir"),
        ),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch("time.sleep"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())

    assert exc_info.value.code == 1
    # The serve child was already up — must be terminated on the way
    # out so we don't leak a process.
    serve_proc.terminate.assert_called_once()


def test_verify_auth_gate_rejects_unauthenticated_server():
    """Codex round-2 BLOCKING: a process started WITHOUT --api-key (e.g.
    a different OpenAI-compatible server that won the port race) returns
    200 for any Authorization header because there is no auth. Our gate
    must NOT trust that 200 — otherwise we'd open a public tunnel to
    someone else's process.

    Simulate the no-auth path: every probe (bad key + good key) returns
    200. The gate must report False.
    """

    class _OkResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    with patch("urllib.request.urlopen", return_value=_OkResp()):
        assert share_cli._verify_auth_gate(18765, "real-key") is False


def test_verify_auth_gate_accepts_properly_protected_server():
    """The positive path: the bad key returns 401 and the real key
    returns 200. That ordering — auth IS enforced, AND we hold the
    correct key — is the only safe state for the gate to allow."""
    call_count = {"n": 0}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        call_count["n"] += 1
        auth = dict(req.header_items()).get("Authorization", "")
        if auth == "Bearer real-key":

            class _Ok:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

            return _Ok()
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", hdrs=None, fp=None
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert share_cli._verify_auth_gate(18765, "real-key") is True
    # Both probes ran: bad-key check + real-key confirmation.
    assert call_count["n"] == 2


def test_verify_auth_gate_handles_unreachable_endpoint():
    """If the answering process is offline / TCP-resets between probes
    the gate must return False (defensive default), not crash."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        assert share_cli._verify_auth_gate(18765, "real-key") is False


def test_share_command_skips_download_gate_for_chat_spawn_child():
    """When the chat REPL spawns ``rapid-mlx share`` as a child, the
    parent has already run the gate and set ``RAPID_MLX_CHAT_SPAWN=1``.
    Re-prompting in the child would deadlock on the non-TTY stdin path
    — mirror the top-level CLI's grandchild safety."""
    with (
        patch.dict("os.environ", {"RAPID_MLX_CHAT_SPAWN": "1"}, clear=False),
        patch(
            "vllm_mlx._download_gate.is_repo_cached",
            side_effect=AssertionError("should not be called"),
        ),
    ):
        share_cli._maybe_confirm_download("mlx-community/Qwen3.5-4B-MLX-4bit")
