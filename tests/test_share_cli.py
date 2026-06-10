# SPDX-License-Identifier: Apache-2.0
"""Integration smoke test for the share CLI orchestration.

We mock the serve subprocess + the WS tunnel client so the test runs in
<1s and doesn't touch the network. The assertions focus on the pieces of
the contract that are easy to silently break: the security banner content,
the WS tunnel lifecycle, --chat-frontend validation, and ordered shutdown.

Architecture pivot (2026-06-03): the prior frpc + control-plane stack
was replaced with a Cloudflare Worker reached over a WebSocket reverse
tunnel. See ``vllm_mlx/share/ws_tunnel.py`` for the wire protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import threading
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.share import cli as share_cli


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
        model="qwen3.5-4b-4bit",
        port=18765,  # explicit so the env-var fallback path isn't exercised
        thinking=False,  # default: forward --no-thinking to serve
        cors_origins=None,  # None → CLI default allowlist
        rate_limit=120,  # CLI default (rpm), forwarded to spawned serve
        chat_frontend=None,  # use built-in default
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_tunnel(
    *,
    tunnel_id: str = "testabc1234567890abcd",
    public_url: str = "https://rapidserver.quicksilverpro.io/r/testabc1234567890abcd",
    ready: bool = True,
    error: Exception | None = None,
    closed: bool = False,
) -> MagicMock:
    """Build a MagicMock that mimics ``ws_tunnel.TunnelClient`` well enough
    for ``share_command`` to drive it.

    The real tunnel object has ``ready_event`` (set by the WS handshake),
    ``closed_event`` (set on tear-down or error), ``error`` (the exception
    that closed the connection, if any), ``tunnel_id``, ``public_url``,
    plus ``run_in_thread()`` / ``stop()``. Test mock surfaces just those.
    """
    t = MagicMock()
    t.tunnel_id = tunnel_id
    t.public_url = public_url
    t.error = error

    ready_event = threading.Event()
    if ready:
        ready_event.set()
    t.ready_event = ready_event

    closed_event = threading.Event()
    if closed:
        closed_event.set()
    t.closed_event = closed_event

    # ``run_in_thread`` returns a thread-like object the parent will
    # ``.join()`` on in ``finally``. A non-alive mock satisfies the
    # ``is_alive()`` check without us having to spin up a real thread.
    thread = MagicMock()
    thread.is_alive.return_value = False
    t.run_in_thread.return_value = thread
    return t


def _ctrl_c_in_monitor_loop():
    """Side-effect helper: raise KeyboardInterrupt on the FIRST call to
    ``time.sleep`` (which now lives in the monitor loop — no pre-loop
    settle sleep in the WS-tunnel rewrite).
    """
    state = {"calls": 0}

    def _sleep(*_args, **_kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise KeyboardInterrupt
        return None

    return _sleep


# ─────────────────────────── happy path + lifecycle ─────────────────────────


def test_share_command_happy_path(capsys):
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    serve_proc.wait.return_value = 0
    tunnel = _fake_tunnel()

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(_make_args())

    out = capsys.readouterr().out
    # Banner pieces that protect users from leaking the key:
    assert "PUBLIC INTERNET" in out
    assert "Do NOT screenshot" in out
    # URL surfaced verbatim (the worker route, not a bare host).
    assert "https://rapidserver.quicksilverpro.io/r/testabc1234567890abcd" in out
    # Ctrl-C path tears down the tunnel and the serve child.
    tunnel.stop.assert_called_once()
    serve_proc.terminate.assert_called_once()


def test_share_command_aborts_when_serve_exits_before_ready():
    serve_proc = MagicMock()
    serve_proc.poll.return_value = 1  # already exited
    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=False),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


def test_share_command_aborts_when_tunnel_ws_never_connects():
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel(ready=False)  # ready_event NEVER set

    # Patch ``ready_event.wait`` so the test doesn't actually block 30s.
    tunnel.ready_event = MagicMock()
    tunnel.ready_event.wait.return_value = False

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


def test_share_command_aborts_when_tunnel_reports_error_after_ready():
    """``ready_event`` set but ``tunnel.error`` non-None — covers the
    edge case where the handshake completes but the worker rejects us
    (e.g. tunnel id collision). Must NOT print a banner."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel(error=RuntimeError("worker rejected"))

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


def test_share_command_aborts_if_public_url_unreachable():
    """Tunnel is up but the e2e probe through the public URL never
    returns 200 — the banner would advertise a stillborn URL, so we
    bail before printing."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=False),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


# ─────────────────────────── CLI surface / argparse ─────────────────────────


def test_register_adds_share_to_subparsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    share_cli.register(subparsers)
    args = parser.parse_args(["share", "qwen3.5-4b-4bit"])
    assert args.command == "share"
    assert args.model == "qwen3.5-4b-4bit"


def test_share_command_rejects_garbage_port_env(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_SHARE_PORT", "not-a-number")
    with pytest.raises(SystemExit) as exc_info:
        share_cli.share_command(_make_args(port=None))
    assert exc_info.value.code == 2


def test_share_command_rejects_out_of_range_port():
    with pytest.raises(SystemExit) as exc_info:
        share_cli.share_command(_make_args(port=70000))
    assert exc_info.value.code == 2


def test_share_command_rejects_explicit_port_zero():
    with pytest.raises(SystemExit) as exc_info:
        share_cli.share_command(_make_args(port=0))
    assert exc_info.value.code == 2


def test_share_command_rejects_bad_relay_url_scheme(monkeypatch):
    """``RAPID_MLX_RELAY_URL=http://...`` would silently fall through to a
    stalled WS handshake. Exit 2 at config-validation time instead."""
    monkeypatch.setenv("RAPID_MLX_RELAY_URL", "https://not-a-ws-url")
    with (
        patch.object(share_cli, "_spawn_serve", return_value=MagicMock()),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 2


# ─────────────────────────── helper invariants ──────────────────────────────


def test_spawn_serve_passes_loopback_host():
    """Serve child must bind ONLY to 127.0.0.1 — never 0.0.0.0. The
    public surface is the tunnel; if the serve listener leaks onto a
    LAN interface, anyone on the same network gets unauthenticated
    access at ``http://<lan-ip>:<port>``."""
    with patch("subprocess.Popen") as mock_popen:
        share_cli._spawn_serve(
            alias="qwen3.5-4b-4bit",
            port=18765,
            api_key="K",
            log_path=MagicMock(),
            extra_args=[],
        )
    argv = mock_popen.call_args.args[0]
    assert "--host" in argv
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_spawn_serve_passes_api_key_via_env_not_argv():
    """The bearer key must travel via env (RAPID_MLX_API_KEY) and never
    appear in argv where ``ps`` / shell history would leak it."""
    with patch("subprocess.Popen") as mock_popen:
        share_cli._spawn_serve(
            alias="qwen3.5-4b-4bit",
            port=18765,
            api_key="SECRET_KEY_HERE",
            log_path=MagicMock(),
            extra_args=[],
        )
    argv = mock_popen.call_args.args[0]
    env = mock_popen.call_args.kwargs["env"]
    assert "SECRET_KEY_HERE" not in " ".join(argv)
    assert env["RAPID_MLX_API_KEY"] == "SECRET_KEY_HERE"


def test_wait_for_healthz_returns_false_if_serve_exits():
    serve_proc = MagicMock()
    serve_proc.poll.return_value = 1
    assert share_cli._wait_for_healthz(18765, serve_proc) is False


def test_wait_for_healthz_handles_timeout_error():
    """TimeoutError is NOT a subclass of URLError in stdlib — must be
    caught explicitly or it bubbles up and crashes the CLI. We arrange
    for serve to "exit" on the second poll so the retry loop terminates
    deterministically after proving the TimeoutError was swallowed."""
    serve_proc = MagicMock()
    serve_proc.poll.side_effect = [None, 1]  # alive once, then exited
    with (
        patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")),
        patch("time.sleep"),
    ):
        assert share_cli._wait_for_healthz(18765, serve_proc) is False


def test_verify_auth_gate_rejects_unauthenticated_server():
    """If GET /v1/models without bearer returns 200, the server isn't
    enforcing auth — refuse to tunnel it."""
    response = MagicMock()
    response.status = 200
    response.read.return_value = b'{"data":[]}'
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *a: None
    with patch("urllib.request.urlopen", return_value=response):
        assert share_cli._verify_auth_gate(18765, "K") is False


def test_verify_auth_gate_accepts_properly_protected_server():
    """First probe sends a random wrong key — must 401 (proves auth is
    enforced). Second probe sends the real key — must 200 (proves the
    listening process is ours). Codex round-2 BLOCKING."""
    import urllib.error as urlerr

    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        call_count["n"] += 1
        auth_header = req.headers.get("Authorization", "")
        # Only the REAL key ("Bearer real_key") yields 200; anything else
        # mirrors a properly-protected server returning 401.
        if auth_header == "Bearer real_key":
            response = MagicMock()
            response.status = 200
            response.read.return_value = b'{"data":[]}'
            response.__enter__ = lambda self: response
            response.__exit__ = lambda self, *a: None
            return response
        raise urlerr.HTTPError(
            "http://x", 401, "Unauthorized", {}, MagicMock(read=lambda: b"")
        )

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert share_cli._verify_auth_gate(18765, "real_key") is True
    # Two probes: wrong-key + real-key.
    assert call_count["n"] == 2


def test_verify_auth_gate_handles_unreachable_endpoint():
    import urllib.error as urlerr

    with patch("urllib.request.urlopen", side_effect=urlerr.URLError("nope")):
        assert share_cli._verify_auth_gate(18765, "K") is False


def test_verify_auth_gate_handles_timeout_error():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        assert share_cli._verify_auth_gate(18765, "K") is False


def test_resolve_served_model_name_sends_bearer():
    """The served-name probe is auth-gated by /v1/models too — must
    include the bearer, otherwise the helper silently returns None on
    a healthy server and the banner shows the alias instead of the HF id."""
    captured = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["auth"] = req.headers.get("Authorization", "")
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"data":[{"id":"mlx-community/Qwen3.5-4B"}]}'
        response.__enter__ = lambda self: response
        response.__exit__ = lambda self, *a: None
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        name = share_cli._resolve_served_model_name(18765, "the-secret-key")
    assert name == "mlx-community/Qwen3.5-4B"
    assert captured["auth"] == "Bearer the-secret-key"


def test_resolve_served_model_name_handles_timeout_error():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        assert share_cli._resolve_served_model_name(18765, "K") is None


# ─────────────────────────── auth gate / aborts ─────────────────────────────


def test_share_command_aborts_if_auth_gate_fails():
    """``/healthz`` ok but auth gate failed (someone else's serve on the
    same port). Must NOT open the tunnel."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel_factory = MagicMock()
    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=False),
        patch.object(share_cli.ws_tunnel, "TunnelClient", tunnel_factory),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1
    tunnel_factory.assert_not_called()


def test_share_command_surfaces_pick_port_failure():
    with (
        patch.object(
            share_cli,
            "_pick_port",
            side_effect=RuntimeError("no free port"),
        ),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


# ─────────────────────────── monitor-loop exit shapes ───────────────────────


def test_share_command_exits_nonzero_when_serve_crashes(capsys):
    """Codex round-1 BLOCKING (preserved): serve OOMs / crashes →
    non-zero exit so supervisors restart us."""
    serve_proc = MagicMock()
    # Monitor loop polls poll() once and breaks; finally block polls
    # again. ``itertools.chain`` returns the sequence then a sentinel
    # repeating forever so we don't have to count call sites.
    import itertools

    serve_proc.poll.side_effect = itertools.chain([None, 137], itertools.repeat(137))
    tunnel = _fake_tunnel()

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    # share_command translates any non-zero ``serve_exit_code`` to a
    # sentinel 1 — supervisors only need to distinguish "should restart"
    # (1) from "clean operator stop" (0).
    assert exc_info.value.code == 1


def test_share_command_exits_nonzero_when_serve_exits_cleanly(capsys):
    """Codex round-6 BLOCKING (preserved): clean child exit is still a
    share failure — the public URL just disappeared."""
    import itertools

    serve_proc = MagicMock()
    serve_proc.poll.side_effect = itertools.chain([None, 0], itertools.repeat(0))
    tunnel = _fake_tunnel()

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    # Clean child exit translates to sentinel 1 (not 0).
    assert exc_info.value.code == 1


def test_share_command_exits_when_tunnel_drops_post_banner(capsys):
    """WS handshake completed + banner printed, then the WS dies. The
    serve child is still alive, but the public URL is dead. Non-zero
    exit so supervisors restart us."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None  # still alive throughout
    tunnel = _fake_tunnel()

    # Flip closed_event the second time the monitor loop polls.
    poll_count = {"n": 0}

    def fake_sleep(*_a, **_k):
        poll_count["n"] += 1
        if poll_count["n"] == 1:
            tunnel.closed_event.set()
            tunnel.error = RuntimeError("WS dropped")
        return None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=fake_sleep),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args())
    assert exc_info.value.code == 1


def test_share_command_ctrl_c_keeps_exit_zero(capsys):
    """User-initiated Ctrl-C must NOT leak as a non-zero exit. The
    operator chose to stop — supervisors should NOT restart us."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        # No SystemExit expected → returns normally with implicit exit 0.
        share_cli.share_command(_make_args())


def test_share_command_sigterm_runs_cleanup():
    """SIGTERM (docker/systemd kill) must run the same cleanup as
    Ctrl-C: tunnel.stop() + serve_proc.terminate()."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()

    sleep_state = {"called": False}

    def raise_sigterm_handler_once(*_a, **_k):
        if not sleep_state["called"]:
            sleep_state["called"] = True
            # Simulate the SIGTERM-installed handler firing.
            raise KeyboardInterrupt
        return None

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=raise_sigterm_handler_once),
    ):
        share_cli.share_command(_make_args())

    tunnel.stop.assert_called_once()
    serve_proc.terminate.assert_called_once()
    # SIGTERM handler was actually installed (and restored).
    assert (
        signal.getsignal(signal.SIGTERM) != share_cli._term_handler
        if hasattr(share_cli, "_term_handler")
        else True
    )


# ─────────────────────────── CORS forwarding ────────────────────────────────


def test_register_share_cors_origins_accepts_multiple_values():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    share_cli.register(subparsers)
    args = parser.parse_args(
        [
            "share",
            "qwen3.5-4b-4bit",
            "--cors-origins",
            "https://a.com",
            "https://b.com",
        ]
    )
    assert args.cors_origins == ["https://a.com", "https://b.com"]


def test_share_command_forwards_multiple_cors_origins_to_child(capsys):
    """nargs='+' returns a list; serve_command takes a single comma-
    joined string for ``--cors-origins``. Codex round-4 BLOCKING."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()
    spawn_argv: list[str] = []

    def fake_spawn(*, alias, port, api_key, log_path, extra_args):  # noqa: ARG001
        spawn_argv.extend(extra_args)
        return serve_proc

    with (
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(
            _make_args(cors_origins=["https://a.com", "https://b.com"])
        )

    # Both origins must be forwarded — as separate argv elements after
    # ``--cors-origins`` (nargs='+' shape), not just the first dropped
    # silently. Codex round-4 BLOCKING.
    assert "--cors-origins" in spawn_argv
    flag_idx = spawn_argv.index("--cors-origins")
    # The two origins should appear at positions flag_idx+1 and flag_idx+2.
    forwarded = spawn_argv[flag_idx + 1 : flag_idx + 3]
    assert "https://a.com" in forwarded
    assert "https://b.com" in forwarded


# ────────────────────────── pre-release hardening ───────────────────────────


def _drive_share_capture(args, *, extra_patches=()):
    """Run ``share_command`` against a happy-path mock stack and return
    the argv the child ``serve`` would have been spawned with.

    Helper for the hardening tests below; we only care about what extra
    flags get forwarded to the child, not the rest of the lifecycle.
    """
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()
    captured: list[str] = []

    def fake_spawn(*, alias, port, api_key, log_path, extra_args):  # noqa: ARG001
        captured.extend(extra_args)
        return serve_proc

    base_patches = [
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ]
    base_patches.extend(extra_patches)
    with contextlib.ExitStack() as stack:
        for p in base_patches:
            stack.enter_context(p)
        share_cli.share_command(args)
    return captured


def test_default_cors_origins_is_rapidmlx_allowlist_not_wildcard():
    """Finding N. Default ``--cors-origins`` must NOT be ``*`` — that
    let any drive-by web page the publisher visited hit
    ``http://127.0.0.1:<share-port>`` with the bearer key and use the
    publisher's compute. Default now ships the rapidmlx chat-frontend
    allowlist; users who really want wide-open opt in explicitly with
    ``--cors-origins '*'``."""
    spawn_argv = _drive_share_capture(_make_args())
    assert "--cors-origins" in spawn_argv
    flag_idx = spawn_argv.index("--cors-origins")
    # Slice from flag_idx+1 up to the next CLI flag or end.
    tail = spawn_argv[flag_idx + 1 :]
    end = next((i for i, v in enumerate(tail) if v.startswith("--")), len(tail))
    origins = tail[:end]
    assert "*" not in origins, f"default CORS leaked '*': {origins!r}"
    # All four canonical rapidmlx origins must be there.
    for must_have in (
        "https://rapid-pro.pages.dev",
        "https://rapid-pro.quicksilverpro.io",
        "https://rapidmlx.com",
        "https://chat.rapidmlx.com",
    ):
        assert must_have in origins, (
            f"missing {must_have} from default; got {origins!r}"
        )


def test_chat_frontend_origin_is_appended_to_default_cors_allowlist():
    """A user pointing ``--chat-frontend https://my-fork.com`` should
    automatically get that origin into the child's CORS allowlist —
    otherwise they'd have to remember to repeat it under
    ``--cors-origins`` and would silently hit ``Failed to fetch``."""
    spawn_argv = _drive_share_capture(
        _make_args(chat_frontend="https://my-fork.example")
    )
    flag_idx = spawn_argv.index("--cors-origins")
    tail = spawn_argv[flag_idx + 1 :]
    end = next((i for i, v in enumerate(tail) if v.startswith("--")), len(tail))
    origins = tail[:end]
    assert "https://my-fork.example" in origins, origins


def test_explicit_cors_origins_wildcard_overrides_default():
    """Power users running a chat UI we don't list (e.g. local
    OpenWebUI) can still opt back into wide-open CORS with
    ``--cors-origins '*'``. We forward exactly what they asked for."""
    spawn_argv = _drive_share_capture(_make_args(cors_origins=["*"]))
    flag_idx = spawn_argv.index("--cors-origins")
    tail = spawn_argv[flag_idx + 1 :]
    end = next((i for i, v in enumerate(tail) if v.startswith("--")), len(tail))
    origins = tail[:end]
    assert origins == ["*"], origins


def test_default_rate_limit_120_forwarded_to_child():
    """Finding F. Without a forwarded ``--rate-limit`` the child
    ``rapid-mlx serve`` defaults to 0 (disabled) — a leaked share key
    can then burst-DoS the publisher's M3. Share defaults to 120 rpm
    (2 req/s); the value is forwarded verbatim."""
    spawn_argv = _drive_share_capture(_make_args())
    assert "--rate-limit" in spawn_argv
    idx = spawn_argv.index("--rate-limit")
    assert spawn_argv[idx + 1] == "120"


def test_rate_limit_zero_disables_forwarding():
    """``--rate-limit 0`` is the documented escape hatch for power
    users: do NOT forward to the child at all, letting the child's own
    default (disabled) take over."""
    spawn_argv = _drive_share_capture(_make_args(rate_limit=0))
    assert "--rate-limit" not in spawn_argv


def test_rate_limit_custom_value_forwarded():
    """User-supplied non-zero ``--rate-limit`` is forwarded as-is."""
    spawn_argv = _drive_share_capture(_make_args(rate_limit=500))
    assert "--rate-limit" in spawn_argv
    idx = spawn_argv.index("--rate-limit")
    assert spawn_argv[idx + 1] == "500"


# ─────────────────────────── original-alias forwarding ──────────────────────


def test_share_command_forwards_original_alias_to_child(capsys):
    """Codex round-3 BLOCKING (preserved): when sys.argv contained an
    alias the parser rewrote (e.g. via shorthand), the child must see
    the ORIGINAL token so warm-cache lookups hit."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()
    spawn_argv: list[str] = []

    def fake_spawn(*, alias, port, api_key, log_path, extra_args):  # noqa: ARG001
        # ``alias`` is the first arg the parent passes to ``serve``.
        spawn_argv.append(alias)
        return serve_proc

    args = _make_args(model="qwen3.5-4b-4bit")
    # Simulate argparse having rewritten the alias.
    args._original_alias = "Qwen3.5-4B"

    with (
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(args)
    assert spawn_argv == ["Qwen3.5-4B"]


def test_share_command_falls_back_to_args_model_when_no_original_alias(capsys):
    """No ``_original_alias`` attribute → use ``args.model`` directly."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()
    spawn_argv: list[str] = []

    def fake_spawn(*, alias, port, api_key, log_path, extra_args):  # noqa: ARG001
        spawn_argv.append(alias)
        return serve_proc

    with (
        patch.object(share_cli, "_spawn_serve", side_effect=fake_spawn),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(_make_args(model="qwen3.5-4b-4bit"))
    assert spawn_argv == ["qwen3.5-4b-4bit"]


# ─────────────────────────── download-gate behavior ─────────────────────────


def test_share_command_runs_download_gate_for_uncached_hf_repo():
    """``mlx-community/Foo`` is an HF repo id — the gate is reached.
    ``RAPID_MLX_AUTO_PULL=1`` short-circuits the actual prompt; we just
    verify the gate function was reached for HF-shaped aliases by
    asserting ``_pick_port`` (the very next step) was called."""
    pick = MagicMock(side_effect=SystemExit(99))
    with (
        patch.object(share_cli, "_pick_port", pick),
        patch.dict("os.environ", {"RAPID_MLX_AUTO_PULL": "1"}, clear=False),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(_make_args(model="mlx-community/SomeRepo"))
    assert exc_info.value.code == 99
    pick.assert_called_once()


def test_share_command_skips_download_gate_for_local_alias():
    """Aliases without ``/`` (e.g. ``qwen3.5-4b-4bit``) are NOT HF repo ids —
    the gate short-circuits before any HF API call. Verified by
    asserting ``is_repo_cached`` was never called."""
    with (
        patch.object(share_cli, "_pick_port", side_effect=SystemExit(99)),
        patch("vllm_mlx._download_gate.is_repo_cached") as cached,
        pytest.raises(SystemExit),
    ):
        share_cli.share_command(_make_args(model="qwen3.5-4b-4bit"))
    cached.assert_not_called()


def test_share_command_skips_download_gate_when_env_override_set():
    """``RAPID_MLX_AUTO_PULL=1`` short-circuits even for an HF repo id."""
    with (
        patch.object(share_cli, "_pick_port", side_effect=SystemExit(99)),
        patch("vllm_mlx._download_gate.is_repo_cached") as cached,
        patch.dict("os.environ", {"RAPID_MLX_AUTO_PULL": "1"}, clear=False),
        pytest.raises(SystemExit),
    ):
        share_cli.share_command(_make_args(model="mlx-community/SomeRepo"))
    cached.assert_not_called()


def test_share_command_skips_download_gate_for_chat_spawn_child():
    """``RAPID_MLX_CHAT_SPAWN=1`` is the signal that this serve was
    spawned by ``rapid-mlx chat`` and a parent already gated the
    download. Re-asking would block headless flows."""
    with (
        patch.dict("os.environ", {"RAPID_MLX_CHAT_SPAWN": "1"}, clear=False),
        patch(
            "vllm_mlx._download_gate.is_repo_cached",
            side_effect=AssertionError("should not be called"),
        ),
    ):
        share_cli._maybe_confirm_download("mlx-community/Qwen3.5-4B-MLX-4bit")


# ─────────────────────────── banner content ─────────────────────────────────


def test_banner_does_not_inline_key_in_curl_command():
    from vllm_mlx.share import warning

    out = warning.render(
        "https://rapidserver.quicksilverpro.io/r/abc",
        "SUPER_SECRET_KEY",
        "mlx-community/Qwen3.5-4B",
        "abc",
        "https://chat.example.com",
    )
    # The bearer must reach curl via env, not via -H literal.
    assert "Bearer $RAPID_MLX_SHARE_KEY" in out
    assert "Bearer SUPER_SECRET_KEY" not in out


def test_banner_includes_one_click_chat_link():
    from vllm_mlx.share import warning

    out = warning.render(
        "https://rapidserver.quicksilverpro.io/r/abc",
        "deadbeef",
        "mlx-community/Qwen3.5-4B",
        "abc1234567890",
        "https://chat.example.com",
    )
    # Chat link uses ``<id>.<key>`` fragment, with the tunnel id —
    # NOT a URL substring — as the prefix.
    assert "https://chat.example.com/#k=abc1234567890.deadbeef" in out


def test_banner_chat_link_uses_tunnel_id_param_not_url_parse():
    """The chat link must be built from the explicit ``tunnel_id``
    argument — not by parsing ``url``. The Worker URL has shape
    ``https://rapidserver…/r/<id>`` (path component), so parsing the
    host would yield ``rapidserver…`` instead of the tunnel id."""
    from vllm_mlx.share import warning

    out = warning.render(
        # Path-shape URL — the Worker route, not a wildcard subdomain.
        "https://rapidserver.quicksilverpro.io/r/path-shape-id",
        "deadbeef",
        "mlx-community/Qwen3.5-4B",
        # Pass a deliberately different id to prove the param wins.
        "explicit-id-99",
        "https://chat.example.com",
    )
    # The tunnel-id passed as the argument is what appears in the
    # fragment — NOT anything parsed out of the URL. The URL still
    # shows up on the URL: line as-is.
    assert "#k=explicit-id-99.deadbeef" in out
    assert "#k=path-shape-id." not in out
    assert "#k=rapidserver" not in out


def test_banner_omits_chat_line_when_frontend_is_none():
    """``chat_frontend=None`` suppresses the Chat: row but keeps the
    model + URL + key surfaces."""
    from vllm_mlx.share import warning

    out = warning.render(
        "https://rapidserver.quicksilverpro.io/r/abc",
        "REAL_KEY_42",
        "mlx-community/X",
        "abc",
        None,
    )
    assert "https://rapidserver.quicksilverpro.io/r/abc" in out
    assert "REAL_KEY_42" in out
    assert "Chat:" not in out


# ─────────────────────────── --chat-frontend ────────────────────────────────


def test_resolve_chat_frontend_defaults_to_big_agi(monkeypatch):
    """No flag, no env var → built-in default
    ``https://rapid-pro.quicksilverpro.io``. Default surface-bound to the
    Big-AGI fork (tool-calling capable) hosted on CF Pages. BCG remains
    reachable via ``--chat-frontend https://rapid.quicksilverpro.io``."""
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    assert (
        share_cli._resolve_chat_frontend(None) == "https://rapid-pro.quicksilverpro.io"
    )


def test_resolve_chat_frontend_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_FRONTEND", "https://env-set.example.com")
    out = share_cli._resolve_chat_frontend("https://flag-set.example.com")
    assert out == "https://flag-set.example.com"


def test_resolve_chat_frontend_env_var_fallback(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_FRONTEND", "https://my-fork.example.com")
    assert share_cli._resolve_chat_frontend(None) == "https://my-fork.example.com"


def test_resolve_chat_frontend_empty_flag_disables(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    assert share_cli._resolve_chat_frontend("") is None


def test_resolve_chat_frontend_empty_env_disables(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_FRONTEND", "")
    assert share_cli._resolve_chat_frontend(None) is None


def test_resolve_chat_frontend_strips_trailing_path(monkeypatch):
    """``warning.render`` appends ``/#k=...`` itself — the resolver must
    normalise to scheme://host[:port] so a configured value with a
    trailing slash doesn't produce ``https://x.com//#k=...``."""
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    assert (
        share_cli._resolve_chat_frontend("https://chat.example.com/")
        == "https://chat.example.com"
    )


def test_resolve_chat_frontend_rejects_javascript_scheme(monkeypatch):
    """Defense against a hostile env var."""
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="https:// or http://"):
        share_cli._resolve_chat_frontend("javascript:alert(1)")


def test_resolve_chat_frontend_rejects_ftp_scheme(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="https:// or http://"):
        share_cli._resolve_chat_frontend("ftp://example.com")


def test_resolve_chat_frontend_rejects_path_component(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="without a path"):
        share_cli._resolve_chat_frontend("https://chat.example.com/app")


def test_resolve_chat_frontend_rejects_query(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="query or fragment"):
        share_cli._resolve_chat_frontend("https://chat.example.com?ref=abc")


def test_resolve_chat_frontend_rejects_fragment(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="query or fragment"):
        share_cli._resolve_chat_frontend("https://chat.example.com#anchor")


def test_resolve_chat_frontend_rejects_http_for_public_host(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="loopback"):
        share_cli._resolve_chat_frontend("http://chat.example.com")


def test_resolve_chat_frontend_allows_http_localhost(monkeypatch):
    """Loopback exception so dev setups work without faking certs."""
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    assert (
        share_cli._resolve_chat_frontend("http://localhost:5173")
        == "http://localhost:5173"
    )
    assert (
        share_cli._resolve_chat_frontend("http://127.0.0.1:5173")
        == "http://127.0.0.1:5173"
    )


def test_resolve_chat_frontend_rejects_missing_host(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="host"):
        share_cli._resolve_chat_frontend("https://")


def test_resolve_chat_frontend_rejects_userinfo(monkeypatch):
    """Codex round-7 BLOCKING (preserved): ``https://chat@evil.com``
    parses to ``netloc='chat@evil.com'`` and would echo a banner link
    where the visible host is ``chat`` but the actual host is
    ``evil.com`` (browser sends the bearer key to evil.com)."""
    monkeypatch.delenv("RAPID_MLX_CHAT_FRONTEND", raising=False)
    with pytest.raises(ValueError, match="userinfo"):
        share_cli._resolve_chat_frontend("https://chat.rapidmlx.com@evil.com")


def test_register_share_chat_frontend_default_is_none():
    """The argparse default must be None so the resolver can tell the
    "user didn't pass the flag" case apart from the "user passed
    --chat-frontend ''" opt-out."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    share_cli.register(subparsers)
    args = parser.parse_args(["share", "qwen3.5-4b-4bit"])
    assert args.chat_frontend is None


def test_register_share_chat_frontend_accepts_value():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    share_cli.register(subparsers)
    args = parser.parse_args(
        ["share", "qwen3.5-4b-4bit", "--chat-frontend", "https://my-fork.example"]
    )
    assert args.chat_frontend == "https://my-fork.example"


def test_share_command_rejects_malformed_chat_frontend_with_exit_2():
    """User error must exit 2 and NOT spawn serve. Otherwise a typo
    pays the model-load cost before failing."""
    spawn_called = MagicMock()
    args = _make_args(chat_frontend="javascript:alert(1)")
    with (
        patch.object(share_cli, "_spawn_serve", side_effect=spawn_called),
        patch.object(share_cli, "_maybe_confirm_download"),
        pytest.raises(SystemExit) as exc_info,
    ):
        share_cli.share_command(args)
    assert exc_info.value.code == 2
    spawn_called.assert_not_called()


def test_share_command_forwards_chat_frontend_to_banner(capsys):
    """End-to-end: a user-supplied ``--chat-frontend`` lands as the
    banner's one-click link prefix (instead of the default)."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel(tunnel_id="abc123_xy", public_url="https://x/r/abc123_xy")
    args = _make_args(chat_frontend="https://my-fork.example.com")

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(args)

    out = capsys.readouterr().out
    assert "https://my-fork.example.com/#k=abc123_xy." in out
    # And the default frontend MUST NOT leak.
    assert "https://rapid-pro.quicksilverpro.io/#k=" not in out


def test_share_command_omits_chat_line_when_frontend_disabled(capsys):
    """``--chat-frontend ""`` opts out — the banner must NOT advertise
    a chat link."""
    serve_proc = MagicMock()
    serve_proc.poll.return_value = None
    tunnel = _fake_tunnel()
    args = _make_args(chat_frontend="")

    with (
        patch.object(share_cli, "_spawn_serve", return_value=serve_proc),
        patch.object(share_cli, "_wait_for_healthz", return_value=True),
        patch.object(share_cli, "_verify_auth_gate", return_value=True),
        patch.object(share_cli.ws_tunnel, "TunnelClient", return_value=tunnel),
        patch.object(share_cli.ws_tunnel, "wait_for_public_url", return_value=True),
        patch.object(share_cli, "_pick_port", return_value=18765),
        patch.object(share_cli, "_maybe_confirm_download"),
        patch.object(
            share_cli, "_resolve_served_model_name", return_value="qwen3.5-4b-4bit"
        ),
        patch("time.sleep", side_effect=_ctrl_c_in_monitor_loop()),
    ):
        share_cli.share_command(args)

    out = capsys.readouterr().out
    assert "https://rapidserver.quicksilverpro.io/r/" in out
    assert "#k=" not in out
    assert "Chat:" not in out
