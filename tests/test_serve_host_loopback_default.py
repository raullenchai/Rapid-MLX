# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx serve --host`` loopback-default behavior.

Pins the v0.8.2-dogfood PortSweep-bypass fix:

* The default ``--host`` is ``127.0.0.1`` (loopback-only). Operators must
  opt-in to wildcard bind (``0.0.0.0``) — defaults stop being silently
  LAN-reachable, and silent wildcard-vs-loopback port collisions stop
  silently shadowing each other.
* The pre-flight port check ALSO probes ``127.0.0.1`` when the operator
  explicitly opts into ``--host 0.0.0.0``. macOS lets a wildcard bind
  coexist with a more-specific loopback bind on the same port — without
  the loopback probe, ``rapid-mlx serve --host 0.0.0.0 --port N`` would
  happily start beside an ``nc -l 127.0.0.1 N`` and loopback clients
  would silently be routed to nc. This was the original PR-#142
  PortSweep gap reproduced in /tmp/v082-dogfood-findings/03-sidecar-cli.md.

Tests mirror the lightweight style of ``test_serve_listen_fd.py``: drive
``cli.main()`` through a ``serve_command`` stub so the assertions ride
the real subparser, not a hand-rolled namespace.
"""

from __future__ import annotations

import socket
import sys
from unittest.mock import patch

import pytest

from vllm_mlx import cli

# ---------------------------------------------------------------------------
# Helpers
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


def _free_loopback_port() -> int:
    """Bind a real socket to an OS-assigned loopback port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Argparse — default is now loopback, not wildcard
# ---------------------------------------------------------------------------


def test_serve_host_default_is_loopback():
    """``rapid-mlx serve <alias>`` (no ``--host``) binds 127.0.0.1.

    Regression pin for v0.8.2 dogfood finding #2 (PortSweep bypass): the
    old ``default="0.0.0.0"`` widened the bind to every interface AND
    silently allowed a wildcard listener to coexist with a more-specific
    loopback listener (e.g. ``nc -l 127.0.0.1 PORT``) on the same port.
    Changing the default to 127.0.0.1 closes both the LAN-exposure gap
    (security) and the dual-bind ambiguity (correctness).
    """
    captured = _capture_serve_args(["rapid-mlx", "serve", "qwen3.5-4b-4bit"])
    assert len(captured) == 1
    ns = captured[0]
    assert ns.host == "127.0.0.1", (
        f"default --host must be loopback-only, got {ns.host!r} — see "
        "PortSweep bypass write-up in test docstring."
    )


def test_serve_host_explicit_wildcard_is_honored():
    """Operators can still opt into ``--host 0.0.0.0`` when they
    actually want LAN exposure (reverse proxy, deliberate dev rig).
    The default just stops being the dangerous one."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--host", "0.0.0.0"]
    )
    assert captured[0].host == "0.0.0.0"


def test_serve_host_explicit_loopback_is_honored():
    """Redundant but harmless: explicit ``--host 127.0.0.1`` passes through."""
    captured = _capture_serve_args(
        ["rapid-mlx", "serve", "qwen3.5-4b-4bit", "--host", "127.0.0.1"]
    )
    assert captured[0].host == "127.0.0.1"


def test_serve_host_help_mentions_loopback_default(capsys):
    """``rapid-mlx serve --help`` must document the loopback-only default
    so operators can self-diagnose ``connection refused`` from a remote
    host without spelunking the source for the changed default."""
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", "--help"]),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--host" in out
    # The help string mentions either the literal default or the loopback
    # phrasing — accept both so future copy edits don't flake the test.
    assert "127.0.0.1" in out or "loopback" in out.lower()


# ---------------------------------------------------------------------------
# Pre-flight collision detection
# ---------------------------------------------------------------------------


def _stub_heavy_serve_deps(monkeypatch):
    """Replicate the minimal stubs needed to drive ``serve_command``
    through to (or past) the pre-flight bind check without booting a
    real engine. Mirrors ``stub_heavy_serve_deps`` in
    ``test_serve_listen_fd.py`` but without the fixture wrapper so the
    individual tests below can choose to short-circuit early."""
    from vllm_mlx import _version_check
    from vllm_mlx import cli as cli_mod
    from vllm_mlx import server as server_mod

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(_version_check, "print_staleness_warning_if_any", lambda: None)
    monkeypatch.setattr(cli_mod, "_ensure_model_downloaded", lambda model: None)
    monkeypatch.setattr(cli_mod, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli_mod, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "configure_logging", lambda level: "info")
    monkeypatch.setattr(server_mod, "load_model", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "configure_cors", lambda *a, **kw: None)
    from vllm_mlx.middleware import auth as auth_mod

    monkeypatch.setattr(auth_mod, "configure_rate_limiter", lambda *a, **kw: None)


def _serve_ns(**overrides):
    """Resolve a serve Namespace via the real subparser, then apply
    field-level overrides."""
    argv = ["rapid-mlx", "serve", "qwen3.5-4b-4bit"]
    for k, v in overrides.items():
        if k == "host":
            argv += ["--host", v]
        elif k == "port":
            argv += ["--port", str(v)]
        elif k == "listen_fd":
            argv += ["--listen-fd", str(v)]
    captured: list = []
    with (
        patch.object(sys, "argv", argv),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


def test_serve_preflight_rejects_wildcard_when_loopback_already_bound(
    monkeypatch, capsys
):
    """The dogfood-finding repro: a loopback-only listener occupies
    ``127.0.0.1:PORT``, then a second ``rapid-mlx serve --host 0.0.0.0
    --port PORT`` arrives. The kernel WILL let the wildcard bind
    succeed (more-specific 127.0.0.1 wins for loopback traffic), so the
    only place to catch this is an explicit pre-flight probe against
    127.0.0.1.

    Before the fix this test would have FAILED — the wildcard bind
    succeeded and serve_command proceeded to load the model. After the
    fix the probe catches it and ``sys.exit(1)`` fires with the
    surfaced host in the error message.
    """
    _stub_heavy_serve_deps(monkeypatch)

    # Capture uvicorn.run so a regression that drops the pre-flight
    # entirely (and reaches uvicorn) gets a distinct failure shape.
    captured_uvicorn: dict = {}

    def fake_run(app, **kwargs):
        captured_uvicorn["app"] = app
        captured_uvicorn.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    # Stage 1: bind a loopback-only listener — the "nc -l 127.0.0.1 PORT"
    # in the dogfood repro.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        # Stage 2: try to serve on 0.0.0.0:PORT. The pre-flight MUST
        # surface the collision via sys.exit(1).
        ns = _serve_ns(host="0.0.0.0", port=blocked_port)
        assert ns.host == "0.0.0.0"
        assert ns.port == blocked_port
        with pytest.raises(SystemExit) as exc:
            cli.serve_command(ns)

    assert exc.value.code == 1, (
        f"expected sys.exit(1) from PortSweep pre-flight, got {exc.value.code}"
    )
    err_out = capsys.readouterr().out
    assert "already in use" in err_out
    # The actionable detail the fix surfaces: which host the collision
    # happened on. Lets the operator distinguish "LAN port busy" from
    # "loopback shadow" without a packet-trace.
    assert "127.0.0.1" in err_out, (
        f"pre-flight error must name the colliding host 127.0.0.1, got: {err_out!r}"
    )
    # And the wildcard branch must NOT have reached uvicorn — that's the
    # whole point of catching it pre-bind.
    assert not captured_uvicorn, (
        f"serve_command must abort before uvicorn.run, got {captured_uvicorn!r}"
    )


def test_serve_preflight_rejects_loopback_when_loopback_already_bound(
    monkeypatch, capsys
):
    """The plain-default flow: ``rapid-mlx serve`` (loopback default) on
    a port a previous loopback listener already owns must also abort.
    This is the path that ALREADY worked pre-fix; pin it so the new
    probe-loop doesn't accidentally swallow the OSError on this branch."""
    _stub_heavy_serve_deps(monkeypatch)

    captured_uvicorn: dict = {}

    def fake_run(app, **kwargs):
        captured_uvicorn["app"] = app
        captured_uvicorn.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        ns = _serve_ns(host="127.0.0.1", port=blocked_port)
        with pytest.raises(SystemExit) as exc:
            cli.serve_command(ns)

    assert exc.value.code == 1
    err_out = capsys.readouterr().out
    assert "already in use" in err_out
    assert "127.0.0.1" in err_out
    assert not captured_uvicorn


def test_serve_preflight_passes_to_uvicorn_on_free_port(monkeypatch):
    """Pin the happy path: when the port is genuinely free, the
    pre-flight does NOT raise and serve_command reaches uvicorn.run.
    Guards against an over-eager probe that mistakes (say) IPv6
    ::1 traffic for an IPv4 collision."""
    _stub_heavy_serve_deps(monkeypatch)

    captured_uvicorn: dict = {}

    def fake_run(app, **kwargs):
        captured_uvicorn["app"] = app
        captured_uvicorn.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    ns = _serve_ns(host="127.0.0.1", port=_free_loopback_port())
    # No SystemExit raised; uvicorn.run captured with the expected kwargs.
    cli.serve_command(ns)

    assert captured_uvicorn.get("host") == "127.0.0.1"
    assert captured_uvicorn.get("port") == ns.port
    assert "fd" not in captured_uvicorn
