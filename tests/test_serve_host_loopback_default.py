# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx serve --host`` loopback-default behavior.

Pins the v0.8.2-dogfood PortSweep-bypass fix:

* The default ``--host`` is ``127.0.0.1`` (loopback-only). Operators must
  opt-in to wildcard bind (``0.0.0.0`` or ``""``) — defaults stop being
  silently LAN-reachable, and silent wildcard-vs-loopback port collisions
  stop silently shadowing each other.
* ``_port_preflight_or_die`` also probes ``127.0.0.1`` when the operator
  explicitly opts into a wildcard alias. macOS lets a wildcard bind
  coexist with a more-specific loopback bind on the same port — without
  the loopback probe, ``rapid-mlx serve --host 0.0.0.0 --port N`` would
  happily start beside an ``nc -l 127.0.0.1 N`` and loopback clients
  would silently be routed to nc. This was the original PR-#142
  PortSweep gap reproduced in /tmp/v082-dogfood-findings/03-sidecar-cli.md.

Two layers of test:

1. **Lightweight argparse + helper** — drive ``cli.main()`` through a
   ``serve_command`` stub (no MLX import) to pin the new default, and
   call ``_port_preflight_or_die`` directly to pin the probe loop. These
   tests run in headless / sandboxed environments (no Metal GPU).
2. **Behavioral via ``serve_command``** — the existing
   ``test_serve_listen_fd.py`` already exercises the full
   ``serve_command`` prologue. We don't duplicate that here; codex
   round-1 MAJOR on PR #848 flagged that importing
   ``vllm_mlx.server`` (which loads MLX) made the new tests
   environment-fragile, so the helper tests below avoid it entirely.
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
# Wildcard-alias normalization
# ---------------------------------------------------------------------------


def test_wildcard_host_aliases_includes_both_spellings():
    """``_port_preflight_or_die`` treats ``"0.0.0.0"`` AND the empty
    string as "bind on every interface" — both are uvicorn-supported
    wildcard spellings, so both must trigger the loopback probe.

    Codex round-1 MAJOR on PR #848: the first version of the gate
    only matched ``"0.0.0.0"``, so ``--host ""`` could still bypass
    the loopback collision check. Fixed by routing through a shared
    alias set; this test pins that set so a future refactor can't
    silently drop ``""`` (or accidentally add ``"localhost"`` to it,
    which would break collision detection on the default path).
    """
    aliases = cli._wildcard_host_aliases()
    assert "0.0.0.0" in aliases
    assert "" in aliases
    # MUST NOT include loopback / DNS-aliased loopback — those name a
    # single host and must be probed exactly once.
    assert "127.0.0.1" not in aliases
    assert "localhost" not in aliases


# ---------------------------------------------------------------------------
# Pre-flight collision detection — exercises ``_port_preflight_or_die``
# directly so the test stays lightweight (no MLX import, no model load).
# ---------------------------------------------------------------------------


def test_preflight_rejects_wildcard_when_loopback_already_bound(capsys):
    """The dogfood-finding repro at the helper level: bind a loopback-only
    listener on ``127.0.0.1:PORT``, then ask the helper to probe
    ``("0.0.0.0", PORT)``. Pre-fix the wildcard probe succeeded and
    nothing else ran — bypass. Post-fix the helper additionally probes
    ``127.0.0.1`` and catches the collision."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        with pytest.raises(SystemExit) as exc:
            cli._port_preflight_or_die("0.0.0.0", blocked_port, model="qwen3.5-4b-4bit")

    assert exc.value.code == 1
    err_out = capsys.readouterr().out
    assert "already in use" in err_out
    assert "127.0.0.1" in err_out, (
        f"pre-flight error must name the colliding host 127.0.0.1, got: {err_out!r}"
    )


def test_preflight_rejects_empty_host_when_loopback_already_bound(capsys):
    """``--host ""`` is the second uvicorn-style wildcard spelling. It
    MUST also trigger the loopback probe — codex round-1 MAJOR caught
    that the first version of the gate string-matched only ``"0.0.0.0"``
    and would have left this bypass open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        with pytest.raises(SystemExit) as exc:
            cli._port_preflight_or_die("", blocked_port, model="qwen3.5-4b-4bit")

    assert exc.value.code == 1
    err_out = capsys.readouterr().out
    assert "already in use" in err_out
    assert "127.0.0.1" in err_out


def test_preflight_rejects_loopback_when_loopback_already_bound(capsys):
    """Pre-existing case (default ``--host 127.0.0.1``): loopback
    listener already bound → collision surfaces directly without the
    extra probe. Pin so the new probe loop doesn't accidentally swallow
    the OSError on this path."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        with pytest.raises(SystemExit) as exc:
            cli._port_preflight_or_die(
                "127.0.0.1", blocked_port, model="qwen3.5-4b-4bit"
            )

    assert exc.value.code == 1
    err_out = capsys.readouterr().out
    assert "already in use" in err_out
    assert "127.0.0.1" in err_out


def test_preflight_passes_on_free_port():
    """Pin the happy path: when the port is genuinely free, the helper
    returns cleanly and does NOT raise SystemExit. Guards against an
    over-eager probe that mistakes (say) IPv6 ::1 traffic for an IPv4
    collision."""
    # Should return None and not raise.
    result = cli._port_preflight_or_die(
        "127.0.0.1", _free_loopback_port(), model="qwen3.5-4b-4bit"
    )
    assert result is None


def test_preflight_wildcard_branch_passes_on_free_port():
    """Wildcard branch also must NOT raise on a fully free port — the
    probe loop runs twice (host + 127.0.0.1) but neither probe should
    collide."""
    result = cli._port_preflight_or_die(
        "0.0.0.0", _free_loopback_port(), model="qwen3.5-4b-4bit"
    )
    assert result is None


def test_preflight_error_uses_friendly_host_display_for_empty(capsys):
    """``--host ""`` is a wildcard alias; the error message must NOT
    print a bare empty string (would look like ``on .`` in the UX). The
    helper substitutes the friendlier ``0.0.0.0`` for display when the
    user passed ``""``.

    Achieve the empty-host collision deterministically by simulating
    only the first probe failing — bind on ``("", port)`` collides with
    any existing wildcard listener.
    """
    # Listen on a wildcard so the empty-string probe (which is also a
    # wildcard bind) collides on the first iteration of the loop.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("", 0))
        blocker.listen(1)
        blocked_port = blocker.getsockname()[1]

        with pytest.raises(SystemExit) as exc:
            cli._port_preflight_or_die("", blocked_port, model="qwen3.5-4b-4bit")

    assert exc.value.code == 1
    err_out = capsys.readouterr().out
    # The display-host substitution must kick in: surface "0.0.0.0",
    # not a bare quote, so the UX stays readable.
    assert "0.0.0.0" in err_out


# ---------------------------------------------------------------------------
# Legacy ``python -m vllm_mlx.server`` entrypoint default
# ---------------------------------------------------------------------------


def test_legacy_server_argparse_host_default_is_loopback():
    """Codex round-1 MAJOR on PR #848 flagged that ``vllm_mlx/server.py``
    is the second supported entrypoint (``python -m vllm_mlx.server``)
    and was also defaulting to ``0.0.0.0``. The fix landed the same
    loopback default there too. This test pins the argparse contract
    WITHOUT importing the heavy ``vllm_mlx.server`` module (which loads
    MLX) — we read the source for the default literal instead, the same
    style ``test_serve_listen_fd.py`` uses for source-level invariants.
    """
    import inspect
    from pathlib import Path

    # Resolve the source file via inspect on the cli module (already
    # imported above) so the test stays self-locating across worktrees.
    pkg_root = Path(inspect.getfile(cli)).parent
    server_src = (pkg_root / "server.py").read_text(encoding="utf-8")

    # Locate the --host argparse block and assert the default is
    # loopback. We pin on a tight slice of source rather than a regex
    # across the whole file to keep the test resilient to unrelated
    # edits.
    idx = server_src.find('"--host"')
    assert idx >= 0, "expected --host argparse arg in vllm_mlx/server.py"
    nearby = server_src[idx : idx + 400]
    assert 'default="127.0.0.1"' in nearby, (
        "vllm_mlx/server.py --host argparse default must be 127.0.0.1; "
        f"got nearby source: {nearby!r}"
    )
    # And the 0.0.0.0 default must be gone — guards against a future
    # refactor that adds a second --host block without dropping the old
    # one.
    assert 'default="0.0.0.0"' not in server_src, (
        "vllm_mlx/server.py must not retain the legacy 0.0.0.0 default"
    )


# ---------------------------------------------------------------------------
# IPv6 preflight — codex round-1 MED #6 on PR #855
# ---------------------------------------------------------------------------


def _ipv6_loopback_supported() -> bool:
    """Skip-guard for IPv6-disabled environments.

    GH Actions runners sometimes disable IPv6 entirely; we don't want a
    real environment limitation to flake the test. A trivial ``::1``
    bind probe tells us whether the family is usable.
    """
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _ipv6_loopback_supported(), reason="IPv6 loopback not available"
)
def test_preflight_passes_for_ipv6_loopback_on_free_port():
    """Codex round-1 MED #6 on PR #855: pre-fix the IPv4-only preflight
    always opened an ``AF_INET`` socket, so ``--host ::1`` raised
    ``OSError`` from ``socket.bind`` and was misreported as "port
    already in use" — blocking a configuration uvicorn otherwise
    supports.

    Post-fix the helper detects IPv6 literals and switches the probe
    family to ``AF_INET6``. On a free port the call must return cleanly
    (no SystemExit) just like the IPv4 happy path."""
    # OS-assigned IPv6 loopback port.
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.bind(("::1", 0))
        free_port = s.getsockname()[1]

    result = cli._port_preflight_or_die("::1", free_port, model="qwen3.5-4b-4bit")
    assert result is None


@pytest.mark.skipif(
    not _ipv6_loopback_supported(), reason="IPv6 loopback not available"
)
def test_preflight_passes_for_ipv6_wildcard_on_free_port():
    """``--host ::`` is the IPv6 wildcard spelling. Same regression as
    ``::1``: pre-fix the AF_INET socket raised ``EAFNOSUPPORT``/
    ``EADDRNOTAVAIL`` and the helper printed "port already in use,"
    masking the real (no-collision) state.

    Note: ``::`` is NOT in ``_wildcard_host_aliases()`` because the
    loopback-shadow probe is IPv4-specific (macOS treats v4 and v6
    loopback as distinct stacks, per the helper's docstring). So the
    fix is family-only — IPv6 wildcards probe themselves once with
    AF_INET6 and don't trigger the secondary 127.0.0.1 probe."""
    # OS-assigned IPv6 port via a transient bind, then release it.
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.bind(("::", 0))
        free_port = s.getsockname()[1]

    result = cli._port_preflight_or_die("::", free_port, model="qwen3.5-4b-4bit")
    assert result is None


def test_ipv6_host_detector_matches_literals_only():
    """Pin the colon-based heuristic ``_is_ipv6_host`` uses.

    IPv6 literals always contain ``:``; IPv4 literals, wildcards
    (``0.0.0.0``, ``""``), and DNS names (``localhost``) never do.
    Keep the detector purely lexical so scoped literals
    (``fe80::1%en0``) that uvicorn accepts still route through the
    AF_INET6 branch (a stricter ``ipaddress.ip_address`` parse would
    reject them)."""
    assert cli._is_ipv6_host("::") is True
    assert cli._is_ipv6_host("::1") is True
    assert cli._is_ipv6_host("2001:db8::1") is True
    assert cli._is_ipv6_host("fe80::1%en0") is True

    assert cli._is_ipv6_host("0.0.0.0") is False
    assert cli._is_ipv6_host("127.0.0.1") is False
    assert cli._is_ipv6_host("") is False
    assert cli._is_ipv6_host("localhost") is False
