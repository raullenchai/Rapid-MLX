# SPDX-License-Identifier: Apache-2.0
"""Test: ``_run_uvicorn`` exits non-zero on ``OSError(EADDRINUSE)`` from
the actual bind step (R13 Sven B1 dogfood).

Why this matters: process supervisors (systemd, launchd, k8s) infer
"server started successfully" from a zero exit code. If
``rapid-mlx serve`` collides on its port and exits 0, the supervisor
silently believes a server is running when nothing is — the symptom
Sven reported in the R13 dogfood pass.

``_port_preflight_or_die`` (CLI prologue) handles the common case, but
two paths still reach ``_run_uvicorn`` with a colliding port:

  1. **TOCTOU race**: another process grabs the port between the
     preflight's ``socket.close()`` and uvicorn's ``loop.create_server``.
  2. **``--listen-fd`` mode**: preflight is skipped by design; a bad
     inherited fd surfaces as an ``OSError`` from inside uvicorn.

The fix is a try/except in ``_run_uvicorn`` (the single CLI-side
chokepoint) that catches ``OSError`` with ``errno == EADDRINUSE``,
prints a Sven-style friendly message, and ``sys.exit(1)``s. This is the
"layer-level fix at the CLI entrypoint" the bug ticket called for —
both the text-model branch (``serve_command``) and the audio/multimodal
branch (``_serve_audio_dispatch``) route through ``_run_uvicorn``, so
the wrap lives in one place.

These tests pin the contract:

* ``EADDRINUSE`` → ``SystemExit(1)`` AND the error message names the
  colliding port (so an operator grepping logs can find it).
* Unrelated ``OSError`` (e.g. ``EACCES``) is **not** swallowed —
  re-raise so disk/permission failures still surface with their
  original trace.
* The blocking socket the test uses to claim a port is closed cleanly
  via the ``socket.socket`` context manager so the test never leaks
  file descriptors into the rest of the suite.
"""

from __future__ import annotations

import errno
import socket
import sys
import types
from unittest.mock import patch

import pytest

from vllm_mlx import cli


def _claim_loopback_port() -> tuple[socket.socket, int]:
    """Bind a TCP socket to ``127.0.0.1`` on an OS-chosen port and return
    ``(sock, port)``. The caller is responsible for closing ``sock``
    (use a ``try/finally`` or ``with``) so the test doesn't leak fds.

    ``SO_REUSEADDR`` is intentionally NOT set: the whole point of the
    fixture is to collide with a later bind that itself sets
    ``SO_REUSEADDR``. On macOS+Linux, a second non-``SO_REUSEPORT`` bind
    on the same loopback (host, port) still fails with EADDRINUSE — which
    is exactly what we want to assert ``_run_uvicorn`` surfaces.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.listen(1)
    return sock, port


def _serve_ns(port: int) -> types.SimpleNamespace:
    """Minimal ``argparse.Namespace`` for ``_run_uvicorn`` —
    the heavy serve prologue (model download, version check) is
    bypassed; we only need the host/port/listen_fd fields the
    dispatcher reads."""
    return types.SimpleNamespace(host="127.0.0.1", port=port, listen_fd=None)


def test_run_uvicorn_exits_nonzero_on_eaddrinuse(monkeypatch, capsys):
    """Real bind path: ``_run_uvicorn`` MUST translate an
    ``OSError(EADDRINUSE)`` from ``uvicorn.run`` into ``SystemExit(1)``
    AND name the port in the message so a supervisor / operator can
    triage from the captured stderr.

    We don't actually start uvicorn — that would require a FastAPI app
    + an asyncio loop + a model. Instead we monkeypatch ``uvicorn.run``
    to raise the exact exception uvicorn would surface if its internal
    ``loop.create_server`` re-raised an ``EADDRINUSE`` past uvicorn's
    own ``sys.exit(1)`` guard (the TOCTOU-race / future-uvicorn-change
    case the wrapper is defense-in-depth for).
    """
    sock, port = _claim_loopback_port()
    try:
        # Simulate uvicorn's bind raising EADDRINUSE. Use a real OSError
        # with the platform errno so ``errno.EADDRINUSE`` matching is
        # exercised end-to-end (not a duck-typed mock).
        def _raise_eaddrinuse(*_args, **_kwargs):
            raise OSError(errno.EADDRINUSE, "Address already in use")

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", _raise_eaddrinuse)

        ns = _serve_ns(port)
        with pytest.raises(SystemExit) as excinfo:
            cli._run_uvicorn(object(), ns, "error")

        assert excinfo.value.code == 1, (
            f"expected SystemExit(1) on EADDRINUSE, got code={excinfo.value.code!r}"
        )

        captured = capsys.readouterr()
        # The supervisor / operator-facing message: must call out the
        # colliding port so triage doesn't need to grep server logs.
        assert str(port) in captured.err, (
            f"expected port {port} in stderr error message, got: {captured.err!r}"
        )
        assert "already in use" in captured.err.lower(), (
            f"expected friendly 'already in use' phrase, got: {captured.err!r}"
        )
    finally:
        sock.close()


def test_run_uvicorn_reraises_unrelated_oserror(monkeypatch):
    """An ``OSError`` that is NOT ``EADDRINUSE`` (e.g. ``EACCES`` when
    a low port is bound without privileges) must **not** be swallowed by
    the wrapper — the user-facing CLI should still surface the original
    trace so the failure is debuggable. The wrap is intentionally narrow.
    """

    def _raise_eacces(*_args, **_kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _raise_eacces)

    ns = _serve_ns(port=80)  # port irrelevant — uvicorn.run is stubbed
    with pytest.raises(OSError) as excinfo:
        cli._run_uvicorn(object(), ns, "error")

    assert excinfo.value.errno == errno.EACCES, (
        f"expected EACCES to propagate, got errno={excinfo.value.errno!r}"
    )


def test_run_uvicorn_eaddrinuse_with_real_blocked_port(monkeypatch, capsys):
    """End-to-end variant: hold the port for real (not mocked) so the
    test exercises the wrap against a true OS-level ``EADDRINUSE``
    raised from uvicorn's bind. We monkeypatch ``uvicorn.run`` to
    actually attempt a ``socket.bind`` (the real failure mode) rather
    than just the exception shape — this catches "the wrapper depends
    on errno being set by the OS" regressions that a hand-rolled
    ``OSError(errno.EADDRINUSE, ...)`` would mask.
    """
    sock, port = _claim_loopback_port()
    try:

        def _try_real_bind(*_args, **kwargs):
            # Mimic the bind uvicorn would do — same family + a real
            # bind() against the host/port the caller passed.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((kwargs["host"], kwargs["port"]))

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", _try_real_bind)

        ns = _serve_ns(port)
        with pytest.raises(SystemExit) as excinfo:
            cli._run_uvicorn(object(), ns, "error")

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert str(port) in captured.err
    finally:
        sock.close()


def test_claim_loopback_port_releases_fd():
    """Self-check on the helper: confirm the holder socket actually
    closes after ``sock.close()`` so the broader suite doesn't pay an
    fd-leak tax. We assert this by re-binding the same port — if the
    OS still considers it held, ``bind`` raises ``OSError`` and the
    test fails loudly. This is a guard for the test infra itself, not
    a production code claim.
    """
    sock, port = _claim_loopback_port()
    sock.close()

    # On macOS/Linux a freshly closed (non-TIME_WAIT) loopback port can
    # be reclaimed immediately when ``SO_REUSEADDR`` is set. Try the
    # rebind to prove the fd is actually released.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))  # would raise if leaked
