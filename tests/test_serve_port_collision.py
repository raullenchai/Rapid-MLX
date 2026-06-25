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
import types

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


def test_run_uvicorn_eaddrinuse_socket_level_discriminator(monkeypatch, capsys):
    """Socket-level discriminator: hold the port for real, then stub
    ``uvicorn.run`` with a hand-written ``socket.bind`` — NOT real
    uvicorn — so the wrap's EADDRINUSE detection is exercised against
    a true OS-set ``errno`` rather than a hand-rolled
    ``OSError(errno.EADDRINUSE, ...)`` (codex round-2 NIT: prior test
    name implied uvicorn coverage that doesn't exist here — the real
    uvicorn-SystemExit-after-bind path is pinned by
    ``test_run_uvicorn_systemexit_from_uvicorn_eaddrinuse_reemits_message``
    below, which is the actual contract for current uvicorn).

    What this test pins: the wrap's ``except OSError`` arm correctly
    reads ``exc.errno`` from a kernel-set error rather than only from
    a synthetic exception — so a future regression that drops the
    ``errno`` comparison still surfaces. Complements (does not
    duplicate) the SystemExit-from-uvicorn test below.
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


def test_run_uvicorn_systemexit_from_uvicorn_eaddrinuse_reemits_message(
    monkeypatch, capsys
):
    """uvicorn>=0.34 catches the bind ``OSError`` inside
    ``Server.startup`` and ``sys.exit(1)``s before our ``except OSError``
    can fire — so the simple ``except OSError`` wrap is dead for the
    normal CLI port-collision case (codex round-1 BLOCKING #2).

    The wrapper must also catch ``SystemExit(1)`` from uvicorn and, if
    a fresh probe confirms ``(host, port)`` is genuinely busy, re-emit
    the friendly Sven-style message so an operator's log grep for
    ``"already in use"`` still hits even when uvicorn was the one who
    logged the raw ``[Errno 48]``.

    Holds the port for real so the post-SystemExit probe (the actual
    discriminator inside the wrapper) sees a true EADDRINUSE rather
    than a mocked one — this is the test that pins the contract codex
    flagged as missing.
    """
    sock, port = _claim_loopback_port()
    try:

        def _raise_sysexit(*_args, **_kwargs):
            # Exactly what uvicorn's ``Server.startup`` does on
            # ``OSError`` from ``loop.create_server``: log, then
            # ``sys.exit(1)``.
            raise SystemExit(1)

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", _raise_sysexit)

        ns = _serve_ns(port)
        with pytest.raises(SystemExit) as excinfo:
            cli._run_uvicorn(object(), ns, "error")

        assert excinfo.value.code == 1, (
            f"expected SystemExit(1) to propagate, got code={excinfo.value.code!r}"
        )
        captured = capsys.readouterr()
        assert str(port) in captured.err
        assert "already in use" in captured.err.lower()
    finally:
        sock.close()


def test_run_uvicorn_probe_failure_does_not_mask_systemexit(monkeypatch):
    """Codex round-2 BLOCKING: if the ``_port_is_busy`` probe raises
    something other than ``OSError`` (e.g. ``TypeError`` from a
    non-string host, ``gaierror`` from a hostname the OS can't resolve
    at probe time), the wrapper MUST NOT replace uvicorn's original
    ``SystemExit(1)`` with the probe's traceback. The supervisor's
    failure-detection contract reads the original exit, not whatever
    the discriminator coincidentally bubbled.

    Force the probe to raise by stubbing it directly — exercises the
    outer ``except BaseException`` in ``_port_is_busy`` that returns
    ``False`` so the caller's ``raise`` re-delivers uvicorn's exit.
    """

    def _raise_sysexit(*_args, **_kwargs):
        raise SystemExit(1)

    def _probe_explodes(*_args, **_kwargs):
        raise TypeError("simulated probe-side failure (bad host type)")

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _raise_sysexit)
    monkeypatch.setattr(cli, "_port_is_busy", _probe_explodes)

    ns = _serve_ns(port=8000)
    with pytest.raises(SystemExit) as excinfo:
        cli._run_uvicorn(object(), ns, "error")

    # The original SystemExit from uvicorn must propagate untouched —
    # NOT the TypeError the probe raised.
    assert excinfo.value.code == 1, (
        f"expected uvicorn SystemExit(1) to propagate, got code={excinfo.value.code!r}"
    )


def test_port_is_busy_returns_false_on_probe_side_exception():
    """Direct unit test on ``_port_is_busy``: when the probe machinery
    fails for ANY reason (host normalization, socket constructor,
    etc.) the helper must return ``False`` so the caller's ``raise``
    re-delivers the original ``SystemExit``. Covers the codex round-2
    BLOCKING contract at the helper boundary, complementing the
    integration test above.

    We pass ``host=None`` which would historically have raised a
    ``TypeError`` from ``probe.bind((None, port))``; the outer
    ``except BaseException`` (or the explicit ``isinstance`` guard)
    must convert that into ``False`` rather than re-raising.
    """
    # Should return False, NOT raise.
    assert cli._port_is_busy(None, 8000) is False  # type: ignore[arg-type]
    assert cli._port_is_busy(12345, 8000) is False  # type: ignore[arg-type]


def test_run_uvicorn_systemexit_passthrough_when_port_not_busy(monkeypatch, capsys):
    """If uvicorn ``SystemExit(1)``s for a reason OTHER than a bind
    collision (e.g. TLS misconfig, lifespan abort), the wrapper MUST
    NOT paper over it with a port-collision message. Pin this so the
    discriminator probe stays narrow: only re-emit when the port is
    actually busy.
    """

    def _raise_sysexit(*_args, **_kwargs):
        raise SystemExit(1)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _raise_sysexit)

    # OS-chosen port that we DON'T hold — the probe will succeed,
    # confirming the SystemExit wasn't a port collision.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # probe closed → port is free → discriminator returns False

    ns = _serve_ns(port)
    with pytest.raises(SystemExit) as excinfo:
        cli._run_uvicorn(object(), ns, "error")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    # The wrapper must NOT have printed its port-collision message —
    # the SystemExit came from uvicorn for an unrelated reason.
    assert "already in use" not in captured.err.lower(), (
        f"wrapper papered over a non-collision SystemExit with a "
        f"port-collision message: {captured.err!r}"
    )


def test_run_uvicorn_listen_fd_eaddrinuse_uses_fd_specific_message(monkeypatch, capsys):
    """In ``--listen-fd`` mode, ``args.port`` is meaningless — the
    supervisor owns the bind, and the inherited fd may not correspond
    to the CLI port at all. The friendly message must therefore NOT
    print ``lsof -i :<args.port>`` (operator would chase the wrong
    socket); it must reference the fd-mode failure instead.

    Codex round-1 NIT #3.
    """

    def _raise_eaddrinuse(*_args, **_kwargs):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _raise_eaddrinuse)

    ns = types.SimpleNamespace(host="127.0.0.1", port=8000, listen_fd=11)
    with pytest.raises(SystemExit) as excinfo:
        cli._run_uvicorn(object(), ns, "error")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    # The fd-mode message must NOT include a port-specific lsof hint
    # since args.port has no relationship to the inherited socket.
    assert "lsof -i :8000" not in captured.err, (
        f"--listen-fd mode must not reference args.port; got: {captured.err!r}"
    )
    assert (
        "listen-fd" in captured.err.lower()
        or "supervisor" in captured.err.lower()
        or "inherited" in captured.err.lower()
    ), f"--listen-fd error must mention the fd / supervisor: {captured.err!r}"


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
