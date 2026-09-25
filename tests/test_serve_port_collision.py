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
import os
import socket
import types
from argparse import Namespace
from contextlib import ExitStack

import pytest

from rapid_mlx import cli
from rapid_mlx.connect import endpoints_from_bind, render_banner


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


def _claim_exact_loopback_port(stack: ExitStack, port: int) -> None:
    """Hold one exact loopback port for the duration of ``stack``."""

    sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    sock.bind(("127.0.0.1", port))
    sock.listen(1)


@pytest.fixture
def scan_base() -> int:
    """Return an OS-selected base whose full candidate range is available."""

    for _attempt in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as seed:
            seed.bind(("127.0.0.1", 0))
            base = seed.getsockname()[1]
        if base + cli.DEFAULT_SERVE_PORT_CANDIDATES > 65536:
            continue
        try:
            with ExitStack() as probes:
                for port in range(base, base + cli.DEFAULT_SERVE_PORT_CANDIDATES):
                    _claim_exact_loopback_port(probes, port)
        except OSError:
            continue
        return base
    pytest.fail("could not reserve an ephemeral serve-port scan range")


def test_implicit_busy_default_selects_next_port_and_stamps_user_urls(
    capsys, scan_base
):
    """An omitted ``--port`` falls forward before any server URL is rendered."""

    with ExitStack() as stack:
        _claim_exact_loopback_port(stack, scan_base)
        resolved = cli._resolve_serve_port(
            "127.0.0.1",
            None,
            model="qwen3.5-4b-4bit",
            scan_base=scan_base,
        )

    assert resolved == scan_base + 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        f"Port {scan_base} is in use; using {scan_base + 1} instead "
        "(pass --port to choose)."
    ]

    args = types.SimpleNamespace(
        host="127.0.0.1", port=resolved, listen_fd=None, lazy_load=False
    )
    assert f"http://127.0.0.1:{scan_base + 1}" in cli._serve_startup_message(args)
    ready = render_banner(
        endpoints_from_bind(args.host, args.port, model="qwen3.5-4b-4bit")
    )
    assert f"Ready: http://127.0.0.1:{scan_base + 1}" in ready


def test_explicit_busy_port_keeps_existing_hard_failure(capsys, scan_base):
    """An explicit collision remains rc 1 with the established message."""

    with ExitStack() as stack:
        _claim_exact_loopback_port(stack, scan_base)
        with pytest.raises(SystemExit) as excinfo:
            cli._resolve_serve_port("127.0.0.1", scan_base, model="qwen3.5-4b-4bit")

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        f"\n  Error: Port {scan_base} is already in use on 127.0.0.1.\n"
        "  Try a different port: rapid-mlx serve qwen3.5-4b-4bit "
        f"--port {scan_base + 1}\n"
    )


def test_implicit_port_fails_when_all_ten_candidates_are_busy(capsys, scan_base):
    """The implicit scan is bounded to ten candidates and retains rc 1."""

    with ExitStack() as stack:
        for port in range(scan_base, scan_base + cli.DEFAULT_SERVE_PORT_CANDIDATES):
            _claim_exact_loopback_port(stack, port)
        with pytest.raises(SystemExit) as excinfo:
            cli._resolve_serve_port(
                "127.0.0.1",
                None,
                model="qwen3.5-4b-4bit",
                scan_base=scan_base,
            )

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    scan_end = scan_base + cli.DEFAULT_SERVE_PORT_CANDIDATES - 1
    assert captured.out == (
        f"Ports {scan_base}-{scan_end} are all in use; "
        "pass --port with a free port outside that range.\n"
    )


def test_implicit_wildcard_scan_detects_loopback_shadow(capsys, scan_base):
    """Wildcard fallback also treats a loopback-only listener as busy."""

    with ExitStack() as stack:
        _claim_exact_loopback_port(stack, scan_base)
        resolved = cli._resolve_serve_port(
            "0.0.0.0", None, model="qwen3.5-4b-4bit", scan_base=scan_base
        )

    assert resolved == scan_base + 1
    assert capsys.readouterr().err.splitlines() == [
        f"Port {scan_base} is in use; using {scan_base + 1} instead "
        "(pass --port to choose)."
    ]


def test_explicit_free_port_has_no_substitution_notice(capsys):
    """A user-selected free port passes through without fallback output."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    assert cli._resolve_serve_port("127.0.0.1", port, model="qwen3.5-4b-4bit") == port
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_implicit_free_scan_base_has_no_notice(capsys, scan_base):
    """The normal omitted-port path keeps 8000 when it is available."""

    assert (
        cli._resolve_serve_port("127.0.0.1", None, model="model", scan_base=scan_base)
        == scan_base
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.parametrize("requested_port", [None, 8000])
def test_port_probe_reports_non_collision_bind_error(
    monkeypatch, capsys, requested_port
):
    error = OSError(errno.EADDRNOTAVAIL, "address not available")

    class FailingProbe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def setsockopt(self, *_args):
            pass

        def bind(self, _address):
            raise error

    monkeypatch.setattr(socket, "socket", lambda *_args: FailingProbe())

    with pytest.raises(SystemExit) as excinfo:
        cli._resolve_serve_port(
            "192.0.2.1",
            requested_port,
            model="model",
            scan_base=8000,
            scan_count=1,
        )

    assert excinfo.value.code == 2
    assert "Invalid --host '192.0.2.1'" in capsys.readouterr().err


def test_product_scan_defaults_remain_8000_through_8009():
    assert cli.DEFAULT_SERVE_PORT == 8000
    assert cli.DEFAULT_SERVE_PORT_CANDIDATES == 10


def test_listen_fd_uses_bound_socket_port_and_keeps_fd_open():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        bound_port = listener.getsockname()[1]

        resolved = cli._resolve_serve_port(
            "127.0.0.1",
            None,
            model="model",
            listen_fd=listener.fileno(),
        )

        assert resolved == bound_port
        assert listener.getsockname()[1] == bound_port


def test_resolve_listen_fd_reports_cli_error_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_listen_fd_port",
        lambda _fd: (_ for _ in ()).throw(OSError("descriptor is closed")),
    )

    with pytest.raises(SystemExit) as caught:
        cli._resolve_serve_port("127.0.0.1", None, model="model", listen_fd=17)

    assert caught.value.code == 2
    assert capsys.readouterr().err == ("Invalid --listen-fd 17: descriptor is closed\n")


def test_listen_fd_rejects_non_tcp_socket():
    left, right = socket.socketpair()
    with left, right, pytest.raises(OSError, match="not bound to a TCP socket"):
        cli._listen_fd_port(left.fileno())


def test_listen_fd_rejects_bound_udp_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.bind(("127.0.0.1", 0))
        with pytest.raises(OSError, match="not bound to a TCP socket"):
            cli._listen_fd_port(udp.fileno())


def test_listen_fd_rejects_tcp_socket_that_is_not_listening():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        with pytest.raises(OSError, match="SO_ACCEPTCONN is false"):
            cli._listen_fd_port(stream.fileno())


def test_listen_fd_plain_file_failure_does_not_leak_dup(tmp_path):
    if not os.path.isdir("/dev/fd"):
        pytest.skip("platform does not expose /dev/fd")

    path = tmp_path / "plain-file"
    path.write_text("not a socket", encoding="utf-8")
    fd = os.open(path, os.O_RDONLY)
    try:
        before = len(os.listdir("/dev/fd"))
        for _ in range(50):
            with pytest.raises(OSError):
                cli._listen_fd_port(fd)
        after = len(os.listdir("/dev/fd"))
        os.fstat(fd)
    finally:
        os.close(fd)

    assert after == before


def _fake_inherited_socket(*, sockname):
    class FakeInheritedSocket:
        family = socket.AF_INET

        def __init__(self, *, fileno):
            self.fileno = fileno

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            os.close(self.fileno)

        def getsockopt(self, _level, option, *_args):
            if option == socket.SO_TYPE:
                return socket.SOCK_STREAM
            return 1

        def getsockname(self):
            return sockname

    return FakeInheritedSocket


class _FakeSocketOptions:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def getsockopt(self, *args):
        self.calls.append(args)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _listener_accepting(sock, *, platform_name="linux", tcp_connection_info=42):
    return cli._listener_accepting(
        sock,
        so_acceptconn=30,
        platform_name=platform_name,
        enoprotoopt=errno.ENOPROTOOPT,
        tcp_connection_info=tcp_connection_info,
        sol_socket=1,
        ipproto_tcp=6,
    )


@pytest.mark.parametrize(("acceptconn", "expected"), [(1, True), (0, False)])
def test_listener_accepting_uses_so_acceptconn(acceptconn, expected):
    sock = _FakeSocketOptions(acceptconn)

    assert _listener_accepting(sock) is expected
    assert sock.calls == [(1, 30)]


@pytest.mark.parametrize(("state", "expected"), [(1, True), (0, False)])
def test_listener_accepting_uses_darwin_tcp_connection_info(state, expected):
    tcp_info = bytes([state]) + bytes(cli._DARWIN_TCP_CONNECTION_INFO_SIZE - 1)
    sock = _FakeSocketOptions(OSError(errno.ENOPROTOOPT, "unsupported"), tcp_info)

    assert _listener_accepting(sock, platform_name="darwin") is expected
    assert sock.calls == [
        (1, 30),
        (6, 42, cli._DARWIN_TCP_CONNECTION_INFO_SIZE),
    ]


def test_listener_accepting_reraises_enoprotoopt_off_darwin():
    error = OSError(errno.ENOPROTOOPT, "unsupported")
    sock = _FakeSocketOptions(error)

    with pytest.raises(OSError) as excinfo:
        _listener_accepting(sock)

    assert excinfo.value is error


def test_listener_accepting_reraises_other_errno_on_darwin():
    error = OSError(errno.EINVAL, "unexpected socket option failure")
    sock = _FakeSocketOptions(error)

    with pytest.raises(OSError) as excinfo:
        _listener_accepting(sock, platform_name="darwin")

    assert excinfo.value is error


def test_listener_accepting_reraises_without_tcp_connection_info():
    error = OSError(errno.ENOPROTOOPT, "unsupported")
    sock = _FakeSocketOptions(error)

    with pytest.raises(OSError) as excinfo:
        _listener_accepting(sock, platform_name="darwin", tcp_connection_info=None)

    assert excinfo.value is error


def test_listener_accepting_fails_closed_without_listener_state_options():
    sock = _FakeSocketOptions()

    assert not cli._listener_accepting(
        sock,
        so_acceptconn=None,
        platform_name="linux",
        enoprotoopt=errno.ENOPROTOOPT,
        tcp_connection_info=None,
        sol_socket=1,
        ipproto_tcp=6,
    )
    assert sock.calls == []


def test_listener_accepting_uses_darwin_fallback_without_so_acceptconn():
    tcp_info = bytes([1]) + bytes(cli._DARWIN_TCP_CONNECTION_INFO_SIZE - 1)
    sock = _FakeSocketOptions(tcp_info)

    assert cli._listener_accepting(
        sock,
        so_acceptconn=None,
        platform_name="darwin",
        enoprotoopt=errno.ENOPROTOOPT,
        tcp_connection_info=42,
        sol_socket=1,
        ipproto_tcp=6,
    )
    assert sock.calls == [(6, 42, cli._DARWIN_TCP_CONNECTION_INFO_SIZE)]


def test_listen_fd_rejects_invalid_inet_sockname(monkeypatch):
    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setattr(
            socket,
            "socket",
            _fake_inherited_socket(sockname="not-an-inet-address"),
        )
        with pytest.raises(OSError, match="not bound to a TCP listener"):
            cli._listen_fd_port(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_serve_lane_port_invariant_rejects_none():
    with pytest.raises(AssertionError, match="unresolved port"):
        cli._resolved_serve_port(types.SimpleNamespace(port=None))


def test_listen_fd_resolves_once_before_audio_lane(monkeypatch):
    seen: list[int | None] = []
    resolve_calls = 0
    real_resolve = cli._resolve_serve_port

    def tracked_resolve(*args, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(cli, "_resolve_serve_port", tracked_resolve)
    monkeypatch.setattr(cli, "_run_optional_runtime_guard", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_offline_hub_mode_active", lambda: False)
    monkeypatch.setattr(
        cli, "_serve_audio_mode", lambda args, _entry: seen.append(args.port)
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        bound_port = listener.getsockname()[1]
        args = Namespace(
            model="kokoro",
            embedding_model=None,
            served_model_name=None,
            no_mllm=True,
            mllm=False,
            max_tokens=None,
            api_key=None,
            timeout=60,
            max_request_bytes=None,
            cors_origins=None,
            rate_limit=0,
            log_level="INFO",
            host="127.0.0.1",
            port=None,
            listen_fd=listener.fileno(),
        )
        cli.serve_command(args)

    assert resolve_calls == 1
    assert seen == [bound_port]


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
