# SPDX-License-Identifier: Apache-2.0
"""Tests for the signal-observability hook installed by the FastAPI lifespan.

Covers:
  * ``install_signal_observability`` registers a handler for SIGTERM /
    SIGHUP / SIGABRT, saves the prior handler, and chains to it.
  * ``faulthandler.dump_traceback`` is invoked on signal receipt.
  * The latch is idempotent (repeat installs don't stack handlers).
  * The C-04 recon symptom (silent server death) is now observable:
    sending SIGTERM to a process running the install + a tiny event loop
    produces the documented WARNING line and a thread-stack dump on
    stderr before the chained default handler runs.

The C-level signals (SIGSEGV, SIGBUS, …) handled by ``faulthandler.enable``
are NOT exercised in unit tests — actually raising them in-process would
kill the test runner. The presence of ``faulthandler.is_enabled()`` after
the install is the smoke-test surface.
"""

from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import sys
import textwrap
import threading
from pathlib import Path


def _read_ready_with_timeout(proc: subprocess.Popen, *, timeout: float = 10.0) -> str:
    """Read a single line from ``proc.stdout`` but give up after
    ``timeout`` seconds even if the child never prints anything.

    Codex r7 BLOCKING #2: the previous tests used
    ``proc.stdout.readline()`` with no timeout, so a child that died
    before printing ``READY`` would hang CI indefinitely (both pipes
    still open in the parent). ``select.select`` on the underlying
    file descriptor bounds the wait safely without needing the
    extra ``communicate(timeout=...)`` dance.

    Returns the line read (with trailing newline stripped) or raises
    ``AssertionError`` on timeout — we'd rather report a fast
    failure than wait for the CI test-runner timeout to fire.
    """
    fd = proc.stdout.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        # Drain whatever the child produced so the AssertionError
        # message tells the operator something useful.
        proc.kill()
        out_tail, err_tail = proc.communicate(timeout=5)
        raise AssertionError(
            f"subprocess did not emit READY within {timeout:.1f}s;"
            f" stdout-tail={out_tail!r}, stderr-tail={err_tail!r}"
        )
    line = proc.stdout.readline()
    if line == "":
        # EOF on stdout — the subprocess died before printing READY.
        # ``select`` returns ready on EOF too, so we hit this branch
        # without a timeout. Surface stderr so the operator sees the
        # actual crash reason instead of a cryptic ``assert '' ==
        # 'READY'`` with no diagnostic context. This is the only
        # signal-test mode where we get an empty readline; under
        # normal operation the child writes ``READY\n`` exactly once
        # before any signal can land.
        try:
            err_tail = proc.stderr.read() or ""
        except (OSError, ValueError):
            err_tail = "<stderr read failed>"
        raise AssertionError(
            "subprocess died before emitting READY;"
            f" returncode={proc.returncode!r}, stderr={err_tail!r}"
        )
    return line


def test_install_is_idempotent_and_saves_prior_handlers():
    """Repeated installs must not stack handlers (each install would
    otherwise add a layer that re-runs the dump on every signal)."""
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    try:
        # Install a sentinel SIGUSR1 prior handler so we can detect
        # chaining behaviour without touching SIGTERM (which would
        # actually kill the test runner if our chain misbehaves).
        sentinel_calls: list[int] = []

        def _sentinel(signum, frame):  # noqa: ARG001
            sentinel_calls.append(signum)

        # Save+restore SIGUSR1 around the test so we don't leak state.
        prior_usr1 = signal.signal(signal.SIGUSR1, _sentinel)
        try:
            ok = so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
            assert ok is True
            handlers_after_first = dict(so._get_installed_handlers())
            assert signal.SIGUSR1 in handlers_after_first
            assert handlers_after_first[signal.SIGUSR1] is _sentinel

            # Second install must be a no-op (idempotent).
            ok2 = so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
            assert ok2 is True
            handlers_after_second = dict(so._get_installed_handlers())
            assert handlers_after_second == handlers_after_first
        finally:
            # Codex r7 BLOCKING #1: ``_reset_for_tests`` restores the
            # handler from its saved-prior map (which is _sentinel)
            # back on top of whatever we set, so it MUST run BEFORE we
            # restore the outer test's prior. Inverting the order
            # leaves _sentinel as the live SIGUSR1 handler.
            so._reset_for_tests()
            signal.signal(signal.SIGUSR1, prior_usr1)
    finally:
        # Belt-and-braces in case the inner try raised before reaching
        # its own ``finally`` — same ordering invariant.
        so._reset_for_tests()


def test_signal_chain_calls_prior_handler():
    """Receiving the signal must invoke the prior handler so uvicorn's
    graceful shutdown still fires."""
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()

    invoked: list[int] = []

    def _prior(signum, frame):  # noqa: ARG001
        invoked.append(signum)

    prior_usr1 = signal.signal(signal.SIGUSR1, _prior)
    try:
        so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
        os.kill(os.getpid(), signal.SIGUSR1)
        # Signal delivery is synchronous on POSIX; by the time
        # ``os.kill`` returns and Python re-acquires the GIL, the
        # handler chain has run.
        assert invoked == [signal.SIGUSR1]
    finally:
        # Codex r7 BLOCKING #1: reset BEFORE restoring our outer prior
        # so ``_reset_for_tests`` doesn't reinstall ``_prior`` on top
        # of the test-provided handler we're about to put back.
        so._reset_for_tests()
        signal.signal(signal.SIGUSR1, prior_usr1)


def test_install_chains_to_sig_dfl_via_restore_and_raise():
    """Codex r2 BLOCKING #1 follow-up: when the prior handler is
    ``SIG_DFL``, the chain must restore the default disposition and
    re-raise via ``signal.raise_signal`` so the kernel-level
    terminate-by-default fires after the WARNING + stack dump. Without
    this, SIGHUP — whose default disposition under uvicorn is SIG_DFL
    because uvicorn only captures SIGINT/SIGTERM — would be silently
    swallowed in production despite the install (the exact silent-death
    shape C-04 is trying to make observable).

    We exercise the mechanism on SIGUSR1 (the prior is SIG_DFL by
    default, but its default action is "terminate" same as SIGHUP, so
    we use it as a safe proxy that won't disturb the test runner's
    SIGTERM / SIGHUP handlers).
    """
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()

    # Use a subprocess so the actual termination fires in isolation
    # without killing the pytest runner. The subprocess installs our
    # observability over a fresh SIG_DFL prior, then sends itself
    # SIGUSR1 — the chain should log + dump + terminate.
    program = textwrap.dedent(
        """
        import faulthandler, logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        # Confirm we start from SIG_DFL.
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        from rapid_mlx._signal_observability import install_signal_observability
        assert install_signal_observability(observed_signals=(signal.SIGUSR1,)) is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGUSR1)
        # Give the signal time to deliver + chain. If we reach the
        # ``os._exit(99)`` below the chain swallowed the signal — that's
        # the failure mode this test is pinning.
        time.sleep(2.0)
        os._exit(99)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_ready_with_timeout(proc)
        assert ready.strip() == "READY", ready
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGUSR1" in stderr, stderr
    # Default disposition for SIGUSR1 is terminate (signed exit). The
    # ``os._exit(99)`` line MUST NOT be reached — if it is, the chain
    # silently swallowed the signal and r2 BLOCKING #1 has regressed.
    assert proc.returncode != 99, (
        f"chain swallowed the signal — process exited via os._exit(99) "
        f"instead of being terminated by SIG_DFL re-raise; stderr={stderr!r}"
    )


def test_install_returns_false_when_no_signals_could_be_installed():
    """Codex r2 BLOCKING #2 + r7 NIT #3: a no-op install must NOT
    register anything. If every ``signal.signal`` call rejected (empty
    list, or all platform-rejected signals), a later legitimate install
    from a different entry point must still succeed.
    """
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    try:
        # Empty signal set → installed_any stays False.
        result = so.install_signal_observability(observed_signals=())
        assert result is False
        # The per-signal map stays empty — a follow-on install for the
        # real default ``(SIGTERM, SIGHUP)`` set can still register.
        assert so._get_installed_handlers() == {}
    finally:
        so._reset_for_tests()


def test_per_signal_latch_does_not_block_later_default_install():
    """Codex r7 NIT #3 follow-up: a narrow custom install (e.g.
    ``(SIGUSR1,)`` from a test) must NOT latch out a subsequent
    install for additional signals. The earlier global-latch version
    regressed here — the second call returned True without actually
    registering the new requested signals.

    Uses SIGUSR1 + SIGUSR2 so we never touch SIGTERM/SIGHUP (which
    pytest reserves for its own teardown signaling).
    """
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()

    # Save both USR slots so we can restore.
    def _sentinel1(signum, frame):  # noqa: ARG001
        pass

    def _sentinel2(signum, frame):  # noqa: ARG001
        pass

    prior_usr1 = signal.signal(signal.SIGUSR1, _sentinel1)
    prior_usr2 = signal.signal(signal.SIGUSR2, _sentinel2)
    try:
        # First: narrow custom install for SIGUSR1.
        ok1 = so.install_signal_observability(observed_signals=(signal.SIGUSR1,))
        assert ok1 is True
        assert signal.SIGUSR1 in so._get_installed_handlers()
        assert signal.SIGUSR2 not in so._get_installed_handlers()

        # Second: add SIGUSR2. Earlier global-latch version returned
        # True without actually installing SIGUSR2 — that's the
        # regression this test pins.
        ok2 = so.install_signal_observability(
            observed_signals=(signal.SIGUSR1, signal.SIGUSR2)
        )
        assert ok2 is True
        handlers = so._get_installed_handlers()
        assert signal.SIGUSR1 in handlers
        assert signal.SIGUSR2 in handlers
    finally:
        so._reset_for_tests()
        signal.signal(signal.SIGUSR1, prior_usr1)
        signal.signal(signal.SIGUSR2, prior_usr2)


def test_install_skipped_off_main_thread():
    """Calling install from a worker thread must return False rather
    than raising — the server must still boot."""
    from rapid_mlx import _signal_observability as so

    result_box: list[bool] = []

    def _worker():
        result_box.append(so.install_signal_observability())

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert result_box == [False]


def test_reset_ignores_signal_restore_failure(monkeypatch):
    from rapid_mlx import _signal_observability as so

    so._prior_handlers[signal.SIGUSR1] = signal.SIG_DFL
    monkeypatch.setattr(
        so.signal,
        "signal",
        lambda *_args: (_ for _ in ()).throw(ValueError("restore failed")),
    )

    so._reset_for_tests()

    assert so._get_installed_handlers() == {}


def test_faulthandler_is_enabled_after_install():
    """``faulthandler.enable`` must fire so SIGSEGV from MLX produces a
    Python traceback rather than a silent core dump.

    Codex r8 NIT: capture the prior enabled-state and restore it in the
    ``finally`` block. The previous revision unconditionally called
    ``faulthandler.disable()`` and never restored it, so if the test
    ran inside a runner that had ``faulthandler`` pre-enabled (the
    ``-X faulthandler`` interpreter flag, the pytest ``--faulthandler``
    option, or any earlier test that turned it on) we'd silently leak
    the disable into the rest of the suite — the next SIGSEGV would
    crash without the traceback the operator relies on.
    """
    import faulthandler

    from rapid_mlx import _signal_observability as so

    was_enabled = faulthandler.is_enabled()
    so._reset_for_tests()
    try:
        # Disable first so we can prove the install enabled it.
        faulthandler.disable()
        assert not faulthandler.is_enabled()
        so.install_signal_observability(observed_signals=())
        assert faulthandler.is_enabled()
    finally:
        so._reset_for_tests()
        # Restore the enabled-state we observed on entry so the test
        # is a pure no-op w.r.t. global faulthandler state. The
        # ``install_signal_observability(observed_signals=())`` call
        # above already left it enabled, so we only need to disable
        # if it was originally off.
        if not was_enabled:
            faulthandler.disable()


def test_crash_file_is_private_rotated_and_previous_crash_reported_once(
    monkeypatch, tmp_path, capsys
):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".rapid-mlx" / "logs"
    log_dir.mkdir(parents=True, mode=0o777)
    for index in range(7):
        path = log_dir / f"crash-20260924T00000{index}Z-100.txt"
        path.write_text(f"old crash {index}\n", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))
    try:
        so.install_signal_observability(observed_signals=())
        files = sorted(log_dir.glob("crash-*.txt"))
        current = max(files, key=lambda path: path.stat().st_mtime_ns)

        inactive = [path for path in files if so._crash_file_pid(path) == 100]
        assert len(files) == 6
        assert len(inactive) == 5
        assert current.stat().st_mode & 0o777 == 0o600
        assert log_dir.stat().st_mode & 0o777 == 0o700
        lines = capsys.readouterr().err.splitlines()
        assert lines == [
            f"Previous run crashed; details in {log_dir / 'crash-20260924T000006Z-100.txt'} "
            "(and macOS DiagnosticReports under ~/Library/Logs/DiagnosticReports)."
        ]
    finally:
        so._reset_for_tests()


def test_rotation_never_unlinks_live_process_crash_files(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    paths = []
    for pid in range(200, 207):
        path = tmp_path / f"crash-20260924T000000000{pid}Z-{pid}.txt"
        path.write_text("traceback\n", encoding="utf-8")
        os.utime(path, (pid, pid))
        paths.append(path)
    live_pid = 200
    monkeypatch.setattr(so, "_pid_is_alive", lambda pid: pid == live_pid)

    so._rotate_crash_files(tmp_path)

    assert paths[0].exists()
    assert len(list(tmp_path.glob("crash-*.txt"))) == 6
    assert not paths[1].exists()


def test_crash_pid_probe_treats_permission_and_unknown_errors_as_alive(monkeypatch):
    from rapid_mlx import _signal_observability as so

    for error in (PermissionError(), OSError("probe failed")):
        monkeypatch.setattr(
            so.os,
            "kill",
            lambda *_args, error=error: (_ for _ in ()).throw(error),
        )
        assert so._pid_is_alive(123) is True


def test_empty_crash_file_is_removed_at_clean_shutdown(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    so.install_signal_observability(observed_signals=())
    files = list((tmp_path / ".rapid-mlx" / "logs").glob("crash-*.txt"))
    assert len(files) == 1
    assert files[0].stat().st_size == 0

    so._cleanup_crash_file()

    assert not files[0].exists()


def test_crash_file_creation_failure_warns_and_does_not_block(
    monkeypatch, tmp_path, caplog
):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        so.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    assert so.install_signal_observability(observed_signals=()) is False
    assert "could not create durable rapid-mlx crash file" in caplog.text


def test_missing_tee_warns_once_and_uses_file_only(monkeypatch, tmp_path, caplog):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    prior_warned = so._tee_fallback_warned
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        so.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("tee")),
    )
    so._tee_fallback_warned = False
    try:
        so.install_signal_observability(observed_signals=())
        path = so._crash_path
        assert path is not None
        assert so._crash_fd is not None
        os.write(so._crash_fd, b"fatal traceback\n")
        so._cleanup_crash_file()

        assert path.stat().st_size > 0

        so.install_signal_observability(observed_signals=())
        so._cleanup_crash_file()
        warnings = [
            record
            for record in caplog.records
            if "using crash file only" in record.getMessage()
        ]
        assert len(warnings) == 1
    finally:
        so._reset_for_tests()
        so._tee_fallback_warned = prior_warned


def test_unsupported_tee_platform_uses_file_only(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    log_dir = tmp_path / ".rapid-mlx" / "logs"
    try:
        with monkeypatch.context() as patch:
            patch.setattr(so, "_crash_logs_dir", lambda: log_dir)
            patch.setattr(so.os, "name", "nt")
            so.install_signal_observability(observed_signals=())
        assert so._crash_fd is not None
        assert so._crash_pipe is None
        assert so._crash_tee is None
    finally:
        so._reset_for_tests()


def test_tee_without_stdin_falls_back_to_file(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    class NoStdinProcess:
        stdin = None

        def wait(self, *, timeout):
            return 0

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        so.subprocess, "Popen", lambda *_args, **_kwargs: NoStdinProcess()
    )
    try:
        so.install_signal_observability(observed_signals=())
        assert so._crash_fd is not None
        assert so._crash_pipe is None
        assert so._crash_tee is None
    finally:
        so._reset_for_tests()


def test_tee_shutdown_errors_are_inert():
    from rapid_mlx import _signal_observability as so

    class BrokenPipe:
        def close(self):
            raise OSError("close failed")

    class StuckProcess:
        terminated = False

        def wait(self, *, timeout):
            raise subprocess.TimeoutExpired("tee", timeout)

        def terminate(self):
            self.terminated = True

    process = StuckProcess()
    so._stop_crash_tee(process, BrokenPipe())

    assert process.terminated is True


def test_crash_file_scan_and_io_failures_are_inert(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    class BadEntry:
        def stat(self):
            raise OSError("stat failed")

    class BadDirectory:
        def glob(self, _pattern):
            raise OSError("glob failed")

    class OneBadEntry:
        def glob(self, _pattern):
            return [BadEntry()]

    assert so._crash_files(BadDirectory()) == []
    assert so._crash_files(OneBadEntry()) == []

    crash = tmp_path / "crash-old.txt"
    crash.write_text("fatal\n", encoding="utf-8")
    monkeypatch.setattr(so, "_crash_files", lambda _directory: [crash] * 6)

    class BrokenStderr:
        closed = False

        def write(self, _text):
            raise ValueError("closed")

        def flush(self):
            raise AssertionError("write should fail first")

    monkeypatch.setattr(so.sys, "stderr", BrokenStderr())
    so._report_previous_crash(tmp_path)

    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")),
    )
    so._rotate_crash_files(tmp_path)


def test_crash_cleanup_and_failed_install_cleanup_errors_are_inert(
    monkeypatch, tmp_path
):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    path = tmp_path / "open-crash.txt"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    so._crash_fd = fd
    prior_enabled = so._faulthandler_was_enabled

    class BrokenPath:
        def stat(self):
            raise OSError("stat failed")

    so._crash_path = BrokenPath()
    so._faulthandler_was_enabled = True
    real_close = os.close

    def close_then_raise(open_fd):
        real_close(open_fd)
        raise OSError("close reported failure")

    monkeypatch.setattr(so.os, "close", close_then_raise)
    monkeypatch.setattr(
        so.faulthandler,
        "disable",
        lambda: (_ for _ in ()).throw(RuntimeError("disable failed")),
    )
    monkeypatch.setattr(
        so.faulthandler,
        "enable",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("enable failed")),
    )
    so._cleanup_crash_file()
    so._faulthandler_was_enabled = prior_enabled

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    real_unlink = Path.unlink

    def unlink_then_raise(target, *args, **kwargs):
        real_unlink(target, *args, **kwargs)
        raise OSError("unlink reported failure")

    monkeypatch.setattr(Path, "unlink", unlink_then_raise)
    so._install_crash_file()


def test_abort_subprocess_leaves_nonempty_durable_crash_file(tmp_path):
    home = tmp_path / "home"
    env = dict(os.environ, HOME=str(home))
    program = textwrap.dedent(
        """
        import os
        from rapid_mlx._signal_observability import install_signal_observability

        install_signal_observability(observed_signals=())
        os.abort()
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    files = list((home / ".rapid-mlx" / "logs").glob("crash-*.txt"))
    assert proc.returncode != 0
    assert len(files) == 1
    assert files[0].stat().st_size > 0
    assert "Fatal Python error" in files[0].read_text(encoding="utf-8")
    assert "Fatal Python error" in proc.stderr


def test_crash_file_survives_later_faulthandler_registration(tmp_path):
    home = tmp_path / "home"
    alternate = tmp_path / "alternate.txt"
    env = dict(os.environ, HOME=str(home))
    program = textwrap.dedent(
        f"""
        import faulthandler
        import os
        from rapid_mlx._signal_observability import install_signal_observability

        install_signal_observability(observed_signals=())
        with open({os.fspath(alternate)!r}, "w", encoding="utf-8") as alternate:
            faulthandler.enable(file=alternate, all_threads=True)
            install_signal_observability(observed_signals=())
            os.abort()
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    files = list((home / ".rapid-mlx" / "logs").glob("crash-*.txt"))
    assert proc.returncode != 0
    assert len(files) == 1
    assert files[0].stat().st_size > 0
    assert "Fatal Python error" in files[0].read_text(encoding="utf-8")
    assert "Fatal Python error" in proc.stderr


def test_package_has_one_faulthandler_enable_call_site():
    package = Path(__file__).resolve().parents[1] / "rapid_mlx"
    call = re.compile(r"^\s*faulthandler\.enable\(")
    matches = [
        (path.relative_to(package), line.strip())
        for path in package.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if call.match(line)
    ]

    assert matches == [
        (
            Path("_signal_observability.py"),
            "faulthandler.enable(file=target_fd, all_threads=True)",
        )
    ]


def test_subprocess_sigterm_emits_warning_and_stack_dump():
    """End-to-end: spawn a child running the install + an idle loop,
    send SIGTERM, assert the WARNING marker + thread-stack dump appear
    on stderr BEFORE the process exits.

    This is the C-04 recon symptom reproduction — without the hook the
    child would die between two stdout writes with no log line. With
    the hook the operator sees a single-line WARNING + per-thread
    traceback even when the SIGTERM landed mid-handler.
    """
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        # Route the standard logger to stderr so a single capture surface
        # picks up BOTH the WARNING marker and the faulthandler dump.
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        # Replace SIG_DFL chain target with a clean exit so we don't
        # produce a misleading exit code (default SIGTERM = killed-by-15).
        def _exit_handler(signum, frame):
            sys.stderr.flush()
            os._exit(0)
        signal.signal(signal.SIGTERM, _exit_handler)

        from rapid_mlx._signal_observability import install_signal_observability
        assert install_signal_observability() is True

        # Tell the parent we're ready to be signalled.
        sys.stdout.write("READY\\n")
        sys.stdout.flush()

        # Idle until signal lands.
        for _ in range(50):
            time.sleep(0.1)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for READY before sending SIGTERM so we don't race the
        # install. ``readline`` blocks until the child flushes.
        ready_line = _read_ready_with_timeout(proc)
        assert ready_line.strip() == "READY", ready_line
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    # Documented WARNING shape.
    assert "received signal SIGTERM" in stderr, stderr
    # faulthandler.dump_traceback shape — the header line it writes
    # starts with ``Current thread`` or ``Thread`` depending on whether
    # all_threads dumped multiple threads.
    assert "Thread" in stderr or "Current thread" in stderr, stderr


def test_subprocess_sighup_default_disposition_dumps_and_terminates():
    """SIGHUP keeps its default termination after the diagnostic dump.

    This is the production baseline under uvicorn, which does not install a
    SIGHUP handler. Swallowing it strands servers when tmux or another
    supervisor closes the controlling session.
    """
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        # Establish the production baseline explicitly: uvicorn does not
        # install a SIGHUP handler, so SIGHUP starts at SIG_DFL. We SET it
        # rather than ASSERT it — a SIG_IGN leaked by an earlier test in the
        # parent pytest process is inherited across exec (POSIX: SIG_IGN
        # survives execve, unlike caught handlers) and would otherwise fail
        # this unrelated precondition, a full-suite ordering flake. The real
        # contract is proven below: observability keeps SIGHUP terminating.
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        from rapid_mlx._signal_observability import install_signal_observability
        assert install_signal_observability() is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        # Reaching this exit means the observability hook swallowed SIGHUP.
        for _ in range(20):
            time.sleep(0.1)
        sys.stdout.write("ALIVE\\n"); sys.stdout.flush()
        os._exit(0)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_line = _read_ready_with_timeout(proc)
        assert ready_line.strip() == "READY", ready_line
        proc.send_signal(signal.SIGHUP)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    # WARNING marker and stacks land before default termination.
    assert "received signal SIGHUP" in stderr, stderr
    # faulthandler.dump_traceback shape — verifies stack-dump fired.
    assert "Thread" in stderr or "Current thread" in stderr, stderr
    assert proc.returncode == -signal.SIGHUP, (
        f"SIGHUP was swallowed: returncode={proc.returncode} "
        f"stdout={stdout!r} stderr={stderr!r}"
    )
    assert "ALIVE" not in stdout


def test_subprocess_sigterm_default_disposition_still_terminates():
    """R7-C1 invariant guard: the SIGHUP-stays-alive change must NOT
    leak into SIGTERM. SIGTERM with a SIG_DFL prior (no uvicorn
    installed yet, e.g. unit tests that mount the lifespan without
    binding the socket) must still terminate the process — that's the
    PR #820 contract Liang r5 verified for graceful drain. Only
    SIGHUP gets the dump-and-stay-alive short-circuit.

    We use SIGUSR1 as a proxy for "SIG_DFL-defaults-to-terminate
    signal that isn't SIGHUP" so the test runs without disturbing the
    test runner's SIGTERM handler. SIGUSR1's default action is
    "terminate" same as SIGTERM/SIGHUP, so it exercises the same
    SIG_DFL chain branch.
    """
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
        from rapid_mlx._signal_observability import install_signal_observability
        assert install_signal_observability(observed_signals=(signal.SIGUSR1,)) is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGUSR1)
        # If the chain incorrectly swallowed the signal (SIGHUP-style
        # short-circuit leaking to other signals), we'd fall through
        # to os._exit(99) and the test would catch it.
        time.sleep(2.0)
        os._exit(99)
        """
    ).strip()

    proc = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready_line = _read_ready_with_timeout(proc)
        assert ready_line.strip() == "READY", ready_line
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGUSR1" in stderr, stderr
    # Process must have been terminated by the SIG_DFL re-raise (NOT
    # by hitting os._exit(99) which would mean the SIGHUP short-
    # circuit leaked to non-SIGHUP signals).
    assert proc.returncode != 99, (
        f"non-SIGHUP signal was swallowed — the R7-C1 stay-alive"
        f" short-circuit must be gated on SIGHUP only;"
        f" stderr={stderr!r}"
    )
