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

import builtins
import json
import os
import re
import select
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _wait_for_nonempty_file(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.stat().st_size > 0:
            return
        time.sleep(0.01)
    raise AssertionError(f"{path} stayed empty for {timeout:.1f}s")


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
    monkeypatch.setattr(so, "process_identity", lambda _pid: None)
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


def test_acknowledged_hard_link_suppresses_repeat_after_unlink_failure(
    monkeypatch, tmp_path, capsys
):
    from rapid_mlx import _signal_observability as so

    crash = tmp_path / "crash-20260924T000000Z-99999999.txt"
    crash.write_text("fatal traceback\n", encoding="utf-8")
    real_unlink = Path.unlink

    def fail_original_unlink(path, *args, **kwargs):
        if path == crash:
            raise PermissionError("simulated unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_original_unlink)

    so._report_previous_crash(tmp_path)
    first = capsys.readouterr().err
    so._report_previous_crash(tmp_path)
    second = capsys.readouterr().err

    assert "Previous run crashed" in first
    assert second == ""
    assert crash.exists()
    assert any(".reported" in path.name for path in tmp_path.iterdir())


def test_tee_keeps_stderr_mirror_when_durable_sink_fails():
    from rapid_mlx import _signal_observability as so

    result = subprocess.run(
        [sys.executable, "-c", so._CRASH_TEE_SCRIPT, "999999"],
        input=b"fatal traceback\nsecond chunk\n",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b"fatal traceback\nsecond chunk\n"


def test_rotation_never_unlinks_live_process_crash_files(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    paths = []
    for pid in range(200, 207):
        path = tmp_path / f"crash-20260924T000000000{pid}Z-{pid}.txt"
        path.write_text("traceback\n", encoding="utf-8")
        os.utime(path, (pid, pid))
        paths.append(path)
    live_pid = 200
    monkeypatch.setattr(
        so,
        "_marker_for_pid",
        lambda _log_dir, pid: {"pid": pid} if pid == live_pid else None,
    )
    monkeypatch.setattr(
        so,
        "is_same_process",
        lambda marker: marker["pid"] == live_pid,
    )
    monkeypatch.setattr(so, "process_identity", lambda _pid: None)

    so._rotate_crash_files(tmp_path)

    assert paths[0].exists()
    assert len(list(tmp_path.glob("crash-*.txt"))) == 6
    assert not paths[1].exists()


def test_rotation_retains_live_process_before_marker_exists(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so
    from rapid_mlx._process_identity import ProcessIdentity

    live_pid = 4567
    live = tmp_path / f"crash-20260924T000000000000Z-{live_pid}.txt"
    live.write_text("", encoding="utf-8")
    for pid in range(100, 106):
        path = tmp_path / f"crash-20260923T000000000{pid}Z-{pid}.txt"
        path.write_text("old\n", encoding="utf-8")
        os.utime(path, (pid, pid))
    monkeypatch.setattr(so, "_marker_for_pid", lambda *_args: None)
    monkeypatch.setattr(
        so,
        "process_identity",
        lambda pid: ProcessIdentity(pid, 1.0, 1.0) if pid == live_pid else None,
    )

    so._rotate_crash_files(tmp_path)

    assert live.exists()


def test_rotation_treats_reused_pid_marker_as_inactive(tmp_path):
    from rapid_mlx import _signal_observability as so
    from rapid_mlx._process_identity import process_identity

    identity = process_identity(os.getpid())
    assert identity is not None
    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    log_dir.mkdir()
    state_dir.mkdir()
    crash = log_dir / f"crash-20260924T000000000000Z-{os.getpid()}.txt"
    crash.write_text("traceback\n", encoding="utf-8")
    marker = state_dir / f"serve-inflight-{os.getpid()}.json"
    marker.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "create_time": identity.create_time,
                "boot_time": identity.boot_time - 100.0,
                "app_version": "0.15.1",
            }
        ),
        encoding="utf-8",
    )

    assert so._crash_file_is_live(crash, log_dir) is False


def test_windows_liveness_never_calls_os_kill(monkeypatch):
    from rapid_mlx import _process_identity as identity
    from rapid_mlx import _signal_observability as so
    from rapid_mlx.telemetry import server_start

    marker = {
        "pid": 123,
        "create_time": 1.0,
        "boot_time": 1.0,
        "app_version": "0.15.1",
    }
    monkeypatch.setattr(identity, "psutil", None)
    monkeypatch.setattr(identity.sys, "platform", "win32")
    monkeypatch.setattr(
        identity.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("os.kill called")),
    )

    assert identity.process_identity(123) is None
    assert identity.is_same_process(marker) is True

    assert server_start.is_same_process is identity.is_same_process
    assert so.is_same_process is identity.is_same_process


def test_windows_without_psutil_identifies_current_process(monkeypatch, tmp_path):
    from rapid_mlx import _process_identity as identity
    from rapid_mlx.telemetry import server_start

    monkeypatch.setattr(identity, "psutil", None)
    monkeypatch.setattr(identity.sys, "platform", "win32")
    current = identity.process_identity(os.getpid())

    assert current == identity.ProcessIdentity(
        os.getpid(), identity._CURRENT_PROCESS_CREATE_TIME, 0.0
    )
    marker = tmp_path / "state" / f"serve-inflight-{os.getpid()}.json"
    server_start._atomic_write_marker(marker)
    assert json.loads(marker.read_text(encoding="utf-8"))["create_time"] == (
        identity._CURRENT_PROCESS_CREATE_TIME
    )

    stale = {
        "pid": os.getpid(),
        "create_time": identity._CURRENT_PROCESS_CREATE_TIME - 10.0,
        "boot_time": 0.0,
        "app_version": "0.15.1",
    }
    assert identity.is_same_process(stale) is False


def test_process_identity_import_without_psutil(monkeypatch):
    from rapid_mlx import _process_identity as identity

    real_import = builtins.__import__

    def import_without_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil unavailable")
        return real_import(name, *args, **kwargs)

    module_name = "rapid_mlx._process_identity_without_psutil"
    module = ModuleType(module_name)
    module.__file__ = identity.__file__
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(builtins, "__import__", import_without_psutil)

    source = Path(identity.__file__).read_text(encoding="utf-8")
    exec(compile(source, identity.__file__, "exec"), module.__dict__)

    assert module._psutil_module is None
    assert module.psutil is None


def test_process_identity_validation_and_probe_fallbacks(monkeypatch):
    from rapid_mlx import _process_identity as identity

    marker = {
        "pid": 123,
        "create_time": 1.0,
        "boot_time": 1.0,
        "app_version": "0.15.1",
    }
    assert identity.marker_identity([]) is None
    assert identity.process_identity(True) is None
    assert identity.is_same_process({}) is False

    no_such_process = type("NoSuchProcess", (Exception,), {})
    zombie_process = type("ZombieProcess", (Exception,), {})
    access_denied = type("AccessDenied", (Exception,), {})

    class MissingProcess:
        def create_time(self):
            raise no_such_process()

    fake_psutil = SimpleNamespace(
        pid_exists=lambda _pid: True,
        Process=lambda _pid: MissingProcess(),
        boot_time=lambda: 1.0,
        NoSuchProcess=no_such_process,
        ZombieProcess=zombie_process,
        AccessDenied=access_denied,
    )
    monkeypatch.setattr(identity, "psutil", fake_psutil)
    assert identity.process_identity(123) is None

    fake_psutil.Process = lambda _pid: SimpleNamespace(create_time=lambda: 1.0)
    fake_psutil.boot_time = lambda: (_ for _ in ()).throw(OSError("probe failed"))
    assert identity.process_identity(123) == identity.ProcessIdentity(123, 0.0, 0.0)
    assert identity.is_same_process(marker) is True

    monkeypatch.setattr(identity, "psutil", None)
    monkeypatch.setattr(identity.sys, "platform", "darwin")
    for error, alive in (
        (ProcessLookupError(), False),
        (PermissionError(), True),
        (OSError("probe failed"), True),
        (None, True),
    ):
        if error is None:
            monkeypatch.setattr(identity.os, "kill", lambda *_args: None)
        else:
            monkeypatch.setattr(
                identity.os,
                "kill",
                lambda *_args, error=error: (_ for _ in ()).throw(error),
            )
        assert (identity.process_identity(123) is not None) is alive
    assert identity.is_same_process(marker) is True

    monkeypatch.setattr(identity, "psutil", fake_psutil)
    monkeypatch.setattr(
        identity,
        "process_identity",
        lambda _pid: (_ for _ in ()).throw(access_denied()),
    )
    assert identity.is_same_process(marker) is True


@pytest.mark.parametrize("denied", [PermissionError("denied"), "access-denied"])
def test_identity_probe_denial_fails_closed_as_alive(monkeypatch, denied):
    from rapid_mlx import _process_identity as identity

    no_such_process = type("NoSuchProcess", (Exception,), {})
    access_denied = type("AccessDenied", (Exception,), {})
    error = access_denied("denied") if denied == "access-denied" else denied
    fake_psutil = SimpleNamespace(
        pid_exists=lambda _pid: True,
        Process=lambda _pid: SimpleNamespace(
            create_time=lambda: (_ for _ in ()).throw(error)
        ),
        boot_time=lambda: 1_600_000_000.0,
        NoSuchProcess=no_such_process,
        AccessDenied=access_denied,
    )

    marker = {
        "pid": 123,
        "create_time": 1_700_000_000.0,
        "boot_time": 1_600_000_000.0,
        "app_version": "0.15.1",
    }
    monkeypatch.setattr(identity, "psutil", fake_psutil)

    assert identity.is_same_process(marker) is True


def test_identity_exception_and_extreme_pid_matrix(monkeypatch, caplog):
    from rapid_mlx import _process_identity as identity

    caplog.set_level("DEBUG", logger="rapid_mlx._process_identity")
    marker = {
        "pid": 123,
        "create_time": 1_700_000_000.0,
        "boot_time": 1_600_000_000.0,
        "app_version": "0.15.1",
    }
    real = identity.psutil
    for exc in (
        PermissionError("denied"),
        real.AccessDenied(123),
        OSError("io"),
        RuntimeError("backend"),
    ):
        fake = SimpleNamespace(
            pid_exists=lambda _pid: True,
            Process=lambda _pid, exc=exc: SimpleNamespace(
                create_time=lambda: (_ for _ in ()).throw(exc)
            ),
            boot_time=lambda: 1_600_000_000.0,
            NoSuchProcess=real.NoSuchProcess,
            AccessDenied=real.AccessDenied,
        )
        monkeypatch.setattr(identity, "psutil", fake)
        assert identity.is_same_process(marker) is True

    fake.Process = lambda _pid: SimpleNamespace(create_time=lambda: 10**1000)
    monkeypatch.setattr(identity, "psutil", fake)
    assert identity.process_identity(123) == identity.ProcessIdentity(123, 0.0, 0.0)
    assert identity.is_same_process(marker) is True

    assert identity.is_same_process({**marker, "pid": 10**200}) is False
    assert "could not probe process identity" in caplog.text


def test_epoch_relative_tolerance_does_not_hide_pid_reuse(monkeypatch):
    from rapid_mlx import _process_identity as identity

    marker = {
        "pid": 123,
        "create_time": 1_700_000_000.0,
        "boot_time": 1_600_000_000.0,
        "app_version": "0.15.1",
    }
    monkeypatch.setattr(
        identity,
        "process_identity",
        lambda _pid: identity.ProcessIdentity(123, 1_700_000_001.0, 1_600_000_001.0),
    )

    assert identity.is_same_process(marker) is False


def test_crash_pointer_is_acknowledged_across_clean_launches(tmp_path, capsys):
    from rapid_mlx import _signal_observability as so

    home = tmp_path / "home"
    log_dir = home / ".rapid-mlx" / "logs"
    log_dir.mkdir(parents=True)
    crash = log_dir / "crash-20260924T000000000000Z-99999999.txt"
    crash.write_text("fatal traceback\n", encoding="utf-8")
    program = (
        "from rapid_mlx._signal_observability import "
        "install_signal_observability, _cleanup_crash_file; "
        "install_signal_observability(observed_signals=()); "
        "_cleanup_crash_file()"
    )
    env = dict(os.environ, HOME=str(home))
    outputs = [
        subprocess.run(
            [sys.executable, "-c", program],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stderr
        for _ in range(2)
    ]

    assert [value.count("Previous run crashed; details in") for value in outputs] == [
        1,
        0,
    ]
    assert not crash.exists()
    assert crash.with_name(f"{crash.stem}.reported{crash.suffix}").exists()

    so._report_previous_crash(log_dir)
    assert capsys.readouterr().err == ""


def test_second_server_does_not_acknowledge_live_server_crash_file(tmp_path, capsys):
    from rapid_mlx import _signal_observability as so

    home = tmp_path / "home"
    first_program = """
import os
from rapid_mlx.telemetry import server_start
server_start._atomic_write_marker(server_start._marker_path())
log_dir = server_start._marker_path().parent.parent / "logs"
log_dir.mkdir()
crash = log_dir / f"crash-20260924T000000000000Z-{os.getpid()}.txt"
crash.write_text("active diagnostic\\n", encoding="utf-8")
print(crash, flush=True)
input()
"""
    env = dict(os.environ, HOME=str(home))
    first = subprocess.Popen(
        [sys.executable, "-c", first_program],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        crash = Path(_read_ready_with_timeout(first).strip())
        so._report_previous_crash(crash.parent)

        assert "Previous run crashed" not in capsys.readouterr().err
        assert crash.exists()
        assert not crash.with_name(f"{crash.stem}.reported{crash.suffix}").exists()
    finally:
        assert first.stdin is not None
        first.stdin.write("stop\n")
        first.stdin.flush()
        first.communicate(timeout=10)


def test_reported_suffix_collision_is_acknowledged_once(tmp_path, capsys):
    from rapid_mlx import _signal_observability as so

    crash = tmp_path / "crash-20260924T000000Z-99999.txt"
    reported = tmp_path / "crash-20260924T000000Z-99999.reported.txt"
    crash.write_text("new-unreported-diagnostic", encoding="utf-8")
    reported.write_text("existing-reported-diagnostic", encoding="utf-8")

    pointer_counts = []
    for _ in range(2):
        so._report_previous_crash(tmp_path)
        pointer_counts.append(capsys.readouterr().err.count("Previous run crashed"))

    assert pointer_counts == [1, 0]
    assert reported.read_text(encoding="utf-8") == "existing-reported-diagnostic"
    assert not crash.exists()
    assert (
        crash.with_name(f"{crash.stem}.reported-1.txt").read_text(encoding="utf-8")
        == "new-unreported-diagnostic"
    )


def test_closed_crash_fd_rearm_recovers_and_reports_success(tmp_path):
    home = tmp_path / "home"
    program = (
        "import os; "
        "from rapid_mlx import _signal_observability as so; "
        "so.install_signal_observability(observed_signals=()); "
        "os.close(so._crash_fd); "
        "print(so.ensure_crash_sink(), flush=True); "
        "os.abort()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, HOME=str(home)),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    files = list((home / ".rapid-mlx" / "logs").glob("crash-*.txt"))

    assert proc.stdout.strip() == "True"
    assert proc.returncode != 0
    assert files and files[0].stat().st_size > 0


def test_repeated_install_recovers_closed_crash_fd(tmp_path):
    home = tmp_path / "home"
    program = (
        "import os; "
        "from rapid_mlx import _signal_observability as so; "
        "so.install_signal_observability(observed_signals=()); "
        "old_fd = so._crash_fd; "
        "os.close(old_fd); "
        "print(so.install_signal_observability(observed_signals=()), flush=True); "
        "print(so._crash_fd != old_fd or os.fstat(so._crash_fd).st_ino, flush=True)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, HOME=str(home)),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0] == "False"
    assert proc.stdout.splitlines()[1]


def test_closed_crash_fd_rearm_recovers_file_only_in_process(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    path = tmp_path / "crash.txt"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(fd)
    previous = (
        so._crash_fd,
        so._crash_fd_identity,
        so._crash_file_identity,
        so._crash_path,
        so._crash_pipe,
        so._crash_tee,
    )
    calls = []
    so._crash_fd = fd
    so._crash_fd_identity = (-1, -1)
    path_stat = path.stat()
    so._crash_file_identity = (path_stat.st_dev, path_stat.st_ino)
    so._crash_path = path
    so._crash_pipe = None
    so._crash_tee = None
    monkeypatch.setattr(so, "_enable_faulthandler", calls.append)
    try:
        assert so.ensure_crash_sink() is True
        assert so._crash_fd is not None
        assert calls == [so._crash_fd]
    finally:
        if so._crash_fd is not None:
            os.close(so._crash_fd)
        (
            so._crash_fd,
            so._crash_fd_identity,
            so._crash_file_identity,
            so._crash_path,
            so._crash_pipe,
            so._crash_tee,
        ) = previous


def test_closed_fd_reused_before_rearm_reopens_installed_crash_file(tmp_path):
    home = tmp_path / "home"
    program = """
import json, os
from pathlib import Path
from rapid_mlx import _signal_observability as so
so.install_signal_observability(observed_signals=())
crash = so._crash_path
old_fd = so._crash_fd
os.close(old_fd)
decoy_path = Path.home() / "decoy.txt"
opened = []
while not opened or opened[-1] < old_fd:
    opened.append(os.open(decoy_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600))
result = so.ensure_crash_sink()
os.write(old_fd, b"decoy remains open\\n")
print(json.dumps({"old_fd": old_fd, "decoy_fd": opened[-1], "result": result, "crash": str(crash)}), flush=True)
os.abort()
"""
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, HOME=str(home)),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    info = json.loads(proc.stdout)

    assert info["decoy_fd"] == info["old_fd"]
    assert info["result"] is True
    assert proc.returncode != 0
    assert Path(info["crash"]).stat().st_size > 0
    assert (home / "decoy.txt").read_text(encoding="utf-8") == "decoy remains open\n"


def test_exited_tee_is_replaced_with_direct_crash_file(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    class ExitedProcess:
        def poll(self):
            return 7

        def wait(self, *, timeout):
            return 7

    path = tmp_path / "crash.txt"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    pipe = os.fdopen(fd, "wb", buffering=0)
    file_stat = os.fstat(fd)
    previous = (
        so._crash_fd,
        so._crash_fd_identity,
        so._crash_file_identity,
        so._crash_path,
        so._crash_pipe,
        so._crash_tee,
    )
    so._crash_fd = fd
    so._crash_fd_identity = (file_stat.st_dev, file_stat.st_ino)
    so._crash_file_identity = (file_stat.st_dev, file_stat.st_ino)
    so._crash_path = path
    so._crash_pipe = pipe
    so._crash_tee = ExitedProcess()
    enabled = []
    monkeypatch.setattr(so, "_enable_faulthandler", enabled.append)
    try:
        assert so.ensure_crash_sink() is True
        assert pipe.closed is True
        assert so._crash_pipe is None
        assert so._crash_tee is None
        assert enabled == [so._crash_fd]
    finally:
        if so._crash_fd is not None:
            os.close(so._crash_fd)
        (
            so._crash_fd,
            so._crash_fd_identity,
            so._crash_file_identity,
            so._crash_path,
            so._crash_pipe,
            so._crash_tee,
        ) = previous


def test_closed_crash_fd_without_path_stops_tee_and_returns_false(tmp_path):
    from rapid_mlx import _signal_observability as so

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    path = tmp_path / "closed.txt"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(fd)
    pipe = Pipe()
    previous = (so._crash_fd, so._crash_path, so._crash_pipe, so._crash_tee)
    so._crash_fd = fd
    so._crash_path = None
    so._crash_pipe = pipe
    so._crash_tee = None
    try:
        assert so.ensure_crash_sink() is False
        assert pipe.closed is False
    finally:
        so._crash_fd, so._crash_path, so._crash_pipe, so._crash_tee = previous


def test_closed_crash_fd_rearm_failure_closes_replacement(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    previous = (
        so._crash_fd,
        so._crash_fd_identity,
        so._crash_file_identity,
        so._crash_path,
        so._crash_pipe,
        so._crash_tee,
    )
    so._crash_fd = 123
    so._crash_file_identity = (1, 2)
    so._crash_path = tmp_path / "crash.txt"
    so._crash_pipe = None
    so._crash_tee = None
    monkeypatch.setattr(
        so.os,
        "fstat",
        lambda fd: (
            (_ for _ in ()).throw(OSError("closed"))
            if fd == 123
            else SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG)
        ),
    )
    monkeypatch.setattr(so.os, "open", lambda *_args: 456)
    monkeypatch.setattr(
        so,
        "_enable_faulthandler",
        lambda _fd: (_ for _ in ()).throw(RuntimeError("rearm failed")),
    )
    monkeypatch.setattr(
        so.os, "close", lambda _fd: (_ for _ in ()).throw(OSError("close failed"))
    )
    try:
        assert so.ensure_crash_sink() is False
    finally:
        (
            so._crash_fd,
            so._crash_fd_identity,
            so._crash_file_identity,
            so._crash_path,
            so._crash_pipe,
            so._crash_tee,
        ) = previous


def test_ensure_crash_sink_without_installed_sink_returns_false():
    from rapid_mlx import _signal_observability as so

    previous = so._crash_fd
    so._crash_fd = None
    try:
        assert so.ensure_crash_sink() is False
    finally:
        so._crash_fd = previous


def test_crash_marker_reader_rejects_oversized_and_growing_files(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    log_dir.mkdir()
    state_dir.mkdir()
    marker = state_dir / "serve-inflight-123.json"
    marker.write_bytes(b"x" * 4097)
    assert so._marker_for_pid(log_dir, 123) is None

    marker.write_bytes(b"x")
    real_fstat = os.fstat
    real_read = os.read
    with monkeypatch.context() as patch:
        patch.setattr(
            so.os,
            "fstat",
            lambda fd: SimpleNamespace(st_mode=real_fstat(fd).st_mode, st_size=1),
        )
        patch.setattr(
            so.os,
            "read",
            lambda fd, size: b"x" * 4097 if size == 4097 else real_read(fd, size),
        )
        assert so._marker_for_pid(log_dir, 123) is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_crash_marker_reader_rejects_fifo_without_blocking(tmp_path):
    from rapid_mlx import _signal_observability as so

    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    log_dir.mkdir()
    state_dir.mkdir()
    os.mkfifo(state_dir / "serve-inflight-123.json")

    assert so._marker_for_pid(log_dir, 123) is None


def test_crash_marker_reader_handles_short_reads(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    log_dir = tmp_path / "logs"
    state_dir = tmp_path / "state"
    log_dir.mkdir()
    state_dir.mkdir()
    marker = state_dir / "serve-inflight-123.json"
    marker.write_text(
        json.dumps(
            {
                "pid": 123,
                "create_time": 1.0,
                "boot_time": 2.0,
                "app_version": "0.15.1",
            }
        ),
        encoding="utf-8",
    )
    real_read = os.read

    monkeypatch.setattr(so.os, "read", lambda fd, size: real_read(fd, min(size, 5)))

    parsed = so._marker_for_pid(log_dir, 123)
    assert parsed is not None
    assert parsed["pid"] == 123


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
    real_open = so.os.open

    def deny_crash_file(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            raise PermissionError("denied")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        so.os,
        "open",
        deny_crash_file,
    )

    assert so.install_signal_observability(observed_signals=()) is False
    assert "could not create durable rapid-mlx crash file" in caplog.text


def test_symlinked_crash_log_directory_is_refused(monkeypatch, tmp_path, caplog):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    home = tmp_path / "home"
    rapid_dir = home / ".rapid-mlx"
    victim = tmp_path / "victim"
    rapid_dir.mkdir(parents=True)
    victim.mkdir(mode=0o755)
    (rapid_dir / "logs").symlink_to(victim, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    fallback_targets = []
    monkeypatch.setattr(so, "_enable_faulthandler", fallback_targets.append)
    try:
        so.install_signal_observability(observed_signals=())

        assert so._crash_fd is None
        assert list(victim.iterdir()) == []
        assert victim.stat().st_mode & 0o777 == 0o755
        assert "crash log directory is unavailable" in caplog.text
        assert fallback_targets == [sys.stderr]
    finally:
        so._reset_for_tests()


def test_crash_log_directory_inode_swap_and_io_failure_are_refused(
    monkeypatch, tmp_path
):
    from rapid_mlx import _signal_observability as so

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(so.os, "fstat", lambda _fd: SimpleNamespace(st_dev=-1, st_ino=-1))
        assert so._prepare_crash_logs_dir(log_dir) is False


def test_crash_log_directory_uses_windows_compatible_validation(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(so.os, "name", "nt")
    monkeypatch.setattr(
        so.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Windows directory validation must not os.open a directory")
        ),
    )

    assert so._prepare_crash_logs_dir(log_dir) is True
    with monkeypatch.context() as patch:
        patch.setattr(
            so.os,
            "lstat",
            lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
        )
        assert so._prepare_crash_logs_dir(log_dir) is False


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


def test_stuck_tee_is_killed_and_reaped(monkeypatch):
    from rapid_mlx import _signal_observability as so

    class BrokenPipe:
        def close(self):
            raise OSError("close failed")

    class StuckProcess:
        terminated = False
        killed = False
        reaped = False
        waits = 0

        def wait(self, *, timeout=None):
            self.waits += 1
            if timeout is None:
                self.reaped = True
                return 0
            raise subprocess.TimeoutExpired("tee", timeout)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    process = StuckProcess()

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(so.threading, "Thread", ImmediateThread)
    so._stop_crash_tee(process, BrokenPipe())

    assert process.terminated is True
    assert process.killed is True
    assert process.reaped is True
    assert process.waits == 4


def test_ensure_crash_sink_rearms_and_warns_on_failure(monkeypatch, caplog, tmp_path):
    from rapid_mlx import _signal_observability as so

    prior_fd = so._crash_fd
    prior_identity = so._crash_fd_identity
    calls = []
    fd = os.open(tmp_path / "crash.txt", os.O_WRONLY | os.O_CREAT, 0o600)
    so._crash_fd = fd
    installed_stat = os.fstat(fd)
    so._crash_fd_identity = (installed_stat.st_dev, installed_stat.st_ino)
    try:
        monkeypatch.setattr(so, "_enable_faulthandler", calls.append)
        assert so.ensure_crash_sink() is True
        assert calls == [fd]

        monkeypatch.setattr(
            so,
            "_enable_faulthandler",
            lambda _fd: (_ for _ in ()).throw(RuntimeError("rearm failed")),
        )
        assert so.ensure_crash_sink() is False
        assert "could not re-arm durable rapid-mlx crash file" in caplog.text
    finally:
        so._crash_fd = prior_fd
        so._crash_fd_identity = prior_identity
        os.close(fd)


@pytest.mark.parametrize("stderr", [None, object()])
def test_unavailable_stderr_object_still_uses_helper(monkeypatch, tmp_path, stderr):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(so.sys, "stderr", stderr)
    try:
        so.install_signal_observability(observed_signals=())
        assert so._crash_path is not None
        assert so._crash_pipe is not None
        assert so._crash_tee is not None
    finally:
        so._reset_for_tests()


def test_crash_file_fd_is_moved_above_standard_fds(monkeypatch):
    from rapid_mlx import _signal_observability as so

    fcntl = pytest.importorskip("fcntl")
    calls = []
    monkeypatch.setattr(fcntl, "fcntl", lambda *args: calls.append(args) or 9)
    monkeypatch.setattr(so.os, "close", lambda fd: calls.append(("close", fd)))

    assert so._move_fd_above_stdio(2) == 9
    assert calls == [(2, fcntl.F_DUPFD_CLOEXEC, 3), ("close", 2)]


def test_crash_fd_is_cloexec_and_tee_is_reaped(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        so.install_signal_observability(observed_signals=())
        fd = so._crash_fd
        tee = so._crash_tee
        assert fd is not None
        assert tee is not None
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import os\ntry: os.fstat({fd}); print('INHERITED')\n"
                "except OSError: print('CLOSED')",
            ],
            capture_output=True,
            text=True,
            close_fds=False,
            check=True,
        )
        so._cleanup_crash_file()

        assert check.stdout.strip() == "CLOSED"
        assert tee.poll() is not None
    finally:
        so._reset_for_tests()


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


def test_acknowledged_inode_scan_tolerates_candidate_and_source_races(
    monkeypatch, tmp_path
):
    from rapid_mlx import _signal_observability as so

    crash = tmp_path / "crash-20260924T000000Z-99999.txt"
    reported = tmp_path / "crash-20260924T000000Z-99999.reported.txt"
    crash.write_text("crash", encoding="utf-8")
    reported.write_text("reported", encoding="utf-8")
    real_lstat = Path.lstat

    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "lstat",
            lambda path: (
                (_ for _ in ()).throw(OSError("candidate disappeared"))
                if path == reported
                else real_lstat(path)
            ),
        )
        assert so._has_acknowledged_inode(crash) is False

    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "lstat",
            lambda path: (
                (_ for _ in ()).throw(OSError("source disappeared"))
                if path == crash
                else real_lstat(path)
            ),
        )
        assert so._has_acknowledged_inode(crash) is False


def test_windows_crash_directory_rejects_identity_swap(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    real = os.lstat(tmp_path)
    stats = iter(
        [
            real,
            SimpleNamespace(
                st_mode=real.st_mode,
                st_dev=real.st_dev,
                st_ino=real.st_ino + 1,
            ),
        ]
    )
    monkeypatch.setattr(so.os, "name", "nt")
    monkeypatch.setattr(so.os, "chmod", lambda *_args: None)
    monkeypatch.setattr(so.os, "lstat", lambda _path: next(stats))

    assert so._prepare_crash_logs_dir(tmp_path) is False


def test_stderr_faulthandler_and_tee_reaper_defensive_paths(monkeypatch):
    from rapid_mlx import _signal_observability as so

    monkeypatch.setattr(so.sys, "stderr", None)
    so._enable_stderr_faulthandler()

    process = SimpleNamespace(
        wait=lambda: (_ for _ in ()).throw(OSError("wait failed"))
    )
    so._reap_crash_tee(process)


def test_crash_install_rejects_nonregular_sink(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(so, "_crash_logs_dir", lambda: tmp_path)
    monkeypatch.setattr(so, "_prepare_crash_logs_dir", lambda _path: True)
    monkeypatch.setattr(so, "_report_previous_crash", lambda _path: None)
    monkeypatch.setattr(so, "_enable_stderr_faulthandler", lambda: None)
    real_fstat = so.os.fstat
    monkeypatch.setattr(
        so.os,
        "fstat",
        lambda fd: SimpleNamespace(st_mode=stat.S_IFIFO, st_dev=1, st_ino=2),
    )

    so._install_crash_file()

    monkeypatch.setattr(so.os, "fstat", real_fstat)
    so._reset_for_tests()


def test_crash_sink_rearm_rejects_replaced_path(monkeypatch, tmp_path):
    from rapid_mlx import _signal_observability as so

    so._reset_for_tests()
    path = tmp_path / "crash.txt"
    path.write_text("diagnostic", encoding="utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    current = os.fstat(fd)
    so._crash_fd = fd
    so._crash_fd_identity = (current.st_dev, current.st_ino)
    so._crash_file_identity = (current.st_dev, current.st_ino + 1)
    so._crash_path = path
    so._crash_tee = SimpleNamespace(poll=lambda: 1)

    try:
        assert so._ensure_crash_sink_locked() is False
    finally:
        so._crash_tee = None
        so._reset_for_tests()


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
    monkeypatch.setattr(so.os, "close", real_close)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    real_unlink = Path.unlink

    def unlink_then_raise(target, *args, **kwargs):
        real_unlink(target, *args, **kwargs)
        raise OSError("unlink reported failure")

    monkeypatch.setattr(Path, "unlink", unlink_then_raise)

    def fail_enable(_target_fd):
        monkeypatch.setattr(so.os, "close", close_then_raise)
        raise RuntimeError("enable failed")

    monkeypatch.setattr(so, "_enable_faulthandler", fail_enable)
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


def test_helper_writes_crash_file_when_fd2_is_closed_at_spawn(tmp_path):
    home = tmp_path / "home"
    env = dict(os.environ, HOME=str(home))
    program = textwrap.dedent(
        """
        import os
        from rapid_mlx import _signal_observability as so

        os.close(2)
        so.install_signal_observability(observed_signals=())
        os.write(so._crash_fd, b"fatal traceback with closed fd 2\\n")
        so._cleanup_crash_file()
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
    assert proc.returncode == 0
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == "fatal traceback with closed fd 2\n"


def test_closed_parent_stderr_pipe_still_writes_crash_file(tmp_path):
    home = tmp_path / "home"
    program = textwrap.dedent(
        """
        import os
        import sys
        from rapid_mlx._signal_observability import install_signal_observability

        install_signal_observability(observed_signals=())
        print("READY", flush=True)
        sys.stdin.buffer.read(1)
        os.abort()
        """
    )
    env = dict(os.environ, HOME=str(home))
    child = subprocess.Popen(
        [sys.executable, "-c", program],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert _read_ready_with_timeout(child) == "READY\n"
    assert child.stderr is not None
    child.stderr.close()
    assert child.stdin is not None
    child.stdin.write("x")
    child.stdin.close()
    child.wait(timeout=10)

    files = list((home / ".rapid-mlx" / "logs").glob("crash-*.txt"))
    assert child.returncode != 0
    assert len(files) == 1
    _wait_for_nonempty_file(files[0])
    assert "Fatal Python error" in files[0].read_text(encoding="utf-8")


def test_r1_later_faulthandler_registration_without_reinstall(tmp_path):
    home = tmp_path / "home"
    alternate = tmp_path / "alternate.txt"
    env = dict(os.environ, HOME=str(home))
    program = textwrap.dedent(
        f"""
        import faulthandler
        import os
        from rapid_mlx._signal_observability import install_signal_observability
        from rapid_mlx.telemetry.server_start import ready

        install_signal_observability(observed_signals=())
        with open({os.fspath(alternate)!r}, "w", encoding="utf-8") as alternate:
            faulthandler.enable(file=alternate, all_threads=True)
            ready()
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
