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
import select
import signal
import subprocess
import sys
import textwrap
import threading


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
    return proc.stdout.readline()


def test_install_is_idempotent_and_saves_prior_handlers():
    """Repeated installs must not stack handlers (each install would
    otherwise add a layer that re-runs the dump on every signal)."""
    from vllm_mlx import _signal_observability as so

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
    from vllm_mlx import _signal_observability as so

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
    from vllm_mlx import _signal_observability as so

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
        from vllm_mlx._signal_observability import install_signal_observability
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
    from vllm_mlx import _signal_observability as so

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
    from vllm_mlx import _signal_observability as so

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
    from vllm_mlx import _signal_observability as so

    result_box: list[bool] = []

    def _worker():
        result_box.append(so.install_signal_observability())

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert result_box == [False]


def test_faulthandler_is_enabled_after_install():
    """``faulthandler.enable`` must fire so SIGSEGV from MLX produces a
    Python traceback rather than a silent core dump."""
    import faulthandler

    from vllm_mlx import _signal_observability as so

    so._reset_for_tests()
    try:
        # Disable first so we can prove the install enabled it.
        faulthandler.disable()
        assert not faulthandler.is_enabled()
        so.install_signal_observability(observed_signals=())
        assert faulthandler.is_enabled()
    finally:
        so._reset_for_tests()


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

        from vllm_mlx._signal_observability import install_signal_observability
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


def test_subprocess_sighup_default_disposition_emits_warning_and_terminates():
    """Codex r2 BLOCKING #3: SIGHUP's default disposition under uvicorn
    is ``SIG_DFL`` because uvicorn only installs handlers for
    SIGINT/SIGTERM (``HANDLED_SIGNALS``). This is the production
    SIGHUP path, and it has to work end-to-end: log the receipt + dump
    stacks + still terminate (so the operator's ``kill -SIGHUP <pid>``
    still works as expected).

    A previous round of this test installed a callable
    ``_exit_handler`` BEFORE calling ``install_signal_observability``,
    which bypassed the SIG_DFL chain entirely. This version
    deliberately does NOT install a prior so we exercise the real
    default-disposition path that production sees.
    """
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        # Confirm we start from SIG_DFL — this is the production
        # baseline for SIGHUP under uvicorn.
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
        from vllm_mlx._signal_observability import install_signal_observability
        assert install_signal_observability() is True
        sys.stdout.write("READY\\n"); sys.stdout.flush()
        # If the chain swallows the signal we'll fall through to
        # os._exit(99) and the test will detect that via returncode.
        for _ in range(50):
            time.sleep(0.1)
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
        proc.send_signal(signal.SIGHUP)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    # WARNING marker must land before termination.
    assert "received signal SIGHUP" in stderr, stderr
    # faulthandler.dump_traceback shape.
    assert "Thread" in stderr or "Current thread" in stderr, stderr
    # The chain MUST have re-raised under SIG_DFL → terminate, NOT
    # swallowed the signal. If we hit os._exit(99) below the signal,
    # the SIG_DFL re-raise branch in _on_signal regressed.
    assert proc.returncode != 99, (
        f"SIGHUP was swallowed — chain failed to re-raise under SIG_DFL"
        f" so the process kept running; stderr={stderr!r}"
    )
