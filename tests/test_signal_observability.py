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
import signal
import subprocess
import sys
import textwrap
import threading


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
            signal.signal(signal.SIGUSR1, prior_usr1)
    finally:
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
        signal.signal(signal.SIGUSR1, prior_usr1)
        so._reset_for_tests()


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
        ready_line = proc.stdout.readline()
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


def test_subprocess_sighup_emits_warning_and_stack_dump():
    """SIGHUP is the canonical "parent shell hung up" signal — the
    Marcus/Karim persona logs in the C-04 recon all ended right after a
    request returned, which matches an interactive shell tab being
    closed (SIGHUP delivered to the process group). Verify the same
    observability shape lands for SIGHUP too."""
    program = textwrap.dedent(
        """
        import logging, os, signal, sys, time
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                            format="%(levelname)s %(name)s: %(message)s")
        def _exit_handler(signum, frame):
            sys.stderr.flush()
            os._exit(0)
        signal.signal(signal.SIGHUP, _exit_handler)
        from vllm_mlx._signal_observability import install_signal_observability
        install_signal_observability()
        sys.stdout.write("READY\\n"); sys.stdout.flush()
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
        ready_line = proc.stdout.readline()
        assert ready_line.strip() == "READY", ready_line
        proc.send_signal(signal.SIGHUP)
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert "received signal SIGHUP" in stderr, stderr
    assert "Thread" in stderr or "Current thread" in stderr, stderr
