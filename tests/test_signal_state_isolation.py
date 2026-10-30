# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for pytest's process signal baseline and isolation."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    _normalize_inherited_signal_dispositions,
    _restore_signal_state,
    _restore_stdlib_callables,
    _snapshot_signal_state,
    _snapshot_stdlib_callables,
)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"), reason="POSIX signals required"
)
def test_normalize_inherited_signals():
    snapshot = _snapshot_signal_state()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        _normalize_inherited_signal_dispositions()
        assert signal.getsignal(signal.SIGINT) == signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
        assert signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    finally:
        _restore_signal_state(snapshot)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"), reason="POSIX signals required"
)
def test_restore_signal_handler_and_mask():
    snapshot = _snapshot_signal_state()
    try:
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        _restore_signal_state(snapshot)
        assert signal.getsignal(signal.SIGTERM) == snapshot[signal.SIGTERM]
        assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == snapshot["mask"]
    finally:
        _restore_signal_state(snapshot)


def test_restore_stdlib_callables():
    snapshot = _snapshot_stdlib_callables()
    real_which = shutil.which
    try:
        shutil.which = lambda _: None
        _restore_stdlib_callables(snapshot)
        assert shutil.which is real_which
    finally:
        shutil.which = real_which


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck required")
def test_quicksilver_does_not_leak_which_to_shellcheck():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTEST_XDIST_") and key != "PYTEST_CURRENT_TEST"
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "tests/test_share_quicksilver.py::test_install_service_uses_running_install_when_path_is_older",
            "tests/test_tart_ci_runner_scripts.py::test_shellcheck",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 passed" in proc.stdout


def test_pytest_normalizes_inherited_ignored_sigint():
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTEST_XDIST_") and key != "PYTEST_CURRENT_TEST"
    }
    script = (
        "import os, signal, sys; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "os.execv(sys.executable, [sys.executable, '-m', 'pytest', '-q', "
        "'-p', 'no:cacheprovider', '-p', 'no:randomly', "
        "'tests/test_pull_cancel.py::test_pull_interrupt_process_exit'])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed" in proc.stdout
