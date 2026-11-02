# SPDX-License-Identifier: Apache-2.0
"""Shared registry for bounded child process groups."""

import os
import signal
import subprocess
import threading
import time

# Reentrant because a signal can interrupt a registry operation on the main thread.
_LOCK = threading.RLock()
_ACTIVE: dict[int, subprocess.Popen] = {}


def register(proc: subprocess.Popen) -> None:
    with _LOCK:
        _ACTIVE[proc.pid] = proc


def wait(proc: subprocess.Popen, timeout: float) -> int:
    """Reap and unregister atomically with respect to group termination."""
    deadline = time.monotonic() + timeout
    while True:
        with _LOCK:
            code = proc.poll()
            if code is not None:
                _ACTIVE.pop(proc.pid, None)
                return code
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        time.sleep(min(0.02, remaining))


def kill_groups() -> set[int]:
    """Kill registered groups before any leader is reaped; never block on wait."""
    with _LOCK:
        for proc in _ACTIVE.values():
            if proc.returncode is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
        return set(_ACTIVE)
