# SPDX-License-Identifier: Apache-2.0
"""Subprocess lifecycle helper for ``rapid-mlx serve`` during bench tier runs.

Previously lived at ``vllm_mlx/doctor/server.py``; PR #622 slimmed the
doctor module to pure env-health and deleted the file, but
``vllm_mlx/bench/tier_runner.py`` still imported it — so every actual
``rapid-mlx bench --tier ...`` invocation (which is exactly the path
that took over the model-validation work) crashed with
``ModuleNotFoundError: No module named 'vllm_mlx.doctor.server'``.

PR #5 (the --tier/--submit unification) moves the helper to the bench
package, which is the only consumer left in-tree. The implementation
is unchanged byte-for-byte from the original except:

- yields ``boot_time_ms`` in the info dict so the tier-smoke probe can
  populate ``smoke_result.boot_time_ms`` for the community-bench
  schema (PR #3 added that field; PR #5 wires the producer).
- ``REPO_ROOT`` / ``python_executable`` are imported from the same
  ``vllm_mlx.doctor.runner`` module that PR #622 kept around — no new
  duplication; if doctor.runner ever moves, both this module and the
  rest of the bench/ stack track it from one place.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..doctor.runner import REPO_ROOT, python_executable

_BOOT_LOG_TAIL_LINES = 30


class ServerStartFailed(RuntimeError):  # noqa: N818 — domain-specific error name
    """Server did not become healthy within the timeout."""


def find_free_port() -> int:
    """Pick a free TCP port on localhost.

    OS-assigned ephemeral; tiny TOCTOU race between close and reuse
    that doesn't matter for a single-host harness.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(
    model: str,
    port: int | None = None,
    log_path: Path | None = None,
    extra_args: list[str] | None = None,
    boot_timeout_s: int = 180,
    model_path: str | Path | None = None,
):
    """Boot ``rapid-mlx serve <model>`` and yield the live base URL.

    Yields a dict with:

    - ``base_url`` — ``http://127.0.0.1:<port>/v1``
    - ``health_url`` — ``http://127.0.0.1:<port>/health``
    - ``port`` — the resolved port (matches the caller's request when set)
    - ``boot_time_ms`` — wall-clock from process spawn to the first 200
      from ``/health``. Populated for the community-bench smoke probe
      (schema v2 ``smoke_result.boot_time_ms``).

    On exit, SIGTERM the whole process group; escalate to SIGKILL after
    10s. ``log_path`` if provided receives the server's combined
    stdout/stderr.
    """
    if port is None:
        port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    health_url = f"{base_url}/health"
    v1_url = f"{base_url}/v1"

    serve_target = str(model_path) if model_path else model
    cmd = [
        python_executable(),
        "-m",
        "vllm_mlx.cli",
        "serve",
        serve_target,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Open the log file BEFORE the try/finally so the Popen call has
    # an fd to point ``stdout=`` at — but guard with a narrow
    # try/except so an exception INSIDE ``Popen`` (path error, OS
    # resource exhaustion, etc.) doesn't leak the file handle.
    # The outer try/finally below then owns the fh for the rest of
    # the lifetime; this guard only covers the spawn window where
    # the outer try hasn't started yet. (Codex PR #623 BLOCKING-2.)
    #
    # If the caller didn't provide a log_path, route output to a temp
    # file anyway so that on a boot crash we can replay the tail in
    # the ``ServerStartFailed`` exception. Older code piped both
    # streams to DEVNULL, which left users staring at
    # ``server exited with code 3`` with nothing to act on (the
    # gemma-4 / gemma3-multimodal aliases hit this — the real error
    # was ``mlx-vlm not installed`` but bench surfaced nothing).
    tmp_log: Path | None = None
    if log_path is not None:
        log_fh = open(log_path, "w")
    else:
        tmp_log = Path(
            tempfile.mkstemp(prefix=f"rapid-mlx-bench-{port}-", suffix=".log")[1]
        )
        log_fh = open(tmp_log, "w")
    try:
        t_spawn = time.perf_counter()
        # Parent-PID watchdog (rapid-desktop #449 sibling fix). A
        # SIGKILL of the bench process before ``_terminate(proc)`` runs
        # would otherwise leave the spawned serve as an orphan holding
        # the model in RAM. The watchdog inside the child polls
        # ``os.getppid()`` and exits when it stops matching this PID.
        # Direct assignment (NOT setdefault) so an inherited / stale
        # env value from a grandparent supervisor cannot mis-target
        # the watchdog at the wrong PID.
        spawn_env = os.environ.copy()
        spawn_env["RAPID_MLX_WATCHDOG_PPID"] = str(os.getpid())
        proc = subprocess.Popen(  # noqa: S603 — args constructed by us
            cmd,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=spawn_env,
            # New process group so we can signal the whole tree on
            # teardown (uvicorn + worker children). POSIX-only; we
            # don't run on win.
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
    except BaseException:
        log_fh.close()
        if tmp_log is not None:
            tmp_log.unlink(missing_ok=True)
        raise

    try:
        try:
            _wait_for_health(health_url, proc, boot_timeout_s)
        except ServerStartFailed as exc:
            log_fh.flush()
            tail = _read_log_tail(log_path or tmp_log, _BOOT_LOG_TAIL_LINES)
            if tail:
                raise ServerStartFailed(
                    f"{exc}\n--- server log tail ---\n{tail}"
                ) from exc
            raise
        boot_time_ms = (time.perf_counter() - t_spawn) * 1000.0
        yield {
            "base_url": v1_url,
            "health_url": health_url,
            "port": port,
            "boot_time_ms": boot_time_ms,
        }
    finally:
        _terminate(proc)
        log_fh.close()
        if tmp_log is not None:
            tmp_log.unlink(missing_ok=True)


def _read_log_tail(log_path: Path | None, max_lines: int) -> str:
    """Return the last ``max_lines`` lines of ``log_path``, best-effort."""
    if log_path is None:
        return ""
    try:
        with open(log_path, errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    return "".join(lines[-max_lines:]).strip()


def _wait_for_health(health_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    """Poll /health until 200 or timeout. Abort early if the process dies.

    Uses ``time.monotonic()`` for the deadline so a system clock
    adjustment mid-boot (NTP step, manual ``date`` change) can't
    shorten or extend the health wait incorrectly (Codex PR #623
    review NIT-2). ``time.time()`` would happily measure negative
    elapsed time if the clock stepped backwards.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ServerStartFailed(
                f"server exited with code {proc.returncode} before becoming healthy"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.5)
    raise ServerStartFailed(
        f"server did not respond at {health_url} within {timeout_s}s"
    )


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort clean teardown of a server process group."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    except (ProcessLookupError, PermissionError):
        pass


__all__ = ["serve", "find_free_port", "ServerStartFailed"]
