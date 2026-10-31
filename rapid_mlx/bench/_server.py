# SPDX-License-Identifier: Apache-2.0
"""Subprocess lifecycle helper for ``rapid-mlx serve`` during bench tier runs.

Previously lived at ``rapid_mlx/doctor/server.py``; PR #622 slimmed the
doctor module to pure env-health and deleted the file, but
``rapid_mlx/bench/tier_runner.py`` still imported it — so every actual
``rapid-mlx bench --tier ...`` invocation (which is exactly the path
that took over the model-validation work) crashed with
``ModuleNotFoundError: No module named 'rapid_mlx.doctor.server'``.

PR #5 (the --tier/--submit unification) moves the helper to the bench
package, which is the only consumer left in-tree. The implementation
is unchanged byte-for-byte from the original except:

- yields ``boot_time_ms`` in the info dict so the tier-smoke probe can
  populate ``smoke_result.boot_time_ms`` for the community-bench
  schema (PR #3 added that field; PR #5 wires the producer).
- ``REPO_ROOT`` / ``python_executable`` are imported from the same
  ``rapid_mlx.doctor.runner`` module that PR #622 kept around — no new
  duplication; if doctor.runner ever moves, both this module and the
  rest of the bench/ stack track it from one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from ..doctor.runner import REPO_ROOT, python_executable

_BOOT_LOG_TAIL_LINES = 30


class ServerStartFailed(RuntimeError):  # noqa: N818 — domain-specific error name
    """Server did not become healthy within the timeout."""


class ServerExitedError(RuntimeError):
    """The local server exited during a benchmark request."""

    def __init__(
        self,
        message: str,
        returncode: int | None,
        signal_name: str | None,
        log_tail: str,
    ):
        super().__init__(message)
        self.returncode = returncode
        self.signal_name = signal_name
        self.log_tail = log_tail


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
    extra_env: Mapping[str, str] | None = None,
    isolate_process_group: bool = True,
    echo_log: TextIO | None = None,
):
    """Boot ``rapid-mlx serve <model>`` and yield the live base URL.

    Yields a dict with:

    - ``base_url`` — ``http://127.0.0.1:<port>/v1``
    - ``health_url`` — ``http://127.0.0.1:<port>/health``
    - ``port`` — the resolved port (matches the caller's request when set)
    - ``boot_time_ms`` — wall-clock from process spawn to the first 200
      from ``/health``. Populated for the community-bench smoke probe
      (schema v2 ``smoke_result.boot_time_ms``).

    ``isolate_process_group=False`` is reserved for a supervisor that already
    launched the benchmark CLI as a dedicated process-group leader. In that
    topology the server must inherit the group so supervisor cancellation owns
    the complete tree; ordinary CLI callers retain the isolated default.

    On exit, SIGTERM the whole process group; escalate to SIGKILL after
    2s on cancellation or 10s otherwise. ``log_path`` receives the server's combined
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
        "rapid_mlx.cli",
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
        if extra_env:
            spawn_env.update(extra_env)
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
            preexec_fn=(
                os.setsid if isolate_process_group and sys.platform != "win32" else None
            ),
        )
    except BaseException:
        log_fh.close()
        if tmp_log is not None:
            tmp_log.unlink(missing_ok=True)
        raise

    stop_echo = threading.Event()
    echo_thread = None
    try:
        if echo_log is not None:

            def echo_output() -> None:
                with open(log_path or tmp_log, errors="replace") as reader:
                    while not stop_echo.is_set():
                        position = reader.tell()
                        line = reader.readline()
                        if line.endswith("\n"):
                            echo_log.write(line)
                            echo_log.flush()
                        else:
                            reader.seek(position)
                            stop_echo.wait(0.2)
                    remaining = reader.read()
                    if remaining:
                        echo_log.write(remaining)
                        echo_log.flush()

            echo_thread = threading.Thread(target=echo_output, daemon=True)
            echo_thread.start()

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

        def exit_error(wait_s: float = 5.0) -> ServerExitedError | None:
            log_fh.flush()
            return describe_server_exit(proc, log_path or tmp_log, wait_s=wait_s)

        yield {
            "base_url": v1_url,
            "health_url": health_url,
            "port": port,
            "boot_time_ms": boot_time_ms,
            "exit_error": exit_error,
        }
    finally:
        from ..community_bench.local_runner import BenchmarkCancelledError

        cancelled = isinstance(
            sys.exc_info()[1],
            (
                KeyboardInterrupt,
                SystemExit,
                BenchmarkCancelledError,
                asyncio.CancelledError,
            ),
        )
        try:
            _terminate(
                proc,
                isolated_process_group=isolate_process_group,
                grace_s=2 if cancelled else 10,
            )
        finally:
            try:
                if echo_thread is not None:
                    stop_echo.set()
                    echo_thread.join()
            finally:
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


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def describe_server_exit(
    proc: subprocess.Popen, log_path: Path | None, *, wait_s: float
) -> ServerExitedError | None:
    """Describe an exited server, waiting briefly for a concurrent exit."""
    deadline = time.monotonic() + wait_s
    while (rc := proc.poll()) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.05, remaining))
    signal_name = None
    if rc < 0:
        signal_name = _signal_name(-rc)
        message = (
            f"The local model server stopped unexpectedly (killed by {signal_name})."
        )
        if signal_name == "SIGKILL":
            message += (
                " On macOS this usually means the system was short on memory and"
                " terminated the largest process. Close other memory-heavy apps and"
                " retry, or run this benchmark on a Mac with more unified memory."
            )
    else:
        message = f"The local model server stopped unexpectedly (exit code {rc})."
    tail = _read_log_tail(log_path, _BOOT_LOG_TAIL_LINES)
    if tail:
        message += f"\n--- server log tail ---\n{tail}"
    else:
        message += " The server wrote no log output before it stopped."
    return ServerExitedError(message, rc, signal_name, tail)


def _wait_for_health(health_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    """Poll /health until 200 or timeout. Abort early if the process dies.

    Uses ``time.monotonic()`` for the deadline so a system clock
    adjustment mid-boot (NTP step, manual ``date`` change) can't
    shorten or extend the health wait incorrectly (Codex PR #623
    raised in review). ``time.time()`` would happily measure negative
    elapsed time if the clock stepped backwards.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if (rc := proc.poll()) is not None:
            if rc < 0:
                name = _signal_name(-rc)
                label = f"{name} (signal {-rc})" if name != f"signal {-rc}" else name
                message = f"The local model server was killed by {label} while loading the model."
                if name == "SIGKILL":
                    message += (
                        " On macOS this usually means the Mac ran out of memory."
                        " Try a smaller model (for example a lower-bit quantization),"
                        " or close other memory-heavy apps and retry."
                    )
            else:
                message = f"The local model server exited with code {rc} while loading the model."
            raise ServerStartFailed(message)
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.5)
    raise ServerStartFailed(
        f"server did not respond at {health_url} within {timeout_s}s"
    )


def _terminate(
    proc: subprocess.Popen, *, isolated_process_group: bool = True, grace_s: float = 10
) -> None:
    """Best-effort clean teardown of a server process group."""
    isolated = isolated_process_group and sys.platform != "win32"
    if (isolated and proc.returncode is not None) or (
        not isolated and proc.poll() is not None
    ):
        return
    # serve() starts an isolated server with setsid(), making its PID the PGID.
    pgid: int = proc.pid
    interrupted = False
    escalate = False
    try:
        if isolated:
            deadline = time.monotonic() + grace_s
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=max(0, deadline - time.monotonic()) if isolated else grace_s)
        if isolated:
            while True:
                try:
                    os.killpg(pgid, 0)
                except (ProcessLookupError, PermissionError):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    escalate = True
                    break
                time.sleep(min(0.05, remaining))
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        escalate = True
    except (ProcessLookupError, PermissionError):
        pass
    if escalate:
        for _ in range(3):
            try:
                if isolated:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                break
            except KeyboardInterrupt:
                interrupted = True
            except (ProcessLookupError, PermissionError):
                break
    while proc.returncode is None:
        try:
            proc.wait(timeout=5)
            break
        except KeyboardInterrupt:
            interrupted = True
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            break
    if interrupted:
        raise KeyboardInterrupt


__all__ = ["serve", "find_free_port", "ServerStartFailed", "ServerExitedError"]
