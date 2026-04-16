# SPDX-License-Identifier: Apache-2.0
"""Subprocess lifecycle helper for ``rapid-mlx serve`` during doctor runs.

Used by check / full / benchmark tiers.  Boots a server on a non-default
port (so it doesn't collide with the user's own running server), polls
``/health`` until ready, and guarantees teardown via context manager.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .runner import REPO_ROOT, python_executable


class ServerStartFailed(RuntimeError):  # noqa: N818 — domain-specific error name
    """Server did not become healthy within the timeout."""


def find_free_port() -> int:
    """Pick a free TCP port on localhost.

    We bind ephemeral=0 to let the OS allocate, then close immediately.
    There is a tiny race between close and the server reusing the port,
    but it's small enough not to matter for a single-host harness.
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

    Args:
        model: alias or HF repo id used for logging/display.
        model_path: optional local path to use *instead* of resolving
            the alias.  When provided, the server loads from this exact
            location and never reaches Hugging Face.  This honors the
            benchmark tier's "no auto-download" contract for models
            installed in non-HF layouts (e.g. LM Studio's
            ~/.lmstudio/models/{org}/{repo}/).

    On exit, send SIGTERM and wait up to 10s; escalate to SIGKILL if the
    process is still alive.  ``log_path`` if provided receives the
    server's combined stdout/stderr — useful for post-mortem after a
    failed boot.
    """
    if port is None:
        port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    health_url = f"{base_url}/health"
    v1_url = f"{base_url}/v1"

    # serve_target is what we pass to the CLI: the explicit local path
    # takes precedence over the alias so the server never tries to
    # download a model that discovery confirmed is on disk.
    serve_target = str(model_path) if model_path else model
    cmd = [
        python_executable(),
        "-m", "vllm_mlx.cli", "serve", serve_target,
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_fh = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(  # noqa: S603 — args constructed by us
        cmd,
        cwd=REPO_ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        # New process group so we can signal the whole tree on teardown
        # (uvicorn + worker children).  POSIX-only; we don't run on win.
        preexec_fn=os.setsid if sys.platform != "win32" else None,
    )

    try:
        _wait_for_health(health_url, proc, boot_timeout_s)
        yield {"base_url": v1_url, "health_url": health_url, "port": port}
    finally:
        _terminate(proc)
        if log_fh is not subprocess.DEVNULL:
            log_fh.close()


def _wait_for_health(health_url: str, proc: subprocess.Popen, timeout_s: int) -> None:
    """Poll /health until 200 or timeout.  Abort early if the process dies."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ServerStartFailed(
                f"server exited with code {proc.returncode} before becoming healthy"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            # Server still booting — keep polling.
            time.sleep(0.5)
    raise ServerStartFailed(
        f"server did not respond at {health_url} within {timeout_s}s"
    )


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort clean teardown of a server process group."""
    if proc.poll() is not None:
        return
    try:
        # Signal the whole process group (children too).
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Escalate.
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    except (ProcessLookupError, PermissionError):
        # Process already gone or can't be signalled — nothing to do.
        pass
