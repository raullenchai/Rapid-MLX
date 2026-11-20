# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP over the private Unix socket; no inference dependencies."""

import contextlib
import http.client
import json
import os
import socket
from pathlib import Path

from . import RUNNER_ENV, SECRET_ENV, SECRET_HEADER, SOCKET_ENV

_startup_terminal = False

MAX_BODY = 16384
TIMEOUT = 2.0


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=TIMEOUT)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect(self.path)
        except BaseException:
            self.sock.close()
            raise


def request(path, secret, method="GET", payload=None):
    body = json.dumps(payload).encode() if payload is not None else b""
    if len(body) > MAX_BODY:
        raise ValueError("control request too large")
    with contextlib.closing(UnixConnection(path)) as conn:
        conn.request(
            method,
            "/rapid/v1/status" if method == "GET" else "/startup",
            body=body,
            headers={SECRET_HEADER: secret, "Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = response.read(MAX_BODY + 1)
        if len(data) > MAX_BODY or response.status != 200:
            raise OSError("invalid control response")
        return json.loads(data)


def report(event, **fields):
    """Best effort observer; never change the legacy runner's failure behavior."""
    global _startup_terminal
    if os.environ.get(RUNNER_ENV) != "1":
        return
    if event in ("bind_failed", "startup_failed") or (
        event == "bound" and fields.get("ready", True)
    ):
        _startup_terminal = True
    try:
        request(
            os.environ[SOCKET_ENV],
            os.environ[SECRET_ENV],
            "POST",
            {"event": event, "pid": os.getpid(), **fields},
        )
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        pass


def runtime_root():
    # tempfile.gettempdir() caches env; use TMPDIR directly for discoverability.
    return Path(os.environ.get("TMPDIR", "/tmp")) / f"rmlx-cp-{os.getuid()}"


def discover(port=None):
    """Return live instances only. Stale files never imply a running service."""
    results = []
    for info in sorted(runtime_root().glob("*/instance.json")):
        try:
            metadata = json.loads(info.read_text())
            status = request(info.parent / "control.sock", metadata["secret"])
            if port is None or status["port"] == port:
                results.append(status)
        except (OSError, ValueError, KeyError, http.client.HTTPException):
            continue
    return results


def status_command(args):
    import sys

    statuses = discover(getattr(args, "port", None))
    if not statuses:
        print("No supervised Rapid-MLX instance found.", file=sys.stderr)
        return 1
    print(json.dumps(statuses[0] if len(statuses) == 1 else statuses, indent=2))
    return 0


def install_startup_report():
    """Cover failures before Uvicorn (including downloads/argument validation)."""
    import atexit

    def exited():
        report_startup_failure("runner exited before readiness")

    atexit.register(exited)


def report_startup_failure(reason):
    if not _startup_terminal:
        report("startup_failed", reason=reason[:512])
