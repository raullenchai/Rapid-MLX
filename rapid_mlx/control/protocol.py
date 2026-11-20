# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP over the private Unix socket; no inference dependencies."""

import contextlib
import http.client
import json
import os
import socket
import stat
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


def discovery_root():
    return Path.home() / ".rapid-mlx" / "run"


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise OSError("control discovery directory must be an owned 0700 directory")
    return path


def discover(port=None):
    """Resolve the stable private index, independent of the caller's TMPDIR."""
    results = []
    root = discovery_root()
    try:
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            return []
    except OSError:
        return []
    for entry in sorted(root.glob("*.json")):
        try:
            info = entry.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                continue
            metadata = json.loads(entry.read_text())
            path = Path(metadata["socket"])
            info = path.parent.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                continue
            status = request(path, metadata["secret"])
            if port is None or status["port"] == port:
                results.append(status)
        except (
            FileNotFoundError,
            ConnectionRefusedError,
            ValueError,
            KeyError,
            TypeError,
        ):
            with contextlib.suppress(OSError):
                entry.unlink()
        except (OSError, http.client.HTTPException):
            # A busy live supervisor may time out; keep its discovery entry.
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
