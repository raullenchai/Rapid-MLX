# SPDX-License-Identifier: Apache-2.0
"""Exclusive model-free profile owner and authenticated local discovery.

The internal module entry point deliberately bypasses inference CLI startup.
No PID from a discovery file is ever used to signal a process.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import re
import secrets
import socket
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

from rapid_mlx.headless_service.config import atomic_write

from . import SCHEMA_VERSION

FORMAT_VERSION = 1
WIRE_VERSION = 1
CAPABILITIES = {
    "streaming": True,
    "replay": True,
    "branches": True,
    "attachments": [
        "text/plain",
        "text/html",
        "text/csv",
        "text/tab-separated-values",
        "application/json",
        "application/yaml",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ],
    "side_effect_tools": os.environ.get("RAPID_MLX_CHAT_ACTION_TOOLS", "1").lower()
    not in {"0", "false", "off"},
    "approvals": True,
    "subagents": {"max_concurrent": 3, "max_depth": 1, "hidden_transcripts": True},
    "reasoning_persisted": False,
    "prompt_surface": True,
}


def profile_directory(profile: str = "default") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", profile):
        raise ValueError("Profile must contain 1–64 letters, digits, dots, _ or -")
    root = Path(
        os.environ.get(
            "RAPID_MLX_CHAT_HOME",
            str(Path.home() / "Library/Application Support/Rapid/chat"),
        )
    ).expanduser()
    return root / profile


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("Chat profile directory must be owned by the current user")
    if info.st_mode & 0o077:
        raise RuntimeError("Chat profile directory must have permissions 0700")


def _open_private(path: Path, *, create: bool = False) -> int:
    flags = os.O_RDWR if create else os.O_RDONLY
    if create:
        flags |= os.O_CREAT
    flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        os.close(fd)
        raise RuntimeError(
            "Chat discovery and credentials must be private regular files"
        )
    return fd


def _read_private(path: Path) -> str:
    with os.fdopen(_open_private(path), "r") as file:
        return file.read(65536)


def _endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or not parsed.port
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Invalid chat owner endpoint")
    return value


def discover(profile: str = "default") -> dict | None:
    """Return a verified owner and its in-memory credential, or no live owner."""
    run = profile_directory(profile) / "run"
    if not run.exists():
        return None
    _private_directory(run)
    try:
        discovery = json.loads(_read_private(run / "discovery.json"))
        if discovery.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("Chat owner discovery requires an upgrade")
        if (
            discovery.get("wire_version") != WIRE_VERSION
            or discovery.get("schema_version") != SCHEMA_VERSION
        ):
            raise RuntimeError("Chat owner protocol requires an upgrade")
        epoch = str(uuid.UUID(discovery["owner_epoch"]))
        credential = discovery["credential_file"]
        if credential != f"bearer-{epoch}":
            raise RuntimeError("Invalid chat owner credential reference")
        endpoint = _endpoint(discovery["endpoint"])
        token = _read_private(run / credential).strip()
        if not token:
            return None
        # Process identity is an additional stale-discovery check, never kill authority.
        process = psutil.Process(discovery["pid"])
        if abs(process.create_time() - discovery["process_start_time"]) > 0.01:
            return None
        with httpx.Client(
            timeout=0.75, trust_env=False, follow_redirects=False
        ) as client:
            response = client.get(
                endpoint + "/v1/rapid/chat/capabilities",
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code != 200:
            return None
        handshake = response.json()
        if (
            handshake.get("owner_epoch") != epoch
            or handshake.get("store_uuid") != discovery["store_uuid"]
        ):
            return None
        if handshake.get("wire_version") != WIRE_VERSION:
            raise RuntimeError("Chat owner protocol requires an upgrade")
        return {**discovery, "token": token}
    except (
        FileNotFoundError,
        ProcessLookupError,
        psutil.NoSuchProcess,
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
    ):
        return None


def discover_or_start(profile: str = "default", *, timeout: float = 20.0) -> dict:
    """Serialize launch; a held lifetime lock always prevents replacement."""
    directory = profile_directory(profile)
    _private_directory(directory)
    run = directory / "run"
    _private_directory(run)
    deadline = time.monotonic() + timeout
    with os.fdopen(_open_private(run / "startup.lock", create=True), "w") as startup:
        while True:
            try:
                fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for chat owner startup")
                time.sleep(0.05)
        existing = discover(profile)
        if existing:
            return existing
        lifetime = _open_private(run / "owner.lock", create=True)
        launched = None
        try:
            try:
                fcntl.flock(lifetime, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass  # Existing owner may still be starting; never kill or steal it.
            else:
                fcntl.flock(lifetime, fcntl.LOCK_UN)
                launched = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "rapid_mlx.chat_runtime.owner",
                        "--profile",
                        profile,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            while time.monotonic() < deadline:
                existing = discover(profile)
                if existing:
                    return existing
                if launched and launched.poll() is not None:
                    raise RuntimeError(
                        "Chat owner could not start. Check profile permissions and schema compatibility; "
                        "run python -m rapid_mlx.chat_runtime.owner --profile "
                        + profile
                    )
                time.sleep(0.05)
            raise RuntimeError(
                "Chat owner is unavailable; its lifetime lock was not overridden"
            )
        finally:
            os.close(lifetime)


class ClientLeases:
    """Renewable leases; expired/crashed clients do not keep the owner forever."""

    def __init__(self, *, ttl: float = 45.0):
        self.ttl = ttl
        self.clients: dict[str, float] = {}
        self.idle_since = time.monotonic()

    def attach(self, client_id: str) -> None:
        self.clients[client_id] = time.monotonic() + self.ttl
        self.idle_since = time.monotonic()

    def detach(self, client_id: str) -> None:
        self.clients.pop(client_id, None)
        if not self.clients:
            self.idle_since = time.monotonic()

    def idle(self, seconds: float, active: bool) -> bool:
        now = time.monotonic()
        expired = [key for key, deadline in self.clients.items() if deadline <= now]
        for key in expired:
            self.clients.pop(key)
        if expired and not self.clients:
            self.idle_since = now
        if active:
            self.idle_since = now
        return not self.clients and not active and now - self.idle_since >= seconds


async def run_owner(profile: str) -> None:
    import uvicorn

    from rapid_mlx import __version__
    from rapid_mlx.chat_runtime.api import create_app
    from rapid_mlx.chat_runtime.runtime import Runtime
    from rapid_mlx.chat_runtime.store import Store

    directory = profile_directory(profile)
    _private_directory(directory)
    run = directory / "run"
    _private_directory(run)
    lifetime = _open_private(run / "owner.lock", create=True)
    try:
        try:
            fcntl.flock(lifetime, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        store = await Store.open(directory)
        runtime = Runtime(store)
        leases = ClientLeases()
        epoch = str(uuid.uuid4())
        token = secrets.token_urlsafe(48)
        credential = run / f"bearer-{epoch}"
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        metadata = {
            "format_version": FORMAT_VERSION,
            "store_uuid": store.store_uuid,
            "owner_epoch": epoch,
            "pid": os.getpid(),
            "process_start_time": psutil.Process().create_time(),
            "endpoint": f"http://127.0.0.1:{listener.getsockname()[1]}",
            "build_version": __version__,
            "wire_version": WIRE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "capabilities": CAPABILITIES,
            "credential_file": credential.name,
        }
        app = create_app(store, runtime, token, metadata=metadata, leases=leases)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_level="critical",
                access_log=False,
                timeout_graceful_shutdown=2,
            )
        )
        idle_seconds = float(os.environ.get("RAPID_MLX_CHAT_IDLE_SECONDS", "300"))

        async def watch_idle():
            while not server.should_exit:
                await asyncio.sleep(min(1.0, max(0.05, idle_seconds / 4)))
                if leases.idle(idle_seconds, bool(runtime.active_turns)):
                    server.should_exit = True

        watcher = None
        try:
            await store.recover_interrupted()
            atomic_write(credential, token.encode())
            atomic_write(run / "discovery.json", json.dumps(metadata).encode())
            watcher = asyncio.create_task(watch_idle())
            await server.serve(sockets=[listener])
        finally:
            if watcher:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            await runtime.close()
            await store.close()
            listener.close()
            credential.unlink(missing_ok=True)
            # Still under the lifetime lock; no new owner's discovery can exist.
            (run / "discovery.json").unlink(missing_ok=True)
    finally:
        os.close(lifetime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal model-free chat owner")
    parser.add_argument("--profile", default="default")
    args = parser.parse_args()
    try:
        asyncio.run(run_owner(args.profile))
    except (RuntimeError, ValueError, OSError) as error:
        # Fixed operational errors only; never dump model prompts or credentials.
        print(f"Chat owner unavailable: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
