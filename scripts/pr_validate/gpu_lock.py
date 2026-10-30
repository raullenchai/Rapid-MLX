# SPDX-License-Identifier: Apache-2.0
"""Shared, legacy-compatible GPU exclusion for the benchmark matrix."""

import fcntl
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GPU_LOCK_PATH = "/tmp/harbor-prvalidate.lock"


def gpu_lock_path() -> Path:
    return Path(os.environ.get("PR_VALIDATE_GPU_LOCK") or DEFAULT_GPU_LOCK_PATH)


@dataclass
class LockInfo:
    path: Path
    mode: str = ""
    waited_seconds: float = 0.0
    held_seconds: float = 0.0

    def describe(self) -> str:
        if self.mode == "held-by-caller":
            return (
                "shared lock: already held by the calling process (legacy whole-run wrapping); "
                f"not re-acquired, ran {self.held_seconds:.1f}s"
            )
        return (
            f"shared lock: acquired after {self.waited_seconds:.1f}s wait, "
            f"held {self.held_seconds:.1f}s"
        )


def _holder_pids(path: Path) -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-t", "--", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return [int(line) for line in result.stdout.splitlines()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _ancestor_pids() -> set[int]:
    pid = os.getppid()
    ancestors: set[int] = set()
    for _ in range(64):
        if pid <= 1 or pid in ancestors:
            break
        ancestors.add(pid)
        try:
            result = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            pid = int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            break
    return ancestors


@contextmanager
def hold_gpu_lock(
    path: Path,
    *,
    log: Callable[[str], None],
    poll_seconds: float = 5.0,
    holder_pids: Callable[[Path], list[int]] | None = None,
    ancestor_pids: Callable[[], set[int]] | None = None,
) -> Iterator[LockInfo]:
    start = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    info = LockInfo(path)
    holder_pids = holder_pids or _holder_pids
    ancestor_pids = ancestor_pids or _ancestor_pids
    last_log = float("-inf")
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info.mode = "acquired"
                break
            except BlockingIOError:
                pids = holder_pids(path)
                if set(pids) & ancestor_pids():
                    log(
                        "shared GPU lock already held by the calling process; wrapping the validator in lockf is no longer needed"
                    )
                    os.close(fd)
                    fd = -1
                    info.mode = "held-by-caller"
                    break
                now = time.monotonic()
                if now - last_log >= 60:
                    log(
                        f"waiting for the shared GPU lock (held by pid(s) {', '.join(map(str, pids)) or 'unknown'})"
                    )
                    last_log = now
                time.sleep(poll_seconds)
        held_start = time.monotonic()
        info.waited_seconds = held_start - start
        try:
            yield info
        finally:
            info.held_seconds = time.monotonic() - held_start
    finally:
        if fd != -1:
            try:
                if info.mode == "acquired":
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
