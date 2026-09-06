# SPDX-License-Identifier: Apache-2.0
"""Bounded stdout/stderr capture for the long-lived service runtime."""

from __future__ import annotations

import os
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


class RotatingLog:
    """A small dependency-free size/age/count rotating binary sink."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int,
        backup_count: int,
        retention_days: int,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.retention_days = retention_days
        self._handle: BinaryIO | None = None
        self._lock = threading.Lock()
        path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o750)
        self._purge()

    def _open(self) -> BinaryIO:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags, 0o640)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise OSError(f"refusing non-regular log target: {self.path}")
        return os.fdopen(fd, "ab", buffering=0)

    def _backups(self) -> list[Path]:
        return sorted(
            self.path.parent.glob(f"{self.path.name}.*"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

    def _purge(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        max_age = self.retention_days * 86400
        for index, backup in enumerate(self._backups()):
            try:
                if index >= self.backup_count or now - backup.stat().st_mtime > max_age:
                    backup.unlink()
            except OSError:
                continue

    def _rotate(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.path.replace(self.path.with_name(f"{self.path.name}.{stamp}"))
        self._purge()

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if self._handle is None:
                self._handle = self._open()
            try:
                size = self._handle.tell()
            except OSError:
                size = self.path.stat().st_size if self.path.exists() else 0
            if size and size + len(data) > self.max_bytes:
                self._rotate()
                self._handle = self._open()
            self._handle.write(data)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
