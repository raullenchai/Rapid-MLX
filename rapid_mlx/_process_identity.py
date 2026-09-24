# SPDX-License-Identifier: Apache-2.0
"""Cross-platform process identity checks for local crash state."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard

try:
    import psutil as _psutil_module
except ImportError:
    _psutil_module = None
psutil: Any = _psutil_module

_MARKER_KEYS = frozenset({"pid", "create_time", "boot_time", "app_version"})
_MAX_APP_VERSION_LENGTH = 256


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    create_time: float
    boot_time: float


def _valid_pid(value: object) -> TypeGuard[int]:
    return type(value) is int and value > 0


def _valid_time(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def marker_identity(marker: object) -> ProcessIdentity | None:
    """Validate the complete on-disk marker payload."""
    if not isinstance(marker, Mapping):
        return None
    if set(marker) != _MARKER_KEYS:
        return None
    pid = marker.get("pid")
    create_time = marker.get("create_time")
    boot_time = marker.get("boot_time")
    app_version = marker.get("app_version")
    if (
        not _valid_pid(pid)
        or not _valid_time(create_time)
        or not _valid_time(boot_time)
        or type(app_version) is not str
        or not app_version
        or len(app_version) > _MAX_APP_VERSION_LENGTH
    ):
        return None
    return ProcessIdentity(
        pid=pid,
        create_time=float(create_time),
        boot_time=float(boot_time),
    )


def process_identity(pid: int) -> ProcessIdentity | None:
    """Return the identity of an existing process, or ``None`` when dead."""
    if not _valid_pid(pid):
        return None
    if psutil is not None:
        if not psutil.pid_exists(pid):
            return None
        try:
            create_time = psutil.Process(pid).create_time()
            boot_time = psutil.boot_time()
        except (psutil.NoSuchProcess, ProcessLookupError):
            return None
        return ProcessIdentity(pid, float(create_time), float(boot_time))

    # CPython implements os.kill(pid, 0) with TerminateProcess on Windows.
    # Never probe that way there; absence of psutil must fail closed as alive.
    if sys.platform == "win32":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return ProcessIdentity(pid, 0.0, 0.0)
    except OSError:
        return ProcessIdentity(pid, 0.0, 0.0)
    return ProcessIdentity(pid, 0.0, 0.0)


def is_same_process(marker: object) -> bool:
    """Return whether a validated marker still names the same live process."""
    expected = marker_identity(marker)
    if expected is None:
        return False
    if psutil is None:
        if sys.platform == "win32":
            return True
        return process_identity(expected.pid) is not None
    try:
        current = process_identity(expected.pid)
    except (psutil.AccessDenied, OSError):
        return True
    if current is None:
        return False
    return math.isclose(
        current.create_time, expected.create_time, rel_tol=0.0, abs_tol=0.01
    ) and math.isclose(current.boot_time, expected.boot_time, rel_tol=0.0, abs_tol=0.01)
