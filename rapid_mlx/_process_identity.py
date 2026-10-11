# SPDX-License-Identifier: Apache-2.0
"""Cross-platform process identity checks for local crash state."""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard

try:
    import psutil as _psutil_module
except ImportError:
    _psutil_module = None
psutil: Any = _psutil_module

_MARKER_KEYS = frozenset({"pid", "create_time", "boot_time", "app_version"})
_MARKER_OPTIONAL_KEYS = frozenset({"startup_terminal"})
_MAX_APP_VERSION_LENGTH = 256
_MAX_PID = 2**31 - 1
_MAX_TIME = float(2**40)

logger = logging.getLogger(__name__)
_CURRENT_PROCESS_CREATE_TIME = time.time()


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    create_time: float
    boot_time: float


def _valid_pid(value: object) -> TypeGuard[int]:
    return type(value) is int and 1 <= value <= _MAX_PID


def _valid_time(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        parsed = float(value)
    except (ValueError, OverflowError, TypeError):
        return False
    return math.isfinite(parsed) and 0.0 <= parsed <= _MAX_TIME


def marker_identity(marker: object) -> ProcessIdentity | None:
    """Validate the complete on-disk marker payload."""
    if not isinstance(marker, Mapping):
        return None
    marker_keys = set(marker)
    if (
        not marker_keys >= _MARKER_KEYS
        or not marker_keys - _MARKER_KEYS <= _MARKER_OPTIONAL_KEYS
    ):
        return None
    pid = marker.get("pid")
    create_time = marker.get("create_time")
    boot_time = marker.get("boot_time")
    app_version = marker.get("app_version")
    startup_terminal = marker.get("startup_terminal", False)
    if not _valid_pid(pid) or type(app_version) is not str:
        return None
    try:
        if (
            not _valid_time(create_time)
            or not _valid_time(boot_time)
            or not app_version
            or len(app_version) > _MAX_APP_VERSION_LENGTH
            or type(startup_terminal) is not bool
        ):
            return None
        return ProcessIdentity(
            pid=pid,
            create_time=float(create_time),
            boot_time=float(boot_time),
        )
    except (ValueError, OverflowError, TypeError):
        return None


def process_identity(pid: int) -> ProcessIdentity | None:
    """Return the identity of an existing process, or ``None`` when dead.

    Without psutil, Windows can identify only this process. Other valid PIDs
    remain unknown and liveness checks conservatively treat them as alive.
    """
    if not _valid_pid(pid):
        return None
    if psutil is not None:
        try:
            if not psutil.pid_exists(pid):
                return None
            create_time = psutil.Process(pid).create_time()
            boot_time = psutil.boot_time()
            return ProcessIdentity(pid, float(create_time), float(boot_time))
        except (psutil.NoSuchProcess, ProcessLookupError):
            return None
        except Exception as exc:
            logger.debug("could not probe process identity for pid %s: %r", pid, exc)
            return ProcessIdentity(pid, 0.0, 0.0)

    # CPython implements os.kill(pid, 0) with TerminateProcess on Windows.
    # Never probe that way there. Preserve enough identity for this process to
    # write a marker; checks for every other PID fail closed as alive below.
    if sys.platform == "win32":
        if pid == os.getpid():
            return ProcessIdentity(pid, _CURRENT_PROCESS_CREATE_TIME, 0.0)
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except Exception as exc:
        logger.debug("could not probe process identity for pid %s: %r", pid, exc)
        return ProcessIdentity(pid, 0.0, 0.0)
    return ProcessIdentity(pid, 0.0, 0.0)


def is_same_process(marker: object) -> bool:
    """Return whether a validated marker still names the same live process."""
    expected = marker_identity(marker)
    if expected is None:
        return False
    if psutil is None:
        if sys.platform == "win32":
            if expected.pid != os.getpid():
                return True
            return math.isclose(
                expected.create_time,
                _CURRENT_PROCESS_CREATE_TIME,
                rel_tol=0.0,
                abs_tol=0.01,
            )
        return process_identity(expected.pid) is not None
    try:
        current = process_identity(expected.pid)
    except Exception as exc:
        logger.debug(
            "could not compare process identity for pid %s: %r", expected.pid, exc
        )
        return True
    if current is None:
        return False
    if current.create_time == 0.0 and current.boot_time == 0.0:
        return True
    if not math.isclose(
        current.boot_time, expected.boot_time, rel_tol=0.0, abs_tol=0.01
    ):
        return False
    if math.isclose(
        current.create_time, expected.create_time, rel_tol=0.0, abs_tol=0.01
    ):
        return True
    if sys.platform != "darwin":
        return False
    # Recent psutil macOS backends adjust create_time against an import-time
    # clock baseline. A long-lived parent can therefore disagree with a fresh
    # child about the same kernel timestamp after clock synchronization. Keep
    # the existing marker/boot checks, then compare the unadjusted timestamp
    # when this backend exposes it. Older backends need no such correction.
    try:
        backend = getattr(psutil.Process(expected.pid), "_proc", None)
        raw_create_time = getattr(backend, "create_time", None)
        if not callable(raw_create_time):
            return False
        raw = raw_create_time(monotonic=True)
    except (AttributeError, TypeError):
        return False  # backend has no raw-time capability
    except (psutil.NoSuchProcess, ProcessLookupError):
        return False
    except (psutil.AccessDenied, PermissionError):
        # Retain diagnostics only while the observed process identity remains
        # stable. It may have exited or its PID may have been reused since the
        # first probe; a permission denial supplies no evidence of liveness.
        try:
            refreshed = process_identity(expected.pid)
            return (
                refreshed is not None
                and refreshed.pid == current.pid
                and math.isclose(
                    refreshed.create_time,
                    current.create_time,
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
                and math.isclose(
                    refreshed.boot_time,
                    current.boot_time,
                    rel_tol=0.0,
                    abs_tol=0.01,
                )
            )
        except Exception as exc:
            logger.debug(
                "could not revalidate denied process identity for pid %s: %r",
                expected.pid,
                exc,
            )
            return False
    except Exception as exc:
        logger.debug(
            "could not probe raw creation time for pid %s: %r", expected.pid, exc
        )
        return False  # an unexpected failure cannot override a known mismatch
    return _valid_time(raw) and math.isclose(
        float(raw), expected.create_time, rel_tol=0.0, abs_tol=0.01
    )
