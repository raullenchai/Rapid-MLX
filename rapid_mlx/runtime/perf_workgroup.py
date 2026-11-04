# SPDX-License-Identifier: Apache-2.0
"""Best-effort macOS performance-core scheduling for individual decode steps.

A single process-lifetime workgroup is created lazily. Membership is scoped to
each interval on its calling thread: every successful join must leave before
that thread can exit. Concurrent steps skip the hint rather than wait. No
frameworks are loaded at import time.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Scheduling hint, not a timeout: roughly one decode round, including GPU wait.
STEP_DEADLINE_MS = 50
_OS_CLOCK_MACH_ABSOLUTE_TIME = 32


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class StepWorkgroup:
    """Best-effort interval guard; the module singleton owns the shared group."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempted = False
        self._disabled = False
        self._wg: int | None = None
        self._deadline_ticks = 0
        self._system: ctypes.CDLL | None = None

    def _disable(self, error: object) -> None:
        if not self._disabled:
            logger.debug("Step workgroup unavailable: %s", error)
        self._disabled = True

    def _initialize(self) -> None:
        self._attempted = True
        if (
            sys.platform != "darwin"
            or os.environ.get("RAPID_MLX_STEP_WORKGROUP") == "0"
        ):
            self._disabled = True
            return
        try:
            audio = ctypes.CDLL(
                "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
            )
            system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            # Resolve every symbol before creating anything (older macOS: no-op).
            audio.AudioWorkIntervalCreate.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ]
            audio.AudioWorkIntervalCreate.restype = ctypes.c_void_p
            for name, args, result in (
                ("os_workgroup_join", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
                ("os_workgroup_leave", [ctypes.c_void_p, ctypes.c_void_p], None),
                (
                    "os_workgroup_interval_start",
                    [
                        ctypes.c_void_p,
                        ctypes.c_uint64,
                        ctypes.c_uint64,
                        ctypes.c_void_p,
                    ],
                    ctypes.c_int,
                ),
                (
                    "os_workgroup_interval_finish",
                    [ctypes.c_void_p, ctypes.c_void_p],
                    ctypes.c_int,
                ),
                ("mach_absolute_time", [], ctypes.c_uint64),
                ("mach_timebase_info", [ctypes.POINTER(_Timebase)], ctypes.c_int),
            ):
                func = getattr(system, name)
                func.argtypes = args
                func.restype = result
            self._system = system
            timebase = _Timebase()
            if system.mach_timebase_info(ctypes.byref(timebase)) != 0 or not (
                timebase.numer and timebase.denom
            ):
                raise RuntimeError("mach_timebase_info failed")
            self._deadline_ticks = max(
                1, STEP_DEADLINE_MS * 1_000_000 * timebase.denom // timebase.numer
            )
            self._wg = audio.AudioWorkIntervalCreate(
                b"rapid-mlx-step", _OS_CLOCK_MACH_ABSOLUTE_TIME, None
            )
            if not self._wg:
                raise RuntimeError("AudioWorkIntervalCreate failed")
        except Exception as exc:
            self._disable(exc)

    @contextmanager
    def interval(self, *, has_work: bool = True) -> Iterator[None]:
        if not has_work or self._disabled or not self._lock.acquire(blocking=False):
            yield
            return
        joined = False
        started = False
        try:
            try:
                token = ctypes.create_string_buffer(256)  # opaque, per-call token
                if not self._attempted:
                    self._initialize()
                if not self._disabled and self._system is not None:
                    if self._system.os_workgroup_join(self._wg, token) != 0:
                        raise RuntimeError("os_workgroup_join failed")
                    joined = True
                    now = self._system.mach_absolute_time()
                    if (
                        self._system.os_workgroup_interval_start(
                            self._wg, now, now + self._deadline_ticks, None
                        )
                        != 0
                    ):
                        raise RuntimeError("os_workgroup_interval_start failed")
                    started = True
            except Exception as exc:
                self._disable(exc)
            yield
        finally:
            # Leave even when start fails, finish fails, or the step raises.
            try:
                try:
                    if started and self._system is not None:
                        if (
                            self._system.os_workgroup_interval_finish(self._wg, None)
                            != 0
                        ):
                            raise RuntimeError("os_workgroup_interval_finish failed")
                except Exception as exc:
                    self._disable(exc)
                finally:
                    try:
                        if joined and self._system is not None:
                            self._system.os_workgroup_leave(self._wg, token)
                    except Exception as exc:
                        self._disable(exc)
            finally:
                self._lock.release()


step_workgroup = StepWorkgroup()
