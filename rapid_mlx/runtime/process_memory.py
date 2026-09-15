# SPDX-License-Identifier: Apache-2.0
"""Authoritative macOS process-memory pressure measurement.

The implementation follows the ``proc_pid_rusage(RUSAGE_INFO_V4)`` approach
used by oMLX. ``phys_footprint`` includes IOAccelerator-backed allocations that
ordinary RSS can omit on Apple Silicon, making it the useful companion to MLX's
active-memory counter for long-context admission decisions.
"""

from __future__ import annotations

import ctypes
import os
import sys


class _RusageInfoV4(ctypes.Structure):
    # Layout from macOS <sys/resource.h>. Keep all fields through
    # ri_interval_max_phys_footprint so the kernel can fill the V4 struct.
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_proc_pid_rusage = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _proc_pid_rusage = _libproc.proc_pid_rusage
        _proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _proc_pid_rusage.restype = ctypes.c_int
    except OSError:
        _proc_pid_rusage = None


def get_phys_footprint(pid: int | None = None) -> int:
    """Return macOS ``phys_footprint`` bytes, or zero when unavailable."""
    if _proc_pid_rusage is None:
        return 0
    info = _RusageInfoV4()
    rc = _proc_pid_rusage(pid or os.getpid(), 4, ctypes.byref(info))
    return int(info.ri_phys_footprint) if rc == 0 else 0
