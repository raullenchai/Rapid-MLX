# SPDX-License-Identifier: Apache-2.0
"""Bounded, stdlib-only process-tree tracking for supervised serve on POSIX."""

import contextlib
import ctypes
import functools
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class _ProcBsdInfo(ctypes.Structure):
    """Public libproc PROC_PIDTBSDINFO layout (136 bytes on macOS)."""

    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "flags",
            "status",
            "xstatus",
            "pid",
            "ppid",
            "uid",
            "gid",
            "ruid",
            "rgid",
            "svuid",
            "svgid",
            "reserved",
        )
    ] + [
        ("comm", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        ("nfiles", ctypes.c_uint32),
        ("pgid", ctypes.c_uint32),
        ("pjobc", ctypes.c_uint32),
        ("tdev", ctypes.c_uint32),
        ("tpgid", ctypes.c_uint32),
        ("nice", ctypes.c_int32),
        ("start_sec", ctypes.c_uint64),
        ("start_usec", ctypes.c_uint64),
    ]


@functools.lru_cache(maxsize=1)
def _libproc():
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.proc_pidinfo.restype = ctypes.c_int
    lib.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.proc_listallpids.restype = ctypes.c_int
    return lib


def _linux_info(stat):
    # comm can contain spaces and ')' characters; fields after its final ')'
    # begin at field 3 (state). starttime is field 22, in kernel clock ticks.
    fields = stat[stat.rindex(")") + 2 :].split()
    return int(fields[1]), fields[0], int(fields[19])


def process_info(pid):
    """Read one PID's (PPID, state, native start identity), never invoking ps."""
    try:
        if sys.platform == "darwin":
            info = _ProcBsdInfo()
            size = ctypes.sizeof(info)
            if _libproc().proc_pidinfo(pid, 3, 0, ctypes.byref(info), size) != size:
                return None
            state = {4: "T", 5: "Z"}.get(info.status, "S")
            return info.ppid, state, (info.start_sec, info.start_usec)
        if sys.platform == "linux":
            return _linux_info(Path(f"/proc/{pid}/stat").read_text())
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_table():
    """Snapshot native process identities, without starting observer processes."""
    if sys.platform == "darwin":
        lib = _libproc()
        count = lib.proc_listallpids(None, 0)
        if count <= 0:
            raise OSError("Cannot enumerate processes")
        # Allow for processes born between sizing and enumeration. A truncated
        # snapshot is safe: descendant cleanup is deliberately best-effort.
        buffer = (ctypes.c_int * (count + 64))()
        count = lib.proc_listallpids(buffer, ctypes.sizeof(buffer))
        if count <= 0:
            raise OSError("Cannot enumerate processes")
        pids = list(buffer[:count])
    elif sys.platform == "linux":
        pids = [
            int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()
        ]
    else:
        raise OSError("Unsupported process enumeration platform")
    return {pid: info for pid in pids if pid > 0 and (info := process_info(pid))}


class ProcessTree:
    def __init__(self, pid):
        self.pid = pid
        info = process_info(pid)
        self.root_identity = info[2] if info is not None else None
        self.known = {pid: self.root_identity} if info is not None else {}
        self.stopped = False
        self.read_failed = False
        self.lock = threading.RLock()
        self.tracking_stop = threading.Event()

    def start_tracking(self):
        """Sample off the event loop, no more often than every five seconds."""

        def track():
            while not self.tracking_stop.is_set():
                self.refresh()
                if self.tracking_stop.wait(5.0):
                    break

        threading.Thread(target=track, daemon=True).start()

    def stop_tracking(self):
        self.tracking_stop.set()

    def update_root_state(self):
        # Job-control status needs a single cheap PID lookup, not a tree scan.
        info = process_info(self.pid)
        self.stopped = info is not None and "T" in info[1].upper()

    def refresh(self):
        with self.lock:
            return self._refresh()

    def _refresh(self):
        try:
            table = process_table()
        except (OSError, ValueError):
            if not self.read_failed:
                logger.warning("Cannot read supervised runner process tree")
            self.read_failed = True
            return None
        self.read_failed = False
        # Retain descendants across reparenting, but never adopt recycled PIDs.
        live = {
            pid: started
            for pid, started in self.known.items()
            if pid in table and table[pid][2] == started
        }
        children = {}
        for pid, (ppid, _, _) in table.items():
            children.setdefault(ppid, []).append(pid)
        pending = list(live)
        while pending:
            for pid in children.get(pending.pop(), ()):
                if pid not in live:
                    live[pid] = table[pid][2]
                    pending.append(pid)
        self.known = live
        self.stopped = self.pid in live and "T" in table[self.pid][1].upper()
        return {pid for pid in live if not table[pid][1].upper().startswith("Z")}

    def send_root(self, signum):
        # Enumeration may fail while one-PID lookup still works. Keep the
        # original identity even after refresh drops a dead/recycled root.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            info = process_info(self.pid)
            if info is not None and info[2] == self.root_identity:
                os.kill(self.pid, signum)

    def send(self, signum, *, include_root=True):
        with self.lock:
            live = self._refresh()
            if live is None:
                # Fail closed when enumeration fails. The supervisor can still
                # revalidate its direct child with a one-PID lookup.
                return set()
            if not include_root:
                live.discard(self.pid)
            sent = set()
            # Descendants first, before the runner can exit and reparent them.
            for pid in sorted(live, key=lambda pid: pid == self.pid):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    # Revalidate immediately before EACH signal, including TERM
                    # and KILL. A tree snapshot alone leaves a PID-reuse window.
                    info = process_info(pid)
                    if info is not None and info[2] == self.known[pid]:
                        os.kill(pid, signum)
                        sent.add(pid)
            return sent


_orphan_tree = None


def start_orphan_tracking():
    """Retain observed descendants across reparenting before parent loss."""
    global _orphan_tree
    if _orphan_tree is None:
        _orphan_tree = ProcessTree(os.getpid())
        _orphan_tree.start_tracking()


def supervised_on_orphan(expected_ppid, observed_ppid):
    """Clean children before signalling self: the watchdog thread is a daemon.

    Only the control runner installs this callback. Desktop's existing orphan
    callback and its shutdown timing remain unchanged.
    """
    from rapid_mlx._parent_watchdog import _default_on_orphan

    # Snapshot before signalling self: TERM may immediately exit the runner,
    # killing this daemon thread and reparenting children that still hold pipes.
    # ProcessTree retains start times and rechecks identity before escalation.
    tree = _orphan_tree or ProcessTree(os.getpid())
    tree.stop_tracking()
    tree.send(signal.SIGTERM, include_root=False)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        live = tree.refresh()
        if live is None or not (live - {os.getpid()}):
            break
        time.sleep(0.05)
    tree.send(signal.SIGKILL, include_root=False)
    # The supervised contract is four seconds including the 0.5 s poll and
    # process-table reads. Do not inherit the desktop's five-second self grace.
    _default_on_orphan(expected_ppid, observed_ppid, grace=1.0)
