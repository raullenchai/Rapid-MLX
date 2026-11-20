# SPDX-License-Identifier: Apache-2.0
"""Bounded, stdlib-only process-tree tracking for supervised serve on POSIX."""

import contextlib
import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)


def process_table():
    """Return PID -> (PPID, state, start time), on both macOS and Linux.

    Keep start times to reject recycled PIDs. lstart is locale-dependent;
    force C so its representation stays stable across snapshots.
    """
    with subprocess.Popen(
        ["/bin/ps", "-Ao", "pid=,ppid=,stat=,lstart="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    ) as ps:
        try:
            output, _ = ps.communicate(timeout=0.1)
        except subprocess.TimeoutExpired:
            ps.kill()
            ps.communicate()
            raise
        if ps.returncode:
            raise subprocess.CalledProcessError(ps.returncode, ps.args)
    table = {}
    for line in output.splitlines():
        pid, ppid, state, started = line.split(None, 3)
        # In the runner watchdog, ps is itself a child of the tracked root.
        # Exclude the observer so an empty tree can finish its grace period.
        if int(pid) != ps.pid:
            table[int(pid)] = (int(ppid), state, started)
    return table


class ProcessTree:
    def __init__(self, pid):
        self.pid = pid
        self.known = {}
        self.root_seen = False
        self.stopped = False
        self.read_failed = False

    def refresh(self):
        try:
            table = process_table()
        except (OSError, subprocess.SubprocessError, ValueError):
            if not self.read_failed:
                logger.warning("Cannot read supervised runner process tree")
            self.read_failed = True
            return None
        self.read_failed = False
        if not self.root_seen and self.pid in table:
            self.known[self.pid] = table[self.pid][2]
            self.root_seen = True
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

    def send(self, signum, *, include_root=True):
        live = self.refresh()
        if live is None:
            # Do not signal stale PIDs when identity cannot be rechecked. The
            # supervisor can still safely signal its unreaped direct child.
            live = set()
        if not include_root:
            live.discard(self.pid)
        # Signal descendants before the runner can exit and reparent them.
        for pid in sorted(live, key=lambda pid: pid == self.pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signum)
        return live


def supervised_on_orphan(expected_ppid, observed_ppid):
    """Clean children before signalling self: the watchdog thread is a daemon.

    Only the control runner installs this callback. Desktop's existing orphan
    callback and its shutdown timing remain unchanged.
    """
    from rapid_mlx._parent_watchdog import _default_on_orphan

    # Snapshot before signalling self: TERM may immediately exit the runner,
    # killing this daemon thread and reparenting children that still hold pipes.
    # ProcessTree retains start times and rechecks identity before escalation.
    tree = ProcessTree(os.getpid())
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
