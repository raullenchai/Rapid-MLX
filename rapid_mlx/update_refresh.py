# SPDX-License-Identifier: Apache-2.0
"""Best-effort periodic update notices and an IO-free health snapshot."""

import os
import sys
import threading
import time

from . import _version_check as vc

_lock = threading.Lock()
_latest: str | None = None
_seen_versions: set[str] = set()
_available = False
_thread: threading.Thread | None = None
_pid: int | None = None


def start(
    *, announced: str | None = None, interval: float = vc.CACHE_TTL_SECONDS
) -> bool:
    """Start at most one background checker per process, unless opted out."""
    global _thread, _pid, _latest, _available
    with _lock:
        if _pid != os.getpid():
            _thread, _pid = None, None
            _seen_versions.clear()
            _latest, _available = None, False
        if vc._explicitly_disabled() or _thread is not None:
            return False
        _thread = threading.Thread(
            target=_run, args=(interval,), daemon=True, name="rapid-mlx-update-check"
        )
        try:
            _thread.start()
        except Exception:
            _thread = None
            return False
        _pid = os.getpid()
        if announced is not None:
            _seen_versions.add(announced)
        return True


def _run(interval):
    while True:
        try:
            _refresh()
        except Exception:
            pass
        time.sleep(interval)


def _refresh():
    global _latest, _available
    if vc._explicitly_disabled():
        with _lock:
            _latest, _available = None, False
        return
    latest = vc.get_latest_version()
    if latest is None:
        return
    installed = vc._installed_version() or ""
    current, candidate = vc._parse_version(installed), vc._parse_version(latest)
    available = current is not None and candidate is not None and candidate > current
    with _lock:
        unseen = latest not in _seen_versions
        _seen_versions.add(latest)
        _latest, _available = latest, available
    if available and unseen:
        print(vc.format_staleness_warning(installed, latest), file=sys.stderr)


def snapshot() -> dict:
    """Read cached status without fetching or accessing the on-disk cache."""
    with _lock:
        if _pid != os.getpid() or _thread is None or vc._explicitly_disabled():
            return {"latest_version": None, "update_available": False}
        return {"latest_version": _latest, "update_available": _available}
