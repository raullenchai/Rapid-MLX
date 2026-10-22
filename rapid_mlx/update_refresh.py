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


def _reset_after_fork():
    global _lock, _thread, _latest, _available, _seen_versions
    _lock = threading.Lock()
    _thread = None
    _latest = None
    _available = False
    _seen_versions = set()


os.register_at_fork(after_in_child=_reset_after_fork)


def start(
    *, announced: str | None = None, interval: float = vc.CACHE_TTL_SECONDS
) -> bool:
    """Start at most one background checker per process, unless opted out."""
    global _thread
    with _lock:
        if vc._explicitly_disabled() or _thread is not None:
            return False
        _thread = threading.Thread(
            target=_run, args=(interval,), daemon=True, name="rapid-mlx-update-check"
        )
        newly_announced = announced is not None and announced not in _seen_versions
        if newly_announced and announced is not None:
            _seen_versions.add(announced)
        try:
            _thread.start()
        except Exception:
            if newly_announced:
                _seen_versions.discard(announced)
            _thread = None
            return False
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
        if _thread is None or vc._explicitly_disabled():
            return {"latest_version": None, "update_available": False}
        return {"latest_version": _latest, "update_available": _available}
