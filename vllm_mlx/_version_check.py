# SPDX-License-Identifier: Apache-2.0
"""Background check for newer ``rapid-mlx`` releases on GitHub.

Surfaces a one-line warning at the top of ``rapid-mlx models`` (and
similarly user-facing entrypoints) when the installed version is at
least 2 patch versions behind the latest GitHub release. Designed to
fail completely silently on network / parse / sandbox errors — staleness
warnings should never break the CLI.

Cache: ``~/.cache/rapid-mlx/version_check.json`` with 24h TTL. Network
fetch is opt-out via ``RAPID_MLX_DISABLE_VERSION_CHECK=1`` or any
non-interactive context (``CI=1``, missing TTY).

Behaviour matrix:

  installed = 0.6.14, latest = 0.6.16 (2 patch behind)
    → warns, suggests ``brew upgrade``

  installed = 0.6.16, latest = 0.6.16 (current)
    → silent

  installed = 0.7.0, latest = 0.6.16 (dev ahead)
    → silent (don't nag developers running their own builds)

  no network / cache miss / GitHub 5xx
    → silent (fail-closed)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

GITHUB_LATEST_API = "https://api.github.com/repos/raullenchai/Rapid-MLX/releases/latest"
CACHE_TTL_SECONDS = 24 * 3600  # 24h
NETWORK_TIMEOUT_SECONDS = 2  # tight — staleness check is best-effort
# Minimum patch lag before warning. Bumping by 1 patch happens often
# enough that a one-version lag is normal noise; 2+ means a feature
# release was missed.
MIN_LAG_PATCH = 2


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "rapid-mlx" / "version_check.json"


def _disabled() -> bool:
    """Skip the check in non-interactive contexts.

    Devs running tests, CI, scripts piped to other tools — none of them
    benefit from a version warning. Only show when stderr is a TTY and
    the user hasn't explicitly opted out.
    """
    if os.environ.get("RAPID_MLX_DISABLE_VERSION_CHECK"):
        return True
    if os.environ.get("CI"):
        return True
    try:
        # ``stderr.isatty()`` matches where we'd print the warning.
        return not sys.stderr.isatty()
    except Exception:  # noqa: BLE001 — stderr might be replaced
        return True


def _parse_version(s: str) -> tuple[int, int, int] | None:
    """Strict-ish ``major.minor.patch`` parse; returns None for anything
    weirder. We deliberately don't try to handle dev/rc suffixes —
    if a user is running a dev build, ``pkg_version`` returns
    ``X.Y.Z.devN`` and we just stay silent.
    """
    parts = s.strip().lstrip("v").split(".")
    if len(parts) < 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _read_cache() -> dict | None:
    p = _cache_path()
    try:
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        with p.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "latest" in data:
            return data
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(latest: str) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump({"latest": latest, "ts": int(time.time())}, f)
    except OSError:
        # Cache write failure is non-fatal — we'll just refetch next time.
        pass


def _fetch_latest_from_github() -> str | None:
    try:
        req = urllib.request.Request(
            GITHUB_LATEST_API,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        tag = data.get("tag_name")
        if not isinstance(tag, str):
            return None
        return tag.lstrip("v")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _installed_version() -> str | None:
    try:
        return pkg_version("rapid-mlx")
    except PackageNotFoundError:
        return None


def get_latest_version(force_refresh: bool = False) -> str | None:
    """Return the latest GitHub release version, or None.

    Cache-first to keep the CLI snappy. ``force_refresh=True`` is for
    tests; production code path always tries cache.
    """
    if not force_refresh:
        cached = _read_cache()
        if cached is not None:
            v = cached.get("latest")
            if isinstance(v, str):
                return v
    latest = _fetch_latest_from_github()
    if latest is not None:
        _write_cache(latest)
    return latest


def staleness_warning() -> str | None:
    """Return a one-line warning string if the installed version is
    ``MIN_LAG_PATCH`` or more patch versions behind the latest release.
    Returns None when no warning is warranted (or check is disabled).
    """
    if _disabled():
        return None
    installed_str = _installed_version()
    if not installed_str:
        return None
    installed = _parse_version(installed_str)
    if installed is None:
        return None  # dev build / unparseable

    latest_str = get_latest_version()
    if not latest_str:
        return None  # offline / GitHub down — be silent
    latest = _parse_version(latest_str)
    if latest is None:
        return None

    # Only warn for patch-level lag inside the same major.minor — across
    # minors there might be intentional API changes the user is staying
    # on for stability. Across majors, definitely silent.
    if (installed[0], installed[1]) != (latest[0], latest[1]):
        return None
    if latest[2] - installed[2] < MIN_LAG_PATCH:
        return None

    return (
        f"⚠ rapid-mlx {installed_str} is behind latest {latest_str} — "
        f"run `brew upgrade rapid-mlx` (or `pip install -U rapid-mlx`) "
        f"to pick up new model aliases / flags."
    )


def print_staleness_warning_if_any() -> None:
    """Best-effort: fetches + prints to stderr. Always silent on errors."""
    try:
        msg = staleness_warning()
        if msg:
            print(msg, file=sys.stderr)
    except Exception:  # noqa: BLE001 — never break the CLI
        pass
