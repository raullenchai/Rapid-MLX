# SPDX-License-Identifier: Apache-2.0
"""Background check for newer ``rapid-mlx`` releases on GitHub.

Surfaces a one-line warning at the top of ``rapid-mlx models``,
``rapid-mlx serve`` and ``rapid-mlx chat`` when the installed version
is at least 2 patch versions behind the latest GitHub release. Designed
to fail completely silently on network / parse / sandbox errors —
staleness warnings should never break the CLI.

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
        f"run `rapid-mlx upgrade` to pick up new model aliases / flags."
    )


def print_staleness_warning_if_any() -> None:
    """Best-effort: fetches + prints to stderr. Always silent on errors."""
    try:
        msg = staleness_warning()
        if msg:
            print(msg, file=sys.stderr)
    except Exception:  # noqa: BLE001 — never break the CLI
        pass


# --- install-method detection (used by ``rapid-mlx upgrade``) -----------


class InstallInfo:
    """Detected install method + the right upgrade command to run.

    ``upgrade_argv`` is the form actually executed (``subprocess.run`` with
    no shell), avoiding the injection risk from interpolating
    ``sys.executable`` (or any other path that might contain spaces) into a
    shell-parsed string. ``upgrade_command`` is the cosmetic form printed
    to the user before they confirm.

    Plain class (not dataclass) so the module stays stdlib-only — staleness
    helper is loaded on every CLI startup, so we keep its import surface
    minimal.
    """

    __slots__ = ("method", "upgrade_command", "upgrade_argv", "binary_path")

    def __init__(
        self,
        method: str,
        upgrade_command: str,
        upgrade_argv: list[str],
        binary_path: str | None = None,
    ) -> None:
        self.method = method  # one of: brew, pip, install_sh
        self.upgrade_command = upgrade_command
        self.upgrade_argv = upgrade_argv
        self.binary_path = binary_path


def detect_install_method() -> InstallInfo:
    """Detect how rapid-mlx was installed and return the right upgrade command.

    Detection order:
      1. brew — ``rapid-mlx`` realpath under ``/Cellar/rapid-mlx``,
         ``/opt/homebrew/`` (macOS) or ``/home/linuxbrew/`` (Linux brew)
         triggers ``brew upgrade raullenchai/tap/rapid-mlx``.
      2. install.sh — binary under ``~/.local/bin`` (or realpath under
         the install.sh venv at ``~/.rapid-mlx/``) triggers a re-run of
         the install.sh script.
      3. pip (default) — uses ``sys.executable -m pip install --upgrade``
         so the upgrade lands in the same env that's currently running
         the CLI.
    """
    import shutil

    binary = shutil.which("rapid-mlx")
    if binary:
        normalized = os.path.realpath(binary)
        brew_markers = ("/Cellar/rapid-mlx", "/opt/homebrew/", "/home/linuxbrew/")
        if any(m in normalized for m in brew_markers):
            return InstallInfo(
                method="brew",
                upgrade_command="brew upgrade raullenchai/tap/rapid-mlx",
                upgrade_argv=["brew", "upgrade", "raullenchai/tap/rapid-mlx"],
                binary_path=binary,
            )
        # install.sh creates ``~/.rapid-mlx`` (venv) and symlinks the
        # entry point into ``~/.local/bin``. Match either side: the
        # symlink path (binary) for fresh installs, the venv root
        # (normalized) for installs where ``~/.local/bin`` was overridden.
        local_bin = str(Path.home() / ".local" / "bin")
        rapid_mlx_dir = str(Path.home() / ".rapid-mlx")
        if binary.startswith(local_bin) or normalized.startswith(rapid_mlx_dir):
            install_sh_pipe = (
                "curl -fsSL https://raullenchai.github.io/Rapid-MLX/install.sh | bash"
            )
            return InstallInfo(
                method="install_sh",
                upgrade_command=install_sh_pipe,
                # Pipe needs a shell — use bash -c explicitly rather than
                # ``shell=True`` (no ambient $SHELL coupling, no PATH-based
                # shell-injection surface beyond the literal string we control).
                upgrade_argv=["bash", "-c", install_sh_pipe],
                binary_path=binary,
            )

    return InstallInfo(
        method="pip",
        upgrade_command=f"{sys.executable} -m pip install --upgrade rapid-mlx",
        upgrade_argv=[
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "rapid-mlx",
        ],
        binary_path=binary,
    )
