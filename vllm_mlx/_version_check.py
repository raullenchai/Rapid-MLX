# SPDX-License-Identifier: Apache-2.0
"""Background check for newer ``rapid-mlx`` releases on GitHub.

Surfaces a one-line warning at the top of ``rapid-mlx models``,
``rapid-mlx serve`` and ``rapid-mlx chat`` whenever the installed
version is behind the latest GitHub release — ANY newer release,
regardless of how far behind (patch, minor, or major). There is no
version-distance threshold: the whole point is that stragglers stuck on
an old line (the version fragmentation we care about) actually see the
nudge on every command, not only the ones a patch or two behind inside
the same minor. Designed to fail completely silently on network / parse
/ sandbox errors — staleness warnings should never break the CLI.

Cache: ``~/.cache/rapid-mlx/version_check.json`` with 24h TTL. Network
fetch is opt-out via ``RAPID_MLX_DISABLE_VERSION_CHECK=1`` or any
non-interactive context (``CI=1``, missing TTY).

Behaviour matrix:

  installed = 0.6.15, latest = 0.6.16 (1 patch behind)
    → warns, suggests ``rapid-mlx upgrade``

  installed = 0.10.8, latest = 0.12.10 (behind by minors)
    → warns (cross-minor stragglers are exactly who we want to reach)

  installed = 0.9.9, latest = 1.0.0 (behind by a major)
    → warns

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
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

# The update check routes through the landing worker rather than hitting
# api.github.com directly. The worker at ``rapidmlx.com/api/cli-update``
# is a passthrough that returns the SAME GitHub ``releases/latest`` JSON,
# so parsing is unchanged; the indirection just lets the server count
# active CLI polls (mirrors what the desktop app now does).
#
# What goes on the wire, precisely: the installed version, URL-encoded as
# the ``v`` query param (so counts bucket by version), PLUS the transport
# metadata every HTTP request unavoidably carries — the client IP and a
# User-Agent. We pin a fixed, non-identifying ``USER_AGENT`` below so the
# UA leaks nothing (urllib would otherwise default to
# ``Python-urllib/<x.y.z>``, exposing the interpreter patch version). This
# is the SAME network exposure the previous direct ``api.github.com`` call
# already had — the only change is the recipient is now our own endpoint.
# Never sent: client id, os/arch, flag values, prompt or generated content.
CLI_UPDATE_ENDPOINT = "https://rapidmlx.com/api/cli-update"
GITHUB_RELEASES_ENDPOINT = (
    "https://api.github.com/repos/raullenchai/Rapid-MLX/releases?per_page=100"
)
# Fixed, non-identifying User-Agent so the poll carries no data beyond the
# ``v`` param + unavoidable IP. Overrides urllib's ``Python-urllib/x.y.z``.
USER_AGENT = "rapid-mlx-cli"
CACHE_TTL_SECONDS = 24 * 3600  # 24h
NETWORK_TIMEOUT_SECONDS = 2  # tight — staleness check is best-effort
_REMOTE_ENGINE_TAG_RE = re.compile(r"^v(\d{1,6})\.(\d{1,6})\.(\d{1,6})$")
_CACHED_VERSION_RE = re.compile(r"^\d{1,6}\.\d{1,6}\.\d{1,6}$")


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
        if (
            isinstance(data, dict)
            and isinstance(data.get("latest"), str)
            and _CACHED_VERSION_RE.fullmatch(data["latest"])
        ):
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


def _fetch_latest() -> str | None:
    """Fetch the latest release tag via the landing worker.

    Routes through ``rapidmlx.com/api/cli-update`` instead of
    api.github.com directly so the poll is countable server-side. The
    worker passes the GitHub ``releases/latest`` JSON straight through,
    so the parse (``tag_name``) is identical to the old direct fetch.

    Privacy: the only application data sent is the installed version,
    URL-encoded as the ``v`` query param (empty string when running from
    an uninstalled source tree). Like any HTTP request it also exposes the
    client IP and a User-Agent — we pin the fixed, non-identifying
    ``USER_AGENT`` so nothing beyond the version + unavoidable transport
    metadata leaves the machine (no client id, no os/arch, no interpreter
    version, no headers that identify the host). Same network exposure as
    the prior direct GitHub call; only the recipient changed. Fail-open:
    any network / parse / sandbox error returns None silently, exactly as
    before.
    """
    try:
        installed = _installed_version() or ""
        query = urllib.parse.urlencode({"v": installed})
        url = f"{CLI_UPDATE_ENDPOINT}?{query}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        tag = data.get("tag_name") if isinstance(data, dict) else None
        if isinstance(tag, str) and _REMOTE_ENGINE_TAG_RE.fullmatch(tag):
            return tag.lstrip("v")

        # GitHub's "latest release" can be a desktop release tagged
        # ``rapid-mac-vX.Y.Z``. That is a different product/version stream,
        # so fall back to the repository tag inventory and select only exact
        # engine tags. This path is exceptional; normal polls remain countable
        # through the landing worker above.
        page = 1
        versions: list[tuple[int, int, int]] = []
        while True:
            separator = "&" if "?" in GITHUB_RELEASES_ENDPOINT else "?"
            fallback = urllib.request.Request(
                f"{GITHUB_RELEASES_ENDPOINT}{separator}page={page}",
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": USER_AGENT,
                },
            )
            with urllib.request.urlopen(
                fallback, timeout=NETWORK_TIMEOUT_SECONDS
            ) as fallback_resp:
                releases = json.loads(fallback_resp.read())
            if not isinstance(releases, list):
                return None
            # GitHub returns releases newest-first. The first exact engine tag
            # is therefore authoritative; desktop releases are simply skipped.
            for item in releases:
                candidate = item.get("tag_name") if isinstance(item, dict) else None
                match = (
                    _REMOTE_ENGINE_TAG_RE.fullmatch(candidate)
                    if isinstance(candidate, str)
                    else None
                )
                if match:
                    versions.append(tuple(map(int, match.groups())))
            if len(releases) < 100:
                return ".".join(map(str, max(versions))) if versions else None
            page += 1
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
    latest = _fetch_latest()
    if latest is not None:
        _write_cache(latest)
    return latest


def staleness_warning() -> str | None:
    """Return a one-line warning string if the installed version is behind
    the latest release by ANY amount (patch, minor, or major). Returns
    None when no warning is warranted (current/ahead, or check disabled).
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

    # Warn on ANY newer release — no version-distance gate. One patch, a
    # whole minor, or a major behind all get the banner: cross-line
    # stragglers (e.g. 0.10.x while 0.12.x ships) are precisely who we
    # want to reach, and they'd never see a same-minor-only warning.
    # This is a passive one-liner (it only *suggests* ``rapid-mlx
    # upgrade``), so unlike the interactive ``prompt_upgrade_if_available``
    # — which auto-runs an upgrade and therefore conservatively skips
    # dev / pre-release builds — it can safely fire for anyone behind.
    # Developers on builds AHEAD of latest still stay silent via the
    # comparison below.
    if latest <= installed:
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


def prompt_upgrade_if_available() -> bool:
    """Interactive Y/n prompt when a newer release is on GitHub.

    Designed for the top of long-lived entry points (``rapid-mlx serve``):
    if the network has a newer release than what's installed, ask once
    before booting the model. On accept, dispatch the right upgrade
    command (brew/pip/install.sh — same dispatcher as ``rapid-mlx
    upgrade``) and return ``True`` so the caller can exit. Returns
    ``False`` when no prompt was shown (disabled, non-TTY, already
    current, dev build, network down) or the user declined.

    Distinct from ``staleness_warning()`` in two ways: (1) prompts on ANY
    newer release, not only ≥2-patch lag, because if we're going to
    interrupt a long-running boot we may as well save the user the
    re-launch; (2) crosses minor-version boundaries, because an
    interactive opt-in is safer than the silent banner's automatic
    cross-minor restraint.
    """
    try:
        if _disabled():
            return False
        # Need stdin for the prompt too — _disabled checks stderr only.
        if not sys.stdin.isatty():
            return False

        installed_str = _installed_version()
        if not installed_str:
            return False
        # Skip pre-release / dev / local-version builds. ``_parse_version``
        # tolerates ``0.6.62.dev1`` → ``(0, 6, 62)`` so the tuple can be
        # compared at all, but for an interactive prompt the dev-base case
        # can fire a false "newer release available" against the dev's
        # own in-progress branch (installed ``0.6.61.dev1`` vs latest
        # ``0.6.62`` → prompt). A clean PEP 440 final release is digits
        # and dots only; anything else (``dev``/``a``/``b``/``rc``/
        # ``post``/``+local``) is non-final and gets the dev-build skip
        # path. DeepSeek finding #1 on PR #428.
        if any(c.isalpha() or c == "+" for c in installed_str.lstrip("v")):
            return False
        installed = _parse_version(installed_str)
        if installed is None:
            return False  # unparseable

        latest_str = get_latest_version()
        if not latest_str:
            return False
        latest = _parse_version(latest_str)
        if latest is None or latest <= installed:
            return False

        import subprocess

        info = detect_install_method()
        print(
            f"\nA newer rapid-mlx is available: {latest_str} "
            f"(current: {installed_str})."
        )
        print(f"  Upgrade command: {info.upgrade_command}")
        try:
            answer = input("  Run it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer and answer not in ("y", "yes"):
            return False

        print()
        try:
            result = subprocess.run(info.upgrade_argv, check=False)
        except FileNotFoundError as e:
            print(f"  Upgrade command not found: {e}\n")
            return False
        except KeyboardInterrupt:
            print("\n  Interrupted.\n")
            return False
        if result.returncode == 0:
            print("\nUpgrade complete. Please re-run your command.\n")
            return True
        # Failed upgrade: return False so the caller continues booting with
        # the current version rather than exiting silently. The user has
        # been shown the exit code and can retry manually if they care.
        # DeepSeek finding #2 on PR #428: returning True here would have
        # `serve_command` do ``sys.exit(0)`` and leave the user with no
        # running server and only an error message.
        print(
            f"\nUpgrade exited with code {result.returncode}; "
            f"continuing with rapid-mlx {installed_str}.\n"
        )
        return False
    except Exception:  # noqa: BLE001 — never break the CLI
        return False


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
         triggers ``brew upgrade rapid-mlx`` (now in homebrew/core).
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
                upgrade_command="brew upgrade rapid-mlx",
                upgrade_argv=["brew", "upgrade", "rapid-mlx"],
                binary_path=binary,
            )
        # install.sh creates ``~/.rapid-mlx`` (venv) and symlinks the
        # entry point into ``~/.local/bin``. Match either side: the
        # symlink path (binary) for fresh installs, the venv root
        # (normalized) for installs where ``~/.local/bin`` was overridden.
        local_bin = str(Path.home() / ".local" / "bin")
        rapid_mlx_dir = str(Path.home() / ".rapid-mlx")
        if binary.startswith(local_bin) or normalized.startswith(rapid_mlx_dir):
            # Prefer the canonical rapidmlx.com host (the same domain the
            # website, docs, and the update poll ``CLI_UPDATE_ENDPOINT`` use),
            # falling back to the raullenchai.github.io Pages mirror when it
            # can't be reached. Two independent hosts, no single point of
            # failure: if rapidmlx.com is down the mirror still upgrades, and —
            # because ``*.github.io`` is frequently unreachable from mainland
            # China — trying rapidmlx.com (Cloudflare) first keeps that path
            # working there.
            #
            # Download to a temp file and execute only a *complete* fetch,
            # rather than ``curl ... | bash``: a naive pipe (a) reports the
            # pipeline's ``bash`` status, so if BOTH hosts fail bash reads empty
            # input and exits 0 — a false "upgrade succeeded"; and (b) streams
            # bytes as they arrive, so a mid-transfer primary failure could feed
            # a partial script to bash before the fallback even runs. ``set -e``
            # + ``curl -o`` fixes both: ``curl -f`` exits non-zero on any HTTP
            # error (the primary's stderr silenced so a clean fallback is quiet;
            # the mirror keeps its stderr), ``||`` reaches the mirror only when
            # the primary fails, ``curl -o`` overwrites the temp file so the two
            # candidates never concatenate, and ``set -e`` aborts (non-zero,
            # before ``bash``) when both fail. The trap cleans up on every exit.
            # Bounded timeouts (``--connect-timeout``/``--max-time``) are what
            # make the fallback actually reachable: a blocked or blackholed host
            # frequently hangs the connection rather than returning a clean HTTP
            # error, and without a cap the primary ``curl`` would wait forever
            # and never reach the mirror. 5s to connect + 30s overall is ample
            # for a few-KB script while failing fast when rapidmlx.com stalls.
            _curl = "curl -fsSL --connect-timeout 5 --max-time 30"
            install_sh_pipe = (
                'set -e; t="$(mktemp)"; trap \'rm -f "$t"\' EXIT; '
                f'{_curl} https://rapidmlx.com/install.sh -o "$t" 2>/dev/null || '
                f'{_curl} https://raullenchai.github.io/Rapid-MLX/install.sh -o "$t"; '
                'bash "$t"'
            )
            return InstallInfo(
                method="install_sh",
                upgrade_command=install_sh_pipe,
                # The script needs a shell — use bash -c explicitly rather than
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
