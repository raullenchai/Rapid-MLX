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
fetch is opt-out via ``RAPID_MLX_DISABLE_VERSION_CHECK=1`` or ``CI=1``.
Interactive commands require a TTY; ``serve`` also writes the passive
notice to non-interactive startup logs so launchd/daemon operators are
not left on an old release with no signal.

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

  latest requires a newer Python
    → explain how to move the install to a supported interpreter

  Homebrew formula has not advanced
    → silent until Homebrew offers a newer version

  no network / cache miss / GitHub 5xx
    → silent (fail-closed)
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import sysconfig
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as pkg_version
from pathlib import Path

from rapid_mlx._env import is_falsey

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
_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
# ``X.Y.Z.devN`` (and other dotted-suffix builds like ``.postN``) keep the
# old ``major.minor.patch``-only behavior: they parse to the base tuple so
# dev builds compare exactly as before. Attached ``rc`` is handled below.
_DOTTED_SUFFIX_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)\.\S+$")
_RC_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)rc(\d+)$")
_prompt_floor_announced: str | None = None


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "rapid-mlx" / "version_check.json"


def _explicitly_disabled() -> bool:
    """Return whether every version check is explicitly disabled.

    These opt-outs also apply to the non-interactive ``serve`` log notice.
    """
    # Any value other than an explicit "off" spelling keeps opting out, as it
    # always has; only ``=0`` / ``=false`` / ... no longer disable the check.
    raw = os.environ.get("RAPID_MLX_DISABLE_VERSION_CHECK", "").strip()
    if raw and not is_falsey(raw):
        return True
    if os.environ.get("CI"):
        return True
    return False


def _disabled() -> bool:
    """Return whether the normal interactive warning should be skipped."""
    if _explicitly_disabled():
        return True
    try:
        # ``stderr.isatty()`` matches where we'd print the warning.
        return not sys.stderr.isatty()
    except Exception:  # noqa: BLE001 — stderr might be replaced
        return True


def _parse_version(
    s: str,
) -> tuple[int, int, int, int | float] | None:
    """Parse a version into a comparison-capable tuple, or None.

    Returns ``(major, minor, patch, prerelease)`` where ``prerelease`` is
    the release-candidate number for an attached ``X.Y.ZrcN`` (e.g.
    ``0.13.0rc1`` → ``(0, 13, 0, 1)``) or ``inf`` for everything else that
    parses to a clean base version — a final release (``0.13.0``) or a
    dotted-suffix build (``0.13.0.dev1`` / ``0.13.0.post1``), which keep
    their historical ``major.minor.patch``-only behavior. The
    ``prerelease`` element makes any ``X.Y.ZrcN`` sort BELOW the matching
    ``X.Y.Z`` final, so an rc user reports as behind once the final ships,
    while a `devN` build stays exactly as it always was (equal to its base
    final, no new warning / no regression). Completely malformed strings
    (``0.13``, ``abc``, attached alpha/beta/``+local``) return None — those
    releases stay silent. Stdlib only (no packaging dependency); a leading
    ``v`` is tolerated.
    """
    s = s.strip()
    m = _RC_VERSION_RE.match(s)
    if m is not None:
        return (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
        )
    # Final, or dotted-suffix (dev/post) build → base tuple + inf.
    m = _VERSION_RE.match(s) or _DOTTED_SUFFIX_RE.match(s)
    if m is not None:
        return (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            float("inf"),
        )
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
            spec = data.get("requires_python")
            if not isinstance(spec, str) or not spec or len(spec) > 200:
                data.pop("requires_python", None)
            brew = data.get("brew_stable")
            if not isinstance(brew, str) or not _CACHED_VERSION_RE.fullmatch(brew):
                data.pop("brew_stable", None)
            if not isinstance(data.get("requires_python_checked"), bool):
                data.pop("requires_python_checked", None)
            failed_at = data.get("requires_python_failed_at")
            if not isinstance(failed_at, int) or isinstance(failed_at, bool):
                data.pop("requires_python_failed_at", None)
            return data
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(
    latest: str,
    requires_python: str | None = None,
    brew_stable: str | None = None,
    ts: int | float | None = None,
    requires_python_checked: bool = True,
    requires_python_failed_at: int | None = None,
) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mtime = p.stat().st_mtime if ts is not None and p.exists() else None
        with p.open("w") as f:
            data = {
                "latest": latest,
                "ts": int(time.time()) if ts is None else ts,
            }
            if requires_python_checked:
                data["requires_python_checked"] = True
            if requires_python is not None:
                data["requires_python"] = requires_python
            if requires_python_failed_at is not None:
                data["requires_python_failed_at"] = requires_python_failed_at
            if brew_stable is not None:
                data["brew_stable"] = brew_stable
            json.dump(data, f)
        if mtime is not None:
            os.utime(p, (mtime, mtime))
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
                    versions.append((int(match[1]), int(match[2]), int(match[3])))
            if len(releases) < 100:
                return ".".join(map(str, max(versions))) if versions else None
            page += 1
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        # http.client.HTTPException covers mid-body transport failures such
        # as IncompleteRead (truncated chunked responses), which are NOT
        # OSError subclasses — without it a flaky connection crashes the CLI
        # instead of failing open (seen live: rapid-mlx upgrade --dry-run).
        http.client.HTTPException,
    ):
        return None


_REQUIRES_PYTHON_FAILED = object()


def _fetch_requires_python(version: str) -> str | None | object:
    if not _CACHED_VERSION_RE.fullmatch(version):
        return _REQUIRES_PYTHON_FAILED
    try:
        req = urllib.request.Request(
            f"https://pypi.org/pypi/rapid-mlx/{version}/json",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        info = data.get("info") if isinstance(data, dict) else None
        spec = info.get("requires_python") if isinstance(info, dict) else None
        return spec if isinstance(spec, str) and spec and len(spec) <= 200 else None
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        http.client.HTTPException,
    ):
        return _REQUIRES_PYTHON_FAILED


def _fetch_brew_formula_version() -> str | None:
    try:
        req = urllib.request.Request(
            "https://formulae.brew.sh/api/formula/rapid-mlx.json",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        versions = data.get("versions") if isinstance(data, dict) else None
        stable = versions.get("stable") if isinstance(versions, dict) else None
        return (
            stable
            if isinstance(stable, str) and _CACHED_VERSION_RE.fullmatch(stable)
            else None
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        http.client.HTTPException,
    ):
        return None


def _python_supports(
    requires_python: str | None, version_info=sys.version_info
) -> bool:
    if not requires_python:
        return True
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
    except ImportError:
        return True
    try:
        return SpecifierSet(requires_python).contains(
            ".".join(str(part) for part in version_info[:3]), prereleases=True
        )
    except InvalidSpecifier:
        return True


class ReleaseInfo:
    __slots__ = ("latest", "requires_python", "brew_stable")

    def __init__(
        self,
        latest: str,
        requires_python: str | None = None,
        brew_stable: str | None = None,
    ):
        self.latest = latest
        self.requires_python = requires_python
        self.brew_stable = brew_stable


def get_release_info(
    force_refresh: bool = False, *, need_brew: bool = False
) -> ReleaseInfo | None:
    cached = None if force_refresh else _read_cache()
    if cached is not None:
        latest = cached["latest"]
        spec = cached.get("requires_python")
        brew = cached.get("brew_stable")
        checked = cached.get("requires_python_checked", False)
        failed_at = cached.get("requires_python_failed_at")
        rewrite = False
        if (
            spec is None
            and not checked
            and (failed_at is None or time.time() - failed_at >= 3600)
        ):
            result = _fetch_requires_python(latest)
            if result is not _REQUIRES_PYTHON_FAILED:
                spec = result
                checked = True
                failed_at = None
            else:
                failed_at = int(time.time())
            rewrite = True
        if need_brew and brew is None:
            brew = _fetch_brew_formula_version()
            if brew is not None:
                rewrite = True
        if rewrite:
            ts = cached.get("ts")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                try:
                    ts = _cache_path().stat().st_mtime
                except OSError:
                    return ReleaseInfo(latest, spec, brew)
            _write_cache(
                latest,
                spec,
                brew,
                ts=ts,
                requires_python_checked=checked,
                requires_python_failed_at=failed_at,
            )
        return ReleaseInfo(latest, spec, brew)
    latest = _fetch_latest()
    if latest is None:
        return None
    result = _fetch_requires_python(latest)
    checked = result is not _REQUIRES_PYTHON_FAILED
    spec = result if isinstance(result, str) else None
    brew = _fetch_brew_formula_version() if need_brew else None
    _write_cache(
        latest,
        spec,
        brew,
        requires_python_checked=checked,
        requires_python_failed_at=None if checked else int(time.time()),
    )
    return ReleaseInfo(latest, spec, brew)


def _installed_version() -> str | None:
    try:
        return pkg_version("rapid-mlx")
    except PackageNotFoundError:
        return None


def get_latest_version(force_refresh: bool = False) -> str | None:
    """Return the latest GitHub release version, or None."""
    release = get_release_info(force_refresh)
    return release.latest if release else None


def staleness_warning(*, allow_non_tty: bool = False) -> str | None:
    """Return a one-line warning string if the installed version is behind
    the latest release by ANY amount (patch, minor, or major). Returns
    None when no warning is warranted (current/ahead, or check disabled).
    """
    versions = _staleness_warning_versions(allow_non_tty=allow_non_tty)
    return _format_available_warning(*versions) if versions else None


def _staleness_warning_versions(*, allow_non_tty: bool) -> tuple[str, str] | None:
    disabled = _explicitly_disabled() if allow_non_tty else _disabled()
    if disabled:
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

    try:
        info = detect_install_method()
        release = get_release_info(need_brew=info.method == "brew")
        if release and release.latest == latest_str:
            block = _upgrade_block(installed_str, release, info)
            if block == "brew_lag":
                return None
            if info.method == "brew" and release.brew_stable:
                latest_str = release.brew_stable
    except Exception:  # noqa: BLE001 — version notices fail open
        pass
    return installed_str, latest_str


def format_staleness_warning(installed: str, latest: str) -> str:
    """Format the shared one-line CLI and server update notice."""
    return (
        f"⚠ rapid-mlx {installed} is behind latest {latest} — "
        f"run `rapid-mlx upgrade` to pick up new model aliases / flags."
    )


def print_staleness_warning_if_any(*, allow_non_tty: bool = False) -> str | None:
    """Best-effort: print to stderr and return the announced version, else None.

    Always silent on errors.

    ``allow_non_tty`` is reserved for ``serve`` startup logs. It keeps the
    explicit environment/CI opt-outs but does not require stderr to be a TTY.
    """
    try:
        versions = _staleness_warning_versions(allow_non_tty=allow_non_tty)
        if versions:
            if versions[1] == _prompt_floor_announced:
                return versions[1]
            print(_format_available_warning(*versions), file=sys.stderr)
            return versions[1]
    except Exception:  # noqa: BLE001 — never break the CLI
        pass
    return None


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
    global _prompt_floor_announced
    try:
        if _disabled():
            return False
        # Need stdin for the prompt too — _disabled checks stderr only.
        if not sys.stdin.isatty():
            return False

        installed_str = _installed_version()
        if not installed_str:
            return False
        # Skip pre-release / dev / local-version builds entirely, before
        # even parsing. For an interactive prompt the dev-base case can
        # fire a false "newer release available" against the dev's own
        # in-progress branch (installed ``0.6.61.dev1`` vs latest
        # ``0.6.62`` → prompt). A clean PEP 440 final release is digits
        # and dots only; anything else (``dev``/``a``/``b``/``rc``/
        # ``post``/``+local``) is non-final and gets the dev-build skip
        # path. This keeps ``rc``/``dev`` builds silent here even though
        # ``_parse_version`` now parses ``X.Y.ZrcN`` for the staleness
        # banner. DeepSeek finding #1 on PR #428.
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

        from rapid_mlx._upgrade_runner import run_upgrade_command

        info = detect_install_method()
        release = get_release_info(need_brew=info.method == "brew")
        if release and release.latest == latest_str:
            block = _upgrade_block(installed_str, release, info)
            if block == "python":
                assert release.requires_python is not None
                if _prompt_floor_announced != latest_str:
                    print(
                        format_python_floor_warning(
                            installed_str, latest_str, release.requires_python, info
                        ),
                        file=sys.stderr,
                    )
                    _prompt_floor_announced = latest_str
                return False
            if block == "brew_lag":
                return False
            if info.method == "brew" and release.brew_stable:
                latest_str = release.brew_stable
        print(
            f"\nA newer rapid-mlx is available: {latest_str} "
            f"(current: {installed_str})."
        )
        print(f"  Upgrade command: {info.upgrade_command}")
        if info.method == "unknown":
            print("  Automatic upgrade unavailable; run that command manually.\n")
            return False
        try:
            answer = input("  Run it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer and answer not in ("y", "yes"):
            return False

        print()
        try:
            returncode = run_upgrade_command(info.upgrade_argv)
        except FileNotFoundError as e:
            print(f"  Upgrade command not found: {e}\n")
            return False
        except KeyboardInterrupt:
            print("\n  Interrupted.\n")
            return False
        if returncode == 0:
            new = installed_version_in_fresh_process(info)
            if new is not None and _version_satisfies_target(new, latest_str):
                print(
                    f"\nUpgrade complete: rapid-mlx {new}. Please re-run your command.\n"
                )
                return True
            if new is None:
                print(
                    f"\nThe upgrade command finished, but the installed version could not be verified; continuing with rapid-mlx {installed_str}. Check with `rapid-mlx --version`.\n"
                )
            else:
                print(
                    f"\nThe upgrade command finished, but rapid-mlx is still {new} (expected {latest_str}); continuing with rapid-mlx {installed_str}.\n"
                )
            return False
        # Failed upgrade: return False so the caller continues booting with
        # the current version rather than exiting silently. The user has
        # been shown the exit code and can retry manually if they care.
        # DeepSeek finding #2 on PR #428: returning True here would have
        # `serve_command` do ``sys.exit(0)`` and leave the user with no
        # running server and only an error message.
        print(
            f"\nUpgrade exited with code {returncode}; "
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
        self.method = method  # brew, uv, pipx, pip, install_sh, or unknown
        self.upgrade_command = upgrade_command
        self.upgrade_argv = upgrade_argv
        self.binary_path = binary_path


def install_method_label(info: InstallInfo) -> str:
    if info.method == "pip" and "--user" in info.upgrade_argv:
        return "pip --user"
    return {
        "brew": "Homebrew",
        "install_sh": "one-line installer",
        "uv": "uv tool",
        "pipx": "pipx",
        "pip": "pip",
    }.get(info.method, "unknown")


def python_upgrade_instructions(info: InstallInfo, requires_python: str) -> list[str]:
    bounds = re.findall(r"(?:^|,)\s*(>=|>|==)\s*(\d+)\.(\d+)(\.\*)?", requires_python)
    versions = [
        (int(major), int(minor) + (op == ">"))
        for op, major, minor, wildcard in bounds
        if op != "==" or wildcard
    ]
    minor = ".".join(map(str, max(versions))) if versions else "3.12"
    try:
        from packaging.specifiers import SpecifierSet

        spec = SpecifierSet(requires_python)
        for candidate in range(10, 31):
            if any(
                spec.contains(f"3.{candidate}.{patch}", prereleases=False)
                for patch in range(31)
            ):
                minor = f"3.{candidate}"
                break
    except (ImportError, ValueError):
        pass
    installer = "curl -fsSL https://rapidmlx.com/install.sh | bash"
    if info.method == "brew":
        return ["brew upgrade rapid-mlx"]
    if info.method == "install_sh":
        return [
            installer,
            f"It rebuilds ~/.rapid-mlx on Python {minor}; your settings and downloaded models are kept.",
        ]
    if info.method == "uv":
        return [f"uv tool upgrade --python {minor} rapid-mlx"]
    if info.method == "pipx":
        return [
            f"Install Python {minor} first, e.g. `brew install python@{minor}`",
            f"pipx reinstall --python python{minor} rapid-mlx",
        ]
    if info.method == "unknown" and info.upgrade_command.startswith("sudo pipx "):
        return [f"sudo pipx reinstall --global --python python{minor} rapid-mlx"]
    venv = f"~/.venvs/rapid-mlx-py{minor.replace('.', '')}"
    return [
        f"Install Python {minor} or newer (e.g. `brew install python@{minor}`), then reinstall Rapid-MLX with it, e.g. in a new environment:",
        f"python{minor} -m venv {venv} && {venv}/bin/pip install rapid-mlx",
        f"Add {venv}/bin to PATH (or call it directly); remove the old environment only after the new one works.",
        f"or use the one-line installer: `{installer}`",
    ]


def _upgrade_block(
    installed: str, release: ReleaseInfo, info: InstallInfo
) -> str | None:
    if info.method != "brew" and not _python_supports(release.requires_python):
        return "python"
    if info.method == "brew" and release.brew_stable:
        current, stable = _parse_version(installed), _parse_version(release.brew_stable)
        if current is not None and stable is not None and stable <= current:
            return "brew_lag"
    return None


def format_python_floor_warning(
    installed: str, latest: str, requires_python: str, info: InstallInfo
) -> str:
    del installed
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines = [
        f"⚠ rapid-mlx {latest} requires Python {requires_python}; this install runs Python {version}, so `rapid-mlx upgrade` cannot move it."
    ]
    lines.extend(
        f"  {line}" for line in python_upgrade_instructions(info, requires_python)
    )
    return "\n".join(lines)


def _format_available_warning(installed: str, latest: str) -> str:
    try:
        info = detect_install_method()
        release = get_release_info(need_brew=info.method == "brew")
        if (
            release
            and release.latest == latest
            and _upgrade_block(installed, release, info) == "python"
        ):
            assert release.requires_python is not None
            return format_python_floor_warning(
                installed, latest, release.requires_python, info
            )
    except Exception:  # noqa: BLE001 — version notices fail open
        pass
    return format_staleness_warning(installed, latest)


def _normalize_version(version: str) -> tuple[tuple[int, ...], str] | None:
    match = re.match(r"(\d+(?:\.\d+)*)(.*)", version.strip().lower().removeprefix("v"))
    if match is None:
        return None
    release = [int(component) for component in match[1].split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    aliases = {
        "alpha": "a",
        "beta": "b",
        "c": "rc",
        "pre": "rc",
        "preview": "rc",
    }
    suffix = re.sub(
        r"^[._-]?(alpha|beta|preview|pre|rc|dev|post|a|b|c)",
        lambda label: aliases.get(label[1], label[1]),
        match[2],
    )
    return tuple(release), suffix


def _versions_equal(installed: str, target: str) -> bool:
    """Compare exact releases without dropping prerelease or build suffixes."""
    try:
        from packaging.version import Version
    except ImportError:
        normalized = _normalize_version(installed)
        return normalized is not None and normalized == _normalize_version(target)
    try:
        return bool(Version(installed) == Version(target))
    except ValueError:
        return False


def _version_satisfies_target(installed: str, target: str) -> bool:
    """Accept the target or a newer release appropriate for its release channel."""
    try:
        from packaging.version import Version
    except ImportError:
        if _versions_equal(installed, target):
            return True
        normalized_installed = _normalize_version(installed)
        normalized_target = _normalize_version(target)
        return bool(
            normalized_installed is not None
            and normalized_target is not None
            and not normalized_installed[1]
            and normalized_installed[0] > normalized_target[0]
        )
    try:
        installed_version = Version(installed)
        target_version = Version(target)
    except ValueError:
        return False
    return bool(
        installed_version == target_version
        or (
            installed_version > target_version
            and (target_version.is_prerelease or not installed_version.is_prerelease)
        )
    )


def installed_version_in_fresh_process(info: InstallInfo) -> str | None:
    import subprocess

    if (
        info.binary_path
        and os.access(info.binary_path, os.X_OK)
        and _launcher_belongs_to_running_install(info.binary_path)
    ):
        argv = [info.binary_path, "--version"]
        pattern = r"\brapid-mlx\s+(\d+(?:\.\d+)*\S*)"
    else:
        import site

        extra_paths = []
        if sys.prefix == sys.base_prefix and site.ENABLE_USER_SITE:
            extra_paths.append(site.getusersitepackages())
        argv = [
            sys.executable,
            "-I",
            "-c",
            "import sys, importlib.metadata as m; "
            "d = next(m.distributions(name='rapid-mlx', path=sys.argv[1:] + sys.path), None); "
            "print(d.version if d else '')",
            *extra_paths,
        ]
        pattern = r"^\s*(\S+)\s*$"
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode:
            return None
        matches = re.findall(pattern, result.stdout, re.MULTILINE)
        version = matches[-1] if matches else None
        return version if version and _versions_equal(version, version) else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def launcher_interpreter(first_line: str, second_line: str) -> str | None:
    """Return the interpreter declared by a console-script launcher."""
    if not first_line.startswith("#!"):
        return None
    if re.fullmatch(r"#!/bin/sh(?:\s+-)?\s*", first_line):
        match = re.match(
            r"^'''exec' (?:'([^']+)'|\"([^\"]+)\"|(\S+)) \"\$0\" \"\$@\"",
            second_line,
        )
        if match and match.group(2) and any(c in match.group(2) for c in "$`\\"):
            return None
        return (
            next((group for group in match.groups() if group), None) if match else None
        )
    parts = first_line[2:].strip().split(maxsplit=1)
    return parts[0] if parts else None


def _python_environment(path: str) -> str:
    """Identify the venv containing an interpreter, or its resolved executable."""
    try:
        current = os.path.abspath(path)
        for _ in range(40):
            directory = os.path.realpath(os.path.dirname(current))
            root = os.path.dirname(directory)
            if os.path.isfile(os.path.join(root, "pyvenv.cfg")):
                return os.path.normcase(root)
            if not os.path.islink(current):
                break
            target = os.readlink(current)
            if not os.path.isabs(target):
                target = os.path.join(os.path.dirname(current), target)
            current = os.path.abspath(target)
    except OSError:
        pass
    return os.path.normcase(os.path.realpath(path))


def _launcher_belongs_to_running_install(binary: str) -> bool:
    """Prove that a PATH launcher is owned by this interpreter's install.

    A same-name launcher elsewhere on PATH is not evidence about the imported
    distribution. The launcher must resolve to the active interpreter even
    when its lexical path is inside that interpreter's scripts directory: a
    stale symlink there may still target a Homebrew or pipx installation.
    """
    try:
        dist = distribution("rapid-mlx")
        if not any(entry.name == "rapid-mlx" for entry in dist.entry_points):
            return False
    except (PackageNotFoundError, OSError, ValueError):
        return False

    resolved = Path(os.path.realpath(binary))
    try:
        with resolved.open("rb") as launcher:
            first_line = launcher.readline(4096).decode("utf-8", "replace")
            second_line = launcher.readline(4096).decode("utf-8", "replace")
    except OSError:
        return False
    shebang = launcher_interpreter(first_line, second_line)
    if not shebang:
        return False
    return _python_environment(shebang) == _python_environment(sys.executable)


def _externally_managed() -> bool:
    try:
        return (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").is_file()
    except OSError:
        return False


def detect_install_method() -> InstallInfo:
    """Detect how rapid-mlx was installed and return the right upgrade command.

    Detection order:
      1. brew — ``rapid-mlx`` realpath under ``/Cellar/rapid-mlx``,
         ``/opt/homebrew/`` (macOS) or ``/home/linuxbrew/`` (Linux brew)
         triggers ``brew upgrade rapid-mlx`` (now in homebrew/core).
      2. install.sh — realpath under the install.sh venv at
         ``~/.rapid-mlx/`` upgrades via that environment's Python/pip.
      3. uv / pipx — realpath in that tool manager's isolated environment
         triggers its native upgrade command.
      4. pip (default) — uses ``sys.executable -m pip install --upgrade``
         so the upgrade lands in the same env that's currently running
         the CLI.
    """
    import importlib.util
    import shlex
    import shutil

    path_binary = shutil.which("rapid-mlx")
    binary = (
        path_binary
        if path_binary is not None and _launcher_belongs_to_running_install(path_binary)
        else None
    )
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
        # install.sh creates ``~/.rapid-mlx`` (venv) and symlinks its entry
        # point into ``~/.local/bin``. uv, pipx, and ``pip install --user``
        # use that SAME bin directory, so the launcher path is not evidence of
        # ownership. Only the resolved target identifies an install.sh venv.
        rapid_mlx_dir = os.path.normcase(os.path.realpath(Path.home() / ".rapid-mlx"))
        try:
            is_install_sh = (
                os.path.commonpath([os.path.normcase(normalized), rapid_mlx_dir])
                == rapid_mlx_dir
            )
        except ValueError:
            is_install_sh = False
        if is_install_sh:
            # Upgrade the existing environment through its package manager.
            # Never fetch and execute mutable website shell scripts here.
            argv = [sys.executable, "-m", "pip", "install", "--upgrade", "rapid-mlx"]
            if importlib.util.find_spec("pip") is None and (uv := shutil.which("uv")):
                argv = [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "--upgrade",
                    "rapid-mlx",
                ]
            return InstallInfo(
                method="install_sh",
                upgrade_command=shlex.join(argv),
                upgrade_argv=argv,
                binary_path=binary,
            )

        # uv and pipx both resolve their shared-bin launcher into a manager-owned
        # venv. Require three signals before dispatching a manager command:
        # known/configured root, exact ``rapid-mlx/bin/rapid-mlx`` target, and
        # that manager's metadata receipt. A coincidentally named ordinary venv
        # must fall back to the running interpreter instead.
        def _is_managed_entry(venv: Path, receipt: str) -> bool:
            expected = venv / "bin" / "rapid-mlx"
            return (
                os.path.normcase(normalized)
                == os.path.normcase(os.path.realpath(expected))
                and (venv / receipt).is_file()
            )

        uv_tool_dir = os.environ.get("UV_TOOL_DIR")
        if uv_tool_dir:
            uv_roots = [Path(uv_tool_dir).expanduser()]
        else:
            data_home = os.environ.get("XDG_DATA_HOME")
            uv_data = (
                Path(data_home).expanduser()
                if data_home
                else Path.home() / ".local/share"
            )
            uv_roots = [uv_data / "uv" / "tools"]
        if any(
            _is_managed_entry(root / "rapid-mlx", "uv-receipt.toml")
            for root in uv_roots
        ):
            return InstallInfo(
                method="uv",
                upgrade_command="uv tool upgrade rapid-mlx",
                upgrade_argv=["uv", "tool", "upgrade", "rapid-mlx"],
                binary_path=binary,
            )

        global_pipx_home = Path(
            os.environ.get("PIPX_GLOBAL_HOME", "/opt/pipx")
        ).expanduser()
        if _is_managed_entry(
            global_pipx_home / "venvs" / "rapid-mlx", "pipx_metadata.json"
        ):
            return InstallInfo(
                method="unknown",
                upgrade_command="sudo pipx upgrade --global rapid-mlx",
                upgrade_argv=[],
                binary_path=binary,
            )

        pipx_home = os.environ.get("PIPX_HOME")
        if pipx_home:
            pipx_homes = [Path(pipx_home).expanduser()]
        else:
            data_home = os.environ.get("XDG_DATA_HOME")
            pipx_homes = [
                Path.home() / ".local/pipx",
                Path.home() / ".local/share/pipx",
            ]
            if data_home:
                pipx_homes.insert(0, Path(data_home).expanduser() / "pipx")
            if sys.platform == "darwin":
                pipx_homes.insert(0, Path.home() / "Library/Application Support/pipx")
        if any(
            _is_managed_entry(home / "venvs" / "rapid-mlx", "pipx_metadata.json")
            for home in pipx_homes
        ):
            return InstallInfo(
                method="pipx",
                upgrade_command="pipx upgrade rapid-mlx",
                upgrade_argv=["pipx", "upgrade", "rapid-mlx"],
                binary_path=binary,
            )

    import site

    import rapid_mlx

    user_site = Path(os.path.realpath(site.getusersitepackages()))
    package_file = Path(os.path.realpath(rapid_mlx.__file__))
    is_user = sys.prefix == sys.base_prefix and user_site in package_file.parents
    argv = [sys.executable, "-m", "pip", "install"]
    if is_user and _externally_managed():
        argv.append("--break-system-packages")
    if is_user:
        argv.append("--user")
    argv.extend(["--upgrade", "rapid-mlx"])
    return InstallInfo(
        method="pip",
        upgrade_command=shlex.join(argv),
        upgrade_argv=argv,
        binary_path=binary,
    )
