#!/usr/bin/env python3
"""The engine and the desktop app ship ONE version number. Prove it.

This monorepo carried two hand-maintained versions with nothing between
them:

* ``pyproject.toml`` ``version`` — the engine. Guarded by
  ``version-check.yml``: only a ``chore: bump version to X.Y.Z`` PR may
  touch it, and merging one publishes to PyPI and tags ``vX.Y.Z``.
* ``apps/rapid-mac/Resources/Info.plist``
  ``CFBundleShortVersionString`` — the desktop app. Edited by hand in
  whatever PR happened to cut a desktop build, and tagged
  ``rapid-mac-vX.Y.Z``.

Nothing compared them, so they drifted, and on 2026-08-07 the drift
reached users: the engine was 0.12.5 while the app was 0.12.6, both
called "the latest release", and rapidmlx.com's install command
(``bash -s 0.12.5``, correct for the engine) read as a stale website to
anyone who had just seen 0.12.6 on GitHub. Every downstream consumer
inherits the ambiguity — the ``/api/desktop-*`` feeds, the changelog, the
release notes, and any user asked "which version are you on?".

``release-local.sh`` already refuses a ``rapid-mac-vX.Y.Z`` tag that
disagrees with the plist. With this check the chain closes:

    git tag  ==  Info.plist  ==  pyproject.toml

Run by ``ci.yml`` on every PR (stdlib only — no mlx, no macOS, so it runs
on the Linux lint runner) and pinned by ``tests/test_version_sync.py``.

SCOPE: this check enforces that the two numbers AGREE. It does not
enforce that they increase — ``version-check.yml`` already does that,
rejecting any ``pyproject.toml`` version that is not strictly greater
than the base branch's. The two compose: a PR that lowers both files in
step is refused by ``version-check.yml``, and a PR that lowers only the
plist is refused here as a mismatch. Re-deriving monotonicity from this
script would mean reading git history from a stdlib-only linter, to
duplicate a rule that is already enforced one layer up.

Exit status: 0 in sync, 1 out of sync or unreadable. "Unreadable" is a
FAILURE, not a skip: a guard that passes when it cannot find its inputs
is indistinguishable from no guard at all, and this one has to survive a
file move.
"""

from __future__ import annotations

import plistlib
import re
import sys
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 and older
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INFO_PLIST = ROOT / "apps" / "rapid-mac" / "Resources" / "Info.plist"

# Both numbers feed release tags and an updater that compares them, so a
# value neither side can order (``0.12``, ``1.0.0-rc1``, an empty string)
# is a defect wherever it appears, not merely a mismatch.
#
# Matched with ``fullmatch`` and an explicit ASCII class, not ``^…$`` and
# ``\d``. ``$`` matches before a trailing newline, so ``"0.12.6\n"`` —
# entirely plausible from a hand-edited ``<string>`` element — would pass
# and then build the tag ``v0.12.6\n``. ``\d`` additionally accepts
# non-ASCII digits, which no release tag can carry.
#
# Leading zeros are refused for a sharper reason than pedantry: PEP 440
# NORMALISES them, so ``01.02.3`` in pyproject.toml publishes to PyPI as
# ``1.2.3`` while the plist and the git tag keep ``01.02.3``. Both files
# would agree and this check would pass, having produced exactly the
# split version it exists to prevent.
_NUM = r"(?:0|[1-9][0-9]*)"
SEMVER = re.compile(rf"{_NUM}\.{_NUM}\.{_NUM}")


class VersionSyncError(Exception):
    """Raised with a message written for whoever has to fix it."""


def engine_version(pyproject: Path = PYPROJECT) -> str:
    """``[project] version`` from ``pyproject.toml``."""
    if not pyproject.is_file():
        raise VersionSyncError(f"{_rel(pyproject)} not found")
    # OSError as well as the parse errors: a PermissionError here would
    # otherwise escape ``main``'s handler as a bare traceback, which is
    # exactly the "unreadable silently isn't a failure" hole this script
    # exists to close.
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        raise VersionSyncError(
            f"{_rel(pyproject)} is not readable TOML: {exc}"
        ) from exc
    # ``[project]`` is only a table by convention. ``project = "x"`` is
    # valid TOML, and ``.get`` on the str would be an AttributeError that
    # never reaches the VersionSyncError path.
    project = data.get("project")
    if not isinstance(project, dict):
        raise VersionSyncError(f"{_rel(pyproject)} has no [project] version")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise VersionSyncError(f"{_rel(pyproject)} has no [project] version")
    if not SEMVER.fullmatch(version):
        raise VersionSyncError(
            f"{_rel(pyproject)} [project] version is {version!r}, "
            "which is not X.Y.Z — release tags are built from it"
        )
    return version


def app_version(info_plist: Path = INFO_PLIST) -> str:
    """``CFBundleShortVersionString`` from the app's ``Info.plist``.

    ``plistlib`` rather than ``PlistBuddy`` so this runs on the Linux
    lint runner; the file is XML, which ``plistlib`` reads anywhere.

    ``CFBundleVersion`` (a monotonic build counter) is deliberately NOT
    checked. It is not a release version, nobody quotes it, and tying it
    to the engine would force a meaningless edit on every engine patch.
    Its value is deliberately not quoted here either — a literal in a
    docstring is a second copy of a number that nothing verifies, which
    is the same class of drift this script exists to prevent.
    """
    if not info_plist.is_file():
        raise VersionSyncError(f"{_rel(info_plist)} not found")
    try:
        with info_plist.open("rb") as fh:
            data = plistlib.load(fh)
    except (plistlib.InvalidFileException, ValueError, OSError) as exc:
        raise VersionSyncError(
            f"{_rel(info_plist)} is not a readable plist: {exc}"
        ) from exc
    # A plist's root need not be a dict — ``<plist><array/></plist>``
    # parses fine and would AttributeError on ``.get``.
    if not isinstance(data, dict):
        raise VersionSyncError(f"{_rel(info_plist)} has no CFBundleShortVersionString")
    version = data.get("CFBundleShortVersionString")
    if not isinstance(version, str) or not version:
        raise VersionSyncError(f"{_rel(info_plist)} has no CFBundleShortVersionString")
    if not SEMVER.fullmatch(version):
        raise VersionSyncError(
            f"{_rel(info_plist)} CFBundleShortVersionString is {version!r}, "
            "which is not X.Y.Z — the rapid-mac-v tag is built from it"
        )
    return version


def check(
    pyproject: Path = PYPROJECT, info_plist: Path = INFO_PLIST
) -> tuple[str, str]:
    """Return the shared version, or raise ``VersionSyncError``."""
    engine = engine_version(pyproject)
    app = app_version(info_plist)
    if engine != app:
        raise VersionSyncError(
            f"engine and desktop app disagree about the version:\n"
            f"  {_rel(pyproject)}  [project] version           = {engine}\n"
            f"  {_rel(info_plist)}  CFBundleShortVersionString = {app}\n"
            f"\n"
            f"They ship as one product and must carry one number. Set both\n"
            f"to the same X.Y.Z in the same PR, and move the number UP — a\n"
            f"rapid-mac-vX.Y.Z tag already exists for every version the app\n"
            f"has shipped, and the in-app updater orders these values.\n"
            f"(The increase itself is enforced by version-check.yml; this\n"
            f"check only enforces that the two files agree.)"
        )
    return engine, app


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    try:
        version, _ = check()
    except VersionSyncError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f"engine and desktop app agree: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
