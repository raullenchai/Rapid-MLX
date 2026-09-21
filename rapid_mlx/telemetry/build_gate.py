# SPDX-License-Identifier: Apache-2.0
"""Official-build gate — telemetry transmits ONLY from official releases.

Telemetry v2 is default-on, which makes the question "is this process an
official release build?" load-bearing: a developer checkout, an editable
install, a CI machine, or a fork's rebuild must never send events, or
contributors' machines pollute product analytics and people who never
saw the consent disclosure get tracked. Mirroring Orca's
``IS_OFFICIAL_BUILD``, this module answers that question and nothing else.

Two INDEPENDENT conditions must hold for :func:`official_build` to return
a stamp; both are needed because each one alone is forgeable by accident:

1. A release stamp exists. ``_release_stamp.json`` is written ONLY by the
   release workflow into ``rapid_mlx/telemetry/`` at publish time and is
   never committed — so a checkout cannot have one. But the stamp alone
   is not sufficient: an official sdist contains everything the wheel
   does, so ``pip install -e`` from an unpacked official sdist would sit
   on a developer's machine with a perfectly valid stamp. (A committed
   flag or a version-string heuristic is even worse: a fork's build or a
   repacked wheel would inherit both.)
2. The install is NOT editable/source-like: the PEP 610
   ``direct_url.json`` of the ``rapid-mlx`` distribution must not say
   ``dir_info.editable == true``, AND the package directory must not live
   inside a git checkout (``.git`` as a directory for clones, as a file
   for worktrees). But this check alone is not sufficient either: a
   locally built wheel ``pip install``ed into a throwaway venv is
   neither editable nor in a checkout, yet it is not an official build —
   only the release workflow's stamp makes it one.

**Failure policy.** Every public function fails closed and never raises:
a missing, unreadable, or malformed stamp, unreadable distribution
metadata, or a hostile filesystem all resolve to "not an official
build" (``None`` / ``True``). Unknown provenance does not transmit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import distribution
from pathlib import Path

#: The stamp file ONLY the release workflow writes into ``rapid_mlx/telemetry/``
#: at publish time. Never committed (see ``.gitignore``); a later PR wires the
#: writing. JSON shape: ``{"channel": "stable" | "rc", "posthog_key": "phc_…"}``.
RELEASE_STAMP_NAME = "_release_stamp.json"

#: Distribution name in ``pyproject.toml`` ([project] name).
_DISTRIBUTION_NAME = "rapid-mlx"

#: PostHog project keys look like ``phc_<20..80 alphanumerics>``. Anything
#: else in the stamp means the file was tampered with or truncated.
_POSTHOG_KEY_RE = re.compile(r"^phc_[A-Za-z0-9]{20,80}$")

#: Release channels eligible to transmit. Nothing else parses.
_VALID_CHANNELS = frozenset({"stable", "rc"})

#: Levels to walk up from the package dir looking for a ``.git`` entry.
#: The package sits ~2-4 levels below a checkout root in every layout we
#: ship (repo, wheel unpack, site-packages); 12 leaves generous headroom
#: while bounding the stat calls on deep, alien directory trees.
_GIT_WALK_MAX_LEVELS = 12


@dataclass(frozen=True)
class ReleaseStamp:
    """A valid release stamp: what channel this build ships on, and the key."""

    channel: str
    posthog_key: str


def _stamp_path() -> Path:
    """Locate the stamp relative to this module.

    A private function only so tests can monkeypatch the location; the
    release workflow writes next to this file in the installed package.
    """
    return Path(__file__).with_name(RELEASE_STAMP_NAME)


def _parse_stamp(raw: str) -> ReleaseStamp | None:
    """Validate stamp text; ``None`` for anything but the exact shape."""
    try:
        obj = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        # Arrays, strings, numbers — a stamp is a JSON object, nothing else.
        return None
    channel = obj.get("channel")
    key = obj.get("posthog_key")
    if not isinstance(channel, str) or channel not in _VALID_CHANNELS:
        return None
    if not isinstance(key, str) or _POSTHOG_KEY_RE.fullmatch(key) is None:
        return None
    return ReleaseStamp(channel=channel, posthog_key=key)


def read_release_stamp() -> ReleaseStamp | None:
    """Read and validate ``_release_stamp.json``; ``None`` if not official.

    Returns ``None`` when the file is missing, unreadable, not JSON, not a
    JSON object, or its fields fail validation. Never raises.
    """
    try:
        raw = _stamp_path().read_text(encoding="utf-8")
    except Exception:
        # Missing, permission-denied, a directory, undecodable bytes —
        # every failure to READ is a failure to transmit.
        return None
    return _parse_stamp(raw)


def _direct_url_says_editable() -> bool | None:
    """PEP 610 verdict: editable install? ``None`` when unknowable.

    ``None`` means the distribution metadata itself could not be read —
    the caller must fail closed. A readable distribution without
    ``direct_url.json`` (the normal PyPI-wheel case) is simply "not
    editable", i.e. ``False``.
    """
    try:
        dist = distribution(_DISTRIBUTION_NAME)
        raw = dist.read_text("direct_url.json")
    except Exception:
        # PackageNotFoundError and any other metadata-read failure:
        # unknown provenance does not transmit.
        return None
    if raw is None:
        return False
    try:
        obj = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(obj, dict):
        return False
    dir_info = obj.get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return dir_info.get("editable") is True


def _inside_git_checkout(start: Path) -> bool:
    """Walk up from *start* looking for a ``.git`` entry (file OR directory).

    ``.git`` is a directory in a plain clone and a file in a git
    worktree; ``exists()`` covers both. Bounded to
    ``_GIT_WALK_MAX_LEVELS`` parents so a package dropped somewhere
    pathological cannot make this walk the filesystem root-to-leaf.
    """
    current = start
    for _ in range(_GIT_WALK_MAX_LEVELS):
        if (current / ".git").exists():
            return True
        parent = current.parent
        if parent == current:
            # Reached the filesystem root without finding a checkout.
            return False
        current = parent
    return False


def is_editable_or_source_install() -> bool:
    """True when this install looks editable or source-like (must not send).

    True when the ``rapid-mlx`` distribution is a PEP 610 editable
    install, when the package directory lives inside a git checkout, or
    when provenance cannot be determined at all (metadata unreadable).
    Never raises.
    """
    if _direct_url_says_editable() is not False:
        # ``True`` = editable; ``None`` = unknown provenance. Both fail closed.
        return True
    try:
        return _inside_git_checkout(Path(__file__).parent)
    except Exception:
        # Hostile filesystem (e.g. unreadable parent directories): the
        # checkout question could not be answered, so do not transmit.
        return True


@lru_cache(maxsize=1)
def official_build() -> ReleaseStamp | None:
    """The release stamp iff this is an official build, else ``None``.

    The single decision point telemetry v2 transmits behind: an event
    leaves the machine only when a valid stamp exists AND the install is
    neither editable nor source-like. Cached per process — the answers
    cannot change mid-run.
    """
    stamp = read_release_stamp()
    if stamp is None:
        return None
    if is_editable_or_source_install():
        return None
    return stamp


def _reset_for_tests() -> None:
    """Clear the process-wide cache so tests re-evaluate from scratch."""
    official_build.cache_clear()
