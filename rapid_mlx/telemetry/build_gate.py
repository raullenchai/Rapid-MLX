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
   release workflow into ``rapid_mlx/telemetry/`` — next to this module,
   inside the installed package — at publish time and is never committed,
   so a checkout cannot have one and a user cannot forge one in the
   current directory. But the stamp alone is not sufficient: an official
   sdist contains everything the wheel does, so ``pip install -e`` from
   an unpacked official sdist would sit on a developer's machine with a
   perfectly valid stamp. (A committed flag or a version-string heuristic
   is even worse: a fork's build or a repacked wheel would inherit both.)
2. The install is NOT editable/source-like. The PEP 610 verdict comes
   from the distribution BOUND TO THE RUNNING CODE: importlib.metadata
   resolves by name through ``sys.path`` order, so a stale
   ``rapid_mlx-*.dist-info`` with ``dir_info.editable: true`` earlier on
   the path could otherwise describe an install we are not — and silence
   a genuine official build forever. A distribution only counts when
   ``locate_file("rapid_mlx")`` resolves to the running package
   directory AND its name is PEP 503-equivalent to ours. The verdict
   must not say ``dir_info.editable == true``, AND
   the ``rapid_mlx`` package directory must not sit beside this project's
   own ``pyproject.toml``. In a source checkout, an unpacked sdist, or an
   editable install, ``rapid_mlx/`` is directly beside
   ``pyproject.toml``; in site-packages it never is. An earlier draft
   walked up looking for ANY ``.git`` entry instead, and that rule was
   rejected because it misclassifies real official installs that live
   inside unrelated git checkouts: ``/opt/homebrew`` itself is a git
   checkout on Apple Silicon (with the package ~10 levels below it in
   ``Cellar/rapid-mlx/<v>/libexec/...``), ``~/.pyenv`` is one, and the
   most common pip layout of all is a project venv
   (``~/code/myproject/.venv/lib/python3.x/site-packages/``) inside the
   user's own repo. Identifying OUR tree — the package's parent holding
   a ``pyproject.toml`` whose ``[project]`` name is ``rapid-mlx`` —
   matches exactly the layouts that must stay silent and never matches
   site-packages. But this check alone is not sufficient either: a
   locally built wheel ``pip install``ed into a throwaway venv is
   neither editable nor in a source tree, yet it is not an official
   build — only the release workflow's stamp makes it one.

**Failure policy.** Every public function fails closed and never raises:
a missing, unreadable, or malformed stamp, no distribution bound to the
running package, metadata that cannot be read, a ``direct_url.json`` that
is PRESENT but unparseable or not a JSON object, an unreadable
``pyproject.toml``, or a hostile filesystem all resolve to "not an
official build" (``None`` / ``True``). Unknown provenance does not
transmit. Only provenance that is positively known is allowed to read as
"not editable / not source": an ABSENT ``direct_url.json`` (the normal
PyPI/wheel case — pip writes the file only for direct-URL, archive, VCS,
and editable installs) and a valid ``direct_url.json`` object without an
editable ``dir_info`` (a direct-URL install is not an editable one).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import Distribution, distributions
from pathlib import Path

#: The stamp file ONLY the release workflow writes into ``rapid_mlx/telemetry/``
#: at publish time. Never committed (see ``.gitignore``); a later PR wires the
#: writing. JSON shape: ``{"channel": "stable" | "rc", "posthog_key": "phc_…"}``.
RELEASE_STAMP_NAME = "_release_stamp.json"

#: Distribution name in ``pyproject.toml`` ([project] name) and in
#: installed metadata — the single name both checks agree on.
_DISTRIBUTION_NAME = "rapid-mlx"


def _pep503_name(name: str) -> str:
    """PEP 503 name normalization: case-fold, collapse ``-``/``_``/``.`` runs."""
    return re.sub(r"[-_.]+", "-", name).lower()


#: ``_DISTRIBUTION_NAME`` is already PEP 503-normalized; precompute so the
#: scan loop does not re-normalize the constant for every distribution.
_DISTRIBUTION_NAME_NORM = _pep503_name(_DISTRIBUTION_NAME)

#: Top-level package directory this module ships in — the argument to
#: ``Distribution.locate_file`` when binding metadata to running code.
_PACKAGE_DIRNAME = "rapid_mlx"

#: PostHog project keys look like ``phc_<20..80 alphanumerics>``. Anything
#: else in the stamp means the file was tampered with or truncated.
_POSTHOG_KEY_RE = re.compile(r"^phc_[A-Za-z0-9]{20,80}$")

#: Release channels eligible to transmit. Nothing else parses.
_VALID_CHANNELS = frozenset({"stable", "rc"})

#: Matches OUR ``[project]`` name assignment in ``pyproject.toml``:
#: ``name = "rapid-mlx"``, tolerant of spacing, quote style, leading
#: indentation, and an optional trailing comment. Anchored to whole lines
#: so ``{name = "Rapid-MLX contributors"}`` inside the authors array (or
#: a like-named package) cannot match. pyproject.toml is read as text
#: rather than parsed with tomllib because this repo supports Python 3.10
#: and tomllib is 3.11+.
_PROJECT_NAME_RE = re.compile(
    r"(?m)^\s*name\s*=\s*[\"']" + re.escape(_DISTRIBUTION_NAME) + r"[\"']\s*(?:#.*)?$"
)


@dataclass(frozen=True)
class ReleaseStamp:
    """A valid release stamp: what channel this build ships on, and the key."""

    channel: str
    posthog_key: str


def _stamp_path() -> Path:
    """Locate the stamp relative to this module.

    A private function only so tests can monkeypatch the location; the
    release workflow writes next to this file in the installed package —
    nowhere else, and in particular never the current directory, where a
    user could forge a stamp.
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


def _running_package_dir() -> Path | None:
    """The ``rapid_mlx`` package directory: ``parent.parent`` of this module.

    ``None`` when it cannot be determined: a frozen/embedded interpreter
    may lack ``__file__`` entirely, and a ``__file__`` of ``None`` breaks
    ``Path()``. Callers treat ``None`` as unknown provenance.
    """
    try:
        return Path(__file__).resolve().parent.parent
    except Exception:
        # NameError (no __file__) / TypeError (__file__ = None) and any
        # path weirdness: the package directory is unknowable.
        return None


def _running_distribution(running_dir: Path) -> Distribution | None:
    """The installed distribution that actually provides THIS ``rapid_mlx``.

    ``importlib.metadata`` resolves by NAME through ``sys.path`` order,
    which can hand back metadata for a DIFFERENT install than the package
    being imported — e.g. a stale ``rapid_mlx-*.dist-info`` with
    ``dir_info.editable: true`` earlier on ``sys.path`` would silence a
    genuine official install forever. Bind the metadata to the running
    code instead: a distribution only counts when ``locate_file()`` maps
    the package directory name onto exactly *running_dir*, and its name
    is PEP 503-equivalent to ours (a repacked ``Name: rapid_mlx`` must
    not blind the gate). Cheap comparison first: ``locate_file`` plus one
    ``resolve()`` avoids parsing METADATA for every installed
    distribution. ``None`` when nothing matches — unknown provenance.
    Never raises; distributions are only probed for ``locate_file`` and,
    after the location matched, ``name``.
    """
    try:
        candidates = list(distributions())
    except Exception:
        return None
    for dist in candidates:
        try:
            located = dist.locate_file(_PACKAGE_DIRNAME)
            if located is None:
                continue
            # str() normalizes SimplePath (str | PathLike) for Path().
            if Path(str(located)).resolve() != running_dir:
                continue
            # The located package dir is ours; only now pay for METADATA.
            if _pep503_name(dist.name) != _DISTRIBUTION_NAME_NORM:
                continue
            return dist
        except Exception:
            continue
    return None


def _direct_url_says_editable(dist: Distribution | None) -> bool | None:
    """PEP 610 verdict for *dist*: editable install? ``None`` when unknowable.

    ``None`` means provenance is UNKNOWN and the caller must fail closed:
    no distribution is bound to the running package, the metadata could
    not be read at all, OR a ``direct_url.json`` is present but
    unparseable / not a JSON object (someone unpacked something we cannot
    vouch for). Provenance that is positively known reads as ``False``
    ("not editable"): the file is ABSENT — pip only writes
    ``direct_url.json`` for direct-URL, archive, VCS, and editable
    installs, so a plain PyPI/wheel install has none — or the JSON object
    has no ``dir_info`` mapping, which is exactly how a direct-URL
    (non-editable) install is recorded.
    """
    if dist is None:
        # Nothing bound to the running package: unknown provenance.
        return None
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        # Any metadata-read failure: unknown provenance does not transmit.
        return None
    if raw is None:
        return False
    try:
        obj = json.loads(raw)
    except ValueError:
        # PRESENT but unparseable: unknown provenance, fail closed.
        return None
    if not isinstance(obj, dict):
        # PRESENT, valid JSON, but not an object: same — fail closed.
        return None
    dir_info = obj.get("dir_info")
    if not isinstance(dir_info, dict):
        # A valid direct-URL install without dir_info: recorded, known,
        # and NOT editable — pip omits dir_info unless installing
        # in-place. This is knowledge, not ignorance, so ``False``.
        return False
    return dir_info.get("editable") is True


def _package_is_in_source_tree(package_dir: Path) -> bool:
    """True when *package_dir* (the ``rapid_mlx`` package) is OUR source.

    *package_dir* is the ``rapid_mlx`` package directory (what
    :func:`_running_package_dir` returns). In a source checkout, an
    unpacked sdist, or an editable install, its parent holds THIS
    project's ``pyproject.toml`` (``[project]`` name ``rapid-mlx``); in
    site-packages it never does. A pyproject.toml for a DIFFERENT project
    (a monorepo vendoring us) does not count.
    """
    try:
        text = (package_dir.parent / "pyproject.toml").read_text(encoding="utf-8")
    except FileNotFoundError:
        # No pyproject.toml beside the package: the site-packages shape.
        return False
    except Exception:
        # PRESENT but unreadable (permissions, undecodable bytes): the
        # source-tree question cannot be answered -> fail closed.
        return True
    return _PROJECT_NAME_RE.search(text) is not None


def is_editable_or_source_install() -> bool:
    """True when this install looks editable or source-like (must not send).

    True when the distribution bound to the running package is a PEP 610
    editable install, when provenance is unknown (no matching
    distribution, unreadable metadata, an unparseable ``direct_url.json``,
    or no usable ``__file__``), or when the package directory sits beside
    this project's own ``pyproject.toml`` (checkout, unpacked sdist,
    editable install). Never raises.
    """
    package_dir = _running_package_dir()
    if package_dir is None:
        # No usable __file__ (frozen/embedded interpreter): unknowable.
        return True
    if _direct_url_says_editable(_running_distribution(package_dir)) is not False:
        # ``True`` = editable; ``None`` = unknown provenance. Both fail closed.
        return True
    return _package_is_in_source_tree(package_dir)


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
