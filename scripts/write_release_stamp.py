#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the telemetry v2 release stamp into the package tree.

Telemetry v2 transmits ONLY from official release builds
(``rapid_mlx/telemetry/build_gate.py``). The gate requires a file
``rapid_mlx/telemetry/_release_stamp.json`` of shape
``{"channel": "stable" | "rc", "posthog_key": "phc_…"}`` that only the
release workflow writes and that is never committed (``.gitignore``
covers it; see #3628). This script is that writer, and nothing else:

    python scripts/write_release_stamp.py --version v0.15.0rc1

The channel is derived from the version/tag:

* a plain ``X.Y.Z`` (optionally leading ``v``) -> ``"stable"``;
* a PEP 440 / tag pre-release (``0.15.0rc1``, ``v0.15.0-rc1``, ``a``,
  ``b``, ``c``, ``.dev`` segments) -> ``"rc"``;
* ANYTHING else (``latest``, ``1.2``, ``1.2.3.4``, local versions,
  unicode digits, surrounding whitespace, ...) -> exit non-zero and
  write nothing. The grammar deliberately mirrors
  ``scripts/release_version.py``: the publish lane binds the tag to the
  wheel/sdist filenames, so a spelling it cannot bind must fail here
  instead of shipping a mislabelled build.

The stamp targets ``<repo>/rapid_mlx/telemetry/_release_stamp.json``
(the package directory that ``python -m build`` then packs into both
the wheel and the sdist). The default destination is derived from THIS
script's location — the repository root two levels up — and not from
the imported ``build_gate.__file__``: a machine with an installed
``rapid-mlx`` must stamp the checkout being released, never
site-packages. ``--dest`` overrides it for tests.

After writing, the file is read back and round-tripped through
``build_gate._parse_stamp`` — the exact validator the runtime gate
uses — and a stamp that does not parse is deleted and fails the run.
An existing stamp with different content is never overwritten unless
``--force``; rewriting identical content is an idempotent no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _ReleaseStamp(Protocol):
    channel: str
    posthog_key: str


class _BuildGate(Protocol):
    RELEASE_STAMP_NAME: str
    _POSTHOG_KEY_RE: re.Pattern[str]

    def _parse_stamp(self, raw: str) -> _ReleaseStamp | None: ...


def _load_build_gate() -> ModuleType:
    """Load this checkout's stdlib-only gate without importing its package."""
    path = _REPO_ROOT / "rapid_mlx" / "telemetry" / "build_gate.py"
    name = "_rapid_mlx_release_build_gate"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load release build gate from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the module by name while executing @dataclass.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


# Importing ``rapid_mlx.telemetry`` executes its package initializer and pulls
# runtime dependencies such as PyYAML. The publish stamp step intentionally has
# only build tooling installed, so load the stdlib-only source file directly.
build_gate = cast(_BuildGate, _load_build_gate())

#: PostHog **public, write-only** project token for the Rapid-MLX project.
#: Verify it in PostHog project settings for project 619833 (US Cloud).
#: PostHog project API keys (``phc_…``) are explicitly safe to embed in
#: public apps — they can only ingest events, never read them, and the
#: stamp exists precisely so only official builds ingest. Override for
#: local/staging sinks via the ``RAPID_MLX_POSTHOG_KEY`` environment
#: variable (validated with the same regex ``build_gate`` enforces).
DEFAULT_POSTHOG_KEY = "phc_pnhbZbU8pZKysBtXPkd2qmF56i3ARfnY5bt9mZFdeEEW"

#: Environment variable overriding :data:`DEFAULT_POSTHOG_KEY`.
POSTHOG_KEY_ENV = "RAPID_MLX_POSTHOG_KEY"

#: Version-number component: no leading zeros (PEP 440, and the same
#: grammar ``scripts/release_version.py`` enforces for release subjects).
_NUM = r"(?:0|[1-9][0-9]*)"
_CORE = rf"{_NUM}\.{_NUM}\.{_NUM}"

#: Plain stable version: exactly three numeric components, optional ``v``.
#: Full-match — surrounding whitespace, local segments, a fourth component
#: or any decoration must fail rather than pass as "stable".
_STABLE_VERSION_RE = re.compile(rf"v?{_CORE}\Z")

#: Pre-release: the same core followed by a PEP 440 pre-release segment —
#: ``rc``/``a``/``b``/``c`` with optional separator, or a ``.dev`` segment.
#: The trailing number is optional (PEP 440 normalizes ``1.2a`` to
#: ``1.2a0``). Matched case-sensitively: the publish lane (and
#: ``scripts/release_version.py``) only bind lowercase ``rc`` tags.
_RC_VERSION_RE = re.compile(rf"v?{_CORE}(?:[-._]?(?:rc|a|b|c)|[-._]?dev)[0-9]*\Z")


class StampError(ValueError):
    """A release stamp could not be derived, validated, or written."""


def derive_channel(version: str) -> str:
    """Return ``"stable"`` or ``"rc"`` for *version* (a tag or a version).

    Raises :class:`StampError` for anything that is neither a plain
    ``v?X.Y.Z`` nor a pre-release of that shape — the caller must write
    nothing in that case.
    """
    if _RC_VERSION_RE.fullmatch(version) is not None:
        return "rc"
    if _STABLE_VERSION_RE.fullmatch(version) is not None:
        return "stable"
    raise StampError(
        f"invalid release version/tag: {version!r} — expected a plain "
        "'X.Y.Z' (stable, optional leading 'v') or a PEP 440 pre-release "
        "such as 'X.Y.ZrcN' / 'X.Y.Z-rcN' / 'X.Y.Za1' / 'X.Y.Z.devN' (rc)"
    )


def resolve_posthog_key(env: Mapping[str, str] | None = None) -> str:
    """The PostHog key for the stamp: env override, else the public token.

    Either value must satisfy the exact regex the runtime gate
    (``build_gate._POSTHOG_KEY_RE``) enforces on read — a key that would
    make the written stamp unparseable must fail here, at write time.
    """
    source = os.environ if env is None else env
    key = source.get(POSTHOG_KEY_ENV, "") or DEFAULT_POSTHOG_KEY
    if not isinstance(key, str) or build_gate._POSTHOG_KEY_RE.fullmatch(key) is None:
        raise StampError(
            f"{POSTHOG_KEY_ENV} is not a valid PostHog project key: {key!r} "
            "(expected 'phc_' followed by 20-80 alphanumerics)"
        )
    return key


def default_dest() -> Path:
    """The stamp path inside THIS repository's package tree."""
    return _REPO_ROOT / "rapid_mlx" / "telemetry" / build_gate.RELEASE_STAMP_NAME


def stamp_document(channel: str, key: str) -> str:
    """Serialize the stamp exactly the way the runtime gate reads it."""
    return json.dumps({"channel": channel, "posthog_key": key}) + "\n"


def write_stamp(
    version: str,
    dest: Path | None = None,
    force: bool = False,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, bool]:
    """Write the stamp for *version*; return ``(path, written)``.

    Refuses to overwrite an existing stamp with different content unless
    *force*; identical content is an idempotent no-op. The written file
    is round-tripped through ``build_gate._parse_stamp`` — a stamp that
    does not parse is removed and raises. Nothing is written when the
    version or key fails validation.
    """
    channel = derive_channel(version)
    key = resolve_posthog_key(env)
    target = dest if dest is not None else default_dest()
    raw = stamp_document(channel, key)

    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise StampError(f"cannot read existing stamp {target}: {exc}") from exc
        if existing != raw and not force:
            raise StampError(
                f"refusing to overwrite {target} with different content; "
                "pass --force to replace it"
            )
        if existing == raw:
            return target, False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw, encoding="utf-8")
    # Round-trip through the runtime gate's own validator: the file on
    # disk must be exactly what official_build() will accept.
    written = target.read_text(encoding="utf-8")
    if build_gate._parse_stamp(written) is None:
        target.unlink()
        raise StampError(
            f"stamp written to {target} failed to round-trip through "
            "build_gate._parse_stamp; removed it"
        )
    return target, True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the telemetry v2 release stamp (official builds only)."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="release tag or version, e.g. v0.15.0rc1 or 0.15.0",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="stamp path (default: <repo>/rapid_mlx/telemetry/_release_stamp.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing stamp that has different content",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target, written = write_stamp(
            version=args.version, dest=args.dest, force=args.force
        )
    except StampError as exc:
        print(f"write_release_stamp: {exc}", file=sys.stderr)
        return 1
    print(f"{'wrote' if written else 'already current'}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
