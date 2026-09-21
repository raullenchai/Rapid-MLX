#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the telemetry release stamp inside the built wheel and sdist.

The stamp written by ``scripts/write_release_stamp.py`` is only useful if
it actually SHIPS: telemetry v2 reads the stamp from the installed
package (``rapid_mlx/telemetry/build_gate.py``), so a wheel or sdist
that left it out — or carries a stamp that does not parse, or whose
channel contradicts the release tag — must fail the release job before
anything reaches PyPI. This script opens the built artifacts directly
(stdlib ``zipfile``/``tarfile``), extracts
``rapid_mlx/telemetry/_release_stamp.json`` from each, validates it with
the runtime gate's own ``build_gate._parse_stamp``, and requires the
stamp's channel to equal the channel derived from the artifact's own
filename version (and, when ``--version`` is given, from the release
tag).

    python scripts/verify_release_stamp.py dist/ --version v0.15.0rc1

The artifact set is enumerated with ``release_manifest.release_files`` —
the same exactly-one-wheel-and-one-sdist contract the manifest step
enforces — so this check can never bless a dist/ layout the release
manifest would reject. It is read-only and runs BEFORE the manifest
step; it does not weaken or replace any existing verification.
"""

from __future__ import annotations

import argparse
import importlib.util
import stat
import sys
import tarfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _ReleaseStamp(Protocol):
    channel: str
    posthog_key: str


class _BuildGate(Protocol):
    RELEASE_STAMP_NAME: str

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


# Do not import ``rapid_mlx.telemetry`` here: its package initializer reaches
# optional/runtime dependencies absent from the release build-tools venv.
build_gate = cast(_BuildGate, _load_build_gate())

# Isolated mode omits even the entry-point script's directory from sys.path.
# Add only scripts/ (never the repo root/package tree) for the two stdlib-only
# sibling helpers this verifier shares with the release workflow.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

if TYPE_CHECKING:
    derive_channel: Callable[[str], str]
    release_files: Callable[[Path], list[Path]]
else:
    from release_manifest import release_files
    from write_release_stamp import derive_channel

#: The last three path components of the stamp inside ANY artifact: the
#: wheel stores it at ``rapid_mlx/telemetry/<name>``; the sdist nests the
#: same path under one ``rapid_mlx-<version>/`` top-level directory.
_STAMP_PARTS = ("rapid_mlx", "telemetry", build_gate.RELEASE_STAMP_NAME)


def _decode(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: release stamp is not valid UTF-8: {exc}") from exc


def _read_stamp_from_wheel(path: Path) -> str:
    """Extract the stamp text from a wheel (a zip archive)."""
    try:
        with zipfile.ZipFile(path) as archive:
            member_name = "/".join(_STAMP_PARTS)
            member = archive.getinfo(member_name)
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if member.is_dir() or (
                member.create_system == 3 and file_type not in (0, stat.S_IFREG)
            ):
                raise KeyError(member_name)
            raw = archive.read(member)
    except KeyError:
        raise ValueError(
            f"{path.name}: wheel is missing {'/'.join(_STAMP_PARTS)} — "
            "telemetry would stay silent in this build"
        ) from None
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name}: not a readable wheel (zip): {exc}") from exc
    return _decode(raw, path.name)


def _read_stamp_from_sdist(path: Path) -> str:
    """Extract the stamp text from an sdist (a gzipped tar archive)."""
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                # The sdist wraps everything in one top-level directory;
                # match on the trailing package-relative path so either
                # spelling is found. A non-regular member named like the stamp
                # is not a stamp; skip it without asking tarfile to dereference
                # a potentially missing or hostile link target.
                if PurePosixPath(member.name).parts[-3:] != _STAMP_PARTS:
                    continue
                if not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:  # pragma: no cover - isfile() guarantees it
                    continue
                return _decode(handle.read(), path.name)
    except tarfile.TarError as exc:
        raise ValueError(f"{path.name}: not a readable sdist (tar.gz): {exc}") from exc
    raise ValueError(
        f"{path.name}: sdist is missing {'/'.join(_STAMP_PARTS)} — "
        "telemetry would stay silent in this build"
    )


def _artifact_version(filename: str) -> str:
    """The PEP 440 version embedded in a wheel/sdist filename."""
    if filename.endswith(".whl"):
        parts = filename.removesuffix(".whl").split("-")
        if len(parts) < 2 or parts[0] != "rapid_mlx" or not parts[1]:
            raise ValueError(f"unrecognized wheel filename: {filename!r}")
        return parts[1]
    if filename.endswith(".tar.gz"):
        stem = filename.removesuffix(".tar.gz")
        if not stem.startswith("rapid_mlx-"):
            raise ValueError(f"unrecognized sdist filename: {filename!r}")
        return stem.removeprefix("rapid_mlx-")
    raise ValueError(
        f"unrecognized artifact filename: {filename!r} (expected .whl or .tar.gz)"
    )


def verify_artifact(path: Path, expected_channel: str | None = None) -> _ReleaseStamp:
    """Assert one artifact carries a valid stamp matching its tag kind.

    *expected_channel* (when given) is the channel derived from the
    release tag; every artifact must agree with it on top of agreeing
    with its own filename.
    """
    if path.name.endswith(".whl"):
        raw = _read_stamp_from_wheel(path)
    elif path.name.endswith(".tar.gz"):
        raw = _read_stamp_from_sdist(path)
    else:
        raise ValueError(
            f"{path.name}: unsupported artifact type (expected .whl or .tar.gz)"
        )
    stamp = build_gate._parse_stamp(raw)
    if stamp is None:
        raise ValueError(
            f"{path.name}: release stamp does not parse "
            "(see rapid_mlx.telemetry.build_gate._parse_stamp)"
        )
    tag_channel = derive_channel(_artifact_version(path.name))
    if stamp.channel != tag_channel:
        raise ValueError(
            f"{path.name}: stamp channel {stamp.channel!r} does not match "
            f"the artifact version's channel {tag_channel!r}"
        )
    if expected_channel is not None and stamp.channel != expected_channel:
        raise ValueError(
            f"{path.name}: stamp channel {stamp.channel!r} does not match "
            f"the release tag's channel {expected_channel!r}"
        )
    return stamp


def verify_dist(
    dist_dir: Path, expected_version: str | None = None
) -> list[tuple[Path, _ReleaseStamp]]:
    """Verify the stamp in the wheel and sdist of *dist_dir*.

    Reuses ``release_manifest.release_files`` so the accepted dist/ shape
    is exactly the one the release manifest step accepts. Raises
    :class:`ValueError` (often as :class:`StampError`) on any problem.
    """
    expected_channel = (
        derive_channel(expected_version) if expected_version is not None else None
    )
    return [
        (path, verify_artifact(path, expected_channel))
        for path in release_files(dist_dir)
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the telemetry release stamp inside built artifacts."
    )
    parser.add_argument("dist_dir", type=Path, help="directory holding the dist/")
    parser.add_argument(
        "--version",
        default=None,
        help="release tag or version; require the stamp channel to match it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.dist_dir.is_dir():
            raise ValueError(f"--dist-dir is not a directory: {args.dist_dir}")
        results = verify_dist(args.dist_dir, expected_version=args.version)
    except ValueError as exc:
        print(f"verify_release_stamp: {exc}", file=sys.stderr)
        return 1
    for path, stamp in results:
        print(f"  {path.name}: stamp channel={stamp.channel}")
    print("release stamp verified in wheel and sdist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
