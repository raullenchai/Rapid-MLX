#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create and verify the Desktop (rapid-mac DMG) release candidate manifest.

The manifest follows the same contract as ``release_manifest.py`` (which binds
the engine's wheel/sdist to a release version and source commit), adapted for
the Desktop app's single DMG artifact. It binds:

  - the requested release ``version`` and the intended ``rapid-mac-v*`` tag,
  - the candidate ``source_sha`` (full 40-character commit the app was built
    from),
  - the versions embedded in the built ``Rapid-MLX Desktop.app``
    (``CFBundleShortVersionString`` and ``CFBundleVersion``),
  - the DMG's filename, byte size and SHA-256, and
  - the signing / notarization / DMG-validation gate identities that the build
    passed and whether the build was signed (not ad-hoc).

It is deliberately tiny and stdlib-only so both the candidate gate and the
publishing job can verify the artifact hand-off without installing or executing
app code. A manifest is only "accepted" when the workflow run that produced it
completed successfully at ``source_sha`` AND ``verify`` agrees with the exact
DMG bytes on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from release_version import parse_version
except ModuleNotFoundError:  # imported by tests as ``scripts.*`` from repo root
    from scripts.release_version import parse_version

SCHEMA = 1
PROJECT = "rapid-mlx"
ARTIFACT_KIND = "desktop-dmg"
APP_TAG_PREFIX = "rapid-mac-v"


def _validation_gates(signed: bool, delta_compared: bool) -> str:
    """The exact gate steps that actually ran to produce this candidate.

    Claims are truthful — a gate that did not run is never recorded:

      * ``signed-build`` is only claimed for a signed (non ad-hoc) build; an
        ad-hoc build is recorded as ``ad-hoc-build`` so no lane claims signing
        or notarisation it never performed.
      * ``app-notarize`` / ``dmg-notarize`` / ``final-validate-dmg`` only run
        (and are only claimed) for a signed build.
      * ``dmg-size-delta`` only runs when a previous release baseline exists; if
        there is no baseline the comparison is skipped and not claimed.
    """

    gates = ["signed-build" if signed else "ad-hoc-build", "bundle-size"]
    if signed:
        gates.append("app-notarize")
    gates.append("dmg-build")
    if delta_compared:
        gates.append("dmg-size-delta")
    gates.append("validate-dmg")
    if signed:
        gates.extend(["dmg-notarize", "final-validate-dmg"])
    return "|".join(gates)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_commit_sha(source_sha: str) -> None:
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source SHA must be a 40-character lowercase Git commit SHA")


def _app_plist(app_dir: Path) -> dict[str, Any]:
    """Return the parsed Info.plist of the built ``Rapid-MLX Desktop.app``."""

    plist_path = app_dir / "build" / "Rapid-MLX Desktop.app" / "Contents" / "Info.plist"
    if not plist_path.is_file():
        raise ValueError(
            f"built app Info.plist not found at {plist_path} — a signed candidate "
            "must be fully built before a manifest can be created"
        )
    try:
        with plist_path.open("rb") as handle:
            plist = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        raise ValueError(
            f"cannot read built app Info.plist {plist_path}: {exc}"
        ) from exc
    if not isinstance(plist, dict):
        raise ValueError(f"built app Info.plist is not a dictionary: {plist_path}")
    return plist


def _embedded_versions(app_dir: Path) -> dict[str, str]:
    """Return ``CFBundleShortVersionString`` and ``CFBundleVersion`` from the app."""

    plist = _app_plist(app_dir)
    short = plist.get("CFBundleShortVersionString")
    build = plist.get("CFBundleVersion")
    if not isinstance(short, str) or not short:
        raise ValueError("built app Info.plist is missing CFBundleShortVersionString")
    if not isinstance(build, str) or not build:
        raise ValueError("built app Info.plist is missing CFBundleVersion")
    return {
        "CFBundleShortVersionString": short,
        "CFBundleVersion": build,
    }


def create_manifest(
    *,
    app_dir: Path,
    dmg_path: Path,
    source_sha: str,
    version: str,
    app_tag: str,
    signed: bool,
    delta_compared: bool,
) -> dict[str, Any]:
    """Return a JSON-serializable manifest for the built Desktop DMG."""

    _assert_commit_sha(source_sha)
    parse_version(version)  # require X.Y.Z or X.Y.Z-rcN
    if app_tag != f"{APP_TAG_PREFIX}{version}":
        raise ValueError(
            f"app tag {app_tag!r} does not match version {version!r} "
            f"(expected {APP_TAG_PREFIX}{version})"
        )
    if not dmg_path.is_file():
        raise ValueError(f"DMG not found at {dmg_path}")
    embedded = _embedded_versions(app_dir)
    if embedded["CFBundleShortVersionString"] != version:
        raise ValueError(
            "built app CFBundleShortVersionString "
            f"{embedded['CFBundleShortVersionString']!r} does not match release "
            f"version {version!r}"
        )
    return {
        "schema": SCHEMA,
        "project": PROJECT,
        "artifact_kind": ARTIFACT_KIND,
        "version": version,
        "app_tag": app_tag,
        "source_sha": source_sha,
        "embedded_version": embedded,
        "signed": bool(signed),
        "dmg_size_delta_compared": bool(delta_compared),
        "validation_gate": _validation_gates(bool(signed), bool(delta_compared)),
        "artifacts": [
            {
                "filename": dmg_path.name,
                "sha256": sha256(dmg_path),
                "size": dmg_path.stat().st_size,
            }
        ],
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    """Write a canonical, review-friendly manifest."""

    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_manifest(
    *, app_dir: Path, dmg_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Verify the built DMG against a stored manifest; return the manifest."""

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read desktop manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema") != SCHEMA or manifest.get("project") != PROJECT:
        raise ValueError("desktop manifest has an unknown schema or project")
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("desktop manifest has an unexpected artifact kind")
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("desktop manifest version must be a non-empty string")
    _assert_commit_sha(str(manifest.get("source_sha") or ""))
    if manifest.get("app_tag") != f"{APP_TAG_PREFIX}{version}":
        raise ValueError(
            "desktop manifest app tag does not match its version: "
            f"{manifest.get('app_tag')!r} vs {version!r}"
        )
    recorded = manifest.get("artifacts")
    if not isinstance(recorded, list) or len(recorded) != 1:
        raise ValueError("desktop manifest must contain exactly one artifact")
    item = recorded[0]
    if not isinstance(item, dict) or item.get("filename") != dmg_path.name:
        raise ValueError(
            f"desktop manifest artifact name {item.get('filename')!r} does not "
            f"match the DMG {dmg_path.name!r}"
        )
    if not dmg_path.is_file():
        raise ValueError(f"DMG not found at {dmg_path}")
    if item.get("size") != dmg_path.stat().st_size:
        raise ValueError(f"DMG size mismatch for {dmg_path.name}")
    if item.get("sha256") != sha256(dmg_path):
        raise ValueError(f"SHA-256 mismatch for {dmg_path.name}")
    embedded = _embedded_versions(app_dir)
    if manifest.get("embedded_version") != embedded:
        raise ValueError(
            "desktop manifest embedded versions do not match the built app: "
            f"{embedded!r}"
        )
    if embedded["CFBundleShortVersionString"] != version:
        raise ValueError(
            "built app CFBundleShortVersionString "
            f"{embedded['CFBundleShortVersionString']!r} does not match manifest "
            f"version {version!r}"
        )
    signed = manifest.get("signed")
    if not isinstance(signed, bool):
        raise ValueError("desktop manifest signed flag must be a boolean")
    delta_compared = manifest.get("dmg_size_delta_compared")
    if not isinstance(delta_compared, bool):
        raise ValueError("desktop manifest dmg_size_delta_compared must be a boolean")
    gates = manifest.get("validation_gate")
    expected_gates = _validation_gates(signed, delta_compared)
    if not isinstance(gates, str) or gates != expected_gates:
        raise ValueError(
            "desktop manifest validation gate does not match what actually ran "
            f"({gates!r} vs expected {expected_gates!r} for signed={signed}, "
            f"delta_compared={delta_compared})"
        )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="write a manifest for the built DMG")
    create.add_argument("--app-dir", type=Path, required=True)
    create.add_argument("--dmg", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--source-sha", required=True)
    create.add_argument("--version", required=True)
    create.add_argument("--app-tag", required=True)
    create.add_argument("--signed", action=argparse.BooleanOptionalAction)
    create.add_argument("--delta-compared", action=argparse.BooleanOptionalAction)

    verify = subparsers.add_parser(
        "verify", help="verify the built DMG against a manifest"
    )
    verify.add_argument("--app-dir", type=Path, required=True)
    verify.add_argument("--dmg", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        manifest = create_manifest(
            app_dir=args.app_dir,
            dmg_path=args.dmg,
            source_sha=args.source_sha,
            version=args.version,
            app_tag=args.app_tag,
            signed=bool(args.signed),
            delta_compared=bool(args.delta_compared),
        )
        write_manifest(manifest, args.output)
        print(
            f"wrote {args.output} for {manifest['version']} at {manifest['source_sha']}"
        )
    else:
        manifest = verify_manifest(
            app_dir=args.app_dir, dmg_path=args.dmg, manifest_path=args.manifest
        )
        print(
            f"verified {manifest['artifact_kind']} {manifest['version']} "
            f"({manifest['source_sha']})"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"desktop manifest: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
