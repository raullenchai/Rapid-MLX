#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create and verify the hash manifest for a Python release candidate.

The manifest is deliberately tiny and stdlib-only so the publishing job can
verify the artifact hand-off without installing or executing project code.
It binds the release version and source commit to the exact wheel and sdist
that passed ``twine check`` and were uploaded as the workflow artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(dist_dir: Path) -> list[Path]:
    """Return the expected rapid-mlx wheel and sdist in stable order."""

    all_files = sorted(path for path in dist_dir.iterdir() if path.is_file())
    files = sorted(
        path
        for path in all_files
        if path.name.startswith("rapid_mlx-") and path.suffix in {".whl", ".gz"}
    )
    wheels = [path for path in files if path.suffix == ".whl"]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(files) != 2 or len(all_files) != 2:
        names = ", ".join(path.name for path in all_files) or "<none>"
        raise ValueError(
            "release dist must contain exactly one rapid_mlx wheel and one "
            f"tar.gz sdist, with no extra files; found: {names}"
        )
    return files


def validate_artifact_versions(files: list[Path], version: str) -> None:
    """Require artifact filenames to bind to the declared release version."""

    wheel = next(path for path in files if path.suffix == ".whl")
    sdist = next(path for path in files if path.name.endswith(".tar.gz"))
    if not wheel.name.startswith(f"rapid_mlx-{version}-"):
        raise ValueError(
            f"wheel filename {wheel.name!r} does not match release version {version!r}"
        )
    expected_sdist = f"rapid_mlx-{version}.tar.gz"
    if sdist.name != expected_sdist:
        raise ValueError(
            f"sdist filename {sdist.name!r} does not match release version {version!r}"
        )


def create_manifest(*, dist_dir: Path, source_sha: str, version: str) -> dict[str, Any]:
    """Return a JSON-serializable manifest for the files in ``dist_dir``."""

    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source SHA must be a 40-character lowercase Git commit SHA")
    if not version:
        raise ValueError("version must not be empty")
    files = release_files(dist_dir)
    validate_artifact_versions(files, version)
    return {
        "schema": 1,
        "project": "rapid-mlx",
        "version": version,
        "source_sha": source_sha,
        "artifacts": [
            {"filename": path.name, "sha256": sha256(path), "size": path.stat().st_size}
            for path in files
        ],
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    """Write a canonical, review-friendly manifest."""

    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_manifest(*, dist_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify artifact names, size and digest against a stored manifest."""

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read release manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema") != 1 or manifest.get("project") != "rapid-mlx":
        raise ValueError("release manifest has an unknown schema or project")
    version = manifest.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("release manifest version must be a non-empty string")
    recorded = manifest.get("artifacts")
    if not isinstance(recorded, list):
        raise ValueError("release manifest artifacts must be a list")
    expected = {
        item.get("filename"): item for item in recorded if isinstance(item, dict)
    }
    if len(recorded) != 2 or len(expected) != 2:
        raise ValueError("release manifest must contain exactly two unique artifacts")
    files = release_files(dist_dir)
    validate_artifact_versions(files, version)
    if set(expected) != {path.name for path in files}:
        raise ValueError("release manifest artifact names do not match dist/")
    for path in files:
        item = expected[path.name]
        if item.get("size") != path.stat().st_size:
            raise ValueError(f"size mismatch for {path.name}")
        if item.get("sha256") != sha256(path):
            raise ValueError(f"SHA-256 mismatch for {path.name}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="write a manifest for dist/")
    create.add_argument("--dist-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--source-sha", required=True)
    create.add_argument("--version", required=True)

    verify = subparsers.add_parser("verify", help="verify dist/ against a manifest")
    verify.add_argument("--dist-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.dist_dir.is_dir():
        raise ValueError(f"--dist-dir is not a directory: {args.dist_dir}")
    if args.command == "create":
        manifest = create_manifest(
            dist_dir=args.dist_dir,
            source_sha=args.source_sha,
            version=args.version,
        )
        write_manifest(manifest, args.output)
        print(f"wrote {args.output}")
    else:
        manifest = verify_manifest(dist_dir=args.dist_dir, manifest_path=args.manifest)
        print(f"verified rapid-mlx {manifest['version']} ({manifest['source_sha']})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"release manifest: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
