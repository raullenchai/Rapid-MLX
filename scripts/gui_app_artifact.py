#!/usr/bin/env python3
"""Create and verify provenance for the Desktop app consumed by GUI CI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1
BUILD_CONFIG = "release"
SIDECAR_MODE = "skipped"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create(archive: Path, source_sha: str, output: Path) -> None:
    if not archive.is_file():
        raise SystemExit(f"archive does not exist: {archive}")
    if not source_sha.strip():
        raise SystemExit("source SHA must not be empty")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "build_config": BUILD_CONFIG,
        "sidecar": SIDECAR_MODE,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify(archive: Path, manifest: Path, expected_source_sha: str) -> None:
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid GUI app manifest: {error}") from error
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source_sha": expected_source_sha,
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "build_config": BUILD_CONFIG,
        "sidecar": SIDECAR_MODE,
    }
    if payload != expected:
        differences = [
            f"{key}: expected {value!r}, found {payload.get(key)!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        unexpected = sorted(set(payload) - set(expected))
        if unexpected:
            differences.append(f"unexpected fields: {', '.join(unexpected)}")
        raise SystemExit(
            "GUI app provenance verification failed:\n- " + "\n- ".join(differences)
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--archive", type=Path, required=True)
    create_parser.add_argument("--source-sha", required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-source-sha", required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "create":
        create(args.archive, args.source_sha, args.output)
    else:
        verify(args.archive, args.manifest, args.expected_source_sha)


if __name__ == "__main__":
    main()
