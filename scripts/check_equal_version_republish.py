#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed unless an equal-version Desktop rerun is byte-identical."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DMG_NAME = "rapid-mlx-desktop.dmg"


def _load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_exact_artifact(
    *, candidate_path: Path, dmg_path: Path, release_path: Path | None
) -> tuple[str, str, int]:
    candidate = _load(candidate_path, "candidate latest.json")
    version = candidate.get("version")
    digest = candidate.get("dmg_sha256")
    size = candidate.get("dmg_size")
    if (
        not isinstance(version, str)
        or not version
        or not dmg_path.is_file()
        or not isinstance(digest, str)
        or len(digest) != 64
        or type(size) is not int
        or size <= 0
        or dmg_path.stat().st_size != size
        or _sha256(dmg_path) != digest
    ):
        raise ValueError("candidate latest.json does not match the exact-run DMG")
    if release_path is not None:
        release = _load(release_path, "existing GitHub Release")
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise ValueError("existing GitHub Release assets are not a list")
        matches = [
            item
            for item in assets
            if isinstance(item, dict) and item.get("name") == DMG_NAME
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one existing {DMG_NAME} asset; found {len(matches)}"
            )
        asset = matches[0]
        if (
            asset.get("state") != "uploaded"
            or asset.get("digest") != f"sha256:{digest}"
            or type(asset.get("size")) is not int
            or asset["size"] != size
        ):
            raise ValueError("existing GitHub Release DMG identity differs")
    return version, digest, size


def verify(
    *,
    current_path: Path,
    candidate_path: Path,
    dmg_path: Path,
    release_path: Path | None,
) -> str:
    version, _, _ = verify_exact_artifact(
        candidate_path=candidate_path, dmg_path=dmg_path, release_path=release_path
    )
    current = _load(current_path, "current latest.json")
    candidate = _load(candidate_path, "candidate latest.json")
    if current.get("version") != version:
        raise ValueError("latest.json versions are not equal")
    identity_fields = ("dmg_url", "dmg_sha256", "dmg_size")
    if any(current.get(field) != candidate.get(field) for field in identity_fields):
        raise ValueError("equal-version latest.json artifact identity differs")
    return f"equal-version {version} is byte-identical; mutable updater no-op"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--release-json", type=Path)
    args = parser.parse_args()
    try:
        if args.current is None:
            version, digest, size = verify_exact_artifact(
                candidate_path=args.candidate,
                dmg_path=args.dmg,
                release_path=args.release_json,
            )
            print(f"exact-run {version} sha256:{digest}/{size} identity accepted")
        else:
            print(
                verify(
                    current_path=args.current,
                    candidate_path=args.candidate,
                    dmg_path=args.dmg,
                    release_path=args.release_json,
                )
            )
    except ValueError as exc:
        print(f"equal-version republish refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
