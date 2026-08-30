#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create and verify provenance for an exact pre-tag Desktop release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from release_version import parse_version
except ModuleNotFoundError:
    from scripts.release_version import parse_version

SCHEMA = 1
LIFECYCLE_STAGE = "pre-tag-candidate"
DMG = "rapid-mlx-desktop.dmg"
DESKTOP_MANIFEST = "rapid-mlx-desktop.manifest.json"
NOTES = "release-notes.md"
APPCAST = "sparkle/appcast.xml"
PROMOTION_MANIFEST = "desktop-promotion-manifest.json"
SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: str, label: str = "source SHA") -> str:
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a 40-character lowercase Git commit SHA")
    return value


def _positive(value: int, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"unsafe promotion payload path: {value!r}")
    return value


def _regular_file(bundle: Path, relative: str) -> Path:
    relative = _safe_relative(relative)
    path = bundle / relative
    if (
        any(parent.is_symlink() for parent in [path, *list(path.parents)[:2]])
        or not path.is_file()
    ):
        raise ValueError(f"promotion payload is missing a regular file: {relative}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _payload_paths(bundle: Path) -> list[str]:
    sparkle = bundle / "sparkle"
    zips = sorted(
        path
        for path in sparkle.glob("*.zip")
        if path.is_file() and not path.is_symlink()
    )
    if len(zips) != 1:
        raise ValueError(
            f"promotion bundle must contain exactly one regular sparkle/*.zip; found {len(zips)}"
        )
    paths = [DMG, DESKTOP_MANIFEST, NOTES, APPCAST, f"sparkle/{zips[0].name}"]
    for relative in paths:
        _regular_file(bundle, relative)
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    allowed = {*paths, PROMOTION_MANIFEST}
    unexpected = actual - allowed
    if unexpected:
        raise ValueError(
            "promotion bundle contains unrecorded payload: "
            + ", ".join(sorted(unexpected))
        )
    if (bundle / NOTES).stat().st_size == 0:
        raise ValueError("promotion release notes must not be empty")
    return paths


def _verify_desktop_manifest(
    bundle: Path, *, source_sha: str, version: str, app_tag: str
) -> None:
    manifest = _load_json(bundle / DESKTOP_MANIFEST, "Desktop manifest")
    if manifest.get("source_sha") != source_sha:
        raise ValueError("Desktop manifest source SHA does not match promotion source")
    if manifest.get("version") != version or manifest.get("app_tag") != app_tag:
        raise ValueError(
            "Desktop manifest version/tag does not match promotion identity"
        )
    if manifest.get("signed") is not True:
        raise ValueError("Desktop manifest is not a signed release candidate")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("Desktop manifest must contain exactly one DMG artifact")
    item = artifacts[0]
    dmg = _regular_file(bundle, DMG)
    if not isinstance(item, dict) or item.get("filename") != DMG:
        raise ValueError("Desktop manifest does not name the canonical DMG")
    if item.get("size") != dmg.stat().st_size or item.get("sha256") != _sha256(dmg):
        raise ValueError("Desktop manifest does not match the canonical DMG bytes")


def _verify_appcast(bundle: Path, *, version: str, zip_relative: str) -> None:
    appcast = _regular_file(bundle, APPCAST)
    try:
        root = ET.parse(appcast).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot parse Sparkle appcast: {exc}") from exc
    item_enclosures = [
        (item, enclosure)
        for item in root.findall(".//item")
        for enclosure in item.findall("enclosure")
    ]
    if len(item_enclosures) != 1:
        raise ValueError(
            "Sparkle appcast must contain exactly one enclosure; "
            f"found {len(item_enclosures)}"
        )
    item, enclosure = item_enclosures[0]
    zip_path = _regular_file(bundle, zip_relative)
    url = enclosure.get("url", "")
    if url.rsplit("/", 1)[-1] != zip_path.name:
        raise ValueError("Sparkle enclosure URL does not name the promoted ZIP")
    if enclosure.get("length") != str(zip_path.stat().st_size):
        raise ValueError("Sparkle enclosure length does not match the promoted ZIP")
    short_version = _sparkle_version_value(item, enclosure, "shortVersionString")
    if short_version != version:
        raise ValueError("Sparkle short version does not match the promotion version")
    build = _sparkle_version_value(item, enclosure, "version")
    if not build.isdigit() or int(build) < 1:
        raise ValueError("Sparkle enclosure has no valid numeric bundle build")
    if not enclosure.get(f"{{{SPARKLE_NS}}}edSignature"):
        raise ValueError("Sparkle enclosure has no EdDSA signature")


def _sparkle_version_value(
    item: ET.Element, enclosure: ET.Element, local_name: str
) -> str:
    """Read one Sparkle identity field without accepting contradictory copies.

    Sparkle 2 emits ``version`` and ``shortVersionString`` as item-level
    elements. Older appcasts may carry the same values as enclosure
    attributes. Sparkle accepts either representation, so promotion must too,
    while rejecting an appcast that supplies both with different values.
    """

    child = item.find(f"{{{SPARKLE_NS}}}{local_name}")
    element_value = (child.text or "").strip() if child is not None else ""
    attribute_value = enclosure.get(f"{{{SPARKLE_NS}}}{local_name}", "").strip()
    if element_value and attribute_value and element_value != attribute_value:
        raise ValueError(f"Sparkle {local_name} values conflict")
    value = element_value or attribute_value
    if not value:
        raise ValueError(f"Sparkle appcast is missing {local_name}")
    return value


def create_manifest(
    *,
    bundle: Path,
    repository: str,
    workflow: str,
    run_id: int,
    run_attempt: int,
    source_sha: str,
    version: str,
    app_tag: str,
) -> dict[str, Any]:
    _sha(source_sha)
    parse_version(version)
    if app_tag != f"rapid-mac-v{version}":
        raise ValueError("promotion tag does not match version")
    if (
        repository.count("/") != 1
        or repository.startswith("/")
        or repository.endswith("/")
    ):
        raise ValueError("repository must be an owner/name identifier")
    if workflow != ".github/workflows/auto-release.yml":
        raise ValueError("promotion producer must be auto-release.yml")
    paths = _payload_paths(bundle)
    _verify_desktop_manifest(
        bundle, source_sha=source_sha, version=version, app_tag=app_tag
    )
    zip_relative = next(path for path in paths if path.endswith(".zip"))
    _verify_appcast(bundle, version=version, zip_relative=zip_relative)
    return {
        "schema": SCHEMA,
        "lifecycle_stage": LIFECYCLE_STAGE,
        "release": {"version": version, "app_tag": app_tag},
        "producer": {
            "repository": repository,
            "workflow": workflow,
            "run_id": _positive(run_id, "run id"),
            "run_attempt": _positive(run_attempt, "run attempt"),
            "source_sha": source_sha,
        },
        "artifacts": [
            {
                "path": relative,
                "sha256": _sha256(_regular_file(bundle, relative)),
                "size": _regular_file(bundle, relative).stat().st_size,
            }
            for relative in paths
        ],
    }


def verify_manifest(
    *,
    bundle: Path,
    manifest_path: Path,
    repository: str,
    workflow: str,
    run_id: int,
    run_attempt: int,
    source_sha: str,
    version: str,
    app_tag: str,
) -> dict[str, Any]:
    parse_version(version)
    if app_tag != f"rapid-mac-v{version}":
        raise ValueError("expected promotion tag does not match version")
    canonical_manifest = _regular_file(bundle, PROMOTION_MANIFEST)
    if manifest_path.absolute() != canonical_manifest.absolute():
        raise ValueError(
            "promotion manifest must be the canonical file inside the bundle"
        )
    manifest = _load_json(canonical_manifest, "promotion manifest")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("promotion manifest has an unknown schema")
    if manifest.get("lifecycle_stage") != LIFECYCLE_STAGE:
        raise ValueError("promotion manifest has an unexpected lifecycle stage")
    release = manifest.get("release")
    producer = manifest.get("producer")
    if not isinstance(release, dict) or release != {
        "version": version,
        "app_tag": app_tag,
    }:
        raise ValueError(
            "promotion manifest release identity does not match expected tag/version"
        )
    if not isinstance(producer, dict):
        raise ValueError("promotion manifest producer must be an object")
    expected = {
        "repository": repository,
        "workflow": workflow,
        "run_id": _positive(run_id, "expected run id"),
        "run_attempt": _positive(run_attempt, "expected run attempt"),
        "source_sha": _sha(source_sha, "expected source SHA"),
    }
    for key, value in expected.items():
        if producer.get(key) != value:
            raise ValueError(
                f"promotion producer {key} does not match the requested run"
            )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("promotion manifest artifacts must be an array")
    paths = _payload_paths(bundle)
    if len(artifacts) != len(paths):
        raise ValueError("promotion manifest artifact roster is incomplete")
    recorded: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("promotion manifest contains a malformed artifact")
        relative = _safe_relative(item["path"])
        if relative in recorded:
            raise ValueError(f"promotion manifest repeats artifact path {relative}")
        recorded[relative] = item
    if set(recorded) != set(paths):
        raise ValueError("promotion manifest artifact roster does not match the bundle")
    for relative in paths:
        path = _regular_file(bundle, relative)
        item = recorded[relative]
        if item.get("size") != path.stat().st_size or item.get("sha256") != _sha256(
            path
        ):
            raise ValueError(
                f"promotion artifact bytes do not match manifest: {relative}"
            )

    _verify_desktop_manifest(
        bundle, source_sha=source_sha, version=version, app_tag=app_tag
    )
    _verify_appcast(
        bundle,
        version=version,
        zip_relative=next(path for path in paths if path.endswith(".zip")),
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bundle", type=Path, required=True)
    common.add_argument("--repository", required=True)
    common.add_argument("--workflow", default=".github/workflows/auto-release.yml")
    common.add_argument("--run-id", type=int, required=True)
    common.add_argument("--source-sha", required=True)
    common.add_argument("--version", required=True)
    common.add_argument("--app-tag", required=True)
    create = sub.add_parser("create", parents=[common])
    create.add_argument("--run-attempt", type=int, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", parents=[common])
    verify.add_argument("--run-attempt", type=int, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = create_manifest(
                bundle=args.bundle,
                repository=args.repository,
                workflow=args.workflow,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
                source_sha=args.source_sha,
                version=args.version,
                app_tag=args.app_tag,
            )
            args.output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            verify_manifest(
                bundle=args.bundle,
                manifest_path=args.manifest,
                repository=args.repository,
                workflow=args.workflow,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
                source_sha=args.source_sha,
                version=args.version,
                app_tag=args.app_tag,
            )
    except (OSError, ValueError) as exc:
        print(f"desktop promotion: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
