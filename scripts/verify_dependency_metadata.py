#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject direct URL dependencies in built release distributions.

Public package indexes reject uploaded distributions whose ``Requires-Dist``
metadata contains a direct reference. Build frontends and ``twine check`` accept
that metadata, so this verifier inspects the exact wheel and sdist selected by
the release manifest before publication.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

try:
    from release_manifest import release_files
except ModuleNotFoundError:  # imported by tests as ``scripts.*``
    from scripts.release_manifest import release_files


_DIRECT_REFERENCE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\s*\[\s*[A-Za-z0-9._-]+(?:\s*,\s*[A-Za-z0-9._-]+)*\s*\])?"
    r"\s*@\s*\S+"
)


def _metadata_member(
    names: list[str], *, pattern: str, label: str, artifact: Path
) -> str:
    matches = [name for name in names if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise ValueError(
            f"{artifact.name}: expected exactly one canonical {label} metadata "
            f"member, found {len(matches)}"
        )
    return matches[0]


def artifact_metadata(artifact: Path) -> bytes:
    """Return the canonical core metadata bytes from a wheel or sdist."""

    try:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                member = _metadata_member(
                    archive.namelist(),
                    pattern=r"[^/]+\.dist-info/METADATA",
                    label="METADATA",
                    artifact=artifact,
                )
                return archive.read(member)
        if artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact, "r:gz") as archive:
                member_name = _metadata_member(
                    archive.getnames(),
                    pattern=r"[^/]+/PKG-INFO",
                    label="PKG-INFO",
                    artifact=artifact,
                )
                member = archive.extractfile(member_name)
                if member is None:
                    raise ValueError(f"{artifact.name}: cannot read {member_name}")
                return member.read()
    except (OSError, tarfile.TarError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(
            f"{artifact.name}: cannot read distribution metadata: {exc}"
        ) from exc
    raise ValueError(f"{artifact.name}: unsupported distribution type")


def direct_references(metadata: bytes, *, artifact: Path) -> list[str]:
    """Return direct-reference dependency declarations from core metadata."""

    try:
        message = BytesParser().parsebytes(metadata)
    except Exception as exc:  # email parser errors are rare and type-specific
        raise ValueError(f"{artifact.name}: invalid core metadata: {exc}") from exc
    requirements = message.get_all("Requires-Dist", [])
    return [value for value in requirements if _DIRECT_REFERENCE.match(value.strip())]


def verify_artifact(artifact: Path) -> None:
    references = direct_references(artifact_metadata(artifact), artifact=artifact)
    if references:
        joined = "; ".join(references)
        raise ValueError(
            f"{artifact.name}: direct URL dependencies are not publishable: {joined}"
        )


def verify_dist(dist_dir: Path) -> None:
    for artifact in release_files(dist_dir):
        verify_artifact(artifact)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject direct URL dependencies in built wheel/sdist metadata."
    )
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        verify_dist(args.dist_dir)
    except (OSError, ValueError) as exc:
        print(f"verify_dependency_metadata: {exc}", file=sys.stderr)
        return 1
    print("dependency metadata verified in wheel and sdist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
