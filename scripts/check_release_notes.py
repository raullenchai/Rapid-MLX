#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the version-bound release-note inputs for a bump PR."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from release_version import VERSION_RE
except ModuleNotFoundError:  # imported by tests as ``scripts.*`` from repo root
    from scripts.release_version import VERSION_RE

CHANGELOG_HEADING_RE = re.compile(r"^## \[([^]]+)](?:\s|$)")


def check_release_notes(version: str, changelog: Path, notes_dir: Path) -> None:
    """Fail when the Desktop changelog and curated notes are not in sync."""

    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid release version: {version!r}")

    changelog_lines = changelog.read_text(encoding="utf-8").splitlines()
    starts = [
        index
        for index, line in enumerate(changelog_lines)
        if (match := CHANGELOG_HEADING_RE.match(line)) and match.group(1) == version
    ]
    if not starts:
        raise ValueError(
            f"{changelog} has no exact '## [{version}]' section; "
            f"add '## [{version}]' with the Desktop release notes"
        )
    if len(starts) != 1:
        raise ValueError(
            f"{changelog} has {len(starts)} '## [{version}]' sections; keep exactly one"
        )

    section_start = starts[0] + 1
    section_end = next(
        (
            index
            for index in range(section_start, len(changelog_lines))
            if changelog_lines[index].startswith("## ")
        ),
        len(changelog_lines),
    )
    if not "\n".join(changelog_lines[section_start:section_end]).strip():
        raise ValueError(
            f"{changelog} has an empty '## [{version}]' section; add the Desktop release notes"
        )

    notes = notes_dir / f"v{version}.md"
    if not notes.is_file():
        raise ValueError(
            f"release notes are missing: {notes}; create it for this bump PR"
        )
    if not notes.read_text(encoding="utf-8").strip():
        raise ValueError(f"release notes are empty: {notes}; add curated release notes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--changelog", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        check_release_notes(args.version, args.changelog, args.notes_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"release-note inputs are synchronized for {args.version}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
