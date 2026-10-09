#!/usr/bin/env python3
"""Render a metadata-only release bump without changing product files."""

from __future__ import annotations

import datetime
import plistlib
import re
from pathlib import Path

try:
    from check_release_notes import check_release_notes
    from release_version import parse_version
except ModuleNotFoundError:
    from scripts.check_release_notes import check_release_notes
    from scripts.release_version import parse_version

PLIST = "apps/rapid-mac/Resources/Info.plist"
CHANGELOG = "apps/rapid-mac/CHANGELOG.md"


def project_version(text: str) -> str:
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    if project is None:
        raise ValueError("missing [project] metadata")
    versions = re.findall(r'^version\s*=\s*"([^"]+)"\s*$', project[1], re.M)
    if len(versions) != 1:
        raise ValueError("expected one quoted project version")
    parse_version(versions[0])
    return versions[0]


def replace_plist(text: str, key: str, value: str) -> str:
    pattern = rf"(<key>{key}</key>\s*<string>)[^<]*(</string>)"
    result, count = re.subn(pattern, lambda m: m[1] + value + m[2], text)
    if count != 1:
        raise ValueError(f"expected one string plist key: {key}")
    return result


def changelog_section(text: str, version: str) -> str:
    match = re.search(
        rf"(?ms)^## \[{re.escape(version)}\][^\n]*\n.*?(?=^## |^\[[^\n]+\]:|\Z)",
        text,
    )
    if match is None:
        raise ValueError(f"published changelog has no section for {version}")
    return match[0].strip() + "\n"


def render(
    *,
    project: str,
    plist: str,
    changelog: str,
    published_changelog: str,
    previous_version: str,
    version: str,
    previous_builds: list[int],
    notes: str,
    highlights: str,
    date: str,
) -> dict[str, str]:
    """Keep source metadata intact; import missing published changelog sections."""
    datetime.date.fromisoformat(date)
    if parse_version(version) <= max(
        parse_version(project_version(project)), parse_version(previous_version)
    ):
        raise ValueError("release version must exceed source and published versions")
    if not notes.strip() or not highlights.strip():
        raise ValueError("curated notes and user-facing highlights cannot be empty")
    if re.search(rf"^## \[{re.escape(version)}\]", changelog, re.M):
        raise ValueError("target release already has a changelog section")
    data = plistlib.loads(plist.encode())
    source_build = data.get("CFBundleVersion", "")
    if not isinstance(source_build, str) or not re.fullmatch(r"[0-9]+", source_build):
        raise ValueError("source Desktop build must be a numeric string")
    if not previous_builds or any(type(n) is not int or n < 0 for n in previous_builds):
        raise ValueError(
            "published Desktop build baselines must be nonnegative integers"
        )
    build = max(int(source_build), *previous_builds) + 1
    updated_plist = replace_plist(plist, "CFBundleShortVersionString", version)
    updated_plist = replace_plist(updated_plist, "CFBundleVersion", str(build))
    # Preserve comments, formatting and all other plist keys byte for byte.
    after = plistlib.loads(updated_plist.encode())
    expected = dict(
        data, CFBundleShortVersionString=version, CFBundleVersion=str(build)
    )
    if after != expected:
        raise ValueError("unexpected plist metadata mutation")
    old_version = project_version(project)
    # Restrict substitution to [project], not dependency/tool version fields.
    match = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", project)
    assert match is not None
    updated_project = (
        project[: match.start(1)]
        + re.sub(
            rf'(?m)^(version\s*=\s*"){re.escape(old_version)}("\s*)$',
            lambda m: m[1] + version + m[2],
            match[1],
        )
        + project[match.end(1) :]
    )
    if len(re.findall(r"^## \[Unreleased\]\s*$", changelog, re.M)) != 1:
        raise ValueError("expected exactly one Unreleased changelog heading")
    # A frozen release can publish ahead of main's metadata. Restore each missing
    # published section/reference, rather than silently comparing to an older tag.
    missing = []
    for item in re.findall(r"^## \[([^]]+)\]", published_changelog, re.M):
        if item != "Unreleased" and not re.search(
            rf"^## \[{re.escape(item)}\]", changelog, re.M
        ):
            missing.append(changelog_section(published_changelog, item))
    if not re.search(rf"^## \[{re.escape(previous_version)}\]", changelog, re.M):
        if not any(s.startswith(f"## [{previous_version}]") for s in missing):
            raise ValueError("published baseline section is missing")
    heading = f"## [{version}] — {date}\n\n{highlights.strip()}\n\n"
    changelog = re.sub(
        r"(?m)^## \[Unreleased\]\s*$",
        lambda m: m[0] + "\n\n" + heading + "\n".join(missing),
        changelog,
        count=1,
    )
    root = "https://github.com/raullenchai/Rapid-MLX/compare/"
    changelog, count = re.subn(
        r"(?m)^\[Unreleased\]:\s+\S+\s*$",
        f"[Unreleased]: {root}rapid-mac-v{version}...HEAD\n"
        f"[{version}]: {root}rapid-mac-v{previous_version}...rapid-mac-v{version}",
        changelog,
    )
    if count != 1:
        raise ValueError("expected one Unreleased comparison reference")
    for section in missing:
        item = re.search(r"^## \[([^]]+)\]", section)[1]
        refs = re.findall(rf"(?m)^\[{re.escape(item)}\]:\s+\S+", published_changelog)
        if len(refs) != 1:
            raise ValueError(f"missing published comparison reference for {item}")
        changelog += "\n" + refs[0] + "\n"
    return {
        "pyproject.toml": updated_project,
        PLIST: updated_plist,
        CHANGELOG: changelog,
        f"docs/release-notes/v{version}.md": notes.strip() + "\n",
    }


def write_metadata(root: Path, files: dict[str, str], version: str) -> None:
    notes = root / f"docs/release-notes/v{version}.md"
    if notes.exists():
        raise ValueError("refusing to overwrite existing release notes")
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    check_release_notes(version, root / CHANGELOG, root / "docs/release-notes")
