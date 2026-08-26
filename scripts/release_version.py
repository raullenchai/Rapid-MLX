#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and compare Rapid release versions and bump subjects.

The source version is SemVer-shaped and may be either a stable ``X.Y.Z`` or
an RC ``X.Y.Z-rcN``.  Packaging tools normalize the latter to PEP 440's
``X.Y.ZrcN`` in artifact filenames, while Git tags and the desktop bundle keep
the human-facing hyphenated spelling.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

VERSION_PATTERN = (
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-rc(?:[1-9][0-9]*))?"
)
VERSION_RE = re.compile(rf"^(?P<version>{VERSION_PATTERN})$")
SUBJECT_RE = re.compile(rf"^chore: bump version to (?P<version>{VERSION_PATTERN})$")
SUBJECT_WITH_PR_RE = re.compile(
    rf"^chore: bump version to (?P<version>{VERSION_PATTERN})(?: \(#[0-9]+\))?$"
)


@dataclass(frozen=True, order=True)
class ReleaseVersion:
    major: int
    minor: int
    patch: int
    stability: int
    rc: int


def parse_version(value: str) -> ReleaseVersion:
    if VERSION_RE.fullmatch(value) is None:
        raise ValueError(f"invalid release version: {value!r}")
    core, separator, rc_text = value.partition("-rc")
    major, minor, patch = (int(part) for part in core.split("."))
    return ReleaseVersion(
        major,
        minor,
        patch,
        0 if separator else 1,  # every RC sorts before its stable release
        int(rc_text) if separator else 0,
    )


def version_from_subject(subject: str, *, allow_pr_suffix: bool = False) -> str:
    pattern = SUBJECT_WITH_PR_RE if allow_pr_suffix else SUBJECT_RE
    match = pattern.fullmatch(subject)
    if match is None:
        raise ValueError(f"invalid release subject: {subject!r}")
    return match.group("version")


def preceding_stable_tag(intended: str, existing_tags: Sequence[str]) -> str | None:
    """Greatest existing stable ``rapid-mac-v*`` tag strictly below ``intended``.

    The desktop DMG size-delta gate compares against the immediately previous
    STABLE rapid-mac release, so:

      * RCs are never a predecessor (they sort before their own stable, and the
        intended pre-tag tag does not exist yet);
      * the intended tag itself is excluded even if a same-version stable tag
        already exists (``strictly below``);
      * ordering uses :class:`ReleaseVersion` (RC < stable), never GNU ``sort
        -V``, so this is portable to macOS runners.

    Returns ``None`` when no stable tag sorts below ``intended`` (no baseline —
    callers then truthfully record that the DMG size-delta gate was skipped).
    """

    target = parse_version(intended)
    best: str | None = None
    best_version: ReleaseVersion | None = None
    for tag in existing_tags:
        if not isinstance(tag, str) or not tag.startswith("rapid-mac-v"):
            continue
        core = tag[len("rapid-mac-v") :]
        if "-rc" in core:  # predecessor is always a stable release
            continue
        try:
            candidate = parse_version(core)
        except ValueError:
            continue
        if candidate < target and (best_version is None or candidate > best_version):
            best, best_version = tag, candidate
    return best


def preceding_release_tag(intended: str, existing_tags: Sequence[str]) -> str | None:
    """Greatest existing ``rapid-mac-v*`` tag strictly below ``intended``.

    Unlike :func:`preceding_stable_tag`, this INCLUDES RC tags (ordered RC <
    stable via :class:`ReleaseVersion`). It is the Sparkle monotonic build
    version predecessor: Sparkle rejects an update whose ``CFBundleVersion``
    does not strictly exceed the build of every already-released app, including
    prior RCs of the same line. So 0.13.0-rc2 must beat 0.13.0-rc1 (not just the
    last stable 0.12.18), and a stable 0.13.0 must beat the latest 0.13.0-rcN.

    Returns ``None`` when no release tag sorts below ``intended``.
    """

    target = parse_version(intended)
    best: str | None = None
    best_version: ReleaseVersion | None = None
    for tag in existing_tags:
        if not isinstance(tag, str) or not tag.startswith("rapid-mac-v"):
            continue
        core = tag[len("rapid-mac-v") :]
        try:
            candidate = parse_version(core)
        except ValueError:
            continue
        if candidate < target and (best_version is None or candidate > best_version):
            best, best_version = tag, candidate
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("version")

    greater = subparsers.add_parser("greater")
    greater.add_argument("base")
    greater.add_argument("candidate")

    subject = subparsers.add_parser("subject")
    subject.add_argument("subject")
    subject.add_argument("--allow-pr-suffix", action="store_true")

    preceding = subparsers.add_parser(
        "preceding",
        help=(
            "given the intended release version and existing rapid-mac-v* tags "
            "on stdin (one per line), print the greatest STABLE tag strictly "
            "below the intended version (the DMG size-delta baseline; RCs have "
            "no published DMG asset), or nothing"
        ),
    )
    preceding.add_argument("version")

    preceding_release = subparsers.add_parser(
        "preceding-release",
        help=(
            "given the intended release version and existing rapid-mac-v* tags "
            "on stdin (one per line), print the greatest prior release tag "
            "strictly below it INCLUDING RCs (the Sparkle monotonic-CFBundleVersion "
            "predecessor: rc2 must beat rc1, a stable must beat its latest rc), "
            "or nothing"
        ),
    )
    preceding_release.add_argument("version")

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            parse_version(args.version)
            print(args.version)
        elif args.command == "greater":
            if parse_version(args.candidate) <= parse_version(args.base):
                raise ValueError(
                    f"candidate {args.candidate!r} is not greater than {args.base!r}"
                )
            print(args.candidate)
        elif args.command == "preceding":
            tags = [line.strip() for line in sys.stdin if line.strip()]
            best = preceding_stable_tag(args.version, tags)
            if best is not None:
                print(best)
        elif args.command == "preceding-release":
            tags = [line.strip() for line in sys.stdin if line.strip()]
            best = preceding_release_tag(args.version, tags)
            if best is not None:
                print(best)
        else:
            print(
                version_from_subject(args.subject, allow_pr_suffix=args.allow_pr_suffix)
            )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
