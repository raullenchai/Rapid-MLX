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
