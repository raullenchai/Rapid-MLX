#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate a candidate commit subject would auto-release cleanly.

The ``auto-release.yml`` workflow watches for a strict subject:

    chore: bump version to X.Y.Z

GitHub's default squash-merge appends ``(#NN)`` to the subject unless
the merger passes ``--subject``. That suffix breaks the regex match and
strands the version between commit-to-main and PyPI/Homebrew publish.

This script is the structural belt-and-suspenders: run it on the bump
PR's title at PR-time, and the workflow refuses to merge until the
title is auto-release-compatible.

Usage:
    python3 scripts/validate_release_subject.py --subject "<text>"

Exit 0 = OK, exit 1 = would not auto-release (with reason on stderr).
"""

from __future__ import annotations

import argparse
import re
import sys

# Mirror auto-release.yml's regex EXACTLY. If you tweak one, tweak both.
SUBJECT_RE = re.compile(r"^chore: bump version to \d+\.\d+\.\d+$")


def diagnose(subject: str) -> list[str]:
    """Return a list of human-readable problems with the subject.

    Empty list means the subject would auto-release.
    """
    problems: list[str] = []
    if not subject:
        problems.append("subject is empty")
        return problems
    if SUBJECT_RE.fullmatch(subject):
        return problems

    if re.search(r"\(#\d+\)\s*$", subject):
        problems.append(
            "subject has a `(#NN)` PR-number suffix — GitHub's default "
            'squash-merge added it. Pass `--subject "chore: bump version '
            'to X.Y.Z"` to `gh pr merge` to strip it.'
        )
    if not subject.startswith("chore: bump version to "):
        problems.append(
            "subject does not start with the literal `chore: bump version to ` prefix"
        )
    if not re.search(r"\b\d+\.\d+\.\d+\b", subject):
        problems.append("subject contains no X.Y.Z version number")
    if "\n" in subject:
        problems.append(
            "subject contains a newline — only the first line is the subject"
        )
    if subject != subject.strip():
        problems.append("subject has leading/trailing whitespace")
    if not problems:
        problems.append(
            "subject doesn't match the auto-release regex "
            f"`{SUBJECT_RE.pattern}` for an unknown reason"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--subject",
        required=True,
        help="The candidate commit subject (typically the bump PR title).",
    )
    args = p.parse_args(argv)

    problems = diagnose(args.subject)
    if not problems:
        print(f"OK: subject would auto-release: {args.subject!r}")
        return 0
    print(f"FAIL: subject would NOT auto-release: {args.subject!r}", file=sys.stderr)
    for prob in problems:
        print(f"  - {prob}", file=sys.stderr)
    print(
        "\nFix: rename the PR to exactly `chore: bump version to X.Y.Z` and, "
        "at merge time, use:\n"
        "  gh pr merge <PR#> --repo raullenchai/Rapid-MLX --squash "
        '--subject "chore: bump version to X.Y.Z" --delete-branch',
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
