#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate a bump-PR title is the CANONICAL auto-release subject.

PF-1 enforces that the bump-PR **title** is exactly the canonical
subject. The post-merge detect step in ``auto-release.yml`` is
deliberately more tolerant: it runs the merged commit subject through
``release_version.py subject --allow-pr-suffix``, so GitHub's default
squash-merge appending ``(#NN)`` no longer strands a release. This
script keeps the *title* strict for HYGIENE — a canonical PR title is
clearer to review and keeps the merged subject clean when ``--subject``
is passed — but a stray ``(#NN)`` suffix on an already-merged commit is
no longer the release-killer it once was.

Usage:
    python3 scripts/validate_release_subject.py --subject "<text>"
    python3 scripts/validate_release_subject.py --subject "<text>" \
        --pr-body "<markdown>" --repository owner/repo --print-preflight-run-id

Exit 0 = OK (title is canonical), exit 1 = not canonical (with reason).
"""

from __future__ import annotations

import argparse
import re
import sys

try:
    from release_version import SUBJECT_RE
except ModuleNotFoundError:  # imported by tests as ``scripts.*`` from repo root
    from scripts.release_version import SUBJECT_RE


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
            "the BUMP-PR title must be the canonical subject WITHOUT a "
            "`(#NN)` suffix. (The post-merge detect step tolerates the suffix "
            "via --allow-pr-suffix, but PF-1 keeps the title clean. To keep the "
            'merged subject canonical too, pass `--subject "chore: bump version '
            'to X.Y.Z[-rcN]"` to `gh pr merge`.)'
        )
    if not subject.startswith("chore: bump version to "):
        problems.append(
            "subject does not start with the literal `chore: bump version to ` prefix"
        )
    if not re.search(r"\b\d+\.\d+\.\d+(?:-rc\d+)?\b", subject):
        problems.append("subject contains no X.Y.Z or X.Y.Z-rcN version number")
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


def extract_preflight_run_id(pr_body: str, repository: str) -> str:
    """Return the single exact-head pre-flight run recorded in a PR body."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError(f"invalid GitHub repository: {repository!r}")
    prefix = f"Release-Preflight: https://github.com/{repository}/actions/runs/"
    matches = [
        line.removeprefix(prefix).strip()
        for line in pr_body.splitlines()
        if line.startswith(prefix)
    ]
    if not matches:
        raise ValueError(
            f"PR body is missing the exact pre-flight evidence line: `{prefix}<run-id>`"
        )
    if len(matches) != 1:
        raise ValueError(
            "PR body must contain exactly one Release-Preflight evidence line"
        )
    run_id = matches[0]
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError(
            "Release-Preflight must be a bare GitHub Actions run URL with no suffix"
        )
    return run_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--subject",
        required=True,
        help="The candidate commit subject (typically the bump PR title).",
    )
    p.add_argument("--pr-body", help="Bump PR body containing pre-flight evidence.")
    p.add_argument("--repository", help="GitHub owner/repository for the evidence URL.")
    p.add_argument(
        "--print-preflight-run-id",
        action="store_true",
        help="Print the validated pre-flight run ID for a caller to verify live.",
    )
    args = p.parse_args(argv)

    problems = diagnose(args.subject)
    if not problems:
        if not args.print_preflight_run_id:
            print(f"OK: subject would auto-release: {args.subject!r}")
        if args.pr_body is None:
            if args.repository is not None or args.print_preflight_run_id:
                p.error("--repository/--print-preflight-run-id require --pr-body")
            return 0
        if args.repository is None:
            p.error("--pr-body requires --repository")
        try:
            run_id = extract_preflight_run_id(args.pr_body, args.repository)
        except ValueError as exc:
            p.error(str(exc))
        if args.print_preflight_run_id:
            print(run_id)
        else:
            print(f"OK: PR body records Release pre-flight run {run_id}")
        return 0
    print(f"FAIL: subject would NOT auto-release: {args.subject!r}", file=sys.stderr)
    for prob in problems:
        print(f"  - {prob}", file=sys.stderr)
    print(
        "\nFix: rename the PR to exactly `chore: bump version to X.Y.Z[-rcN]` and, "
        "at merge time, use:\n"
        "  gh pr merge <PR#> --repo raullenchai/Rapid-MLX --squash "
        '--subject "chore: bump version to X.Y.Z[-rcN]" --delete-branch',
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
