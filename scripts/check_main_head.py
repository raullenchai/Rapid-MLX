#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live main-head identity gate for the RC tag claim (#2301).

The tag must be claimed at the commit that BOTH (a) was validated by the desktop
candidate lane, AND (b) is still the live ``refs/heads/main`` head at the moment
of tagging. This prevents a second reproduction of the rc1/ordering regression:
version-bump commit A starts the candidate build, packaging fix B lands on main
while A is still validating, A passes its own gates and would otherwise be tagged
despite now being behind the intended validated head B.

The caller resolves ``refs/heads/main`` from the GitHub API (peeling annotated
tags is unnecessary — main is a branch, so the ref points straight at a commit).
This helper only evaluates the three SHAs and fails closed on any disagreement or
malformed value, so it is fully testable offline:

  * live main head == accepted candidate SHA == release SHA  -> pass
  * main advanced past the accepted/release commit (A then B) -> fail (not yet
    merged candidate)
  * malformed or non-identical SHAs / query failure            -> fail closed
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


class MainHeadGateError(Exception):
    """Raised on any fail-closed main-head condition."""


def _is_full_sha(value: str) -> bool:
    return len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def check_live_head(*, main_sha: str, accepted_sha: str, release_sha: str) -> list[str]:
    """Verify main_sha == accepted_sha == release_sha; return evidence or raise."""

    for label, value in (
        ("main", main_sha),
        ("accepted", accepted_sha),
        ("release", release_sha),
    ):
        if not isinstance(value, str) or not _is_full_sha(value):
            raise MainHeadGateError(
                f"{label} SHA must be a full 40-character commit, got {value!r}"
            )
    if not (main_sha == accepted_sha == release_sha):
        raise MainHeadGateError(
            "live main head is no longer the validated candidate: "
            f"main={main_sha} accepted={accepted_sha} release={release_sha}. "
            "A newer commit may have landed on main while the candidate was "
            "validating — refusing to tag a candidate that is behind the "
            "intended validated head (#2301). Re-trigger on the new head so the "
            "candidate and the tag share the same commit."
        )
    return [
        f"live main head == accepted == release: {main_sha}",
        "tag identity binds the validated candidate AND the current main head",
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--accepted-sha", required=True)
    parser.add_argument("--release-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = check_live_head(
            main_sha=args.main_sha,
            accepted_sha=args.accepted_sha,
            release_sha=args.release_sha,
        )
    except MainHeadGateError as exc:
        print(f"main head: {exc}", file=sys.stderr)
        return 1
    print("\n".join(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
