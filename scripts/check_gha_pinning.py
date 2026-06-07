#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enforce 40-char SHA pinning on third-party GitHub Actions.

Mutable tags (``actions/checkout@v4``) are a supply-chain compromise
vector: if the action's repo is taken over, every workflow that pins by
tag automatically picks up the malicious version on the next CI run.
This bit Trivy in 2026 and is the recurring class of attack against
auto-publishing release pipelines (which Rapid-MLX is).

Allowlist: official ``actions/*`` and ``github/*`` org actions are
permitted at tag refs because GitHub backs their immutability via a
separate guarantee. Everything else must be a 40-char commit SHA, with
the human-readable version comment on the same line:

    uses: codecov/codecov-action@1234567890abcdef1234567890abcdef12345678  # v4.5.0

Run on every PR touching ``.github/workflows/`` (gate in ci.yml).
Standalone: ``python3 scripts/check_gha_pinning.py``.

Exit 0 = all good, exit 1 = violations (printed to stderr).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Owners whose tags we trust as immutable (per GitHub's own guarantee).
TRUSTED_OWNERS = {"actions", "github"}

# uses: <owner>/<repo>[/path]@<ref>
USES_RE = re.compile(
    r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)",
    re.MULTILINE,
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def violations_in_file(path: Path) -> list[str]:
    """Return a list of human-readable violations for one workflow file."""
    out: list[str] = []
    text = path.read_text()
    for m in USES_RE.finditer(text):
        action, ref = m.group(1), m.group(2)
        owner = action.split("/", 1)[0]
        if owner in TRUSTED_OWNERS:
            continue
        if SHA_RE.fullmatch(ref):
            continue
        line_no = text[: m.start()].count("\n") + 1
        out.append(
            f"{path}:{line_no}: uses: {action}@{ref} — non-allowlisted "
            "third-party action must pin to a 40-char SHA, not a tag/branch"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--workflows-dir",
        default=".github/workflows",
        help="Directory of GitHub Actions workflow YAML files",
    )
    args = p.parse_args(argv)

    root = Path(args.workflows_dir)
    if not root.is_dir():
        print(f"FAIL: {root} is not a directory", file=sys.stderr)
        return 1

    workflows = sorted(p for p in root.iterdir() if p.suffix in {".yml", ".yaml"})
    if not workflows:
        print(f"OK: no workflows in {root}")
        return 0

    all_violations: list[str] = []
    for wf in workflows:
        all_violations.extend(violations_in_file(wf))

    if not all_violations:
        print(
            f"OK: {len(workflows)} workflows clean — every third-party "
            "`uses:` is a 40-char SHA or an allowlisted owner."
        )
        return 0

    print(
        f"FAIL: {len(all_violations)} GitHub Actions SHA-pinning violation(s):",
        file=sys.stderr,
    )
    for v in all_violations:
        print(f"  {v}", file=sys.stderr)
    print(
        "\nFix: replace the tag/branch with the commit SHA from the action's "
        "GitHub release page, keeping the tag as a trailing comment:\n"
        "  - uses: foo/bar@<40-char-sha>  # v1.2.3",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
