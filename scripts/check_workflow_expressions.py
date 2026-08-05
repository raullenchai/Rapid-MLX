#!/usr/bin/env python3
"""Reject GitHub Actions workflows containing an invalid ``${{ }}`` expression.

GitHub evaluates ``${{ ... }}`` in workflow VALUES — including inside ``run:``
scripts, where a leading ``#`` is a *shell* comment and therefore still gets
interpolated. An expression that does not parse invalidates the whole
workflow: the run fails instantly with ZERO jobs and no log, so it never
surfaces as a failing check on the PR that introduced it.

That is how a ``run:`` comment reading "never passes through `${{ }}`" took
the release pipeline down — an empty expression is a syntax error.

Scans parsed YAML values only, so genuine YAML comments (which the parser
drops before Actions ever sees them) are correctly ignored. Deliberately
narrow: it checks the one mistake that silently breaks an entire workflow.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.S)


def walk(node):
    """Yield every string scalar in a parsed YAML document."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from walk(k)
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def problems(text: str) -> list[str]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"YAML does not parse: {exc}"]

    found: list[str] = []
    for s in walk(doc):
        for m in EXPR.finditer(s):
            if not m.group(1).strip():
                found.append(
                    "empty expression `${{ }}` — invalid, breaks the whole workflow"
                )
        # An unclosed opener is equally fatal.
        if s.count("${{") > len(EXPR.findall(s)):
            found.append("unclosed `${{` — invalid, breaks the whole workflow")
    return found


def candidate_lines(text: str) -> list[int]:
    """Raw-text lines holding a literal ``${{ }}``.

    Detection is done on parsed YAML (authoritative); this is only to point a
    human at the right place. A genuine YAML comment can appear here too, so
    all candidates are listed rather than guessing which one is fatal.
    """
    return [
        i
        for i, line in enumerate(text.split("\n"), 1)
        if re.search(r"\$\{\{\s*\}\}", line)
    ]


def main() -> int:
    root = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    files = sorted(root.glob("*.y*ml"))
    failed = False
    for path in files:
        text = path.read_text(encoding="utf-8")
        probs = problems(text)
        if probs:
            rel = path.relative_to(root.parent.parent)
            for why in probs:
                print(f"{rel}: {why}", file=sys.stderr)
            lines = candidate_lines(text)
            if lines:
                print(
                    f"  literal `${{{{ }}}}` appears at line(s): "
                    f"{', '.join(map(str, lines))} "
                    "(one inside a `run:` block is the fatal one; a real YAML "
                    "comment is harmless)",
                    file=sys.stderr,
                )
            failed = True
    if failed:
        print(
            "\nTo write those characters in prose, say "
            "'GitHub Actions expression interpolation' instead.",
            file=sys.stderr,
        )
        return 1
    print(f"workflow expressions OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
