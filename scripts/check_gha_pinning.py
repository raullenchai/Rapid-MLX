#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enforce 40-char SHA pinning on GitHub Actions.

Mutable tags (``actions/checkout@v4``) are a supply-chain compromise
vector: if the action's repo is taken over, every workflow that pins by
tag automatically picks up the malicious version on the next CI run.
This bit Trivy in 2026 and is the recurring class of attack against
auto-publishing release pipelines (which Rapid-MLX is).

Every action must use a 40-char commit SHA, including official
``actions/*`` and ``github/*`` actions. Tags can be moved or replaced;
the human-readable version belongs in a trailing comment on the same
line:

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

import yaml

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# A container image digest, e.g. ``docker://ghcr.io/org/img@sha256:<64 hex>``.
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


class _LineLoader(yaml.SafeLoader):
    """SafeLoader that records the 1-based line number of every mapping value.

    Parsing the YAML (rather than regex-scanning raw text) means a quoted key
    (``"uses": ...``) or a quoted value (``uses: "actions/checkout@v4"``) is
    validated exactly like the bare form — the previous ``^\\s*uses:`` regex
    silently skipped both, leaving a mutable action reference green.
    """


def _construct_mapping(loader: _LineLoader, node: yaml.MappingNode):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        value = loader.construct_object(value_node)
        mapping[key] = value
        if key == "uses":
            # Stash the source line so a violation can point at it.
            mapping.setdefault("__uses_line__", value_node.start_mark.line + 1)
    return mapping


_LineLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _iter_uses(node: object):
    """Yield ``(uses_value, line_no)`` for every ``uses:`` mapping in the tree."""
    if isinstance(node, dict):
        if "uses" in node and isinstance(node["uses"], str):
            yield node["uses"], node.get("__uses_line__")
        for value in node.values():
            yield from _iter_uses(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_uses(item)


def _is_pinned(uses: str) -> bool:
    """True iff ``uses`` is an acceptably-immutable reference.

    * Local actions (``./path`` or ``.`` — same-repo, no supply-chain hop) pass.
    * Container actions pinned by ``@sha256:<digest>`` pass.
    * Remote ``owner/repo[/path]@<ref>`` must pin ``<ref>`` to a 40-char SHA.
    """
    if uses.startswith("./") or uses == ".":
        return True
    if uses.startswith("docker://"):
        return bool(DIGEST_RE.search(uses))
    if "@" not in uses:
        # No ref at all on a remote action — mutable by definition.
        return False
    ref = uses.rsplit("@", 1)[1]
    return bool(SHA_RE.fullmatch(ref))


def violations_in_file(path: Path) -> list[str]:
    """Return a list of human-readable violations for one workflow file."""
    out: list[str] = []
    text = path.read_text()
    try:
        documents = list(yaml.load_all(text, Loader=_LineLoader))
    except yaml.YAMLError as exc:
        return [f"{path}: unparseable YAML — cannot verify action pinning ({exc})"]
    for document in documents:
        for uses, line_no in _iter_uses(document):
            if _is_pinned(uses):
                continue
            where = f"{path}:{line_no}" if line_no else str(path)
            out.append(
                f"{where}: uses: {uses} — action must pin to a 40-char SHA "
                "(or sha256 digest for container actions), not a tag/branch"
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
        print(f"OK: {len(workflows)} workflows clean — every `uses:` is a 40-char SHA.")
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
