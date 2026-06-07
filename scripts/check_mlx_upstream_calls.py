#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Surface dangerous module-scope MLX API calls in installed mlx-*.

Release SOP Gate 10 catches the class of bug that #404 shipped: a new
top-level ``mx.new_thread_local_stream(mx.default_device())`` lands at
module-load time in ``mlx_lm/generate.py``, works on M1-M4, crashes on
M5. The dev env's MLX is whatever's checked out; the release wheel
might pull a newer mlx-lm that has the new call, and the dev gates run
green while production users on the wrong chip family ``ImportError``.

This script does NOT diff against an "old" version — it scans the
currently-installed mlx-lm / mlx-vlm tree for ALL module-scope (i.e.
import-time) calls to a known-dangerous API set, and emits a warning
list. The CI gate fires only when ``pyproject.toml``'s mlx-* deps
change in the PR; the release-er must read the list and either:

  (a) explicitly clear it in the PR description ("audited for #404
      class — none of these call into hardware-specific paths"), or
  (b) add a ``vllm_mlx/_mlx_compat.py``-style probe-and-cache shim for
      the affected call before merging.

The script does NOT auto-fail — it surfaces. Hard-failing on every
new mx.* import-time call would block legitimate releases on
upstream's cleanup PRs. The human decision is the gate; this is the
information feed to that decision.

Usage:
    python3 scripts/check_mlx_upstream_calls.py
        # scans installed mlx_lm + mlx_vlm; prints findings + exits 0

    python3 scripts/check_mlx_upstream_calls.py --strict
        # exits 1 if any findings — for CI runs where the bump must be
        # human-reviewed before merge

Exit 0 = scan complete (or no findings in --strict).
Exit 1 = scan error (or findings in --strict).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

# API surface that has historically caused chip-family-specific failures
# at module load time. Add to this list when a new landmine is found.
#
# Format: ("<attribute path>", "<rationale>"). The attribute path is the
# AST-rendered dotted form (``mx.metal.foo``, ``mlx.core.default_device``)
# we match against any Call's func chain.
DANGEROUS_CALLS = [
    ("mx.metal", "device-introspection / chip-family-specific"),
    ("mlx.core.metal", "device-introspection / chip-family-specific"),
    ("mx.new_thread_local_stream", "single-stream M5 #404 root cause"),
    ("mlx.core.new_thread_local_stream", "single-stream M5 #404 root cause"),
    ("mx.default_device", "device-introspection at import time"),
    ("mlx.core.default_device", "device-introspection at import time"),
    ("mx.set_default_device", "mutates global device state at import"),
    ("mlx.core.set_default_device", "mutates global device state at import"),
]


def _attr_chain(node: ast.AST) -> str:
    """Render ``mx.metal.foo`` from the nested Attribute/Name AST."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _is_module_scope(parents: list[ast.AST]) -> bool:
    """A call is module-scope if no enclosing FunctionDef/ClassDef."""
    return not any(
        isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for p in parents
    )


def _walk_with_parents(node: ast.AST, parents: list[ast.AST] | None = None):
    if parents is None:
        parents = []
    yield node, parents
    for child in ast.iter_child_nodes(node):
        yield from _walk_with_parents(child, parents + [node])


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of ``(line_no, dangerous_chain, rationale)`` for path."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    findings: list[tuple[int, str, str]] = []
    for node, parents in _walk_with_parents(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_module_scope(parents):
            continue
        chain = _attr_chain(node.func)
        for needle, rationale in DANGEROUS_CALLS:
            if chain == needle or chain.startswith(needle + "."):
                findings.append((node.lineno, chain, rationale))
                break
    return findings


def scan_package(pkg_name: str) -> list[tuple[Path, int, str, str]]:
    """Scan an installed package by name; return [(path, lineno, chain, why)]."""
    spec = importlib.util.find_spec(pkg_name)
    if spec is None or spec.origin is None:
        return []
    root = Path(spec.origin).parent
    out: list[tuple[Path, int, str, str]] = []
    for py in sorted(root.rglob("*.py")):
        for line_no, chain, why in scan_file(py):
            out.append((py, line_no, chain, why))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--packages",
        nargs="+",
        default=["mlx_lm", "mlx_vlm"],
        help="Packages to scan (default: mlx_lm mlx_vlm).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any dangerous module-scope calls found.",
    )
    args = p.parse_args(argv)

    total = 0
    for pkg in args.packages:
        findings = scan_package(pkg)
        if not findings:
            print(f"OK: {pkg}: no module-scope calls into known-dangerous MLX API.")
            continue
        print(
            f"⚠  {pkg}: {len(findings)} module-scope call(s) into known-dangerous MLX API:"
        )
        for path, line_no, chain, why in findings:
            print(f"    {path}:{line_no}: {chain}()  — {why}")
        total += len(findings)

    if total == 0:
        return 0

    print(
        "\nThese are CANDIDATES for cross-chip-family review — see release "
        "workflow Gate 10. For each finding, decide:\n"
        "  (a) add an `_mlx_compat.py`-style probe-and-cache shim if it's a "
        "real landmine on M3/M4/M5, or\n"
        "  (b) explicitly clear in the PR description: '#404-audited — "
        "<file>:<line> is OK because <reason>'."
    )
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
