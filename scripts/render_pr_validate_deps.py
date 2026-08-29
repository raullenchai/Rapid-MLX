#!/usr/bin/env python3
"""Render pr-validate's CI test-dependency set from the declared `[ci-linux]` extra.

pr-validate.yml used to hand-copy a ``pip install pytest ... transformers tomli-w``
line. That is the same drift bug #2489 fixes for ci.yml, but without even the
synced-requirements protection — if a dependency (or a version floor) changed in
pyproject.toml, nobody updated the copy and pr_validate ran against a stale set.

This script regenerates ``config/pr-validate-deps.txt`` from the canonical
``[ci-linux]`` optional-dependency extra using jinja2, and (with ``--check``)
fails closed if the committed file doesn't match the render. pr-validate.yml
consumes it via ``pip install --requirement``, so the ad hoc list is gone and
the test deps live in exactly one declared place.

The pipeline's OWN tools (``pip-audit`` / ``ruff`` / ``rich`` / ``mcp``) and the
runtime ``tomli-w`` (agents/adapter.py config merge) are NOT test deps and stay
explicitly pinned in pr-validate.yml, not in ``[ci-linux]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jinja2 import BaseLoader, Environment

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
OUT = ROOT / "config" / "pr-validate-deps.txt"

TEMPLATE = """\
# GENERATED FILE — do not edit.
# Rendered from pyproject.toml ``[project.optional-dependencies].ci-linux`` by
# scripts/render_pr_validate_deps.py. pr-validate.yml installs this set so the
# CI pipeline runs against exactly the declared test dependencies (no ad hoc
# list). Re-run ``python scripts/render_pr_validate_deps.py`` after changing
# ``[ci-linux]``; ``--check`` (wired into pr-validate CI) fails on a stale copy.
{% for dep in ci_linux_deps %}{{ dep }}
{% endfor %}"""


def _ci_linux_extra() -> list[str]:
    """Read the canonical ``[ci-linux]`` extra on every supported Python.

    ``tomllib`` is stdlib on 3.11+ only; this script also runs on 3.10 (the
    test-matrix includes it), where the declared CI dependency set installs
    the API-compatible ``tomli`` backport.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]["ci-linux"]


def render() -> str:
    env = Environment(loader=BaseLoader, keep_trailing_newline=True)
    template = env.from_string(TEMPLATE)
    return template.render(ci_linux_deps=_ci_linux_extra())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="fail (exit 1) if the committed file differs from the render",
    )
    args = ap.parse_args()

    rendered = render()
    if args.check:
        if OUT.read_text() != rendered:
            sys.stderr.write(
                f"❌ {OUT.relative_to(ROOT)} is stale. Re-run "
                f"`python scripts/render_pr_validate_deps.py` so it matches the "
                f"declared `[ci-linux]` extra.\n"
            )
            return 1
        print(f"✓ {OUT.relative_to(ROOT)} in sync with [ci-linux].")
        return 0

    OUT.write_text(rendered)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
