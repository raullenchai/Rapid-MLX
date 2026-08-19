"""Docs-defaults pin test (issue #2071).

The 2026-08-18 documentation dogfood found six documented defaults that
had silently drifted from the code (``--timeout`` 300 vs 1800, bench
``--num-prompts`` 5 vs 10, bench ``--max-tokens`` 256 vs 100, ...).
This test pins the specific Default cells that drifted against the real
argparse defaults, so the next default change goes red here instead of
rotting in the docs — the same pattern that keeps the
model-recommendation mirrors honest.

Deliberately surgical: it pins the flags issue #2071 fixed, not a
general docs-vs-argparse framework.
"""

import argparse
from pathlib import Path

import pytest

from vllm_mlx.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent

TIMEOUT_DOCS = [
    REPO_ROOT / "docs" / "guides" / "server.md",
    REPO_ROOT / "docs" / "reference" / "cli.md",
    REPO_ROOT / "docs" / "reference" / "configuration.md",
]
CLI_REFERENCE = REPO_ROOT / "docs" / "reference" / "cli.md"


def _subparser(name: str) -> argparse.ArgumentParser:
    """Return the ``name`` subcommand parser from the real CLI parser."""
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise AssertionError("CLI parser has no subparsers")


def _doc_default_cells(
    path: Path, flag: str, description_fragment: str | None = None
) -> list[str]:
    """Default-column cells of Markdown table rows whose first cell is *flag*.

    ``description_fragment`` narrows to rows whose Description cell
    contains it — needed for ``--max-tokens``, which appears in several
    command tables with different (correct) defaults.
    """
    cells = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        columns = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(columns) < 2 or columns[0] != f"`{flag}`":
            continue
        if description_fragment is not None and description_fragment not in columns[1]:
            continue
        cells.append(columns[-1].strip("`").strip())
    return cells


@pytest.mark.parametrize("doc", TIMEOUT_DOCS, ids=lambda p: p.name)
def test_timeout_default_matches_serve_parser(doc: Path) -> None:
    """`--timeout` Default cells in all three docs match the argparse default."""
    code_default = _subparser("serve").get_default("timeout")
    cells = _doc_default_cells(doc, "--timeout")
    assert cells, f"{doc}: no `--timeout` table row found"
    for cell in cells:
        assert float(cell) == code_default, (
            f"{doc}: documents `--timeout` default {cell!r} but "
            f"`rapid-mlx serve` uses {code_default} (vllm_mlx/cli.py)"
        )


def test_bench_num_prompts_default_matches_parser() -> None:
    """cli.md bench `--num-prompts` Default cell matches the argparse default."""
    code_default = _subparser("bench").get_default("num_prompts")
    cells = _doc_default_cells(CLI_REFERENCE, "--num-prompts")
    assert cells, f"{CLI_REFERENCE}: no `--num-prompts` table row found"
    for cell in cells:
        assert int(cell) == code_default, (
            f"{CLI_REFERENCE}: documents bench `--num-prompts` default {cell!r} "
            f"but `rapid-mlx bench` uses {code_default} (vllm_mlx/cli.py)"
        )


def test_bench_max_tokens_default_matches_parser() -> None:
    """cli.md bench `--max-tokens` Default cell matches the argparse default.

    Keyed on the bench row's Description ("Max tokens per prompt") so the
    serve and chat `--max-tokens` rows — different flags with different,
    correct defaults — stay out of scope.
    """
    code_default = _subparser("bench").get_default("max_tokens")
    cells = _doc_default_cells(
        CLI_REFERENCE, "--max-tokens", description_fragment="Max tokens per prompt"
    )
    assert cells, f"{CLI_REFERENCE}: no bench `--max-tokens` table row found"
    for cell in cells:
        assert int(cell) == code_default, (
            f"{CLI_REFERENCE}: documents bench `--max-tokens` default {cell!r} "
            f"but `rapid-mlx bench` uses {code_default} (vllm_mlx/cli.py)"
        )
