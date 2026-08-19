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

The defaults are read from ``vllm_mlx/cli.py`` SOURCE via ``ast`` —
never by importing it. ``vllm_mlx.cli`` transitively imports mlx and
probes the host at import time, which breaks collection on the Linux
validation runner; source-level extraction runs identically everywhere
(the same convention as ``test_no_mllm_flag.py``).
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_SOURCE = REPO_ROOT / "vllm_mlx" / "cli.py"

TIMEOUT_DOCS = [
    REPO_ROOT / "docs" / "guides" / "server.md",
    REPO_ROOT / "docs" / "reference" / "cli.md",
    REPO_ROOT / "docs" / "reference" / "configuration.md",
]
CLI_REFERENCE = REPO_ROOT / "docs" / "reference" / "cli.md"


def _argparse_default(subcommand: str, flag: str):
    """Literal ``default=`` of *flag* inside *subcommand*'s parser region.

    Subcommand regions are delimited by the ``add_parser("<name>", ...)``
    call sites in source order; an ``add_argument`` call belongs to the
    most recent ``add_parser`` above it. Only constant defaults are
    supported — that is all the pinned flags use.
    """
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))

    regions: list[tuple[int, str]] = []  # (lineno, subcommand name)
    defaults: list[tuple[int, str, object]] = []  # (lineno, flag, default)

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "add_parser":
                    names = [
                        a.value
                        for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    ]
                    if names:
                        regions.append((node.lineno, names[0]))
                elif func.attr == "add_argument":
                    names = [
                        a.value
                        for a in node.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    ]
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            for name in names:
                                defaults.append((node.lineno, name, kw.value.value))
            self.generic_visit(node)

    Visitor().visit(tree)
    regions.sort()

    matches = []
    for lineno, name, value in defaults:
        if name != flag:
            continue
        owner = None
        for region_line, region_name in regions:
            if region_line <= lineno:
                owner = region_name
            else:
                break
        if owner == subcommand:
            matches.append(value)
    assert matches, f"no constant default for {flag} in the {subcommand} parser"
    assert len(matches) == 1, f"ambiguous {flag} in {subcommand}: {matches}"
    return matches[0]


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
    code_default = _argparse_default("serve", "--timeout")
    cells = _doc_default_cells(doc, "--timeout")
    assert cells, f"{doc}: no `--timeout` table row found"
    for cell in cells:
        assert float(cell) == code_default, (
            f"{doc}: documents `--timeout` default {cell!r} but "
            f"`rapid-mlx serve` uses {code_default} (vllm_mlx/cli.py)"
        )


def test_bench_num_prompts_default_matches_parser() -> None:
    """cli.md bench `--num-prompts` Default cell matches the argparse default."""
    code_default = _argparse_default("bench", "--num-prompts")
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
    code_default = _argparse_default("bench", "--max-tokens")
    cells = _doc_default_cells(
        CLI_REFERENCE, "--max-tokens", description_fragment="Max tokens per prompt"
    )
    assert cells, f"{CLI_REFERENCE}: no bench `--max-tokens` table row found"
    for cell in cells:
        assert int(cell) == code_default, (
            f"{CLI_REFERENCE}: documents bench `--max-tokens` default {cell!r} "
            f"but `rapid-mlx bench` uses {code_default} (vllm_mlx/cli.py)"
        )
