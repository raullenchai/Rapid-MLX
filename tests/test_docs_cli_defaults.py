"""Keep the few documented CLI defaults aligned with the parser."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_SOURCE = REPO_ROOT / "rapid_mlx" / "cli_parser.py"
SERVER_GUIDE = REPO_ROOT / "docs" / "guides" / "server.md"
CLI_REFERENCE = REPO_ROOT / "docs" / "reference" / "cli.md"
CONFIG_REFERENCE = REPO_ROOT / "docs" / "reference" / "configuration.md"


def _argparse_default(subcommand: str, flag: str):
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))
    regions: list[tuple[int, str]] = []
    defaults: list[tuple[int, str, object]] = []

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
                            defaults.extend(
                                (node.lineno, name, kw.value.value) for name in names
                            )
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
    assert len(matches) == 1, f"expected one parser default for {subcommand} {flag}"
    return matches[0]


def test_server_guide_timeout_default_matches_parser() -> None:
    expected = _argparse_default("serve", "--timeout")
    rows = [
        line
        for line in SERVER_GUIDE.read_text().splitlines()
        if line.startswith("| `--timeout` |")
    ]
    assert len(rows) == 1
    assert float(rows[0].split("|")[-2].strip()) == expected


def test_references_direct_users_to_installed_help_for_defaults() -> None:
    for path in (CLI_REFERENCE, CONFIG_REFERENCE):
        text = path.read_text()
        assert "rapid-mlx serve --help" in text
        assert "curl -fsSL https://rapidmlx.com/install.sh | bash" in text
        assert not any(line.startswith("| `--") for line in text.splitlines())
