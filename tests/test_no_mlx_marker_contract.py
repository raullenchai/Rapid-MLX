# SPDX-License-Identifier: Apache-2.0
"""Contracts for automatic no-MLX Linux test discovery.

The Linux matrix intentionally installs Rapid-MLX without its Apple-only MLX
dependencies, then discovers the unit suite by directory. A module that needs
MLX owns that fact next to the test: it carries ``requires_mlx`` and guards
collection with ``pytest.importorskip("mlx")`` before importing MLX-bound code.
This keeps new cross-platform tests enrolled automatically without maintaining
a second list in the workflow.

The AST pass is deliberately limited to module-scope MLX imports. It does not
execute test modules and therefore remains safe in the no-MLX lane. The live
``pytest --collect-only``/unit run remains the authoritative check for
transitive imports through Rapid-MLX modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

_MLX_MODULE_PREFIXES = ("mlx", "mlx.", "mlx_lm", "mlx_core", "mlx_vlm")
_NON_UNIT_ROOTS = {
    REPO_ROOT / "tests" / "integrations",
    REPO_ROOT / "tests" / "headless_mlx",
}


def _module_scope_mlx_imports(tree: ast.Module) -> list[str]:
    """Return MLX-family names imported at module scope."""
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith(_MLX_MODULE_PREFIXES)
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith(_MLX_MODULE_PREFIXES):
                imports.append(node.module)
    return imports


def _has_module_scope_guard(tree: ast.Module) -> bool:
    """Return whether ``importorskip`` precedes every top-level MLX import."""
    mlx_import_lines = [
        node.lineno
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name.startswith(_MLX_MODULE_PREFIXES) for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith(_MLX_MODULE_PREFIXES)
        )
    ]
    if not mlx_import_lines:
        return False
    first_mlx_import = min(mlx_import_lines)
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
        if not isinstance(value, ast.Call):
            continue
        call = value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "importorskip"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
            and call.args[0].value.startswith(("mlx", "mlx_lm", "mlx_vlm"))
            and node.lineno < first_mlx_import
        ):
            return True
    return False


def _is_requires_mlx_marker(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "requires_mlx"


def _has_module_scope_requires_mlx_marker(tree: ast.Module) -> bool:
    """Return whether the effective module-level marker includes MLX.

    Python assignment semantics matter: a later ``pytestmark = ...`` replaces
    an earlier one. Inspect only the final assignment so a duplicate cannot
    silently erase ``requires_mlx`` and put the module back into Linux CI.
    """
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "pytestmark"
    ]
    if not assignments:
        return False
    value = assignments[-1].value
    if _is_requires_mlx_marker(value):
        return True
    if isinstance(value, (ast.List, ast.Tuple)):
        return any(_is_requires_mlx_marker(item) for item in value.elts)
    return False


def _ordinary_test_files() -> list[Path]:
    """Return every test module owned by the automatic Linux unit surface."""
    files: list[Path] = []
    for path in sorted((REPO_ROOT / "tests").rglob("test_*.py")):
        if any(root in path.parents for root in _NON_UNIT_ROOTS):
            continue
        files.append(path)
    return files


def _lint_one(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    imports = _module_scope_mlx_imports(tree)
    if not imports:
        return []

    problems: list[str] = []
    if not _has_module_scope_guard(tree):
        problems.append(
            f"{path.relative_to(REPO_ROOT)}: top-level MLX import(s) {imports} "
            "must be preceded by pytest.importorskip('mlx') so collection "
            "succeeds when MLX is unavailable"
        )
    if not _has_module_scope_requires_mlx_marker(tree):
        problems.append(
            f"{path.relative_to(REPO_ROOT)}: MLX-bound module must declare "
            "pytest.mark.requires_mlx"
        )
    return problems


def _linux_run_text() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["test-matrix"]["steps"]
    return next(
        step["run"]
        for step in steps
        if step.get("name") == "Run unit tests (no MLX required)"
    )


def test_contract_rules_are_loaded() -> None:
    ini = (REPO_ROOT / "pytest.ini").read_text()
    assert "requires_mlx" in ini
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text()
    assert "requires_mlx" in conftest
    assert "HAS_MLX" in conftest
    assert "pytest_collection_modifyitems" in conftest


def test_every_direct_mlx_import_is_guarded_and_marked() -> None:
    files = _ordinary_test_files()
    assert len(files) > 500, "automatic test discovery unexpectedly collapsed"
    problems = [problem for path in files for problem in _lint_one(path)]
    assert not problems, "no-MLX collection contract violations:\n" + "\n".join(
        problems
    )


def test_module_level_pytestmark_is_never_reassigned() -> None:
    problems: list[str] = []
    for path in _ordinary_test_files():
        tree = ast.parse(path.read_text())
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
        ]
        if len(assignments) > 1:
            problems.append(
                f"{path.relative_to(REPO_ROOT)}: duplicate pytestmark "
                "assignments overwrite earlier markers"
            )
    assert not problems, "pytest marker override hazards:\n" + "\n".join(problems)


def test_linux_workflow_uses_directories_not_a_file_roster() -> None:
    run = _linux_run_text()
    assert "pytest \\\n  tests \\" in run
    assert "--ignore=tests/integrations" in run
    assert "--ignore=tests/headless_mlx" in run
    assert "pytest \\\n  tests/headless_mlx \\" in run
    assert "tests/test_" not in run
    assert (
        '-m "not requires_mlx and not real_hf_cache and not requires_network '
        'and not slow and not integration and not needle"' in run
    )


def test_contract_recognizes_guard_and_composed_marker() -> None:
    tree = ast.parse(
        "import pytest\n"
        "pytest.importorskip('mlx')\n"
        "pytestmark = [pytest.mark.property, pytest.mark.requires_mlx]\n"
        "import mlx.core as mx\n"
    )
    assert _module_scope_mlx_imports(tree) == ["mlx.core"]
    assert _has_module_scope_guard(tree)
    assert _has_module_scope_requires_mlx_marker(tree)

    overridden = ast.parse(
        "import pytest\n"
        "pytestmark = pytest.mark.requires_mlx\n"
        "pytestmark = pytest.mark.slow\n"
    )
    assert not _has_module_scope_requires_mlx_marker(overridden)

    late_guard = ast.parse(
        "import pytest\n"
        "import mlx.core as mx\n"
        "pytest.importorskip('mlx')\n"
        "pytestmark = pytest.mark.requires_mlx\n"
    )
    assert not _has_module_scope_guard(late_guard)
