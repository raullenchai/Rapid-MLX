# SPDX-License-Identifier: Apache-2.0
"""Contract: the no-MLX CI leg never collects a test module that crashes on
an unguarded top-level ``import mlx``.

The Linux test-matrix step ("Run unit tests (no MLX required)") runs a roster
of tests on a host where mlx (the Apple-Silicon runtime) is NOT installed. Its
comment demands every collected module be no-MLX-safe. Over time that roster is
hand-edited, and a file dropped into it that does an unguarded top-level
``import mlx`` (directly, or via ``mlx_lm`` / ``mlx_core`` / ``mlx_vlm``) makes
the whole leg fail at collection. This test is the machine-readable tripwire
for that exact failure mode.

It is deliberately an AST scan rather than an import check: importing every
roster file in-process to see whether it pulls mlx would itself require mlx on
the Apple side and would run arbitrary test-module code during collection of
THIS suite. An AST pass is pure-stdlib, runs on the no-MLX leg, and inspects
only module-scope imports — the ones that crash collection. Imports nested
inside function bodies cannot crash the leg at collection (they only run when
their test runs, and such tests guard themselves), so they are intentionally
not flagged: that keeps the contract free of false positives.

The marker mechanism interacts with this as the companion piece:
``tests/conftest.py`` auto-skips any test marked ``requires_mlx`` when mlx is
absent. A module that imports mlx LAZILY (inside functions) and is marked
``requires_mlx`` therefore drops out cleanly on the no-MLX leg. But a module
with an UNGUARDED top-level ``import mlx`` fails collection *before* the
conftest skip can run, so such a module must instead protect itself with a
top-level ``pytest.importorskip("mlx")`` (which short-circuits collection) or
simply not be collected on the no-MLX leg. This test enforces those rules.

Pure-pytest, Linux-friendly, no MLX import (stdlib ``ast`` only).
"""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.train_gates_parser import parse_linux_pytest_args

REPO_ROOT = Path(__file__).resolve().parents[1]

# Any top-level import whose base module is (or begins with) one of these is a
# no-MLX-leg import hazard. ``mlx_*`` and ``mlx_lm`` cover the MLX family; a
# bare ``mlx`` import is the canonical case. Everything else (`transformers`,
# `numpy`, `vllm_mlx` itself) does not require mlx at import time.
_MLX_MODULE_PREFIXES = ("mlx", "mlx.", "mlx_lm", "mlx_core", "mlx_vlm")


def _module_scope_mlx_imports(tree: ast.AST) -> list[str]:
    """Return the mlx-bound names imported at MODULE scope (top level)."""
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(_MLX_MODULE_PREFIXES):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith(_MLX_MODULE_PREFIXES):
                imports.append(node.module)
    return imports


def _has_module_scope_guard(tree: ast.AST) -> bool:
    """True if the module self-guards at top level with importorskip('mlx').

    ``pytest.importorskip("mlx")`` at module scope aborts collection cleanly
    when mlx is absent, so a module that calls it (before any unguarded mlx
    import) is no-MLX-leg-safe even though it references mlx. We match the
    exact idiomatic form ``pytest.importorskip("mlx")`` at top level.
    """
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "importorskip"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in ("mlx", "mlx.core")
            ):
                return True
    return False


def _has_module_scope_requires_mlx_marker(tree: ast.AST) -> bool:
    """True if the module carries ``pytestmark = pytest.mark.requires_mlx``.

    A module that marks itself ``requires_mlx`` but still does an unguarded
    top-level ``import mlx`` will still fail COLLECTION on the no-MLX leg (the
    conftest skip runs after collection), so this exemption is only meaningful
    when combined with a lazy import or a self-guard. We still treat it as an
    explicit author signal and exempt it — mirroring the "must be marked
    requires_mlx" escape hatch the design documents — but the stronger,
    actually-correct protection for a top-level importer is ``importorskip``.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
            if not (isinstance(target, ast.Name) and target.id == "pytestmark"):
                continue
            # ``pytestmark = pytest.mark.requires_mlx``
            if isinstance(value, ast.Attribute) and value.attr == "requires_mlx":
                return True
            # ``pytestmark = [pytest.mark.requires_mlx]`` (a marker list)
            if (
                isinstance(value, ast.List)
                and value.elts
                and all(
                    isinstance(e, ast.Attribute) and e.attr == "requires_mlx"
                    for e in value.elts
                )
            ):
                return True
    return False


def _no_mlx_leg_test_files() -> list[Path]:
    """The concrete test paths the no-MLX CI leg actually collects.

    Sourced from ci.yml via the shared parser rather than hardcoded here, so a
    change to the roster (adding/removing a file, or collapsing a block) is
    automatically picked up — the same single source of truth the local
    ``train_gates.sh`` Gate 1 uses. Keeping this contract on the LEG's actual
    surface (not the whole tests/ tree) is what makes it free of false
    positives: the thousands of mlx-bound tests that never run on the no-MLX
    leg are intentionally out of scope here.
    """
    files: list[Path] = []
    for invocation in parse_linux_pytest_args():
        for token in invocation["paths"]:
            # Strip any ``::Class::test`` selector down to the file path.
            file_part = token.split("::", 1)[0]
            files.append(REPO_ROOT / file_part)
    return files


def _lint_one(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    problems: list[str] = []
    mlx_imports = _module_scope_mlx_imports(tree)
    if not mlx_imports:
        return problems
    guarded = _has_module_scope_guard(tree)
    marked = _has_module_scope_requires_mlx_marker(tree)
    if not (guarded or marked):
        problems.append(
            f"{path.name}: unguarded top-level mlx import(s) {mlx_imports}; "
            "this crashes collection on the no-MLX Linux leg. Guard with a "
            'top-level ``pytest.importorskip("mlx")`` before the import, or '
            "mark the module ``pytestmark = pytest.mark.requires_mlx`` "
            "(lazy imports + marker), or take it off the no-MLX roster."
        )
    return problems


def test_contract_rules_are_loaded() -> None:
    # Guard the marker + conftest mechanism exists; if the marker were renamed
    # or the conftest auto-skip removed, this contract still names the escape
    # hatches consistently and should fail loudly rather than silently rot.
    ini = (REPO_ROOT / "pytest.ini").read_text()
    assert "requires_mlx" in ini
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text()
    assert "requires_mlx" in conftest
    assert "HAS_MLX" in conftest
    assert "pytest_collection_modifyitems" in conftest


def test_no_mlx_leg_contract() -> None:
    # Every file the no-MLX leg collects must be no-MLX-safe at module scope:
    # either it imports nothing mlx-bound at top level, or it self-guards via
    # importorskip / declares requires_mlx. A violation is a latent no-MLX-leg
    # collection crash — this test turns it into a hard PR-time failure.
    files = _no_mlx_leg_test_files()
    assert files, "no-MLX leg surfaced no test files; ci.yml layout drifted"
    problems: list[str] = []
    for path in files:
        assert path.is_file(), f"no-MLX leg references missing file: {path}"
        try:
            problems.extend(_lint_one(path))
        except SyntaxError as exc:  # pragma: no cover - defensive
            problems.append(f"{path}: could not parse ({exc})")
    assert not problems, "no-MLX leg contract violations:\n" + "\n".join(problems)


def test_contract_handles_marked_and_guarded_patterns() -> None:
    # Meta-test: exercise the two exemption paths on tiny synthetic snippets so
    # the AST logic is pinned (a regression in the scanner itself cannot be
    # masked by an empty real-world roster).
    guarded = "import pytest\npytest.importorskip('mlx')\nimport numpy as np\n"
    tree = ast.parse(guarded)
    assert _module_scope_mlx_imports(tree) == []
    # Clearing: importorskip itself is not an import node; a bare mlx import
    # after a guard is what the guard protects.
    mixed = "import pytest\npytest.importorskip('mlx')\nimport mlx.core as mc\n"
    tree = ast.parse(mixed)
    assert _module_scope_mlx_imports(tree) == ["mlx.core"]
    assert _has_module_scope_guard(tree) is True
    assert _has_module_scope_requires_mlx_marker(tree) is False

    marked = "import pytest\npytestmark = pytest.mark.requires_mlx\n"
    tree = ast.parse(marked)
    assert _has_module_scope_requires_mlx_marker(tree) is True

    plain = "import os\n"
    tree = ast.parse(plain)
    assert _module_scope_mlx_imports(tree) == []
    assert _has_module_scope_guard(tree) is False
    assert _has_module_scope_requires_mlx_marker(tree) is False
