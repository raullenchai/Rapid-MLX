# SPDX-License-Identifier: Apache-2.0
"""Detect test modules that cannot be imported on Python 3.10.

The opt-in chat harness depends on ``pydantic_ai``, which is installed only on
Python 3.11+ (see the ``pydantic-ai-slim`` marker in ``pyproject.toml``).
``tests/conftest.py`` uses :func:`skip_on_python310` on Python 3.10 to skip a
test file *before* importing it, so a new harness test cannot break the 3.10
CI leg by being missing from a hand-maintained list. A minimal escape hatch
covers tests that only reach the harness at runtime.

Rules:

* Only imports executed while the module is imported count: the module body,
  including ``if``/``try``/``with``/``for`` blocks and class bodies. Imports in
  function bodies and lambdas are deferred and ignored.
* ``if TYPE_CHECKING:`` blocks are ignored, and so are imports inside a ``try``
  whose handlers catch ``ImportError``/``ModuleNotFoundError`` (or a broader
  ``Exception``/bare ``except``): such a module already copes with the
  dependency being absent.
* Imports of modules inside this repository (``rapid_mlx``, ``tests``, ...) are
  followed transitively, including the parent packages' ``__init__`` files, so
  importing a harness submodule that pulls in ``pydantic_ai`` counts, while a
  3.10-safe sibling submodule does not.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level third-party modules whose distributions are installed only on
# Python >= 3.11 and that project/test code imports at module scope.
PY311_ONLY_MODULES: tuple[str, ...] = ("pydantic_ai",)

# These files import only 3.10-safe modules at import time but exercise the
# harness at runtime: lazy model_client imports or owner subprocesses need
# pydantic_ai/openai. Import scanning cannot detect these; keep the set minimal.
RUNTIME_ONLY_TEST_FILES: frozenset[str] = frozenset(
    {"test_chat_harness_client.py", "test_chat_runtime_owner.py"}
)

_GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _is_py311_only(name: str) -> bool:
    return any(
        name == root or name.startswith(root + ".") for root in PY311_ONLY_MODULES
    )


def _module_file(name: str) -> Path | None:
    """Return the in-repo source file for a dotted module name, if any."""
    base = REPO_ROOT.joinpath(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _package_of(path: Path) -> str | None:
    """Dotted package name containing ``path`` (for relative imports)."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _handler_guards_import(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for node in types:
        name = (
            node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
        )
        if name in _GUARD_EXCEPTIONS:
            return True
    return False


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _executed_imports(tree: ast.AST, package: str | None) -> list[str]:
    """Absolute module names imported while the module body executes."""
    found: list[str] = []

    def resolve_from(node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module or ""
        if package is None:
            return None
        parts = package.split(".")
        if node.level - 1 > len(parts):
            return None
        base = parts[: len(parts) - (node.level - 1)]
        if node.module:
            base.append(node.module)
        return ".".join(base)

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Try) and any(
            _handler_guards_import(h) for h in node.handlers
        ):
            # Imports in a guarded try body tolerate the module being missing.
            for child in [*node.handlers, *node.orelse, *node.finalbody]:
                visit(child)
            return
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
            return
        if isinstance(node, ast.ImportFrom):
            module = resolve_from(node)
            if module is None:
                return
            found.append(module)
            # ``from pkg import sub`` may import the submodule ``pkg.sub``.
            found.extend(
                f"{module}.{alias.name}" for alias in node.names if alias.name != "*"
            )
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def _with_parents(names: list[str]) -> set[str]:
    # Importing ``a.b.c`` also executes ``a/__init__`` and ``a/b/__init__``.
    out: set[str] = set()
    for name in names:
        parts = name.split(".")
        out.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return out


@cache
def _module_imports(name: str) -> frozenset[str]:
    """Modules (with parents) imported while in-repo module ``name`` executes."""
    path = _module_file(name)
    if path is None:
        return frozenset()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return frozenset()
    return frozenset(_with_parents(_executed_imports(tree, _package_of(path))))


def requires_python311(source: str, path: Path | None = None) -> bool:
    """Return True if importing ``source`` would import a 3.11-only module.

    ``path`` (the file ``source`` came from) is used to resolve relative imports.
    """
    package = _package_of(path) if path is not None else None
    pending = list(_with_parents(_executed_imports(ast.parse(source), package)))
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if _is_py311_only(name):
            return True
        pending.extend(_module_imports(name) - seen)
    return False


def skip_on_python310(path: Path) -> bool:
    """Return whether this test file must be skipped on Python 3.10."""
    return path.name in RUNTIME_ONLY_TEST_FILES or requires_python311(
        path.read_text(encoding="utf-8"), path
    )
