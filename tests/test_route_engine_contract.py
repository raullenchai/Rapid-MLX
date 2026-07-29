# SPDX-License-Identifier: Apache-2.0
"""Bug-class AST gates for ``vllm_mlx/routes/*.py`` engine access.

These gates keep route code on the public ``BaseEngine`` contract instead of
silently probing optional methods or reaching into concrete-engine internals.
They are intentionally AST-based so refactors do not bypass them.

Pattern A — ``hasattr(engine, X)`` is banned unless ``X`` is on an explicit
allowlist of genuinely optional methods.

Pattern B — ``engine.X.Y`` (two-level access) is banned unless ``X`` is
explicitly declared on ``BaseEngine``. The only ``X`` that qualifies
today is ``tokenizer``. Anything else (``engine.model``, ``engine.scheduler``,
…) is reaching into private engine internals from the route layer.

Pattern C — every method called as ``engine.METHOD(...)`` from a route
must exist on ``BaseEngine``. Catches "I added a new route helper that
calls a method only one engine implements" before it ships.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROUTES_DIR = pathlib.Path(__file__).parent.parent / "vllm_mlx" / "routes"


# Methods/properties the routes may legitimately ``hasattr``-guard. Keep
# this list small. Anything that warrants a real conditional code path
# belongs on ``BaseEngine`` with a sensible default instead.
HASATTR_ENGINE_ALLOWLIST: frozenset[str] = frozenset(
    {
        # No entries today. Route-facing methods belong on BaseEngine.
    }
)

# Two-level engine accesses (``engine.X.Y``) that are sanctioned because
# ``X`` is on the BaseEngine contract.
TWO_LEVEL_ENGINE_ROOT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "tokenizer",  # declared abstract on BaseEngine
    }
)


def _route_files() -> list[pathlib.Path]:
    return sorted(p for p in ROUTES_DIR.glob("*.py") if p.name != "__init__.py")


def _parse(file: pathlib.Path) -> ast.AST:
    return ast.parse(file.read_text(), filename=str(file))


def _base_engine_public_attrs() -> set[str]:
    """Names declared **on** ``BaseEngine`` itself that routes may use.

    Reads only ``vars(BaseEngine)`` — not the MRO — so inherited names
    from ``ABC`` (``__subclasshook__``, etc.) and ``object`` (``__init__``,
    ``__repr__``, etc.) don't accidentally become part of the "contract"
    and let route-layer typos through (codex round-1 review on PR #502).

    Includes ``@abstractmethod`` methods, ``@property`` declarations, and
    concrete methods with default implementations declared in this class.
    """
    from vllm_mlx.engine import base as base_mod

    return {name for name in vars(base_mod.BaseEngine) if not name.startswith("_")}


# ---------------------------------------------------------------------------
# Pattern A — hasattr(engine, X) guards
# ---------------------------------------------------------------------------


class _HasattrEngineVisitor(ast.NodeVisitor):
    def __init__(self):
        self.findings: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "engine"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self.findings.append((node.lineno, node.args[1].value))
        self.generic_visit(node)


@pytest.mark.parametrize("route_file", _route_files(), ids=lambda p: p.name)
def test_no_hasattr_engine_guard(route_file: pathlib.Path) -> None:
    """``hasattr(engine, "X")`` is banned unless ``X`` is on the
    allowlist. When the engine class evolves and the method goes away, such
    guards turn False and silently disable the feature.

    If a future engine genuinely doesn't support some optional feature,
    declare it on ``BaseEngine`` with a sensible default (e.g.
    ``supports_X = False``) and branch on the value, not on attribute
    existence.
    """
    visitor = _HasattrEngineVisitor()
    visitor.visit(_parse(route_file))
    illegal = [
        (line, name)
        for line, name in visitor.findings
        if name not in HASATTR_ENGINE_ALLOWLIST
    ]
    assert not illegal, (
        f"{route_file.name} has hasattr(engine, ...) guards on non-"
        f"allowlisted methods (silent-skip shape of #500):\n"
        + "\n".join(f"  line {ln}: hasattr(engine, {name!r})" for ln, name in illegal)
        + f"\nAllowlist: {sorted(HASATTR_ENGINE_ALLOWLIST) or '(empty)'}."
        " Hoist the method onto BaseEngine with a default and remove the"
        " guard, or add to the allowlist with a documented justification."
    )


# ---------------------------------------------------------------------------
# Pattern B — engine.X.Y (two-level engine access)
# ---------------------------------------------------------------------------


class _TwoLevelEngineVisitor(ast.NodeVisitor):
    def __init__(self):
        self.findings: list[tuple[int, str, str]] = []  # (line, X, full)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # match ``engine.X.Y`` — i.e. an Attribute whose ``value`` is itself
        # an Attribute whose ``value`` is a Name("engine").
        inner = node.value
        if (
            isinstance(inner, ast.Attribute)
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "engine"
        ):
            self.findings.append(
                (
                    node.lineno,
                    inner.attr,
                    f"engine.{inner.attr}.{node.attr}",
                )
            )
        self.generic_visit(node)


@pytest.mark.parametrize("route_file", _route_files(), ids=lambda p: p.name)
def test_no_two_level_engine_access(route_file: pathlib.Path) -> None:
    """``engine.X.Y`` is banned unless ``X`` is declared on
    ``BaseEngine``. Concrete-engine internals such as ``engine.scheduler``
    are not part of the route contract.

    If you genuinely need ``engine.X.Y``, declare ``X`` on
    ``BaseEngine`` as an abstract property so every engine that ships
    has it.
    """
    visitor = _TwoLevelEngineVisitor()
    visitor.visit(_parse(route_file))
    illegal = [
        (line, x, full)
        for line, x, full in visitor.findings
        if x not in TWO_LEVEL_ENGINE_ROOT_ALLOWLIST
    ]
    assert not illegal, (
        f"{route_file.name} reaches into engine internals via two-level"
        f" attribute access (shape of v0.6.70 hotfix):\n"
        + "\n".join(f"  line {ln}: {full}" for ln, _, full in illegal)
        + f"\nAllowed roots: {sorted(TWO_LEVEL_ENGINE_ROOT_ALLOWLIST)}."
        " Hoist the helper onto BaseEngine as ``engine.Y(...)`` so every"
        " engine that ships implements it (and a missing one fails at"
        " instantiation, not at request time)."
    )


# ---------------------------------------------------------------------------
# Pattern C — engine.METHOD(...) must be on BaseEngine
# ---------------------------------------------------------------------------


class _DirectEngineMethodVisitor(ast.NodeVisitor):
    def __init__(self):
        self.findings: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        # match ``engine.METHOD(...)``
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "engine"
        ):
            self.findings.append((node.lineno, func.attr))
        self.generic_visit(node)


@pytest.mark.parametrize("route_file", _route_files(), ids=lambda p: p.name)
def test_route_engine_method_calls_are_on_base_contract(
    route_file: pathlib.Path,
) -> None:
    """Every ``engine.METHOD(...)`` call in a route file must resolve to
    a name declared on ``BaseEngine``. Catches the "new route helper
    calls a BatchedEngine-only method" failure mode at PR time, before
    the missing method ships as a silent route-level skip.
    """
    base_names = _base_engine_public_attrs()
    visitor = _DirectEngineMethodVisitor()
    visitor.visit(_parse(route_file))
    illegal = [
        (line, name) for line, name in visitor.findings if name not in base_names
    ]
    assert not illegal, (
        f"{route_file.name} calls engine methods that are NOT on the"
        f" BaseEngine contract:\n"
        + "\n".join(f"  line {ln}: engine.{name}(...)" for ln, name in illegal)
        + "\nAdd the method to vllm_mlx/engine/base.py — either as"
        " @abstractmethod (every engine must implement) or with a default"
        " body (engines opt in by overriding)."
    )


# ---------------------------------------------------------------------------
# Smoke — confirm the gate works against the original buggy shapes.
# ---------------------------------------------------------------------------


def test_gates_catch_pre_fix_shapes(tmp_path: pathlib.Path) -> None:
    """Synthesize the two original bug shapes in a temp file and assert
    each gate's AST visitor flags them. Without this, a future refactor
    could silently weaken the visitors and we'd never know."""
    src = (
        "def f(engine):\n"
        '    if hasattr(engine, "build_prompt"):\n'
        '        engine.build_prompt("x")\n'
        "    return engine.scheduler.cancel('x')\n"
    )
    tree = ast.parse(src)

    a = _HasattrEngineVisitor()
    a.visit(tree)
    assert a.findings == [(2, "build_prompt")]

    b = _TwoLevelEngineVisitor()
    b.visit(tree)
    assert b.findings == [(4, "scheduler", "engine.scheduler.cancel")]

    c = _DirectEngineMethodVisitor()
    c.visit(tree)
    assert c.findings == [(3, "build_prompt")]
