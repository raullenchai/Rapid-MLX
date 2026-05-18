# SPDX-License-Identifier: Apache-2.0
"""Structural audit: every ``@pytest.mark.xfail`` must justify itself.

Issue #320 surfaced the regression class: a non-strict xfail is allowed
to drift into a deterministically-failing test without anyone noticing,
because xfail "expected failure" mutes the run-time signal. That is how
``test_concurrent_same_prompt`` slipped — bisected to PR #280, only
caught because a maintainer happened to look.

The bar this audit enforces, for every ``@pytest.mark.xfail`` in
``tests/``:

  1. ``strict=True``, **or**
  2. ``strict=False`` plus a ``reason=`` string that explicitly contains
     the substring ``strict=False`` (forcing the author to acknowledge
     they are taking the muting tradeoff on purpose, with a
     human-readable rationale).

Calls to ``pytest.xfail(...)`` from inside test bodies are also
flagged — those are unconditional skips and should be plain
``pytest.skip()`` with a reason instead.

This is a pure AST audit; no test bodies execute. Runs in well under
a second and is part of every ``make smoke`` / pr_validate cycle.
"""

from __future__ import annotations

import ast
import pathlib

TESTS_ROOT = pathlib.Path(__file__).resolve().parent


def _is_xfail_call(node: ast.AST) -> bool:
    """Match ``pytest.mark.xfail(...)`` regardless of import shape."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # ``pytest.mark.xfail(...)`` → Attribute(Attribute(Name("pytest"), "mark"), "xfail")
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "xfail"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mark"
    ):
        return True
    return False


def _is_bare_xfail_attr(node: ast.AST) -> bool:
    """Match a bare ``@pytest.mark.xfail`` decorator (no parens / kwargs).

    Always non-strict (xfail's default is ``strict=False``) and has no
    reason — never acceptable under this audit.
    """
    if isinstance(node, ast.Attribute) and node.attr == "xfail":
        if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
            return True
    return False


def _kwargs(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _const_str(node: ast.expr | None) -> str | None:
    """Pull a string literal out of ``reason=`` even when it's been
    line-broken across multiple parenthesised pieces."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Common pattern: reason=( "..." "..." ) — implicit string concat shows
    # up as a single Constant after ast.parse. So the only awkward case is
    # explicit ``+`` joining, which we conservatively handle here.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _const_str(node.left)
        right = _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _const_bool(node: ast.expr | None) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _audit_file(path: pathlib.Path) -> list[str]:
    """Return a list of violation messages (empty when clean)."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(TESTS_ROOT)
    violations: list[str] = []

    for node in ast.walk(tree):
        # 1) ``pytest.xfail("...")`` mid-test — should be skip()
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "xfail"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ):
            violations.append(
                f"{rel}:{node.lineno}: bare `pytest.xfail(...)` call — use "
                f"`pytest.skip(...)` instead (xfail at runtime mutes the "
                f"signal the same way a non-strict marker does)"
            )
            continue

        # 2) ``@pytest.mark.xfail(...)`` with kwargs.  Bare decorator
        #    form (no parens) needs decorator-position context, handled
        #    in the second pass over FunctionDef/ClassDef below.
        if _is_xfail_call(node):
            kw = _kwargs(node)
            reason_str = _const_str(kw.get("reason"))
            strict = _const_bool(kw.get("strict"))

            if strict is True:
                continue  # acceptable: regression will surface as XPASS=FAIL

            # strict missing → defaults to False (pytest behaviour)
            if reason_str is None or not reason_str.strip():
                violations.append(
                    f"{rel}:{node.lineno}: `@pytest.mark.xfail(...)` with no "
                    f"`reason=` — every non-strict xfail must explain why."
                )
                continue

            if "strict=False" not in reason_str:
                violations.append(
                    f"{rel}:{node.lineno}: `@pytest.mark.xfail(strict=False, ...)` "
                    f"must include the substring `strict=False` in its `reason=` "
                    f"to acknowledge the regression-muting tradeoff (see #320). "
                    f"Current reason starts with: "
                    f"{reason_str.strip().splitlines()[0][:80]!r}"
                )

    # Second pass: bare ``@pytest.mark.xfail`` decorators (no parens) on
    # any FunctionDef / AsyncFunctionDef / ClassDef. Always non-strict +
    # no reason — never acceptable.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for deco in node.decorator_list:
                if _is_bare_xfail_attr(deco):
                    violations.append(
                        f"{rel}:{deco.lineno}: bare `@pytest.mark.xfail` "
                        f"decorator (no parens) — defaults to strict=False "
                        f"AND has no reason. Convert to `strict=True` or "
                        f"add `reason=...` containing `strict=False`."
                    )

    return violations


def test_every_xfail_is_justified():
    """Walk ``tests/`` and assert every xfail marker meets the bar.

    Excludes this audit file itself — its docstring contains the
    substring ``strict=False`` to explain the rule, and the AST visitor
    would otherwise flag it as a Call form (it isn't, but the exclusion
    is belt-and-suspenders).
    """
    self_path = pathlib.Path(__file__).resolve()
    violations: list[str] = []
    for path in TESTS_ROOT.rglob("*.py"):
        if path.parent.name == "__pycache__":
            continue
        if path == self_path:
            continue
        violations.extend(_audit_file(path))

    assert not violations, (
        "xfail audit failed (#320): every non-strict xfail must justify "
        "itself with `reason=...` containing the substring `strict=False`, "
        "and every xfail decorator must use parens with kwargs. "
        "Without this gate, a regression can silently land under "
        "`xfail strict=False` and the test suite stays green.\n\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
