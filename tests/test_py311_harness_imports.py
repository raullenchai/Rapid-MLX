# SPDX-License-Identifier: Apache-2.0
"""Contracts for the Python 3.10 harness collection guard."""

import ast
import re
from pathlib import Path

import pytest

from tests.py311_harness_imports import (
    PY311_ONLY_MODULES,
    RUNTIME_ONLY_TEST_FILES,
    _executed_imports,
    _is_py311_only,
    _package_of,
    requires_python311,
    skip_on_python310,
)

TESTS_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pydantic_ai", True),
        ("from pydantic_ai.messages import X", True),
        ("import pydantic_ai as pa", True),
        ("import json", False),
        ("def f():\n    import pydantic_ai", False),
        ("class C:\n    import pydantic_ai", True),
        ("if TYPE_CHECKING:\n    import pydantic_ai", False),
        ("try:\n    import pydantic_ai\nexcept ImportError:\n    pass", False),
        ("try:\n    import pydantic_ai\nexcept ImportError:\n    raise", True),
        (
            'try:\n    import pydantic_ai\nexcept ImportError:\n    raise RuntimeError("x") from None',
            True,
        ),
        ("if cond:\n    import pydantic_ai", True),
        ("from rapid_mlx.chat_runtime.model_client import inference_url", True),
        ("from rapid_mlx.chat_runtime.progress import tool_progress", False),
        ("from rapid_mlx.chat_runtime import model_client", True),
    ],
)
def test_requires_python311(source, expected):
    assert requires_python311(source) is expected


def test_direct_imports_are_skipped_and_runtime_exceptions_exist():
    paths = list(TESTS_DIR.rglob("test_*.py"))
    assert {path.name for path in paths} >= RUNTIME_ONLY_TEST_FILES
    for path in paths:
        source = path.read_text(encoding="utf-8")
        imports = _executed_imports(ast.parse(source), _package_of(path))
        if any(_is_py311_only(name) for name in imports):
            assert requires_python311(source, path), path
            assert skip_on_python310(path), path


def test_python311_only_dependency_marker():
    pyproject = (TESTS_DIR.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert PY311_ONLY_MODULES == ("pydantic_ai",)
    assert re.search(
        r"pydantic-ai-slim[^\n]*;\s*python_version\s*>=\s*['\"]3\.11['\"]",
        pyproject,
    )
