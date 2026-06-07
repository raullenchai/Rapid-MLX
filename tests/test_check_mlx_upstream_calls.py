# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``scripts/check_mlx_upstream_calls.py``.

Pure stdlib AST testing — synthesize tiny Python files and assert the
scanner classifies them correctly. No real mlx-lm install needed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import textwrap

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_mlx_upstream_calls.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_mlx_upstream_calls", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cmu():
    return _load_module()


def _write(tmp_path: pathlib.Path, name: str, body: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


# ---------- module-scope dangerous calls flagged --------------------


def test_module_scope_new_thread_local_stream_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        _stream = mx.new_thread_local_stream(mx.default_device())
        """,
    )
    findings = cmu.scan_file(f)
    chains = [chain for _, chain, _ in findings]
    assert "mx.new_thread_local_stream" in chains
    assert "mx.default_device" in chains


def test_module_scope_mx_metal_is_available_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        HAS_METAL = mx.metal.is_available()
        """,
    )
    findings = cmu.scan_file(f)
    assert any("mx.metal" in chain for _, chain, _ in findings)


def test_set_default_device_at_module_scope_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        """,
    )
    findings = cmu.scan_file(f)
    chains = [chain for _, chain, _ in findings]
    assert "mx.set_default_device" in chains


# ---------- inside-function calls are NOT flagged -------------------


def test_dangerous_call_inside_function_not_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        def get_stream():
            return mx.new_thread_local_stream(mx.default_device())
        """,
    )
    assert cmu.scan_file(f) == []


def test_dangerous_call_inside_class_method_not_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        class Foo:
            def __init__(self):
                self.stream = mx.new_thread_local_stream(mx.default_device())
        """,
    )
    assert cmu.scan_file(f) == []


def test_dangerous_call_inside_class_body_is_known_gap(cmu, tmp_path):
    # Class body executes at import — same risk as module top-level.
    # AST-wise it's inside ClassDef, so per the strict reading of
    # _is_module_scope (no enclosing FunctionDef/ClassDef) this is
    # currently NOT flagged. Document the behavior so the reader
    # knows the limit — class-body initializers should be rare in
    # mlx-lm anyway.
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        class Foo:
            stream = mx.new_thread_local_stream(mx.default_device())
        """,
    )
    # Known-gap: scanner only catches top-of-module right now.
    assert cmu.scan_file(f) == []


# ---------- non-dangerous calls not flagged -------------------------


def test_mlx_array_call_not_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core as mx
        ZERO = mx.array([0])
        """,
    )
    assert cmu.scan_file(f) == []


def test_non_mlx_call_not_flagged(cmu, tmp_path):
    f = _write(
        tmp_path,
        "a.py",
        """
        import os
        _ = os.environ.get("FOO")
        """,
    )
    assert cmu.scan_file(f) == []


# ---------- robustness ----------------------------------------------


def test_invalid_python_returns_empty(cmu, tmp_path):
    f = _write(tmp_path, "a.py", "this is not python code }}\n")
    assert cmu.scan_file(f) == []


def test_full_mlx_core_path_also_matched(cmu, tmp_path):
    # Some files import as ``from mlx.core import ...`` and write
    # ``mlx.core.default_device()`` explicitly.
    f = _write(
        tmp_path,
        "a.py",
        """
        import mlx.core
        DEV = mlx.core.default_device()
        """,
    )
    chains = [chain for _, chain, _ in cmu.scan_file(f)]
    assert "mlx.core.default_device" in chains
