# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``vllm_mlx`` deprecation shim (package rename to
``rapid_mlx``).

Contract under test (see ``vllm_mlx/__init__.py``):

1. ``import vllm_mlx`` still works and emits a ``DeprecationWarning``.
2. Attribute access forwards to ``rapid_mlx`` lazily — importing the shim
   must NOT eagerly import ``mlx.core`` (same contract as
   ``import rapid_mlx`` itself; see test_mlx_compat.py).
3. ``import vllm_mlx.<sub>`` resolves to the SAME module object as
   ``rapid_mlx.<sub>`` (no duplicate module instances — isinstance
   identity must survive the aliasing).
4. Unknown submodules raise the real ``ImportError``.
5. ``python -m vllm_mlx.server`` still runs (pre-rename invocation found
   in older scripts and docs).

The shim is scheduled for removal after a deprecation window; when that
happens, delete this file together with ``vllm_mlx/``.
"""

import importlib
import pathlib
import subprocess
import sys

import pytest

import rapid_mlx


def _fresh_import_vllm_mlx(monkeypatch):
    """Import ``vllm_mlx`` as if fresh, without disturbing other tests."""
    monkeypatch.delitem(sys.modules, "vllm_mlx", raising=False)
    return importlib.import_module("vllm_mlx")


def test_shim_import_emits_deprecation_warning(monkeypatch):
    with pytest.warns(DeprecationWarning, match="renamed to 'rapid_mlx'"):
        shim = _fresh_import_vllm_mlx(monkeypatch)
    assert shim.__version__ == rapid_mlx.__version__


def test_shim_import_does_not_pull_in_mlx_core(monkeypatch):
    """Same laziness contract as ``import rapid_mlx``: no mlx.core import."""
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    _fresh_import_vllm_mlx(monkeypatch)
    assert "mlx.core" not in sys.modules


def test_shim_forwards_attributes_to_rapid_mlx(monkeypatch):
    shim = _fresh_import_vllm_mlx(monkeypatch)
    # Direct (non-lazy) attribute on rapid_mlx.
    assert shim.__version__ == rapid_mlx.__version__
    # Unknown attribute raises AttributeError, not something stranger.
    with pytest.raises(AttributeError):
        _ = shim.definitely_not_a_real_attribute_12345


def test_shim_submodule_aliases_same_module_object(monkeypatch):
    # chip_tier is stdlib-only — safe on the no-MLX CI leg.
    legacy = importlib.import_module("vllm_mlx.chip_tier")
    modern = importlib.import_module("rapid_mlx.chip_tier")
    assert legacy is modern
    assert sys.modules["vllm_mlx.chip_tier"] is sys.modules["rapid_mlx.chip_tier"]
    # Submodule must also be reachable as an attribute of the shim package.
    shim = _fresh_import_vllm_mlx(monkeypatch)
    assert shim.chip_tier is modern


def test_shim_unknown_submodule_raises_import_error():
    with pytest.raises(ImportError, match="no_such_submodule_xyz"):
        importlib.import_module("vllm_mlx.no_such_submodule_xyz")


@pytest.mark.requires_mlx
def test_python_dash_m_vllm_mlx_server_still_runs():
    """``python -m vllm_mlx.server --help`` must keep working (issue #3511)."""
    python = sys.executable
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [python, "-m", "vllm_mlx.server", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
    )
    assert result.returncode == 0, (
        f"python -m vllm_mlx.server --help failed:\n{result.stderr}"
    )
