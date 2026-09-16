# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``vllm_mlx`` deprecation shim (package rename to
``rapid_mlx``).

Contract under test (see ``vllm_mlx/__init__.py``):

1. ``import vllm_mlx`` still works and emits a ``DeprecationWarning``.
2. Attribute access forwards to ``rapid_mlx`` lazily — importing the shim
   must NOT eagerly import ``mlx.core`` (same contract as
   ``import rapid_mlx`` itself; see test_mlx_compat.py).
3. ``import vllm_mlx.<sub>`` — at ANY depth — resolves to the SAME module
   object as ``rapid_mlx.<sub>`` (no duplicate module instances;
   isinstance identity must survive the aliasing). Regression guard: a
   finder registered after PathFinder lets nested names like
   ``vllm_mlx.launch.cli`` double-execute the target file.
4. Aliasing must not corrupt the shared target module's import metadata:
   ``__spec__``/``__loader__`` stay the real ones, so
   ``importlib.reload(rapid_mlx.<sub>)`` keeps re-executing the real file
   (a no-op reload was a real bug in an earlier shim revision).
5. Import errors stay honest: a missing ``rapid_mlx`` submodule reports
   ``No module named 'vllm_mlx.<sub>'``; a target whose own dependencies
   fail propagates THAT error (e.g. ``No module named 'yaml'``), not a
   masked "no module named vllm_mlx.<sub>".
6. ``python -m vllm_mlx.server`` still runs (pre-rename invocation found
   in older scripts and docs), executing the module file once.

The shim is scheduled for removal after a deprecation window; when that
happens, delete this file together with ``vllm_mlx/``.
"""

import importlib
import importlib.resources
import importlib.util
import pathlib
import pkgutil
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
    # Unknown attribute raises AttributeError, not something stranger.
    with pytest.raises(AttributeError):
        _ = shim.definitely_not_a_real_attribute_12345


@pytest.mark.requires_mlx
def test_shim_star_import_matches_pre_rename_contract(monkeypatch):
    """``from vllm_mlx import *`` yields the pre-rename engine names.

    Without the shim's own ``__all__``, star-import would fall back to the
    module dict and leak the shim's private imports (``sys``,
    ``warnings``, ...) instead of the engine names the old package
    exported.
    """
    shim = _fresh_import_vllm_mlx(monkeypatch)
    assert shim.__all__ == list(rapid_mlx.__all__)
    star_names: dict = {}
    exec("from vllm_mlx import *", star_names)
    for name in rapid_mlx.__all__:
        assert name in star_names, f"star import lost {name!r}"
    for leaked in ("sys", "warnings", "importlib", "importlib_util"):
        assert leaked not in star_names, f"star import leaked {leaked!r}"


def test_shim_submodule_aliases_same_module_object():
    # Top-level submodule (chip_tier is stdlib-only — safe on no-MLX CI).
    legacy = importlib.import_module("vllm_mlx.chip_tier")
    modern = importlib.import_module("rapid_mlx.chip_tier")
    assert legacy is modern
    assert sys.modules["vllm_mlx.chip_tier"] is sys.modules["rapid_mlx.chip_tier"]


def test_shim_preserves_package_discovery_and_top_level_resources(monkeypatch):
    """External integrations may discover modules/data through the old package."""
    shim = _fresh_import_vllm_mlx(monkeypatch)

    discovered = {entry.name for entry in pkgutil.iter_modules(shim.__path__)}
    assert {"cli", "engine", "server"} <= discovered

    root = importlib.resources.files(shim)
    assert root.joinpath("aliases.json").is_file()
    assert root.joinpath("model_recommendations.json").is_file()


def test_shim_reload_does_not_stack_alias_finders(monkeypatch):
    shim = _fresh_import_vllm_mlx(monkeypatch)

    def installed_finders():
        return [
            finder
            for finder in sys.meta_path
            if getattr(finder, "_rapid_mlx_alias_finder", False)
        ]

    before = installed_finders()
    importlib.reload(shim)
    after = installed_finders()

    assert len(after) == len(before) == 1
    assert after[0] is before[0]


def test_shim_nested_submodule_aliases_same_module_object():
    """Regression: nested names must alias, not double-execute the file.

    The aliased parent's ``__path__`` points into the real ``rapid_mlx/``
    tree, so a finder consulted AFTER PathFinder lets the machinery load
    e.g. ``launch/cli.py`` a second time as a distinct
    ``vllm_mlx.launch.cli`` module — duplicate state, broken isinstance.
    """
    legacy_cli = importlib.import_module("vllm_mlx.launch.cli")
    modern_cli = importlib.import_module("rapid_mlx.launch.cli")
    assert legacy_cli is modern_cli
    assert sys.modules["vllm_mlx.launch.cli"] is sys.modules["rapid_mlx.launch.cli"]


def test_shim_preserves_target_module_import_metadata():
    """Aliasing must not stamp the alias spec onto the shared module.

    Earlier shim revision let ``_init_module_attrs`` overwrite
    ``rapid_mlx.<sub>.__spec__`` with the alias spec: that made
    ``importlib.reload(rapid_mlx.<sub>)`` a silent no-op and made
    ``__spec__.loader.get_code`` recurse into the alias loader.
    """
    modern = importlib.import_module("rapid_mlx.launch.cli")
    importlib.import_module("vllm_mlx.launch.cli")

    assert modern.__spec__.name == "rapid_mlx.launch.cli"
    assert modern.__spec__.parent == "rapid_mlx.launch"
    assert modern.__loader__ is modern.__spec__.loader

    # reload() re-executes the real file, not a no-op: mutate the source
    # on disk, reload, and observe the new module-level binding.
    source_path = pathlib.Path(modern.__file__)
    original_source = source_path.read_text()
    try:
        source_path.write_text(original_source + "\n_shim_reload_probe = 1\n")
        importlib.reload(modern)
        assert getattr(modern, "_shim_reload_probe", None) == 1
    finally:
        source_path.write_text(original_source)


def test_shim_propagates_target_dependency_errors(monkeypatch):
    """A target whose own import fails must surface ITS error.

    With ``yaml`` forced to fail, ``import vllm_mlx.agents`` must report
    the real cause (``yaml``), not the masked
    ``No module named 'vllm_mlx.agents'``.
    """
    monkeypatch.setitem(sys.modules, "yaml", None)
    monkeypatch.delitem(sys.modules, "vllm_mlx.agents", raising=False)
    monkeypatch.delitem(sys.modules, "rapid_mlx.agents", raising=False)
    with pytest.raises(ModuleNotFoundError, match="yaml"):
        importlib.import_module("vllm_mlx.agents")


def test_shim_unknown_submodule_raises_import_error():
    with pytest.raises(ModuleNotFoundError, match="vllm_mlx.no_such_submodule_xyz"):
        importlib.import_module("vllm_mlx.no_such_submodule_xyz")


def test_shim_propagates_intermediate_parent_failures(monkeypatch):
    """A broken dep of an INTERMEDIATE parent package must surface honestly.

    ``importlib.util.find_spec`` executes intermediate parent packages, so
    e.g. a yaml failure inside ``rapid_mlx.audio``'s ``__init__`` raises
    ``ModuleNotFoundError('yaml')`` while locating
    ``rapid_mlx.audio.tts`` — the shim must re-raise it, not swallow it
    into ``No module named 'vllm_mlx.audio.tts'``. Simulated at the same
    call site by intercepting ``importlib.util.find_spec``.
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "rapid_mlx.audio":
            raise ModuleNotFoundError(
                "No module named 'fakebrokenlib'", name="fakebrokenlib"
            )
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.delitem(sys.modules, "rapid_mlx.audio", raising=False)
    monkeypatch.delitem(sys.modules, "vllm_mlx.audio", raising=False)
    with pytest.raises(ModuleNotFoundError, match="fakebrokenlib"):
        importlib.import_module("vllm_mlx.audio")


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
