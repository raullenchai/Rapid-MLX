# SPDX-License-Identifier: Apache-2.0
"""Covers the lazy ``rapid_mlx.__getattr__`` dispatch and version fallback.

Every branch of the package ``__getattr__`` imports a submodule lazily so
that ``import rapid_mlx`` stays free of mlx.core. The branches are
exercised here through stubbed submodules (``sys.modules``), which keeps
the test runnable on hosts without mlx — the changed-lines coverage gate
requires every line the package rename touched to execute somewhere.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import types

import rapid_mlx

# module name -> attribute names dispatched to it by __getattr__
_LAZY_BRANCHES = {
    "rapid_mlx.request": (
        "Request",
        "RequestOutput",
        "RequestStatus",
        "SamplingParams",
    ),
    "rapid_mlx.scheduler": ("Scheduler", "SchedulerConfig", "SchedulerOutput"),
    "rapid_mlx.engine_core": ("EngineCore", "AsyncEngineCore", "EngineConfig"),
    "rapid_mlx.prefix_cache": (
        "PrefixCacheManager",
        "PrefixCacheStats",
        "BlockAwarePrefixCache",
    ),
    "rapid_mlx.paged_cache": (
        "PagedCacheManager",
        "CacheBlock",
        "BlockTable",
        "CacheStats",
    ),
    "rapid_mlx.model_registry": ("get_registry", "ModelOwnershipError"),
}


def test_every_lazy_branch_forwards_to_its_submodule(monkeypatch):
    """Each ``__getattr__`` branch resolves through its lazy submodule."""
    for module_name, attr_names in _LAZY_BRANCHES.items():
        stub = types.ModuleType(module_name)
        for attr in attr_names:
            setattr(stub, attr, object())
        monkeypatch.setitem(sys.modules, module_name, stub)
        # A sibling test may already have imported the real submodule,
        # which binds it as an attribute on the package — and the package
        # attribute would win over the lazy import. Remove it so the
        # branch under test is the code path that resolves the name.
        monkeypatch.delattr(rapid_mlx, module_name.rsplit(".", 1)[-1], raising=False)
        for attr in attr_names:
            assert getattr(rapid_mlx, attr) is getattr(stub, attr)


def test_lazy_vlm_aliases_map_to_mllm_names(monkeypatch):
    """Legacy ``VLM*`` names resolve to the MLLM cache module's MLLM names."""
    stub = types.ModuleType("rapid_mlx.mllm_cache")
    stub.MLLMCacheManager = object()
    stub.MLLMCacheStats = object()
    monkeypatch.setitem(sys.modules, "rapid_mlx.mllm_cache", stub)
    monkeypatch.delattr(rapid_mlx, "mllm_cache", raising=False)

    assert rapid_mlx.VLMCacheManager is stub.MLLMCacheManager
    assert rapid_mlx.VLMCacheStats is stub.MLLMCacheStats


def test_version_fallback_when_package_metadata_is_missing(monkeypatch):
    """Editable installs without dist metadata degrade to "0.0.0"."""

    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    with monkeypatch.context() as m:
        m.setattr(importlib.metadata, "version", _raise)
        importlib.reload(rapid_mlx)
        assert rapid_mlx.__version__ == "0.0.0"
    # Reload once more OUTSIDE the patched context so the real package
    # version is restored for every later test (a reload inside the
    # context would re-take the fallback path and leave "0.0.0" behind).
    importlib.reload(rapid_mlx)
    assert rapid_mlx.__version__ != "0.0.0"
