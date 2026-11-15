# SPDX-License-Identifier: Apache-2.0
"""Exercise admission ownership without importing MLX or requiring Metal."""

import copy
import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path

import pytest


@pytest.fixture
def fastpath(monkeypatch):
    import rapid_mlx

    class KVCache:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0

    class RotatingKVCache(KVCache):
        pass

    class NativeLayer:
        _rapid_native_batch_surface = True

        def __init__(self, caches=None):
            self.state = [1, 2, 3]
            if caches is not None:
                self.caches = caches

        def filter(self, indices):
            pass

        def extract(self, index):
            return copy.deepcopy(self)

        def extend(self, other):
            pass

    stubs = {
        "mlx": types.ModuleType("mlx"),
        "mlx.core": types.ModuleType("mlx.core"),
        "mlx_lm": types.ModuleType("mlx_lm"),
        "mlx_lm.models": types.ModuleType("mlx_lm.models"),
        "mlx_lm.models.cache": types.SimpleNamespace(
            KVCache=KVCache, RotatingKVCache=RotatingKVCache
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(
        rapid_mlx,
        "_mlx_compat",
        types.SimpleNamespace(install=lambda: None),
        raising=False,
    )
    path = Path(__file__).parents[1] / "rapid_mlx/singleton_cache_fastpath.py"
    spec = importlib.util.spec_from_file_location("rapid_mlx._test_singleton", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gen = types.SimpleNamespace(_merge_caches=lambda c: c, _extend_cache=lambda a, b: b)
    assert module._install_locked(gen)
    return gen, NativeLayer, KVCache, module


def test_owned_layers_keep_identity_at_extend(fastpath):
    gen, native, kv, _ = fastpath
    for raw in (native(), kv(), native([native()])):
        [owned] = gen._merge_caches([[raw]])
        assert gen._extend_cache([], [owned])[0] is owned


def test_raw_and_copied_payloads_are_detached(fastpath):
    gen, native, _, _ = fastpath
    [owned] = gen._merge_caches([[native()]])
    for raw in (native(), owned.extract(0), copy.copy(owned)):
        [admitted] = gen._extend_cache([], [raw])
        assert admitted is not raw
        assert admitted.state is not raw.state


@pytest.mark.parametrize("rotating", [False, True], ids=["kv", "rotating"])
def test_extracted_exact_layer_is_unowned_and_detached(fastpath, monkeypatch, rotating):
    gen, _, kv, module = fastpath

    class Array(list):
        dtype = "float32"

        @property
        def shape(self):
            return (1, 1, len(self), 1)

        def __getitem__(self, index):
            if isinstance(index, tuple):
                return Array(super().__getitem__(index[1]))
            return super().__getitem__(index)

    monkeypatch.setattr(module.mx, "contiguous", Array, raising=False)
    monkeypatch.setattr(
        module.mx, "zeros", lambda shape, dtype: Array([0] * shape[2]), raising=False
    )
    monkeypatch.setattr(
        module.mx,
        "concatenate",
        lambda arrays, axis: Array(value for array in arrays for value in array),
        raising=False,
    )
    kv.step = 256
    cache_type = module.RotatingKVCache if rotating else kv
    [owned] = gen._merge_caches([[cache_type()]])
    owned.keys, owned.values = Array([1, 2]), Array([3, 4])
    owned.offset = 2

    payload = owned.extract(0)
    assert payload not in module._admitted_layers
    [admitted] = gen._extend_cache([], [payload])
    assert admitted is not payload
    assert admitted in module._admitted_layers
    admitted.keys[0], admitted.values[0] = 9, 10
    assert payload.keys == [1, 2] and payload.values == [3, 4]


def test_unknown_wrapper_with_owned_child_still_detaches(fastpath):
    gen, native, _, _ = fastpath
    [child] = gen._merge_caches([[native()]])
    wrapper = native([child])
    [admitted] = gen._extend_cache([], [wrapper])
    assert admitted is not wrapper
    assert admitted.caches[0] is not child
    assert gen._extend_cache([], [admitted])[0] is admitted


def test_ownership_does_not_keep_layers_alive(fastpath):
    gen, native, _, _ = fastpath
    [owned] = gen._merge_caches([[native()]])
    ref = weakref.ref(owned)
    del owned
    gc.collect()
    assert ref() is None


def test_value_equal_layers_cannot_inherit_ownership(fastpath):
    gen, native, _, _ = fastpath

    class EqualLayer(native):
        def __eq__(self, other):
            return isinstance(other, EqualLayer)

        def __hash__(self):
            return 0

    [owned] = gen._merge_caches([[EqualLayer()]])
    raw = EqualLayer()
    assert raw == owned
    assert gen._extend_cache([], [raw])[0] is not raw


def test_slots_wrapper_without_weakref_keeps_copy_on_admit(fastpath):
    gen, native, _, _ = fastpath

    class Wrapper:
        __slots__ = ("caches",)
        filter = native.filter
        extract = native.extract
        extend = native.extend

        def __init__(self, child):
            self.caches = [child]

    [owned] = gen._merge_caches([[Wrapper(native())]])
    [admitted] = gen._extend_cache([], [owned])
    assert admitted is not owned
    assert admitted.caches[0] is not owned.caches[0]


def test_deepcopy_of_owned_exact_layer_keeps_ownership(fastpath, monkeypatch):
    gen, _, kv, module = fastpath
    [owned] = gen._merge_caches([[kv()]])
    owned.keys, owned.values = [1, 2], [3, 4]
    owned.offset = 2
    dup = copy.deepcopy(owned)
    assert dup in module._admitted_layers
    assert dup.keys is not owned.keys and dup.values is not owned.values

    def unexpected_detach(cache):
        pytest.fail("an owned deepcopy must not be detached again")

    monkeypatch.setattr(module, "_detach_layer", unexpected_detach)
    assert gen._extend_cache([], [dup])[0] is dup
    dup.keys[0], dup.values[0] = 9, 10
    assert owned.keys == [1, 2] and owned.values == [3, 4]


def test_deepcopy_of_unowned_exact_layer_still_detaches(fastpath, monkeypatch):
    gen, _, kv, module = fastpath
    raw = kv()
    raw.keys, raw.values = [1, 2], [3, 4]
    raw.offset = 2
    # A surface alone does not confer ownership (e.g. a shallow copy).
    module._bind_singleton_surface(raw)
    dup = copy.deepcopy(raw)
    assert raw not in module._admitted_layers
    assert dup not in module._admitted_layers
    detached = []

    def copy_layer(cache, *, headroom=0):
        detached.append(cache)
        clone = kv()
        clone.keys = copy.deepcopy(cache.keys)
        clone.values = copy.deepcopy(cache.values)
        clone.offset = cache.offset
        return clone

    kv.step = 256
    monkeypatch.setattr(module, "_copy_exact_layer", copy_layer)
    [admitted] = gen._extend_cache([], [dup])
    assert detached == [dup]
    assert admitted is not dup
    assert admitted.keys is not dup.keys and admitted.values is not dup.values
