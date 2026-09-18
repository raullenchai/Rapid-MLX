# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored APC support modules: ``apc_coordinator``,
``apc_storage``, ``kv_quant``, ``_stream_cleanup`` and ``vision_cache``.

The coordinator/storage/kv_quant modules are consumed by the vendored APC
engine vendored in the next PR of the stack, but their import wiring is
established here: the coordinator must resolve the vendored adapters, and
kv_quant must stay behavior-identical to upstream (redirect-parity probe).
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.models.mlx_vlm_vendored import (
    _stream_cleanup,
    apc_adapters,
    apc_coordinator,
    apc_storage,
)
from rapid_mlx.models.mlx_vlm_vendored import cache as vendored_cache
from rapid_mlx.models.mlx_vlm_vendored import kv_quant as vendored_kv_quant
from rapid_mlx.models.mlx_vlm_vendored import vision_cache as vendored_vision_cache


class _FakeLM:
    def make_cache(self):
        return [vendored_cache.KVCache() for _ in range(2)]


def _coordinator_model():
    import types

    return types.SimpleNamespace(language_model=_FakeLM())


def test_coordinator_resolves_the_vendored_adapters():
    """The module-level import must bind the vendored sibling, not upstream's."""
    assert (
        apc_coordinator.build_prefix_cache_plan is apc_adapters.build_prefix_cache_plan
    )
    assert apc_coordinator.PrefixCachePlan is apc_adapters.PrefixCachePlan


def test_engine_resolves_the_vendored_family():
    """apc.py's top-level relative imports must bind the vendored siblings."""
    from rapid_mlx.models.mlx_vlm_vendored import apc

    assert apc.APCCoordinator is apc_coordinator.APCCoordinator
    assert apc.APCNode is apc_storage.APCNode


def test_coordinator_builds_plan_from_vendored_caches():
    coordinator = apc_coordinator.APCCoordinator(
        manager=None, model=_coordinator_model()
    )
    assert coordinator.plan.restorable
    assert coordinator.plan.strategy == "block"
    assert len(coordinator.plan.components) == 2
    # ``enabled`` requires a manager as well as a restorable plan.
    assert coordinator.enabled is False
    fresh = coordinator.fresh_cache()
    assert len(fresh) == 2
    assert all(type(c) is vendored_cache.KVCache for c in fresh)


def test_coordinator_upstream_namespace_plan_parity():
    """Same plan shape for upstream-typed caches (the producer namespace
    during the transition)."""
    pytest.importorskip("mlx_vlm.models.cache")
    from mlx_vlm.models import cache as upstream_cache

    class _UpstreamLM:
        def make_cache(self):
            return [upstream_cache.KVCache() for _ in range(2)]

    import types

    coordinator = apc_coordinator.APCCoordinator(
        manager=None, model=types.SimpleNamespace(language_model=_UpstreamLM())
    )
    assert coordinator.plan.restorable
    assert coordinator.plan.strategy == "block"


def test_kv_quant_from_legacy_uniform_policy():
    policy = vendored_kv_quant.from_legacy(8.0)
    assert policy is not None
    assert policy.bits == 8.0
    assert not policy.is_turboquant
    assert policy.is_homogeneous
    assert policy.scheme == vendored_kv_quant.UNIFORM_SCHEME
    assert policy.group_size == 64
    assert vendored_kv_quant.from_legacy(None) is None


def test_kv_quant_from_config_roundtrip():
    policy = vendored_kv_quant.from_config({"bits": 4, "group_size": 32})
    assert policy is not None
    assert policy.bits == 4 and policy.group_size == 32
    config = policy.to_config()
    restored = vendored_kv_quant.from_config(config)
    assert restored.bits == policy.bits
    assert restored.group_size == policy.group_size
    assert restored.scheme == policy.scheme


def test_kv_quant_fingerprint_parity_with_upstream():
    """The redirect-parity probe: vendored kv_quant must produce identical
    fingerprints to the pinned upstream for a spread of inputs."""
    pytest.importorskip("mlx_vlm.kv_quant")
    from mlx_vlm import kv_quant as upstream_kv_quant

    cases = [
        (8.0, 64, "uniform", 0),
        (4.0, 32, "uniform", 128),
        (None, 64, None, 0),
        (3.5, 64, "uniform", 16),
    ]
    for kv_bits, group_size, scheme, start in cases:
        assert vendored_kv_quant.kv_quant_fingerprint(
            kv_bits, group_size, scheme, start
        ) == upstream_kv_quant.kv_quant_fingerprint(kv_bits, group_size, scheme, start)
        vendored = vendored_kv_quant.from_legacy(kv_bits, scheme, group_size)
        upstream = upstream_kv_quant.from_legacy(kv_bits, scheme, group_size)
        assert (vendored is None) == (upstream is None)
        if vendored is not None:
            assert vendored.fingerprint(start) == upstream.fingerprint(start)


def test_apc_storage_kv_handle_roundtrip():
    import mlx.core as mx

    handle = apc_storage.KVBlockHandle(
        keys=[mx.ones((2, 3))], values=[mx.zeros((2, 4))]
    )
    assert handle.resident_bytes() > 0
    handle.release()
    assert handle.keys is None and handle.values is None
    assert handle.resident_bytes() == 0


def test_apc_storage_node_components():
    import mlx.core as mx

    class _Node(apc_storage.APCNode):
        def __init__(self):
            self.components = {}

    node = _Node()
    assert node.kv_handle() is None
    node.set_kv([mx.ones((1, 2))], [mx.zeros((1, 2))])
    assert node.kv_handle() is not None
    assert node.keys is not None and node.values is not None
    assert node.resident_bytes() > 0
    node.release_components()
    assert node.components == {}
    assert node.resident_bytes() == 0


def test_stream_cleanup_runs():
    # ``clear_streams`` releases the *calling thread's* MLX streams; calling
    # it on the main thread poisons the process-wide default stream for every
    # later test. Upstream servers invoke it from request worker threads, so
    # mirror that here.
    import threading

    errors: list[BaseException] = []

    def _run():
        try:
            _stream_cleanup.clear_mlx_streams()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    worker = threading.Thread(target=_run)
    worker.start()
    worker.join()
    assert not errors


def test_vision_feature_cache_lru_contract():
    cache = vendored_vision_cache.VisionFeatureCache(max_size=2)
    cache.put("a", "fa")
    cache.put("b", "fb")
    assert cache.get("a") == "fa"  # refresh a's recency
    cache.put("c", "fc")  # evicts b
    assert "b" not in cache
    assert cache.get("a") == "fa" and cache.get("c") == "fc"
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_vision_feature_cache_key_shapes():
    cache = vendored_vision_cache.VisionFeatureCache(max_size=4)
    assert cache._make_key("path/img.png") == "s12:path/img.png"
    assert cache._make_key(["a", "b"]) == "l2:s1:as1:b"

    class _Blob:
        def tobytes(self):
            return b"payload"

    key = cache._make_key(_Blob())
    assert key.startswith("p:")
    assert cache._make_key(_Blob()) == key  # content-addressed
    # bytes-like sources are accepted directly and content-addressed.
    assert cache._make_key(b"payload") == key


def test_vision_feature_cache_empty_and_nested_lists_do_not_collide():
    """The child count keeps empty children unambiguous: bare ``l`` tags
    would collapse [[], []] and [[[]]] onto the same key."""
    cache = vendored_vision_cache.VisionFeatureCache(max_size=8)
    assert cache._make_key([[], []]) != cache._make_key([[[]]])
    assert cache._make_key([]) != cache._make_key([[]])
    cache.put([[], []], "pair")
    assert cache.get([[[]]]) is None
    assert cache.get([[], []]) == "pair"


def test_vision_feature_cache_pathlike_sources():
    """The docstring promises Path sources; upstream only accepted str (a
    Path fell into the obj:id fallback). The vendored copy normalizes
    PathLike via os.fspath."""
    import pathlib

    cache = vendored_vision_cache.VisionFeatureCache(max_size=4)
    path = pathlib.Path("/tmp/img.png")
    assert cache._make_key(path) == cache._make_key("/tmp/img.png")


def test_vision_feature_cache_list_keys_do_not_collide():
    """Type-tagged composite keys: distinct sources must never share a key —
    upstream's bare "|" join collided ["a|b", "c"] with ["a", "b|c"], and
    even length-prefixing collides ["1:a", "b"] with [["a"], "b"]."""
    cache = vendored_vision_cache.VisionFeatureCache(max_size=8)
    assert cache._make_key(["a|b", "c"]) != cache._make_key(["a", "b|c"])
    assert cache._make_key(["1:a", "b"]) != cache._make_key([["a"], "b"])
    cache.put(["a", "b|c"], "first")
    cache.put(["1:a", "b"], "second")
    cache.put([["a"], "b"], "third")
    assert cache.get(["a|b", "c"]) is None
    assert cache.get(["a", "b|c"]) == "first"
    assert cache.get(["1:a", "b"]) == "second"
    assert cache.get([["a"], "b"]) == "third"


def test_vision_feature_cache_rejects_unsupported_source_types():
    """Upstream's ``obj:{id(...)}`` fallback could hand a recycled id to an
    unrelated object; the vendored copy fails loudly instead."""
    cache = vendored_vision_cache.VisionFeatureCache(max_size=4)
    with pytest.raises(TypeError, match="unsupported image source"):
        cache._make_key(object())
