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
    assert cache._make_key("path/img.png") == "path/img.png"
    assert cache._make_key(["a", "b"]) == "a|b"

    class _Blob:
        def tobytes(self):
            return b"payload"

    key = cache._make_key(_Blob())
    assert key.startswith("pil:")
    assert cache._make_key(_Blob()) == key  # content-addressed
