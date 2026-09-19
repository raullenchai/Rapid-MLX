# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored APC support modules: ``apc_coordinator``,
``apc_storage``, ``kv_quant``, ``_stream_cleanup`` and ``vision_cache``.

The coordinator/storage/kv_quant modules are consumed by the vendored APC
engine. Their import wiring is explicit: the coordinator must resolve the
vendored adapters, and kv_quant must stay behavior-identical to upstream
(redirect-parity probe).
"""

import hashlib
import inspect

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

_APC_UPSTREAM_SHA256 = (
    "5b2b940852f11f34f7b4daf627bc31fc701f8abffc72d40189bc3e5ac57f878c"
)
_APC_VENDORED_SHA256 = (
    "74c227cb9def17a60410a40f0cbffe113b42c55a05d2ef3b046019074fa603a8"
)
_APC_DEVIATION_COUNT = 22


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
    assert apc.ComponentId is apc_storage.ComponentId
    assert apc.StateHandle is apc_storage.StateHandle
    assert apc.clear_mlx_streams is _stream_cleanup.clear_mlx_streams
    assert apc.kv_quant_from_config is vendored_kv_quant.from_config
    assert apc.kv_quant_fingerprint is vendored_kv_quant.kv_quant_fingerprint


def test_engine_matches_reviewed_vendored_source():
    """Fail closed if either provenance anchor or reviewed copy drifts.

    The step-2b-3 engine intentionally differs from mlx-vlm 0.7.1: it has
    dual-namespace support plus three repro-tested upstream bug fixes.  Pin
    both sources and the deviation-sentinel inventory so future edits cannot
    silently hide inside the large coverage-exempt file.
    """
    import mlx_vlm.apc as upstream_apc

    from rapid_mlx.models.mlx_vlm_vendored import apc as vendored_apc

    upstream_source = inspect.getsource(upstream_apc)
    vendored_source = inspect.getsource(vendored_apc)
    assert hashlib.sha256(upstream_source.encode()).hexdigest() == _APC_UPSTREAM_SHA256

    assert hashlib.sha256(vendored_source.encode()).hexdigest() == _APC_VENDORED_SHA256
    assert vendored_source.count("# VENDOR-DEVIATION") == _APC_DEVIATION_COUNT


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

    # The legacy API intentionally treats fractional bits as TurboQuant even
    # when an old caller passes the nominal ``uniform`` string.  Pin that
    # slightly surprising compatibility rule explicitly instead of hiding it
    # inside the generic table above.
    vendored_fractional = vendored_kv_quant.from_legacy(3.5, "uniform", 64)
    upstream_fractional = upstream_kv_quant.from_legacy(3.5, "uniform", 64)
    assert vendored_fractional is not None and upstream_fractional is not None
    assert vendored_fractional.is_turboquant
    assert upstream_fractional.is_turboquant
    assert vendored_fractional.fingerprint(16) == upstream_fractional.fingerprint(16)


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
    assert cache._make_key(b"payload") == cache._make_key(bytearray(b"payload"))
    assert cache._make_key(b"payload") == cache._make_key(memoryview(b"payload"))
    # tobytes() sources hash type/mode/size metadata together with the raw
    # bytes, so they intentionally do NOT collide with a bare bytes source
    # carrying the same byte string.
    assert cache._make_key(b"payload") != key


def test_vision_feature_cache_tobytes_keys_include_mode_and_size():
    """Upstream hashed only ``tobytes()``, so images with identical raw
    bytes but different mode or size collided and one image's features were
    served for the other. The vendored copy folds stable type/mode/size
    metadata into the hashed payload."""
    cache = vendored_vision_cache.VisionFeatureCache(max_size=8)

    class _Image:
        def __init__(self, mode, size):
            self.mode = mode
            self.size = size

        def tobytes(self):
            return b"\x00\x00"

    gray_wide = _Image("L", (2, 1))
    rgb_square = _Image("RGB", (1, 1))
    assert cache._make_key(gray_wide) != cache._make_key(rgb_square)
    cache.put(gray_wide, "gray-features")
    assert cache.get(rgb_square) is None  # no cross-serve
    assert cache.get(gray_wide) == "gray-features"
    # The same image object (same type/mode/size/bytes) still hits across
    # calls — that is the cache's entire purpose.
    assert cache.get(_Image("L", (2, 1))) == "gray-features"


def test_vision_feature_cache_palette_images_do_not_collide():
    """Palette (``"P"``) images hash to palette indices in ``tobytes()``;
    two same-sized images with identical indices but different palettes
    render different content yet collided upstream (raw-bytes keying). The
    vendored copy folds the effective palette into the digest."""
    pytest.importorskip("PIL.Image")
    from PIL import Image

    red = Image.new("P", (2, 1))
    red.putpalette([200, 30, 30] * 256)
    red.putdata([0, 1])
    blue = Image.new("P", (2, 1))
    blue.putpalette([30, 30, 200] * 256)
    blue.putdata([0, 1])
    # Precondition: upstream's key material (raw tobytes) is identical.
    assert red.tobytes() == blue.tobytes()

    cache = vendored_vision_cache.VisionFeatureCache(max_size=8)
    assert cache._make_key(red) != cache._make_key(blue)
    cache.put(red, "red-palette-features")
    assert cache.get(blue) is None  # no cross-serve
    assert cache.get(red) == "red-palette-features"


def test_vision_feature_cache_palette_transparency_does_not_collide():
    """The same indices and RGB palette can render differently when a
    different palette entry is transparent; that metadata is not part of
    ``tobytes()`` or ``getpalette()``."""
    pytest.importorskip("PIL.Image")
    from PIL import Image

    first = Image.new("P", (2, 1))
    first.putpalette([200, 30, 30] * 256)
    first.putdata([0, 1])
    first.info["transparency"] = 0
    second = first.copy()
    second.info["transparency"] = 1
    assert first.tobytes() == second.tobytes()
    assert first.getpalette() == second.getpalette()

    cache = vendored_vision_cache.VisionFeatureCache(max_size=8)
    assert cache._make_key(first) != cache._make_key(second)
    cache.put(first, "first-transparency-features")
    assert cache.get(second) is None
    assert cache.get(first) == "first-transparency-features"


def test_vision_feature_cache_zero_max_size_disables_storage():
    """Upstream raised KeyError (``popitem()`` on an empty mapping) when
    ``put`` was called on a cache constructed with ``max_size <= 0``; the
    constructor accepted any integer. The vendored copy treats zero (or
    negative) as storage disabled."""
    cache = vendored_vision_cache.VisionFeatureCache(max_size=0)
    cache.put("a", "fa")  # must not raise
    assert len(cache) == 0
    assert cache.get("a") is None
    assert "a" not in cache
    negative = vendored_vision_cache.VisionFeatureCache(max_size=-3)
    negative.put("a", "fa")  # must not raise
    assert len(negative) == 0


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
    PathLike via os.fsdecode."""
    import pathlib

    cache = vendored_vision_cache.VisionFeatureCache(max_size=4)
    path = pathlib.Path("/tmp/img.png")
    assert cache._make_key(path) == cache._make_key("/tmp/img.png")


def test_vision_feature_cache_byte_valued_paths_do_not_collide():
    """``os.fspath`` can return ``bytes`` for byte-valued paths; routing
    those into the image-content hash branch collided a byte path with a
    raw image payload equal to the path bytes. The vendored copy decodes
    byte paths (surrogateescape, injective) onto the str-path key space."""
    cache = vendored_vision_cache.VisionFeatureCache(max_size=8)

    class BytesPath:
        def __fspath__(self):
            return b"/tmp/img.png"

    byte_path = BytesPath()
    assert cache._make_key(byte_path) == cache._make_key("/tmp/img.png")
    # The collision that motivated the fix: a raw image payload equal to
    # the path bytes must never receive the path's cached features.
    assert cache._make_key(byte_path) != cache._make_key(b"/tmp/img.png")
    cache.put(byte_path, "path-features")
    assert cache.get(b"/tmp/img.png") is None
    assert cache.get("/tmp/img.png") == "path-features"


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


def test_storage_fallback_accounting_and_non_kv_component(monkeypatch):
    """Exercise defensive accounting and the node view for foreign handles."""

    class _WithoutNBytes:
        size = 3
        itemsize = 4

    class _BrokenSize:
        @property
        def size(self):
            raise RuntimeError("no size")

    monkeypatch.setattr(apc_storage, "mx", type("MX", (), {"array": object}))
    assert apc_storage._array_bytes(_WithoutNBytes()) == 12
    assert apc_storage._array_bytes(_BrokenSize()) == 0

    class _OtherHandle:
        def resident_bytes(self):
            return 7

        def release(self):
            self.released = True

    class _Node(apc_storage.APCNode):
        def __init__(self):
            self.components = {"other": _OtherHandle()}

    node = _Node()
    assert node.kv_handle() is None
    assert node.keys is None and node.values is None
    assert node.resident_bytes() == 7


def test_kv_quant_mixed_policy_validation_and_explicit_splits():
    mixed = vendored_kv_quant.from_legacy(
        4,
        kv_key_bits=3,
        kv_value_bits=5,
        kv_key_scheme="uniform",
        kv_value_scheme="turboquant",
    )
    assert mixed is not None
    assert mixed.has_split_override
    assert not mixed.is_homogeneous
    assert not mixed.is_turboquant
    with pytest.raises(ValueError, match="mixes quantization schemes"):
        _ = mixed.scheme
    config = mixed.to_config()
    assert config["key_bits"] == 3 and config["value_bits"] == 5
    assert config["key_scheme"] == "uniform"
    assert config["value_scheme"] == "turboquant"
    assert "-ksuniform-vsturboquant" in mixed.fingerprint(12)
    assert vendored_kv_quant.kv_quant_fingerprint(4, 64, "uniform", 0, 3, 5).endswith(
        "-k3-v5"
    )

    with pytest.raises(ValueError, match="unknown KV quantization scheme"):
        vendored_kv_quant.from_legacy(4, kv_key_scheme="unknown")
    with pytest.raises(ValueError, match="requires integer bits"):
        vendored_kv_quant.from_legacy(
            3.5,
            kv_key_scheme="uniform",
            kv_key_bits=3.5,
        )
    converted = vendored_kv_quant.from_legacy(4, kv_key_scheme="turboquant")
    assert converted is not None and converted.key.bits == 4
    assert vendored_kv_quant.from_config({}) is None


def test_vision_feature_cache_bytes_transparency_and_existing_key_refresh():
    pytest.importorskip("PIL.Image")
    from PIL import Image

    image = Image.new("P", (1, 1))
    image.putpalette([1, 2, 3] * 256)
    image.info["transparency"] = b"\x00\xff"
    cache = vendored_vision_cache.VisionFeatureCache(max_size=1)
    key = cache._make_key(image)
    assert key.startswith("p:")
    cache.put("same", "one")
    cache.put("same", "two")
    assert cache.get("same") == "two"


class _CoordinatorManager:
    def __init__(self):
        import threading
        import types

        self.prepared = []
        self.released = []
        self.exact_cache_guard_tokens = 1
        self.checkpoint_interval_tokens = 4
        self._exact_cache_max = 3
        self.disk = None
        self.block_size = 2
        self.exact_cache_min_tokens = 2
        self.lock = threading.Lock()
        self.stats = types.SimpleNamespace(memory_skips=0)

    def prepare_prefill(self, count):
        self.prepared.append(count)

    def release(self, blocks):
        self.released.append(tuple(blocks))

    def store_exact_cache(self, token_ids, snapshot, *, extra_hash):
        self.stored = (list(token_ids), snapshot, extra_hash)
        return True

    def _make_room(self, _size):
        return True


def test_coordinator_block_and_checkpoint_paths(monkeypatch):
    import types

    import mlx_vlm.apc as upstream_apc

    manager = _CoordinatorManager()
    block = apc_coordinator.APCCoordinator(manager, _coordinator_model())
    assert block.enabled and block.strategy == "block" and not block.is_checkpoint
    assert block.legacy_mode == "block"
    block.prepare_prefill(9)
    assert manager.prepared == [9]

    monkeypatch.setattr(
        upstream_apc,
        "apc_lookup_plan",
        lambda *_a, **_kw: {"matched_blocks": ["b"]},
    )
    hit = block.lookup(
        [1, 2, 3],
        extra_hash=7,
        safe_lookup_min=1,
        suffix_is_text_only=lambda _n: True,
        prefix_has_media=lambda _n: False,
    )
    assert hit is not None and hit["cache_plan"] is block.plan
    monkeypatch.setattr(upstream_apc, "apc_lookup_plan", lambda *_a, **_kw: None)
    assert (
        block.lookup(
            [1],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _n: True,
            prefix_has_media=lambda _n: False,
        )
        is None
    )

    monkeypatch.setattr(
        upstream_apc,
        "make_warm_batch_kv_cache_multi",
        lambda picks, **_kw: (["merged-block"], len(picks)),
    )
    assert block.merge_rows([None, None], [0, 0]) == (["merged-block"], 2)
    monkeypatch.setattr(
        upstream_apc,
        "make_warm_kv_cache",
        lambda blocks, **_kw: ["warm", *blocks],
    )
    assert block.materialize_single(
        {"warm_cache": ["ready"]}, min_capacity_tokens=1
    ) == ["ready"]
    assert block.materialize_single(
        {"matched_blocks": ["cold"]}, min_capacity_tokens=1
    ) == ["warm", "cold"]
    monkeypatch.setattr(upstream_apc, "commit_prefix_blocks", lambda *_a, **_kw: None)
    assert block.commit(["cache"], [1, 2], blocks_in_use=["lease"])
    block.release_hit({"matched_blocks": ["lease"]})
    block.release_hit(None)

    checkpoint_model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            make_cache=lambda: [vendored_cache.ArraysCache(1)]
        )
    )
    checkpoint = apc_coordinator.APCCoordinator(manager, checkpoint_model)
    assert checkpoint.is_checkpoint and checkpoint.legacy_mode == "exact"
    monkeypatch.setattr(
        upstream_apc,
        "adjust_prefix_to_text_suffix_boundary",
        lambda _tokens, boundary, _media, **_kw: boundary,
    )
    assert checkpoint.checkpoint_len(list(range(10)), set()) == 9
    assert checkpoint.checkpoint_lengths(list(range(10)), set()) == [4, 8, 9]
    manager.checkpoint_interval_tokens = 0
    assert checkpoint.checkpoint_lengths(list(range(10)), set()) == [9]

    monkeypatch.setattr(
        upstream_apc,
        "make_warm_batch_exact_cache_multi",
        lambda rows, _lens, **_kw: (rows, 4),
    )
    rows, prefix = checkpoint.merge_rows([None, {"warm_cache": ["warm"]}], [0, 4])
    assert prefix == 4 and len(rows) == 2

    monkeypatch.setattr(upstream_apc, "_prompt_cache_is_batch_shaped", lambda _c: False)
    monkeypatch.setattr(
        upstream_apc, "snapshot_prompt_cache_row", lambda *_a, **_kw: ["snap"]
    )
    assert checkpoint.store_checkpoint([1, 2], ["cache"], extra_hash=8)
    assert checkpoint.commit(["cache"], [1, 2], blocks_in_use=["checkpoint-lease"])
    assert manager.released[-1] == ("checkpoint-lease",)


def test_coordinator_disabled_and_storage_guard_paths(monkeypatch):
    import types

    import mlx_vlm.apc as upstream_apc

    disabled = apc_coordinator.APCCoordinator(None, _coordinator_model())
    disabled.prepare_prefill(1)
    assert disabled.strategy is None and disabled.legacy_mode is None
    assert (
        disabled.lookup(
            [1],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _n: True,
            prefix_has_media=lambda _n: False,
        )
        is None
    )
    assert disabled.checkpoint_len([1], set()) == 0
    assert not disabled.commit([], [])
    disabled.release_hit({"matched_blocks": [1]})

    manager = _CoordinatorManager()
    manager.disk = types.SimpleNamespace(
        flush=lambda: setattr(manager, "flushed", True)
    )
    manager._make_room = lambda _size: False
    checkpoint_model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            make_cache=lambda: [vendored_cache.ArraysCache(1)]
        )
    )
    checkpoint = apc_coordinator.APCCoordinator(manager, checkpoint_model)
    monkeypatch.setattr(upstream_apc, "_prompt_cache_is_batch_shaped", lambda _c: True)
    monkeypatch.setattr(upstream_apc, "_cache_nbytes", lambda _c: 10)
    assert not checkpoint.store_checkpoint([1], ["cache"])
    assert manager.flushed and manager.stats.memory_skips == 1
    manager._make_room = lambda _size: True
    monkeypatch.setattr(
        upstream_apc, "snapshot_prompt_cache_row", lambda *_a, **_kw: None
    )
    assert not checkpoint.store_checkpoint([1], ["cache"])
    # A non-checkpoint plan cannot store exact snapshots.
    block = apc_coordinator.APCCoordinator(manager, _coordinator_model())
    assert not block.store_checkpoint([1], ["cache"])

    monkeypatch.setattr(checkpoint, "checkpoint_len", lambda *_a, **_kw: 0)
    assert checkpoint.checkpoint_lengths([1], set()) == []


def test_coordinator_fresh_cache_falls_back_to_upstream_factory(monkeypatch):
    import types

    import mlx_vlm.models.cache as upstream_cache

    model = types.SimpleNamespace(language_model=object())
    coordinator = object.__new__(apc_coordinator.APCCoordinator)
    coordinator.manager = None
    coordinator.model = model
    monkeypatch.setattr(
        upstream_cache,
        "make_prompt_cache",
        lambda _model: [vendored_cache.KVCache()],
    )
    fresh = coordinator.fresh_cache()
    assert len(fresh) == 1 and type(fresh[0]) is vendored_cache.KVCache
