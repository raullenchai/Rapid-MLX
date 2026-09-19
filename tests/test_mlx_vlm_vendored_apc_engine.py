# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored APC engine's dual-namespace recognition and the
documented ``upstream-bugfix`` hunks in ``mlx_vlm_vendored/apc.py``.

The dual-namespace helpers keep the engine's exact-type tables, snapshot/
restore constructors, and checkpoint-class resolver namespace-complete
during the mlx-vlm transition (see the vendor-mllm-primitives design note).
Each ``upstream-bugfix`` test reproduces a defect against the pinned
upstream 0.7.1 source.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.mlx_vlm_vendored import apc
from rapid_mlx.models.mlx_vlm_vendored import cache as vendored_cache

_SHARD = apc.DiskBlockStore.SHARD_PREFIX + "0" * 32


def _upstream_cache_module():
    try:
        from mlx_vlm.models import cache as upstream_cache
    except ImportError:  # pragma: no cover - stripped installation
        pytest.skip("mlx_vlm not installed")
    return upstream_cache


def _dense_kv(tokens: int, *, offset: int | None = None):
    c = vendored_cache.KVCache()
    c.keys = mx.zeros((1, 1, tokens, 2))
    c.values = mx.zeros((1, 1, tokens, 2))
    c.offset = tokens if offset is None else offset
    return c


@pytest.fixture()
def store(tmp_path):
    s = apc.DiskBlockStore(tmp_path)
    yield s
    s.close()


def test_dense_checkpoint_trimmable_accepts_both_namespaces(store):
    upstream_cache = _upstream_cache_module()

    assert apc._dense_checkpoint_trimmable([_dense_kv(4)], 4)

    u = upstream_cache.KVCache()
    u.keys = mx.zeros((1, 1, 4, 2))
    u.values = mx.zeros((1, 1, 4, 2))
    u.offset = 4
    assert apc._dense_checkpoint_trimmable([u], 4)
    assert apc._dense_checkpoint_trimmable([_dense_kv(4), u], 4)
    assert not apc._dense_checkpoint_trimmable([_dense_kv(4, offset=2)], 4)
    assert not apc._dense_checkpoint_trimmable([], 4)


def test_exact_snapshot_records_vendored_namespace(store):
    upstream_cache = _upstream_cache_module()

    arrays: dict = {}
    metadata: dict = {}
    assert store._snapshot_exact_cache_entry(
        vendored_cache.KVCache(), "c0", arrays, metadata
    )
    assert metadata["c0_kind"] == "kv"
    assert metadata["c0_ns"] == "v"

    arrays_up: dict = {}
    metadata_up: dict = {}
    assert store._snapshot_exact_cache_entry(
        upstream_cache.KVCache(), "c1", arrays_up, metadata_up
    )
    assert metadata_up["c1_kind"] == "kv"
    assert "c1_ns" not in metadata_up


def test_exact_restore_is_namespace_faithful(store):
    upstream_cache = _upstream_cache_module()

    metadata: dict = {}
    assert store._snapshot_exact_cache_entry(
        vendored_cache.KVCache(), "c0", {}, metadata
    )
    restored = store._load_exact_cache_entry(
        None,
        {},
        metadata,
        0,
        "c0",
        min_capacity_tokens=None,
        eval_targets=[],
    )
    assert type(restored) is vendored_cache.KVCache

    metadata_up: dict = {}
    assert store._snapshot_exact_cache_entry(
        upstream_cache.KVCache(), "c1", {}, metadata_up
    )
    restored_up = store._load_exact_cache_entry(
        None,
        {},
        metadata_up,
        0,
        "c1",
        min_capacity_tokens=None,
        eval_targets=[],
    )
    assert type(restored_up) is upstream_cache.KVCache

    unknown_namespace = dict(metadata_up)
    unknown_namespace["c1_ns"] = "future"
    assert (
        store._load_exact_cache_entry(
            None,
            {},
            unknown_namespace,
            0,
            "c1",
            min_capacity_tokens=None,
            eval_targets=[],
        )
        is None
    )


def test_resolve_checkpoint_class_allows_only_reviewed_cache_classes(store):
    from mlx_vlm import turboquant
    from mlx_vlm.models import cache as upstream_cache
    from mlx_vlm.turboquant import TurboQuantKVCache

    for module, base in (
        (upstream_cache, upstream_cache._BaseCache),
        (turboquant, upstream_cache._BaseCache),
        (vendored_cache, vendored_cache._BaseCache),
    ):
        concrete_cache_classes = {
            name
            for name, value in vars(module).items()
            if not name.startswith("_")
            and isinstance(value, type)
            and value.__module__ == module.__name__
            and issubclass(value, base)
        }
        assert apc._CHECKPOINT_CLASS_ALLOWLIST[module.__name__] == frozenset(
            concrete_cache_classes
        )

    assert (
        apc._resolve_checkpoint_class(
            "rapid_mlx.models.mlx_vlm_vendored.cache", "KVCache"
        )
        is vendored_cache.KVCache
    )
    assert apc._resolve_checkpoint_class("os", "system") is None
    assert apc._resolve_checkpoint_class("builtins", "exec") is None
    assert apc._resolve_checkpoint_class("mlx_vlm.apc", "APCCoordinator") is None
    assert apc._resolve_checkpoint_class("mlx_vlm.models.cache", "_BaseCache") is None
    assert (
        apc._resolve_checkpoint_class("mlx_vlm.models.cache", "KVCache.__class__")
        is None
    )
    assert (
        apc._resolve_checkpoint_class(
            "rapid_mlx.models.mlx_vlm_vendored.apc", "DiskBlockStore"
        )
        is None
    )
    assert (
        apc._resolve_checkpoint_class("mlx_vlm.models.cache", "KVCache")
        is upstream_cache.KVCache
    )
    assert (
        apc._resolve_checkpoint_class("mlx_vlm.turboquant", "TurboQuantKVCache")
        is TurboQuantKVCache
    )


def test_rebuild_index_excludes_dropped_shard_bytes(store):
    """upstream-bugfix: an unreadable shard must not inflate _disk_bytes."""
    good = store.dir / f"{_SHARD}{store.SUFFIX}"
    mx.save_safetensors(
        str(good), {"k": mx.zeros((2, 2))}, metadata={"block_hashes": "11,12"}
    )
    good_size = good.stat().st_size
    corrupt = store.dir / f"{apc.DiskBlockStore.SHARD_PREFIX}{'a' * 32}{store.SUFFIX}"
    corrupt.write_bytes(b"not a safetensors header")
    invalid_exact = (
        store.dir / f"{apc.DiskBlockStore.EXACT_PREFIX}{'b' * 32}{store.SUFFIX}"
    )
    mx.save_safetensors(
        str(invalid_exact), {"k": mx.zeros((1, 1))}, metadata={"cache_hash": "bad"}
    )
    invalid_block = (
        store.dir / f"{apc.DiskBlockStore.SHARD_PREFIX}{'c' * 32}{store.SUFFIX}"
    )
    mx.save_safetensors(
        str(invalid_block),
        {"k": mx.zeros((1, 1))},
        metadata={"block_hashes": "not-an-int"},
    )

    total = store._rebuild_index()
    assert total == good_size
    assert not corrupt.exists()
    assert not invalid_exact.exists()
    assert not invalid_block.exists()
    # Mirror the production call site, which assigns the returned total.
    store._disk_bytes = store._rebuild_index()
    assert store.disk_bytes == good_size


def test_rebuild_index_counts_invalid_shard_when_unlink_fails(store, monkeypatch):
    invalid = store.dir / f"{apc.DiskBlockStore.EXACT_PREFIX}{'d' * 32}{store.SUFFIX}"
    mx.save_safetensors(
        str(invalid), {"k": mx.zeros((1, 1))}, metadata={"cache_hash": "bad"}
    )
    invalid_size = invalid.stat().st_size
    original_unlink = Path.unlink

    def fail_for_invalid(path, *args, **kwargs):
        if path == invalid:
            raise OSError("injected unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_for_invalid)

    assert store._rebuild_index() == invalid_size
    assert invalid.exists()
    assert store.num_exact_indexed == 0


def test_finish_write_release_requires_event_ownership(store):
    """upstream-bugfix: overlapping writers must not erase another's entry."""
    ev_owner, ev_foreign = threading.Event(), threading.Event()
    store._in_flight[7] = ev_owner

    store._finish_write([7], ev_foreign)
    assert store._in_flight.get(7) is ev_owner
    assert not ev_owner.is_set()

    store._finish_write([7], ev_owner)
    assert 7 not in store._in_flight
    assert ev_owner.is_set()


def test_save_layer_major_shard_cleans_temp_on_failure(store, monkeypatch):
    """upstream-bugfix: a failed write must not leak the temporary shard."""
    path = store.dir / f"{_SHARD}{store.SUFFIX}"
    blocks = [SimpleNamespace(block_hash=1)]
    written_temps = []

    def fail_after_creating_temp(path, *_args, **_kwargs):
        tmp = Path(path)
        tmp.write_bytes(b"partial shard")
        written_temps.append(tmp)
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(mx, "save_safetensors", fail_after_creating_temp)

    with pytest.raises(RuntimeError, match="injected serialization failure"):
        store._save_layer_major_shard(
            path,
            blocks,
            {},
            [mx.zeros((1, 1, 4, 2))],
            [mx.zeros((1, 1, 4, 2))],
            4,
        )
    assert len(written_temps) == 1
    assert not written_temps[0].exists()
    assert list(store.dir.glob(f"*{store.SUFFIX}")) == []
