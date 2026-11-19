"""Partial manual cache release keeps warm prefixes across cache backends."""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from tests.test_prefix_cache_pressure_eviction import (
    TestBlockAwareCacheEviction as _PagedFixture,
)
from tests.test_prefix_cache_pressure_eviction import (
    _kv_layer_states,
    _make_scheduler_with_legacy_cache,
    _make_scheduler_with_memory_aware_cache,
)


@pytest.mark.parametrize("variant", ["memory", "trie", "paged"])
@pytest.mark.parametrize("touch_oldest", [False, True])
def test_keep_recent_preserves_newest_and_warm_followup(variant, touch_oldest):
    if variant == "memory":
        sched = _make_scheduler_with_memory_aware_cache()
    elif variant == "trie":
        sched = _make_scheduler_with_legacy_cache()
    else:
        sched = _PagedFixture()._make_paged_scheduler()
    for i in range(3):
        tokens = [i + 1] * 8
        states = _kv_layer_states(8, base=float(i))
        if variant == "paged":
            sched.block_aware_cache.store_cache(str(i), tokens, states)
        elif variant == "memory":
            sched.memory_aware_cache.store(tokens, states)
        else:
            from mlx_lm.models.cache import KVCache

            cache = KVCache()
            cache.state = states[0]["state"]
            sched.prefix_cache.store_cache(tokens, [cache])
    newest = 3
    if touch_oldest:
        newest = 1
        if variant == "paged":
            table, _ = sched.block_aware_cache.fetch_cache("touch", [1] * 8 + [4])
            assert table is not None
            sched.block_aware_cache.release_cache("touch")
        elif variant == "memory":
            assert sched.memory_aware_cache.fetch([1] * 8 + [4])[0] is not None
        else:
            assert sched.prefix_cache.fetch_cache([1] * 8 + [4])[0] is not None
    result = sched.retain_prefix_cache(1)
    assert result["entries_freed"] == 2
    assert result["entries_kept"] == 1
    assert result["bytes_freed"] > 0
    assert result["bytes_kept"] > 0
    assert sched.get_cache_stats()["last_release"] == result
    if variant == "paged":
        table, remaining = sched.block_aware_cache.fetch_cache(
            "followup", [newest] * 8 + [4]
        )
        assert table is not None
    elif variant == "memory":
        cache, remaining = sched.memory_aware_cache.fetch([newest] * 8 + [4])
        assert cache is not None
    else:
        cache, remaining = sched.prefix_cache.fetch_cache([newest] * 8 + [4])
        assert cache is not None
    assert remaining == [4]


def test_active_paged_fetch_survives_partial_release():
    import mlx.core as mx

    sched = _PagedFixture()._make_paged_scheduler()
    cache = sched.block_aware_cache
    for i in range(3):
        cache.store_cache(str(i), [i + 1] * 8, _kv_layer_states(8, base=float(i)))
    table, remaining = cache.fetch_cache("active", [1] * 8 + [4])
    assert table is not None and remaining == [4]
    sched.requests["active"] = object()
    blocks = [cache.paged_cache.allocated_blocks[bid] for bid in table.block_ids]
    snapshots = [mx.array(block.cache_data[0][0]) for block in blocks]
    # Make a different conversation MRU so the active prefix's stored owner
    # is evicted. Its fetch-held references must keep its tensors intact.
    assert cache.fetch_cache("touch", [3] * 8 + [4])[0] is not None
    cache.release_cache("touch")
    result = sched.retain_prefix_cache(1)
    assert result["entries_freed"] == 2
    for block, expected in zip(blocks, snapshots, strict=True):
        assert block.cache_data is not None
        assert block.ref_count == 1
        assert mx.array_equal(block.cache_data[0][0], expected).item()
    assert cache.reconstruct_cache(table) is not None
