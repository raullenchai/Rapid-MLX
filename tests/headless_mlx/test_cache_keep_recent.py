"""Partial release contracts without requiring tensor evaluation or Metal."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.engine_core import AsyncEngineCore, EngineCore
from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.paged_cache import PagedCacheManager
from rapid_mlx.prefix_cache import (
    BlockAwarePrefixCache,
    BlockCacheEntry,
    PrefixCacheManager,
)
from rapid_mlx.routes import health
from rapid_mlx.scheduler import Scheduler


def scheduler(variant):
    sched = Scheduler.__new__(Scheduler)
    sched.memory_aware_cache = None
    sched.prefix_cache = None
    sched.block_aware_cache = None
    sched.requests = {}
    sched.running = {}
    if variant == "memory":
        sched.memory_aware_cache = MemoryAwarePrefixCache(
            object(), MemoryCacheConfig(max_memory_mb=64)
        )
    elif variant == "trie":
        sched.prefix_cache = PrefixCacheManager(object())
    elif variant == "paged":
        sched.block_aware_cache = BlockAwarePrefixCache(
            object(), PagedCacheManager(block_size=4, max_blocks=32)
        )
    return sched


def populate(sched):
    for i in range(3):
        array = SimpleNamespace(nbytes=128)
        if sched.memory_aware_cache is not None:
            # Exercise the actual eviction ledger without invoking tensors.
            cache = sched.memory_aware_cache
            from rapid_mlx.memory_cache import _CacheEntry

            entry = _CacheEntry(tokens=(i,), cache=[], memory_bytes=128)
            cache._entries[(i,)] = entry
            cache._sorted_keys.append((i,))
            cache._ledger_add(entry)
        elif sched.prefix_cache is not None:
            sched.prefix_cache.store_cache([i], [SimpleNamespace(state=(array, array))])
        else:
            cache = sched.block_aware_cache
            paged = cache.paged_cache
            table = paged.create_block_table(str(i))
            block = paged.allocate_block()
            block.cache_data = [(array, array)]
            table.block_ids.append(block.block_id)
            table.num_tokens = 4
            cache._request_tables[str(i)] = BlockCacheEntry(table, last_access=float(i))


@pytest.mark.parametrize("variant", ["memory", "trie", "paged"])
@pytest.mark.parametrize("keep", [0, 1, 3, 10])
def test_release_counts_bytes_and_mru(variant, keep):
    sched = scheduler(variant)
    populate(sched)
    result = sched.retain_prefix_cache(keep)
    kept = min(keep, 3)
    size = 128 if variant == "memory" else 256
    assert result == {
        "entries_freed": 3 - kept,
        "entries_kept": kept,
        "bytes_freed": (3 - kept) * size,
        "bytes_kept": kept * size,
    }
    assert sched.get_cache_stats()["last_release"] == result
    if keep == 1:
        if variant == "memory":
            assert list(sched.memory_aware_cache._entries) == [(2,)]
        elif variant == "trie":
            assert sched.prefix_cache.fetch_cache([2, 5])[1] == [5]
        else:
            assert list(sched.block_aware_cache._request_tables) == ["2"]


@pytest.mark.parametrize("keep", [-1, True, 1.5, "1"])
def test_invalid_count(keep):
    with pytest.raises(ValueError, match="nonnegative integer"):
        scheduler("none").retain_prefix_cache(keep)


def test_no_cache():
    sched = scheduler("none")
    assert sched.retain_prefix_cache(1) == dict(
        entries_freed=0, entries_kept=0, bytes_freed=0, bytes_kept=0
    )
    assert sched.get_cache_stats() is None


def test_active_paged_table_and_shared_blocks_survive():
    sched = scheduler("paged")
    populate(sched)
    cache = sched.block_aware_cache
    sched.requests["0"] = object()
    # Entry 1 shares its block with an active fetch which is not a stored entry.
    shared = cache.paged_cache.allocated_blocks[
        cache._request_tables["1"].block_table.block_ids[0]
    ]
    shared.ref_count += 1
    result = sched.retain_prefix_cache(1)
    assert set(cache._request_tables) == {"0", "2"}
    assert result["entries_freed"] == 1
    assert result["bytes_freed"] == 0
    assert shared.cache_data is not None
    assert shared.ref_count == 1
    assert result["bytes_kept"] == 768


def test_pinned_trie_entry_survives():
    sched = scheduler("trie")
    populate(sched)
    assert sched.prefix_cache.pin_prefix([0])
    assert sched.retain_prefix_cache(0)["entries_kept"] == 1
    assert sched.prefix_cache.fetch_cache([0, 9])[1] == [9]


def test_worker_and_engine_forwarding():
    sched = scheduler("none")
    core = EngineCore.__new__(EngineCore)
    core.scheduler = sched
    calls = []
    original = sched.retain_prefix_cache

    def release(keep_recent):
        calls.append(threading.get_ident())
        return original(keep_recent)

    sched.retain_prefix_cache = release
    with ThreadPoolExecutor(max_workers=1) as executor:
        core._mlx_executor = executor
        wrapper = AsyncEngineCore.__new__(AsyncEngineCore)
        wrapper.engine = core
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = None
        engine._engine = wrapper
        assert engine.retain_prefix_cache(1)["entries_kept"] == 0
        assert calls == [executor.submit(threading.get_ident).result()]
        assert calls[0] != threading.get_ident()
    engine._engine = None
    with pytest.raises(RuntimeError, match="unavailable"):
        engine.retain_prefix_cache(1)
    engine._mllm_scheduler = object()
    with pytest.raises(RuntimeError, match="unavailable"):
        engine.retain_prefix_cache(1)


def client(monkeypatch, engine, api_key=None):
    monkeypatch.setattr(health, "get_config", lambda: SimpleNamespace(engine=engine))
    from rapid_mlx.config import get_config

    monkeypatch.setattr(get_config(), "api_key", api_key)
    app = FastAPI()
    app.include_router(health.admin_router)
    app.include_router(health.router)
    install_exception_handlers(app)
    return TestClient(app)


def test_route_and_text_stats(monkeypatch):
    sched = scheduler("trie")
    populate(sched)
    engine = SimpleNamespace(
        retain_prefix_cache=sched.retain_prefix_cache,
        get_cache_stats=sched.get_cache_stats,
        is_mllm=False,
    )
    c = client(monkeypatch, engine)
    response = c.post("/v1/cache/clear", json={"keep_recent": 1})
    assert response.status_code == 200
    data = response.json()
    assert data.pop("status") == "ok"
    assert data["entries_freed"] == 2
    assert c.get("/v1/cache/stats").json()["prefix_cache"]["last_release"] == data


@pytest.mark.parametrize(
    "body",
    [
        {"keep_recent": -1},
        {"keep_recent": True},
        {"keep_recent": "1"},
        {"keep_recent": 1.1},
        {"keep_recent": None},
        {"keep_recent": []},
        {"keep_recent": {}},
        {},
    ],
)
def test_route_validation(monkeypatch, body):
    engine = SimpleNamespace(retain_prefix_cache=Mock())
    response = client(monkeypatch, engine).post("/v1/cache/clear", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"].startswith("Invalid request body: keep_recent:")
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_request"
    assert error["param"] == "keep_recent"
    engine.retain_prefix_cache.assert_not_called()


def test_route_validation_redacts_unknown_fields_and_values(monkeypatch):
    engine = SimpleNamespace(retain_prefix_cache=Mock())
    c = client(monkeypatch, engine)
    response = c.post(
        "/v1/cache/clear",
        json={"keep_recent": "private-value", "private-key": "private-value"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "keep_recent"
    assert "private-key" not in response.text
    assert "private-value" not in response.text
    response = c.post(
        "/v1/cache/clear", json={"keep_recent": 1, "private-key": "private-value"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] is None
    assert (
        "<field>: Extra inputs are not permitted" in response.json()["error"]["message"]
    )
    assert "private-key" not in response.text
    assert "private-value" not in response.text
    engine.retain_prefix_cache.assert_not_called()


def test_route_missing_engine_backend_and_conflict(monkeypatch):
    assert (
        client(monkeypatch, None)
        .post("/v1/cache/clear", json={"keep_recent": 1})
        .status_code
        == 503
    )
    assert (
        client(monkeypatch, object())
        .post("/v1/cache/clear", json={"keep_recent": 1})
        .status_code
        == 409
    )
    engine = SimpleNamespace(
        retain_prefix_cache=Mock(side_effect=RuntimeError("unavailable"))
    )
    assert (
        client(monkeypatch, engine)
        .post("/v1/cache/clear", json={"keep_recent": 1})
        .status_code
        == 409
    )


def test_route_requires_existing_admin_auth(monkeypatch):
    c = client(monkeypatch, object(), api_key="test-secret")
    assert c.post("/v1/cache/clear", json={"keep_recent": 1}).status_code == 401


def test_no_body_preserves_full_clear(monkeypatch):
    engine = SimpleNamespace(clear_prefix_cache=Mock(return_value=True))
    response = client(monkeypatch, engine).post("/v1/cache/clear")
    assert response.status_code == 200
    assert response.json()["scheduler_cache_cleared"] is True
    engine.clear_prefix_cache.assert_called_once_with(reset_stats=False)


@pytest.mark.parametrize("active_map", ["requests", "running"])
def test_keep_zero_does_not_release_active_sole_owner(active_map):
    sched = scheduler("paged")
    populate(sched)
    getattr(sched, active_map)["0"] = object()
    result = sched.retain_prefix_cache(0)
    assert result["entries_kept"] == 1
    assert result["bytes_kept"] == 256
    assert set(sched.block_aware_cache._request_tables) == {"0"}


def test_pinned_paged_block_survives_entry_eviction():
    sched = scheduler("paged")
    populate(sched)
    cache = sched.block_aware_cache
    block_id = cache._request_tables["0"].block_table.block_ids[0]
    block = cache.paged_cache.allocated_blocks[block_id]
    block.is_pinned = True
    result = sched.retain_prefix_cache(0)
    assert result["entries_kept"] == 0
    assert result["bytes_kept"] == 256
    assert block.cache_data is not None


@pytest.mark.parametrize("variant", ["memory", "trie", "paged"])
def test_access_order_overrides_insertion_order(variant, monkeypatch):
    sched = scheduler(variant)
    populate(sched)
    if variant == "memory":
        assert sched.memory_aware_cache.fetch([0, 9])[1] == [9]
    elif variant == "trie":
        assert sched.prefix_cache.fetch_cache([0, 9])[1] == [9]
    else:
        cache = sched.block_aware_cache
        table = cache._request_tables["0"].block_table
        monkeypatch.setattr(cache, "_acquire_candidate_blocks", lambda *_: (table, [9]))
        monkeypatch.setattr(cache, "_reconstruct_from_table", lambda *_: [object()])
        assert cache.fetch_cache("touch", [0, 9])[1] == [9]
    sched.retain_prefix_cache(1)
    if variant == "memory":
        assert list(sched.memory_aware_cache._entries) == [(0,)]
    elif variant == "trie":
        assert sched.prefix_cache.fetch_cache([0, 9])[1] == [9]
    else:
        assert list(sched.block_aware_cache._request_tables) == ["0"]


def test_allocator_error_preserves_release_accounting(monkeypatch):
    sched = scheduler("trie")
    populate(sched)
    monkeypatch.setattr(
        "rapid_mlx.scheduler.mx",
        SimpleNamespace(
            clear_cache=Mock(side_effect=RuntimeError("allocator failure"))
        ),
    )
    with pytest.raises(RuntimeError, match="allocator failure"):
        sched.retain_prefix_cache(1)
    assert sched.get_cache_stats()["last_release"]["entries_freed"] == 2
