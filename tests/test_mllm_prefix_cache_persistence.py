# SPDX-License-Identifier: Apache-2.0
"""Lifespan-only persistence of the vision scheduler's text exact snapshots."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.mllm_batch_generator import MLLMBatchGenerator
from rapid_mlx.mllm_scheduler import MLLMScheduler
from rapid_mlx.models.mlx_vlm_vendored import apc
from rapid_mlx.models.mlx_vlm_vendored.cache import ArraysCache, KVCache


def generator(capacity=8):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.model = SimpleNamespace(config=SimpleNamespace(image_token_id=999))
    gen._prefix_cache = apc.from_env(
        overrides={"enabled": True, "num_blocks": 0, "disk_enabled": False}
    )
    gen._prefix_cache.exact_cache_min_tokens = 1
    gen._prefix_cache._exact_cache_max = capacity
    gen._prefix_cache_mode = "exact"
    gen._prefix_cache_enabled = True
    gen._prefix_cache_extra_hash = 17
    gen._prefix_cache_max_bytes = 0
    gen._prefix_cache_budget_evictions = 0
    return gen


def tensors(tag=1):
    kv = KVCache()
    kv.update_and_fetch(mx.full((1, 1, 3, 4), float(tag)), mx.ones((1, 1, 3, 4)))
    arrays = ArraysCache(2)
    arrays.cache = [mx.full((1, 2), float(tag)), mx.zeros((1, 3))]
    return [kv, arrays]


def add(gen, tokens=(1, 2, 3), salt=17, tag=1):
    assert gen._prefix_cache.store_exact_cache(tokens, tensors(tag), extra_hash=salt)
    return next(reversed(gen._prefix_cache._exact_cache))


def scheduler(gen, executor=None, injected=None):
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.batch_generator = gen
    sched._step_executor = executor
    sched._injected_step_executor = injected
    return sched


def test_round_trip(tmp_path):
    source = generator()
    add(source)
    add(source, (4, 5, 6), tag=2)
    original_keys = list(source._prefix_cache._exact_cache)
    assert source.save_exact_prefix_cache_to_disk(tmp_path / "cache")
    assert source._prefix_cache.disk is None  # no request-time disk store
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(tmp_path / "cache") == 2
    assert list(fresh._prefix_cache._exact_cache) == original_keys
    for tokens, tag in [((1, 2, 3), 1), ((4, 5, 6), 2)]:
        hit, length = fresh._prefix_cache.lookup_exact_cache(
            (*tokens, 7), extra_hash=17
        )
        assert length == 3
        assert mx.array_equal(hit[0].keys[..., :3, :], tensors(tag)[0].keys[..., :3, :])
        assert mx.array_equal(
            hit[0].values[..., :3, :], tensors(tag)[0].values[..., :3, :]
        )
        for actual, expected in zip(hit[1].cache, tensors(tag)[1].cache):
            assert mx.array_equal(actual, expected)
    # Reading must not change persisted LRU order.
    assert generator().load_exact_prefix_cache_from_disk(tmp_path / "cache") == 2


@pytest.mark.parametrize("execution", ["direct", "owned", "injected"])
def test_engine_delegates_on_step_executor(tmp_path, execution, monkeypatch):
    source = generator()
    add(source)
    caller = threading.get_ident()
    seen = []
    original_save = source.save_exact_prefix_cache_to_disk
    original_load = source.load_exact_prefix_cache_from_disk

    def save(*args):
        seen.append(threading.get_ident())
        return original_save(*args)

    def load(*args):
        seen.append(threading.get_ident())
        return original_load(*args)

    monkeypatch.setattr(source, "save_exact_prefix_cache_to_disk", save)
    monkeypatch.setattr(source, "load_exact_prefix_cache_from_disk", load)
    with ThreadPoolExecutor(max_workers=1) as worker:
        sched = scheduler(
            source,
            worker if execution == "owned" else None,
            worker if execution == "injected" else None,
        )
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = True
        engine._engine = None
        engine._mllm_scheduler = sched
        base = str(tmp_path / "cache")
        assert engine.save_cache_to_disk(base + "/")
        assert not Path(base).exists()
        assert Path(base + ".mllm-exact/index.json").is_file()
        source._prefix_cache._exact_cache.clear()
        assert engine.load_cache_from_disk(base) == 1
    assert all((ident == caller) == (execution == "direct") for ident in seen)


def test_image_and_salt_guards_save_and_load(tmp_path, caplog):
    caplog.set_level("INFO")
    source = generator()
    good = add(source)
    image = add(source, (1, 999, 3))
    foreign = add(source, (4, 5, 6), salt=18)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    assert json.loads((path / "index.json").read_text()) == [good]
    store = apc.DiskBlockStore(path)
    try:
        for key in (image, foreign):
            entry = source._prefix_cache._exact_cache[key]
            assert store.save_exact_cache(
                key,
                entry.token_ids,
                entry.extra_hash,
                entry.prompt_cache,
                synchronous=True,
            )
    finally:
        store.close()
    (path / "index.json").write_text(json.dumps([foreign, image, good]))
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert list(fresh._prefix_cache._exact_cache) == [good]
    assert "foreign extra_hash" in caplog.text
    assert "media tokens" in caplog.text


def test_abort_preserves_previous_and_partial_prefers_mru(tmp_path):
    gen = generator()
    oldest = add(gen)
    path = tmp_path / "cache"
    assert gen.save_exact_prefix_cache_to_disk(path)
    before = (path / "index.json").read_bytes()
    predictions = []

    def abort(predicted):
        predictions.append(predicted)
        return True

    assert not gen.save_exact_prefix_cache_to_disk(path, abort)
    assert predictions[0] > 0
    assert (path / "index.json").read_bytes() == before
    assert generator().load_exact_prefix_cache_from_disk(path) == 1
    newest = add(gen, (4, 5, 6))
    calls = iter([False, True])
    assert gen.save_exact_prefix_cache_to_disk(path, lambda _: next(calls))
    assert json.loads((path / "index.json").read_text()) == [newest]
    assert oldest != newest


def test_live_entries_capacity_and_byte_budget(tmp_path):
    source = generator()
    first = add(source)
    second = add(source, (4, 5, 6), tag=2)
    third = add(source, (7, 8, 9), tag=3)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    fresh = generator(capacity=2)
    add(fresh, tag=42)
    live = fresh._prefix_cache._exact_cache[first]
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert fresh._prefix_cache._exact_cache[first] is live
    assert set(fresh._prefix_cache._exact_cache) == {first, third}
    assert second not in fresh._prefix_cache._exact_cache
    fresh = generator()
    add(fresh, (7, 8, 9), tag=42)  # newest key is skipped, never overwritten
    assert fresh.load_exact_prefix_cache_from_disk(path) == 2
    assert (
        fresh._prefix_cache._exact_cache[third].prompt_cache[1].cache[0][0, 0].item()
        == 42
    )
    fresh = generator()
    fresh._prefix_cache_max_bytes = 1
    assert fresh.load_exact_prefix_cache_from_disk(path) == 0
    assert not fresh._prefix_cache._exact_cache
    assert fresh._prefix_cache_budget_evictions == 0


@pytest.mark.parametrize("budget_owner", ["lane", "apc"])
@pytest.mark.parametrize("live_kind", ["empty", "text", "media"])
def test_restore_admits_bytes_before_loading(
    tmp_path, monkeypatch, budget_owner, live_kind
):
    source = generator()
    oldest = add(source)
    newest = add(source, (4, 5, 6))
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    fresh = generator()
    live_bytes = 0
    if live_kind == "text":
        live_key = add(fresh, (7, 8, 9))
        live_bytes = fresh._exact_entry_bytes(
            fresh._prefix_cache._exact_cache[live_key].prompt_cache
        )
    elif live_kind == "media":
        live_bytes = 100
        fresh._media_boundary_entries = {
            "media": SimpleNamespace(cache_bytes=live_bytes)
        }
    # The newest entry is oversized; skip it and admit only the oldest.
    monkeypatch.setattr(
        apc.DiskBlockStore,
        "exact_cache_bytes",
        lambda self, key: 201 if key == newest else 200,
    )
    budget = live_bytes + 200
    if budget_owner == "lane":
        fresh._prefix_cache_max_bytes = budget
    else:
        fresh._prefix_cache.memory_max_bytes = budget
    loaded_keys = []
    original_load = apc.DiskBlockStore.load_exact_cache

    def load(store, key):
        loaded_keys.append(key)
        return original_load(store, key)

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", load)
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert loaded_keys == [oldest]


@pytest.mark.parametrize("budget_owner", ["lane", "apc"])
def test_restore_counts_pending_bytes(tmp_path, monkeypatch, budget_owner):
    source = generator()
    add(source)
    newest = add(source, (4, 5, 6))
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    fresh = generator()
    if budget_owner == "lane":
        fresh._prefix_cache_max_bytes = 200
    else:
        fresh._prefix_cache.memory_max_bytes = 200
    monkeypatch.setattr(apc.DiskBlockStore, "exact_cache_bytes", lambda *args: 200)
    loaded_keys = []
    original_load = apc.DiskBlockStore.load_exact_cache

    def load(store, key):
        loaded_keys.append(key)
        return original_load(store, key)

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", load)
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert loaded_keys == [newest]


@pytest.mark.parametrize("free_ram", [None, 1200, 1199])
def test_restore_free_ram_floor(tmp_path, monkeypatch, free_ram):
    source = generator()
    add(source)
    add(source, (4, 5, 6))
    newest = add(source, (7, 8, 9))
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    fresh = generator()
    fresh._prefix_cache._disk_min_free_ram_bytes = 1000
    monkeypatch.setattr(apc.DiskBlockStore, "exact_cache_bytes", lambda *args: 200)
    # First entry fits exactly; second stops restoration if below floor + size.
    readings = iter([1200, free_ram, 1200])
    monkeypatch.setattr(apc, "_free_ram_bytes", lambda: next(readings))
    loaded_keys = []
    original_load = apc.DiskBlockStore.load_exact_cache

    def load(store, key):
        loaded_keys.append(key)
        return original_load(store, key)

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", load)
    expected = 1 if free_ram == 1199 else 3
    assert fresh.load_exact_prefix_cache_from_disk(path) == expected
    assert len(loaded_keys) == expected
    assert loaded_keys[0] == newest


@pytest.mark.parametrize("suffix", [".new", ".old"])
def test_crash_recovery(tmp_path, suffix):
    source = generator()
    add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    path.rename(str(path) + suffix)
    if suffix == ".old":
        staging = Path(str(path) + ".new")
        staging.mkdir()
        (staging / "index.json.tmp").write_text("[")
    # An incomplete target must not hide the recoverable completed snapshot.
    path.mkdir()
    assert generator().load_exact_prefix_cache_from_disk(path) == 1
    assert (path / "index.json").is_file()


@pytest.mark.parametrize(
    "disabled", ["manager", "mode", "enabled", "config", "invalid", "unknown"]
)
def test_disabled_and_unresolved_fail_closed(tmp_path, disabled, caplog):
    caplog.set_level("INFO")
    gen = generator()
    add(gen)
    if disabled == "manager":
        gen._prefix_cache = None
    elif disabled == "mode":
        gen._prefix_cache_mode = "block"
    elif disabled == "enabled":
        gen._prefix_cache_enabled = False
    elif disabled == "config":
        gen.model = None
    elif disabled == "invalid":
        gen.model.config.image_token_id = "invalid"
    else:
        gen.model.config = SimpleNamespace()
    assert not gen.save_exact_prefix_cache_to_disk(tmp_path / "cache")
    assert gen.load_exact_prefix_cache_from_disk(tmp_path / "cache") == 0
    assert not list(tmp_path.iterdir())
    assert "disabled" in caplog.text or "unresolved media token IDs" in caplog.text


def test_empty_missing_failed_write_and_unreadable_entry(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    gen = generator()
    path = tmp_path / "cache"
    assert gen.load_exact_prefix_cache_from_disk(path) == 0
    assert not gen.save_exact_prefix_cache_to_disk(path)
    assert "no text-only exact prefix entries to persist" in caplog.text
    add(gen)
    with monkeypatch.context() as patch:
        patch.setattr(apc.DiskBlockStore, "save_exact_cache", lambda *a, **kw: False)
        assert not gen.save_exact_prefix_cache_to_disk(path)
    assert gen.save_exact_prefix_cache_to_disk(path)
    for tensor_file in path.rglob("*.safetensors"):
        tensor_file.unlink()
    assert generator().load_exact_prefix_cache_from_disk(path) == 0


def test_missing_scheduler_and_generator(tmp_path, caplog):
    caplog.set_level("INFO")
    sched = scheduler(None)
    assert not sched.save_prefix_cache_to_disk(str(tmp_path))
    assert sched.load_prefix_cache_from_disk(str(tmp_path)) == 0
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._engine = None
    assert not engine.save_cache_to_disk(str(tmp_path))
    assert engine.load_cache_from_disk(str(tmp_path)) == 0
    assert "no exact prefix scheduler available" in caplog.text
