# SPDX-License-Identifier: Apache-2.0
"""Lifespan-only persistence of the vision scheduler's text exact snapshots."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = [
    pytest.mark.requires_mlx,
    pytest.mark.usefixtures("deterministic_apc_host_memory"),
]

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


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        ("[", "Expecting value"),
        ('{"a":1}', "list of integers"),
        ('["x"]', "list of integers"),
        ("null", "list of integers"),
        ("1", "list of integers"),
        ("[true]", "list of integers"),
        ("[1, false]", "list of integers"),
        ("[1.0]", "list of integers"),
        ("[[]]", "list of integers"),
    ],
)
def test_invalid_index_preserves_live_entries(tmp_path, caplog, contents, reason):
    gen = generator()
    key = add(gen)
    live = gen._prefix_cache._exact_cache[key]
    path = tmp_path / "cache"
    path.mkdir()
    (path / "index.json").write_text(contents)

    assert gen.load_exact_prefix_cache_from_disk(path) == 0
    assert list(gen._prefix_cache._exact_cache) == [key]
    assert gen._prefix_cache._exact_cache[key] is live
    assert "invalid snapshot index" in caplog.text
    assert reason in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


def test_unreadable_index_preserves_live_entries(tmp_path, monkeypatch, caplog):
    gen = generator()
    key = add(gen)
    live = gen._prefix_cache._exact_cache[key]
    path = tmp_path / "cache"
    path.mkdir()
    (path / "index.json").write_text("[]")

    def fail_read(self):
        raise OSError("snapshot read failed")

    monkeypatch.setattr(Path, "read_text", fail_read)
    assert gen.load_exact_prefix_cache_from_disk(path) == 0
    assert list(gen._prefix_cache._exact_cache) == [key]
    assert gen._prefix_cache._exact_cache[key] is live
    assert "invalid snapshot index" in caplog.text
    assert "snapshot read failed" in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


def test_missing_scheduler_and_generator(tmp_path, caplog):
    caplog.set_level("INFO")
    sched = scheduler(None)
    assert not sched.save_prefix_cache_to_disk(str(tmp_path))
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._engine = None
    assert not engine.save_cache_to_disk(str(tmp_path))
    assert engine.load_cache_from_disk(str(tmp_path)) == 0
    assert "no exact prefix scheduler available" in caplog.text


@pytest.mark.parametrize("with_executor", [False, True])
def test_restore_builds_lazy_generator_first(tmp_path, with_executor):
    # The startup restore runs before the first request creates the batch
    # generator; it must build it (on the owner thread) instead of dropping
    # the snapshot.
    source = generator()
    key = add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    with ThreadPoolExecutor(max_workers=1) as worker:
        sched = scheduler(None, worker if with_executor else None)
        built = []

        def ensure():
            if sched.batch_generator is None:
                built.append(threading.get_ident())
                sched.batch_generator = generator()

        sched._ensure_batch_generator = ensure
        assert sched.load_prefix_cache_from_disk(str(path)) == 1
    assert list(sched.batch_generator._prefix_cache._exact_cache) == [key]
    assert (built[0] != threading.get_ident()) == with_executor


@pytest.mark.parametrize(
    "failure", ["construction", "entry", "index", "index_rename", "backup", "final"]
)
def test_save_oserror_preserves_committed_snapshot(
    tmp_path, monkeypatch, caplog, failure
):
    gen = generator()
    old_key = add(gen)
    path = tmp_path / "cache"
    assert gen.save_exact_prefix_cache_to_disk(path)
    before = {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }
    add(gen, (4, 5, 6), tag=2)
    original_rename = Path.rename
    original_write = Path.write_text

    def fail(*args, **kwargs):
        raise OSError("injected snapshot failure")

    def write(self, *args, **kwargs):
        if self.name == "index.json.tmp":
            fail()
        return original_write(self, *args, **kwargs)

    def rename(self, target):
        if (
            (failure == "index_rename" and self.name == "index.json.tmp")
            or (failure == "backup" and self == path)
            or (failure == "final" and self == Path(str(path) + ".new"))
        ):
            fail()
        return original_rename(self, target)

    with monkeypatch.context() as patch:
        if failure == "construction":
            patch.setattr(apc, "DiskBlockStore", fail)
        elif failure == "entry":
            patch.setattr(apc.DiskBlockStore, "save_exact_cache", fail)
        elif failure == "index":
            patch.setattr(Path, "write_text", write)
        else:
            patch.setattr(Path, "rename", rename)
        assert gen.save_exact_prefix_cache_to_disk(path) is False
    assert not Path(str(path) + ".new").exists()
    assert not Path(str(path) + ".old").exists()
    assert {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    } == before
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert list(fresh._prefix_cache._exact_cache) == [old_key]
    # Match the failure record itself: a test that ran earlier in the same
    # process may have enabled INFO logging, so the reload above can log later.
    failures = [r for r in caplog.records if "snapshot save failed" in r.getMessage()]
    assert failures and all(r.levelname == "WARNING" for r in failures)
    assert "snapshot save failed: injected snapshot failure" in caplog.text


def test_save_rollback_oserror_keeps_backup_recoverable(tmp_path, monkeypatch, caplog):
    gen = generator()
    old_key = add(gen)
    path = tmp_path / "cache"
    assert gen.save_exact_prefix_cache_to_disk(path)
    before = (path / "index.json").read_bytes()
    add(gen, (4, 5, 6))
    original_rename = Path.rename

    def rename(self, target):
        if self in (Path(str(path) + ".new"), Path(str(path) + ".old")):
            raise OSError("rename unavailable")
        return original_rename(self, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", rename)
        assert gen.save_exact_prefix_cache_to_disk(path) is False
    assert not Path(str(path) + ".new").exists()
    assert (Path(str(path) + ".old") / "index.json").read_bytes() == before
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert list(fresh._prefix_cache._exact_cache) == [old_key]
    assert "snapshot rollback failed: rename unavailable" in caplog.text


@pytest.mark.parametrize("failure", ["construction", ".new", ".old", "close"])
def test_load_oserror_preserves_live_entries(tmp_path, monkeypatch, caplog, failure):
    source = generator()
    add(source)
    add(source, (4, 5, 6))
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    before = (path / "index.json").read_bytes()
    snapshot_path = path
    if failure in (".new", ".old"):
        snapshot_path = path.rename(str(path) + failure)
    fresh = generator()
    live_key = add(fresh, (7, 8, 9), tag=42)
    live = fresh._prefix_cache._exact_cache[live_key]
    original_close = apc.DiskBlockStore.close

    def fail(*args, **kwargs):
        raise OSError("injected snapshot failure")

    def close(store):
        original_close(store)
        fail()

    with monkeypatch.context() as patch:
        if failure == "construction":
            patch.setattr(apc, "DiskBlockStore", fail)
        elif failure == "close":
            patch.setattr(apc.DiskBlockStore, "close", close)
        else:
            patch.setattr(Path, "rename", fail)
        assert fresh.load_exact_prefix_cache_from_disk(path) == (
            2 if failure == "close" else 0
        )
    assert fresh._prefix_cache._exact_cache[live_key] is live
    if failure != "close":
        assert list(fresh._prefix_cache._exact_cache) == [live_key]
    assert (snapshot_path / "index.json").read_bytes() == before
    assert "snapshot load failed: injected snapshot failure" in caplog.text
    assert caplog.records[-1].levelname == "WARNING"


def test_unreadable_entry_is_skipped_and_others_restore(tmp_path, monkeypatch, caplog):
    source = generator()
    first = add(source)
    second = add(source, (4, 5, 6), tag=2)
    assert source.save_exact_prefix_cache_to_disk(tmp_path / "cache")
    real_load = apc.DiskBlockStore.load_exact_cache

    def flaky(store, key, **kwargs):
        if key == second:
            raise ValueError("corrupt tensor header")
        return real_load(store, key, **kwargs)

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", flaky)
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(tmp_path / "cache") == 1
    assert list(fresh._prefix_cache._exact_cache) == [first]
    assert "skipped unreadable snapshot entry: corrupt tensor header" in caplog.text


def test_entry_that_becomes_live_during_restore_is_not_overwritten(
    tmp_path, monkeypatch
):
    source = generator()
    key = add(source)
    assert source.save_exact_prefix_cache_to_disk(tmp_path / "cache")
    fresh = generator()
    real_load = apc.DiskBlockStore.load_exact_cache

    def racing(store, cache_hash, **kwargs):
        snapshot = real_load(store, cache_hash, **kwargs)
        add(fresh, tag=5)  # a live store of the same prefix lands meanwhile
        return snapshot

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", racing)
    assert fresh.load_exact_prefix_cache_from_disk(tmp_path / "cache") == 0
    live = fresh._prefix_cache._exact_cache[key]
    assert mx.array_equal(live.prompt_cache[1].cache[0], mx.full((1, 2), 5.0))


def test_restore_rejects_decoded_footprint_over_budget_without_evicting_live(
    tmp_path, monkeypatch
):
    source = generator()
    add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    fresh = generator()
    live_key = add(fresh, (7, 8, 9), tag=3)
    live = fresh._prefix_cache._exact_cache[live_key]
    live_bytes = fresh._exact_entry_bytes(live.prompt_cache)
    # The on-disk size understates the decoded footprint: the read is admitted
    # but the decoded entry must not be inserted, and the live entry survives.
    monkeypatch.setattr(apc.DiskBlockStore, "exact_cache_bytes", lambda self, key: 1)
    fresh._prefix_cache_max_bytes = live_bytes + 2
    assert fresh.load_exact_prefix_cache_from_disk(path) == 0
    assert list(fresh._prefix_cache._exact_cache) == [live_key]
    assert fresh._prefix_cache._exact_cache[live_key] is live


def test_restore_skips_entry_whose_size_probe_raises(tmp_path, monkeypatch, caplog):
    source = generator()
    add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)

    def broken(self, key):
        raise ValueError("bad entry metadata")

    monkeypatch.setattr(apc.DiskBlockStore, "exact_cache_bytes", broken)
    assert generator().load_exact_prefix_cache_from_disk(path) == 0
    assert "skipped unreadable snapshot entry: bad entry metadata" in caplog.text


def test_restore_rechecks_budget_at_insertion(tmp_path, monkeypatch):
    source = generator()
    add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    fresh = generator()
    fresh._prefix_cache_max_bytes = 1 << 20
    footprints = iter([0, 0, 1 << 20])  # pre-read, post-read, insertion

    # Simulate live growth that lands after the entry was read and admitted.
    monkeypatch.setattr(fresh, "_exact_cache_footprint_bytes", lambda: next(footprints))
    assert fresh.load_exact_prefix_cache_from_disk(path) == 0
    assert not fresh._prefix_cache._exact_cache


def test_malformed_new_does_not_shadow_valid_old(tmp_path, caplog):
    source = generator()
    key = add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    path.rename(str(path) + ".old")
    broken = Path(str(path) + ".new")
    broken.mkdir()
    (broken / "index.json").write_text("[")
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert list(fresh._prefix_cache._exact_cache) == [key]
    assert "invalid snapshot index in" in caplog.text


def test_restore_rejects_entry_under_a_mismatched_key(tmp_path, monkeypatch, caplog):
    source = generator()
    add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)
    real_load = apc.DiskBlockStore.load_exact_cache

    def swapped(store, cache_hash, **kwargs):
        tokens, salt, prompt_cache = real_load(store, cache_hash, **kwargs)
        return (4, 5, 6), salt, prompt_cache  # KV state under another prefix

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", swapped)
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(path) == 0
    assert not fresh._prefix_cache._exact_cache
    assert "snapshot key does not match its tokens" in caplog.text


def test_save_serialization_error_preserves_committed_snapshot(
    tmp_path, monkeypatch, caplog
):
    gen = generator()
    key = add(gen)
    path = tmp_path / "cache"
    assert gen.save_exact_prefix_cache_to_disk(path)
    before = (path / "index.json").read_bytes()

    def explode(*args, **kwargs):
        raise RuntimeError("cannot serialize cache state")

    # Scope the patch so restoring it keeps the module's pinned host memory.
    with monkeypatch.context() as patch:
        patch.setattr(apc.DiskBlockStore, "save_exact_cache", explode)
        assert not gen.save_exact_prefix_cache_to_disk(path)
    assert (path / "index.json").read_bytes() == before
    assert not Path(str(path) + ".new").exists()
    assert "snapshot save failed: cannot serialize cache state" in caplog.text
    fresh = generator()
    assert fresh.load_exact_prefix_cache_from_disk(path) == 1
    assert list(fresh._prefix_cache._exact_cache) == [key]


def test_restore_memory_error_is_not_swallowed(tmp_path, monkeypatch):
    source = generator()
    add(source)
    path = tmp_path / "cache"
    assert source.save_exact_prefix_cache_to_disk(path)

    def oom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(apc.DiskBlockStore, "load_exact_cache", oom)
    fresh = generator()
    with pytest.raises(MemoryError):
        fresh.load_exact_prefix_cache_from_disk(path)
    assert not fresh._prefix_cache._exact_cache


def test_engine_sibling_dir_stays_absolute_for_root(monkeypatch):
    seen = []
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._engine = None
    engine._mllm_scheduler = SimpleNamespace(
        save_prefix_cache_to_disk=lambda d, abort: seen.append(d) or True,
        load_prefix_cache_from_disk=lambda d: seen.append(d) or 0,
    )
    assert engine.save_cache_to_disk("/")
    assert engine.load_cache_from_disk("/") == 0
    assert seen == ["/.mllm-exact", "/.mllm-exact"]
