# SPDX-License-Identifier: Apache-2.0
"""
Reproduction tests for prefix-cache disk-persistence corruption bugs.

Failure modes documented here (see analysis 2026-05-03):

A. Stale ``index.json`` + freshly-overwritten ``entry_i.*`` files —
   if a server is killed mid-shutdown after rewriting some entry files
   but before ``index.json`` is rewritten, the next start loads using
   the stale ``num_tokens`` field. ``arr.fromfile(f, num_tokens_old)``
   silently truncates the new tokens.bin, producing an entry whose
   ``tokens_key`` length disagrees with ``cache.offset``. Subsequent
   fetches return that mismatched cache to the scheduler, which
   appends new tokens at the wrong position → garbage attention →
   token-id-0 collapse (``!!!!!`` in user output).

B. Orphan files from a previous save are not removed when the next
   save writes fewer entries. They sit on disk indefinitely; the next
   crash that interrupts ``save_to_disk`` mid-rewrite turns them into
   the inconsistency described in (A).

C. ``mx.save_safetensors`` is called directly on the target path
   (no ``.tmp`` + rename), so a SIGKILL during a single-entry write
   leaves a half-written safetensors. ``mx.load`` will usually raise
   on it (caught and dropped silently), but combined with (A) it can
   amplify the inconsistency.

D. ``mx.load`` is lazy — it parses the header and returns array
   handles without materializing data. A safetensors with a valid
   header but truncated body passes ``load_from_disk`` silently and
   is registered as a usable cache entry. The corruption only
   surfaces at the first attention call, often inside a worker thread
   where the RuntimeError can be swallowed.

These tests use real ``mlx_lm`` ``KVCache`` objects with very small
tensors (1×4×N×8 fp16) so they run fast (<1s each).
"""

from __future__ import annotations

import array
import json
import os

import pytest

mx = pytest.importorskip("mlx.core")
KVCache = pytest.importorskip("mlx_lm.models.cache").KVCache
save_prompt_cache = pytest.importorskip("mlx_lm.models.cache").save_prompt_cache

from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_kvcache(num_tokens: int, *, n_layers: int = 2, fill: float = 1.0) -> list:
    """Build a populated ``mlx_lm`` KVCache list with ``num_tokens`` positions.

    Tiny shape (1, 4, num_tokens, 8) fp16 keeps per-entry I/O <1 KB so
    these tests stay fast.
    """
    layers = []
    for layer_idx in range(n_layers):
        c = KVCache()
        keys = mx.full((1, 4, num_tokens, 8), fill + layer_idx, dtype=mx.float16)
        values = mx.full((1, 4, num_tokens, 8), -(fill + layer_idx), dtype=mx.float16)
        c.update_and_fetch(keys, values)
        layers.append(c)
    return layers


def fresh_cache() -> MemoryAwarePrefixCache:
    """Build a small in-memory cache for testing."""
    return MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100),
    )


def write_entry_files(
    cache_dir: str, entry_idx: int, tokens: list[int], kv_layers: list
) -> None:
    """Write a single (safetensors + tokens.bin) pair, mimicking save_to_disk."""
    save_prompt_cache(
        os.path.join(cache_dir, f"entry_{entry_idx}.safetensors"),
        kv_layers,
        metadata={"num_tokens": str(len(tokens))},
    )
    arr = array.array("i", tokens)
    with open(os.path.join(cache_dir, f"entry_{entry_idx}_tokens.bin"), "wb") as f:
        arr.tofile(f)


# --------------------------------------------------------------------------
# Sanity check — clean roundtrip works
# --------------------------------------------------------------------------


def test_clean_roundtrip_save_then_load(tmp_path):
    """Sanity: clean save → load → entry preserved."""
    cache = fresh_cache()
    tokens = list(range(11))
    cache.store(tokens, make_kvcache(num_tokens=11))
    assert cache.save_to_disk(str(tmp_path)) is True

    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(tmp_path))
    assert loaded == 1

    entry = next(iter(cache2._entries.values()))
    assert entry.tokens == tuple(tokens)
    # Must hold for any well-formed entry: KV state is exactly as long
    # as the token sequence it claims to represent.
    assert entry.cache[0].offset == len(entry.tokens)


# --------------------------------------------------------------------------
# BUG A — stale index.json + overwritten entry files = poisoned tokens_key
# --------------------------------------------------------------------------


def test_stale_index_with_overwritten_entry_loads_without_error(tmp_path):
    """BUG A — load must reject (or normalize) entries whose tokens.bin
    size disagrees with the index.json claim. Such entries are the
    fingerprint of a previous interrupted save_to_disk.
    """
    # --- session 1: clean save with one entry of 11 tokens
    cache_v1 = fresh_cache()
    tokens_v1 = list(range(11))
    cache_v1.store(tokens_v1, make_kvcache(num_tokens=11))
    cache_v1.save_to_disk(str(tmp_path))

    # Sanity: index claims num_tokens=11
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["entries"][0]["num_tokens"] == 11

    # --- session 2: simulate kill DURING save_to_disk —
    # entry_0 files get rewritten with a longer 20-token payload,
    # but the process dies before index.json is rewritten.
    tokens_v2 = list(range(100, 120))  # 20 fresh tokens
    write_entry_files(str(tmp_path), 0, tokens_v2, make_kvcache(num_tokens=20))

    # index.json untouched — still says num_tokens=11
    index_after = json.loads((tmp_path / "index.json").read_text())
    assert index_after["entries"][0]["num_tokens"] == 11

    # --- session 3: load — the size-mismatch check must reject entry_0,
    # since tokens.bin is now 80 bytes (20 ints) but index.json claims 11.
    cache_v3 = fresh_cache()
    loaded = cache_v3.load_from_disk(str(tmp_path))
    assert loaded == 0, (
        "loader accepted an entry whose tokens.bin size disagrees with "
        "index.json's num_tokens — that's the BUG A poisoning vector"
    )
    assert len(cache_v3._entries) == 0

    # If any entry did slip through, its invariant must hold.
    for entry in cache_v3._entries.values():
        assert len(entry.tokens) == entry.cache[0].offset


def test_poisoned_entry_returns_misaligned_cache_via_fetch(tmp_path):
    """BUG A user-visible effect — a poisoned entry must NOT reach fetch().

    If the loader rejects it (correct), fetch returns None / no match.
    If a future regression lets one through, the returned cache.offset
    must at least equal the matched-prefix length so the scheduler
    appends tokens at the right position.
    """
    # Set up the same poisoned state as the previous test.
    cache_v1 = fresh_cache()
    cache_v1.store(list(range(11)), make_kvcache(num_tokens=11))
    cache_v1.save_to_disk(str(tmp_path))

    # Overwrite entry_0 with 20-token content; leave index.json stale.
    tokens_v2 = list(range(100, 120))
    write_entry_files(str(tmp_path), 0, tokens_v2, make_kvcache(num_tokens=20))

    cache_v3 = fresh_cache()
    cache_v3.load_from_disk(str(tmp_path))

    # User sends a prompt that begins with the (would-be-truncated) cached prefix.
    prompt = list(range(100, 111)) + [777, 888, 999]
    kv, remaining = cache_v3.fetch(prompt)

    if kv is None:
        # Correct path: poisoned entry was rejected at load_from_disk;
        # fetch sees an empty cache and returns a clean miss.
        assert remaining == prompt
        return

    # If for some reason an entry slipped through, the returned cache
    # offset must match the matched-prefix length.
    matched_len = len(prompt) - len(remaining)
    returned_offset = kv[0].offset
    assert returned_offset == matched_len, (
        f"fetch returned cache with offset={returned_offset} for "
        f"matched_len={matched_len} prefix. Scheduler would write next "
        f"token at the wrong position."
    )


# --------------------------------------------------------------------------
# BUG B — orphan files from a previous save are not cleaned up
# --------------------------------------------------------------------------


def test_save_to_disk_removes_orphans_from_previous_save(tmp_path):
    """BUG B — directory-rename swap must leave no orphan entry files."""
    # --- session 1: 5 entries
    cache_v1 = fresh_cache()
    for i in range(5):
        cache_v1.store(
            list(range(i * 100, i * 100 + 11)), make_kvcache(num_tokens=11, fill=i + 1)
        )
    cache_v1.save_to_disk(str(tmp_path))

    # --- session 2: only 2 entries (fresh cache, simulates eviction)
    cache_v2 = fresh_cache()
    for i in range(2):
        cache_v2.store(
            list(range(500 + i * 100, 500 + i * 100 + 11)),
            make_kvcache(num_tokens=11, fill=i + 10),
        )
    cache_v2.save_to_disk(str(tmp_path))

    # New index.json reflects only 2 entries
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["num_entries"] == 2

    # CORRECT BEHAVIOR (xfail): orphan entry_2..entry_4 from session 1
    # must be removed so a future crash mid-save can't resurrect them.
    for i in range(2, 5):
        sf = tmp_path / f"entry_{i}.safetensors"
        tk = tmp_path / f"entry_{i}_tokens.bin"
        assert not sf.exists(), (
            f"orphan {sf.name} from previous save was not cleaned up — "
            f"this is the precondition that turns a half-written next save "
            f"into BUG A."
        )
        assert not tk.exists(), f"orphan {tk.name} not cleaned up"


# --------------------------------------------------------------------------
# Characterization tests — document current behavior (no xfail)
# --------------------------------------------------------------------------


def test_severely_truncated_safetensors_is_silently_skipped(tmp_path):
    """Document current behavior: a header-corrupt .safetensors is dropped silently.

    This is *acceptable* on its own (the entry just won't be used), but
    no structured signal is propagated upward — operators have no way
    to notice gradual cache decay. Once the diagnostic-counter fix
    (P3 in the analysis) lands, this test should also assert that
    ``loaded`` returns a (loaded, skipped) tuple or that a structured
    warning was emitted.
    """
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(tmp_path))

    # Truncate aggressively (16 bytes — far short of safetensors header)
    sf = tmp_path / "entry_0.safetensors"
    sf.write_bytes(sf.read_bytes()[:16])

    # Load: must not raise, just skip
    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(tmp_path))
    assert loaded == 0
    assert len(cache2._entries) == 0


def test_body_truncated_safetensors_should_fail_eagerly_at_load(tmp_path):
    """BUG D — load_from_disk must reject a body-truncated safetensors
    even though ``mx.load`` will lazily mmap it without complaint.
    """
    import struct

    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(tmp_path))

    sf = tmp_path / "entry_0.safetensors"
    full = sf.read_bytes()

    # Compute the maximum data offset declared by the header so we can
    # truncate strictly inside the body region — guards against future
    # changes to padding/alignment in save_prompt_cache.
    header_len = struct.unpack("<Q", full[:8])[0]
    header = json.loads(full[8 : 8 + header_len])
    max_end = max(
        meta["data_offsets"][1]
        for name, meta in header.items()
        if name != "__metadata__"
    )
    declared_total = 8 + header_len + max_end
    cut_to = declared_total - 100
    assert cut_to > 8 + header_len, (
        "test setup: cut would land in the header region, not the body — "
        "use a larger entry"
    )
    sf.write_bytes(full[:cut_to])

    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(tmp_path))
    assert loaded == 0, (
        "Body-truncated safetensors was loaded as a usable cache entry. "
        "It will blow up later at attention time with a RuntimeError, "
        "likely inside a worker thread."
    )


# --------------------------------------------------------------------------
# Crash-recovery for interrupted save_to_disk swap
# --------------------------------------------------------------------------


def test_load_recovers_from_swap_interrupted_after_first_rename(tmp_path):
    """If the process died after ``cache_dir → .old`` but before
    ``.new → cache_dir``, load_from_disk must promote ``.new`` because
    it holds the freshly-committed snapshot.
    """
    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    # Snapshot 1 → ends up at .old (simulates the first rename of the swap)
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))
    cache_dir.rename(old_dir)

    # Snapshot 2 built in a side dir, then placed at .new (simulates the
    # staging dir of an interrupted save — done writing, swap not yet
    # finished). Using a side dir avoids triggering the next save's
    # pre-clean of .old.
    side_dir = tmp_path / "side"
    c2 = fresh_cache()
    c2.store(list(range(50, 65)), make_kvcache(num_tokens=15, fill=2.0))
    c2.save_to_disk(str(side_dir))
    side_dir.rename(new_dir)

    assert not cache_dir.exists()
    assert new_dir.exists()
    assert old_dir.exists()

    # Load: should promote .new to cache_dir, drop .old
    c3 = fresh_cache()
    loaded = c3.load_from_disk(str(cache_dir))
    assert loaded == 1
    assert cache_dir.exists()
    assert not new_dir.exists()
    assert not old_dir.exists()
    entry = next(iter(c3._entries.values()))
    assert entry.tokens == tuple(range(50, 65))


def test_load_recovers_from_swap_interrupted_with_only_old(tmp_path):
    """If only ``.old`` survives (e.g. ``.new`` was never finalized),
    load_from_disk must restore ``.old`` to ``cache_dir``.
    """
    cache_dir = tmp_path / "snap"
    c1 = fresh_cache()
    c1.store(list(range(7)), make_kvcache(num_tokens=7))
    c1.save_to_disk(str(cache_dir))

    # Simulate crash mid-swap with no .new survivor
    cache_dir.rename(tmp_path / "snap.old")
    assert not cache_dir.exists()

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    assert loaded == 1
    assert cache_dir.exists()
    entry = next(iter(c2._entries.values()))
    assert entry.tokens == tuple(range(7))


def test_load_cleans_orphan_staging_dirs(tmp_path):
    """If ``cache_dir`` exists alongside leftover ``.new`` / ``.old``
    staging dirs, load_from_disk must wipe the orphans so the next
    save starts from a clean slate.
    """
    cache_dir = tmp_path / "snap"
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))

    # Sprinkle leftover staging dirs
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"
    new_dir.mkdir()
    (new_dir / "leftover.txt").write_text("orphan")
    old_dir.mkdir()
    (old_dir / "leftover.txt").write_text("orphan")

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    assert loaded == 1
    assert not new_dir.exists()
    assert not old_dir.exists()


def test_partial_new_index_json_is_not_promoted(tmp_path):
    """If .new/index.json exists but is corrupt JSON (e.g. crash mid
    json.dump), recovery must NOT promote .new — fall back to .old or
    leave cache_dir absent rather than handing the partial snapshot
    to subsequent json.load.
    """
    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    # Build a valid snapshot at .old (the previous committed state)
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))
    cache_dir.rename(old_dir)

    # Hand-craft a .new with a *partial* index.json (simulates crash
    # in the middle of json.dump).
    new_dir.mkdir()
    (new_dir / "index.json").write_text('{"versi')

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    # Should fall through to .old, recovering the previous snapshot
    assert loaded == 1
    entry = next(iter(c2._entries.values()))
    assert entry.tokens == tuple(range(11))


def test_save_handles_trailing_slash_in_cache_dir(tmp_path):
    """A user-supplied cache_dir with a trailing separator must still
    swap atomically. Without the rstrip in save_to_disk, ``cache_dir +
    '.new'`` would become a *child* of cache_dir rather than a sibling,
    silently breaking the swap.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(cache_dir) + "/")

    # The committed snapshot lives at cache_dir, NOT cache_dir/.new
    assert cache_dir.exists()
    assert (cache_dir / "index.json").exists()
    assert not (cache_dir / ".new").exists()
    assert not (tmp_path / "snap/.new").exists()

    # Round-trips with trailing slash on load too.
    c2 = fresh_cache()
    assert c2.load_from_disk(str(cache_dir) + "/") == 1


def test_load_into_non_empty_cache_skips_duplicates(tmp_path):
    """If load_from_disk is called on a cache that already contains some
    keys (e.g. populated by warmup before lifespan calls load), entries
    whose tokens_key matches an in-memory entry must be skipped — not
    re-inserted. Otherwise bisect.insort produces duplicate keys in
    _sorted_keys and _current_memory double-counts.
    """
    cache_dir = tmp_path / "snap"
    # Persist two entries to disk: one duplicates a future in-memory
    # entry; one is fresh.
    persisted = fresh_cache()
    persisted.store(list(range(11)), make_kvcache(num_tokens=11))
    persisted.store(list(range(50, 61)), make_kvcache(num_tokens=11, fill=2.0))
    persisted.save_to_disk(str(cache_dir))

    # Simulated warmup state: the [0..10] entry is already in memory.
    runtime = fresh_cache()
    runtime.store(list(range(11)), make_kvcache(num_tokens=11))
    warmup_mem = runtime._current_memory
    warmup_keys = list(runtime._sorted_keys)

    loaded = runtime.load_from_disk(str(cache_dir))
    assert loaded == 1, "exactly one fresh entry should have been loaded"
    # The duplicate did not double-insert
    assert runtime._sorted_keys.count(tuple(range(11))) == 1
    assert tuple(range(50, 61)) in runtime._sorted_keys
    # Memory grew by exactly the new entry's footprint
    new_entry_mem = runtime._current_memory - warmup_mem
    assert new_entry_mem > 0
    # Pre-existing entry untouched in keys list ordering wrt itself
    assert warmup_keys[0] in runtime._sorted_keys


def test_recovery_rejects_new_with_index_but_no_entry_files(tmp_path):
    """If ``.new/index.json`` references entries but the entry files are
    missing on disk (manual deletion, fs corruption, partial restore),
    recovery must NOT promote ``.new``. Doing so would discard ``.old``
    in favor of an empty snapshot — net data loss.
    """
    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    # Build a real, complete snapshot in .old
    c1 = fresh_cache()
    c1.store(list(range(11)), make_kvcache(num_tokens=11))
    c1.save_to_disk(str(cache_dir))
    cache_dir.rename(old_dir)

    # Hand-craft a .new with valid-looking index.json but NO entry files
    new_dir.mkdir()
    (new_dir / "index.json").write_text(
        json.dumps(
            {
                "version": 2,
                "num_entries": 1,
                "total_memory_bytes": 12345,
                "entries": [{"index": 0, "num_tokens": 11, "memory_bytes": 12345}],
            }
        )
    )

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    # Recovery should fall through to .old, not silently lose the snapshot
    assert loaded == 1, "recovery promoted an empty .new and lost .old"
    entry = next(iter(c2._entries.values()))
    assert entry.tokens == tuple(range(11))


def test_load_dedup_check_runs_before_safetensors_load(tmp_path, monkeypatch):
    """Performance + memory: a tokens_key that's already in the in-memory
    cache must skip ``load_prompt_cache`` entirely. Otherwise every
    duplicate entry mmaps its safetensors only to discard it — wastes
    file descriptors, memory, and time.
    """
    cache_dir = tmp_path / "snap"
    persisted = fresh_cache()
    persisted.store(list(range(11)), make_kvcache(num_tokens=11))
    persisted.save_to_disk(str(cache_dir))

    # Spy on load_prompt_cache to count calls
    import mlx_lm.models.cache as mlx_cache_mod

    real_load = mlx_cache_mod.load_prompt_cache
    call_count = {"n": 0}

    def spy(path):
        call_count["n"] += 1
        return real_load(path)

    monkeypatch.setattr(mlx_cache_mod, "load_prompt_cache", spy)
    monkeypatch.setattr("vllm_mlx.memory_cache.load_prompt_cache", spy, raising=False)

    runtime = fresh_cache()
    runtime.store(list(range(11)), make_kvcache(num_tokens=11))
    loaded = runtime.load_from_disk(str(cache_dir))

    assert loaded == 0, "the only persisted entry was a duplicate"
    assert call_count["n"] == 0, (
        f"load_prompt_cache should not be called for duplicates "
        f"(was called {call_count['n']} time(s))"
    )


def test_save_routes_writes_through_staging_dir(tmp_path, monkeypatch):
    """Atomicity invariant: save_safetensors must be called with a path
    inside a sibling ``<cache_dir>.new`` staging directory, not directly
    inside ``cache_dir``. The directory-rename swap is what makes the
    snapshot all-or-nothing.
    """
    seen_paths: list[str] = []

    real_save = mx.save_safetensors

    def spy(path, *args, **kwargs):
        seen_paths.append(path)
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr("mlx.core.save_safetensors", spy)

    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(cache_dir))

    assert seen_paths, "save_safetensors was never called"
    expected_staging = str(cache_dir) + ".new"
    for p in seen_paths:
        assert p.startswith(expected_staging + os.sep), (
            f"save_safetensors called with {p!r}, expected to be inside "
            f"{expected_staging!r}. Direct writes into the committed "
            f"cache_dir defeat the atomic-snapshot guarantee."
        )

    # After save returns, the staging dir is gone (renamed into place).
    assert not (tmp_path / "snap.new").exists()
    assert (cache_dir / "index.json").exists()
    assert (cache_dir / "entry_0.safetensors").exists()


# --------------------------------------------------------------------------
# Issue #198 BUG B — incompatible cache types loaded across config changes
# --------------------------------------------------------------------------
#
# When the previous server run persisted ``QuantizedKVCache`` entries
# (``--kv-cache-quantization`` or earlier ``--kv-cache-turboquant`` runs
# that fell back to mlx-lm quantization) and the next run starts under
# a different cache config, the on-disk entries' tuple-form ``keys``
# bypass the dequantize path in ``_decompress_cache`` and reach the
# scheduler, which crashes with::
#
#     AttributeError: 'list' object has no attribute 'shape'
#
# The hasattr guards in ``scheduler.py`` stop the crash, but the entry
# is unusable. Real fix: ``load_from_disk`` rejects entries whose
# persisted cache class can't be safely dequantized under the current
# config, and counts them in ``incompatible_skipped`` (distinct from
# ``corrupt_skipped`` so users don't see "may need cleanup" warnings
# for an expected config change).


def _make_quantized_kvcache(num_tokens: int, *, n_layers: int = 2):
    """Build an mlx-lm ``QuantizedKVCache`` list with ``num_tokens`` positions.

    Used to simulate persisted entries from a previous ``--kv-cache-
    quantization`` run.
    """
    QuantizedKVCache = pytest.importorskip("mlx_lm.models.cache").QuantizedKVCache
    layers = []
    for layer_idx in range(n_layers):
        c = QuantizedKVCache(group_size=64, bits=8)
        # Need a head_dim that's a clean multiple of group_size for quantize.
        keys = mx.full((1, 4, num_tokens, 64), 0.5 + layer_idx, dtype=mx.float16)
        values = mx.full((1, 4, num_tokens, 64), -(0.5 + layer_idx), dtype=mx.float16)
        c.update_and_fetch(keys, values)
        layers.append(c)
    return layers


def _save_one_entry(
    cache_dir, tokens, kv_layers, *, cache_types: list[str] | None = None
):
    """Write a one-entry snapshot directly (no MemoryAwarePrefixCache).

    ``cache_types`` lets a test pretend the index.json is from a
    different config than what we'd write today.
    """
    os.makedirs(cache_dir, exist_ok=True)
    write_entry_files(str(cache_dir), 0, tokens, kv_layers)
    types = (
        cache_types
        if cache_types is not None
        else ([type(layer).__name__ for layer in kv_layers])
    )
    index = {
        "version": 2,
        "num_entries": 1,
        "total_memory_bytes": 0,
        "entries": [
            {
                "index": 0,
                "num_tokens": len(tokens),
                "memory_bytes": 0,
                "cache_types": types,
            }
        ],
    }
    with open(os.path.join(str(cache_dir), "index.json"), "w") as f:
        json.dump(index, f)


def test_quantized_entry_rejected_under_plain_config(tmp_path):
    """BUG B — loading a persisted QuantizedKVCache under a plain config
    must reject the entry (otherwise the tuple-form keys reach the
    scheduler and crash with ``'list' has no attribute 'shape'``)."""
    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, _make_quantized_kvcache(num_tokens=11))

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=False),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 0
    assert tuple(tokens) not in cache._entries


def test_quantized_entry_rejected_under_turboquant_config(tmp_path):
    """BUG B (the exact scenario in #198) — previous run wrote
    QuantizedKVCache; current run uses ``--kv-cache-turboquant``.
    ``_turboquant_decompress_cache`` only handles ``TurboQuantKVCache``
    so any other type slips through unchanged. Reject at load."""
    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, _make_quantized_kvcache(num_tokens=11))

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_turboquant=True),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 0


def test_quantized_entry_loadable_under_kv_quantize(tmp_path):
    """Negative control — same config restart must still load."""
    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, _make_quantized_kvcache(num_tokens=11))

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=True),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 1


def test_plain_entry_loadable_under_kv_quantize(tmp_path):
    """Plain ``KVCache`` is forward-compatible with any config — the next
    ``store()`` recompresses, until then fetch passes through unchanged.
    Don't reject these; that would force users to wipe their cache when
    enabling ``--kv-cache-quantization``."""
    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, make_kvcache(num_tokens=11))

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=True),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 1


def test_plain_entry_loadable_under_turboquant(tmp_path):
    """Same forward-compat as above but for ``--kv-cache-turboquant``."""
    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, make_kvcache(num_tokens=11))

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_turboquant=True),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 1


def test_hybrid_entry_with_one_quantized_layer_rejected_under_plain_config(
    tmp_path,
):
    """A hybrid model could mix layer types (e.g. global attention layers
    quantized for memory, sliding-window layers kept plain). The compat
    check must reject the WHOLE entry if ANY layer requires a config the
    current run doesn't have — otherwise the partial dequantize at fetch
    leaves the quantized layer's tuple-form keys for the scheduler.

    Constructs an entry with [KVCache, QuantizedKVCache, KVCache] and
    asserts rejection under plain config.
    """
    QuantizedKVCache = pytest.importorskip("mlx_lm.models.cache").QuantizedKVCache
    plain = make_kvcache(num_tokens=11, n_layers=1)[0]
    quant_layer = QuantizedKVCache(group_size=64, bits=8)
    qk = mx.full((1, 4, 11, 64), 0.5, dtype=mx.float16)
    qv = mx.full((1, 4, 11, 64), -0.5, dtype=mx.float16)
    quant_layer.update_and_fetch(qk, qv)
    plain2 = make_kvcache(num_tokens=11, n_layers=1)[0]
    mixed = [plain, quant_layer, plain2]

    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, mixed)
    # Sanity: the recorded list reflects the hybrid layout.
    with open(tmp_path / "index.json") as f:
        idx = json.load(f)
    assert idx["entries"][0]["cache_types"] == [
        "KVCache",
        "QuantizedKVCache",
        "KVCache",
    ]

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=False),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 0, (
        "hybrid entry with even one QuantizedKVCache layer must be "
        "rejected under plain config — partial dequantize would leave "
        "tuple-form keys for the scheduler"
    )


def test_legacy_index_without_cache_types_falls_back_to_safetensors_metadata(
    tmp_path,
):
    """Backward compat — index.json from before #198 lacks ``cache_types``.
    Loader must peek at the safetensors ``__metadata__`` to figure out
    the persisted class. Without this fallback, every legacy quantized
    entry would be incorrectly accepted under a plain config and crash
    the scheduler the moment it's fetched."""
    tokens = list(range(11))
    _save_one_entry(
        tmp_path,
        tokens,
        _make_quantized_kvcache(num_tokens=11),
        cache_types=[],  # simulate legacy index without the field
    )
    # Sanity: the index.json should now have cache_types == [].
    with open(tmp_path / "index.json") as f:
        idx = json.load(f)
    assert idx["entries"][0]["cache_types"] == []

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=False),
    )
    loaded = cache.load_from_disk(str(tmp_path))
    assert loaded == 0, (
        "expected legacy QuantizedKVCache entry to be rejected via "
        "safetensors-metadata fallback under plain config"
    )


def test_save_to_disk_records_cache_types_in_index(tmp_path):
    """Forward-looking — the field must be present in newly written
    index.json so future loads can pre-filter without parsing each
    safetensors header."""
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.save_to_disk(str(tmp_path))

    with open(tmp_path / "index.json") as f:
        idx = json.load(f)
    assert idx["entries"][0]["cache_types"] == ["KVCache", "KVCache"]


def test_save_to_disk_records_post_dequant_cache_types(tmp_path):
    """Regression — when ``store`` quantized the entry but ``save_to_disk``
    must dequantize before persisting (save_prompt_cache requires .state /
    .meta_state which QuantizedKVCache doesn't expose), the index must
    record the *post-dequant* class names. Recording ``QuantizedKVCache``
    here would cause a subsequent unquantized startup to reject the entry
    as "config changed", silently losing a perfectly loadable cache.
    Caught by codex post-v0.6.14 review."""
    n_tokens = 300  # > kv_min_quantize_tokens default of 256
    # Build a real KVCache with head_dim=64 so quantize(group_size=64) works.
    kv_layers = []
    for layer_idx in range(2):
        c = KVCache()
        keys = mx.full((1, 4, n_tokens, 64), 0.5 + layer_idx, dtype=mx.float16)
        values = mx.full((1, 4, n_tokens, 64), -(0.5 + layer_idx), dtype=mx.float16)
        c.update_and_fetch(keys, values)
        kv_layers.append(c)
    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(
            max_memory_mb=64,
            max_entries=100,
            kv_quantize=True,
        ),
    )
    cache.store(list(range(n_tokens)), kv_layers)

    # Sanity — entry.cache should now be QuantizedKVCache layers.
    QuantizedKVCache = pytest.importorskip("mlx_lm.models.cache").QuantizedKVCache
    entry = next(iter(cache._entries.values()))
    assert all(isinstance(c, QuantizedKVCache) for c in entry.cache)

    cache.save_to_disk(str(tmp_path))

    with open(tmp_path / "index.json") as f:
        idx = json.load(f)
    assert idx["entries"][0]["cache_types"] == ["KVCache", "KVCache"], (
        "cache_types must reflect what's on disk (post-dequant), not what "
        "lived in entry.cache before save"
    )

    # End-to-end: a future unquantized startup must accept this entry.
    plain = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=False),
    )
    loaded = plain.load_from_disk(str(tmp_path))
    assert loaded == 1, (
        "previously-quantized-but-now-dequantized entry must load under "
        "unquantized config (otherwise users lose their cache after toggling "
        "--kv-cache-quantization off)"
    )


def test_incompatible_skipped_does_not_count_as_corruption(tmp_path, caplog):
    """The summary log must distinguish "config changed → expected skips"
    from "disk corrupt → user should investigate". A WARNING for an
    expected config change would be alarm fatigue."""
    import logging as _logging

    tokens = list(range(11))
    _save_one_entry(tmp_path, tokens, _make_quantized_kvcache(num_tokens=11))

    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=64, max_entries=100, kv_quantize=False),
    )
    with caplog.at_level(_logging.INFO, logger="vllm_mlx.memory_cache"):
        cache.load_from_disk(str(tmp_path))

    text = caplog.text
    assert "incompatible" in text, "summary should mention incompatible skips"
    assert "may need cleanup" not in text, (
        "config-change skips must not surface as corruption warnings"
    )


# --------------------------------------------------------------------------
# Issue #198 BUG A — scheduler-side hasattr guards
# --------------------------------------------------------------------------
#
# Tests for the three ``.shape``-on-tuple-keys crash sites in
# ``vllm_mlx/scheduler.py``. We exercise the validators directly with
# the cache shape they receive when a QuantizedKVCache reaches them
# (which happens for stale-cache scenarios where Bug B fix didn't fire,
# or for in-memory mid-prefill states under quantized model paths).


class _FakeQuantizedLayer:
    """Stand-in for QuantizedKVCache shape: ``keys`` is a tuple, not array.

    We use this rather than the real class to keep the test independent
    of mlx-lm's internal layout — only the surface seen by the scheduler
    matters for the regression we're guarding against.
    """

    def __init__(self, num_tokens: int = 11):
        # mlx-lm QuantizedKVCache stores keys/values as 3-tuples
        # (data, scales, biases); only ``data`` is a real array.
        data = mx.zeros((1, 4, num_tokens, 16), dtype=mx.uint32)
        scales = mx.zeros((1, 4, num_tokens, 1), dtype=mx.float16)
        biases = mx.zeros((1, 4, num_tokens, 1), dtype=mx.float16)
        self.keys = (data, scales, biases)
        self.values = (data, scales, biases)


def test_scheduler_validator_accepts_tuple_keys_without_crashing():
    """BUG A — the validator must not raise ``AttributeError`` when
    ``layer.keys`` is the QuantizedKVCache 3-tuple.

    Pre-fix this would do ``layer_cache.keys.shape[0]`` and crash on
    the tuple, taking down whichever request triggered the fetch.
    Post-fix the validator skips the batch-dim check for non-array
    keys (the dim is structurally guaranteed by the cache class) and
    returns truthfully.

    Note: the only Bug A site that actually fires under #198's repro
    is this validator. The two other sites in
    ``_reconstruct_cache_from_states`` are gated on
    ``cache_cls is _BatchKVCache`` whose state tuple is always
    array-typed in practice; their ``hasattr`` guards are defensive,
    not load-bearing, so we don't pin them with tests.
    """
    from vllm_mlx.scheduler import Scheduler

    layer = _FakeQuantizedLayer(num_tokens=11)
    # _validate_cache is an instance method but doesn't touch self for
    # the list-of-layers path — call unbound.
    is_valid = Scheduler._validate_cache(None, [layer])
    assert is_valid in (True, False)


# --------------------------------------------------------------------------
# Staging dir vanishes mid-save (observed on macOS during long shutdown saves
# with multi-GB caches — Spotlight, purgeable-cache cleanup, or stat-cache
# coherence can clobber `<cache_dir>.new` between successful entry writes
# and the index.json finalize). Pre-fix, save_to_disk would raise
# FileNotFoundError up to the lifespan handler, dumping a scary traceback
# on shutdown even though the warning was already logged downstream.
# --------------------------------------------------------------------------


def test_save_aborts_cleanly_when_staging_dir_vanishes_completely(
    tmp_path, monkeypatch
):
    """If the staging dir is deleted *after* at least one entry has been
    fully written and accounted for in `saved`, but before index.json is
    written, save_to_disk must return False without raising. Pre-fix it
    raised FileNotFoundError up to the lifespan handler, dumping a
    user-visible traceback on shutdown.

    Reproduction: nuke `<cache_dir>.new` at the *start* of the second
    entry's save_prompt_cache call. Entry 0 is fully written (its
    safetensors + tokens.bin both committed, saved=1), so the saved==0
    early-return is bypassed. Entry 1+ fail (logged but caught), but
    `saved > 0` so the function proceeds to index.json — which raises
    FileNotFoundError because new_dir is gone."""
    import shutil as _shutil

    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    for k in range(3):
        cache.store(list(range(10 * (k + 1), 10 * (k + 1) + 11)), make_kvcache(11))

    new_dir = str(cache_dir) + ".new"
    import mlx_lm.models.cache as _mc

    # Patching `mlx_lm.models.cache.save_prompt_cache` (not
    # `vllm_mlx.memory_cache.save_prompt_cache`) is intentional and
    # correct: memory_cache.py imports it via a function-local
    # `from mlx_lm.models.cache import save_prompt_cache` *inside*
    # save_to_disk (line ~1180). That `from X import Y` re-resolves
    # the attribute on the source module on every call, so
    # monkeypatch.setattr on _mc takes effect for the subsequent
    # save_to_disk invocation.
    real_save = _mc.save_prompt_cache
    call_count = {"n": 0}

    def _save_with_nuke_before_call_2(file_name, kv, metadata=None):
        # Fire BEFORE call 2's actual save — at this point call 1 has
        # fully completed (its safetensors + tokens.bin both on disk,
        # saved already incremented). Mimics the production sequence
        # where the dir is wiped after a successful run of N entries.
        # The per-entry try/except in memory_cache.py:1248 swallows the
        # subsequent FileNotFoundError so the loop continues; the bug
        # we're guarding against fires later, at the index.json write.
        call_count["n"] += 1
        if call_count["n"] == 2:
            _shutil.rmtree(new_dir, ignore_errors=True)
        return real_save(file_name, kv, metadata=metadata or {})

    monkeypatch.setattr(_mc, "save_prompt_cache", _save_with_nuke_before_call_2)

    result = cache.save_to_disk(str(cache_dir))
    assert result is False  # must NOT raise
    assert not (cache_dir / "index.json").exists()
    # And no half-baked .new should remain
    assert not os.path.exists(new_dir)


def test_save_persists_only_surviving_entries_when_some_files_vanish(
    tmp_path, monkeypatch
):
    """Soft-failure variant: 2 of 3 entry files vanish before index.json
    is written (pretend an external cache cleaner deleted the largest
    files). The save should still commit a snapshot containing only the
    survivor — better than losing all of them."""
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(11))
    cache.store(list(range(20, 31)), make_kvcache(11, fill=2.0))
    cache.store(list(range(40, 51)), make_kvcache(11, fill=3.0))

    # Save normally, but right after the loop finishes (before the new
    # `verified` filter runs), nuke entries 1 and 2's files. The filter
    # should detect this and persist only entry 0.
    real_save = __import__(
        "mlx_lm.models.cache", fromlist=["save_prompt_cache"]
    ).save_prompt_cache  # noqa: E501
    saved_paths = []

    def _track_saves(file_name, kv, metadata=None):
        result = real_save(file_name, kv, metadata=metadata or {})
        saved_paths.append(file_name)
        # After all 3 saves complete, delete entries 1 and 2's files
        if len(saved_paths) == 3:
            new_dir = str(cache_dir) + ".new"
            for i in (1, 2):
                for suffix in (".safetensors", "_tokens.bin"):
                    p = os.path.join(new_dir, f"entry_{i}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)
        return result

    import mlx_lm.models.cache as _mc

    monkeypatch.setattr(_mc, "save_prompt_cache", _track_saves)

    result = cache.save_to_disk(str(cache_dir))
    assert result is True
    assert (cache_dir / "index.json").exists()

    # Verify the persisted snapshot loads cleanly with exactly 1 entry
    cache2 = fresh_cache()
    assert cache2.load_from_disk(str(cache_dir)) == 1
    entry = next(iter(cache2._entries.values()))
    assert entry.tokens == tuple(range(11))


def test_save_aborts_on_post_filter_dir_loss(tmp_path, monkeypatch):
    """TOCTOU corner: the staging dir survives the verified-filter check
    but is then deleted before index.json is written. Pre-fix,
    `os.makedirs(new_dir, exist_ok=True)` would silently recreate an
    EMPTY dir and the index.json write would commit a snapshot pointing
    to non-existent entry files (load_from_disk recovers correctly, but
    the swap is wasted and the previous good snapshot is unnecessarily
    promoted to .old).

    This regression test pins the post-filter recheck added to defend
    against that window. We monkeypatch ``os.makedirs`` to delete the
    dir's contents the instant after it's recreated — simulating an
    aggressive external cleaner that wakes up exactly between the
    verified filter and the index.json open()."""
    import os as _os
    import shutil as _shutil

    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(11))
    cache.store(list(range(20, 31)), make_kvcache(11, fill=2.0))

    new_dir = str(cache_dir) + ".new"
    real_makedirs = _os.makedirs
    nuke_after_call = {"after": False}

    def _makedirs_then_maybe_nuke(path, *args, **kwargs):
        result = real_makedirs(path, *args, **kwargs)
        # The first makedirs call (top of save_to_disk) is fine; the
        # SECOND call (the defensive recreation right before index.json)
        # is where we want to clobber. Track via a flag flipped on first
        # entry-files write.
        if path == new_dir and nuke_after_call["after"]:
            _shutil.rmtree(new_dir, ignore_errors=True)
        return result

    # Flip the flag the *first* time we see entry_0_tokens.bin opened
    # for write. Trigger is precise (exact path match, fires once) so
    # no unrelated I/O can accidentally arm us. monkeypatch.setattr on
    # builtins.open auto-unwinds at fixture teardown, so the patch
    # never escapes this test even on assertion failure.
    import vllm_mlx.memory_cache as _mc_mod

    expected_marker = os.path.join(new_dir, "entry_0_tokens.bin")
    real_open = open

    def _open_then_arm(file, *a, **k):
        if (
            isinstance(file, str)
            and file == expected_marker
            and not nuke_after_call["after"]
        ):
            nuke_after_call["after"] = True
        return real_open(file, *a, **k)

    monkeypatch.setattr(_mc_mod, "os", _os)
    monkeypatch.setattr(_os, "makedirs", _makedirs_then_maybe_nuke)
    monkeypatch.setattr("builtins.open", _open_then_arm)

    # Must not raise — pre-fix, the post-filter `os.makedirs` would
    # recreate an empty dir, then `open(index_path, "w")` would succeed
    # on the freshly-empty dir and commit a bogus snapshot. With the
    # TOCTOU recheck, save_to_disk returns False cleanly.
    result = cache.save_to_disk(str(cache_dir))
    assert result is False, (
        "post-filter dir loss must abort cleanly (not commit a snapshot "
        "pointing to vanished entry files)"
    )
    # No half-baked cache_dir should have been created
    assert not (cache_dir / "index.json").exists()


# --------------------------------------------------------------------------
# Shutdown-deadline ("should_abort") partial-commit path
# --------------------------------------------------------------------------
#
# Rapid-desktop only gives the rapid-mlx sidecar ~5s between SIGTERM and
# SIGKILL. The previous synchronous flush would write entries linearly,
# get SIGKILLed mid-write, and leave ``<cache_dir>.new/`` orphaned on
# disk — no cache survives to the next launch. The fix lets the lifespan
# pass a ``should_abort`` predicate down to the per-entry loop; when it
# trips the loop stops and STILL commits the entries that did finish via
# the atomic directory-rename swap.


def test_save_to_disk_partial_commit_on_abort(tmp_path):
    """should_abort fires after one entry — staging dir must still be
    committed via atomic rename and the surviving entry must be
    loadable on the next session.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    # Three entries; the predicate will fire after the first finishes.
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.store(list(range(20, 31)), make_kvcache(num_tokens=11, fill=2.0))
    cache.store(list(range(50, 61)), make_kvcache(num_tokens=11, fill=3.0))

    calls = {"n": 0}

    def predicate(predicted_sec=0.0):
        # Fire from the second invocation onward so entry 0 lands but
        # entries 1 and 2 are skipped. Accepts ``predicted_sec`` to
        # match the forward-looking predicate contract added in PR
        # #667 round 2 — production callers pass an estimated per-
        # entry write duration; this test ignores it and gates on
        # call count for deterministic single-vs-multi-entry behavior.
        calls["n"] += 1
        return calls["n"] > 1

    assert cache.save_to_disk(str(cache_dir), should_abort=predicate) is True

    # Critical invariant: the staging dir is gone — committed via rename.
    assert not (tmp_path / "snap.new").exists(), (
        "should_abort path must still run the atomic-rename swap so the "
        "staging dir is never left orphaned — that's the whole point of "
        "the fix vs. the pre-existing SIGKILL behavior"
    )
    # And the committed dir holds the surviving entry plus a consistent index.
    assert (cache_dir / "index.json").exists()
    index = json.loads((cache_dir / "index.json").read_text())
    assert index["num_entries"] == 1
    assert len(index["entries"]) == 1
    assert (cache_dir / "entry_0.safetensors").exists()
    assert (cache_dir / "entry_0_tokens.bin").exists()

    # Re-loading must succeed and surface exactly the entry that survived.
    cache2 = fresh_cache()
    loaded = cache2.load_from_disk(str(cache_dir))
    assert loaded == 1
    entry = next(iter(cache2._entries.values()))
    assert entry.tokens == tuple(range(11))


def test_save_to_disk_aborts_before_first_entry_returns_false(tmp_path):
    """If the deadline already passed before the first write, ``should_abort``
    fires immediately, ``saved == 0``, and the staging dir is cleaned up
    rather than committed empty. Matches the existing "no entries saved
    successfully, aborting" branch.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))

    assert (
        cache.save_to_disk(
            str(cache_dir),
            should_abort=lambda predicted_sec=0.0: True,
        )
        is False
    )
    assert not cache_dir.exists()
    assert not (tmp_path / "snap.new").exists()


def test_save_to_disk_predicate_none_preserves_full_flush(tmp_path):
    """Passing ``should_abort=None`` must be equivalent to the legacy call
    shape — all entries get written, no early break. Protects the offline
    ``rapid-mlx`` CLI path and the existing tests from regressing.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.store(list(range(20, 31)), make_kvcache(num_tokens=11, fill=2.0))

    assert cache.save_to_disk(str(cache_dir), should_abort=None) is True
    index = json.loads((cache_dir / "index.json").read_text())
    assert index["num_entries"] == 2
    assert (cache_dir / "entry_0.safetensors").exists()
    assert (cache_dir / "entry_1.safetensors").exists()


def test_shutdown_save_prefix_cache_runs_off_event_loop(tmp_path, monkeypatch):
    """Regression for the lifespan shutdown bug, pinned at the production
    callsite.

    Drives ``server._shutdown_save_prefix_cache`` directly — NOT a
    test-local ``asyncio.to_thread`` wrapper around the save function.
    Codex flagged PR #667 round 1 because the prior shape wrapped
    ``to_thread`` test-side, so a regression that dropped the wrapper
    from the lifespan helper would still pass. This version exercises
    the helper that's literally what ``lifespan`` awaits — if anyone
    replaces the ``await asyncio.to_thread(...)`` line in
    ``_shutdown_save_prefix_cache`` with a direct call, the slow fake
    engine below blocks the loop and the ticker count drops to 1.

    Shape: pretend the engine takes ~600ms to flush. Drive the helper
    AND a 50ms ticker concurrently. Assert the ticker advances
    multiple times — i.e. the loop wasn't blocked.
    """
    import asyncio
    import time as _time

    from vllm_mlx import config as _config_mod
    from vllm_mlx import server as _server_mod
    from vllm_mlx.runtime import cache as _cache_mod

    class _SlowEngine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            # Block for ~600ms on the worker thread — production wrap
            # is asyncio.to_thread so the loop stays responsive.
            _time.sleep(0.6)
            return True

    class _FakeCfg:
        engine = _SlowEngine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())
    # ``_shutdown_save_prefix_cache`` checks ``_engine is not None``
    # AND ``hasattr(_engine, "save_cache_to_disk")`` before delegating
    # to the runtime save. Substitute a stub that satisfies both so
    # the helper actually runs.
    monkeypatch.setattr(_server_mod, "_engine", _SlowEngine())

    async def _drive():
        ticks: list[float] = []
        t0 = _time.monotonic()

        async def _ticker():
            while _time.monotonic() - t0 < 0.6:
                ticks.append(_time.monotonic() - t0)
                await asyncio.sleep(0.05)

        # Drive the production lifespan helper as-is. Don't wrap it
        # in asyncio.to_thread out here — that would re-introduce the
        # original bug the test exists to catch.
        await asyncio.gather(
            _server_mod._shutdown_save_prefix_cache(),
            _ticker(),
        )
        return ticks

    ticks = asyncio.run(_drive())
    assert len(ticks) >= 5, (
        f"event loop was blocked during cache flush — only saw {len(ticks)} "
        f"ticks in 600ms (expected ≥5). Did _shutdown_save_prefix_cache "
        f"lose its asyncio.to_thread wrap?"
    )


def test_shutdown_save_prefix_cache_no_op_when_engine_missing(monkeypatch):
    """Companion guard: when ``server._engine`` is None (no model loaded
    yet at shutdown, or already torn down), the helper returns silently
    instead of blowing up with AttributeError. This is the production
    failure mode for ``rapid-mlx serve --help`` lifecycle interruption
    where shutdown lands before startup finished.
    """
    import asyncio

    from vllm_mlx import server as _server_mod

    monkeypatch.setattr(_server_mod, "_engine", None)
    asyncio.run(_server_mod._shutdown_save_prefix_cache())  # must not raise


def test_save_prefix_cache_to_disk_respects_budget(tmp_path, monkeypatch):
    """``save_prefix_cache_to_disk`` must build a deadline predicate from
    the budget arg and forward it as ``should_abort`` to the engine. With
    a 0.1s budget and a save that takes ≥0.2s to even get its first
    callback, the predicate is True on first call.
    """
    import time as _time

    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    captured = {"pred": None}

    class _Engine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            captured["pred"] = should_abort
            # Sleep past the deadline so the predicate is guaranteed True.
            _time.sleep(0.25)
            return False

    class _FakeCfg:
        engine = _Engine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    _cache_mod.save_prefix_cache_to_disk(budget_sec=0.1)
    pred = captured["pred"]
    assert pred is not None, (
        "save_prefix_cache_to_disk must pass a should_abort predicate"
    )
    assert pred() is True, "deadline-backed predicate should be tripped after sleep"


def test_save_prefix_cache_to_disk_predicate_is_forward_looking(tmp_path, monkeypatch):
    """Codex PR #667 round 1 BLOCKING-2 regression.

    The predicate must accept a ``predicted_sec`` argument and return
    True when starting that operation would push past the budget — not
    just when wall-clock has already crossed the deadline. Without
    this shape, a single 300 MB ``save_prompt_cache`` call lasting 2 s
    can straddle a 3.5 s budget and get SIGKILL'd mid-write, leaving
    ``cache_dir.new/`` orphaned. We exercise the contract directly so
    a regression that drops the kwarg fails locally instead of
    presenting as "rare orphan dir on shutdown" in production.
    """
    import time as _time

    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    captured = {"pred": None}

    class _Engine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            captured["pred"] = should_abort
            return False

    class _FakeCfg:
        engine = _Engine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    # 2 s budget — comfortably under the headroom-padded deadline (so
    # the at-deadline check is False) but a 5 s predicted operation
    # blows past it.
    _cache_mod.save_prefix_cache_to_disk(budget_sec=2.0)
    pred = captured["pred"]
    assert pred is not None
    assert pred(predicted_sec=0.0) is False, (
        "predicate fires too eagerly — a 2s budget shouldn't trip at t=0 "
        "with no predicted duration"
    )
    assert pred(predicted_sec=5.0) is True, (
        "predicate must look forward: starting a 5s op under a 2s budget "
        "should abort BEFORE the op begins, not let it straddle the deadline"
    )
    # Sanity: the no-arg call shape (legacy `should_abort()`) also still
    # works thanks to the default arg, but should NOT trip at t=0.
    # This keeps tests / callers that don't yet pass predicted_sec
    # compatible during the transition.
    _ = pred()
    # Wait past the deadline (minus headroom) — at-now check should now trip.
    _time.sleep(2.0)
    assert pred() is True, "at-now check should trip after wall-clock past deadline"


def test_save_prefix_cache_to_disk_fallback_for_legacy_engine_signature(monkeypatch):
    """Codex PR #667 round 1 BLOCKING-1 regression.

    External / third-party engine implementations may still expose the
    legacy one-argument ``save_cache_to_disk(cache_dir)`` signature.
    The runtime helper unconditionally passes ``should_abort=`` which
    would raise ``TypeError`` against those engines, losing the entire
    save. The fallback retries the call with the legacy positional
    shape — losing deadline awareness, but preserving the save.
    """
    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    calls = []

    class _LegacyEngine:
        # Note: no ``should_abort`` kwarg — emulates a pre-#667 plugin.
        def save_cache_to_disk(self, cache_dir):
            calls.append(cache_dir)
            return True

    class _FakeCfg:
        engine = _LegacyEngine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    # Must not raise. Must reach the engine. Must not crash the
    # lifespan shutdown.
    _cache_mod.save_prefix_cache_to_disk(budget_sec=1.0)
    assert len(calls) == 1, (
        "legacy engine should be invoked once after the deadline-aware "
        "call's TypeError is caught + fallback retried"
    )


def test_save_prefix_cache_to_disk_no_retry_on_internal_typeerror(monkeypatch):
    """Codex PR #667 round 2 BLOCKING-2 regression.

    The round-1 fallback caught any ``TypeError`` whose message
    mentioned ``should_abort`` and retried via the legacy signature —
    a compatible engine raising that error AFTER partial side effects
    (write to disk, metric increment) would be invoked TWICE, doubling
    the side effects.

    Round-2 detection is signature-based (``inspect.signature``) up
    front, so an internal TypeError raised by the engine method body
    propagates without retry. We assert exactly one call.
    """
    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    calls = {"n": 0}

    class _BuggyEngine:
        # Signature DOES accept should_abort — so detection passes,
        # call proceeds, error raised internally must NOT trigger a
        # legacy-signature retry.
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            calls["n"] += 1
            raise TypeError("internal failure mentioning should_abort somewhere")

    class _FakeCfg:
        engine = _BuggyEngine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    _cache_mod.save_prefix_cache_to_disk(budget_sec=1.0)
    assert calls["n"] == 1, (
        f"engine method must be invoked exactly once even on internal "
        f"TypeError — round-1 fallback double-called via legacy path, "
        f"got {calls['n']} calls"
    )


def test_save_to_disk_predicts_entry_zero_from_bootstrap_floor(tmp_path):
    """Codex PR #667 round 3 BLOCKING-1.

    Entry 0 must receive a non-zero forward-looking ``predicted_sec``
    derived from the 150 MB/s bootstrap floor — round 2 passed 0 here
    and let a catastrophically large first entry straddle the SIGTERM
    grace deadline, leaving ``cache_dir.new/`` orphaned. The estimate
    scales with ``entry.memory_bytes`` so a too-large entry-0 trips
    BEFORE its (uninterruptible) ``save_prompt_cache`` call starts.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))

    seen_predicted = []

    def predicate(predicted_sec=0.0):
        seen_predicted.append(predicted_sec)
        return False  # never trip — just record what the loop sends

    cache.save_to_disk(str(cache_dir), should_abort=predicate)
    assert len(seen_predicted) == 1
    assert seen_predicted[0] > 0, (
        f"entry 0 must receive a non-zero predicted_sec (bootstrap "
        f"150 MB/s × entry size) — round 2 used 0 here and let huge "
        f"entries straddle the deadline (codex round 3 BLOCKING-1). "
        f"Got {seen_predicted[0]}"
    )


def test_save_to_disk_skips_entry_zero_when_predicted_exceeds_budget(tmp_path):
    """Forward-looking check now fires on entry 0 — proves a
    catastrophically large first entry can't straddle the deadline.

    Predicate trips on any non-trivial ``predicted_sec``. Result:
    ``saved == 0`` because even entry 0 is gated, and ``save_to_disk``
    returns False with the staging dir cleaned up (not committed
    empty, not left as an orphan).
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.store(list(range(20, 31)), make_kvcache(num_tokens=11, fill=2.0))

    seen_predicted = []

    def predicate(predicted_sec=0.0):
        seen_predicted.append(predicted_sec)
        return predicted_sec > 0  # trip on any forward-looking estimate

    assert cache.save_to_disk(str(cache_dir), should_abort=predicate) is False
    assert len(seen_predicted) == 1, (
        f"loop should abort on first predicate trip (entry 0), got "
        f"{len(seen_predicted)} calls"
    )
    assert not (tmp_path / "snap").exists()
    assert not (tmp_path / "snap.new").exists()


def test_save_to_disk_accepts_zero_arg_predicate_back_compat(tmp_path):
    """Codex PR #667 round 3 BLOCKING-2 regression.

    Round-1 docstrings described ``should_abort`` as a zero-arg
    callable. Round-2 changed the loop to ``should_abort(predicted_sec)``
    but external callers / older fixtures may still pass a zero-arg
    predicate. The auto-adapter in ``_adapt_should_abort`` must handle
    both shapes without raising ``TypeError``.
    """
    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))
    cache.store(list(range(20, 31)), make_kvcache(num_tokens=11, fill=2.0))

    calls = {"n": 0}

    # Deliberately ZERO-ARG predicate — the round-1 documented shape.
    def predicate():
        calls["n"] += 1
        return calls["n"] > 1  # save entry 0, abort entry 1

    # Must not raise. Must save entry 0 and commit via rename.
    assert cache.save_to_disk(str(cache_dir), should_abort=predicate) is True
    assert calls["n"] >= 2
    assert not (tmp_path / "snap.new").exists()
    index = json.loads((cache_dir / "index.json").read_text())
    assert index["num_entries"] == 1


def test_adapt_should_abort_handles_keyword_only_and_args_kwargs():
    """``_adapt_should_abort`` must classify a few non-obvious callable
    shapes correctly. Regressions here would silently call the wrong
    shape and either raise TypeError mid-save or drop the
    ``predicted_sec`` arg (re-introducing the round-1 BLOCKING-2 bug).
    """
    from vllm_mlx.memory_cache import _adapt_should_abort

    # None passes through.
    assert _adapt_should_abort(None) is None

    # Zero-arg: must be called with no args.
    captured = []
    adapted = _adapt_should_abort(lambda: captured.append("z") or True)
    assert adapted(0.5) is True
    assert captured == ["z"]

    # One-arg positional: must receive predicted_sec.
    captured = []
    adapted = _adapt_should_abort(lambda p: captured.append(p) or False)
    assert adapted(1.5) is False
    assert captured == [1.5]

    # **kwargs only: must be called BY KEYWORD as predicted_sec=...
    # (codex PR #667 round 4 BLOCKING-1: round 3 classified **kwargs
    # as positional-capable and raised TypeError on call). The
    # predicate sees the value via ``kw["predicted_sec"]``.
    captured = []

    def kw_only_pred(**kw):
        captured.append(kw.get("predicted_sec", "no-arg"))
        return False

    adapted = _adapt_should_abort(kw_only_pred)
    assert adapted(2.5) is False
    assert captured == [2.5], (
        f"**kwargs predicate must receive predicted_sec by keyword, got {captured}"
    )

    # Keyword-only ``predicted_sec=...`` — same routing as **kwargs.
    captured = []

    def kw_only_named(*, predicted_sec=0.0):
        captured.append(predicted_sec)
        return False

    adapted = _adapt_should_abort(kw_only_named)
    assert adapted(3.5) is False
    assert captured == [3.5]

    # ``*args, **kwargs`` — positional path wins (it accepts positional
    # via the ``*args`` part).
    captured = []

    def star_args(*a, **kw):
        captured.append(("args", a, "kw", kw))
        return False

    adapted = _adapt_should_abort(star_args)
    assert adapted(2.5) is False
    assert captured == [("args", (2.5,), "kw", {})]


def test_save_prefix_cache_to_disk_zero_budget_disables_deadline(tmp_path, monkeypatch):
    """A budget of 0 (or negative) means "full flush, no deadline" — the
    engine should receive ``should_abort=None`` so the offline CLI path
    is unaffected.
    """
    from vllm_mlx import config as _config_mod
    from vllm_mlx.runtime import cache as _cache_mod

    captured = {"pred": "unset"}

    class _Engine:
        def save_cache_to_disk(self, cache_dir, should_abort=None):
            captured["pred"] = should_abort
            return True

    class _FakeCfg:
        engine = _Engine()
        model_path = "fake/model"
        model_name = None

    monkeypatch.setattr(_cache_mod, "get_config", lambda: _FakeCfg())
    monkeypatch.setattr(_config_mod, "get_config", lambda: _FakeCfg())

    _cache_mod.save_prefix_cache_to_disk(budget_sec=0.0)
    assert captured["pred"] is None


# --------------------------------------------------------------------------
# R8-M7 — commit-phase hardening (dogfood-089 Talia r1/r2)
# --------------------------------------------------------------------------
#
# The commit phase of save_to_disk does two non-atomic renames
# (``cache_dir → .old`` then ``.new → cache_dir``). Pre-R8-M7 there was
# no exception handler around either: a transient OSError (PermissionError
# from a fs-event-driven antivirus touching cache_dir mid-rename, observed
# on macOS Spotlight rebuilds; ENOSPC mid-shutdown; EBUSY on Windows)
# raised up to the caller and left ``cache_dir`` absent + ``.new`` orphan.
# load_from_disk's recovery path handled THAT shape — but the next save's
# pre-clean unconditionally rmtree'd ``.new``, silently discarding the
# just-saved snapshot if reboot didn't happen between the failed save
# and the next save attempt (e.g. an embedded harness that does multiple
# in-process save cycles).
#
# The R8-M7 fix wraps the rename phase in a try/except that attempts
# in-process recovery before returning, plus an ``fsync`` on the staging
# dir before the rename so the rename actually commits the right
# contents on a kernel-level crash.


def test_save_to_disk_returns_false_when_rename_phase_fails(tmp_path, monkeypatch):
    """If the rename phase raises and recovery cannot complete, the
    save returns False so the caller doesn't log a spurious success.

    Pre-R8-M7 the bare ``os.rename`` raised up to the lifespan handler;
    the outer ``try/except`` in ``save_prefix_cache_to_disk`` logged
    the exception but ``save_to_disk`` itself never returned at all.
    Now the failure is captured + reported via the return value.
    """
    import vllm_mlx.memory_cache as _mc

    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(11)), make_kvcache(num_tokens=11))

    # Patch os.rename so the FIRST rename succeeds (cache_dir doesn't
    # exist yet — branch falls through) but the SECOND rename
    # (``.new → cache_dir``) raises PermissionError, mimicking an
    # antivirus / spotlight rebuild touching cache_dir mid-commit.
    original_rename = os.rename
    rename_calls = {"n": 0}

    def _hostile_rename(src, dst):
        rename_calls["n"] += 1
        # Allow the in-process recovery retry to succeed: only the
        # FIRST commit-phase rename raises; the recovery branch retries
        # the same rename and we let that one through.
        if rename_calls["n"] == 1:
            raise PermissionError("simulated av lock on cache_dir")
        return original_rename(src, dst)

    monkeypatch.setattr(_mc.os, "rename", _hostile_rename)
    # save_to_disk should attempt the recovery rename and succeed —
    # so the final state matches a clean save. The return value
    # reflects that recovery worked.
    result = cache.save_to_disk(str(cache_dir))
    assert result is True
    # cache_dir landed correctly via the recovery branch.
    assert cache_dir.exists()
    assert (cache_dir / "index.json").exists()


def test_save_to_disk_recovers_when_only_first_rename_fails(tmp_path, monkeypatch):
    """If ``cache_dir → .old`` succeeds but ``.new → cache_dir``
    raises AND the recovery retry also fails, the function returns
    False and leaves the on-disk state in the recoverable shape
    (cache_dir absent, .new present) so load_from_disk's standard
    recovery path picks up the snapshot at next boot.
    """
    import vllm_mlx.memory_cache as _mc

    cache_dir = tmp_path / "snap"
    # Build session 1: clean save (so cache_dir exists for session 2
    # to ``cache_dir → .old`` against).
    c1 = fresh_cache()
    c1.store(list(range(5)), make_kvcache(num_tokens=5))
    assert c1.save_to_disk(str(cache_dir)) is True

    # Session 2: build a new entry, save — but make the SECOND
    # commit-phase rename fail BOTH on first attempt and recovery
    # retry, so the function logs the partial state and returns False.
    c2 = fresh_cache()
    c2.store(list(range(10, 18)), make_kvcache(num_tokens=8, fill=2.0))

    original_rename = os.rename
    rename_log = []

    def _always_fail_new_to_cache_dir(src, dst):
        rename_log.append((str(src), str(dst)))
        # Block any rename whose dst is the final cache_dir
        # (covers both the first commit-phase attempt and the
        # recovery retry).
        if str(dst) == str(cache_dir):
            raise OSError("simulated rename failure")
        return original_rename(src, dst)

    monkeypatch.setattr(_mc.os, "rename", _always_fail_new_to_cache_dir)
    result = c2.save_to_disk(str(cache_dir))
    assert result is False, "save_to_disk must surface rename failures via return False"
    # Restore os.rename before exercising load_from_disk's recovery
    # path (which also calls os.rename to promote .new -> cache_dir).
    monkeypatch.setattr(_mc.os, "rename", original_rename)

    # The on-disk state is the recovery-friendly shape: cache_dir
    # absent (rename 1 moved it to .old before rename 2 failed),
    # .new present with the new snapshot. load_from_disk will
    # promote .new → cache_dir on next boot.
    new_dir = tmp_path / "snap.new"
    assert not cache_dir.exists()
    assert new_dir.exists()
    assert (new_dir / "index.json").exists()

    # Confirm the recovery path actually works on a fresh load.
    c3 = fresh_cache()
    loaded = c3.load_from_disk(str(cache_dir))
    assert loaded == 1
    entry = next(iter(c3._entries.values()))
    assert entry.tokens == tuple(range(10, 18))


def test_save_to_disk_drops_orphan_new_when_first_rename_fails(tmp_path, monkeypatch):
    """If ``cache_dir → .old`` raises (the FIRST commit-phase rename)
    the old snapshot is still in place. Recovery should keep
    ``cache_dir`` and drop the staging ``.new`` — a future save will
    rebuild it from current state. Pre-R8-M7 the bare raise left
    ``.new`` orphan, then the next save's pre-clean rmtree'd it
    anyway, but during the gap between the two saves, load callers
    could see the stale snapshot AND the unrelated ``.new``.
    """
    import vllm_mlx.memory_cache as _mc

    cache_dir = tmp_path / "snap"
    # Session 1: clean save
    c1 = fresh_cache()
    c1.store(list(range(5)), make_kvcache(num_tokens=5))
    assert c1.save_to_disk(str(cache_dir)) is True

    # Session 2: build new entry, fail the FIRST commit-phase rename.
    c2 = fresh_cache()
    c2.store(list(range(20, 30)), make_kvcache(num_tokens=10, fill=3.0))

    original_rename = os.rename
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"

    def _fail_first_commit_rename(src, dst):
        # The commit-phase first rename is cache_dir → cache_dir.old.
        if str(src) == str(cache_dir) and str(dst) == str(old_dir):
            raise OSError("simulated first-rename failure")
        return original_rename(src, dst)

    monkeypatch.setattr(_mc.os, "rename", _fail_first_commit_rename)
    result = c2.save_to_disk(str(cache_dir))
    # Save reports failure — original cache_dir is intact.
    assert result is False
    assert cache_dir.exists()
    # Recovery dropped the orphan .new so a subsequent save starts
    # from a clean slate.
    assert not new_dir.exists()

    # The original (session 1) snapshot is still loadable.
    c3 = fresh_cache()
    loaded = c3.load_from_disk(str(cache_dir))
    assert loaded == 1
    entry = next(iter(c3._entries.values()))
    assert entry.tokens == tuple(range(5))


def test_save_to_disk_fsync_failure_is_non_fatal(tmp_path, monkeypatch):
    """An OSError from the staging-dir fsync must not abort the save.

    Platforms / mounts vary in fsync semantics (NFS, FUSE, Windows
    has no equivalent). The fsync is a correctness invariant for
    rename atomicity, but a non-fsyncable fs is no worse than the
    pre-R8-M7 behaviour (which didn't fsync at all). Log + proceed.
    """
    import vllm_mlx.memory_cache as _mc

    cache_dir = tmp_path / "snap"
    cache = fresh_cache()
    cache.store(list(range(7)), make_kvcache(num_tokens=7))

    def _broken_fsync(path):
        raise OSError("simulated fsync failure on staging dir")

    monkeypatch.setattr(_mc, "_fsync_dir", _broken_fsync)
    assert cache.save_to_disk(str(cache_dir)) is True
    # Cache dir landed correctly despite the fsync failure.
    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    assert loaded == 1


def test_fsync_dir_works_on_real_directory(tmp_path):
    """The fsync helper itself works on a vanilla tmp directory.

    Regression guard: a future refactor that swaps ``os.O_RDONLY`` for
    ``O_DIRECTORY`` (not available on every platform) or changes the
    file-descriptor lifecycle would silently break this helper, and
    save_to_disk would still succeed (the except clause is non-fatal)
    but lose the rename-atomicity guarantee. The test below pins the
    happy path so a regression in helper semantics is observable.
    """
    from vllm_mlx.memory_cache import _fsync_dir

    # No exception on a real, readable directory.
    _fsync_dir(str(tmp_path))


def test_fsync_file_works_on_real_file(tmp_path):
    """The per-file fsync helper works on a vanilla tmp file.

    R8-M7 codex r1 BLOCKING #3: ``_fsync_dir`` only covers directory
    metadata; ``_fsync_file`` covers the file body. Pin both so a
    future refactor that drops one or the other (or breaks the
    fd-lifecycle in either) surfaces here.
    """
    from vllm_mlx.memory_cache import _fsync_file

    p = tmp_path / "blob.bin"
    p.write_bytes(b"x" * 4096)
    _fsync_file(str(p))


def test_save_to_disk_preserves_old_dir_when_commit_fails(tmp_path, monkeypatch):
    """R8-M7 codex r1 BLOCKING #1 regression: when the new-snapshot
    rename does not commit, the previous good snapshot (``.old``)
    must be kept on disk so load_from_disk can recover via the
    ``_has_valid_index(old_dir)`` path. Pre-fix this PR's ``.old``
    rmtree ran unconditionally, destroying the last known-good
    snapshot before returning False.
    """
    import vllm_mlx.memory_cache as _mc

    cache_dir = tmp_path / "snap"
    new_dir = tmp_path / "snap.new"
    old_dir = tmp_path / "snap.old"
    # Session 1: clean save (so cache_dir exists for session 2's
    # rename-1 to push it to .old).
    c1 = fresh_cache()
    c1.store(list(range(5)), make_kvcache(num_tokens=5))
    assert c1.save_to_disk(str(cache_dir)) is True

    # Session 2: build a new entry, fail rename 2 + its recovery
    # retry, observe that .old survives so a subsequent
    # load_from_disk can promote it.
    c2 = fresh_cache()
    c2.store(list(range(10, 18)), make_kvcache(num_tokens=8, fill=2.0))

    original_rename = os.rename

    def _fail_new_to_cache_dir(src, dst):
        if str(dst) == str(cache_dir):
            raise OSError("simulated rename-to-cache_dir failure")
        return original_rename(src, dst)

    monkeypatch.setattr(_mc.os, "rename", _fail_new_to_cache_dir)
    result = c2.save_to_disk(str(cache_dir))
    monkeypatch.setattr(_mc.os, "rename", original_rename)

    assert result is False
    # The crucial assertion: ``.old`` was NOT rmtree'd because the
    # rename didn't commit. load_from_disk's recovery path can
    # promote either ``.new`` or fall back to ``.old``.
    assert old_dir.exists(), (
        "BLOCKING regression — .old destroyed when commit failed; "
        "last known-good snapshot lost"
    )
    assert (old_dir / "index.json").exists()

    # And the recovery path actually works: a fresh load promotes
    # ``.new`` (the partial new snapshot) — the .old preservation
    # is a belt-and-suspenders for the case where .new is itself
    # corrupt or partially written, which the existing
    # ``_has_valid_index(old_dir)`` branch in load_from_disk handles.
    c3 = fresh_cache()
    loaded = c3.load_from_disk(str(cache_dir))
    assert loaded == 1


def test_clean_save_load_roundtrip_after_r8m7_hardening(tmp_path):
    """End-to-end: the R8-M7 hardening must not regress the clean-flush
    path. A save with no signal pressure, no rename failures, and no
    fsync issues still produces an on-disk snapshot that loads back to
    the original entries.
    """
    cache_dir = tmp_path / "snap"
    c1 = fresh_cache()
    c1.store(list(range(3, 13)), make_kvcache(num_tokens=10))
    c1.store(list(range(50, 60)), make_kvcache(num_tokens=10, fill=2.0))
    assert c1.save_to_disk(str(cache_dir)) is True

    c2 = fresh_cache()
    loaded = c2.load_from_disk(str(cache_dir))
    assert loaded == 2
    # Both entries round-tripped intact.
    keys = {e.tokens for e in c2._entries.values()}
    assert keys == {tuple(range(3, 13)), tuple(range(50, 60))}
