# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``vllm_mlx.runtime.radix_index.RadixPrefixIndex`` (R15-P1,
task #303).

These tests cover the radix-tree data structure in isolation — no model,
no scheduler, no MLX runtime needed. They are CI-cheap (sub-second) and
exercise:

- Insert / remove / clear / __len__ / __contains__
- ``longest_prefix`` (exact, prefix, divergent, empty)
- Refcount + node-pruning invariants on remove
- Dedup-bytes accounting (the headline footprint-reduction metric)
- Persistence: round-trip save → load → save → load equivalence
- ``rebuild_from_keys`` for the cold-boot fallback path
- Thread-safety smoke test (concurrent insert + lookup)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from vllm_mlx.runtime.radix_index import (
    _BYTES_PER_TOKEN_INT32,
    RadixPrefixIndex,
    RadixStats,
)


class TestRadixStats:
    def test_empty_stats(self):
        stats = RadixStats()
        d = stats.to_dict()
        assert d["hits"] == 0
        assert d["misses"] == 0
        assert d["node_count"] == 0
        assert d["entry_count"] == 0
        assert d["deduped_prefix_bytes_saved"] == 0
        assert d["lookup_p50_seconds"] == 0.0
        assert d["lookup_p99_seconds"] == 0.0

    def test_lookup_percentiles_populated(self):
        stats = RadixStats()
        for ns in (100, 200, 300, 400, 500):
            stats.record_lookup(ns)
        d = stats.to_dict()
        # p50 of [100,200,300,400,500] is 300 ns → 3e-7 s
        assert d["lookup_p50_seconds"] == pytest.approx(3e-7)
        # p99 with 5 samples lands at idx 3 → 400 ns → 4e-7 s
        assert d["lookup_p99_seconds"] == pytest.approx(4e-7)


class TestRadixInsertLookup:
    def test_empty_insert_noop(self):
        idx = RadixPrefixIndex()
        idx.insert([])
        assert len(idx) == 0
        assert idx.stats()["node_count"] == 0

    def test_single_insert_and_exact_match(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        assert [1, 2, 3] in idx
        assert [1, 2] not in idx
        assert len(idx) == 1
        matched, key = idx.longest_prefix([1, 2, 3])
        assert matched == [1, 2, 3]
        assert key == (1, 2, 3)

    def test_prefix_match_returns_shorter_terminal(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        # Query is a SUPERSEQUENCE of the stored entry — longest stored
        # prefix is [1,2,3].
        matched, key = idx.longest_prefix([1, 2, 3, 4, 5])
        assert matched == [1, 2, 3]
        assert key == (1, 2, 3)

    def test_query_shorter_than_stored_returns_partial_walk(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3, 4, 5])
        # Query is a strict prefix of the stored entry; nothing terminal
        # along the walk, so we return ``(query, None)``.
        matched, key = idx.longest_prefix([1, 2, 3])
        assert matched == [1, 2, 3]
        assert key is None

    def test_diverging_query_returns_empty(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        matched, key = idx.longest_prefix([9, 9, 9])
        assert matched == []
        assert key is None

    def test_two_entries_share_prefix_returns_longer_terminal(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        idx.insert([1, 2, 3, 4, 5])
        matched, key = idx.longest_prefix([1, 2, 3, 4, 5, 6])
        # Longest stored prefix of the query is [1,2,3,4,5].
        assert matched == [1, 2, 3, 4, 5]
        assert key == (1, 2, 3, 4, 5)

    def test_branching(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        idx.insert([1, 2, 4])
        idx.insert([1, 5, 6])
        # Walk for [1,2,4,9]
        matched, key = idx.longest_prefix([1, 2, 4, 9])
        assert matched == [1, 2, 4]
        assert key == (1, 2, 4)
        # Walk for [1,5,7]
        matched, key = idx.longest_prefix([1, 5, 7])
        # We don't terminate at [1,5] (no terminal) so this is the
        # partial-walk path: matched=[1,5], key=None.
        assert matched == [1, 5]
        assert key is None


class TestRadixDedupAccounting:
    def test_first_insert_records_no_dedup(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        assert idx.stats()["deduped_prefix_bytes_saved"] == 0

    def test_shared_prefix_records_dedup(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3, 4])  # +0
        idx.insert([1, 2, 3, 5])  # shares [1,2,3] → +3 tokens worth
        assert idx.stats()["deduped_prefix_bytes_saved"] == 3 * _BYTES_PER_TOKEN_INT32

    def test_shared_prefix_accumulates_across_inserts(self):
        # 10 tenants on a 100-token shared system prompt, each with a
        # different 5-token continuation. Dedup should be
        # 9 inserts × 100 tokens × _BYTES_PER_TOKEN_INT32 (first insert
        # gets nothing, the other 9 each share the full 100 tokens with
        # the existing entry).
        idx = RadixPrefixIndex()
        shared = list(range(1, 101))
        for i in range(10):
            idx.insert(shared + [1000 + i, 2000 + i, 3000 + i, 4000 + i, 5000 + i])
        deduped = idx.stats()["deduped_prefix_bytes_saved"]
        assert deduped == 9 * 100 * _BYTES_PER_TOKEN_INT32


class TestRadixRemove:
    def test_remove_existing_returns_true(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        assert idx.remove([1, 2, 3]) is True
        assert [1, 2, 3] not in idx
        assert len(idx) == 0

    def test_remove_missing_returns_false(self):
        idx = RadixPrefixIndex()
        assert idx.remove([1, 2, 3]) is False
        idx.insert([1, 2, 3])
        assert idx.remove([1, 2, 4]) is False
        assert idx.remove([1, 2]) is False  # not terminal
        assert [1, 2, 3] in idx

    def test_remove_prunes_empty_nodes(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        assert idx.stats()["node_count"] == 3
        idx.remove([1, 2, 3])
        assert idx.stats()["node_count"] == 0

    def test_remove_keeps_shared_path(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        idx.insert([1, 2, 4])
        # Shared nodes [1] and [2], plus the two leaves [3] and [4].
        assert idx.stats()["node_count"] == 4
        idx.remove([1, 2, 3])
        # [1] and [2] survive because [1,2,4] still threads through them.
        assert idx.stats()["node_count"] == 3
        assert [1, 2, 4] in idx
        assert [1, 2, 3] not in idx


class TestRadixPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        idx.insert([1, 2, 4, 5])
        idx.insert([7, 8, 9])
        path = str(tmp_path / "subdir" / "radix.index")
        idx.save(path)
        assert os.path.exists(path)
        # Hydrate a fresh index from the file.
        idx2 = RadixPrefixIndex()
        assert idx2.load(path) is True
        assert len(idx2) == 3
        assert [1, 2, 3] in idx2
        assert [1, 2, 4, 5] in idx2
        assert [7, 8, 9] in idx2

    def test_load_missing_returns_false(self, tmp_path):
        idx = RadixPrefixIndex()
        assert idx.load(str(tmp_path / "does_not_exist.index")) is False

    def test_load_corrupt_json_returns_false(self, tmp_path):
        path = str(tmp_path / "radix.index")
        with open(path, "w") as f:
            f.write("{not json")
        idx = RadixPrefixIndex()
        assert idx.load(path) is False

    def test_load_wrong_version_returns_false(self, tmp_path):
        path = str(tmp_path / "radix.index")
        with open(path, "w") as f:
            json.dump({"version": 99, "keys": [[1, 2, 3]]}, f)
        idx = RadixPrefixIndex()
        assert idx.load(path) is False
        assert len(idx) == 0

    def test_save_atomicity_no_partial_file(self, tmp_path):
        # The .tmp pattern: after save, .tmp must NOT exist.
        idx = RadixPrefixIndex()
        idx.insert([1, 2])
        path = str(tmp_path / "radix.index")
        idx.save(path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

    def test_rebuild_from_keys(self):
        idx = RadixPrefixIndex()
        idx.rebuild_from_keys([(1, 2, 3), (1, 2, 4), (5, 6)])
        assert len(idx) == 3
        assert [1, 2, 3] in idx
        assert [1, 2, 4] in idx
        assert [5, 6] in idx

    def test_rebuild_idempotent(self):
        # Calling rebuild twice with the same input must converge.
        idx = RadixPrefixIndex()
        keys = [(1, 2, 3), (1, 2, 4), (5, 6)]
        idx.rebuild_from_keys(keys)
        nodes_first = idx.stats()["node_count"]
        idx.rebuild_from_keys(keys)
        nodes_second = idx.stats()["node_count"]
        assert nodes_first == nodes_second


class TestRadixClear:
    def test_clear_resets_state(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        idx.insert([4, 5, 6])
        idx.clear()
        assert len(idx) == 0
        s = idx.stats()
        assert s["node_count"] == 0
        assert s["entry_count"] == 0
        assert s["max_depth"] == 0


class TestRadixThreadSafety:
    """Smoke test for concurrent mutation + lookup.

    The radix uses an RLock for mutations and lock-free reads (Python's
    GIL gives us atomic dict.get()). We don't assert a particular timing
    here — just that N parallel inserts + N parallel lookups never raise
    and produce a coherent final state.
    """

    def test_concurrent_insert_lookup_no_crash(self):
        idx = RadixPrefixIndex()
        prefix = list(range(50))  # shared 50-token "system prompt"
        N = 32

        def inserter(tid: int) -> None:
            idx.insert(prefix + [1000 + tid, 2000 + tid])

        def reader(tid: int) -> tuple[list[int], object]:
            return idx.longest_prefix(prefix + [1000 + tid, 2000 + tid, 9999])

        with ThreadPoolExecutor(max_workers=8) as ex:
            insert_futures = [ex.submit(inserter, t) for t in range(N)]
            for f in as_completed(insert_futures):
                f.result()
            # Now do parallel lookups against the populated tree.
            read_futures = [ex.submit(reader, t) for t in range(N)]
            for f in as_completed(read_futures):
                matched, _ = f.result()
                # Every reader must at least see the shared prefix.
                assert matched[: len(prefix)] == prefix

        assert len(idx) == N


class TestRadixMaxDepthGauge:
    def test_max_depth_tracks_longest_insert(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        assert idx.stats()["max_depth"] == 3
        idx.insert([1, 2, 3, 4, 5, 6])
        assert idx.stats()["max_depth"] == 6
        # A shorter subsequent insert must not lower the gauge.
        idx.insert([7])
        assert idx.stats()["max_depth"] == 6


class TestRadixHitMissCounters:
    def test_hits_and_misses_tracked(self):
        idx = RadixPrefixIndex()
        idx.insert([1, 2, 3])
        idx.longest_prefix([1, 2, 3, 4])  # hit
        idx.longest_prefix([1, 2, 3])  # hit
        idx.longest_prefix([9, 9])  # miss (empty match)
        idx.longest_prefix([1, 2])  # miss (no terminal yet)
        s = idx.stats()
        assert s["hits"] == 2
        assert s["misses"] == 2
