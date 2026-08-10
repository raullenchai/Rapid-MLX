# SPDX-License-Identifier: Apache-2.0
"""Tests for the disk-streaming MoE expert-weight cache (vllm_mlx/expert_cache.py).

Ported from the disk-streaming MoE spike's `test_expert_cache.py` (see
PRD-rapid-mlx-integration.md) — same 9 cases, no behavior changes, adapted
to this repo's class-grouped/docstring pytest style (see test_memory_cache.py
for the precedent this follows).

All bundles here are fake/dummy: a tiny dataclass carrying a controlled
`.nbytes`. No real model, safetensors checkpoint, disk I/O, or MLX import
is needed -- exercising the point of the module (it only knows key/value/
LRU/byte-budget, nothing about where a real bundle comes from).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_mlx.expert_cache import ExpertCache


@dataclass
class FakeBundle:
    """Stand-in for a real weight bundle: just a name + controllable size."""

    tag: str
    nbytes: int = 10


class CountingFetcher:
    """`fetch_fn` stub that records every call so tests can assert
    fetch_fn was (or wasn't) invoked on a given `get()`.
    """

    def __init__(self, nbytes: int = 10):
        self.calls: list[tuple[int, int]] = []
        self.nbytes = nbytes

    def __call__(self, layer_idx: int, expert_id: int) -> FakeBundle:
        self.calls.append((layer_idx, expert_id))
        return FakeBundle(tag=f"L{layer_idx}E{expert_id}", nbytes=self.nbytes)


class _Sized:
    """Minimal `.nbytes`-carrying leaf for the nested-dict size_fn test."""

    def __init__(self, n: int):
        self.nbytes = n


def _sized(n: int) -> _Sized:
    return _Sized(n)


class TestExpertCache:
    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError, match="must be positive"):
            ExpertCache(lambda layer, expert: object(), budget_bytes=0)

        with pytest.raises(ValueError, match="must be positive"):
            ExpertCache(lambda layer, expert: object(), budget_bytes=-1)

    """Tests for ExpertCache: miss/hit correctness, LRU eviction order,
    recency refresh on hit, and the byte-budget invariant.
    """

    def test_get_returns_correct_bundle_on_miss_and_calls_fetch_fn(self):
        fetcher = CountingFetcher()
        cache = ExpertCache(fetcher, budget_bytes=1_000)

        bundle = cache.get(2, 5)
        assert bundle.tag == "L2E5"
        assert fetcher.calls == [(2, 5)]

    def test_get_returns_same_bundle_on_hit_without_calling_fetch_fn_again(self):
        fetcher = CountingFetcher()
        cache = ExpertCache(fetcher, budget_bytes=1_000)

        first = cache.get(2, 5)
        second = cache.get(2, 5)

        assert second is first  # identical cached object, not a re-fetch
        assert fetcher.calls == [(2, 5)]  # fetch_fn called exactly once

    def test_cache_never_exceeds_budget_bytes(self):
        fetcher = CountingFetcher(nbytes=10)
        cache = ExpertCache(fetcher, budget_bytes=25)  # room for ~2 entries

        for expert_id in range(10):  # force many evictions
            cache.get(0, expert_id)
        assert cache.total_bytes <= 25

    def test_oversized_bundle_is_returned_but_not_cached(self):
        fetcher = CountingFetcher(nbytes=30)
        cache = ExpertCache(fetcher, budget_bytes=25)

        first = cache.get(0, 0)
        second = cache.get(0, 0)

        assert first.tag == second.tag == "L0E0"
        assert len(fetcher.calls) == 2
        assert cache.total_bytes == 0
        assert not cache._store

        # Repeated hits on the same key must not grow total_bytes either.
        for _ in range(5):
            cache.get(0, 9)
            assert cache.total_bytes <= 25

    def test_budget_holds_across_mixed_hit_miss_sequence(self):
        fetcher = CountingFetcher(nbytes=10)
        cache = ExpertCache(fetcher, budget_bytes=30)

        sequence = [(0, 0), (0, 1), (0, 0), (0, 2), (0, 3), (0, 1), (0, 4)]
        for layer_idx, expert_id in sequence:
            cache.get(layer_idx, expert_id)
            assert cache.total_bytes <= 30

    def test_overflow_evicts_least_recently_used_entry_first(self):
        fetcher = CountingFetcher(nbytes=10)
        cache = ExpertCache(fetcher, budget_bytes=25)  # fits 2 entries, not 3

        cache.get(0, 0)  # LRU order: [0]
        cache.get(0, 1)  # LRU order: [0, 1]
        cache.get(0, 2)  # overflow: evicts 0 (oldest); order: [1, 2]

        assert (0, 0) not in cache._store
        assert (0, 1) in cache._store
        assert (0, 2) in cache._store

        # Confirm eviction really happened: re-`get`ing (0, 0) is a fresh fetch.
        fetcher.calls.clear()
        cache.get(0, 0)
        assert fetcher.calls == [(0, 0)]

    def test_hit_refreshes_recency_so_it_is_not_next_eviction_candidate(self):
        fetcher = CountingFetcher(nbytes=10)
        cache = ExpertCache(fetcher, budget_bytes=25)  # fits 2 entries, not 3

        cache.get(0, 0)  # order: [0]
        cache.get(0, 1)  # order: [0, 1]
        cache.get(0, 0)  # hit -> refreshed; order: [1, 0]
        cache.get(0, 2)  # overflow: evicts 1 (now oldest), not 0

        assert (0, 0) in cache._store  # survived because the hit refreshed it
        assert (0, 1) not in cache._store  # evicted -- was LRU after the hit
        assert (0, 2) in cache._store

    def test_hit_miss_counters_across_known_sequence(self):
        fetcher = CountingFetcher(nbytes=10)
        cache = ExpertCache(fetcher, budget_bytes=25)  # fits 2 entries

        cache.get(0, 0)  # miss 1
        cache.get(0, 1)  # miss 2
        cache.get(0, 2)  # miss 3, evicts (0, 0)
        assert cache.hits == 0 and cache.misses == 3

        cache.get(0, 1)  # hit 1 (still cached)
        cache.get(0, 2)  # hit 2 (still cached)
        assert cache.hits == 2 and cache.misses == 3

        cache.get(0, 0)  # miss 4, evicted earlier -> re-fetch, evicts LRU
        assert cache.hits == 2 and cache.misses == 4


class TestExpertCacheSizeFn:
    """Decoupling: default size_fn duck-types nested dict bundles too (the
    shape a real gate/up/down-merged 9-array bundle would have), and a
    custom size_fn can be supplied for exotic shapes.
    """

    def test_default_size_fn_sums_nested_dict_bundle(self):
        def fetch(layer_idx, expert_id):
            return {
                "gate": {"weight": _sized(4), "scales": _sized(2), "biases": _sized(2)},
                "up": {"weight": _sized(4), "scales": _sized(2), "biases": _sized(2)},
                "down": {"weight": _sized(4), "scales": _sized(2), "biases": _sized(2)},
            }

        cache = ExpertCache(fetch, budget_bytes=1_000)
        cache.get(0, 0)
        assert cache.total_bytes == 4 * 3 + 2 * 6

    def test_custom_size_fn_is_used_instead_of_default(self):
        def fetch(layer_idx, expert_id):
            return "opaque-blob-not-a-dict"

        cache = ExpertCache(fetch, budget_bytes=1_000, size_fn=lambda b: 7)
        cache.get(0, 0)
        assert cache.total_bytes == 7
