# SPDX-License-Identifier: Apache-2.0
"""R6-H6: prefix-cache eviction counter wiring.

The 0.8.7 dogfood (Hiro R2) flagged the
``rapid_mlx_prefix_cache_evictions_total`` and
``rapid_mlx_prefix_cache_pressure_evictions_total`` Prometheus series
both stuck at 0 even after 50 distinct ~1K-token system prompts had
ballooned the memory-aware cache to 31 GB and Metal allocated 35.5 GB.
Two interacting root causes:

1. ``MemoryCacheConfig.compute_memory_limit`` honored only the
   ``max_memory_percent`` heuristic (default 20% of available RAM) or
   the programmatic ``max_memory_mb`` override. On a 256 GB Apple
   Silicon box that's a 51 GB cache budget — far higher than the
   prefix cache should ever grow, so the LRU evict-on-insert path
   inside ``MemoryAwarePrefixCache.store`` never fired and the
   ``evictions`` stat (surfaced as
   ``rapid_mlx_prefix_cache_evictions_total``) stayed at 0.
2. ``Scheduler.evict_prefix_cache_under_pressure`` short-circuited
   when ``gpu_memory_utilization`` was unset (default 0.0), so the
   pressure-eviction trigger never fired and
   ``rapid_mlx_prefix_cache_pressure_evictions_total`` stayed at 0
   regardless of how much memory the cache had pinned.

The fix:
* Add a ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES`` env-var override so
  operators can bound the cache to a known ceiling.
* Add a cache-self-pressure trigger to
  ``evict_prefix_cache_under_pressure`` that fires when
  ``memory_aware_cache._current_memory`` crosses
  ``metal_pressure_evict_fraction × _max_memory`` — INDEPENDENT of
  ``gpu_memory_utilization``.

These tests exercise both paths against fakes so they run on any
machine without spinning up a model load.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

# ─── compute_memory_limit env-var override ──────────────────────────


def test_env_override_takes_precedence_over_heuristic(monkeypatch):
    """``RAPID_MLX_PREFIX_CACHE_MAX_BYTES`` wins over the
    ``max_memory_percent`` heuristic so operators can bound the cache
    even on large-memory hosts where 20% of RAM is excessive."""
    from vllm_mlx.memory_cache import MemoryCacheConfig

    # Use a value safely above the 100 MiB floor so the test asserts
    # the env override directly rather than the floor.
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(500 * 1024 * 1024))

    cfg = MemoryCacheConfig(max_memory_percent=0.99)  # heuristic would be huge
    assert cfg.compute_memory_limit() == 500 * 1024 * 1024


def test_env_override_takes_precedence_over_max_memory_mb(monkeypatch):
    """The env var is the operator's ultimate ceiling — programmatic
    ``max_memory_mb`` from the CLI / config plumbing must NOT override
    it. This lets operators pin a hard cap that survives across
    code changes that wire in new programmatic defaults."""
    from vllm_mlx.memory_cache import MemoryCacheConfig

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(200 * 1024 * 1024))

    cfg = MemoryCacheConfig(max_memory_mb=10000)  # 10 GiB programmatic
    assert cfg.compute_memory_limit() == 200 * 1024 * 1024


def test_env_unset_falls_back_to_legacy_max_memory_mb(monkeypatch):
    """When the env var is unset, the programmatic override still
    wins over the heuristic — keeps callers that already set
    ``max_memory_mb`` working unchanged."""
    from vllm_mlx.memory_cache import _BYTES_PER_MB, MemoryCacheConfig

    monkeypatch.delenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", raising=False)

    cfg = MemoryCacheConfig(max_memory_mb=512)
    assert cfg.compute_memory_limit() == 512 * _BYTES_PER_MB


def test_env_invalid_value_falls_through_silently(monkeypatch):
    """A misconfigured env var (e.g. ``"5GB"`` instead of bytes) must
    NOT crash the server — fall through to the legacy heuristic so
    a typo in the operator's env still boots a working server."""
    from vllm_mlx.memory_cache import _BYTES_PER_MB, MemoryCacheConfig

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", "5GB")

    cfg = MemoryCacheConfig(max_memory_mb=512)
    # Falls through to max_memory_mb (legacy heuristic).
    assert cfg.compute_memory_limit() == 512 * _BYTES_PER_MB


def test_env_zero_or_negative_value_falls_through(monkeypatch):
    """Zero/negative values must NOT silently disable the cache;
    they fall through so the legacy heuristic applies."""
    from vllm_mlx.memory_cache import _BYTES_PER_MB, MemoryCacheConfig

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", "0")

    cfg = MemoryCacheConfig(max_memory_mb=256)
    assert cfg.compute_memory_limit() == 256 * _BYTES_PER_MB

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", "-100")
    assert cfg.compute_memory_limit() == 256 * _BYTES_PER_MB


def test_env_value_passes_through_unclamped(monkeypatch):
    """An explicit env value is the operator's ceiling — we honor it
    verbatim, NOT clamped to ``_MIN_MEMORY_BYTES``. That floor only
    exists to protect the heuristic ``percent × available_RAM`` path
    from underestimating on a memory-starved host; once an operator
    has typed a specific number, they want that specific number (for
    instance, deterministic test fixtures that drive eviction
    against a known cap).
    """
    from vllm_mlx.memory_cache import MemoryCacheConfig

    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", "1024")

    cfg = MemoryCacheConfig()
    assert cfg.compute_memory_limit() == 1024


# ─── LRU evictions counter ticks on cap pressure ─────────────────────


class _FakeCacheLayer:
    """Minimal cache layer that satisfies the memory estimator's
    ``state -> (keys, values)`` branch. Keeps the test free of mlx-lm
    by exposing two fake arrays whose ``shape × dtype.size`` matches
    the byte size we want to charge against the cache budget.
    """

    class _FakeDtype:
        size = 4  # fp32-sized — the estimator multiplies shape × size.

    class _FakeArr:
        def __init__(self, n: int):
            self.shape = (n,)
            self.dtype = _FakeCacheLayer._FakeDtype()
            self.nbytes = n * 4

    def __init__(self, byte_size: int):
        # Split the requested bytes evenly between K and V (each fp32-sized).
        n = max(1, byte_size // (2 * 4))
        keys = self._FakeArr(n)
        values = self._FakeArr(n)
        # The estimator's ``hasattr(layer_cache, "state")`` branch
        # unpacks ``keys, values = layer_cache.state`` — so state
        # must be a 2-tuple, not 3.
        self.state = (keys, values)
        self.offset = n

    def is_trimmable(self) -> bool:
        return False


def _make_cache_entry(byte_size: int):
    return [_FakeCacheLayer(byte_size)]


def test_lru_evictions_total_ticks_when_cache_exceeds_env_cap(monkeypatch):
    """The end-to-end claim of R6-H6: with the env-var override set
    to a small ceiling, inserting more entries than the ceiling can
    hold should evict LRU entries and tick the ``evictions`` stat
    that the metrics route surfaces as
    ``rapid_mlx_prefix_cache_evictions_total``."""
    # Use a small cap so a handful of fake entries trip eviction.
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(8 * 1024 * 1024))

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    cfg = MemoryCacheConfig()
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)
    assert cache.get_stats()["evictions"] == 0

    # Each entry is ~3 MiB; with an 8 MiB cap, the 3rd insertion must
    # evict at least one earlier entry. 5 distinct token sequences keeps
    # the math clearly past the cap regardless of overhead.
    per_entry_bytes = 3 * 1024 * 1024
    for i in range(5):
        tokens = list(range(i * 100, i * 100 + 64))
        cache.store(tokens, _make_cache_entry(per_entry_bytes))

    stats = cache.get_stats()
    assert stats["evictions"] >= 1, (
        f"LRU-on-cap evictions did not fire; cache stat snapshot: {stats!r}"
    )
    # Cache memory must NOT exceed the configured cap (the whole point
    # of the env override).
    assert stats["current_memory_mb"] <= stats["max_memory_mb"] + 1.0


# ─── pressure evictions counter ticks on cache-self pressure ─────────


class _StubMemoryAwareCache:
    """Minimal stand-in for MemoryAwarePrefixCache that exposes only
    the surface ``evict_prefix_cache_under_pressure`` consults: an
    ``_entries`` ledger, a ``_current_memory`` byte count, a
    ``_max_memory`` cap, a re-entrant ``_lock``, and an ``_evict_lru``
    that pops the LRU entry. Lets the scheduler test exercise the
    cache-self-pressure path without loading mlx-lm.
    """

    def __init__(self, max_memory: int, per_entry_bytes: int):
        self._entries: dict[tuple[int, ...], object] = {}
        self._current_memory = 0
        self._max_memory = max_memory
        self._per_entry_bytes = per_entry_bytes
        self._lock = threading.RLock()
        self._evict_calls = 0

    def insert(self, key: tuple[int, ...]) -> None:
        self._entries[key] = object()
        self._current_memory += self._per_entry_bytes

    def _evict_lru(self) -> None:
        if not self._entries:
            return
        oldest = next(iter(self._entries))
        del self._entries[oldest]
        self._current_memory = max(0, self._current_memory - self._per_entry_bytes)
        self._evict_calls += 1


class _StubBatchGenerator:
    """No-op batch generator surface the Scheduler constructor pokes."""


def _make_scheduler_with_stub_cache(stub_cache):
    """Build a real Scheduler instance with a stub model + stub
    memory_aware_cache attached. We bypass __init__ to avoid the full
    BatchGenerator wiring (which needs an mlx-lm model) and only set
    the attributes ``evict_prefix_cache_under_pressure`` reads.
    """
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    sched = Scheduler.__new__(Scheduler)
    sched.config = SchedulerConfig(
        gpu_memory_utilization=0.0,  # Metal cap disabled — the R6-H6 case
        metal_pressure_evict_fraction=0.9,
    )
    sched.memory_aware_cache = stub_cache
    sched.prefix_cache = None
    sched.block_aware_cache = None
    sched.num_prefix_cache_pressure_evictions = 0
    sched._metal_cap_bytes = 0
    sched._metal_cap_bytes_resolved = True
    return sched


def test_pressure_evictions_total_ticks_on_cache_self_pressure():
    """Cache-self trigger: when ``_current_memory`` crosses
    ``fraction × _max_memory`` and ``gpu_memory_utilization`` is the
    default 0.0, eviction MUST still fire and tick
    ``num_prefix_cache_pressure_evictions`` so the metric series
    surfaces real activity instead of a stuck-zero line.
    """
    cap = 10 * 1024 * 1024  # 10 MiB
    per_entry = 2 * 1024 * 1024  # 2 MiB
    stub_cache = _StubMemoryAwareCache(max_memory=cap, per_entry_bytes=per_entry)
    # Fill the cache PAST the 0.9 × 10 MiB = 9 MiB pressure threshold.
    for i in range(5):
        stub_cache.insert((i,))
    assert stub_cache._current_memory == 5 * per_entry

    sched = _make_scheduler_with_stub_cache(stub_cache)

    # Patch mx.clear_cache so the test doesn't need a Metal device.
    with patch("vllm_mlx.scheduler.mx.clear_cache"):
        evicted = sched.evict_prefix_cache_under_pressure()

    assert evicted >= 1, (
        "cache-self-pressure trigger did not fire under "
        f"gpu_memory_utilization=0; entries={len(stub_cache._entries)} "
        f"current={stub_cache._current_memory} max={cap}"
    )
    assert sched.num_prefix_cache_pressure_evictions == evicted, (
        "pressure_evictions_total counter did not match actual evictions: "
        f"counter={sched.num_prefix_cache_pressure_evictions} evicted={evicted}"
    )
    # Loop must stop once the ledger drops below the threshold so a
    # transient burst does not wipe the entire cache.
    assert stub_cache._current_memory < int(cap * 0.9), (
        "loop did not stop when cache-self pressure dropped below threshold: "
        f"current={stub_cache._current_memory} threshold={int(cap * 0.9)}"
    )


def test_pressure_eviction_loop_short_circuits_when_no_trigger_configured():
    """No memory-aware cache + ``gpu_memory_utilization=0`` → the
    pressure loop returns 0 immediately without scanning anything.
    Keeps the no-op cost off the path on engines with neither
    trigger configured."""
    sched = _make_scheduler_with_stub_cache(stub_cache=None)
    # No cache → both triggers stay zero.
    with patch("vllm_mlx.scheduler.mx.clear_cache") as clear_cache_mock:
        assert sched.evict_prefix_cache_under_pressure() == 0
        clear_cache_mock.assert_not_called()
    assert sched.num_prefix_cache_pressure_evictions == 0


def test_pressure_eviction_max_evict_bounds_a_single_tick():
    """``max_evict`` caps how many entries one tick can remove so a
    transient pressure spike does not wipe the entire prefix cache
    on one engine-loop tick."""
    cap = 10 * 1024 * 1024
    per_entry = 1 * 1024 * 1024
    stub_cache = _StubMemoryAwareCache(max_memory=cap, per_entry_bytes=per_entry)
    # Pin 50 entries so the loop has plenty to evict; cap is 10 MiB so
    # the pressure threshold (9 MiB) is well below current (50 MiB).
    for i in range(50):
        stub_cache.insert((i,))

    sched = _make_scheduler_with_stub_cache(stub_cache)

    with patch("vllm_mlx.scheduler.mx.clear_cache"):
        evicted = sched.evict_prefix_cache_under_pressure(max_evict=3)

    assert evicted == 3
    assert sched.num_prefix_cache_pressure_evictions == 3


def test_pressure_eviction_stops_when_cache_drops_below_threshold():
    """When the cache-self ledger drops below
    ``fraction × _max_memory`` mid-loop, eviction stops — avoids
    draining the whole cache on a borderline pressure tick.
    """
    cap = 10 * 1024 * 1024
    per_entry = 1 * 1024 * 1024
    stub_cache = _StubMemoryAwareCache(max_memory=cap, per_entry_bytes=per_entry)
    # 10 entries × 1 MiB = 10 MiB current. Threshold = 9 MiB. Need to
    # evict 2 entries to drop below the threshold (current → 8 MiB).
    for i in range(10):
        stub_cache.insert((i,))

    sched = _make_scheduler_with_stub_cache(stub_cache)

    with patch("vllm_mlx.scheduler.mx.clear_cache"):
        evicted = sched.evict_prefix_cache_under_pressure(max_evict=64)

    # The loop should stop as soon as current < 9 MiB. With per_entry=1
    # MiB starting at 10, that means exactly 2 evictions.
    assert evicted == 2
    assert sched.num_prefix_cache_pressure_evictions == 2
    assert stub_cache._current_memory == 8 * 1024 * 1024


def test_cache_self_pressure_respects_env_override(monkeypatch):
    """The env-var override is what configures the cache's
    ``_max_memory``. The cache-self-pressure threshold (and so the
    pressure counter) therefore tracks the env override — so an
    operator can drive eviction by lowering the env value alone,
    without touching ``gpu_memory_utilization``.
    """
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(8 * 1024 * 1024))

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    cfg = MemoryCacheConfig()
    cache = MemoryAwarePrefixCache(model=object(), config=cfg)
    # ``_max_memory`` is the env override (8 MiB).
    assert cache._max_memory == 8 * 1024 * 1024


# ─── Stats integration: get_stats wires through the counters ────────


def test_get_stats_surfaces_evictions(monkeypatch):
    """``MemoryAwarePrefixCache.get_stats`` must surface the
    ``evictions`` key in the shape that the metrics route reads.
    Locks in the contract the metrics serializer depends on so a
    refactor that renames the key trips this test instead of silently
    flat-lining the Prometheus series.
    """
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(8 * 1024 * 1024))

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    cache = MemoryAwarePrefixCache(model=object(), config=MemoryCacheConfig())

    # Insert past the cap so evictions fire.
    for i in range(5):
        cache.store(
            list(range(i * 100, i * 100 + 64)), _make_cache_entry(3 * 1024 * 1024)
        )

    stats = cache.get_stats()
    assert "evictions" in stats
    assert stats["evictions"] >= 1


# ─── R7-H7: cap admission MUST evict-LRU-until-fits, not reject-new ──


def test_r7_h7_near_full_cache_admits_fresh_inserts_via_lru_eviction(
    monkeypatch,
):
    """R7-H7 (dogfood-088 Talia r1/r2): when the cache is preloaded
    to near-cap, fresh prefix stores must STILL succeed by evicting
    LRU entries — not be rejected (``stored=False``) without freeing
    room. Talia's repro shape:

      1. Preload cache to 95% of cap (mimics ``load_from_disk``
         repopulating from a previous run).
      2. Insert 50 distinct fresh prefixes.
      3. Assert all 50 returned ``stored=True``.
      4. Assert ``evictions`` counter incremented.

    Pre-R7 (Talia's claim), the eviction path fired only on prefix-
    subset replacement (``evict_prefixes=True`` shrinks the cache
    when a new entry SUPERSETS an old one); cap-driven LRU eviction
    on a fresh-prefix insert was missing. The code review confirmed
    the LRU loop is present in ``store()`` lines 1168-1173 — this
    test pins the integration end-to-end so a refactor that moves the
    insert path can't silently lose the loop.

    8 MiB cap is small enough that the math is obvious: 1 MiB fake
    entries × ~7 = preload to 87% (close to "95% of cap" without
    burning fixture time on a long preload).
    """
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(8 * 1024 * 1024))

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    cache = MemoryAwarePrefixCache(model=object(), config=MemoryCacheConfig())

    per_entry_bytes = 1 * 1024 * 1024  # 1 MiB
    # Preload to ~87% (7 × 1 MiB out of 8 MiB cap).
    for i in range(7):
        cache.store(
            list(range(i * 100, i * 100 + 64)),
            _make_cache_entry(per_entry_bytes),
        )

    preload_stats = cache.get_stats()
    preload_evictions = preload_stats["evictions"]
    # Sanity: preload populated the cache.
    assert preload_stats["entry_count"] >= 5, preload_stats

    # Insert 50 distinct fresh prefixes. NONE may be rejected; LRU
    # eviction must free room for every one of them.
    rejected = 0
    accepted = 0
    for i in range(50):
        tokens = list(range(10_000 + i * 100, 10_000 + i * 100 + 64))
        ok = cache.store(tokens, _make_cache_entry(per_entry_bytes))
        if ok:
            accepted += 1
        else:
            rejected += 1

    final_stats = cache.get_stats()
    assert rejected == 0, (
        f"R7-H7 regression: {rejected}/50 fresh prefix stores rejected"
        f" without LRU eviction. The cap admission policy must"
        f" evict-LRU-until-fits, not reject-new. Final stats:"
        f" {final_stats!r}"
    )
    assert accepted == 50
    # At least one cap-driven eviction must have ticked the counter.
    new_evictions = final_stats["evictions"] - preload_evictions
    assert new_evictions >= 1, (
        f"R7-H7 regression: evictions counter did not tick across the"
        f" fresh-insert burst. preload_evictions={preload_evictions},"
        f" final_evictions={final_stats['evictions']}"
    )
    # Cap must hold — current memory cannot exceed the configured ceiling.
    assert final_stats["current_memory_mb"] <= final_stats["max_memory_mb"] + 0.5


def test_r7_h7_lru_ordering_least_recently_touched_evicted_first(
    monkeypatch,
):
    """R7-H7 LRU-ordering invariant: the entry evicted on cap pressure
    must be the LEAST recently touched (read OR write). A fetch hit
    must promote the entry to MRU so subsequent cap pressure evicts
    the older neighbours first.

    Strategy:
      1. Fill the cache to capacity with 3 entries.
      2. Fetch entry[0] (move it to MRU).
      3. Insert one new entry that requires eviction.
      4. Assert entry[1] (the new least-recently-touched) is gone but
         entry[0] (just fetched) and the new entry are still present.

    Pre-r7-D the LRU ordering wasn't tested explicitly — this test
    pins the contract so a refactor that swaps OrderedDict for a
    dict (loses insertion order) or skips the ``move_to_end`` on
    fetch silently broken-evicts the wrong entries.
    """
    # 3 MiB cap, 1 MiB per existing entry → 3 fit exactly at cap;
    # the 4th MUST force eviction. The 4th entry is the SAME 1 MiB
    # size as the others so the LRU loop only has to evict ONE entry
    # to make room.
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_MAX_BYTES", str(3 * 1024 * 1024))

    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    cache = MemoryAwarePrefixCache(model=object(), config=MemoryCacheConfig())

    # Three distinct prefixes whose token sequences DO NOT share a
    # common prefix — the ``evict_prefixes`` branch only removes
    # entries whose key is a strict prefix of the new key, so we
    # avoid that path here to isolate the LRU-on-cap behaviour. The
    # first token of each list must differ to skip the prefix sweep.
    tokens_a = [1, 100, 101, 102, 103]
    tokens_b = [2, 200, 201, 202, 203]
    tokens_c = [3, 300, 301, 302, 303]
    tokens_d = [4, 400, 401, 402, 403]

    per_entry_bytes = 1024 * 1024  # 1 MiB

    assert cache.store(tokens_a, _make_cache_entry(per_entry_bytes))
    assert cache.store(tokens_b, _make_cache_entry(per_entry_bytes))
    assert cache.store(tokens_c, _make_cache_entry(per_entry_bytes))
    # Cache is now AT cap (3 MiB / 3 MiB).

    # Touch A via fetch — must promote A to MRU. (Exact-match fetch
    # bumps via ``move_to_end`` inside fetch.)
    cache.fetch(tokens_a)

    # Insert D — this MUST evict the least-recently-touched entry
    # (B, since A was just fetched and C is newer than B). The
    # while-loop in store() should drop the LRU until room exists.
    assert cache.store(tokens_d, _make_cache_entry(per_entry_bytes))

    # Direct ledger inspection: keys present must be {A, C, D}.
    present_keys = set(cache._entries.keys())  # noqa: SLF001 — test asserts internals
    assert tuple(tokens_a) in present_keys, (
        f"LRU regression: just-fetched entry was evicted; present={present_keys}"
    )
    assert tuple(tokens_c) in present_keys, (
        f"LRU regression: middle-aged entry was evicted instead of LRU;"
        f" present={present_keys}"
    )
    assert tuple(tokens_d) in present_keys, (
        f"newly inserted entry missing from cache; present={present_keys}"
    )
    assert tuple(tokens_b) not in present_keys, (
        f"R7-H7 LRU ordering regression: the least-recently-touched"
        f" entry was NOT evicted. Present keys: {present_keys}."
        f" Expected B (tokens_b={tokens_b}) to be the eviction victim."
    )
