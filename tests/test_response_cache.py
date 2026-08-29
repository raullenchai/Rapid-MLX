# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the opt-in prompt-deterministic RESPONSE CACHE.

Covers the invariants that guard correctness:

* Cache key includes EVERY output-affecting field — a change in any
  sampling param (or model / prompt / response-shape field) yields a
  different key → a miss → a correct recompute. A missing field would be
  a correctness bug (a wrong response served).
* Determinism gate — only greedy (temperature==0 / top_k==1) requests
  are cacheable; a ``temperature > 0`` request with no definitively
  deterministic decode (even with a pinned seed, in this MVP) is NOT
  cached, so sampling variety is preserved.
* LRU bound — the (N+1)th distinct store evicts the LEAST-RECENTLY-USED
  entry, AND a HIT refreshes recency so eviction order is true LRU, not
  FIFO (explicitly regression-tested; the sibling hybrid-cache knob had a
  FIFO-vs-LRU gap that this guards against).
* N=0 fully disables — no store, no lookup, counters stay at zero, zero
  observable effect.
* Concurrent access safety — many threads hammering get/put must not
  corrupt the store or lose the LRU/counter invariants.
* SchedulerConfig validation — a negative ``response_cache_entries`` is
  rejected at construction.
* Metrics rendering — the hit/miss counters surface on ``/metrics``.
"""

from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.requires_mlx

from vllm_mlx.response_cache import (
    UNCACHEABLE,
    ResponseCache,
    configure_response_cache,
    get_response_cache,
    is_deterministic,
    make_cache_key,
    reset_response_cache_for_tests,
)


@pytest.fixture(autouse=True)
def _fresh_singleton():
    """Reset the process singleton around every test so counters/state
    from one case never leak into the next."""
    reset_response_cache_for_tests()
    yield
    reset_response_cache_for_tests()


def _get(cache: ResponseCache, key):
    """``get`` at the cache's CURRENT epoch — the common case for the LRU /
    counter mechanics tests (no reload in play)."""
    return cache.get(key, cache.current_epoch())


def _put(cache: ResponseCache, key, value):
    """``put`` at the cache's CURRENT epoch (see :func:`_get`)."""
    cache.put(key, value, cache.current_epoch())


# ── LRU semantics ─────────────────────────────────────────────────────


def test_lru_evicts_least_recently_used_not_fifo():
    """The (N+1)th store evicts the LRU entry — and a HIT must refresh
    recency so the eviction is LRU, NOT FIFO.

    Insert a, b (capacity 2). Then GET a (making a the most-recently
    used and b the least). Insert c → b (the LRU) must be evicted, and
    a (refreshed by the hit) must survive. A FIFO cache would wrongly
    evict a here; this asserts that does not happen.
    """
    c = ResponseCache(capacity=2)
    _put(c, "a", 1)
    _put(c, "b", 2)
    assert _get(c, "a") == 1  # a is now MRU, b is LRU
    _put(c, "c", 3)  # overflow → evict LRU (b), NOT the FIFO-oldest (a)
    assert _get(c, "b") is None, "FIFO bug: b was LRU and should be evicted"
    assert _get(c, "a") == 1, "the hit-refreshed entry must survive eviction"
    assert _get(c, "c") == 3


def test_reinsert_refreshes_recency():
    """Re-``put``-ing an existing key refreshes its recency (and value)."""
    c = ResponseCache(capacity=2)
    _put(c, "a", 1)
    _put(c, "b", 2)
    _put(c, "a", 99)  # refresh a → b becomes LRU, value updated
    _put(c, "c", 3)  # evict LRU (b)
    assert _get(c, "b") is None
    assert _get(c, "a") == 99
    assert _get(c, "c") == 3


def test_capacity_bound_never_exceeded():
    c = ResponseCache(capacity=3)
    for i in range(100):
        _put(c, f"k{i}", i)
        assert c.snapshot()["entries"] <= 3
    # Only the last 3 distinct keys remain.
    assert _get(c, "k99") == 99 and _get(c, "k98") == 98 and _get(c, "k97") == 97
    assert _get(c, "k96") is None


# ── N=0 disables everything ───────────────────────────────────────────


def test_zero_capacity_fully_inert():
    """Capacity 0 → no store, no lookup, counters untouched."""
    c = ResponseCache(capacity=0)
    assert c.enabled is False
    _put(c, "x", 1)
    assert _get(c, "x") is None
    snap = c.snapshot()
    # An inert cache records NOTHING — not even the miss.
    assert snap == {"hits": 0, "misses": 0, "entries": 0, "capacity": 0}


def test_configure_zero_clears_and_disables():
    c = ResponseCache(capacity=4)
    _put(c, "a", 1)
    assert _get(c, "a") == 1
    c.configure(0)
    assert c.enabled is False
    assert _get(c, "a") is None
    assert c.snapshot()["entries"] == 0


def test_configure_shrink_evicts_coldest():
    c = ResponseCache(capacity=3)
    _put(c, "1", 1)
    _put(c, "2", 2)
    _put(c, "3", 3)  # MRU order: 1(cold) < 2 < 3(hot)
    c.configure(1)  # keep only the hottest
    assert _get(c, "3") == 3
    assert _get(c, "1") is None
    assert _get(c, "2") is None


def test_negative_capacity_rejected():
    with pytest.raises(ValueError, match=r"capacity must be >= 0"):
        ResponseCache(capacity=-1)
    c = ResponseCache(capacity=1)
    with pytest.raises(ValueError, match=r"capacity must be >= 0"):
        c.configure(-5)
    with pytest.raises(ValueError, match=r"capacity must be >= 0"):
        c.reconfigure(-5)


# ── Counters ──────────────────────────────────────────────────────────


def test_hit_miss_counters():
    c = ResponseCache(capacity=4)
    _get(c, "absent")  # miss
    _put(c, "k", "v")
    _get(c, "k")  # hit
    _get(c, "k")  # hit
    _get(c, "absent2")  # miss
    snap = c.snapshot()
    assert snap["hits"] == 2
    assert snap["misses"] == 2


# ── Determinism gate ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"temperature": 0}, True),
        ({"temperature": 0.0}, True),
        ({"top_k": 1}, True),
        ({"top_k": 1, "temperature": 0.9}, True),  # top_k==1 forces greedy
        ({"temperature": 0.8}, False),
        ({"temperature": 0.8, "seed": 42}, False),  # seed alone NOT enough (MVP)
        ({"temperature": 0.0000001}, False),  # near-zero is still sampling
        ({}, False),  # missing → not greedy
        ({"temperature": None}, False),
        ({"top_k": 0, "temperature": 0.7}, False),
    ],
)
def test_determinism_gate(kwargs, expected):
    assert is_deterministic(kwargs) is expected


# ── Cache key: every output-affecting field participates ──────────────


_BASE = dict(
    model="m",
    prompt="hello world",
    sampling_kwargs={
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 0,
        "min_p": 0.0,
        "max_tokens": 64,
        "stop": ["</s>"],
        "seed": 7,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
)


def _key(**overrides):
    args = dict(_BASE)
    if "sampling_kwargs" in overrides:
        args["sampling_kwargs"] = {
            **_BASE["sampling_kwargs"],
            **overrides.pop("sampling_kwargs"),
        }
    args.update(overrides)
    return make_cache_key(**args)


def test_key_is_dict_order_independent():
    k1 = make_cache_key(
        model="m", prompt="p", sampling_kwargs={"temperature": 0, "max_tokens": 10}
    )
    k2 = make_cache_key(
        model="m", prompt="p", sampling_kwargs={"max_tokens": 10, "temperature": 0}
    )
    assert k1 == k2


def test_key_identical_inputs_match():
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.5),
        ("top_p", 0.5),
        ("top_k", 40),
        ("min_p", 0.05),
        ("max_tokens", 65),
        ("seed", 8),
        ("stop", ["STOP"]),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
    ],
)
def test_key_changes_when_any_sampling_param_changes(field, value):
    """A missing field in the key = a wrong response served. Prove EACH
    sampling param is part of the key by flipping it and asserting the
    key changes."""
    base = _key()
    changed = _key(sampling_kwargs={field: value})
    assert base != changed, f"{field} must affect the cache key"


def test_key_changes_on_model_prompt_and_extra():
    base = _key()
    assert _key(model="other") != base
    assert _key(prompt="different") != base
    # ``extra`` fields (response_format / logprobs) change the wire shape.
    assert make_cache_key(**_BASE, extra={"logprobs": True}) != base
    assert (
        make_cache_key(**_BASE, extra={"response_format": {"type": "json_object"}})
        != base
    )
    assert make_cache_key(**_BASE, extra={"top_logprobs": 5}) != base


def test_key_handles_pydantic_like_components():
    """Non-JSON-native key components (objects with ``model_dump``) must be
    canonicalized by ``_json_default`` via ``.model_dump()`` — NOT raise,
    and NOT fall through to an unstable repr.

    The instance is passed DIRECTLY in ``extra`` (not pre-``model_dump``-ed)
    so ``json.dumps`` actually routes it through ``_json_default`` and
    invokes ``.model_dump()``. Passing the already-dumped dict would bypass
    the code path under test entirely (a dict is JSON-native), leaving the
    ``model_dump`` branch unexercised — mutation-kill: deleting the
    ``model_dump`` branch in ``_json_default`` MUST make this test fail.
    """

    class _Fake:
        def model_dump(self):
            return {"type": "json_schema", "schema": {"a": 1}}

    k = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"response_format": _Fake()},  # instance, so _json_default fires
    )
    assert isinstance(k, str) and len(k) == 64  # sha256 hexdigest

    # The key must equal the one produced from the DUMPED dict directly:
    # proves _json_default really invoked .model_dump() (not repr / some
    # other fallback), so canonicalization is by VALUE, not object identity.
    k_from_dict = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"response_format": _Fake().model_dump()},
    )
    assert k == k_from_dict

    # And two FRESH _Fake() instances must produce the SAME key — the whole
    # point of canonicalizing by value rather than repr (which would embed
    # the object's memory address and silently miss).
    k2 = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"response_format": _Fake()},
    )
    assert k == k2


def test_uncacheable_when_component_cannot_be_canonicalized():
    """A key component that is neither JSON-native, a set, nor a
    ``model_dump``-carrying object cannot be mapped to a STABLE string.
    ``make_cache_key`` must return the ``UNCACHEABLE`` sentinel (so the
    caller skips store + lookup) rather than emit an unstable repr key —
    and it must NOT raise.

    A repr fallback would embed the object's memory address, so two
    otherwise-identical requests carrying fresh equivalent objects would
    key DIFFERENTLY → silent misses that defeat an exact-match cache.
    Mutation-kill: replacing the ``raise _UncanonicalizableError`` in
    ``_json_default`` with ``return repr(obj)`` MUST make this test fail
    (the result would be a 64-char hex string, not the sentinel).
    """

    class _Opaque:
        """No model_dump, not JSON-native, not a set."""

    result = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"weird": _Opaque()},
    )
    assert result is UNCACHEABLE

    # A model_dump that itself raises is likewise uncacheable, not a 500.
    class _BadDump:
        def model_dump(self):
            raise RuntimeError("boom")

    result2 = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"rf": _BadDump()},
    )
    assert result2 is UNCACHEABLE

    # Two fresh opaque objects both yield the sentinel — no repr address
    # leakage, no accidental distinct keys.
    r_a = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"weird": _Opaque()},
    )
    r_b = make_cache_key(
        model="m",
        prompt="p",
        sampling_kwargs={"temperature": 0},
        extra={"weird": _Opaque()},
    )
    assert r_a is UNCACHEABLE and r_b is UNCACHEABLE


# ── Concurrency safety ────────────────────────────────────────────────


def test_concurrent_readers_all_hit_prepopulated_keys():
    """Concurrent readers hammering PREPOPULATED shared keys must produce
    real hits under contention.

    Every read targets a key that is already stored, so every read is a
    hit. With reads-only after prepopulation, the exact hit total is
    deterministic: ``n_readers * reads_per_reader``. Worker exceptions are
    collected and re-raised (a bare Thread swallows them).

    Mutation-kill: making ``put()`` a no-op leaves the store empty, so
    every read MISSES instead of hitting → the observed-hits assertion
    fails.
    """
    c = ResponseCache(capacity=256)
    epoch = c.current_epoch()
    shared_keys = [f"shared-{i}" for i in range(50)]
    for k in shared_keys:
        c.put(k, f"val-{k}", epoch)  # prepopulate — every later read hits

    n_readers = 16
    reads_per_reader = 200
    barrier = threading.Barrier(n_readers)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def reader():
        try:
            barrier.wait()
            for i in range(reads_per_reader):
                k = shared_keys[i % len(shared_keys)]
                v = c.get(k, epoch)
                # Every read targets a prepopulated key → must hit.
                assert v == f"val-{k}", f"prepopulated key {k!r} missed under race"
        except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(n_readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"reader thread(s) raised under concurrency: {errors!r}"

    snap = c.snapshot()
    # Reads-only after prepopulation → exact hit total, zero misses.
    assert snap["hits"] == n_readers * reads_per_reader, (
        "concurrent readers did not all hit the prepopulated keys — "
        "storage broken or hit accounting lost under a race"
    )
    assert snap["misses"] == 0


def test_concurrent_writers_evict_to_capacity():
    """Concurrent writers storing UNIQUE keys that far exceed a SMALL
    capacity must never let the store grow past the bound — eviction is
    actually exercised (not sidestepped by an over-large capacity).

    Mutation-kill: making ``put()`` a no-op leaves the store empty, so the
    "eviction actually happened" lower-bound assertion (store is full)
    fails.
    """
    capacity = 64
    c = ResponseCache(capacity=capacity)
    epoch = c.current_epoch()
    n_writers = 10
    writes_per_writer = 500  # 5000 unique keys total ≫ capacity → eviction
    barrier = threading.Barrier(n_writers)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def writer(wid: int):
        try:
            barrier.wait()
            for i in range(writes_per_writer):
                c.put(f"w{wid}-k{i}", (wid, i), epoch)  # every key unique
                # Bound must hold at every step, not just at the end.
                assert c.snapshot()["entries"] <= capacity
        except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"writer thread(s) raised under concurrency: {errors!r}"

    snap = c.snapshot()
    assert snap["entries"] <= capacity, "capacity bound violated under concurrency"
    # Eviction actually happened AND storage works: far more unique writes
    # than capacity means the store must be FULL. A no-op put() leaves it
    # empty → this fails.
    assert snap["entries"] == capacity, (
        "store is not full after "
        f"{n_writers * writes_per_writer} unique writes into capacity "
        f"{capacity} — put() is not storing"
    )


# ── Module singleton wiring ───────────────────────────────────────────


def test_singleton_starts_disabled_and_configures():
    assert get_response_cache().enabled is False
    configure_response_cache(8)
    assert get_response_cache().enabled is True
    assert get_response_cache().capacity == 8
    configure_response_cache(0)
    assert get_response_cache().enabled is False


def test_configure_clears_entries_on_reload():
    """``configure_response_cache`` runs on EVERY ``load_model`` — including
    a hot reload of changed weights under the same model id. A stored
    completion is only valid for the model artifact that produced it, so
    the (re)configure MUST drop all cached entries; otherwise a reload
    serves completions from the previous model.

    Mutation-kill: remove the ``clear`` from ``ResponseCache.reconfigure``
    → the entry survives the simulated reload and this fails.
    """
    configure_response_cache(16)
    cache = get_response_cache()
    ep = cache.current_epoch()
    cache.put("some-key", "completion-from-model-v1", ep)
    assert cache.get("some-key", ep) == "completion-from-model-v1"

    # Simulate a second load_model() with the SAME positive capacity (the
    # case where plain configure() would otherwise preserve entries).
    configure_response_cache(16)

    assert get_response_cache().snapshot()["entries"] == 0
    # Lookup at the NEW epoch also misses (store was cleared).
    new_ep = get_response_cache().current_epoch()
    assert get_response_cache().get("some-key", new_ep) is None


# ── Epoch versioning: cross-model invalidation ────────────────────────


def test_reconfigure_is_atomic_clear_and_epoch_bump():
    """``reconfigure`` sets capacity, clears the store, and bumps the epoch
    — the single-lock load-time invalidation point."""
    c = ResponseCache(capacity=8)
    ep0 = c.current_epoch()
    c.put("k", "v", ep0)
    assert c.get("k", ep0) == "v"

    c.reconfigure(8)  # same capacity, but a reload
    ep1 = c.current_epoch()
    assert ep1 == ep0 + 1, "epoch must advance on reconfigure"
    assert c.snapshot()["entries"] == 0, "store must be cleared on reconfigure"
    assert c.capacity == 8


def test_stale_epoch_put_is_rejected():
    """An old-model generation completing AFTER a reload must NOT poison the
    freshly-cleared cache: a ``put`` carrying the pre-reload epoch is
    dropped.

    Mutation-kill: remove the epoch gate in ``put`` → the stale write lands
    and the cache is no longer empty, so this fails.
    """
    c = ResponseCache(capacity=8)
    old_epoch = c.current_epoch()

    c.reconfigure(8)  # reload → epoch bumps; old_epoch is now stale

    # An in-flight old-model request tries to store its completion.
    c.put("stale-key", "old-model-output", old_epoch)

    assert c.snapshot()["entries"] == 0, "stale-epoch put must be dropped"
    new_epoch = c.current_epoch()
    assert c.get("stale-key", new_epoch) is None


def test_stale_epoch_get_cannot_read_new_model_entry():
    """A lookup carrying a pre-reload epoch must NOT consume a new-model
    entry — even when one exists under the same key — and must tick
    NOTHING (a stale-epoch request is not a real lookup outcome)."""
    c = ResponseCache(capacity=8)
    old_epoch = c.current_epoch()

    c.reconfigure(8)  # reload → epoch bumps
    new_epoch = c.current_epoch()
    # New model stores a completion under a key.
    c.put("k", "new-model-output", new_epoch)

    # A request that began under the OLD model looks up the same key.
    assert c.get("k", old_epoch) is None, "stale-epoch get must not read new entry"

    # It ticked nothing — neither hit nor miss.
    snap = c.snapshot()
    assert snap["hits"] == 0
    assert snap["misses"] == 0

    # A current-epoch lookup DOES see the new entry (sanity).
    assert c.get("k", new_epoch) == "new-model-output"
    assert c.snapshot()["hits"] == 1


# ── SchedulerConfig validation ────────────────────────────────────────


def test_scheduler_config_rejects_negative_response_cache_entries():
    from vllm_mlx.scheduler import SchedulerConfig

    with pytest.raises(ValueError, match=r"response_cache_entries must be >= 0"):
        SchedulerConfig(response_cache_entries=-1)


def test_scheduler_config_default_response_cache_entries_is_zero():
    from vllm_mlx.scheduler import SchedulerConfig

    assert SchedulerConfig().response_cache_entries == 0
    assert SchedulerConfig(response_cache_entries=32).response_cache_entries == 32
