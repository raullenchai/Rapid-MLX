# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the compiled-matcher template cache (#558 perf follow-up).

``get_request_matcher`` must build the expensive ``LLMatcher`` automaton at most
once per distinct ``(tokenizer, grammar)`` and hand each caller its OWN matcher
via ``deep_copy`` — so repeated identical schemas skip the per-request automaton
construction WITHOUT sharing stateful parse cursors between requests. These tests
drive the cache with a fake ``LLMatcher`` (no model / no llguidance needed) so the
build-count, per-request isolation, broken-grammar bypass, and LRU bound are all
asserted deterministically.

The concurrency tests use an event-latch (NOT sleeps): the sole builder blocks in
``__init__`` until the test confirms — by instrumenting the under-lock
``_cache_hit_copy_locked`` entry — that every competing caller has committed
inside the single-flight critical section. Only then is the builder released, so
"burst builds once" holds by construction rather than by timing (codex #1155).
"""

import threading
from contextlib import contextmanager

import pytest

import vllm_mlx.api.tool_grammar as tg


class _FakeMatcher:
    """Records how many templates were CONSTRUCTED vs deep-copied."""

    builds = 0
    _builds_lock = threading.Lock()
    # The sole builder blocks here until a concurrency test releases it, so the
    # single-flight slot stays open until every competitor has committed to wait
    # on it (deterministic; no sleeps). Default set => non-concurrency tests
    # construct synchronously.
    proceed = threading.Event()
    proceed.set()

    def __init__(self, lltok, grammar):
        with _FakeMatcher._builds_lock:
            _FakeMatcher.builds += 1
        _FakeMatcher.proceed.wait()
        self.lltok = lltok
        self.grammar = grammar
        self.is_copy = False
        # A mutable parse cursor, so a test can prove that consuming on one
        # per-request copy does NOT advance any other copy or the cached template.
        self.consumed = 0
        # A grammar carrying the BROKEN marker reports a compile error, mirroring
        # llguidance's never-raise "error is stored on the matcher" contract.
        self._error = "boom" if "BROKEN" in grammar else ""

    def get_error(self):
        return self._error

    def consume_token(self, tok_id):
        self.consumed += 1
        return True

    def deep_copy(self):
        c = _FakeMatcher.__new__(_FakeMatcher)
        c.lltok = self.lltok
        c.grammar = self.grammar
        c._error = self._error
        c.consumed = self.consumed  # clone COPIES the cursor at copy time
        c.is_copy = True
        return c


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    monkeypatch.setattr(tg, "LLMatcher", _FakeMatcher)
    tg._compiled_matcher_cache.clear()
    tg._compiled_matcher_building.clear()
    tg._compiled_matcher_cache_bytes = 0
    _FakeMatcher.builds = 0
    _FakeMatcher.proceed.set()
    yield
    tg._compiled_matcher_cache.clear()
    tg._compiled_matcher_building.clear()
    tg._compiled_matcher_cache_bytes = 0
    _FakeMatcher.proceed.set()


@contextmanager
def _single_flight_latch(monkeypatch, n):
    """Hold the sole builder until all ``n`` callers commit in the critical path.

    ``_cache_hit_copy_locked`` runs UNDER ``_compiled_matcher_lock`` for every
    caller's first-pass (one thread registers the build slot; the rest read it as
    present and commit to WAIT on it). Counting those entries lets the test
    release the builder only once the whole burst is provably sharing one build —
    replacing the previous timing-dependent ``sleep`` with a deterministic latch.
    Yields a callable that blocks until all ``n`` have entered, then releases the
    builder.
    """
    entered = {"n": 0}
    cond = threading.Condition()
    real_hit = tg._cache_hit_copy_locked

    def _counting_hit(key):
        with cond:
            entered["n"] += 1
            cond.notify_all()
        return real_hit(key)

    monkeypatch.setattr(tg, "_cache_hit_copy_locked", _counting_hit)
    _FakeMatcher.proceed.clear()

    def _release_when_all_entered():
        with cond:
            if not cond.wait_for(lambda: entered["n"] >= n, timeout=5):
                raise AssertionError("not all workers entered the single-flight path")
        _FakeMatcher.proceed.set()

    try:
        yield _release_when_all_entered
    finally:
        _FakeMatcher.proceed.set()  # never wedge threads if an assertion fails


def _run_burst(n, target):
    """Start ``n`` worker threads that each call ``target()`` after a barrier.

    Captures per-worker exceptions (so a swallowed thread failure can't let the
    test pass on a subset) and returns ``(results, errors)`` with both threads
    joined (codex #1155).
    """
    barrier = threading.Barrier(n)
    results: list = []
    errors: list = []
    lock = threading.Lock()

    def worker():
        try:
            barrier.wait()
            m = target()
            with lock:
                results.append(m)
        except Exception as exc:  # noqa: BLE001 - record, never swallow silently
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    return threads, results, errors


def test_same_key_builds_template_once_and_returns_distinct_copies():
    lltok = object()
    g = "start: TAG\nTAG: /x/"
    m1 = tg.get_request_matcher(lltok, g)
    m2 = tg.get_request_matcher(lltok, g)
    # Automaton constructed exactly ONCE (the template); both requests got copies.
    assert _FakeMatcher.builds == 1
    assert m1.is_copy and m2.is_copy
    assert m1 is not m2  # per-request isolation — no shared parse cursor

    # Prove STATE isolation, not just object identity: consuming on m1 must not
    # advance m2's cursor NOR the cached template's (the template must stay at
    # its initial, never-consumed state so future clones start fresh).
    m1.consume_token(7)
    m1.consume_token(7)
    assert m1.consumed == 2
    assert m2.consumed == 0, "consuming on one copy must not advance another"
    cached_template = tg._compiled_matcher_cache[(id(lltok), g)][1]
    assert cached_template.consumed == 0, "the cached template must never be mutated"
    m3 = tg.get_request_matcher(lltok, g)
    assert m3.consumed == 0, "a later clone must start from the initial state"


def test_concurrent_cold_burst_builds_template_exactly_once(monkeypatch):
    # Per-key single-flight (codex #1155): a burst of N concurrent requests for
    # the SAME uncached key must construct the expensive automaton exactly ONCE,
    # not once-per-thread. The event-latch holds the sole builder until all N
    # callers have committed inside the critical section, so the collision is
    # forced deterministically (no sleep / no timing dependence).
    lltok = object()
    g = "start: HOT\nHOT: /h/"
    n = 8
    with _single_flight_latch(monkeypatch, n) as release:
        threads, results, errors = _run_burst(
            n, lambda: tg.get_request_matcher(lltok, g)
        )
        release()
        for t in threads:
            t.join(timeout=5)

    assert not errors, f"worker(s) raised: {errors}"
    assert _FakeMatcher.builds == 1, "single-flight must build the automaton once"
    assert len(results) == n, "every worker must have returned a matcher"
    assert all(m.is_copy for m in results), "every request gets its own deep_copy"
    assert len({id(m) for m in results}) == n, "no two requests share a matcher"


def test_concurrent_burst_of_broken_grammar_builds_once(monkeypatch):
    # codex #1155: a concurrent burst of the SAME broken grammar must not
    # serialize into N compilations. The builder publishes the (inert) broken
    # matcher via the _BuildSlot; waiters share it instead of rebuilding. Because
    # broken results are UNCACHED, this "builds once" property is guaranteed only
    # while every waiter has committed to the slot before the builder retires it —
    # which the latch enforces deterministically (was a flaky 50 ms sleep before).
    lltok = object()
    g = "start: BROKEN"
    n = 6
    with _single_flight_latch(monkeypatch, n) as release:
        threads, results, errors = _run_burst(
            n, lambda: tg.get_request_matcher(lltok, g)
        )
        release()
        for t in threads:
            t.join(timeout=5)

    assert not errors, f"worker(s) raised: {errors}"
    assert len(results) == n, "every worker must have returned a matcher"
    assert _FakeMatcher.builds == 1, "broken grammar burst must compile once"
    assert all(m.get_error() for m in results), "all requests see the compile error"
    assert (id(lltok), g) not in tg._compiled_matcher_cache  # broken stays uncached


def test_distinct_grammars_build_distinct_templates():
    lltok = object()
    tg.get_request_matcher(lltok, "start: A\nA: /a/")
    tg.get_request_matcher(lltok, "start: B\nB: /b/")
    assert _FakeMatcher.builds == 2


def test_distinct_tokenizers_do_not_share_a_template():
    g = "start: A\nA: /a/"
    tg.get_request_matcher(object(), g)
    tg.get_request_matcher(object(), g)
    # Same grammar, different tokenizer identity -> vocab-specific automaton
    # must be rebuilt (never share a compiled matcher across tokenizers).
    assert _FakeMatcher.builds == 2


def test_broken_grammar_is_not_cached():
    lltok = object()
    g = "start: BROKEN"
    m1 = tg.get_request_matcher(lltok, g)
    m2 = tg.get_request_matcher(lltok, g)
    # Broken template returned as-is (uncached) each time, so is_broken() handling
    # in GrammarLogitsProcessor is unchanged and no bad template poisons the cache.
    assert m1.get_error() and m2.get_error()
    assert not m1.is_copy  # returned directly, not a deep_copy of a cached template
    assert _FakeMatcher.builds == 2
    assert (id(lltok), g) not in tg._compiled_matcher_cache


def test_cache_is_bounded_lru(monkeypatch):
    monkeypatch.setattr(tg, "_COMPILED_MATCHER_CACHE_MAX", 4)
    lltok = object()
    for i in range(10):
        tg.get_request_matcher(lltok, f"start: R{i}\nR{i}: /{i}/")
    assert len(tg._compiled_matcher_cache) <= 4


def test_eviction_is_recency_ordered_not_fifo(monkeypatch):
    # Prove LRU (not FIFO / arbitrary): fill to capacity, TOUCH the oldest key so
    # it becomes most-recently-used, then insert one past capacity. The evicted
    # entry must be the now-least-recently-used key, NOT the touched one.
    monkeypatch.setattr(tg, "_COMPILED_MATCHER_CACHE_MAX", 3)
    lltok = object()
    g = [f"start: K{i}\nK{i}: /{i}/" for i in range(4)]
    for i in range(3):  # fill: K0 (oldest) .. K2 (newest)
        tg.get_request_matcher(lltok, g[i])
    # Touch K0 -> it becomes most-recently-used; K1 is now the LRU victim.
    tg.get_request_matcher(lltok, g[0])
    # Insert K3, forcing one eviction.
    tg.get_request_matcher(lltok, g[3])
    keys = {k[1] for k in tg._compiled_matcher_cache}
    assert g[0] in keys, "touched (recently-used) key must survive — this is LRU"
    assert g[1] not in keys, "the least-recently-used key must be the one evicted"
    assert g[3] in keys, "the newest inserted key must be present"


def test_oversized_grammar_is_not_cached(monkeypatch):
    # A single grammar larger than the whole byte budget must NOT be cached (it
    # would evict everything and still overflow); it is served fresh each call.
    monkeypatch.setattr(tg, "_COMPILED_MATCHER_CACHE_MAX_BYTES", 64)
    lltok = object()
    big = "start: BIG\nBIG: /" + ("x" * 200) + "/"
    m1 = tg.get_request_matcher(lltok, big)
    m2 = tg.get_request_matcher(lltok, big)
    assert m1.is_copy and m2.is_copy  # still a usable per-request matcher
    assert (id(lltok), big) not in tg._compiled_matcher_cache
    assert _FakeMatcher.builds == 2  # rebuilt each time (uncached)


def test_byte_budget_counts_utf8_bytes_not_code_points(monkeypatch):
    # A non-ASCII grammar's UTF-8 size exceeds its code-point count, so budgeting
    # must use encoded bytes: a grammar of few code points but many UTF-8 bytes
    # can exceed the budget and be refused caching (codex #1155).
    monkeypatch.setattr(tg, "_COMPILED_MATCHER_CACHE_MAX_BYTES", 40)
    lltok = object()
    # 20 code points, each a 3-byte CJK char = 60 UTF-8 bytes > 40 budget.
    g = "描" * 20
    assert len(g) == 20 and len(g.encode("utf-8")) == 60
    m = tg.get_request_matcher(lltok, g)
    assert m.is_copy  # usable matcher returned
    # Refused caching because its UTF-8 byte size (60) exceeds the 40 budget —
    # would be wrongly cached if the code-point count (20) were used.
    assert (id(lltok), g) not in tg._compiled_matcher_cache


def test_byte_budget_evicts_before_count_cap(monkeypatch):
    # With a generous count cap but a tight byte budget, entries evict on BYTES.
    monkeypatch.setattr(tg, "_COMPILED_MATCHER_CACHE_MAX", 100)
    monkeypatch.setattr(tg, "_COMPILED_MATCHER_CACHE_MAX_BYTES", 120)
    lltok = object()
    for i in range(6):
        tg.get_request_matcher(lltok, f"g{i}:" + ("y" * 40))  # ~44 bytes each
    assert tg._compiled_matcher_cache_bytes <= 120
    assert len(tg._compiled_matcher_cache) < 6  # byte budget bound before count
