# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the compiled-matcher template cache (#558 perf follow-up).

``get_request_matcher`` must build the expensive ``LLMatcher`` automaton at most
once per distinct ``(tokenizer, grammar)`` and hand each caller its OWN matcher
via ``deep_copy`` — so repeated identical schemas skip the per-request automaton
construction WITHOUT sharing stateful parse cursors between requests. These tests
drive the cache with a fake ``LLMatcher`` (no model / no llguidance needed) so the
build-count, per-request isolation, broken-grammar bypass, and LRU bound are all
asserted deterministically.
"""

import pytest

import vllm_mlx.api.tool_grammar as tg


class _FakeMatcher:
    """Records how many templates were CONSTRUCTED vs deep-copied."""

    builds = 0

    def __init__(self, lltok, grammar):
        _FakeMatcher.builds += 1
        self.lltok = lltok
        self.grammar = grammar
        self.is_copy = False
        # A grammar carrying the BROKEN marker reports a compile error, mirroring
        # llguidance's never-raise "error is stored on the matcher" contract.
        self._error = "boom" if "BROKEN" in grammar else ""

    def get_error(self):
        return self._error

    def deep_copy(self):
        c = _FakeMatcher.__new__(_FakeMatcher)
        c.lltok = self.lltok
        c.grammar = self.grammar
        c._error = self._error
        c.is_copy = True
        return c


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch):
    monkeypatch.setattr(tg, "LLMatcher", _FakeMatcher)
    tg._compiled_matcher_cache.clear()
    _FakeMatcher.builds = 0
    yield
    tg._compiled_matcher_cache.clear()


def test_same_key_builds_template_once_and_returns_distinct_copies():
    lltok = object()
    g = "start: TAG\nTAG: /x/"
    m1 = tg.get_request_matcher(lltok, g)
    m2 = tg.get_request_matcher(lltok, g)
    # Automaton constructed exactly ONCE (the template); both requests got copies.
    assert _FakeMatcher.builds == 1
    assert m1.is_copy and m2.is_copy
    assert m1 is not m2  # per-request isolation — no shared parse cursor


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
