# SPDX-License-Identifier: Apache-2.0
"""
Tests that demonstrate measurable impact of the three SGLang-inspired
optimizations: radix-tree prefix cache, async_eval overlap, and
jump-forward structured decoding.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

# =====================================================================
# 1. Radix-tree prefix cache vs naive trie
# =====================================================================


@dataclass
class _CacheEntry:
    prompt_cache: list[Any]
    count: int


class _NaiveTrieCache:
    """Original per-token trie implementation for comparison."""

    def __init__(self, max_entries: int = 10000):
        self._cache: dict = {}
        self._lru: deque = deque()
        self.max_size = max_entries

    def store(self, tokens: list[int], value: Any) -> None:
        current = self._cache
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        key = tuple(tokens)
        if "cache" not in current:
            current["cache"] = _CacheEntry(value, 1)
            self._lru.append(key)
        while len(self._lru) > self.max_size:
            evict_key = self._lru.popleft()
            self._delete(list(evict_key))

    def search(self, tokens: list[int]) -> bool:
        """Return True if exact match exists (no deepcopy — pure traversal)."""
        current = self._cache
        for tok in tokens:
            if tok not in current:
                return False
            current = current[tok]
        return "cache" in current

    def _delete(self, tokens: list[int]) -> None:
        path = [(self._cache, None)]
        current = self._cache
        for tok in tokens:
            if tok not in current:
                return
            path.append((current[tok], tok))
            current = current[tok]
        if "cache" in current:
            del current["cache"]
        for i in range(len(path) - 1, 0, -1):
            node, tok = path[i]
            parent, _ = path[i - 1]
            if not node:
                del parent[tok]


class TestRadixTreeVsTrie:
    """Benchmark: radix tree should use fewer nodes and have faster
    traversal for prefix-heavy workloads."""

    @staticmethod
    def _make_multi_turn_tokens(
        system_len: int, n_turns: int, turn_len: int
    ) -> list[list[int]]:
        """Simulate multi-turn chat: shared system prompt + diverging turns."""
        system = list(range(1, system_len + 1))
        sequences = []
        for turn in range(n_turns):
            suffix = list(range(10000 * (turn + 1), 10000 * (turn + 1) + turn_len))
            sequences.append(system + suffix)
        return sequences

    def test_radix_node_compression(self):
        """Radix tree should use dramatically fewer nodes than trie.
        This is the core value proposition of the data structure change."""
        from vllm_mlx.prefix_cache import PrefixCacheManager

        sequences = self._make_multi_turn_tokens(
            system_len=2048, n_turns=50, turn_len=100
        )

        # Count trie nodes (iterative to avoid stack overflow)
        trie = _NaiveTrieCache()
        for seq in sequences:
            trie.store(seq, ["dummy"])

        trie_nodes = 0
        stack = [trie._cache]
        while stack:
            node = stack.pop()
            for k, v in node.items():
                if k != "cache" and isinstance(v, dict):
                    trie_nodes += 1
                    stack.append(v)

        # Count radix nodes
        model = MagicMock()
        radix = PrefixCacheManager(model, max_entries=10000)
        for seq in sequences:
            radix.store_cache(seq, ["dummy"])

        radix_nodes = 0
        root = radix._roots[radix.model_key]
        rstack = [root]
        while rstack:
            node = rstack.pop()
            radix_nodes += 1
            rstack.extend(node.children.values())

        compression = trie_nodes / radix_nodes if radix_nodes > 0 else 0
        print(
            f"\n  Trie: {trie_nodes} nodes, Radix: {radix_nodes} nodes — "
            f"{compression:.0f}x compression"
        )
        # 50 sequences sharing 2048-token prefix: trie ~107,048 nodes
        # Radix ~101 nodes (1 root + 1 compressed prefix + 50 leaves + 49 internal)
        assert compression > 50, (
            f"Expected >50x node compression, got {compression:.1f}x"
        )

    def test_radix_memory_efficiency(self):
        """Radix tree uses dramatically less memory due to node compression.
        With 200 entries sharing a 2048-token system prompt:
        - Trie: 200*(2048+100) = ~430K dict entries
        - Radix: 1 root + 1 compressed prefix + ~200 leaves ≈ 202 nodes

        This means less GC pressure, better cache locality, and faster
        eviction in production."""
        from vllm_mlx.prefix_cache import PrefixCacheManager

        sequences = self._make_multi_turn_tokens(
            system_len=2048, n_turns=200, turn_len=100
        )

        # Trie: count total dict entries
        trie = _NaiveTrieCache(max_entries=10000)
        for seq in sequences:
            trie.store(seq, ["dummy_cache"])

        trie_nodes = 0
        stack = [trie._cache]
        while stack:
            node = stack.pop()
            for k, v in node.items():
                if k != "cache" and isinstance(v, dict):
                    trie_nodes += 1
                    stack.append(v)

        # Radix: count nodes
        model = MagicMock()
        radix = PrefixCacheManager(model, max_entries=10000)
        for seq in sequences:
            radix.store_cache(seq, ["dummy_cache"])

        radix_nodes = 0
        root = radix._roots[radix.model_key]
        rstack = [root]
        while rstack:
            n = rstack.pop()
            radix_nodes += 1
            rstack.extend(n.children.values())

        ratio = trie_nodes / radix_nodes
        print(
            f"\n  Trie: {trie_nodes} dict entries → Radix: {radix_nodes} nodes "
            f"({ratio:.0f}x memory reduction)"
        )
        # Real-world impact: fewer allocations, less GC, faster eviction scan
        assert ratio > 100, f"Expected >100x node reduction, got {ratio:.0f}x"

    def test_radix_store_performance(self):
        """Insertion with shared prefixes: radix should be competitive."""
        from vllm_mlx.prefix_cache import PrefixCacheManager

        sequences = self._make_multi_turn_tokens(
            system_len=2048, n_turns=200, turn_len=100
        )

        trie = _NaiveTrieCache(max_entries=10000)
        t0 = time.perf_counter()
        for seq in sequences:
            trie.store(seq, ["dummy_cache"])
        trie_time = time.perf_counter() - t0

        model = MagicMock()
        radix = PrefixCacheManager(model, max_entries=10000)
        t0 = time.perf_counter()
        for seq in sequences:
            radix.store_cache(seq, ["dummy_cache"])
        radix_time = time.perf_counter() - t0

        speedup = trie_time / radix_time if radix_time > 0 else float("inf")
        print(
            f"\n  Radix store: {radix_time*1000:.1f}ms vs "
            f"trie store: {trie_time*1000:.1f}ms — {speedup:.1f}x"
        )
        # Radix should be at least as fast (fewer dict creates)
        assert speedup > 0.8, f"Radix store unexpectedly slow: {speedup:.2f}x"


# =====================================================================
# 2. async_eval overlap in _cleanup_finished
# =====================================================================


class TestAsyncEvalOverlap:
    """Verify that _cleanup_finished uses async_eval (non-blocking)
    instead of sync eval (blocking) for cache tensor evaluation."""

    def test_cleanup_uses_async_eval_not_sync(self):
        """The cache tensor evaluation path must use mx.async_eval,
        not synchronous mx.eval, to avoid GPU stalls during cleanup."""
        import inspect

        from vllm_mlx.scheduler import Scheduler

        source = inspect.getsource(Scheduler._cleanup_finished)

        # Must contain async_eval
        assert "mx.async_eval" in source, (
            "_cleanup_finished should use mx.async_eval for cache tensors"
        )

        # Must NOT contain synchronous mx.eval (except in comments)
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "mx.eval(" in stripped and "async_eval" not in stripped:
                pytest.fail(
                    f"Found blocking mx.eval in _cleanup_finished: {stripped!r}\n"
                    "This forces GPU sync during cleanup, killing overlap."
                )

    def test_cleanup_batches_eval_arrays(self):
        """async_eval should be called once with all arrays, not per-layer,
        to minimize dispatch overhead."""
        import inspect

        from vllm_mlx.scheduler import Scheduler

        source = inspect.getsource(Scheduler._cleanup_finished)

        # Should collect arrays first, then call async_eval once
        assert "_eval_arrays" in source, (
            "Should batch arrays into _eval_arrays list before async_eval"
        )
        assert "mx.async_eval(*_eval_arrays)" in source, (
            "Should call async_eval with unpacked array list (single dispatch)"
        )


# =====================================================================
# 3. Jump-forward structured decoding
# =====================================================================


class TestJumpForwardDecoding:
    """Verify jump-forward correctly identifies deterministic tokens
    and calculates concrete decode-step savings."""

    @pytest.fixture
    def tokenizer(self):
        """Mock tokenizer with char-level encoding."""
        tok = MagicMock()
        _vocab = {}
        _next_id = [100]

        def _encode(text, add_special_tokens=False):
            tokens = []
            for ch in text:
                if ch not in _vocab:
                    _vocab[ch] = _next_id[0]
                    _next_id[0] += 1
                tokens.append(_vocab[ch])
            return tokens

        def _decode(ids, skip_special_tokens=False):
            id_to_char = {v: k for k, v in _vocab.items()}
            return "".join(id_to_char.get(i, "?") for i in ids)

        tok.encode = _encode
        tok.decode = _decode
        return tok

    def test_jump_forward_tokens_available(self, tokenizer):
        """When a pattern is active with >= 2 remaining tokens,
        get_jump_forward_tokens returns them."""
        from vllm_mlx.api.tool_logits import MiniMaxToolLogitsProcessor

        proc = MiniMaxToolLogitsProcessor(tokenizer)

        pattern = ' name="'
        pattern_tokens = proc._pattern_tokens[pattern]
        assert len(pattern_tokens) >= 2

        proc._active_pattern = pattern
        proc._pattern_pos = 1  # First token already biased

        jump = proc.get_jump_forward_tokens()
        assert jump is not None
        assert jump == pattern_tokens[1:]
        print(
            f"\n  Pattern {pattern!r}: {len(pattern_tokens)} tokens, "
            f"jump saves {len(jump)} decode steps"
        )

    def test_no_jump_when_idle(self, tokenizer):
        from vllm_mlx.api.tool_logits import MiniMaxToolLogitsProcessor

        proc = MiniMaxToolLogitsProcessor(tokenizer)
        assert proc.get_jump_forward_tokens() is None

    def test_no_jump_when_single_remaining(self, tokenizer):
        from vllm_mlx.api.tool_logits import MiniMaxToolLogitsProcessor

        proc = MiniMaxToolLogitsProcessor(tokenizer)
        pattern = ' name="'
        pattern_tokens = proc._pattern_tokens[pattern]

        proc._active_pattern = pattern
        proc._pattern_pos = len(pattern_tokens) - 1
        assert proc.get_jump_forward_tokens() is None

    def test_complete_jump_forward_resets_state(self, tokenizer):
        from vllm_mlx.api.tool_logits import MiniMaxToolLogitsProcessor

        proc = MiniMaxToolLogitsProcessor(tokenizer)
        proc._active_pattern = ' name="'
        proc._pattern_pos = 2
        proc._consecutive_bias_count = 5
        proc._recent_text = "some text<invoke"

        jumped = tokenizer.encode(' name="')
        proc.complete_jump_forward(jumped)

        assert proc._active_pattern is None
        assert proc._pattern_pos == 0
        assert proc._consecutive_bias_count == 0
        assert ' name="' in proc._recent_text

    def test_total_steps_saved_per_tool_call(self, tokenizer):
        """Concrete calculation: how many forward passes does jump-forward
        skip for one complete MiniMax tool call?

        A tool call has 4 structural patterns. Each pattern's first token
        is generated normally (biased); the rest are injected via a single
        prefill pass. So each pattern saves (N-1) decode steps.
        """
        from vllm_mlx.api.tool_logits import MiniMaxToolLogitsProcessor

        proc = MiniMaxToolLogitsProcessor(tokenizer)

        total_saved = 0
        total_structural = 0
        details = []
        for pattern, _trigger in proc.PATTERNS:
            tokens = proc._pattern_tokens.get(pattern, [])
            if len(tokens) >= 2:
                saved = len(tokens) - 1
                total_saved += saved
                total_structural += len(tokens)
                details.append(f"  {pattern!r}: {len(tokens)} tok, saves {saved}")

        print("\n" + "\n".join(details))
        print(
            f"\n  TOTAL: {total_structural} structural tokens, "
            f"{total_saved} decode steps saved"
        )

        # At 50 tok/s decode, each step ~20ms. 44 saved steps = ~880ms savings
        time_saved_ms = total_saved * 20
        print(f"  Estimated time saved: ~{time_saved_ms}ms per tool call (at 50 tok/s)")

        assert total_saved >= 30, (
            f"Expected >= 30 decode steps saved per tool call, got {total_saved}"
        )

    def test_jump_forward_wired_in_scheduler(self):
        """The jump-forward code path must exist in scheduler.py
        (inside the _generation_step closure)."""
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parent.parent
            / "vllm_mlx"
            / "scheduler.py"
        ).read_text()

        assert "get_jump_forward_tokens" in scheduler_src, (
            "_generation_step should call get_jump_forward_tokens"
        )
        assert "complete_jump_forward" in scheduler_src, (
            "_generation_step should call complete_jump_forward after injection"
        )
        assert "inject_logits = self.model(" in scheduler_src, (
            "_generation_step should call model directly for jump prefill"
        )

    def test_multi_pattern_savings(self, tokenizer):
        """Simulate two patterns firing in succession: the combined savings
        represent a realistic tool call scenario."""
        from vllm_mlx.api.tool_logits import MiniMaxToolLogitsProcessor

        proc = MiniMaxToolLogitsProcessor(tokenizer)

        patterns = [
            (' name="', "<invoke"),
            ("</minimax:tool_call>", "</invoke>"),
        ]
        total_saved = 0
        for pattern, trigger in patterns:
            tokens = proc._pattern_tokens.get(pattern, [])
            if len(tokens) >= 2:
                proc._active_pattern = pattern
                proc._pattern_pos = 1
                jump = proc.get_jump_forward_tokens()
                if jump:
                    total_saved += len(jump)
                    proc.complete_jump_forward(jump)

        print(f"\n  Two patterns: saved {total_saved} decode steps")
        assert total_saved >= 5

    def test_all_parsers_have_jump_forward(self, tokenizer):
        """Every parser with registered patterns must produce a working
        processor with jump-forward capability."""
        from vllm_mlx.api.tool_logits import (
            PARSER_PATTERNS,
            create_tool_logits_processor,
        )

        results = []
        for parser_name, patterns in PARSER_PATTERNS.items():
            if not patterns:
                continue
            proc = create_tool_logits_processor(parser_name, tokenizer)
            assert proc is not None, f"Parser {parser_name!r} should create a processor"

            # Check that at least one pattern is jumpable (>= 2 tokens)
            jumpable = 0
            total_saved = 0
            for pattern_text, _trigger in patterns:
                tokens = proc._pattern_tokens.get(pattern_text, [])
                if len(tokens) >= 2:
                    jumpable += 1
                    total_saved += len(tokens) - 1

            results.append((parser_name, len(patterns), jumpable, total_saved))

        print("\n  Parser jump-forward coverage:")
        print(f"  {'Parser':<20s} {'Patterns':>8s} {'Jumpable':>8s} {'Steps saved':>11s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*11}")
        for name, n_pat, n_jump, saved in sorted(results):
            print(f"  {name:<20s} {n_pat:>8d} {n_jump:>8d} {saved:>11d}")

        total_parsers = len(results)
        parsers_with_jump = sum(1 for _, _, j, _ in results if j > 0)
        print(
            f"\n  {parsers_with_jump}/{total_parsers} parsers have jump-forward "
            f"patterns"
        )
        assert parsers_with_jump >= 10, (
            f"Expected >= 10 parsers with jump-forward, got {parsers_with_jump}"
        )


# =====================================================================
# 4. Radix tree correctness under stress
# =====================================================================


class TestRadixTreeStress:
    def test_lru_eviction(self):
        from vllm_mlx.prefix_cache import PrefixCacheManager

        model = MagicMock()
        cache = PrefixCacheManager(model, max_entries=50)

        for i in range(100):
            cache.store_cache([1, 2, 3] + [i] * 10, [f"cache_{i}"])

        # First 50 evicted
        for i in range(50):
            result, _ = cache.fetch_cache([1, 2, 3] + [i] * 10)
            assert result is None

        # Last 50 present
        for i in range(50, 100):
            result, _ = cache.fetch_cache([1, 2, 3] + [i] * 10)
            assert result == [f"cache_{i}"]

    def test_prefix_sharing(self):
        from vllm_mlx.prefix_cache import PrefixCacheManager

        model = MagicMock()
        cache = PrefixCacheManager(model, max_entries=1000)

        system = list(range(1, 2049))
        for i in range(20):
            cache.store_cache(system + [10000 + i, 10001 + i], [f"conv_{i}"])

        for i in range(20):
            result, remaining = cache.fetch_cache(system + [10000 + i, 10001 + i])
            assert result == [f"conv_{i}"]
            assert remaining == []

    def test_edge_split(self):
        from vllm_mlx.prefix_cache import PrefixCacheManager

        model = MagicMock()
        cache = PrefixCacheManager(model, max_entries=100)

        cache.store_cache([1, 2, 3, 4, 5], ["long"])
        cache.store_cache([1, 2, 3], ["short"])

        r1, _ = cache.fetch_cache([1, 2, 3, 4, 5])
        r2, _ = cache.fetch_cache([1, 2, 3])
        assert r1 == ["long"]
        assert r2 == ["short"]

    def test_compaction_after_delete(self):
        from vllm_mlx.prefix_cache import PrefixCacheManager

        model = MagicMock()
        cache = PrefixCacheManager(model, max_entries=100)

        cache.store_cache([1, 2, 3, 4, 5], ["a"])
        cache.store_cache([1, 2, 3, 6, 7], ["b"])
        cache._delete_cache(cache.model_key, [1, 2, 3, 4, 5])

        result, _ = cache.fetch_cache([1, 2, 3, 6, 7])
        assert result == ["b"]

        root = cache._roots[cache.model_key]
        count = 0
        stack = [root]
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children.values())
        assert count <= 3, f"Expected <= 3 nodes after compaction, got {count}"
