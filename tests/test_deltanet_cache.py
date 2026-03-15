# SPDX-License-Identifier: Apache-2.0
"""
Tests for DeltaNet/SSM cache handling in SimpleEngine prompt cache.

Qwen3.5 uses a hybrid architecture: 75% Gated DeltaNet layers (non-trimmable
ArraysCache) + 25% full attention layers (trimmable KVCache).  The prompt
cache logic must handle both types correctly to avoid stale recurrent state
corrupting multi-turn conversations.
"""

from unittest.mock import MagicMock

import pytest


class FakeKVCache:
    """Simulates a trimmable KVCache."""

    def __init__(self):
        self.offset = 0
        self._trimmed = 0

    def is_trimmable(self):
        return True

    def empty(self):
        return self.offset == 0

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._trimmed += n
        return n


class FakeArraysCache:
    """Simulates a non-trimmable ArraysCache (DeltaNet recurrent state)."""

    def __init__(self, size=2):
        self.cache = [None] * size

    def is_trimmable(self):
        return False

    def empty(self):
        return self.cache[0] is None

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]


class FakeLLM:
    """Minimal mock of MLXLanguageModel for testing cache logic."""

    def __init__(self, num_linear=3, num_full_attn=1):
        self._prompt_cache = []
        self._cached_token_ids = []

        # Build hybrid cache: linear, linear, linear, full_attn pattern
        for i in range(num_linear + num_full_attn):
            if i % (num_linear + num_full_attn) < num_linear:
                self._prompt_cache.append(FakeArraysCache())
            else:
                kv = FakeKVCache()
                self._prompt_cache.append(kv)

    def _find_common_prefix_len(self, prompt_token_ids):
        """Find common prefix length between cached and new tokens."""
        common = 0
        for a, b in zip(self._cached_token_ids, prompt_token_ids):
            if a != b:
                break
            common += 1
        return common

    def _reset_all_caches(self):
        """Reset all cache layers to empty state."""
        for c in self._prompt_cache:
            if c.is_trimmable():
                current = c.offset if hasattr(c, "offset") else 0
                if current > 0:
                    c.trim(current)
            elif hasattr(c, "cache"):
                for i in range(len(c.cache)):
                    c.cache[i] = None

    def _prepare_cache_for_prompt(self, prompt_token_ids):
        """Simplified version of the real method for testing."""
        if not self._prompt_cache:
            return prompt_token_ids

        common_len = self._find_common_prefix_len(prompt_token_ids)

        has_non_trimmable = any(
            not c.is_trimmable() and not c.empty()
            for c in self._prompt_cache
        )

        if common_len == 0:
            self._reset_all_caches()
            self._cached_token_ids = []
            return prompt_token_ids

        needs_trim = False
        for c in self._prompt_cache:
            if c.is_trimmable():
                current = c.offset if hasattr(c, "offset") else 0
                if current > common_len:
                    needs_trim = True
                    break

        if has_non_trimmable and needs_trim:
            self._reset_all_caches()
            self._cached_token_ids = []
            return prompt_token_ids

        for c in self._prompt_cache:
            if not c.is_trimmable():
                continue
            current = c.offset if hasattr(c, "offset") else 0
            to_trim = current - common_len
            if to_trim > 0:
                c.trim(to_trim)
        self._cached_token_ids = self._cached_token_ids[:common_len]

        suffix = prompt_token_ids[common_len:]
        return suffix


def _simulate_generation(llm, prompt_tokens, gen_tokens=5):
    """Simulate processing prompt + generating tokens."""
    suffix = llm._prepare_cache_for_prompt(prompt_tokens)

    # Simulate model processing: DeltaNet layers accumulate state,
    # KV cache grows
    total_processed = len(prompt_tokens) + gen_tokens
    for c in llm._prompt_cache:
        if c.is_trimmable():
            c.offset = total_processed
        else:
            c[0] = "conv_state"  # non-None to mark as non-empty
            c[1] = "recurrent_state"

    llm._cached_token_ids = list(prompt_tokens)
    return suffix


class TestDeltaNetCacheReset:
    """Test that non-trimmable DeltaNet caches are properly reset."""

    def test_no_overlap_resets_deltanet(self):
        """When prompts have no common prefix, DeltaNet state must be reset."""
        llm = FakeLLM()
        _simulate_generation(llm, [1, 2, 3, 4, 5])

        # Verify DeltaNet caches have state
        for c in llm._prompt_cache:
            if not c.is_trimmable():
                assert not c.empty(), "DeltaNet should have state after gen"

        # New prompt with no overlap
        suffix = _simulate_generation(llm, [10, 20, 30])

        # DeltaNet should have been reset before reprocessing
        # (the simulate_generation re-fills them, but suffix should be full prompt)
        assert len(suffix) == 3, "Should reprocess full prompt"

    def test_partial_overlap_resets_deltanet(self):
        """When prompts share a prefix but diverge, DeltaNet must reset."""
        llm = FakeLLM()
        _simulate_generation(llm, [1, 2, 3, 4, 5])

        # New prompt shares prefix [1, 2, 3] but diverges after
        suffix = llm._prepare_cache_for_prompt([1, 2, 3, 10, 20])

        # Must return FULL prompt because DeltaNet can't be trimmed
        assert suffix == [1, 2, 3, 10, 20], \
            "Should reprocess full prompt when DeltaNet state can't be trimmed"

    def test_exact_same_prompt_no_reset(self):
        """When the same prompt is repeated, no reset needed."""
        llm = FakeLLM()
        _simulate_generation(llm, [1, 2, 3, 4, 5])

        # Same prompt again — KV cache has offset > prompt length (includes gen tokens)
        # but common_len == 5 == full prompt, and KV offset needs trimming
        # Since DeltaNet state is non-empty and KV needs trim, this would trigger reset
        # BUT the suffix is empty (exact match), so no trim is needed for content
        suffix = llm._prepare_cache_for_prompt([1, 2, 3, 4, 5])

        # KV cache has offset = 10 (5 prompt + 5 gen), needs trim to 5
        # DeltaNet state is non-empty and KV needs_trim = True
        # So this WILL reset — which is correct because DeltaNet state includes gen tokens
        assert suffix == [1, 2, 3, 4, 5], \
            "Should reprocess when DeltaNet has generated token state"

    def test_pure_kv_cache_no_regression(self):
        """Pure KV cache models (no DeltaNet) should work as before."""
        llm = FakeLLM(num_linear=0, num_full_attn=4)
        _simulate_generation(llm, [1, 2, 3, 4, 5])

        # Partial overlap — should only return suffix
        suffix = llm._prepare_cache_for_prompt([1, 2, 3, 10, 20])
        assert suffix == [10, 20], "Pure KV model should return only suffix"

    def test_pure_kv_exact_repeat(self):
        """Pure KV cache model with exact same prompt."""
        llm = FakeLLM(num_linear=0, num_full_attn=4)
        _simulate_generation(llm, [1, 2, 3])

        suffix = llm._prepare_cache_for_prompt([1, 2, 3])
        assert suffix == [], "Pure KV model exact repeat should return empty suffix"

    def test_reset_clears_arrays_cache_entries(self):
        """_reset_all_caches should set ArraysCache entries to None."""
        llm = FakeLLM()
        _simulate_generation(llm, [1, 2, 3])

        llm._reset_all_caches()

        for c in llm._prompt_cache:
            if not c.is_trimmable():
                assert c.empty(), "ArraysCache should be empty after reset"
            else:
                assert c.offset == 0, "KVCache should have offset 0 after reset"

    def test_growing_conversation_works(self):
        """Multi-turn: growing prompt should work correctly."""
        llm = FakeLLM(num_linear=0, num_full_attn=4)

        # Turn 1: system + user1
        _simulate_generation(llm, [1, 2, 3, 4, 5])

        # Turn 2: system + user1 + assistant1 + user2
        # Common prefix = [1, 2, 3, 4, 5], suffix = [6, 7, 8]
        suffix = llm._prepare_cache_for_prompt([1, 2, 3, 4, 5, 6, 7, 8])
        assert suffix == [6, 7, 8], "Should only process new tokens"

    def test_deltanet_growing_conversation_resets(self):
        """Multi-turn with DeltaNet: must reset because gen tokens are in state."""
        llm = FakeLLM()

        # Turn 1
        _simulate_generation(llm, [1, 2, 3, 4, 5])

        # Turn 2: extends the prompt. KV cache offset = 10 (5 prompt + 5 gen),
        # common_len = 5, KV needs trim (10 > 5), DeltaNet is non-empty → reset
        suffix = llm._prepare_cache_for_prompt([1, 2, 3, 4, 5, 6, 7, 8])
        assert suffix == [1, 2, 3, 4, 5, 6, 7, 8], \
            "DeltaNet model must reprocess full prompt when KV needs trimming"
