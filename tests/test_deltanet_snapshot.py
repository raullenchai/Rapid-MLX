"""Tests for DeltaNet/hybrid cache snapshot logic in MLXLanguageModel."""

import copy
from unittest.mock import MagicMock

import pytest


class FakeKVCache:
    """Mock trimmable KVCache layer."""

    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n

    def size(self):
        return self.offset


class FakeArraysCache:
    """Mock non-trimmable ArraysCache layer (DeltaNet/Gated DeltaNet)."""

    def __init__(self, state_id=0):
        self.state_id = state_id
        self.data = [0.0] * 10  # simulate some state

    def is_trimmable(self):
        return False

    def __deepcopy__(self, memo):
        new = FakeArraysCache(self.state_id)
        new.data = list(self.data)
        return new


def make_model():
    """Create a minimal MLXLanguageModel without loading a real model."""
    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel.__new__(MLXLanguageModel)
    model._prompt_cache = None
    model._cached_token_ids = []
    model._cache_lock = False
    model._rnn_state_snapshot = None
    model._snapshot_prefix_ids = []
    model._main_cache_len = 0
    model.prefill_step_size = 2048
    model.model = None
    model.draft_model = None
    return model


def make_hybrid_cache(num_rnn=3, num_kv=1, kv_offset=100):
    """Create a hybrid cache: mix of ArraysCache + KVCache layers."""
    layers = []
    for i in range(num_rnn):
        layers.append(FakeArraysCache(state_id=i))
    for _ in range(num_kv):
        kv = FakeKVCache()
        kv.offset = kv_offset
        layers.append(kv)
    return layers


class TestIsHybridCache:
    def test_none_cache(self):
        model = make_model()
        assert model._is_hybrid_cache() is False

    def test_pure_trimmable(self):
        model = make_model()
        model._prompt_cache = [FakeKVCache(), FakeKVCache()]
        assert model._is_hybrid_cache() is False

    def test_pure_non_trimmable(self):
        model = make_model()
        model._prompt_cache = [FakeArraysCache(), FakeArraysCache()]
        assert model._is_hybrid_cache() is False

    def test_hybrid(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache()
        assert model._is_hybrid_cache() is True


class TestSnapshotRnnLayers:
    def test_snapshot_saves_non_trimmable(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache(num_rnn=2, num_kv=1)
        prefix = [1, 2, 3, 4, 5]

        model._snapshot_rnn_layers(prefix)

        assert model._snapshot_prefix_ids == prefix
        assert model._rnn_state_snapshot is not None
        assert len(model._rnn_state_snapshot) == 3  # 2 rnn + 1 kv
        # Non-trimmable layers are deep-copied
        assert model._rnn_state_snapshot[0] is not None
        assert model._rnn_state_snapshot[1] is not None
        # Trimmable layers are None placeholders
        assert model._rnn_state_snapshot[2] is None

    def test_snapshot_is_independent_copy(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache(num_rnn=1, num_kv=1)
        prefix = [10, 20, 30]

        model._snapshot_rnn_layers(prefix)

        # Mutate the original cache
        model._prompt_cache[0].data[0] = 999.0

        # Snapshot should be unaffected
        assert model._rnn_state_snapshot[0].data[0] == 0.0


class TestRestoreRnnLayers:
    def test_restore_succeeds_with_exact_matching_prefix(self):
        """Restore succeeds when common_len == snap_len (no gap tokens)."""
        model = make_model()
        cache = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=100)
        model._prompt_cache = cache
        model._cached_token_ids = [1, 2, 3, 4, 5]

        # Take snapshot at exactly 5 tokens
        model._snapshot_rnn_layers([1, 2, 3, 4, 5])

        # Mutate the cache (simulating generation)
        cache[0].data[0] = 999.0
        cache[2].offset = 150  # KV grew during generation

        # Now restore with a prompt that shares the same 5-token prefix
        prompt = [1, 2, 3, 4, 5, 6, 7]  # same prefix + new suffix
        common_len = 5  # matches snap_len exactly → no gap tokens
        result = model._restore_rnn_layers(prompt, common_len)

        assert result is True
        # Non-trimmable layers should be restored from snapshot
        assert model._prompt_cache[0].data[0] == 0.0  # restored, not 999.0
        # KV layers should be trimmed to snap_len (== common_len here)
        assert model._prompt_cache[2].offset == 5

    def test_restore_fails_with_short_common_len(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache()
        model._snapshot_rnn_layers([1, 2, 3, 4, 5])

        # common_len < snapshot length
        result = model._restore_rnn_layers([1, 2, 9, 9, 9], 2)
        assert result is False

    def test_restore_fails_with_no_snapshot(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache()

        result = model._restore_rnn_layers([1, 2, 3], 3)
        assert result is False

    def test_restore_fails_with_token_mismatch(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache()
        model._snapshot_rnn_layers([1, 2, 3])

        # Same length but different tokens
        result = model._restore_rnn_layers([1, 2, 9, 4, 5], 3)
        assert result is False

    def test_restore_is_independent_copy(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache(num_rnn=1, num_kv=1, kv_offset=50)
        model._cached_token_ids = [1, 2, 3]
        model._snapshot_rnn_layers([1, 2, 3])

        # Restore
        model._restore_rnn_layers([1, 2, 3, 4], 3)

        # Mutate restored cache
        model._prompt_cache[0].data[0] = 888.0

        # Snapshot should be unaffected (deepcopy on restore)
        assert model._rnn_state_snapshot[0].data[0] == 0.0


class TestPrepareCacheHybrid:
    def test_hybrid_cache_restores_from_snapshot(self):
        model = make_model()
        cache = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=50)
        model._prompt_cache = cache
        model._cached_token_ids = [1, 2, 3, 4, 5]

        # Take snapshot at exactly 5 tokens (same as cached)
        model._snapshot_rnn_layers([1, 2, 3, 4, 5])

        # Simulate generation mutating the cache
        cache[0].data[0] = 999.0
        cache[2].offset = 80

        original_is_trimmable = model._cache_is_trimmable

        def mock_not_trimmable():
            return False

        model._cache_is_trimmable = mock_not_trimmable

        # Prepare with overlapping prompt (common_len == snap_len == 5)
        prompt = [1, 2, 3, 4, 5, 10, 11, 12]
        suffix = model._prepare_cache_for_prompt(prompt)

        assert suffix == [10, 11, 12]
        # RNN layers restored
        assert model._prompt_cache[0].data[0] == 0.0
        # KV trimmed to snap_len (5)
        assert model._prompt_cache[2].offset == 5

        model._cache_is_trimmable = original_is_trimmable

    def test_hybrid_cache_no_snapshot_no_overlap_recreates(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=50)
        model._cached_token_ids = [1, 2, 3]

        original_is_trimmable = model._cache_is_trimmable

        def mock_not_trimmable():
            return False

        model._cache_is_trimmable = mock_not_trimmable

        # Mock _make_fresh_cache
        fresh = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=0)
        model._make_fresh_cache = lambda: fresh

        # Completely different prompt — no overlap
        prompt = [99, 98, 97, 96]
        suffix = model._prepare_cache_for_prompt(prompt)

        # No overlap, no snapshot → full prompt returned
        assert suffix == prompt
        assert model._cached_token_ids == []

        model._cache_is_trimmable = original_is_trimmable

    def test_hybrid_cache_exact_repeat_returns_one_suffix_token(self):
        """Exact-repeat on hybrid cache must NOT return empty suffix.

        The generic trim(1) path only rolls back trimmable KV layers;
        non-trimmable RNN layers cannot be trimmed, so the last prompt
        token would be processed twice in recurrent layers.  Instead,
        effective_len is capped at len(prompt) - 1, ensuring at least
        1 suffix token that passes through both RNN and KV once.

        When snap_len == common_len, the reduced effective_len (common_len - 1)
        is less than snap_len, so restore fails and falls through to
        _prefill_and_snapshot which needs a model.  To test without a real
        model, we set snap_len = common_len - 1 (as if a previous
        exact-repeat already built this snapshot).
        """
        model = make_model()
        cache = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=50)
        model._prompt_cache = cache
        model._cached_token_ids = [1, 2, 3, 4, 5]

        # Snapshot at 4 tokens (simulating a prior exact-repeat that built
        # a snapshot at common_len - 1)
        model._snapshot_rnn_layers([1, 2, 3, 4])

        # Simulate generation mutating the cache
        cache[0].data[0] = 999.0
        cache[2].offset = 80

        model._cache_is_trimmable = lambda: False

        # EXACT same prompt → common_len == len(prompt) == 5
        # effective_len = 5 - 1 = 4 == snap_len → restore succeeds
        prompt = [1, 2, 3, 4, 5]
        suffix = model._prepare_cache_for_prompt(prompt)

        # Must NOT be empty — should be [5] (last token as suffix)
        assert len(suffix) >= 1
        assert suffix == [5]
        # RNN restored from snapshot
        assert model._prompt_cache[0].data[0] == 0.0

    def test_hybrid_cache_different_prefix_recreates(self):
        model = make_model()
        model._prompt_cache = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=50)
        model._cached_token_ids = [1, 2, 3, 4, 5]
        model._snapshot_rnn_layers([1, 2, 3, 4, 5])

        original_is_trimmable = model._cache_is_trimmable

        def mock_not_trimmable():
            return False

        model._cache_is_trimmable = mock_not_trimmable

        fresh = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=0)
        model._make_fresh_cache = lambda: fresh

        # Completely different prompt
        prompt = [99, 98, 97]
        suffix = model._prepare_cache_for_prompt(prompt)

        # No overlap → full recreate
        assert suffix == prompt
        assert model._cached_token_ids == []

        model._cache_is_trimmable = original_is_trimmable


class TestEstimateNewTokensHybrid:
    def _setup_model_with_snapshot(self):
        model = make_model()
        model._loaded = True
        model._prompt_cache = make_hybrid_cache(num_rnn=2, num_kv=1, kv_offset=50)
        model._cached_token_ids = [100, 200, 300, 400, 500]
        model._snapshot_rnn_layers([100, 200, 300, 400, 500])

        # Mock tokenizer
        model.tokenizer = MagicMock()
        model.tokenizer.bos_token = None

        # Mock _cache_is_trimmable
        model._cache_is_trimmable = lambda: False

        return model

    def test_estimate_with_snapshot_match(self):
        model = self._setup_model_with_snapshot()
        # Prompt shares full prefix + 3 new tokens
        model.tokenizer.encode.return_value = [100, 200, 300, 400, 500, 601, 602, 603]

        total, new = model.estimate_new_tokens("dummy")

        assert total == 8
        assert new == 3  # only suffix tokens

    def test_estimate_without_snapshot(self):
        model = self._setup_model_with_snapshot()
        model._rnn_state_snapshot = None  # no snapshot

        model.tokenizer.encode.return_value = [100, 200, 300, 400, 500, 601]

        total, new = model.estimate_new_tokens("dummy")

        assert total == 6
        assert new == 6  # full prefill since no snapshot

    def test_estimate_exact_repeat_reports_one_new_token(self):
        model = self._setup_model_with_snapshot()
        # Exact same tokens as cached → common_len == len(prompt)
        model.tokenizer.encode.return_value = [100, 200, 300, 400, 500]

        total, new = model.estimate_new_tokens("dummy")

        assert total == 5
        assert new == 1  # exact-repeat caps reuse at common_len - 1

    def test_estimate_with_partial_prefix_below_snapshot(self):
        model = self._setup_model_with_snapshot()
        # Only 2 tokens match, but snapshot needs 5
        model.tokenizer.encode.return_value = [100, 200, 999, 888]

        total, new = model.estimate_new_tokens("dummy")

        assert total == 4
        assert new == 4  # can't use snapshot, full prefill
