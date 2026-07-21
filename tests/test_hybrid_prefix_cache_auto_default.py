"""Regression test for #1122: prefix cache auto-defaults for hybrid models.

When --enable-prefix-cache is on and the model is hybrid (GatedDeltaNet
etc.), hybrid_cache_entries should auto-default to 8 so the cache
actually stores entries. Without this fix, every hybrid entry is
silently dropped at store time (stored=False).

Tests cover:
  1. CLI auto-default logic (_resolve_hybrid_cache_entries)
  2. Cache-layer behavior with hybrid_reuse_max_entries=0 vs >0
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.cli import _DEFAULT_HYBRID_CACHE_ENTRIES, _resolve_hybrid_cache_entries
from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

# ---------------------------------------------------------------------------
# Mock cache layers (mirrors test_hybrid_prefix_cache_growth.py)
# ---------------------------------------------------------------------------


class _MockArray:
    def __init__(self, nbytes: int = 100):
        self.nbytes = nbytes


class _TrimmableLayer:
    """Stands in for KVCache (transformer attention layer)."""

    def __init__(self, nbytes: int = 200, offset: int = 0):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)
        self._offset = offset

    @property
    def offset(self):
        return self._offset

    def is_trimmable(self):
        return True


class _NonTrimmableLayer:
    """Stands in for ArraysCache (DeltaNet/Mamba RNN state)."""

    def __init__(self, nbytes: int = 200):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)

    def is_trimmable(self):
        return False


def _hybrid_cache():
    """Hybrid model cache: 3 trimmable + 2 non-trimmable layers."""
    return [_TrimmableLayer() for _ in range(3)] + [
        _NonTrimmableLayer() for _ in range(2)
    ]


def _dense_cache():
    """Pure transformer cache: all trimmable."""
    return [_TrimmableLayer() for _ in range(5)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHybridCacheAutoDefault:
    """Test that hybrid_reuse_max_entries controls hybrid entry storage."""

    def test_hybrid_entry_dropped_when_zero(self):
        """Default 0 drops hybrid entries — reproduces #1122."""
        config = MemoryCacheConfig(max_memory_mb=10, hybrid_reuse_max_entries=0)
        cache = MemoryAwarePrefixCache(MagicMock(), config)

        stored = cache.store(list(range(100)), _hybrid_cache())
        assert stored is False, "hybrid entry should be dropped when limit=0"

    def test_hybrid_entry_stored_when_nonzero(self):
        """With hybrid_reuse_max_entries=8, hybrid entries are stored."""
        config = MemoryCacheConfig(max_memory_mb=10, hybrid_reuse_max_entries=8)
        cache = MemoryAwarePrefixCache(MagicMock(), config)

        stored = cache.store(list(range(100)), _hybrid_cache())
        assert stored is True, "hybrid entry should be stored when limit>0"

    def test_hybrid_entry_fetchable_after_store(self):
        """Stored hybrid entry can be fetched on exact match."""
        config = MemoryCacheConfig(max_memory_mb=10, hybrid_reuse_max_entries=8)
        cache = MemoryAwarePrefixCache(MagicMock(), config)

        tokens = list(range(100))
        cache.store(tokens, _hybrid_cache())

        result = cache.fetch(tokens)
        assert result is not None, "exact-match fetch should hit stored hybrid entry"

    def test_dense_cache_unaffected(self):
        """Dense (non-hybrid) entries are stored regardless of the flag."""
        config = MemoryCacheConfig(max_memory_mb=10, hybrid_reuse_max_entries=0)
        cache = MemoryAwarePrefixCache(MagicMock(), config)

        stored = cache.store(list(range(100)), _dense_cache())
        assert stored is True, "dense entry should be stored even when hybrid limit=0"


# ---------------------------------------------------------------------------
# CLI auto-default logic tests
# ---------------------------------------------------------------------------


class TestResolveHybridCacheEntries:
    """Test _resolve_hybrid_cache_entries — the CLI-layer auto-default."""

    def test_auto_defaults_for_hybrid_model(self, monkeypatch):
        """Hybrid model + prefix cache → auto-default to 8."""
        _patch_resolve_profile(monkeypatch, is_hybrid=True)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=False,
            model_name="qwen3.5-9b-4bit",
        )
        assert result == _DEFAULT_HYBRID_CACHE_ENTRIES

    def test_no_auto_default_for_non_hybrid(self, monkeypatch):
        """Non-hybrid model → stays 0."""
        _patch_resolve_profile(monkeypatch, is_hybrid=False)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=False,
            model_name="llama-3-8b-4bit",
        )
        assert result == 0

    def test_no_auto_default_without_prefix_cache(self, monkeypatch):
        """Hybrid model but prefix cache disabled → stays 0."""
        _patch_resolve_profile(monkeypatch, is_hybrid=True)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=False,
            explicit_value=0,
            user_set_explicit=False,
            model_name="qwen3.5-9b-4bit",
        )
        assert result == 0

    def test_explicit_zero_honored(self, monkeypatch):
        """User explicitly set --hybrid-cache-entries 0 → stays 0."""
        _patch_resolve_profile(monkeypatch, is_hybrid=True)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=True,
            model_name="qwen3.5-9b-4bit",
        )
        assert result == 0

    def test_explicit_nonzero_honored(self, monkeypatch):
        """User set --hybrid-cache-entries 16 → keeps 16."""
        _patch_resolve_profile(monkeypatch, is_hybrid=True)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=16,
            user_set_explicit=True,
            model_name="qwen3.5-9b-4bit",
        )
        assert result == 16

    def test_unknown_model_stays_zero(self, monkeypatch):
        """Unknown model (resolve_profile returns None) → stays 0."""
        monkeypatch.setattr(
            "vllm_mlx.model_aliases.resolve_profile", lambda _name: None
        )
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=False,
            model_name="unknown-model",
        )
        assert result == 0


def _patch_resolve_profile(monkeypatch, *, is_hybrid: bool):
    """Monkeypatch resolve_profile to return a mock alias profile."""
    mock_profile = MagicMock()
    mock_profile.is_hybrid = is_hybrid
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile", lambda _name: mock_profile
    )
