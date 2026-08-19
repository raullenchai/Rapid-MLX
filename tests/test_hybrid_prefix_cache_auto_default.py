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

    def test_auto_defaults_for_deepseek_v4_0731_local_path(self, monkeypatch):
        """DeepSeek's pooling/rotating caches need bounded trim-free reuse."""
        monkeypatch.setattr(
            "vllm_mlx.model_aliases.resolve_profile", lambda _name: None
        )
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=False,
            model_name=("/models/DeepSeek-V4-Flash-0731-MXFP4-MLX"),
        )
        assert result == _DEFAULT_HYBRID_CACHE_ENTRIES

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

    def test_auto_defaults_for_pinned_nonhybrid_recurrent(self, monkeypatch):
        """Dense GatedDeltaNet pinned is_hybrid=False + is_hybrid_explicit=True
        (qwen3.5/3.6 dense, Ternary-Bonsai) → auto-default to 8.

        These have non-trimmable ArraysCache layers but are deliberately kept
        off the hybrid scheduler (metal::malloc wedge). Before this fix the
        #1122 auto-default keyed only on ``is_hybrid`` and skipped them, so
        ``--enable-prefix-cache`` stored nothing and every agent turn
        re-prefilled the whole context.
        """
        _patch_resolve_profile(monkeypatch, is_hybrid=False, is_hybrid_explicit=True)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=False,
            model_name="qwen3.5-9b-4bit",
        )
        assert result == _DEFAULT_HYBRID_CACHE_ENTRIES

    def test_pinned_nonhybrid_recurrent_respects_explicit_zero(self, monkeypatch):
        """Even for a pinned-nonhybrid recurrent model, an explicit 0 wins."""
        _patch_resolve_profile(monkeypatch, is_hybrid=False, is_hybrid_explicit=True)
        result = _resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=True,
            model_name="qwen3.5-9b-4bit",
        )
        assert result == 0


class TestNeedsBoundedTrimFreeReuseRealAliases:
    """End-to-end against the real alias registry (no mocks) — the #1122 gap.

    These assert the actual shipped aliases.json classification, so they fail
    on ``main`` (dense recurrent models returned False) and pass with the fix.
    """

    def test_dense_recurrent_qwen35_9b_needs_bounded_reuse(self):
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse

        assert _needs_bounded_trim_free_reuse("qwen3.5-9b-4bit") is True

    def test_dense_recurrent_qwen36_27b_needs_bounded_reuse(self):
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse

        assert _needs_bounded_trim_free_reuse("qwen3.6-27b-4bit") is True

    def test_lfm25_26b_hybrid_alias_needs_bounded_reuse(self):
        """LFM2.5's declared ArraysCache routing must retain cache entries."""
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse
        from vllm_mlx.model_aliases import resolve_profile

        profile = resolve_profile("lfm2.5-2.6b-4bit")
        assert profile is not None
        assert profile.is_hybrid is True
        assert _needs_bounded_trim_free_reuse("lfm2.5-2.6b-4bit") is True
        assert (
            _resolve_hybrid_cache_entries(
                enable_prefix_cache=True,
                explicit_value=0,
                user_set_explicit=False,
                model_name="lfm2.5-2.6b-4bit",
            )
            == _DEFAULT_HYBRID_CACHE_ENTRIES
        )

    def test_moe_hybrid_still_needs_bounded_reuse(self):
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse

        # MoE A3B flagship is is_hybrid=True — unchanged path.
        assert _needs_bounded_trim_free_reuse("qwen3.6-35b-4bit") is True

    def test_pure_attention_alias_does_not_need_bounded_reuse(self):
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse

        # A genuine pure-attention alias (no ArraysCache layers, not pinned
        # is_hybrid_explicit) must stay on the ordinary trimmable prefix cache.
        assert _needs_bounded_trim_free_reuse("qwen3-coder-30b-4bit") is False


def _patch_resolve_profile(
    monkeypatch, *, is_hybrid: bool, is_hybrid_explicit: bool = False
):
    """Monkeypatch resolve_profile to return a mock alias profile.

    ``is_hybrid_explicit`` must be set on the mock (not left as an
    auto-truthy ``MagicMock`` attribute) because
    ``_needs_bounded_trim_free_reuse`` now reads it to detect dense
    recurrent models pinned non-hybrid (#1122 gap for qwen3.5/3.6 dense).
    """
    mock_profile = MagicMock()
    mock_profile.is_hybrid = is_hybrid
    mock_profile.is_hybrid_explicit = is_hybrid_explicit
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile", lambda _name: mock_profile
    )


# ---------------------------------------------------------------------------
# #2061: sliding-window (RotatingKVCache) families need bounded trim-free reuse
# ---------------------------------------------------------------------------


class TestConfigDeclaresSlidingWindow:
    """Unit tests for the architecture-driven sliding-window probe (no I/O)."""

    def test_layer_types_sliding_is_detected(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        assert _config_declares_sliding_window(
            {"layer_types": ["sliding_attention", "full_attention"]}
        )

    def test_positive_sliding_window_int_is_detected(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        assert _config_declares_sliding_window({"sliding_window": 512})

    def test_nested_text_config_is_inspected(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        # Gemma VLM checkpoints nest the language config under ``text_config``.
        assert _config_declares_sliding_window(
            {"text_config": {"sliding_window": 1024}}
        )

    def test_full_attention_only_is_not_detected(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        assert not _config_declares_sliding_window(
            {"layer_types": ["full_attention", "full_attention"]}
        )

    def test_layer_types_wins_over_inert_sliding_window_scalar(self):
        """An authoritative all-full-attention ``layer_types`` must NOT be
        overridden by a leftover ``sliding_window`` scalar — that field is inert
        without a sliding layer to apply it (codex #2064)."""
        from vllm_mlx.cli import _config_declares_sliding_window

        assert not _config_declares_sliding_window(
            {"layer_types": ["full_attention"], "sliding_window": 4096}
        )

    def test_layer_types_sliding_still_detected_with_sliding_window(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        assert _config_declares_sliding_window(
            {
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 1024,
            }
        )

    def test_nested_language_config_wins_over_top_level_scalar(self):
        """The language backbone (``text_config``) is authoritative: an all-
        full-attention LM must not be forced sliding by a top-level
        ``sliding_window`` scalar (often a vision/default field) — codex #2064."""
        from vllm_mlx.cli import _config_declares_sliding_window

        assert not _config_declares_sliding_window(
            {"sliding_window": 4096, "text_config": {"layer_types": ["full_attention"]}}
        )

    def test_top_level_used_when_nested_lm_config_has_no_signal(self):
        """If ``text_config`` carries no attention signal at all, root-level
        fields still count (checkpoints that place them at the root)."""
        from vllm_mlx.cli import _config_declares_sliding_window

        assert _config_declares_sliding_window(
            {"sliding_window": 512, "text_config": {"hidden_size": 1}}
        )

    def test_use_sliding_window_false_disables_scalar(self):
        """Qwen2 / Mistral carry a ``sliding_window`` scalar gated behind
        ``use_sliding_window`` — a positive scalar with the flag off is inert and
        must NOT trigger bounded snapshots (memory regression) — codex #2064."""
        from vllm_mlx.cli import _config_declares_sliding_window

        assert not _config_declares_sliding_window(
            {"use_sliding_window": False, "sliding_window": 32768}
        )

    def test_use_sliding_window_true_keeps_scalar(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        assert _config_declares_sliding_window(
            {"use_sliding_window": True, "sliding_window": 4096}
        )

    def test_empty_or_zero_or_none_is_not_detected(self):
        from vllm_mlx.cli import _config_declares_sliding_window

        assert not _config_declares_sliding_window({})
        assert not _config_declares_sliding_window(None)
        assert not _config_declares_sliding_window({"sliding_window": 0})


class TestSlidingWindowNeedsBoundedReuse:
    """#2061: Gemma-2/3/4 sliding layers run on RotatingKVCache, which is
    non-trimmable once the ring rotates past its window. They must take the
    bounded snapshot path or --enable-prefix-cache is a silent no-op and every
    agentic turn re-prefills the whole context. Detection is config-driven so a
    bare local path (with no alias) is covered too — exactly how #2061 was
    served (an lm-studio gemma-4 checkpoint path)."""

    _SLIDING = {"layer_types": ["sliding_attention", "full_attention"]}
    _FULL = {"layer_types": ["full_attention", "full_attention"]}

    def test_sliding_window_alias_needs_bounded_reuse(self, monkeypatch):
        # Reference cli through the live module object (not a load-time import)
        # and patch the probe on that SAME object, so a sibling test that
        # reload/pops ``vllm_mlx.cli`` cannot desync the two (per the string-
        # patch-target rule in testing-gotchas).
        import vllm_mlx.cli as cli

        _patch_resolve_profile(monkeypatch, is_hybrid=False)
        monkeypatch.setattr(
            cli, "_resolve_checkpoint_config", lambda _name, _profile: self._SLIDING
        )
        assert cli._needs_bounded_trim_free_reuse("gemma-4-26b-4bit") is True

    def test_sliding_window_alias_auto_defaults_entries(self, monkeypatch):
        import vllm_mlx.cli as cli

        _patch_resolve_profile(monkeypatch, is_hybrid=False)
        monkeypatch.setattr(
            cli, "_resolve_checkpoint_config", lambda _name, _profile: self._SLIDING
        )
        result = cli._resolve_hybrid_cache_entries(
            enable_prefix_cache=True,
            explicit_value=0,
            user_set_explicit=False,
            model_name="gemma-4-26b-4bit",
        )
        assert result == cli._DEFAULT_HYBRID_CACHE_ENTRIES

    def test_sliding_window_bare_path_no_alias(self, monkeypatch):
        """#2061 exactly: served by a bare checkpoint path (resolve_profile
        returns None), so only the config probe can classify it."""
        import vllm_mlx.cli as cli

        monkeypatch.setattr(
            "vllm_mlx.model_aliases.resolve_profile", lambda _name: None
        )
        monkeypatch.setattr(
            cli, "_resolve_checkpoint_config", lambda _name, _profile: self._SLIDING
        )
        assert (
            cli._needs_bounded_trim_free_reuse(
                "/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"
            )
            is True
        )

    def test_full_attention_config_stays_trimmable(self, monkeypatch):
        """A pure full-attention checkpoint must NOT be pushed onto the bounded
        snapshot path — its KVCache is ordinarily trimmable."""
        import vllm_mlx.cli as cli

        monkeypatch.setattr(
            "vllm_mlx.model_aliases.resolve_profile", lambda _name: None
        )
        monkeypatch.setattr(
            cli, "_resolve_checkpoint_config", lambda _name, _profile: self._FULL
        )
        assert cli._needs_bounded_trim_free_reuse("/models/llama-3-8b") is False

    @staticmethod
    def _write_config(dir_path, config: dict) -> str:
        import json

        (dir_path / "config.json").write_text(json.dumps(config))
        return str(dir_path)

    def test_bare_path_real_config_discovery_sliding(self, tmp_path):
        """End-to-end #2061: a bare on-disk checkpoint dir (no alias, nothing
        mocked) whose config.json declares sliding attention is classified as
        needing bounded reuse via the real offline config probe."""
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse

        path = self._write_config(
            tmp_path,
            {
                "model_type": "gemma4",
                "sliding_window": 512,
                "layer_types": ["sliding_attention", "full_attention"],
            },
        )
        assert _needs_bounded_trim_free_reuse(path) is True

    def test_bare_path_real_config_discovery_full_attention(self, tmp_path):
        """The same real path, for a full-attention checkpoint, stays off the
        bounded path — guards against the probe defaulting to True."""
        from vllm_mlx.cli import _needs_bounded_trim_free_reuse

        path = self._write_config(
            tmp_path, {"model_type": "llama", "layer_types": ["full_attention"]}
        )
        assert _needs_bounded_trim_free_reuse(path) is False
