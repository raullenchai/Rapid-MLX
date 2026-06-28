# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``TurboQuantKVCache.state`` (surfaced 2026-06-27).

Reproducer
----------

Before the fix, the graceful-shutdown prefix-cache flush walked the
radix entries and called ``mlx_lm.models.cache.save_prompt_cache``,
which does ``[c.state for c in cache]``. ``TurboQuantKVCache`` had no
``state`` property → per-entry write raised
``AttributeError: 'TurboQuantKVCache' object has no attribute 'state'``
and ``save_to_disk`` committed **zero** entries to disk. Every K8V4
restart paid a full re-prefill cost on shared system prompts.

Evidence: ``/tmp/k8v4_bench/cfg3-radix-k8v4-server.log``::

    failed to save entry N: 'TurboQuantKVCache' object has no attribute 'state'
    [lifespan] No cache to save

Tests
-----

1. ``state`` / ``meta_state`` shape + idempotent in-process round-trip
   (V4 and K8V4). No filesystem.
2. End-to-end through ``mlx_lm.save_prompt_cache`` →
   ``mlx_lm.load_prompt_cache`` (V4 and K8V4) — exercises the SAME
   path the radix shutdown flush takes, just on a tempfile instead of
   the user cache dir. Byte-identical decode after the round trip.
3. End-to-end through ``MemoryAwarePrefixCache.save_to_disk`` →
   ``load_from_disk`` with a populated ``TurboQuantKVCache`` entry —
   reproduces the bug scenario directly: pre-fix this committed
   ``0 entries``, post-fix it commits the entry and reloads it.
4. ``mlx_lm.models.cache`` globals registration is idempotent — the
   import-time injection must not clobber an existing global of the
   same name (defensive against a future upstream introducing one).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

mx = pytest.importorskip("mlx.core")
mlx_cache = pytest.importorskip("mlx_lm.models.cache")
save_prompt_cache = mlx_cache.save_prompt_cache
load_prompt_cache = mlx_cache.load_prompt_cache
KVCache = mlx_cache.KVCache

import numpy as np  # noqa: E402

from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)
from vllm_mlx.turboquant import (  # noqa: E402
    TurboQuantConfig,
    TurboQuantKVCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populated_kv(head_dim: int = 128, seq_len: int = 32, n_heads: int = 8):
    """Build a deterministic mock KVCache-like object."""
    rng = np.random.RandomState(0)
    kv = MagicMock()
    kv.keys = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv.values = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv.offset = seq_len
    return kv


def _populated_real_kv(head_dim: int = 128, seq_len: int = 32, n_heads: int = 8):
    """Build a real ``mlx_lm.models.cache.KVCache`` (needed by the
    MemoryAwarePrefixCache integration test — ``store`` paths run through
    code that touches a real KVCache before any TurboQuant compression).
    """
    rng = np.random.RandomState(0)
    keys = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    values = mx.array(rng.randn(1, n_heads, seq_len, head_dim).astype(np.float16))
    kv = KVCache()
    kv.update_and_fetch(keys, values)
    return kv


# ---------------------------------------------------------------------------
# 1. state / meta_state shape and in-process round-trip
# ---------------------------------------------------------------------------


class TestStateShape:
    def test_v4_state_has_keys_and_compressed_values(self):
        """V4: ``keys`` (fp16) + V-side ``v_*``. No K-compressed entries."""
        kv = _populated_kv()
        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="v4"))
        state = tq.state
        assert set(state.keys()) == {"keys", "v_indices", "v_scales", "v_zeros"}
        assert all(isinstance(a, mx.array) for a in state.values())

    def test_k8v4_state_has_only_compressed(self):
        """K8V4: no fp16 ``keys`` slot; ``k_*`` and ``v_*`` only."""
        kv = _populated_kv()
        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="k8v4"))
        state = tq.state
        assert "keys" not in state
        assert set(state.keys()) == {
            "v_indices",
            "v_scales",
            "v_zeros",
            "k_packed",
            "k_norms",
            "k_scales",
        }

    def test_meta_state_round_trip_preserves_config(self):
        """``meta_state`` carries enough to reconstruct codec config + dtype."""
        kv = _populated_kv()
        cfg = TurboQuantConfig(bits=4, group_size=32, rotation_seed=7, mode="k8v4")
        tq = TurboQuantKVCache.from_kv_cache(kv, cfg)
        meta = tq.meta_state
        # 7-element shape (offset, head_dim, bits, group_size, seed, mode, dtype)
        assert len(meta) == 7
        # tree_unflatten turns the tuple into a list — emulate that here.
        tq2 = TurboQuantKVCache.from_state(tq.state, list(meta))
        assert tq2.offset == tq.offset
        assert tq2.head_dim == tq.head_dim
        assert tq2.config.bits == cfg.bits
        assert tq2.config.group_size == cfg.group_size
        assert tq2.config.rotation_seed == cfg.rotation_seed
        assert tq2.config.mode == cfg.mode
        assert tq2.original_dtype == tq.original_dtype

    def test_empty_cache_state_is_empty(self):
        """Empty caches (``from_kv_cache`` on a never-populated KV) emit
        empty markers on both axes — symmetric with ``_BaseCache``."""
        kv = MagicMock()
        kv.keys = None
        kv.values = None
        kv.offset = 0
        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig())
        assert tq.state == {}
        assert tq.meta_state == ()
        tq2 = TurboQuantKVCache.from_state(tq.state, list(tq.meta_state))
        assert tq2.offset == 0
        assert tq2.head_dim == 0
        assert tq2.keys is None
        assert tq2.keys_compressed is None


class TestStateSetterValidation:
    def test_non_dict_state_rejected(self):
        """A list-shaped ``state`` would mean the snapshot was written by
        a future codec — fail loudly, never silently produce garbage."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        with pytest.raises(TypeError, match="must be a dict"):
            tq.state = [1, 2, 3]

    @pytest.mark.parametrize("empty", [{}, []])
    def test_empty_dict_and_empty_list_both_accepted_as_empty_marker(self, empty):
        """``tree_unflatten([])`` returns ``[]`` even when the state was
        saved as ``{}``, so both empty markers must round-trip to the
        empty-cache shape. Anything ELSE falsy (None/0/'') must NOT."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        tq.state = empty
        assert tq.keys is None
        assert tq.values_compressed == (None, None, None)
        assert tq.keys_compressed is None

    @pytest.mark.parametrize("bad", [None, 0, ""])
    def test_non_empty_marker_falsy_state_rejected(self, bad):
        """Codex round 1 BLOCKING #2: ``not v`` was too permissive — a
        ``None`` / ``0`` / ``""`` snapshot must not silently load as
        empty. Only ``{}`` / ``[]`` qualify as the empty marker."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        with pytest.raises(TypeError, match="must be a dict"):
            tq.state = bad

    def test_partial_v_fields_rejected(self):
        """Codex round 1 BLOCKING #1: a snapshot with some but not all
        v_* fields must fail at the setter, not defer the crash to
        ``to_kv_cache()``. The setter signal routes through the
        persistence ``corrupt_skipped`` counter for /metrics."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        # Construct a fake v_indices array so the check sees a non-empty
        # state but with the v_* triple incomplete.
        arr = mx.array([1, 2], dtype=mx.uint8)
        with pytest.raises(ValueError, match="partial v_\\* fields"):
            tq.state = {"v_indices": arr, "v_scales": arr}

    def test_partial_k_fields_rejected(self):
        """Same all-or-nothing rule for the K8 side."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        arr_v = mx.array([1, 2], dtype=mx.uint8)
        arr_k = mx.array([3, 4], dtype=mx.uint8)
        with pytest.raises(ValueError, match="partial k_\\* fields"):
            tq.state = {
                "v_indices": arr_v,
                "v_scales": arr_v,
                "v_zeros": arr_v,
                "k_packed": arr_k,
                "k_norms": arr_k,
                # missing k_scales
            }

    def test_keys_only_state_without_v_rejected(self):
        """V is the one thing TurboQuant ALWAYS emits — fp16 keys
        without v_* is structurally impossible. Fail at the setter
        rather than at decode."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        arr = mx.array([1, 2], dtype=mx.float16)
        with pytest.raises(ValueError, match="no V compression"):
            tq.state = {"keys": arr}

    def test_meta_state_wrong_length_rejected(self):
        """Same defense for ``meta_state`` — only the 7-element shape this
        codec emits is acceptable."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        with pytest.raises(ValueError, match="expects 7 elements"):
            tq.meta_state = ["1", "2", "3"]

    def test_meta_state_invalid_dtype_rejected(self):
        """An unknown dtype name in ``meta_state`` → ``ValueError`` so the
        persistence layer's ``corrupt_skipped`` counter fires."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        with pytest.raises(ValueError, match="unknown dtype"):
            tq.meta_state = ["32", "128", "4", "32", "42", "v4", "not_a_real_dtype"]

    @pytest.mark.parametrize("empty", [(), []])
    def test_meta_state_empty_markers_accepted(self, empty):
        """``()`` and ``[]`` are both legitimate empty markers (tuple
        emitted by ``meta_state`` getter, list produced by tree_unflatten)."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        tq.meta_state = empty
        assert tq.offset == 0
        assert tq.head_dim == 0

    @pytest.mark.parametrize("bad", [None, 0, ""])
    def test_meta_state_non_empty_marker_falsy_rejected(self, bad):
        """Codex round 1 BLOCKING #3: a non-tuple non-list falsy value
        must NOT silently load with default config — that path bypasses
        the documented 7-element validation."""
        tq = TurboQuantKVCache.__new__(TurboQuantKVCache)
        with pytest.raises(TypeError, match="must be a tuple/list"):
            tq.meta_state = bad


# ---------------------------------------------------------------------------
# 2. End-to-end through mlx_lm save_prompt_cache / load_prompt_cache
# ---------------------------------------------------------------------------


class TestSavePromptCacheRoundTrip:
    """Pre-fix this raised ``AttributeError`` at line 51 of upstream
    ``cache.py`` (``cache_data = [c.state for c in cache]``)."""

    def test_v4_byte_identical_decode_after_safetensors_roundtrip(self, tmp_path):
        kv = _populated_kv()
        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="v4"))

        path = str(tmp_path / "v4.safetensors")
        save_prompt_cache(path, [tq], metadata={"num_tokens": "32"})
        loaded = load_prompt_cache(path)

        assert len(loaded) == 1
        tq2 = loaded[0]
        assert isinstance(tq2, TurboQuantKVCache)
        assert tq2.config.mode == "v4"

        # Byte-identical decode: same codec inputs (state) → same output.
        kv1 = tq.to_kv_cache()
        kv2 = tq2.to_kv_cache()
        assert mx.array_equal(kv1.keys, kv2.keys)
        assert mx.array_equal(kv1.values, kv2.values)

    def test_k8v4_byte_identical_decode_after_safetensors_roundtrip(self, tmp_path):
        kv = _populated_kv()
        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="k8v4"))

        path = str(tmp_path / "k8v4.safetensors")
        save_prompt_cache(path, [tq], metadata={"num_tokens": "32"})
        loaded = load_prompt_cache(path)

        assert len(loaded) == 1
        tq2 = loaded[0]
        assert isinstance(tq2, TurboQuantKVCache)
        assert tq2.config.mode == "k8v4"
        # K8V4 stores K in ``keys_compressed``; ``keys`` stays None.
        assert tq2.keys is None
        assert tq2.keys_compressed is not None

        kv1 = tq.to_kv_cache()
        kv2 = tq2.to_kv_cache()
        assert mx.array_equal(kv1.keys, kv2.keys)
        assert mx.array_equal(kv1.values, kv2.values)

    def test_bf16_original_dtype_round_trips(self, tmp_path):
        """``original_dtype`` is the only non-string scalar in
        ``meta_state``. bf16 is the operationally-important case
        (Qwen-class native dtype); float16 is exercised in the V4 test
        above."""
        rng = np.random.RandomState(0)
        keys_bf16 = mx.array(rng.randn(1, 8, 32, 128).astype(np.float32)).astype(
            mx.bfloat16
        )
        values_bf16 = mx.array(rng.randn(1, 8, 32, 128).astype(np.float32)).astype(
            mx.bfloat16
        )
        kv = MagicMock()
        kv.keys = keys_bf16
        kv.values = values_bf16
        kv.offset = 32

        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="v4"))
        assert tq.original_dtype == mx.bfloat16

        path = str(tmp_path / "bf16.safetensors")
        save_prompt_cache(path, [tq], metadata={"num_tokens": "32"})
        tq2 = load_prompt_cache(path)[0]
        assert tq2.original_dtype == mx.bfloat16
        assert tq2.to_kv_cache().keys.dtype == mx.bfloat16

    def test_multi_layer_mixed_modes_round_trip(self, tmp_path):
        """Realistic: a model whose decision layer is V4 and rest is K8V4
        (or any mix). Each layer's mode rides in its own ``meta_state``."""
        kv = _populated_kv()
        layers = [
            TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="v4")),
            TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode="k8v4")),
            TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=3, mode="v4")),
        ]
        path = str(tmp_path / "mixed.safetensors")
        save_prompt_cache(path, layers, metadata={"num_tokens": "32"})
        loaded = load_prompt_cache(path)
        assert [type(c).__name__ for c in loaded] == ["TurboQuantKVCache"] * 3
        assert [c.config.mode for c in loaded] == ["v4", "k8v4", "v4"]
        assert [c.config.bits for c in loaded] == [4, 4, 3]


# ---------------------------------------------------------------------------
# 3. Integration: MemoryAwarePrefixCache.save_to_disk reproduces the bug
# ---------------------------------------------------------------------------


class TestMemoryAwarePrefixCacheTurboQuant:
    """This is the failure mode the bug report cites directly: the
    radix shutdown flush walks ``_entries`` and calls
    ``MemoryAwarePrefixCache.save_to_disk`` → ``save_prompt_cache``. Pre-fix
    the per-entry write raised ``AttributeError`` and ZERO entries
    persisted. Post-fix the entry survives → reloads on next boot."""

    def _store_turboquant_entry(
        self, cache: MemoryAwarePrefixCache, tokens, mode: str
    ) -> TurboQuantKVCache:
        kv = _populated_real_kv()
        tq = TurboQuantKVCache.from_kv_cache(kv, TurboQuantConfig(bits=4, mode=mode))
        cache.store(tokens, [tq])
        return tq

    def _assert_full_round_trip(self, cache, cache2, tokens, original_tq, mode):
        """Shared post-load invariants — codex round 1 NIT #4 strengthening:
        verify the reloaded entry is a TurboQuantKVCache with the same mode
        and decodes byte-identically to the originally stored entry, not
        merely that the token-key exists."""
        assert tuple(tokens) in cache2._entries
        entry = cache2._entries[tuple(tokens)]
        # ``_CacheEntry.cache`` is the per-layer list; we stored a single
        # TurboQuantKVCache layer.
        assert len(entry.cache) == 1
        loaded_layer = entry.cache[0]
        assert isinstance(loaded_layer, TurboQuantKVCache), (
            f"reload produced {type(loaded_layer).__name__}, not TurboQuantKVCache"
        )
        assert loaded_layer.config.mode == mode
        assert loaded_layer.offset == original_tq.offset
        assert loaded_layer.head_dim == original_tq.head_dim
        # Byte-identical decode: same compressed inputs → same output.
        kv_orig = original_tq.to_kv_cache()
        kv_loaded = loaded_layer.to_kv_cache()
        assert mx.array_equal(kv_orig.keys, kv_loaded.keys)
        assert mx.array_equal(kv_orig.values, kv_loaded.values)

    def test_k8v4_entry_survives_save_to_disk_and_reloads(self, tmp_path):
        # Pre-fix repro: save_to_disk would log
        # 'failed to save entry N: ... no attribute state'
        # and return False with zero files written. Post-fix the entry
        # commits and the next load_from_disk finds it.
        cache = MemoryAwarePrefixCache(
            model=object(),
            config=MemoryCacheConfig(
                max_memory_mb=64, max_entries=100, kv_turboquant=True
            ),
        )
        tokens = list(range(32))
        original_tq = self._store_turboquant_entry(cache, tokens, mode="k8v4")
        assert len(cache) == 1

        ok = cache.save_to_disk(str(tmp_path))
        assert ok is True

        # Per-entry files exist on disk — pre-fix this directory was empty.
        files = set(os.listdir(tmp_path))
        assert "index.json" in files
        assert "entry_0.safetensors" in files
        assert "entry_0_tokens.bin" in files

        # Reload into a fresh cache — same config required because
        # ``_cache_classes_compatible`` gates ``TurboQuantKVCache``
        # on ``kv_turboquant=True``.
        cache2 = MemoryAwarePrefixCache(
            model=object(),
            config=MemoryCacheConfig(
                max_memory_mb=64, max_entries=100, kv_turboquant=True
            ),
        )
        loaded = cache2.load_from_disk(str(tmp_path))
        assert loaded == 1
        self._assert_full_round_trip(cache, cache2, tokens, original_tq, mode="k8v4")

    def test_v4_entry_survives_save_to_disk_and_reloads(self, tmp_path):
        """V4 path uses the same ``state`` property — covering it guards
        against a regression that fixes K8V4 in isolation."""
        cache = MemoryAwarePrefixCache(
            model=object(),
            config=MemoryCacheConfig(
                max_memory_mb=64, max_entries=100, kv_turboquant=True
            ),
        )
        tokens = list(range(32))
        original_tq = self._store_turboquant_entry(cache, tokens, mode="v4")
        assert cache.save_to_disk(str(tmp_path)) is True

        cache2 = MemoryAwarePrefixCache(
            model=object(),
            config=MemoryCacheConfig(
                max_memory_mb=64, max_entries=100, kv_turboquant=True
            ),
        )
        assert cache2.load_from_disk(str(tmp_path)) == 1
        self._assert_full_round_trip(cache, cache2, tokens, original_tq, mode="v4")


# ---------------------------------------------------------------------------
# 4. Idempotent registration in mlx_lm.models.cache globals
# ---------------------------------------------------------------------------


class TestUpstreamGlobalsRegistration:
    def test_class_visible_to_upstream_load(self):
        """``load_prompt_cache`` resolves ``globals()[type_name]`` in the
        upstream module — our import-time hook must make ``TurboQuantKVCache``
        discoverable there."""
        assert mlx_cache.__dict__.get("TurboQuantKVCache") is TurboQuantKVCache

    def test_registration_does_not_clobber_existing(self):
        """``setdefault`` semantics — if a future upstream adds its own
        ``TurboQuantKVCache``, our import-time hook must NOT overwrite it.
        Verified by injecting a sentinel and re-running the registration."""
        from vllm_mlx.turboquant import _register_in_mlx_lm_cache_globals

        original = mlx_cache.__dict__.get("TurboQuantKVCache")
        sentinel = object()
        try:
            mlx_cache.__dict__["TurboQuantKVCache"] = sentinel
            _register_in_mlx_lm_cache_globals()
            assert mlx_cache.__dict__["TurboQuantKVCache"] is sentinel
        finally:
            # Restore so subsequent tests see the real class.
            if original is None:
                mlx_cache.__dict__.pop("TurboQuantKVCache", None)
            else:
                mlx_cache.__dict__["TurboQuantKVCache"] = original
