# SPDX-License-Identifier: Apache-2.0
"""Tests for the startup probe that blocks hybrid models from --mllm mode (#352).

The user-facing contract:
- MLLM continuous batching cannot merge ArraysCache / MambaCache (only
  KVCache / RotatingKVCache).
- We must fail at startup with a message naming the actual incompatibility
  (hybrid backbone), not the misleading "disable --kv-cache-quantization"
  the user saw in #352 with Qwen3.6-35B-A3B-UD --mllm.
"""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from vllm_mlx.engine.batched import _probe_mllm_cache_type, _resolve_mllm_cache_policy
from vllm_mlx.mllm_cache_compat import first_incompatible_mllm_cache_type
from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig


class _FakeArraysCache:
    """Stand-in for mlx-lm's ArraysCache that doesn't import the real one
    (which requires building a layer). Type name is what the probe uses
    for the error message, so a class with the right __name__ is enough."""

    pass


_FakeArraysCache.__name__ = "ArraysCache"


class _FakeMambaCache:
    pass


_FakeMambaCache.__name__ = "MambaCache"


def _model_with_cache(cache_obj):
    """Build a fake language_model whose make_prompt_cache hook returns
    a single-layer cache containing ``cache_obj``."""
    m = MagicMock()
    # mlx-lm's make_prompt_cache calls model.make_cache() when present;
    # both that and a direct prompt-cache list need to return the layers.
    m.make_cache = MagicMock(return_value=[cache_obj])
    # mlx-lm 0.31's make_prompt_cache also walks layers/n_kv_heads etc.
    # for fallback construction — we only need the make_cache path here,
    # which short-circuits the fallback.
    return m


def test_probe_returns_none_for_kvcache():
    """Standard text models produce KVCache → probe returns None and
    startup proceeds normally."""
    model = _model_with_cache(KVCache())
    assert _probe_mllm_cache_type(model) is None


def test_probe_returns_none_for_rotating_kvcache():
    """Sliding-window models (e.g. Gemma 4 with RotatingKVCache) are
    also MLLM-compatible."""
    model = _model_with_cache(RotatingKVCache(max_size=256))
    assert _probe_mllm_cache_type(model) is None


@pytest.mark.parametrize("cache_name", ["KVCache", "RotatingKVCache"])
def test_probe_accepts_mlx_vlm_native_cache_classes(monkeypatch, cache_name):
    """mlx-vlm 0.6.4+ owns cache classes distinct from mlx-lm's classes.

    The MLLM model returns those native caches from ``make_cache()``.  They
    have the batching API Rapid-MLX needs, so rejecting them solely because
    an ``isinstance`` check names mlx-lm's parallel class crashes every
    affected VLM during startup (rapid-desktop #603).
    """
    from mlx_vlm.models import cache as vlm_cache

    native_cache_class = type(cache_name, (), {})
    monkeypatch.setattr(vlm_cache, cache_name, native_cache_class)

    model = _model_with_cache(native_cache_class())
    assert _probe_mllm_cache_type(model) is None


def test_probe_returns_class_name_for_arrayscache():
    """Hybrid models (Qwen3.5/3.6/Nemotron) produce ArraysCache; probe
    must return the offending type name so the caller can quote it in
    the startup error."""
    model = _model_with_cache(_FakeArraysCache())
    assert _probe_mllm_cache_type(model) == "ArraysCache"


def test_compatibility_check_inspects_every_cache_layer():
    """A mixed backbone may put a supported KV layer before ArraysCache."""
    assert (
        first_incompatible_mllm_cache_type([KVCache(), _FakeArraysCache()])
        == "ArraysCache"
    )


def test_arrayscache_is_accepted_only_for_serialized_compatibility():
    """The runtime allowlist is explicit; startup probing stays fail-closed."""
    cache = _FakeArraysCache()
    assert first_incompatible_mllm_cache_type([cache]) == "ArraysCache"
    # A name-only stand-in is deliberately not enough to enter the allowlist.
    # Production accepts only mlx-lm's concrete ArraysCache class.
    assert (
        first_incompatible_mllm_cache_type([cache], allow_arrays_cache=True)
        == "ArraysCache"
    )

    real_cache = ArraysCache(2)
    assert (
        first_incompatible_mllm_cache_type(
            [KVCache(), real_cache], allow_arrays_cache=True
        )
        is None
    )


def test_mlx_vlm_arrayscache_is_accepted_in_serialized_mode(monkeypatch):
    """The VLM namespace owns a class distinct from mlx-lm's ArraysCache."""
    from mlx_vlm.models import cache as vlm_cache

    native_arrays_cache = type("ArraysCache", (), {})
    monkeypatch.setattr(vlm_cache, "ArraysCache", native_arrays_cache)

    cache = native_arrays_cache()
    assert first_incompatible_mllm_cache_type([cache]) == "ArraysCache"
    assert first_incompatible_mllm_cache_type([cache], allow_arrays_cache=True) is None


def test_arrayscache_scheduler_mode_requires_singleton_limits():
    config = MLLMSchedulerConfig(
        max_num_seqs=1,
        prefill_batch_size=1,
        completion_batch_size=1,
        allow_arrays_cache=True,
    )
    assert config.allow_arrays_cache is True

    with pytest.raises(ValueError, match="all be 1"):
        MLLMSchedulerConfig(
            max_num_seqs=2,
            prefill_batch_size=1,
            completion_batch_size=1,
            allow_arrays_cache=True,
        )


def test_engine_policy_serializes_only_arrayscache():
    assert _resolve_mllm_cache_policy("ArraysCache", 16, 8, 32) == (1, 1, 1, True)
    assert _resolve_mllm_cache_policy(None, 16, 8, 32) == (16, 8, 32, False)
    assert _resolve_mllm_cache_policy("MambaCache", 16, 8, 32) == (
        16,
        8,
        32,
        False,
    )


def test_arrayscache_singleton_primitives_preserve_recurrent_state():
    """The admitted cache supports the lifecycle used by MLLMBatch."""
    first = ArraysCache(2)
    first[0] = mx.array([[1.0, 2.0]])
    first[1] = mx.array([[3.0]])

    merged = ArraysCache.merge([first])
    assert merged.batch_size == 1
    assert merged[0].tolist() == [[1.0, 2.0]]

    merged.filter(mx.array([0], mx.int32))
    extracted = merged.extract(0)
    assert extracted.batch_size == 1
    assert extracted[1].tolist() == [[3.0]]


def test_serialized_mode_does_not_admit_other_hybrid_caches():
    assert (
        first_incompatible_mllm_cache_type([_FakeMambaCache()], allow_arrays_cache=True)
        == "MambaCache"
    )


def test_compatibility_check_without_optional_mlx_vlm(monkeypatch):
    """Text-only installs have mlx-lm but intentionally omit mlx-vlm."""
    original_import = builtins.__import__
    blocked = False

    def import_without_mlx_vlm(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal blocked
        if name == "mlx_vlm.models" and "cache" in fromlist:
            blocked = True
            raise ImportError("mlx-vlm is not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_mlx_vlm)
    assert first_incompatible_mllm_cache_type([KVCache()]) is None
    assert blocked is True


def test_probe_returns_none_when_make_prompt_cache_raises():
    """A model that crashes on cache construction (e.g. mid-load,
    incompatible mlx-lm version) is best-effort skipped — runtime path
    surfaces the real error rather than us masking it with #352's text."""
    broken = MagicMock()
    broken.make_cache = MagicMock(side_effect=RuntimeError("not loaded"))
    # mlx-lm's make_prompt_cache will try fallbacks; we expect *some* path
    # to raise inside it. The probe swallows that and returns None.
    result = _probe_mllm_cache_type(broken)
    assert result is None


def test_probe_returns_none_for_empty_cache():
    """Defensive: zero-layer model returns an empty list. Probe must not
    IndexError into sample = cache[0]."""
    model = MagicMock()
    model.make_cache = MagicMock(return_value=[])
    assert _probe_mllm_cache_type(model) is None
