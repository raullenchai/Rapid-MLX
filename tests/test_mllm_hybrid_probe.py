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

from unittest.mock import MagicMock

from mlx_lm.models.cache import KVCache, RotatingKVCache

from vllm_mlx.engine.batched import _probe_mllm_cache_type


class _FakeArraysCache:
    """Stand-in for mlx-lm's ArraysCache that doesn't import the real one
    (which requires building a layer). Type name is what the probe uses
    for the error message, so a class with the right __name__ is enough."""

    pass


_FakeArraysCache.__name__ = "ArraysCache"


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


def test_probe_returns_class_name_for_arrayscache():
    """Hybrid models (Qwen3.5/3.6/Nemotron) produce ArraysCache; probe
    must return the offending type name so the caller can quote it in
    the startup error."""
    model = _model_with_cache(_FakeArraysCache())
    assert _probe_mllm_cache_type(model) == "ArraysCache"


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
