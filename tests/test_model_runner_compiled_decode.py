# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the compiled-decode-replay lane gate in ``model_runner``.

These are CPU-only and do not load a model. They cover the fail-closed
eligibility gate, the manifest requirement, the warm-restore plan decision,
and the publish-back discipline. Device correctness (bit-identity to eager,
engagement, warm-restore TTFT, throughput) is covered by the GPU gate script,
not here.
"""

import types

import pytest

from vllm_mlx import model_runner as mr
from vllm_mlx.model_runner import MLXModelRunner


def _runner(*, enabled=True, vllm_config=None):
    """A runner with the compiled-lane attributes set, bypassing model load."""
    r = MLXModelRunner.__new__(MLXModelRunner)
    r._enable_compiled_decode = enabled
    r.model = None
    r._compiled_apc_tokens = None
    r._compiled_apc_cache = None
    r.vllm_config = vllm_config or types.SimpleNamespace(speculative_config=None)
    return r


def _params(**kw):
    kw.setdefault("temperature", 0.0)
    kw.setdefault("top_p", 1.0)
    kw.setdefault("n", 1)
    return types.SimpleNamespace(**kw)


def test_env_truthy():
    assert mr._env_truthy("1")
    assert mr._env_truthy("true")
    assert mr._env_truthy("YES")
    for falsey in ("", "0", "false", "no", "off", None):
        assert not mr._env_truthy(falsey)


def test_disabled_lane_declines(monkeypatch):
    monkeypatch.delenv(mr._COMPILED_QUALIFICATION_ENV, raising=False)
    r = _runner(enabled=False)
    reason = r._compiled_decode_decline_reason([1, 2, 3], _params(), 16)
    assert reason == "compiled-decode lane disabled"
    assert r._plan_compiled_decode([1, 2, 3], _params(), 16) is None


def test_manifest_required(monkeypatch):
    monkeypatch.delenv(mr._COMPILED_QUALIFICATION_ENV, raising=False)
    r = _runner(enabled=True)
    reason = r._compiled_decode_decline_reason([1, 2, 3], _params(), 16)
    # Either the build lacks the compiled surface, or (surface present) the
    # missing manifest is the blocker. Both are fail-closed to eager.
    available, _ = mr._compiled_decode_available()
    if available:
        assert "qualification manifest" in reason
    else:
        assert reason is not None


def _require_surface():
    available, why = mr._compiled_decode_available()
    if not available:
        pytest.skip(f"mlx-lm compiled-decode surface unavailable: {why}")


def test_speculative_declined(monkeypatch):
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    r = _runner(enabled=True, vllm_config=types.SimpleNamespace(
        speculative_config=object()
    ))
    reason = r._compiled_decode_decline_reason([1, 2, 3], _params(), 16)
    assert "speculative" in reason


def test_batched_declined(monkeypatch):
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    r = _runner(enabled=True)
    assert "width 1" in r._compiled_decode_decline_reason([1, 2], _params(n=2), 16)
    assert "width 1" in r._compiled_decode_decline_reason(
        [1, 2], _params(best_of=4), 16
    )


def test_prompt_lookup_declined(monkeypatch):
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    monkeypatch.setenv("VLLM_MLX_PROMPT_LOOKUP", "1")
    r = _runner(enabled=True)
    assert "PLD" in r._compiled_decode_decline_reason([1, 2, 3], _params(), 16)


def test_kv_bits_declined(monkeypatch):
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    r = _runner(enabled=True)
    assert "kv_bits" in r._compiled_decode_decline_reason(
        [1, 2, 3], _params(kv_bits=4), 16
    )
    assert "max_kv_size" in r._compiled_decode_decline_reason(
        [1, 2, 3], _params(max_kv_size=2048), 16
    )


def test_out_of_policy_context_declined(monkeypatch):
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    monkeypatch.delenv("MLX_LM_COMPILED_DECODE_CONTEXT_POLICY", raising=False)
    r = _runner(enabled=True)
    # short profile stops at 4096; 5000 + 16 exceeds it.
    reason = r._compiled_decode_decline_reason(list(range(5000)), _params(), 16)
    assert reason is not None and "short" in reason


def test_plan_warm_restore_hit_reuses_stored_cache(monkeypatch):
    """A stored prefix that this request extends reuses the stored cache and
    feeds only the suffix, without touching make_prompt_cache."""
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    r = _runner(enabled=True)
    sentinel_cache = ["STORED-CACHE"]
    r._compiled_apc_tokens = (1, 2, 3)
    r._compiled_apc_cache = sentinel_cache

    # Guard: the hit branch must not call make_prompt_cache.
    import mlx_lm.models.cache as cache_mod

    def _boom(*a, **k):  # pragma: no cover - only fires on a bug
        raise AssertionError("make_prompt_cache must not run on a warm hit")

    monkeypatch.setattr(cache_mod, "make_prompt_cache", _boom)

    plan = r._plan_compiled_decode([1, 2, 3, 4, 5], _params(), 16)
    assert plan is not None
    assert plan["hit"] is True
    assert plan["cache"] is sentinel_cache
    assert plan["feed"] == [4, 5]
    assert plan["prefix_len"] == 3
    # The reference is cleared while in flight (re-published on finish).
    assert r._compiled_apc_cache is None
    assert r._compiled_apc_tokens is None


def test_plan_no_hit_on_divergent_prefix(monkeypatch):
    _require_surface()
    monkeypatch.setenv(mr._COMPILED_QUALIFICATION_ENV, "/tmp/fake-manifest.json")
    r = _runner(enabled=True)
    r._compiled_apc_tokens = (1, 2, 9)
    r._compiled_apc_cache = ["STORED"]

    import mlx_lm.models.cache as cache_mod

    monkeypatch.setattr(cache_mod, "make_prompt_cache", lambda *a, **k: ["FRESH"])
    plan = r._plan_compiled_decode([1, 2, 3, 4], _params(), 16)
    assert plan is not None and plan["hit"] is False
    assert plan["cache"] == ["FRESH"]
    assert plan["feed"] == [1, 2, 3, 4]


def test_publish_back_converts_ring_to_stock():
    """Publish-back stores the stock KVCache form, never a RingKVCache."""
    from mlx_lm.models.cache import KVCache, RingKVCache

    r = _runner(enabled=True)
    ring = RingKVCache()  # empty is fine: to_kv_cache() returns an empty KVCache
    arrays_like = object()
    plan = {"cache": [ring, arrays_like], "feed": [4], "hit": False, "prefix_len": 0}
    r._record_compiled_result(
        plan, [1, 2, 3], [4], status={"used": True, "decline_reason": None}
    )
    assert r._compiled_apc_tokens == (1, 2, 3, 4)
    assert r._compiled_apc_cache is not None
    assert isinstance(r._compiled_apc_cache[0], KVCache)
    assert not any(isinstance(c, RingKVCache) for c in r._compiled_apc_cache)
    assert r._compiled_apc_cache[1] is arrays_like
