# SPDX-License-Identifier: Apache-2.0
"""``Scheduler.projected_memory_max_context`` — the memory-fitted context
ceiling behind the ``/v1/models`` ``max_model_len`` field.

The method is exercised in isolation: a Scheduler built via ``__new__`` (so no
engine spin-up) with just the two attributes it reads (``model``, ``config``),
and a stubbed ``mx`` so the device-memory budget and residency are
deterministic. That keeps the test on the arithmetic — dim resolution, the
architecture-aware footprint, and the binary-search inversion — not on the
host GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx.scheduler import Scheduler, _read_kv_dims

# Qwen3-0.6B-ish dense dims — the real shape that first surfaced the
# ``.args``-not-``.config`` gap this method has to handle.
DENSE_ARGS = SimpleNamespace(
    num_hidden_layers=28,
    num_key_value_heads=8,
    head_dim=128,
    dtype="bfloat16",
)
# 2 (K+V) * layers * kv_heads * head_dim * dtype_bytes(bf16=2)
PER_TOKEN = 2 * 28 * 8 * 128 * 2  # 229_376 bytes/token


def _make_scheduler(
    model,
    util=0.0,
    *,
    sched_per_tok=0,
    sched_fixed=0,
    sched_slot=0,
    sched_window=0,
):
    """A Scheduler stub. The scheduler's cached KV terms default to 0 so the
    method takes the ``_read_kv_dims`` fallback (the mlx-lm ``.args`` case);
    pass ``sched_per_tok`` etc. to exercise the primary "prefer the scheduler's
    own resolved terms" path instead.
    """
    sched = Scheduler.__new__(Scheduler)
    sched.model = model
    sched.config = SimpleNamespace(
        gpu_memory_utilization=util,
        metal_cap_kv_bytes_per_token=0,
        kv_cache_dtype="bf16",
    )
    sched._kv_bytes_per_token_resolved = True
    sched._kv_bytes_per_token = sched_per_tok
    sched._kv_fixed_baseline_bytes = sched_fixed
    sched._kv_sliding_slot_bytes = sched_slot
    sched._kv_sliding_window = sched_window
    return sched


def _stub_mx(monkeypatch, *, base_bytes, resident_bytes, metal=True):
    fake = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: metal),
        device_info=lambda: {"max_recommended_working_set_size": base_bytes},
        get_active_memory=lambda: resident_bytes,
    )
    monkeypatch.setattr("vllm_mlx.scheduler.mx", fake)


def test_read_kv_dims_from_mlx_args() -> None:
    """mlx-lm models carry dims on ``.args``, not ``.config`` — the case the
    scheduler's own cached path misses."""
    model = SimpleNamespace(args=DENSE_ARGS)  # no ``.config`` attribute
    dims = _read_kv_dims(model)
    assert dims is not None
    layers, kv_heads, head_dim, cfg = dims
    assert (layers, kv_heads, head_dim) == (28, 8, 128)
    assert cfg is DENSE_ARGS


def test_read_kv_dims_none_for_stub_model() -> None:
    assert _read_kv_dims(SimpleNamespace()) is None


def test_read_kv_dims_prefers_text_tower_over_decoy_outer() -> None:
    """A multimodal shape carries decoy vision dims on the outer config and the
    real text-tower dims under ``text_config`` — the text tower must win, or
    max_model_len would be computed from the wrong architecture."""
    outer = SimpleNamespace(
        num_hidden_layers=40,  # decoy vision-tower depth
        num_key_value_heads=16,
        head_dim=80,
        text_config=SimpleNamespace(
            num_hidden_layers=28,
            num_key_value_heads=8,
            head_dim=128,
            dtype="bfloat16",
        ),
    )
    dims = _read_kv_dims(SimpleNamespace(args=outer))
    assert dims is not None
    assert dims[:3] == (28, 8, 128)


def test_native_window_fits_returns_native(monkeypatch) -> None:
    """When the whole native window fits the budget, report the native window
    (memory is not the binding constraint)."""
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=100 * 10**9, resident_bytes=0)
    assert sched.projected_memory_max_context(40960) == 40960


def test_memory_capped_below_native(monkeypatch) -> None:
    """A budget too small for the native window returns the largest T that
    fits — the exact inverse of the per-token footprint."""
    base = 2_550_000_000
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=base, resident_bytes=0)
    available = int(base * 0.90)  # util defaults to 0.90 when cap disabled
    expected = available // PER_TOKEN  # dense: footprint(T) = PER_TOKEN * T
    result = sched.projected_memory_max_context(40960)
    assert result == expected
    assert 0 < result < 40960


def test_resident_weights_reduce_the_ceiling(monkeypatch) -> None:
    """Already-resident bytes (weights + live cache) come out of the budget."""
    base = 20 * 10**9
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=base, resident_bytes=10 * 10**9)
    available = int(base * 0.90) - 10 * 10**9
    assert sched.projected_memory_max_context(1_000_000) == available // PER_TOKEN


def test_capped_at_native_even_with_huge_budget(monkeypatch) -> None:
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=10**12, resident_bytes=0)
    assert sched.projected_memory_max_context(5000) == 5000


def test_configured_utilization_is_honored(monkeypatch) -> None:
    """When the operator set a Metal cap fraction, use it (not the 0.90
    reporting default)."""
    base = 10 * 10**9
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS), util=0.5)
    _stub_mx(monkeypatch, base_bytes=base, resident_bytes=0)
    assert sched.projected_memory_max_context(10**6) == int(base * 0.5) // PER_TOKEN


def test_prefers_scheduler_resolved_terms(monkeypatch) -> None:
    """When the scheduler already resolved its footprint terms (config-backed
    models, and where the operator KV override applies), use THOSE — so the
    advertised ceiling matches admission — not a re-derivation. Proven by a
    model exposing no dims at all: a fallback to _read_kv_dims would be None."""
    base = 2_550_000_000
    sched = _make_scheduler(SimpleNamespace(), sched_per_tok=PER_TOKEN)
    _stub_mx(monkeypatch, base_bytes=base, resident_bytes=0)
    assert sched.projected_memory_max_context(40960) == int(base * 0.90) // PER_TOKEN


def test_none_when_no_dims(monkeypatch) -> None:
    sched = _make_scheduler(SimpleNamespace())  # stub model, no dims
    _stub_mx(monkeypatch, base_bytes=100 * 10**9, resident_bytes=0)
    assert sched.projected_memory_max_context(40960) is None


def test_none_when_metal_unavailable(monkeypatch) -> None:
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=100 * 10**9, resident_bytes=0, metal=False)
    assert sched.projected_memory_max_context(40960) is None


def test_none_when_budget_exhausted(monkeypatch) -> None:
    """Residency at or above the budget → no headroom → None, not 0."""
    base = 10 * 10**9
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=base, resident_bytes=int(base * 0.90))
    assert sched.projected_memory_max_context(40960) is None


@pytest.mark.parametrize("native", [None, 0, -5])
def test_no_native_hint_uses_generous_ceiling(monkeypatch, native) -> None:
    """Without a usable native cap the search still terminates and returns a
    positive fit."""
    base = 2_550_000_000
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    _stub_mx(monkeypatch, base_bytes=base, resident_bytes=0)
    result = sched.projected_memory_max_context(native)
    assert result == int(base * 0.90) // PER_TOKEN
