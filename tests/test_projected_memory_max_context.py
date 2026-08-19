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


def _make_scheduler(model, util=0.0):
    sched = Scheduler.__new__(Scheduler)
    sched.model = model
    sched.config = SimpleNamespace(
        gpu_memory_utilization=util,
        metal_cap_kv_bytes_per_token=0,
        kv_cache_dtype="bf16",
    )
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
