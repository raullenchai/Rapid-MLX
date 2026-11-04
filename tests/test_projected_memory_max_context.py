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

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


from types import SimpleNamespace

from rapid_mlx.scheduler import Scheduler, _read_kv_dims

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
    sched._model_weights_bytes = 0
    sched.waiting = []
    sched.running = {}
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
    monkeypatch.setattr("rapid_mlx.scheduler.mx", fake)


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
    sched._model_weights_bytes = 10 * 10**9
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
    sched._model_weights_bytes = int(base * 0.90)
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


def test_live_kv_cannot_shrink_stable_weights_ceiling(monkeypatch):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS), util=0.5)
    sched._model_weights_bytes = 10**9
    _stub_mx(monkeypatch, base_bytes=10**10, resident_bytes=9 * 10**9)
    expected = (5 * 10**9 - 10**9) // PER_TOKEN
    assert sched.projected_memory_max_context(1_000_000) == expected
    # Cached probes on the frontend do not touch Metal, even while another
    # request's KV occupies all currently available memory.
    monkeypatch.setattr(
        "rapid_mlx.scheduler.mx.device_info", lambda: pytest.fail("hot-path probe")
    )
    assert sched.projected_memory_max_context(1_000_000) == expected


def test_unknown_weight_baseline_does_not_use_transient_active(monkeypatch):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    sched._model_weights_bytes = None
    _stub_mx(monkeypatch, base_bytes=10**10, resident_bytes=10**9)
    assert sched.projected_memory_max_context(40960) is None


def test_memory_ceiling_requires_active_admission_cap(monkeypatch):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    monkeypatch.setattr(sched, "_resolve_metal_cap_bytes", lambda: 0)
    assert sched.memory_context_ceiling(40960) is None
    monkeypatch.setattr(sched, "_resolve_metal_cap_bytes", lambda: 100)
    monkeypatch.setattr(
        sched, "projected_memory_max_context", lambda native: native // 2
    )
    assert sched.memory_context_ceiling(40960) == 20480


@pytest.mark.parametrize("holder", ["config", "args"])
def test_offline_and_runtime_share_nested_text_config_terms(holder):
    from rapid_mlx.kv_estimation import footprint_from_config

    config = {
        "num_hidden_layers": 40,
        "num_key_value_heads": 32,
        "head_dim": 64,
        "dtype": "float16",
        "text_config": {
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
    }
    sched = _make_scheduler(SimpleNamespace(**{holder: config}))
    sched._kv_bytes_per_token_resolved = False
    expected = footprint_from_config(config)
    assert expected is not None
    assert sched._footprint_from_model_dims() == expected
    assert sched._resolve_kv_bytes_per_token() == expected.per_token_growth_bytes
    assert sched._resolve_kv_fixed_baseline_bytes() == expected.fixed_baseline_bytes
    assert sched._kv_sliding_slot_bytes == expected.sliding_slot_bytes
    assert sched._kv_sliding_window == expected.sliding_window


def test_idle_floor_ratchets_and_invalidates_cached_ceiling(
    monkeypatch, single_model_process
):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS), util=0.5)
    sched.requests = {}
    sched.memory_aware_cache = sched.prefix_cache = None
    sched.block_aware_cache = sched.paged_cache_manager = None
    sched._model_weights_bytes = 10**9
    _stub_mx(monkeypatch, base_bytes=10**10, resident_bytes=2 * 10**9)
    before = sched.projected_memory_max_context(1_000_000)
    sched.observe_idle_resident_floor()
    after = sched.projected_memory_max_context(1_000_000)
    assert after == (3 * 10**9 - 1) // PER_TOKEN
    assert after < before
    sched.observe_idle_resident_floor(1)
    assert sched.projected_memory_max_context(1_000_000) == after
    sched.requests = {"live": object()}
    sched.observe_idle_resident_floor(4 * 10**9)
    assert sched.projected_memory_max_context(1_000_000) == after


@pytest.mark.parametrize(
    "cache",
    ["memory_aware_cache", "prefix_cache", "paged_cache_manager", "block_aware_cache"],
)
def test_cached_kv_never_raises_floor(cache, single_model_process):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    sched.requests = {}
    sched.memory_aware_cache = sched.prefix_cache = None
    sched.block_aware_cache = sched.paged_cache_manager = None
    sched._steady_resident_bytes = 100
    paged = SimpleNamespace(blocks=[SimpleNamespace(cache_data=object())])
    value = {
        "memory_aware_cache": SimpleNamespace(_entries={"cached": object()}),
        "prefix_cache": SimpleNamespace(_lru={"cached": object()}),
        "paged_cache_manager": paged,
        "block_aware_cache": SimpleNamespace(paged_cache=paged),
    }[cache]
    setattr(sched, cache, value)
    sched.observe_idle_resident_floor(1000)
    assert sched._steady_resident_bytes == 100


def test_warmup_records_idle_residency_after_forward():
    from unittest.mock import Mock

    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._model = Mock()
    engine._tokenizer = SimpleNamespace(encode=lambda _: [1])
    scheduler = SimpleNamespace(observe_idle_resident_floor=Mock())
    engine._engine = SimpleNamespace(
        engine=SimpleNamespace(scheduler=scheduler, _mlx_executor=None)
    )
    engine.generate_warmup()
    scheduler.observe_idle_resident_floor.assert_called_once_with()


def test_observed_floor_is_usable_when_parameter_size_is_unknown(monkeypatch):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS), util=0.5)
    sched._model_weights_bytes = None
    sched._steady_resident_bytes = 10**9
    _stub_mx(monkeypatch, base_bytes=10**10, resident_bytes=10**9)
    assert sched.projected_memory_max_context(1_000_000) == (4 * 10**9 - 1) // PER_TOKEN


@pytest.fixture
def single_model_process(monkeypatch):
    from rapid_mlx import model_registry
    from rapid_mlx.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg, "model_registry", None)
    monkeypatch.setattr(cfg, "residency_manager", None)
    monkeypatch.setattr(cfg, "embedding_engine", None)
    monkeypatch.setattr(cfg, "rerank_engine", None)
    registry = object.__new__(model_registry.ModelRegistry)
    registry._init()
    monkeypatch.setattr(model_registry, "_registry", registry)
    return cfg, registry


@pytest.mark.parametrize("busy", [False, True])
def test_multi_engine_never_learns_process_residency(single_model_process, busy):
    from unittest.mock import Mock, patch

    from rapid_mlx.engine.batched import BatchedEngine

    cfg, registry = single_model_process
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    sched.requests = {}
    sched._model_weights_bytes = sched._steady_resident_bytes = 100
    sched.memory_aware_cache = sched.prefix_cache = None
    sched.block_aware_cache = sched.paged_cache_manager = None
    other = SimpleNamespace(
        scheduler=SimpleNamespace(requests={"live": 1} if busy else {})
    )
    # Use weak-referenceable owners like the real ownership registry.
    owner = Mock()
    owner.scheduler = sched
    other_owner = Mock()
    other_owner.scheduler = other.scheduler
    registry.acquire(sched.model, owner, "first")
    registry.acquire(object(), other_owner, "second")
    sched.observe_idle_resident_floor(900)
    assert sched._steady_resident_bytes == 100
    # The same guard runs after a warmup forward on this engine's worker.
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._model = Mock()
    engine._tokenizer = SimpleNamespace(encode=lambda _: [1])
    engine._engine = SimpleNamespace(
        engine=SimpleNamespace(scheduler=sched, _mlx_executor=None)
    )
    with patch.object(sched, "_current_metal_active_bytes", return_value=900):
        engine.generate_warmup()
    assert sched._steady_resident_bytes == 100


@pytest.mark.parametrize(
    "mode", ["model_registry", "residency_manager", "embedding_engine", "rerank_engine"]
)
def test_managed_loading_never_learns_floor(single_model_process, mode):
    cfg, _ = single_model_process
    setattr(cfg, mode, object())
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    sched._steady_resident_bytes = 100
    sched.observe_idle_resident_floor(900)
    assert sched._steady_resident_bytes == 100


def test_other_sole_owner_prevents_learning(single_model_process):
    from unittest.mock import Mock

    _, registry = single_model_process
    owner = Mock()
    registry.acquire(object(), owner, "other")
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    sched._steady_resident_bytes = 100
    sched.observe_idle_resident_floor(900)
    assert sched._steady_resident_bytes == 100


@pytest.mark.parametrize("queue", ["waiting", "running"])
def test_queued_work_prevents_floor_learning(single_model_process, queue):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS))
    sched.requests = {}
    setattr(sched, queue, [object()])
    sched._steady_resident_bytes = 100
    sched.observe_idle_resident_floor(900)
    assert sched._steady_resident_bytes == 100


@pytest.mark.parametrize("reading", [None, 0, -1, 10**9])
def test_unknown_floor_requires_valid_idle_measurement(
    reading, monkeypatch, single_model_process
):
    sched = _make_scheduler(SimpleNamespace(args=DENSE_ARGS), util=0.5)
    sched._model_weights_bytes = sched._steady_resident_bytes = None
    sched.requests = {}
    sched.memory_aware_cache = sched.prefix_cache = None
    sched.block_aware_cache = sched.paged_cache_manager = None
    _stub_mx(monkeypatch, base_bytes=10**10, resident_bytes=reading)
    if reading is None:

        def unavailable():
            raise RuntimeError("measurement failed")

        monkeypatch.setattr("rapid_mlx.scheduler.mx.get_active_memory", unavailable)
    assert sched.projected_memory_max_context(1_000_000) is None
    assert sched._memory_context_cache[0][-1] is None
    sched.observe_idle_resident_floor()
    if reading is not None and reading > 0:
        assert sched._steady_resident_bytes == reading
        assert (
            sched.projected_memory_max_context(1_000_000)
            == (4 * 10**9 - 1) // PER_TOKEN
        )
    else:
        assert sched._steady_resident_bytes is None
        assert sched.projected_memory_max_context(1_000_000) is None
