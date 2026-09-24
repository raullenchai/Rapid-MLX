"""Contracts for the benchmark-only Qwen3.8 serialized-MLLM fused-GDN A/B."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import benchmark_qwen38_mllm_fused_gdn as bench


class FakeArray:
    def __init__(self, shape: tuple[int, ...], dtype: str):
        self.shape = shape
        self.dtype = dtype

    def __mul__(self, _value):
        return FakeArray(self.shape, self.dtype)

    def astype(self, dtype):
        return FakeArray(self.shape, dtype)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeRandom:
    @staticmethod
    def key(value):
        return value

    @staticmethod
    def normal(shape, key=None):
        assert key is not None
        return FakeArray(shape, FakeMx.bfloat16)


class FakeMx:
    bfloat16 = "bf16"
    float32 = "fp32"
    array = FakeArray
    random = FakeRandom()

    @staticmethod
    def concatenate(values, axis=0):
        assert axis == 1
        left, right = values
        return FakeArray(
            (left.shape[0], left.shape[1] + right.shape[1], left.shape[2]),
            left.dtype,
        )

    @staticmethod
    def zeros(shape, dtype):
        return FakeArray(shape, dtype)

    @staticmethod
    def eval(*_values):
        return None

    @staticmethod
    def array_equal(left, right):
        return FakeScalar(left.shape == right.shape and left.dtype == right.dtype)


class FakeCache:
    def __init__(self, size=2):
        self.cache = [None] * size
        self._left_padding = None
        self._left_padding_advance = 0
        self._lengths = None
        self._lengths_advance = 0
        self._speculation = None
        self._speculation_generation = 0
        self.metadata_revision = 0
        self.window_updates = 0
        self.recurrent_updates = 0
        self.advances = 0
        self.fail_recurrent = False
        self.raise_speculation = False
        self.raise_history = False

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    @property
    def left_padding(self):
        return self._left_padding

    @property
    def lengths(self):
        return self._lengths

    @property
    def is_speculating(self):
        if self.raise_speculation:
            raise RuntimeError("is_speculating unavailable")
        return self._speculation is not None

    @property
    def history_capacity(self):
        if self.raise_history:
            raise RuntimeError("history_capacity unavailable")
        return 0

    def update_window(self, index, source, width, *, lengths=None):
        assert lengths is None
        self.window_updates += 1
        self.cache[index] = FakeArray((1, width, source.shape[2]), source.dtype)
        return self.cache[index]

    def update_recurrent(self, index, length, update):
        self.recurrent_updates += 1
        if self.fail_recurrent:
            raise RuntimeError("post-commit failure")
        output, state = update(self.cache[index], None)
        self.cache[index] = state
        return output, state

    def advance(self, amount):
        self.advances += amount


class FakeGdn:
    stock_calls = 0

    def __init__(self):
        self.hidden_size = bench.HIDDEN_SIZE
        self.training = False
        self.conv1d = SimpleNamespace(
            weight=FakeArray((bench.CONV_DIM, bench.CONV_KERNEL, 1), FakeMx.bfloat16)
        )
        self.A_log = FakeArray((bench.NUM_VALUE_HEADS,), FakeMx.float32)
        self.dt_bias = FakeArray((bench.NUM_VALUE_HEADS,), FakeMx.bfloat16)
        self.norm = SimpleNamespace(
            weight=FakeArray((bench.VALUE_HEAD_DIM,), FakeMx.bfloat16), eps=1e-6
        )
        self.out_proj_raises = False

    def __call__(self, inputs, mask=None, cache=None):
        type(self).stock_calls += 1
        return "stock"

    def in_proj_qkv(self, _inputs):
        return FakeArray((1, 1, bench.CONV_DIM), FakeMx.bfloat16)

    def in_proj_z(self, _inputs):
        return FakeArray((1, 1, bench.VALUE_DIM), FakeMx.bfloat16)

    def _project_gates(self, _inputs):
        shape = (1, 1, bench.NUM_VALUE_HEADS)
        return FakeArray(shape, FakeMx.bfloat16), FakeArray(shape, FakeMx.bfloat16)

    def out_proj(self, output):
        if self.out_proj_raises:
            raise RuntimeError("out projection failed")
        assert output.shape == (1, 1, bench.VALUE_DIM)
        return FakeArray((1, 1, bench.HIDDEN_SIZE), FakeMx.bfloat16)


def _kernel(*_args, **kwargs):
    assert kwargs["num_key_heads"] == bench.NUM_KEY_HEADS
    assert kwargs["num_value_heads"] == bench.NUM_VALUE_HEADS
    assert kwargs["qwen35_semantics"] is True
    return (
        FakeArray((1, 1, bench.VALUE_DIM), FakeMx.bfloat16),
        FakeArray((1, bench.CONV_KERNEL - 1, bench.CONV_DIM), FakeMx.bfloat16),
        FakeArray(
            (
                1,
                bench.NUM_VALUE_HEADS,
                bench.VALUE_HEAD_DIM,
                bench.KEY_HEAD_DIM,
            ),
            FakeMx.float32,
        ),
    )


def _patch(kernel=_kernel):
    layers = [FakeGdn() for _ in range(bench.EXPECTED_GDN_LAYERS)]
    patch = bench.FusedGdnPatch(
        FakeGdn,
        FakeCache,
        layers,
        FakeMx,
        kernel,
        lambda _cache, _steps: None,
        lambda _cache, _steps: None,
    )
    patch.threadgroup_y = 32
    patch.qualified = True
    patch.install()
    return patch, layers


def _cache():
    cache = FakeCache()
    cache[0] = FakeArray((1, bench.CONV_KERNEL - 1, bench.CONV_DIM), FakeMx.bfloat16)
    cache[1] = FakeArray(
        (
            1,
            bench.NUM_VALUE_HEADS,
            bench.VALUE_HEAD_DIM,
            bench.KEY_HEAD_DIM,
        ),
        FakeMx.float32,
    )
    return cache


def _input():
    return FakeArray((1, 1, bench.HIDDEN_SIZE), FakeMx.bfloat16)


def test_baseline_restores_exact_original_class_method():
    patch, _ = _patch()
    try:
        patch.set_candidate(False)
        assert FakeGdn.__call__ is patch.original
        patch.set_candidate(True)
        assert FakeGdn.__call__ is patch.wrapped
        patch.set_candidate(False)
        assert FakeGdn.__call__ is patch.original
    finally:
        patch.close()


def test_candidate_commits_through_exact_cache_apis():
    patch, layers = _patch()
    cache = _cache()
    try:
        output = patch._candidate_call(layers[0], _input(), None, cache)
        assert output.shape == (1, 1, bench.HIDDEN_SIZE)
        assert cache.window_updates == 1
        assert cache.recurrent_updates == 1
        assert cache.advances == 1
        assert patch.hits == 1
        assert patch.layer_hits == [1] + [0] * 47
    finally:
        patch.close()


def test_precommit_out_projection_failure_falls_back_without_cache_touch():
    patch, layers = _patch()
    cache = _cache()
    layers[0].out_proj_raises = True
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
        assert cache.recurrent_updates == 0
        assert cache.advances == 0
        assert patch.hits == 0
    finally:
        patch.close()


def test_precommit_kernel_failure_falls_back_without_cache_touch():
    def fail(*_args, **_kwargs):
        raise RuntimeError("kernel construction failed")

    patch, layers = _patch(fail)
    cache = _cache()
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
        assert cache.recurrent_updates == 0
        assert cache.advances == 0
    finally:
        patch.close()


def test_postcommit_failure_propagates_without_stock_replay():
    patch, layers = _patch()
    cache = _cache()
    cache.fail_recurrent = True
    FakeGdn.stock_calls = 0
    try:
        with pytest.raises(RuntimeError, match="post-commit"):
            patch._candidate_call(layers[0], _input(), None, cache)
        assert cache.window_updates == 1
        assert cache.recurrent_updates == 1
        assert cache.advances == 0
        assert FakeGdn.stock_calls == 0
    finally:
        patch.close()


@pytest.mark.parametrize(
    ("mutation", "stock_calls"),
    [
        (lambda cache, layer, value: setattr(value, "shape", (2, 1, 5120)), 1),
        (lambda cache, layer, value: setattr(cache, "_speculation", {}), 1),
        (lambda cache, layer, value: cache.cache.append(None), 1),
        (lambda cache, layer, value: setattr(cache, "_lengths", object()), 1),
        (lambda cache, layer, value: setattr(cache, "_left_padding", object()), 1),
        (lambda cache, layer, value: setattr(layer, "training", True), 1),
    ],
)
def test_ineligible_runtime_shape_is_untouched_stock(mutation, stock_calls):
    patch, layers = _patch()
    cache = _cache()
    value = _input()
    mutation(cache, layers[0], value)
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], value, None, cache) == "stock"
        assert FakeGdn.stock_calls == stock_calls
        assert cache.window_updates == 0
        assert cache.recurrent_updates == 0
    finally:
        patch.close()


@pytest.mark.parametrize("field", ["raise_speculation", "raise_history"])
def test_raising_cache_metadata_marker_falls_back_stock(field):
    patch, layers = _patch()
    cache = _cache()
    setattr(cache, field, True)
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
    finally:
        patch.close()


def test_missing_cache_metadata_marker_falls_back_stock():
    patch, layers = _patch()
    cache = _cache()
    del cache._speculation
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
    finally:
        patch.close()


def test_cache_subclass_falls_back_stock():
    class CacheSubclass(FakeCache):
        pass

    patch, layers = _patch()
    cache = CacheSubclass()
    cache.cache = _cache().cache
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
    finally:
        patch.close()


class ProbeGdn(FakeGdn):
    def __call__(self, inputs, mask=None, cache=None):
        assert inputs.shape == (1, 1, bench.HIDDEN_SIZE)
        cache[0] = FakeArray(
            (1, bench.CONV_KERNEL - 1, bench.CONV_DIM), FakeMx.bfloat16
        )
        cache[1] = FakeArray(
            (
                1,
                bench.NUM_VALUE_HEADS,
                bench.VALUE_HEAD_DIM,
                bench.KEY_HEAD_DIM,
            ),
            FakeMx.float32,
        )
        cache.advance(1)
        return FakeArray((1, 1, bench.HIDDEN_SIZE), FakeMx.bfloat16)


def test_parity_probe_continues_after_one_threadgroup_exception():
    layers = [ProbeGdn() for _ in range(bench.EXPECTED_GDN_LAYERS)]

    def kernel(*_args, **kwargs):
        if kwargs["threadgroup_y"] == 32:
            raise RuntimeError("unsupported threadgroup")
        return _kernel(*_args, **kwargs)

    patch = bench.FusedGdnPatch(
        ProbeGdn,
        FakeCache,
        layers,
        FakeMx,
        kernel,
        lambda _cache, _steps: None,
        lambda _cache, _steps: None,
    )
    result = bench.run_real_weight_parity_probe(patch, steps=2)
    assert result["pass"] is True
    assert result["threadgroup_y"] == 16
    assert result["candidate_hits"] == 2
    assert patch.qualified is True


def _sample(mode: str, tps: float, token_hash: str = "same") -> dict[str, Any]:
    candidate = mode == "candidate"
    per_layer = bench.EXPECTED_TOKENS + 1 if candidate else 0
    return {
        "mode": mode,
        "completion_tokens": bench.EXPECTED_TOKENS,
        "token_sha256": token_hash,
        "text_sha256": "text",
        "ttft_s": 1.0 if not candidate else 1.05,
        "elapsed_s": 10.0 if not candidate else 9.5,
        "decode_tps": tps,
        "singleton_batch_delta": 1,
        "fused_hits": bench.EXPECTED_GDN_LAYERS * per_layer,
        "fused_layer_hits": [per_layer] * bench.EXPECTED_GDN_LAYERS,
        "memory": {"active_bytes": 10, "peak_bytes": 20},
        "system_memory": {
            "physical_footprint": {
                "available": True,
                "current_bytes": 30,
                "peak_bytes": 40,
                "error": None,
            },
            "swap": {"available": True, "used_bytes": 100, "error": None},
        },
    }


def _idle_status(*, paused=False):
    return {
        "paused": paused,
        "admitted_requests": 0,
        "running_requests": 0,
        "queued_requests": 0,
    }


def _checkpoint():
    return {
        "phase": "before_engine_construction",
        "mlx": {
            "available": True,
            "active_bytes": 10,
            "peak_bytes": 20,
            "cache_bytes": 5,
            "error": None,
        },
        "aggregate_run_max": {
            "active_bytes": 10,
            "peak_bytes": 20,
            "cache_bytes": 5,
        },
        "physical_footprint": {
            "available": True,
            "current_bytes": 30,
            "peak_bytes": 40,
            "error": None,
        },
        "swap": {"available": True, "used_bytes": 100, "error": None},
    }


def _passing_receipt() -> dict[str, Any]:
    pairs = []
    for index in range(bench.EXPECTED_STRATA):
        order = (
            ["baseline", "candidate", "candidate", "baseline"]
            if index % 2 == 0
            else ["candidate", "baseline", "baseline", "candidate"]
        )
        pairs.append(
            {
                "index": index + 1,
                "category": bench.PROMPTS[index][0],
                "order": order,
                "baseline": [_sample("baseline", 100), _sample("baseline", 100)],
                "candidate": [
                    _sample("candidate", 105),
                    _sample("candidate", 105),
                ],
            }
        )
    return {
        "pairs": pairs,
        "real_weight_parity": {"pass": True, "steps": 32},
        "identity": {"same_model": True, "same_executor": True},
        "configuration": {
            "prefix_cache": False,
            "temperature": 0.0,
            "thinking": False,
            "measured_ignore_eos": True,
        },
        "artifact": {"verified": True},
        "source": {"dirty": False, "source_tree_match": True},
        "media_recovery": {
            "pass": True,
            "stock_before": {"singleton_batch_delta": 1},
            "candidate_image": {"singleton_batch_delta": 1},
            "candidate_text": {"singleton_batch_delta": 1},
            "stock_after": {"singleton_batch_delta": 1},
        },
        "abort_recovery": {
            "cycles": [
                {
                    "cutoff_tokens": (index % 16) + 1,
                    "observed_tokens": (index % 16) + 1,
                    "closed": True,
                    "idle_status": _idle_status(),
                    "recovery": {
                        **_sample("candidate", 10),
                        "completion_tokens": 8,
                        "fused_hits": bench.EXPECTED_GDN_LAYERS * 9,
                        "fused_layer_hits": [9] * bench.EXPECTED_GDN_LAYERS,
                    },
                }
                for index in range(bench.EXPECTED_ABORT_CYCLES)
            ],
            "final_status": _idle_status(),
        },
        "cache_recovery": {
            "allocator_clear_attempted": True,
            "post_clear": _checkpoint(),
            "recovery": {
                **_sample("candidate", 10),
                "completion_tokens": 16,
                "fused_hits": bench.EXPECTED_GDN_LAYERS * 17,
                "fused_layer_hits": [17] * bench.EXPECTED_GDN_LAYERS,
            },
            "final_status": _idle_status(),
        },
        "lifecycle": {
            "pause": _idle_status(paused=True),
            "resume": _idle_status(paused=False),
            "resume_stock_recovery": _sample("baseline", 10),
            "reload": {
                "pause": _idle_status(paused=True),
                "stopped": True,
                "started": True,
                "resume": _idle_status(paused=False),
                "model_replaced": True,
                "executor_replaced": True,
                "parity": {"pass": True},
                "stock_recovery": _sample("baseline", 10),
                "candidate_recovery": {
                    **_sample("candidate", 10),
                    "completion_tokens": 16,
                    "fused_hits": bench.EXPECTED_GDN_LAYERS * 17,
                    "fused_layer_hits": [17] * bench.EXPECTED_GDN_LAYERS,
                },
            },
            "final_stop": {
                "pause": _idle_status(paused=True),
                "completed": True,
                "loaded": False,
            },
        },
        "hardware": {
            "verified": True,
            "expected_memory_gib": 48,
            "expected_chip": "Apple M4 Pro",
        },
        "runtime_contract": {
            "packages": {
                "mlx": bench.EXPECTED_MLX_VERSION,
                "mlx-lm": bench.EXPECTED_MLX_LM_VERSION,
                "mlx-vlm": bench.EXPECTED_MLX_VLM_VERSION,
            },
            "kernel": {
                "source_tree_match": True,
                "source_sha256": bench.EXPECTED_KERNEL_SHA256,
            },
            "stock_method": {"source_sha256": bench.EXPECTED_LANGUAGE_SHA256},
            "cache_class": {"source_sha256": bench.EXPECTED_CACHE_SHA256},
        },
        "memory_checkpoints": {
            name: _checkpoint()
            for name in ("pre", "probe", "peak", "post_clear", "post_stop")
        },
        "errors": [],
    }


def test_all_decision_gates_pass_on_complete_exact_receipt():
    gates = bench.evaluate_gates(_passing_receipt())
    assert gates["pass"] is True
    assert all(gates["checks"].values())


def test_parent_v1_gate_keys_remain_additively_compatible():
    parent_v1 = {
        "six_complete_prompt_strata",
        "all_twenty_four_samples_are_256_tokens",
        "median_decode_speedup_gte_1_03",
        "five_of_six_strata_positive",
        "median_wall_speedup_gte_1_03",
        "five_of_six_wall_strata_positive",
        "paired_ratio_cv_lte_0_05",
        "paired_wall_ratio_cv_lte_0_05",
        "exact_token_ids_all_runs_per_stratum",
        "median_ttft_ratio_lte_1_10",
        "active_delta_lte_64_mib",
        "isolated_peak_delta_lte_64_mib",
        "exact_candidate_hits_per_layer_and_zero_baseline",
        "singleton_fastpath_engaged",
        "real_weight_32_step_bit_exact",
        "same_model_and_executor",
        "prefix_cache_disabled",
        "greedy_thinking_off",
        "measured_ignore_eos_enabled",
        "artifact_exact_b0_verified",
        "source_clean",
        "source_tree_match",
        "vlm_candidate_and_stock_recovery",
        "no_errors",
    }
    assert parent_v1 <= set(bench.evaluate_gates(_passing_receipt())["checks"])
    assert bench.evaluate_gates(_passing_receipt(), 64 * bench.MIB)["pass"] is True


@pytest.mark.parametrize(
    ("field", "legacy_gate", "new_gate"),
    [
        (
            "active_bytes",
            "active_delta_lte_64_mib",
            "active_delta_lte_max_512_mib_or_3_percent",
        ),
        (
            "peak_bytes",
            "isolated_peak_delta_lte_64_mib",
            "isolated_peak_delta_lte_max_512_mib_or_3_percent",
        ),
    ],
)
def test_schema_v1_memory_limit_remains_independent(field, legacy_gate, new_gate):
    receipt = _passing_receipt()
    receipt["pairs"][0]["candidate"][0]["memory"][field] += 100 * bench.MIB

    gates = bench.evaluate_gates(receipt, 64 * bench.MIB)
    assert gates["checks"][new_gate] is True
    assert gates["checks"][legacy_gate] is False
    assert gates["pass"] is False

    relaxed_legacy = bench.evaluate_gates(receipt, 128 * bench.MIB)
    assert relaxed_legacy["checks"][legacy_gate] is True
    assert relaxed_legacy["pass"] is True


def test_cleanup_residuals_allow_equal_nonzero_imported_process_baseline():
    receipt = _passing_receipt()
    checkpoints = receipt["memory_checkpoints"]
    baseline = 700 * bench.MIB
    run_peak = 20 * bench.GIB
    checkpoints["pre"]["mlx"].update(
        active_bytes=baseline, peak_bytes=baseline, cache_bytes=baseline
    )
    checkpoints["pre"]["physical_footprint"].update(
        current_bytes=baseline, peak_bytes=baseline
    )
    checkpoints["peak"]["mlx"].update(
        active_bytes=run_peak, peak_bytes=run_peak, cache_bytes=bench.GIB
    )
    checkpoints["peak"]["aggregate_run_max"].update(
        active_bytes=run_peak, peak_bytes=run_peak, cache_bytes=bench.GIB
    )
    checkpoints["peak"]["physical_footprint"].update(
        current_bytes=run_peak, peak_bytes=run_peak
    )
    checkpoints["post_stop"]["mlx"].update(
        active_bytes=baseline, peak_bytes=run_peak, cache_bytes=baseline
    )
    checkpoints["post_stop"]["physical_footprint"].update(
        current_bytes=baseline, peak_bytes=run_peak
    )

    gates = bench.evaluate_gates(receipt)
    assert gates["checks"]["post_stop_mlx_active_within_cleanup_bound"] is True
    assert gates["checks"]["post_stop_mlx_cache_within_cleanup_bound"] is True
    assert gates["checks"]["post_stop_footprint_within_cleanup_bound"] is True
    assert gates["checks"]["stop_does_not_raise_mlx_peak_beyond_allowance"] is True
    assert (
        gates["checks"]["stop_does_not_raise_footprint_peak_beyond_allowance"]
        is True
    )
    assert gates["metrics"]["post_stop_mlx_active_residual_bytes"] == 0
    assert gates["metrics"]["post_stop_mlx_cache_residual_bytes"] == 0
    assert gates["metrics"]["post_stop_physical_footprint_residual_bytes"] == 0
    assert gates["pass"] is True


@pytest.mark.parametrize(
    ("mutate", "failed_gate"),
    [
        (
            lambda receipt: receipt["pairs"][0]["candidate"][0][
                "fused_layer_hits"
            ].__setitem__(0, 256),
            "exact_candidate_hits_per_layer_and_zero_baseline",
        ),
        (
            lambda receipt: receipt["pairs"][0]["candidate"][0].__setitem__(
                "completion_tokens", 255
            ),
            "all_twenty_four_samples_are_256_tokens",
        ),
        (
            lambda receipt: receipt["real_weight_parity"].__setitem__("pass", False),
            "real_weight_32_step_bit_exact",
        ),
        (
            lambda receipt: receipt["source"].__setitem__("dirty", True),
            "source_clean",
        ),
        (
            lambda receipt: receipt["media_recovery"].__setitem__("pass", False),
            "vlm_candidate_and_stock_recovery",
        ),
        (
            lambda receipt: receipt["pairs"][0]["baseline"][0].__setitem__(
                "singleton_batch_delta", 0
            ),
            "singleton_fastpath_engaged",
        ),
        (
            lambda receipt: receipt["media_recovery"]["candidate_image"].__setitem__(
                "singleton_batch_delta", 0
            ),
            "singleton_fastpath_engaged",
        ),
        (
            lambda receipt: [
                sample.__setitem__("elapsed_s", 11.0)
                for pair in receipt["pairs"]
                for sample in pair["candidate"]
            ],
            "median_wall_speedup_gte_1_05",
        ),
        (
            lambda receipt: receipt["pairs"][0]["candidate"][0].__setitem__(
                "elapsed_s", 5.0
            ),
            "paired_wall_ratio_cv_lte_0_01",
        ),
        (
            lambda receipt: receipt["abort_recovery"]["final_status"].__setitem__(
                "running_requests", 1
            ),
            "fifty_randomized_abort_recovery_cycles",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "physical_footprint"
            ].__setitem__("available", False),
            "all_five_physical_footprint_checkpoints_available",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "mlx"
            ].__setitem__("active_bytes", 60 * bench.GIB),
            "post_stop_mlx_active_within_cleanup_bound",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "mlx"
            ].__setitem__("cache_bytes", 60 * bench.GIB),
            "post_stop_mlx_cache_within_cleanup_bound",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "mlx"
            ].__setitem__("peak_bytes", 64 * bench.GIB),
            "stop_does_not_raise_mlx_peak_beyond_allowance",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "physical_footprint"
            ].update(
                current_bytes=60 * bench.GIB,
                peak_bytes=64 * bench.GIB,
            ),
            "post_stop_footprint_within_cleanup_bound",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "physical_footprint"
            ].__setitem__("peak_bytes", 64 * bench.GIB),
            "stop_does_not_raise_footprint_peak_beyond_allowance",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["post_stop"][
                "swap"
            ].__setitem__("used_bytes", 32 * bench.GIB),
            "end_to_end_extra_swap_lte_256_mib",
        ),
        (
            lambda receipt: receipt["pairs"][0].__setitem__(
                "order", ["candidate", "baseline", "baseline", "candidate"]
            ),
            "exact_abba_baab_order_index_category",
        ),
        (
            lambda receipt: receipt["pairs"][1].__setitem__("index", 99),
            "exact_abba_baab_order_index_category",
        ),
        (
            lambda receipt: receipt["pairs"][2].__setitem__("category", "wrong"),
            "exact_abba_baab_order_index_category",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["pre"].__setitem__(
                "phase", "after_engine_start"
            ),
            "pre_memory_captured_before_engine_construction",
        ),
        (
            lambda receipt: receipt["memory_checkpoints"]["peak"].pop(
                "aggregate_run_max"
            ),
            "aggregate_mlx_peak_covers_run",
        ),
        (
            lambda receipt: receipt["runtime_contract"]["packages"].__setitem__(
                "mlx", "0.0.0"
            ),
            "exact_pinned_runtime_contract",
        ),
        (
            lambda receipt: receipt["hardware"].update(
                expected_memory_gib=64, expected_chip="Apple M4 Pro"
            ),
            "target_48_or_64_gib_hardware",
        ),
    ],
)
def test_decision_gates_fail_closed(mutate, failed_gate):
    receipt = _passing_receipt()
    mutate(receipt)
    gates = bench.evaluate_gates(receipt)
    assert gates["pass"] is False
    assert gates["checks"][failed_gate] is False


def _media_sample(mode: str, *, completion_tokens=41):
    sample = _sample(mode, 10.0)
    sample.update(
        {
            "completion_tokens": completion_tokens,
            "token_sha256": "image-token-hash",
            "text_sha256": "image-text-hash",
            "text": "A cheetah is visible.",
        }
    )
    if mode == "candidate":
        per_layer = completion_tokens + 1
        sample["fused_layer_hits"] = [per_layer] * bench.EXPECTED_GDN_LAYERS
        sample["fused_hits"] = per_layer * bench.EXPECTED_GDN_LAYERS
    return sample


def _passing_media():
    return {
        "image": {"width": 1920, "height": 1080},
        "expected": "cheetah",
        "candidate_text_expected": "100",
        "stock_before": _media_sample("baseline"),
        "candidate_image": _media_sample("candidate"),
        "candidate_text": {
            **_media_sample(
                "candidate", completion_tokens=bench.EXPECTED_LONG_TEXT_TOKENS
            ),
            "text": "The original price is 100.",
        },
        "stock_after": _media_sample("baseline"),
    }


def test_media_sequence_requires_exact_candidate_and_reversible_stock():
    assert bench.media_sequence_pass(_passing_media()) is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda media: media["candidate_image"]["fused_layer_hits"].__setitem__(0, 1),
        lambda media: media["candidate_text"]["fused_layer_hits"].__setitem__(0, 1),
        lambda media: media["stock_after"].__setitem__("fused_hits", 1),
        lambda media: media["stock_after"].__setitem__("token_sha256", "different"),
        lambda media: media.__setitem__("expected", "   "),
        lambda media: media["candidate_text"].__setitem__(
            "text", "The arithmetic answer is 80."
        ),
    ],
)
def test_media_sequence_fails_closed(mutation):
    media = _passing_media()
    mutation(media)
    assert bench.media_sequence_pass(media) is False


def test_pinned_installed_gdn_and_cache_abi_provenance():
    metadata = pytest.importorskip("importlib.metadata")
    if metadata.version("mlx-vlm") != bench.EXPECTED_MLX_VLM_VERSION:
        pytest.skip("requires the pinned benchmark mlx-vlm environment")
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    provenance = bench._stock_method_provenance(Qwen3_5GatedDeltaNet)
    assert provenance["source_sha256"] == bench.EXPECTED_LANGUAGE_SHA256
    cache_provenance = bench._cache_class_provenance(ArraysCache)
    assert cache_provenance["source_sha256"] == bench.EXPECTED_CACHE_SHA256
    assert ArraysCache.__module__ == bench.EXPECTED_CACHE_MODULE
    assert len(ArraysCache(size=2).cache) == 2


def test_pinned_kernel_source_hash_matches_contract():
    kernel = (
        bench.SCRIPT.parent.parent
        / "rapid_mlx"
        / "kernels"
        / "qwen4_fused_gdn_decode.py"
    )
    assert (
        hashlib.sha256(kernel.read_bytes()).hexdigest() == bench.EXPECTED_KERNEL_SHA256
    )


def test_validate_args_rejects_blank_image_expect(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    args = argparse.Namespace(
        model=model,
        image_path=image,
        image_expect="  ",
        max_memory_delta_mib=64,
    )
    with pytest.raises(SystemExit):
        bench._validate_args(bench.build_parser(), args)


def test_mlx_probe_records_unavailable_instead_of_fabricating_zero():
    probe = bench._mlx_memory_snapshot(SimpleNamespace())
    assert probe["available"] is False
    assert "unavailable" in probe["error"]
    assert "active_bytes" not in probe


def test_footprint_probe_parses_current_and_peak(monkeypatch):
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(bench.shutil, "which", lambda _name: "/usr/bin/footprint")

    def runner(*_args, **_kwargs):
        return SimpleNamespace(
            stdout="Footprint: 1.5 GB\nphys_footprint_peak: 2048 MB\n"
        )

    probe = bench._process_footprint_snapshot(runner=runner)
    assert probe == {
        "available": True,
        "current_bytes": int(1.5 * bench.GIB),
        "peak_bytes": 2048 * bench.MIB,
        "error": None,
    }


def test_monotonic_swap_growth_is_rejected_but_plateau_is_not():
    assert bench._no_monotonic_growth([1, 2, 2, 3]) is False
    assert bench._no_monotonic_growth([1, 1, 1, 1]) is True
    assert bench._no_monotonic_growth([1, 2, 1, 2]) is True


def test_raw_receipt_writer_never_replaces_existing_file(tmp_path: Path):
    output = tmp_path / "receipt.json"
    bench._write_new_receipt(output, {"pass": False})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        bench._write_new_receipt(output, {"pass": True})
    assert output.read_bytes() == original


@pytest.mark.asyncio
async def test_fifty_abort_cycles_close_streams_and_recover_with_fakes(monkeypatch):
    class FakePatch:
        def set_candidate(self, _enabled):
            return None

    class FakeEngine:
        def stream_chat(self, **kwargs):
            async def stream():
                for index in range(kwargs["max_tokens"]):
                    yield SimpleNamespace(tokens=[index], completion_tokens=index + 1)

            return stream()

        def lifecycle_status(self):
            return _idle_status()

    async def fake_sample(*_args, **_kwargs):
        return {
            **_sample("candidate", 10),
            "completion_tokens": 8,
            "fused_hits": bench.EXPECTED_GDN_LAYERS * 9,
            "fused_layer_hits": [9] * bench.EXPECTED_GDN_LAYERS,
        }

    monkeypatch.setattr(bench, "_run_sample", fake_sample)

    async def fake_memory(*_args, **_kwargs):
        return {"active_bytes": 1, "peak_bytes": 2, "cache_bytes": 0}

    monkeypatch.setattr(bench.common, "_worker_memory", fake_memory)
    result = await bench.run_abort_recovery(FakeEngine(), FakePatch(), timeout=0.1)
    assert result["pass"] is True
    assert len(result["cycles"]) == 50
    assert all(cycle["closed"] for cycle in result["cycles"])
    assert bench._request_counts_zero(result["final_status"])


def test_cli_rejects_cross_paired_hardware_before_touching_paths():
    class PoisonPath:
        def is_dir(self):
            raise AssertionError("model path was touched")

    args = argparse.Namespace(
        model=PoisonPath(),
        image_path=PoisonPath(),
        image_expect="expected",
        expected_memory_gib=64,
        expected_chip="Apple M4 Pro",
        lifecycle_timeout=120.0,
        max_memory_delta_mib=None,
        output=PoisonPath(),
    )
    with pytest.raises(SystemExit):
        bench._validate_args(bench.build_parser(), args)


def test_cli_accepts_legacy_max_memory_delta_spelling():
    args = bench.build_parser().parse_args(
        [
            "--model",
            "/model",
            "--image-path",
            "/image.png",
            "--image-expect",
            "subject",
            "--output",
            "/receipt.json",
            "--expected-memory-gib",
            "48",
            "--expected-chip",
            "Apple M4 Pro",
            "--max-memory-delta-mib",
            "64",
        ]
    )
    assert args.max_memory_delta_mib == 64


@pytest.mark.asyncio
async def test_hardware_mismatch_precedes_artifact_or_engine_path(monkeypatch):
    touched = False

    def inspect_artifact(_model):
        nonlocal touched
        touched = True
        raise AssertionError("artifact path was touched")

    monkeypatch.setattr(
        bench,
        "_hardware_snapshot",
        lambda *_args: {
            "expected_memory_gib": 48,
            "expected_chip": "Apple M4 Pro",
            "physical_memory_bytes": 64 * bench.GIB,
            "chip": "Apple M1 Max",
            "verified": False,
            "errors": [],
        },
    )
    monkeypatch.setattr(bench.common, "_inspect_exact_artifact", inspect_artifact)
    args = argparse.Namespace(
        model=object(), expected_memory_gib=48, expected_chip="Apple M4 Pro"
    )
    with pytest.raises(bench.QualificationRunError) as failure:
        await bench.run_benchmark(args)
    assert touched is False
    assert failure.value.receipt["stage"] == "hardware_preflight"
    assert failure.value.receipt["hardware"]["verified"] is False


@pytest.mark.parametrize("mismatched", ["mlx", "mlx-lm", "mlx-vlm"])
@pytest.mark.asyncio
async def test_runtime_version_mismatch_precedes_artifact_load(monkeypatch, mismatched):
    expected = {
        "mlx": bench.EXPECTED_MLX_VERSION,
        "mlx-lm": bench.EXPECTED_MLX_LM_VERSION,
        "mlx-vlm": bench.EXPECTED_MLX_VLM_VERSION,
    }
    monkeypatch.setattr(
        bench,
        "_hardware_snapshot",
        lambda *_args: {
            "expected_memory_gib": 48,
            "expected_chip": "Apple M4 Pro",
            "physical_memory_bytes": 48 * bench.GIB,
            "chip": "Apple M4 Pro",
            "verified": True,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        bench.importlib.metadata,
        "version",
        lambda package: "0.0.0" if package == mismatched else expected[package],
    )
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(
            stdout="deadbeef\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        ),
    )
    monkeypatch.setattr(
        bench.common,
        "_inspect_exact_artifact",
        lambda _model: pytest.fail("artifact must not be inspected"),
    )
    args = argparse.Namespace(
        model=object(), expected_memory_gib=48, expected_chip="Apple M4 Pro"
    )
    with pytest.raises(bench.QualificationRunError) as failure:
        await bench.run_benchmark(args)
    assert failure.value.receipt["stage"] == "runtime_preflight"
    assert mismatched in failure.value.receipt["errors"][0]


@pytest.mark.asyncio
async def test_partial_receipt_preserves_primary_and_cleanup_errors(monkeypatch):
    async def fail(_args, state):
        state.update(
            stage="abort_recovery",
            hardware={"verified": True},
            source={"dirty": False, "source_tree_match": True},
            memory_checkpoints={"pre": _checkpoint()},
            cleanup_errors=[
                {
                    "type": "OSError",
                    "message": (
                        "cleanup failed at /private/tmp/q38-secret/cache.bin "
                        "using token=cleanup-secret"
                    ),
                }
            ],
        )
        raise ValueError(
            "primary at /private/tmp/q38-secret/model.bin via "
            "https://alice:password@private.example/run?token=url-secret "
            "with api_key=loose-secret"
        )

    monkeypatch.setattr(bench, "_run_benchmark_impl", fail)
    with pytest.raises(bench.QualificationRunError) as failure:
        await bench.run_benchmark(SimpleNamespace())
    receipt = failure.value.receipt
    assert receipt["stage"] == "abort_recovery"
    assert receipt["failure"]["primary"]["type"] == "ValueError"
    assert receipt["failure"]["cleanup"][0]["type"] == "OSError"
    serialized = str(receipt)
    for secret in (
        "/private/tmp/q38-secret",
        "alice:password",
        "private.example",
        "url-secret",
        "loose-secret",
        "cleanup-secret",
    ):
        assert secret not in serialized
    assert "<redacted-path>" in serialized
    assert "<redacted-url>" in serialized
    assert "token=<redacted>" in serialized
    assert "api_key=<redacted>" in serialized
    assert receipt["gates"]["pass"] is False
