# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from vllm_mlx.kernels import qwen4_fused_gdn_decode as fused_gdn
from vllm_mlx.models import qwen4_exp


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeCache:
    def __init__(self, conv_state=None, recurrent_state=None):
        self.cache = [conv_state, recurrent_state]
        self.lengths = None
        self.advanced = 0

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    def advance(self, amount):
        self.advanced += amount


def production_values(dtype=mx.bfloat16):
    return {
        "qkv": FakeArray((1, 1, 10240), dtype),
        "z": FakeArray((1, 1, 6144), dtype),
        "beta": FakeArray((1, 1, 48), dtype),
        "alpha": FakeArray((1, 1, 48), dtype),
        "conv_state": FakeArray((1, 3, 10240), dtype),
        "recurrent_state": FakeArray((1, 48, 128, 128), mx.float32),
        "conv_weight": FakeArray((10240, 4, 1), dtype),
        "a_log": FakeArray((48,), mx.float32),
        "dt_bias": FakeArray((48,), dtype),
        "norm_weight": FakeArray((128,), dtype),
    }


def admission(**overrides):
    values = production_values()
    values.update(overrides)
    return fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        mask=None,
        cache_lengths=None,
        record_rollback=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="sigmoid",
    )


def tiny_args():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1.0e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )


class Identity:
    def __call__(self, value):
        return value


def test_production_single_token_decode_is_admitted():
    result = admission()
    assert result.accepted, result.reason


def test_batch_prefill_mask_ragged_and_speculation_fall_back():
    result = admission(qkv=FakeArray((2, 1, 10240), mx.bfloat16))
    assert not result.accepted
    assert "qkv shape" in result.reason

    values = production_values()
    base = {
        **values,
        "training": False,
        "sharded": False,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "conv_kernel": 4,
        "gate_activation": "sigmoid",
    }
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **base,
        mask=object(),
        cache_lengths=None,
        record_rollback=False,
    )
    assert result.reason == "masked decode"
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **base,
        mask=None,
        cache_lengths=object(),
        record_rollback=False,
    )
    assert result.reason == "ragged cache lengths"
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **base,
        mask=None,
        cache_lengths=None,
        record_rollback=True,
    )
    assert result.reason == "speculative rollback"


def test_dtype_and_geometry_are_strict():
    result = admission(a_log=FakeArray((48,), mx.float16))
    assert not result.accepted
    assert "A_log" in result.reason
    assert admission(a_log=FakeArray((48,), mx.bfloat16)).accepted

    values = production_values()
    result = fused_gdn.admit_qwen4_fused_gdn_decode(
        **values,
        mask=None,
        cache_lengths=None,
        record_rollback=False,
        training=False,
        sharded=False,
        num_key_heads=24,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="sigmoid",
    )
    assert not result.accepted
    assert "unsupported geometry" in result.reason


def test_admission_rejects_training_gate_and_state_dtypes():
    values = production_values()
    base = {
        **values,
        "mask": None,
        "cache_lengths": None,
        "record_rollback": False,
        "training": False,
        "sharded": False,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "key_head_dim": 128,
        "value_head_dim": 128,
        "conv_kernel": 4,
        "gate_activation": "sigmoid",
    }

    assert (
        fused_gdn.admit_qwen4_fused_gdn_decode(**{**base, "training": True}).reason
        == "training"
    )
    assert (
        fused_gdn.admit_qwen4_fused_gdn_decode(
            **{**base, "gate_activation": "silu"}
        ).reason
        == "output gate 'silu'"
    )
    assert (
        "activation dtype"
        in fused_gdn.admit_qwen4_fused_gdn_decode(
            **{**base, "qkv": FakeArray((1, 1, 10240), mx.float16)}
        ).reason
    )
    assert (
        "z dtype"
        in fused_gdn.admit_qwen4_fused_gdn_decode(
            **{**base, "z": FakeArray((1, 1, 6144), mx.float16)}
        ).reason
    )
    assert (
        fused_gdn.admit_qwen4_fused_gdn_decode(
            **{
                **base,
                "recurrent_state": FakeArray((1, 48, 128, 128), mx.bfloat16),
            }
        ).reason
        == "recurrent_state must be float32"
    )


def test_kernel_dispatch_is_one_threadgroup_per_value_head():
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return [
            FakeArray(shape, dtype)
            for shape, dtype in zip(
                kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
            )
        ]

    values = production_values()
    with patch.object(fused_gdn, "_kernel", return_value=fake_kernel):
        outputs = fused_gdn.qwen4_fused_gdn_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            values["conv_state"],
            values["conv_weight"],
            values["a_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=16,
        )
    assert calls[0]["grid"] == (32, 16, 48)
    assert calls[0]["threadgroup"] == (32, 16, 1)
    assert outputs[0].shape == (1, 1, 6144)
    assert outputs[1].shape == (1, 3, 10240)
    assert outputs[2].shape == (1, 48, 128, 128)
    assert ("RATIO", 3) in calls[0]["template"]

    with pytest.raises(ValueError, match="unsupported threadgroup_y"):
        fused_gdn.qwen4_fused_gdn_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            values["conv_state"],
            values["conv_weight"],
            values["a_log"],
            values["dt_bias"],
            values["recurrent_state"],
            values["norm_weight"],
            1.0e-6,
            threadgroup_y=2,
        )


def test_runtime_capability_and_probe_fail_closed_without_metal():
    assert isinstance(fused_gdn.fused_gdn_runtime_supported(), bool)

    with (
        patch.object(fused_gdn, "_PROBE_COMPLETE", True),
        patch.object(fused_gdn, "_PROBED_THREADGROUP_Y", 8),
    ):
        assert fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16) == 8

    with (
        patch.object(fused_gdn, "_PROBE_COMPLETE", False),
        patch.object(fused_gdn, "_PROBED_THREADGROUP_Y", None),
        patch.object(fused_gdn, "_PROBE_LOCK", Lock()),
        patch.object(fused_gdn, "fused_gdn_runtime_supported", return_value=False),
    ):
        assert fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16) is None


def test_probe_skips_threadgroup_resource_errors_and_stops_on_other_value_error():
    with (
        patch.object(fused_gdn, "_PROBE_COMPLETE", False),
        patch.object(fused_gdn, "_PROBED_THREADGROUP_Y", None),
        patch.object(fused_gdn, "_PROBE_LOCK", Lock()),
        patch.object(fused_gdn, "fused_gdn_runtime_supported", return_value=True),
        patch.object(fused_gdn.mx, "zeros", return_value=object()),
        patch.object(fused_gdn.mx, "ones", return_value=object()),
        patch.object(
            fused_gdn,
            "qwen4_fused_gdn_decode",
            side_effect=[
                ValueError("threads per threadgroup exceeded"),
                ValueError("kernel source rejected"),
            ],
        ) as execute,
    ):
        assert fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16) is None

    assert [call.kwargs["threadgroup_y"] for call in execute.call_args_list] == [32, 16]


def test_concurrent_probe_publishes_only_after_initialization():
    entered = Event()
    release = Event()
    calls = 0

    def blocking_kernel(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return (object(), object(), object())

    with (
        patch.object(fused_gdn, "_PROBE_COMPLETE", False),
        patch.object(fused_gdn, "_PROBED_THREADGROUP_Y", None),
        patch.object(fused_gdn, "_PROBE_LOCK", Lock()),
        patch.object(fused_gdn, "fused_gdn_runtime_supported", return_value=True),
        patch.object(fused_gdn, "qwen4_fused_gdn_decode", side_effect=blocking_kernel),
        patch.object(fused_gdn.mx, "eval"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(fused_gdn.probe_qwen4_fused_gdn_decode, mx.bfloat16)
        assert entered.wait(timeout=5)
        second = pool.submit(fused_gdn.probe_qwen4_fused_gdn_decode, mx.bfloat16)
        release.set()
        assert first.result(timeout=5) == 32
        assert second.result(timeout=5) == 32

    assert calls == 1


def test_probe_tries_smaller_threadgroup_after_runtime_error():
    successful_outputs = (object(), object(), object())
    with (
        patch.object(fused_gdn, "_PROBE_COMPLETE", False),
        patch.object(fused_gdn, "_PROBED_THREADGROUP_Y", None),
        patch.object(fused_gdn, "_PROBE_LOCK", Lock()),
        patch.object(fused_gdn, "fused_gdn_runtime_supported", return_value=True),
        patch.object(
            fused_gdn,
            "qwen4_fused_gdn_decode",
            side_effect=[RuntimeError("threadgroup resources"), successful_outputs],
        ) as execute,
        patch.object(fused_gdn.mx, "eval"),
    ):
        assert fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16) == 16

    assert [call.kwargs["threadgroup_y"] for call in execute.call_args_list] == [32, 16]


def test_resident_switch_preserves_weights_and_defaults_stock():
    with patch.object(qwen4_exp, "_FUSED_GDN_DEFAULT", False):
        layer = qwen4_exp.GatedDeltaNet(tiny_args())
    weight = layer.conv1d.weight
    assert qwen4_exp.qwen4_fused_gdn_mode_counts(layer) == {
        "stock": 1,
        "fused": 0,
    }
    assert qwen4_exp.set_qwen4_fused_gdn_mode(layer, "fused") == 1
    assert layer.conv1d.weight is weight
    assert layer.fused_gdn_decode_mode == "fused"
    assert qwen4_exp.set_qwen4_fused_gdn_mode(layer, "stock") == 1
    assert layer.conv1d.weight is weight

    with pytest.raises(ValueError, match="unknown fused GDN decode mode"):
        layer.set_fused_gdn_decode_mode("invalid")
    with pytest.raises(ValueError, match="unknown fused GDN decode mode"):
        qwen4_exp.set_qwen4_fused_gdn_mode(layer, "invalid")

    layer.fused_gdn_decode_calls = 3
    layer.fused_gdn_decode_fallbacks = 2
    layer.fused_gdn_decode_last_fallback = "Metal runtime unavailable"
    assert qwen4_exp.qwen4_fused_gdn_stats(layer) == {
        "fused_calls": 3,
        "fallbacks": 2,
        "last_fallbacks": {"Metal runtime unavailable": 1},
    }


def test_uninitialized_and_speculative_cache_do_not_probe_metal():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    values = production_values()
    with patch.object(qwen4_exp, "fused_gdn_runtime_supported") as runtime:
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            FakeCache(),
            record_rollback=False,
        )
    assert result is None
    runtime.assert_not_called()
    assert layer.fused_gdn_decode_last_fallback == "uninitialized cache"

    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    result = layer._try_fused_decode(
        values["qkv"],
        values["z"],
        values["beta"],
        values["alpha"],
        None,
        cache,
        record_rollback=True,
    )
    assert result is None
    assert layer.fused_gdn_decode_last_fallback == "speculative rollback"


def test_sharded_layer_falls_back_before_probe():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    layer.sharding_group = object()
    values = production_values()
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    with patch.object(qwen4_exp, "probe_qwen4_fused_gdn_decode") as probe:
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
            record_rollback=False,
        )
    assert result is None
    probe.assert_not_called()
    assert layer.fused_gdn_decode_last_fallback == "distributed sharding"


def test_runtime_and_probe_declines_preserve_cache():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    values = production_values()
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")

    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=False),
    ):
        assert (
            layer._try_fused_decode(
                values["qkv"],
                values["z"],
                values["beta"],
                values["alpha"],
                None,
                cache,
                record_rollback=False,
            )
            is None
        )
    assert layer.fused_gdn_decode_last_fallback == "Metal runtime unavailable"

    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=None),
    ):
        assert (
            layer._try_fused_decode(
                values["qkv"],
                values["z"],
                values["beta"],
                values["alpha"],
                None,
                cache,
                record_rollback=False,
            )
            is None
        )
    assert layer.fused_gdn_decode_last_fallback == "Metal kernel probe declined"


def test_call_returns_fused_result_before_stock_path():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    inputs = FakeArray((1, 1, 16), mx.bfloat16)
    fused = object()
    layer.in_proj_qkv = Identity()
    layer.in_proj_z = Identity()
    layer.in_proj_b = Identity()
    layer.in_proj_a = Identity()
    with patch.object(layer, "_try_fused_decode", return_value=fused):
        assert layer.__call__(inputs) is fused


def test_admitted_path_updates_cache_and_counter_without_real_kernel():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    layer.out_proj = Identity()
    values = production_values()
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    fused_output = FakeArray((1, 1, 6144), mx.bfloat16)
    next_conv = object()
    next_state = object()
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8),
        patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            return_value=(fused_output, next_conv, next_state),
        ) as execute,
    ):
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
            record_rollback=False,
        )
    assert result is fused_output
    assert cache[0] is next_conv
    assert cache[1] is next_state
    assert cache.advanced == 1
    assert layer.fused_gdn_decode_calls == 1
    assert execute.call_args.kwargs["threadgroup_y"] == 8


def test_synchronous_dispatch_failure_preserves_cache():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    values = production_values()
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(qwen4_exp, "probe_qwen4_fused_gdn_decode", return_value=8),
        patch.object(
            qwen4_exp,
            "qwen4_fused_gdn_decode",
            side_effect=RuntimeError("dispatch rejected"),
        ),
    ):
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
            record_rollback=False,
        )
    assert result is None
    assert cache[0] is values["conv_state"]
    assert cache[1] is values["recurrent_state"]
    assert cache.advanced == 0
    assert layer.fused_gdn_decode_calls == 0
    assert layer.fused_gdn_decode_last_fallback == (
        "Metal kernel dispatch failed: RuntimeError"
    )


def test_probe_exception_preserves_cache():
    layer = qwen4_exp.GatedDeltaNet(tiny_args())
    layer.eval()
    layer.set_fused_gdn_decode_mode("fused")
    values = production_values()
    cache = FakeCache(values["conv_state"], values["recurrent_state"])
    accepted = fused_gdn.FusedGdnAdmission(True, "eligible")
    with (
        patch.object(qwen4_exp, "admit_qwen4_fused_gdn_decode", return_value=accepted),
        patch.object(qwen4_exp, "fused_gdn_runtime_supported", return_value=True),
        patch.object(
            qwen4_exp,
            "probe_qwen4_fused_gdn_decode",
            side_effect=ValueError("probe rejected"),
        ),
        patch.object(qwen4_exp, "qwen4_fused_gdn_decode") as execute,
    ):
        result = layer._try_fused_decode(
            values["qkv"],
            values["z"],
            values["beta"],
            values["alpha"],
            None,
            cache,
            record_rollback=False,
        )
    assert result is None
    execute.assert_not_called()
    assert cache[0] is values["conv_state"]
    assert cache[1] is values["recurrent_state"]
    assert cache.advanced == 0
    assert layer.fused_gdn_decode_calls == 0
    assert layer.fused_gdn_decode_last_fallback == (
        "Metal kernel dispatch failed: ValueError"
    )
