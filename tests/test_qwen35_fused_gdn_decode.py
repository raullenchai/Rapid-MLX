# SPDX-License-Identifier: Apache-2.0
"""Contracts for Qwen3.5-family fused GDN single-token decode."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

import mlx.nn as nn

from vllm_mlx import gdn_in_proj_fusion
from vllm_mlx import qwen35_fused_gdn_decode as fused
from vllm_mlx.kernels import qwen4_fused_gdn_decode as shared_kernel


class _Cache:
    def __init__(self):
        self.cache = [
            mx.zeros((1, 3, fused._CONV_DIM), dtype=mx.bfloat16),
            mx.zeros((1, 32, 128, 128), dtype=mx.float32),
        ]
        self.lengths = None
        self.advanced = 0

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    def advance(self, amount):
        self.advanced += amount


def _layer(**overrides):
    values = {
        fused._TAG: True,
        "training": False,
        "hidden_size": 2048,
        "sharding_group": None,
        "in_proj_fused": object(),
        "num_k_heads": 16,
        "num_v_heads": 32,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "conv_kernel_size": 4,
        "conv1d": SimpleNamespace(
            weight=mx.zeros((fused._CONV_DIM, 4, 1), dtype=mx.bfloat16)
        ),
        "A_log": mx.zeros((32,), dtype=mx.bfloat16),
        "dt_bias": mx.zeros((32,), dtype=mx.bfloat16),
        "norm": SimpleNamespace(weight=mx.ones((128,), dtype=mx.bfloat16), eps=1e-6),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_admission_is_single_token_bf16_and_complete_cache_only():
    layer = _layer()
    cache = _Cache()
    decode = mx.zeros((1, 1, 2048), dtype=mx.bfloat16)
    assert fused._eligible(layer, decode, None, cache)
    assert not fused._eligible(
        layer, mx.zeros((1, 2, 2048), dtype=mx.bfloat16), None, cache
    )
    assert not fused._eligible(
        layer, mx.zeros((1, 1, 2048), dtype=mx.float16), None, cache
    )
    assert not fused._eligible(layer, decode, mx.ones((1, 1)), cache)
    cache.lengths = mx.array([1])
    assert not fused._eligible(layer, decode, None, cache)

    class BrokenInput:
        @property
        def shape(self):
            raise OSError("shape unavailable")

    assert not fused._eligible(layer, BrokenInput(), None, _Cache())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_v_heads", 48),
        ("head_k_dim", 64),
        ("conv_kernel_size", 3),
        ("hidden_size", 4096),
        ("in_proj_fused", None),
    ],
)
def test_structural_gate_rejects_unknown_geometry(field, value):
    layer = _layer(**{field: value})
    if field == "in_proj_fused":
        delattr(layer, field)
    assert not fused._structurally_eligible(layer)


def test_structural_gate_fails_closed_on_unknown_object():
    assert not fused._structurally_eligible(object())


def test_install_honors_kill_switch(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN35_FUSED_GDN_DECODE", "0")
    assert fused.install_qwen35_fused_gdn_decode(object()) == 0


def test_install_ignores_non_module_placeholder():
    assert fused.install_qwen35_fused_gdn_decode(object()) == 0


def test_install_fails_closed_when_probe_raises(monkeypatch):
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    layer = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(layer)
    for name, value in vars(_layer()).items():
        if name not in {"training", fused._TAG}:
            setattr(layer, name, value)
    layer.eval()
    model = SimpleNamespace(named_modules=lambda: iter((("gdn", layer),)))
    monkeypatch.setattr(
        fused,
        "probe_qwen35_fused_gdn_decode",
        lambda: (_ for _ in ()).throw(OSError("probe unavailable")),
    )

    assert fused.install_qwen35_fused_gdn_decode(model) == 0
    assert not getattr(layer, fused._TAG, False)


def test_install_fails_closed_when_probe_has_no_exact_candidate(monkeypatch):
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    layer = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(layer)
    for name, value in vars(_layer()).items():
        if name not in {"training", fused._TAG}:
            setattr(layer, name, value)
    layer.eval()
    model = SimpleNamespace(named_modules=lambda: iter((("gdn", layer),)))
    monkeypatch.setattr(fused, "probe_qwen35_fused_gdn_decode", lambda: None)

    assert fused.install_qwen35_fused_gdn_decode(model) == 0
    assert not getattr(layer, fused._TAG, False)


def test_install_tags_only_exact_qwen35_layers(monkeypatch):
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    layer = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(layer)
    for name, value in vars(_layer()).items():
        if name in {"training", fused._TAG}:
            continue
        setattr(layer, name, value)
    layer.eval()
    model = SimpleNamespace(named_modules=lambda: iter((("gdn", layer),)))
    monkeypatch.setattr(fused, "probe_qwen35_fused_gdn_decode", lambda: 16)
    monkeypatch.setattr(fused, "_patch_class", lambda _class: None)
    assert fused.install_qwen35_fused_gdn_decode(model) == 1
    assert getattr(layer, fused._TAG)
    assert layer._rapid_qwen35_fused_gdn_threadgroup_y == 16


def test_shared_kernel_keeps_qwen4_and_qwen35_compile_time_paths():
    source = shared_kernel._SOURCE
    assert "if constexpr (Q35)" in source
    assert "float gate = zg * mlx_sigmoid_precise<float>(zg);" in source
    assert "float x = gate * float(normalized);" in source
    assert (
        "float x = float(normalized) * "
        "mlx_sigmoid_precise<float>(float(z[hv * DV + d]));"
    ) in source


def test_shared_kernel_rejects_non_integral_head_ratio_before_dispatch():
    with pytest.raises(ValueError, match="divisible"):
        shared_kernel.fused_gdn_decode(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            1e-6,
            threadgroup_y=4,
            num_key_heads=3,
            num_value_heads=4,
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_real_metal_probe_matches_output_and_both_states(monkeypatch):
    monkeypatch.setattr(fused, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(fused, "_PROBED_THREADGROUP_Y", None)
    assert fused.probe_qwen35_fused_gdn_decode() in {4, 8, 16, 32}


def test_probe_returns_process_cached_result(monkeypatch):
    monkeypatch.setattr(fused, "_PROBE_COMPLETE", True)
    monkeypatch.setattr(fused, "_PROBED_THREADGROUP_Y", 8)
    assert fused.probe_qwen35_fused_gdn_decode() == 8


def test_probe_observes_result_completed_while_waiting_for_lock(monkeypatch):
    class CompletingLock:
        def __enter__(self):
            fused._PROBE_COMPLETE = True
            fused._PROBED_THREADGROUP_Y = 16

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(fused, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(fused, "_PROBED_THREADGROUP_Y", None)
    monkeypatch.setattr(fused, "_PROBE_LOCK", CompletingLock())
    assert fused.probe_qwen35_fused_gdn_decode() == 16


def test_probe_skips_candidate_exception_and_fails_closed(monkeypatch):
    monkeypatch.setattr(fused, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(fused, "_PROBED_THREADGROUP_Y", None)
    monkeypatch.setattr(fused, "_THREADGROUP_Y_CANDIDATES", (4,))
    monkeypatch.setattr(fused, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(
        fused,
        "_probe_candidate",
        lambda _candidate: (_ for _ in ()).throw(OSError("compile failed")),
    )
    assert fused.probe_qwen35_fused_gdn_decode() is None


def test_real_probe_rejects_mismatched_kernel_output(monkeypatch):
    def wrong(*args, **kwargs):
        return (
            mx.zeros((1, 1, fused._VALUE_DIM), dtype=mx.bfloat16),
            mx.zeros((1, 3, fused._CONV_DIM), dtype=mx.bfloat16),
            mx.zeros((1, 32, 128, 128), dtype=mx.float32),
        )

    monkeypatch.setattr(fused, "fused_gdn_decode", wrong)
    assert not fused._probe_candidate(4)


def test_patched_call_falls_back_when_request_is_not_eligible():
    class FakeGdn:
        def __call__(self, inputs, mask=None, cache=None):
            return "stock"

    fused._patch_class(FakeGdn)
    assert FakeGdn()(mx.zeros((1, 2, 8), dtype=mx.bfloat16)) == "stock"
    fused._patch_class(FakeGdn)


def test_patched_call_commits_fresh_cache_only_after_dispatch(monkeypatch):
    class FakeGdn:
        def __call__(self, inputs, mask=None, cache=None):
            return "stock"

    layer = FakeGdn()
    template = _layer()
    for name, value in vars(template).items():
        setattr(layer, name, value)
    layer._rapid_qwen35_fused_gdn_threadgroup_y = 32
    layer.out_proj = lambda value: value
    cache = _Cache()
    qkv = mx.zeros((1, 1, fused._CONV_DIM), dtype=mx.bfloat16)
    z = mx.zeros((1, 1, fused._VALUE_DIM), dtype=mx.bfloat16)
    gates = mx.zeros((1, 1, fused._NUM_VALUE_HEADS), dtype=mx.bfloat16)
    next_conv = mx.ones_like(cache[0])
    next_state = mx.ones_like(cache[1])
    output = mx.ones((1, 1, fused._VALUE_DIM), dtype=mx.bfloat16)
    monkeypatch.setattr(
        gdn_in_proj_fusion, "_fused_projections", lambda *_: (qkv, z, gates, gates)
    )
    monkeypatch.setattr(
        fused,
        "fused_gdn_decode",
        lambda *args, **kwargs: (output, next_conv, next_state),
    )
    fused._patch_class(FakeGdn)
    result = layer(mx.zeros((1, 1, 2048), dtype=mx.bfloat16), cache=cache)
    assert result is output
    assert cache[0] is next_conv
    assert cache[1] is next_state
    assert cache.advanced == 1


def test_patched_call_falls_back_before_cache_mutation_on_projection_error(monkeypatch):
    class FakeGdn:
        def __call__(self, inputs, mask=None, cache=None):
            return "stock"

    layer = FakeGdn()
    for name, value in vars(_layer()).items():
        setattr(layer, name, value)
    layer._rapid_qwen35_fused_gdn_threadgroup_y = 32
    cache = _Cache()
    original_conv, original_state = cache[0], cache[1]
    monkeypatch.setattr(
        gdn_in_proj_fusion,
        "_fused_projections",
        lambda *_: (_ for _ in ()).throw(OSError("projection unavailable")),
    )
    fused._patch_class(FakeGdn)

    assert layer(mx.zeros((1, 1, 2048), dtype=mx.bfloat16), cache=cache) == "stock"
    assert cache[0] is original_conv
    assert cache[1] is original_state
    assert cache.advanced == 0


def test_engine_installs_gdn_decode_only_after_projection_fusion():
    from vllm_mlx.engine.batched import BatchedEngine

    source = inspect.getsource(BatchedEngine._start_llm)
    projection = source.index("fuse_gdn_in_proj")
    decode = source.index("install_qwen35_fused_gdn_decode")
    assert projection < decode
