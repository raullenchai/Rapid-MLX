# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
# Tiny configuration and checkpoint conversion adapted from the source in
# rapid_mlx/models/inkling2/NOTICE, under its accompanying MIT license.
"""Text-only catalog, loop/cache, quantization and loader regressions."""

import functools
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rapid_mlx.model_aliases import resolve_model, resolve_profile, resolve_revision
from rapid_mlx.model_auto_config import detect_model_config

REPO = "mlx-community/Inkling-2-Mini-4bit"
REVISION = "9856dc53edead71405f8348930937c3638ff21a2"
REDUCED_PRECISION_MODEL_TOLERANCE = 2e-2

CONFIG = {
    "model_type": "inkling2",
    "image_token_id": 127,
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
        "sliding_window_size": 8,
        "local_layer_ids": [0, 1, 2, 3, 4, 6],
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 3.0,
        "rel_l2_clip_threshold": 2.0,
        "sconv_kernel_size": 4,
        "n_layers_per_attn_res_block": 3,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 64,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 2.0,
        "final_logit_softcapping": 30.0,
    },
}


CONFIG["model_type"] = "inkling2_loop"
CONFIG["text_config"]["loop_config"] = {
    "core_start": 3,
    "core_end": 6,
    "num_passes": 3,
    "gate_rank": 8,
    "is_varloop": False,
}


# Complete reviewed source of the five numerical-core files omitted by
# .coveragerc. See NOTICE for provenance. Review edits before updating pins;
# this contract reads bytes only and requires neither MLX nor Metal.
_NUMERICAL_CORE_SHA256 = {
    "layers.py": "f30e646d49cc4f36450cf99f4465862a22448e1998d9d83d020f5f01217c9217",
    "kernels.py": "355670b093d3cdf23e435fd90b14bfb12bdd0d681d8c2e940dc9bf603e2195bc",
    "model.py": "4dee052b0aaaf5df2a90b9f1d80b215fd4546b7ee55367589fb30d84a3e9cd08",
    "switch_layers.py": "88aa71e9d0c5681eb418a6c0398b3b6e205f2584496fef9f3ff5a5daa54e99d1",
    "checkpoint.py": "7ba19519e069ce47319641f473a696dd3cfa82c246f2eec40949268a6bd2935f",
}


def test_numerical_core_source_contract():
    import configparser

    root = Path(__file__).parents[1]
    coverage = configparser.ConfigParser()
    coverage.read(root / ".coveragerc")
    prefix = "rapid_mlx/models/inkling2/"
    omitted = {
        path.strip().removeprefix(prefix)
        for path in coverage["run"]["omit"].splitlines()
        if path.strip().startswith(prefix)
    }
    assert omitted == _NUMERICAL_CORE_SHA256.keys()
    for filename, expected in _NUMERICAL_CORE_SHA256.items():
        assert hashlib.sha256((root / prefix / filename).read_bytes()).hexdigest() == (
            expected
        ), filename


@pytest.fixture
def glue_loader(monkeypatch):
    """Execute glue in isolation with CPU array/import doubles, even with MLX installed.

    No production package import (which initializes Metal kernels), distributed
    process group, or GPU is needed. All sys.modules changes are scoped to a test.
    """
    import rapid_mlx

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    class Array(np.ndarray):
        def __new__(cls, value, dtype=None):
            return np.asarray(value, dtype=dtype).view(cls)

    mx = module(
        "mlx.core",
        array=Array,
        Dtype=np.dtype,
        float32=np.float32,
        float16=np.float16,
        contiguous=Mock(side_effect=np.ascontiguousarray),
        eval=Mock(),
        concatenate=Mock(side_effect=np.concatenate),
        distributed=SimpleNamespace(Group=object, all_gather=Mock()),
    )
    module("mlx", core=mx, nn=SimpleNamespace(Module=object))
    shim = module("rapid_mlx._mlx_compat", install=Mock())
    monkeypatch.setattr(rapid_mlx, "_mlx_compat", shim, raising=False)
    module("mlx_lm", __path__=[])
    module("mlx_lm.models", __path__=[])
    module("mlx_lm.kernels", __path__=[])

    def load(filename):
        name = f"rapid_mlx.models.inkling2.{filename}"
        path = (
            Path(__file__).parents[1] / "rapid_mlx/models/inkling2" / f"{filename}.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
        return result

    return SimpleNamespace(load=load, module=module, mx=mx, shim=shim)


@pytest.fixture
def decode_cpu(glue_loader, monkeypatch):
    """Import the real numerical modules with NumPy arrays and inert Metal APIs."""
    g, mx = glue_loader, glue_loader.mx
    for name, value in dict(
        bfloat16=np.float16,
        bool_=np.bool_,
        int32=np.int32,
        uint32=np.uint32,
        ones=np.ones,
        zeros=np.zeros,
        compile=lambda fn=None, **kw: fn if fn is not None else lambda f: f,
        fast=SimpleNamespace(metal_kernel=Mock(return_value=Mock())),
    ).items():
        monkeypatch.setattr(mx, name, value, raising=False)
    g.module(
        "rapid_mlx.models.inkling2.compat",
        SwitchLinear=object,
        _gather_sort=Mock(),
        _scatter_unsort=Mock(),
        expert_routing=None,
        scalar=lambda x, dtype=np.float32: np.array(x, dtype=dtype),
        row_threads=lambda size: 32,
    )
    g.module("rapid_mlx.models.inkling2.config", TextArgs=object)
    g.module("mlx_lm.models.cache", BatchRotatingKVCache=object, RotatingKVCache=object)
    g.module("mlx.nn", Module=object)
    g.module("mlx.nn.layers", __path__=[])
    g.module("mlx.nn.layers.distributed", sum_gradients=Mock())
    kernels = g.load("kernels")
    layers = g.load("layers")
    monkeypatch.setattr(layers, "_use_kernels", lambda: True)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    return SimpleNamespace(kernels=kernels, layers=layers, mx=mx)


@pytest.mark.parametrize(
    "failure", [None, "construction", "probe", "evaluation", "invocation"]
)
@pytest.mark.parametrize("masked", [False, True])
def test_expert_conv_cpu_guards(decode_cpu, failure, masked):
    k, mx = decode_cpu.kernels, decode_cpu.mx
    x = np.ones((1, 1, 2, 32), np.float32)
    mix = np.ones((1, 1, 2), np.float32)
    state = np.ones((1, 3, 32), np.float32)
    weight = np.ones((32, 4, 1), np.float32)
    outer = np.ones((1, 1, 32), np.float32)
    mask, lengths = (
        (np.ones((1, 1), bool), np.ones((1,), np.int32)) if masked else (None, None)
    )
    pending = [outer, state]
    launch = Mock(return_value=pending)
    mx.fast.metal_kernel.reset_mock()
    mx.fast.metal_kernel.return_value = launch
    if failure == "construction":
        mx.fast.metal_kernel.side_effect = RuntimeError("construction")
    elif failure == "probe":
        launch.side_effect = RuntimeError("probe invocation")
    elif failure == "evaluation":
        mx.eval.side_effect = RuntimeError("probe evaluation")
    elif failure == "invocation":
        launch.side_effect = [
            pending,
            RuntimeError("real invocation"),
            RuntimeError("real invocation"),
        ]
    for _ in range(2):
        result = k.expert_conv_residual_decode(
            x, mix, state, weight, mask, lengths, outer, np.float32
        )
        assert (result is None) == (failure is not None)
        if result is not None:
            assert result is pending
    mx.fast.metal_kernel.assert_called_once()
    assert launch.call_count == (
        0
        if failure == "construction"
        else 1
        if failure in ("probe", "evaluation")
        else 3
    )
    assert mx.eval.call_count == (0 if failure in ("construction", "probe") else 1)
    # Shape rejection must bypass even construction/probing.
    mx.fast.metal_kernel.reset_mock()
    assert (
        k.expert_conv_residual_decode(
            x[:, :, :1], mix, state, weight, mask, lengths, outer, np.float32
        )
        is None
    )
    mx.fast.metal_kernel.assert_not_called()


def test_expert_conv_cpu_reprobes_promoted_experts(decode_cpu):
    """An unexpected expert dtype still gets its own evaluated specialization."""
    k, mx = decode_cpu.kernels, decode_cpu.mx
    weight = np.ones((32, 4, 1), np.float32)
    outer = np.ones((1, 1, 32), np.float32)
    state = np.ones((1, 3, 32), np.float32)
    launch = Mock(return_value=[outer, state])
    mx.fast.metal_kernel.return_value = launch
    ready = k._expert_conv_residual_ready(
        np.float32, weight, 2, False, False, np.float32
    )
    for _ in range(2):
        assert (
            k.expert_conv_residual_decode(
                np.ones((1, 1, 2, 32), np.float16),
                np.ones((1, 1, 2), np.float32),
                state,
                weight,
                None,
                None,
                outer,
                np.float32,
                kernel=ready,
            )
            is not None
        )
    assert mx.eval.call_count == 2
    assert launch.call_count == 4
    assert {dict(c.kwargs["template"])["T"] for c in launch.call_args_list} == {
        np.float32
    }


@pytest.mark.parametrize("failure", [None, "construction", "probe", "evaluation"])
def test_route_topk_cpu_probe(decode_cpu, failure):
    k, mx = decode_cpu.kernels, decode_cpu.mx
    launch = Mock(return_value=[np.zeros((1, 1, 2)), np.zeros((1, 1, 4))])
    mx.fast.metal_kernel.return_value = launch
    if failure == "construction":
        mx.fast.metal_kernel.side_effect = RuntimeError("construction")
    elif failure == "probe":
        launch.side_effect = RuntimeError("probe")
    elif failure == "evaluation":
        mx.eval.side_effect = RuntimeError("evaluation")
    for _ in range(2):
        assert (k._route_topk(4, 2) is None) == (failure is not None)
    assert launch.call_count == (0 if failure == "construction" else 1)
    assert mx.eval.call_count == (0 if failure in ("construction", "probe") else 1)


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "construction",
        "probe",
        "evaluation",
        "invocation",
        "ineligible",
        "disabled",
    ],
)
@pytest.mark.parametrize("boundary", ["first", "append", "inner"])
def test_decoder_cpu_evaluates_mlp_once(decode_cpu, monkeypatch, failure, boundary):
    """A failed probe uses stock MoE; failed invocation reuses evaluated experts."""
    layers, mx = decode_cpu.layers, decode_cpu.mx
    x = np.ones((1, 1, 32), np.float32)
    expert_states = np.full((1, 1, 2, 32), 2, np.float32)
    weights = np.full((1, 1, 2), 0.5, np.float32)
    mlp = layers.SparseMoeBlock.__new__(layers.SparseMoeBlock)
    mlp.sharding_group = None
    mlp.gate = Mock(return_value=(np.zeros((1, 1, 2), np.uint32), weights))
    mlp.gate.top_k = 2
    mlp.experts = Mock(return_value=expert_states)
    mlp.state = {}
    stock = Mock(wraps=mlp.__call__)

    # Delegate __call__ while retaining the real _forward implementation.
    class MLP:
        sharding_group = None
        gate = mlp.gate
        _forward = mlp._forward
        __call__ = staticmethod(stock)

    class Conv:
        kernel_size = 4
        conv1d = SimpleNamespace(weight=np.ones((32, 4, 1), np.float32))

        def __call__(self, hidden, history, mask, *, lengths):
            return hidden, history

    class Cache(list):
        lengths = None
        advance = Mock()

        @property
        def cache(self):
            return self

    layer = layers.DecoderLayer.__new__(layers.DecoderLayer)
    layer.mlp = MLP()
    layer.attention_input = None if boundary == "first" else Mock(return_value=x)
    layer.is_block_boundary = boundary != "inner"
    layer._self_attention = lambda hidden, *args: hidden
    layer._mlp_states = Mock(side_effect=lambda hidden, *args: hidden)
    layer.post_attention_convolution = layer.post_mlp_convolution = Conv()
    monkeypatch.setattr(layers, "unweighted_rms_norm", lambda hidden: hidden)
    monkeypatch.setattr(layers, "_conv_state", lambda window, *args: window)
    history = np.zeros((1, 3, 32), np.float32)
    cache = [SimpleNamespace(offset=3), Cache([history] * 4)]
    expected = np.full_like(x, 4 if boundary == "inner" else 3)
    outputs = [expected, history]
    launch = Mock(return_value=outputs)
    mx.fast.metal_kernel.return_value = launch
    if failure == "construction":
        mx.fast.metal_kernel.side_effect = RuntimeError("construction")
    elif failure == "probe":
        launch.side_effect = RuntimeError("probe")
    elif failure == "evaluation":
        mx.eval.side_effect = RuntimeError("evaluation")
    elif failure == "invocation":
        launch.side_effect = [
            outputs,
            RuntimeError("invocation"),
            RuntimeError("invocation"),
        ]
    elif failure == "ineligible":
        layer.mlp.gate.top_k = 1
    elif failure == "disabled":
        monkeypatch.setenv("RAPID_MLX_INKLING2_EXPERT_CONV_RESIDUAL", "0")
    for step in range(2):
        block_states = None if boundary == "first" else x[None]
        actual, block = layer(x, block_states, cache=cache)
        assert block.shape[0] == (2 if boundary == "append" else 1)
        assert layer._mlp_states.call_args.args[-1] is True
        np.testing.assert_array_equal(actual, expected)
        assert mlp.experts.call_count == mlp.gate.call_count == step + 1
    assert stock.call_count == (0 if failure in (None, "invocation") else 2)
    assert launch.call_count == (
        0
        if failure in ("construction", "ineligible", "disabled")
        else 1
        if failure in ("probe", "evaluation")
        else 3
    )
    assert mx.eval.call_count == (
        0 if failure in ("construction", "ineligible", "disabled", "probe") else 1
    )


@pytest.mark.parametrize(
    "feature", ["MESSAGE_NORM", "ROUTE_TOPK", "EXPERT_CONV_RESIDUAL"]
)
def test_decode_cpu_independent_switches(decode_cpu, monkeypatch, feature):
    layers = decode_cpu.layers
    features = ("MESSAGE_NORM", "ROUTE_TOPK", "EXPERT_CONV_RESIDUAL")
    for name in features:
        monkeypatch.delenv(f"RAPID_MLX_INKLING2_{name}", raising=False)
    assert all(layers._fast_decode_enabled(name) for name in features)
    monkeypatch.setenv(f"RAPID_MLX_INKLING2_{feature}", "0")
    assert layers._fast_decode_enabled()
    for name in features:
        assert layers._fast_decode_enabled(name) is (name != feature)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    assert not any(layers._fast_decode_enabled(name) for name in (*features, None))


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("unavailable", [False, True])
def test_route_topk_cpu_switch_and_fallback(
    decode_cpu, monkeypatch, disabled, unavailable
):
    layers, mx = decode_cpu.layers, decode_cpu.mx
    router = layers.TopKRouter.__new__(layers.TopKRouter)
    router.weight = np.ones((4, 32), np.float32)
    router.e_score_correction_bias = np.zeros((4,), np.float32)
    router.top_k, router.routed_scaling_factor = 2, 2.0
    expected = (np.array([[[0, 1]]], np.uint32), np.ones((1, 1, 2), np.float32))
    fused = Mock(return_value=None if unavailable else expected)
    stock = Mock(return_value=expected)
    monkeypatch.setattr(layers, "route_topk_decode", fused)
    monkeypatch.setattr(layers, "_route_weights", stock)
    monkeypatch.setattr(mx, "sigmoid", lambda x: 1 / (1 + np.exp(-x)), raising=False)
    monkeypatch.setenv("RAPID_MLX_INKLING2_ROUTE_TOPK", "0" if disabled else "1")
    assert router(np.ones((1, 1, 32), np.float32), cached_decode=True) is expected
    assert fused.call_count == int(not disabled)
    assert stock.call_count == int(disabled or unavailable)


@pytest.fixture
def compat_loader(glue_loader):
    def load(*, global_scale=True, routing=True):
        class Native:
            if global_scale:

                def to_quantized(self, *, global_scale=False):
                    pass
            else:

                def to_quantized(self):
                    pass

        native = glue_loader.module(
            "mlx_lm.models.switch_layers",
            SwitchLinear=Native,
            _gather_sort=object(),
            _scatter_unsort=object(),
        )
        fallback = glue_loader.module(
            "rapid_mlx.models.inkling2.switch_layers",
            SwitchLinear=object(),
            _gather_sort=object(),
            _scatter_unsort=object(),
        )
        if routing:
            router = glue_loader.module(
                "mlx_lm.kernels.expert_routing", expert_routing=object()
            ).expert_routing
        else:
            # Missing export forces ImportError even if the real helper was loaded.
            glue_loader.module("mlx_lm.kernels.expert_routing")
            router = None
        result = glue_loader.load("compat")
        selected = native if global_scale else fallback
        for name in ("SwitchLinear", "_gather_sort", "_scatter_unsort"):
            assert getattr(result, name) is getattr(selected, name)
        assert result.expert_routing is router
        glue_loader.shim.install.assert_called_once_with()
        return result

    return load


@pytest.fixture
def message_cpu(glue_loader, compat_loader, monkeypatch):
    """Load the real wrappers with NumPy arrays and inert Metal construction."""
    compat_loader()
    mx = glue_loader.mx
    mx.bfloat16 = object()  # Distinct supported dtype; NumPy has no bf16.
    mx.ones = lambda shape, dtype=np.float32: mx.array(np.ones(shape, dtype=dtype))
    mx.compile = lambda fn, **kwargs: fn
    mx.fast = SimpleNamespace(
        metal_kernel=Mock(return_value=Mock()),
        rms_norm=lambda x, w, eps: (
            (
                x
                / np.sqrt(
                    np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps
                )
            )
            * (1 if w is None else w)
        ),
    )
    kernels = glue_loader.load("kernels")
    glue_loader.module("mlx.nn.layers.distributed", sum_gradients=Mock())
    glue_loader.module(
        "mlx_lm.models.cache", BatchRotatingKVCache=object, RotatingKVCache=object
    )
    glue_loader.module("rapid_mlx.models.inkling2.config", TextArgs=object)
    layers = glue_loader.load("layers")
    monkeypatch.setattr(layers, "_use_kernels", lambda: True)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    return mx, kernels, layers


@pytest.mark.parametrize(
    "failure", [None, "construction", "probe", "evaluation", "call"]
)
def test_message_norm_cpu_probe_and_fallback(message_cpu, monkeypatch, failure):
    mx, kernels, layers = message_cpu
    x, w = mx.ones((1, 1, 17)), mx.ones((17,))
    probe_output = object()
    launches = []

    def launch(**kw):
        launches.append(kw)
        if len(launches) == 1:
            if failure == "probe":
                raise RuntimeError("probe invocation")
            return [probe_output]
        if failure == "call":
            raise RuntimeError("real invocation")
        a, b, weight, eps = kw["inputs"]
        output = dict(kw["template"])["OUTPUT"]
        value = mx.fast.rms_norm(a, weight if output else None, eps)
        if not output:
            value = np.concatenate((value, mx.fast.rms_norm(b, None, eps)), axis=-1)
        return [value.astype(kw["output_dtypes"][0])]

    factory = Mock(
        side_effect=RuntimeError("construction") if failure == "construction" else None,
        return_value=launch,
    )
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    if failure == "evaluation":
        mx.eval.side_effect = RuntimeError("probe evaluation")
    for helper, args in (
        (layers._message_input, (x, x, mx.float16)),
        (layers._norm_cast, (x, w, 1e-6, mx.float16)),
    ):
        launches.clear()
        for _ in range(2):
            np.testing.assert_array_equal(helper(*args, decode=True), helper(*args))
        assert len(launches) == (
            0
            if failure == "construction"
            else 1
            if failure in ("probe", "evaluation")
            else 3
        )
    assert factory.call_count == 2  # Cached success/failure, once per variant.
    assert mx.eval.call_count == (0 if failure in ("construction", "probe") else 2)
    if failure not in ("construction", "probe"):
        assert all(call.args == (probe_output,) for call in mx.eval.call_args_list)


def test_message_norm_cpu_guards(message_cpu, monkeypatch):
    mx, kernels, layers = message_cpu
    forbidden = Mock(side_effect=AssertionError("unsupported input launched a kernel"))
    monkeypatch.setattr(kernels, "_get_message_norm", forbidden)
    x = mx.ones((1, 1, 17))
    # Output mode's default weight must return the sentinel, not dereference None.
    assert kernels.message_norm_decode(x, None, mx.float32) is None
    for value, message, dtype, weight in (
        (mx.ones((1, 17)), x, mx.float32, None),
        (mx.ones((2, 1, 17)), x, mx.float32, None),
        (mx.ones((1, 1, 0)), x, mx.float32, None),
        (mx.ones((1, 1, 4097)), x, mx.float32, None),
        (x.astype(np.int32), x, mx.float32, None),
        (x, x, np.int32, None),
        (x, mx.ones((1, 1, 16)), mx.float32, None),
        (x, x.astype(np.float16), mx.float32, None),
        (x.astype(np.float16), None, mx.float32, mx.ones((17,))),
        (x, None, mx.float32, mx.ones((16,))),
        (x, None, mx.float32, mx.ones((17,), np.float16)),
    ):
        assert kernels.message_norm_decode(value, message, dtype, weight) is None
    forbidden.assert_not_called()
    monkeypatch.setattr(layers, "message_norm_decode", forbidden)
    for decode, enabled, device in (
        (False, "1", True),
        (True, "0", True),
        (True, "1", False),
    ):
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", enabled)
        monkeypatch.setattr(layers, "_use_kernels", lambda device=device: device)
        layers._message_input(x, x, mx.float16, decode)
        layers._norm_cast(x, mx.ones((17,)), 1e-6, mx.float16, decode)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    monkeypatch.setenv("RAPID_MLX_INKLING2_MESSAGE_NORM", "0")
    monkeypatch.setattr(layers, "_use_kernels", lambda: True)
    layers._message_input(x, x, mx.float16, True)
    layers._norm_cast(x, mx.ones((17,)), 1e-6, mx.float16, True)
    forbidden.assert_not_called()


@pytest.mark.parametrize("kind", ["SlidingDecoderLayer", "FullDecoderLayer"])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("with_messages", [False, True])
def test_message_norm_cpu_forwarding(
    message_cpu, monkeypatch, kind, decode, with_messages
):
    """Exercise real layer call sites with CPU state and recorded leaf operations."""
    mx, kernels, layers = message_cpu
    x = mx.ones((1, 1, 17))
    messages = mx.ones((1, 1, 4, 17)) if with_messages else None
    norm = SimpleNamespace(weight=mx.ones((17,)), eps=1e-6)
    cls = getattr(layers, kind)
    layer = cls.__new__(cls)
    layer.state = {}
    layer.input_layernorm = norm
    layer.q_input_layernorm = layer.k_input_layernorm = layer.v_input_layernorm = norm
    layer.post_attention_layernorm = norm
    gates = [Mock(side_effect=lambda a, b, d: a + b) for _ in range(4)]
    (
        layer.q_message_gate,
        layer.k_message_gate,
        layer.v_message_gate,
        layer.mlp_message_gate,
    ) = gates
    layer.mlp_input = Mock(side_effect=lambda block, a: a[..., None, :])
    layer.self_attn = Mock(side_effect=lambda q, k, v, **kw: q)
    layer.attention_input = None
    layer.is_block_boundary = True
    layer.mlp = Mock(side_effect=lambda a, **kw: a)
    layer.post_attention_convolution = layer.post_mlp_convolution = (
        lambda a, state, mask, **kw: (a, state)
    )
    calls = Mock(return_value=None)
    monkeypatch.setattr(layers, "message_norm_decode", calls)
    cache = [SimpleNamespace(offset=0), None] if decode else None
    result, block = layer(x, None, cache=cache, messages=messages)
    assert result.shape == x.shape and block.shape == (1, *x.shape)
    for gate in gates:
        assert gate.call_count == int(with_messages)
        if with_messages:
            assert gate.call_args.args[-1] is decode
    assert calls.call_count == (4 if decode and with_messages else 0)
    # Also cover the attention-input path with independently supplied Q/K/V.
    attention = np.stack((x, x, x), axis=-2)
    layer._self_attention(x, attention, cache, [None] * 4, None, messages)
    for gate in gates[:3]:
        assert gate.call_count == (2 if with_messages else 0)
    # MessageGate must pass decode to the normalization helper itself.
    gate = layers.MessageGate.__new__(layers.MessageGate)
    gate.down_proj = Mock(side_effect=lambda a: a[..., :17])
    gate.down_proj.weight = x
    gate.forget_gate_proj = gate.message_gate_proj = Mock()
    gate.forget_gate_proj.weight = x.astype(np.float16)  # Use eager expansion.
    gate.forget_gate_proj.return_value = x
    mx.tanh = np.tanh
    seen = Mock(wraps=layers._message_input)
    monkeypatch.setattr(layers, "_message_input", seen)
    assert gate(x, x, decode).shape == x.shape
    seen.assert_called_once_with(x, x, x.dtype, decode)


def _expert_layer(switch_layers, mx, *, global_scale):
    mx.random.seed(0)
    layer = switch_layers.QuantizedSwitchLinear(
        128, 16, 4, bias=False, group_size=16, mode="nvfp4", global_scale=global_scale
    )
    x = mx.random.normal((2, 3, 1, 1, 128))
    indices = mx.array([[[0, 3], [1, 2], [3, 3]], [[2, 0], [1, 1], [0, 2]]])
    return layer, x, indices


def test_expert_global_scale_fallback_matches_reference(monkeypatch):
    """Without gather_qmm(global_scale=...), rescaling the output is exact.

    The reference decodes each expert on its own, so this also runs on MLX
    builds that lack the native argument.
    """
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models.inkling2 import switch_layers

    layer, x, indices = _expert_layer(switch_layers, mx, global_scale=True)
    # Give every expert a distinct scale so a wrong gather would show.
    layer.global_scale = layer.global_scale * mx.array([0.5, 1.0, 3.0, 7.0])
    dense = mx.stack(
        [
            mx.dequantize(
                layer.weight[e],
                layer.scales[e],
                None,
                layer.group_size,
                layer.bits,
                mode=layer.mode,
                global_scale=layer.global_scale[e],
            )
            for e in range(layer.num_experts)
        ]
    )
    reference = x @ dense[indices].swapaxes(-1, -2)
    native = layer(x, indices) if switch_layers._GATHER_QMM_HAS_GLOBAL_SCALE else None

    real = mx.gather_qmm
    calls = []

    def old_gather_qmm(*args, **kwargs):
        calls.append(set(kwargs))
        assert "global_scale" not in kwargs
        return real(*args, **kwargs)

    monkeypatch.setattr(switch_layers.mx, "gather_qmm", old_gather_qmm)
    monkeypatch.setattr(switch_layers, "_GATHER_QMM_HAS_GLOBAL_SCALE", False)
    fallback = layer(x, indices)

    assert calls
    assert fallback.shape == reference.shape
    # The quantized kernel and a dense decode agree to about 0.2%; a wrong or
    # missing per-expert scale is off by at least a factor of two.
    error = mx.abs(fallback - reference).max() / mx.abs(reference).max()
    assert error.item() < 1e-2
    if native is not None:
        assert fallback.dtype == native.dtype
        assert mx.allclose(fallback, native, rtol=1e-5, atol=1e-6).item()

    # Half precision: the unscaled product overflows, the fallback must not.
    layer.global_scale = mx.full(layer.global_scale.shape, 0.1)
    half = layer(mx.ones((1, 1, 1, 1, 128), dtype=mx.float16) * 40, mx.array([[[0]]]))
    assert half.dtype == mx.float16
    assert mx.all(mx.isfinite(half)).item()


def test_expert_without_global_scale_never_passes_it(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models.inkling2 import switch_layers

    layer, x, indices = _expert_layer(switch_layers, mx, global_scale=False)
    assert layer.global_scale is None
    real = mx.gather_qmm
    seen = []

    def spy(*args, **kwargs):
        seen.append(set(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(switch_layers.mx, "gather_qmm", spy)
    mx.eval(layer(x, indices))
    assert seen and all("global_scale" not in kwargs for kwargs in seen)


@pytest.mark.parametrize("global_scale", [False, True])
@pytest.mark.parametrize("routing", [False, True])
def test_compat_version_fallbacks(compat_loader, global_scale, routing):
    compat_loader(global_scale=global_scale, routing=routing)


def test_compat_scalar_and_row_threads(compat_loader):
    compat = compat_loader()
    value = compat.scalar(1.5)
    assert value.shape == () and value.dtype == np.float32 and value.item() == 1.5
    assert compat.scalar(1.5) is value
    assert compat.scalar(1.5, np.float16).dtype == np.float16
    for dim, expected in [(1, 32), (511, 32), (512, 64), (8192, 1024), (16384, 1024)]:
        assert compat.row_threads(dim) == expected


def test_kernel_templates_ignore_decode_sizes(glue_loader, compat_loader):
    """Exercise every launch wrapper without needing Metal or model weights."""
    compat_loader()
    mx = glue_loader.mx
    mx.int32 = np.int32
    mx.result_type = np.result_type
    calls = {}

    def factory(*, name, **kwargs):
        calls[name] = []

        def launch(**kwargs):
            calls[name].append(kwargs)
            return [
                mx.array(np.zeros(shape, dtype=dtype))
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"]
                )
            ]

        return launch

    mx.fast = SimpleNamespace(metal_kernel=factory)
    kernels = glue_loader.load("kernels")

    def zeros(shape):
        return mx.array(np.zeros(shape, dtype=np.float32))

    for step in range(40):
        for batch in (1, 2, 3):
            size, length = step + 7, step + 1
            x = zeros((batch, 1, 64))
            kernels.message_mix_decode(
                x, x, zeros((batch, 1, 8)), zeros((64, 8)), zeros((64, 8))
            )
            kernels.attention_mask_decode(
                zeros((batch, 1, 4, 8)),
                mx.array(np.arange(batch) + size, dtype=mx.int32),
                step,
                size,
                0,
                mx.float32,
                rotating=False,
            )
            kernels.route_weights_decode(
                zeros((batch, 1, 4)), zeros((batch, 1, 2)), 2.0
            )
            kernels.short_convolution_decode(
                x, zeros((batch, 3, 64)), zeros((64, 4)), None, 4
            )
            kernels.depth_attention(
                zeros((2, batch, 1, 64)), x, zeros((4, 64)), scale=0.125
            )
            kernels.relative_attention(
                zeros((batch, 4, length, 16)),
                zeros((batch, 2, size, 16)),
                zeros((batch, 2, size, 16)),
                zeros((batch, length, 4, 8)),
                zeros((batch, length, 4)),
                scale=0.25,
                window=8,
                key_start=mx.array(np.arange(batch), dtype=mx.int32),
            )

    assert len(calls) == 6
    for name, launches in calls.items():
        assert len(launches) == 120
        assert len({tuple(call["template"]) for call in launches}) == 1, name
    # The changing dimensions must reach the device as int32 data, while the
    # dispatch grid and output shape still follow the current batch and cache.
    for step in range(40):
        for batch in (1, 2, 3):
            size = step + 7
            call = calls["inkling2_attention_mask_decode"][step * 3 + batch - 1]
            args = dict(zip(["bias", "ends", "index", "sizes"], call["inputs"]))
            np.testing.assert_array_equal(args["sizes"], [batch, size])
            assert args["sizes"].dtype == np.int32
            assert args["index"].dtype == np.int32 and args["index"].item() == step
            np.testing.assert_array_equal(args["ends"], np.arange(batch) + size)
            assert call["grid"] == (batch * 4 * size, 1, 1)
            assert call["output_shapes"] == [(batch, 4, 1, size)]


def test_compat_gather_rank_order(compat_loader, glue_loader):
    compat = compat_loader()
    x = glue_loader.mx.array(np.arange(12).reshape(2, 3, 2))
    group = SimpleNamespace(size=lambda: 2)
    glue_loader.mx.distributed.all_gather.return_value = np.concatenate(
        [x.reshape(-1, 2), x.reshape(-1, 2) + 100], axis=0
    )
    actual = compat.gather_last_axis(x, group)
    np.testing.assert_array_equal(actual, np.concatenate([x, x + 100], axis=-1))
    args, kwargs = glue_loader.mx.distributed.all_gather.call_args
    np.testing.assert_array_equal(args[0], x.reshape(-1, 2))
    assert kwargs == {"group": group}


@pytest.mark.parametrize("sharding", ["all-to-sharded", "sharded-to-all"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("chunked", [False, True])
def test_compat_shard_experts(compat_loader, glue_loader, sharding, rank, chunked):
    compat = compat_loader()
    mx = glue_loader.mx

    class LargeArray(mx.array):
        # Report a large logical allocation to exercise the chunk budget with
        # tiny storage. Slicing/concatenation still operate on real CPU values.
        @property
        def size(self):
            return self.shape[0] * (1 << 30)

    weights = mx.array(np.arange(3 * 4 * 6).reshape(3, 4, 6))
    if chunked:
        weights = weights.view(LargeArray)
    scales = mx.array([1.0, 2.0, 3.0])
    nested = {"weight": weights}
    module = SimpleNamespace(
        parameters=lambda: {"weight": weights, "scales": scales, "nested": nested},
        update=Mock(),
    )
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    compat.shard_experts_inplace(module, sharding, group)
    updates = module.update.call_args.args[0]
    assert set(updates) == {"weight"}
    expected = (
        weights[:, rank * 2 : (rank + 1) * 2]
        if sharding == "all-to-sharded"
        else weights[..., rank * 3 : (rank + 1) * 3]
    )
    np.testing.assert_array_equal(updates["weight"], expected)
    assert mx.eval.call_count == mx.contiguous.call_count == (3 if chunked else 1)
    assert mx.concatenate.call_count == int(chunked)


@pytest.mark.parametrize("loop", [None, {}, {"core_start": 3, "core_end": 6}])
@pytest.mark.parametrize("vision", [None, {"hidden_size": 1024}])
def test_config_optional_fields(glue_loader, loop, vision):
    class BaseModelArgs:
        @classmethod
        def from_dict(cls, config):
            return cls(**config)

    glue_loader.module("mlx_lm.models.base", BaseModelArgs=BaseModelArgs)
    config = glue_loader.load("config")
    text = {**CONFIG["text_config"], "loop_config": loop}
    args = config.ModelArgs(
        model_type="inkling2_loop", text_config=text, vision_config=vision
    )
    assert isinstance(args.text, config.TextArgs)
    assert args.text.hidden_size == 64
    assert list(args.text.loop_core) == ([3, 4, 5] if loop else [])
    assert args.text.loop_config == loop
    assert args.vision is None
    assert args.vision_config == vision
    assert args.image_token_id == 200054
    # Exercise omission of all optional fields, including loop_config.
    text.pop("loop_config")
    defaults = config.ModelArgs("inkling2_loop", text, image_token_id=127)
    assert defaults.text.loop_config is None and list(defaults.text.loop_core) == []
    assert defaults.vision_config is None and defaults.image_token_id == 127


@pytest.fixture(scope="module")
def runtime():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    from rapid_mlx.models import inkling2

    return mx, inkling2


@functools.cache
def _reduced_precision_fp32_gemm(device):
    """Probe the current default device; cache separately for CPU and GPU."""
    import mlx.core as mx

    # Use local NumPy randomness so probing cannot change model/test seeds.
    # M > 1 exercises GEMM: single-row GEMV can retain full fp32 precision.
    rng = np.random.default_rng(42)
    a = rng.standard_normal((8, 512)).astype(np.float32)
    b = rng.standard_normal((512, 512)).astype(np.float32)
    expected = a.astype(np.float64) @ b.astype(np.float64)
    actual = np.asarray(mx.array(a) @ mx.array(b))
    # Peak-normalized error avoids cancellation near zero. Full fp32 is far
    # below 1e-5; reduced-precision GEMM errors are about 1e-3 on M5/MLX 0.32.4.
    return np.max(np.abs(actual - expected)) > 1e-5 * np.max(np.abs(expected))


@pytest.fixture
def metal_runtime(runtime, monkeypatch):
    mx, _ = runtime
    if not mx.metal.is_available():
        pytest.skip("Inkling decode kernels require Metal")
    mx.set_default_device(mx.gpu)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    return runtime


@pytest.fixture
def compile_traces(runtime):
    """Count Python graph traces, rather than calls to compiled wrappers."""
    from rapid_mlx.models.inkling2 import layers

    names = {
        name for name, fn in vars(layers).items() if hasattr(fn, "__wrapped__")
    } | {
        "_decode_log_scale",
        "_l2_clip",
        "_softcapped_swiglu",
        "_forward",
        "_mlp_states",
        "_attention_states",
    }
    traces = Counter()
    previous = sys.getprofile()

    def profile(frame, event, arg):
        code = frame.f_code
        if (
            event == "call"
            and code.co_filename == layers.__file__
            and code.co_name in names
        ):
            traces[code.co_name] += 1
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield traces
    finally:
        sys.setprofile(previous)


@pytest.mark.requires_mlx
def test_decode_attention_bias_broadcasts_shared_bias_per_batch(metal_runtime):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    # The decode mask must follow each call's batch size and key count, not
    # the shapes of the first call traced (here a larger batch).
    for batch, keys in ((3, 9), (2, 9), (3, 10), (1, 9), (2, 7)):
        bias = layers._decode_attention_bias(
            mx.zeros((batch, 1, 4, 8)),
            mx.broadcast_to(mx.arange(keys), (batch, 1, 1, keys)),
            mx.ones((batch, 1, 1, keys), dtype=mx.bool_),
            8,
        )
        mx.eval(bias)
        assert bias.shape == (batch, 4, 1, keys)


@pytest.mark.requires_mlx
def test_compiled_helpers_ignore_batch_and_offsets(metal_runtime, compile_traces):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    # Include the CPU/fallback router even on machines with a fused router.
    # k=7 and these dimensions keep this independent of other tests' warmups.
    expected_traces = None
    for step in range(40):
        for batch in (1, 2, 3):
            x = mx.ones((batch, 1, 24))
            experts = mx.ones((batch, 1, 7, 24))
            weights = mx.ones((batch, 1, 7))
            offsets = mx.full((batch, 1), step, dtype=mx.int32)
            indices, routes = layers._route_weights_reference(
                x, mx.zeros((24,)), 7, 2.0
            )
            outputs = [
                layers._norm(x, mx.ones((24,)), 1e-6),
                layers._value_norm(x[..., None, :], 1e-6),
                layers._decode_attention_bias(
                    mx.zeros((batch, 1, 4, 8)),
                    mx.broadcast_to(mx.arange(step + 7), (batch, 1, 1, step + 7)),
                    mx.ones((batch, 1, 1, step + 7), dtype=mx.bool_),
                    8,
                ),
                layers._scale_states(x, mx.array(1.13)),
                layers._gate_attention(x, x),
                layers._routing_order(x, mx.zeros((24,))),
                layers._combine_experts(experts, weights),
                layers._normalized_message_input(x, x),
                layers._message_mix(x, x, x, x),
                layers._residual_output(x, x),
                layers._depth_attention_reference(
                    x, mx.ones((2, batch, 1, 24)), mx.ones((4, 24)), 0.2
                ),
                layers._decode_log_scale(offsets, mx.array(4.0), mx.array(0.1)),
                layers._l2_clip(x, 1.0, 3.0),
                layers._softcapped_swiglu(x, x, 4.0, 25.0),
                indices,
                routes,
            ]
            mx.eval(outputs)
            assert indices.shape == routes.shape == (batch, 1, 7)
            assert mx.allclose(routes, mx.full(routes.shape, 2 / 7)).item()
            assert [out.shape[0] for out in outputs] == [batch] * len(outputs)
            if expected_traces is None:
                expected_traces = compile_traces.copy()
            assert compile_traces == expected_traces


def _assert_conv_close(mx, actual, expected):
    mx.eval(actual, expected)
    if actual.dtype in (mx.bfloat16, mx.float16):
        # Compare representable values, including across exponent boundaries
        # and signed zero. An absolute 1e-2 tolerance would hide small outputs.
        def ordered(x):
            bits = np.asarray(x.view(mx.uint16)).astype(np.int32)
            return np.where(bits & 0x8000, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)

        assert np.max(np.abs(ordered(actual) - ordered(expected))) <= 1
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("width", [2, 4, 5])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("history", ["empty", "warm", "masked", "ragged"])
def test_short_conv_decode_parity(
    metal_runtime, monkeypatch, dynamic, batch, width, dtype, history
):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2.layers import (
        DynamicShortConvolution,
        ShortConvolution,
        _conv_state,
    )

    mx.random.seed(29)
    dtype = getattr(mx, dtype)
    channels, heads = 24, 3
    if dynamic:
        conv = DynamicShortConvolution(channels, 16, heads, width)
        conv.static_weight = mx.random.normal((width, channels)).astype(dtype)
        conv.weight_proj.weight = mx.random.normal((heads * width, 16)).astype(dtype)
    else:
        conv = ShortConvolution(channels, width)
        conv.conv1d.weight = mx.random.normal((channels, width, 1)).astype(dtype)

    # Exercise strided inputs and FP32 state, including values not representable
    # in the weight dtype. State must retain them without a BF16 round trip.
    x = mx.random.normal((batch, 1, 2 * channels)).astype(dtype)[..., ::2]
    state = (
        None
        if history == "empty"
        else mx.random.normal((batch, width - 1, 2 * channels))[..., ::2]
    )
    mask = (
        mx.array([[i % 2 == 1] for i in range(batch)])
        if history in ("masked", "ragged")
        else None
    )
    lengths = mx.array([i - 1 for i in range(batch)]) if history == "ragged" else None
    dynamic_input = mx.random.normal((batch, 1, 16)).astype(dtype)
    for step in range(3):
        args = (x, dynamic_input, state, mask) if dynamic else (x, state, mask)
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
        expected, window = conv(*args, lengths=lengths)
        expected_state = _conv_state(window, lengths, width - 1)
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
        actual, next_state = conv(*args, lengths=lengths)
        _assert_conv_close(mx, actual, expected)
        assert next_state.shape == (batch, width - 1, channels)
        assert next_state.dtype == mx.float32
        assert mx.array_equal(next_state, expected_state).item()
        assert _conv_state(next_state, lengths, width - 1) is next_state
        state = next_state
        x = (x.astype(mx.float32) * 0.75 + step * 0.125).astype(dtype)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("experts", [2, 4, 8])
def test_expert_conv_residual_exact(metal_runtime, monkeypatch, dtype, experts):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    launch = Mock(wraps=kernels._expert_conv_residual_factory())
    monkeypatch.setattr(kernels, "_expert_conv_residual_factory", lambda: launch)
    dtype = getattr(mx, dtype)
    for seed in (3, 29, 71):
        mx.random.seed(seed)
        for scale, channels, width in ((0.001, 32, 2), (1, 64, 4), (100, 2048, 5)):
            conv = layers.ShortConvolution(channels, width)
            conv.conv1d.weight = mx.random.normal((channels, width, 1)).astype(dtype)
            x = (mx.random.normal((1, 1, experts, channels * 2)) * scale).astype(dtype)[
                ..., ::2
            ]
            mix = mx.random.uniform(shape=(1, 1, experts))
            state = mx.random.normal((1, width - 1, channels * 2))[..., ::2] * scale
            residual = mx.random.normal((1, 1, channels)) * scale
            for mask, lengths in ((None, None), (mx.array([[False]]), mx.array([0]))):
                combined = layers._combine_experts(x, mix).astype(dtype)
                convolved, window = conv(combined, state, mask, lengths=lengths)
                expected = layers._residual_output(convolved, residual)
                launch.reset_mock()
                actual, retained = kernels.expert_conv_residual_decode(
                    x, mix, state, conv.conv1d.weight, mask, lengths, residual, dtype
                )
                assert any(c.kwargs["inputs"][0] is x for c in launch.call_args_list)
                assert actual.dtype == expected.dtype == dtype
                assert mx.array_equal(actual, expected).item()
                bits = mx.uint32 if dtype == mx.float32 else mx.uint16
                assert mx.array_equal(actual.view(bits), expected.view(bits)).item()
                assert retained.dtype == mx.float32
                assert mx.array_equal(
                    retained, layers._conv_state(window, lengths, width - 1)
                ).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_expert_conv_residual_cached_logits(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    model = create_model(metal_runtime, dtype=getattr(mx, dtype))
    fused = layers.expert_conv_residual_decode

    def checked_fusion(*args, **kwargs):
        result = fused(*args, **kwargs)
        assert result is not None
        return result

    calls = Mock(side_effect=checked_fusion)
    factory = layers._expert_conv_residual_ready
    for seed in (3, 29, 71):
        mx.random.seed(seed)
        tokens = mx.random.randint(0, 120, (1, 10))
        caches = [model.make_cache(), model.make_cache()]
        for cache in caches:
            mx.eval(model(tokens[:, :6], cache=cache))
        for step in range(6, 10):
            monkeypatch.setattr(
                layers, "_expert_conv_residual_ready", lambda *args: None
            )
            expected = model(tokens[:, step : step + 1], cache=caches[0])
            mx.eval(expected)
            monkeypatch.setattr(layers, "_expert_conv_residual_ready", factory)
            monkeypatch.setattr(layers, "expert_conv_residual_decode", calls)
            calls.reset_mock()
            actual = model(tokens[:, step : step + 1], cache=caches[1])
            assert calls.call_count > 0
            assert mx.array_equal(actual, expected).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "failure", ["construction", "probe", "invocation", "evaluation"]
)
def test_expert_conv_residual_fallback(metal_runtime, monkeypatch, failure):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    unavailable = Mock(side_effect=AssertionError("unsupported input reached kernel"))
    weight = mx.zeros((64, 4, 1))
    with monkeypatch.context() as patch:
        patch.setattr(kernels, "_expert_conv_residual_factory", unavailable)
        for batch, length, experts, kind in (
            (2, 1, 2, "shape"),
            (1, 2, 2, "shape"),
            (1, 1, 16, "shape"),
            (1, 1, 2, "dtype"),
            (1, 1, 2, "cache"),
        ):
            x = mx.zeros((batch, length, experts, 64))
            mix = mx.zeros(
                x.shape[:3], dtype=mx.float16 if kind == "dtype" else mx.float32
            )
            state = None if kind == "cache" else mx.zeros((batch, 3, 64))
            outer = mx.zeros((batch, length, 64))
            assert (
                kernels.expert_conv_residual_decode(
                    x, mix, state, weight, None, None, outer, mx.float32
                )
                is None
            )
    unavailable.assert_not_called()
    model = create_model(metal_runtime, dtype=mx.float32, loop=False)
    cache, reference = model.make_cache(), model.make_cache()
    for bank in (cache, reference):
        mx.eval(model(mx.array([[1, 2, 3]]), cache=bank))
    kernels._expert_conv_residual_factory.cache_clear()
    kernels._probe_expert_conv_residual.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(layers, "_expert_conv_residual_ready", lambda *args: None)
        expected = model(mx.array([[4]]), cache=reference)
        mx.eval(expected)
    pending, real_eval = mx.zeros((1,)), mx.eval
    broken = Mock(side_effect=RuntimeError("kernel invocation unavailable"))
    factory = Mock(return_value=broken)
    evaluation = Mock(side_effect=RuntimeError("probe evaluation unavailable"))

    def evaluate(*arrays):
        if any(a is pending for a in arrays):
            return evaluation()
        return real_eval(*arrays)

    if failure == "construction":
        factory.side_effect = RuntimeError("kernel construction unavailable")
    if failure == "evaluation":
        broken.side_effect, broken.return_value = None, [pending, pending]
    with monkeypatch.context() as patch:
        patch.setattr(mx.fast, "metal_kernel", factory)
        patch.setattr(mx, "eval", evaluate)
        if failure == "invocation":
            patch.setattr(
                kernels, "_probe_expert_conv_residual", lambda kernel, *args: kernel
            )
        actual = model(mx.array([[4]]), cache=cache)
        assert mx.array_equal(actual, expected).item()
        factory.assert_called_once()
        expected_calls = {"construction": 0, "invocation": len(model.layers)}
        assert broken.call_count == expected_calls.get(failure, 1)
        assert evaluation.call_count == (1 if failure == "evaluation" else 0)
    kernels._expert_conv_residual_factory.cache_clear()
    kernels._probe_expert_conv_residual.cache_clear()
    # The existing setting and uncached/multi-token/batched calls bypass fusion.
    monkeypatch.setattr(
        layers,
        "expert_conv_residual_decode",
        Mock(side_effect=AssertionError("unexpected fusion")),
    )
    layer = model.layers[0]
    x = mx.ones((1, 1, 64))
    with monkeypatch.context() as patch:
        patch.setattr(layer, "_self_attention", lambda hidden, *args: hidden)
        expected = layer(x, None)[0]
        actual = layer(x, None, cache=[model.make_cache()[0][0], None])[0]
        assert mx.array_equal(actual, expected).item()
    for batch, length in ((1, 1), (1, 3), (2, 1)):
        ids = mx.ones((batch, length), dtype=mx.int32)
        mx.eval(model(ids))
        cache = model.make_cache()
        mx.eval(model(mx.ones((batch, 3), dtype=mx.int32), cache=cache))
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
        mx.eval(model(ids, cache=cache))
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
        if batch > 1 or length > 1:
            mx.eval(model(ids, cache=cache))


@pytest.mark.requires_mlx
def test_short_conv_weight_rounding(metal_runtime, monkeypatch):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2.layers import ShortConvolution

    conv = ShortConvolution(32, 4)
    mx.random.seed(71)
    conv.conv1d.weight = mx.random.normal((32, 4, 1)).astype(mx.bfloat16)
    state = mx.random.normal((2, 3, 32))
    x = mx.random.normal((2, 1, 32))  # FP32 input, BF16 convolution weights.
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    expected, window = conv(x, state, None)
    rounded = conv.conv1d(window.astype(mx.bfloat16)).astype(mx.float32)
    # Show that the fixture detects omitting the convolution output rounding.
    unrounded = (
        window.astype(mx.bfloat16).astype(mx.float32)
        * conv.conv1d.weight[..., 0].T.astype(mx.float32)[None]
    ).sum(axis=1, keepdims=True)
    assert mx.any(rounded != unrounded).item()
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    actual, _ = conv(x, state, None)
    _assert_conv_close(mx, actual, expected)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_compiled_decode_helpers(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    dtype = getattr(mx, dtype)
    mx.random.seed(32)
    # Changing dimensions in one trace exercises shapeless compilation; these
    # helpers must receive learned weights as arguments rather than capture them.
    for batch, dims in [(1, 16), (3, 32), (2, 16)]:
        x = mx.random.normal((batch, 1, dims)).astype(dtype)
        message = mx.random.normal(x.shape).astype(dtype)
        cases = [
            (layers._norm_cast, (x, mx.random.normal((dims,)), 1e-6, dtype)),
            (layers._value_norm, (x[..., None, :], 1e-6)),
            (layers._scale_states, (x, mx.array(1.13))),
            (layers._gate_attention, (x, message)),
            (
                layers._route_weights,
                (mx.sigmoid(x.astype(mx.float32)), mx.zeros((dims,)), 2, 2.0),
            ),
            (
                layers._combine_experts,
                (
                    mx.random.normal((batch, 1, 3, dims)).astype(dtype),
                    mx.random.uniform(shape=(batch, 1, 3)),
                ),
            ),
            (layers._message_input, (x, message, dtype)),
            (layers._message_mix, (x, message, message * 0.5, x * 0.25)),
            (
                layers._depth_attention_reference,
                (
                    x.astype(mx.float32),
                    mx.random.normal((3, batch, 1, dims)),
                    mx.random.normal((2, dims)),
                    dims**-0.5,
                ),
            ),
        ]
        for fn, args in cases:
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
            reference = fn(*args)
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
            actual = fn(*args)
            if fn is layers._route_weights:
                assert mx.array_equal(actual[0], reference[0]).item()
                actual, reference = actual[1], reference[1]
            assert actual.dtype == reference.dtype
            if actual.dtype in (mx.bfloat16, mx.float16):
                _assert_conv_close(mx, actual, reference)
            else:
                assert mx.allclose(actual, reference, atol=2e-6, rtol=2e-6).item(), fn

        # Output precision can differ from the input (FP32 residuals feeding
        # lower-precision projections). Alternate it on the same compiled helper.
        for output_dtype in (mx.bfloat16, mx.float32):
            for fn, args in (
                (layers._norm_cast, (x, mx.ones((dims,)), 1e-6, output_dtype)),
                (layers._message_input, (x, message, output_dtype)),
            ):
                monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
                reference = fn(*args)
                monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
                actual = fn(*args)
                assert actual.dtype == output_dtype
                if output_dtype == mx.bfloat16:
                    _assert_conv_close(mx, actual, reference)
                else:
                    assert mx.allclose(actual, reference, atol=2e-6, rtol=2e-6).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_decode_message_controller(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2.layers import MessageGate

    dtype = getattr(mx, dtype)
    mx.random.seed(38)
    for batch, dims, rank in [(1, 17, 7), (3, 64, 8), (2, 2304, 64)]:
        gate = MessageGate(dims, rank)
        gate.down_proj.weight = (
            mx.random.normal((rank, 2 * dims)) / (2 * dims) ** 0.5
        ).astype(dtype)
        # Strided FP32 expansion weights, as retained by cast_predicate.
        gate.forget_gate_proj.weight = (mx.random.normal((dims, 2 * rank)) / rank**0.5)[
            :, ::2
        ]
        gate.message_gate_proj.weight = (
            mx.random.normal((dims, 2 * rank)) / rank**0.5
        )[:, ::2]
        for state_dtype in (mx.float32, dtype):
            x = mx.random.normal((batch, 1, 2 * dims)).astype(state_dtype)[..., ::2]
            message = mx.random.normal((batch, 1, 2 * dims)).astype(state_dtype)[
                ..., ::2
            ]
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
            expected = gate(x, message)
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
            actual = gate(x, message)
            assert actual.dtype == expected.dtype == mx.float32
            # The fused expansion uses fp32 per-row reductions; the reference
            # uses GEMM for batch > 1. Expansion weights/output remain fp32
            # even for bf16/fp16 down projections (observed error <= 1.85e-3).
            tolerance = (
                5e-3
                if batch > 1 and _reduced_precision_fp32_gemm(str(mx.default_device()))
                else 5e-6
            )
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_message_norm_exact(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    select = kernels._get_message_norm
    launches = Mock()

    def selected(*args):
        kernel = select(*args)
        assert kernel is not None
        return lambda **kw: (launches(), kernel(**kw))[1]

    monkeypatch.setattr(kernels, "_get_message_norm", selected)
    dtype = getattr(mx, dtype)
    for seed in (0, 17, 38):
        mx.random.seed(seed)
        for dims in (17, 64, 2304, 4096):
            for scale in (0.0, 1e-4, 1.0, 1000.0):
                # Non-contiguous inputs exercise the controller message views.
                x, message = [
                    (mx.random.normal((1, 1, 2 * dims)) * scale).astype(dtype)[..., ::2]
                    for _ in range(2)
                ]
                for target in (mx.float32, mx.float16, mx.bfloat16):
                    expected = layers._message_input(x, message, target)
                    launches.reset_mock()
                    actual = kernels.message_norm_decode(x, message, target)
                    launches.assert_called_once()
                    assert actual is not None and actual.dtype == expected.dtype
                    assert mx.array_equal(actual, expected).item()
                    assert mx.array_equal(
                        actual.view(mx.uint8), expected.view(mx.uint8)
                    ).item()
                mixed = x.astype(mx.float32)
                weight = mx.random.normal((2 * dims,))[::2]
                expected = layers._norm_cast(mixed, weight, 1e-6, dtype)
                launches.reset_mock()
                actual = kernels.message_norm_decode(mixed, None, dtype, weight)
                launches.assert_called_once()
                assert actual is not None and actual.dtype == expected.dtype
                assert mx.array_equal(actual, expected).item()
                assert mx.array_equal(
                    actual.view(mx.uint8), expected.view(mx.uint8)
                ).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_message_norm_cached_logits_exact(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    # Probe outside compilation; retrace each step so instrumentation runs.
    kernels._get_message_norm.cache_clear()
    for source in (mx.float32, getattr(mx, dtype)):
        for target in (mx.float32, getattr(mx, dtype)):
            for output in (False, True):
                assert kernels._get_message_norm(64, output, source, target)
    stock = create_model(metal_runtime, dtype=getattr(mx, dtype))
    fused = create_model(metal_runtime, dtype=getattr(mx, dtype))
    stock_cache, fused_cache = stock.make_cache(), fused.make_cache()
    tokens = mx.random.randint(0, 120, (1, 14))
    assert mx.array_equal(
        stock(tokens[:, :6], cache=stock_cache), fused(tokens[:, :6], cache=fused_cache)
    ).item()
    norm = layers.message_norm_decode
    seen = set()

    def record(x, message, *args):
        out = norm(x, message, *args)
        if out is not None:
            seen.add(message is None)
        return out

    # Disable only this delta: all existing decode fusions stay enabled.
    for step in range(6, 14):
        monkeypatch.setattr(layers, "message_norm_decode", lambda *args: None)
        expected = stock(tokens[:, step : step + 1], cache=stock_cache)
        monkeypatch.setattr(layers, "message_norm_decode", record)
        seen.clear()
        for module in fused.modules():
            for name in list(vars(module)):
                if name.startswith("_block_compiled_"):
                    delattr(module, name)
        actual = fused(tokens[:, step : step + 1], cache=fused_cache)
        # Output fusion requires FP32 norm weights; cast fixtures use stock.
        assert seen == ({False, True} if dtype == "float32" else {False})
        assert actual.dtype == expected.dtype
        assert mx.array_equal(actual, expected).item()
        assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError])
def test_message_norm_invocation_fallback(metal_runtime, monkeypatch, error):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    kernel = Mock(side_effect=error("kernel invocation unavailable"))
    monkeypatch.setattr(kernels, "_get_message_norm", lambda *args: kernel)
    x = mx.ones((1, 1, 64))
    for helper, args in (
        (layers._message_input, (x, x, mx.float16)),
        (layers._norm_cast, (x, mx.ones((64,)), 1e-6, mx.float16)),
    ):
        assert mx.array_equal(helper(*args, decode=True), helper(*args)).item()
    assert kernel.call_count == 2


@pytest.mark.requires_mlx
@pytest.mark.parametrize("failure", ["construction", "invocation", "evaluation"])
def test_message_norm_guards_and_probe_fallback(metal_runtime, monkeypatch, failure):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    kernels._get_message_norm.cache_clear()
    failed = mx.zeros((1,))
    evaluate = mx.eval
    probes = []

    def unavailable(*outputs):
        if any(out is failed for out in outputs):
            probes.append(True)
            raise RuntimeError("kernel evaluation unavailable")
        return evaluate(*outputs)

    monkeypatch.setattr(mx, "eval", unavailable)

    probe_kernel = Mock(
        side_effect=RuntimeError("probe invocation unavailable")
        if failure == "invocation"
        else None,
        return_value=[failed],
    )

    def construct(**kwargs):
        if failure == "construction":
            raise RuntimeError("kernel construction unavailable")
        return probe_kernel

    factory = Mock(side_effect=construct)
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    x = mx.ones((1, 1, 64))
    for helper, args in (
        (layers._message_input, (x, x, mx.float16)),
        (layers._norm_cast, (x, mx.ones((64,)), 1e-6, mx.float16)),
    ):
        for _ in range(2):
            assert mx.array_equal(helper(*args, decode=True), helper(*args)).item()
    assert factory.call_count == 2  # One probe attempt per kernel variant.
    assert probe_kernel.call_count == (0 if failure == "construction" else 2)
    assert len(probes) == (2 if failure == "evaluation" else 0)
    kernels._get_message_norm.cache_clear()
    monkeypatch.setattr(
        kernels, "_get_message_norm", lambda *args: pytest.fail("unsupported path")
    )
    for shape in ((2, 1, 64), (1, 2, 64), (1, 1, 4097)):
        value = mx.ones(shape)
        assert kernels.message_norm_decode(value, value, mx.float32) is None
        assert (
            kernels.message_norm_decode(value, None, mx.float32, mx.ones(shape[-1:]))
            is None
        )
    assert kernels.message_norm_decode(x.astype(mx.int32), x, mx.float32) is None
    assert kernels.message_norm_decode(x, x, mx.int32) is None
    monkeypatch.setattr(
        layers,
        "message_norm_decode",
        lambda *args: pytest.fail("bypassed fast function"),
    )
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    mx.eval(layers._message_input(x, x, mx.float16, decode=True))
    mx.eval(layers._norm_cast(x, mx.ones((64,)), 1e-6, mx.float16, decode=True))
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    mx.eval(layers._message_input(x, x, mx.float16))
    # A singleton forward without a caller-provided cache must remain stock.
    mx.eval(create_model(metal_runtime)(mx.array([[1]])))


@pytest.mark.requires_mlx
@pytest.mark.parametrize("seed", [0, 17, 53])
def test_router_topk_exact(metal_runtime, monkeypatch, seed):
    mx, inkling2 = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    monkeypatch.setattr(layers, "expert_routing", None)
    mx.random.seed(seed)
    for n, k in [(4, 2), (16, 8), (65, 33), (128, 8), (128, 128)]:
        router = layers.TopKRouter(inkling2.ModelArgs.from_dict(CONFIG).text)
        router.weight, router.top_k = mx.eye(n), k
        for dtype in (mx.float32, mx.bfloat16):
            random = mx.random.normal((1, 1, n)).astype(dtype)
            for logits, bias in (
                (random, mx.random.normal((n,))),
                (mx.zeros_like(random), mx.zeros((n,))),
                (random, -mx.sigmoid(random.astype(mx.float32)).reshape(n)),
                (random, (mx.arange(n) % 3 - 1.0) * 1e20),
                ((random * 1000).astype(dtype), mx.arange(n) % 3 - 1.0),
                (mx.full(random.shape, -100.0, dtype=dtype), mx.zeros((n,))),
            ):
                router.e_score_correction_bias = bias
                expected = router(logits)
                scores = mx.sigmoid(logits.astype(mx.float32))
                indices, actual_scores = kernels._route_topk(n, k)(
                    inputs=[logits.astype(mx.float32), bias],
                    template=[("N", n), ("K", k)],
                    grid=(128, 1, 1),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(1, 1, k), logits.shape],
                    output_dtypes=[mx.uint32, mx.float32],
                )
                assert mx.array_equal(indices, expected[0]).item()
                assert mx.array_equal(
                    actual_scores.view(mx.uint32), scores.view(mx.uint32)
                ).item()
                actual = router(logits, True)
                for a, e in zip(actual, expected):
                    assert a.dtype == e.dtype
                    assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("failure", ["construction", "invocation", "evaluation"])
def test_router_topk_gates_and_failure(metal_runtime, monkeypatch, failure):
    mx, inkling2 = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    monkeypatch.setattr(layers, "expert_routing", None)
    router = layers.TopKRouter(inkling2.ModelArgs.from_dict(CONFIG).text)

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported router used decode kernel")

    with monkeypatch.context() as patch:
        patch.setattr(layers, "route_topk_decode", unexpected)
        for shape, cached in [
            ((1, 3, 64), True),
            ((2, 1, 64), True),
            ((1, 1, 64), False),
        ]:
            mx.eval(router(mx.zeros(shape), cached))
        patch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
        mx.eval(router(mx.zeros((1, 1, 64)), True))
        patch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
        patch.setattr(layers, "_use_kernels", lambda: False)
        mx.eval(router(mx.zeros((1, 1, 64)), True))
        patch.setattr(layers, "_use_kernels", lambda: True)
        router.weight = mx.zeros((129, 64))
        router.e_score_correction_bias = mx.zeros((129,))
        mx.eval(router(mx.zeros((1, 1, 64)), True))

    attempts = []
    pending = mx.zeros((1,))
    real_eval = mx.eval

    def broken(*args, **kwargs):
        attempts.append("construction")
        if failure == "construction":
            raise RuntimeError("kernel construction failed")

        def launch(**kwargs):
            attempts.append("invocation")
            if failure == "invocation":
                raise RuntimeError("kernel invocation failed")
            return [pending, pending]

        return launch

    def evaluate(*arrays):
        if any(a is pending for a in arrays):
            attempts.append("evaluation")
            raise RuntimeError("lazy kernel evaluation failed")
        return real_eval(*arrays)

    router = layers.TopKRouter(inkling2.ModelArgs.from_dict(CONFIG).text)
    x = mx.ones((1, 1, 64))
    expected = router(x)
    kernels._route_topk.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(mx.fast, "metal_kernel", broken)
            patch.setattr(mx, "eval", evaluate)
            for _ in range(2):
                actual = router(x, True)
                mx.eval(*actual)
                for a, e in zip(actual, expected):
                    assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()
            stages = ["construction", "invocation", "evaluation"]
            assert attempts == stages[: stages.index(failure) + 1]
    finally:
        kernels._route_topk.cache_clear()


@pytest.mark.requires_mlx
def test_router_topk_real_invocation_failure(metal_runtime, monkeypatch):
    mx, inkling2 = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    monkeypatch.setattr(layers, "expert_routing", None)
    router = layers.TopKRouter(inkling2.ModelArgs.from_dict(CONFIG).text)
    x = mx.ones((1, 1, 64))
    expected = router(x)
    # Warm the cache with a successful construction and evaluated probe.
    assert kernels._route_topk(4, 2) is not None
    launch = Mock(side_effect=RuntimeError("real kernel invocation failed"))
    monkeypatch.setattr(kernels, "_route_topk", lambda n, k: launch)
    actual = router(x, True)
    mx.eval(*actual)
    launch.assert_called_once()
    for a, e in zip(actual, expected):
        assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_router_topk_cached_logits_exact(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    monkeypatch.setattr(layers, "expert_routing", None)
    models = [create_model(metal_runtime, dtype=getattr(mx, dtype)) for _ in range(2)]
    caches = [model.make_cache() for model in models]
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    calls = Mock(wraps=layers.route_topk_decode)
    for step in range(10):
        outputs = []
        for i, (model, cache) in enumerate(zip(models, caches)):
            with monkeypatch.context() as patch:
                patch.setattr(
                    layers, "route_topk_decode", calls if i else lambda *a: None
                )
                ids = tokens[:, step : step + 1]
                out = model(ids, cache=cache)
                mx.eval(out)
                outputs.append(out)
        assert mx.array_equal(
            outputs[0].view(mx.uint32), outputs[1].view(mx.uint32)
        ).item()
        if step == 0:
            assert calls.call_count == 0  # A one-token prefill has no cached history.
    assert calls.call_count > 0


@pytest.mark.requires_mlx
def test_decode_router_and_positions(metal_runtime, monkeypatch):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    mx.random.seed(67)
    for batch, experts, k in [(1, 128, 8), (3, 16, 2), (2, 65, 33)]:
        scores = mx.sigmoid(mx.random.normal((batch, 1, experts)))
        for bias in (mx.zeros((experts,)), mx.random.normal((experts,))):
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
            expected_idx, expected = layers._route_weights(scores, bias, k, 2.3)
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
            actual_idx, actual = layers._route_weights(scores, bias, k, 2.3)
            assert mx.array_equal(actual_idx, expected_idx).item()
            _assert_conv_close(mx, actual, expected)
        # Every expert ties: preserve argpartition's choice and ordering.
        tied = mx.full((batch, 1, experts), 0.5)
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
        expected_idx, expected = layers._route_weights(
            tied, mx.zeros((experts,)), k, 2.3
        )
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
        actual_idx, actual = layers._route_weights(tied, mx.zeros((experts,)), k, 2.3)
        assert mx.array_equal(actual_idx, expected_idx).item()
        _assert_conv_close(mx, actual, expected)

    # All values that change between tokens are array inputs to the trace.
    for offset in (0, 3, 5, 8192, 100000):
        for offsets in (mx.array([[offset]]), mx.array([[offset], [offset + 19]])):
            for floor, alpha in [(4.0, 0.1), (8192.0, 0.25)]:
                expected = 1 + alpha * mx.log(
                    mx.maximum((offsets.astype(mx.float32) + 1) / floor, 1)
                )
                actual = layers._decode_log_scale(
                    offsets, mx.array(floor), mx.array(alpha)
                )
                _assert_conv_close(mx, actual, expected)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_decode_sdpa_ragged_ring(metal_runtime, monkeypatch, batch, sliding, dtype):
    mx, inkling2 = metal_runtime
    from mlx_lm.models.cache import BatchRotatingKVCache

    from rapid_mlx.models.inkling2.layers import (
        Attention,
        _attention_reference,
        _key_positions,
        _positions,
    )

    args = inkling2.ModelArgs.from_dict(CONFIG).text
    attention = Attention(args, sliding=sliding)
    mx.random.seed(46)
    dtype = getattr(mx, dtype)
    hq, hkv, initial_dim = (
        attention.num_heads,
        attention.num_key_value_heads,
        attention.head_dim,
    )
    # Head dim 16 exercises SDPA's fallback; 64 exercises the native decode
    # kernel. Vary extent too, so the compiled mask cannot capture its first R.
    for dim, size, extent in [(initial_dim, 8, 4), (64, 13, 8), (64, 8, 4)]:
        for idx in (0, 1, size - 1):
            query = (0.5 * mx.random.normal((batch, hq, 1, dim))).astype(dtype)
            key = (0.5 * mx.random.normal((batch, hkv, size, dim))).astype(dtype)
            value = mx.random.normal((batch, hkv, size, dim)).astype(dtype)
            bias = mx.random.normal((batch, 1, hq, extent)).astype(dtype)
            sinks = mx.random.normal((batch, 1, hq)).astype(dtype)
            # Include a row with only the sink visible (end=0), partial history,
            # and history beyond the window. Keys use physical ring ordering.
            end_values = (
                [0, 3, 29]
                if batch > 1
                else [0 if idx == 0 else (3 if idx == 1 else 29)]
            )
            ends = mx.array(end_values)
            cache = BatchRotatingKVCache(max_size=size, left_padding=[0] * batch)
            cache.offset, cache._idx = ends, idx
            reference = _attention_reference(
                query,
                key,
                value,
                bias,
                sinks,
                scale=attention.scaling,
                window=attention.sliding_window,
                query_positions=_positions(ends - 1, 1),
                key_positions=_key_positions(cache, size, 1, ends),
            )
            actual = attention._attention(query, key, value, bias, sinks, cache)
            mx.eval(actual, reference)
            tolerance = 1e-2 if dtype == mx.bfloat16 else 2e-6
            assert mx.allclose(
                actual.astype(mx.float32),
                reference.astype(mx.float32),
                atol=tolerance,
                rtol=tolerance,
            ).item()
            if end_values[0] == 0:
                assert mx.array_equal(actual[0], mx.zeros_like(actual[0])).item()
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
            fallback = attention._attention(query, key, value, bias, sinks, cache)
            assert mx.array_equal(fallback, reference).item()
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_decode_mask_exact(metal_runtime, dtype):
    mx, _ = metal_runtime
    from mlx_lm.models.cache import BatchRotatingKVCache

    from rapid_mlx.models.inkling2 import kernels, layers

    mx.random.seed(71)
    dtype = getattr(mx, dtype)
    for batch, size, extent in [(1, 13, 4), (3, 8, 16), (2, 1024, 128)]:
        # Bias may be strided and have a different dtype from Q/K/V.
        bias = mx.random.normal((batch, 1, 4, 2 * extent))[..., ::2]
        for rotating in (False, True):
            cache = BatchRotatingKVCache(max_size=size, left_padding=[0] * batch)
            for idx in (0, 1, size):
                for end in (3, 29, mx.array([0] + [29] * (batch - 1))):
                    cache._idx = idx
                    positions = layers._key_positions(
                        cache if rotating else None, size, 1, end
                    )
                    queries = layers._positions(end - 1, 1)
                    distance = queries[..., :, None] - positions[..., None, :]
                    for window in (0, 4):
                        visible = (distance >= 0) & (positions[..., None, :] >= 0)
                        if window:
                            visible &= distance < window
                        distance_rows = mx.broadcast_to(distance, (batch, 1, size))[
                            :, None
                        ]
                        visible_rows = mx.broadcast_to(visible, (batch, 1, size))[
                            :, None
                        ]
                        expected = layers._decode_attention_mask(
                            bias, distance_rows, visible_rows, dtype, extent
                        )
                        actual = kernels.attention_mask_decode(
                            bias, end, idx, size, window, dtype, rotating=rotating
                        )
                        assert mx.array_equal(actual, expected).item()


@pytest.mark.requires_mlx
def test_decode_kill_switch_and_prefill(metal_runtime, monkeypatch):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    def unexpected(*args, **kwargs):
        pytest.fail("decode convolution called for prefill or with kill switch off")

    for name in (
        "short_convolution_decode",
        "message_mix_decode",
        "attention_mask_decode",
        "route_weights_decode",
        "_decode_log_scale",
    ):
        monkeypatch.setattr(layers, name, unexpected)
    for conv in (
        layers.ShortConvolution(16, 4),
        layers.DynamicShortConvolution(16, 16, 2, 4),
    ):
        for enabled, length in [("1", 3), ("0", 1)]:
            monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", enabled)
            x = mx.ones((2, length, 16))
            args = (
                (x, x, None, None)
                if isinstance(conv, layers.DynamicShortConvolution)
                else (x, None, None)
            )
            out, window = conv(*args)
            mx.eval(out, window)
            assert window.shape == (2, length + 3, 16)


def create_model(runtime, *, dtype=None, loop=True):
    mx, inkling2 = runtime
    from mlx.utils import tree_flatten, tree_unflatten

    config = json.loads(json.dumps(CONFIG))
    if not loop:
        config["text_config"].pop("loop_config")
    mx.random.seed(0)
    model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        if key.endswith(("norm.weight", "l2_clip.scale")):
            value = 1 + 0.1 * mx.random.normal(value.shape)
        else:
            fan_in = value.shape[-1] if value.ndim > 1 else 16
            value = mx.random.normal(value.shape) * fan_in**-0.5
        # Exercise learned queries, router and message gates rather than their
        # zero initialization. Keep precision-sensitive parameters in float32.
        if dtype is not None and model.cast_predicate(key):
            value = value.astype(dtype)
        weights.append((key, value))
    model.update(tree_unflatten(weights))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_catalog():
    assert resolve_model("inkling-2-mini-4bit") == REPO
    assert resolve_revision("inkling-2-mini-4bit") == REVISION
    for name in ("inkling-2-mini-4bit", REPO):
        p = resolve_profile(name)
        assert p.model_type == "inkling2_loop"
        assert p.is_moe and p.is_text_only and p.modality == "text"
        assert p.is_hybrid and p.is_hybrid_explicit
        assert not p.supports_spec_decode and not p.mtp_default_enabled
        assert p.tool_call_parser == "inkling" and p.reasoning_parser == "inkling"
        assert detect_model_config(name) == p


@pytest.mark.parametrize("directory", ["checkpoint", "Qwen3-8B"])
def test_local_metadata(tmp_path, directory):
    tmp_path = tmp_path / directory
    tmp_path.mkdir()
    config = {**CONFIG, "vision_config": {"hidden_size": 1024}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    p = detect_model_config(str(tmp_path))
    assert p.is_moe and p.is_text_only and p.is_hybrid and p.is_hybrid_explicit
    assert not p.supports_spec_decode
    assert p.tool_call_parser == "inkling" and p.reasoning_parser == "inkling"


def test_preflight():
    from rapid_mlx.byom.preflight import RAPID_MODEL_TYPES

    assert "inkling2_loop" in RAPID_MODEL_TYPES


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.requires_mlx
def test_forward(runtime, dtype):
    mx, _ = runtime
    model = create_model(runtime, dtype=getattr(mx, dtype))
    logits = model(mx.array([[1, 2, 3], [4, 5, 6]]))
    mx.eval(logits)
    assert logits.shape == (2, 3, 120)
    assert logits.dtype == mx.float32
    assert mx.all(mx.isfinite(logits)).item()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.requires_mlx
def test_cached_decode_matches_full(runtime, batch, loop):
    mx, _ = runtime
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    reduced_precision = _reduced_precision_fp32_gemm(str(mx.default_device()))
    if loop and batch == 2 and reduced_precision:
        pytest.skip("Routing flips amplify reduced-precision GEMM through loop passes")

    # Full-model cache parity uses float32, as in the source regression.
    # In bf16, prefill/decode use different attention kernels; rounding can
    # change top-k expert selection and amplify differences through loop passes.
    model = create_model(runtime, dtype=mx.float32, loop=loop)
    tokens = mx.random.randint(0, 120, (batch, 20))
    full = model(tokens)
    cache = model.make_cache()
    visits = model.model.language_model.layer_visits
    assert len(cache) == (14 if loop else 8)
    assert any(isinstance(c[0], RotatingKVCache) for c in cache)
    assert any(type(c[0]) is KVCache for c in cache)
    assert len({id(c) for c in cache}) == len(visits)
    # Chunked prefill crosses the short sliding window, followed by single
    # tokens to cover ring-buffer wraparound and the singleton SDPA path.
    chunks = [model(tokens[:, :6], cache=cache), model(tokens[:, 6:12], cache=cache)]
    chunks += [model(tokens[:, i : i + 1], cache=cache) for i in range(12, 20)]
    actual = mx.concatenate(chunks, axis=1)
    mx.eval(full, actual)
    # Full sequence, chunks and singleton decode use different GEMM row counts
    # and SDPA kernels. M5 fp32 SDPA alone can differ by about 1.5e-3.
    tolerance = REDUCED_PRECISION_MODEL_TOLERANCE if reduced_precision else 1e-4
    assert mx.allclose(actual, full, atol=tolerance, rtol=tolerance).item()
    assert all(c[0].offset == 20 for c in cache)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("tie_delta", [0.0, 2**-20, -(2**-20)])
def test_compiled_router_preserves_cutoff_ties(
    metal_runtime, monkeypatch, batch, dtype, tie_delta
):
    mx, inkling2 = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    # Exercise the production fallback even when the optional helper exists.
    monkeypatch.setattr(layers, "expert_routing", None)
    router = layers.TopKRouter(inkling2.ModelArgs.from_dict(CONFIG).text)
    mx.random.seed(17)
    weight = 0.1 * mx.random.normal(router.weight.shape)
    # Duplicate rows make the logits identical for experts 1 and 2 for every
    # token, including after bf16 input rounding. Biases force expert 0 above
    # and expert 3 below them, placing the tie at the top-2 cutoff.
    router.weight = mx.concatenate([weight[:2], weight[1:2], weight[3:]], axis=0)
    router.e_score_correction_bias = mx.array([2.0, 0.125, 0.125 + tie_delta, -2.0])
    hidden = mx.random.normal((batch, 1, router.weight.shape[-1])).astype(
        getattr(mx, dtype)
    )
    mx.eval(router.parameters(), hidden)

    corrected = (
        mx.sigmoid(hidden.astype(mx.float32) @ router.weight.T)
        + router.e_score_correction_bias
    )
    if tie_delta == 0:
        assert mx.array_equal(corrected[..., 1], corrected[..., 2]).item()
    else:
        gap = corrected[..., 2] - corrected[..., 1]
        assert mx.all(gap * tie_delta > 0).item()
        assert mx.all(mx.abs(gap) < 2**-18).item()
    assert mx.all(corrected[..., 0] > corrected[..., 1:3].max(axis=-1)).item()
    assert mx.all(corrected[..., 3] < corrected[..., 1:3].min(axis=-1)).item()

    eager_indices, eager_weights = router(hidden)
    mx.eval(eager_indices, eager_weights)
    compiled = mx.compile(lambda x: router(x), inputs=router.state, shapeless=False)
    compiled_indices, compiled_weights = compiled(hidden)
    mx.eval(compiled_indices, compiled_weights)
    assert mx.array_equal(compiled_indices, eager_indices).item()
    assert compiled_weights.dtype == eager_weights.dtype == mx.float32
    assert mx.array_equal(
        compiled_weights.view(mx.uint32), eager_weights.view(mx.uint32)
    ).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("loop", [False, True])
def test_block_compile_decode_parity(metal_runtime, monkeypatch, batch, loop):
    mx, _ = metal_runtime
    from mlx.utils import tree_flatten, tree_unflatten

    from rapid_mlx.models.inkling2 import layers

    monkeypatch.setattr(layers, "expert_routing", None)
    model = create_model(metal_runtime, dtype=mx.float32, loop=loop)
    tokens = mx.random.randint(0, 120, (batch, 16))
    reference_cache, compiled_cache = model.make_cache(), model.make_cache()
    # Loop messages use batched eager GEMM expansion versus per-row fp32
    # fused reductions. Differences propagate into logits and both cache banks.
    tolerance = (
        REDUCED_PRECISION_MODEL_TOLERANCE
        if loop and batch > 1 and _reduced_precision_fp32_gemm(str(mx.default_device()))
        else 1e-4
    )

    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    reference = model(tokens[:, :6], cache=reference_cache)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    actual = model(tokens[:, :6], cache=compiled_cache)
    assert mx.array_equal(actual, reference).item()
    decoder_layers = model.model.language_model.layers
    assert all(
        not hasattr(layer.mlp, "_block_compiled__forward") for layer in decoder_layers
    )

    # Cross the rotating KV window and exercise messages on repeated layer visits.
    for step in range(6, 16):
        if step == 10:
            # Same shapes, new arrays: warmed graphs must read live weights,
            # including nested expert weights, rather than frozen trace values.
            model.update(
                tree_unflatten(
                    [
                        (name, weight * 1.01)
                        for name, weight in tree_flatten(model.parameters())
                    ]
                )
            )
            mx.eval(model.parameters())
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
        reference = model(tokens[:, step : step + 1], cache=reference_cache)
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
        actual = model(tokens[:, step : step + 1], cache=compiled_cache)
        assert mx.allclose(actual, reference, atol=tolerance, rtol=tolerance).item()
        for compiled, eager in zip(compiled_cache, reference_cache):
            assert compiled[0].offset == eager[0].offset == step + 1
            for compiled_state, eager_state in zip(compiled[0].state, eager[0].state):
                assert mx.allclose(
                    compiled_state, eager_state, atol=tolerance, rtol=tolerance
                ).item()
            for compiled_state, eager_state in zip(compiled[1].cache, eager[1].cache):
                assert mx.allclose(
                    compiled_state, eager_state, atol=tolerance, rtol=tolerance
                ).item()

    for layer in decoder_layers:
        assert hasattr(layer.mlp, "_block_compiled__forward")
        assert hasattr(layer, "_block_compiled__mlp_states")
        assert hasattr(layer, "_block_compiled__attention_states")


@pytest.mark.requires_mlx
@pytest.mark.parametrize("batch", [1, 3])
def test_fast_decode_model_parity(metal_runtime, monkeypatch, batch):
    mx, _ = metal_runtime
    model = create_model(metal_runtime, dtype=mx.float32)
    tokens = mx.random.randint(0, 120, (batch, 14))
    reference_cache, fast_cache = model.make_cache(), model.make_cache()
    # As in block-compile parity, batched GEMM expansion and per-row fused
    # reductions can differ; preserve the strict singleton comparison.
    tolerance = (
        REDUCED_PRECISION_MODEL_TOLERANCE
        if batch > 1 and _reduced_precision_fp32_gemm(str(mx.default_device()))
        else 1e-4
    )
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    reference_prefill = model(tokens[:, :6], cache=reference_cache)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    fast_prefill = model(tokens[:, :6], cache=fast_cache)
    assert mx.array_equal(fast_prefill, reference_prefill).item()
    # Follow the same teacher-forced tokens through loop messages and all four
    # convolution states at each visit, crossing the rotating cache's wrap.
    for step in range(6, 14):
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
        reference = model(tokens[:, step : step + 1], cache=reference_cache)
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
        actual = model(tokens[:, step : step + 1], cache=fast_cache)
        assert mx.allclose(actual, reference, atol=tolerance, rtol=tolerance).item()
        for fast, slow in zip(fast_cache, reference_cache):
            assert fast[0].offset == slow[0].offset == step + 1
            for fast_state, slow_state in zip(fast[1].cache, slow[1].cache):
                assert mx.allclose(
                    fast_state, slow_state, atol=tolerance, rtol=tolerance
                ).item()


@pytest.mark.requires_mlx
def test_fast_decode_specializations_stay_bounded(
    metal_runtime, monkeypatch, compile_traces
):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    # Exercise all local decode kernels, including the routing fallback used
    # when the optional upstream fused routing helper is unavailable.
    monkeypatch.setattr(layers, "expert_routing", None)
    model = create_model(metal_runtime, dtype=mx.float32, loop=True)
    tokens = [mx.random.randint(0, 120, (batch, 46)) for batch in (1, 2, 3)]
    caches = [model.make_cache() for _ in tokens]
    for ids, cache in zip(tokens, caches):
        mx.eval(model(ids[:, :6], cache=cache))

    keys = {}

    def record(name, kernel):
        keys[name] = set()

        def launch(**kwargs):
            keys[name].add(tuple(kwargs["template"]))
            return kernel(**kwargs)

        return launch

    for name, kernel in vars(kernels).copy().items():
        if name.endswith("_kernel"):
            monkeypatch.setattr(kernels, name, record(name, kernel))
    topk = record("route_topk", kernels._route_topk(4, 2))
    monkeypatch.setattr(kernels, "_route_topk", lambda n, k: topk)

    expected_keys = expected_traces = None
    for step in range(6, 46):
        for ids, cache in zip(tokens, caches):
            logits = model(ids[:, step : step + 1], cache=cache)
            mx.eval(logits)
            assert mx.all(mx.isfinite(logits)).item()
            assert all(c[0].offset == step + 1 for c in cache)
            if expected_keys is not None:
                assert keys == expected_keys
            traces = compile_traces.copy()
            if expected_traces is not None:
                assert traces == expected_traces
        if expected_traces is None:
            # Warm every batch size once: block graphs trace per input shape.
            # Growing KV banks and revisiting these sizes must add no traces.
            expected_traces = traces.copy()
            expected_keys = {name: seen.copy() for name, seen in keys.items()}
    assert all(
        seen for name, seen in keys.items() if name != "_relative_attention_kernel"
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.requires_mlx
def test_bfloat16_attention_cache_matches_full(runtime, batch, sliding):
    mx, inkling2 = runtime
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from rapid_mlx.models.inkling2.layers import Attention

    args = inkling2.ModelArgs.from_dict(CONFIG).text
    attention = Attention(args, sliding=sliding)
    mx.random.seed(3)
    length = 20
    hq, hkv, dim = (
        attention.num_heads,
        attention.num_key_value_heads,
        attention.head_dim,
    )
    query = (0.5 * mx.random.normal((batch, hq, length, dim))).astype(mx.bfloat16)
    key = (0.5 * mx.random.normal((batch, hkv, length, dim))).astype(mx.bfloat16)
    value = mx.random.normal((batch, hkv, length, dim)).astype(mx.bfloat16)
    bias = mx.random.normal((batch, length, hq, args.rel_extent)).astype(mx.bfloat16)
    sinks = mx.random.normal((batch, length, hq)).astype(mx.bfloat16)
    full = attention._attention(query, key, value, bias, sinks, None)
    cache = RotatingKVCache(max_size=args.sliding_window_size) if sliding else KVCache()
    chunks = []
    # Identical bf16 projections isolate cache positions and kernel numerics
    # from projection rounding and discontinuous expert routing downstream.
    for start, end in [(0, 6), (6, 12)] + [(i, i + 1) for i in range(12, length)]:
        keys, values = cache.update_and_fetch(
            key[:, :, start:end], value[:, :, start:end]
        )
        chunks.append(
            attention._attention(
                query[:, :, start:end],
                keys,
                values,
                bias[:, start:end],
                sinks[:, start:end],
                cache,
            )
        )
    actual = mx.concatenate(chunks, axis=1)
    mx.eval(full, actual)
    # The source kernel tests use TOL[mx.bfloat16] == 1e-2, with a float32
    # comparison. Whole-model bf16 logits have no upstream parity tolerance.
    assert mx.allclose(
        actual.astype(mx.float32), full.astype(mx.float32), atol=1e-2, rtol=1e-2
    ).item()
    assert cache.offset == length


@pytest.mark.parametrize("prefix", ["model.visual.", "model.vision_tower."])
@pytest.mark.requires_mlx
def test_sanitize_drops_vision(runtime, prefix):
    mx, inkling2 = runtime
    config = {**CONFIG, "vision_config": {"hidden_size": 1024}}
    model = inkling2.Model(inkling2.ModelArgs.from_dict(config))
    assert model.args.vision is None
    assert "vision_tower" not in model.model
    text_weight = mx.ones((128, 64))
    weights = {
        "model.llm.embed.weight": text_weight,
        prefix + "backbone.layers.0.weight": mx.zeros((1,)),
    }
    sanitized = model.sanitize(weights)
    assert set(sanitized) == {"model.language_model.embed_tokens.weight"}
    assert mx.array_equal(next(iter(sanitized.values())), text_weight).item()


_CHECKPOINT_NAMES = [
    (r"^lm_head\.", "model.llm.unembed."),
    (r"^model\.language_model\.embed_tokens\.", "model.llm.embed."),
    (r"^model\.language_model\.", "model.llm."),
    (r"\.final_input\.query$", ".attn_res.output_query"),
    (r"\.layers\.(\d+)\.mlp_input\.query$", r".attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.", ".attn.wv_dv."),
    (r"\.self_attn\.r_proj\.", ".attn.wr_du."),
    (r"\.self_attn\.o_proj\.", ".attn.wo_ud."),
    (r"\.self_attn\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.", ".attn."),
    (r"\.q_message_gate\.", ".controllers.attn."),
    (r"\.(k|v|mlp)_message_gate\.", r".controllers.\1."),
    (r"(\.controllers\.\w+)\.down_proj\.weight$", r"\1.down_weight"),
    (r"(\.controllers\.\w+)\.forget_gate_proj\.weight$", r"\1.forget_weight"),
    (r"(\.controllers\.\w+)\.message_gate_proj\.weight$", r"\1.gain_weight"),
    (r"\.post_attention_convolution\.", ".attn_sconv."),
    (r"\.post_mlp_convolution\.", ".mlp_sconv."),
    (r"\.conv1d\.weight$", ".weight"),
    (r"sconv\.static_weight$", "sconv.static_w"),
    (r"sconv\.weight_proj\.weight$", "sconv.weight_proj"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
]


def checkpoint(model) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    weights = dict(tree_flatten(model.parameters()))
    ckpt = {}
    denominator = mx.array([7.5])
    for k, v in weights.items():
        if k.endswith(".message_input.query"):
            q, kv = mx.concatenate([v[:, :1], v[:, 3:]], axis=1), v[:, 1:3]
            ckpt["model.llm.middle_cycle_cross_pass_queries"] = q.flatten(0, 1)
            ckpt["model.llm.middle_cycle_cross_pass_kv_queries"] = kv.flatten(0, 1)
            continue
        if m := re.match(r"(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$", k):
            prefix, name, param = m.groups()
            prefix = prefix.replace("model.language_model.", "model.llm.")
            w = f"{prefix}.{'w13' if name == 'gate_up' else 'w2'}_weight"
            if param == "weight":
                ckpt[w] = v.view(mx.uint8)
            elif param == "scales":
                ckpt[f"{w}_scale"] = v
            else:
                ckpt[f"{w}_amax"] = v * denominator / 2688.0
                ckpt[f"{w}_outer_scale_denominator"] = denominator
            continue
        if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
            ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
            continue
        for pattern, replacement in _CHECKPOINT_NAMES:
            k = re.sub(pattern, replacement, k)
        if k.endswith("sconv.weight"):
            v = v.swapaxes(1, 2)
        ckpt[k] = v
    return ckpt


@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.requires_mlx
def test_sanitize_mixed_experts(runtime, converted, legacy, monkeypatch):
    mx, inkling2 = runtime
    from mlx import nn
    from mlx.utils import tree_flatten

    if legacy:
        from rapid_mlx.models.inkling2 import checkpoint as checkpoint_module
        from rapid_mlx.models.inkling2 import layers, switch_layers

        monkeypatch.setattr(layers, "SwitchLinear", switch_layers.SwitchLinear)
        monkeypatch.setattr(
            checkpoint_module, "SwitchLinear", switch_layers.SwitchLinear
        )

    model = create_model(runtime)

    def predicate(path, module):
        # The source quantization helper applies this capability guard before
        # invoking its expert predicate. nn.quantize calls ours directly.
        if not hasattr(module, "to_quantized") or ".mlp.experts." not in path:
            return False
        if ".layers.7." in path:
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}

    nn.quantize(model, class_predicate=predicate)
    expected = dict(tree_flatten(model.parameters()))
    weights = expected if converted else checkpoint(model)
    target = inkling2.Model(inkling2.ModelArgs.from_dict(CONFIG))
    sanitized = target.sanitize(weights)
    target.load_weights(list(sanitized.items()), strict=True)
    assert set(sanitized) == set(expected)
    for name, value in expected.items():
        assert mx.allclose(sanitized[name], value, atol=1e-6).item(), name
    tokens = mx.array([[1, 2, 3]])
    # Native mixed-expert quantization/loading can select different matmul
    # kernels. Keep tensor conversion checks above strict.
    tolerance = (
        REDUCED_PRECISION_MODEL_TOLERANCE
        if _reduced_precision_fp32_gemm(str(mx.default_device()))
        else 1e-4
    )
    assert mx.allclose(
        model(tokens), target(tokens), atol=tolerance, rtol=tolerance
    ).item()


@pytest.mark.requires_mlx
def test_public_loader_text_only(runtime, tmp_path):
    mx, _ = runtime
    from mlx import nn
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    source = create_model(runtime)
    nn.quantize(
        source,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: (
            hasattr(m, "to_quantized") and source.quant_predicate(p, m)
        ),
    )
    config = {
        **CONFIG,
        "vision_config": {"hidden_size": 1024},
        "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
        "model_file": "untrusted.py",
        "auto_map": {"AutoModel": "untrusted.Model"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "untrusted.py").write_text(
        "raise AssertionError('checkpoint code ran')"
    )
    weights = dict(tree_flatten(source.parameters()))
    weights["model.vision_tower.weight"] = mx.zeros((2, 2))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    Tokenizer(WordLevel({"<unk>": 0, "hi": 1, "<eos>": 2})).save(
        str(tmp_path / "tokenizer.json")
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>", "unk_token": "<unk>"})
    )
    # A fresh process enters through the production loader before importing the
    # architecture; no direct model import can hide missing registration.
    script = """
import sys
import mlx.core as mx
from rapid_mlx.utils.tokenizer import load_model_with_fallback, _VENDORED_MODEL_TYPES
model, tokenizer = load_model_with_fallback(sys.argv[1])
from rapid_mlx.models.inkling2 import Model
from mlx_lm.utils import _get_classes
assert isinstance(model, Model)
assert _get_classes({"model_type": "inkling2_loop"})[0] is Model
assert "inkling2_loop" in _VENDORED_MODEL_TYPES
mx.save(sys.argv[2], model(mx.array([[1, 2, 3]])))
"""
    output = tmp_path / "logits.npy"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(output)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert mx.allclose(
        source(mx.array([[1, 2, 3]])), mx.load(str(output)), atol=1e-4
    ).item()
