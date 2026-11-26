# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
# Tiny configuration and checkpoint conversion adapted from the source in
# rapid_mlx/models/inkling2/NOTICE, under its accompanying MIT license.
"""Text-only catalog, loop/cache, quantization and loader regressions."""

import copy
import functools
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import ModuleType, SimpleNamespace
from unittest.mock import DEFAULT, Mock

import numpy as np
import pytest

from rapid_mlx.model_aliases import resolve_model, resolve_profile, resolve_revision
from rapid_mlx.model_auto_config import detect_model_config

REPO = "mlx-community/Inkling-2-Mini-4bit"
REVISION = "9856dc53edead71405f8348930937c3638ff21a2"
REDUCED_PRECISION_MODEL_TOLERANCE = 2e-2
# For <=128 positive FP32 terms: plain ops use up to 64 serial additions,
# decode uses <=4 per lane + a 32-lane tree. Their rounding bounds plus
# division fit 5e-6 relative; use 1e-8 absolute near zero. No plain-op bit parity.
ROUTER_PLAIN_OP_RTOL = 5e-6
ROUTER_PLAIN_OP_ATOL = 1e-8

# Frozen origin/main Metal sources: the shipped two-dispatch decode baseline.
# Keep these independent of the fused implementation and the plain-op path.
_MAIN_ROUTE_SELECT_SOURCE = """
                #pragma clang fp contract(off)
                const uint tid = thread_position_in_threadgroup.x;
                threadgroup float scores[N], order[N];
                if (tid < N) {
                    float x = logits[tid * logits_strides[2]];
                    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    scores[tid] = x < 0.0f ? y : 1.0f - y;
                    out_scores[tid] = scores[tid];
                    order[tid] = -(scores[tid] + bias[tid * bias_strides[0]]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < N) {
                    uint rank = 0;
                    const float a = order[tid];
                    for (uint j = 0; j < N; ++j) {
                        float b = order[j];
                        bool less = metal::isnan(a) ? !metal::isnan(b)
                            : !metal::isnan(b) && b < a;
                        bool equal = a == b || (metal::isnan(a) && metal::isnan(b));
                        rank += less || (equal && j < tid);
                    }
                    if (rank < K) {
                        indices[rank] = tid;
                    }
                }
            """
_MAIN_ROUTE_WEIGHTS_SOURCE = """
    #pragma clang fp contract(off)
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    float total = 0.0f;
    for (uint j = lane; j < K; j += 32) {
        total += scores[row * scores_strides[0]
                        + indices[row * indices_strides[0] + j * indices_strides[1]] * scores_strides[1]];
    }
    total = simd_sum(total) + 1e-20f;
    for (uint j = lane; j < K; j += 32) {
        const float s = scores[row * scores_strides[0]
                        + indices[row * indices_strides[0] + j * indices_strides[1]] * scores_strides[1]];
        out[row * K + j] = (s / total) * route_scale;
    }
"""
_MAIN_ROUTE_WEIGHTS_BLOCK_SHA256 = (
    "ce7f5b683d2c13e7ecbcf6e00a0966f817ad3c13e41446e603cfb8f63404dc18"
)


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


# Complete reviewed source of the five omitted numerical-core files and the
# decode kernels. See NOTICE for provenance. Review edits before updating pins;
# this contract reads bytes only and requires neither MLX nor Metal.
_NUMERICAL_CORE_SHA256 = {
    "layers.py": "945ffee6cbce2afde0166b854c578209c01c3380eb21f80695a18d3523802d99",
    "kernels.py": "557e6fd704d311cfb615173845bf0057aa64b7f2f0b284286164fc005ccdb7ae",
    "model.py": "4dee052b0aaaf5df2a90b9f1d80b215fd4546b7ee55367589fb30d84a3e9cd08",
    "switch_layers.py": "1df554e66c4a627ffb1e6f91989abe65c4663a457977d05fdffbe25771e7fc47",
    "checkpoint.py": "7ba19519e069ce47319641f473a696dd3cfa82c246f2eec40949268a6bd2935f",
    "nvfp4_decode.py": "b40d0dee816f30612de4b3541097f1f3de692251418d4b5c7f3493063d750360",
    "visit_norm.py": "389750faffd0d7843e92ad1550a7f80790e3122ed26f40000be8ab200a75bc4d",
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
    assert omitted == _NUMERICAL_CORE_SHA256.keys() - {
        "nvfp4_decode.py",
        "visit_norm.py",
    }
    assert _NUMERICAL_CORE_SHA256.keys() == {
        path.name for path in (root / prefix).glob("*.py")
    } - {"__init__.py", "compat.py", "config.py"}
    for filename, expected in _NUMERICAL_CORE_SHA256.items():
        assert hashlib.sha256((root / prefix / filename).read_bytes()).hexdigest() == (
            expected
        ), filename


def test_main_decode_normalization_source_unchanged(decode_cpu):
    """Pin main's exact normalization source, construction and launch wrapper."""
    source = (
        Path(__file__).parents[1] / "rapid_mlx/models/inkling2/kernels.py"
    ).read_text()
    block = source[
        source.index("_ROUTE_WEIGHTS_SOURCE =") : source.index("_route_topk_lock =")
    ]
    assert (
        hashlib.sha256(block.encode()).hexdigest() == _MAIN_ROUTE_WEIGHTS_BLOCK_SHA256
    )
    assert decode_cpu.kernels._ROUTE_WEIGHTS_SOURCE == _MAIN_ROUTE_WEIGHTS_SOURCE
    assert (
        hashlib.sha256(_MAIN_ROUTE_SELECT_SOURCE.encode()).hexdigest()
        == "875073aafb57396901e7c524f012014b8467023ce8796edb348e57e04d6940ee"
    )


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
    g.load("visit_norm")
    layers = g.load("layers")
    monkeypatch.setattr(layers, "_use_kernels", lambda: True)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    return SimpleNamespace(kernels=kernels, layers=layers, mx=mx)


@pytest.fixture
def visit_cpu(decode_cpu):
    decode_cpu.mx.fast.metal_kernel.reset_mock()
    visit = sys.modules["rapid_mlx.models.inkling2.visit_norm"]
    args = (
        np.ones((2, 1, 1, 64), np.float32),
        np.ones((1, 1, 64), np.float16),
        np.ones((3, 64), np.float32),
    )
    return SimpleNamespace(**vars(decode_cpu), visit=visit, args=args)


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "failure", [None, "construction", "probe", "evaluation", "invocation"]
)
def test_visit_norm_cpu_lifecycle(visit_cpu, append, failure):
    c = visit_cpu
    probe, output = [object() for _ in range(3)], [object() for _ in range(3)]
    kernel = Mock(side_effect=[probe, output, output, RuntimeError("invoke")])
    c.mx.fast.metal_kernel.return_value = kernel
    if failure == "construction":
        c.mx.fast.metal_kernel.side_effect = RuntimeError("build")
    elif failure == "probe":
        kernel.side_effect = RuntimeError("probe")
    elif failure == "evaluation":
        c.mx.eval.side_effect = RuntimeError("eval")
    elif failure == "invocation":
        kernel.side_effect = [probe, RuntimeError("invoke")]
    fn = functools.partial(c.visit.visit_norm_decode, *c.args, append_block=append)
    if failure is None:
        assert fn() == output and fn() == output
        # Real lazy outputs are returned without evaluating them.
        c.mx.eval.assert_called_once_with(*probe)
        assert fn() is None
    for _ in range(2):
        assert fn() is None
    c.mx.fast.metal_kernel.assert_called_once()
    assert (
        kernel.call_count
        == {None: 4, "construction": 0, "probe": 1, "evaluation": 1, "invocation": 2}[
            failure
        ]
    )
    assert c.mx.eval.call_count == int(failure not in ("construction", "probe"))
    if failure != "construction":
        launch = kernel.call_args.kwargs
        assert launch["output_shapes"] == [
            (1, 1, 3, 64),
            (3, 1, 1, 64) if append else (1, 1, 64),
            (1, 1, 64),
        ]
        if failure in (None, "invocation"):
            assert all(a is b for a, b in zip(launch["inputs"], c.args))
            probe_launch = kernel.call_args_list[0].kwargs
            for name in launch.keys() - {"inputs"}:
                assert probe_launch[name] == launch[name]
            for probe_input, real_input in zip(
                probe_launch["inputs"], launch["inputs"]
            ):
                assert probe_input is not real_input
                assert (probe_input.shape, probe_input.dtype) == (
                    real_input.shape,
                    real_input.dtype,
                )
        else:
            assert all(a is not b for a, b in zip(launch["inputs"], c.args))
    factory = c.mx.fast.metal_kernel.call_args.kwargs
    assert factory["ensure_row_contiguous"] is True
    assert "_strides" not in factory["source"]


@pytest.mark.parametrize("failure", [False, True])
def test_visit_norm_cpu_concurrent_first_use(visit_cpu, failure):
    c, barrier = visit_cpu, Barrier(8)
    kernel = Mock(return_value=[object(), object(), object()])
    c.mx.fast.metal_kernel.return_value = kernel
    if failure:
        c.mx.eval.side_effect = RuntimeError("evaluation")

    def run(_):
        barrier.wait(timeout=10)
        return c.visit.visit_norm_decode(*c.args)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(8)))
    assert all((result is None) == failure for result in results)
    c.mx.fast.metal_kernel.assert_called_once()
    c.mx.eval.assert_called_once()
    assert kernel.call_count == (1 if failure else 9)


def test_visit_norm_cpu_concurrent_invocation_success(visit_cpu):
    c, barrier = visit_cpu, Barrier(2)
    output = [object(), object(), object()]
    # Pre-probe so the barrier checks dispatch concurrency, not compilation.
    kernel = Mock(return_value=output)
    c.mx.fast.metal_kernel.return_value = kernel
    assert c.visit.visit_norm_decode(*c.args) == output
    assert c.visit.visit_norm_decode(*c.args, append_block=True) == output

    def invoke(**kwargs):
        barrier.wait(timeout=10)
        assert not c.visit._LOCK.locked()
        return output

    kernel.side_effect = invoke
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(
            pool.map(
                lambda append: c.visit.visit_norm_decode(*c.args, append_block=append),
                (False, True),
            )
        ) == [output, output]
    assert c.mx.fast.metal_kernel.call_count == 2
    assert c.mx.eval.call_count == 2


def test_visit_norm_cpu_concurrent_invocation_failure(visit_cpu, monkeypatch):
    c = visit_cpu
    entered, contender, disabled = Event(), Event(), Event()
    first_thread = []
    kernel = Mock(return_value=[object(), object(), object()])
    c.mx.fast.metal_kernel.return_value = kernel
    assert c.visit.visit_norm_decode(*c.args) is not None
    kernel.reset_mock()

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if entered.is_set() and get_ident() != first_thread[0]:
                contender.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()
            if entered.is_set() and get_ident() != first_thread[0]:
                assert disabled.wait(timeout=10)

    state = next(iter(c.visit._KERNELS.values()))
    monkeypatch.setattr(state, "lock", ObservedLock())
    # The global lock must not be acquired on a warm variant lookup or dispatch.
    monkeypatch.setattr(c.visit, "_LOCK", None)

    def invoke(**kwargs):
        first_thread.append(get_ident())
        entered.set()
        assert contender.wait(timeout=10)
        raise RuntimeError("invoke")

    def fail():
        try:
            return c.visit.visit_norm_decode(*c.args)
        finally:
            disabled.set()

    kernel.side_effect = invoke
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(fail)
        assert entered.wait(timeout=10)
        second = pool.submit(c.visit.visit_norm_decode, *c.args)
        assert first.result(timeout=10) is None
        assert second.result(timeout=10) is None
    assert kernel.call_count == 1  # No dispatch after the first caller disables it.
    c.mx.fast.metal_kernel.assert_called_once()
    c.mx.eval.assert_called_once()


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("case", ["none", "shape", "dtype"])
def test_visit_norm_cpu_guards(visit_cpu, position, case):
    c, args = visit_cpu, list(visit_cpu.args)
    args[position] = {
        "none": None,
        "shape": np.ones((2, 2), np.float32),
        "dtype": args[position].astype(np.int32),
    }[case]
    assert c.visit.visit_norm_decode(*args) is None
    c.mx.fast.metal_kernel.assert_not_called()


def test_visit_norm_cpu_variant_isolation(visit_cpu):
    c = visit_cpu
    kernel = Mock(return_value=[object(), object(), object()])
    c.mx.fast.metal_kernel.return_value = kernel
    assert c.visit.visit_norm_decode(*c.args) is not None
    c.mx.fast.metal_kernel.side_effect = RuntimeError("other variant")
    assert c.visit.visit_norm_decode(*c.args, append_block=True) is None
    assert c.visit.visit_norm_decode(*c.args) is not None
    assert c.mx.fast.metal_kernel.call_count == 2


@pytest.mark.parametrize("variant", ["width", "bank", "queries", "dtype", "append"])
def test_visit_norm_cpu_probe_evaluation_variant_isolation(visit_cpu, variant):
    c = visit_cpu
    output = [object(), object(), object()]
    kernel = Mock(return_value=output)
    c.mx.fast.metal_kernel.return_value = kernel
    assert c.visit.visit_norm_decode(*c.args) == output
    dims, k, q = {"width": (129, 2, 3), "bank": (64, 3, 3), "queries": (64, 2, 1)}.get(
        variant, (64, 2, 3)
    )
    args = (
        np.ones((k, 1, 1, dims), np.float32),
        np.ones((1, 1, dims), np.float32 if variant == "dtype" else np.float16),
        np.ones((q, dims), np.float32),
    )
    c.mx.eval.side_effect = RuntimeError("variant execution")
    for _ in range(2):
        assert (
            c.visit.visit_norm_decode(*args, append_block=variant == "append") is None
        )
    failed_probe = kernel.call_args.kwargs
    c.mx.eval.side_effect = None
    assert c.visit.visit_norm_decode(*c.args) == output
    assert c.mx.fast.metal_kernel.call_count == c.mx.eval.call_count == 2
    assert kernel.call_count == 4  # Two probes, two real calls on the healthy variant.
    for probe_input, real_input in zip(failed_probe["inputs"], args):
        assert probe_input is not real_input
        assert (probe_input.shape, probe_input.dtype) == (
            real_input.shape,
            real_input.dtype,
        )
        assert not np.any(probe_input)
    template = dict(failed_probe["template"])
    assert (template["D"], template["K"], template["Q"], template["APPEND"]) == (
        dims,
        k,
        q,
        variant == "append",
    )


@pytest.mark.parametrize(
    "dims,k,q", [(0, 1, 1), (4097, 1, 1), (64, 0, 1), (64, 33, 1), (64, 1, 2)]
)
def test_visit_norm_cpu_bounds(visit_cpu, dims, k, q):
    c = visit_cpu
    assert (
        c.visit.visit_norm_decode(
            np.zeros((k, 1, 1, dims), np.float32),
            np.zeros((1, 1, dims), np.float32),
            np.zeros((q, dims), np.float32),
        )
        is None
    )
    c.mx.fast.metal_kernel.assert_not_called()


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "construction",
        "probe",
        "evaluation",
        "invocation",
        "disabled",
        "prefill",
        "no_attention",
        "no_bank",
        "no_gpu",
    ],
)
def test_visit_norm_cpu_layer_fallback(visit_cpu, monkeypatch, append, failure):
    c, layers = visit_cpu, visit_cpu.layers
    blocks, x, queries = c.args
    normed = x.astype(np.float32) * 0.5
    bank = np.concatenate([blocks, normed[None]]) if append else normed
    attention = np.ones((1, 1, 3, 64), np.float32)
    layer = layers.DecoderLayer.__new__(layers.DecoderLayer)
    layer.attention_input = Mock(return_value=attention)
    layer.attention_input.query = queries
    layer.is_block_boundary = append
    layer._self_attention = Mock(side_effect=lambda hidden, *args: hidden)
    layer._mlp_states = lambda hidden, *args: hidden
    layer.mlp = lambda hidden, **kw: hidden
    layer.post_attention_convolution = layer.post_mlp_convolution = (
        lambda hidden, state, *a, **kw: (hidden, state)
    )
    stock_norm = Mock(return_value=normed)
    monkeypatch.setattr(layers, "unweighted_rms_norm", stock_norm)
    kernel = Mock(return_value=[attention, bank, x.astype(np.float32)])
    c.mx.fast.metal_kernel.return_value = kernel
    if failure == "construction":
        c.mx.fast.metal_kernel.side_effect = RuntimeError("build")
    elif failure == "probe":
        kernel.side_effect = RuntimeError("probe")
    elif failure == "evaluation":
        c.mx.eval.side_effect = RuntimeError("eval")
    elif failure == "invocation":
        kernel.side_effect = [
            [attention, bank, x.astype(np.float32)],
            RuntimeError("invoke"),
        ]
    elif failure == "disabled":
        monkeypatch.setitem(layers._FAST_DECODE_FEATURES, "VISIT_NORM", False)
    elif failure == "no_gpu":
        monkeypatch.setattr(layers, "_use_kernels", lambda: False)
    elif failure == "no_attention":
        layer.attention_input = None
    elif failure == "no_bank":
        layer.attention_input = None
        blocks = None
        layer.is_block_boundary = True
    cache = [SimpleNamespace(offset=0 if failure == "prefill" else 3), None]
    for step in range(2):
        result, result_bank = layer(x, blocks, cache=cache)
        expected = x.astype(np.float32) * (2 if layer.is_block_boundary else 4)
        np.testing.assert_array_equal(result, expected)
        expected_bank = (
            normed[None]
            if blocks is None
            else np.concatenate([blocks, normed[None]])
            if append
            else blocks
        )
        np.testing.assert_array_equal(result_bank, expected_bank)
        assert stock_norm.call_count == (0 if failure is None else step + 1)
        if layer.attention_input is not None:
            assert layer.attention_input.call_count == (
                0 if failure is None else step + 1
            )
    assert layer._self_attention.call_count == 2
    assert kernel.call_count == {
        None: 3,
        "probe": 1,
        "evaluation": 1,
        "invocation": 2,
    }.get(failure, 0)
    assert c.mx.fast.metal_kernel.call_count == int(
        failure in (None, "construction", "probe", "evaluation", "invocation")
    )


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize(
    "dims,k,q",
    [
        (1, 1, 1),
        (31, 3, 3),
        (64, 2, 1),
        (127, 3, 3),
        (128, 1, 1),
        (129, 2, 3),
        (255, 3, 1),
        (256, 1, 3),
        (257, 2, 1),
        (1152, 3, 3),
        (2304, 32, 1),
        (4096, 3, 3),
    ],
)
def test_visit_norm_bitwise(metal_runtime, monkeypatch, dtype, append, dims, k, q):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers, visit_norm

    real_ready, calls = visit_norm._ready, []

    def ready(*args):
        state = real_ready(*args)
        assert state.kernel is not None
        call = Mock(wraps=state.kernel)
        calls.append(call)
        monkeypatch.setattr(state, "kernel", call)
        return state

    monkeypatch.setattr(visit_norm, "_ready", ready)
    mx.random.seed(42)
    for layout in ("contiguous", "stepped", "reversed", "broadcast", "transposed"):

        def values(shape, dtype, layout=layout):
            if layout == "stepped":
                return mx.random.normal((*shape[:-1], shape[-1] * 2)).astype(dtype)[
                    ..., ::2
                ]
            x = (mx.random.normal(shape) * 0.7).astype(dtype)
            if layout == "reversed":
                return x[..., ::-1]
            if layout == "broadcast":
                return mx.broadcast_to(x[..., :1], shape)
            if layout == "transposed" and len(shape) > 1:
                return mx.random.normal(shape[::-1]).astype(dtype).transpose()
            return x

        blocks = values((k, 1, 1, dims), mx.float32)
        partial = values((1, 1, dims), getattr(mx, dtype))
        queries = values((q, dims), mx.float32)
        normed = layers.unweighted_rms_norm(partial.astype(mx.float32))
        expected = [
            kernels.depth_attention(blocks, normed, queries, scale=dims**-0.5),
            mx.concatenate([blocks, normed[None]]) if append else normed,
            partial.astype(mx.float32),
        ]
        mx.eval(*expected)
        actual = visit_norm.visit_norm_decode(
            blocks, partial, queries, append_block=append
        )
        assert actual is not None
        calls[-1].assert_called_once()
        for a, b in zip(actual, expected):
            assert a.shape == b.shape and a.dtype == b.dtype
            assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item(), layout


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_visit_norm_cached_logits_exact(metal_runtime, monkeypatch, dtype):
    import copy

    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    stock = create_model(metal_runtime, dtype=getattr(mx, dtype))
    fused = create_model(metal_runtime, dtype=getattr(mx, dtype))
    stock_cache, fused_cache = stock.make_cache(), fused.make_cache()
    tokens = mx.random.randint(0, 120, (1, 10))
    assert mx.array_equal(
        stock(tokens[:, :6], cache=stock_cache), fused(tokens[:, :6], cache=fused_cache)
    ).item()
    real = layers.visit_norm_decode
    calls = []

    def record(*args, **kwargs):
        result = real(*args, **kwargs)
        assert result is not None
        calls.append(kwargs["append_block"])
        return result

    for step in range(6, 10):
        monkeypatch.setitem(layers._FAST_DECODE_FEATURES, "VISIT_NORM", False)
        expected = stock(tokens[:, step : step + 1], cache=stock_cache)
        monkeypatch.setitem(layers._FAST_DECODE_FEATURES, "VISIT_NORM", True)
        monkeypatch.setattr(layers, "visit_norm_decode", record)
        calls.clear()
        actual = fused(tokens[:, step : step + 1], cache=fused_cache)
        assert set(calls) == {False, True}
        assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()
    # Serving copies warm caches; only arrays and ordinary state may live there.
    copy.deepcopy(fused_cache)


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


@pytest.mark.parametrize("index", [0, 1, 2, 3, 6])
def test_expert_conv_cpu_missing_inputs(decode_cpu, index):
    k, mx = decode_cpu.kernels, decode_cpu.mx
    mx.fast.metal_kernel.reset_mock()
    args = [
        np.ones((1, 1, 2, 32), np.float32),
        np.ones((1, 1, 2), np.float32),
        np.ones((1, 3, 32), np.float32),
        np.ones((32, 4, 1), np.float32),
        None,
        None,
        np.ones((1, 1, 32), np.float32),
        np.float32,
    ]
    args[index] = None
    assert k.expert_conv_residual_decode(*args) is None
    mx.fast.metal_kernel.assert_not_called()


@pytest.mark.parametrize(
    "index,value",
    [
        (3, np.ones((32, 4), np.float32)),
        (4, np.ones((0, 1), bool)),
        (4, np.ones((1, 1), np.float32)),
        (5, np.ones((0,), np.int32)),
        (5, np.ones((1,), np.float32)),
    ],
)
def test_expert_conv_cpu_unsafe_indices(decode_cpu, index, value):
    k, mx = decode_cpu.kernels, decode_cpu.mx
    mx.fast.metal_kernel.reset_mock()
    args = [
        np.ones((1, 1, 2, 32), np.float32),
        np.ones((1, 1, 2), np.float32),
        np.ones((1, 3, 32), np.float32),
        np.ones((32, 4, 1), np.float32),
        None,
        None,
        np.ones((1, 1, 32), np.float32),
        np.float32,
    ]
    args[index] = value
    assert k.expert_conv_residual_decode(*args) is None
    mx.fast.metal_kernel.assert_not_called()


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
        calls = Mock()

        def __call__(self, hidden, history, mask, *, lengths):
            self.calls()
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
    layer.post_attention_convolution, layer.post_mlp_convolution = Conv(), Conv()
    layer.post_mlp_convolution.calls = Mock()
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
        monkeypatch.setitem(layers._FAST_DECODE_FEATURES, "EXPERT_CONV_RESIDUAL", False)
    for step in range(2):
        block_states = None if boundary == "first" else x[None]
        actual, block = layer(x, block_states, cache=cache)
        assert block.shape[0] == (2 if boundary == "append" else 1)
        assert layer._mlp_states.call_args.args[-1] is True
        np.testing.assert_array_equal(actual, expected)
        assert mlp.experts.call_count == mlp.gate.call_count == step + 1
    assert stock.call_count == (0 if failure in (None, "invocation") else 2)
    assert layer.post_mlp_convolution.calls.call_count == (0 if failure is None else 2)
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
    assert all(layers._fast_decode_enabled(name) for name in features)
    monkeypatch.setitem(layers._FAST_DECODE_FEATURES, feature, False)
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
    monkeypatch.setitem(layers._FAST_DECODE_FEATURES, "ROUTE_TOPK", not disabled)
    assert router(np.ones((1, 1, 32), np.float32), cached_decode=True) is expected
    assert fused.call_count == int(not disabled)
    assert stock.call_count == int(disabled or unavailable)


@pytest.mark.parametrize(
    "failure", [None, "construction", "probe", "evaluation", "invocation"]
)
def test_router_complete_cpu_lifecycle(decode_cpu, monkeypatch, failure):
    k, layers, mx = decode_cpu.kernels, decode_cpu.layers, decode_cpu.mx
    router = layers.TopKRouter.__new__(layers.TopKRouter)
    router.weight = np.eye(4, dtype=np.float32)
    router.e_score_correction_bias = np.zeros(4, np.float32)
    router.top_k, router.routed_scaling_factor = 2, 2.0
    x = np.zeros((1, 1, 4), np.float32)
    expected = (np.array([[[0, 1]]], np.uint32), np.ones((1, 1, 2), np.float32))
    stock = Mock(return_value=expected)
    monkeypatch.setattr(layers, "_route_weights", stock)
    monkeypatch.setattr(mx, "sigmoid", lambda x: x, raising=False)
    monkeypatch.setattr(k, "route_weights_decode", Mock(side_effect=AssertionError))
    launch = Mock(return_value=expected)
    factory = Mock(return_value=launch)
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    if failure == "construction":
        factory.side_effect = RuntimeError("construction")
    elif failure == "probe":
        launch.side_effect = RuntimeError("probe")
    elif failure == "evaluation":
        mx.eval.side_effect = RuntimeError("evaluation")
    elif failure == "invocation":
        launch.side_effect = [expected, RuntimeError("invocation")]
    for _ in range(3):
        actual = router(x, cached_decode=True)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a.view(np.uint32), e.view(np.uint32))
    assert stock.call_count == (3 if failure else 0)
    assert factory.call_count == 1
    assert (
        launch.call_count
        == {None: 4, "construction": 0, "probe": 1, "evaluation": 1, "invocation": 2}[
            failure
        ]
    )
    assert mx.eval.call_count == int(failure not in ("construction", "probe"))
    if failure != "construction":
        spec = factory.call_args.kwargs
        assert spec["ensure_row_contiguous"] is True
        assert "_strides" not in spec["source"]
        call = launch.call_args.kwargs
        assert call["output_shapes"] == [(1, 1, 2)] * 2
        assert len(call["inputs"]) == 3
    k.route_weights_decode.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_router_complete_cpu_concurrent_first_use(decode_cpu, monkeypatch, failure):
    k, mx = decode_cpu.kernels, decode_cpu.mx
    barrier = Barrier(8)
    expected = (np.zeros((1, 1, 2), np.uint32), np.ones((1, 1, 2), np.float32))

    launch = Mock(return_value=expected)
    factory = Mock(return_value=launch)
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    if failure:
        mx.eval.side_effect = RuntimeError("evaluation")
    x, bias = np.ones((1, 1, 4), np.float32), np.zeros(4, np.float32)

    def run(_):
        barrier.wait(timeout=10)
        return k.route_topk_decode(x, bias, 2, 2.0)

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, range(8)))
    assert all((r is None) == failure for r in results)
    assert factory.call_count == mx.eval.call_count == 1
    assert launch.call_count == (1 if failure else 9)
    if not failure:

        def invoke(**kw):
            assert k._route_topk_lock.acquire(blocking=False)
            k._route_topk_lock.release()
            return expected

        launch.side_effect = invoke
        assert k.route_topk_decode(x, bias, 2, 2.0) is not None
        launch.side_effect = None
    # A distinct width is a distinct evaluated variant, including after failure.
    k.route_topk_decode(np.ones((1, 1, 5), np.float32), np.zeros(5, np.float32), 2, 1.0)
    assert factory.call_count == mx.eval.call_count == 2
    k.route_topk_decode(x, bias, 1, 1.0)
    assert factory.call_count == mx.eval.call_count == 3


@pytest.mark.parametrize(
    "shape,bias_shape,dtype,bias_dtype,top_k",
    [
        (None, (4,), np.float32, np.float32, 2),
        ((1, 1, 4), None, np.float32, np.float32, 2),
        ((), (4,), np.float32, np.float32, 2),
        ((1, 4), (4,), np.float32, np.float32, 2),
        ((2, 1, 4), (4,), np.float32, np.float32, 2),
        ((1, 2, 4), (4,), np.float32, np.float32, 2),
        ((1, 1, 4), (1, 4), np.float32, np.float32, 2),
        ((1, 1, 4), (4,), np.float16, np.float32, 2),
        ((1, 1, 4), (4,), np.float32, np.float16, 2),
        ((1, 1, 4), (4,), np.float32, np.float32, 0),
        ((1, 1, 4), (4,), np.float32, np.float32, 5),
        ((1, 1, 129), (129,), np.float32, np.float32, 2),
    ],
)
def test_router_complete_cpu_rejected(
    decode_cpu, shape, bias_shape, dtype, bias_dtype, top_k
):
    k, mx = decode_cpu.kernels, decode_cpu.mx
    mx.fast.metal_kernel.reset_mock()
    logits = None if shape is None else np.ones(shape, dtype)
    bias = None if bias_shape is None else np.ones(bias_shape, bias_dtype)
    assert k.route_topk_decode(logits, bias, top_k, 2.0) is None
    mx.fast.metal_kernel.assert_not_called()
    mx.eval.assert_not_called()


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

        glue_loader.module(
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
        for name in ("SwitchLinear", "_gather_sort", "_scatter_unsort"):
            assert getattr(result, name) is getattr(fallback, name)
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
    glue_loader.load("visit_norm")
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
    monkeypatch.setitem(layers._FAST_DECODE_FEATURES, "MESSAGE_NORM", False)
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


@pytest.fixture
def decode_launches(metal_runtime, monkeypatch):
    from rapid_mlx.models.inkling2 import nvfp4_decode

    prepare, calls = nvfp4_decode.prepare, []

    def traced(*args):
        launch = prepare(*args)
        assert launch.kernel is not None
        launch.kernel = Mock(wraps=launch.kernel)
        calls.append(launch.kernel)
        return launch

    monkeypatch.setattr(nvfp4_decode, "prepare", traced)
    return calls


@pytest.fixture
def cpu_decode(glue_loader, monkeypatch):
    """Run launch plumbing with NumPy metadata and inert kernels, without MLX."""
    mx = glue_loader.mx
    for name, value in dict(
        int32=np.int32,
        uint32=np.uint32,
        uint8=np.uint8,
        bfloat16=np.float16,
        zeros=np.zeros,
        full=np.full,
        fast=SimpleNamespace(metal_kernel=Mock()),
    ).items():
        monkeypatch.setattr(mx, name, value, raising=False)
    decode = glue_loader.load("nvfp4_decode")
    x = np.zeros((1, 1, 128), np.float16)
    w = np.zeros((4, 32, 16), np.uint32)
    scales = np.zeros((4, 32, 8), np.uint8)
    indices = np.array([0, 2], np.int32)
    return SimpleNamespace(
        mx=mx, decode=decode, args=(x, w, scales, indices), probe=object(), out=object()
    )


@pytest.mark.parametrize("has_scale", [False, True])
def test_nvfp4_cpu_probe_and_guarded_invocation(cpu_decode, has_scale):
    """Each variant evaluates one probe; real outputs remain lazy and unwrapped."""
    c = cpu_decode
    scale = np.ones(4) if has_scale else None
    kernel = Mock(side_effect=[[c.probe], [c.out], [c.out], RuntimeError("invoke")])
    c.mx.fast.metal_kernel.return_value = kernel
    for _ in range(2):
        assert c.decode.project(*c.args, scale) is c.out
    c.mx.fast.metal_kernel.assert_called_once()
    c.mx.eval.assert_called_once_with(c.probe)
    inputs = kernel.call_args.kwargs["inputs"]
    assert all(a is b for a, b in zip(inputs[:4], c.args))
    assert inputs[4] is (scale if has_scale else c.args[2])
    assert kernel.call_args.kwargs["output_shapes"] == [(2, 1, 32)]
    assert c.decode.project(*c.args, scale) is None
    assert kernel.call_count == 4
    # An exception evaluating a real output is outside the fallback guarantee.
    c.mx.eval.side_effect = RuntimeError("lazy output")
    with pytest.raises(RuntimeError, match="lazy output"):
        c.mx.eval(c.out)


@pytest.mark.parametrize("failure", ["construction", "invocation", "evaluation"])
def test_nvfp4_cpu_probe_failure_cached_per_variant(cpu_decode, failure):
    c = cpu_decode
    kernel = Mock(return_value=[c.probe])
    c.mx.fast.metal_kernel.return_value = kernel
    # A successful specialization must survive failure of a different variant.
    assert c.decode._get_kernel() is kernel
    c.mx.fast.metal_kernel.reset_mock()
    kernel.reset_mock()
    c.mx.eval.reset_mock()
    error = RuntimeError(failure)
    if failure == "construction":
        c.mx.fast.metal_kernel.side_effect = error
    elif failure == "invocation":
        kernel.side_effect = error
    else:
        c.mx.eval.side_effect = error
    for _ in range(2):
        assert c.decode.project(*c.args, None) is None
    c.mx.fast.metal_kernel.assert_called_once()
    assert kernel.call_count == (failure != "construction")
    assert c.mx.eval.call_count == (failure == "evaluation")
    assert c.decode._get_kernel() is kernel


@pytest.mark.parametrize("failure", [False, True])
def test_nvfp4_cpu_layer_dispatch(cpu_decode, glue_loader, monkeypatch, failure):
    """Exercise the layer's cached dispatch and sticky invocation fallback on CPU."""
    c = cpu_decode

    class Module(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    monkeypatch.setattr(
        sys.modules["mlx"], "nn", glue_loader.module("mlx.nn", Module=Module)
    )
    for name, value in dict(
        gather_qmm=Mock(return_value=c.out),
        default_device=lambda: "fake-gpu",
        gpu="fake-gpu",
        integer=np.integer,
        issubdtype=np.issubdtype,
        quantize=Mock(return_value=c.args[1:3]),
    ).items():
        monkeypatch.setattr(c.mx, name, value, raising=False)
    sl = glue_loader.load("switch_layers")
    layer = sl.QuantizedSwitchLinear.__new__(sl.QuantizedSwitchLinear)
    layer.mode, layer.bits, layer.group_size = "nvfp4", 4, 16
    layer._quantize(c.args[1])
    kernel = (
        Mock(side_effect=[[c.probe], RuntimeError("invoke")])
        if failure
        else Mock(return_value=[c.out])
    )
    c.mx.fast.metal_kernel.return_value = kernel
    monkeypatch.delenv("RAPID_MLX_INKLING2_FAST_DECODE", raising=False)
    for _ in range(2):
        assert layer(c.args[0], c.args[3]) is c.out
    c.mx.fast.metal_kernel.assert_called_once()
    assert kernel.call_count == (2 if failure else 3)
    assert c.mx.gather_qmm.call_count == (2 if failure else 0)
    assert (layer._nvfp4_decode.kernel is None) == failure
    # Unsupported inputs bypass even a warmed launch.
    kernel.reset_mock()
    layer(c.args[0].astype(np.float32), c.args[3])
    kernel.assert_not_called()
    assert c.mx.gather_qmm.call_count == (3 if failure else 1)


@pytest.mark.parametrize(
    "r,n,m,p",
    [
        (r, n, m, p)
        for n, m, p in [
            (16, 32, 2),
            (64, 3, 2),
            (96, 17, 16),
            (512, 17, 2),
            (528, 3, 2),
            (1152, 2304, 8),
            (2304, 1152, 8),
            (2560, 2304, 8),
            (768, 17, 16),
        ]
        for r in (1, p)
    ],
)
@pytest.mark.parametrize("scale", [None, "native", "fallback"])
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize("wide", [False, True])
def test_nvfp4_decode_matches_stock(
    metal_runtime, monkeypatch, decode_launches, r, n, m, p, scale, seed, wide
):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import switch_layers as sl

    if scale == "native" and not sl._GATHER_QMM_HAS_GLOBAL_SCALE:
        pytest.skip("Native tensor scale unavailable")
    mx.random.seed(seed)
    layer = sl.QuantizedSwitchLinear(n, m, 20, False, 16, 4, "nvfp4", scale is not None)
    if scale is not None:
        layer.global_scale *= mx.linspace(0.25, 4.0, 20)
    if scale == "fallback":
        monkeypatch.setattr(sl, "_GATHER_QMM_HAS_GLOBAL_SCALE", False)
    x = mx.random.normal((1, 1, r, 1, n))
    if wide:
        # Exercise rounding within qdot, beyond nearly exact bf16 products
        # of similarly sized values, as well as cancellation across groups.
        x *= 2.0 ** mx.random.randint(-16, 17, x.shape).astype(mx.float32)
    x = x.astype(mx.bfloat16)
    indices = mx.random.randint(0, 20, (1, 1, p))
    monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", False)
    expected = layer(x, indices)
    monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", True)
    actual = layer(x, indices)
    assert len(decode_launches) == 1
    decode_launches[0].assert_called_once()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == mx.bfloat16
    assert mx.array_equal(actual, expected).item()
    repeated = layer(x, indices)
    assert len(decode_launches) == 1 and decode_launches[0].call_count == 2
    assert mx.array_equal(repeated, expected).item()


def test_nvfp4_decode_formats(metal_runtime):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import nvfp4_decode

    kernel = mx.fast.metal_kernel(
        name="inkling2_nvfp4_formats",
        input_names=["encoded"],
        output_names=["values", "scales"],
        header=nvfp4_decode._HEADER,
        source="""
            uint i = thread_position_in_grid.x;
            uint c = encoded[i];
            if (i < 16) values[i] = fp4(c);
            scales[i] = scale8(c);
        """,
    )
    values, scales = kernel(
        inputs=[mx.arange(256, dtype=mx.uint32)],
        grid=(256, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16,), (256,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    # Every E2M1 code and E4M3 scale, including subnormals and signed zero.
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expected_values = mx.array(magnitudes + [-v for v in magnitudes], mx.float32)
    expected_scales = []
    for code in range(256):
        e, m = (code >> 3) & 15, code & 7
        value = m / 512 if e == 0 else (1 + m / 8) * 2 ** (e - 7)
        expected_scales.append(-value if code & 128 else value)
    finite = mx.array([c for c in range(256) if c & 127 != 127])
    expected_scales = mx.array(expected_scales, mx.float32)
    assert mx.array_equal(
        values.view(mx.uint32), expected_values.view(mx.uint32)
    ).item()
    assert mx.array_equal(
        scales[finite].view(mx.uint32), expected_scales[finite].view(mx.uint32)
    ).item()
    assert mx.all(mx.isnan(scales[mx.array([127, 255])])).item()


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "case", ["float32", "sorted", "multi", "batched", "affine", "disabled"]
)
def test_nvfp4_decode_unsupported(metal_runtime, monkeypatch, case, warm):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import nvfp4_decode
    from rapid_mlx.models.inkling2 import switch_layers as sl

    mode, group = ("affine", 32) if case == "affine" else ("nvfp4", 16)
    layer = sl.QuantizedSwitchLinear(128, 32, 4, False, group, 4, mode)
    if warm and mode == "nvfp4":
        # Unsupported calls must still fall back after warming the layer cache.
        mx.eval(
            layer(
                mx.ones((1, 1, 1, 1, 128), mx.bfloat16),
                mx.zeros((1, 1, 2), mx.uint32),
            )
        )
    shape = (2 if case == "batched" else 1, 1, 1, 2 if case == "multi" else 1, 128)
    x = mx.ones(shape, mx.float32 if case == "float32" else mx.bfloat16)
    indices = mx.zeros((1, 1, 2), mx.uint32)
    if case == "disabled":
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    dispatch = Mock(side_effect=lambda *a, **kw: pytest.fail("kernel dispatched"))
    monkeypatch.setattr(nvfp4_decode, "prepare", dispatch)
    if layer._nvfp4_decode is not None:
        monkeypatch.setattr(layer._nvfp4_decode, "kernel", dispatch)
    mx.eval(layer(x, indices, sorted_indices=case == "sorted"))
    dispatch.assert_not_called()


@pytest.mark.parametrize("n,m", [(128, 32), (16, 1)])
@pytest.mark.parametrize("failure", ["build", "call", "eval"])
def test_nvfp4_decode_nondefault_build_failure(
    metal_runtime, monkeypatch, failure, n, m
):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import nvfp4_decode
    from rapid_mlx.models.inkling2 import switch_layers as sl

    layer = sl.QuantizedSwitchLinear(n, m, 4, False, 16, 4, "nvfp4")
    x = mx.ones((1, 1, 1, 1, n), mx.bfloat16)
    indices = mx.zeros((1, 1, 2), mx.uint32)
    monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", False)
    expected = layer(x, indices)
    mx.eval(expected)
    monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", True)
    error = RuntimeError("unavailable")
    kernel = Mock(side_effect=error if failure == "call" else None)
    build = Mock(side_effect=error if failure == "build" else None, return_value=kernel)
    factory = functools.cache(nvfp4_decode._get_kernel.__wrapped__)
    monkeypatch.setattr(nvfp4_decode, "_get_kernel", factory)
    assert factory() is not None
    monkeypatch.setattr(mx.fast, "metal_kernel", build)
    with monkeypatch.context() as probe:
        if failure == "eval":
            kernel.return_value = [mx.zeros((1,), mx.bfloat16)]
            evaluate = Mock(side_effect=error)
            probe.setattr(mx, "eval", evaluate)
        actual = [layer(x, indices), layer(x, indices)]
    for output in actual:
        assert mx.array_equal(output, expected).item()
    assert factory(n, m, True, False, mx.uint32) is None and factory() is not None
    build.assert_called_once()
    assert kernel.call_count == (failure != "build")
    if failure == "eval":
        evaluate.assert_called_once()


def test_nvfp4_decode_invocation_failure(metal_runtime, monkeypatch):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import nvfp4_decode
    from rapid_mlx.models.inkling2 import switch_layers as sl

    layer = sl.QuantizedSwitchLinear(128, 32, 4, False, 16, 4, "nvfp4")
    x = mx.ones((1, 1, 1, 1, 128), mx.bfloat16)
    indices = mx.zeros((1, 1, 2), mx.uint32)
    monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", False)
    expected = layer(x, indices)
    mx.eval(expected)
    monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", True)
    factory = functools.cache(nvfp4_decode._get_kernel.__wrapped__)
    monkeypatch.setattr(nvfp4_decode, "_get_kernel", factory)
    # The one-time probe succeeds; only the real invocation raises.
    kernel = Mock(side_effect=[[mx.zeros((1,), mx.bfloat16)], RuntimeError("invoke")])
    monkeypatch.setattr(mx.fast, "metal_kernel", Mock(return_value=kernel))
    for _ in range(2):
        actual = layer(x, indices)
        assert mx.array_equal(actual, expected).item()
    assert factory(128, 32, True, False, mx.uint32) is kernel
    assert kernel.call_count == 2
    assert kernel.call_args.kwargs["inputs"][0] is x
    assert kernel.call_args.kwargs["inputs"][3] is indices


@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_nvfp4_invalid_indices_no_crash_no_stock_parity(
    metal_runtime, decode_launches, dtype
):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import nvfp4_decode
    from rapid_mlx.models.inkling2 import switch_layers as sl

    layer = sl.QuantizedSwitchLinear(128, 32, 4, False, 16, 4, "nvfp4", True)
    x = mx.ones((1, 1, 128), mx.bfloat16)
    indices = mx.array([-1, 4, 2**31 - 1, 2**31, 2**32, 2**32 + 3], mx.int64).astype(
        getattr(mx, dtype)
    )
    output = nvfp4_decode.project(
        x, layer.weight, layer.scales, indices, layer.global_scale
    )
    mx.eval(output)
    assert output.shape == (indices.size, 1, 32)
    assert mx.all(mx.isfinite(output)).item()
    expected = nvfp4_decode.project(
        x,
        layer.weight,
        layer.scales,
        mx.clip(indices.astype(mx.int32), 0, 3),
        layer.global_scale,
    )
    assert len(decode_launches) == 2
    for kernel in decode_launches:
        kernel.assert_called_once()
    assert mx.array_equal(output, expected).item()


def test_nvfp4_decode_model_exact(metal_runtime, monkeypatch, decode_launches):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import switch_layers as sl

    model = create_model(metal_runtime, dtype=mx.bfloat16)
    for layer in model.model.language_model.layers:
        for name in ("gate_up_proj", "down_proj"):
            proj = getattr(layer.mlp.experts, name)
            setattr(layer.mlp.experts, name, proj.to_quantized(16, 4, "nvfp4", True))
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(sl, "_NVFP4_DECODE_ENABLED", enabled)
        for layer in model.model.language_model.layers:
            if hasattr(layer.mlp, "_block_compiled__forward"):
                delattr(layer.mlp, "_block_compiled__forward")
        cache = model.make_cache()
        mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
        outputs.append(model(mx.array([[4]]), cache=cache))
        mx.eval(outputs[-1])
    assert len(decode_launches) == 2 * len(model.model.language_model.layers)
    assert all(kernel.call_count > 0 for kernel in decode_launches)
    assert mx.array_equal(*outputs).item()


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


@pytest.fixture
def main_decode_router(metal_runtime):
    """Run main's existing select+normalize kernels directly, with no fallback.

    This is the bitwise contract. The plain-op path has a different FP32
    accumulation order and is compared separately with a tight tolerance.
    Callers evaluate outputs, including when routing is traced inside an MoE.
    """
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2.compat import scalar

    select = Mock(
        wraps=mx.fast.metal_kernel(
            name="inkling2_main_route_topk_decode",
            input_names=["logits", "bias"],
            output_names=["indices", "out_scores"],
            source=_MAIN_ROUTE_SELECT_SOURCE,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    )
    normalize = Mock(
        wraps=mx.fast.metal_kernel(
            name="inkling2_main_route_weights_decode",
            input_names=["scores", "indices", "route_scale"],
            output_names=["out"],
            source=_MAIN_ROUTE_WEIGHTS_SOURCE,
            ensure_row_contiguous=False,
            compile_options={"math_mode": "safe"},
        )
    )

    def run(logits, bias, k, scale):
        n = logits.shape[-1]
        select.reset_mock()
        normalize.reset_mock()
        indices, scores = select(
            inputs=[logits, bias],
            template=[("N", n), ("K", k)],
            grid=(128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, 1, k), logits.shape],
            output_dtypes=[mx.uint32, mx.float32],
        )
        weights = normalize(
            inputs=[scores.reshape(-1, n), indices.reshape(-1, k), scalar(scale)],
            template=[("K", k)],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[indices.shape],
            output_dtypes=[mx.float32],
        )[0]
        select.assert_called_once()
        normalize.assert_called_once()
        return indices, weights

    return run


@pytest.mark.requires_mlx
@pytest.mark.parametrize("k", [1, 4, 31, 32, 33, 63, 64, 65, 127, 128])
def test_router_main_decode_reduction_exact(metal_runtime, main_decode_router, k):
    """Fused routing must be bit-identical to main's existing decode kernels."""
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels

    mx.random.seed(97)
    for values in (
        [0.0] + [-18.0] * (k - 1),
        mx.random.normal((k,)).tolist(),
    ):
        logits = mx.array(values).reshape(1, 1, k)
        bias = mx.zeros((k,))
        for scale in (0.0, 0.3, 2.0):
            expected = main_decode_router(logits, bias, k, scale)
            actual = kernels.route_topk_decode(logits, bias, k, scale)
            assert actual is not None  # Compile and run the fused variant.
            mx.eval(*actual)
            for a, e in zip(actual, expected):
                assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("seed", [0, 17, 53])
def test_router_topk_exact(metal_runtime, main_decode_router, monkeypatch, seed):
    """Bit-identical to main decode; plain-op parity uses the tolerance test."""
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
                expected = main_decode_router(
                    logits.astype(mx.float32),
                    bias,
                    k,
                    router.routed_scaling_factor,
                )
                indices, actual_weights = kernels._route_topk(n, k)(
                    inputs=[logits.astype(mx.float32), bias, mx.array([2.0])],
                    template=[("N", n), ("K", k)],
                    grid=(128, 1, 1),
                    threadgroup=(128, 1, 1),
                    output_shapes=[(1, 1, k)] * 2,
                    output_dtypes=[mx.uint32, mx.float32],
                )
                assert mx.array_equal(indices, expected[0]).item()
                assert mx.array_equal(
                    actual_weights.view(mx.uint32), expected[1].view(mx.uint32)
                ).item()
                with monkeypatch.context() as patch:
                    patch.setattr(
                        layers, "_route_weights", Mock(side_effect=AssertionError)
                    )
                    patch.setattr(
                        kernels,
                        "route_weights_decode",
                        Mock(side_effect=AssertionError),
                    )
                    actual = router(logits, True)
                for a, e in zip(actual, expected):
                    assert a.dtype == e.dtype
                    assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 1),
        (31, 3),
        (31, 31),
        (32, 32),
        (33, 31),
        (33, 33),
        (63, 63),
        (64, 64),
        (65, 64),
        (65, 65),
        (127, 127),
        (128, 127),
        (128, 128),
    ],
)
def test_router_complete_strided_exact(metal_runtime, main_decode_router, n, k):
    """Strided inputs retain main decode bits; plain ops have tolerance parity."""
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels

    mx.random.seed(73)
    # Non-contiguous inputs must be copied by the kernel API before flat loads.
    logits = mx.random.normal((1, 1, n * 2))[..., ::2]
    bias = mx.random.normal((n * 2,))[::2]
    for values, correction in (
        (logits, bias),
        (mx.zeros_like(logits), mx.zeros_like(bias)),
        (logits * 1000, bias),
        (logits, mx.where(mx.arange(n) % 2 == 0, float("nan"), bias)),
        (mx.full(logits.shape, -100.0), bias),
    ):
        for scale in (0.0, 0.3, 2.0):
            expected = main_decode_router(values, correction, k, scale)
            actual = kernels.route_topk_decode(values, correction, k, scale)
            assert actual is not None  # No silent fallback can satisfy parity.
            for a, e in zip(actual, expected):
                assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("k", [1, 4, 31, 32, 33, 63, 64, 65, 127, 128])
def test_router_plain_op_tolerance(metal_runtime, k):
    """Decode is within tolerance of plain ops, whose sum order differs.

    Main's existing decode kernels and the fused kernel sum lane-strided terms
    then SIMD-reduce. Plain ops sum sequentially through K=64, then reduce
    four consecutive terms per lane. Even the K=4 example differs in bits.
    """
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    mx.random.seed(97)
    for values in (
        [0.0] + [-18.0] * (k - 1),
        mx.random.normal((k,)).tolist(),
    ):
        logits = mx.array(values).reshape(1, 1, k)
        scores, bias = mx.sigmoid(logits), mx.zeros((k,))
        for scale in (0.0, 0.3, 2.0):
            indices, expected = layers._route_weights_reference(scores, bias, k, scale)
            actual = kernels.route_topk_decode(logits, bias, k, scale)
            assert actual is not None  # No fallback may satisfy this comparison.
            mx.eval(*actual)
            assert mx.array_equal(actual[0], indices).item()
            for weights in (
                actual[1],
                kernels.route_weights_decode(scores, indices, scale),
            ):
                assert mx.allclose(
                    weights,
                    expected,
                    rtol=ROUTER_PLAIN_OP_RTOL,
                    atol=ROUTER_PLAIN_OP_ATOL,
                ).item()
                if k == 4 and values[0] == 0.0 and scale != 0.0:
                    # Each tiny sequential addend rounds away; SIMD sum keeps
                    # their combined contribution. Preserve main's latter order.
                    assert scores.sum(axis=-1).view(mx.uint32).item() == 0x3F000000
                    assert not mx.array_equal(
                        weights.view(mx.uint32), expected.view(mx.uint32)
                    ).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("n", [4, 33, 65, 128])
@pytest.mark.parametrize("nan_input", ["logits", "bias"])
def test_router_nan_total_order(metal_runtime, main_decode_router, n, nan_input):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels

    for positions in ([0], [n // 2], [n - 1], [0, n // 2, n - 1], list(range(n))):
        logits, bias = [0.0] * n, [0.0] * n
        for position in positions:
            (logits if nan_input == "logits" else bias)[position] = float("nan")
        logits, bias = mx.array(logits).reshape(1, 1, n), mx.array(bias)
        # Ascending stock routing order is stable, with all NaNs last.
        order = [i for i in range(n) if i not in positions] + sorted(positions)
        for k in (1, n // 2, n):
            expected = main_decode_router(logits, bias, k, 2.0)
            assert expected[0].reshape(-1).tolist() == order[:k]
            for _ in range(3):
                actual = kernels.route_topk_decode(logits, bias, k, 2.0)
                assert actual is not None
                mx.eval(*actual)
                assert actual[0].reshape(-1).tolist() == order[:k]
                assert len(set(actual[0].reshape(-1).tolist())) == k
                for a, e in zip(actual, expected):
                    assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("k", [4, 64, 65, 128, 129, 256])
def test_route_weights_plain_op_rows_tolerance(metal_runtime, k):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels

    mx.random.seed(101)
    scores = mx.sigmoid(mx.random.normal((3, 1, 256)))
    indices = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    expected = (weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)) * 0.3
    actual = kernels.route_weights_decode(scores, indices, 0.3)
    assert mx.allclose(
        actual, expected, rtol=ROUTER_PLAIN_OP_RTOL, atol=ROUTER_PLAIN_OP_ATOL
    ).item()


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
    kernels._route_topk_kernels.clear()
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
        kernels._route_topk_kernels.clear()


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
    monkeypatch.setitem(kernels._route_topk_kernels, (4, 2), launch)
    try:
        for _ in range(3):
            actual = router(x, True)
            mx.eval(*actual)
            for a, e in zip(actual, expected):
                assert mx.array_equal(a.view(mx.uint32), e.view(mx.uint32)).item()
        launch.assert_called_once()
    finally:
        kernels._route_topk_kernels.clear()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_router_topk_cached_logits_exact(
    metal_runtime, main_decode_router, monkeypatch, dtype
):
    """Cached model logits match main decode bits; plain ops only within tolerance."""
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    monkeypatch.setattr(layers, "expert_routing", None)
    models = [create_model(metal_runtime, dtype=getattr(mx, dtype)) for _ in range(2)]
    caches = [model.make_cache() for model in models]
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    real_route = layers.route_topk_decode

    def required_fast(*args):
        result = real_route(*args)
        assert result is not None
        return result

    calls = Mock(side_effect=required_fast)
    for step in range(10):
        outputs = []
        for i, (model, cache) in enumerate(zip(models, caches)):
            with monkeypatch.context() as patch:
                patch.setattr(
                    layers, "route_topk_decode", calls if i else main_decode_router
                )
                ids = tokens[:, step : step + 1]
                out = model(ids, cache=cache)
                mx.eval(out)
                outputs.append(out)
        assert mx.array_equal(
            outputs[0].view(mx.uint32), outputs[1].view(mx.uint32)
        ).item()
        if step == 4:
            # Serving copies warmed caches; process-level kernel locks must
            # never become attributes of any cache object.
            caches[1] = copy.deepcopy(caches[1])
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


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(
    "mode,group", [("affine", 64), ("mxfp4", 32), ("nvfp4", 16), ("mxfp8", 32)]
)
def test_packed_quantized_projections_exact(
    metal_runtime, monkeypatch, dtype, mode, group
):
    import gc

    from mlx import nn

    from rapid_mlx.models.inkling2.layers import _packed_quantized_projections

    mx, _ = metal_runtime

    def packed_call(owner, projections, x, *, launches=2):
        with monkeypatch.context() as m:
            launch = Mock(wraps=mx.quantized_matmul)
            m.setattr(mx, "quantized_matmul", launch)
            output = _packed_quantized_projections(owner, "query", projections, x)
            assert launch.call_count == launches
            assert output is not None
            return output

    dtype = getattr(mx, dtype)
    for seed, width, scale in ((3, 64, 0.001), (17, 128, 1.0), (91, 4096, 100.0)):
        mx.random.seed(seed)
        owner = nn.Module()
        sizes = (width // 2, width // 16, width // 2, max(4, width // 128))
        projections = [nn.Linear(width, n, bias=False) for n in sizes]
        for p in projections:
            p.weight = mx.random.normal(p.weight.shape).astype(dtype)
        projections = [p.to_quantized(group_size=group, mode=mode) for p in projections]
        x = (mx.random.normal((1, 1, width * 2)) * scale).astype(dtype)[..., ::2]
        expected = [p(x) for p in projections]
        mx.eval(expected, [p.parameters() for p in projections])
        actual = packed_call(owner, projections, x)
        mx.eval(actual)
        for a, b in zip(actual, expected):
            assert a.dtype == b.dtype
            assert mx.array_equal(a, b).item()
            assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()
        del actual, a, b
        gc.collect()
        mx.synchronize()
        before = mx.get_active_memory()
        packed = owner._decode_pack_query
        actual = packed_call(owner, projections, x, launches=1)
        mx.eval(actual)
        del actual
        gc.collect()
        mx.synchronize()
        # Reusing a pack must not retain another allocation.
        assert mx.get_active_memory() <= before
        assert owner._decode_pack_query is packed
        projections[0].scales = projections[0].scales + mx.array(
            1, dtype=projections[0].scales.dtype
        )
        expected = projections[0](x)
        actual = packed_call(owner, projections, x)[0]
        assert mx.array_equal(actual, expected).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("failure", ["concatenate", "first_eval", "second_eval"])
def test_packed_projection_fallbacks(metal_runtime, monkeypatch, failure):
    from mlx import nn

    from rapid_mlx.models.inkling2 import layers

    mx, inkling2 = metal_runtime
    pack = layers._packed_quantized_projections
    projections = [nn.Linear(64, 32, bias=False).to_quantized() for _ in range(2)]
    x = mx.ones((1, 1, 64))
    owner = nn.Module()
    original = [p[f] for p in projections for f in ("weight", "scales", "biases")]
    expected = [p(x) for p in projections]
    mx.eval(expected)
    with monkeypatch.context() as m:
        target = "concatenate" if failure == "concatenate" else "eval"
        failure_at = 3 if failure == "second_eval" else 1
        evaluate = Mock(
            wraps=getattr(mx, target),
            side_effect=[DEFAULT] * (failure_at - 1) + [RuntimeError("packing failed")],
        )
        m.setattr(mx, target, evaluate)
        assert pack(owner, "query", projections, x) is None
        assert pack(owner, "query", projections, x) is None
        assert evaluate.call_count == failure_at
    for p in projections:
        for f in ("weight", "scales", "biases"):
            assert p[f] is original.pop(0)
    assert all(mx.array_equal(p(x), y).item() for p, y in zip(projections, expected))
    with monkeypatch.context() as m:
        m.setattr(mx, "quantized_matmul", lambda *a, **kw: pytest.fail("packed launch"))
        scales = projections[1].scales
        for invalid in (scales.astype(mx.float16), scales[:, :0]):
            projections[1].scales = invalid
            assert pack(nn.Module(), "query", projections, x) is None
        projections[1].scales = scales
        projections[1].bits = 8
        assert pack(nn.Module(), "query", projections, x) is None
        projections[1].bits = 4
        projections[1].bias = mx.zeros((32,))
        assert pack(nn.Module(), "query", projections, x) is None
        assert pack(nn.Module(), "query", [nn.Linear(64, 32)] * 2, x) is None
    attn = layers.SlidingAttention(inkling2.ModelArgs.from_dict(CONFIG).text)
    spy = Mock(side_effect=lambda *a: pytest.fail("excluded packed path"))
    monkeypatch.setattr(layers, "_packed_quantized_projections", spy)
    for batch, length, cache, enabled, gpu in (
        (1, 2, [SimpleNamespace(offset=2)], "1", True),
        (2, 1, [SimpleNamespace(offset=2)], "1", True),
        (1, 1, None, "1", True),
        (1, 1, [None], "1", True),
        (1, 1, [SimpleNamespace(offset=0)], "1", True),
        (1, 1, [SimpleNamespace(offset=2)], "0", True),
        (1, 1, [SimpleNamespace(offset=2)], "1", False),
    ):
        monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", enabled)
        monkeypatch.setattr(layers, "_use_kernels", lambda gpu=gpu: gpu)
        h = mx.ones((batch, length, 64))
        attn._project_inputs(h, h, h, cache)
    spy.assert_not_called()
    spy.side_effect, spy.return_value = None, None
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")
    monkeypatch.setattr(layers, "_use_kernels", lambda: True)
    attn._project_inputs(x, x, x + 1, [SimpleNamespace(offset=2)])
    assert [call.args[1] for call in spy.call_args_list] == ["query"]


@pytest.mark.requires_mlx
def test_packed_projection_concurrent_first_use(metal_runtime, monkeypatch):
    from mlx import nn

    from rapid_mlx.models.inkling2.layers import _packed_quantized_projections

    mx, _ = metal_runtime
    projections = [nn.Linear(64, 32, bias=False).to_quantized() for _ in range(2)]
    x = mx.ones((1, 1, 64))
    mx.eval(x, [p.parameters() for p in projections])
    owner, barrier = nn.Module(), Barrier(2)
    concatenate = Mock(wraps=mx.concatenate)
    monkeypatch.setattr(mx, "concatenate", concatenate)

    def first_use(_):
        with mx.stream(mx.new_stream(mx.gpu)):
            barrier.wait(timeout=10)
            output = _packed_quantized_projections(owner, "query", projections, x)
            assert output is not None
            mx.eval(output)
            return output

    with ThreadPoolExecutor(2) as pool:
        outputs = list(pool.map(first_use, range(2)))
    mx.eval(outputs)
    assert concatenate.call_count == len(owner._decode_pack_query[2])
    assert all(mx.array_equal(a, b).item() for a, b in zip(*outputs))


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("loop", [False, True])
def test_packed_projection_cached_logits_exact(metal_runtime, monkeypatch, dtype, loop):
    from mlx import nn

    from rapid_mlx.models.inkling2 import layers

    mx, _ = metal_runtime
    model = create_model(metal_runtime, dtype=getattr(mx, dtype), loop=loop)
    nn.quantize(
        model,
        group_size=64,
        class_predicate=lambda p, m: any(
            p.endswith("." + n + "_proj")
            for n in ("q", "k", "v", "r", "gate", "sink", "weight")
        ),
    )
    reference_cache, packed_cache = model.make_cache(), model.make_cache()
    tokens = mx.random.randint(0, 120, (1, 12))
    full = model(tokens)
    mx.eval(full)
    pack = layers._packed_quantized_projections

    def checked_pack(*args):
        output = pack(*args)
        assert output is not None
        return output

    spy = Mock(side_effect=checked_pack)
    for start, end in [(0, 6)] + [(i, i + 1) for i in range(6, 12)]:
        with monkeypatch.context() as m:
            m.setattr(layers, "_packed_quantized_projections", lambda *a: None)
            expected = model(tokens[:, start:end], cache=reference_cache)
            mx.eval(expected)
        spy.reset_mock()
        with monkeypatch.context() as m:
            m.setattr(layers, "_packed_quantized_projections", spy)
            spy.side_effect = (
                checked_pack if start else lambda *a: pytest.fail("prefill")
            )
            actual = model(tokens[:, start:end], cache=packed_cache)
            assert {c.args[1] for c in spy.call_args_list} == (
                {"query", "k", "v"} if start else set()
            )
        assert mx.array_equal(actual, expected).item()
        assert mx.array_equal(actual.view(mx.uint32), expected.view(mx.uint32)).item()
    assert layers._packed_quantized_projections is pack
    assert any(hasattr(layer.self_attn, "_decode_pack_query") for layer in model.layers)
    assert any(hasattr(layer.self_attn, "_decode_pack_k") for layer in model.layers)
    assert mx.array_equal(model(tokens), full).item()


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


@pytest.fixture
def projection_cpu(glue_loader, monkeypatch):
    """Load the real layer code with NumPy projections and inert kernel imports."""

    class Module(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None

        def __setattr__(self, name, value):
            self[name] = value

    class QuantizedLinear(Module):
        def __init__(self, width=4, size=2, *, biases=True):
            super().__init__()
            self.group_size, self.bits, self.mode = 4, 4, "affine"
            self.weight = np.arange(size * width, dtype=np.float32).reshape(size, width)
            self.scales = np.ones((size, 1), dtype=np.float32)
            if biases:
                self.biases = np.zeros((size, 1), dtype=np.float32)
            self.calls = Mock()

        def __call__(self, x):
            self.calls(x)
            return x @ self.weight.T

    nn = SimpleNamespace(
        Module=Module,
        QuantizedLinear=QuantizedLinear,
        Linear=lambda width, size, **kw: QuantizedLinear(width, size),
        Conv1d=Mock(),
    )
    glue_loader.module("mlx.nn", **vars(nn))
    monkeypatch.setattr(sys.modules["mlx"], "nn", nn)
    glue_loader.module("mlx.nn.layers", __path__=[])
    glue_loader.module("mlx.nn.layers.distributed", sum_gradients=lambda x: x)
    glue_loader.module(
        "mlx_lm.models.cache",
        RotatingKVCache=type("Ring", (), {}),
        BatchRotatingKVCache=type("BatchRing", (), {}),
    )
    glue_loader.module(
        "rapid_mlx.models.inkling2.compat",
        SwitchLinear=Module,
        _gather_sort=Mock(),
        _scatter_unsort=Mock(),
        expert_routing=None,
        scalar=lambda x, *a: np.asarray(x),
    )
    glue_loader.module("rapid_mlx.models.inkling2.config", TextArgs=object)
    kernels = glue_loader.module(
        "rapid_mlx.models.inkling2.kernels",
        **{
            name: Mock()
            for name in (
                "_expert_conv_residual_ready",
                "attend_clip_decode",
                "expert_conv_residual_decode",
                "attention_mask_decode",
                "depth_attention",
                "message_mix_decode",
                "message_norm_decode",
                "relative_attention",
                "route_topk_decode",
                "route_weights_decode",
                "short_convolution_decode",
            )
        },
    )
    mx = glue_loader.mx
    for name in (
        "zeros",
        "ones",
        "zeros_like",
        "stack",
        "log",
        "maximum",
        "arange",
        "int32",
    ):
        monkeypatch.setattr(mx, name, getattr(np, name), raising=False)
    monkeypatch.setattr(
        mx, "split", lambda x, ends, axis: np.split(x, ends, axis), raising=False
    )
    monkeypatch.setattr(mx, "compile", lambda fn, **kw: fn, raising=False)
    monkeypatch.setattr(
        mx,
        "quantized_matmul",
        Mock(side_effect=lambda x, w, **kw: x @ w.T),
        raising=False,
    )
    glue_loader.module(
        "rapid_mlx.models.inkling2.visit_norm",
        visit_norm_decode=Mock(return_value=None),
    )
    layers = glue_loader.load("layers")
    monkeypatch.setattr(layers, "_use_kernels", lambda: True)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "1")

    def attention(dynamic=False):
        cls = layers.FullAttention if dynamic else layers.SlidingAttention
        attn = cls(SimpleNamespace(**CONFIG["text_config"]))
        for name in ("q", "k", "v", "r", "gate", "sink"):
            setattr(attn, name + "_proj", QuantizedLinear())
        for name in ("k_sconv", "v_sconv"):
            conv = (
                layers.DynamicShortConvolution.__new__(layers.DynamicShortConvolution)
                if dynamic
                else Mock()
            )
            if dynamic:
                conv.weight_proj = QuantizedLinear()
            setattr(attn, name, conv)
        return attn

    return SimpleNamespace(
        layers=layers, mx=mx, nn=nn, attention=attention, kernels=kernels
    )


@pytest.mark.parametrize("biases", [False, True])
def test_packed_projection_cpu_reuse_and_replacement(projection_cpu, biases):
    cpu = projection_cpu
    owner = cpu.nn.Module()
    projections = [cpu.nn.QuantizedLinear(biases=biases) for _ in range(2)]
    x = np.ones((1, 1, 4), dtype=np.float32)
    pack = cpu.layers._packed_quantized_projections
    for step in range(3):
        if step == 2:
            projections[0].weight = projections[0].weight + 1
        expected = [p(x) for p in projections]
        actual = pack(owner, "test", projections, x)
        assert actual is not None
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        assert cpu.mx.quantized_matmul.call_count == (2, 3, 5)[step]
        assert cpu.mx.eval.call_count == (3, 3, 6)[step]
    for p in projections:
        assert np.shares_memory(p.weight, owner._decode_pack_test[2]["weight"])


@pytest.mark.parametrize(
    "failure",
    ["construction", "probe_invocation", "probe_evaluation", "slices"],
)
def test_packed_projection_cpu_failure_decode(projection_cpu, monkeypatch, failure):
    cpu = projection_cpu
    attn = cpu.attention()
    x = np.ones((1, 1, 4), dtype=np.float32)
    cache = [SimpleNamespace(offset=2)]
    pending = object()
    real_eval = cpu.mx.eval
    real_launch = cpu.mx.quantized_matmul
    originals = [attn.q_proj.weight, attn.k_proj.weight]
    for name in ("q", "k", "v", "r", "gate", "sink"):
        getattr(attn, name + "_proj").calls.reset_mock()
    attempts = Mock()

    def evaluate(*arrays):
        if any(a is pending for a in arrays):
            attempts("probe_evaluation")
            raise RuntimeError("lazy packed output failed during probe")
        if failure == "slices" and len(arrays) == 1 and isinstance(arrays[0], list):
            raise RuntimeError("slice evaluation failed")
        return real_eval(*arrays)

    def launch(*args, **kwargs):
        attempts("launch")
        if failure == "probe_invocation":
            raise RuntimeError("packed invocation failed")
        return (
            pending if failure == "probe_evaluation" else real_launch(*args, **kwargs)
        )

    if failure == "construction":
        monkeypatch.setattr(
            cpu.mx,
            "concatenate",
            Mock(side_effect=RuntimeError("pack construction failed")),
        )
    monkeypatch.setattr(cpu.mx, "quantized_matmul", launch)
    monkeypatch.setattr(cpu.mx, "eval", evaluate)
    for _ in range(2):
        q, k, v, r, gates, kw, vw = attn._project_inputs(x, x, x, cache)
        for name, actual in (("q", q), ("k", k), ("v", v), ("r", r)):
            projection = getattr(attn, name + "_proj")
            np.testing.assert_array_equal(actual, x @ projection.weight.T)
            assert projection.calls.call_count > 0  # Unpacked projections ran.
        assert gates == [] and kw is vw is None
    assert attn._decode_pack_query is False and attn._decode_pack_kv is False
    assert attn.q_proj.weight is originals[0] and attn.k_proj.weight is originals[1]
    if failure == "probe_evaluation":
        assert [c.args[0] for c in attempts.call_args_list] == [
            "launch",
            "probe_evaluation",
        ] * 2
    if failure == "construction":
        assert attempts.call_count == 0


@pytest.mark.parametrize("dynamic", [False, True])
def test_projection_cpu_attention_paths(projection_cpu, monkeypatch, dynamic):
    cpu = projection_cpu
    attn = cpu.attention(dynamic)
    x = np.ones((1, 1, 4), dtype=np.float32)
    cache = [SimpleNamespace(offset=2), SimpleNamespace(lengths=None)]
    expected = [getattr(attn, n + "_proj")(x) for n in ("q", "k", "v", "r")]
    for n in ("q", "k", "v", "r", "gate", "sink"):
        getattr(attn, n + "_proj").calls.reset_mock()
    actual = attn._project_inputs(x, x, x, cache)
    for a, e in zip(actual[:4], expected):
        np.testing.assert_array_equal(a, e)
    assert len(actual[4]) == 2
    for n in ("q", "k", "v", "r", "gate", "sink"):
        getattr(attn, n + "_proj").calls.assert_not_called()
    assert cpu.mx.quantized_matmul.call_count == (6 if dynamic else 4)
    if dynamic:
        assert actual[5] is not None and actual[6] is not None
    for name in ("k_sconv", "v_sconv"):
        setattr(attn, name, Mock(side_effect=lambda h, *a, **kw: (h, h)))
    object.__setattr__(attn, "_attend", Mock(return_value=x))
    attn.log_scaling_n_floor, attn.log_scaling_alpha = 4, 0.1
    for length in (1, 2):
        h = np.ones((1, length, 4), dtype=np.float32)
        assert attn(h, h, h, cache=cache, conv_states=[None, None], conv_mask=None) is x
        assert len(attn._attend.call_args.kwargs["gate_sinks"]) == (
            2 if length == 1 else 0
        )


@pytest.mark.parametrize("dynamic", [None, "provided"])
@pytest.mark.parametrize("fast", [False, True])
def test_projection_cpu_dynamic_convolution(projection_cpu, monkeypatch, dynamic, fast):
    cpu = projection_cpu
    layer = cpu.layers.DynamicShortConvolution.__new__(
        cpu.layers.DynamicShortConvolution
    )
    layer.kernel_size, layer.num_heads = 2, 1
    layer.static_weight = np.zeros((2, 2), dtype=np.float32)
    layer.weight_proj = Mock(return_value=np.ones((1, 1, 2), dtype=np.float32))
    monkeypatch.setattr(cpu.layers, "_use_kernels", lambda: fast)
    x = np.ones((1, 1, 2), dtype=np.float32)
    supplied = None if dynamic is None else np.ones((1, 1, 2), dtype=np.float32)
    layer(x, x, None, None, dynamic=supplied)
    assert layer.weight_proj.call_count == (1 if supplied is None else 0)
    assert cpu.kernels.short_convolution_decode.call_count == int(fast)
    if fast:
        np.testing.assert_array_equal(
            cpu.kernels.short_convolution_decode.call_args.kwargs["dynamic"],
            np.ones_like(x),
        )


def test_projection_cpu_guards_and_lock_recheck(projection_cpu, monkeypatch):
    cpu = projection_cpu
    x = np.ones((1, 1, 4), dtype=np.float32)
    pack = cpu.layers._packed_quantized_projections
    p, q = [cpu.nn.QuantizedLinear() for _ in range(2)]
    for field, invalid in (
        ("bits", 8),
        ("biases", None),
        ("scales", np.ones((2, 0))),
        ("scales", q.scales.astype(np.float16)),
        ("bias", np.zeros(2)),
    ):
        with monkeypatch.context() as patch:
            patch.setitem(q, field, invalid)
            assert pack(cpu.nn.Module(), "test", (p, q), x) is None
    assert pack(cpu.nn.Module(), "test", (p, Mock()), x) is None
    cpu.mx.quantized_matmul.assert_not_called()
    # Simulate another caller publishing while this caller waits for the lock.
    owner = cpu.nn.Module()

    class RacingLock:
        def __enter__(self):
            object.__setattr__(owner, "_decode_pack_test", False)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(cpu.layers, "_DECODE_PACK_LOCK", RacingLock())
    assert pack(owner, "test", (p, q), x) is None
    cpu.mx.quantized_matmul.assert_not_called()
    spy = Mock(side_effect=AssertionError("excluded packed path"))
    monkeypatch.setattr(cpu.layers, "_packed_quantized_projections", spy)
    attn = cpu.attention()
    for batch, length, cache, enabled, gpu in (
        (1, 2, [SimpleNamespace(offset=2)], True, True),
        (2, 1, [SimpleNamespace(offset=2)], True, True),
        (1, 1, None, True, True),
        (1, 1, [None], True, True),
        (1, 1, [SimpleNamespace(offset=0)], True, True),
        (1, 1, [SimpleNamespace(offset=2)], False, True),
        (1, 1, [SimpleNamespace(offset=2)], True, False),
    ):
        monkeypatch.setattr(
            cpu.layers, "_fast_decode_enabled", lambda enabled=enabled: enabled
        )
        monkeypatch.setattr(cpu.layers, "_use_kernels", lambda gpu=gpu: gpu)
        h = np.ones((batch, length, 4), dtype=np.float32)
        attn._project_inputs(h, h, h, cache)
    spy.assert_not_called()
    monkeypatch.setattr(cpu.layers, "_fast_decode_enabled", lambda: True)
    monkeypatch.setattr(cpu.layers, "_use_kernels", lambda: True)
    spy.side_effect, spy.return_value = None, None
    attn._project_inputs(x, x, x + 1, [SimpleNamespace(offset=2)])
    assert [c.args[1] for c in spy.call_args_list] == ["query"]


@pytest.mark.parametrize("provided", [False, True])
def test_projection_cpu_attend_gate_sinks(projection_cpu, monkeypatch, provided):
    cpu = projection_cpu
    attn = cpu.attention()
    attn.num_heads = attn.num_key_value_heads = 1
    attn.eps = 1e-6
    for name in ("q_l2_clip", "k_l2_clip", "rel_l2_clip", "rel_logits_proj", "o_proj"):
        setattr(attn, name, lambda x: x)
    monkeypatch.setattr(cpu.layers, "_value_norm", lambda x, eps: x)
    monkeypatch.setattr(cpu.layers, "_gate_attention", lambda x, gate: x + gate)
    object.__setattr__(
        attn, "_attention", Mock(side_effect=lambda q, *a: q.transpose(0, 2, 1, 3))
    )
    x = np.ones((1, 1, 4), dtype=np.float32)
    gates = (attn.gate_proj(x), attn.sink_proj(x)) if provided else None
    attn.gate_proj.calls.reset_mock()
    attn.sink_proj.calls.reset_mock()
    result = attn._attend(
        x, x[..., :2], x[..., :2], x[..., :2], x[..., :2], None, gate_sinks=gates
    )
    np.testing.assert_array_equal(result, x[..., :2] + x @ attn.gate_proj.weight.T)
    assert (
        attn.gate_proj.calls.call_count
        == attn.sink_proj.calls.call_count
        == int(not provided)
    )


def test_message_norm_shared_decode_switch(message_cpu, monkeypatch):
    mx, _, layers = message_cpu
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    forbidden = Mock(side_effect=AssertionError("disabled message normalization"))
    monkeypatch.setattr(layers, "message_norm_decode", forbidden)
    x = mx.ones((1, 1, 17))
    np.testing.assert_array_equal(
        layers._message_input(x, x, mx.float16, True),
        layers._message_input(x, x, mx.float16),
    )
    np.testing.assert_array_equal(
        layers._norm_cast(x, mx.ones((17,)), 1e-6, mx.float16, True),
        layers._norm_cast(x, mx.ones((17,)), 1e-6, mx.float16),
    )
    assert not layers._fast_decode_enabled()
    forbidden.assert_not_called()


def test_projection_shared_decode_switch(projection_cpu, monkeypatch):
    cpu = projection_cpu
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    forbidden = Mock(side_effect=AssertionError("disabled packed projection"))
    monkeypatch.setattr(cpu.layers, "_packed_quantized_projections", forbidden)
    x = np.ones((1, 1, 4), dtype=np.float32)
    attn = cpu.attention()
    actual = attn._project_inputs(x, x, x, [SimpleNamespace(offset=2)])
    for name, value in zip(("q", "k", "v", "r"), actual):
        np.testing.assert_array_equal(value, getattr(attn, name + "_proj")(x))
    assert not cpu.layers._fast_decode_enabled()
    forbidden.assert_not_called()


def test_router_shared_decode_switch(projection_cpu, monkeypatch):
    cpu = projection_cpu
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0")
    router = cpu.layers.TopKRouter(SimpleNamespace(**CONFIG["text_config"]))
    x = np.ones((1, 1, 64), dtype=np.float32)
    forbidden = Mock(side_effect=AssertionError("disabled router top-k"))
    monkeypatch.setattr(cpu.layers, "route_topk_decode", forbidden)
    monkeypatch.setattr(
        cpu.mx, "sigmoid", lambda x: 1 / (1 + np.exp(-x)), raising=False
    )
    reference = (np.array([0, 1]), np.array([0.5, 0.5]))
    fallback = Mock(return_value=reference)
    monkeypatch.setattr(cpu.layers, "_route_weights", fallback)
    assert router(x, cached_decode=True) is reference
    assert not cpu.layers._fast_decode_enabled()
    fallback.assert_called_once()
    forbidden.assert_not_called()


@pytest.fixture
def attend_cpu(message_cpu, monkeypatch):
    mx, kernels, layers = message_cpu
    for name in ("sqrt", "minimum"):
        monkeypatch.setattr(mx, name, getattr(np, name), raising=False)
    return mx, kernels, layers


def _attend_args(mx, dk=16, dr=4, dtype=None, tau=True):
    dtype = mx.float32 if dtype is None else dtype
    return [
        mx.ones((1, 1, 4, dk), dtype=dtype),
        mx.ones((1, 1, 2, dk), dtype=dtype),
        mx.ones((1, 1, 4, dr), dtype=dtype),
        mx.ones((4, dk)),
        mx.ones((4, dr)),
        3.0,
        2.0,
        mx.ones((1,)) if tau else None,
    ]


@pytest.mark.parametrize("tau", [False, True])
@pytest.mark.parametrize(
    "failure", [None, "construction", "probe", "evaluation", "call"]
)
def test_attend_clip_cpu_failures(attend_cpu, monkeypatch, tau, failure):
    mx, kernels, layers = attend_cpu
    args = _attend_args(mx, tau=tau)
    probe = [object(), object(), object()]
    events = []

    def launch(**kw):
        events.append("launch")
        if len(events) == 1:
            if failure == "probe":
                raise RuntimeError("probe invocation")
            return probe
        if failure == "call":
            raise RuntimeError("real invocation")
        q, k, r, qs, rs, qt, rt, t = kw["inputs"]
        out = [
            layers._l2_clip(x, s, threshold)
            for x, s, threshold in ((q, qs, qt), (k, 1.0, qt), (r, rs, rt))
        ]
        if tau:
            out[0], out[2] = [layers._scale_states(out[i], t) for i in (0, 2)]
        return out

    def evaluate(*outputs):
        assert outputs == tuple(probe)
        events.append("evaluate")
        if failure == "evaluation":
            raise RuntimeError("lazy compilation")

    factory = Mock(
        side_effect=RuntimeError("construction") if failure == "construction" else None,
        return_value=launch,
    )
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    mx.eval.side_effect = evaluate
    for _ in range(2):
        actual = kernels.attend_clip_decode(*args, row_contiguous=True)
        if failure:
            assert actual is None
        else:
            assert actual is not None
            np.testing.assert_array_equal(
                actual[0], layers._l2_clip(args[0], args[3], 3)
            )
    factory.assert_called_once()
    assert factory.call_args.kwargs["ensure_row_contiguous"] is True
    assert "_strides" not in factory.call_args.kwargs["source"]
    if failure == "construction":
        assert events == []
    elif failure == "probe":
        assert events == ["launch"]
    elif failure == "evaluation":
        assert events == ["launch", "evaluate"]
    elif failure == "call":
        assert events == ["launch", "evaluate", "launch"]
    else:
        assert events == ["launch", "evaluate", "launch", "launch"]
    assert len(kernels._ATTEND_CLIP_CACHE) == 1


@pytest.mark.parametrize("tau", [False, True])
def test_attend_clip_cpu_invocation_failure_is_variant_local(
    attend_cpu, monkeypatch, tau
):
    mx, kernels, _ = attend_cpu

    class LockedCache(dict):
        def __setitem__(self, variant, value):
            assert kernels._ATTEND_CLIP_LOCK.locked()
            super().__setitem__(variant, value)

    monkeypatch.setattr(kernels, "_ATTEND_CLIP_CACHE", LockedCache())
    outputs = [object(), object(), object()]
    broken = Mock(side_effect=[outputs, RuntimeError("real invocation")])
    working = Mock(return_value=outputs)
    factory = Mock(side_effect=[broken, working])
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    args = _attend_args(mx, tau=tau)
    for _ in range(2):
        assert kernels.attend_clip_decode(*args, row_contiguous=True) is None
    factory.assert_called_once()
    assert broken.call_count == 2  # Probe succeeds; only the first real call runs.
    assert list(kernels._ATTEND_CLIP_CACHE.values()) == [None]
    args[-1] = None if tau else mx.ones((1,))
    assert kernels.attend_clip_decode(*args, row_contiguous=True) is outputs
    assert factory.call_count == working.call_count == mx.eval.call_count == 2
    args[-1] = mx.ones((1,)) if tau else None
    assert kernels.attend_clip_decode(*args, row_contiguous=True) is None
    assert factory.call_count == broken.call_count == 2


@pytest.mark.parametrize("failed", [False, True])
def test_attend_clip_cpu_concurrent_first_use(attend_cpu, monkeypatch, failed):
    mx, kernels, _ = attend_cpu
    barrier = Barrier(2)
    lock = kernels._ATTEND_CLIP_LOCK

    class ContendedLock:
        def __enter__(self):
            # Both callers have missed the unlocked lookup before either probes.
            barrier.wait(timeout=5)
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    monkeypatch.setattr(kernels, "_ATTEND_CLIP_LOCK", ContendedLock())
    outputs = [object(), object(), object()]
    factory = Mock(side_effect=lambda **kw: Mock(return_value=outputs))
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    if failed:
        mx.eval.side_effect = RuntimeError("lazy probe failed")
    variant = (4, 2, 16, 4, mx.float32, mx.float32, mx.float32, False)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: kernels._get_attend_clip(*variant), range(2)))
    assert results[0] is results[1]
    assert (results[0] is None) == failed
    factory.assert_called_once()
    mx.eval.assert_called_once_with(*outputs)
    assert len(kernels._ATTEND_CLIP_CACHE) == 1
    if not failed:
        results[0].assert_called_once()
    # Cached failures and successes never enter the contended lock again.
    assert kernels._get_attend_clip(*variant) is results[0]


@pytest.mark.parametrize("case", range(26))
def test_attend_clip_cpu_guards(attend_cpu, monkeypatch, case):
    mx, kernels, _ = attend_cpu
    args = _attend_args(mx)
    changes = [
        *[(i, None) for i in range(7)],
        (0, mx.ones((1, 4, 16))),
        (0, mx.ones((2, 1, 4, 16))),
        (0, mx.ones((1, 2, 4, 16))),
        (1, mx.ones((1, 1, 2, 16), dtype=mx.float16)),
        (0, mx.ones((1, 1, 4, 16), dtype=np.int32)),
        (0, mx.ones((1, 1, 0, 16))),
        (0, mx.ones((1, 1, 129, 16))),
        (1, mx.ones((1, 1, 0, 16))),
        (1, mx.ones((1, 1, 129, 16))),
        (2, mx.ones((1, 1, 3, 4))),
        (1, mx.ones((1, 1, 2, 32))),
        (0, mx.ones((1, 1, 4, 8))),
        (2, mx.ones((1, 1, 4, 128))),
        (3, mx.ones((4, 15))),
        (4, mx.ones((4, 3))),
        (3, mx.ones((4, 16), dtype=np.int32)),
        (4, mx.ones((4, 4), dtype=np.int32)),
        (7, mx.ones((2,))),
        (7, mx.ones((1,), dtype=mx.float16)),
    ]
    i, value = changes[case]
    args[i] = value
    forbidden = Mock(side_effect=AssertionError("unsupported kernel"))
    monkeypatch.setattr(kernels, "_get_attend_clip", forbidden)
    assert kernels.attend_clip_decode(*args, row_contiguous=True) is None
    forbidden.assert_not_called()


@pytest.mark.parametrize("tau", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        "fast",
        "fallback",
        "prefill",
        "uncached",
        "off",
        "cpu",
        "table",
        "construction",
        "probe",
        "evaluation",
        "call",
        "unproven",
        "reverse_q",
        "reverse_k",
        "reverse_r",
        "reverse_qs",
        "reverse_rs",
        "transpose_k",
    ],
)
def test_attend_clip_cpu_attention(attend_cpu, monkeypatch, tau, mode):
    mx, kernels, layers = attend_cpu
    q, k, r, qs, rs, qt, rt, t = _attend_args(mx, tau=tau)
    unproven = mode == "unproven" or mode.startswith(("reverse_", "transpose_"))
    if unproven:
        # Losing the trailing ones in a serial FP32 sum changes clip(3).
        q[..., 0] = k[..., 0] = r[..., 0] = 4096
        args = [q, k, r, qs, rs]
        if mode.startswith("reverse_"):
            i = ("q", "k", "r", "qs", "rs").index(mode.removeprefix("reverse_"))
            args[i] = args[i][..., ::-1]
        elif mode == "transpose_k":
            args[1] = np.ascontiguousarray(k.swapaxes(-1, -2)).swapaxes(-1, -2)
        q, k, r, qs, rs = args
        forbidden = Mock(side_effect=AssertionError("unproven layout probed"))
        monkeypatch.setattr(kernels, "_get_attend_clip", forbidden)
        assert kernels.attend_clip_decode(q, k, r, qs, rs, qt, rt, t) is None
    if mode == "prefill":
        q, k, r = [np.repeat(a, 2, axis=1) for a in (q, k, r)]
    shape = q.shape[:2] + (64,)
    hidden = mx.ones(shape)
    clipped = [
        layers._l2_clip(x, s, threshold)
        for x, s, threshold in ((q, qs, qt), (k, 1.0, qt), (r, rs, rt))
    ]
    if tau:
        clipped[0], clipped[2] = [layers._scale_states(clipped[i], t) for i in (0, 2)]
    failed = mode in ("construction", "probe", "evaluation", "call")
    if failed:
        outputs = [object(), object(), object()]
        kernel = Mock(
            side_effect=RuntimeError("probe") if mode == "probe" else None,
            return_value=outputs,
        )
        if mode == "call":
            kernel.side_effect = [outputs, RuntimeError("real invocation")]
        factory = Mock(
            side_effect=RuntimeError("construction")
            if mode == "construction"
            else None,
            return_value=kernel,
        )
        monkeypatch.setattr(mx.fast, "metal_kernel", factory)
        if mode == "evaluation":
            mx.eval.side_effect = RuntimeError("lazy compilation")
        fast = Mock(wraps=kernels.attend_clip_decode)
    elif unproven:
        fast = Mock(wraps=kernels.attend_clip_decode)
    else:
        fast = Mock(return_value=clipped if mode == "fast" else None)
    monkeypatch.setattr(layers, "attend_clip_decode", fast)
    monkeypatch.setenv("RAPID_MLX_INKLING2_FAST_DECODE", "0" if mode == "off" else "1")
    monkeypatch.setattr(layers, "_use_kernels", lambda: mode != "cpu")
    monkeypatch.setitem(layers._DECODE_FUSIONS, "attend_clip", mode != "table")

    def clip(scale, threshold):
        result = Mock(side_effect=lambda x: layers._l2_clip(x, scale, threshold))
        result.scale, result.threshold = scale, threshold
        return result

    attn = SimpleNamespace(
        num_heads=4,
        num_key_value_heads=2,
        eps=1e-6,
        q_l2_clip=clip(qs, qt),
        k_l2_clip=clip(1.0, qt),
        rel_l2_clip=clip(rs, rt),
        gate_proj=Mock(return_value=hidden),
        sink_proj=Mock(return_value=hidden),
        rel_logits_proj=Mock(side_effect=lambda x: x),
        _attention=Mock(side_effect=lambda q, *a: q.transpose(0, 2, 1, 3)),
        o_proj=Mock(side_effect=lambda x: x),
    )
    cache = (
        None
        if mode == "uncached"
        else SimpleNamespace(update_and_fetch=Mock(side_effect=lambda k, v: (k, v)))
    )
    norm = Mock(side_effect=lambda x, eps: x)
    monkeypatch.setattr(layers, "_value_norm", norm)
    monkeypatch.setattr(layers, "_gate_attention", lambda x, gate: x)
    result = layers.Attention._attend(
        attn,
        hidden,
        q,
        k,
        k,
        r,
        cache,
        t,
        **({} if unproven else {"row_contiguous": True}),
    )
    assert fast.call_count == int(mode in ("fast", "fallback") or failed or unproven)
    if unproven:
        forbidden.assert_not_called()
    if failed:
        factory.assert_called_once()
        assert kernel.call_count == (
            0 if mode == "construction" else 2 if mode == "call" else 1
        )
        assert mx.eval.call_count == (0 if mode in ("construction", "probe") else 1)
    for c in (attn.q_l2_clip, attn.k_l2_clip, attn.rel_l2_clip):
        assert c.call_count == (0 if mode == "fast" else 1)
    norm.assert_called_once()
    np.testing.assert_array_equal(
        attn._attention.call_args.args[0], clipped[0].transpose(0, 2, 1, 3)
    )
    np.testing.assert_array_equal(attn.rel_logits_proj.call_args.args[0], clipped[2])
    np.testing.assert_array_equal(
        attn._attention.call_args.args[1], clipped[1].transpose(0, 2, 1, 3)
    )
    if mode == "call":
        # The failed invocation and every later visit use exactly the existing path.
        repeated = layers.Attention._attend(
            attn, hidden, q, k, k, r, cache, t, row_contiguous=True
        )
        assert (
            result.tobytes()
            == repeated.tobytes()
            == clipped[0].reshape(result.shape).tobytes()
        )
        assert kernel.call_count == 2  # One probe and one failed real invocation.
        factory.assert_called_once()
        assert mx.eval.call_count == 1
        for actual, expected in (
            (attn._attention.call_args.args[0], clipped[0].transpose(0, 2, 1, 3)),
            (attn._attention.call_args.args[1], clipped[1].transpose(0, 2, 1, 3)),
            (attn.rel_logits_proj.call_args.args[0], clipped[2]),
        ):
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("heads", [(4, 2), (32, 32)])
def test_attend_clip_exact(metal_runtime, monkeypatch, dtype, heads):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    original = kernels._get_attend_clip
    launches = Mock()

    def selected(*args):
        kernel = original(*args)
        assert kernel is not None
        return lambda **kw: (launches(), kernel(**kw))[1]

    monkeypatch.setattr(kernels, "_get_attend_clip", selected)
    dtype = getattr(mx, dtype)
    hq, hk = heads  # Exercise both row_reduce_looped and row_reduce_simple at 128.
    mx.random.seed(41)
    for dk, dr in ((4, 4), (16, 16), (32, 32), (64, 64), (128, 4), (128, 64)):
        for tau in (None, mx.array([1.137])):
            for magnitude in (0.0, 0.1, 10.0):
                for scale_dtype in (mx.float32, dtype):
                    q, k, r = [
                        (mx.random.normal((1, 1, h, d)) * magnitude).astype(dtype)
                        for h, d in ((hq, dk), (hk, dk), (hq, dr))
                    ]
                    qs, rs = [
                        mx.random.normal((hq, d)).astype(scale_dtype) for d in (dk, dr)
                    ]
                    expected = [
                        layers._l2_clip(x, s, threshold)
                        for x, s, threshold in (
                            (q, qs, 3.0),
                            (k, 1.0, 3.0),
                            (r, rs, 2.0),
                        )
                    ]
                    if tau is not None:
                        expected[0], expected[2] = [
                            layers._scale_states(expected[i], tau) for i in (0, 2)
                        ]
                    launches.reset_mock()
                    actual = kernels.attend_clip_decode(
                        q, k, r, qs, rs, 3.0, 2.0, tau, row_contiguous=True
                    )
                    launches.assert_called_once()
                    assert actual is not None
                    for a, e in zip(actual, expected):
                        assert a.dtype == e.dtype
                        assert mx.array_equal(a.view(mx.uint8), e.view(mx.uint8)).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_attend_clip_layouts(metal_runtime, monkeypatch, dtype):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import kernels, layers

    get_kernel = Mock(wraps=kernels._get_attend_clip)
    monkeypatch.setattr(kernels, "_get_attend_clip", get_kernel)
    for dk in (4, 16, 32, 64, 128):
        args = _attend_args(mx, dk, min(dk, 64), getattr(mx, dtype), tau=False)
        # The review counterexample: every head starts [4096, 1, ..., 1].
        for i in range(3):
            values = np.ones(args[i].shape, dtype=np.float32)
            values[..., 0] = 4096
            args[i] = mx.array(values).astype(getattr(mx, dtype))
        args[6] = 3.0
        row_key = layers._l2_clip(args[1], 1.0, 3.0)
        for layout in ("row", "transpose_k", *range(5)):
            inputs = list(args)
            if layout == "transpose_k":
                inputs[1] = mx.contiguous(inputs[1].swapaxes(-1, -2)).swapaxes(-1, -2)
                assert not memoryview(inputs[1]).c_contiguous
            elif isinstance(layout, int):
                inputs[layout] = inputs[layout][..., ::-1]
                assert memoryview(inputs[layout]).strides[-1] < 0
            get_kernel.reset_mock()
            actual = kernels.attend_clip_decode(
                *inputs, **({"row_contiguous": True} if layout == "row" else {})
            )
            expected = [
                layers._l2_clip(x, s, threshold)
                for x, s, threshold in (
                    (inputs[0], inputs[3], 3.0),
                    (inputs[1], 1.0, 3.0),
                    (inputs[2], inputs[4], 3.0),
                )
            ]
            mx.eval(*expected)
            if layout == "row":
                get_kernel.assert_called_once()
                assert actual is not None
                for a, e in zip(actual, expected):
                    assert mx.array_equal(a.view(mx.uint8), e.view(mx.uint8)).item()
            else:
                assert actual is None  # Caller must run the existing path.
                get_kernel.assert_not_called()
                if layout == "transpose_k" and dtype == "float32" and dk == 16:
                    assert not mx.array_equal(expected[1], row_key).item()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("quantized", [False, True])
def test_attend_clip_cached_logits_exact(metal_runtime, monkeypatch, dtype, quantized):
    mx, _ = metal_runtime
    from mlx import nn

    from rapid_mlx.models.inkling2 import layers

    stock = create_model(metal_runtime, dtype=getattr(mx, dtype))
    fused = create_model(metal_runtime, dtype=getattr(mx, dtype))
    if quantized:
        for model in (stock, fused):
            nn.quantize(
                model,
                group_size=64,
                class_predicate=lambda p, m: any(
                    p.endswith("." + n + "_proj")
                    for n in ("q", "k", "v", "r", "gate", "sink", "weight")
                ),
            )
    caches = stock.make_cache(), fused.make_cache()
    tokens = mx.array([[1, 17, 35, 7, 8, 11, 12, 13]])
    assert mx.array_equal(
        stock(tokens[:, :4], cache=caches[0]), fused(tokens[:, :4], cache=caches[1])
    ).item()
    fast = layers.attend_clip_decode
    seen = []

    def record(*args, **kwargs):
        # Test-only evaluation exposes actual storage, including the nonzero
        # offsets of packed Q/R slices and both static/dynamic K convolutions.
        mx.eval(*args[:5])
        for x in args[:5]:
            layout = memoryview(x)
            assert layout.c_contiguous
            assert layout.strides[-1] == x.itemsize
            assert layout.strides[-2] == x.shape[-1] * x.itemsize
        assert kwargs == {"row_contiguous": True}
        out = fast(*args, **kwargs)
        assert out is not None
        seen.append(args[-1] is not None)
        return out

    monkeypatch.setattr(layers, "attend_clip_decode", record)
    for step in range(4, 8):
        monkeypatch.setitem(layers._DECODE_FUSIONS, "attend_clip", False)
        expected = stock(tokens[:, step : step + 1], cache=caches[0])
        monkeypatch.setitem(layers._DECODE_FUSIONS, "attend_clip", True)
        seen.clear()
        actual = fused(tokens[:, step : step + 1], cache=caches[1])
        assert set(seen) == {False, True}
        assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()


def test_attend_clip_cpu_probe_variants(attend_cpu, monkeypatch):
    mx, kernels, _ = attend_cpu
    kernel = Mock(return_value=[object(), object(), object()])
    factory = Mock(return_value=kernel)
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    variants = []
    for dk, dr, dtype, qs_dtype, rs_dtype, tau in (
        (16, 4, mx.float32, mx.float32, mx.float32, False),
        (16, 4, mx.float32, mx.float32, mx.float32, True),
        (128, 4, mx.float32, mx.float32, mx.float32, False),
        (16, 64, mx.float32, mx.float32, mx.float32, False),
        (16, 4, mx.float16, mx.float32, mx.float32, False),
        (16, 4, mx.float32, mx.float16, mx.float32, False),
        (16, 4, mx.float32, mx.float32, mx.float16, False),
    ):
        args = _attend_args(mx, dk, dr, dtype, tau)
        args[3], args[4] = args[3].astype(qs_dtype), args[4].astype(rs_dtype)
        variants.append(args)
    # Head counts also belong to the evaluated specialization.
    args = _attend_args(mx, tau=False)
    args[0], args[2], args[3], args[4] = [args[i][..., :1, :] for i in (0, 2, 3, 4)]
    args[1] = args[1][..., :1, :]
    variants.append(args)
    for index, args in enumerate(variants, 1):
        for _ in range(2):
            assert kernels.attend_clip_decode(*args, row_contiguous=True) is not None
        assert factory.call_count == mx.eval.call_count == index
        probe, real, repeated = kernel.call_args_list[-3:]
        for call in (probe, real, repeated):
            kw = call.kwargs
            assert kw["grid"] == ((2 * args[0].shape[2] + args[1].shape[2]) * 32, 1, 1)
            assert kw["threadgroup"] == (32, 1, 1)
            assert kw["output_shapes"] == [a.shape for a in args[:3]]
            assert kw["output_dtypes"] == [a.dtype for a in args[:3]]
    assert kernel.call_count == 3 * len(variants)
