# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
# Tiny configuration and checkpoint conversion adapted from the source in
# rapid_mlx/models/inkling2/NOTICE, under its accompanying MIT license.
"""Text-only catalog, loop/cache, quantization and loader regressions."""

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
    "layers.py": "11ab8c3510582229a222742960d64465dcf555fc2fa7a93586d0e40947137acb",
    "kernels.py": "528f3396a1f46c6abd0ff65ceb99deacba9b69253eb5475674df01fd42b9f169",
    "model.py": "d4244b2ab9864a3d99c2122ba27897c945a45004c6802ebf37f4a0bd7b99a091",
    "switch_layers.py": "31fb0b6e31112276694f4fa8ba25a92cfc994f01bb90780792e54fd8fd4c54c9",
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

    assert len(calls) == 3
    for name, launches in calls.items():
        assert len(launches) == 120
        assert len({tuple(call["template"]) for call in launches}) == 1, name


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
    } | {"_l2_clip", "_softcapped_swiglu"}
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
def test_compiled_helpers_ignore_batch_and_offsets(metal_runtime, compile_traces):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    # Exercise the eager router alongside the compiled helpers.
    # k=7 and these dimensions keep this independent of other tests' warmups.
    expected_traces = None
    for step in range(40):
        for batch in (1, 2, 3):
            x = mx.ones((batch, 1, 24))
            experts = mx.ones((batch, 1, 7, 24))
            weights = mx.ones((batch, 1, 7))
            indices, routes = layers._route_weights(x, mx.zeros((24,)), 7, 2.0)
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
                layers._combine_experts(experts, weights),
                layers._normalized_message_input(x, x),
                layers._message_mix(x, x, x, x),
                layers._depth_attention_reference(
                    x, mx.ones((2, batch, 1, 24)), mx.ones((4, 24)), 0.2
                ),
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
def test_decode_kill_switch_and_prefill(metal_runtime, monkeypatch):
    mx, _ = metal_runtime
    from rapid_mlx.models.inkling2 import layers

    def unexpected(*args, **kwargs):
        pytest.fail("decode convolution called for prefill or with kill switch off")

    monkeypatch.setattr(layers, "short_convolution_decode", unexpected)
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
    assert mx.allclose(actual, full, atol=1e-4, rtol=1e-4).item()
    assert all(c[0].offset == 20 for c in cache)


@pytest.mark.requires_mlx
@pytest.mark.parametrize("batch", [1, 3])
def test_fast_decode_model_parity(metal_runtime, monkeypatch, batch):
    mx, _ = metal_runtime
    model = create_model(metal_runtime, dtype=mx.float32)
    tokens = mx.random.randint(0, 120, (batch, 14))
    reference_cache, fast_cache = model.make_cache(), model.make_cache()
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
        assert mx.allclose(actual, reference, atol=1e-4, rtol=1e-4).item()
        for fast, slow in zip(fast_cache, reference_cache):
            assert fast[0].offset == slow[0].offset == step + 1
            for fast_state, slow_state in zip(fast[1].cache, slow[1].cache):
                assert mx.allclose(fast_state, slow_state, atol=1e-4, rtol=1e-4).item()


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

    expected_keys = expected_traces = None
    for step in range(6, 46):
        for ids, cache in zip(tokens, caches):
            logits = model(ids[:, step : step + 1], cache=cache)
            mx.eval(logits)
            assert mx.all(mx.isfinite(logits)).item()
            assert all(c[0].offset == step + 1 for c in cache)
            if expected_keys is None:
                expected_keys = {name: seen.copy() for name, seen in keys.items()}
                expected_traces = compile_traces.copy()
            # A single warmed decode step covers every model-constant layer
            # variant. Neither growing KV banks nor changing B adds a key.
            assert keys == expected_keys
            assert compile_traces == expected_traces
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
    assert mx.allclose(model(tokens), target(tokens), atol=1e-4, rtol=1e-4).item()


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
