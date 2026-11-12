# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 opt-in policy, model-local dispatch and production enrollment."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx import lattice
from rapid_mlx.lattice import defaults
from rapid_mlx.patches import gemma4_moe_decode_fusion as fusion

NAME = "gemma4_moe_decode_fusion"


class Tensor:
    def __init__(self, shape, dtype="fp16"):
        self.shape, self.dtype, self.ndim = shape, dtype, len(shape)


def test_registration_and_schema_parity(monkeypatch, tmp_path):
    assert lattice.KNOBS[NAME].default is False
    assert lattice.KNOBS[NAME].valid(True)
    assert not lattice.KNOBS[NAME].valid(1)
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "proto/model-runtime/v1/lattice-profile.schema.json",
        root / "rapid_mlx/catalog/schemas/lattice-profile.schema.json",
    )
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert f'"{NAME}"' in paths[0].read_text()
    cells = [c for c in lattice.builtin_profile().cells if NAME in c["knobs"]]
    # Enabled only on measured generation/tier cells; unknown hardware stays off.
    assert {c["id"]: c["knobs"][NAME] for c in cells} == {
        "gemma4-moe-decode-15-ultra": True,
        "gemma4-moe-decode-16-pro": True,
        "gemma4-moe-decode-17-max": True,
    }
    ctx = lattice.runtime_context({}, model=lattice.ModelProfile(NAME + "_install"))
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    monkeypatch.delenv("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION", raising=False)
    monkeypatch.delenv("RAPID_MLX_BENCHMARK_KNOBS", raising=False)
    assert lattice.resolve(NAME, ctx).value is False
    cell = {"id": "ab", "knobs": {NAME: True}, "provenance": {"evidence": "test"}}
    profile_path = tmp_path / "overlay.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_version": "gemma4-ab.1",
                "generated_at": "2026-11-12T00:00:00Z",
                "cells": [cell],
            }
        )
    )
    profile = defaults.load_profile(profile_path, lattice.KNOBS)
    assert profile.cells == (cell,)
    monkeypatch.setattr(lattice, "overlay_profile", lambda: profile)
    assert lattice.resolve(NAME, ctx).value is True
    assert lattice.resolve(NAME, lattice.runtime_context({})).value is False
    monkeypatch.setenv("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION", "0")
    assert lattice.resolve(NAME, ctx).value is False


@pytest.fixture
def setup(monkeypatch):
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        default_device=lambda: "gpu",
        gpu="gpu",
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        int32="i32",
        uint32="u32",
        int64="i64",
        uint64="u64",
        device_info=lambda: {},
        clear_cache=lambda: None,
    )
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    gate_up = importlib.import_module("rapid_mlx.moe_fusion")
    monkeypatch.setattr(gate_up, "mx", mx)

    class Switch:
        pass

    class Experts:
        training = False

        def __init__(self):
            self.switch_glu = Mock(return_value=Tensor((1, 1, 8, 128)))
            self.layer = Switch()

        def __call__(self, x, top_k_indices, top_k_weights):
            return "stock"

    upstream = ModuleType("mlx_vlm.models.gemma4.language")
    upstream.Experts = Experts
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    block = Experts()
    model = SimpleNamespace(
        named_modules=lambda: [("expert", block), ("glu", block.layer)]
    )
    model.discovery = gate_up._switch_layer_families
    monkeypatch.setattr(
        gate_up,
        "_switch_layer_families",
        lambda **_: ((object, Switch, object, None, None, None),),
    )
    monkeypatch.setattr(
        gate_up, "_can_fuse", lambda b, *_: not hasattr(b, "gate_up_proj")
    )
    monkeypatch.setattr(gate_up, "_ensure_call_patch", lambda *_: None)
    monkeypatch.setattr(
        gate_up, "_fuse_one", lambda b, _: setattr(b, "gate_up_proj", True)
    )
    monkeypatch.setattr(lattice, "overlay_profile", lambda: defaults.Profile())
    for env in (
        "RAPID_MLX_MOE_DECODE_FUSION",
        "RAPID_MLX_MOE_GATE_UP_FUSION",
        "RAPID_MLX_BENCHMARK_KNOBS",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION", "1")
    monkeypatch.setattr(fusion, "combine", Mock(return_value="fused"))
    return mx, model, block, Experts


@pytest.mark.parametrize(
    "model_type",
    [
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "gemma4_assistant",
        "qwen3_5_moe",
        "gemma4_future",
    ],
)
def test_enrollment(setup, model_type):
    _, model, block, base = setup
    expected = model_type in {"gemma4", "gemma4_text"}
    assert fusion.install_gemma4_moe_decode_fusion(model, model_type) == int(expected)
    assert (type(block) is base) != expected
    assert hasattr(block.layer, "gate_up_proj") == expected


def test_exact_type_idempotence_and_dispatch(setup, monkeypatch):
    _, model, block, base = setup

    class Subclass(base):
        pass

    sub, other = Subclass(), base()
    original = model.named_modules
    model.named_modules = lambda: original() + [("sub", sub)]
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 1
    assert fusion.is_installed(model)
    assert type(sub) is Subclass
    assert type(other) is base
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 0
    monkeypatch.setattr(
        fusion,
        "install_enabled",
        Mock(side_effect=AssertionError("per-token resolution")),
    )
    x, indices, weights = (
        Tensor((1, 1, 128)),
        Tensor((1, 1, 8), "i32"),
        Tensor((1, 1, 8)),
    )
    assert block(x, indices, weights) == "fused"
    for shape in [(1, 2, 128), (2, 1, 128), (2, 2, 128)]:
        assert block(Tensor(shape), indices, weights) == "stock"


@pytest.mark.parametrize(
    "env,decode,gate",
    [
        ("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION", False, False),
        ("RAPID_MLX_MOE_DECODE_FUSION", False, True),
        ("RAPID_MLX_MOE_GATE_UP_FUSION", True, False),
    ],
)
def test_opt_outs(setup, monkeypatch, env, decode, gate):
    _, model, block, _ = setup
    monkeypatch.setenv(env, "0")
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == int(decode)
    assert hasattr(block.layer, "gate_up_proj") == gate


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "no_metal",
        "training",
        "sharded",
        "dtype",
        "indices",
        "k",
        "shape",
        "y_dtype",
        "kernel",
    ],
)
def test_safety_fallback(setup, monkeypatch, case):
    mx, model, block, _ = setup
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 1
    x, indices, weights = (
        Tensor((1, 1, 128)),
        Tensor((1, 1, 8), "i32"),
        Tensor((1, 1, 8)),
    )
    if case == "cpu":
        mx.default_device = lambda: "cpu"
    elif case == "no_metal":
        mx.metal.is_available = lambda: False
    elif case == "training":
        block.training = True
    elif case == "sharded":
        block.sharding_group = object()
    elif case == "dtype":
        weights.dtype = "fp64"
    elif case == "indices":
        indices.dtype = "fp16"
    elif case == "k":
        indices = Tensor((1, 1, 9), "i32")
        weights = Tensor((1, 1, 9))
    elif case == "shape":
        block.switch_glu.return_value = Tensor((2, 1, 8, 128))
    elif case == "y_dtype":
        block.switch_glu.return_value.dtype = "fp64"
    else:
        monkeypatch.setattr(fusion, "combine", Mock(side_effect=RuntimeError("kernel")))
    assert block(x, indices, weights) == "stock"


@pytest.mark.parametrize(
    "family",
    [
        "mlx_lm.models.gemma4_text",
        "mlx_vlm.models.gemma4_text.language",
        "rapid_mlx.models.gemma4_vendored.language",
    ],
)
def test_loaded_expert_families(setup, monkeypatch, family):
    _, model, _, base = setup
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.gemma4.language")
    monkeypatch.setitem(sys.modules, family, SimpleNamespace(Experts=base))
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 1


def test_default_off_at_install(setup, monkeypatch):
    _, model, block, base = setup
    monkeypatch.delenv("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION")
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 0
    assert type(block) is base
    assert not hasattr(block.layer, "gate_up_proj")


def test_standard_vlm_switch_family_supported(setup, monkeypatch):
    class SwitchGLU:
        def __call__(self, x, indices):
            pass

    switch = SimpleNamespace(
        SwitchGLU=SwitchGLU,
        QuantizedSwitchLinear=type("Quantized", (), {}),
        SwitchLinear=type("Plain", (), {}),
        _gather_sort=lambda *a: None,
        _scatter_unsort=lambda *a: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.switch_layers",
        ModuleType("mlx_lm.models.switch_layers"),
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.switch_layers", switch)
    # Use the production discovery function, rather than the fixture's stub.
    families = setup[1].discovery()
    assert any(family[1] is SwitchGLU and family[-1] is None for family in families)


def test_install_failure_retains_original(setup, monkeypatch):
    _, model, block, base = setup
    monkeypatch.setattr(
        fusion, "install_enabled", Mock(side_effect=RuntimeError("policy"))
    )
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 0
    assert type(block) is base
    assert not hasattr(block.layer, "gate_up_proj")


def test_install_fires_on_real_serve_import_path(tmp_path):
    script = Path(__file__).with_name("fixtures") / "gemma4_moe_serve_stub.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "RAPID_MLX_GEMMA4_MOE_DECODE_FUSION": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.requires_mlx
@pytest.mark.parametrize("route", ["text", "mllm"])
def test_gpu_greedy_parity(route, monkeypatch):
    """Explicit local snapshot opt-in; never download weights during unit tests."""
    path = os.environ.get("RAPID_MLX_GEMMA4_MOE_PARITY_MODEL")
    if not path or not Path(path).is_dir():
        pytest.skip(
            "set RAPID_MLX_GEMMA4_MOE_PARITY_MODEL to a local Gemma 4 MoE snapshot"
        )
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    monkeypatch.delenv("RAPID_MLX_MOE_DECODE_FUSION", raising=False)
    monkeypatch.delenv("RAPID_MLX_MOE_GATE_UP_FUSION", raising=False)
    if route == "text":
        from mlx_lm import load

        model, tok = load(path)
    else:
        from mlx_vlm import load

        outer, processor = load(path)
        model, tok = outer.language_model, processor.tokenizer
    from mlx_lm.models.cache import make_prompt_cache

    def greedy(prompt):
        cache = make_prompt_cache(model)
        inputs = mx.array([tok.encode(prompt)])
        tokens = []
        for _ in range(32):
            logits = model(inputs, cache=cache)
            logits = getattr(logits, "logits", logits)
            inputs = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
            mx.eval(inputs)
            tokens.append(inputs.item())
        return tokens

    prompts = [
        "Write a story about a lighthouse.",
        "Explain CPU caches.",
        "List facts about Rome.",
    ]
    monkeypatch.setenv("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION", "0")
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 0
    expected = [greedy(prompt) for prompt in prompts]
    assert not fusion.is_installed(model)
    monkeypatch.setenv("RAPID_MLX_GEMMA4_MOE_DECODE_FUSION", "1")
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") > 0
    assert [greedy(prompt) for prompt in prompts] == expected


class _Probe:
    """Stand-in probe array: every construction step returns itself."""

    def astype(self, _):
        return self

    def __getitem__(self, _):
        return self


@pytest.fixture
def layers(monkeypatch):
    class DecoderLayer:
        enable_moe = True
        per_layer_input_gate = None
        layer_scalar = 1

        def __init__(self, moe=True):
            self.enable_moe = moe
            self.pre_feedforward_layernorm = SimpleNamespace(
                weight=SimpleNamespace(dtype="bf16", size=4)
            )

        def input_layernorm(self, x):
            return x

        def self_attn(self, h, mask, cache, **shared):
            return "attended", shared["shared_kv"], shared["offset"]

        def __call__(self, x, mask=None, cache=None, per_layer_input=None, **kw):
            return "stock"

    upstream = ModuleType("mlx_lm.models.gemma4_text")
    upstream.DecoderLayer = DecoderLayer
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    probe = _Probe()
    mx = SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
        compile=lambda f: f,
        arange=lambda *a, **k: probe,
        sin=lambda a: a,
        cos=lambda a: a,
        array_equal=lambda a, b: SimpleNamespace(item=lambda: a == b),
    )
    monkeypatch.setattr(fusion, "_ffn", lambda layer, x, a: ("ffn", a))
    built = [DecoderLayer(), DecoderLayer(), DecoderLayer(moe=False)]
    model = SimpleNamespace(
        named_modules=lambda: [(str(i), m) for i, m in enumerate(built)]
    )
    return mx, model, built, DecoderLayer


def test_compiled_layer_dispatch_and_fallbacks(layers):
    mx, model, (first, second, dense), base = layers
    assert fusion._compile_layers(model, mx) == 2
    assert type(first) is not base and isinstance(first, base)
    assert type(dense) is base
    assert fusion._compile_layers(model, mx) == 0  # idempotent
    x = Tensor((1, 1, 4), "bf16")
    assert first(x, shared_kv="kv", offset=3) == (("ffn", "attended"), "kv", 3)
    for call in (
        lambda: first(Tensor((1, 2, 4), "bf16")),
        lambda: first(Tensor((2, 1, 4), "bf16")),
        lambda: first(Tensor((1, 1, 4), "fp16")),
        lambda: first(x, per_layer_input=object()),
        lambda: first(x, shared_cache=object()),
    ):
        assert call() == "stock"


def test_compiled_layer_probe_mismatch_restores_all(layers):
    mx, model, built, base = layers
    calls = []

    def compile_(f):
        calls.append(f)
        return f if len(calls) == 1 else (lambda x, a: "drift")

    mx.compile = compile_
    assert fusion._compile_layers(model, mx) == 0
    for layer in built:
        assert type(layer) is base
        assert "_rapid_gemma4_compiled_ffn" not in layer.__dict__


def test_compiled_layer_declines_unsupported_dtype(layers):
    mx, model, built, base = layers
    built[1].pre_feedforward_layernorm.weight.dtype = "fp32"
    assert fusion._compile_layers(model, mx) == 0
    assert all(type(layer) is base for layer in built)


def test_ffn_composes_gemma4_post_attention_half():
    layer = SimpleNamespace(
        post_attention_layernorm=lambda a: a * 2,
        pre_feedforward_layernorm=lambda h: h + 1,
        mlp=lambda h: h * 3,
        post_feedforward_layernorm_1=lambda d: d + 5,
        router=lambda h: ("indices", h - 1),
        pre_feedforward_layernorm_2=lambda h: h * 7,
        experts=lambda h, indices, weights: h + weights,
        post_feedforward_layernorm_2=lambda r: r * 11,
        post_feedforward_layernorm=lambda b: b + 13,
        layer_scalar=2,
    )
    real = importlib.import_module(fusion.__name__)._ffn
    # h = 1 + 2*2 = 5; dense = (5+1)*3+5 = 23; routed = 5*7 + 4 = 39
    # out = (5 + (23 + 39*11) + 13) * 2
    assert real(layer, 1, 2) == (5 + (23 + 39 * 11) + 13) * 2


def test_install_declines_without_metal_gpu(setup):
    mx, model, block, base = setup
    mx.metal.is_available = lambda: False
    assert fusion.install_gemma4_moe_decode_fusion(model, "gemma4") == 0
    assert type(block) is base


@pytest.mark.parametrize("model_type", ["gemma4", "qwen3_5"])
def test_text_engine_installs_after_dense_fusion(monkeypatch, model_type):
    pytest.importorskip("mlx.core")
    import asyncio

    from rapid_mlx import engine_core
    from rapid_mlx.engine import batched
    from rapid_mlx.runtime import cache
    from rapid_mlx.utils import tokenizer

    class BoundaryError(RuntimeError):
        pass

    model = SimpleNamespace(model_type=model_type)
    order = []

    async def start(**kwargs):
        raise BoundaryError()

    monkeypatch.setattr(engine_core, "_init_mlx_step_thread", lambda: None)
    monkeypatch.setattr(
        tokenizer, "load_model_with_fallback", lambda *a, **kw: (model, object(), "/m")
    )
    monkeypatch.setattr(cache, "pin_prefix_cache_identity", lambda *a, **kw: None)
    for name in (
        "rapid_mlx.moe_fusion.fuse_gate_up",
        "rapid_mlx.qwen35_moe_router.install_qwen35_moe_router",
        "rapid_mlx.patches.qwen35_affine4.install_qwen35_affine4",
        "rapid_mlx.patches.moe_decode_fusion.install_moe_decode_fusion",
        "rapid_mlx.kernels.lane_matmul.install_lane_matmul",
        "rapid_mlx.gdn_in_proj_fusion.fuse_gdn_in_proj",
        "rapid_mlx.qwen35_fused_gdn_decode.install_qwen35_fused_gdn_decode",
    ):
        monkeypatch.setattr(name, lambda *a, **kw: None)
    monkeypatch.setattr(
        "rapid_mlx.dense_decode_fusion.install_dense_decode_fusion",
        lambda *a, **kw: order.append("dense"),
    )
    monkeypatch.setattr(
        fusion,
        "install_gemma4_moe_decode_fusion",
        lambda m, t: order.append(("gemma4", m is model, t)),
    )
    monkeypatch.setattr(
        "rapid_mlx.execution_reference.freeze_load_receipt",
        lambda *a, **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        engine_core,
        "AsyncEngineCore",
        lambda **kw: SimpleNamespace(
            engine=SimpleNamespace(scheduler=SimpleNamespace(), start=start)
        ),
    )
    engine = object.__new__(batched.BatchedEngine)
    engine._model_name = "gemma"
    engine._trust_remote_code = False
    engine._scheduler_config = SimpleNamespace(spec_decode="none")
    engine._model_load_executor = None
    for name in (
        "_profile_name",
        "_stream_interval",
        "_gpu_memory_utilization",
        "_tool_logits_processor_factory",
        "_force_hybrid",
        "_no_hybrid",
        "_force_spec_decode",
        "_no_spec_decode",
    ):
        setattr(engine, name, None)
    engine._install_resolved_metal_budget = lambda: None
    engine._run_metal_admission_preflight = lambda: None
    engine._bind_prompt_host_cache = lambda engine: None
    try:
        with pytest.raises(BoundaryError):
            asyncio.run(engine._start_llm())
    finally:
        if engine._model_load_executor is not None:
            engine._model_load_executor.shutdown(wait=True)
    expected = ["dense"] + (
        [("gemma4", True, "gemma4")] if model_type == "gemma4" else []
    )
    assert order == expected
