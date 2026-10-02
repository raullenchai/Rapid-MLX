"""Row-invariant lane matmul (rapid_mlx/kernels/lane_matmul).

Real-Metal tests run the backend this GPU selects (``mpp`` on M5, ``simd`` on
M1-M4) and the ``simd`` kernels on any Apple GPU: every row of a multi-row call
must be bitwise equal to the same row computed alone, including the partial
last 32-row blocks (33-48 rows) where TensorFold 0.5.0 fixed an out-of-bounds
read in its own kernel.  The rest covers installation, the numerical-law
receipt, the environment switch and the engine wiring.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytestmark = [pytest.mark.requires_mlx]

from rapid_mlx.kernels import lane_matmul as lane  # noqa: E402
from rapid_mlx.kernels.lane_matmul import installer as inst  # noqa: E402
from rapid_mlx.kernels.lane_matmul import matmul as lm  # noqa: E402
from rapid_mlx.kernels.lane_matmul import simd  # noqa: E402

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
ROWS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 40, 48)


def _quantized(k, n, bits, gs, seed=0):
    w = (mx.random.normal((n, k), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    module = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
    module.weight, module.scales, module.biases = mx.quantize(
        w, group_size=gs, bits=bits
    )
    return module


@pytest.fixture(autouse=True)
def _reset():
    simd.mma_one_row.clear()
    simd.bits_fallback.clear()
    yield
    simd.mma_one_row.clear()
    simd.bits_fallback.clear()
    lm.force_backend(None)


def _assert_row_invariant(module):
    lw = lm.prepare(module)
    k = lw.k
    base = np.asarray(
        mx.random.normal((max(ROWS), k), key=mx.random.key(11)).astype(mx.float32)
    )
    alone = mx.concatenate(
        [
            lm.lane_matmul(mx.array(base[i : i + 1]).astype(mx.bfloat16), lw)
            for i in range(max(ROWS))
        ]
    )
    for rows in ROWS:
        fresh = mx.array(np.ascontiguousarray(base[:rows])).astype(
            mx.bfloat16
        )  # exactly `rows`
        together = lm.lane_matmul(fresh, lw)
        assert mx.array_equal(together, alone[:rows]).item(), (
            lw.backend,
            lw.format,
            rows,
        )
    deq = mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
    ).astype(mx.float32)
    ref = mx.array(base[:8]) @ deq.T
    stock = mx.quantized_matmul(
        mx.array(base[:8]).astype(mx.bfloat16),
        module.weight,
        module.scales,
        module.biases,
        transpose=True,
        group_size=module.group_size,
        bits=module.bits,
    ).astype(mx.float32)
    lane_err = float(mx.max(mx.abs(alone[:8].astype(mx.float32) - ref)).item())
    stock_err = float(mx.max(mx.abs(stock - ref)).item())
    assert lane_err <= 2 * stock_err + float(mx.max(mx.abs(ref)).item()) * 2.0**-7
    return lw


FORMATS = [(4, 64), (4, 32), (5, 64), (6, 64), (8, 64), (3, 128), (2, 32)]


@metal
@pytest.mark.parametrize("bits,gs", FORMATS)
def test_simd_backend_rows_equal_the_row_alone(bits, gs):
    lm.force_backend("simd")
    module = _quantized(512, 256, bits, gs, seed=bits)
    assert (
        simd.check(module["weight"], module["scales"], module["biases"], gs, bits)
        is not None
    )
    assert _assert_row_invariant(module).backend == "simd"


@metal
@pytest.mark.parametrize("bits,gs", [(4, 64), (5, 64), (8, 32)])
def test_this_gpus_backend_rows_equal_the_row_alone(bits, gs):
    assert lane.backend() in ("mpp", "simd")
    _assert_row_invariant(_quantized(512, 256, bits, gs, seed=bits))


def test_off_is_the_default_and_bad_values_fail(monkeypatch):
    monkeypatch.delenv(lane.ENABLE_ENV, raising=False)
    assert lane.mode_from_env() == "off"
    assert lane.install_lane_matmul(object()) is None
    for value in ("0", "off", "false"):
        monkeypatch.setenv(lane.ENABLE_ENV, value)
        assert lane.mode_from_env() == "off"
    monkeypatch.setenv(lane.ENABLE_ENV, "Crossover")
    assert lane.mode_from_env() == "crossover"
    monkeypatch.setenv(lane.ENABLE_ENV, "fast")
    with pytest.raises(ValueError, match=lane.ENABLE_ENV):
        lane.mode_from_env()
    with pytest.raises(ValueError, match="mode"):
        lane.install_lane_matmul(object(), mode="sometimes")


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _quantized(256, 64, 4, 64, seed=1)
        self.k_proj = _quantized(256, 32, 4, 64, seed=2)
        self.v_proj = _quantized(256, 32, 4, 64, seed=3)
        self.o_proj = _quantized(64, 256, 4, 64, seed=4)

    def __call__(self, x):
        return self.o_proj(
            self.q_proj(x) + mx.concatenate([self.k_proj(x), self.v_proj(x)], -1)
        )


class _SwitchGLU(nn.Module):
    pass


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.experts = _SwitchGLU()


@metal
def test_exact_mode_installs_groups_and_keeps_rows_equal_to_decode(monkeypatch):
    model = _Attention()
    weights = {name: model[name]["weight"] for name in ("q_proj", "k_proj", "v_proj")}
    receipt = lane.install_lane_matmul(model, mode="exact")
    assert receipt["mode"] == "exact" and receipt["backend"] == lane.backend()
    assert receipt["law_id"].startswith(inst.LAW_IDS[lane.backend()])
    assert receipt["law_id"].endswith("+grouped")
    assert receipt["covered"] == {"affine-q4-g64": 4}
    assert receipt["groups"] == {"affine-q4-g64x3": 1}
    for name, before in weights.items():  # members are views of the stack
        assert mx.array_equal(model[name]["weight"], before).item()
    x = mx.random.normal((1, 6, 256), key=mx.random.key(5)).astype(mx.bfloat16)
    together = model(x)
    alone = mx.concatenate([model(x[:, i : i + 1]) for i in range(6)], axis=1)
    assert mx.array_equal(together, alone).item()
    counts = lane.stats()
    assert counts["group_launches"] >= 7 and counts["group_reuses"] >= 14
    assert inst.uninstall(model) == 4
    assert type(model.q_proj) is nn.QuantizedLinear


@metal
def test_crossover_keeps_short_calls_on_stock():
    model = _Attention()
    x = mx.random.normal((1, 4, 256), key=mx.random.key(6)).astype(mx.bfloat16)
    before = model(x)
    receipt = lane.install_lane_matmul(model, mode="crossover")
    assert receipt["min_rows"] == lane.CROSSOVER
    # q/k/v below the crossover run as one stacked stock launch, proven
    # bitwise equal to three separate stock calls for 1-7 rows on this GPU.
    assert receipt["stock_stacked"] == {"groups": 1, "unproven": 0}
    group = inst._group(model.q_proj)
    assert group.stock_stacked
    launches = []
    real = inst._stock_matmul
    inst._stock_matmul = lambda g, v: launches.append(v.shape) or real(g, v)
    try:
        assert mx.array_equal(model(x), before).item()  # 4 rows < 8: stock arithmetic
    finally:
        inst._stock_matmul = real
    assert len(launches) == 1  # one stacked stock launch served q, k and v
    assert group.stock_last is None  # released once all three took their columns
    lane.uninstall(model)


@metal
def test_lane_group_results_are_released_after_every_member_took_them():
    model = _Attention()
    lane.install_lane_matmul(model, mode="exact")
    group = inst._group(model.q_proj)
    x = mx.random.normal((1, 5, 256), key=mx.random.key(9)).astype(mx.bfloat16)
    mx.eval(model(x))
    assert group.last is None and group.size == 3
    mx.eval(model.q_proj(x))  # a lone member call keeps one result until reused
    assert group.last is not None and group.last[2] == 1
    lane.uninstall(model)


def test_an_install_failure_leaves_the_model_on_stock(monkeypatch, caplog):
    monkeypatch.setattr(lane, "available", lambda: True)

    def boom(model, **_kw):
        inst.install(model, min_rows_by_format={"q4": 1}, groups=())
        raise RuntimeError("metal compile failed")

    monkeypatch.setattr(lane, "install", boom)
    monkeypatch.setattr(inst, "_check_simd_twins", lambda model: {})
    model = _Attention()
    assert lane.install_lane_matmul(model, mode="exact") is None
    assert type(model.q_proj) is nn.QuantizedLinear
    assert "serving with stock kernels" in caplog.text


@metal
def test_a_stack_that_changes_stock_bits_keeps_separate_calls(monkeypatch):
    real = inst._stock_matmul
    monkeypatch.setattr(inst, "_stock_matmul", lambda group, x: real(group, x) + 1)
    model = _Attention()
    x = mx.random.normal((1, 3, 256), key=mx.random.key(7)).astype(mx.bfloat16)
    before = model(x)
    receipt = lane.install_lane_matmul(model, mode="crossover")
    assert receipt["stock_stacked"] == {"groups": 0, "unproven": 1}
    assert not inst._group(model.q_proj).stock_stacked
    assert mx.array_equal(model(x), before).item()
    lane.uninstall(model)


def test_exact_mode_has_no_stock_rows_to_stack(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "check", lambda *a, **k: True)
    receipt = inst.install(_Attention(), min_rows_by_format={"q4": 1})
    assert "stock_stacked" not in receipt


def test_moe_models_and_uncovered_models_are_skipped(monkeypatch):
    monkeypatch.setattr(lane, "available", lambda: True)
    assert lane.is_moe(_MoE()) and not lane.is_moe(_Attention())
    assert lane.install_lane_matmul(_MoE(), mode="crossover") is None

    class _Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(100, 64)  # K not a multiple of 64: refused

    model = _Dense()
    assert lane.install_lane_matmul(model, mode="exact") is None
    assert type(model.proj) is nn.Linear


def test_no_backend_is_reported_not_installed(monkeypatch, caplog):
    monkeypatch.setattr(lane, "available", lambda: False)
    assert lane.install_lane_matmul(_Attention(), mode="exact") is None
    assert "no lane backend" in caplog.text


def _fake_qmm(differ):
    def qmm(x2, weight, scales, biases, group_size, bits, *, kind=None):
        value = 1.0 if (differ and kind == "scalar") else 0.0
        return mx.full(
            (int(x2.shape[0]), int(weight.shape[0])), value, dtype=mx.bfloat16
        )

    return qmm


def test_simd_install_law_twins_and_backend_switch(monkeypatch):
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=False))
    model = _Attention()
    for name in ("simd", "mpp", "simd"):
        monkeypatch.setattr(lm, "backend", lambda name=name: name)
        monkeypatch.setattr(inst, "backend", lambda name=name: name)
        receipt = inst.install(model, min_rows_by_format={"q4": 1})
        assert receipt["backend"] == name
        assert receipt["law_id"] == f"{inst.LAW_IDS[name]}+stock-below[q4:1]+grouped"
        assert all(
            inst._prepared(model[a]).backend == name
            for a in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        assert inst._group(model.q_proj).lw.backend == name
        if name == "simd":
            assert inst._group(model.q_proj).lw.scale_bias is None  # no scale/bias copy
            assert receipt["simd_twins"] == {"shapes": 4, "rerouted": 0}
        else:
            assert "simd_twins" not in receipt
    inst.uninstall(model)


def test_a_high_bit_affine_fallback_joins_the_law(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=True))

    class _Up(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = _quantized(256, 64, 6, 64, seed=9)
            self.down = _quantized(256, 64, 4, 64, seed=8)

    model = _Up()
    first = inst.install(model, min_rows_by_format={"q4": 1, "q6": 1})
    assert first["simd_twins"] == {
        "shapes": 2,
        "rerouted": 2,
        "affine": ["64x256q6g64"],
        "mma": ["64x256q4g64"],
    }
    assert "+simd-affine[" in first["law_id"]
    assert simd.path(6, 64, 64, 256, mx.bfloat16) == "affine"
    again = inst.install(model, min_rows_by_format={"q4": 1, "q6": 1})
    assert (
        again["law_id"] == first["law_id"]
        and again["simd_twins"] == first["simd_twins"]
    )
    assert not simd._scalar_kind(
        1, 64, 256, 64
    )  # 4-bit one-row calls use the matrix kernel
    inst.uninstall(model)


def test_formats_without_a_threshold_stay_stock(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "check", lambda *a, **k: True)
    model = _Attention()
    model.head = nn.Linear(256, 128, bias=False)
    model.head.weight = model.head.weight.astype(mx.bfloat16)
    receipt = inst.install(
        model, min_rows_by_format={"q4": 8, "bf16": 16}, max_rows=16, groups=()
    )
    assert receipt["refused"] == {"unquantized weights need the M5 tensor units": 1}
    assert receipt["law_id"] == "lane-simd-v1+stock-below[bf16:16,q4:8]+rows-le-16"
    assert model.q_proj._lane_min_rows == 8
    with pytest.raises(ValueError, match="max_rows"):
        inst.install(model, min_rows_by_format={"q4": 1}, max_rows=0)
    inst.uninstall(model)
    receipt = inst.install(_Attention(), min_rows_by_format={"q8": 8})
    assert receipt["covered"] == {} and receipt["refused"] == {
        "no threshold for this format": 4
    }


@pytest.mark.parametrize(
    "bits,gs,n,k,dtype,expected",
    [
        (4, 64, 64, 256, "bfloat16", "simd4"),
        (4, 32, 64, 256, "bfloat16", "simd4"),
        (4, 128, 64, 256, "bfloat16", "affine"),
        (4, 64, 60, 256, "bfloat16", "affine"),
        (4, 64, 64, 256, "float16", "affine"),
        (6, 64, 64, 256, "bfloat16", "simd_bits"),
        (8, 32, 64, 256, "bfloat16", "affine"),
    ],
)
def test_kernel_family_is_fixed_per_weight(bits, gs, n, k, dtype, expected):
    assert simd.path(bits, gs, n, k, getattr(mx, dtype)) == expected


def test_admission_refuses_layouts_the_kernels_cannot_read():
    base = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "n": 64,
        "k": 256,
        "weight_dtype": mx.uint32,
        "scales_dtype": mx.bfloat16,
    }
    simd.check_geometry(**base, biases_dtype=mx.bfloat16)
    for extra, message in (
        ({"biases_dtype": mx.float16}, "share one dtype"),
        ({"weight_ndim": 3}, "rank 2"),
        ({"mode": "mxfp4"}, "affine"),
        ({"group_size": 16}, "groups of 16"),
        ({"weight_dtype": mx.uint8}, "uint32"),
        ({"scales_dtype": mx.int32}, "scales"),
        ({"k": 250}, "multiple"),
        ({"n": 0}, "positive"),
    ):
        with pytest.raises(simd.SimdUnsupportedError, match=message):
            simd.check_geometry(**{**base, **extra})
    module = _quantized(256, 64, 4, 64)
    module.biases = module.biases.astype(mx.float16)
    with pytest.raises(lane.LaneUnsupportedError, match="share one dtype"):
        lm.prepare(module, "simd")
    with pytest.raises(lane.LaneUnsupportedError, match="M5 tensor units"):
        lm.prepare(nn.Linear(128, 64), "simd")
    with pytest.raises(simd.SimdUnsupportedError, match="bf16"):
        simd.qmm(
            mx.zeros((1, 256), dtype=mx.float16),
            module["weight"],
            module["scales"],
            module["scales"],
            64,
            4,
        )
    with pytest.raises(ValueError, match="backend"):
        lm.force_backend("cuda")


def test_thread_limit_probing_steps_down_on_m1_m2(monkeypatch):
    """M1/M2 pipelines can take fewer threads than a launch asks (TensorFold's threads.fit)."""
    monkeypatch.setattr(simd, "_probing", lambda: True)
    simd._fitted.clear()
    tried = []

    def launch(size):
        tried.append(size)
        if size > 256:
            raise ValueError("maximum allowed threads per threadgroup (256)")
        return size

    assert simd._fit(("probe-test",), (512, 256, 128), launch) == 256
    assert tried == [512, 256] and simd._fitted[("probe-test",)] == 256
    assert (
        simd._fit(("probe-test",), (512, 256, 128), launch) == 256
    )  # cached, no re-probe
    assert tried == [512, 256, 256]

    def never(size):
        raise ValueError("maximum allowed threads per threadgroup (32)")

    with pytest.raises(simd.SimdUnsupportedError, match="allows 32 threads"):
        simd._fit(("probe-never",), (64,), never)

    def traced(size):
        if size > 256:
            raise ValueError("[compile] function transformations cannot launch")
        return size

    assert simd._fit(("probe-traced",), (512, 128), traced) == 128
    with pytest.raises(ValueError, match="unrelated"):
        simd._fit(
            ("probe-other",),
            (64,),
            lambda size: (_ for _ in ()).throw(ValueError("unrelated")),
        )
    monkeypatch.setattr(simd, "_probing", lambda: False)
    assert simd._fit(("probe-fast",), (512, 256), lambda size: size) == 512
    simd._fitted.clear()


def test_the_law_joins_the_prefix_cache_identity():
    from rapid_mlx.runtime.cache import pin_prefix_cache_identity

    engine = SimpleNamespace()
    stock = pin_prefix_cache_identity(
        engine, raw_model_name="m", checkpoint_source="/nonexistent", kv_dtype="bf16"
    )
    lawful = pin_prefix_cache_identity(
        engine,
        raw_model_name="m",
        checkpoint_source="/nonexistent",
        kv_dtype="bf16",
        numerical_law="lane-simd-v1",
    )
    assert lawful == stock + "\0law=lane-simd-v1"
    assert engine._rapid_mlx_prefix_cache_identity == lawful


class _SentinelError(Exception):
    pass


def test_start_llm_installs_lane_before_gdn_fusion_and_binds_the_law(monkeypatch):
    from rapid_mlx import gdn_in_proj_fusion
    from rapid_mlx.engine import batched
    from rapid_mlx.runtime import cache
    from rapid_mlx.scheduler import SchedulerConfig
    from rapid_mlx.utils import tokenizer as tokenizer_mod

    engine = object.__new__(batched.BatchedEngine)
    engine._model_name = "local/qwen3.8-27b-4bit"
    engine._trust_remote_code = False
    engine._scheduler_config = SchedulerConfig()
    engine._gpu_memory_utilization = 0.90
    engine._is_mllm = False
    engine._model = engine._tokenizer = engine._engine = None
    engine._loaded = engine._engine_started = False
    model = object()
    order = []

    def fake_load(name, tokenizer_config=None, *, return_source=False, **_kw):
        return model, SimpleNamespace(eos_token_id=0), "/fake/snapshot"

    def fake_lane(target):
        order.append(("lane", target))
        return {"law_id": "lane-matmul-v1+test"}

    def fake_fuse(target):
        order.append(("fuse", target))
        return 0

    def fake_pin(engine_, **kwargs):
        order.append(("pin", kwargs["numerical_law"]))
        raise _SentinelError

    monkeypatch.setattr(tokenizer_mod, "load_model_with_fallback", fake_load)
    monkeypatch.setattr(lane, "install_lane_matmul", fake_lane)
    monkeypatch.setattr(gdn_in_proj_fusion, "fuse_gdn_in_proj", fake_fuse)
    monkeypatch.setattr(cache, "pin_prefix_cache_identity", fake_pin)
    try:
        with pytest.raises(_SentinelError):
            asyncio.run(engine._start_llm())
    finally:
        engine._model_load_executor.shutdown(wait=True)
    assert order == [("lane", model), ("fuse", model), ("pin", "lane-matmul-v1+test")]
    assert engine._lane_matmul_receipt == {"law_id": "lane-matmul-v1+test"}


def test_simd_refuses_non_bf16_checkpoints():
    module = _quantized(256, 64, 4, 64)
    module.scales = module.scales.astype(mx.float16)
    module.biases = module.biases.astype(mx.float16)
    with pytest.raises(lane.LaneUnsupportedError, match="bf16 checkpoint"):
        lm.prepare(module, "simd")
    assert lm.prepare(module, "mpp").backend == "mpp"  # the M5 kernels take fp16


def test_a_narrower_reinstall_restores_formats_it_no_longer_covers(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "check", lambda *a, **k: True)
    model = _Attention()
    inst.install(model, min_rows_by_format={"q4": 1})
    assert type(model.q_proj) is inst.LaneQuantizedLinear
    receipt = inst.install(model, min_rows_by_format={"q8": 8})
    assert receipt["covered"] == {} and receipt["groups"] == {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert type(model[name]) is nn.QuantizedLinear
        assert inst._prepared(model[name]) is None


@metal
def test_the_stock_stack_probe_covers_fp16_activations(monkeypatch):
    real = inst._stock_matmul
    calls = []

    def spy(group, x):
        calls.append(x.dtype)
        return real(group, x)

    monkeypatch.setattr(inst, "_stock_matmul", spy)
    lane.install_lane_matmul(_Attention(), mode="crossover")
    assert mx.bfloat16 in calls and mx.float16 in calls


def test_a_config_that_declares_experts_is_moe_whatever_its_module_names():
    from types import SimpleNamespace

    model = _Attention()
    assert not lane.is_moe(model)
    model.args = SimpleNamespace(num_experts=0, text_config={"n_routed_experts": 64})
    assert lane.is_moe(model)
    model.args = SimpleNamespace(num_local_experts=8)
    assert lane.is_moe(model)
    model.args = SimpleNamespace(num_experts=True)  # a bool is not an expert count
    assert not lane.is_moe(model)
    wrapper = SimpleNamespace(
        named_modules=model.named_modules,
        language_model=SimpleNamespace(args={"moe_num_experts": 4}),
    )
    assert lane.is_moe(wrapper)


class _BiasedAttention(nn.Module):
    """Qwen2-style attention: q/k/v carry an additive bias."""

    def __init__(self):
        super().__init__()
        for name, n, seed in (
            ("q_proj", 64, 21),
            ("k_proj", 32, 22),
            ("v_proj", 32, 23),
        ):
            m = _quantized(256, n, 4, 64, seed=seed)
            m.bias = mx.random.normal((n,), key=mx.random.key(seed + 100)).astype(
                mx.bfloat16
            )
            setattr(self, name, m)

    def __call__(self, x):
        return mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], -1)


class _TwoBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = _Attention()
        self.b = _Attention()


@metal
def test_biased_groups_add_each_members_bias_and_stay_row_invariant():
    model = _BiasedAttention()
    receipt = lane.install_lane_matmul(model, mode="exact")
    assert receipt["groups"] == {"affine-q4-g64x3": 1}
    x = mx.random.normal((1, 6, 256), key=mx.random.key(12)).astype(mx.bfloat16)
    together = model(x)
    alone = mx.concatenate([model(x[:, i : i + 1]) for i in range(6)], axis=1)
    assert mx.array_equal(together, alone).item()
    lane.uninstall(model)


@metal
def test_calls_outside_the_lane_window_or_format_fall_back_to_stock():
    model = _Attention()
    lane.install_lane_matmul(model, mode="exact")  # 1..32 rows take the lane
    before = dict(lane.stats())
    wide = mx.random.normal((40, 64), key=mx.random.key(13)).astype(mx.bfloat16)
    y = model.o_proj(wide)  # 40 rows > max_rows: stock
    stock = nn.QuantizedLinear.__call__(model.o_proj, wide)
    assert mx.array_equal(y, stock).item()
    f32 = mx.random.normal((3, 64), key=mx.random.key(14))  # fp32 activations: refused
    assert mx.array_equal(
        model.o_proj(f32), nn.QuantizedLinear.__call__(model.o_proj, f32)
    ).item()
    after = lane.stats()
    assert after["stock_above_max_rows"] == before.get("stock_above_max_rows", 0) + 1
    assert after["stock_unsupported"] == before.get("stock_unsupported", 0) + 1
    inst._LIVE[0] = False  # a device without a backend: every call stays stock
    try:
        model.o_proj(mx.zeros((3, 64), dtype=mx.bfloat16))
        assert lane.stats()["stock_disabled"] == before.get("stock_disabled", 0) + 1
    finally:
        inst._LIVE[0] = True
    lane.uninstall(model)


@metal
def test_the_stock_stack_probe_runs_once_per_shape():
    real = inst._probe_stock_stack
    probes = []

    def spy(members, group, rows_below, seen):
        result = real(members, group, rows_below, seen)
        probes.append(len(seen))
        return result

    inst._probe_stock_stack = spy
    try:
        receipt = lane.install_lane_matmul(_TwoBlocks(), mode="crossover")
    finally:
        inst._probe_stock_stack = real
    assert receipt["stock_stacked"] == {"groups": 2, "unproven": 0}
    assert probes == [1, 1]  # the second, same-shaped group reused the verdict


def test_unquantized_siblings_stack_and_run_stock_as_one_matmul(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "mpp")
    monkeypatch.setattr(inst, "backend", lambda: "mpp")
    monkeypatch.setattr(inst, "available", lambda: True)

    class _Dense(nn.Module):
        def __init__(self):
            super().__init__()
            for name, n in (("q_proj", 64), ("k_proj", 64), ("v_proj", 64)):
                layer = nn.Linear(128, n, bias=False)
                layer.weight = layer.weight.astype(mx.bfloat16)
                setattr(self, name, layer)

    model = _Dense()
    x = mx.random.normal((2, 128), key=mx.random.key(15)).astype(mx.bfloat16)
    before = [model[name](x) for name in ("q_proj", "k_proj", "v_proj")]
    receipt = inst.install(model, min_rows_by_format={"bf16": 16})
    assert receipt["groups"] == {"unquantizedx3": 1}
    assert type(model.q_proj) is inst.LaneLinear
    group = inst._group(model.q_proj)
    assert "scales" not in group.stack
    stacked = inst._stock_matmul(group, x)
    assert mx.array_equal(stacked, mx.concatenate(before, axis=-1)).item()
    after = [
        model[name](x) for name in ("q_proj", "k_proj", "v_proj")
    ]  # 2 rows < 16: stock
    assert all(mx.array_equal(a, b).item() for a, b in zip(before, after))
    inst.uninstall(model)


def test_mixed_format_siblings_only_stack_within_a_format(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "check", lambda *a, **k: True)
    model = _Attention()
    model.v_proj = _quantized(256, 32, 8, 64, seed=30)  # q8 beside q4 q/k
    receipt = inst.install(model, min_rows_by_format={"q4": 1, "q8": 1})
    assert receipt["groups"] == {"affine-q4-g64x2": 1}
    assert inst._group(model.v_proj) is None
    inst.uninstall(model)


def test_law_ids_buckets_and_format_classes():
    assert inst.law_id(1, "simd") == "lane-simd-v1"
    assert inst.law_id(8, "mpp") == "lane-matmul-v1+stock-below-8"
    assert [inst._bucket(r) for r in (1, 4, 8, 16, 33, 128)] == [
        "1-3",
        "4-7",
        "8-15",
        "16-32",
        "33-128",
        "33-128",
    ]
    assert inst.format_class(nn.RMSNorm(8)) is None
    assert inst.format_class(nn.Linear(8, 8)) is None  # fp32 weights: no class
