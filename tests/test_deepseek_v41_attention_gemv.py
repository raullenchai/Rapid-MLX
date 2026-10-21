"""Single-row stock parity, lazy qualification and attention projection wiring."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from rapid_mlx.models.deepseek_v41_native import attention_gemv as gemv
from rapid_mlx.models.deepseek_v41_native.attention import GroupedOutputLinear

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# The shipped target layouts (N, K, groups); wo_a always uses FP32 input.
LAYOUTS = [
    ("wq_a", 1280, 5120, 1),
    ("wq_b", 32768, 1280, 1),
    ("wkv", 512, 5120, 1),
    ("wo_a", 1024, 4096, 8),
    ("wo_b", 5120, 8192, 1),
]


def projection(bits=8, n=19, k=320, groups=1, dtype=None):
    dtype = dtype or mx.bfloat16
    layer = (
        GroupedOutputLinear(k, n, groups) if groups > 1 else nn.Linear(k, n, bias=False)
    )
    layer.weight = layer.weight.astype(dtype)
    return layer.to_quantized(group_size=64, bits=bits)


def inputs(rows=2, k=320, groups=1, dtype=None):
    shape = (1, rows, groups, 1, k) if groups > 1 else (1, rows, k)
    return mx.random.normal(shape).astype(dtype or mx.float32)


def equal(a, b):
    mx.eval(a, b)
    return bool(mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item())


def per_row_stock(layer, x):
    """Match qualification's single-row calls, retaining grouped row dimensions."""
    row_shape = x.shape[-3:] if layer.weight.ndim == 3 else x.shape[-1:]
    rows = x.reshape(-1, *row_shape)
    return mx.concatenate(
        [layer(rows[r : r + 1]) for r in range(rows.shape[0])], axis=0
    ).reshape(*x.shape[:-1], layer.scales.shape[-2])


def test_layout_rejections_and_cpu_fallback(monkeypatch):
    layer, x = projection(), inputs()
    assert gemv._layout(layer, x) == (2, 1, 19, 320)
    assert gemv._layout(SimpleNamespace(), x) is None
    biased = nn.QuantizedLinear(320, 19, bias=True, group_size=64, bits=8)
    assert gemv._layout(biased, x) is None
    for key, value in [
        ("bits", 2),
        ("group_size", 32),
        ("mode", "mxfp8"),
        ("weight", mx.zeros((19, 80), dtype=mx.float32)),
        ("scales", mx.zeros((19, 4))),
        ("biases", mx.zeros((19, 5), dtype=mx.float16)),
    ]:
        layer = projection()
        setattr(layer, key, value)
        assert gemv._layout(layer, x) is None
    layer = projection()
    for bad in [
        mx.zeros((320,)),
        inputs(9),
        inputs(k=256),
        mx.zeros((1, 320), dtype=mx.int32),
        inputs(0),
    ]:
        assert gemv._layout(layer, bad) is None
    assert gemv._layout(projection(k=128), inputs(k=128)) is None
    assert gemv._layout(projection(groups=3), x) is None
    assert gemv._layout(projection(groups=3), mx.zeros((3, 1, 320))) is None
    layer.weight = layer.weight[:, :-1]
    assert gemv._layout(layer, x) is None
    with pytest.raises(ValueError, match="unsupported"):
        gemv._launch(layer, x)
    layer = projection()
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not gemv.supports(layer, x)
        assert equal(gemv.project(layer, x), layer(x))
    finally:
        mx.set_default_device(previous)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    assert gemv.supports(layer, x)
    monkeypatch.setenv("RAPID_MLX_DSV41_FUSED", "0")
    assert not gemv.supports(layer, x)
    monkeypatch.delenv("RAPID_MLX_DSV41_FUSED")
    monkeypatch.setattr(gemv, "ENABLED", False)
    assert not gemv.supports(layer, x)


@pytest.mark.parametrize("groups", [1, 3])
def test_factory_and_dispatch_with_cpu_standin(monkeypatch, groups):
    seen, built = {}, {}

    def call(**kw):
        seen.update(kw)
        x, w, s, b = kw["inputs"]
        return [mx.quantized_matmul(x, w, s, b, group_size=64, bits=8)]

    def factory(**kw):
        built.update(kw)
        return call

    gemv._kernel.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", factory)
    layer, x = projection(groups=groups), inputs(groups=groups)
    gemv._launch(layer, x)
    assert seen["inputs"][0] is x
    assert seen["grid"] == (32, 8, groups)
    assert seen["threadgroup"] == (32, 4, 1)
    assert dict(seen["template"])["ROWS"] == 2
    assert dict(seen["template"])["VPT"] == 4
    assert built["ensure_row_contiguous"] is True
    assert "uint2" in built["source"]
    gemv._kernel.cache_clear()


def test_qualification_caches_by_actual_shape_dtype_and_weight(monkeypatch, caplog):
    monkeypatch.setattr(gemv, "supports", lambda *a: True)
    launches = []

    def launch(m, x):
        launches.append(x.shape)
        return per_row_stock(m, x)

    monkeypatch.setattr(gemv, "_launch", launch)
    layer = projection()
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            x = inputs()
            assert equal(
                gemv.project(layer, x, "layer 0 wq_a"), per_row_stock(layer, x)
            )
    # Two qualification probes, then two hot launches.
    assert len(launches) == 4
    assert "coverage" in caplog.text and "exact=True" in caplog.text
    assert gemv._CHECK_ATTR not in layer.parameters()
    for x in (inputs(3), inputs(dtype=mx.float16), mx.zeros((2, 320))):
        before = len(launches)
        assert equal(gemv.project(layer, x), per_row_stock(layer, x))
        assert len(launches) == before + 3
    for name in ("weight", "scales", "biases"):
        old_ref = getattr(layer, gemv._CHECK_ATTR)[0][
            ("weight", "scales", "biases").index(name)
        ]
        setattr(layer, name, getattr(projection(), name))
        assert old_ref() is None
        before = len(launches)
        gemv.project(layer, inputs())
        assert len(launches) == before + 3
    layer.group_size = 32
    # Deliberately replace launch with a matching stand-in: settings invalidate
    # the verdict independently of the array identity and shape.
    monkeypatch.setattr(gemv, "_launch", per_row_stock)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        mx.ones((19, 320)), group_size=32, bits=8
    )
    assert gemv._qualified(layer, inputs(), "settings changed")


@pytest.mark.parametrize("groups", [1, 3])
def test_qualification_uses_single_row_stock_for_every_lane(monkeypatch, groups):
    layer = projection(groups=groups)
    calls = []

    class Stock:
        weight, scales, biases = layer.weight, layer.scales, layer.biases
        bits, group_size, mode = layer.bits, layer.group_size, "affine"

        def __call__(self, x):
            calls.append(x.shape)
            # Deliberately model a different stock multi-row algorithm.
            rows = x.size // (groups * x.shape[-1])
            return layer(x) + (1 if rows > 1 else 0)

    stock = Stock()
    shape = (2, 4, groups, 1, 320) if groups > 1 else (2, 4, 320)
    x = mx.random.normal(shape)
    launches = []

    def launch(m, probe):
        launches.append(probe.shape)
        return per_row_stock(layer, probe)

    monkeypatch.setattr(gemv, "_launch", launch)
    monkeypatch.setattr(gemv, "supports", lambda *a: True)
    assert gemv._qualified(stock, x, "per-row reference")
    assert launches == [shape, shape]
    single_shape = (1, groups, 1, 320) if groups > 1 else (1, 320)
    assert calls == [single_shape] * 16
    assert equal(gemv.project(stock, x), per_row_stock(layer, x))
    assert not equal(per_row_stock(layer, x), stock(x))

    def wrong_last_row(m, probe):
        result = per_row_stock(layer, probe).reshape(8, groups, -1)
        return mx.concatenate([result[:-1], result[-1:] + 1], axis=0).reshape(
            *probe.shape[:-1], layer.scales.shape[-2]
        )

    monkeypatch.setattr(gemv, "_launch", wrong_last_row)
    assert not gemv._qualified(Stock(), x, "last lane mismatch")


def test_mismatch_and_compile_failure_stay_stock(monkeypatch, caplog):
    monkeypatch.setattr(gemv, "supports", lambda *a: True)
    calls = []

    def wrong(m, x):
        calls.append(x.shape)
        return mx.zeros((*x.shape[:-1], m.scales.shape[-2]))

    monkeypatch.setattr(gemv, "_launch", wrong)
    layer, x = projection(), inputs()
    for _ in range(2):
        assert equal(gemv.project(layer, x, "wq_b"), layer(x))
    assert len(calls) == 2 and "stock fallback" in caplog.text

    def broken(*a):
        raise RuntimeError("compile failure")

    monkeypatch.setattr(gemv, "_launch", broken)
    layer = projection()
    assert equal(gemv.project(layer, x), layer(x))
    assert "unavailable" in caplog.text


def test_hot_launch_failure_disables_only_that_shape(monkeypatch, caplog):
    monkeypatch.setattr(gemv, "supports", lambda *a: True)
    layer, x = projection(), inputs()
    monkeypatch.setattr(gemv, "_launch", per_row_stock)
    assert equal(gemv.project(layer, x), per_row_stock(layer, x))

    def broken(*a):
        raise RuntimeError("graph failure")

    monkeypatch.setattr(gemv, "_launch", broken)
    assert equal(gemv.project(layer, x), layer(x))
    assert not gemv._qualified(layer, x, "wkv")
    assert "launch fallback" in caplog.text


@needs_metal
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("name,n,k,groups", LAYOUTS)
def test_shipped_layouts_rows_1_to_8_against_single_row_stock(bits, name, n, k, groups):
    mx.set_default_device(mx.gpu)
    layer = projection(bits, n, k, groups)
    for dtype in [mx.float32, mx.bfloat16]:
        for rows in range(1, 9):
            x = inputs(rows, k, groups, dtype)
            assert gemv._qualified(layer, x, name), (
                f"{rows}-row {name}/{bits}/{dtype} failed qualification"
            )
            stock = per_row_stock(layer, x)
            assert equal(gemv._launch(layer, x), stock)
            assert equal(gemv.project(layer, x, name), stock)
            # Stock multi-row QMM uses different arithmetic; it is not the
            # bit reference for sequential-decode parity.


@needs_metal
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
def test_odd_output_and_k_tails_and_noncontiguous_rows(bits, groups, dtype):
    mx.set_default_device(mx.gpu)
    layer = projection(bits, 19, 320, groups, dtype=mx.float16)
    for rows in range(1, 9):
        full = inputs(rows * 2, 320, groups, dtype)
        x = full[:, ::2]
        stock = per_row_stock(layer, x)
        assert gemv._qualified(layer, x, "tails")
        assert equal(gemv._launch(layer, x), stock)
        assert equal(gemv.project(layer, x), stock)


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_decode_verify_and_batch_projection_wiring(monkeypatch, rows):
    from rapid_mlx.models.deepseek_v41_native.cache import BatchModelCache
    from scripts.bench_deepseek_v41_batched_decode import make_tiny_model

    model = make_tiny_model()
    seen = []

    def spy(m, x, label):
        seen.append((label.rsplit(" ", 1)[-1], x.shape))
        return m(x)

    monkeypatch.setattr(gemv, "project", spy)
    cache = model.make_cache(max_seq_len=16)
    mx.eval(model(mx.ones((1, rows), dtype=mx.int32), cache))
    assert {name for name, _ in seen} == {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}
    assert all(shape[1] == rows for _, shape in seen)
    seen.clear()
    caches = [model.make_cache(max_seq_len=16) for _ in range(rows)]
    batch = BatchModelCache.from_caches(caches)
    mx.eval(model(mx.ones((rows, 1), dtype=mx.int32), batch))
    assert {name for name, _ in seen} == {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}
    assert all(shape[0] == rows for _, shape in seen)


def test_qualification_compares_signed_zero_bits(monkeypatch):
    layer, x = projection(), inputs()
    layer.scales = mx.zeros_like(layer.scales)
    layer.biases = mx.zeros_like(layer.biases)
    monkeypatch.setattr(gemv, "_launch", lambda m, x: -mx.zeros((*x.shape[:-1], 19)))
    assert not gemv._qualified(layer, x, "signed zero")


def test_microbenchmark_synchronizes_and_reports_production_fallback(monkeypatch):
    from scripts import bench_deepseek_v41_attention_gemv as bench

    monkeypatch.setattr(gemv, "supports", lambda *a: True)
    monkeypatch.setattr(gemv, "_launch", per_row_stock)
    monkeypatch.setitem(bench.LAYOUTS, "wq_a", (19, 320, 1))
    monkeypatch.setitem(bench.LAYOUTS, "wo_a", (19, 320, 3))
    syncs = []
    monkeypatch.setattr(mx, "synchronize", lambda: syncs.append(True))
    for name in ("wq_a", "wo_a"):
        results = bench.benchmark(name, 4, [1, 3], mx.bfloat16, 1, 2)
        assert [r["rows"] for r in results] == [1, 3]
        assert all(r["qualified"] for r in results)
        assert all(r["sample_exact"] for r in results if r["rows"] == 1)
        assert all(r["weight_bytes"] > 0 and r["candidate_GB_s"] > 0 for r in results)
    assert len(syncs) == 2 * 2 * 3 * 5


def test_microbenchmark_cli_and_entrypoint(monkeypatch, capsys):
    import argparse
    import runpy

    from scripts import bench_deepseek_v41_attention_gemv as bench

    for argv in (["--warmup", "-1"], ["--repeats", "0"]):
        with pytest.raises(SystemExit):
            bench.main(argv)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        bench.main([])
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setenv("RAPID_MLX_DSV41_FUSED", "0")
    with pytest.raises(SystemExit):
        bench.main([])
    monkeypatch.delenv("RAPID_MLX_DSV41_FUSED")
    monkeypatch.setattr(mx, "set_default_device", lambda *a: None)
    monkeypatch.setattr(mx.metal, "device_info", lambda: {"device_name": "test"})
    monkeypatch.setattr(bench, "benchmark", lambda *a: [{"qualified": True}])
    bench.main(["--projection", "wq_a", "--bits", "4", "--rows", "1"])
    assert '"qualified": true' in capsys.readouterr().out
    args = SimpleNamespace(warmup=0, repeats=1, projection=[])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *a, **kw: args)
    runpy.run_path(bench.__file__, run_name="__main__")


def test_projection_wiring_also_reaches_fused_attention(monkeypatch):
    from rapid_mlx.models.deepseek_v41_native import decode_attention
    from scripts.bench_deepseek_v41_batched_decode import make_tiny_model

    model = make_tiny_model()
    monkeypatch.setattr(decode_attention, "supports", lambda *a: True)
    monkeypatch.setattr(decode_attention, "attend", lambda q, *a: mx.zeros_like(q))
    mx.eval(model(mx.ones((1, 3), dtype=mx.int32), model.make_cache(max_seq_len=16)))
