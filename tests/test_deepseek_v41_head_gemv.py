from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from rapid_mlx.models.deepseek_v41_native import head_gemv  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.head_gemv import (  # noqa: E402
    head_logits,
    supports,
)
from rapid_mlx.models.deepseek_v41_native.model import Model  # noqa: E402

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal kernel requires a GPU"
)


def _head(outputs: int = 1003, inputs: int = 512) -> nn.Linear:
    head = nn.Linear(inputs, outputs, bias=False)
    head.weight = (mx.random.normal((outputs, inputs)) * 0.02).astype(mx.bfloat16)
    return head


def _exact_logits(head: nn.Linear, x: mx.array) -> mx.array:
    # head_logits silently falls back to head(x); pin the kernel path so a
    # broken kernel cannot pass these comparisons through the fallback.
    assert supports(head, x)
    assert head_gemv._matches_stock(head)
    return head_gemv._launch(head.weight, x)


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("Metal kernel requires a GPU")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@needs_metal
@pytest.mark.parametrize("shape", [(1, 1), (1,), (1, 1, 1)])
def test_single_rows_are_bit_exact_with_stock_projection(gpu, shape) -> None:
    head = _head()
    x = mx.random.normal((*shape, 512))
    actual = _exact_logits(head, x)
    expected = head(x)
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected).item()


@needs_metal
@pytest.mark.parametrize("shared_min", [head_gemv.SHARED_READ_MIN_OUTPUTS, 1])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_each_row_matches_its_single_row_stock_result(
    gpu, monkeypatch, rows: int, shared_min: int
) -> None:
    # Multi-row stock GEMM may use a different (even reduced-precision) kernel;
    # the exact head instead gives every row its single-row decode result,
    # whether the rows read the weight separately or share one read.
    monkeypatch.setattr(head_gemv, "SHARED_READ_MIN_OUTPUTS", shared_min)
    head = _head(outputs=4096, inputs=1024)
    x = mx.random.normal((1, rows, 1024))
    actual = _exact_logits(head, x)
    expected = mx.concatenate([head(x[:, i : i + 1]) for i in range(rows)], axis=1)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@needs_metal
def test_strided_last_token_rows_match_single_row_stock(gpu) -> None:
    head = _head(outputs=4096, inputs=1024)
    hidden = mx.random.normal((2, 3, 1024))
    last = hidden[:, -1:]
    actual = _exact_logits(head, last)
    expected = mx.concatenate([head(hidden[b : b + 1, -1:]) for b in range(2)])
    mx.eval(actual, expected)
    assert actual.shape == (2, 1, 4096)
    assert mx.array_equal(actual, expected).item()


def _tiny_model() -> Model:
    args = ModelArgs(
        dim=128,
        vocab_size=40,
        n_layers=0,
        n_heads=1,
        head_dim=128,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
        dspark_target_layer_ids=(0,),
    )
    model = Model(args)
    model.head.weight = model.head.weight.astype(mx.bfloat16)

    class IdentityBlock:
        engram = None

        def __call__(self, h, pre_mix, *_args):
            return h, pre_mix

    model.layers = [IdentityBlock()]
    return model


@needs_metal
@pytest.mark.parametrize("last_logit_only", [True, False])
def test_model_forward_uses_exact_head(gpu, monkeypatch, last_logit_only) -> None:
    model = _tiny_model()
    tokens = mx.array([[1, 2, 3]])
    calls = []
    original = head_gemv.head_logits

    def spy(head, x):
        calls.append(x.shape)
        return original(head, x)

    launched = []
    launch = head_gemv._launch

    def launch_spy(weight, x, shared=None):
        launched.append(x.shape)
        return launch(weight, x, shared)

    monkeypatch.setattr("rapid_mlx.models.deepseek_v41_native.model.head_logits", spy)
    monkeypatch.setattr(head_gemv, "_launch", launch_spy)
    cache = model.make_cache(max_seq_len=8)
    logits = model(tokens, cache, last_logit_only=last_logit_only)
    mx.eval(logits)
    assert calls and calls[0][-1] == 128
    # the forward's own rows went through the kernel, not the stock fallback
    # (the self-check's probe launches have a different shape)
    assert calls[0] in launched
    rows = 1 if last_logit_only else 3
    assert logits.shape == (1, rows, 40)
    monkeypatch.setattr(head_gemv, "ENABLED", False)
    stock = model(
        tokens, model.make_cache(max_seq_len=8), last_logit_only=last_logit_only
    )
    mx.eval(stock)
    if rows == 1:
        assert mx.array_equal(logits, stock).item()


def test_unsupported_inputs_fall_back_to_stock(monkeypatch) -> None:
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    head = _head()
    rows = mx.random.normal((9, 512))
    assert not supports(head, rows)
    assert not supports(head, rows[:1].astype(mx.bfloat16))
    assert not supports(head, mx.random.normal((1, 256)))
    assert not supports(head, mx.zeros((0, 512)))
    assert not supports(head, mx.array(1.0))
    assert not supports(_head(inputs=200), mx.random.normal((1, 200)))
    # K >= 16 * N selects a split-K stock geometry.
    assert not supports(_head(outputs=32, inputs=512), mx.random.normal((1, 512)))
    biased = nn.Linear(512, 64)
    biased.weight = biased.weight.astype(mx.bfloat16)
    assert not supports(biased, rows[:1])
    fp32 = nn.Linear(512, 64, bias=False)
    assert not supports(fp32, rows[:1])
    quantized = nn.QuantizedLinear(512, 64, bias=False)
    assert not supports(quantized, rows[:1])
    weightless = nn.Linear(512, 64, bias=False)
    del weightless["weight"]
    assert not supports(weightless, rows[:1])
    flat = nn.Linear(512, 64, bias=False)
    flat.weight = mx.zeros((512,), dtype=mx.bfloat16)
    assert not supports(flat, rows[:1])
    actual = head_logits(head, rows)
    expected = head(rows)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_cpu_and_disabled_switch_use_stock(monkeypatch) -> None:
    head = _head()
    x = mx.random.normal((1, 512))
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not head_gemv._gpu_default()
        assert not supports(head, x)
    finally:
        mx.set_default_device(previous)
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    assert supports(head, x)
    monkeypatch.setattr(head_gemv, "ENABLED", False)
    assert not supports(head, x)


def test_dispatch_geometry_without_metal(monkeypatch) -> None:
    """Exercise the kernel factory and launch wrapper with a stand-in on any host."""
    seen = {}
    built = {}

    def fake_kernel(**kwargs):
        seen.update(kwargs)
        x, weight = kwargs["inputs"]
        rows = [x[i : i + 1] @ weight.astype(mx.float32).T for i in range(x.shape[0])]
        return [mx.concatenate(rows)]

    def fake_metal_kernel(**kwargs):
        built[kwargs["name"]] = kwargs
        return fake_kernel

    head_gemv._kernel.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", fake_metal_kernel)
    monkeypatch.setattr(head_gemv, "supports", lambda head, x: True)
    head = _head(outputs=1003)
    logits = head_logits(head, mx.zeros((2, 3, 512)))
    reference, ok = getattr(head, head_gemv._CHECK_ATTR)
    assert ok is True and reference() is head.weight
    assert logits.shape == (2, 3, 1003)
    assert seen["grid"] == (251 * 32 * 4, 6, 1)
    assert seen["threadgroup"] == (128, 1, 1)
    assert dict(seen["template"]) == {"K": 512, "N": 1003, "OUTPUTS": 4}
    assert seen["inputs"][0].shape == (6, 512)
    assert list(built) == ["rapid_v41_fp32_bf16_head_gemv"]
    assert built["rapid_v41_fp32_bf16_head_gemv"]["ensure_row_contiguous"] is True

    # A head at or above the threshold reads its weight once for all rows;
    # a single row keeps the per-row kernel.
    monkeypatch.setattr(head_gemv, "SHARED_READ_MIN_OUTPUTS", 1003)
    head = _head(outputs=1003)
    logits = head_logits(head, mx.zeros((2, 3, 512)))
    assert logits.shape == (2, 3, 1003)
    assert seen["grid"] == (251 * 32 * 4, 1, 1)
    assert dict(seen["template"]) == {"K": 512, "N": 1003, "OUTPUTS": 4, "MAX_ROWS": 8}
    shared = built["rapid_v41_fp32_bf16_head_gemv_shared"]
    assert "input_shape[0]" in shared["source"] and shared["ensure_row_contiguous"]
    head_logits(head, mx.zeros((1, 512)))
    assert seen["grid"] == (251 * 32 * 4, 1, 1) and "MAX_ROWS" not in dict(
        seen["template"]
    )
    head_gemv._kernel.cache_clear()


def test_self_check_mismatch_routes_head_to_stock(monkeypatch) -> None:
    launches = []

    def wrong_kernel(**kwargs):
        launches.append(kwargs["inputs"][0].shape)
        rows, outputs = kwargs["output_shapes"][0]
        return [mx.zeros((rows, outputs), dtype=mx.float32)]

    monkeypatch.setattr(head_gemv, "_kernel", lambda shared=False: wrong_kernel)
    monkeypatch.setattr(head_gemv, "supports", lambda head, x: True)
    head = _head()
    x = mx.random.normal((1, 512))
    for _ in range(2):
        actual = head_logits(head, x)
        assert mx.array_equal(actual, head(x)).item()
    # One probe launch only; this weight then stays on the stock path.
    assert launches == [(8, 512)]
    assert getattr(head, head_gemv._CHECK_ATTR)[1] is False
    assert "_rapid_exact_head_check" not in head.parameters()
    # A replaced weight of the same shape is validated again, and the old
    # weight is not kept alive by the check.
    old = getattr(head, head_gemv._CHECK_ATTR)[0]
    head.weight = _head().weight
    assert old() is None
    head_logits(head, x)
    assert launches == [(8, 512), (8, 512)]
    assert getattr(head, head_gemv._CHECK_ATTR)[0]() is head.weight


def test_self_check_launch_failure_falls_back(monkeypatch) -> None:
    def broken_kernel(shared=False):
        raise RuntimeError("unable to build metal library")

    monkeypatch.setattr(head_gemv, "_kernel", broken_kernel)
    monkeypatch.setattr(head_gemv, "supports", lambda head, x: True)
    head = _head()
    x = mx.random.normal((1, 512))
    assert mx.array_equal(head_logits(head, x), head(x)).item()
    assert getattr(head, head_gemv._CHECK_ATTR)[1] is False


def test_self_check_compiles_the_hot_path_specialization(monkeypatch) -> None:
    """The real self-check evaluates the specialization every later launch uses."""
    launches = []

    def recording_kernel(shared=False):
        def call(*, inputs, template, **_kwargs):
            x, weight = inputs
            # A Metal pipeline is specialized by its template and input dtypes.
            launches.append(
                (tuple(template), x.dtype, weight.dtype, shared, x.shape[0])
            )
            # each row's own stock projection (the kernel's contract), so the
            # self-check passes
            return [mx.concatenate([x[r : r + 1] @ weight.T for r in range(len(x))])]

        return call

    monkeypatch.setattr(head_gemv, "_kernel", recording_kernel)
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    head = _head()
    for rows in (1, 3, 8):
        mx.eval(head_logits(head, mx.random.normal((rows, 512))))
    assert getattr(head, head_gemv._CHECK_ATTR)[1] is True
    probe, *hot = launches
    assert probe[4] == head_gemv.MAX_ROWS  # the self-check ran first, on the full grid
    assert [rows for *_, rows in hot] == [1, 3, 8]
    assert {launch[:4] for launch in launches} == {probe[:4]}

    # A shared-read head probes both variants, then only uses probed ones.
    launches.clear()
    monkeypatch.setattr(head_gemv, "SHARED_READ_MIN_OUTPUTS", 1)
    head = _head()
    for rows in (1, 3, 8):
        mx.eval(head_logits(head, mx.random.normal((rows, 512))))
    probes, hot = launches[:2], launches[2:]
    assert [(p[3], p[4]) for p in probes] == [(False, 8), (True, 8)]
    assert [(h[3], h[4]) for h in hot] == [(False, 1), (True, 3), (True, 8)]
    assert {h[:4] for h in hot} <= {p[:4] for p in probes}
