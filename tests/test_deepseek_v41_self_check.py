"""CPU regression tests for first-use kernel qualification and fallback."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import (
    affine_route_qmv as route,
)
from rapid_mlx.models.deepseek_v41_native import (
    decode_attention as attention,
)
from rapid_mlx.models.deepseek_v41_native import (
    rope_quant as rope,
)
from rapid_mlx.models.deepseek_v41_native.self_check import SelfCheck


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    for module in (route, attention, rope):
        monkeypatch.setattr(module, "_CHECK", SelfCheck(module.__name__))
    yield
    mx.set_default_device(device)


@pytest.mark.parametrize("result", [True, False, RuntimeError("launch")])
def test_serialized_probe_is_cached_and_warns_once(result, caplog):
    check = SelfCheck("test kernel")
    barrier = Barrier(4)
    probe = Mock(
        side_effect=result if isinstance(result, Exception) else None,
        return_value=result,
    )

    def qualify(_):
        barrier.wait()
        return check.qualified((1,), probe)

    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(qualify, range(4))) == [result is True] * 4
    assert probe.call_count == 1
    assert check.qualified((1,), probe) == (result is True)
    assert len(caplog.records) == (0 if result is True else 1)
    if isinstance(result, Exception):
        assert caplog.records[0].exc_info[1] is result
    check.qualified((2,), probe)
    assert probe.call_count == 2


def gpu_gate(monkeypatch):
    # Only the eligibility predicate sees a GPU; all real arrays stay on CPU.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)


@pytest.mark.parametrize("kind", ["rope", "attention", "route"])
@pytest.mark.parametrize("failure", ["pass", "mismatch", "launch", "lazy"])
def test_first_use_dispatch(kind, failure, monkeypatch, caplog):
    x = mx.ones((1, 1, 128))
    cos, sin = mx.ones((1, 32)), mx.zeros((1, 32))
    ring = mx.zeros((1, 8, 128))
    from types import SimpleNamespace

    w, s, b = mx.quantize(mx.ones((2, 8, 64)), group_size=64, bits=2)
    module = SimpleNamespace(
        weight=w,
        scales=s.astype(mx.float16),
        biases=b.astype(mx.float16),
        bits=2,
        group_size=64,
    )
    inputs, indices = mx.ones((1, 64)), mx.zeros((1, 1), dtype=mx.uint32)
    if kind == "rope":
        target = rope
        invoke = lambda: rope.rope_fake_quant(x, 64, cos, sin, "fp8_ue8m0")
        expected = rope.FORMATS["fp8_ue8m0"][2](rope.rope_tail(x, 64, cos, sin), 32)
        fused = Mock(return_value=expected)
    elif kind == "route":
        target = route
        invoke = lambda: route.affine2_route_down_qmv(module, inputs, indices)
        expected = route._reference(module, inputs, indices)
        fused = Mock(return_value=expected)
    else:
        target = attention
        invoke = lambda: attention.supports(x, ring, 8, 128, 64)
        fused = Mock()
    mx.eval(x, ring, cos, sin, inputs, indices)
    calls = []

    def probe(*args):
        calls.append(args)
        if failure == "launch":
            raise RuntimeError("compile/launch")
        if failure == "lazy":
            # Failure only when the lazy result is materialized.
            Mock(item=Mock(side_effect=RuntimeError("evaluation"))).item()
        return failure == "pass"

    monkeypatch.setattr(target, "_probe", probe)
    monkeypatch.setattr(target, "_fused", fused)
    gpu_gate(monkeypatch)
    for _ in range(2):
        result = invoke()
        if kind == "attention":
            assert result == (failure == "pass")
        else:
            assert mx.array_equal(result, expected).item()
    assert len(calls) == 1
    assert len(caplog.records) == (0 if failure == "pass" else 1)
    if kind != "attention":
        assert fused.call_count == (2 if failure == "pass" else 0)
    monkeypatch.setenv("RAPID_MLX_DSV41_FUSED", "0")
    target._CHECK.verdicts.clear()
    result = invoke()
    assert not target._CHECK.verdicts
    assert len(calls) == 1
    if kind == "attention":
        assert result is False
    else:
        assert mx.array_equal(result, expected).item()


@pytest.mark.parametrize("fmt", sorted(rope.FORMATS))
@pytest.mark.parametrize("good", [True, False])
def test_rope_probe_evaluates_bit_equality(fmt, good, monkeypatch):
    def fused(x, rd, cos, sin, fmt, dtype, **kw):
        result = rope.FORMATS[fmt][2](
            rope.rope_tail(x, rd, cos, sin), rope.FORMATS[fmt][0]
        ).astype(dtype)
        return result if good else result + 1

    monkeypatch.setattr(rope, "_fused", fused)
    assert rope._probe((2, 2, 128), mx.float32, 64, fmt, mx.float16) == good


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("good", [True, False])
def test_route_probe_evaluates_bit_equality(bits, good, monkeypatch):
    def fused(module, x, indices):
        result = route._reference(module, x, indices)
        return result if good else result + 1

    monkeypatch.setattr(route, "_fused", fused)
    assert route._probe(mx.float16, bits, 6, 2, 64, 8) == good


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("good", [True, False])
def test_attention_probe_evaluates_output_and_preserves_inputs(
    dtype, good, monkeypatch
):
    seen = []

    def fused(q, ring, start, comp, picks, offset, sink, cos, sin, scale, rd, old):
        seen.append((ring.shape, picks.tolist(), old.shape))
        return mx.full(q.shape, 100.0)

    # Capture the independent reference graph; the mismatch case uses the real
    # comparator, while the pass exercises the cache/index preservation checks.
    monkeypatch.setattr(attention, "_fused", fused)
    if good:
        monkeypatch.setattr(mx, "allclose", lambda *a, **kw: mx.array(True))
    assert attention._probe(dtype, mx.float32, 8, 128, 64, 8) == good
    assert seen[0][0] == (1, 8, 128)
    assert seen[0][1][0][0] == [18, -1, 16, 19]


@pytest.mark.parametrize(
    "module,args",
    [
        (rope, ((1, 128), mx.float32, 64, "fp8_ue8m0", mx.float32)),
        (route, (mx.float16, 2, 1, 1, 64, 8)),
        (attention, (mx.float32, mx.float32, 8, 128, 64, 8)),
    ],
)
def test_probe_lazy_comparison_failure_falls_back(module, args, monkeypatch, caplog):
    # A fused graph can be built successfully and fail only at comparison eval.
    monkeypatch.setattr(module, "_fused", lambda *a, **kw: mx.array([1.0]))

    def fail(*a, **kw):
        return Mock(item=Mock(side_effect=RuntimeError("lazy launch")))

    monkeypatch.setattr(mx, "array_equal", fail)
    monkeypatch.setattr(mx, "allclose", fail)
    assert not module._CHECK.qualified((1,), module._probe, *args)
    assert not module._CHECK.qualified((1,), module._probe, *args)
    assert len(caplog.records) == 1


def test_waiter_rechecks_verdict_after_acquiring_lock():
    check = SelfCheck("waiting kernel")

    class PublishedWhileWaiting:
        def __enter__(self):
            check.verdicts[(1,)] = False

        def __exit__(self, *args):
            return False

    check.lock = PublishedWhileWaiting()
    probe = Mock()
    assert not check.qualified((1,), probe)
    probe.assert_not_called()


class EligibleCPU:
    """Expose GPU eligibility only to the module under test, never launch Metal."""

    def default_device(self):
        return mx.gpu

    @property
    def metal(self):
        return Mock(is_available=lambda: True)

    def __getattr__(self, name):
        return getattr(mx, name)


@pytest.mark.parametrize("target", [attention, rope])
@pytest.mark.parametrize("failure", [False, RuntimeError("lazy launch")])
def test_model_failure_preserves_greedy_tokens_and_cache(target, failure, monkeypatch):
    from rapid_mlx.models.deepseek_v41_native.model import Model
    from tests.test_deepseek_v41_decode_attention import _tiny_args

    mx.random.seed(5032)
    model = Model(_tiny_args())
    model.set_dtype(mx.float32)
    prompt = mx.array([[2, 4, 6, 8, 10]])

    def run():
        cache = model.make_cache(max_seq_len=32)
        tokens = []
        logits = model(prompt, cache, last_logit_only=True)
        for _ in range(3):
            token = mx.argmax(logits[:, -1], axis=-1)[:, None]
            tokens.append(token)
            logits = model(token, cache)
        values = [mx.concatenate(tokens, axis=-1)]
        values.extend(layer.win_kv for layer in cache.layers)
        mx.eval(*values)
        return values

    expected = run()
    monkeypatch.setattr(target, "mx", EligibleCPU())
    probe = Mock(
        return_value=failure,
        side_effect=failure if isinstance(failure, Exception) else None,
    )
    monkeypatch.setattr(target, "_probe", probe)
    # The request must never construct the failed kernel after qualification.
    monkeypatch.setattr(
        target,
        "_kernels" if target is attention else "_kernel",
        Mock(side_effect=AssertionError("request launched kernel")),
    )
    actual = run()
    assert probe.called
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_installed_route_failure_preserves_stock_glu(bits, monkeypatch):
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from rapid_mlx.models.deepseek_v41_native.serving import ExactDirectDownSwitchGLU

    mx.random.seed(5032)
    layer = SwitchGLU(128, 64, 2, bias=False)
    nn.quantize(layer, group_size=64, bits=bits)
    layer.down_proj.scales = layer.down_proj.scales.astype(mx.float16)
    layer.down_proj.biases = layer.down_proj.biases.astype(mx.float16)
    x = mx.ones((2, 128))
    indices = mx.array([[0, 1], [1, 0]], dtype=mx.uint32)
    expected = layer(x, indices)
    mx.eval(expected)
    layer.__class__ = ExactDirectDownSwitchGLU
    monkeypatch.setattr(route, "mx", EligibleCPU())
    monkeypatch.setattr(route, "_probe", Mock(side_effect=RuntimeError("launch")))
    monkeypatch.setattr(
        route, "_kernel", Mock(side_effect=AssertionError("request launched kernel"))
    )
    actual = layer(x, indices)
    assert mx.array_equal(actual, expected).item()


def test_rope_probe_keeps_specialization_but_bounds_sequence(monkeypatch):
    launches = []

    def kernel(**kwargs):
        launches.append(kwargs)
        return [mx.zeros(kwargs["output_shapes"][0], dtype=kwargs["output_dtypes"][0])]

    monkeypatch.setattr(rope, "_kernel", lambda: kernel)
    assert not rope._probe((4096, 2, 128), mx.float32, 64, "fp8_ue8m0", mx.float32)
    launch = launches[0]
    assert dict(launch["template"])["S"] == 4096
    assert launch["output_shapes"] == [(1, 2, 2, 128)]
    assert launch["grid"] == (512, 1, 1)
