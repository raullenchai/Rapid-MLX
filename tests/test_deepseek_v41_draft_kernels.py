"""CPU wiring/fallback contracts and Metal checks for acceptance-qualified draft kernels."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import draft_kernels as kernels  # noqa: E402
from rapid_mlx.models.deepseek_v41_native import serving  # noqa: E402
from scripts import bench_deepseek_v41_draft as bench  # noqa: E402

pytestmark = pytest.mark.requires_mlx


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def exact(a, b):
    assert a.dtype == b.dtype
    assert mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.0])
def test_supported_dispatch_has_no_probes_or_host_sync(delta, monkeypatch):
    owner = SimpleNamespace(_rapid_draft_pending=[])
    x = mx.ones((5, 32))
    calls = []

    def candidate(value):
        calls.append(value)
        return value + delta

    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("draft dispatch synchronized")

        patch.setattr(mx, "eval", forbidden)
        patch.setattr(mx, "synchronize", forbidden)
        actual = kernels._checked(
            owner, "mtp.2.ffn", x, (4, 4, 4), candidate, forbidden
        )
    exact(actual, x + delta)
    assert calls == [x]
    assert bool(owner._rapid_draft_pending[0][3])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_deferred_fallback_is_per_layer_permanent_and_warns_once(bad, caplog):
    draft = SimpleNamespace(adapter=SimpleNamespace(), w=SimpleNamespace())
    x = mx.ones((5, 4))
    kernels.begin_round(draft)
    owner = draft.adapter
    bad_output = kernels._checked(
        owner, "mtp.1.ffn", x, (2, 3, 4), lambda v: v * bad, lambda v: v
    )
    # Propagating non-finite input must not disable a healthy next component.
    kernels._checked(draft.w, "head", bad_output, "bfloat16", lambda v: v, lambda v: v)
    kernels._checked(owner, "mtp.0.ffn", x, (4, 4, 4), lambda v: v + 1, lambda v: v)
    draft._draft_kernel_status = kernels.round_status(draft)
    mx.eval(draft._draft_kernel_status)
    assert kernels.disable_nonfinite(draft)
    assert not kernels.disable_nonfinite(draft)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "mtp.1.ffn" in caplog.text and "bits=(2, 3, 4)" in caplog.text
    assert not hasattr(draft.w, "_rapid_draft_disabled")
    kernels.begin_round(draft)
    actual = kernels._checked(
        owner,
        "mtp.1.ffn",
        x[:1],
        (8, 8, 8),
        lambda v: pytest.fail("disabled"),
        lambda v: v,
    )
    exact(actual, x[:1])
    draft._draft_kernel_status = kernels.round_status(draft)
    mx.eval(draft._draft_kernel_status)
    assert not kernels.disable_nonfinite(draft)
    assert not kernels.disable_nonfinite(SimpleNamespace())


def test_head_wiring_and_unsupported_fallback(monkeypatch):
    weight = mx.ones((64, 128)).astype(mx.bfloat16)
    calls = []
    weights = SimpleNamespace(
        read=lambda _: weight, entries={}, linear=lambda _, x: x @ weight.T
    )
    x = mx.ones((5, 128))
    monkeypatch.setattr(kernels.head_gemv, "_supports_weight", lambda *_: True)

    def launch(w, v):
        calls.append(1)
        return v @ w.T + 1

    monkeypatch.setattr(kernels.head_gemv, "_launch", launch)
    exact(kernels.head(weights, x), weights.linear("head", x) + 1)
    assert len(calls) == 1
    for key in ("head.scales", "head.bias"):
        weights.entries = {key: {}}
        exact(kernels.head(weights, x), weights.linear("head", x))
    weights.entries = {}
    monkeypatch.setattr(kernels.head_gemv, "_supports_weight", lambda *_: False)
    exact(kernels.head(weights, x), weights.linear("head", x))
    monkeypatch.setattr(kernels, "ENABLED", False)
    exact(kernels.head(weights, x), weights.linear("head", x))


def install_tiny():
    _, draft = bench.tiny_setup()
    serving._install_packed_mtp_moe(draft)
    return draft


def test_packed_moe_keeps_draft_routing_and_post_down_weights(monkeypatch):
    draft = install_tiny()
    base = "mtp.0.ffn"
    x = mx.random.normal((5, 32))
    reference = draft.adapter.moe(base, x)
    seen = []
    monkeypatch.setattr(kernels.moe_decode, "supports", lambda _: True)
    # The tiny fixture is intentionally not a real Metal layout. This test
    # isolates wiring; actual kernel arithmetic is checked separately below.
    monkeypatch.setattr(
        kernels.moe_decode, "_layout", lambda _, **kwargs: (32, 32, 8, 4, 4, 4)
    )

    def experts(layer, layout, value, selected, picks):
        seen.append((selected, picks))
        scores = mx.sqrt(
            mx.logaddexp(
                value @ layer.gate.weight.astype(mx.float32).T, mx.zeros((5, 4))
            )
        )
        expected_picks = mx.argsort(scores + layer.gate.bias, axis=-1)[:, -2:]
        exact(picks, expected_picks.astype(mx.uint32))
        expected_weights = mx.take_along_axis(scores, expected_picks, axis=-1)
        expected_weights = expected_weights / (
            mx.sum(expected_weights, axis=-1, keepdims=True) + 1e-20
        )
        exact(selected, expected_weights * draft.c["routed_scaling_factor"])
        with monkeypatch.context() as patch:
            patch.setattr(kernels, "ENABLED", False)
            return draft.adapter.moe(base, value)

    monkeypatch.setattr(kernels.moe_decode, "_experts", experts)
    exact(draft.adapter.moe(base, x), reference)
    assert len(seen) == 1
    exact(draft.adapter.moe(base, x), reference)
    assert len(seen) == 2
    # Supported settings dispatch directly without qualification.
    draft.c["swiglu_limit"] = 8
    draft.adapter.moe(base, x)
    assert len(seen) == 3


def test_unsupported_shared_and_bias_fall_back(monkeypatch, caplog):
    draft = install_tiny()
    base = "mtp.0.ffn"
    x = mx.ones((5, 32))
    reference = draft.adapter.moe(base, x)
    monkeypatch.setattr(kernels.moe_decode, "supports", lambda _: True)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported draft layout dispatched fused experts")

    monkeypatch.setattr(kernels.moe_decode, "_experts", forbidden)
    exact(draft.adapter.moe(base, x), reference)
    assert not caplog.records
    # Bias-bearing quantized shared experts are also ineligible.
    for name in ("w1", "w2", "w3"):
        prefix = f"{base}.shared_experts.{name}"
        w, s, b = mx.quantize(draft.w.read(prefix + ".weight"), group_size=32, bits=4)
        for suffix, value in (
            ("weight", w),
            ("scales", s),
            ("biases", b),
            ("bias", mx.ones((32,))),
        ):
            draft.w._arrays[prefix + "." + suffix] = value
            draft.w.entries[prefix + "." + suffix] = {"shape": value.shape}
    with monkeypatch.context() as patch:
        patch.setattr(kernels, "ENABLED", False)
        reference = draft.adapter.moe(base, x)
    exact(draft.adapter.moe(base, x), reference)
    assert not caplog.records


@pytest.mark.parametrize("enabled", [False, True])
def test_component_timer_boundaries_and_restoration(enabled, monkeypatch):
    events = []
    original_sync, original_eval = mx.synchronize, mx.eval

    def sync():
        events.append("sync")
        original_sync()

    def evaluate(value):
        events.append("eval")
        original_eval(value)

    monkeypatch.setattr(mx, "synchronize", sync)
    monkeypatch.setattr(mx, "eval", evaluate)

    class Draft:
        def head(self, value):
            events.append("call")
            return value

    draft = Draft()
    draft.attention = draft.head
    draft.markov = draft.head
    draft.adapter = SimpleNamespace(moe=draft.head)
    original = draft.attention
    with (
        pytest.raises(RuntimeError, match="restore"),
        bench.draft_component_timing(draft, enabled) as totals,
    ):
        draft.head(mx.ones((1,)))
        assert events == (
            ["eval", "sync", "call", "eval", "sync"] if enabled else ["call"]
        )
        assert totals["head"] >= 0
        raise RuntimeError("restore")
    assert draft.attention is original
    assert "head" not in vars(draft)


def test_component_timer_cli_and_stats(capsys):
    assert bench.main(["--tokens", "2", "--repeats", "1", "--component-timers"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["component_timers"] is True
    assert result["runs"][0]["greedy_equal"]
    means = result["runs"][0]["draft_on"]["mean_round_ms"]
    for name in ("attention", "moe", "head", "markov"):
        assert means[f"draft_{name}_ms"] >= 0


def supported_moe_fixture():
    """Small real 4-bit routed / 8-bit shared layout with FP32 draft inputs."""
    from tests.test_deepseek_v41_mixed_precision_fallback import _layer

    layer = _layer(4, down_bits=4, shared_bits=8)
    _, draft = bench.tiny_setup()
    weights = draft.w
    weights._arrays = {}
    weights.resident = {}
    weights.q = {"bits": 4, "group_size": 64, "modules": {}}
    draft.c["dspark_n_routed_experts"] = 64
    draft.c["dspark_num_experts_per_tok"] = 4
    base = "mtp.0.ffn"
    for name in ("weight", "bias"):
        weights._arrays[f"{base}.gate.{name}"] = getattr(layer.gate, name)
    for name, attr in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
        for index in range(64):
            for suffix in ("weight", "scales", "biases"):
                weights._arrays[f"{base}.experts.{index}.{name}.{suffix}"] = getattr(
                    getattr(layer.experts, attr), suffix
                )[index]
        prefix = f"{base}.shared_experts.{name}"
        weights.q["modules"][prefix] = {"bits": 8, "group_size": 64}
        for suffix in ("weight", "scales", "biases"):
            weights._arrays[f"{prefix}.{suffix}"] = getattr(
                getattr(layer.shared_experts, name), suffix
            )
    weights.entries = {
        key: {"shape": value.shape} for key, value in weights._arrays.items()
    }
    serving._install_packed_mtp_moe(draft)
    return draft, base


@pytest.mark.parametrize("delta", [0, 1])
def test_metal_supported_moe_is_finite_without_bitwise_qualification(
    delta, monkeypatch
):
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    mx.set_default_device(mx.gpu)
    draft, base = supported_moe_fixture()
    x = mx.random.normal((5, 512))
    original = kernels.moe_decode._experts
    calls = []

    def experts(*args):
        calls.append(1)
        return original(*args) + delta

    monkeypatch.setattr(kernels.moe_decode, "_experts", experts)
    kernels.begin_round(draft)
    actual = draft.adapter.moe(base, x)
    assert bool(mx.all(mx.isfinite(actual)))
    assert len(calls) == 1
    draft._draft_kernel_status = kernels.round_status(draft)
    mx.eval(draft._draft_kernel_status)
    assert not kernels.disable_nonfinite(draft)


@pytest.mark.parametrize("delta", [0, 1])
@pytest.mark.parametrize("rows", range(1, 6))
def test_metal_head_is_finite_without_bitwise_qualification(delta, rows, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    mx.set_default_device(mx.gpu)
    _, draft = bench.tiny_setup()
    draft.w._arrays["head.weight"] = mx.random.normal((16384, 512)).astype(mx.bfloat16)
    x = mx.random.normal((rows, 512))
    calls = []
    variants = []
    original = kernels.head_gemv._launch
    original_kernel = kernels.head_gemv._kernel

    def kernel(shared=False):
        variants.append(shared)
        return original_kernel(shared)

    def launch(*args):
        calls.append(1)
        return original(*args) + delta

    monkeypatch.setattr(kernels.head_gemv, "_launch", launch)
    monkeypatch.setattr(kernels.head_gemv, "_kernel", kernel)
    kernels.begin_round(draft)
    assert bool(mx.all(mx.isfinite(draft.head(x))))
    assert len(calls) == 1
    assert variants == [rows > 1]


def test_external_routes_do_not_require_target_router_gemv():
    from tests.test_deepseek_v41_mixed_precision_fallback import _layer

    layer = _layer(4, down_bits=4, shared_bits=8)
    # The draft has fewer experts than the target; 5120 >= 16 * 128.
    # Model this boundary with four of the fixture's 64 routed experts.
    for projection in (
        layer.experts.gate_proj,
        layer.experts.up_proj,
        layer.experts.down_proj,
    ):
        for suffix in ("weight", "scales", "biases"):
            setattr(projection, suffix, getattr(projection, suffix)[:4])
    layer.gate.topk = 4
    layer.gate.weight = layer.gate.weight[:4].astype(mx.float32)
    layer.gate.bias = layer.gate.bias[:4]
    assert kernels.moe_decode._layout(layer) is None
    assert kernels.moe_decode._layout(layer, external_routing=True) == (
        512,
        256,
        8,
        4,
        4,
        4,
    )


@pytest.mark.parametrize("seed", [7, 11, 42])
@pytest.mark.parametrize("failure", [None, "mtp.0.ffn", "head"])
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_serving_acceptance_and_retry_equal_draft_off(
    seed, failure, depth, monkeypatch, caplog
):
    model, draft = bench.tiny_setup(seed)
    serving._install_vectorized_mtp_attention(draft)
    serving._install_packed_mtp_moe(draft)
    expected, _ = bench.plain_decode(model, [1, 2, 3], 12)
    original = kernels._checked
    counts = {}

    def dispatch(owner, label, x, bits, candidate, stock):
        if label == "head":
            assert 1 <= x.shape[0] <= depth

        def fused(value):
            counts[label] = counts.get(label, 0) + 1
            result = stock(value)
            if label == failure:
                return mx.full(result.shape, float("nan"))
            # Deliberately change proposals; equality to stock is not required.
            return -result * 3

        return original(owner, label, x, bits, fused, stock)

    monkeypatch.setattr(kernels, "_checked", dispatch)
    monkeypatch.setattr(kernels.moe_decode, "supports", lambda _: True)
    # Force only draft dispatch: patching the shared module's support predicate
    # also sends the tiny target's FP32 head/router weights into a BF16 kernel.
    monkeypatch.setattr(
        kernels,
        "head_gemv",
        SimpleNamespace(
            _supports_weight=lambda *_: True, _launch=kernels.head_gemv._launch
        ),
    )
    monkeypatch.setenv("RAPID_MLX_DSV41_DRAFT_DEPTH", str(depth))
    runtime = serving.DSparkRuntime(draft, "synthetic", "synthetic", "synthetic")
    actual = [
        chunk.token
        for chunk in serving.stream_generate(
            model,
            bench.TinyTokenizer(),
            [1, 2, 3],
            runtime=runtime,
            max_tokens=12,
            ignore_eos=True,
        )
    ]
    assert actual == expected
    assert counts["head"] >= 1
    warnings = [
        record
        for record in caplog.records
        if record.name == kernels.logger.name and record.levelname == "WARNING"
    ]
    if failure:
        assert counts[failure] == 1
        assert len(warnings) == 1
        assert failure in warnings[0].getMessage()
    else:
        assert not warnings


@pytest.mark.parametrize("failure", ["mtp.0.ffn", "head"])
def test_diagnostic_proposals_retry_nonfinite(failure, monkeypatch, caplog):
    _, draft = bench.tiny_setup()
    serving._install_packed_mtp_moe(draft)
    draft.observe_many(mx.random.normal((1, 3, 96)), 0)
    original = kernels._checked

    def dispatch(owner, label, x, bits, candidate, stock):
        candidate = lambda value: (
            mx.full(stock(value).shape, float("inf"))
            if label == failure
            else stock(value)
        )
        return original(owner, label, x, bits, candidate, stock)

    monkeypatch.setattr(kernels, "_checked", dispatch)
    monkeypatch.setattr(kernels.moe_decode, "supports", lambda _: True)
    monkeypatch.setattr(kernels.head_gemv, "_supports_weight", lambda *_: True)
    proposals, confidence = draft.propose(4)
    assert len(proposals) == 6
    assert bool(mx.all(mx.isfinite(confidence)))
    assert len(caplog.records) == 1
    draft.propose(4)
    assert len(caplog.records) == 1
