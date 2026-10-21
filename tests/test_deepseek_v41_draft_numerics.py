"""Strict CPU contracts and Metal numerical checks for draft verification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native import dspark, serving
from rapid_mlx.models.deepseek_v41_native.hyper_connections import (
    hc_mixes,
    hc_post,
    hc_pre,
)
from rapid_mlx.models.deepseek_v41_native.layers import (
    RMSNorm,
    precompute_freqs_cis,
    rope_tail,
)
from rapid_mlx.models.deepseek_v41_native.moe import Gate, SharedExpert
from rapid_mlx.models.deepseek_v41_native.sparse_attention import sparse_attn
from scripts import bench_deepseek_v41_draft as bench


@pytest.fixture(autouse=True)
def restore_default_device():
    """The tiny benchmark CLI selects the CPU; never leak that to later modules."""
    previous = mx.default_device()
    yield
    mx.set_default_device(previous)


@pytest.fixture
def cpu_device():
    """Keep algebraic/bit-exact checks independent of the default device."""
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(
    params=[
        pytest.param(mx.cpu, id="cpu"),
        pytest.param(
            mx.gpu,
            id="metal",
            marks=pytest.mark.skipif(
                not mx.metal.is_available(), reason="Metal GPU unavailable"
            ),
        ),
    ]
)
def numerical_device(request):
    previous = mx.default_device()
    try:
        mx.set_default_device(request.param)
        yield request.param
    finally:
        mx.set_default_device(previous)


def equal(a, b, atol=0):
    assert a.dtype == b.dtype
    assert bool(mx.allclose(a, b, atol=atol, rtol=0)), float(mx.max(mx.abs(a - b)))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.usefixtures("cpu_device")
def test_rms_rope_quantization_hyper_connections(dtype):
    _, draft = bench.tiny_setup()
    x = mx.random.normal((5, 4, 32)).astype(dtype)
    weight = mx.random.normal((32,))
    norm = RMSNorm(32, 1e-20)
    norm.weight = weight
    equal(dspark._rms_impl(x[:, 0], weight, 1e-20), norm(x[:, 0]))
    pre, post, combine = draft.adapter.mixes("mtp.0", "attn", x)
    reference = hc_mixes(
        x[None],
        draft.w.read("mtp.0.hc_attn_fn"),
        draft.w.read("mtp.0.hc_attn_scale"),
        draft.w.read("mtp.0.hc_attn_base"),
        4,
        3,
        1e-20,
        1e-6,
    )
    for a, b in zip((pre, post, combine), reference):
        equal(a, b[0])
    collapsed = dspark._hc_pre(x, pre)
    equal(collapsed, hc_pre(x[None], pre[None])[0])
    equal(
        dspark._hc_post(collapsed, x, post, combine),
        hc_post(collapsed[None], x[None], post[None], combine[None])[0],
    )
    cos, sin = precompute_freqs_cis(8, 16, 0, 10000, 1, 32, 1)
    for inverse in (False, True):
        actual = dspark.rotate_positions(x, mx.arange(5) + 7, 8, 10000, inverse)
        equal(actual, rope_tail(x[None], 8, cos[7:12], sin[7:12], inverse)[0])
        equal(actual[0], dspark._rotary(x[0], 7, 8, 10000, inverse))
    equal(dspark.quantize_cache(x, 8, 32), dspark.fake_quant_fp8_ue8m0(x, 32))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_attention_sink_matches_target_and_vectorized_projection(
    dtype, numerical_device
):
    # M5 Max / MLX 0.32.4: Metal reduction order differs by up to 8.4e-4
    # in fp32. Keep the CPU algebraic bound and allow 1e-3 only on Metal.
    atol = 2e-6 if numerical_device == mx.cpu else 1e-3
    if dtype == mx.bfloat16:
        atol = 0.0078125
    _, draft = bench.tiny_setup()
    draft.observe_many(mx.random.normal((1, 3, 96)), 0)
    x = mx.random.normal((5, 32)).astype(dtype)
    q = mx.random.normal((5, 2, 32)).astype(dtype)
    kv = mx.random.normal((8, 32)).astype(dtype)
    sink = mx.array([0.5, -0.25])
    idx = mx.broadcast_to(mx.arange(8)[None, None], (1, 5, 8))
    equal(
        dspark.draft_attention(q, kv, sink),
        sparse_attn(q[None], kv[None], sink, idx, 32**-0.5)[0],
        atol,
    )
    expected = draft.attention("mtp.0.attn", x, 0)
    serving._install_vectorized_mtp_attention(draft)
    equal(
        draft.attention("mtp.0.attn", x, 0),
        expected,
        atol,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("normalize,topk,limit", [(True, 2, 10.0), (False, 1, 0.0)])
@pytest.mark.usefixtures("cpu_device")
def test_expert_routing_and_packed_moe_match_target(dtype, normalize, topk, limit):
    model, draft = bench.tiny_setup()
    draft.c.update(
        norm_topk_prob=normalize, dspark_num_experts_per_tok=topk, swiglu_limit=limit
    )
    x = (mx.random.normal((1, 32)) * 20).astype(dtype)
    base = "mtp.0.ffn"
    expert = SharedExpert(32, 32, limit)
    for name in ("w1", "w2", "w3"):
        getattr(expert, name).weight = draft.w.read(
            f"{base}.shared_experts.{name}.weight"
        ).astype(dtype)
        draft.w._arrays[f"{base}.shared_experts.{name}.weight"] = getattr(
            expert, name
        ).weight
    equal(draft.adapter.expert(base + ".shared_experts", x), expert(x))
    equal(
        draft.adapter.expert(base + ".shared_experts", x, 0.37),
        expert(x).astype(mx.float32) * 0.37,
    )
    gate = Gate(model.args)
    gate.topk, gate.norm_topk_prob = topk, normalize
    gate.weight, gate.bias = (
        draft.w.read(base + ".gate.weight"),
        draft.w.read(base + ".gate.bias"),
    )
    weights, indices = gate(x)
    reference = mx.zeros_like(x).astype(mx.float32)
    for i, w in zip(indices.tolist()[0], weights.tolist()[0]):
        reference += (
            draft.adapter.expert(f"{base}.experts.{i}", x).astype(mx.float32) * w
        )
    reference = (reference + expert(x).astype(mx.float32)).astype(dtype)
    expected = draft.adapter.moe(base, x)
    equal(expected, reference, 1e-4 if dtype == mx.float32 else 0.03125)
    assert serving._install_packed_mtp_moe(draft) > 0
    equal(
        draft.adapter.moe(base, x), expected, 1e-4 if dtype == mx.float32 else 0.03125
    )


@pytest.mark.usefixtures("cpu_device")
def test_observe_many_matches_serial_windows_and_one_eval(monkeypatch):
    # Metal GEMM/GEMV rounding can cross an 8-bit cache quantization boundary
    # (observed max difference 0.125); bit-exact batching is a CPU contract.
    _, serial = bench.tiny_setup()
    _, batched = bench.tiny_setup()
    hidden = mx.random.normal((1, 12, 96))
    for i in range(12):
        serial.observe(hidden[:, i], i)
    evals = []
    original = mx.eval
    monkeypatch.setattr(
        mx, "eval", lambda *args: (evals.append(1), original(*args))[-1]
    )
    batched.observe_many(hidden, 0)
    assert len(evals) == 1
    assert batched.position == serial.position == 11
    for layer in serial.layers:
        assert len(batched.windows[layer]) == 8
        equal(
            mx.concatenate(batched.windows[layer]),
            mx.concatenate(serial.windows[layer]),
        )
    with pytest.raises(ValueError, match="contiguous"):
        batched.observe_many(hidden, 14)
    for shape in ((2, 2, 96), (1, 0, 96)):
        with pytest.raises(ValueError, match="one nonempty stream"):
            batched.observe_many(mx.zeros(shape), 12)
    assert batched.position == 11


def test_markov_chain_keeps_device_dependencies_and_checks_finite(monkeypatch):
    _, draft = bench.tiny_setup()
    with pytest.raises(ValueError, match="two verified"):
        draft.propose(1)
    draft.observe_many(mx.random.normal((1, 2, 96)), 0)
    proposals, confidence = draft.propose(4)
    assert len(proposals) == 6 and proposals[0] == 4
    assert confidence.shape == (5, 1)
    again, _ = draft.propose(4)
    assert proposals == again
    original = draft.w.linear

    def broken(base, x):
        result = original(base, x)
        return result * float("nan") if "confidence_head" in base else result

    monkeypatch.setattr(draft.w, "linear", broken)
    with pytest.raises(FloatingPointError, match="non-finite"):
        draft.propose(4)


@pytest.mark.parametrize(
    "candidate,expected,eos,emitted,result",
    [
        ([1], [], 9, False, ([1], None, False, 0)),
        ([1, 2, 3], [2, 3], 9, False, ([1, 2, 3], None, False, 2)),
        ([1, 2, 3], [4, 3], 9, True, ([4], 1, False, 0)),
        ([1, 2, 3], [2, 4], 4, False, ([1, 2, 4], 2, True, 1)),
        ([1, 2, 3], [2, 3], 2, True, ([2], None, True, 1)),
    ],
)
def test_match_one_readback(candidate, expected, eos, emitted, result):
    logits = mx.full((1, max(1, len(expected)), 10), -1.0)
    for i, token in enumerate(expected):
        logits[:, i, token] = 1
    assert serving._match(candidate, logits, eos, emitted) == result


@pytest.mark.parametrize("seed", [7, 11, 42])
def test_verify_chunk_vs_sequential_and_hidden_handoff(seed, numerical_device):
    dtype = mx.float32
    model, _ = bench.tiny_setup(seed)
    model.set_dtype(dtype)
    batch_cache, serial_cache = (
        model.make_cache(max_seq_len=24),
        model.make_cache(max_seq_len=24),
    )
    prompt = mx.array([[1, 2, 3]])
    model(prompt, batch_cache)
    model(prompt, serial_cache)
    ids = mx.array([[4, 5, 6, 7, 8]])
    batch, hidden = model(
        ids, batch_cache, return_dspark_hidden=True, enable_rollback=True
    )
    sequential, states = [], []
    for i in range(5):
        logits, state = model(
            ids[:, i : i + 1], serial_cache, return_dspark_hidden=True
        )
        sequential.append(logits)
        states.append(state)
    sequential = mx.concatenate(sequential, axis=1)
    error = float(mx.max(mx.abs(batch - sequential)))
    print(
        f"verify seed={seed} dtype={dtype} max_abs={error} argmax_equal={bool(mx.all(mx.argmax(batch, axis=-1) == mx.argmax(sequential, axis=-1)))}"
    )
    # M5 Max / MLX 0.32.4: fp32 Metal chunk/serial errors are 0.0062--0.0087.
    # Allow 0.02 for accumulated reduction-order error, while requiring the
    # same greedy tokens. CPU retains the tight numerical regression bound.
    atol = 2e-5 if numerical_device == mx.cpu else 0.02
    assert error <= atol
    assert bool(mx.all(mx.argmax(batch, axis=-1) == mx.argmax(sequential, axis=-1)))
    equal(hidden, mx.concatenate(states, axis=1), atol)
    # Capture is mean over copies BEFORE the selected block and its normalization.
    model.args.dspark_target_layer_ids = (2, 0, 1)
    cache = model.make_cache(max_seq_len=24)
    captured = []
    original_layers = model.layers

    class Capture:
        def __init__(self, block):
            self.block = block
            self.engram = block.engram

        def __call__(self, h, *args):
            captured.append(mx.mean(h, axis=2))
            return self.block(h, *args)

    model.layers = [Capture(block) for block in original_layers]
    _, handoff = model(prompt, cache, return_dspark_hidden=True)
    equal(handoff, mx.concatenate([captured[i] for i in (2, 0, 1)], axis=-1))


@pytest.mark.parametrize("mismatch", [None, 1, 2, 3, 4])
def test_production_stream_greedy_rollback_observes_only_committed(mismatch):
    model, draft = bench.tiny_setup()
    observed = []
    original_observe = draft.observe_many

    def observe(hidden, position):
        observed.extend(range(position, position + hidden.shape[1]))
        original_observe(hidden, position)

    draft.observe_many = observe
    # An oracle draft makes acceptance/rejection deterministic for every boundary.
    prefix = [1, 2, 3, 4]
    plain, _ = bench.plain_decode(model, prefix, 18)

    def propose(seed):
        position = draft.position + 1 - len(prefix)
        candidates = [seed] + plain[position + 1 : position + 5]
        if mismatch is not None and len(candidates) > mismatch:
            candidates[mismatch] = (candidates[mismatch] + 1) % 32
        return candidates, mx.ones((5, 1))

    draft.propose = propose
    runtime = serving.DSparkRuntime(draft, "synthetic", "synthetic", "synthetic")
    output, stats = bench.speculative_decode(
        serving, model, bench.TinyTokenizer(), runtime, prefix, 18
    )
    assert output == plain
    assert observed == list(range(len(observed)))
    assert stats["rounds"] >= 1
    assert stats["mean_round_ms"]["host_ms"] >= 0
    # Last emitted replacement may not be in cache yet; no rejected suffix observed.
    assert len(observed) <= len(prefix) + len(output)


def test_observe_provider_fallback():
    observed = []
    serving._observe_verified(
        SimpleNamespace(observe=lambda h, p: observed.append((h.tolist(), p))),
        mx.array([[[1], [2]]]),
        3,
    )
    assert observed == [([[1]], 3), ([[2]], 4)]


def test_benchmark_main_tiny_and_checkpoint_provider(monkeypatch, capsys):
    assert bench.main(["--tokens", "2", "--repeats", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["_schema_version"] == 1
    assert len(result["_methodology_hash"]) == 64
    model, draft = bench.tiny_setup()
    runtime = serving.DSparkRuntime(draft, "synthetic", "synthetic", "synthetic")
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model, bench.TinyTokenizer(), runtime

    monkeypatch.setattr(serving, "load_product_runtime", load)
    assert (
        bench.main(
            [
                "--model",
                "checkpoint",
                "--draft",
                "sidecar",
                "--tokens",
                "1",
                "--repeats",
                "1",
                "--label",
                "old",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "checkpoint" and result["label"] == "old"
    assert calls[0][0] == ("checkpoint", "sidecar")
    with pytest.raises(SystemExit):
        bench.main(["--tokens", "0"])


@pytest.mark.usefixtures("cpu_device")
def test_packed_batch_proposals_match_serial_components():
    _, draft = bench.tiny_setup()
    draft.observe_many(mx.random.normal((1, 3, 96)), 0)
    proposals, confidence = draft.propose(4)
    x = mx.random.normal((5, 32))
    serial = mx.concatenate([draft.adapter.moe("mtp.0.ffn", row[None]) for row in x])
    serving._install_vectorized_mtp_attention(draft)
    serving._install_packed_mtp_moe(draft)
    equal(draft.adapter.moe("mtp.0.ffn", x), serial, 1e-6)
    actual, actual_confidence = draft.propose(4)
    assert proposals == actual
    equal(confidence, actual_confidence, 1e-6)


def test_benchmark_script_entrypoint(monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(
        sys, "argv", [bench.__file__, "--tokens", "1", "--repeats", "1"]
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(bench.__file__, run_name="__main__")
    assert result.value.code == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["greedy_equal"]


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("projection_dtype", [mx.float32, mx.bfloat16])
@pytest.mark.usefixtures("cpu_device")
def test_full_native_moe_matches_draft_with_identical_quantized_weights(
    dtype, projection_dtype
):
    import mlx.nn as nn

    model, draft = bench.tiny_setup()
    native = model.layers[0].ffn
    nn.quantize(native.experts, group_size=32, bits=4)
    base = "mtp.0.ffn"
    for key in draft.w._arrays:
        if (
            key.startswith(base)
            and ".gate." not in key
            and draft.w._arrays[key].dtype != mx.uint32
        ):
            draft.w._arrays[key] = draft.w._arrays[key].astype(projection_dtype)
    for name in ("weight", "bias"):
        setattr(native.gate, name, draft.w.read(f"{base}.gate.{name}"))
    for target, source in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        for suffix in ("weight", "scales", "biases"):
            value = mx.stack(
                [
                    draft.w.read(f"{base}.experts.{i}.{source}.{suffix}")
                    for i in range(4)
                ]
            )
            getattr(native.experts, target)[suffix] = value
    for projection in ("w1", "w2", "w3"):
        getattr(native.shared_experts, projection).weight = draft.w.read(
            f"{base}.shared_experts.{projection}.weight"
        )
    x = mx.random.normal((5, 32)).astype(dtype)
    expected = native(x)
    scalar = mx.concatenate([draft.adapter.moe(base, row[None]) for row in x])
    equal(scalar, expected, 1e-6 if dtype == mx.float32 else 0.0078125)
    serving._install_packed_mtp_moe(draft)
    equal(
        draft.adapter.moe(base, x), expected, 1e-6 if dtype == mx.float32 else 0.0078125
    )


@pytest.mark.parametrize("input_dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("weight_dtype", [mx.float32, mx.bfloat16])
@pytest.mark.usefixtures("cpu_device")
def test_dense_linear_matches_native_without_premature_input_rounding(
    input_dtype, weight_dtype
):
    import mlx.nn as nn

    _, draft = bench.tiny_setup()
    x = mx.random.normal((5, 32)).astype(input_dtype)
    linear = nn.Linear(32, 32, bias=False)
    linear.weight = mx.random.normal((32, 32)).astype(weight_dtype)
    draft.w._arrays["dense.weight"] = linear.weight
    equal(draft.w.linear("dense", x), linear(x))


@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize("serial_rows,chunk_rows", [(5, 6), (3, 12), (8, 1)])
def test_observe_many_appends_into_populated_window(serial_rows, chunk_rows):
    _, serial = bench.tiny_setup()
    _, batched = bench.tiny_setup()
    hidden = mx.random.normal((1, serial_rows + chunk_rows, 96))
    for i in range(serial_rows):
        serial.observe(hidden[:, i], i)
        batched.observe(hidden[:, i], i)
    for i in range(serial_rows, serial_rows + chunk_rows):
        serial.observe(hidden[:, i], i)
    batched.observe_many(hidden[:, serial_rows:], serial_rows)
    assert batched.position == serial.position == serial_rows + chunk_rows - 1
    for layer in serial.layers:
        assert len(batched.windows[layer]) == len(serial.windows[layer])
        equal(
            mx.concatenate(batched.windows[layer]),
            mx.concatenate(serial.windows[layer]),
        )


@pytest.mark.parametrize("eos_index", [0, 2, 6])
@pytest.mark.parametrize("mismatch", [None, 1, 3])
def test_production_stream_stops_at_target_eos_with_draft(eos_index, mismatch):
    model, draft = bench.tiny_setup()
    prefix = [1, 2, 3, 4]
    plain, _ = bench.plain_decode(model, prefix, 18)
    eos = plain[eos_index]
    # first occurrence, so a repeated token cannot stop the stream earlier
    assert plain.index(eos) == eos_index

    def propose(seed):
        position = draft.position + 1 - len(prefix)
        candidates = [seed] + plain[position + 1 : position + 5]
        if mismatch is not None and len(candidates) > mismatch:
            candidates[mismatch] = (candidates[mismatch] + 1) % 32
        return candidates, mx.ones((5, 1))

    draft.propose = propose

    class EosTokenizer(bench.TinyTokenizer):
        eos_token_id = eos
        all_special_ids = [eos]

    runtime = serving.DSparkRuntime(draft, "synthetic", "synthetic", "synthetic")
    output = [
        chunk.token
        for chunk in serving.stream_generate(
            model, EosTokenizer(), prefix, runtime=runtime, max_tokens=18
        )
    ]
    assert output == plain[: plain.index(eos) + 1]
