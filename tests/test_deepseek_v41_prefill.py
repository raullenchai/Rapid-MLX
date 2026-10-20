"""CPU-verifiable prefill equivalence with production cache and observation math."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native import serving
from scripts import bench_deepseek_v41_prefill as bench


@pytest.fixture(autouse=True)
def cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def assert_state_equal(left, right):
    if isinstance(left, (mx.array, np.ndarray)):
        np.testing.assert_array_equal(np.array(left), np.array(right))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif hasattr(left, "__dict__"):
        assert_state_equal(vars(left), vars(right))
    else:
        assert left == right


@pytest.mark.parametrize("length", [1, 2, 127, 128, 129, 511, 512, 513, 1100, 2049])
def test_prefill_cache_logits_hidden_and_decode_exact(length):
    model = bench.tiny_model()
    ids = (np.arange(length) % 64).tolist()
    old = bench.prefill(model, ids, optimized=False)
    new = bench.prefill(model, ids, optimized=True)
    assert_state_equal(old, new)
    assert np.isfinite(np.array(new[0])).all()
    # No draft: identical 32-step greedy continuation and every cache field.
    logits_old, cache_old, _ = old
    logits_new, cache_new, _ = new
    for _ in range(32):
        token_old = int(mx.argmax(logits_old))
        assert token_old == int(mx.argmax(logits_new))
        logits_old = model(mx.array([[token_old]]), cache_old)
        logits_new = model(mx.array([[token_old]]), cache_new)
        mx.eval(logits_old, logits_new)
        assert_state_equal(logits_old, logits_new)
    assert_state_equal(cache_old, cache_new)


@pytest.mark.parametrize("seed", [1, 42, 73])
@pytest.mark.parametrize("quantized", [False, True])
def test_batched_observe_exact_windows_and_contiguity(seed, quantized):
    old = bench.tiny_drafter(seed, quantized)
    new = bench.tiny_drafter(seed, quantized)
    mx.random.seed(seed + 1)
    position = 0
    for count in (1, 2, 127, 128, 129, 511, 512, 513):
        hidden = mx.random.normal((1, count, 96))
        mx.eval(hidden)
        for i in range(count):
            old.observe(hidden[:, i], position + i)
        new.observe_many(hidden, position)
        assert_state_equal(old.windows, new.windows)
        assert old.position == new.position == position + count - 1
        position += count
    before = new.position
    with pytest.raises(ValueError, match="contiguous"):
        new.observe_many(mx.zeros((1, 2, 96)), 0)
    for shape in ((2, 2, 96), (1, 0, 96)):
        with pytest.raises(ValueError, match="nonempty"):
            new.observe_many(mx.zeros(shape), before + 1)
    assert new.position == before


class TinyProcessor:
    eos_token_id = -1
    all_special_ids = []
    clean_up_tokenization_spaces = False

    def encode(self, _text, **_kwargs):
        return [1, 2]

    def decode(self, tokens, **_kwargs):
        return " ".join(map(str, tokens))


def test_stream_prefill_and_draft_enabled_greedy_32_steps(monkeypatch):
    # Production observe/observe_many and real target rollback. The synthetic
    # proposal deliberately causes rejections to exercise continuation state.
    model = bench.tiny_model()
    processor = TinyProcessor()
    outputs, states, captures = [], [], []
    for optimized in (False, True):
        draft = bench.tiny_drafter()
        seen = []
        original_many = draft.observe_many
        original_one = draft.observe

        def observe_many(hidden, start, seen=seen, original_many=original_many):
            seen.extend(np.array(hidden[0]))
            return original_many(hidden, start)

        def observe(hidden, position, seen=seen, original_one=original_one):
            seen.extend(np.array(hidden))
            return original_one(hidden, position)

        draft.observe_many = observe_many
        draft.observe = observe
        draft.propose = lambda seed: ([seed, (seed + 1) % 64, 3, 4, 5], None)
        runtime = serving.DSparkRuntime(draft, "tiny", "tiny", "tiny")
        chunks = list(
            serving.stream_generate(
                model,
                processor,
                (np.arange(513) % 64).tolist(),
                runtime=runtime,
                max_tokens=32,
                ignore_eos=True,
                optimize_prefill=optimized,
            )
        )
        outputs.append([chunk.token for chunk in chunks])
        states.append(draft.windows)
        captures.append(seen)
    assert len(outputs[0]) == 32
    assert outputs[0] == outputs[1]
    assert_state_equal(states[0], states[1])
    assert_state_equal(captures[0], captures[1])


def test_serving_flushes_cache_and_batches_observation(monkeypatch):
    model = bench.tiny_model()
    calls = []

    class SpyModel:
        make_cache = model.make_cache

        def __call__(self, ids, cache, **kwargs):
            calls.append((ids.shape[1], kwargs["last_logit_only"]))
            return model(ids, cache, **kwargs)

    draft = bench.tiny_drafter()

    # Stop precisely at the prefill/decode boundary.
    def stop(seed):
        raise RuntimeError("prefill complete")

    draft.propose = stop
    runtime = serving.DSparkRuntime(draft, "tiny", "tiny", "tiny")
    with pytest.raises(RuntimeError, match="prefill complete"):
        next(
            serving.stream_generate(
                SpyModel(),
                TinyProcessor(),
                [1] * 513,
                runtime=runtime,
                max_tokens=1,
            )
        )
    assert calls == [(512, False), (1, False)]
    assert draft.position == 512


def test_tiny_bench_json_and_counts(capsys, monkeypatch):
    model = bench.tiny_model()
    ids = [1] * 513
    old = bench.profile_prefill(model, ids, False)
    new = bench.profile_prefill(model, ids, True)
    assert old["eval_calls"] == 515
    assert new["eval_calls"] == 4
    assert old["vocab_head_rows"] == 513
    assert new["vocab_head_rows"] == 1
    assert new["graph_nodes_at_eval"] < old["graph_nodes_at_eval"]
    # Keep CLI smoke coverage inexpensive while exercising real benchmark work.
    benchmark = bench.benchmark
    monkeypatch.setattr(
        bench, "benchmark", lambda m, ls, r, f, p: benchmark(m, [2], r, f, p)
    )
    assert bench.main(["--repeats", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["_schema_version"] == 1
    assert payload["logits_and_hidden_bit_exact"]
    assert len(payload["rows"]) == 2


@pytest.mark.parametrize("draft_layout", ["none", "colocated", "explicit"])
def test_checkpoint_bench_modes_with_synthetic_runtime(
    tmp_path, monkeypatch, capsys, draft_layout
):
    import importlib

    from transformers import PreTrainedTokenizerFast

    native_load = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
    model = bench.tiny_model()
    processor = TinyProcessor()
    draft = bench.tiny_drafter()
    draft.propose = lambda seed: ([seed, 1, 2, 3, 4], None)
    runtime = serving.DSparkRuntime(draft, "tiny", "tiny", "tiny")
    loads = []

    def load_runtime(target, mtp, **kwargs):
        loads.append((target, mtp))
        return model, processor, runtime

    monkeypatch.setattr(serving, "load_product_runtime", load_runtime)
    monkeypatch.setattr(native_load, "load", lambda *a, **kw: (model, model.args))
    monkeypatch.setattr(
        PreTrainedTokenizerFast, "from_pretrained", lambda *a, **kw: processor
    )
    benchmark = bench.benchmark
    requested_lengths = []

    def small_benchmark(m, lengths, repeats, factory, tokenizer):
        requested_lengths.extend(lengths)
        return benchmark(m, [2], repeats, factory, tokenizer)

    monkeypatch.setattr(bench, "benchmark", small_benchmark)
    args = ["--model", str(tmp_path), "--repeats", "1"]
    if draft_layout != "none":
        directory = tmp_path / ("dspark" if draft_layout == "colocated" else "draft")
        directory.mkdir()
        if draft_layout == "explicit":
            args += ["--drafter", str(directory)]
    assert bench.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "checkpoint"
    assert payload["cache_and_drafter_bit_exact"]
    assert requested_lengths == [1024, 4096]
    assert bool(loads) == (draft_layout != "none")
    for row in payload["rows"]:
        if draft_layout == "none":
            assert row["ttft_seconds"] is None
        else:
            assert row["ttft_seconds"] > 0


def test_benchmark_fingerprints_checkpoint_bfloat16_without_numpy_conversion():
    value = mx.array([[1.0, 2.0]], dtype=mx.bfloat16)
    assert bench.state_digest(value) == bench.state_digest(value + mx.zeros_like(value))
    assert bench.state_digest(value) != bench.state_digest(value + 1)
    assert bench.state_digest(value) != bench.state_digest(value.astype(mx.float32))
