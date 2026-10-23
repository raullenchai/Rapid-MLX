"""CPU contracts for layer-major scheduling and raw-source teacher weights."""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from mlx.utils import tree_flatten

from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from rapid_mlx.models.deepseek_v41_native.engram import build_layout
from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_pre_norm
from rapid_mlx.models.deepseek_v41_native.model import Block, Model
from scripts.deepseek_v41_streaming_teacher import (
    SourceWeights,
    install_cpu_gather,
    install_prefill_probe,
    memory_estimate,
    save_scores,
    stream_forward,
)


def tiny_config():
    return dict(
        hidden_size=64,
        vocab_size=32,
        num_hidden_layers=4,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=2,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        hc_mult=4,
        compress_ratios=[2, 2, 1, 1],
        kv_source_layer_ids=[0, 2],
        index_source_layer_ids=[0, 1, 2, 3],
        index_n_heads=2,
        index_head_dim=32,
        index_topk=3,
        candidate_source_layer_id=2,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        sliding_window=4,
        engram_layer_ids=[1, 3],
        engram_vocab_size=7,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_max_ngram_size=3,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
    )


def write_source(root):
    """Complete tiny checkpoint: FP4 experts, FP8 shared, raw FP8 Engram."""
    mx.set_default_device(mx.cpu)
    mx.random.seed(1017)
    config = tiny_config()
    args = ModelArgs.from_dict(config)
    primes, _ = build_layout(args)
    config["engram_num_embeddings"] = primes.reshape(2, -1).sum(-1).tolist()
    config["quantization_config"] = dict(expert_dtype="fp4")
    args = ModelArgs.from_dict(config)
    model = Model(args, token_map=list(range(32)))
    raw = {}
    reverse = dict(gate_proj="w1", up_proj="w3", down_proj="w2")
    for name, value in tree_flatten(model.parameters()):
        if ".ffn.experts." in name:
            proj = reverse[name.split(".")[-2]]
            for e in range(args.n_routed_experts):
                base = name.split(".ffn.experts.")[0] + f".ffn.experts.{e}.{proj}"
                raw[base + ".weight"] = mx.random.randint(
                    0, 256, shape=(64, 32)
                ).astype(mx.uint8)
                raw[base + ".scale"] = mx.full((64, 2), 121, mx.uint8)
        elif ".engram.embed." in name:
            raw[name] = mx.full(
                value.shape, 56 if name.endswith("weight") else 122, mx.uint8
            )
        elif name.endswith(".shared_experts.w1.weight"):
            raw[name] = mx.full(value.shape, 56, mx.uint8)
            raw[name.removesuffix(".weight") + ".scale"] = mx.full(
                (2, 2), 122, mx.uint8
            )
        else:
            if ".hc_" in name:
                value = 0.02 * mx.random.normal(value.shape)
            if name.endswith(".wo_a.weight"):
                value = value.reshape(-1, value.shape[-1])
            # Preserve FP32 correction, norms and HC as the source does.
            raw[name] = value.astype(
                mx.float32
                if ".hc_" in name or name.endswith("bias") or name.endswith("bias_vl")
                else mx.bfloat16
            )
    root.mkdir()
    mx.save_safetensors(str(root / "source.safetensors"), raw)
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map={n: "source.safetensors" for n in raw}))
    )
    return args, raw


def independent_model(args, raw):
    from rapid_mlx.models.deepseek_v41_native.convert import sanitize_group
    from rapid_mlx.models.deepseek_v41_native.load import reshape_grouped_wo_a

    weights = sanitize_group(list(raw), raw)
    weights = reshape_grouped_wo_a(list(weights.items()), args)
    model = Model(args, token_map=list(range(args.vocab_size)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    model.eval()
    for block in model.layers:
        install_cpu_gather(block)
    return model


def error(a, b):
    a = np.asarray(a.astype(mx.float32))
    b = np.asarray(b.astype(mx.float32))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return dict(
        max_abs=float(np.max(np.abs(a - b))),
        max_rel=float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8))),
    )


@pytest.mark.parametrize("length", [8, 70])
def test_source_stream_equals_native_with_all_cross_layer_states(
    tmp_path, monkeypatch, length
):
    args, raw = write_source(tmp_path / "source")
    model = independent_model(args, raw)
    native_hidden = {}
    original = Block.__call__

    def captured(self, *xs):
        h, pre = original(self, *xs)
        mx.eval(h, pre)
        native_hidden[self.layer_id] = (mx.array(h), mx.array(pre))
        return h, pre

    monkeypatch.setattr(Block, "__call__", captured)
    ids = [np.arange(length, dtype=np.int32) % 31 + 1, np.array([7, 2, 1, 8, 9])]
    expected, snapshots = [], []
    for seq in ids:
        expected.append(
            model(mx.array(seq[None]), model.make_cache(max_seq_len=len(seq)))
        )
        mx.eval(expected[-1])
        snapshots.append(dict(native_hidden))
    monkeypatch.setattr(Block, "__call__", original)
    weights = SourceWeights(tmp_path / "source")
    loaded = []

    def loader(i):
        loaded.append(i)
        return weights.layer(i)

    observed = []

    def check(d, layer, h, pre):
        observed.append(error(h, snapshots[d][layer][0]))
        assert error(h, snapshots[d][layer][0])["max_abs"] == 0
        assert error(pre, snapshots[d][layer][1])["max_abs"] == 0

    streams = stream_forward(
        args,
        ids,
        lambda: model.embed,
        loader,
        hasher=model.engram_hasher,
        hidden_observer=check,
        observer=lambda *xs: None,
    )
    for seq, want in zip(streams, expected):
        got = model.head(
            hc_pre_norm(seq.h, seq.pre, model.norm.weight, args.norm_eps).astype(
                mx.float32
            )
        )
        assert error(got, want)["max_abs"] == 0
        assert all(c is None for c in seq.cache.layers)
        assert seq.shared.topk_idxs is not None and seq.shared.candidates is not None
        assert seq.shared.index_src_cache is not None
    assert loaded == [0, 1, 2, 3]
    print("synthetic hidden/logits max abs/rel = 0 / 0")


def test_prefill_probe_preserves_outputs_and_single_token_scores(tmp_path):
    args, raw = write_source(tmp_path / "source")
    model = independent_model(args, raw)
    moe = model.layers[0].ffn
    x = mx.random.normal((1, 7, args.dim)).astype(mx.bfloat16)
    expected = moe(x)
    records = []
    install_prefill_probe(model.layers[0], lambda *v: records.append(v))
    moe._stream_domain = 0
    actual = moe(x)
    assert error(actual, expected)["max_abs"] == 0
    batch = records.pop()
    for i in range(7):
        moe(x[:, i : i + 1])
    for col in (2, 3, 4):
        assert np.allclose(
            np.concatenate([r[col] for r in records]), batch[col], rtol=0.02, atol=0.02
        )


def test_score_output_allocator_contract(tmp_path):
    shape = (2, 2, 3, 4)
    counts = np.ones(shape, np.int64)
    path = tmp_path / "scores.npz"
    save_scores(
        path,
        counts * 2.0,
        counts * 3.0,
        counts * 9.0,
        counts,
        dict(model_revision="test", routing_topk=2, slice_names=["a", "b"]),
        3,
        8,
    )
    with np.load(path) as data:
        assert data["reap_halves"].shape == shape
        assert data["counts_halves_by_slice"].shape == shape
        assert data["split_keep_overlap"].shape == (2, 3)
        assert np.all(data["man_by_slice"] == 3.0)


def test_estimate_is_header_only(tmp_path, monkeypatch):
    from scripts import deepseek_v41_expert_quant_error as qe

    args, _ = write_source(tmp_path / "source")
    w = SourceWeights(tmp_path / "source")
    monkeypatch.setattr(qe, "read_tensor", lambda *_: pytest.fail("body read"))
    small = memory_estimate(w, 2, 8)
    larger = memory_estimate(w, 2, 16)
    assert larger["estimated_peak_bytes"] > small["estimated_peak_bytes"]
    assert small["source_pass_bytes"] > 0


def test_cli_calibration_and_exact_kl_cache(tmp_path, monkeypatch):
    import sys

    from tokenizers import Tokenizer, models, pre_tokenizers

    from scripts import deepseek_v41_streaming_teacher as teacher
    from scripts.evaluate_deepseek_v41_rd import compare

    source = tmp_path / "source"
    args, _ = write_source(source)
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(source / "tokenizer.json"))
    corpus = tmp_path / "slice.txt"
    corpus.write_text(" ".join(f"t{i}" for i in range(1, 10)))
    slices = tmp_path / "slices.json"
    slices.write_text(json.dumps(dict(slices=dict(a=["slice.txt"], b=["slice.txt"]))))
    for mode, output in [
        ("calibrate", tmp_path / "scores.npz"),
        ("reference", tmp_path / "logits"),
    ]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "teacher",
                mode,
                "--source",
                str(source),
                "--slices",
                str(slices),
                "--output",
                str(output),
                "--tokens",
                "8",
                "--keep-experts",
                "3",
                "--cpu",
                "--logit-chunk",
                "3",
            ],
        )
        teacher.main()
    with np.load(tmp_path / "scores.npz") as data:
        assert (
            data["counts_halves_by_slice"].sum()
            == 2 * 8 * args.n_layers * args.n_activated_experts
        )
        assert (
            data["counts_halves_by_slice"][:, 0].sum()
            == 2 * 4 * args.n_layers * args.n_activated_experts
        )
        assert json.loads(str(data["metadata"]))["engram"] == "source_fp8_rowwise"
    metrics = compare(tmp_path / "logits", tmp_path / "logits")
    assert metrics["worst_domain_kl_nats"] == 0.0
    assert metrics["slices"]["a"]["top1_flip_rate"] == 0.0
    # Compare saved chunked head logits against native prefill from identical raw weights.
    from rapid_mlx.models.deepseek_v41_native.load import load_token_map

    native = independent_model(args, mx.load(str(source / "source.safetensors")))
    native.set_token_map(load_token_map(str(source), args, cache_dir=str(tmp_path)))
    ids = np.load(tmp_path / "logits/a-tokens.npy")
    expected = native(mx.array(ids[None]), native.make_cache(max_seq_len=len(ids)))
    saved = np.load(tmp_path / "logits/a-logits.npy")
    chunk_error = error(mx.array(saved), expected[0])
    print("chunked cached head error:", chunk_error)
    assert chunk_error["max_abs"] <= 3e-7


def test_memory_guard_precedes_body_reads(tmp_path, monkeypatch):
    import sys

    from scripts import deepseek_v41_streaming_teacher as teacher

    source = tmp_path / "source"
    write_source(source)
    manifest = tmp_path / "slices.json"
    manifest.write_text(json.dumps(dict(slices=dict(a=["unused.txt"]))))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            "reference",
            "--source",
            str(source),
            "--slices",
            str(manifest),
            "--output",
            str(tmp_path / "output"),
            "--tokens",
            "8",
            "--keep-experts",
            "3",
            "--memory-limit-gib",
            ".001",
        ],
    )
    monkeypatch.setattr(
        teacher.SourceWeights, "tensor", lambda *_: pytest.fail("body read")
    )
    with pytest.raises(ValueError, match="estimated memory"):
        teacher.main()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("chunk", [1, 3, 8, 16])
def test_chunked_prefill_equals_single_pass(tmp_path, chunk):
    """Cross window wraps, partial compression groups, sharing and Engram history."""
    args, raw = write_source(tmp_path / "source")
    model = independent_model(args, raw)
    ids = [np.arange(35, dtype=np.int32) % 31 + 1, np.array([7, 2, 1, 8, 9])]
    weights = SourceWeights(tmp_path / "source")
    expected = stream_forward(
        args, ids, lambda: model.embed, weights.layer, hasher=model.engram_hasher
    )
    chunked = stream_forward(
        args,
        ids,
        lambda: model.embed,
        weights.layer,
        hasher=model.engram_hasher,
        prefill_chunk=chunk,
    )
    for full, part in zip(expected, chunked):

        def head(seq):
            return model.head(
                hc_pre_norm(seq.h, seq.pre, model.norm.weight, args.norm_eps).astype(
                    mx.float32
                )
            )

        lhs, rhs = head(full), head(part)
        delta = error(lhs, rhs)
        print("chunk", chunk, "error", delta)
        np.testing.assert_allclose(
            np.asarray(lhs), np.asarray(rhs), atol=3e-6, rtol=3e-6
        )
        assert np.array_equal(
            np.asarray(mx.argmax(lhs, -1)), np.asarray(mx.argmax(rhs, -1))
        )


def test_compact_reference_cli_and_online_candidate(tmp_path, monkeypatch):
    import importlib
    import sys

    from tokenizers import Tokenizer, models, pre_tokenizers

    from rapid_mlx.models.deepseek_v41_native.load import load_token_map
    from scripts import deepseek_v41_streaming_teacher as teacher
    from scripts import evaluate_deepseek_v41_rd as evaluator

    source = tmp_path / "source"
    args, raw = write_source(source)
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(source / "tokenizer.json"))
    np.save(tmp_path / "a.npy", np.arange(19, dtype=np.int32))
    slices = tmp_path / "slices.json"
    slices.write_text(json.dumps(dict(slices=dict(a=["a.npy"]), ruler="tiny")))
    ref = tmp_path / "ref"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teacher",
            "reference",
            "--cpu",
            "--source",
            str(source),
            "--slices",
            str(slices),
            "--tokens",
            "19",
            "--keep-experts",
            "3",
            "--output",
            str(ref),
            "--prefill-chunk",
            "3",
            "--compact-k",
            "32",
            "--model-revision",
            "test",
        ],
    )
    teacher.main()
    assert not (ref / "a-logits.npy").exists()
    model = independent_model(args, raw)
    model.set_token_map(load_token_map(str(source), args, cache_dir=str(tmp_path)))
    load_module = importlib.import_module("rapid_mlx.models.deepseek_v41_native.load")
    monkeypatch.setattr(load_module, "load", lambda *a, **kw: (model, args))
    cand = tmp_path / "cand"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "collect",
            "--model",
            str(source),
            "--slices",
            str(slices),
            "--tokens",
            "19",
            "--output",
            str(cand),
            "--prefill-chunk",
            "3",
            "--reference",
            str(ref),
            "--model-revision",
            "test",
        ],
    )
    evaluator.main()
    assert not (cand / "a-logits.npy").exists()
    result = evaluator.compare(ref, cand)
    assert result["kl_nats"] < 3e-7
    assert result["top1_agreement"] == 1
    assert result["rulers"]["tiny"]["tokens"] == 19


@pytest.fixture(autouse=True)
def cpu_device():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(old)
