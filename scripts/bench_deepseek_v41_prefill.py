#!/usr/bin/env python3
"""Compare bounded native prefill against the original full-logit path."""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from rapid_mlx.models.deepseek_v41_native.dspark import DSpark
from rapid_mlx.models.deepseek_v41_native.model import Model
from rapid_mlx.models.deepseek_v41_native.serving import DSparkRuntime, stream_generate
from scripts.bench_metadata import format_bench_json


def tiny_model():
    """Exercise shared ratio-2/ratio-1 caches, index keys, ring wrap and Engram."""
    mx.random.seed(41)
    args = ModelArgs(
        vocab_size=64,
        dim=32,
        n_layers=3,
        moe_inter_dim=32,
        n_heads=1,
        head_dim=32,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=1,
        window_size=128,
        n_routed_experts=2,
        n_activated_experts=1,
        compress_ratios=(2, 2, 1),
        kv_source_layers=(0, 2),
        index_source_layers=(0, 2),
        index_n_heads=1,
        index_head_dim=32,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        engram_layer_ids=(1,),
        engram_num_embeddings=(12,),
        engram_vocab_size=4,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_max_ngram_size=3,
        dspark_target_layer_ids=(0, 1, 2),
    )
    model = Model(args, token_map=np.arange(64))
    # Nonzero lookup rows exercise the gated Engram contribution too.
    model.layers[1].engram.embed = nn.Embedding(12, 32)
    model.eval_interval = 40
    mx.eval(model.parameters())
    return model


def tiny_drafter(seed=42, quantized=False):
    """Use the production observation math with small synthetic projections."""
    mx.random.seed(seed)
    projections = {"mtp.0.main_proj": nn.Linear(96, 32, bias=False)}
    for stage in range(3):
        projections[f"mtp.{stage}.attn.wkv"] = nn.Linear(32, 32, bias=False)
    if quantized:
        projections = {
            name: layer.to_quantized(group_size=32, bits=4)
            for name, layer in projections.items()
        }
    weights = SimpleNamespace(
        entries={f"mtp.{stage}.weight": {} for stage in range(3)},
        linear=lambda name, x: projections[name](x),
        read=lambda name: mx.ones((32,)),
    )
    config = dict(
        dspark_block_size=5,
        sliding_window=128,
        qk_rope_head_dim=8,
        rope_theta=10000,
        rms_norm_eps=1e-20,
    )
    return DSpark(SimpleNamespace(w=weights, c=config), pin_weights=False)


def prefill(model, ids, *, optimized, drafter=None):
    """Isolate prefill, excluding decode, allocation reuse and tokenization."""
    cache = model.make_cache(max_seq_len=len(ids) + 40)
    hidden_chunks = []
    for start in range(0, len(ids), 512):
        logits, hidden = model(
            mx.array([ids[start : start + 512]]),
            cache,
            last_logit_only=False,
            return_dspark_hidden=True,
        )
        if optimized:
            state = []
            for layer in cache.layers:
                state.extend(
                    value
                    for value in (layer.win_kv, layer.comp_kv, layer.index_k)
                    if value is not None
                )
                if layer.comp_state is not None:
                    state.extend(
                        (layer.comp_state.kv_state, layer.comp_state.score_state)
                    )
            if start + 512 >= len(ids):
                mx.eval(logits, hidden, state)
            else:
                mx.eval(hidden, state)
        else:
            mx.eval(logits, hidden)
        hidden_chunks.append(hidden)
        if drafter is not None:
            if optimized:
                drafter.observe_many(hidden, start)
            else:
                for index in range(hidden.shape[1]):
                    drafter.observe(hidden[:, index], start + index)
    return logits[:, -1], cache, mx.concatenate(hidden_chunks, axis=1)


def profile_prefill(model, ids, optimized):
    counts = {"eval_calls": 0, "graph_nodes_at_eval": 0}
    evaluate = mx.eval

    def counted(*values):
        counts["eval_calls"] += 1
        graph = io.StringIO()
        mx.export_to_dot(graph, *values)
        counts["graph_nodes_at_eval"] += sum(
            "label" in line and "->" not in line
            for line in graph.getvalue().splitlines()
        )
        evaluate(*values)

    with patch.object(mx, "eval", counted):
        prefill(model, ids, optimized=optimized, drafter=tiny_drafter())
    counts["vocab_head_rows"] = ((len(ids) - 1) % 512 + 1) if optimized else len(ids)
    return counts


def state_digest(value):
    """Fingerprint every cache field, including inactive ring and partial slots."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, mx.array):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(np.array(item.view(mx.uint8)).tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key, child in sorted(item.items()):
                digest.update(str(key).encode())
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__"):
            visit(vars(item))
        else:
            digest.update(str(item).encode())

    visit(value)
    return digest.hexdigest()


def benchmark(model, lengths, repeats, drafter_factory=None, processor=None):
    rows = []
    for length in lengths:
        ids = (np.arange(length) % model.args.vocab_size).tolist()
        outputs = []
        for optimized in (False, True):
            times, ttfts = [], []
            for iteration in range(repeats + 1):
                draft = drafter_factory() if drafter_factory else None
                begin = time.perf_counter()
                logits, cache, hidden = prefill(
                    model, ids, optimized=optimized, drafter=draft
                )
                elapsed = time.perf_counter() - begin
                cache_digest = state_digest(cache)
                draft_digest = (
                    state_digest(draft.windows) if draft is not None else None
                )
                if iteration:
                    times.append(elapsed)
                if draft is not None and processor is not None:
                    # Reusing the drafter is safe: stream_generate calls
                    # runtime.reset() (position -1, empty windows) first.
                    runtime = DSparkRuntime(draft, "local", "local", "local")
                    begin = time.perf_counter()
                    stream = stream_generate(
                        model,
                        processor,
                        ids,
                        runtime=runtime,
                        max_tokens=32,
                        ignore_eos=True,
                        optimize_prefill=optimized,
                    )
                    next(stream)
                    ttft = time.perf_counter() - begin
                    stream.close()
                    if iteration:
                        ttfts.append(ttft)
            outputs.append(
                (
                    np.array(logits),
                    np.array(hidden.astype(mx.float32)),
                    cache_digest,
                    draft_digest,
                )
            )
            rows.append(
                dict(
                    tokens=length,
                    path="optimized" if optimized else "baseline",
                    prefill_seconds=statistics.median(times),
                    prefill_tok_s=length / statistics.median(times),
                    ttft_seconds=statistics.median(ttfts) if ttfts else None,
                    **(
                        profile_prefill(model, ids, optimized)
                        if processor is None
                        else {}
                    ),
                )
            )
            del cache, hidden
        for old, new in zip(outputs[0][:2], outputs[1][:2]):
            np.testing.assert_array_equal(old, new)
        if outputs[0][2:] != outputs[1][2:]:
            raise AssertionError("prefill cache or drafter state differs")
    return rows


def main(argv=None):
    """Run the benchmark; tiny mode's CPU default device is restored on exit."""
    previous = mx.default_device()
    try:
        return _main(argv)
    finally:
        mx.set_default_device(previous)


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--drafter", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.drafter and not args.model:
        parser.error("--drafter requires --model")
    if args.model:
        from transformers import PreTrainedTokenizerFast

        from rapid_mlx.models.deepseek_v41_native.load import load
        from rapid_mlx.models.deepseek_v41_native.serving import load_product_runtime

        draft_path = args.drafter or args.model / "dspark"
        factory = None
        if draft_path.is_dir():
            model, processor, runtime = load_product_runtime(
                str(args.model),
                str(draft_path),
                target_revision="local",
                mtp_revision="local",
                mixed_target_precision=True,
            )

            def factory():
                runtime.reset()
                return runtime.drafter
        else:
            model, _ = load(str(args.model), lazy=False)
            model.eval_interval = 40
            processor = PreTrainedTokenizerFast.from_pretrained(
                args.model,
                local_files_only=True,
            )
        lengths = [1024, 4096]
    else:
        mx.set_default_device(mx.cpu)
        model, processor, factory = tiny_model(), None, tiny_drafter
        lengths = [128, 513, 1024]
    rows = benchmark(model, lengths, args.repeats, factory, processor)
    print(
        format_bench_json(
            dict(
                mode="checkpoint" if args.model else "tiny-cpu",
                platform=platform.platform(),
                python=platform.python_version(),
                mlx=mx.__version__,
                repeats=args.repeats,
                chunk_size=512,
                rows=rows,
                logits_and_hidden_bit_exact=True,
                cache_and_drafter_bit_exact=True,
                ttft_note="Includes first verification forward; null when no real drafter is loaded.",
            ),
            __file__,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
