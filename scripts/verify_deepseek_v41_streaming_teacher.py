#!/usr/bin/env python3
"""Bounded CPU faithfulness check from cached source; no downloads or model writes.

Truncates depth and routed-expert axis, retains full attention/HC/Engram/vocabulary.
This verifies scheduling with real weights, not full-model release accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def errors(actual, expected):
    import mlx.core as mx

    a, b = [np.asarray(x.astype(mx.float32)) for x in (actual, expected)]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite comparison")
    delta = np.abs(a - b)
    return dict(
        max_abs=float(delta.max()),
        max_rel=float((delta / np.maximum(np.abs(b), 1e-8)).max()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--tokens", type=int, default=4)
    a = p.parse_args()
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.engram import EngramHasher
    from rapid_mlx.models.deepseek_v41_native.hyper_connections import hc_pre_norm
    from rapid_mlx.models.deepseek_v41_native.load import load_token_map
    from rapid_mlx.models.deepseek_v41_native.model import Block, Model
    from scripts.deepseek_v41_streaming_teacher import SourceWeights, stream_forward

    mx.set_default_device(mx.cpu)
    if not (2 <= a.layers <= 4 and 6 <= a.experts <= 16 and 1 <= a.tokens <= 8):
        raise ValueError("bounded check supports 2-4 layers, 6-16 experts, 1-8 tokens")

    class TruncatedSource(SourceWeights):
        def tensor(self, name):
            w = super().tensor(name)
            if ".ffn.gate." in name:
                w = w[: a.experts]
            return w

    weights = TruncatedSource(a.source)
    original_args = weights.args
    engram = [
        (i, n)
        for i, n in zip(
            original_args.engram_layer_ids, original_args.engram_num_embeddings
        )
        if i < a.layers
    ]
    args = replace(
        original_args,
        n_layers=a.layers,
        n_routed_experts=a.experts,
        engram_layer_ids=tuple(i for i, _ in engram),
        engram_num_embeddings=tuple(n for _, n in engram),
        dspark_target_layer_ids=(),
    )
    weights.args = args
    a.output.parent.mkdir(parents=True, exist_ok=True)
    token_map = load_token_map(str(a.source), args, cache_dir=str(a.output.parent))
    hasher = EngramHasher(args, token_map)
    native = Model(replace(args, engram_num_embeddings=tuple(1 for _ in engram)))
    native.engram_hasher = hasher
    native.layers = [weights.layer(i) for i in range(a.layers)]
    native.embed.weight = weights.tensor("embed.weight")
    native.norm.weight = weights.tensor("norm.weight")
    native.head.weight = weights.tensor("head.weight")
    native.eval()
    mx.eval(native.parameters())
    # Two independent slices, with distinct order/history. IDs are real vocabulary IDs.
    tokens = [
        np.arange(10, 10 + a.tokens, dtype=np.int32),
        np.arange(20, 20 + a.tokens, dtype=np.int32)[::-1].copy(),
    ]
    snapshots, expected = [], []
    current = {}
    original_call = Block.__call__

    def capture(self, *xs):
        h, pre = original_call(self, *xs)
        mx.eval(h, pre)
        current[self.layer_id] = (mx.array(h), mx.array(pre))
        return h, pre

    Block.__call__ = capture
    try:
        for ids in tokens:
            y = native(mx.array(ids[None]), native.make_cache(max_seq_len=len(ids)))
            mx.eval(y)
            snapshots.append(dict(current))
            expected.append(y)
    finally:
        Block.__call__ = original_call
    # Release resident blocks before streaming; expected small tensors remain.
    native.layers = []
    gc.collect()
    mx.clear_cache()
    report = dict(
        source_revision=a.source.resolve().name,
        layers=a.layers,
        experts=a.experts,
        source_experts=original_args.n_routed_experts,
        dim=args.dim,
        vocab_size=args.vocab_size,
        tokens_per_slice=a.tokens,
        slices=len(tokens),
        device="CPU",
        arithmetic="source BF16 weights; F32 CPU GatherMM with original output casts",
        relative_denominator_floor=1e-8,
        hidden=[],
        pre_mix=[],
        logits=[],
    )

    def check(domain, layer, h, pre):
        report["hidden"].append(
            dict(slice=domain, layer=layer, **errors(h, snapshots[domain][layer][0]))
        )
        report["pre_mix"].append(
            dict(slice=domain, layer=layer, **errors(pre, snapshots[domain][layer][1]))
        )

    result = stream_forward(
        args,
        tokens,
        lambda: native.embed,
        weights.layer,
        hasher=hasher,
        hidden_observer=check,
        observer=lambda *xs: None,
        progress=lambda layer, t: print(
            json.dumps(dict(layer=layer, seconds=t)), flush=True
        ),
    )
    for seq, want in zip(result, expected):
        got = native.head(
            hc_pre_norm(seq.h, seq.pre, native.norm.weight, args.norm_eps).astype(
                mx.float32
            )
        )
        report["logits"].append(errors(got, want))
    report["passed"] = all(
        r["max_abs"] == 0 for k in ("hidden", "pre_mix", "logits") for r in report[k]
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
