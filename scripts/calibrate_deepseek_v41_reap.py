#!/usr/bin/env python3
"""Collect conservative routing saliency from the cached V4.1 checkpoint.

This calibration executes checkpoint-bundled Python and therefore requires an
explicit trust flag. Saliency is the sum of normalized routing weights for each
selected expert. Two disjoint token halves are retained so a pruning recipe can
be rejected when the keep set is unstable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path
from types import MethodType

import mlx.core as mx
import numpy as np


def _load_runtime(model: Path):
    path = model / "runtime" / "runtime.py"
    spec = importlib.util.spec_from_file_location(
        "rapid_deepseek_v41_calibration_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import checkpoint runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus_tokens(runtime, paths: list[Path], count: int) -> list[int]:
    text = "\n\n".join(path.read_text(errors="replace") for path in paths)
    ids = runtime.tokenizer.encode(text).ids
    if len(ids) < count:
        raise ValueError(f"corpus has {len(ids)} tokens, need {count}")
    return ids[:count]


def _overlap(a: np.ndarray, b: np.ndarray, keep: int) -> float:
    values = []
    for layer in range(a.shape[0]):
        left = set(np.argsort(-a[layer])[:keep].tolist())
        right = set(np.argsort(-b[layer])[:keep].tolist())
        values.append(len(left & right) / keep)
    return float(np.mean(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--keep-experts", type=int, default=336)
    parser.add_argument("--trust-checkpoint-runtime", action="store_true")
    parser.add_argument("corpus", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trust_checkpoint_runtime:
        raise SystemExit(
            "refusing to execute checkpoint-bundled Python without "
            "--trust-checkpoint-runtime"
        )
    model = args.model.resolve()
    module = _load_runtime(model)
    runtime = module.TextRuntime(
        model,
        max_tokens=args.tokens,
        resident_backbone=True,
        execution_mode="compiled",
    )
    layers = int(runtime.c["num_hidden_layers"])
    experts = int(runtime.c["n_routed_experts"])
    if not 1 <= args.keep_experts <= experts:
        raise SystemExit(f"--keep-experts must be in [1, {experts}]")
    ids = _corpus_tokens(runtime, args.corpus, args.tokens)
    saliency_halves = np.zeros((2, layers, experts), dtype=np.float64)
    counts_halves = np.zeros((2, layers, experts), dtype=np.int64)
    layer_re = re.compile(r"^layers\.(\d+)\.ffn$")

    def instrumented_moe(self, base, x):
        match = layer_re.match(base)
        if match is None:
            raise ValueError(f"unexpected MoE path: {base}")
        layer = int(match.group(1))
        config = self.c
        logits = (
            x.astype(mx.float32)
            @ self.w.read(base + ".gate.weight").astype(mx.float32).T
        )
        scores = mx.sqrt(mx.logaddexp(logits, mx.zeros_like(logits)))
        topk = config["num_experts_per_tok"]
        picks = mx.argsort(scores + self.w.read(base + ".gate.bias"), axis=-1)[
            :, -topk:
        ]
        selected = mx.take_along_axis(scores, picks, axis=-1)
        if config["norm_topk_prob"] and topk > 1:
            selected = selected / (mx.sum(selected, axis=-1, keepdims=True) + 1e-20)
        selected = selected * config["routed_scaling_factor"]
        mx.eval(picks, selected)
        chosen = picks.tolist()[0]
        weights = selected.tolist()[0]
        half = min(1, self.position * 2 // args.tokens)
        for expert, weight in zip(chosen, weights):
            saliency_halves[half, layer, expert] += weight
            counts_halves[half, layer, expert] += 1

        result = mx.zeros_like(x).astype(mx.float32)
        for expert, weight in zip(chosen, weights):
            result = result + self.expert(
                base + f".experts.{expert}", x, weight
            ).astype(mx.float32)
        return (
            result + self.expert(base + ".shared_experts", x).astype(mx.float32)
        ).astype(x.dtype)

    runtime.moe = MethodType(instrumented_moe, runtime)
    started = time.monotonic()
    for index, token in enumerate(ids, 1):
        runtime.step(token)
        if index % 32 == 0 or index == len(ids):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "tokens": index,
                        "total": len(ids),
                        "tok_s": index / elapsed,
                        "active_gb": mx.get_active_memory() / 1e9,
                        "peak_gb": mx.get_peak_memory() / 1e9,
                    }
                ),
                flush=True,
            )

    saliency = saliency_halves.sum(axis=0)
    counts = counts_halves.sum(axis=0)
    normalized_halves = saliency_halves / np.maximum(
        saliency_halves.sum(axis=2, keepdims=True), 1e-12
    )
    # Conservative across domains: an expert is low-value only when both
    # corpus halves assign it little or no normalized routing mass.
    robust_saliency = np.maximum(normalized_halves[0], normalized_halves[1])
    split_overlap = _overlap(saliency_halves[0], saliency_halves[1], args.keep_experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        saliency=saliency,
        robust_saliency=robust_saliency,
        counts=counts,
        saliency_halves=saliency_halves,
        counts_halves=counts_halves,
        tokens=np.array(args.tokens),
        keep_experts=np.array(args.keep_experts),
        split_keep_overlap=np.array(split_overlap),
        method=np.array("normalized_routing_weight"),
        selection_policy=np.array("max_normalized_half"),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tokens": args.tokens,
                "keep_experts": args.keep_experts,
                "split_keep_overlap": split_overlap,
                "never_routed": int((counts == 0).sum()),
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
