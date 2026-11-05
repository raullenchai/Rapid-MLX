#!/usr/bin/env python3
"""M5 QSA A/B kernels, complete synthetic attention layers, and optional TTFT.

Defaults use random tensors only. TTFT requires an explicit local checkpoint;
no model is downloaded. Timings include input normalization and metadata work.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path


def median_ms(mx, fn, warmups, runs):
    for _ in range(warmups):
        mx.eval(fn())
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def route_counts(model):
    return {
        name: module.m5_nax_route_constructions
        for name, module in model.named_modules()
        if hasattr(module, "m5_nax_route_constructions")
    }


def run_synthetic(args, mx, nax):
    from mlx_lm.models.cache import KVCache

    from rapid_mlx.kernels.qsa_block_sparse import block_sparse_attention
    from rapid_mlx.kernels.qsa_stage1 import _select_scores, qsa_stage1_select
    from rapid_mlx.models.qwen4_exp import QSAAttention
    from rapid_mlx.models.qwen4_exp_cache import QSAIndexCache
    from scripts.bench_qwen4_qsa_stage1_indexer import model_args

    mx.random.seed(args.seed)
    layer = QSAAttention(model_args())
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    mx.eval(layer.parameters())
    for total in args.tokens:
        length = min(total, args.chunk_size)
        q = mx.random.normal((1, length, 4, 128)).astype(mx.bfloat16)
        pooled = mx.random.normal((1, total // 4, 128)).astype(mx.bfloat16)
        aq = mx.random.normal((1, 24, length, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, total, 256)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape).astype(mx.bfloat16)
        hidden = mx.random.normal((1, total, 256)).astype(mx.bfloat16)
        pos = (total - length + mx.arange(length))[None].astype(mx.int32)
        mx.eval(q, pooled, aq, k, v, hidden, pos)
        os.environ[nax.ENABLE_ENV] = "1"
        s = nax.scores(q, pooled)
        ids = nax.select(q, pooled, pos)
        counts = mx.minimum((pos + 1) // 4, 512).astype(mx.int32)
        starts = mx.sort(ids, axis=-1) * 4
        tails = ((pos + 1) // 4 * 4)[..., None] + mx.arange(4, dtype=mx.int32)
        tc = ((pos + 1) % 4).astype(mx.int32)
        mx.eval(s, ids, counts, starts, tails, tc)
        compact = [starts, counts, tails, tc]
        timings = {}
        funcs = {
            "scores_current": lambda q=q, pooled=pooled: (
                mx.maximum(
                    (q.transpose(0, 2, 1, 3) @ pooled.swapaxes(-1, -2)[:, None]).astype(
                        mx.float32
                    ),
                    0,
                ).sum(axis=1)
                / 128**0.5
            ),
            "scores_m5": lambda q=q, pooled=pooled: nax.scores(q, pooled),
            "radix_selection": lambda s=s, pos=pos: _select_scores(
                s[0], pos[0], topk=512, compress_ratio=4
            ),
            "stage1_current": lambda q=q, pooled=pooled, pos=pos: qsa_stage1_select(
                q, pooled, pos, block_topk=512, compress_ratio=4
            ),
            "stage1_m5": lambda q=q, pooled=pooled, pos=pos: nax.select(q, pooled, pos),
            "attention_current_block_sparse": lambda aq=aq, k=k, v=v, compact=compact: (
                block_sparse_attention(aq, k, v, *compact, block_size=4)
            ),
            "attention_m5": lambda aq=aq, k=k, v=v, compact=compact: nax.attention(
                aq, k, v, *compact, scale=1 / 16
            ),
        }
        for name, fn in funcs.items():
            timings[name] = median_ms(mx, fn, args.warmups, args.runs)

        import numpy as np

        current_ids = np.asarray(
            qsa_stage1_select(q, pooled, pos, block_topk=512, compress_ratio=4)
        ).reshape(-1, 512)
        candidate_ids = np.asarray(ids).reshape(-1, 512)
        agreement = float(
            np.mean(
                [
                    len(set(a) & set(b)) / 512
                    for a, b in zip(current_ids, candidate_ids, strict=True)
                ]
            )
        )
        # Widen in MLX: NumPy does not support its bf16 buffer format.
        old_out = np.asarray(
            funcs["attention_current_block_sparse"]().astype(mx.float32)
        )
        new_out = np.asarray(funcs["attention_m5"]().astype(mx.float32))
        rms = float(np.sqrt(np.mean((new_out - old_out) ** 2)))
        if not np.isfinite(rms) or rms >= 0.003 or agreement < 0.995:
            raise RuntimeError(
                f"kernel parity failed: RMS={rms}, selection={agreement}"
            )

        # Complete QSA layer prefill from an empty cache, including projection,
        # normalization, RoPE, compression, selection, attention, gate and output.
        for enabled in (False, True):
            os.environ[nax.ENABLE_ENV] = str(int(enabled))
            before = route_counts(layer)

            def prefill(total=total, hidden=hidden):
                cache = [KVCache(), QSAIndexCache(4)]
                out = None
                for start in range(0, total, args.chunk_size):
                    out = layer(hidden[:, start : start + args.chunk_size], cache=cache)
                    mx.eval(out)
                return out

            timings[f"qsa_layer_prefill_{'m5' if enabled else 'current'}"] = median_ms(
                mx, prefill, args.warmups, args.runs
            )
            after = route_counts(layer)
            if enabled and sum(after.values()) <= sum(before.values()):
                raise RuntimeError("M5 layer benchmark did not enter the route")
        print(
            json.dumps(
                {
                    "tokens": total,
                    "query_chunk": length,
                    "milliseconds": timings,
                    "route_counts": route_counts(layer),
                    "selected_set_agreement": agreement,
                    "attention_output_rms": rms,
                }
            ),
            flush=True,
        )
        mx.clear_cache()


def run_ttft(args, mx, nax):
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    from rapid_mlx.utils.tokenizer import _register_vendored_archs

    config = json.loads((args.model_path / "config.json").read_text())
    if (
        config.get("model_type") != "qwen4_exp"
        and config.get("text_config", {}).get("model_type") != "qwen4_exp"
    ):
        raise SystemExit("TTFT requires a local qwen4_exp checkpoint")
    _register_vendored_archs()
    model, tokenizer = load(str(args.model_path.resolve()))
    model.eval()
    encoded = tokenizer.encode(args.prompt)
    if not encoded:
        raise SystemExit("prompt encodes to no tokens")
    mx.eval(model.parameters())
    for total in args.tokens:
        prompt = mx.array(
            (encoded * ((total + len(encoded) - 1) // len(encoded)))[:total]
        )
        for repeat in range(args.warmups + args.runs):
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                os.environ[nax.ENABLE_ENV] = str(int(enabled))
                before = route_counts(model)
                started = time.perf_counter()
                gen = generate_step(
                    prompt,
                    model,
                    max_tokens=1,
                    sampler=make_sampler(temp=0),
                    prefill_step_size=args.chunk_size,
                )
                try:
                    token, _ = next(gen)
                    mx.eval(token)
                    elapsed = time.perf_counter() - started
                finally:
                    gen.close()
                after = route_counts(model)
                delta = sum(after.values()) - sum(before.values())
                if enabled and delta == 0:
                    raise RuntimeError("TTFT did not exercise M5 QSA")
                if repeat >= args.warmups:
                    print(
                        json.dumps(
                            {
                                "tokens": total,
                                "mode": "m5" if enabled else "current",
                                "repeat": repeat - args.warmups,
                                "ttft_seconds": elapsed,
                                "first_token": int(token),
                                "m5_route_constructions": delta,
                            }
                        ),
                        flush=True,
                    )
                mx.clear_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--mode", choices=["synthetic", "ttft"], default="synthetic")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--prompt", default="Explain how sparse attention works. ")
    args = p.parse_args()
    if (
        args.runs < 1
        or args.warmups < 0
        or args.chunk_size < 64
        or args.chunk_size % 4
        or any(n < 4096 or n % 4 for n in args.tokens)
    ):
        p.error(
            "require runs>=1, warmups>=0, chunk>=64 divisible by 4, tokens>=4096 divisible by 4"
        )
    if args.mode == "ttft" and (
        args.model_path is None or not args.model_path.is_dir()
    ):
        p.error("TTFT requires --model-path to an existing local checkpoint")
    import mlx.core as mx

    from rapid_mlx.kernels import qsa_m5_nax as nax

    if not nax.hardware_supported():
        raise SystemExit("M5 tensor units and macOS 26.2+ required; no benchmark run")
    mx.set_default_device(mx.gpu)
    # Keep other opt-ins out of the baseline; capture the complete experiment.
    for flag in (
        "RAPID_MLX_QSA_STAGE1",
        "RAPID_MLX_QSA_BLOCK_SPARSE",
        "RAPID_MLX_QSA_INDEXED_SPLITK",
    ):
        os.environ[flag] = "0"
    print(
        json.dumps(
            {
                "device": mx.device_info(),
                "mlx": version("mlx"),
                "macos": platform.mac_ver()[0],
                "seed": args.seed,
                "mode": args.mode,
                "chunk_size": args.chunk_size,
                "baseline": "default layer path; kernel attention baseline is block_sparse",
            }
        ),
        flush=True,
    )
    if args.mode == "ttft":
        run_ttft(args, mx, nax)
    else:
        run_synthetic(args, mx, nax)


if __name__ == "__main__":
    main()
