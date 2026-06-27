# SPDX-License-Identifier: Apache-2.0
"""K8V4 quality spot-check for task #332.

Why this script exists
----------------------
``mlx_lm.perplexity`` measures PPL during a prefill-only forward pass,
which does NOT exercise the K8V4 prefix-cache roundtrip. The PPL number
is therefore invariant to the K8V4 setting and cannot gate the
default-on flip on its own.

What this script does
---------------------
1. Loads a target alias (must be locally cached — no downloads).
2. Prefills a long sequence to populate the KVCache.
3. Snapshots the K/V slabs from a sample of attention layers.
4. Runs each slab through ``TurboQuantKVCache.from_kv_cache(mode="k8v4")``
   → ``.to_kv_cache()`` (the actual prefix-cache compression path).
5. Reports per-layer K-side and V-side cosine similarity + RMSE
   between the FP16 original and the K8V4 roundtrip.
6. Also runs ``mlx_lm.perplexity.eval_ppl`` at the requested sequence
   length as an absolute baseline — this confirms model sanity but is
   NOT a K8V4-gating number (the prefill path doesn't fetch the cache).

Run::

    python scripts/spike_k8v4_spot_check.py \
        --model qwen3.5-35b-8bit --seq-len 8192 --num-samples 4

The script emits a JSON report on stdout and a markdown table summary
suitable for pasting into a PR body.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import mlx.core as mx
import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity along the last axis."""
    flat_a = a.reshape(-1, a.shape[-1]).astype(np.float32)
    flat_b = b.reshape(-1, b.shape[-1]).astype(np.float32)
    num = np.sum(flat_a * flat_b, axis=-1)
    den = np.linalg.norm(flat_a, axis=-1) * np.linalg.norm(flat_b, axis=-1) + 1e-8
    return float(np.mean(num / den))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Alias name or HF path (must be locally cached).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Prefill length in tokens (default: 8192).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of attention layers to spot-check (default: 4).",
    )
    parser.add_argument(
        "--skip-baseline-ppl",
        action="store_true",
        help=(
            "Skip the mlx_lm.perplexity baseline (saves ~1 min on the "
            "35B model; the K8V4 roundtrip stats are the gating signal "
            "anyway)."
        ),
    )
    args = parser.parse_args()

    from mlx_lm import load

    from vllm_mlx.model_aliases import resolve_model
    from vllm_mlx.turboquant import TurboQuantConfig, TurboQuantKVCache

    hf_path = resolve_model(args.model)
    print(f"[1/4] Loading {args.model} → {hf_path}", flush=True)
    t0 = time.time()
    model, tokenizer = load(hf_path)
    print(f"      loaded in {time.time() - t0:.1f}s", flush=True)

    # ---- Prefill a long sequence ----------------------------------
    print(
        f"[2/4] Prefilling seq_len={args.seq_len} tokens to populate the KV cache",
        flush=True,
    )
    # Repeat a stable seed string until we reach the requested length —
    # gives reproducible K/V slab shapes without needing a dataset.
    seed_text = (
        "Apple Silicon decode is memory-bandwidth-bound; the KV cache "
        "dominates the per-step bandwidth. TurboQuant compresses the "
        "prefix cache so cache fetches cost less and longer histories "
        "fit in unified memory. This is a benign warm-up prompt. "
    )
    encoded = tokenizer.encode(seed_text * 200)
    ids = encoded[: args.seq_len]
    if len(ids) < args.seq_len:
        ids = ids + ids[: args.seq_len - len(ids)]
    x = mx.array([ids], dtype=mx.int32)

    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    t0 = time.time()
    _ = model(x, cache=cache)
    mx.eval(_)  # force materialization so timing is honest
    print(
        f"      prefill done in {time.time() - t0:.1f}s, cache layers: {len(cache)}",
        flush=True,
    )

    # ---- K8V4 roundtrip per layer ---------------------------------
    print(
        f"[3/4] Running K8V4 encode/decode roundtrip on {args.layers} sample layers",
        flush=True,
    )
    config = TurboQuantConfig(mode="k8v4", bits=4)

    # Only attention layers carry K/V slabs eligible for K8V4
    # compression. Hybrid Qwen3.5 stacks mix GatedDeltaNet (ArraysCache)
    # with attention (KVCache); the linear layers store a fused state
    # vector that the V-quant codebook can't address.
    from mlx_lm.models.cache import KVCache

    kv_layer_idxs = [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
    print(
        f"      attention layers: {len(kv_layer_idxs)} / {len(cache)} "
        f"(remainder is linear-attention / GatedDeltaNet — K8V4-ineligible)",
        flush=True,
    )

    # Pick evenly-spaced KV layers across the stack so we cover early /
    # middle / late layer-class behaviour rather than only the layer-0
    # outlier.
    if not kv_layer_idxs:
        per_layer_stats = []
        layer_idxs = []
    else:
        picks = (
            np.linspace(
                0, len(kv_layer_idxs) - 1, num=min(args.layers, len(kv_layer_idxs))
            )
            .round()
            .astype(int)
            .tolist()
        )
        layer_idxs = [kv_layer_idxs[p] for p in picks]
    per_layer_stats = []
    for li in layer_idxs:
        layer = cache[li]
        if layer.keys is None or layer.values is None:
            continue
        offset = layer.offset
        if offset <= 0:
            continue
        head_dim = layer.keys.shape[-1]

        # Snapshot the populated slab. Force to float16 first because
        # qwen3.5 attention layers store K/V as bfloat16 and numpy has
        # no native bf16; the K8V4 encoder upcasts internally so this
        # cast is purely for the numpy comparison, not for the K8V4
        # path itself.
        k = layer.keys[..., :offset, :].astype(mx.float16)
        v = layer.values[..., :offset, :].astype(mx.float16)
        k_np = np.array(k)
        v_np = np.array(v)

        # Construct a minimal KVCache-shaped stub for from_kv_cache.
        class _Stub:
            pass

        stub = _Stub()
        stub.keys = k
        stub.values = v
        stub.offset = offset

        tq = TurboQuantKVCache.from_kv_cache(stub, config)
        rt = tq.to_kv_cache()
        rk = np.array(rt.keys)
        rv = np.array(rt.values)

        per_layer_stats.append(
            {
                "layer": int(li),
                "head_dim": int(head_dim),
                "k_cosine": _cosine(k_np, rk),
                "k_rmse": _rmse(k_np, rk),
                "v_cosine": _cosine(v_np, rv),
                "v_rmse": _rmse(v_np, rv),
            }
        )

    # ---- Optional mlx_lm.perplexity baseline ----------------------
    ppl_baseline = None
    if not args.skip_baseline_ppl:
        print(
            f"[4/4] mlx_lm.perplexity baseline at seq_len={args.seq_len} "
            "(sanity only — does NOT exercise K8V4 prefix-cache path)",
            flush=True,
        )
        try:
            from mlx_lm.perplexity import eval_ppl, load_data

            data = load_data(
                tokenizer,
                "allenai/tulu-3-sft-mixture",
                num_samples=16,
                sequence_length=min(args.seq_len, 4096),
            )
            t0 = time.time()
            ppl, se = eval_ppl(model, data, batch_size=1)
            print(
                f"      ppl={ppl:.4f} (se={se:.4f}) in {time.time() - t0:.1f}s",
                flush=True,
            )
            ppl_baseline = {"ppl": float(ppl), "se": float(se)}
        except Exception as exc:
            print(f"      baseline skipped: {exc}", flush=True)

    # ---- Report ----------------------------------------------------
    report = {
        "model": args.model,
        "hf_path": hf_path,
        "seq_len": args.seq_len,
        "k8v4_roundtrip": per_layer_stats,
        "ppl_baseline_note": (
            "mlx_lm.perplexity prefill PPL is invariant to K8V4 because "
            "the K8V4 path lives in the prefix-cache fetch loop, not in "
            "the prefill forward pass. Number is included as a sanity "
            "baseline only."
        ),
        "ppl_baseline": ppl_baseline,
    }

    print("\n========== JSON ==========")
    print(json.dumps(report, indent=2))

    if per_layer_stats:
        kc = [s["k_cosine"] for s in per_layer_stats]
        vc = [s["v_cosine"] for s in per_layer_stats]
        kr = [s["k_rmse"] for s in per_layer_stats]
        vr = [s["v_rmse"] for s in per_layer_stats]
        print("\n========== Summary ==========")
        print(f"K8V4 K cosine: mean={statistics.mean(kc):.4f} min={min(kc):.4f}")
        print(f"K8V4 V cosine: mean={statistics.mean(vc):.4f} min={min(vc):.4f}")
        print(f"K8V4 K RMSE:   mean={statistics.mean(kr):.4f} max={max(kr):.4f}")
        print(f"K8V4 V RMSE:   mean={statistics.mean(vr):.4f} max={max(vr):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
