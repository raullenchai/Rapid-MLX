# SPDX-License-Identifier: Apache-2.0
"""K8V4 perf bench for task #332 verification campaign.

Why this script exists
----------------------
``spike_k8v4_spot_check.py`` measures K8V4 round-trip *correctness*
(K/V cosine + RMSE) against the FP16 baseline. This script answers
the complementary question: after the K8V4 round-trip is applied to a
populated prefix cache, does subsequent decode still hit baseline
tok/s, and does the next-token logit distribution stay aligned with
the FP16 control?

K8V4 architecture recap (from ``vllm_mlx/turboquant.py``)
--------------------------------------------------------
* ``TurboQuantKVCache.from_kv_cache(kv_cache, config)`` — encode a
  fully populated KVCache into compressed K/V.
* ``TurboQuantKVCache.to_kv_cache()`` — restore back to a regular
  ``KVCache`` with decompressed K/V (lossy round-trip).
* The decode path consumes a regular ``KVCache``; so post-restore
  decode tok/s is architecturally invariant to the K8V4 codec — what
  shifts is the cache *values*, not the cache *layout*.

What this script does
---------------------
1. Loads the target alias (must be locally cached).
2. Prefills a 2048-token prompt to populate the KV cache.
3. Snapshots the populated cache.
4. **Baseline run**: decodes ``--decode-tokens`` tokens from the
   snapshot, measures tok/s + top-1 token sequence.
5. **K8V4 run**: round-trips every attention-cache slab through
   ``TurboQuantKVCache.from_kv_cache(mode="k8v4")`` →
   ``.to_kv_cache()`` (matches the prefix-cache restore path), then
   decodes the same number of tokens from the K8V4-restored cache.
6. Reports:
   * baseline_tok_s vs k8v4_tok_s (must satisfy
     ``k8v4_tok_s >= 0.97 * baseline_tok_s``)
   * top-1 token sequence alignment (Hamming distance, 0 is best).

Run::

    python scripts/spike_k8v4_perf_bench.py \\
        --model qwen3.5-35b-4bit --decode-tokens 64

The script emits a single JSON line on stdout suitable for the PR
verification table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import mlx.core as mx


def _generate_step_loop(
    model, cache, x_seed, num_tokens: int
) -> tuple[list[int], float]:
    """Greedy decode ``num_tokens`` from the current cache state."""
    out_tokens: list[int] = []
    t0 = time.time()
    next_token = x_seed
    for _ in range(num_tokens):
        logits = model(next_token, cache=cache)
        if isinstance(logits, tuple):
            logits = logits[0]
        # Take the last position's logits; greedy argmax
        token_id = int(mx.argmax(logits[..., -1, :], axis=-1).item())
        out_tokens.append(token_id)
        next_token = mx.array([[token_id]], dtype=mx.int32)
        mx.eval(next_token)
    mx.eval(next_token)
    elapsed = time.time() - t0
    return out_tokens, elapsed


def _snapshot_cache(cache):
    """Return a list of (layer_idx, keys_copy, values_copy, offset) for
    every ``KVCache`` entry — the only kind K8V4 is wired to handle.
    Linear-attention layers (``ArraysCache``) are skipped; we mark them
    so the rebuild step can leave their state untouched."""
    from mlx_lm.models.cache import KVCache

    snap = []
    for li, layer in enumerate(cache):
        if isinstance(layer, KVCache) and layer.keys is not None:
            snap.append(
                {
                    "layer": li,
                    "keys": mx.array(layer.keys),
                    "values": mx.array(layer.values),
                    "offset": layer.offset,
                }
            )
    return snap


def _restore_cache(cache, snap):
    """Restore ``cache`` slabs from the snapshot in-place."""
    for entry in snap:
        layer = cache[entry["layer"]]
        layer.keys = mx.array(entry["keys"])
        layer.values = mx.array(entry["values"])
        layer.offset = entry["offset"]


def _apply_k8v4_roundtrip(cache, snap, config, cast_back: str | None = None):
    """For every snapped layer, replace its slab with the K8V4
    round-tripped values. If ``cast_back`` is given, cast K/V to that
    dtype before storing — useful for proving the bf16-vs-fp16 perf hit.
    """
    from vllm_mlx.turboquant import TurboQuantKVCache

    cast_target = None
    if cast_back is not None:
        cast_target = {
            "bf16": mx.bfloat16,
            "fp16": mx.float16,
            "fp32": mx.float32,
        }[cast_back]

    for entry in snap:
        layer = cache[entry["layer"]]

        class _Stub:
            pass

        stub = _Stub()
        stub.keys = entry["keys"]
        stub.values = entry["values"]
        stub.offset = entry["offset"]

        tq = TurboQuantKVCache.from_kv_cache(stub, config)
        rt = tq.to_kv_cache()
        out_keys = rt.keys
        out_values = rt.values
        if cast_target is not None:
            out_keys = out_keys.astype(cast_target)
            out_values = out_values.astype(cast_target)
        # Force materialization of the round-tripped K/V slabs so the
        # K8V4 codec cost lands here, not silently amortized into the
        # first decode step's tok/s number.
        mx.eval([out_keys, out_values])
        layer.keys = out_keys
        layer.values = out_values
        layer.offset = rt.offset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Alias name or HF path (must be locally cached).",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=2048,
        help="Tokens to prefill before measuring decode (default: 2048).",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=64,
        help="Greedy decode tokens to time per run (default: 64).",
    )
    parser.add_argument(
        "--cast-back",
        type=str,
        default=None,
        choices=[None, "bf16", "fp16", "fp32"],
        help=(
            "If set, cast K8V4-decoded K/V back to this dtype before "
            "decode. Probes whether the K8V4 codec's fp16 output "
            "causes a dtype-mismatch perf hit against a bf16 model."
        ),
    )
    args = parser.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from vllm_mlx.model_aliases import resolve_model
    from vllm_mlx.turboquant import TurboQuantConfig

    hf_path = resolve_model(args.model)
    print(f"[1/4] Loading {args.model} → {hf_path}", flush=True)
    t0 = time.time()
    model, tokenizer = load(hf_path)
    print(f"      loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[2/4] Prefilling {args.prefill_tokens} tokens", flush=True)
    seed_text = (
        "Apple Silicon decode is memory-bandwidth-bound; the KV cache "
        "dominates the per-step bandwidth. TurboQuant compresses the "
        "prefix cache so cache fetches cost less and longer histories "
        "fit in unified memory. This is a benign warm-up prompt. "
    )
    encoded = tokenizer.encode(seed_text * 50)
    ids = encoded[: args.prefill_tokens]
    if len(ids) < args.prefill_tokens:
        ids = ids + ids[: args.prefill_tokens - len(ids)]
    x = mx.array([ids], dtype=mx.int32)

    cache = make_prompt_cache(model)
    t0 = time.time()
    logits = model(x, cache=cache)
    mx.eval(logits)
    print(f"      prefill done in {time.time() - t0:.1f}s", flush=True)

    # Pick the next seed token from the prefill final logits — same seed
    # for both runs so the comparison is honest.
    last_logits = (
        logits[..., -1, :] if not isinstance(logits, tuple) else logits[0][..., -1, :]
    )
    seed_id = int(mx.argmax(last_logits, axis=-1).item())
    seed_x = mx.array([[seed_id]], dtype=mx.int32)

    # Snapshot the populated cache so we can rewind between runs.
    snap = _snapshot_cache(cache)
    print(
        f"      snapped {len(snap)} attention layers (linear-attn layers skipped)",
        flush=True,
    )

    # ---- Baseline run --------------------------------------------
    print(
        f"[3/4] Baseline decode ({args.decode_tokens} tokens, K8V4 off)",
        flush=True,
    )
    _restore_cache(cache, snap)
    # Warmup: 1 step to prime mlx kernels
    _ = _generate_step_loop(model, cache, seed_x, num_tokens=1)
    # Re-snap (warmup mutated cache) — actually we want the warm pure
    # measurement; so re-restore from snap and decode fresh.
    _restore_cache(cache, snap)
    baseline_tokens, baseline_elapsed = _generate_step_loop(
        model, cache, seed_x, num_tokens=args.decode_tokens
    )
    baseline_tok_s = args.decode_tokens / baseline_elapsed
    print(
        f"      baseline: {baseline_tok_s:.2f} tok/s ({baseline_elapsed:.2f}s)",
        flush=True,
    )

    # ---- K8V4 run -------------------------------------------------
    print(
        f"[4/4] K8V4-restored decode ({args.decode_tokens} tokens)",
        flush=True,
    )
    _restore_cache(cache, snap)
    config = TurboQuantConfig(mode="k8v4", bits=4)
    _apply_k8v4_roundtrip(cache, snap, config, cast_back=args.cast_back)
    # Warmup 1 step on K8V4-restored cache too — primes any decode
    # kernels that need re-specialization on the slightly different
    # numerical values.
    _ = _generate_step_loop(model, cache, seed_x, num_tokens=1)
    _restore_cache(cache, snap)
    _apply_k8v4_roundtrip(cache, snap, config, cast_back=args.cast_back)
    k8v4_tokens, k8v4_elapsed = _generate_step_loop(
        model, cache, seed_x, num_tokens=args.decode_tokens
    )
    k8v4_tok_s = args.decode_tokens / k8v4_elapsed
    print(
        f"      k8v4:     {k8v4_tok_s:.2f} tok/s ({k8v4_elapsed:.2f}s)",
        flush=True,
    )

    # ---- Compare --------------------------------------------------
    ratio = k8v4_tok_s / baseline_tok_s if baseline_tok_s > 0 else 0.0
    hamming = sum(1 for a, b in zip(baseline_tokens, k8v4_tokens) if a != b)
    perf_pass = ratio >= 0.97

    report = {
        "model": args.model,
        "hf_path": hf_path,
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "baseline_tok_s": baseline_tok_s,
        "k8v4_tok_s": k8v4_tok_s,
        "k8v4_over_baseline_ratio": ratio,
        "top1_hamming_distance": hamming,
        "perf_pass_at_0.97": perf_pass,
    }
    print("\n========== JSON ==========")
    print(json.dumps(report, indent=2))
    print(
        "\n========== Summary ==========\n"
        f"baseline={baseline_tok_s:.2f} tok/s, k8v4={k8v4_tok_s:.2f} tok/s, "
        f"ratio={ratio:.3f} ({'PASS' if perf_pass else 'FAIL'}), "
        f"top1_hamming={hamming}/{args.decode_tokens}"
    )
    return 0 if perf_pass else 1


if __name__ == "__main__":
    sys.exit(main())
