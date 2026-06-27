# SPDX-License-Identifier: Apache-2.0
"""Real DFlash spec-decode run on Qwen3.5-9B-4bit + drafter, capturing
per-position (draft_logits, target_logits) so we can compute the
TV-distance confidence c_k* = 1 - 0.5 * ||p_d - p_t||_1 (paper Eq. 8).

The script writes a JSON trace: one entry per spec-decode round with:
    - draft_tokens: list[int]
    - target_argmax: list[int]      (greedy decision of target at each pos)
    - confidence: list[float]        (c_k* per position, length = block_size)
    - accepted_real: int             (longest prefix where draft == target_argmax)
    - block_size: int
    - wall_clock_ms: float           (round end-to-end)

This trace is the input to bench_phaseA.py which replays it under
Algorithm 1 vs fixed gamma.

We DELIBERATELY don't modify any code under vllm_mlx/. The spike loop
mirrors mlx_vlm.speculative.dflash._dflash_rounds but injects a
capture-then-sample sampler so we get logits at every position.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.utils import load
from mlx_vlm.speculative.drafters import load_drafter


def _softmax_row(logits_1d: mx.array) -> mx.array:
    return mx.softmax(logits_1d.astype(mx.float32), axis=-1)


def _tv_confidence(p_d: mx.array, p_t: mx.array) -> float:
    """1 - 0.5 * L1 over the vocabulary axis. Both inputs shape (V,)."""
    l1 = mx.sum(mx.abs(p_d - p_t))
    mx.eval(l1)
    c = 1.0 - 0.5 * float(l1.item())
    return max(0.0, min(1.0, c))


def trace_one_prompt(
    target_model: nn.Module,
    drafter: nn.Module,
    prompt_ids: mx.array,
    max_tokens: int,
) -> dict:
    """Run one full DFlash spec-decode trace; return rounds + tokens.

    target_model is the mlx_vlm top-level model (e.g.
    mlx_vlm.models.qwen3_5.Model); we drive .language_model directly
    because that's the surface with capture_layer_ids+
    rollback_speculative_cache.
    """
    from mlx_lm.models.cache import make_prompt_cache

    lm = target_model.language_model
    cache = make_prompt_cache(lm)
    target_layer_ids = list(drafter.config.target_layer_ids)
    block_size_total = int(drafter.config.block_size)

    # ---- Prefill -----------------------------------------------------
    t0 = time.perf_counter()
    last_out = lm(
        prompt_ids[None],
        cache=cache,
        capture_layer_ids=target_layer_ids,
    )
    hidden = mx.concatenate(last_out.hidden_states, axis=-1)
    first_token = int(mx.argmax(last_out.logits[0, -1], axis=-1).item())
    mx.eval(first_token, hidden)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    # ---- Drafter init -----------------------------------------------
    draft_cache = drafter.reset(target_model)

    emitted: list[int] = [first_token]
    b = first_token
    rounds: list[dict] = []
    bs = block_size_total  # fixed for the bench; drafter advertises this

    while len(emitted) < max_tokens:
        # ---- Draft block (capture draft logits) ---------------------
        # We bypass drafter.draft_block by inlining its body so we can
        # tap the pre-sample logits. This matches mlx-vlm 0.6.3's source.
        mask_id = int(drafter.config.mask_token_id)
        token_dtype = prompt_ids.dtype
        block_in = mx.array(
            [[int(b)] + [mask_id] * (bs - 1)],
            dtype=token_dtype,
        )
        t_round = time.perf_counter()
        draft_hidden = drafter._hidden(block_in, hidden, draft_cache)
        # `draft_logits` shape: (1, bs, V) — predicts positions bs of the block
        draft_logits = drafter._logits(draft_hidden[:, 1:])

        # Greedy sample the draft (paper bench uses temp=0).
        draft_tokens = mx.argmax(draft_logits, axis=-1).astype(token_dtype)

        # ---- Verify (capture target logits) -------------------------
        verify_input = mx.concatenate(
            [mx.array([[b]], dtype=token_dtype), draft_tokens],
            axis=1,
        )
        verify_out = lm(
            verify_input,
            cache=cache,
            capture_layer_ids=target_layer_ids,
        )
        hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
        target_argmax = mx.argmax(verify_out.logits, axis=-1).astype(token_dtype)

        mx.eval(draft_tokens, target_argmax, hidden, draft_logits, verify_out.logits)
        round_ms = (time.perf_counter() - t_round) * 1000.0

        # ---- Speculative walk ---------------------------------------
        # drafter emits bs-1 tokens for a length-bs input (mask scheme).
        # verify_input length is bs, verify_argmax length is bs:
        #   - argmax[0..bs-2] are predictions matched against draft[0..bs-2]
        #   - argmax[bs-1] is the always-emitted post-block bonus on full accept
        n_draft = bs - 1
        d_list = [int(x) for x in draft_tokens[0, :].tolist()]
        t_list = [int(x) for x in target_argmax[0, :].tolist()]
        accepted = 0
        while accepted < n_draft and t_list[accepted] == d_list[accepted]:
            accepted += 1

        # ---- Compute TV confidence per position ---------------------
        conf: list[float] = []
        for k in range(n_draft):
            p_d = _softmax_row(draft_logits[0, k])
            p_t = _softmax_row(verify_out.logits[0, k])
            conf.append(_tv_confidence(p_d, p_t))

        # ---- Emit accepted prefix + bonus ---------------------------
        new_tokens: list[int]
        if accepted < n_draft:
            # Partial accept: emit accepted draft tokens + corrective bonus.
            new_tokens = d_list[:accepted] + [t_list[accepted]]
            # Trim verify hidden to the accepted prefix (mirrors mlx-vlm).
            hidden = hidden[:, : accepted + 1, :]
            # mlx-vlm relies on rollback_speculative_cache to do BOTH the
            # KV trim and the offset rewind — don't double-rewind.
            if hasattr(lm, "rollback_speculative_cache"):
                lm.rollback_speculative_cache(
                    cache, verify_out.gdn_states, accepted, bs
                )
        else:
            # Full accept (accepted == n_draft = bs - 1): emit accepted
            # draft tokens AND the post-block bonus token at output
            # position bs - 1 (predicts the token after the last
            # accepted one). That's t_list[n_draft] == t_list[bs - 1].
            new_tokens = d_list[:] + [t_list[n_draft]]
            # Keep full block hidden as the drafter context for next round.

        for t in new_tokens:
            emitted.append(int(t))
            if len(emitted) >= max_tokens:
                break

        rounds.append(
            dict(
                draft_tokens=d_list,
                target_argmax=t_list,
                confidence=conf,
                accepted=accepted,
                block_size=bs,
                wall_clock_ms=round_ms,
            )
        )
        b = emitted[-1]

        if len(rounds) >= 64:  # cap trace length for spike
            break

    return dict(
        first_token=first_token,
        emitted=emitted,
        rounds=rounds,
        prefill_ms=prefill_ms,
        block_size=block_size_total,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="mlx-community/Qwen3.5-9B-4bit",
        help="Target HF path (mlx-quantized).",
    )
    parser.add_argument(
        "--drafter",
        default="z-lab/Qwen3.5-9B-DFlash",
        help="DFlash drafter HF path.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Write a 10-line Python function that returns the n-th Fibonacci number using memoization.",
            "Explain in two paragraphs why O(n log n) is the lower bound for comparison-based sorting.",
            "Translate to French: 'The quick brown fox jumps over the lazy dog'.",
        ],
        help="Prompts to run (one trace per prompt).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        default="spike/dspark-mvp/bench/trace_qwen3.5-9b-4bit.json",
    )
    args = parser.parse_args()

    print(f"[trace] loading target: {args.target}")
    target_model, processor = load(args.target)
    target_model.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    print(f"[trace] loading drafter: {args.drafter}")
    drafter, kind = load_drafter(args.drafter, kind="dflash")
    drafter.eval()
    print(f"[trace] drafter kind={kind} block_size={drafter.config.block_size}")

    out = {
        "target": args.target,
        "drafter": args.drafter,
        "block_size": int(drafter.config.block_size),
        "traces": [],
    }
    for prompt in args.prompts:
        ids = tokenizer.encode(prompt)
        ids_arr = mx.array(ids, dtype=mx.int32)
        print(f"[trace] running prompt (len={len(ids)}): {prompt[:60]}...")
        result = trace_one_prompt(
            target_model, drafter, ids_arr, max_tokens=args.max_tokens
        )
        result["prompt"] = prompt
        out["traces"].append(result)
        n_rounds = len(result["rounds"])
        total_acc = sum(r["accepted"] for r in result["rounds"])
        mean_acc = total_acc / max(1, n_rounds)
        mean_round_ms = sum(r["wall_clock_ms"] for r in result["rounds"]) / max(
            1, n_rounds
        )
        print(
            f"[trace]   rounds={n_rounds}  mean_accept={mean_acc:.2f}/"
            f"{result['block_size']}  mean_round_ms={mean_round_ms:.1f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[trace] wrote {args.out}")


if __name__ == "__main__":
    main()
