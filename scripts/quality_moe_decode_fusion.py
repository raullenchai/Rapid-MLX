#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced MoE parity with a changed-prefill-chunking noise control.

Run on a leased Metal host with a local checkpoint and a real text file. This
loads one set of weights and holds three independent caches: original, original
with different prefill chunking, and fused. Each sees the same forced tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import cast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error("model must be an existing local checkpoint directory")
    if args.tokens < 1024:
        parser.error("the quality gate requires at least 1024 forced tokens")

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from rapid_mlx.patches.moe_decode_fusion import install_moe_decode_fusion

    os.environ["RAPID_MLX_MOE_DECODE_FUSION"] = "0"
    from rapid_mlx.moe_fusion import fuse_gate_up
    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    model, tokenizer = load_model_with_fallback(args.model)
    fuse_gate_up(model)
    os.environ["RAPID_MLX_MOE_DECODE_FUSION"] = "1"
    enrolled = install_moe_decode_fusion(model)
    if not enrolled:
        raise RuntimeError("no supported MoE blocks enrolled")

    tokens = tokenizer.encode(args.text.read_text())[: 256 + args.tokens]
    if len(tokens) < 256 + args.tokens:
        raise ValueError("text is too short for the prefill and forced continuation")
    caches = [make_prompt_cache(model) for _ in range(3)]
    os.environ["RAPID_MLX_MOE_DECODE_FUSION"] = "0"
    for cache, chunk in zip(caches, (128, 64, 128), strict=True):
        for offset in range(0, 256, chunk):
            out = model(mx.array([tokens[offset : offset + chunk]]), cache=cache)
            mx.eval(getattr(out, "logits", out))

    stats = {
        name: {"top1_matches": 0, "kl_sum": 0.0, "max_logit_difference": 0.0}
        for name in ("fused", "noise")
    }
    for token in tokens[256:]:
        logits = []
        for index, cache in enumerate(caches):
            os.environ["RAPID_MLX_MOE_DECODE_FUSION"] = "1" if index == 2 else "0"
            out = model(mx.array([[token]]), cache=cache)
            row = getattr(out, "logits", out)[0, -1].astype(mx.float32)
            mx.eval(row)
            logits.append(row)
        reference = logits[0] - mx.logsumexp(logits[0])
        probability = mx.exp(reference)
        for name, row in (("noise", logits[1]), ("fused", logits[2])):
            result = stats[name]
            result["top1_matches"] += int(
                bool((mx.argmax(logits[0]) == mx.argmax(row)).item())
            )
            result["kl_sum"] += cast(
                float,
                mx.sum(probability * (reference - (row - mx.logsumexp(row)))).item(),
            )
            result["max_logit_difference"] = max(
                result["max_logit_difference"],
                cast(float, mx.abs(logits[0] - row).max().item()),
            )
    report = {
        "model": args.model,
        "tokens": args.tokens,
        "enrolled_blocks": enrolled,
        "prefill_tokens": 256,
        "prefill_chunks": [128, 64, 128],
        "stats": {
            name: {
                **result,
                "top1_agreement": result["top1_matches"] / args.tokens,
                "mean_kl": result["kl_sum"] / args.tokens,
            }
            for name, result in stats.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
