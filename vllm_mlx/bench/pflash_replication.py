# SPDX-License-Identifier: Apache-2.0
"""TTFT replication harness for PFlash (#287).

@michaelasper's fork reported a cold-prefill TTFT win of 126.6 s →
11.2 s (≈11×) on a 135 k-token Qwen3.6-35B-A3B-4bit prompt. The fork
shipped that as a single anecdotal probe; this harness re-runs the
comparison so maintainers can verify the speedup on their own
hardware before un-drafting the PR.

Usage::

    uv run python -m vllm_mlx.bench.pflash_replication \\
        --model mlx-community/Qwen3-4B-4bit \\
        --context-tokens 65536 \\
        --keep-ratio 0.20

The harness is deliberately small — it builds a synthetic long prompt
via ``cli._build_benchmark_context``, runs one cold prefill with
PFlash off and one with PFlash on, and reports both TTFT numbers and
the delta. Output is single-line JSON so a CI runner can consume it.

This module is opt-in; nothing in the default ``rapid-mlx`` flow
imports it. It exists so a maintainer running the replication run has
a single command to invoke.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


def _build_engine(model_name: str, *, pflash_mode: str, keep_ratio: float):
    """Construct an AsyncEngineCore configured with the requested
    PFlash mode. Heavy imports stay lazy so ``--help`` does not load
    MLX."""
    from mlx_lm import load

    from ..engine_core import AsyncEngineCore, EngineConfig
    from ..pflash import PFlashConfig
    from ..scheduler import SchedulerConfig

    model, tokenizer = load(model_name)
    scheduler_config = SchedulerConfig(
        pflash_config=PFlashConfig(
            mode=pflash_mode,
            threshold=1,  # Always-on for the replication run.
            keep_ratio=keep_ratio,
        ),
    )
    engine_config = EngineConfig(
        model_name=model_name,
        scheduler_config=scheduler_config,
    )
    return model, tokenizer, AsyncEngineCore(model, tokenizer, engine_config)


async def _run_one(
    model_name: str, prompt: str, *, pflash_mode: str, keep_ratio: float
):
    from ..request import SamplingParams

    _model, _tokenizer, engine_ctx = _build_engine(
        model_name, pflash_mode=pflash_mode, keep_ratio=keep_ratio
    )
    params = SamplingParams(max_tokens=1, temperature=0.0)
    async with engine_ctx as engine:
        await asyncio.sleep(0.1)  # warm-up
        start = time.perf_counter()
        output = await engine.generate(prompt=prompt, sampling_params=params)
        elapsed = time.perf_counter() - start
        # ``output.prompt_tokens`` is the logical (pre-compression) count
        # — the number of tokens in the user's prompt. The actual
        # prefill workload is shorter on PFlash-on runs.
        #
        # Surface a ratio-only estimate so the replication JSON makes
        # the intended workload reduction inspectable (codex r4 NIT).
        # IMPORTANT: this is NOT an upper or lower bound on the real
        # post-compression count. The real compressor (see
        # ``pflash.compress_tokens``) applies ``min_keep_tokens`` floor,
        # ``sink_tokens`` + ``tail_tokens`` preservation, threshold
        # short-circuits, and block-truncation rounding — any of which
        # can move the actual count above OR below the ratio number.
        # Codex r6 NIT flagged the old ``_upper_bound`` naming as
        # misleading for exactly this reason. Wiring an exact
        # post-compression count through ``RequestOutput`` for an
        # opt-in replication harness would be more code than the
        # harness itself; the maintainer running the bench can compare
        # this ratio number against the TTFT speedup row in the PR
        # body to sanity-check.
        is_compressed_mode = pflash_mode != "off"
        if is_compressed_mode:
            from math import ceil

            ratio_estimate = max(1, ceil(output.prompt_tokens * keep_ratio))
        else:
            ratio_estimate = output.prompt_tokens
        return {
            "mode": pflash_mode,
            "keep_ratio": keep_ratio,
            "ttft_s": elapsed,
            "prompt_tokens": output.prompt_tokens,
            # Ratio-only estimate. NOT a bound — the real compressor's
            # ``min_keep_tokens`` / ``sink_tokens`` / ``tail_tokens``
            # floors and threshold short-circuits can move the actual
            # count in either direction (codex r6 NIT).
            "model_prompt_tokens_ratio_estimate": ratio_estimate,
            "model_prompt_tokens_estimate_caveat": (
                "ratio-only number = ceil(prompt_tokens * keep_ratio); "
                "not an upper or lower bound — the real compressor also "
                "applies min_keep_tokens / sink / tail floors and "
                "threshold short-circuits that can move the actual "
                "post-compression count in either direction"
            ),
            "completion_tokens": output.completion_tokens,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="PFlash TTFT replication for #287.")
    parser.add_argument("--model", required=True, help="Model to load.")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=65_536,
        help="Approximate filler tokens to prepend to the prompt.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.20,
        help="PFlash keep_ratio for the compressed run (default 0.20).",
    )
    args = parser.parse_args()

    from ..cli import _build_benchmark_context

    haystack = _build_benchmark_context(args.context_tokens)
    prompt = f"{haystack}\n\nUser request:\nSummarize the key reference points in one sentence."

    off = asyncio.run(
        _run_one(args.model, prompt, pflash_mode="off", keep_ratio=args.keep_ratio)
    )
    on = asyncio.run(
        _run_one(args.model, prompt, pflash_mode="always", keep_ratio=args.keep_ratio)
    )

    delta = off["ttft_s"] / on["ttft_s"] if on["ttft_s"] > 0 else float("inf")
    report = {
        "model": args.model,
        "context_tokens": args.context_tokens,
        "off": off,
        "on": on,
        "delta_x": round(delta, 2),
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
