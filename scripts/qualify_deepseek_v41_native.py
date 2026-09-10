#!/usr/bin/env python3
"""Qualify a native DeepSeek-V4.1 build on one resident model instance.

This is deliberately a direct-runtime gate rather than a server benchmark: it
checks the model's required chat framing, greedy token stability across Metal
evaluation intervals, decode throughput, and peak memory before any catalog or
server integration is attempted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deepseek_v41_native.load import load  # noqa: E402

BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
CHAT_START = "</think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--fuse-moe-gate-up", action="store_true")
    parser.add_argument(
        "--prompt",
        default="What is the capital of France? Answer in one short sentence.",
    )
    return parser.parse_args()


def framed_prompt(text: str) -> str:
    """DeepSeek-V4.1 chat mode framing from the release encoding contract."""
    return f"{BOS}{USER}{text}{ASSISTANT}{CHAT_START}"


def run_once(model, token_ids: list[int], output_tokens: int, eos_id: int) -> dict:
    cache = model.make_cache(
        bsz=1,
        max_seq_len=len(token_ids) + output_tokens + 8,
        dtype=mx.bfloat16,
    )
    started = time.monotonic()
    logits = model(mx.array([token_ids]), cache, last_logit_only=True)
    mx.eval(logits)
    prefill_seconds = time.monotonic() - started

    generated: list[int] = []
    decode_times: list[float] = []
    token = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(output_tokens):
        token_id = int(token[0])
        generated.append(token_id)
        if token_id == eos_id:
            break
        started = time.monotonic()
        logits = model(token[:, None], cache, last_logit_only=True)
        mx.eval(logits)
        decode_times.append(time.monotonic() - started)
        token = mx.argmax(logits[:, -1], axis=-1)

    decode_seconds = sum(decode_times)
    decode_tokens = len(decode_times)
    steady_times = decode_times[len(decode_times) // 2 :]
    return {
        "input_tokens": len(token_ids),
        "output_tokens": generated,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_transitions": decode_tokens,
        "decode_tok_s": decode_tokens / decode_seconds if decode_seconds else None,
        "steady_last_half_tok_s": (
            len(steady_times) / sum(steady_times) if steady_times else None
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    model, _ = load(
        str(args.model.resolve()),
        lazy=False,
        fuse_moe_gate_up=args.fuse_moe_gate_up,
    )
    load_seconds = time.monotonic() - started
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.model.resolve() / "tokenizer.json")
    )
    prompt = framed_prompt(args.prompt)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(
        json.dumps(
            {
                "event": "loaded",
                "load_seconds": load_seconds,
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
                "prompt": prompt,
                "prompt_tokens": token_ids,
            }
        ),
        flush=True,
    )

    reference_tokens = None
    for interval in args.intervals:
        model.eval_interval = interval
        result = run_once(model, token_ids, args.tokens, eos_id=1)
        result.update(
            {
                "event": "interval",
                "eval_interval": interval,
                "text": tokenizer.decode(result["output_tokens"]),
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        )
        if reference_tokens is None:
            reference_tokens = result["output_tokens"]
            result["greedy_matches_first_interval"] = True
        else:
            result["greedy_matches_first_interval"] = (
                result["output_tokens"] == reference_tokens
            )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
