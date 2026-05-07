#!/usr/bin/env python3
"""Streaming detokenizer benchmark — naive ``decode([t])`` vs incremental
streaming detokenizer over a real generated token stream.

Internal dev micro-bench. Loads a small model, generates ~2K tokens once,
then times both detokenization paths over those tokens. Lives in
``scripts/`` rather than the user-facing CLI because the number it
reports is meaningful only when tuning the streaming detokenizer
implementation — end users care about end-to-end TTFT/decode tok/s,
which ``rapid-mlx bench`` covers.

Usage:
    python3 scripts/bench_detok.py
    python3 scripts/bench_detok.py mlx-community/Llama-3.2-1B-Instruct-4bit --iterations 5
"""

from __future__ import annotations

import argparse
import statistics
import time

from mlx_lm import load
from mlx_lm.generate import generate


def run(model_id: str, iterations: int) -> None:
    print("=" * 70)
    print(" Streaming Detokenizer Benchmark")
    print("=" * 70)
    print()

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)

    prompt = (
        "Write a detailed explanation of how machine learning works "
        "and its applications in modern technology."
    )
    print(f"Generating tokens with prompt: {prompt[:50]}...")

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=2000,
        verbose=False,
    )

    prompt_tokens = tokenizer.encode(prompt)
    all_tokens = tokenizer.encode(output)
    generated_tokens = all_tokens[len(prompt_tokens) :]
    print(f"Generated {len(generated_tokens)} tokens for benchmark")
    print()

    # Naive decode (one decode() call per token).
    print("Benchmarking Naive Decode (OLD method)...")
    naive_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for t in generated_tokens:
            _ = tokenizer.decode([t])
        naive_times.append(time.perf_counter() - start)

    naive_mean = statistics.mean(naive_times) * 1000

    # Incremental streaming detokenizer (one instance, ``add_token`` per token).
    print("Benchmarking Streaming Detokenizer (NEW method)...")
    streaming_times = []
    detok_class = tokenizer._detokenizer_class
    for _ in range(iterations):
        detok = detok_class(tokenizer)
        detok.reset()
        start = time.perf_counter()
        for t in generated_tokens:
            detok.add_token(t)
            _ = detok.last_segment
        detok.finalize()
        streaming_times.append(time.perf_counter() - start)

    streaming_mean = statistics.mean(streaming_times) * 1000

    speedup = naive_mean / streaming_mean
    time_saved = naive_mean - streaming_mean

    print()
    print("=" * 70)
    print(f" RESULTS: {len(generated_tokens)} tokens, {iterations} iterations")
    print("=" * 70)
    print(f"{'Method':<25} {'Time':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Naive decode():':<25} {naive_mean:>10.2f}ms {'1.00x':>10}")
    print(f"{'Streaming detokenizer:':<25} {streaming_mean:>10.2f}ms {speedup:>9.2f}x")
    print("-" * 70)
    print(f"{'Time saved per request:':<25} {time_saved:>10.2f}ms")
    print(
        f"{'Per-token savings:':<25} {(time_saved / len(generated_tokens) * 1000):>10.1f}µs"
    )
    print()

    # Sanity check: streaming output should match batch decode (modulo BPE
    # boundary noise on leading/trailing spaces).
    print("Verifying correctness...")
    detok = detok_class(tokenizer)
    detok.reset()
    for t in generated_tokens:
        detok.add_token(t)
    detok.finalize()

    batch_result = tokenizer.decode(generated_tokens)
    streaming_stripped = detok.text.strip()
    batch_stripped = batch_result.strip()
    if streaming_stripped == batch_stripped:
        print("  ✓ Streaming output matches batch decode")
    elif streaming_stripped in batch_stripped or batch_stripped in streaming_stripped:
        print("  ✓ Streaming output matches (minor BPE edge case)")
    else:
        common_len = min(len(streaming_stripped), len(batch_stripped)) - 10
        if (
            common_len > 0
            and streaming_stripped[:common_len] == batch_stripped[:common_len]
        ):
            print("  ✓ Streaming output matches (BPE boundary difference)")
        else:
            print("  ✗ MISMATCH! Results differ")
            print(f"    Streaming: {detok.text[:100]!r}...")
            print(f"    Batch: {batch_result[:100]!r}...")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "model",
        nargs="?",
        default="mlx-community/Qwen3-0.6B-8bit",
        help="Model to use for tokenizer (default: mlx-community/Qwen3-0.6B-8bit)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Benchmark iterations (default: 5)",
    )
    args = p.parse_args()
    run(args.model, args.iterations)


if __name__ == "__main__":
    main()
