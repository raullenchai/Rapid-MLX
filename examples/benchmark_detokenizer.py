#!/usr/bin/env python3
"""
Benchmark: Streaming Detokenizer vs Naive Decode

Compares performance of:
1. Old method: tokenizer.decode([token]) for each token
2. New method: StreamingDetokenizer.add_token() + last_segment

Run:
    python examples/benchmark_detokenizer.py
"""

import statistics
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import (
    load as load_tokenizer,
)
from transformers import AutoTokenizer


def benchmark_naive_decode(tokenizer, tokens, iterations=10):
    """Benchmark naive decode approach (old method)."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Old method: decode each token individually
        texts = []
        for t in tokens:
            text = tokenizer.decode([t])
            texts.append(text)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": "naive_decode",
        "mean_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def benchmark_streaming_detokenizer(
    tokenizer, tokens, detokenizer_class, iterations=10
):
    """Benchmark streaming detokenizer approach (new method)."""
    times = []

    for _ in range(iterations):
        detok = detokenizer_class(tokenizer)
        detok.reset()

        start = time.perf_counter()

        # New method: streaming detokenizer
        for t in tokens:
            detok.add_token(t)
            _ = detok.last_segment  # Get incremental text

        detok.finalize()
        _ = detok.text  # Get final text

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": detokenizer_class.__name__,
        "mean_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


def main():
    print("=" * 70)
    print(" Detokenizer Performance Benchmark")
    print("=" * 70)
    print()

    # Load tokenizer using mlx-lm's optimized loader
    print("Loading tokenizer with mlx-lm...")
    model_path = Path(snapshot_download("mlx-community/Qwen3-0.6B-8bit"))
    tokenizer_wrapper = load_tokenizer(model_path)

    # Also get raw tokenizer for naive decode
    raw_tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen3-0.6B-8bit")

    print(f"Tokenizer type: {type(tokenizer_wrapper._detokenizer_class).__name__}")
    print()

    # Test with different sequence lengths (targeting realistic generation sizes)
    base_text = "The development of large language models has revolutionized natural language processing and artificial intelligence. These models can understand context, generate coherent text, and perform a wide variety of tasks. "
    test_texts = [
        ("Short", "Hello, how are you doing today?"),
        ("Medium", base_text * 3),  # ~100 tokens
        ("Long", base_text * 15),  # ~500 tokens
        ("1K tokens", base_text * 35),  # ~1000 tokens
        ("2K tokens", base_text * 70),  # ~2000 tokens
        ("4K tokens", base_text * 140),  # ~4000 tokens
    ]

    results = []

    for name, text in test_texts:
        tokens = raw_tokenizer.encode(text)
        actual_tokens = len(tokens)
        print(f"{name} ({actual_tokens} tokens)")
        print("-" * 50)

        # Benchmark all methods
        naive_result = benchmark_naive_decode(raw_tokenizer, tokens, iterations=20)

        # Use mlx-lm's auto-selected detokenizer (via wrapper)
        optimized_result = benchmark_streaming_detokenizer(
            tokenizer_wrapper,
            tokens,
            tokenizer_wrapper._detokenizer_class,
            iterations=20,
        )

        # Calculate speedup
        speedup = (
            naive_result["mean_ms"] / optimized_result["mean_ms"]
            if optimized_result["mean_ms"] > 0
            else float("inf")
        )

        print(f"  Naive decode():      {naive_result['mean_ms']:8.3f}ms")
        print(f"  {optimized_result['method']}: {optimized_result['mean_ms']:8.3f}ms")
        print(f"  Speedup:             {speedup:8.2f}x")
        print()

        results.append(
            {
                "name": name,
                "tokens": actual_tokens,
                "naive_ms": naive_result["mean_ms"],
                "optimized_ms": optimized_result["mean_ms"],
                "optimized_name": optimized_result["method"],
                "speedup": speedup,
            }
        )

    # Summary table
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print(
        f"{'Sequence':<12} {'Tokens':>8} {'decode()':>12} {'Streaming':>12} {'Speedup':>10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<12} {r['tokens']:>8} {r['naive_ms']:>11.3f}ms {r['optimized_ms']:>11.3f}ms {r['speedup']:>9.2f}x"
        )

    # Average speedup
    avg_speedup = statistics.mean([r["speedup"] for r in results])
    print("-" * 70)
    print(f"{'Average speedup:':<55} {avg_speedup:>9.2f}x")
    print()

    # Verify correctness
    print("Verifying correctness...")
    for name, text in test_texts[:1]:
        tokens = raw_tokenizer.encode(text)

        # Streaming
        detok = tokenizer_wrapper._detokenizer_class(tokenizer_wrapper)
        detok.reset()
        for t in tokens:
            detok.add_token(t)
        detok.finalize()

        batch_result = raw_tokenizer.decode(tokens)

        if detok.text == batch_result:
            print(f"  ✓ {name}: Streaming matches batch decode")
        else:
            print(f"  ✗ {name}: MISMATCH!")
            print(f"    Streaming: {repr(detok.text)}")
            print(f"    Batch:     {repr(batch_result)}")


if __name__ == "__main__":
    main()
