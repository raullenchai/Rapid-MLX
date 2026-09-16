# SPDX-License-Identifier: Apache-2.0
"""Benchmark all text models for README."""

import time


def benchmark_model(model_name: str):
    """Benchmark a single model and return results."""
    from mlx_lm import load

    from rapid_mlx import EngineConfig, EngineCore, SamplingParams, SchedulerConfig

    base_prompts = [
        "What is 2+2?",
        "Name 3 colors.",
        "What is Python?",
        "Capital of Japan?",
        "Who wrote Hamlet?",
    ]

    params = SamplingParams(max_tokens=50, temperature=0.7)

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {model_name}")
    print("=" * 60)

    print("Loading model...")
    model, tokenizer = load(model_name)

    def format_prompt(p):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )

    formatted = [format_prompt(p) for p in base_prompts]

    # Create engine with explicit ID for tracking
    config = EngineConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=256,
            prefill_batch_size=8,
            completion_batch_size=32,
        )
    )
    engine = EngineCore(model, tokenizer, config)

    try:
        # Test 1: Single request throughput
        print("\n1. Single request throughput...")
        single_times = []
        single_tokens = []
        for p in formatted[:3]:
            start = time.perf_counter()
            result = engine.generate_batch_sync([p], params)[0]
            elapsed = time.perf_counter() - start
            single_times.append(elapsed)
            single_tokens.append(result.completion_tokens)

        single_tps = sum(single_tokens) / sum(single_times)
        print(f"   Single: {single_tps:.1f} tok/s")

        # Test 2: Batch throughput (5 concurrent)
        print("2. Batch throughput (5 concurrent)...")
        engine.scheduler.reset()

        # Warmup
        _ = engine.generate_batch_sync(formatted[:1], params)

        # Reset for clean measurement
        engine.scheduler.reset()

        start = time.perf_counter()
        results = engine.generate_batch_sync(formatted, params)
        elapsed = time.perf_counter() - start

        total_tokens = sum(r.completion_tokens for r in results)
        batch_tps = total_tokens / elapsed
        print(f"   Batch:  {batch_tps:.1f} tok/s")

        speedup = batch_tps / single_tps

        # Test 3: Speed measurement
        print("3. Generation speed...")
        engine.scheduler.reset()

        start = time.perf_counter()
        result = engine.generate_batch_sync(
            [formatted[0]], SamplingParams(max_tokens=30, temperature=0.0)
        )[0]
        elapsed = time.perf_counter() - start

        ttft_ms = (
            elapsed / result.completion_tokens * 1000
            if result.completion_tokens > 0
            else 0
        )
        gen_tps = result.completion_tokens / elapsed if elapsed > 0 else 0

        print(f"   TTFT:   ~{ttft_ms:.1f}ms (estimated)")
        print(f"   Speed:  {gen_tps:.1f} tok/s")

        return {
            "model": model_name.split("/")[-1],
            "single_tps": single_tps,
            "batch_tps": batch_tps,
            "speedup": speedup,
            "ttft_ms": ttft_ms,
            "gen_tps": gen_tps,
        }
    finally:
        # Always close the engine to release model ownership
        engine.close()


def main():
    models = [
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Qwen3-0.6B-8bit",
        "mlx-community/Qwen3-30B-A3B-4bit",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    ]

    results = []
    for model_name in models:
        try:
            result = benchmark_model(model_name)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print("\n### Continuous Batching Results\n")
    print("| Model | Single | Batch (5 req) | Speedup |")
    print("|-------|--------|---------------|---------|")
    for r in results:
        print(
            f"| {r['model']} | {r['single_tps']:.1f} tok/s | {r['batch_tps']:.1f} tok/s | **{r['speedup']:.2f}x** |"
        )

    print("\n### Generation Speed\n")
    print("| Model | TTFT | Speed |")
    print("|-------|------|-------|")
    for r in results:
        print(f"| {r['model']} | ~{r['ttft_ms']:.1f}ms | {r['gen_tps']:.1f} tok/s |")


if __name__ == "__main__":
    main()
