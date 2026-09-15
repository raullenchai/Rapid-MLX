# SPDX-License-Identifier: Apache-2.0
"""Test generate_batch_sync() performance."""

import asyncio
import time


def main():
    from mlx_lm import load

    from rapid_mlx import EngineConfig, EngineCore, SamplingParams, SchedulerConfig

    MODEL = "mlx-community/Qwen3-0.6B-8bit"
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)

    base_prompts = [
        "What is 2+2?",
        "Name 3 colors.",
        "What is Python?",
        "Capital of Japan?",
        "Who wrote Hamlet?",
    ]

    def format_prompt(p):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )

    params = SamplingParams(max_tokens=50, temperature=0.7)

    print("\n" + "=" * 70)
    print("BATCH SIZE SCALING TEST: generate_batch_sync()")
    print("=" * 70)
    print(
        f"{'Batch':>6} | {'Time':>8} | {'Tokens':>7} | {'Tok/s':>8} | {'% README':>8}"
    )
    print("-" * 70)

    for multiplier in [1, 2, 4, 8, 16]:
        # Create fresh engine for each test to avoid cache state issues
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=256,
                prefill_batch_size=8,
                completion_batch_size=32,
            )
        )
        engine = EngineCore(model, tokenizer, config)

        prompts = base_prompts * multiplier
        formatted = [format_prompt(p) for p in prompts]

        start = time.perf_counter()
        results = engine.generate_batch_sync(formatted, params)
        elapsed = time.perf_counter() - start

        total_tokens = sum(r.completion_tokens for r in results)
        throughput = total_tokens / elapsed
        pct = throughput / 1003.7 * 100

        print(
            f"{len(prompts):>6} | {elapsed:>7.2f}s | {total_tokens:>7} | {throughput:>7.1f} | {pct:>7.1f}%"
        )

    print("-" * 70)
    print("README benchmark: 1003.7 tok/s (5 prompts, 50 max_tokens)")

    # Async comparison
    print("\n" + "=" * 70)
    print("ASYNC generate() COMPARISON (5 prompts)")
    print("=" * 70)

    async def run_async():
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=256,
                prefill_batch_size=8,
                completion_batch_size=32,
            )
        )
        engine = EngineCore(model, tokenizer, config)
        await engine.start()

        try:
            formatted = [format_prompt(p) for p in base_prompts]

            start = time.perf_counter()
            tasks = [engine.generate(p, params) for p in formatted]
            outputs = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start

            total_tokens = sum(r.completion_tokens for r in outputs)
            return total_tokens, elapsed
        finally:
            await engine.stop()

    tokens, elapsed = asyncio.run(run_async())
    throughput = tokens / elapsed
    pct = throughput / 1003.7 * 100

    print(f"Tokens: {tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} tok/s ({pct:.1f}% of README)")


if __name__ == "__main__":
    main()
