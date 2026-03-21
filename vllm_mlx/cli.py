#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for vllm-mlx.

Commands:
    vllm-mlx serve <model> --port 8000    Start OpenAI-compatible server
    vllm-mlx bench <model>                Run benchmark

Usage:
    vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
    vllm-mlx bench mlx-community/Llama-3.2-1B-Instruct-4bit --num-prompts 10
"""

import argparse
import sys


def serve_command(args):
    """Start the OpenAI-compatible server."""
    import logging
    import os
    import sys

    import uvicorn

    # Import unified server
    from . import server
    from .scheduler import SchedulerConfig
    from .server import RateLimiter, app, load_model

    logger = logging.getLogger(__name__)

    # Validate tool calling arguments
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        print("Error: --enable-auto-tool-choice requires --tool-call-parser")
        print("Example: --enable-auto-tool-choice --tool-call-parser mistral")
        sys.exit(1)

    # Validate gpu-memory-utilization range
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        print(
            "Error: --gpu-memory-utilization must be between 0.0 (exclusive) and 1.0 (inclusive)"
        )
        sys.exit(1)

    # Pass alias info to server (for /v1/models)
    server._model_alias = getattr(args, "_original_alias", None)

    # Configure server security settings
    server._api_key = args.api_key
    server._default_timeout = args.timeout
    if args.rate_limit > 0:
        server._rate_limiter = RateLimiter(
            requests_per_minute=args.rate_limit, enabled=True
        )

    # Configure GC control
    gc_control = args.gc_control and not args.no_gc_control
    server._gc_control = gc_control

    # Configure system prompt pinning
    server._pin_system_prompt = args.pin_system_prompt

    # Configure tool calling
    if args.enable_auto_tool_choice and args.tool_call_parser:
        server._enable_auto_tool_choice = True
        server._tool_call_parser = args.tool_call_parser
        server._enable_tool_logits_bias = getattr(
            args, "enable_tool_logits_bias", False
        )
    else:
        server._enable_auto_tool_choice = False
        server._tool_call_parser = None
        server._enable_tool_logits_bias = False

    # Configure generation defaults
    if args.default_temperature is not None:
        server._default_temperature = args.default_temperature
    if args.default_top_p is not None:
        server._default_top_p = args.default_top_p

    # Configure reasoning parser
    if args.reasoning_parser:
        try:
            from .reasoning import get_parser

            parser_cls = get_parser(args.reasoning_parser)
            server._reasoning_parser = parser_cls()
            server._reasoning_parser_name = args.reasoning_parser
            logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: Failed to import reasoning module: {e}")
            sys.exit(1)
        except Exception as e:
            print(
                f"Error: Failed to initialize reasoning parser "
                f"'{args.reasoning_parser}': {e}"
            )
            sys.exit(1)
    else:
        server._reasoning_parser = None

    # Startup summary
    print()
    print("  Rapid-MLX")
    print("  ─────────")
    features = []
    if args.enable_auto_tool_choice:
        bias_info = (
            " + logits bias" if getattr(args, "enable_tool_logits_bias", False) else ""
        )
        features.append(f"tools: {args.tool_call_parser}{bias_info}")
    if args.reasoning_parser:
        features.append(f"reasoning: {args.reasoning_parser}")
    if args.api_key:
        features.append("auth: on")
    if args.rate_limit > 0:
        features.append(f"rate-limit: {args.rate_limit}/min")
    if args.cloud_model:
        features.append(f"cloud: {args.cloud_model}")
    if gc_control:
        features.append("gc-control")
    if args.pin_system_prompt:
        features.append("pin-system-prompt")
    if features:
        print(f"  Features: {', '.join(features)}")

    print(f"  Model: {args.model}")
    if args.draft_model:
        print(f"  Speculative: {args.draft_model} ({args.num_draft_tokens} draft tokens)")
    # Store MCP config path for FastAPI startup
    if args.mcp_config:
        print(f"MCP config: {args.mcp_config}")
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Pre-load embedding model if specified
    if args.embedding_model:
        print(f"Pre-loading embedding model: {args.embedding_model}")
        server.load_embedding_model(args.embedding_model, lock=True)
        print(f"Embedding model loaded: {args.embedding_model}")

    # Build scheduler config for batched mode
    scheduler_config = None
    if args.continuous_batching:
        # Handle prefix cache flags
        enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # Chunked prefill
            chunked_prefill_tokens=args.chunked_prefill_tokens,
            # MTP
            enable_mtp=args.enable_mtp,
            mtp_num_draft_tokens=args.mtp_num_draft_tokens,
            mtp_optimistic=args.mtp_optimistic,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        )

        print("Mode: Continuous batching (for multiple concurrent users)")
        if args.chunked_prefill_tokens > 0:
            print(f"Chunked prefill: {args.chunked_prefill_tokens} tokens per step")
        if args.enable_mtp:
            print(f"MTP: enabled, draft_tokens={args.mtp_num_draft_tokens}")
        print(f"Stream interval: {args.stream_interval} tokens")
        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )
        elif enable_prefix_cache and not args.no_memory_aware_cache:
            cache_info = (
                f"{args.cache_memory_mb}MB"
                if args.cache_memory_mb
                else f"{args.cache_memory_percent * 100:.0f}% of RAM"
            )
            print(f"Memory-aware cache: {cache_info}")
            if args.kv_cache_quantization:
                print(
                    f"KV cache quantization: {args.kv_cache_quantization_bits}-bit, "
                    f"group_size={args.kv_cache_quantization_group_size}"
                )
        elif enable_prefix_cache:
            print(f"Prefix cache: max_entries={args.prefix_cache_size}")
    else:
        print("Mode: Simple (maximum throughput)")

    # Load model with unified server
    load_model(
        args.model,
        use_batching=args.continuous_batching,
        scheduler_config=scheduler_config,
        stream_interval=args.stream_interval if args.continuous_batching else 1,
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
        gpu_memory_utilization=args.gpu_memory_utilization,
        draft_model=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        cloud_model=args.cloud_model,
        cloud_threshold=args.cloud_threshold,
        cloud_api_base=args.cloud_api_base,
        cloud_api_key=args.cloud_api_key,
    )

    # Start server
    print()
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"  Ready: http://{host_display}:{args.port}/v1")
    print(f"  Docs:  http://{host_display}:{args.port}/docs")
    print()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def bench_command(args):
    """Run benchmark."""
    import asyncio
    import time

    from mlx_lm import load

    from .engine_core import AsyncEngineCore, EngineConfig
    from .request import SamplingParams
    from .scheduler import SchedulerConfig

    # Handle prefix cache flags
    enable_prefix_cache = args.enable_prefix_cache and not args.disable_prefix_cache

    async def run_benchmark():
        print(f"Loading model: {args.model}")
        model, tokenizer = load(args.model)

        scheduler_config = SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            prefill_batch_size=args.prefill_batch_size,
            completion_batch_size=args.completion_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_size=args.prefix_cache_size,
            # Memory-aware cache options
            use_memory_aware_cache=not args.no_memory_aware_cache,
            cache_memory_mb=args.cache_memory_mb,
            cache_memory_percent=args.cache_memory_percent,
            # Paged cache options
            use_paged_cache=args.use_paged_cache,
            paged_cache_block_size=args.paged_cache_block_size,
            max_cache_blocks=args.max_cache_blocks,
            # KV cache quantization
            kv_cache_quantization=args.kv_cache_quantization,
            kv_cache_quantization_bits=args.kv_cache_quantization_bits,
            kv_cache_quantization_group_size=args.kv_cache_quantization_group_size,
            kv_cache_min_quantize_tokens=args.kv_cache_min_quantize_tokens,
        )
        engine_config = EngineConfig(
            model_name=args.model,
            scheduler_config=scheduler_config,
        )

        if args.use_paged_cache:
            print(
                f"Paged cache: block_size={args.paged_cache_block_size}, max_blocks={args.max_cache_blocks}"
            )

        # Generate prompts
        prompts = [
            f"Write a short poem about {topic}."
            for topic in [
                "nature",
                "love",
                "technology",
                "space",
                "music",
                "art",
                "science",
                "history",
                "food",
                "travel",
            ][: args.num_prompts]
        ]

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
        )

        print(
            f"\nRunning benchmark with {len(prompts)} prompts, max_tokens={args.max_tokens}"
        )
        print("-" * 50)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        async with AsyncEngineCore(model, tokenizer, engine_config) as engine:
            await asyncio.sleep(0.1)  # Warm up

            start_time = time.perf_counter()

            # Add all requests
            request_ids = []
            for prompt in prompts:
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect all outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=120):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            total_time = time.perf_counter() - start_time

        # Calculate stats
        for r in results:
            if r:
                total_prompt_tokens += r.prompt_tokens
                total_completion_tokens += r.completion_tokens

        total_tokens = total_prompt_tokens + total_completion_tokens

        print("\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Prompts/second: {len(prompts) / total_time:.2f}")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total completion tokens: {total_completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Tokens/second: {total_completion_tokens / total_time:.2f}")
        print(f"  Throughput: {total_tokens / total_time:.2f} tok/s")

    asyncio.run(run_benchmark())


def bench_detok_command(args):
    """Benchmark streaming detokenizer optimization."""
    import statistics
    import time

    from mlx_lm import load
    from mlx_lm.generate import generate

    print("=" * 70)
    print(" Streaming Detokenizer Benchmark")
    print("=" * 70)
    print()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Generate tokens for benchmark
    prompt = "Write a detailed explanation of how machine learning works and its applications in modern technology."
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

    iterations = args.iterations

    # Benchmark naive decode (old method)
    print("Benchmarking Naive Decode (OLD method)...")
    naive_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for t in generated_tokens:
            _ = tokenizer.decode([t])
        elapsed = time.perf_counter() - start
        naive_times.append(elapsed)

    naive_mean = statistics.mean(naive_times) * 1000

    # Benchmark streaming decode (new method)
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
        elapsed = time.perf_counter() - start
        streaming_times.append(elapsed)

    streaming_mean = statistics.mean(streaming_times) * 1000

    # Results
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

    # Verify correctness (strip for BPE edge cases with leading/trailing spaces)
    print("Verifying correctness...")
    detok = detok_class(tokenizer)
    detok.reset()
    for t in generated_tokens:
        detok.add_token(t)
    detok.finalize()

    batch_result = tokenizer.decode(generated_tokens)
    # BPE tokenizers may have minor edge case differences with spaces
    # Compare stripped versions for functional correctness
    streaming_stripped = detok.text.strip()
    batch_stripped = batch_result.strip()
    if streaming_stripped == batch_stripped:
        print("  ✓ Streaming output matches batch decode")
    elif streaming_stripped in batch_stripped or batch_stripped in streaming_stripped:
        print("  ✓ Streaming output matches (minor BPE edge case)")
    else:
        # Check if most of the content matches (BPE edge cases at boundaries)
        common_len = min(len(streaming_stripped), len(batch_stripped)) - 10
        if (
            common_len > 0
            and streaming_stripped[:common_len] == batch_stripped[:common_len]
        ):
            print("  ✓ Streaming output matches (BPE boundary difference)")
        else:
            print("  ✗ MISMATCH! Results differ")
            print(f"    Streaming: {repr(detok.text[:100])}...")
            print(f"    Batch: {repr(batch_result[:100])}...")


def bench_kv_cache_command(args):
    """Benchmark KV cache quantization memory savings and quality."""
    import time

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from .memory_cache import (
        _dequantize_cache,
        _quantize_cache,
        estimate_kv_cache_memory,
    )

    print("=" * 70)
    print(" KV Cache Quantization Benchmark")
    print("=" * 70)
    print()

    n_layers = args.layers
    seq_len = args.seq_len
    n_heads = args.heads
    head_dim = args.head_dim

    print(
        f"Config: {n_layers} layers, seq_len={seq_len}, "
        f"n_heads={n_heads}, head_dim={head_dim}"
    )
    print()

    # Create synthetic KV cache with random data
    print("Creating synthetic KV cache...")
    cache = []
    for _ in range(n_layers):
        kv = KVCache()
        kv.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.offset = seq_len
        cache.append(kv)
    mx.eval(*[kv.keys for kv in cache], *[kv.values for kv in cache])

    fp16_mem = estimate_kv_cache_memory(cache)
    print(f"FP16 cache memory: {fp16_mem / 1024 / 1024:.2f} MB")
    print()

    # Test each bit width
    results = []
    for bits in [8, 4]:
        group_size = args.group_size

        # Quantize
        start = time.perf_counter()
        quantized = _quantize_cache(cache, bits=bits, group_size=group_size)
        mx.eval(
            *[
                layer.keys[0]
                for layer in quantized
                if hasattr(layer, "keys") and layer.keys is not None
            ]
        )
        quant_time = (time.perf_counter() - start) * 1000

        quant_mem = estimate_kv_cache_memory(quantized)

        # Dequantize
        start = time.perf_counter()
        restored = _dequantize_cache(quantized)
        mx.eval(
            *[
                layer.keys
                for layer in restored
                if hasattr(layer, "keys") and layer.keys is not None
            ]
        )
        dequant_time = (time.perf_counter() - start) * 1000

        # Measure quality
        total_error = 0.0
        max_error = 0.0
        count = 0
        for orig, rest in zip(cache, restored):
            if orig.keys is not None and rest.keys is not None:
                mx.eval(orig.keys, rest.keys, orig.values, rest.values)
                key_err = mx.abs(orig.keys - rest.keys).mean().item()
                val_err = mx.abs(orig.values - rest.values).mean().item()
                key_max = mx.abs(orig.keys - rest.keys).max().item()
                val_max = mx.abs(orig.values - rest.values).max().item()
                total_error += (key_err + val_err) / 2
                max_error = max(max_error, key_max, val_max)
                count += 1

        mean_error = total_error / count if count > 0 else 0.0
        ratio = fp16_mem / quant_mem if quant_mem > 0 else 0.0

        results.append(
            {
                "bits": bits,
                "mem_mb": quant_mem / 1024 / 1024,
                "ratio": ratio,
                "mean_err": mean_error,
                "max_err": max_error,
                "quant_ms": quant_time,
                "dequant_ms": dequant_time,
            }
        )

    # Print results
    fp16_mb = fp16_mem / 1024 / 1024
    print(
        f"{'Mode':<12} {'Memory':>10} {'Savings':>10} "
        f"{'Mean Err':>10} {'Max Err':>10} {'Quant':>10} {'Dequant':>10}"
    )
    print("-" * 72)
    print(
        f"{'FP16':<12} {fp16_mb:>8.2f}MB {'1.00x':>10} "
        f"{'0.000':>10} {'0.000':>10} {'-':>10} {'-':>10}"
    )

    for r in results:
        print(
            f"{r['bits']}-bit{'':<7} {r['mem_mb']:>8.2f}MB "
            f"{r['ratio']:>9.2f}x "
            f"{r['mean_err']:>10.5f} {r['max_err']:>10.5f} "
            f"{r['quant_ms']:>8.1f}ms {r['dequant_ms']:>8.1f}ms"
        )

    print()

    # Recommendation
    best = results[0]  # 8-bit
    print(
        f"Recommendation: 8-bit quantization gives {best['ratio']:.1f}x memory savings "
        f"with mean error {best['mean_err']:.5f}"
    )
    print(
        f"Use 4-bit for maximum compression if quality loss of "
        f"{results[1]['mean_err']:.4f} is acceptable."
    )
    print()
    print("Usage:")
    print("  rapid-mlx serve <model> --continuous-batching --kv-cache-quantization")
    print(
        "  rapid-mlx serve <model> --continuous-batching --kv-cache-quantization "
        "--kv-cache-quantization-bits 4"
    )


def models_command(_args):
    """List available model aliases."""
    from vllm_mlx.model_aliases import list_aliases

    aliases = list_aliases()
    print()
    print("  Available model aliases")
    print("  " + "─" * 50)
    for short, full in sorted(aliases.items()):
        print(f"  {short:<20} → {full}")
    print()
    print(f"  {len(aliases)} aliases available")
    print("  Usage: rapid-mlx serve <alias>")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Rapid-MLX: AI inference for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rapid-mlx serve qwen3.5-9b --port 8000
  rapid-mlx serve mlx-community/Qwen3.5-9B-4bit --port 8000
  rapid-mlx models
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible server")
    serve_parser.add_argument("model", type=str, help="Model to serve")
    serve_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    serve_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching for multiple concurrent users (slower for single user)",
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of device memory for Metal allocation limit and emergency "
        "cache clear threshold (0.0-1.0, default: 0.90). Increase to 0.95 for "
        "large models (200GB+) that need more memory headroom.",
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Chunked prefill
    serve_parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help="Max prefill tokens per scheduler step (0=disabled). "
        "Prevents starvation of active requests during long prefills.",
    )
    # MTP (Multi-Token Prediction)
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help="Enable MTP (Multi-Token Prediction) for models with built-in MTP heads. "
        "Uses cache snapshot/restore for speculative generation.",
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help="Number of draft tokens per MTP step (default: 1)",
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help="Skip MTP acceptance check for maximum speed. "
        "~5-10%% wrong tokens. Best for chat, not for code.",
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )
    # Security options
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable auto tool choice for supported models. Use --tool-call-parser to specify which parser to use.",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=[
            "auto",
            "mistral",
            "qwen",
            "qwen3_coder",
            "llama",
            "hermes",
            "deepseek",
            "kimi",
            "granite",
            "nemotron",
            "xlam",
            "functionary",
            "glm47",
            "minimax",
            "harmony",
            "gpt-oss",
        ],
        help=(
            "Select the tool call parser for the model. Options: "
            "auto (auto-detect), mistral, qwen, qwen3_coder, llama, hermes, "
            "deepseek, kimi, granite, nemotron, xlam, functionary, glm47, minimax, "
            "harmony/gpt-oss. "
            "Required for --enable-auto-tool-choice."
        ),
    )
    # Tool logits bias (jump-forward decoding for tool call structural tokens)
    serve_parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=False,
        help="Bias logits toward structural tool call tokens for faster generation. "
        "Only active when --tool-call-parser is also set. Currently supports minimax.",
    )
    # Speculative decoding options
    serve_parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Draft model for speculative decoding (must use same tokenizer as main model)",
    )
    serve_parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="Number of tokens to generate speculatively per step (default: 4)",
    )
    serve_parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Tokens to process per prefill chunk in simple mode (default: 2048)",
    )
    serve_parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="KV cache quantization bits for simple mode (4 or 8). Reduces memory for long contexts.",
    )
    serve_parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization in simple mode (default: 64)",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Extracts <think>...</think> tags into reasoning_content field. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    # GC control (Tier 0 optimization)
    serve_parser.add_argument(
        "--gc-control",
        action="store_true",
        default=True,
        help="Disable Python GC during generation to avoid latency spikes (default: enabled)",
    )
    serve_parser.add_argument(
        "--no-gc-control",
        action="store_true",
        help="Disable GC control (allow normal GC during generation)",
    )
    # Pinned prefix cache (Tier 0 optimization)
    serve_parser.add_argument(
        "--pin-system-prompt",
        action="store_true",
        default=False,
        help="Auto-pin system prompt in prefix cache to prevent eviction under memory pressure",
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns",
    )
    # Generation defaults
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Override default temperature for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Override default top_p for all requests (default: use model default)",
    )
    # Cloud routing options
    serve_parser.add_argument(
        "--cloud-model",
        type=str,
        default=None,
        help="Cloud model string for litellm (e.g. 'anthropic/claude-sonnet-4-5-20250929'). "
        "When set, large-context requests are routed to the cloud provider.",
    )
    serve_parser.add_argument(
        "--cloud-threshold",
        type=int,
        default=20000,
        help="New token threshold to trigger cloud routing (default: 20000). "
        "Only requests with more new (uncached) tokens than this are routed.",
    )
    serve_parser.add_argument(
        "--cloud-api-base",
        type=str,
        default=None,
        help="Custom API base URL for cloud model (for OpenAI-compatible providers like Zhipu).",
    )
    serve_parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="API key for cloud model (overrides environment variable).",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/embeddinggemma-300m-6bit)",
    )
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("model", type=str, help="Model to benchmark")
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    bench_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    bench_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )

    # Detokenizer benchmark
    detok_parser = subparsers.add_parser(
        "bench-detok", help="Benchmark streaming detokenizer optimization"
    )
    detok_parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default="mlx-community/Qwen3-0.6B-8bit",
        help="Model to use for tokenizer (default: mlx-community/Qwen3-0.6B-8bit)",
    )
    detok_parser.add_argument(
        "--iterations", type=int, default=5, help="Benchmark iterations (default: 5)"
    )

    # KV cache quantization benchmark
    kv_cache_parser = subparsers.add_parser(
        "bench-kv-cache", help="Benchmark KV cache quantization memory savings"
    )
    kv_cache_parser.add_argument(
        "--layers", type=int, default=32, help="Number of layers (default: 32)"
    )
    kv_cache_parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length (default: 512)"
    )
    kv_cache_parser.add_argument(
        "--heads", type=int, default=32, help="Number of attention heads (default: 32)"
    )
    kv_cache_parser.add_argument(
        "--head-dim", type=int, default=128, help="Head dimension (default: 128)"
    )
    kv_cache_parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )

    # Models command
    subparsers.add_parser("models", help="List available model aliases")

    args = parser.parse_args()

    # Resolve model aliases before dispatch
    if hasattr(args, "model") and args.model:
        from vllm_mlx.model_aliases import resolve_model

        resolved = resolve_model(args.model)
        if resolved != args.model:
            print(f"  Alias: {args.model} → {resolved}")
            args._original_alias = args.model
            args.model = resolved

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "bench-detok":
        bench_detok_command(args)
    elif args.command == "bench-kv-cache":
        bench_kv_cache_command(args)
    elif args.command == "models":
        models_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
