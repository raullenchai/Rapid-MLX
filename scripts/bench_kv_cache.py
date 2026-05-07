#!/usr/bin/env python3
"""KV cache quantization micro-bench — synthetic FP16 cache → 8-bit/4-bit
quantize/dequantize → memory ratio + reconstruction error.

Internal dev micro-bench. Operates on random tensors of the requested
shape; does NOT load a real model. Lives in ``scripts/`` rather than the
user-facing CLI because the numbers it reports are meaningful only when
tuning the KV-cache quantization implementation — end users see the
effect via ``--kv-cache-quantization`` on a real ``rapid-mlx serve`` and
the integration benchmarks under ``rapid-mlx bench``.

Usage:
    python3 scripts/bench_kv_cache.py
    python3 scripts/bench_kv_cache.py --layers 64 --seq-len 4096 --heads 16
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from vllm_mlx.memory_cache import (
    _dequantize_cache,
    _quantize_cache,
    estimate_kv_cache_memory,
)


def run(
    n_layers: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    group_size: int,
) -> None:
    print("=" * 70)
    print(" KV Cache Quantization Benchmark")
    print("=" * 70)
    print()
    print(
        f"Config: {n_layers} layers, seq_len={seq_len}, "
        f"n_heads={n_heads}, head_dim={head_dim}"
    )
    print()

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

    results = []
    for bits in [8, 4]:
        # Quantize.
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

        # Dequantize.
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

        # Reconstruction error.
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--layers", type=int, default=32, help="Number of layers (default: 32)"
    )
    p.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length (default: 512)"
    )
    p.add_argument(
        "--heads", type=int, default=32, help="Number of attention heads (default: 32)"
    )
    p.add_argument(
        "--head-dim", type=int, default=128, help="Head dimension (default: 128)"
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    args = p.parse_args()
    run(args.layers, args.seq_len, args.heads, args.head_dim, args.group_size)


if __name__ == "__main__":
    main()
