#!/usr/bin/env python3
"""
Attention roofline benchmark — answers "do we need FlashAttention on Metal?"

Measures the compute throughput of ``mx.fast.scaled_dot_product_attention``
on the local Apple Silicon GPU at prefill-relevant configurations
(causal, fp16/bf16, B=1) and reports % of the chip's theoretical peak.

Background: see GitHub issue #257. Apple's MLX team rejected an explicit
FlashAttention integration (ml-explore/mlx#2955) on the grounds that the
existing SDPA kernel already implements tiled IO-aware attention. This
script gives that claim a number on whatever Apple Silicon you happen to
be running on, so the answer to "would FA-2 help?" is data, not hand-wave.

Usage:
    python3.12 scripts/bench_attention.py
    python3.12 scripts/bench_attention.py --json out.json   # save raw numbers

What we read off the output:
    - achieved TFLOPs/s vs theoretical peak
    - if we are < 50% of peak at long context (16K/32K prefill), there is
      room for a custom attention kernel; if we are > 70%, MLX SDPA is
      already saturating the GPU and no FA port will move the needle.

The peak FLOP estimate is based on Apple's published GPU core count and a
1.4 GHz clock at 256 fp16 MACs/cycle/core (M-series GPU shader datasheet
estimate; Apple does not publish official peak FLOPs). Treat as a
ballpark — the achieved/peak ratio is what matters for the conclusion.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

try:
    from scripts.bench_metadata import write_bench_json
except ModuleNotFoundError:  # Direct execution outside the repository root.
    from bench_metadata import write_bench_json
from rapid_mlx.optimizations import detect_hardware


def measure_matmul_peak(dtype: mx.Dtype, sizes=(4096, 8192)) -> float:
    """Calibrate the practical fp16 compute ceiling via a square matmul.

    Theoretical peak FLOPs from spec sheets (~57 TFLOPs on M3 Ultra) is
    not achievable in practice — Apple GPUs trade compute peak for
    register pressure / cache flexibility. The achievable ceiling for
    pure ``A @ B`` of large fp16 matrices is what we should compare
    SDPA throughput against, since SDPA is fundamentally two matmuls
    bracketing a softmax. Measured ceiling is typically ~30% of the
    spec-sheet peak.

    Returns the best (peak) TFLOPs/s observed across the candidate sizes.
    """
    best = 0.0
    for n in sizes:
        try:
            a = mx.random.normal((n, n)).astype(dtype)
            b = mx.random.normal((n, n)).astype(dtype)
            mx.eval(a, b)
            for _ in range(2):
                c = a @ b
                mx.eval(c)
            t0 = time.perf_counter()
            iters = 3 if n >= 8192 else 5
            for _ in range(iters):
                c = a @ b
                mx.eval(c)
            dt = (time.perf_counter() - t0) / iters
            tflops = 2 * (n**3) / dt / 1e12
            best = max(best, tflops)
        except Exception:
            continue
    return best


def causal_attention_flops(B: int, H: int, N: int, D: int) -> float:
    """Forward-pass FLOPs for causal scaled-dot-product attention.

    Two matmuls dominate: QK^T (B*H*N*N*D MACs ≈ 2*B*H*N*N*D FLOPs) and
    attn@V (same shape). Causal mask halves both. Softmax/scale are O(N²)
    and dominated by the matmuls — ignored.
    """
    matmul_flops = 4 * B * H * N * N * D
    return matmul_flops / 2.0  # causal triangular


def time_call(fn, warmup: int = 2, repeats: int = 5) -> float:
    """Median wall time over ``repeats`` calls after ``warmup`` discarded calls.

    Uses ``mx.eval`` to force completion before measuring — MLX is lazy by
    default and ``time.perf_counter`` would miss the actual GPU work
    otherwise.
    """
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def make_call(B: int, H: int, N: int, D: int, dtype: mx.Dtype, causal: bool):
    """Build a closure that runs one SDPA call. Q/K/V shapes (B, H, N, D)."""
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    mx.eval(q, k, v)
    scale = 1.0 / (D**0.5)
    mask = "causal" if causal else None

    def call():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    return call


# Each tuple = (label, B, H, D)
# H = num query heads (we use the same value for KV — pure MHA, no GQA —
# because mx.fast.scaled_dot_product_attention handles GQA via a separate
# kv_heads kwarg we don't model here. The roofline doesn't change with GQA;
# only the KV memory footprint does, and prefill attention compute is
# governed by query head count anyway.)
SHAPES = [
    ("Llama-3 8B (H=32, D=128)", 1, 32, 128),
    ("Qwen3 8B (H=32, D=128)", 1, 32, 128),
    ("Llama-3 70B (H=64, D=128)", 1, 64, 128),
    ("DeepSeek MLA-style (H=128, D=64)", 1, 128, 64),
]
SEQ_LENS = [1024, 4096, 16384, 32768]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="dtype for Q/K/V (default: float16)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=SEQ_LENS,
        help="prefill lengths to benchmark (default: 1024 4096 16384 32768)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed iterations per config; we take the median (default 5)",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="disable causal masking (doubles FLOPs, more pessimistic roofline)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="if set, write raw results JSON to this path",
    )
    args = parser.parse_args()

    dtype = {"float16": mx.float16, "bfloat16": mx.bfloat16}[args.dtype]
    causal = not args.no_causal

    hw = detect_hardware()

    print("# Attention SDPA roofline benchmark")
    print()
    print(
        f"- chip: **{hw.chip_name}** ({hw.gpu_cores} GPU cores, "
        f"{hw.memory_bandwidth_gbs} GB/s)"
    )
    print(f"- dtype: {args.dtype}, causal: {causal}, repeats: {args.repeats}")
    print()
    print("Calibrating practical fp16 compute ceiling via square matmul...")
    peak_tflops = measure_matmul_peak(dtype)
    print(
        f"- **measured matmul peak: {peak_tflops:.1f} TFLOPs/s** "
        f"(this is what SDPA can realistically saturate; "
        f"spec-sheet peak is ~3× this but unachievable in practice)"
    )
    print()
    print("| shape | seq_len | latency_ms | TFLOPs/s | % of matmul peak |")
    print("|---|---:|---:|---:|---:|")

    raw: list[dict] = []
    for label, B, H, D in SHAPES:
        for N in args.seq_lens:
            try:
                call = make_call(B, H, N, D, dtype, causal)
                latency = time_call(call, repeats=args.repeats)
                flops = (
                    causal_attention_flops(B, H, N, D)
                    if causal
                    else (2 * causal_attention_flops(B, H, N, D))
                )
                tflops_s = flops / latency / 1e12
                pct = tflops_s / peak_tflops * 100
                raw.append(
                    {
                        "shape": label,
                        "B": B,
                        "H": H,
                        "D": D,
                        "N": N,
                        "latency_ms": round(latency * 1000, 3),
                        "tflops_s": round(tflops_s, 2),
                        "pct_of_peak": round(pct, 1),
                    }
                )
                print(
                    f"| {label} | {N} | {latency * 1000:.2f} | "
                    f"{tflops_s:.2f} | {pct:.1f}% |"
                )
            except Exception as exc:
                print(f"| {label} | {N} | FAIL: {type(exc).__name__}: {exc} | | |")
                raw.append(
                    {
                        "shape": label,
                        "N": N,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    print()
    valid_pcts = [r["pct_of_peak"] for r in raw if "pct_of_peak" in r]
    if valid_pcts:
        long_ctx = [r for r in raw if r.get("N", 0) >= 16384 and "pct_of_peak" in r]
        long_avg = (
            sum(r["pct_of_peak"] for r in long_ctx) / len(long_ctx) if long_ctx else 0
        )
        print("## Verdict")
        print()
        print(
            f"- mean % of matmul peak across all configs: "
            f"**{sum(valid_pcts) / len(valid_pcts):.1f}%**"
        )
        if long_ctx:
            print(
                f"- mean % of matmul peak at seq_len ≥ 16K "
                f"(long-context prefill): **{long_avg:.1f}%**"
            )
        print()
        # Thresholds relative to MEASURED matmul peak (not spec-sheet peak),
        # so 90% means "SDPA is matmul-saturated; a custom kernel cannot
        # win on raw compute — only on memory-traffic / fusion".
        if long_avg >= 85:
            print(
                "MLX SDPA is at the matmul ceiling at long-context prefill — it is "
                "fully saturating the GPU's compute throughput, and a custom "
                "FlashAttention kernel cannot win on raw compute. Any FA-style port "
                "would have to win via softmax+matmul fusion, which is exactly what "
                "MLX's SDPA already does internally. Recommendation: close #257 with "
                "this data; revisit only if a benchmark on real models shows wall-clock "
                "headroom that points to attention specifically."
            )
        elif long_avg >= 50:
            print(
                "MLX SDPA reaches a moderate fraction of the matmul peak at long "
                "context — there is some headroom (likely from softmax/mask overhead). "
                "Worth trying mlx-mfa as an optional backend (--flash-attention=mfa) "
                "and benchmarking side-by-side before any deeper investment."
            )
        else:
            print(
                "MLX SDPA is well below matmul peak at long context — meaningful "
                "headroom exists for a custom attention kernel. Recommendation: "
                "prototype an mlx-mfa or philipturner/MFA wrapper as a Phase-2 "
                "experiment."
            )

    if args.json:
        payload = {
            "hardware": {
                "chip": hw.chip_name,
                "gpu_cores": hw.gpu_cores,
                "memory_bandwidth_gbs": hw.memory_bandwidth_gbs,
                "estimated_peak_tflops": peak_tflops,
            },
            "config": {
                "dtype": args.dtype,
                "causal": causal,
                "repeats": args.repeats,
            },
            "results": raw,
        }
        write_bench_json(args.json, payload, __file__)
        print()
        print(f"raw results → {args.json}")


if __name__ == "__main__":
    main()
