#!/usr/bin/env python3.12
# SPDX-License-Identifier: Apache-2.0
"""Benchmark DiffusionGemma 26B-A4B-4bit on rapid-mlx's text-diffusion lane.

Why a separate script (not an entry in ``bench_qwen36_35b_engines.py``):
  - mlx-lm has no diffusion path — only ``mlx-vlm`` does, and rapid-mlx
    is a thin wrapper over ``mlx_vlm.generate.diffusion``. Cross-engine
    decode tok/s is therefore not apples-to-apples with autoregressive
    engines; the right framing is sweeping output length on rapid-mlx
    alone and surfacing how diffusion behaves vs an AR baseline of
    similar parameter count.
  - DiffusionEngine serializes via ``asyncio.Lock`` (mlx-vlm requires
    a single in-flight generator), so B=4 concurrent collapses to ~B=1
    aggregate. We measure B=1 sequential explicitly and skip the
    misleading concurrent column.

Metrics per run (B=1, all OpenAI Chat Completions over localhost):
  TTFT       seconds from request send to first SSE chunk with content
  E2E        seconds from request send to ``data: [DONE]``
  Aggregate TPS  output_tokens / e2e

Diffusion language models emit tokens in **whole denoising blocks** —
all 64 tokens of a single block arrive at the SSE layer in one chunk
once the block completes, not one token at a time. That makes the
classic ``decode_tps = tokens / (e2e − ttft)`` metric meaningless here
(``e2e ≈ ttft`` → division blowup). The metric users actually feel is
``aggregate_tps = total_tokens / total_wall_time`` — how many tokens
land in the chat window per second of real time, regardless of how they
were chunked. That is the column we report.

We sweep ``max_tokens`` across 64 / 256 / 1024 so the report shows how
diffusion scales: aggregate tok/s climbs sharply with output length
because the per-step denoising cost amortizes across more emitted
tokens.

Median of 3 measured runs (1 warmup discarded).

Usage::

    # Assumes a rapid-mlx server is already running on port 18761 with
    # diffusion-gemma-26b loaded. Spawn separately::
    #     rapid-mlx serve diffusion-gemma-26b --port 18761
    python3.12 scripts/bench_diffusion_gemma.py
    python3.12 scripts/bench_diffusion_gemma.py --port 8765 --runs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
PROMPT = (
    "Explain what a diffusion language model is and how it differs from an "
    "autoregressive language model. Cover: token emission order, parallelism, "
    "training objective, and one concrete strength + one concrete weakness."
)


def _measure(base: str, max_tokens: int) -> dict[str, float]:
    """Single B=1 streamed request, return TTFT / E2E / decode_tps / tokens."""

    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    ttft: float | None = None
    completion_tokens: int = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[len(b"data: ") :].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    ttft = time.monotonic() - t0
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])
    e2e = time.monotonic() - t0
    if ttft is None:
        ttft = e2e
    aggregate_tps = completion_tokens / e2e if e2e > 0 else 0.0
    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "aggregate_tps": aggregate_tps,
        "tokens": float(completion_tokens),
    }


def _median(samples: list[dict[str, float]], key: str) -> float:
    return statistics.median(s[key] for s in samples)


def _sweep(base: str, max_tokens: int, runs: int) -> dict[str, float]:
    # 1 warmup discard + ``runs`` measured.
    print(f"  warmup ({max_tokens=})…", flush=True)
    _measure(base, max_tokens)
    samples: list[dict[str, float]] = []
    for i in range(runs):
        print(f"  run {i + 1}/{runs} ({max_tokens=})…", end=" ", flush=True)
        s = _measure(base, max_tokens)
        print(
            f"ttft={s['ttft_s']:.2f}s e2e={s['e2e_s']:.2f}s "
            f"agg={s['aggregate_tps']:.1f}tps tokens={int(s['tokens'])}",
            flush=True,
        )
        samples.append(s)
    return {
        "max_tokens": float(max_tokens),
        "median_ttft_s": _median(samples, "ttft_s"),
        "median_e2e_s": _median(samples, "e2e_s"),
        "median_aggregate_tps": _median(samples, "aggregate_tps"),
        "median_tokens": _median(samples, "tokens"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18761)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--max-tokens-sweep",
        default="64,256,1024",
        help="Comma list of max_tokens values to sweep.",
    )
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    sweep = [int(x) for x in args.max_tokens_sweep.split(",") if x.strip()]
    print(f"DiffusionGemma 26B-A4B-4bit bench (B=1, base={base})")
    print(f"Sweep max_tokens={sweep}, runs={args.runs} (+1 warmup)")
    rows: list[dict[str, float]] = []
    for mt in sweep:
        rows.append(_sweep(base, mt, args.runs))
    print()
    print(
        "| max_tokens | median TTFT (s) | median E2E (s) | "
        "median aggregate tok/s | median tokens |"
    )
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {int(r['max_tokens'])} | {r['median_ttft_s']:.2f} | "
            f"{r['median_e2e_s']:.2f} | {r['median_aggregate_tps']:.1f} | "
            f"{int(r['median_tokens'])} |"
        )
    out = {"model": MODEL, "base": base, "runs": args.runs, "sweep": rows}
    with open("/tmp/diffgemma_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nRaw JSON: /tmp/diffgemma_bench.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
