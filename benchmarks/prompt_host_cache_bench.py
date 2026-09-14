#!/usr/bin/env python3
"""Measure exact repeated render+tokenize host work without loading weights.

Example (uses an already-cached tokenizer snapshot; never downloads):

    python benchmarks/prompt_host_cache_bench.py \
      --tokenizer-path ~/.cache/huggingface/hub/models--ORG--MODEL/snapshots/REV \
      --characters 4096 32768 125833 --repetitions 50 --out result.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.prompt_host_cache import PromptHostCache
from vllm_mlx.scheduler import Scheduler


def _participants(tokenizer, *, enabled: bool):
    cache = PromptHostCache(enabled=enabled)
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._processor = None
    engine._tokenizer = tokenizer
    engine._model_name = str(getattr(tokenizer, "name_or_path", "benchmark"))
    engine._prompt_host_cache = cache
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = tokenizer
    scheduler.config = SimpleNamespace(model_name=engine._model_name)
    scheduler.prompt_host_cache = cache
    return engine, scheduler, cache


def _run_once(engine, scheduler, messages):
    prompt = engine._apply_chat_template(messages, enable_thinking=False)
    return prompt, scheduler._encode_prompt_string(prompt)


def _median_us(values):
    return statistics.median(values) / 1000.0


def benchmark(tokenizer, characters: int, repetitions: int):
    unit = "Review this local document and retain every exact clause. "
    content = (unit * (characters // len(unit) + 1))[:characters]
    messages = [{"role": "user", "content": content}]
    baseline = _participants(tokenizer, enabled=False)
    candidate = _participants(tokenizer, enabled=True)

    # Warm tokenizer internals and seed the candidate's exact host plane.
    baseline_reference = _run_once(*baseline[:2], messages)
    candidate_reference = _run_once(*candidate[:2], messages)
    if baseline_reference != candidate_reference:
        raise RuntimeError("cached host plane changed prompt or token IDs")

    timings = {"uncached": [], "cached": []}
    last = {}
    for repetition in range(repetitions):
        order = (
            ("uncached", "cached") if repetition % 2 == 0 else ("cached", "uncached")
        )
        for label in order:
            engine, scheduler, _cache = baseline if label == "uncached" else candidate
            started = time.perf_counter_ns()
            last[label] = _run_once(engine, scheduler, messages)
            timings[label].append(time.perf_counter_ns() - started)
    if last["uncached"] != last["cached"]:
        raise RuntimeError("interleaved cached output changed")

    uncached_us = _median_us(timings["uncached"])
    cached_us = _median_us(timings["cached"])
    return {
        "characters": characters,
        "tokens": len(candidate_reference[1]),
        "repetitions": repetitions,
        "uncached_median_us": round(uncached_us, 3),
        "cached_median_us": round(cached_us, 3),
        "speedup": round(uncached_us / cached_us, 3),
        "saved_median_us": round(uncached_us - cached_us, 3),
        "exact": True,
        "cache_stats": candidate[2].stats(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--characters", nargs="+", type=int, default=[32768])
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.tokenizer_path.is_dir():
        parser.error("--tokenizer-path must be an existing local snapshot")
    if args.repetitions < 1 or any(value < 1 for value in args.characters):
        parser.error("characters and repetitions must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    result = {
        "benchmark": "prompt_host_render_and_tokenize",
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "rows": [
            benchmark(tokenizer, characters, args.repetitions)
            for characters in args.characters
        ],
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)


if __name__ == "__main__":
    main()
