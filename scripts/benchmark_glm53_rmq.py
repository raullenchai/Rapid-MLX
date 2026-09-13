#!/usr/bin/env python3
"""Compare BF16, uniform Q4, and RMQ GLM checkpoints on one Apple host.

The default prompts are an architecture smoke, not a model-quality benchmark.
For release evidence pass a reviewed JSON prompt list with ``--prompts`` and
run the same command in fresh processes on the full checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from vllm_mlx.patches.glm5_next_forget_gate_quant import (
    install_glm5_next_forget_gate_quant_fix,
)
from vllm_mlx.patches.glm5_next_runtime import install_glm5_next_runtime_fix

DEFAULT_PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "If a contract starts on January 1 and lasts 90 days, describe the end-date calculation.",
    "Solve 37 * 19 and show a compact check.",
]


def _load(path: Path):
    from mlx_vlm.utils import load

    started = time.perf_counter()
    model, processor = load(str(path), lazy=False, strict=True)
    return model, processor, time.perf_counter() - started


def _log_softmax(logits: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _teacher_forced(reference, candidate, processor, prompts: list[str]) -> dict:
    agreements = []
    kls = []
    rmses = []
    for prompt in prompts:
        ids = processor.encode(prompt, add_special_tokens=True)
        tokens = mx.array([ids])
        ref = reference(tokens).logits
        got = candidate(tokens).logits
        ref_logp = _log_softmax(ref)
        got_logp = _log_softmax(got)
        ref_prob = mx.exp(ref_logp)
        agreement = mx.mean(mx.argmax(ref, axis=-1) == mx.argmax(got, axis=-1))
        kl = mx.mean(mx.sum(ref_prob * (ref_logp - got_logp), axis=-1))
        rmse = mx.sqrt(mx.mean((ref.astype(mx.float32) - got.astype(mx.float32)) ** 2))
        mx.eval(agreement, kl, rmse)
        agreements.append(float(agreement.item()))
        kls.append(float(kl.item()))
        rmses.append(float(rmse.item()))
    return {
        "top1_agreement": statistics.fmean(agreements),
        "kl_divergence": statistics.fmean(kls),
        "logit_rmse": statistics.fmean(rmses),
    }


def _generation(model, processor, prompts: list[str], max_tokens: int) -> dict:
    from mlx_vlm import generate

    # Compile/materialize the common single-request path before measurement.
    generate(model, processor, "Warm up.", max_tokens=4, temperature=0, verbose=False)
    prompt_tps = []
    generation_tps = []
    outputs = []
    for prompt in prompts:
        result = generate(
            model,
            processor,
            prompt,
            max_tokens=max_tokens,
            temperature=0,
            verbose=False,
        )
        prompt_tps.append(float(result.prompt_tps))
        generation_tps.append(float(result.generation_tps))
        outputs.append(result.text)
    digest = hashlib.sha256("\n---\n".join(outputs).encode()).hexdigest()
    return {
        "median_prompt_tps": statistics.median(prompt_tps),
        "median_generation_tps": statistics.median(generation_tps),
        "output_sha256": digest,
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--q4", type=Path, required=True)
    parser.add_argument("--rmq", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.prompts:
        prompts = json.loads(args.prompts.read_text())
        if (
            not isinstance(prompts, list)
            or not prompts
            or not all(isinstance(prompt, str) and prompt for prompt in prompts)
        ):
            raise SystemExit("--prompts must contain a non-empty JSON string list")
    else:
        prompts = DEFAULT_PROMPTS

    install_glm5_next_runtime_fix()
    install_glm5_next_forget_gate_quant_fix()
    models = {}
    processors = {}
    report = {"models": {}, "prompt_count": len(prompts)}
    for label, path in (("bf16", args.bf16), ("q4", args.q4), ("rmq", args.rmq)):
        models[label], processors[label], load_seconds = _load(path)
        report["models"][label] = {
            "path": str(path.resolve()),
            "load_seconds": load_seconds,
            "active_memory_bytes_after_load": int(mx.get_active_memory()),
        }

    for label in ("q4", "rmq"):
        report["models"][label]["teacher_forced_vs_bf16"] = _teacher_forced(
            models["bf16"], models[label], processors["bf16"], prompts
        )
    for label in ("bf16", "q4", "rmq"):
        report["models"][label]["generation"] = _generation(
            models[label], processors[label], prompts, args.max_tokens
        )
    report["warning"] = (
        "Tiny-fixture throughput is an architecture smoke and must not be used as a "
        "full GLM-5.3 performance claim."
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
