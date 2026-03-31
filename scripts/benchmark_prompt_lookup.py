#!/usr/bin/env python3.12
# SPDX-License-Identifier: Apache-2.0
"""
A/B benchmark: standard generation vs prompt_lookup_generate_step.

Loads a real model once, runs each prompt through both paths, measures:
  - tok/s (decode throughput)
  - TTFT (time to first token)
  - acceptance rate (prompt lookup only)
  - total wall time

Usage:
    python3.12 benchmark_prompt_lookup.py [--model MODEL_PATH] [--max-tokens 512]

No server needed. Runs at the model layer directly.
"""

import argparse
import gc
import logging
import time
from dataclasses import dataclass

import mlx.core as mx

# ---------------------------------------------------------------------------
# Benchmark prompts — designed to produce output with varying repetitiveness
# ---------------------------------------------------------------------------

PROMPTS: list[tuple[str, str, str]] = [
    # (name, category, prompt)
    # HIGH repetition — prompt lookup should shine
    (
        "getter_setter",
        "high_repeat",
        "Generate a Python class called User with 8 getter/setter method pairs "
        "for fields: name, age, email, phone, address, city, state, zip_code. "
        "Each getter returns self._field, each setter assigns. Include type hints. "
        "Output ONLY code.",
    ),
    (
        "json_array",
        "high_repeat",
        "Generate a JSON array with 15 user objects. Each object has: "
        "id (int), name (string), email (string), role (admin/user/editor), "
        "active (boolean). Output ONLY valid JSON.",
    ),
    (
        "sql_inserts",
        "high_repeat",
        "Generate 15 SQL INSERT INTO products (id, name, price, category) VALUES "
        "statements with realistic data. Output ONLY SQL.",
    ),
    (
        "html_list",
        "high_repeat",
        "Generate an HTML <ul> with 20 <li> items, each containing an <a href='#'> "
        "with a world capital city name. Output ONLY HTML.",
    ),
    # MEDIUM repetition
    (
        "markdown_table",
        "med_repeat",
        "Write a markdown table of 10 programming languages with columns: "
        "Name, Year, Paradigm, Typing, Speed. Output ONLY the table.",
    ),
    (
        "csv_data",
        "med_repeat",
        "Generate CSV with header and 15 rows. Columns: id, name, department, "
        "salary. Use realistic data. Output ONLY CSV.",
    ),
    # LOW repetition — prompt lookup should NOT help much
    (
        "creative_story",
        "low_repeat",
        "Write a short creative story (about 200 words) about a robot discovering "
        "music for the first time. Be vivid and original.",
    ),
    (
        "explain_concept",
        "low_repeat",
        "Explain how TCP/IP works in 200 words. Be technical but clear.",
    ),
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    name: str
    category: str
    method: str  # "baseline" or "prompt_lookup"
    completion_tokens: int = 0
    elapsed_sec: float = 0.0
    ttft_sec: float = 0.0
    tok_per_sec: float = 0.0
    # prompt lookup specific
    draft_attempts: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    acceptance_rate: float = 0.0
    generated_text: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Baseline: standard mlx-lm generate_step
# ---------------------------------------------------------------------------


def run_baseline(model, tokenizer, prompt_str: str, max_tokens: int) -> RunResult:
    """Run standard autoregressive decoding via mlx-lm generate_step."""
    from mlx_lm.generate import generate_step
    from mlx_lm.models import cache as mlx_cache

    result = RunResult(name="", category="", method="baseline")

    tokens = mx.array(tokenizer.encode(prompt_str), mx.uint32)
    kv_cache = mlx_cache.make_prompt_cache(model)

    t_start = time.perf_counter()
    t_first = None

    generated_ids = []

    for (token_id, _logprobs), _ in zip(
        generate_step(
            tokens,
            model,
            prompt_cache=kv_cache,
            max_tokens=max_tokens,
        ),
        range(max_tokens),
    ):
        if t_first is None:
            t_first = time.perf_counter()
        tid = token_id if isinstance(token_id, int) else int(token_id)
        generated_ids.append(tid)

        # Check EOS
        if tid == tokenizer.eos_token_id:
            break

    t_end = time.perf_counter()

    result.completion_tokens = len(generated_ids)
    result.elapsed_sec = t_end - t_start
    result.ttft_sec = (t_first - t_start) if t_first else result.elapsed_sec
    decode_time = t_end - (t_first or t_start)
    decode_tokens = max(result.completion_tokens - 1, 1)
    result.tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0
    result.generated_text = tokenizer.decode(generated_ids)

    return result


# ---------------------------------------------------------------------------
# Prompt Lookup: our speculative path
# ---------------------------------------------------------------------------


def run_prompt_lookup(
    model,
    tokenizer,
    prompt_str: str,
    max_tokens: int,
    num_draft: int = 4,
    ngram_size: int = 3,
) -> RunResult:
    """Run prompt lookup speculative decoding."""
    from vllm_mlx.speculative.prompt_lookup import prompt_lookup_generate_step

    result = RunResult(name="", category="", method="prompt_lookup")

    tokens = mx.array(tokenizer.encode(prompt_str), mx.uint32)

    t_start = time.perf_counter()
    t_first = None

    generated_ids = []
    draft_accepted_total = 0
    draft_proposed_total = 0
    draft_attempts = 0

    for token_id, logprobs, from_draft in prompt_lookup_generate_step(
        tokens,
        model,
        num_draft_tokens=num_draft,
        ngram_size=ngram_size,
        max_tokens=max_tokens,
    ):
        if t_first is None:
            t_first = time.perf_counter()

        generated_ids.append(token_id if isinstance(token_id, int) else token_id.item())

        if from_draft:
            draft_accepted_total += 1

        # Check EOS
        if generated_ids[-1] == tokenizer.eos_token_id:
            break

    t_end = time.perf_counter()

    result.completion_tokens = len(generated_ids)
    result.elapsed_sec = t_end - t_start
    result.ttft_sec = (t_first - t_start) if t_first else result.elapsed_sec
    decode_time = t_end - (t_first or t_start)
    decode_tokens = max(result.completion_tokens - 1, 1)
    result.tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0
    result.generated_text = tokenizer.decode(generated_ids)
    result.draft_tokens_accepted = draft_accepted_total
    # We can't easily get proposed count from the generator, but the
    # PromptLookupDecoder inside logs it. Estimate from accepted tokens.
    result.acceptance_rate = (
        draft_accepted_total / result.completion_tokens
        if result.completion_tokens > 0
        else 0
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Apply chat template to get the full prompt string."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"User: {prompt}\nAssistant:"


def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark: baseline vs prompt lookup"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model path or HF repo",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--num-draft", type=int, default=4, help="Draft tokens for lookup"
    )
    parser.add_argument(
        "--ngram-size", type=int, default=3, help="N-gram size for lookup"
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Run only these prompt names (default: all)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()
    
    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load model once
    print(f"Loading model: {args.model}")
    try:
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(args.model)
    except (ValueError, Exception) as e:
        if "parameters not in model" in str(e) or "Missing" in str(e):
            print(
                f"  Standard load failed ({e.__class__.__name__}), retrying with strict=False..."
            )
            from pathlib import Path

            from mlx_lm.utils import load_model, load_tokenizer

            model_path = Path(args.model)
            model, _ = load_model(model_path, strict=False)
            tokenizer = load_tokenizer(model_path, {})
        else:
            raise
    print(f"Model loaded. Max tokens per prompt: {args.max_tokens}\n")

    # Filter prompts
    prompts = PROMPTS
    if args.prompts:
        prompts = [(n, c, p) for n, c, p in PROMPTS if n in args.prompts]
        if not prompts:
            print(f"No matching prompts. Available: {[n for n, _, _ in PROMPTS]}")
            return

    results: list[RunResult] = []

    # Warmup
    print("Warming up...")
    warmup_prompt = apply_chat_template(tokenizer, "Say hi.")
    run_baseline(model, tokenizer, warmup_prompt, max_tokens=10)
    gc.collect()
    mx.clear_cache()

    for name, category, raw_prompt in prompts:
        prompt_str = apply_chat_template(tokenizer, raw_prompt)

        # --- Baseline ---
        print(f"  {name} [baseline] ...", end="", flush=True)
        gc.collect()
        mx.clear_cache()

        r_base = run_baseline(model, tokenizer, prompt_str, args.max_tokens)
        r_base.name = name
        r_base.category = category
        results.append(r_base)
        print(
            f" {r_base.completion_tokens} tok, "
            f"{r_base.tok_per_sec:.1f} tok/s, "
            f"TTFT {r_base.ttft_sec:.3f}s"
        )

        # --- Prompt Lookup ---
        print(f"  {name} [prompt_lookup] ...", end="", flush=True)
        gc.collect()
        mx.clear_cache()

        r_lookup = run_prompt_lookup(
            model,
            tokenizer,
            prompt_str,
            args.max_tokens,
            num_draft=args.num_draft,
            ngram_size=args.ngram_size,
        )
        r_lookup.name = name
        r_lookup.category = category
        results.append(r_lookup)
        print(
            f" {r_lookup.completion_tokens} tok, "
            f"{r_lookup.tok_per_sec:.1f} tok/s, "
            f"TTFT {r_lookup.ttft_sec:.3f}s, "
            f"draft_accepted={r_lookup.draft_tokens_accepted} "
            f"({r_lookup.acceptance_rate:.0%})"
        )
        print()

    # --- Summary table ---
    print()
    print("=" * 100)
    print(f"  MODEL: {args.model}")
    print(
        f"  SETTINGS: max_tokens={args.max_tokens}, num_draft={args.num_draft}, ngram={args.ngram_size}"
    )
    print("=" * 100)
    print(
        f"  {'Prompt':<18} {'Category':<12} {'Method':<15} "
        f"{'Tokens':>7} {'Time(s)':>8} {'Tok/s':>8} {'TTFT':>7} "
        f"{'Drafts':>7} {'Accept%':>8}"
    )
    print("-" * 100)

    # Group by prompt name for side-by-side comparison
    prompt_names_seen = []
    for name, _, _ in prompts:
        if name not in prompt_names_seen:
            prompt_names_seen.append(name)

    speedups = []

    for pname in prompt_names_seen:
        pair = [r for r in results if r.name == pname]
        for r in pair:
            draft_str = (
                str(r.draft_tokens_accepted) if r.method == "prompt_lookup" else "--"
            )
            accept_str = (
                f"{r.acceptance_rate:.0%}" if r.method == "prompt_lookup" else "--"
            )
            print(
                f"  {r.name:<18} {r.category:<12} {r.method:<15} "
                f"{r.completion_tokens:>7} {r.elapsed_sec:>8.2f} "
                f"{r.tok_per_sec:>8.1f} {r.ttft_sec:>7.3f} "
                f"{draft_str:>7} {accept_str:>8}"
            )
        # Speedup
        base = [r for r in pair if r.method == "baseline"]
        lookup = [r for r in pair if r.method == "prompt_lookup"]
        if base and lookup and base[0].tok_per_sec > 0:
            sp = lookup[0].tok_per_sec / base[0].tok_per_sec
            speedups.append((pname, pair[0].category, sp))
            print(
                f"  {'':>18} {'':>12} {'>>> SPEEDUP':<15} {'':>7} {'':>8} {sp:>7.2f}x"
            )
        print()

    # --- Aggregate ---
    print("-" * 100)
    if speedups:
        high = [s for n, c, s in speedups if c == "high_repeat"]
        med = [s for n, c, s in speedups if c == "med_repeat"]
        low = [s for n, c, s in speedups if c == "low_repeat"]
        all_sp = [s for _, _, s in speedups]

        def avg(xs):
            return sum(xs) / len(xs) if xs else 0

        print(f"  Average speedup (high_repeat):  {avg(high):.2f}x  (n={len(high)})")
        print(f"  Average speedup (med_repeat):   {avg(med):.2f}x  (n={len(med)})")
        print(f"  Average speedup (low_repeat):   {avg(low):.2f}x  (n={len(low)})")
        print(
            f"  Average speedup (ALL):          {avg(all_sp):.2f}x  (n={len(all_sp)})"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
