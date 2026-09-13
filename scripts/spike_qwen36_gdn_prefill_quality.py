#!/usr/bin/env python3
"""Real-weight stock/fast quality probe for Qwen3.6 GDN long-prompt prefill."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
from pathlib import Path


def _hash(tokens: list[int]) -> str:
    payload = ",".join(map(str, tokens)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _answer(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def _json_answer(text: str):
    answer = _answer(text)
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.I)
    return json.loads(answer)


def _valid_safe_divide(text: str) -> bool:
    answer = re.sub(r"^```(?:python)?\s*|\s*```$", "", _answer(text), flags=re.I)
    try:
        tree = ast.parse(answer)
    except SyntaxError:
        return False
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "safe_divide":
        return False
    returns = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Return)]
    return len(returns) == 1 and isinstance(returns[0].value, ast.IfExp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("quality", "performance", "both"), default="both"
    )
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--max-cv", type=float, default=0.05)
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        parser.error("--model must be an existing cached snapshot")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.models import gated_delta as gd
    from mlx_lm.sample_utils import make_sampler

    from vllm_mlx import gdn_prefill
    from vllm_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
    from vllm_mlx.moe_fusion import fuse_gate_up
    from vllm_mlx.qwen35_moe_router import install_qwen35_moe_router

    model, tokenizer = load(model_path)
    fuse_gate_up(model)
    fuse_gdn_in_proj(model)
    install_qwen35_moe_router(model)

    stock_kernel = getattr(gd.gated_delta_kernel, "_stock", gd.gated_delta_kernel)
    gd.gated_delta_kernel = stock_kernel
    gdn_prefill._installed = False
    gdn_prefill._original_kernel = None
    if not gdn_prefill.install():
        raise RuntimeError("GDN prefill fast path did not install")
    fast_kernel = gd.gated_delta_kernel
    sampler = make_sampler(temp=0.0)

    cases = {
        "retrieval": (
            "The exact deployment canary is ORBIT-CEDAR-7319. ",
            "Return only the exact deployment canary.",
            lambda text: _answer(text).rstrip(".") == "ORBIT-CEDAR-7319",
        ),
        "reasoning": (
            "Crate A weighs 17 kg. Crate B weighs 29 kg. Crate C weighs 11 kg. ",
            "Return only the total weight of crates A and C in kilograms.",
            lambda text: bool(re.fullmatch(r"28(?:\s*kg)?\.?", _answer(text), re.I)),
        ),
        "coding": (
            "The required Python fix is `def safe_divide(a, b): return None if b == 0 else a / b`. ",
            "Return only the required corrected Python function.",
            _valid_safe_divide,
        ),
        "creative": (
            "Story brief: protagonist Mara; setting an obsidian orchard; past tense; no dialogue; exactly 90 to 100 words. ",
            "Write only the scene and follow every constraint in the story brief.",
            lambda text: (
                "Mara" in _answer(text)
                and "obsidian" in _answer(text).lower()
                and 90 <= len(re.findall(r"\b[\w’'-]+\b", _answer(text))) <= 100
                and '"' not in _answer(text)
            ),
        ),
        "json": (
            "The number 221 equals 13 times 17. ",
            'Return only compact JSON exactly matching this schema: {"prime": boolean}.',
            lambda text: _json_answer(text) == {"prime": False},
        ),
        "tool": (
            "Available tool: weather(location: string). The requested city is Kyoto. ",
            'Return only compact JSON with exactly keys "name" and "arguments", where arguments has only location.',
            lambda text: (
                _json_answer(text)
                == {"name": "weather", "arguments": {"location": "Kyoto"}}
            ),
        ),
        "ordering": (
            "Priority records: cedar=4, amber=1, cobalt=3, birch=2. ",
            "Return only the record names from lowest to highest priority, comma-separated.",
            lambda text: (
                re.sub(r"\s", "", _answer(text).rstrip("."))
                == "amber,birch,cobalt,cedar"
            ),
        ),
    }
    filler = (
        "Archive note: the northern relay remained nominal after routine inspection. "
        "This sentence is irrelevant background and must not replace explicit instructions. "
    )

    def prompt_tokens(prefix: str, question: str) -> list[int]:
        def render(repeats: int):
            content = prefix + filler * repeats + question
            messages = [{"role": "user", "content": content}]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )

        low, high = 0, args.context_tokens
        best = render(0)
        while low <= high:
            mid = (low + high) // 2
            candidate = render(mid)
            if len(candidate) <= args.context_tokens:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return [int(token) for token in best]

    def run(
        kernel, prompt: list[int], *, max_tokens: int | None = None
    ) -> dict[str, object]:
        gd.gated_delta_kernel = kernel
        mx.random.seed(0)
        responses = list(
            stream_generate(
                model,
                tokenizer,
                prompt,
                sampler=sampler,
                max_tokens=args.max_tokens if max_tokens is None else max_tokens,
            )
        )
        tokens = [int(response.token) for response in responses]
        return {
            "tokens": tokens,
            "text": "".join(response.text for response in responses),
            "prompt_tps": float(responses[-1].prompt_tps),
            "generation_tps": float(responses[-1].generation_tps),
            "hash": _hash(tokens),
        }

    results: dict[str, object] = {}
    all_valid = True
    if args.mode in {"quality", "both"}:
        quality_results = {}
        stock_passes = 0
        fast_passes = 0
        fast_only_failures = 0

        def is_valid(validator, text: str) -> bool:
            try:
                return bool(validator(text))
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                return False

        for index, (name, (prefix, question, validate)) in enumerate(cases.items()):
            prompt = prompt_tokens(prefix, question)
            # Alternate order to avoid systematically favoring the second path.
            if index % 2:
                fast = run(fast_kernel, prompt)
                stock = run(stock_kernel, prompt)
            else:
                stock = run(stock_kernel, prompt)
                fast = run(fast_kernel, prompt)

            stock_valid = is_valid(validate, str(stock["text"]))
            fast_valid = is_valid(validate, str(fast["text"]))
            exact = stock["tokens"] == fast["tokens"]
            stock_passes += int(stock_valid)
            fast_passes += int(fast_valid)
            fast_only_failures += int(stock_valid and not fast_valid)
            quality_results[name] = {
                "prompt_tokens": len(prompt),
                "stock_valid": stock_valid,
                "fast_valid": fast_valid,
                "same_tokens": exact,
                "stock_hash": stock["hash"],
                "fast_hash": fast["hash"],
                "stock_prompt_tps": stock["prompt_tps"],
                "fast_prompt_tps": fast["prompt_tps"],
                "stock_generation_tps": stock["generation_tps"],
                "fast_generation_tps": fast["generation_tps"],
                "stock_answer": _answer(str(stock["text"])),
                "fast_answer": _answer(str(fast["text"])),
                "stock_answer_words": len(
                    re.findall(r"\b[\w’'-]+\b", _answer(str(stock["text"])))
                ),
                "fast_answer_words": len(
                    re.findall(r"\b[\w’'-]+\b", _answer(str(fast["text"])))
                ),
            }
        quality_valid = fast_only_failures == 0
        all_valid &= quality_valid
        results["quality"] = {
            "cases": quality_results,
            "stock_passes": stock_passes,
            "fast_passes": fast_passes,
            "fast_only_failures": fast_only_failures,
            "non_regression_valid": quality_valid,
        }

    if args.mode in {"performance", "both"}:
        if args.pairs < 3:
            parser.error("--pairs must be at least 3")
        perf_prompt = prompt_tokens(
            "The exact deployment canary is ORBIT-CEDAR-7319. ",
            "Return only the exact deployment canary.",
        )
        # Compile and warm both paths outside the measured pairs.
        run(stock_kernel, perf_prompt, max_tokens=1)
        run(fast_kernel, perf_prompt, max_tokens=1)
        pairs = []
        for index in range(args.pairs):
            if index % 2:
                fast = run(fast_kernel, perf_prompt, max_tokens=1)
                stock = run(stock_kernel, perf_prompt, max_tokens=1)
            else:
                stock = run(stock_kernel, perf_prompt, max_tokens=1)
                fast = run(fast_kernel, perf_prompt, max_tokens=1)
            stock_tps = float(stock["prompt_tps"])
            fast_tps = float(fast["prompt_tps"])
            pairs.append(
                {
                    "stock_prompt_tps": stock_tps,
                    "fast_prompt_tps": fast_tps,
                    "speedup_pct": (fast_tps / stock_tps - 1.0) * 100.0,
                }
            )

        stock_values = [float(pair["stock_prompt_tps"]) for pair in pairs]
        fast_values = [float(pair["fast_prompt_tps"]) for pair in pairs]
        speedups = [float(pair["speedup_pct"]) for pair in pairs]

        def cv(values: list[float]) -> float:
            mean = statistics.mean(values)
            return statistics.stdev(values) / mean if mean else float("inf")

        stock_cv = cv(stock_values)
        fast_cv = cv(fast_values)
        positive_pairs = sum(speedup > 0 for speedup in speedups)
        performance_valid = (
            stock_cv <= args.max_cv
            and fast_cv <= args.max_cv
            and positive_pairs >= args.pairs - 1
        )
        all_valid &= performance_valid
        results["performance"] = {
            "prompt_tokens": len(perf_prompt),
            "pairs": pairs,
            "stock_median_prompt_tps": statistics.median(stock_values),
            "fast_median_prompt_tps": statistics.median(fast_values),
            "median_paired_speedup_pct": statistics.median(speedups),
            "stock_cv": stock_cv,
            "fast_cv": fast_cv,
            "positive_pairs": positive_pairs,
            "valid": performance_valid,
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
