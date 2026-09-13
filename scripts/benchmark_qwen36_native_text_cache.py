#!/usr/bin/env python3
"""Qualify Qwen3.6 native-cache text routing against its MLLM fallback."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

PROMPTS = {
    "coding": (
        "Return only Python code, under 30 lines, implementing "
        "two_sum(nums, target) in O(n) time. Include exactly three assert tests."
    ),
    "creative": "Write a vivid scene about a lighthouse keeper hearing music below the ice.",
    "reasoning": "A shop marks an item up 25%, then discounts it 20%. Explain the net percentage change.",
    "json": 'Return only compact JSON with keys "prime" and "explanation" for whether 221 is prime.',
    "tool": "You have a weather(location) tool. State the exact tool call needed for weather in Kyoto.",
}


def _fenced_payload(text: str) -> str:
    match = re.search(r"```(?:python|json)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _behavioral_pass(name: str, text: str) -> bool:
    """Check task requirements without demanding one exact wording."""

    lowered = text.lower()
    if name == "coding":
        try:
            tree = ast.parse(_fenced_payload(text))
        except SyntaxError:
            return False
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        return (
            "two_sum" in functions
            and sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) == 3
        )
    if name == "creative":
        return (
            ("lighthouse" in lowered or " light" in lowered)
            and any(word in lowered for word in ("ice", "frozen"))
            and any(word in lowered for word in ("music", "song", "melody"))
        )
    if name == "reasoning":
        return bool(re.search(r"(?:0\s*%|no net (?:percentage )?change)", lowered))
    if name == "json":
        try:
            payload = json.loads(_fenced_payload(text))
        except (json.JSONDecodeError, TypeError):
            return False
        return payload.get("prime") is False and bool(payload.get("explanation"))
    if name == "tool":
        try:
            payload = json.loads(_fenced_payload(text))
        except (json.JSONDecodeError, TypeError):
            return False
        return payload == {
            "name": "weather",
            "arguments": {"location": "Kyoto"},
        }
    raise ValueError(f"unknown quality prompt: {name}")


def _sample(
    text: str, tokens: int, elapsed: float, *, include_text: bool = False
) -> dict[str, object]:
    sample: dict[str, object] = {
        "tokens": tokens,
        "elapsed": elapsed,
        "tok_s": tokens / elapsed,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    if include_text:
        sample["text"] = text
    return sample


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--quality-max-tokens", type=int, default=512)
    parser.add_argument("--include-quality-text", action="store_true")
    parser.add_argument("--mmlu-path", type=Path)
    parser.add_argument("--mmlu-samples", type=int, default=0)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--image-expect", default="")
    args = parser.parse_args()

    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.scheduler import SchedulerConfig

    engine = BatchedEngine(
        str(args.model.expanduser().resolve()),
        force_mllm=True,
        scheduler_config=SchedulerConfig(enable_prefix_cache=False),
    )

    async def native(
        prompt: str, max_tokens: int, *, include_text: bool = False
    ) -> dict[str, object]:
        rendered = engine._apply_chat_template(
            [{"role": "user", "content": prompt}], enable_thinking=False
        )
        started = time.perf_counter()
        output = await engine.generate(
            rendered,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        elapsed = time.perf_counter() - started
        return _sample(
            output.raw_text or output.text or "",
            output.completion_tokens,
            elapsed,
            include_text=include_text,
        )

    async def fallback(
        prompt: str, max_tokens: int, *, include_text: bool = False
    ) -> dict[str, object]:
        rendered = engine._apply_chat_template(
            [{"role": "user", "content": prompt}], enable_thinking=False
        )
        native_engine = engine._engine
        engine._engine = None
        try:
            started = time.perf_counter()
            output = await engine.generate(
                rendered,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            elapsed = time.perf_counter() - started
        finally:
            engine._engine = native_engine
        return _sample(
            output.raw_text or output.text or "",
            output.completion_tokens,
            elapsed,
            include_text=include_text,
        )

    async def memory_snapshot() -> dict[str, int]:
        def _snapshot() -> dict[str, int]:
            import mlx.core as mx

            return {
                "active_bytes": int(mx.get_active_memory()),
                "cache_bytes": int(mx.get_cache_memory()),
                "peak_bytes": int(mx.get_peak_memory()),
            }

        return await engine.execute_on_model_worker(_snapshot)

    try:
        await engine.start()
        if not engine._mllm_native_text_engine:
            raise RuntimeError(
                "checkpoint did not qualify for the native-cache text lane"
            )

        startup_memory = await memory_snapshot()
        await fallback(PROMPTS["coding"], 64)
        fallback_warm_memory = await memory_snapshot()
        await native(PROMPTS["coding"], 64)
        native_warm_memory = await memory_snapshot()

        quality = {}
        quality_exact = True
        for name, prompt in PROMPTS.items():
            baseline = await fallback(
                prompt, args.quality_max_tokens, include_text=True
            )
            candidate = await native(prompt, args.quality_max_tokens, include_text=True)
            exact = baseline["sha256"] == candidate["sha256"]
            quality_exact = quality_exact and exact
            quality[name] = {
                "fallback": baseline,
                "native": candidate,
                "exact": exact,
                "fallback_pass": _behavioral_pass(name, str(baseline["text"])),
                "native_pass": _behavioral_pass(name, str(candidate["text"])),
            }

        pairs = []
        for index in range(args.pairs):
            order = ("fallback", "native") if index % 2 == 0 else ("native", "fallback")
            samples = {}
            for lane in order:
                run = fallback if lane == "fallback" else native
                samples[lane] = await run(PROMPTS["coding"], args.max_tokens)
            exact = samples["fallback"]["sha256"] == samples["native"]["sha256"]
            pairs.append({"order": order, **samples, "exact": exact})

        media = None
        if args.image_path is not None:
            image_path = args.image_path.expanduser().resolve()
            started = time.perf_counter()
            image_output = await engine.chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Read the main heading in this screenshot. Reply with only the heading.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": str(image_path)},
                            },
                        ],
                    }
                ],
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                enable_thinking=False,
            )
            image_elapsed = time.perf_counter() - started
            image_text = image_output.raw_text or image_output.text or ""
            expected = args.image_expect.strip().casefold()
            image_pass = bool(image_text.strip()) and (
                not expected or expected in image_text.casefold()
            )
            recovery = await native(
                PROMPTS["json"], args.quality_max_tokens, include_text=True
            )
            media = {
                "image": str(image_path),
                "text": image_text,
                "tokens": image_output.completion_tokens,
                "elapsed": image_elapsed,
                "pass": image_pass,
                "text_recovery_pass": _behavioral_pass(
                    "json", str(recovery.get("text", ""))
                ),
            }

        mmlu = None
        if args.mmlu_path is not None and args.mmlu_samples > 0:
            import pandas as pd

            frame = pd.read_parquet(args.mmlu_path.expanduser().resolve())
            selected = frame.drop_duplicates("subject").head(args.mmlu_samples)
            rows = []
            for index, row in enumerate(selected.itertuples(index=False)):
                choices = "\n".join(
                    f"{chr(65 + choice_index)}. {choice}"
                    for choice_index, choice in enumerate(row.choices)
                )
                prompt = (
                    "Answer this multiple-choice question with only the letter "
                    "A, B, C, or D.\n\n"
                    f"{row.question}\n{choices}\nAnswer:"
                )
                order = (
                    ("fallback", "native")
                    if index % 2 == 0
                    else (
                        "native",
                        "fallback",
                    )
                )
                samples = {}
                for lane in order:
                    run = fallback if lane == "fallback" else native
                    samples[lane] = await run(prompt, 8, include_text=True)
                expected = chr(65 + int(row.answer))

                def _answer(sample: dict[str, object]) -> str | None:
                    match = re.search(r"\b([A-D])\b", str(sample["text"]).upper())
                    return match.group(1) if match else None

                fallback_answer = _answer(samples["fallback"])
                native_answer = _answer(samples["native"])
                rows.append(
                    {
                        "subject": row.subject,
                        "expected": expected,
                        "fallback": fallback_answer,
                        "native": native_answer,
                        "fallback_correct": fallback_answer == expected,
                        "native_correct": native_answer == expected,
                    }
                )
            mmlu = {
                "samples": len(rows),
                "fallback_correct": sum(row["fallback_correct"] for row in rows),
                "native_correct": sum(row["native_correct"] for row in rows),
                "native_regressions": sum(
                    row["fallback_correct"] and not row["native_correct"]
                    for row in rows
                ),
                "native_fixes": sum(
                    not row["fallback_correct"] and row["native_correct"]
                    for row in rows
                ),
                "rows": rows,
            }

        behavioral_pass = all(
            result["fallback_pass"] and result["native_pass"]
            for result in quality.values()
        )
        if not args.include_quality_text:
            for result in quality.values():
                result["fallback"].pop("text", None)
                result["native"].pop("text", None)
        mmlu_pass = mmlu is None or (
            mmlu["native_correct"] >= mmlu["fallback_correct"]
            and mmlu["native_regressions"] == 0
        )

        ratios = [pair["native"]["tok_s"] / pair["fallback"]["tok_s"] for pair in pairs]
        result = {
            "model": str(args.model.expanduser().resolve()),
            "quality": quality,
            "pairs": pairs,
            "quality_exact": quality_exact,
            "behavioral_pass": behavioral_pass,
            "mmlu_pass": mmlu_pass,
            "performance_exact": all(pair["exact"] for pair in pairs),
            "positive_pairs": sum(ratio > 1.0 for ratio in ratios),
            "median_ratio": statistics.median(ratios),
            "fallback_median_tok_s": statistics.median(
                pair["fallback"]["tok_s"] for pair in pairs
            ),
            "native_median_tok_s": statistics.median(
                pair["native"]["tok_s"] for pair in pairs
            ),
            "mmlu": mmlu,
            "media": media,
            "memory": {
                "startup": startup_memory,
                "after_fallback_warmup": fallback_warm_memory,
                "after_native_warmup": native_warm_memory,
                "final": await memory_snapshot(),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if (
            not result["behavioral_pass"]
            or not result["mmlu_pass"]
            or (
                result["media"] is not None
                and (
                    not result["media"]["pass"]
                    or not result["media"]["text_recovery_pass"]
                )
            )
            or result["positive_pairs"] < max(1, args.pairs - 1)
            or result["median_ratio"] < 1.30
        ):
            raise SystemExit(1)
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(_main())
