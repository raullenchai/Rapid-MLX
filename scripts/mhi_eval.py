#!/usr/bin/env python3
"""
MHI Eval — Model-Harness Index
=======================================
Minimal eval suite for benchmarking model × agent harness combinations.

Uses curated subsets of established benchmarks:
  - TAU-bench (10 tasks)  — multi-turn agent tool use
  - HumanEval (10 tasks)  — code generation
  - tinyMMLU  (10 tasks)  — knowledge baseline (harness degradation check)

Usage:
    # Full MHI (30 tasks, ~30 min)
    python3 scripts/mhi_eval.py --base-url http://localhost:8000/v1

    # Single suite
    python3 scripts/mhi_eval.py --base-url http://localhost:8000/v1 --suite tau

    # Custom model name
    python3 scripts/mhi_eval.py --base-url http://localhost:8000/v1 --model qwopus-27b-4bit
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# OpenAI client helper
# ---------------------------------------------------------------------------

def get_client(base_url: str, api_key: str = "not-needed"):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def detect_model(client) -> str:
    models = client.models.list()
    return models.data[0].id if models.data else "unknown"


# ---------------------------------------------------------------------------
# TAU-bench: 10 representative retail agent tasks
# ---------------------------------------------------------------------------

TAU_TASK_IDS = [24, 10, 5, 17, 33, 14, 15, 20, 30, 4]

def run_tau_bench(base_url: str, model: str, api_key: str = "not-needed") -> dict:
    """Run 10 curated TAU-bench retail tasks."""
    try:
        from tau_bench.envs import get_env
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        from tau_bench.types import EnvRunResult
    except ImportError:
        return {"error": "tau-bench not installed. pip install tau-bench @ git+https://github.com/sierra-research/tau-bench.git"}

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url

    # Initialize env (user sim also uses local model)
    env = get_env(
        "retail",
        user_strategy="llm",
        user_model=model,
        user_provider="openai",
        task_split="test",
    )

    agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=model,
        provider="openai",
        temperature=0.0,
    )

    results = []
    for idx in TAU_TASK_IDS:
        isolated_env = get_env(
            "retail",
            user_strategy="llm",
            user_model=model,
            user_provider="openai",
            task_split="test",
            task_index=idx,
        )
        t0 = time.time()
        try:
            res = agent.solve(env=isolated_env, task_index=idx)
            reward = res.reward
            error = None
        except Exception as e:
            reward = 0.0
            error = str(e)
        elapsed = time.time() - t0

        status = "PASS" if reward == 1.0 else "FAIL"
        print(f"  [TAU] Task {idx:3d}: {status} ({elapsed:.1f}s)")
        results.append({
            "task_id": idx,
            "reward": reward,
            "elapsed_s": round(elapsed, 1),
            "error": error,
        })

    passed = sum(1 for r in results if r["reward"] == 1.0)
    score = passed / len(results)
    return {
        "suite": "tau_bench",
        "score": round(score, 3),
        "passed": passed,
        "total": len(results),
        "details": results,
    }


# ---------------------------------------------------------------------------
# HumanEval: 10 code generation tasks
# ---------------------------------------------------------------------------

HUMANEVAL_IDS = [
    "HumanEval/0", "HumanEval/1", "HumanEval/2", "HumanEval/3", "HumanEval/4",
    "HumanEval/5", "HumanEval/6", "HumanEval/7", "HumanEval/8", "HumanEval/9",
]


def run_humaneval(base_url: str, model: str, api_key: str = "not-needed") -> dict:
    """Run 10 HumanEval code generation tasks."""
    try:
        from human_eval.data import read_problems
    except ImportError:
        return {"error": "human-eval not installed. pip install human-eval"}

    client = get_client(base_url, api_key)
    problems = read_problems()

    results = []
    for task_id in HUMANEVAL_IDS:
        p = problems[task_id]
        prompt = p["prompt"]
        test_code = p["test"]
        entry_point = p["entry_point"]

        t0 = time.time()
        try:
            # Use completions endpoint — direct continuation of the function
            # This avoids chat template issues (thinking tokens leaking into content)
            resp = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
                stop=["\ndef ", "\nclass ", "\n#", "\nif __name__"],
            )
            completion = resp.choices[0].text or ""
        except Exception as e:
            results.append({"task_id": task_id, "passed": False, "elapsed_s": round(time.time() - t0, 1), "error": str(e)})
            print(f"  [HumanEval] {task_id}: FAIL (API error)")
            continue

        # Completions endpoint returns direct continuation of the prompt
        # Clean up: stop tokens may leave partial code
        completion = completion.rstrip()
        # Remove trailing incomplete lines (e.g. "def" without body)
        lines = completion.split("\n")
        while lines and lines[-1].strip() in ("def", "class", "#", ""):
            lines.pop()
        completion = "\n".join(lines)
        code = prompt + completion

        # Execute and check
        passed = _check_humaneval(code, test_code, entry_point)
        elapsed = time.time() - t0

        status = "PASS" if passed else "FAIL"
        print(f"  [HumanEval] {task_id}: {status} ({elapsed:.1f}s)")
        results.append({
            "task_id": task_id,
            "passed": passed,
            "elapsed_s": round(elapsed, 1),
            "error": None,
        })

    passed_count = sum(1 for r in results if r.get("passed"))
    score = passed_count / len(results)
    return {
        "suite": "humaneval",
        "score": round(score, 3),
        "passed": passed_count,
        "total": len(results),
        "details": results,
    }


def _extract_code(completion: str, prompt: str, entry_point: str) -> str:
    """Extract usable Python code from model completion."""
    # Strip markdown code blocks
    code = completion
    if "```python" in code:
        code = code.split("```python", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1]
        if "```" in code:
            code = code.split("```", 1)[0]

    # If the model repeated the full function (including signature), use as-is
    if f"def {entry_point}" in code:
        return code.strip()

    # Otherwise, prepend the prompt (which has the function signature)
    return prompt + code.strip()


def _check_humaneval(code: str, test_code: str, entry_point: str) -> bool:
    """Execute code + tests in subprocess, return pass/fail."""
    full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


# ---------------------------------------------------------------------------
# tinyMMLU: 10 knowledge questions (degradation check)
# ---------------------------------------------------------------------------

# Indices spanning different subjects for diversity
MMLU_INDICES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def run_mmlu(base_url: str, model: str, api_key: str = "not-needed") -> dict:
    """Run 10 tinyMMLU questions."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets not installed. pip install datasets"}

    client = get_client(base_url, api_key)
    ds = load_dataset("tinyBenchmarks/tinyMMLU", split="test")

    results = []
    for idx in MMLU_INDICES:
        item = ds[idx]
        question = item["question"]
        choices = item["choices"]
        correct_idx = item["answer"]
        correct_letter = "ABCD"[correct_idx]
        subject = item["subject"]

        # Use the pre-formatted 5-shot prompt from tinyMMLU
        formatted = item.get("input_formatted", "")
        if not formatted:
            choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            formatted = f"{question}\n{choices_text}\nAnswer:"

        t0 = time.time()
        try:
            # Use completions endpoint for MMLU (more reliable for letter extraction)
            resp = client.completions.create(
                model=model,
                prompt=formatted,
                temperature=0.0,
                max_tokens=4,
            )
            answer = resp.choices[0].text or ""
        except Exception as e:
            results.append({"idx": idx, "correct": False, "elapsed_s": round(time.time() - t0, 1), "error": str(e)})
            print(f"  [MMLU] Q{idx} ({subject}): FAIL (API error)")
            continue

        # Extract answer letter
        predicted = _extract_letter(answer)
        correct = predicted == correct_letter
        elapsed = time.time() - t0

        status = "PASS" if correct else f"FAIL (got {predicted}, expected {correct_letter})"
        print(f"  [MMLU] Q{idx} ({subject}): {status} ({elapsed:.1f}s)")
        results.append({
            "idx": idx,
            "subject": subject,
            "correct": correct,
            "predicted": predicted,
            "expected": correct_letter,
            "elapsed_s": round(elapsed, 1),
            "error": None,
        })

    correct_count = sum(1 for r in results if r.get("correct"))
    score = correct_count / len(results)
    return {
        "suite": "tinyMMLU",
        "score": round(score, 3),
        "passed": correct_count,
        "total": len(results),
        "details": results,
    }


def _extract_letter(text: str) -> str:
    """Extract A/B/C/D from model response."""
    import re
    text = text.strip()
    # Direct single letter
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    # "The answer is B" / "Answer: B" / "correct answer is C"
    m = re.search(r'(?:answer|option)\s*(?:is|:)\s*([A-Da-d])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "B." or "B)" at start of line
    m = re.search(r'^([A-Da-d])[.\):]', text, re.MULTILINE)
    if m:
        return m.group(1).upper()
    # Last single letter A-D in the text (models often explain then conclude)
    letters = re.findall(r'\b([A-Da-d])\b', text)
    # Filter to only A-D
    valid = [l.upper() for l in letters if l.upper() in "ABCD"]
    if valid:
        return valid[-1]  # Take last mentioned letter
    return "?"


# ---------------------------------------------------------------------------
# MHI composite score
# ---------------------------------------------------------------------------

WEIGHTS = {
    "tau_bench": 0.50,   # Agent tool use — highest signal for model×harness
    "humaneval": 0.30,   # Code generation
    "tinyMMLU": 0.20,    # Knowledge baseline
}


def compute_mhi(suite_results: dict) -> float:
    """Compute weighted MHI score."""
    total = 0.0
    for suite, weight in WEIGHTS.items():
        if suite in suite_results and "score" in suite_results[suite]:
            total += weight * suite_results[suite]["score"]
    return round(total * 100, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MHI Eval — Model-Harness Index")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if not set)")
    parser.add_argument("--api-key", default="not-needed", help="API key")
    parser.add_argument("--suite", default="all", choices=["all", "tau", "humaneval", "mmlu"], help="Which suite to run")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--label", default=None, help="Label for this run (e.g. 'qwopus27b+hermes')")
    args = parser.parse_args()

    # Detect model
    client = get_client(args.base_url, args.api_key)
    model = args.model or detect_model(client)
    # Generate readable label from model path
    if args.label:
        label = args.label
    else:
        name = model.split("/")[-1]
        # Strip hash-like suffixes
        if len(name) == 40 and all(c in "0123456789abcdef" for c in name):
            # HF snapshot hash — try parent dir
            parts = model.split("/")
            for p in reversed(parts):
                if not all(c in "0123456789abcdef" for c in p) and len(p) > 5:
                    name = p.replace("models--", "").replace("--", "/")
                    break
        label = name[:50]

    print(f"\n{'='*60}")
    print(f"  MHI Eval — Model-Harness Index")
    print(f"  Model: {model}")
    print(f"  Label: {label}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Suite: {args.suite}")
    print(f"{'='*60}\n")

    results = {}
    t_start = time.time()

    # TAU-bench
    if args.suite in ("all", "tau"):
        print("[1/3] TAU-bench (10 agent tasks)...")
        results["tau_bench"] = run_tau_bench(args.base_url, model, args.api_key)
        _print_suite_result(results["tau_bench"])

    # HumanEval
    if args.suite in ("all", "humaneval"):
        print("[2/3] HumanEval (10 code tasks)...")
        results["humaneval"] = run_humaneval(args.base_url, model, args.api_key)
        _print_suite_result(results["humaneval"])

    # tinyMMLU
    if args.suite in ("all", "mmlu"):
        print("[3/3] tinyMMLU (10 knowledge tasks)...")
        results["tinyMMLU"] = run_mmlu(args.base_url, model, args.api_key)
        _print_suite_result(results["tinyMMLU"])

    total_time = time.time() - t_start

    # Compute MHI
    mhi_score = compute_mhi(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"  MHI Score: {mhi_score}")
    print(f"  Label: {label}")
    print(f"  Time: {total_time:.0f}s")
    print()
    for suite, weight in WEIGHTS.items():
        if suite in results and "score" in results[suite]:
            r = results[suite]
            print(f"  {suite:12s}: {r['passed']}/{r['total']} ({r['score']:.0%}) × {weight:.0%} weight")
    print(f"{'='*60}\n")

    # Save results
    output = {
        "mhi_score": mhi_score,
        "label": label,
        "model": model,
        "base_url": args.base_url,
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": round(total_time, 1),
        "weights": WEIGHTS,
        "suites": results,
    }

    out_path = args.output or f"reports/mhi/{label.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")

    return mhi_score


def _print_suite_result(result: dict):
    if "error" in result:
        print(f"  ERROR: {result['error']}\n")
        return
    print(f"  Score: {result['passed']}/{result['total']} ({result['score']:.0%})\n")


if __name__ == "__main__":
    main()
