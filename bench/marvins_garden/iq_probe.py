# SPDX-License-Identifier: Apache-2.0
"""Marvin's Garden — general-intelligence "IQ tax" probe.

A/B the BASE model vs the SAME base with the Marvin decision adapter
loaded, on the repo's own standard eval suites (generation mode, not
readout): reasoning.json (10 MATH-500), general.json (10 MMLU-Pro MC),
coding.json (10 tasks with executed test_code).

Grading uses the OFFICIAL evals/run_eval.py functions (extract_answer,
normalize_answer, extract_python_code, check_general_response) and the
same prompt wrappers, so numbers are comparable with repo eval history.
Generation is direct via mlx-lm (no server), greedy decoding.

Usage::

    python bench/marvins_garden/iq_probe.py --model <model> [--adapter <dir>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_PROMPTS = REPO_ROOT / "evals" / "prompts"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("mg_run_eval", REPO_ROOT / "evals" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_eval = _load_run_eval()


def _strip(text: str) -> str:
    try:
        return run_eval._strip_thinking(text)
    except AttributeError:
        return text


def _apply_chat(tokenizer, user_content: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **kwargs
            )
        except TypeError:
            continue
    raise RuntimeError("apply_chat_template failed")


def _strip_think_blocks(text: str) -> str:
    """Guard against qwen3-style templates that still open a think block."""
    if "idensea" in text:
        return text.rsplit("idensea", 1)[-1].strip()
    return text


def _generate(model, tokenizer, prompt_text: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    return str(generate(model, tokenizer, prompt=prompt_text, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0)))


REASONING_WRAPPER = (
    "Solve this math problem step by step. "
    'At the end, write your final answer after "####". '
    "If the answer is a fraction, write it as a/b.\n\n"
    "Problem: {question}\n\nSolution:"
)

GENERAL_SYSTEM = (
    "You are taking a multiple choice test. Read the question and options "
    "carefully, then respond with just the letter of your answer. Do not "
    "show your thinking process."
)


def run_arm(model, tokenizer, suites) -> dict:
    per_suite: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for suite, rows in suites.items():
        results = []
        for row in rows:
            if suite == "reasoning":
                prompt_text = _apply_chat(tokenizer, REASONING_WRAPPER.format(question=row["question"]))
                response = _strip_think_blocks(_strip(_generate(model, tokenizer, prompt_text, 512)))
                got = run_eval.normalize_answer(run_eval.extract_answer(response))
                want = run_eval.normalize_answer(row["answer"])
                ok = got is not None and want is not None and got == want
            elif suite == "general":
                prompt_text = _apply_chat(tokenizer, row["prompt"], system=GENERAL_SYSTEM)
                response = _strip_think_blocks(_strip(_generate(model, tokenizer, prompt_text, 256)))
                ok, _reason = run_eval.check_general_response(response, row.get("checks", {}))
            else:  # coding — mirrors run_coding_suite's exec harness
                prompt_text = _apply_chat(tokenizer, row["prompt"])
                response = _strip_think_blocks(_generate(model, tokenizer, prompt_text, 512))
                code = run_eval.extract_python_code(response)
                full_code = code + "\n\n" + row["test_code"]
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
                    fh.write(full_code + "\n")
                    path = fh.name
                try:
                    proc = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
                    ok = proc.returncode == 0
                except subprocess.TimeoutExpired:
                    ok = False
                finally:
                    Path(path).unlink(missing_ok=True)
            results.append(bool(ok))
            print(f"  [{suite}] {row['id']}: {'PASS' if ok else 'FAIL'}", flush=True)
        per_suite[suite] = {
            "n": len(results),
            "passed": sum(results),
            "accuracy": sum(results) / len(results),
        }
    return {
        "per_suite": per_suite,
        "overall": {
            "n": sum(s["n"] for s in per_suite.values()),
            "accuracy": sum(s["passed"] for s in per_suite.values())
            / sum(s["n"] for s in per_suite.values()),
        },
        "wall_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from mlx_lm import load

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    suites = {
        name: json.loads((EVALS_PROMPTS / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("reasoning", "general", "coding")
    }
    report = run_arm(model, tokenizer, suites)
    report["model"] = args.model
    report["adapter"] = args.adapter
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
