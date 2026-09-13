#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a small, reproducible GLM-5.3 real-task qualification suite.

This is deliberately not a synthetic token-throughput benchmark.  Every task
has either an executable/structured grader or an explicit blind-review flag.
Two artifacts can be compared offline so a speculative path must preserve task
success and deterministic output, rather than merely report high acceptance.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Grade:
    passed: bool
    score: float
    detail: str
    manual_review_required: bool = False


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    prompt: str
    max_tokens: int
    thinking_budget: int
    grade: Callable[[str], Grade]


def _json_object(text: str) -> dict[str, Any] | None:
    """Extract one JSON object without accepting trailing prose as an answer."""

    stripped = text.strip()
    candidates = [stripped]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _grade_exact_json(expected: dict[str, Any]) -> Callable[[str], Grade]:
    def grade(text: str) -> Grade:
        value = _json_object(text)
        if value is None:
            return Grade(False, 0.0, "response is not one JSON object")
        correct = sum(value.get(key) == answer for key, answer in expected.items())
        score = correct / len(expected)
        extras = sorted(set(value) - set(expected))
        passed = score == 1.0 and not extras and len(value) == len(expected)
        detail = f"{correct}/{len(expected)} fields correct"
        if extras:
            detail += f"; unexpected keys: {extras}"
        return Grade(passed, score if not extras else score * 0.9, detail)

    return grade


def _grade_knowledge(text: str) -> Grade:
    value = _json_object(text)
    if value is None:
        return Grade(False, 0.0, "response is not one JSON object")
    aliases = {
        "element_74": {"tungsten", "wolfram"},
        "treaty_1648": {"peace of westphalia", "treaty of westphalia"},
        "largest_moon": {"ganymede"},
        "author_beloved": {"toni morrison"},
        "capital_burkina_faso": {"ouagadougou"},
    }
    correct = sum(
        isinstance(value.get(key), str) and value[key].strip().casefold() in accepted
        for key, accepted in aliases.items()
    )
    extras = sorted(set(value) - set(aliases))
    score = correct / len(aliases)
    passed = correct == len(aliases) and len(value) == len(aliases) and not extras
    return Grade(
        passed, score if not extras else score * 0.9, f"{correct}/5 facts correct"
    )


def _grade_instruction(text: str) -> Grade:
    stripped = text.strip()
    expected = {"alpha": "desserts", "beta": 6, "gamma": [23, 29, 31]}
    value = _json_object(stripped)
    if value is None:
        return Grade(False, 0.0, "response is not one JSON object")
    field_score = sum(value.get(key) == answer for key, answer in expected.items())
    raw_json = stripped.startswith("{") and stripped.endswith("}")
    ordered = list(value) == list(expected)
    exact_keys = set(value) == set(expected) and len(value) == len(expected)
    score = (field_score + int(raw_json) + int(ordered)) / 5
    failures = []
    if not raw_json:
        failures.append("Markdown or prose present")
    if not ordered:
        failures.append("key order")
    if not exact_keys:
        failures.append("key set")
    detail = f"{field_score}/3 values correct"
    if failures:
        detail += "; failed: " + ", ".join(failures)
    return Grade(
        field_score == 3 and raw_json and ordered and exact_keys, score, detail
    )


_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.Try,
    ast.With,
)
_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}


def _python_code(text: str) -> str | None:
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = matches[0].strip() if matches else text.strip()
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def _safe_python(tree: ast.AST) -> str | None:
    if isinstance(tree, ast.Module):
        for statement in tree.body:
            is_docstring = (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
            if not isinstance(statement, ast.FunctionDef) and not is_docstring:
                return f"forbidden top-level syntax: {type(statement).__name__}"
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            return f"forbidden syntax: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"forbidden dunder name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"forbidden dunder attribute: {node.attr}"
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and abs(node.value) > 10_000_000
        ):
            return "forbidden oversized integer literal"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                return f"forbidden call: {node.func.id}"
    return None


def _grade_interval_code(text: str) -> Grade:
    code = _python_code(text)
    if code is None:
        return Grade(False, 0.0, "no parseable Python program")
    tree = ast.parse(code)
    safety_error = _safe_python(tree)
    if safety_error:
        return Grade(False, 0.0, safety_error)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "coalesce_intervals" not in names:
        return Grade(False, 0.0, "missing coalesce_intervals function")

    hidden_tests = """
assert coalesce_intervals([]) == []
assert coalesce_intervals([(1, 2)]) == [(1, 2)]
assert coalesce_intervals([(5, 7), (1, 3), (2, 6)]) == [(1, 7)]
assert coalesce_intervals([(1, 2), (2, 4), (8, 9)]) == [(1, 4), (8, 9)]
assert coalesce_intervals([(-5, -2), (-3, 1), (10, 11)]) == [(-5, 1), (10, 11)]
assert coalesce_intervals([(3, 3), (1, 1), (2, 2)]) == [(1, 1), (2, 2), (3, 3)]
original = [(4, 8), (1, 2)]
snapshot = list(original)
assert coalesce_intervals(original) == [(1, 2), (4, 8)]
assert original == snapshot
"""
    sandbox = shutil.which("sandbox-exec")
    if sandbox is None:
        return Grade(False, 0.0, "coding grader requires macOS sandbox-exec")

    def limit_candidate() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))

    with tempfile.TemporaryDirectory(prefix="rapid-glm53-code-") as directory:
        program = Path(directory) / "candidate.py"
        program.write_text(code + "\n" + hidden_tests, encoding="utf-8")
        profile = "\n".join(
            (
                "(version 1)",
                "(allow default)",
                "(deny network*)",
                "(deny file-write*)",
                f'(allow file-write* (subpath "{directory}"))',
            )
        )
        try:
            completed = subprocess.run(
                [sandbox, "-p", profile, sys.executable, "-I", "-S", str(program)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
                preexec_fn=limit_candidate,
            )
        except subprocess.TimeoutExpired:
            return Grade(False, 0.0, "hidden tests timed out")
    if completed.returncode:
        error = completed.stderr.strip().splitlines()[-1:] or ["unknown failure"]
        return Grade(False, 0.0, f"hidden tests failed: {error[0][:160]}")
    return Grade(True, 1.0, "8/8 hidden tests passed")


def _grade_creative(text: str) -> Grade:
    stripped = text.strip()
    words = re.findall(r"\b[\w'-]+\b", stripped)
    checks = {
        "120-170 words": 120 <= len(words) <= 170,
        "second person": bool(re.search(r"\b(?:you|your|yours)\b", stripped, re.I)),
        "required phrase": "the elevator remembered" in stripped.lower(),
        "no dream cliché": not bool(re.search(r"\bdream\w*\b", stripped, re.I)),
        "exact ending": stripped.endswith("The doors opened onto Tuesday."),
        "no heading": not stripped.startswith("#") and "Title:" not in stripped[:80],
    }
    score = sum(checks.values()) / len(checks)
    failures = [name for name, ok in checks.items() if not ok]
    detail = f"{sum(checks.values())}/{len(checks)} constraints; {len(words)} words"
    if failures:
        detail += "; failed: " + ", ".join(failures)
    return Grade(not failures, score, detail, manual_review_required=True)


def _long_context_prompt() -> str:
    clauses = []
    for number in range(1, 81):
        if number == 47:
            body = (
                "If cumulative verified delay exceeds 45 calendar days, the Supplier "
                "must fund the first 3,200,000 dollars of acceleration costs. The Owner "
                "must issue written notice within 7 business days after verification."
            )
        else:
            body = (
                f"Package {number:02d} is reviewed at the monthly coordination meeting. "
                f"Routine records are retained for {90 + number} days, and ordinary "
                "questions receive a written response within 12 business days."
            )
        clauses.append(f"CLAUSE {number:02d}. {body}")
    contract = "\n\n".join(clauses)
    return f"""Read the contract excerpt below. Return exactly one JSON object with keys
clause, trigger_days, acceleration_cap_usd, and notice_business_days. Use JSON numbers,
not words. Do not include commentary.

{contract}

Question: Which clause assigns acceleration costs after cumulative verified delay, and
what are its trigger, cap, and notice deadline?"""


def build_suite() -> list[Task]:
    return [
        Task(
            "coding.interval_coalescing",
            "coding",
            """Write a complete Python function named coalesce_intervals(intervals).
Each item is a (start, end) integer tuple with start <= end. Return sorted, merged tuples;
touching intervals merge, but gaps of one do not. Do not mutate the input. Use no imports.
            Return only executable Python code, preferably in one Python fence.""",
            1024,
            256,
            _grade_interval_code,
        ),
        Task(
            "knowledge.atomic_facts",
            "knowledge",
            """Answer from general knowledge. Return exactly one JSON object, no prose,
with these keys: element_74, treaty_1648, largest_moon, author_beloved,
and capital_burkina_faso. Values must be the conventional English answer strings.""",
            384,
            128,
            _grade_knowledge,
        ),
        Task(
            "math.inventory_recurrence",
            "math",
            """A warehouse starts Monday with 240 units. Each evening it discards 10%
of the units then present. Tuesday morning it receives 34 units, Wednesday morning 35,
and Thursday morning 36, each before that day's discard. Return exactly
one JSON object with keys monday_close, tuesday_close, wednesday_close, thursday_close.
Use JSON integers and no commentary.""",
            384,
            192,
            _grade_exact_json(
                {
                    "monday_close": 216,
                    "tuesday_close": 225,
                    "wednesday_close": 234,
                    "thursday_close": 243,
                }
            ),
        ),
        Task(
            "instruction.exact_transform",
            "instruction_following",
            """Return exactly one JSON object with exactly three keys in this order:
alpha, beta, gamma. alpha is the reverse of 'stressed'. beta is the number of vowels in
'coordination'. gamma is an array containing the first three prime numbers greater than
20. Do not use a Markdown fence or add commentary.""",
            256,
            96,
            _grade_instruction,
        ),
        Task(
            "creative.constrained_flash",
            "creative_writing",
            """Write a 120-170 word second-person speculative-fiction scene. Include
the exact phrase 'the elevator remembered'. Never use the word dream or its variants.
Do not add a title or heading. End with this exact sentence: The doors opened onto Tuesday.""",
            1024,
            192,
            _grade_creative,
        ),
        Task(
            "long_context.contract_clause",
            "long_context",
            _long_context_prompt(),
            384,
            128,
            _grade_exact_json(
                {
                    "clause": 47,
                    "trigger_days": 45,
                    "acceleration_cap_usd": 3_200_000,
                    "notice_business_days": 7,
                }
            ),
        ),
    ]


def _post_task(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    task: Task,
    use_thinking_budget: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task.prompt}],
        "temperature": 0,
        "max_tokens": task.max_tokens,
        "stream": False,
        "enable_thinking": True,
    }
    if use_thinking_budget:
        payload["thinking_budget"] = task.thinking_budget
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    server_metrics: dict[str, Any] = {}
    try:
        metrics_url = httpx.URL(base_url).copy_with(path="/metrics", query=None)
        metrics_response = client.get(metrics_url)
        metrics_response.raise_for_status()
        latest = metrics_response.json().get("latest") or {}
        server_metrics = {
            key: latest.get(key)
            for key in (
                "prompt_eval_time_s",
                "prefill_tok_s",
                "ttft_s",
                "decode_elapsed_s",
                "request_elapsed_s",
                "request_tok_s",
                "decode_tok_s",
                "peak_memory_gb",
            )
        }
    except (httpx.HTTPError, json.JSONDecodeError, AttributeError):
        pass
    choice = data["choices"][0]
    message = choice["message"]
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    grade = task.grade(text)
    return {
        "task_id": task.task_id,
        "category": task.category,
        "max_tokens": task.max_tokens,
        "thinking_budget": task.thinking_budget if use_thinking_budget else None,
        "elapsed_s": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "client_completion_tps": completion_tokens / elapsed if elapsed else 0.0,
        "finish_reason": choice.get("finish_reason"),
        "output": text,
        "reasoning": reasoning,
        "server_metrics": server_metrics,
        "grade": asdict(grade),
    }


def run_suite(args: argparse.Namespace) -> int:
    selected = set(args.task or [])
    suite = [task for task in build_suite() if not selected or task.task_id in selected]
    unknown = selected - {task.task_id for task in suite}
    if unknown:
        raise SystemExit(f"unknown task ids: {sorted(unknown)}")
    results = []
    with httpx.Client(timeout=args.timeout) as client:
        if not args.no_warmup:
            warmup = Task(
                "warmup",
                "warmup",
                "Return exactly the word READY.",
                128,
                64,
                lambda text: Grade(text.strip() == "READY", 1.0, "warmup"),
            )
            _post_task(
                client,
                base_url=args.base_url,
                model=args.model,
                task=warmup,
                use_thinking_budget=not args.omit_thinking_budget,
            )
        for task in suite:
            result = _post_task(
                client,
                base_url=args.base_url,
                model=args.model,
                task=task,
                use_thinking_budget=not args.omit_thinking_budget,
            )
            results.append(result)
            grade = result["grade"]
            mark = "PASS" if grade["passed"] else "FAIL"
            print(
                f"{mark:4} {task.task_id:36} score={grade['score']:.3f} "
                f"{result['client_completion_tps']:.2f} tok/s "
                f"({result['completion_tokens']} tokens, {result['elapsed_s']:.2f}s)"
            )
    artifact = {
        "schema_version": 1,
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "thinking": True,
        "thinking_budget_mode": ("off" if args.omit_thinking_budget else "per-task"),
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    passed = sum(result["grade"]["passed"] for result in results)
    print(f"wrote {args.output}; {passed}/{len(results)} tasks passed")
    return 0 if passed == len(results) else 2


def compare_artifacts(paths: list[str]) -> int:
    left, right = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if left.get("thinking_budget_mode") != right.get("thinking_budget_mode"):
        raise SystemExit("artifacts use different thinking-budget policies")
    left_rows = {row["task_id"]: row for row in left["results"]}
    right_rows = {row["task_id"]: row for row in right["results"]}
    if set(left_rows) != set(right_rows):
        raise SystemExit("artifacts contain different task ids")
    rows = []
    for task_id in sorted(left_rows):
        baseline = left_rows[task_id]
        candidate = right_rows[task_id]
        same_request = all(
            baseline.get(key) == candidate.get(key)
            for key in ("max_tokens", "thinking_budget", "prompt_tokens")
        )
        exact_output = baseline["output"] == candidate["output"]
        exact_reasoning = baseline.get("reasoning", "") == candidate.get(
            "reasoning", ""
        )
        ratio = (
            candidate["client_completion_tps"] / baseline["client_completion_tps"]
            if baseline["client_completion_tps"]
            else 0.0
        )
        score_delta = candidate["grade"]["score"] - baseline["grade"]["score"]
        rows.append(
            {
                "task_id": task_id,
                "category": baseline["category"],
                "same_request": same_request,
                "exact_output": exact_output,
                "exact_reasoning": exact_reasoning,
                "baseline_passed": baseline["grade"]["passed"],
                "candidate_passed": candidate["grade"]["passed"],
                "score_delta": score_delta,
                "throughput_ratio": ratio,
            }
        )
        print(
            f"{'SAME' if same_request and exact_output and exact_reasoning else 'DIFF':4} "
            f"{task_id:36} "
            f"quality_delta={score_delta:+.3f} speed={ratio:.3f}x"
        )
    ratios = [row["throughput_ratio"] for row in rows]
    gate_passed = all(
        row["same_request"]
        and row["exact_output"]
        and row["exact_reasoning"]
        and row["baseline_passed"]
        and row["candidate_passed"]
        and row["score_delta"] >= 0
        and row["throughput_ratio"] >= 0.95
        for row in rows
    )
    summary = {
        "schema_version": 1,
        "baseline": paths[0],
        "candidate": paths[1],
        "gate_passed": gate_passed,
        "exact_outputs": sum(row["exact_output"] for row in rows),
        "exact_reasoning": sum(row["exact_reasoning"] for row in rows),
        "task_count": len(rows),
        "median_throughput_ratio": statistics.median(ratios),
        "minimum_throughput_ratio": min(ratios),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    return 0 if gate_passed else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8465/v1")
    parser.add_argument("--model", default="glm5.3-flash-4bit")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--output", default="glm53-real-tasks.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--task", action="append", help="run only this task id")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--omit-thinking-budget",
        action="store_true",
        help="omit thinking_budget for runtimes that cannot combine it with MTP",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_JSON", "CANDIDATE_JSON"),
        help="compare two existing artifacts instead of calling a server",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_artifacts(args.compare)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
