#!/usr/bin/env python3
"""Run and score the private Flash-Next launch correctness battery."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

TOOL_SCHEMAS = {
    "get_weather": {
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    "calculator": {
        "description": "Evaluate a basic arithmetic operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
            "additionalProperties": False,
        },
    },
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if case.get("route") not in {"openai", "anthropic"}:
            raise ValueError(f"{path}:{line_no}: unsupported route")
        unknown_tools = set(case.get("tools", ())) - TOOL_SCHEMAS.keys()
        if unknown_tools:
            raise ValueError(f"{path}:{line_no}: unknown tools {sorted(unknown_tools)}")
        for scorer in case.get("scorers", ()):
            if scorer.get("type") == "regex":
                re.compile(scorer["pattern"])
        cases.append(case)
    ids = [case["id"] for case in cases]
    if len(ids) < 40:
        raise ValueError(f"expected at least 40 cases, found {len(ids)}")
    duplicates = [name for name, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
    return cases


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids", ())
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, Sequence)
        and encoded
        and isinstance(encoded[0], Sequence)
        and not isinstance(encoded[0], (str, bytes))
    ):
        encoded = encoded[0]
    return len(encoded)


def expand_needle(case: dict[str, Any], tokenizer: Any) -> list[dict[str, str]]:
    spec = case["expand"]
    target = int(spec["target_tokens"])
    filler = (
        "Stable context sentence for deterministic long-context recall. "
        "Every sentence is irrelevant except the explicitly labeled fact. "
    )

    def messages(chars: int) -> list[dict[str, str]]:
        left = (filler * (chars // len(filler) + 1))[: chars // 2]
        right = (filler * (chars // len(filler) + 1))[: chars - len(left)]
        return [
            {
                "role": "system",
                "content": "Read the context and answer the final question exactly.",
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{left}\nIMPORTANT FACT: {spec['needle']}\n"
                    f"{right}\nQUESTION: {spec['question']}"
                ),
            },
        ]

    low, high = 0, target * 12
    while low < high:
        mid = (low + high + 1) // 2
        if token_count(tokenizer, messages(mid)) <= target:
            low = mid
        else:
            high = mid - 1
    rendered = messages(low)
    actual = token_count(tokenizer, rendered)
    if actual < target - 16 or actual > target:
        raise RuntimeError(f"{case['id']}: target={target}, rendered={actual}")
    case["rendered_prompt_tokens"] = actual
    return rendered


def generic_tools(names: list[str], *, anthropic: bool) -> list[dict[str, Any]]:
    tools = []
    for name in names:
        schema = TOOL_SCHEMAS[name]
        if anthropic:
            tools.append({"name": name, **schema})
        else:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": schema["description"],
                        "parameters": schema["input_schema"],
                    },
                }
            )
    return tools


def build_payload(
    case: dict[str, Any], model: str, tokenizer: Any | None
) -> tuple[str, dict[str, Any]]:
    route = case["route"]
    messages = case.get("messages")
    if "expand" in case:
        if tokenizer is None:
            raise ValueError("--tokenizer-path is required for long-context cases")
        messages = expand_needle(case, tokenizer)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(case.get("max_tokens", 256)),
        "temperature": 0.0,
    }
    thinking = case.get("enable_thinking")
    if thinking is None:
        thinking = case["category"] in {"reasoning_math", "zh_reasoning"}
    payload["enable_thinking"] = thinking

    if route == "anthropic":
        endpoint = "/v1/messages"
        payload["system"] = case.get("system", "")
        if case.get("stop"):
            payload["stop_sequences"] = case["stop"]
        if case.get("tools"):
            payload["tools"] = generic_tools(case["tools"], anthropic=True)
            choice = case.get("tool_choice", "auto")
            payload["tool_choice"] = (
                {"type": "auto"}
                if choice == "auto"
                else {"type": "tool", "name": choice}
            )
    else:
        endpoint = "/v1/chat/completions"
        if case.get("stop"):
            payload["stop"] = case["stop"]
        if case.get("tools"):
            payload["tools"] = generic_tools(case["tools"], anthropic=False)
            choice = case.get("tool_choice", "auto")
            payload["tool_choice"] = (
                "auto"
                if choice == "auto"
                else {"type": "function", "function": {"name": choice}}
            )
        if case.get("response_schema"):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {**case["response_schema"], "strict": True},
            }
    return endpoint, payload


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def normalize_response(route: str, response: dict[str, Any]) -> dict[str, Any]:
    if route == "anthropic":
        blocks = response.get("content") or []
        text = "".join(
            str(block.get("text") or "")
            for block in blocks
            if block.get("type") == "text"
        )
        tool_names = [
            str(block.get("name"))
            for block in blocks
            if block.get("type") == "tool_use"
        ]
    else:
        message = response["choices"][0]["message"]
        text = str(message.get("content") or "")
        tool_names = [
            str(call.get("function", {}).get("name"))
            for call in (message.get("tool_calls") or [])
        ]
    return {"text": text.strip(), "tool_names": tool_names}


def json_path(value: Any, path: str) -> Any:
    for segment in path.split("."):
        value = value[segment]
    return value


def run_project(text: str, timeout: float) -> tuple[bool, str]:
    try:
        payload = json.loads(text)
        files = payload["files"]
        if set(files) != {"todo.py", "test_todo.py"}:
            return False, f"unexpected files: {sorted(files)}"
        if not all(isinstance(content, str) for content in files.values()):
            return False, "file content must be strings"
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"invalid project JSON: {exc}"

    with tempfile.TemporaryDirectory(prefix="flash-next-todo-") as tmp:
        root = Path(tmp)
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            "TMPDIR": tmp,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "test_todo.py"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        detail = (result.stdout + "\n" + result.stderr).strip()[-4000:]
        return result.returncode == 0, detail


def score_one(
    scorer: dict[str, Any], normalized: dict[str, Any], *, allow_project_exec: bool
) -> tuple[bool, str]:
    text = normalized["text"]
    kind = scorer["type"]
    if kind == "exact":
        passed = text == str(scorer["value"])
        return passed, f"expected exact {scorer['value']!r}"
    if kind == "contains":
        passed = str(scorer["value"]).casefold() in text.casefold()
        return passed, f"expected contains {scorer['value']!r}"
    if kind == "not_contains":
        passed = str(scorer["value"]).casefold() not in text.casefold()
        return passed, f"expected absence of {scorer['value']!r}"
    if kind == "regex":
        passed = re.search(scorer["pattern"], text) is not None
        return passed, f"expected regex {scorer['pattern']!r}"
    if kind == "json_valid":
        try:
            json.loads(text)
            return True, "valid JSON"
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
    if kind == "json_path_exact":
        try:
            actual = json_path(json.loads(text), scorer["path"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            return False, f"JSON path error: {exc}"
        passed = actual == scorer["value"]
        return passed, f"{scorer['path']}={actual!r}, expected {scorer['value']!r}"
    if kind == "tool_name":
        passed = scorer["value"] in normalized["tool_names"]
        return passed, f"tool names={normalized['tool_names']!r}"
    if kind == "tool_name_absent":
        passed = not normalized["tool_names"]
        return passed, f"tool names={normalized['tool_names']!r}"
    if kind == "project_files":
        try:
            actual = sorted(json.loads(text)["files"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            return False, f"project files error: {exc}"
        expected = sorted(scorer["files"])
        return actual == expected, f"files={actual!r}, expected={expected!r}"
    if kind == "project_unittest":
        if not allow_project_exec:
            return False, "project execution requires --allow-project-exec"
        return run_project(text, timeout=60.0)
    raise ValueError(f"unknown scorer: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "results.jsonl")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--allow-project-exec", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.prompts)
    categories = Counter(case["category"] for case in cases)
    print(f"validated {len(cases)} cases: {dict(sorted(categories.items()))}")
    if args.dry_run:
        return 0
    if any(case["category"] == "small_project" for case in cases):
        if not args.allow_project_exec:
            parser.error(
                "the battery includes generated-code execution; pass --allow-project-exec"
            )

    tokenizer = load_tokenizer(args.tokenizer_path) if args.tokenizer_path else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed_cases = 0
    with args.output.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            try:
                endpoint, payload = build_payload(case, args.model, tokenizer)
                response = post_json(
                    args.base_url.rstrip("/") + endpoint,
                    payload,
                    args.api_key,
                    args.timeout,
                )
                normalized = normalize_response(case["route"], response)
                scores = []
                for scorer in case["scorers"]:
                    passed, detail = score_one(
                        scorer,
                        normalized,
                        allow_project_exec=args.allow_project_exec,
                    )
                    scores.append(
                        {"type": scorer["type"], "passed": passed, "detail": detail}
                    )
                passed = all(score["passed"] for score in scores)
                error = None
            except Exception as exc:
                normalized = {"text": "", "tool_names": []}
                scores = []
                passed = False
                error = f"{type(exc).__name__}: {exc}"
            elapsed = round(time.perf_counter() - started, 3)
            row = {
                "id": case["id"],
                "category": case["category"],
                "route": case["route"],
                "passed": passed,
                "elapsed_seconds": elapsed,
                "rendered_prompt_tokens": case.get("rendered_prompt_tokens"),
                "text": normalized["text"],
                "tool_names": normalized["tool_names"],
                "scores": scores,
                "error": error,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            passed_cases += int(passed)
            print(
                f"[{index:02d}/{len(cases)}] {'PASS' if passed else 'FAIL'} {case['id']} ({elapsed}s)"
            )

    print(f"SUMMARY {passed_cases}/{len(cases)} passed; results={args.output}")
    return 0 if passed_cases == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
