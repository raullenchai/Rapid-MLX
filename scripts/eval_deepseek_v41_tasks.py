#!/usr/bin/env python3
"""CPU-only OpenAI HTTP smoke, timing, and frozen task evaluation.

Never loads a model or starts a server. Run ``prefetch`` first; subsequent runs
use the self-contained manifest without fetching datasets. See the phase-6
runbook for serving limitations, metrics, sandbox, and comparison caveats.

Examples (from repository root, after the owner starts a compatible server)::

    .venv/bin/python scripts/eval_deepseek_v41_tasks.py prefetch
    .venv/bin/python scripts/eval_deepseek_v41_tasks.py smoke --model dsv41-t256
    .venv/bin/python scripts/eval_deepseek_v41_tasks.py speed --runs 3
    .venv/bin/python scripts/eval_deepseek_v41_tasks.py bench --build-label t256-ar

Reuse the same --manifest across builds. Results include requests, raw replies,
seed/mode/caps and manifest/harness hashes. Generation caps total 139,968 tokens
for the default 814 items (2.59 hours of decode at 15 tok/s, plus prefill).
Thinking mode uses the same caps unless --max-tokens overrides them. Truncation
counts as a wrong answer. Official few-shot base scores are not instruct-chat
baselines. BFCL uses strict JSON/AST-value matching, not its full official scorer.

HumanEval requires macOS sandbox-exec, denies network and host writes, and has
CPU/output/wall limits. There is no unsandboxed fallback. Exit 0 means a complete
benchmark (wrong answers are normal), 1 means failed smoke/speed checks, 2 means
benchmark infrastructure failure. Partial JSONL is flushed after every item.

Serving contract: the product alias is greedy, text-only and limited to 8192
input tokens. Use --omit-seed for that lane. Full smoke includes unsupported
tools and a 16K needle; failures are capability results, not permission to relax
server limits. Text-only bench tasks can be selected explicitly. Local build
qualification uses RAPID_MLX_V41_MODEL_DIR as documented in the serving runbook.
This client never starts a runtime or bypasses server request validation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

VERSION = 1
CAPS = {"gsm8k": 512, "mmlu_pro": 512, "humaneval": 512, "tools": 192}
DEFAULT_ROOT = (
    Path(
        os.environ.get(
            "DSV41_SCRATCH_ROOT",
            str(Path(tempfile.gettempdir()) / "rapid-mlx-dsv41-1017"),
        )
    )
    / "phase6"
)
DATASETS = {
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet"),
    "humaneval": (
        "openai/openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet",
    ),
}
BFCL = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def select(rows, count, seed):
    """Sample stable source indices, retaining source ordering."""
    return [
        rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), count))
    ]


def stratified(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    names = sorted(groups)
    selected = []
    for i, name in enumerate(names):
        selected.extend(
            select(
                groups[name], count // len(names) + (i < count % len(names)), seed + i
            )
        )
    return selected


def json_lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def prefetch(path, seed):
    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, hf_hub_download

    if path.exists():
        raise ValueError(f"Refusing to replace frozen manifest: {path}")
    api = HfApi()
    sources, items = {}, []
    for task, (repo, filename) in DATASETS.items():
        revision = api.dataset_info(repo).sha
        local = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        rows = parquet.read_table(local).to_pylist()
        for i, row in enumerate(rows):
            row["_source_index"] = i
        sources[task] = {
            "repo": repo,
            "revision": revision,
            "files": [filename],
            "split": "test",
        }
        chosen = (
            select(rows, 250, seed)
            if task == "gsm8k"
            else stratified(rows, 300, seed)
            if task == "mmlu_pro"
            else rows
        )
        for row in chosen:
            ident = str(
                row.get("task_id", row.get("question_id", row["_source_index"]))
            )
            items.append({"task": task, "id": ident, "data": row})
    revision = api.dataset_info(BFCL).sha
    files = []
    # Equal category allocation: 25 each = 100 single/multiple/parallel cases.
    for i, category in enumerate(
        ("simple", "multiple", "parallel", "parallel_multiple")
    ):
        filename = f"BFCL_v3_{category}.json"
        answerfile = f"possible_answer/{filename}"
        files.extend([filename, answerfile])
        rows = json_lines(
            hf_hub_download(BFCL, filename, repo_type="dataset", revision=revision)
        )
        answers = {
            r["id"]: r["ground_truth"]
            for r in json_lines(
                hf_hub_download(
                    BFCL, answerfile, repo_type="dataset", revision=revision
                )
            )
        }
        # These files are the Python AST categories; exclude any foreign-language cases.
        rows = [r for r in rows if r.get("language", "Python").lower() == "python"]
        for row in select(rows, 25, seed + i):
            row["ground_truth"] = answers[row["id"]]
            row["category"] = category
            items.append({"task": "tools", "id": row["id"], "data": row})
    sources["tools"] = {"repo": BFCL, "revision": revision, "files": files}
    repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    revision = api.model_info(repo).sha
    tokenizer = hf_hub_download(repo, "tokenizer.json", revision=revision)
    manifest = {
        "version": VERSION,
        "seed": seed,
        "sources": sources,
        "tokenizer": {
            "repo": repo,
            "revision": revision,
            "file": tokenizer,
            "sha256": hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest(),
        },
        "counts": dict(Counter(x["task"] for x in items)),
        "items": items,
        "protocol": {
            "caps": CAPS,
            "temperature": 0,
            "default_mode": "non-thinking",
            "prompting": "zero-shot chat, concise answers; not official evaluation",
        },
    }
    manifest["sha256"] = digest(manifest)
    dump(path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": manifest["sha256"],
                "counts": manifest["counts"],
            }
        )
    )


class Client:
    def __init__(
        self,
        url,
        model,
        mode="non-thinking",
        seed=1017,
        timeout=900,
        api_key=None,
        send_seed=True,
    ):
        self.url, self.model = url.rstrip("/"), model
        self.mode, self.seed, self.timeout = mode, seed, timeout
        self.api_key, self.send_seed = api_key, send_seed

    def chat(self, messages, max_tokens=128, stream=False, **extra):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": stream,
            "enable_thinking": self.mode == "thinking",
            "chat_template_kwargs": {"enable_thinking": self.mode == "thinking"},
        }
        if self.send_seed:
            payload["seed"] = self.seed
        payload.update(extra)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.url + "/chat/completions", json.dumps(payload).encode(), headers
        )
        start = time.monotonic()
        result = {
            "request": payload,
            "raw": [],
            "message": {"role": "assistant", "content": ""},
            "usage": {},
            "finish_reason": None,
            "ttft": None,
        }
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if not stream:
                    raw = response.read().decode()
                    result["raw"].append(raw)
                    obj = json.loads(raw)
                    choice = obj["choices"][0]
                    if not isinstance(choice.get("message"), dict):
                        raise ValueError("missing assistant message")
                    result.update(
                        message=choice["message"],
                        usage=obj.get("usage") or {},
                        finish_reason=choice.get("finish_reason"),
                    )
                else:
                    calls = {}
                    done = False
                    for line in response:
                        line = line.decode().rstrip("\r\n")
                        result["raw"].append(line)
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        obj = json.loads(data)
                        if obj.get("usage"):
                            result["usage"] = obj["usage"]
                        for choice in obj.get("choices", []):
                            delta = choice.get("delta", {})
                            if (
                                delta.get("content")
                                or delta.get("reasoning_content")
                                or delta.get("tool_calls")
                            ):
                                if result["ttft"] is None:
                                    result["ttft"] = time.monotonic() - start
                            for key in ("content", "reasoning_content"):
                                if delta.get(key):
                                    result["message"][key] = (
                                        result["message"].get(key, "") + delta[key]
                                    )
                            for call in delta.get("tool_calls", []):
                                target = calls.setdefault(
                                    call["index"],
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if call.get("id"):
                                    target["id"] += call["id"]
                                for key in ("name", "arguments"):
                                    target["function"][key] += call.get(
                                        "function", {}
                                    ).get(key, "")
                            if choice.get("finish_reason"):
                                result["finish_reason"] = choice["finish_reason"]
                    if calls:
                        result["message"]["tool_calls"] = [
                            calls[i] for i in sorted(calls)
                        ]
                    if not done or not result["finish_reason"]:
                        raise ValueError(
                            "incomplete SSE stream (missing DONE or finish_reason)"
                        )
        except HTTPError as error:
            result["error"] = (
                f"HTTP {error.code}: {error.read().decode(errors='replace')}"
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed"] = time.monotonic() - start
        return result


def text(result):
    return result.get("message", {}).get("content") or ""


def valid(result):
    return not result.get("error") and result.get("finish_reason") in (
        "stop",
        "tool_calls",
    )


def numeric(text):
    # Prefer explicit final marker. Otherwise use the last number, as GSM8K EM.
    text = text.rsplit("####", 1)[-1].replace(",", "")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    try:
        return Decimal(matches[-1]) if matches else None
    except InvalidOperation:
        return None


def letter(text):
    explicit = re.findall(r"\banswer\s*(?:is|:)\s*\(?([A-J])\)?\b", text, re.I)
    standalone = re.findall(r"^\s*\(?([A-J])\)?[.)]?\s*$", text, re.M)
    return (explicit or standalone)[-1].upper() if explicit or standalone else None


def schema(value):
    """BFCL Python types -> JSON Schema, retaining descriptions/required/defaults."""
    if isinstance(value, list):
        return [schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: schema(v) for k, v in value.items()}
    aliases = {
        "dict": "object",
        "float": "number",
        "int": "integer",
        "str": "string",
        "bool": "boolean",
        "list": "array",
        "tuple": "array",
    }
    if result.get("type") == "any":
        result.pop("type")
    if isinstance(result.get("type"), str):
        result["type"] = aliases.get(result["type"], result["type"])
    return result


def parse_calls(message):
    calls = message.get("tool_calls", [])
    if not calls:
        raise ValueError("missing tool_calls")
    parsed = []
    for call in calls:
        function = call["function"]
        arguments = json.loads(function["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        parsed.append((function["name"], arguments))
    return parsed


def argument_match(actual, expected):
    """Strict recursive value match; BFCL outer lists enumerate allowed values."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(argument_match(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(argument_match(a, e) for a, e in zip(actual, expected))
        )
    return (
        type(actual) is type(expected)
        and actual == expected
        or (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
            and actual == expected
        )
    )


def match_call(actual, expected):
    name, arguments = actual
    if len(expected) != 1 or name not in expected:
        return False
    allowed = expected[name]
    if set(arguments) - set(allowed):
        return False
    for key, choices in allowed.items():
        # Empty string in BFCL's oracle permits omission of an optional argument.
        if key not in arguments:
            if "" not in choices:
                return False
        elif not any(argument_match(arguments[key], value) for value in choices):
            return False
    return True


def tool_match(message, oracle):
    actual = parse_calls(message)
    remaining = list(oracle)

    # Backtracking avoids greedy mismatches for repeated parallel function names.
    def match(index, expected):
        if index == len(actual):
            return not expected
        return any(
            match(index + 1, expected[:i] + expected[i + 1 :])
            for i, candidate in enumerate(expected)
            if match_call(actual[index], candidate)
        )

    return len(actual) == len(remaining) and match(0, remaining)


def python_code(output, prompt, entry):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.S)
    code = blocks[0] if blocks else output
    # The chat prompt requests a full function. Also accept classic body completions.
    if not re.search(rf"^def\s+{re.escape(entry)}\s*\(", code, re.M):
        code = prompt + code
    ast.parse(code)
    return code


def execute_code(code, tests, entry, timeout, scratch):
    """macOS Seatbelt subprocess; no network, host writes, or child processes.

    Fail closed when unavailable. Uses the repo's existing sandbox-exec pattern,
    adds file-read restrictions and bounds output files as well as CPU time.
    """
    sandbox = shutil.which("sandbox-exec")
    if sandbox is None:
        return {
            "passed": False,
            "error": "sandbox-exec unavailable; no unsafe fallback",
        }
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="humaneval-") as directory:
        directory = Path(directory).resolve()
        program = directory / "runner.py"
        nonce = secrets.token_bytes(32)
        # Keep candidate globals separate from the test module.
        program.write_text(
            "import os, resource\n"
            f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
            "resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))\n"
            "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
            f'ns = {{}}\nexec(compile({code!r}, "candidate", "exec"), ns)\n'
            f'exec({tests!r})\ncheck(ns[{entry!r}])\nos.write(int(os.environ["RESULT_FD"]), {nonce!r})\n'
        )
        # Read-only runtime paths only; workspace/model/cache and home are hidden.
        roots = {
            str(directory),
            str(Path(sys.executable).resolve().parent.parent),
            str(Path(sys.base_prefix).resolve()),
            "/System",
            "/usr",
            "/Library",
            "/dev",
            "/private/var/db/dyld",
        }
        profile = (
            "(version 1)(deny default)(allow process-exec)(allow signal (target self))"
            "(allow sysctl-read)(allow mach-lookup)"
            # dyld/Python startup stat the root directory entry itself.
            '(allow file-read* (literal "/") '
            + " ".join(f"(subpath {json.dumps(p)})" for p in sorted(roots))
            # Interpreter startup realpath()s its prefix; stat each ancestor only.
            + ")(allow file-read-metadata "
            + " ".join(
                f"(literal {json.dumps(str(parent))})"
                for parent in sorted(
                    {str(a) for p in roots for a in Path(p).parents} - {"/"}
                )
            )
            + ")(allow file-write-data "
            + " ".join(
                f"(literal {json.dumps(str(directory / name))})"
                for name in ("stdout", "stderr", "completion")
            )
            + ")"
        )
        # Permit only these existing output files. RLIMIT_FSIZE bounds each
        # file; pipes would allow unbounded parent memory growth.
        with (
            (directory / "stdout").open("wb") as out,
            (directory / "stderr").open("wb") as err,
            (directory / "completion").open("wb") as completion,
        ):
            process = subprocess.Popen(
                [
                    sandbox,
                    "-p",
                    profile,
                    str(Path(sys.executable).resolve()),
                    "-I",
                    "-S",
                    str(program),
                ],
                cwd=directory,
                stdout=out,
                stderr=err,
                env={
                    "PATH": "/usr/bin:/bin",
                    "TMPDIR": str(directory),
                    "RESULT_FD": str(completion.fileno()),
                },
                pass_fds=(completion.fileno(),),
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout + 2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {"passed": False, "detail": "execution timeout"}
        stdout = (directory / "stdout").read_bytes()
        stderr = (directory / "stderr").read_bytes()
        if b"sandbox-exec:" in stderr:
            return {"passed": False, "error": stderr.decode(errors="replace")[-4000:]}
        return {
            "passed": process.returncode == 0
            and (directory / "completion").read_bytes() == nonce,
            "returncode": process.returncode,
            "stdout": stdout.decode(errors="replace")[-4000:],
            "stderr": stderr.decode(errors="replace")[-4000:],
        }


def prompt_item(item):
    row, task = item["data"], item["task"]
    if task == "gsm8k":
        return [
            {
                "role": "user",
                "content": row["question"]
                + "\nGive only the final numeric answer, prefixed with ####.",
            }
        ], {}
    if task == "mmlu_pro":
        options = "\n".join(f"{chr(65 + i)}. {v}" for i, v in enumerate(row["options"]))
        return [
            {
                "role": "user",
                "content": f"{row['question']}\n{options}\nReturn only the correct answer letter (A-J).",
            }
        ], {}
    if task == "humaneval":
        return [
            {
                "role": "user",
                "content": "Complete this Python function. Return the full function and any required imports in one Python code block.\n"
                + row["prompt"],
            }
        ], {}
    return row["question"][0], {
        "tools": [{"type": "function", "function": schema(f)} for f in row["function"]],
        "tool_choice": "auto",
    }


def grade(item, result, args):
    if not valid(result):
        return {
            "passed": False,
            **(
                {"error": result["error"]}
                if result.get("error")
                else {"detail": "truncated or unfinished response"}
            ),
        }
    row, task, output = item["data"], item["task"], text(result)
    if task == "gsm8k":
        prediction, expected = numeric(output), numeric(row["answer"])
        return {
            "passed": prediction is not None and prediction == expected,
            "prediction": str(prediction),
            "expected": str(expected),
        }
    if task == "mmlu_pro":
        prediction = letter(output)
        return {
            "passed": prediction == row["answer"],
            "prediction": prediction,
            "expected": row["answer"],
        }
    if task == "tools":
        try:
            return {"passed": tool_match(result["message"], row["ground_truth"])}
        except (ValueError, TypeError, KeyError) as error:
            return {"passed": False, "detail": f"invalid function call: {error}"}
    try:
        code = python_code(output, row["prompt"], row["entry_point"])
    except (SyntaxError, ValueError) as error:
        return {"passed": False, "detail": f"invalid Python: {error}"}
    return execute_code(
        code,
        row["test"],
        row["entry_point"],
        args.code_timeout,
        args.scratch,
    )


def long_prompt(tokenizer, target, needle=False, salt="phase6"):
    prefix = f"Document {salt}. Read the records below.\n"
    suffix = (
        "\nWhat is the secret access code? Return only the code."
        if needle
        else "\nWrite a long numbered list of animals, at least 400 entries."
    )
    marker = "\nThe secret access code is SILVER-OTTER-7319.\n" if needle else "\n"

    def build(n):
        lines = [
            f"Record {i}: the ordinary parcel remains in storage.\n" for i in range(n)
        ]
        lines.insert(n // 2, marker)
        return prefix + "".join(lines) + suffix

    low, high = 0, target
    while low + 1 < high:
        middle = (low + high) // 2
        if len(tokenizer.encode(build(middle)).ids) <= target:
            low = middle
        else:
            high = middle
    prompt = build(low)
    return prompt, len(tokenizer.encode(prompt).ids)


SMOKE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    for name, description in [
        ("get_temperature", "Get current Celsius temperature."),
        ("get_population", "Get current population."),
    ]
]


def smoke(client, tokenizer):
    checks = []
    for stream in (False, True):
        for language, prompt, expected in [
            ("en", "What is 2+2? Reply only with 4.", "4"),
            ("zh", "请仅回答“中国首都是北京”。", "北京"),
        ]:
            result = client.chat(
                [{"role": "user", "content": prompt}], max_tokens=64, stream=stream
            )
            checks.append(
                {
                    "id": f"{language}_{stream}",
                    "passed": valid(result) and expected in text(result),
                    "response": result,
                }
            )
        messages = [
            {
                "role": "user",
                "content": "Use both tools to get the temperature and population of Paris. Then report both numbers.",
            }
        ]
        first = client.chat(messages, max_tokens=256, stream=stream, tools=SMOKE_TOOLS)
        check = {"id": f"tools_{stream}", "passed": False, "responses": [first]}
        try:
            calls = parse_calls(first["message"])
            assert valid(first) and sorted(name for name, _ in calls) == [
                "get_population",
                "get_temperature",
            ]
            assert all(arguments == {"city": "Paris"} for _, arguments in calls)
            raw_calls = first["message"]["tool_calls"]
            ids = [c["id"] for c in raw_calls]
            assert all(ids) and len(set(ids)) == 2
            messages.append(first["message"])
            for call in raw_calls:
                value = (
                    {"temperature_c": 17}
                    if call["function"]["name"] == "get_temperature"
                    else {"population": 2100000}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(value),
                    }
                )
            final = client.chat(
                messages, max_tokens=128, stream=stream, tools=SMOKE_TOOLS
            )
            check["responses"].append(final)
            check["passed"] = (
                valid(final)
                and not final["message"].get("tool_calls")
                and "17" in text(final)
                and "2100000" in text(final).replace(",", "").replace(" ", "")
            )
        except (ValueError, KeyError, AssertionError, TypeError) as error:
            check["error"] = f"{type(error).__name__}: {error}"
        checks.append(check)
        prompt, count = long_prompt(tokenizer, 16384, needle=True)
        result = client.chat(
            [{"role": "user", "content": prompt}], max_tokens=64, stream=stream
        )
        actual = result["usage"].get("prompt_tokens", 0)
        checks.append(
            {
                "id": f"needle_{stream}",
                "body_tokens": count,
                "server_prompt_tokens": actual,
                "passed": valid(result)
                and "SILVER-OTTER-7319" in text(result)
                and 15500 <= actual <= 17500,
                "response": result,
            }
        )
    return checks


def speed(client, tokenizer, runs):
    results = []
    for size in (1024, 8192):
        for run in range(runs):
            # Unique first record reduces prefix reuse; server cache state still matters.
            prompt, count = long_prompt(
                tokenizer, size, salt=f"speed-{size}-{run}-{time.time_ns()}"
            )
            result = client.chat(
                [{"role": "user", "content": prompt}], max_tokens=256, stream=True
            )
            usage, ttft = result["usage"], result["ttft"]
            tokens = usage.get("completion_tokens", 0)
            elapsed = result["elapsed"]
            ok = (
                not result.get("error")
                and ttft is not None
                and tokens > 1
                and result["finish_reason"] in ("stop", "length")
                and abs(usage.get("prompt_tokens", 0) - size) < 128
            )
            results.append(
                {
                    "id": f"{size}_{run}",
                    "passed": ok,
                    "body_tokens": count,
                    "decode_tok_s": (tokens - 1) / (elapsed - ttft)
                    if ok and elapsed > ttft
                    else None,
                    "prefill_tok_s_client_estimate": usage.get("prompt_tokens", 0)
                    / ttft
                    if ok and ttft
                    else None,
                    "response": result,
                }
            )
    return results


def load_manifest(path):
    manifest = json.loads(path.read_text())
    expected = manifest.pop("sha256")
    if digest(manifest) != expected:
        raise ValueError("manifest digest mismatch")
    manifest["sha256"] = expected
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prefetch", "smoke", "speed", "bench"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="dsv41-t256")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.json")
    parser.add_argument(
        "--official-scores", type=Path, default=DEFAULT_ROOT / "official-scores.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New run directory; existing results are never overwritten",
    )
    parser.add_argument("--scratch", type=Path, default=DEFAULT_ROOT / "sandbox")
    parser.add_argument(
        "--mode", choices=["non-thinking", "thinking"], default="non-thinking"
    )
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument(
        "--omit-seed",
        action="store_true",
        help="Do not send seed (greedy lanes that reject it); recorded",
    )
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--code-timeout", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", choices=list(CAPS), default=list(CAPS))
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug subset per task, recorded; omit for full benchmark",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Override every task cap; recorded"
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=3,
        help="Stop bench after consecutive HTTP/protocol/grader errors",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer.json; defaults to prefetched official tokenizer",
    )
    parser.add_argument(
        "--build-label",
        default="",
        help="Checkpoint/build identity, including draft on/off",
    )
    args = parser.parse_args(argv)
    if (
        args.code_timeout < 1
        or args.runs < 1
        or args.timeout <= 0
        or args.max_consecutive_errors < 1
        or (args.max_tokens is not None and args.max_tokens < 1)
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("timeouts, runs and limit must be positive")
    if args.command == "prefetch":
        prefetch(args.manifest, args.seed)
        return 0
    manifest = load_manifest(args.manifest)
    if args.seed != manifest["seed"]:
        parser.error("--seed must match frozen manifest seed")
    output = (
        args.output or DEFAULT_ROOT / f"{args.command}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        manifest_sha256=manifest["sha256"],
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caps={k: args.max_tokens or v for k, v in CAPS.items()},
        python=sys.version,
    )
    dump(output / "config.json", config)
    client = Client(
        args.base_url,
        args.model,
        args.mode,
        args.seed,
        args.timeout,
        os.getenv("OPENAI_API_KEY"),
        send_seed=not args.omit_seed,
    )
    started = time.monotonic()
    if args.command == "bench":
        if "humaneval" in args.tasks:
            probe = execute_code(
                "def identity(x):\n return x",
                "def check(f):\n assert f(7) == 7",
                "identity",
                args.code_timeout,
                args.scratch,
            )
            dump(output / "sandbox-probe.json", probe)
            if not probe["passed"]:
                dump(
                    output / "summary.json",
                    {
                        "config": config,
                        "error": "HumanEval sandbox unavailable",
                        "sandbox_probe": probe,
                        "complete": False,
                    },
                )
                print(
                    "HumanEval sandbox preflight failed; see sandbox-probe.json",
                    file=sys.stderr,
                )
                return 2
        results = []
        counts = Counter()
        consecutive_errors = 0
        with (output / "items.jsonl").open("w") as handle:
            for item in manifest["items"]:
                task = item["task"]
                if task not in args.tasks or args.limit and counts[task] >= args.limit:
                    continue
                counts[task] += 1
                messages, extra = prompt_item(item)
                response = client.chat(
                    messages, max_tokens=config["caps"][task], **extra
                )
                try:
                    grading = grade(item, response, args)
                except Exception as error:
                    grading = {
                        "passed": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                result = {
                    "task": task,
                    "id": item["id"],
                    "category": item["data"].get("category"),
                    "response": response,
                    **grading,
                }
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                consecutive_errors = (
                    consecutive_errors + 1 if grading.get("error") else 0
                )
                print(
                    f"{task}/{item['id']}: {'PASS' if result['passed'] else 'FAIL'}",
                    flush=True,
                )
                if consecutive_errors >= args.max_consecutive_errors:
                    break
    else:
        from tokenizers import Tokenizer

        tokenfile = args.tokenizer or Path(manifest["tokenizer"]["file"])
        config["actual_tokenizer_sha256"] = hashlib.sha256(
            tokenfile.read_bytes()
        ).hexdigest()
        dump(output / "config.json", config)
        tokenizer = Tokenizer.from_file(str(tokenfile))
        results = (
            smoke(client, tokenizer)
            if args.command == "smoke"
            else speed(client, tokenizer, args.runs)
        )
        for result in results:
            dump(output / f"{result['id']}.json", result)
    groups = defaultdict(list)
    for result in results:
        groups[result.get("task", args.command)].append(result)
    summary = {
        "config": config,
        "elapsed_seconds": time.monotonic() - started,
        "results": {
            key: {
                "n": len(rows),
                "passed": sum(r["passed"] for r in rows),
                "score_percent": 100 * sum(r["passed"] for r in rows) / len(rows),
                "errors": sum(
                    bool(r.get("error") or r.get("response", {}).get("error"))
                    for r in rows
                ),
                "truncated": sum(
                    r.get("response", {}).get("finish_reason") == "length" for r in rows
                ),
            }
            for key, rows in groups.items()
        },
        "comparison_note": "Frozen zero-shot chat subsets and strict BFCL argument matching are not official leaderboard protocols.",
    }
    if args.command == "bench":
        expected = sum(
            min(manifest["counts"][task], args.limit or manifest["counts"][task])
            for task in args.tasks
        )
        summary["complete"] = len(results) == expected
        summary["expected_items"] = expected
        summary["completion_tokens"] = sum(
            r["response"]["usage"].get("completion_tokens", 0) for r in results
        )
        if args.official_scores.exists():
            official = json.loads(args.official_scores.read_text())
            summary["published_reference"] = {
                "source_file": str(args.official_scores),
                "sha256": hashlib.sha256(args.official_scores.read_bytes()).hexdigest(),
                "scores": official["scores"],
                "comparison_warning": official["comparison_warning"],
            }
    if args.command == "speed":
        summary["timings"] = {}
        for size in (1024, 8192):
            rows = [
                r for r in results if r["id"].startswith(f"{size}_") and r["passed"]
            ]
            summary["timings"][str(size)] = {
                key: statistics.mean(r[key] for r in rows) if rows else None
                for key in ("decode_tok_s", "prefill_tok_s_client_estimate")
            }
    dump(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.command == "bench" and (
        not summary["complete"] or any(r.get("error") for r in results)
    ):
        return 2
    return (
        1
        if args.command in ("smoke", "speed") and any(not r["passed"] for r in results)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
