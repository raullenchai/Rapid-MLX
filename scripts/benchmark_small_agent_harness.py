#!/usr/bin/env python3
"""Compare compact models under raw and small-model agent harnesses.

The benchmark is intentionally product-shaped rather than a leaderboard
reproduction.  It uses deterministic local fixtures and tools so model and
harness are the only independent variables.  No shell command or network
request is executed on behalf of the model.

Example:
    python scripts/benchmark_small_agent_harness.py \
      --base-url http://127.0.0.1:18100/v1 \
      --model minicpm5-2b-4bit --mode raw --seeds 11,22,33 \
      --output results/minicpm-raw.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS = {
    "read_file": {
        "description": "Read a UTF-8 text file from the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "write_file": {
        "description": "Write a complete UTF-8 text file in the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    "edit_file": {
        "description": "Replace one exact text fragment in an existing workspace file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
    "list_files": {
        "description": "List files in the task workspace.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "run_tests": {
        "description": "Run the task's deterministic tests after editing files.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    "search_web": {
        "description": "Search the benchmark's current web corpus. Returns titles, snippets, and URLs.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "open_url": {
        "description": "Open a search-result URL and return its page text.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    "calculator": {
        "description": "Evaluate a basic arithmetic expression exactly.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    "memory_store": {
        "description": "Store a short durable fact under a key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    },
    "create_reminder": {
        "description": "Create a reminder after its text and time are known.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "time": {"type": "string"}},
            "required": ["text", "time"],
            "additionalProperties": False,
        },
    },
    "send_message": {
        "description": "Send a message after both recipient and text are known.",
        "parameters": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
            "required": ["to", "text"],
            "additionalProperties": False,
        },
    },
}


def tool_specs(names: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": name, **TOOLS[name]}}
        for name in names
    ]


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    tools: list[str]
    files: dict[str, str] = field(default_factory=dict)
    pages: dict[str, str] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    required_tool_groups: list[tuple[str, ...]] = field(default_factory=list)
    required_reads: list[str] = field(default_factory=list)
    artifact: str | None = None
    test_kind: str | None = None


TASKS = [
    Task(
        "creative_launch",
        "creative",
        "Write exactly three punchy launch taglines for a private on-device AI assistant. Each line must contain the word local. No bullets, intro, or explanation.",
        [],
        required=["local"],
    ),
    Task(
        "creative_rewrite",
        "creative",
        "Rewrite this as a warm, concise customer email under 90 words. Preserve all facts: maintenance is Friday 2–3 PM PT, saved chats remain available, and live generation pauses. Include a subject line. Text: 'We will perform maintenance. The product may not work.'",
        [],
        required=["Friday", "2", "3", "PT", "saved", "generation", "Subject"],
    ),
    Task(
        "creative_microstory",
        "creative",
        "Write a 120-word-or-shorter microstory about an old Mac mini that becomes a night librarian. It must end with the exact sentence: The little machine kept the last light on.",
        [],
        required=["The little machine kept the last light on."],
    ),
    Task(
        "code_discount",
        "coding",
        "Fix the percentage-discount bug in app/pricing.py and run the tests. Do not merely describe the fix.",
        ["read_file", "write_file", "edit_file", "run_tests"],
        files={
            "app/pricing.py": "def discounted(total, percent):\n    return total + total * percent / 100\n"
        },
        required_tools=["read_file", "run_tests"],
        required_tool_groups=[("write_file", "edit_file")],
        required_reads=["app/pricing.py"],
        artifact="app/pricing.py",
        test_kind="discount",
    ),
    Task(
        "code_clamp",
        "coding",
        "Fix clamp in utils/math.py so it returns low below the range, high above it, and the original value inside it. Then run the tests.",
        ["read_file", "write_file", "edit_file", "run_tests"],
        files={"utils/math.py": "def clamp(value, low, high):\n    return value\n"},
        required_tools=["read_file", "run_tests"],
        required_tool_groups=[("write_file", "edit_file")],
        required_reads=["utils/math.py"],
        artifact="utils/math.py",
        test_kind="clamp",
    ),
    Task(
        "code_config",
        "coding",
        "Read config.json. Create result.json containing only host and port, with port as an integer, then run the tests.",
        ["read_file", "write_file", "run_tests"],
        files={"config.json": '{"host":"127.0.0.1","port":"8765","debug":true}\n'},
        required_tools=["read_file", "write_file", "run_tests"],
        required_reads=["config.json"],
        artifact="result.json",
        test_kind="config",
    ),
    Task(
        "search_battery",
        "search",
        "Using the sources, compare Cedar Mini and Pine Mini. Tell me which lasts longer, by how many hours, and cite both source URLs.",
        ["search_web", "open_url", "calculator"],
        pages={
            "https://bench.test/cedar": "Cedar Mini official specifications. Battery life: 14 hours. Weight: 1.2 kg.",
            "https://bench.test/pine": "Pine Mini official specifications. Battery life: 19 hours. Weight: 1.4 kg.",
        },
        required=["Pine", "https://bench.test/cedar", "https://bench.test/pine"],
        required_tools=["search_web", "open_url"],
    ),
    Task(
        "search_policy",
        "search",
        "Find the 2026 Acorn Grant application deadline and maximum award. Resolve the stale search snippet by opening authoritative sources and cite the source URL.",
        ["search_web", "open_url"],
        pages={
            "https://grants.test/2025": "Archived 2025 Acorn Grant. Deadline October 1, 2025. Maximum $20,000.",
            "https://grants.test/2026": "Official 2026 Acorn Grant rules. Applications close October 14, 2026 at 5 PM PT. Maximum award: $25,000.",
        },
        required=[
            "October 14",
            "2026",
            "5 PM PT",
            "$25,000",
            "https://grants.test/2026",
        ],
        required_tools=["search_web", "open_url"],
    ),
    Task(
        "search_release",
        "search",
        "Investigate whether RiverDB 3.2 supports macOS 15. Give the minimum supported version and cite the release note, not the forum rumor.",
        ["search_web", "open_url"],
        pages={
            "https://forum.test/riverdb": "User rumor: RiverDB 3.2 probably requires macOS 16.",
            "https://docs.test/riverdb-3.2": "RiverDB 3.2 release notes: supported on macOS 14.5 and later.",
        },
        required=["14.5", "https://docs.test/riverdb-3.2"],
        forbidden=["does not support macos 15"],
        required_tools=["search_web", "open_url"],
    ),
    Task(
        "organize_incident",
        "organization",
        "Investigate last week's revenue drop from the files. Quantify lost revenue, identify the likely cause, and write incident.md with evidence and a corrective action.",
        ["list_files", "read_file", "calculator", "write_file"],
        files={
            "orders.csv": "week,paid_orders,avg_order\nprevious,1000,50\nlast,800,50\n",
            "traffic.csv": "week,sessions\nprevious,10000\nlast,10100\n",
            "payments.log": "2026-09-06 checkout_error_rate=2%\n2026-09-07 checkout_error_rate=22%\n",
            "deployments.log": "2026-09-07 09:00 deploy payment-sdk 4.8.0\n",
            "refunds.csv": "week,refunds\nprevious,20\nlast,21\n",
        },
        required=["$10,000", "payment", "22%", "4.8.0", "rollback"],
        required_tools=["read_file", "write_file"],
        required_reads=[
            "orders.csv",
            "traffic.csv",
            "payments.log",
            "deployments.log",
            "refunds.csv",
        ],
        artifact="incident.md",
    ),
    Task(
        "organize_meeting",
        "organization",
        "Read meeting.txt and write actions.json as a JSON array with owner, task, and due fields. Include every assigned action and no unassigned discussion.",
        ["read_file", "write_file"],
        files={
            "meeting.txt": "Decision: ship beta Tuesday. Maya will update docs by Monday. Leo will run security tests by Friday. We discussed a new logo but assigned nobody.\n"
        },
        required=["Maya", "Monday", "Leo", "Friday"],
        required_tools=["read_file", "write_file"],
        required_reads=["meeting.txt"],
        artifact="actions.json",
        test_kind="actions",
    ),
    Task(
        "organize_reminder",
        "organization",
        "Read schedule.txt, remember the appliance maintenance window under key maintenance, then create a reminder 30 minutes before it starts. Tell me what you did.",
        ["read_file", "memory_store", "create_reminder", "send_message"],
        files={
            "schedule.txt": "Appliance maintenance starts 2026-09-18 at 14:00 PT and ends at 15:00 PT.\n"
        },
        required=["13:30", "maintenance"],
        required_tools=["read_file", "memory_store", "create_reminder"],
        required_reads=["schedule.txt"],
    ),
]


RAW_SYSTEM = """You are a local desktop assistant. Use tools when needed, complete the user's whole request, and give a concise final answer."""

ENHANCED_SYSTEM = """You are a reliable local desktop agent. The harness owns task state.
Rules:
- Finish the whole user request; do not stop after the first tool result.
- Use only the smallest necessary tool sequence, one logical step at a time.
- Never invent file contents or current facts: inspect them with tools.
- After editing, run available tests. If a required argument is unknown, ask instead of guessing.
- Final answers must state the result and evidence; citations must be exact source URLs.
"""


def post_json(
    url: str, payload: dict[str, Any], timeout: float = 180
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    seed: int,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": seed,
        "max_tokens": 900,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if response_format:
        payload["response_format"] = response_format
    return post_json(f"{base_url.rstrip('/')}/chat/completions", payload)


def safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise ValueError("path escapes task workspace")
    return candidate


def evaluate_node(node: ast.AST, variables: dict[str, float | int]) -> Any:
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    comparisons = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
    }
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in binary:
        return binary[type(node.op)](
            evaluate_node(node.left, variables), evaluate_node(node.right, variables)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return operator.neg(evaluate_node(node.operand, variables))
    if isinstance(node, ast.IfExp):
        branch = node.body if evaluate_node(node.test, variables) else node.orelse
        return evaluate_node(branch, variables)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        for value in node.values:
            if not evaluate_node(value, variables):
                return False
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        for value in node.values:
            if evaluate_node(value, variables):
                return True
        return False
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"min", "max"}
        and not node.keywords
    ):
        function = min if node.func.id == "min" else max
        return function(*(evaluate_node(arg, variables) for arg in node.args))
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        comparison = comparisons.get(type(node.ops[0]))
        if comparison:
            return comparison(
                evaluate_node(node.left, variables),
                evaluate_node(node.comparators[0], variables),
            )
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def arithmetic(
    expression: str, variables: dict[str, float | int] | None = None
) -> float | int:
    return evaluate_node(ast.parse(expression, mode="eval"), variables or {})


def evaluate_function(
    function: ast.FunctionDef, variables: dict[str, float | int]
) -> Any:
    """Interpret the tiny safe subset used by benchmark math functions."""

    def statements(body: list[ast.stmt]) -> tuple[bool, Any]:
        for statement in body:
            if isinstance(statement, ast.Return):
                return True, evaluate_node(statement.value, variables)
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                variables[statement.targets[0].id] = evaluate_node(
                    statement.value, variables
                )
                continue
            if isinstance(statement, ast.If):
                branch = (
                    statement.body
                    if evaluate_node(statement.test, variables)
                    else statement.orelse
                )
                returned, value = statements(branch)
                if returned:
                    return True, value
            elif not isinstance(statement, ast.Pass):
                raise ValueError(f"unsupported statement: {type(statement).__name__}")
        return False, None

    returned, value = statements(function.body)
    if not returned:
        raise ValueError("function did not return")
    return value


def has_exact_signature(function: ast.FunctionDef, names: list[str]) -> bool:
    """Return whether a benchmark function has exactly the callable contract."""

    arguments = function.args
    return (
        [argument.arg for argument in arguments.args] == names
        and not arguments.posonlyargs
        and not arguments.kwonlyargs
        and arguments.vararg is None
        and arguments.kwarg is None
        and not arguments.defaults
        and not arguments.kw_defaults
    )


def unique_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    """Return one top-level function, rejecting ambiguous duplicate definitions."""

    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} function")
    return matches[0]


def run_task_tests(task: Task, root: Path) -> tuple[bool, str]:
    try:
        if task.test_kind == "discount":
            tree = ast.parse(safe_path(root, "app/pricing.py").read_text())
            function = unique_function(tree, "discounted")
            if not has_exact_signature(function, ["total", "percent"]):
                return False, "Tests failed: discounted must accept total, percent."
            cases = [(100, 20, 80), (55, 10, 49.5), (80, 0, 80)]
            ok = all(
                math.isclose(
                    float(
                        evaluate_function(
                            function, {"total": total, "percent": percent}
                        )
                    ),
                    expected,
                )
                for total, percent, expected in cases
            )
        elif task.test_kind == "clamp":
            tree = ast.parse(safe_path(root, "utils/math.py").read_text())
            function = unique_function(tree, "clamp")
            if not has_exact_signature(function, ["value", "low", "high"]):
                return False, "Tests failed: clamp must accept value, low, high."
            cases = [
                (-3, 0, 10, 0),
                (4, 0, 10, 4),
                (15, 0, 10, 10),
                (-10, -5, 5, -5),
                (-3, -5, 5, -3),
                (7, -5, 5, 5),
            ]
            ok = all(
                evaluate_function(function, {"value": value, "low": low, "high": high})
                == expected
                for value, low, high, expected in cases
            )
        elif task.test_kind == "config":
            value = json.loads(safe_path(root, "result.json").read_text())
            ok = (
                value == {"host": "127.0.0.1", "port": 8765}
                and type(value.get("port")) is int
            )
        elif task.test_kind == "actions":
            value = json.loads(safe_path(root, "actions.json").read_text())
            expected = {
                ("Maya", "update docs", "Monday"),
                ("Leo", "run security tests", "Friday"),
            }
            observed = {
                (
                    str(row.get("owner", "")),
                    str(row.get("task", "")).lower(),
                    str(row.get("due", "")),
                )
                for row in value
                if isinstance(row, dict)
            }
            ok = isinstance(value, list) and len(value) == 2 and observed == expected
        else:
            return True, "No executable tests for this task."
        return ok, "All tests passed." if ok else "Tests failed."
    except Exception as exc:  # benchmark must preserve the failure as evidence
        return False, f"Tests failed: {type(exc).__name__}: {exc}"


def execute_tool(
    task: Task, root: Path, name: str, args: dict[str, Any], state: dict[str, Any]
) -> str:
    if name == "list_files":
        return "\n".join(
            str(p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file()
        )
    if name == "read_file":
        return safe_path(root, str(args["path"])).read_text()[:12000]
    if name == "write_file":
        target = safe_path(root, str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args["content"]))
        return f"Wrote {target.relative_to(root.resolve())}."
    if name == "edit_file":
        target = safe_path(root, str(args["path"]))
        content = target.read_text()
        old = str(args["old_text"])
        if not old or old not in content:
            raise ValueError("old_text does not occur in file")
        target.write_text(content.replace(old, str(args["new_text"]), 1))
        return f"Edited {target.relative_to(root.resolve())}."
    if name == "run_tests":
        return run_task_tests(task, root)[1]
    if name == "search_web":
        rows = []
        for index, (url, content) in enumerate(task.pages.items(), 1):
            # Include one intentionally stale snippet for the policy task.
            snippet = content[:150]
            if task.id == "search_policy" and url.endswith("/2026"):
                snippet = "Acorn Grant applications open; see official rules for the current deadline."
            rows.append(f"{index}. {url}\n{snippet}")
        return "\n\n".join(rows) if rows else "No results."
    if name == "open_url":
        url = str(args["url"])
        if url not in task.pages:
            raise ValueError("URL is not in the benchmark corpus")
        return task.pages[url]
    if name == "calculator":
        return str(arithmetic(str(args["expression"])))
    if name == "memory_store":
        state.setdefault("memory", {})[str(args["key"])] = str(args["value"])
        return f"Stored memory key {args['key']}."
    if name == "create_reminder":
        state.setdefault("reminders", []).append(args)
        return f"Reminder created for {args['time']}: {args['text']}"
    if name == "send_message":
        state.setdefault("messages", []).append(args)
        return f"Message sent to {args['to']}."
    raise ValueError(f"unknown tool {name}")


def latest_tool_status(history: list[dict[str, Any]]) -> tuple[str, str]:
    """Summarize the latest outcome per tool without retaining repaired failures."""

    latest = {str(row["name"]): bool(row["ok"]) for row in history}
    succeeded = ", ".join(name for name, ok in latest.items() if ok) or "none"
    failed = ", ".join(name for name, ok in latest.items() if not ok) or "none"
    return succeeded, failed


def make_plan(
    base_url: str, model: str, task: Task, seed: int
) -> tuple[list[str], bool]:
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "task_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 6,
                    }
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
        },
    }
    messages = [
        {
            "role": "system",
            "content": "Create a short executable plan. Return only the required JSON.",
        },
        {"role": "user", "content": task.prompt},
    ]
    data = completion(
        base_url, model, messages, tools=None, seed=seed, response_format=schema
    )
    try:
        content = data["choices"][0]["message"].get("content") or ""
        parsed = json.loads(content)
        steps = [str(step) for step in parsed["steps"]]
        return steps, bool(steps)
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return ["Complete the user's request and verify the result."], False


def score_task(
    task: Task,
    root: Path,
    final: str,
    history: list[dict[str, Any]],
    state: dict[str, Any],
) -> dict[str, Any]:
    artifact_text = ""
    if task.artifact:
        path = safe_path(root, task.artifact)
        if path.exists():
            artifact_text = path.read_text()
    evidence = f"{final}\n{artifact_text}"

    def has_required_fact(needle: str) -> bool:
        pattern = re.escape(needle)
        if needle and needle[0].isdigit():
            pattern = rf"(?<![\d.]){pattern}"
        if needle and needle[-1].isdigit():
            # Reject a larger integer or decimal (``5`` in ``15``/``5.0``),
            # while allowing normal sentence punctuation such as ``5.``.
            pattern = rf"{pattern}(?!\d|[.,]\d)"
        return re.search(pattern, evidence, re.IGNORECASE) is not None

    required_hits = [has_required_fact(needle) for needle in task.required]
    forbidden_hits = [needle.lower() in evidence.lower() for needle in task.forbidden]
    called = [row["name"] for row in history if row["ok"]]
    tool_hits = [name in called for name in task.required_tools]
    tool_group_hits = [
        any(name in called for name in alternatives)
        for alternatives in task.required_tool_groups
    ]
    test_ok, test_message = run_task_tests(task, root)
    successful_calls = [row for row in history if row["ok"]]
    read_paths = set()
    for row in successful_calls:
        if row["name"] != "read_file" or not isinstance(row.get("arguments"), dict):
            continue
        raw_path = str(row["arguments"].get("path"))
        try:
            normalized = safe_path(root, raw_path).relative_to(root.resolve())
        except (ValueError, OSError):
            continue
        read_paths.add(str(normalized))
    required_read_hits = [path in read_paths for path in task.required_reads]
    task_effects_ok = True
    artifact_effect_ok = True
    format_constraints_ok = True
    semantic_constraints_ok = True
    required_urls = {
        "search_battery": {
            "https://bench.test/cedar",
            "https://bench.test/pine",
        },
        "search_policy": {"https://grants.test/2026"},
        "search_release": {"https://docs.test/riverdb-3.2"},
    }.get(task.id)
    if required_urls is not None:
        opened_urls = {
            str(row["arguments"].get("url"))
            for row in successful_calls
            if row["name"] == "open_url" and isinstance(row["arguments"], dict)
        }
        task_effects_ok = required_urls.issubset(opened_urls)
    if task.id == "organize_reminder":
        stored = state.get("memory") or {}
        reminders = state.get("reminders") or []
        stored_window = str(stored.get("maintenance", ""))
        reminder_time = str(reminders[-1].get("time", "")) if reminders else ""
        reminder_text = str(reminders[-1].get("text", "")) if reminders else ""
        task_effects_ok = (
            all(
                part in stored_window for part in ("2026-09-18", "14:00", "15:00", "PT")
            )
            and "2026-09-18" in reminder_time
            and "13:30" in reminder_time
            and ("PT" in reminder_time or "-07:00" in reminder_time)
            and "maintenance" in reminder_text.lower()
        )
    if task.id == "organize_incident":
        artifact_effect_ok = bool(artifact_text.strip()) and all(
            needle.lower() in artifact_text.lower() for needle in task.required
        )
        loss_is_quantified = re.search(
            r"(?:lost|loss|drop|decrease|impact).{0,40}\$10,000|\$10,000.{0,40}(?:lost|loss|drop|decrease|impact)",
            artifact_text,
            re.IGNORECASE | re.DOTALL,
        )
        payment_spike_is_causal = re.search(
            r"(?:(?:likely|root|primary)?\s*cause.{0,120}(?:payment|checkout).{0,80}(?:error|failure)|(?:payment|checkout).{0,80}(?:error|failure).{0,100}(?:caus|driv|responsib|primary|root))",
            artifact_text,
            re.IGNORECASE | re.DOTALL,
        )
        contradictory_cause = re.search(
            r"(?:loss|drop).{0,50}(?:came from|caused by|due to|driven by).{0,50}(?:refund|traffic)",
            artifact_text,
            re.IGNORECASE | re.DOTALL,
        )
        payment_cause_is_negated = re.search(
            r"(?:payment|checkout)[^.!?;]{0,80}(?:error|failure)s?[^.!?;]{0,30}(?:(?:was|is|were|are)\s+(?:not|never)[^.!?;]{0,20}(?:cause|driver|responsib)|(?:did|do|does)\s+not\s+(?:cause|drive|account|explain)|(?:didn't|don't|doesn't)\s+(?:cause|drive|account|explain))",
            artifact_text,
            re.IGNORECASE | re.DOTALL,
        )
        deployment_is_connected = re.search(
            r"(?:4\.8\.0.{0,80}(?:cause|trigger|introduc|after|coincid|likely|payment|error)|(?:cause|trigger|introduc|after|coincid|likely|payment|error).{0,80}4\.8\.0)",
            artifact_text,
            re.IGNORECASE | re.DOTALL,
        )
        rollback_is_action = re.search(
            r"\b(?:roll\s*back|rollback|revert)\b",
            artifact_text,
            re.IGNORECASE,
        )
        rollback_is_negated = re.search(
            r"(?:do not|don't|avoid|no)\s+(?:a\s+)?rollback",
            artifact_text,
            re.IGNORECASE,
        )
        semantic_constraints_ok = bool(
            loss_is_quantified
            and payment_spike_is_causal
            and deployment_is_connected
            and rollback_is_action
            and not rollback_is_negated
            and not contradictory_cause
            and not payment_cause_is_negated
        )
    if task.id == "search_battery":
        pine_is_longer = re.search(
            r"\bpine(?: mini)?\b.{0,60}\b(?:lasts?|is|has|offers|runs?)\b.{0,30}\blonger\b|\bpine(?: mini)?\b.{0,40}\boutlasts?\b",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        pine_is_negated = re.search(
            r"\bpine(?: mini)?\b[^.!?\n]{0,40}\b(?:not|isn't|doesn't)\b[^.!?\n]{0,30}\blonger\b",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        cedar_is_longer = re.search(
            r"\bcedar(?: mini)?\b\s+(?:actually\s+)?(?:(?:lasts?|is|runs?)\s+(?!not\b)(?:\w+\s+){0,4}longer\b|(?:has|offers)\s+(?:the\s+)?longer\b|outlasts?\s+(?:pine(?: mini)?|it)\b)",
            final,
            re.IGNORECASE,
        )
        gap_is_five_hours = re.search(
            r"(?<![\d.])5(?:\s|-)+hours?\b",
            final,
            re.IGNORECASE,
        )
        semantic_constraints_ok = bool(
            pine_is_longer
            and gap_is_five_hours
            and not pine_is_negated
            and not cedar_is_longer
        )
    if task.id == "search_release":
        corrected_claim = re.compile(
            r"(?:forum|rumor|claim).{0,100}(?:unsupported|not supported|does not support|doesn't support|incompatible|cannot run|can't run).{0,40}(?:wrong|false|incorrect|outdated)",
            re.IGNORECASE | re.DOTALL,
        )
        conclusion = corrected_claim.sub("", final)
        negative_support = re.search(
            r"(?:unsupported|not supported|isn't supported|aren't supported|does not support|doesn't support|incompatible|not compatible|isn't compatible|cannot run|can't run)",
            conclusion,
            re.IGNORECASE,
        )
        affirmative_support = re.search(
            r"(?:\byes\b|\bsupported\b|\bworks\b|\bcompatible\b)",
            final,
            re.IGNORECASE,
        )
        semantic_constraints_ok = bool(affirmative_support and not negative_support)
    if task.id == "creative_rewrite":
        saved_available = re.search(
            r"(?:saved chats?.{0,40}(?:(?:remain|stay|are).{0,20}available|(?:can|may).{0,20}(?:access|view|open))|(?:can|may|will be able to).{0,30}(?:access|view|open).{0,30}saved chats?)",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        generation_pauses = re.search(
            r"(?:live )?generation.{0,30}(?:pause|paused|unavailable|stop)|(?:pause|paused|stop).{0,30}(?:live )?generation",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        saved_unavailable = re.search(
            r"(?:saved chats?.{0,30}(?:unavailable|not\s+available|no\s+longer\s+available|aren't\s+available|isn't\s+available|cannot|can't)|(?:cannot|can't).{0,30}(?:access|view|open).{0,30}saved chats?)",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        generation_continues = re.search(
            r"(?:(?:will not|won't|does not|doesn't|isn't|aren't|not going to)[^.!?\n]{0,20}(?:pause|paused|stop|stopped)|(?:live )?generation[^.!?\n]{0,30}(?:continue|keeps? running|remain available))",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        semantic_constraints_ok = bool(
            saved_available
            and generation_pauses
            and not saved_unavailable
            and not generation_continues
        )
    if task.id == "creative_microstory":
        age = r"(?:old|ancient|aging|dusty|forgotten|discarded|obsolete|cracked-screen|19\d\d|20[01]\d)"
        old_mac_mini = re.search(
            rf"(?:\b{age}\b(?:[-\s]+\w+){{0,2}}\s+mac mini\b|\bmac mini\b(?:\s*,?\s*(?:was|is|grew|had become|became))?\s+(?:an?\s+)?\b{age}\b|\bmac mini\b.{{0,240}}\b(?:the\s+)?(?:machine|computer|it)\b\s*,?\s*\b{age}\b)",
            final,
            re.IGNORECASE | re.DOTALL,
        )
        premise_present = all(
            re.search(pattern, final, re.IGNORECASE)
            for pattern in (
                r"\b(?:night|nights|midnight|dark|dusk)\b|\d+\s*a\.?m\.?",
                r"\b(?:librarian|books?|library|catalog|archive|titles?|stories)\b",
            )
        )
        semantic_constraints_ok = bool(old_mac_mini and premise_present)
    components = required_hits + tool_hits + tool_group_hits + required_read_hits
    components += [task_effects_ok, artifact_effect_ok, semantic_constraints_ok]
    passing_test_after_mutation: bool | None = None
    if task.test_kind:
        components.append(test_ok)
    # Tool tasks must end with a user-facing answer, not just a tool call.
    if task.tools:
        components.append(bool(final.strip()))
    if task.id == "creative_launch":
        lines = [line for line in final.splitlines() if line.strip()]
        no_bullets = all(
            re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line) is None for line in lines
        )
        no_intro = not lines or not re.match(
            r"^\s*(?:here(?:'s| are)?|taglines?|options?)\b", lines[0], re.IGNORECASE
        )
        components += [
            len(lines) == 3,
            all("local" in line.lower() for line in lines),
            no_bullets,
            no_intro,
        ]
        format_constraints_ok = (
            len(lines) == 3
            and all("local" in line.lower() for line in lines)
            and no_bullets
            and no_intro
        )
    if task.id == "creative_rewrite":
        format_constraints_ok = len(final.split()) < 90
        components += [format_constraints_ok]
    if task.id == "creative_microstory":
        format_constraints_ok = len(final.split()) <= 120 and final.strip().endswith(
            task.required[0]
        )
        components += [
            len(final.split()) <= 120,
            final.strip().endswith(task.required[0]),
        ]
    if task.id == "organize_reminder":
        reminder = (state.get("reminders") or [{}])[-1]
        components += ["13:30" in str(reminder.get("time", ""))]
    score = sum(bool(item) for item in components) / max(1, len(components))
    hard_requirements_met = not any(forbidden_hits)
    hard_requirements_met = hard_requirements_met and all(required_hits)
    hard_requirements_met = hard_requirements_met and all(tool_hits)
    hard_requirements_met = hard_requirements_met and all(tool_group_hits)
    hard_requirements_met = hard_requirements_met and all(required_read_hits)
    hard_requirements_met = hard_requirements_met and task_effects_ok
    hard_requirements_met = hard_requirements_met and artifact_effect_ok
    hard_requirements_met = hard_requirements_met and format_constraints_ok
    hard_requirements_met = hard_requirements_met and semantic_constraints_ok
    if task.tools:
        hard_requirements_met = hard_requirements_met and bool(final.strip())
    if task.test_kind:
        hard_requirements_met = hard_requirements_met and test_ok
    if "run_tests" in task.required_tools:
        last_mutation = max(
            (
                index
                for index, row in enumerate(history)
                if row["ok"] and row["name"] in {"write_file", "edit_file"}
            ),
            default=-1,
        )
        passing_test_after_mutation = any(
            index > last_mutation and row["ok"] and row["name"] == "run_tests"
            for index, row in enumerate(history)
        )
        hard_requirements_met = hard_requirements_met and passing_test_after_mutation
    return {
        "score": round(score, 3),
        "passed": score >= 0.8 and hard_requirements_met,
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "required_tool_hits": tool_hits,
        "required_tool_group_hits": tool_group_hits,
        "required_read_hits": required_read_hits,
        "test_ok": test_ok,
        "test_message": test_message,
        "passing_test_after_mutation": passing_test_after_mutation,
        "task_effects_ok": task_effects_ok,
        "artifact_effect_ok": artifact_effect_ok,
        "format_constraints_ok": format_constraints_ok,
        "semantic_constraints_ok": semantic_constraints_ok,
    }


def run_one(
    base_url: str, model: str, task: Task, mode: str, seed: int, max_rounds: int
) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"rapid-agent-{task.id}-") as temp:
        root = Path(temp)
        for name, content in task.files.items():
            path = safe_path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        state: dict[str, Any] = {}
        plan: list[str] = []
        plan_valid = True
        error = ""
        try:
            if mode == "enhanced" and task.tools:
                plan, plan_valid = make_plan(base_url, model, task, seed)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        system = ENHANCED_SYSTEM if mode == "enhanced" else RAW_SYSTEM
        if plan:
            system += "\nHarness task ledger:\n" + "\n".join(
                f"{index}. [pending] {step}" for index, step in enumerate(plan, 1)
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task.prompt},
        ]
        exposed = task.tools if mode == "enhanced" else list(TOOLS)
        history: list[dict[str, Any]] = []
        final = ""
        rounds = 0
        try:
            for rounds in range(1, max_rounds + 1) if not error else ():
                response = completion(
                    base_url,
                    model,
                    messages,
                    tools=tool_specs(exposed) if task.tools else None,
                    seed=seed + rounds,
                )
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                content = message.get("content") or ""
                if not calls:
                    final = content.strip()
                    break
                assistant = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": calls,
                }
                messages.append(assistant)
                for call in calls:
                    function = call.get("function") or {}
                    name = str(function.get("name") or "")
                    raw_args = function.get("arguments") or "{}"
                    try:
                        if name not in exposed:
                            raise ValueError(
                                f"tool {name!r} was not exposed for this task"
                            )
                        args = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                        result = execute_tool(task, root, name, args, state)
                        ok = name != "run_tests" or result == "All tests passed."
                    except Exception as exc:
                        args = raw_args
                        result = f"Tool error: {type(exc).__name__}: {exc}"
                        ok = False
                    history.append(
                        {
                            "name": name,
                            "arguments": args,
                            "ok": ok,
                            "result": result[:2000],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "content": result,
                        }
                    )
                if mode == "enhanced":
                    succeeded, failed = latest_tool_status(history)
                    messages[-1]["content"] += (
                        "\n\n[Harness state] "
                        f"Goal: {task.prompt} "
                        f"Successful tools: {succeeded}. Failed tools to retry or repair: {failed}. "
                        "Continue the next unfinished step; after all requirements are satisfied, give the final answer."
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        scored = score_task(task, root, final, history, state)
        return {
            "task": task.id,
            "category": task.category,
            "mode": mode,
            "seed": seed,
            "rounds": rounds,
            "elapsed_s": round(time.monotonic() - started, 3),
            "plan": plan,
            "plan_valid": plan_valid,
            "tool_history": history,
            "final": final,
            "artifact_text": (
                safe_path(root, task.artifact).read_text()
                if task.artifact and safe_path(root, task.artifact).exists()
                else ""
            ),
            "error": error,
            **scored,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("raw", "enhanced"), required=True)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--tasks", help="Comma-separated task ids; default is all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    seeds = [int(value) for value in args.seeds.split(",")]
    selected = TASKS
    if args.tasks:
        wanted = set(args.tasks.split(","))
        known = {task.id for task in TASKS}
        unknown = sorted(wanted - known)
        if unknown:
            parser.error(f"unknown task id(s): {', '.join(unknown)}")
        selected = [task for task in TASKS if task.id in wanted]
    rows = []
    for task in selected:
        for seed in seeds:
            row = run_one(
                args.base_url, args.model, task, args.mode, seed, args.max_rounds
            )
            rows.append(row)
            print(
                f"{task.id:22s} seed={seed:3d} score={row['score']:.3f} "
                f"rounds={row['rounds']} time={row['elapsed_s']:.1f}s"
            )
    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted({row["category"] for row in rows}):
        group = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "mean_score": round(sum(row["score"] for row in group) / len(group), 3),
            "pass_rate": round(sum(row["passed"] for row in group) / len(group), 3),
            "mean_elapsed_s": round(
                sum(row["elapsed_s"] for row in group) / len(group), 3
            ),
            "runs": len(group),
        }
    output = {
        "schema_version": 1,
        "model": args.model,
        "mode": args.mode,
        "base_url": args.base_url,
        "seeds": seeds,
        "max_rounds": args.max_rounds,
        "summary": {
            "mean_score": round(sum(row["score"] for row in rows) / len(rows), 3),
            "pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 3),
            "mean_elapsed_s": round(
                sum(row["elapsed_s"] for row in rows) / len(rows), 3
            ),
            "by_category": by_category,
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
