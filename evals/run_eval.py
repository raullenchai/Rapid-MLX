#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unified Model Evaluation Runner for vllm-mlx.

Runs 4 test suites against a running vllm-mlx server and produces a
standardized JSON result file for model comparison.

Suites:
  A. Speed     — TTFT, decode tok/s, memory
  B. Tools     — 30 tool-calling scenarios (9 categories)
  C. Coding    — 10 code generation tasks (auto-graded)
  D. Reasoning — 10 math problems (MATH-500 subset)
  E. General   — 10 MMLU-Pro multiple choice questions

Usage:
    # Start server first:
    vllm-mlx serve <model> --port 8000

    # Run all suites:
    python evals/run_eval.py --model <display-name> --port 8000

    # Run specific suites:
    python evals/run_eval.py --model <name> --suite speed tool_calling

    # Specify parser (for tool calling):
    python evals/run_eval.py --model <name> --parser hermes

Community contributors: see evals/README.md for how to submit results.
"""

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

try:
    import httpx

    _HTTPX = True
except ImportError:
    _HTTPX = False

EVALS_DIR = Path(__file__).parent
PROMPTS_DIR = EVALS_DIR / "prompts"
RESULTS_DIR = EVALS_DIR / "results"

# Reuse tool definitions from test_tool_call_e2e
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Execute shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process",
            "description": "Process management",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list",
                            "poll",
                            "log",
                            "write",
                            "kill",
                            "clear",
                            "remove",
                        ],
                    },
                    "sessionId": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_store",
            "description": "Store key-value pair",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_get",
            "description": "Get value by key",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message",
            "parameters": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                "required": ["to", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": "Create a reminder",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "time": {"type": "string"}},
                "required": ["text", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create",
            "description": "Create calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Browse a URL",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_run",
            "description": "Run code snippet",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_gen",
            "description": "Generate an image",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "to": {"type": "string"}},
                "required": ["text", "to"],
            },
        },
    },
]


# =============================================================================
# Helpers
# =============================================================================


def server_available(host: str, port: int) -> bool:
    if not _HTTPX:
        print("ERROR: httpx is required. Install with: pip install httpx")
        return False
    try:
        r = httpx.get(f"http://{host}:{port}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def detect_hardware() -> dict:
    """Detect Apple Silicon hardware info."""
    hw = {"chip": "Unknown", "memory_gb": 0, "os": platform.platform()}
    try:
        sp = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        hw["chip"] = sp.stdout.strip()
    except Exception:
        pass
    try:
        sp = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        hw["memory_gb"] = round(int(sp.stdout.strip()) / (1024**3))
    except Exception:
        pass
    return hw


def chat_request(
    host: str,
    port: int,
    messages: list,
    *,
    tools=None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stream: bool = False,
    timeout: float = 120.0,
    enable_thinking: bool = False,
) -> dict:
    """Send a chat completion request (non-streaming by default)."""
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "enable_thinking": enable_thinking,
    }
    if tools:
        body["tools"] = tools

    resp = httpx.post(
        f"http://{host}:{port}/v1/chat/completions",
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def stream_chat(
    host: str,
    port: int,
    messages: list,
    *,
    tools=None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    enable_thinking: bool = False,
):
    """Stream a chat completion. Returns (content, tool_calls, ttft, elapsed)."""
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        body["tools"] = tools

    content = ""
    tool_calls_by_index: dict[int, dict] = {}
    ttft = None
    start = time.perf_counter()

    with httpx.stream(
        "POST",
        f"http://{host}:{port}/v1/chat/completions",
        json=body,
        timeout=timeout,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "choices" not in chunk or not chunk["choices"]:
                continue
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta and delta["content"]:
                if ttft is None:
                    ttft = time.perf_counter() - start
                content += delta["content"]
            if "tool_calls" in delta and delta["tool_calls"]:
                if ttft is None:
                    ttft = time.perf_counter() - start
                # Merge streaming tool_call deltas by index
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc_delta.get("id", f"call_{idx}"),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_by_index[idx]
                    if "id" in tc_delta and tc_delta["id"]:
                        tc["id"] = tc_delta["id"]
                    fn_delta = tc_delta.get("function", {})
                    if fn_delta.get("name"):
                        tc["function"]["name"] = fn_delta["name"]
                    if fn_delta.get("arguments"):
                        tc["function"]["arguments"] += fn_delta["arguments"]

    elapsed = time.perf_counter() - start
    # Return tool_calls sorted by index
    tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return content, tool_calls, ttft or elapsed, elapsed


# =============================================================================
# Suite A: Speed
# =============================================================================


def run_speed_suite(host: str, port: int, verbose: bool = False) -> dict:
    """Measure TTFT (cold/warm) and decode tok/s."""
    print("\n--- Suite A: Speed ---")
    results = {}

    # -- TTFT cold (first request, no cache) --
    print("  TTFT cold (first request)...", end=" ", flush=True)
    cold_msgs = [{"role": "user", "content": "Hello, how are you?"}]
    _, _, ttft_cold, _ = stream_chat(host, port, cold_msgs, max_tokens=20)
    results["ttft_cold_s"] = round(ttft_cold, 3)
    print(f"{ttft_cold * 1000:.0f} ms")

    # -- TTFT warm (repeat same prefix) --
    print("  TTFT warm (cached prefix)...", end=" ", flush=True)
    _, _, ttft_warm, _ = stream_chat(host, port, cold_msgs, max_tokens=20)
    results["ttft_warm_s"] = round(ttft_warm, 3)
    print(f"{ttft_warm * 1000:.0f} ms")

    # -- Decode tok/s --
    # Use end-to-end non-streaming request for accurate token counts from usage,
    # then compute effective tok/s (includes TTFT overhead, which is what users experience).
    print("  Decode short (<100 tok)...", end=" ", flush=True)
    short_msg = [
        {
            "role": "user",
            "content": "Write a brief greeting in 2 sentences. Be concise, no thinking.",
        }
    ]
    short_start = time.perf_counter()
    short_resp = chat_request(host, port, short_msg, max_tokens=100, temperature=0.0)
    short_elapsed = time.perf_counter() - short_start
    short_tokens = short_resp.get("usage", {}).get("completion_tokens", 0)
    short_tps = short_tokens / short_elapsed if short_elapsed > 0 else 0
    results["decode_short_tps"] = round(short_tps, 1)
    print(f"{short_tps:.1f} tok/s ({short_tokens} tok in {short_elapsed:.2f}s)")

    print("  Decode long (300+ tok)...", end=" ", flush=True)
    long_msg = [
        {
            "role": "user",
            "content": "Write a detailed explanation of how transformers work in deep learning. Cover attention mechanisms, positional encoding, and the encoder-decoder architecture. Be thorough. No thinking, just answer directly.",
        }
    ]
    long_start = time.perf_counter()
    long_resp = chat_request(host, port, long_msg, max_tokens=500, temperature=0.0)
    long_elapsed = time.perf_counter() - long_start
    long_tokens = long_resp.get("usage", {}).get("completion_tokens", 0)
    long_tps = long_tokens / long_elapsed if long_elapsed > 0 else 0
    results["decode_long_tps"] = round(long_tps, 1)
    print(f"{long_tps:.1f} tok/s ({long_tokens} tok in {long_elapsed:.2f}s)")

    # -- RAM usage from /v1/status Metal metrics --
    try:
        r = httpx.get(f"http://{host}:{port}/v1/status", timeout=5.0)
        r.raise_for_status()
        metal = r.json().get("metal", {})
        ram_active = metal.get("active_memory_gb")
        ram_peak = metal.get("peak_memory_gb")
        if ram_active is not None:
            results["ram_active_gb"] = round(ram_active, 1)
            print(f"  RAM active: {ram_active:.1f} GB")
        if ram_peak is not None:
            results["ram_peak_gb"] = round(ram_peak, 1)
            print(f"  RAM peak:   {ram_peak:.1f} GB")
    except Exception:
        pass

    results["_summary"] = (
        f"TTFT cold={results['ttft_cold_s']}s warm={results['ttft_warm_s']}s | Decode {results['decode_short_tps']}/{results['decode_long_tps']} tok/s"
    )
    print(f"  Summary: {results['_summary']}")
    return results


# =============================================================================
# Suite B: Tool Calling
# =============================================================================


def fuzzy_match_args(expected: dict, actual: dict) -> float:
    """Score 0-1 how well actual args match expected. Fuzzy with word-overlap fallback."""
    if not expected or not actual:
        return 0.0
    matches = 0
    total = len(expected)
    for key, exp_val in expected.items():
        act_val = actual.get(key, "")
        if isinstance(exp_val, str) and isinstance(act_val, str):
            exp_lower = exp_val.lower()
            act_lower = act_val.lower()
            # Exact substring match
            if exp_lower in act_lower or act_lower in exp_lower:
                matches += 1
            else:
                # Word-overlap: check if key words from expected appear in actual
                exp_words = set(re.findall(r"\w+", exp_lower))
                act_words = set(re.findall(r"\w+", act_lower))
                if exp_words and act_words:
                    overlap = len(exp_words & act_words) / len(exp_words)
                    matches += overlap  # partial credit
        elif exp_val == act_val:
            matches += 1
    return matches / total if total > 0 else 0.0


def _check_tool_call(tool_calls, scenario, step_prefix="") -> dict:
    """Check a single tool call against expected values. Returns grading dict."""
    expected_key = f"{step_prefix}expected_tool" if step_prefix else "expected_tool"

    tool_detected = len(tool_calls) > 0
    correct_name = False
    valid_json_args = False
    arg_score = 0.0

    if tool_detected:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        expected_name = scenario.get(expected_key, "")
        correct_name = tool_name == expected_name

        try:
            actual_args = json.loads(fn.get("arguments", "{}"))
            valid_json_args = True
            # Only grade arg content on first step (followups just check name)
            if not step_prefix and "expected_args" in scenario:
                match_mode = scenario.get("arg_match_mode", "fuzzy")
                if match_mode == "exact":
                    arg_score = 1.0 if actual_args == scenario["expected_args"] else 0.0
                else:
                    arg_score = fuzzy_match_args(scenario["expected_args"], actual_args)
            else:
                arg_score = 1.0  # followup steps: just check tool name
        except (json.JSONDecodeError, TypeError):
            valid_json_args = False

    return {
        "tool_detected": tool_detected,
        "correct_name": correct_name,
        "valid_json_args": valid_json_args,
        "arg_score": round(arg_score, 2),
    }


def _check_parallel_calls(tool_calls, scenario) -> dict:
    """Check parallel tool calls against expected_tools list. Returns grading dict."""
    expected = scenario.get("expected_tools", [])
    if not expected:
        return {
            "tool_detected": False,
            "correct_name": False,
            "valid_json_args": False,
            "arg_score": 0.0,
        }

    # Build list of actual calls
    actual_calls = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
            actual_calls.append({"name": name, "args": args, "valid_json": True})
        except (json.JSONDecodeError, TypeError):
            actual_calls.append({"name": name, "args": {}, "valid_json": False})

    # Match each expected tool to best actual call (greedy)
    matched = 0
    used = set()
    for exp in expected:
        exp_name = exp["tool"]
        exp_args = exp.get("expected_args", {})
        match_mode = exp.get("arg_match_mode", "fuzzy")
        best_score = -1
        best_idx = -1
        for i, act in enumerate(actual_calls):
            if i in used or act["name"] != exp_name:
                continue
            if match_mode == "exact":
                score = 1.0 if act["args"] == exp_args else 0.0
            elif match_mode == "contains":
                score = (
                    1.0
                    if all(
                        str(v) in str(act["args"].get(k, ""))
                        for k, v in exp_args.items()
                    )
                    else 0.0
                )
            else:
                score = fuzzy_match_args(exp_args, act["args"])
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= 0.5:
            matched += 1
            used.add(best_idx)

    total_expected = len(expected)
    fraction = matched / total_expected if total_expected > 0 else 0.0
    all_json_valid = (
        all(a["valid_json"] for a in actual_calls) if actual_calls else False
    )

    return {
        "tool_detected": len(actual_calls) > 0,
        "correct_name": matched == total_expected,
        "valid_json_args": all_json_valid,
        "arg_score": round(fraction, 2),
        "matched": matched,
        "expected_count": total_expected,
        "actual_count": len(actual_calls),
    }


def run_tool_calling_suite(host: str, port: int, verbose: bool = False) -> dict:
    """Run tool-calling scenarios with multi-turn, parallel, irrelevance, and error recovery support."""
    # NOTE: GPT-OSS-20B scored 3% tools before SUPPORTS_NATIVE_TOOL_FORMAT=True fix (harmony parser).
    #       After fix, scores 80% — the model needs native tool message format for multi-turn.
    print("\n--- Suite B: Tool Calling ---")

    prompts_file = PROMPTS_DIR / "tool_calling.json"
    scenarios = json.loads(prompts_file.read_text())

    details = []
    passed = 0

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. You have access to tools. "
            "Use them when appropriate to help the user. "
            "If a request is missing critical information needed for a tool call, "
            "ask the user for clarification instead of guessing. "
            "If a tool returns an error, explain the issue or try an alternative approach. "
            "After getting tool results, provide a concise final answer."
        ),
    }

    for sc in scenarios:
        sc_id = sc["id"]
        sc_type = sc.get("type", "standard")
        print(
            f"  {sc_id} (L{sc['level']}): {sc['description']}...", end=" ", flush=True
        )

        messages = [system_msg] + sc["messages"]
        result = {
            "id": sc_id,
            "level": sc["level"],
            "description": sc["description"],
            "category": sc.get("category", ""),
        }

        try:
            # ── Irrelevance / Missing Params: expect NO tool call ──
            if sc_type in ("irrelevance", "missing_params"):
                content, tool_calls, ttft, elapsed = stream_chat(
                    host, port, messages, tools=TOOLS, max_tokens=512, temperature=0.0
                )
                no_tool = not tool_calls
                has_content = bool(content and content.strip())
                ok = no_tool and has_content
                result.update(
                    {
                        "fully_correct": ok,
                        "no_tool_called": no_tool,
                        "has_content": has_content,
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                if ok:
                    passed += 1
                    print("PASS (no tool, text response)")
                else:
                    reason = "called a tool" if not no_tool else "empty response"
                    if not no_tool and tool_calls:
                        fn_name = tool_calls[0].get("function", {}).get("name", "?")
                        reason = f"called {fn_name}"
                    print(f"FAIL ({reason})")
                details.append(result)
                continue

            # ── Parallel: expect multiple tool calls in one response ──
            if sc_type == "parallel":
                content, tool_calls, ttft, elapsed = stream_chat(
                    host, port, messages, tools=TOOLS, max_tokens=512, temperature=0.0
                )
                grade = _check_parallel_calls(tool_calls, sc)
                ok = (
                    grade["correct_name"]
                    and grade["valid_json_args"]
                    and grade["arg_score"] >= 0.5
                )
                result.update(grade)
                result["fully_correct"] = ok
                result["elapsed_s"] = round(elapsed, 2)
                if ok:
                    passed += 1
                    print(f"PASS ({grade['matched']}/{grade['expected_count']} tools)")
                else:
                    print(
                        f"FAIL ({grade['matched']}/{grade['expected_count']} matched, {grade['actual_count']} called)"
                    )
                details.append(result)
                continue

            # ── Error Recovery: feed error result, check model adapts ──
            if sc_type == "error_recovery":
                content, tool_calls, ttft, elapsed = stream_chat(
                    host, port, messages, tools=TOOLS, max_tokens=512, temperature=0.0
                )
                grade = _check_tool_call(tool_calls, sc)
                first_ok = grade["tool_detected"] and grade["correct_name"]
                result.update(grade)
                result["elapsed_s"] = round(elapsed, 2)

                if first_ok and tool_calls:
                    # Feed error result back
                    tc = tool_calls[0]
                    fn = tc.get("function", {})
                    error_text = sc.get(
                        "error_result", "Error: Unknown error occurred."
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc.get("id", "call_eval"),
                                    "type": "function",
                                    "function": fn,
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": error_text,
                            "tool_call_id": tc.get("id", "call_eval"),
                        }
                    )

                    # Check recovery: model should either call recovery tool or explain
                    content2, tc2, _, elapsed2 = stream_chat(
                        host,
                        port,
                        messages,
                        tools=TOOLS,
                        max_tokens=512,
                        temperature=0.0,
                    )

                    recovery_tool = sc.get("recovery_expected_tool")
                    if recovery_tool:
                        # Check if model called the recovery tool
                        recovery_ok = any(
                            t.get("function", {}).get("name") == recovery_tool
                            for t in tc2
                        )
                        text_recovery = bool(content2 and content2.strip())
                        ok = recovery_ok or text_recovery
                        result["recovery_tool_called"] = recovery_ok
                    else:
                        # Just expect a text explanation
                        ok = bool(content2 and content2.strip())

                    result["recovery_text"] = bool(content2 and content2.strip())
                    result["fully_correct"] = ok
                else:
                    # Model didn't call the first tool — not error recovery
                    ok = False
                    result["fully_correct"] = False

                if ok:
                    passed += 1
                    print("PASS (recovered)")
                else:
                    print("FAIL (no recovery)")
                details.append(result)
                continue

            # ── Standard / Sequential: existing logic ──
            content, tool_calls, ttft, elapsed = stream_chat(
                host, port, messages, tools=TOOLS, max_tokens=512, temperature=0.0
            )

            grade = _check_tool_call(tool_calls, sc)
            first_ok = (
                grade["tool_detected"]
                and grade["correct_name"]
                and grade["valid_json_args"]
                and grade["arg_score"] >= 0.5
            )

            result.update(grade)
            result["elapsed_s"] = round(elapsed, 2)
            steps_passed = [first_ok]

            # --- Followup rounds (sequential scenarios) ---
            followup_prefixes = ["followup_", "followup2_"]
            if first_ok and tool_calls:
                # Feed fake_result back for the first tool call
                tc = tool_calls[0]
                fn = tc.get("function", {})
                fake_result = sc.get(
                    "fake_result", f"Tool {fn.get('name', '?')} executed successfully."
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", "call_eval"),
                                "type": "function",
                                "function": fn,
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": fake_result,
                        "tool_call_id": tc.get("id", "call_eval"),
                    }
                )

                # Check each followup step
                for prefix in followup_prefixes:
                    expected_key = f"{prefix}expected_tool"
                    if expected_key not in sc:
                        break

                    content2, tc2, _, elapsed2 = stream_chat(
                        host,
                        port,
                        messages,
                        tools=TOOLS,
                        max_tokens=512,
                        temperature=0.0,
                    )

                    fgrade = _check_tool_call(tc2, sc, step_prefix=prefix)
                    step_ok = fgrade["tool_detected"] and fgrade["correct_name"]

                    # Check followup args if specified
                    if step_ok and f"{prefix}expected_args" in sc and tc2:
                        fn_f = tc2[0].get("function", {})
                        try:
                            actual_args = json.loads(fn_f.get("arguments", "{}"))
                            match_mode = sc.get(f"{prefix}arg_match_mode", "fuzzy")
                            if match_mode == "exact":
                                arg_ok = actual_args == sc[f"{prefix}expected_args"]
                            elif match_mode == "contains":
                                arg_ok = all(
                                    str(v) in str(actual_args.get(k, ""))
                                    for k, v in sc[f"{prefix}expected_args"].items()
                                )
                            else:
                                arg_ok = (
                                    fuzzy_match_args(
                                        sc[f"{prefix}expected_args"], actual_args
                                    )
                                    >= 0.5
                                )
                            step_ok = step_ok and arg_ok
                        except (json.JSONDecodeError, TypeError):
                            step_ok = False

                    steps_passed.append(step_ok)

                    if step_ok and tc2:
                        # Feed this step's fake result back
                        tc_f = tc2[0]
                        fn_f = tc_f.get("function", {})
                        fake_key = f"{prefix}fake_result"
                        fake_r = sc.get(
                            fake_key,
                            f"Tool {fn_f.get('name', '?')} executed successfully.",
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tc_f.get("id", "call_eval"),
                                        "type": "function",
                                        "function": fn_f,
                                    }
                                ],
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "content": fake_r,
                                "tool_call_id": tc_f.get("id", "call_eval"),
                            }
                        )
                    else:
                        break  # stop if a followup step fails

            fully_correct = all(steps_passed)
            result["fully_correct"] = fully_correct
            result["steps_passed"] = sum(steps_passed)
            result["steps_total"] = len(steps_passed)

            if fully_correct:
                passed += 1
                label = f"PASS ({len(steps_passed)} step{'s' if len(steps_passed) > 1 else ''})"
                print(label)
            else:
                reasons = []
                if not grade["tool_detected"]:
                    reasons.append("no tool call")
                elif not grade["correct_name"]:
                    tc_name = (
                        tool_calls[0].get("function", {}).get("name", "?")
                        if tool_calls
                        else "?"
                    )
                    reasons.append(f"wrong tool ({tc_name})")
                elif not grade["valid_json_args"]:
                    reasons.append("invalid JSON args")
                elif grade["arg_score"] < 0.5:
                    reasons.append(f"low arg match ({grade['arg_score']:.1f})")
                elif not all(steps_passed):
                    failed_step = steps_passed.index(False) + 1
                    reasons.append(f"step {failed_step} failed")
                print(f"FAIL ({', '.join(reasons)})")

        except Exception as e:
            result.update(
                {
                    "tool_detected": False,
                    "correct_name": False,
                    "valid_json_args": False,
                    "arg_score": 0.0,
                    "fully_correct": False,
                    "error": str(e),
                }
            )
            print(f"ERROR ({e})")

        details.append(result)

    score = passed / len(scenarios)
    print(f"  Score: {passed}/{len(scenarios)} = {score:.0%}")
    return {
        "score": round(score, 2),
        "passed": passed,
        "total": len(scenarios),
        "details": details,
    }


# =============================================================================
# Suite C: Coding
# =============================================================================


def extract_python_code(text: str) -> str:
    """Extract Python code from model response (handles markdown fences)."""
    # Try to find fenced code block
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no fence, try to find function/class definitions
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)
    # Last resort: return everything
    return text


def run_coding_suite(host: str, port: int, verbose: bool = False) -> dict:
    """Run 10 coding tasks, auto-grade by executing test code."""
    # TODO: MiniMax-M2.5 scores 10% coding despite 87% tools / 80% reasoning / 90% general.
    #       Likely a code extraction or formatting issue — investigate response format.
    print("\n--- Suite C: Coding ---")

    prompts_file = PROMPTS_DIR / "coding.json"
    tasks = json.loads(prompts_file.read_text())

    details = []
    passed = 0

    for task in tasks:
        tid = task["id"]
        print(f"  {tid}: {task['description']}...", end=" ", flush=True)
        result = {"id": tid, "description": task["description"]}

        try:
            resp = chat_request(
                host,
                port,
                [{"role": "user", "content": task["prompt"]}],
                max_tokens=4096,
                temperature=0.0,
                enable_thinking=False,
            )
            output = _strip_thinking(resp["choices"][0]["message"]["content"])
            code = extract_python_code(output)

            # Write code + test to temp file and run
            full_code = code + "\n\n" + task["test_code"]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                tmp_path = f.name

            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                runs_ok = proc.returncode == 0
                correct = "PASS" in proc.stdout
                error_msg = proc.stderr.strip() if not runs_ok else None
            except subprocess.TimeoutExpired:
                runs_ok = False
                correct = False
                error_msg = "Timeout (10s)"
            finally:
                os.unlink(tmp_path)

            result.update(
                {
                    "runs_without_error": runs_ok,
                    "correct_output": correct,
                    "fully_correct": runs_ok and correct,
                }
            )
            if error_msg and verbose:
                result["error"] = error_msg[:200]

            if runs_ok and correct:
                passed += 1
                print("PASS")
            elif runs_ok:
                print("FAIL (wrong output)")
            else:
                print("FAIL (runtime error)")
                if verbose and error_msg:
                    print(f"        {error_msg[:120]}")

        except Exception as e:
            result.update(
                {
                    "runs_without_error": False,
                    "correct_output": False,
                    "fully_correct": False,
                    "error": str(e),
                }
            )
            print(f"ERROR ({e})")

        details.append(result)

    score = passed / len(tasks)
    print(f"  Score: {passed}/{len(tasks)} = {score:.0%}")
    return {
        "score": round(score, 2),
        "passed": passed,
        "total": len(tasks),
        "details": details,
    }


# =============================================================================
# Suite D: Reasoning (MATH-500)
# =============================================================================


def extract_answer(text: str):
    """Extract numerical answer from model response. Supports integers, fractions, and LaTeX."""
    # Priority 1: #### marker (with optional fraction or number)
    m = re.search(r"####\s*\\frac\{(\d+)\}\{(\d+)\}", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"####\s*\$?(\d+(?:/\d+)?(?:,\d+)*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")

    # Priority 2: "answer is ..." patterns
    m = re.search(r"answer is[:\s]+\\frac\{(\d+)\}\{(\d+)\}", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(
        r"answer is[:\s]+\$?(\d+(?:/\d+)?(?:,\d+)*(?:\.\d+)?)", text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "")

    # Priority 3: boxed LaTeX \boxed{\frac{a}{b}} or \boxed{N}
    m = re.search(r"\\boxed\{\\frac\{(\d+)\}\{(\d+)\}\}", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"\\boxed\{(\d+(?:/\d+)?)\}", text)
    if m:
        return m.group(1)

    # Priority 4: LaTeX \frac{a}{b} (last occurrence)
    fracs = re.findall(r"\\frac\{(\d+)\}\{(\d+)\}", text)
    if fracs:
        a, b = fracs[-1]
        return f"{a}/{b}"

    # Priority 5: plain fraction a/b (last occurrence)
    frac_matches = re.findall(r"(\d+/\d+)", text)
    if frac_matches:
        return frac_matches[-1]

    # Priority 6: plain number at end of line
    patterns = [
        r"=\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)\s*$",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")

    # Fallback: last number in text
    numbers = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def normalize_answer(answer: str) -> Fraction | None:
    """Normalize answer to a Fraction for exact comparison."""
    if answer is None:
        return None
    answer = answer.replace(",", "").replace("$", "").strip()
    try:
        if "/" in answer:
            parts = answer.split("/")
            return Fraction(int(parts[0]), int(parts[1]))
        num = float(answer)
        return Fraction(num).limit_denominator(10000)
    except (ValueError, ZeroDivisionError):
        return None


def run_reasoning_suite(host: str, port: int, verbose: bool = False) -> dict:
    """Run 10 MATH-500 problems."""
    print("\n--- Suite D: Reasoning (MATH-500) ---")

    prompts_file = PROMPTS_DIR / "reasoning.json"
    problems = json.loads(prompts_file.read_text())

    details = []
    passed = 0

    for prob in problems:
        pid = prob["id"]
        print(f"  {pid}...", end=" ", flush=True)

        prompt = (
            "Solve this math problem step by step. "
            'At the end, write your final answer after "####". '
            "If the answer is a fraction, write it as a/b.\n\n"
            f"Problem: {prob['question']}\n\nSolution:"
        )

        try:
            resp = chat_request(
                host,
                port,
                [
                    {
                        "role": "system",
                        "content": "You are a math tutor. Solve problems step by step, then give the final answer after ####. Use fractions when appropriate.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.0,
                enable_thinking=False,
            )
            output = _strip_thinking(resp["choices"][0]["message"]["content"])
            extracted = extract_answer(output)
            expected = normalize_answer(prob["answer"])
            got = normalize_answer(extracted)
            correct = got is not None and expected is not None and got == expected

            result = {
                "id": pid,
                "expected": str(expected),
                "got": str(got),
                "correct": correct,
            }

            if correct:
                passed += 1
                print(f"PASS (={expected})")
            else:
                print(f"FAIL (expected={expected}, got={got})")

        except Exception as e:
            result = {
                "id": pid,
                "expected": prob["answer"],
                "got": None,
                "correct": False,
                "error": str(e),
            }
            print(f"ERROR ({e})")

        details.append(result)

    score = passed / len(problems)
    print(f"  Score: {passed}/{len(problems)} = {score:.0%}")
    return {
        "score": round(score, 2),
        "passed": passed,
        "total": len(problems),
        "details": details,
    }


# =============================================================================
# Suite E: General Knowledge / Instruction Following
# =============================================================================


def _strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks from model output.

    Handles:
    - <think>...</think> tags (Qwen3 style)
    - "Thinking Process:..." preamble (Qwen3.5 style — outputs thinking as plain text)

    Strategy: if the output starts with thinking patterns, try to find the
    actual answer. If not, return as-is.
    """
    # Strip <think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # If the output doesn't start with thinking patterns, return as-is
    if not re.match(
        r"^(?:Thinking Process|##?\s*Thinking|Let me think|The user\b|\d+\.\s+\*\*Analy)",
        text,
        re.IGNORECASE,
    ):
        return text

    # Look for explicit answer delimiters (last match wins)
    answer_patterns = [
        r"\n---\n",
        r"\n\*\*Answer:?\*\*:?\s*\n",
        r"\n\*\*Response:?\*\*:?\s*\n",
        r"\n\*\*Final Answer:?\*\*:?\s*\n",
        r"\n\*\*Output:?\*\*:?\s*\n",
        r"\n## Answer\s*\n",
        r"\n## Response\s*\n",
        r"\n## Final Answer\s*\n",
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            text = text[matches[-1].end() :]
            return text.strip()

    # Qwen3.5 "Thinking Process:" pattern — these models have numbered
    # sections like "1. **Analyze:**", "4. **Final Version:**", etc.
    # Find ALL "Final" sections and take the content after the last one.
    final_sections = list(
        re.finditer(
            r"\d+\.\s+\*\*Final[^*]*\*\*[:\s]*(?:\([^)]*\)\s*)?\.?\n",
            text,
            re.IGNORECASE,
        )
    )
    if final_sections:
        # Try the last "Final" section first; if too short, try second-to-last
        for idx in range(len(final_sections) - 1, -1, -1):
            candidate = text[final_sections[idx].end() :].strip()
            # Remove any subsequent numbered "Final" sections (verification, etc.)
            if idx < len(final_sections) - 1:
                # Only keep up to the next final section
                next_start = final_sections[idx + 1].start() - final_sections[idx].end()
                candidate = candidate[:next_start].strip()
            else:
                # Last section — remove trailing thinking artifacts
                candidate = re.sub(
                    r"\n\s*\d+\.\s+\*\*(?:Final |Verif|Count|Check).*$",
                    "",
                    candidate,
                    flags=re.DOTALL,
                )
            candidate = re.sub(
                r"\n\s*\*(?:Count|Verification|Check|Wait|Revised)[:\*,].*$",
                "",
                candidate,
                flags=re.DOTALL,
            )
            candidate = re.sub(
                r"^\s*\*(?:Wait|Revised|Note)[^*]*\*\s*\n",
                "",
                candidate,
                flags=re.MULTILINE,
            )
            # Accept if substantial enough (>40 chars to avoid picking up tiny fragments)
            if len(candidate.strip()) > 40:
                return candidate.strip()

    # Last resort: return original
    return text.strip()


def _extract_answer_letter(text: str) -> str | None:
    """Extract MCQ answer letter (A-J) from model response.

    Strategy: search the full text for unambiguous patterns first (explicit
    "answer is X", standalone letter on its own line).  For ambiguous patterns
    (parenthesised letters, bare letters) only search the tail of the response
    to avoid matching option re-listings that verbose models emit.
    """
    # Priority 1: explicit "answer is X" / "answer: X" (last match wins)
    matches = list(
        re.finditer(
            r"answer\s+is\s*[:\s]*\(?([A-Ja-j])\)?",
            text,
            re.IGNORECASE,
        )
    )
    if matches:
        return matches[-1].group(1).upper()

    # Priority 2: standalone letter on its own line (last match)
    matches = list(re.finditer(r"^\s*([A-Ja-j])\s*$", text, re.MULTILINE))
    if matches:
        return matches[-1].group(1).upper()

    # For remaining (ambiguous) patterns, only search the last 300 chars
    # to avoid picking up letters from option re-listings.
    tail = text[-500:] if len(text) > 500 else text

    # Priority 3: letter followed by period/closing-paren at start of line
    matches = list(
        re.finditer(
            r"^\s*([A-Ja-j])[\.\)]",
            tail,
            re.MULTILINE,
        )
    )
    if matches:
        return matches[-1].group(1).upper()

    # Priority 4: parenthesised letter "(X)" in the tail
    matches = list(re.finditer(r"\(([A-Ja-j])\)", tail))
    if matches:
        return matches[-1].group(1).upper()

    # Priority 5: last letter A-J in tail preceded by word boundary,
    # excluding the pronoun "I"/"i" when followed by common verb patterns.
    candidates = list(re.finditer(r"\b([A-Ja-j])\b", tail))
    for c in reversed(candidates):
        letter = c.group(1)
        if letter.upper() == "I":
            after = tail[c.end() : c.end() + 15].lstrip()
            if re.match(
                r"(?:think|believe|choose|would|will|am|'m|'d|'ll)\b",
                after,
                re.IGNORECASE,
            ):
                continue
        return letter.upper()

    return None


def check_general_response(response: str, checks: dict) -> tuple[bool, str]:
    """Check a general response against its checks. Returns (pass, reason)."""
    text = _strip_thinking(response).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    if "min_length" in checks and len(text) < checks["min_length"]:
        return False, f"too short ({len(text)} < {checks['min_length']})"

    if "max_length" in checks and len(text) > checks["max_length"]:
        return False, f"too long ({len(text)} > {checks['max_length']})"

    if "contains_all" in checks:
        for word in checks["contains_all"]:
            if word.lower() not in text.lower():
                return False, f"missing '{word}'"

    if "contains_any" in checks:
        found = any(w.lower() in text.lower() for w in checks["contains_any"])
        if not found:
            return False, f"none of {checks['contains_any']} found"

    if "ordered_contains" in checks:
        words = checks["ordered_contains"]
        last_pos = -1
        for word in words:
            pos = text.lower().find(word.lower())
            if pos == -1:
                return False, f"missing '{word}' for ordered check"
            if pos <= last_pos:
                return False, f"'{word}' out of order"
            last_pos = pos

    if "valid_json" in checks and checks["valid_json"]:
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if "json_has_keys" in checks:
                    for key in checks["json_has_keys"]:
                        if key not in parsed:
                            return False, f"JSON missing key '{key}'"
            else:
                return False, "no JSON object found"
        except json.JSONDecodeError:
            return False, "invalid JSON"

    if "sentence_count" in checks:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) != checks["sentence_count"]:
            return (
                False,
                f"sentence count {len(sentences)} != {checks['sentence_count']}",
            )

    if "max_sentences" in checks:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) > checks["max_sentences"]:
            return (
                False,
                f"too many sentences ({len(sentences)} > {checks['max_sentences']})",
            )

    # Check numbered items (e.g. "1. ..." or "1) ...")
    if "min_items" in checks or "max_items" in checks:
        numbered = [l for l in lines if re.match(r"^\d+[.)]\s", l)]
        count = len(numbered)
        if "min_items" in checks and count < checks["min_items"]:
            return False, f"too few items ({count} < {checks['min_items']})"
        if "max_items" in checks and count > checks["max_items"]:
            return False, f"too many items ({count} > {checks['max_items']})"

    if "line_count_range" in checks:
        lo, hi = checks["line_count_range"]
        if not (lo <= len(lines) <= hi):
            return False, f"line count {len(lines)} not in [{lo}, {hi}]"

    if "contains_none" in checks:
        for word in checks["contains_none"]:
            if word.lower() in text.lower():
                return False, f"should not contain '{word}'"

    if "answer_letter" in checks:
        expected_letter = checks["answer_letter"]
        got_letter = _extract_answer_letter(text)
        if got_letter != expected_letter:
            return False, f"expected {expected_letter}, got {got_letter or 'none'}"

    return True, "ok"


def run_general_suite(host: str, port: int, verbose: bool = False) -> dict:
    """Run 10 general knowledge / instruction following tasks."""
    # TODO: GLM-4.7-Flash scores 50% general despite 100% coding / 90% reasoning.
    #       May struggle with MMLU-Pro 10-option multiple choice format — check answer extraction.
    print("\n--- Suite E: General Knowledge ---")

    prompts_file = PROMPTS_DIR / "general.json"
    tasks = json.loads(prompts_file.read_text())

    details = []
    passed = 0

    # System prompt for MMLU-Pro multiple choice questions
    no_think_sys = {
        "role": "system",
        "content": "You are taking a multiple choice test. Read the question and options carefully, then respond with just the letter of your answer. Do not show your thinking process.",
    }

    for task in tasks:
        tid = task["id"]
        print(f"  {tid}: {task['description']}...", end=" ", flush=True)

        try:
            resp = chat_request(
                host,
                port,
                [no_think_sys, {"role": "user", "content": task["prompt"]}],
                max_tokens=2048,
                temperature=0.0,
                enable_thinking=False,
            )
            output = _strip_thinking(resp["choices"][0]["message"]["content"])
            ok, reason = check_general_response(output, task.get("checks", {}))

            result = {
                "id": tid,
                "description": task["description"],
                "correct": ok,
                "reason": reason,
            }

            if ok:
                passed += 1
                print("PASS")
            else:
                print(f"FAIL ({reason})")

        except Exception as e:
            result = {
                "id": tid,
                "description": task["description"],
                "correct": False,
                "reason": str(e),
            }
            print(f"ERROR ({e})")

        details.append(result)

    score = passed / len(tasks)
    print(f"  Score: {passed}/{len(tasks)} = {score:.0%}")
    return {
        "score": round(score, 2),
        "passed": passed,
        "total": len(tasks),
        "details": details,
    }


# =============================================================================
# Main
# =============================================================================

ALL_SUITES = ["speed", "tool_calling", "coding", "reasoning", "general"]


def main():
    parser = argparse.ArgumentParser(
        description="vllm-mlx Model Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evals/run_eval.py --model "Qwen3.5-122B-mxfp4" --port 8000
    python evals/run_eval.py --model "GPT-OSS-20B-4bit" --suite tool_calling coding
    python evals/run_eval.py --model "GLM-4.7-Flash" --parser glm47 --verbose
        """,
    )
    parser.add_argument("--model", required=True, help="Model display name for results")
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--parser", default=None, help="Tool parser name (e.g. hermes, glm47, harmony)"
    )
    parser.add_argument(
        "--quantization", default=None, help="Quantization label (e.g. 4bit, mxfp4)"
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        default=ALL_SUITES,
        choices=ALL_SUITES,
        help="Which suites to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file (default: evals/results/<model>.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="Hardware description override (e.g. 'Mac Studio M3 Ultra 256GB')",
    )
    parser.add_argument(
        "--server-flags",
        default=None,
        help="Server flags used (e.g. '--enable-auto-tool-choice --tool-call-parser harmony')",
    )
    parser.add_argument(
        "--model-path", default=None, help="Local model path (for reproducibility)"
    )
    parser.add_argument(
        "--engine",
        default="simple",
        choices=["simple", "batched"],
        help="Engine mode (default: simple)",
    )
    parser.add_argument(
        "--notes", default=None, help="Free-form notes about this eval run"
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
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Check server
    if not server_available(args.host, args.port):
        print(f"ERROR: No vllm-mlx server at http://{args.host}:{args.port}")
        print("Start one with: vllm-mlx serve <model> --port 8000")
        sys.exit(1)

    # Detect hardware
    hw = detect_hardware()
    hw_label = args.hardware or f"{hw['chip']} ({hw['memory_gb']}GB)"

    print("=" * 60)
    print("vllm-mlx Model Evaluation")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Hardware: {hw_label}")
    print(f"  Server:   http://{args.host}:{args.port}")
    print(f"  Parser:   {args.parser or 'auto'}")
    print(f"  Suites:   {', '.join(args.suite)}")
    print(f"  Date:     {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    print("=" * 60)

    # Build result object
    result = {
        "model": args.model,
        "quantization": args.quantization,
        "parser": args.parser or "auto",
        "hardware": hw_label,
        "hardware_detail": hw,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "server": f"http://{args.host}:{args.port}",
        "server_flags": args.server_flags,
        "model_path": args.model_path,
        "engine": args.engine,
        "notes": args.notes,
    }

    start_time = time.perf_counter()

    def _bust_cache(host, port):
        """Clear server prompt cache between suites."""
        try:
            url = f"http://{host}:{port}/v1/cache/clear"
            if _HTTPX:
                httpx.post(url, timeout=10)
            else:
                import urllib.request

                req = urllib.request.Request(url, method="POST")
                urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    # Run selected suites
    if "speed" in args.suite:
        result["speed"] = run_speed_suite(args.host, args.port, verbose=args.verbose)

    if "tool_calling" in args.suite:
        _bust_cache(args.host, args.port)
        result["tool_calling"] = run_tool_calling_suite(
            args.host, args.port, verbose=args.verbose
        )

    if "coding" in args.suite:
        _bust_cache(args.host, args.port)
        result["coding"] = run_coding_suite(args.host, args.port, verbose=args.verbose)

    if "reasoning" in args.suite:
        _bust_cache(args.host, args.port)
        result["reasoning"] = run_reasoning_suite(
            args.host, args.port, verbose=args.verbose
        )

    if "general" in args.suite:
        _bust_cache(args.host, args.port)
        result["general"] = run_general_suite(
            args.host, args.port, verbose=args.verbose
        )

    total_time = time.perf_counter() - start_time
    result["total_eval_time_s"] = round(total_time, 1)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Hardware:   {hw_label}")
    print(f"  Total time: {total_time:.0f}s")
    print()

    for suite_name in ["speed", "tool_calling", "coding", "reasoning", "general"]:
        if suite_name in result:
            suite_data = result[suite_name]
            if "score" in suite_data:
                print(
                    f"  {suite_name:15s} {suite_data['score']:.0%} ({suite_data['passed']}/{suite_data['total']})"
                )
            elif "_summary" in suite_data:
                print(f"  {suite_name:15s} {suite_data['_summary']}")

    print("=" * 60)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        # Sanitize model name for filename
        safe_name = re.sub(r"[^\w\-.]", "-", args.model.lower()).strip("-")
        out_path = RESULTS_DIR / f"{safe_name}.json"

    # Remove internal keys before saving
    save_result = {k: v for k, v in result.items()}
    if "speed" in save_result and "_summary" in save_result["speed"]:
        save_result["speed"] = {
            k: v for k, v in save_result["speed"].items() if not k.startswith("_")
        }

    out_path.write_text(json.dumps(save_result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
