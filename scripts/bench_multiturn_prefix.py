#!/usr/bin/env python3
"""Measure ten agent turns over OpenAI and Anthropic streaming routes.

Run each route against a fresh server/cache, then compare the per-turn JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

REPOSITORY = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a coding assistant working on a Python package. Keep answers under "
    "two sentences and use the available repository tools when useful.\n\n"
    "Repository guidance:\n"
    + (REPOSITORY / "AGENTS.md").read_text()
    + "\nProject overview:\n"
    + (REPOSITORY / "README.md").read_text()[:25000]
)
TOOL_NAMES = (
    "read_file",
    "search",
    "list_files",
    "run_tests",
    "edit_file",
    "git_diff",
    "read_config",
    "read_test",
    "read_docs",
    "find_symbol",
    "find_references",
    "inspect_imports",
    "inspect_types",
    "inspect_errors",
    "inspect_logs",
    "list_branches",
    "git_status",
    "git_show",
    "git_blame",
    "run_lint",
    "run_format",
    "run_typecheck",
    "run_smoke",
    "write_test",
    "write_docs",
    "compare_files",
    "inspect_dependencies",
    "inspect_routes",
    "inspect_cache",
    "summarize_change",
)
TOOL_DESC = (
    "Use this repository operation to inspect or modify the package. "
    "Return precise paths and short results; preserve existing behavior. " * 3
)
QUESTIONS = (
    "Find the function that formats cache usage in the API response.",
    "Which tests cover that usage field?",
    "Summarize the expected result of a warm second request.",
    "Check whether the tool reply format changes that result.",
    "Identify a small regression test for that case.",
    "What should the test assert about token prefixes?",
    "Review the proposed assertion for a false positive.",
    "Name the focused test command.",
    "Summarize any remaining risk in one sentence.",
    "Give a concise final report.",
)


def _tools(api: str) -> list[dict]:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    if api == "chat":
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_DESC,
                    "parameters": schema,
                },
            }
            for name in TOOL_NAMES
        ]
    return [
        {"name": name, "description": TOOL_DESC, "input_schema": schema}
        for name in TOOL_NAMES
    ]


def _stream(
    client: httpx.Client, path: str, payload: dict
) -> tuple[float, dict, str, list[dict]]:
    started = time.perf_counter()
    first = None
    usage: dict = {}
    content = ""
    calls: dict[int, dict] = {}
    finished = False
    final_usage = False
    with client.stream("POST", path, json=payload, timeout=180) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line == "data: [DONE]":
                finished = True
                break
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("error") or event.get("type") == "error":
                raise RuntimeError(f"stream error: {event}")
            if path.endswith("chat/completions"):
                if event.get("usage") is not None:
                    usage = event["usage"]
                    final_usage = True
                delta = (event.get("choices") or [{}])[0].get("delta") or {}
                fragment = delta.get("content") or ""
                if fragment and first is None:
                    first = time.perf_counter() - started
                content += fragment
                for call in delta.get("tool_calls") or []:
                    if first is None:
                        first = time.perf_counter() - started
                    slot = calls.setdefault(
                        call["index"],
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    slot["id"] += call.get("id") or ""
                    function = call.get("function") or {}
                    slot["function"]["name"] += function.get("name") or ""
                    slot["function"]["arguments"] += function.get("arguments") or ""
            else:
                event_type = event.get("type")
                if event_type == "message_stop":
                    finished = True
                if event_type == "message_start":
                    usage = event.get("message", {}).get("usage") or usage
                elif event_type == "message_delta":
                    if event.get("usage") is not None:
                        usage.update(event["usage"])
                        final_usage = True
                elif event_type == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        calls[event["index"]] = {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": "",
                        }
                        if first is None:
                            first = time.perf_counter() - started
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    fragment = delta.get("text") or ""
                    if fragment and first is None:
                        first = time.perf_counter() - started
                    content += fragment
                    if delta.get("type") == "input_json_delta":
                        calls[event["index"]]["input"] += (
                            delta.get("partial_json") or ""
                        )
    if not finished:
        raise RuntimeError(f"stream ended without terminal event: {path}")
    if not final_usage:
        raise RuntimeError(f"stream ended without terminal usage: {path}")
    return first or time.perf_counter() - started, usage, content, list(calls.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api", choices=("chat", "messages"), required=True)
    args = parser.parse_args()
    client = httpx.Client(base_url=args.base_url)
    tools = _tools(args.api)
    messages = [{"role": "system", "content": SYSTEM}] if args.api == "chat" else []
    for turn, question in enumerate(QUESTIONS, 1):
        messages.append({"role": "user", "content": question})
        payload = {
            "model": args.model,
            "messages": messages,
            "tools": tools,
            "max_tokens": 64,
            "temperature": 0,
            "stream": True,
        }
        if args.api == "chat":
            payload["stream_options"] = {"include_usage": True}
            path = "/v1/chat/completions"
        else:
            payload["system"] = SYSTEM
            path = "/v1/messages"
        ttft, usage, content, calls = _stream(client, path, payload)
        if args.api == "chat":
            assistant = {"role": "assistant", "content": content or None}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            for call in calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "Found relevant source and tests.",
                    }
                )
            prompt = usage.get("prompt_tokens", 0)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        else:
            blocks = [{"type": "text", "text": content}] if content else []
            for call in calls:
                value = json.loads(call["input"] or "{}")
                blocks.append({**call, "input": value})
            messages.append({"role": "assistant", "content": blocks})
            if calls:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": "Found relevant source and tests.",
                            }
                            for call in calls
                        ],
                    }
                )
            cached = usage.get("cache_read_input_tokens", 0)
            prompt = usage.get("input_tokens", 0) + cached
        if not prompt or not (content or calls):
            raise RuntimeError(f"incomplete turn {turn}: usage={usage}, calls={calls}")
        print(
            json.dumps(
                {
                    "turn": turn,
                    "api": args.api,
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "ttft_s": round(ttft, 3),
                    "tool_calls": len(calls),
                    "content_chars": len(content),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
