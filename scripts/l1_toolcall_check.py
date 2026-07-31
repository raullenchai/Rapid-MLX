#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L1 tool-call FORMAT check — assert a forced ``tool_choice`` yields a
well-formed tool call.

This guards the tool-call *plumbing*, not the small model's discretion to call
a tool. ``tool_choice="required"`` with a single tool forces the call via a
grammar-constrained forced-function prefix (see ``vllm_mlx/routes/chat.py``), so
the result is deterministic at temperature 0 — the check measures whether the
server emits a structurally valid ``tool_calls`` entry (function name present,
arguments parseable as a JSON object), the class of regression where a model
emits a call but the parser mangles it into free text or breaks the JSON args.

Companion to ``evals/coherence_gate.py``; both are driven by
``scripts/l1_smoke.sh`` on the free macos-14 L1 runner. Requires a
``rapid-mlx serve`` already listening — it does not boot one.

Exit codes:
    0 — a well-formed forced tool call came back
    1 — no/malformed tool call, or the server was unreachable
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Paris"}
            },
            "required": ["location"],
        },
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8123/v1")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    body = {
        "model": "default",
        "messages": [
            {"role": "user", "content": "What is the weather in Paris? Use the tool."}
        ],
        "tools": [_TOOL],
        # Single tool + "required" => forced call, grammar-constrained. Makes the
        # FORMAT deterministic regardless of a small model's tool-calling mood.
        "tool_choice": "required",
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        r = httpx.post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            json=body,
            timeout=args.timeout,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"FAIL: request error: {exc}", file=sys.stderr)
        return 1

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        print(
            f"FAIL: malformed response ({exc}): {json.dumps(data)[:300]}",
            file=sys.stderr,
        )
        return 1

    calls = msg.get("tool_calls") or []
    if not calls:
        content = (msg.get("content") or "")[:200]
        print(
            "FAIL: forced tool_choice produced NO tool_calls "
            f"(the call likely leaked into text): content={content!r}",
            file=sys.stderr,
        )
        return 1

    fn = (calls[0] or {}).get("function") or {}
    name = fn.get("name")
    if name != "get_weather":
        print(f"FAIL: tool name {name!r} != 'get_weather'", file=sys.stderr)
        return 1

    raw_args = fn.get("arguments")
    # OpenAI convention: arguments is a JSON *string*. Accept a pre-parsed dict
    # too so the check asserts structure, not serialization style.
    if isinstance(raw_args, dict):
        parsed = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            print(
                f"FAIL: arguments not valid JSON ({exc}): {raw_args!r}", file=sys.stderr
            )
            return 1
    else:
        print(
            f"FAIL: arguments must be a JSON string or object, got "
            f"{type(raw_args).__name__}",
            file=sys.stderr,
        )
        return 1

    if not isinstance(parsed, dict):
        print(
            f"FAIL: arguments JSON is {type(parsed).__name__}, expected an object",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: well-formed forced tool_call get_weather(arguments={parsed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
