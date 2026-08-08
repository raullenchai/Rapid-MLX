#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L1 small-model tool-loop contract check.

This guards the tool-call *plumbing*, not the small model's discretion to call
a tool. A named ``tool_choice`` forces the call via a grammar-constrained
forced-function prefix (see ``vllm_mlx/routes/chat.py``), so
the result is deterministic at temperature 0 — the check measures whether the
server emits a structurally valid ``tool_calls`` entry and can render that call
plus its ``role=tool`` result on the next turn in both non-streaming and
streaming modes.  The replay catches chat-template shape regressions that only
appear after a successful call (for example #1676's DeepSeek-R1 ``str + dict``
500), while deliberately making no assertion about the small model's answer.

Companion to ``evals/coherence_gate.py``; both are driven by
``scripts/l1_smoke.sh`` on the free macos-14 L1 runner. Requires a
``rapid-mlx serve`` already listening — it does not boot one.

Exit codes:
    0 — forced call plus streaming/non-streaming tool-result replay succeeded
    1 — malformed call, replay failure, or the server was unreachable
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

_TOOL = {
    "type": "function",
    "function": {
        "name": "release_probe",
        "description": "Return a fixed release-gate marker.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

_WIRE_MARKERS = (
    "<tool_call",
    "<toolcall",
    "<｜tool▁call",
    "<｜tool▁sep｜>",
    "[TOOL_CALLS]",
    "<think>",
    "</think>",
)


def _visible_wire_marker(text: str) -> str | None:
    return next((marker for marker in _WIRE_MARKERS if marker in text), None)


def _error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"; body={exc.response.text[:500]}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8123/v1")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "Call release_probe now."}],
        "tools": [_TOOL],
        # A named zero-argument call factors model competence out completely:
        # the engine need not invent a required value before it can prove the
        # parser/template contract.
        "tool_choice": {
            "type": "function",
            "function": {"name": "release_probe"},
        },
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
    if name != "release_probe":
        print(f"FAIL: tool name {name!r} != 'release_probe'", file=sys.stderr)
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

    visible = f"{msg.get('content') or ''}\n{msg.get('reasoning_content') or ''}"
    leaked = _visible_wire_marker(visible)
    if leaked:
        print(
            f"FAIL: native wire marker leaked into non-stream content: {leaked!r}",
            file=sys.stderr,
        )
        return 1

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    try:
        with httpx.stream(
            "POST", endpoint, json={**body, "stream": True}, timeout=args.timeout
        ) as forced_stream:
            forced_stream.raise_for_status()
            forced_lines = [
                line for line in forced_stream.iter_lines() if line.startswith("data:")
            ]
        forced_wire = "\n".join(forced_lines)
        if '"tool_calls"' not in forced_wire:
            raise ValueError("forced stream returned no tool_calls delta")
        if not forced_lines or forced_lines[-1].strip() != "data: [DONE]":
            raise ValueError("forced stream did not terminate with data: [DONE]")
        leaked = _visible_wire_marker(forced_wire)
        if leaked:
            raise ValueError(f"native wire marker leaked into stream: {leaked!r}")
    except Exception as exc:
        print(
            f"FAIL: streaming forced-call error: {exc}{_error_detail(exc)}",
            file=sys.stderr,
        )
        return 1

    # Replay the exact OpenAI-wire assistant message returned by the server.
    # This is the important second half of the contract: templates differ on
    # whether historical arguments must be a mapping or JSON text, and the
    # first turn cannot expose that mismatch.
    call_id = (calls[0] or {}).get("id")
    if not isinstance(call_id, str) or not call_id:
        print(f"FAIL: tool call has no string id: {calls[0]!r}", file=sys.stderr)
        return 1
    replay_messages = [
        body["messages"][0],
        msg,
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "release_probe",
            "content": "RELEASE_PROBE_OK",
        },
        {"role": "user", "content": "Acknowledge the completed probe."},
    ]

    replay = {
        "model": "default",
        "messages": replay_messages,
        "tools": [_TOOL],
        "tool_choice": "auto",
        "max_tokens": 32,
        "temperature": 0.0,
    }
    try:
        nonstream = httpx.post(
            endpoint, json={**replay, "stream": False}, timeout=args.timeout
        )
        nonstream.raise_for_status()
        nonstream_data = nonstream.json()
        if not (nonstream_data.get("choices") or []):
            raise ValueError("non-stream replay returned no choices")

        with httpx.stream(
            "POST",
            endpoint,
            json={**replay, "stream": True},
            timeout=args.timeout,
        ) as stream_response:
            stream_response.raise_for_status()
            stream_lines = [
                line
                for line in stream_response.iter_lines()
                if line.startswith("data:")
            ]
        if not stream_lines or stream_lines[-1].strip() != "data: [DONE]":
            raise ValueError("stream replay did not terminate with data: [DONE]")
    except Exception as exc:
        print(
            f"FAIL: tool-result replay error: {exc}{_error_detail(exc)}",
            file=sys.stderr,
        )
        return 1

    print(
        "PASS: forced release_probe tool_call + tool-result replay "
        "(non-stream + stream)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
