# SPDX-License-Identifier: Apache-2.0
"""
End-to-end tool calling integration test.

Simulates a multi-round agent (like OpenClaw) that sends streaming requests
with 14 tools, executes tool results, and feeds them back.  Verifies that
the server correctly parses tool calls (including text-format fallback)
and returns structured SSE responses.

Usage:
    # Requires a running vllm-mlx server on localhost:8000
    python3.12 -m pytest tests/test_tool_call_e2e.py -v -s

    # Or run directly as a script for interactive debugging:
    python3.12 tests/test_tool_call_e2e.py ["custom prompt"]

Skip condition: Tests are skipped if no server is running on localhost:8000.
"""

import json
import subprocess
import sys
import time

import pytest

try:
    import httpx

    _HTTPX = True
except ImportError:
    _HTTPX = False

BASE_URL = "http://localhost:8000/v1/chat/completions"

SYSTEM_PROMPT = """You are Claw, a helpful AI assistant running on the user's local machine.

Current time: Thursday, February 26, 2026 8:40 PM PST
Location: Palo Alto, CA

## Rules
- Use exec or web_search for real-time info (weather, stocks, news).
- After getting tool results, give a FINAL TEXT ANSWER immediately.
- Do NOT call more tools unless absolutely necessary.
- Keep responses concise (under 200 words).
- If a command returns "still running", poll once then give best answer with available info.
"""

# 14 tools — same as OpenClaw's real tool set
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
            "description": "Process management: list/poll/log/write/kill/clear/remove",
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

MAX_ROUNDS = 8


def _server_available() -> bool:
    """Check if a vllm-mlx server on :8000 is reachable AND unauthenticated.

    The tests POST to ``/v1/chat/completions`` without any Authorization
    header. A server with an API key configured will respond 401 there,
    but its ``/health`` endpoint stays 200 (intentionally — health probes
    must not require auth). Probing ``/health`` alone therefore reports
    "available" against an auth-protected server, the tests then run,
    each request bounces off auth, ``run_agent_loop`` returns
    ``content=None``, and every test fails with ``AssertionError:
    Expected text response``.

    We additionally probe ``/v1/models`` — the surface the tests
    actually use — and require it to answer 200 unauthenticated. Any
    non-200 (401, 503, connection error) is treated as "no usable
    server" and the suite skips cleanly.
    """
    if not _HTTPX:
        return False
    try:
        h = httpx.get("http://localhost:8000/health", timeout=2.0)
        if h.status_code != 200:
            return False
        m = httpx.get("http://localhost:8000/v1/models", timeout=2.0)
        return m.status_code == 200
    except Exception:
        return False


def stream_request(messages):
    """Stream a request and return (content, tool_calls, raw_chunks, elapsed)."""
    content = ""
    tool_calls = []
    raw_chunks = []
    start = time.time()

    with httpx.stream(
        "POST",
        BASE_URL,
        json={
            "model": "default",
            "stream": True,
            "messages": messages,
            "tools": TOOLS,
        },
        timeout=120.0,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                raw_chunks.append("[DONE]")
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            raw_chunks.append(chunk)
            if "choices" not in chunk or not chunk["choices"]:
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            if "content" in delta and delta["content"]:
                content += delta["content"]
            if "tool_calls" in delta and delta["tool_calls"]:
                tool_calls.extend(delta["tool_calls"])

    elapsed = time.time() - start
    return content, tool_calls, raw_chunks, elapsed


def execute_tool(name, arguments):
    """Simulate tool execution with deterministic results."""
    args = json.loads(arguments) if isinstance(arguments, str) else arguments

    if name == "read":
        path = args.get("path", "")
        try:
            with open(path) as f:
                return f.read()[:2000]
        except Exception as e:
            return f"Error reading {path}: {e}"

    elif name == "exec":
        cmd = args.get("command", "")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            output = result.stdout + result.stderr
            return output[:2000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Command still running. Use process tool to check status."
        except Exception as e:
            return f"Error: {e}"

    elif name == "process":
        action = args.get("action", "")
        sid = args.get("sessionId", "unknown")
        if action == "poll":
            return "(no new output)\n\nProcess still running."
        elif action == "list":
            return "No active processes."
        return f"Process {action} on {sid}: OK"

    elif name == "web_search":
        query = args.get("query", "")
        return (
            f"Search results for '{query}': "
            "Palo Alto tonight: Clear skies, 54F (12C), wind 3mph NW, "
            "humidity 58%. Sunset was at 6:05 PM."
        )

    else:
        return f"Tool {name} executed with args: {json.dumps(args)}"


def run_agent_loop(user_msg, max_rounds=MAX_ROUNDS):
    """Run the full agent loop and return (rounds, final_content, tool_history).

    Returns:
        rounds: number of rounds completed
        final_content: the final text response, or None
        tool_history: list of (tool_name, tool_args) tuples
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Thu 2026-02-26 20:40 PST] {user_msg}"},
    ]
    tool_history = []

    for round_num in range(1, max_rounds + 1):
        content, tool_calls, raw_chunks, elapsed = stream_request(messages)

        if tool_calls:
            tc = tool_calls[0]
            fn = tc["function"]
            tool_history.append((fn["name"], fn["arguments"]))

            result = execute_tool(fn["name"], fn["arguments"])

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": fn}
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc["id"],
                }
            )
            continue

        if content:
            return round_num, content, tool_history

        # No content and no tool calls — check for error
        for c in raw_chunks:
            if isinstance(c, dict) and "error" in c:
                raise RuntimeError(f"Server error: {c['error']}")

        return round_num, None, tool_history

    return max_rounds, None, tool_history


# ---------------------------------------------------------------------------
# Pytest tests (skipped if server not running)
# ---------------------------------------------------------------------------

server_required = pytest.mark.skipif(
    not _server_available(),
    reason="vllm-mlx server not running on localhost:8000",
)


@server_required
class TestToolCallE2E:
    """End-to-end streaming tool call tests against a live server."""

    def test_simple_exec(self):
        """Model should call exec and return a text answer."""
        rounds, content, tools = run_agent_loop("帮我查看 ~/Desktop 下有什么文件")
        assert content is not None, "Expected text response"
        assert rounds <= 4, f"Should complete in <=4 rounds, got {rounds}"
        assert any(t[0] == "exec" for t in tools), "Should have called exec"

    def test_weather_with_fallback(self):
        """Model may call exec (curl), get 'still running', then web_search."""
        rounds, content, tools = run_agent_loop("今晚出去跑步合适吗")
        assert content is not None, "Expected text response"
        assert rounds <= 6, f"Should complete in <=6 rounds, got {rounds}"
        tool_names = [t[0] for t in tools]
        assert any(n in tool_names for n in ("exec", "web_search")), (
            f"Should use exec or web_search, got {tool_names}"
        )

    def test_no_tool_needed(self):
        """Pure reasoning should return text without tool calls."""
        rounds, content, tools = run_agent_loop("解释一下什么是MoE模型")
        assert content is not None, "Expected text response"
        # Model may or may not use tools — but should produce content
        assert len(content) > 20, "Response too short"

    def test_multi_step_tool_chain(self):
        """Multi-step: exec + create_reminder."""
        rounds, content, tools = run_agent_loop(
            "帮我看下我电脑的 python 版本，然后创建一个提醒明天下午3点升级 python"
        )
        assert content is not None, "Expected text response"
        assert rounds <= 6, f"Should complete in <=6 rounds, got {rounds}"
        tool_names = [t[0] for t in tools]
        assert "exec" in tool_names, "Should call exec for python version"

    def test_sse_format_valid(self):
        """Every SSE chunk should be valid JSON with expected structure."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "[Thu 2026-02-26 20:40 PST] what time is it"},
        ]
        content, tool_calls, raw_chunks, elapsed = stream_request(messages)

        assert len(raw_chunks) > 0, "Should have chunks"
        assert raw_chunks[-1] == "[DONE]", "Last chunk should be [DONE]"

        for chunk in raw_chunks:
            if isinstance(chunk, str):
                assert chunk == "[DONE]"
                continue
            # Must have standard OpenAI fields
            assert "id" in chunk
            assert "object" in chunk
            assert chunk["object"] == "chat.completion.chunk"

    def test_tool_call_has_valid_id(self):
        """Tool call chunks should have a call ID starting with 'call_'."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "[Thu 2026-02-26 20:40 PST] list files in /tmp",
            },
        ]
        content, tool_calls, raw_chunks, elapsed = stream_request(messages)

        if tool_calls:
            tc = tool_calls[0]
            assert "id" in tc, "Tool call should have id"
            assert tc["id"].startswith("call_"), (
                f"ID should start with call_, got {tc['id']}"
            )
            assert "function" in tc, "Tool call should have function"
            assert "name" in tc["function"], "Function should have name"
            assert "arguments" in tc["function"], "Function should have arguments"
            # Arguments should be valid JSON
            args = json.loads(tc["function"]["arguments"])
            assert isinstance(args, dict), "Arguments should be a dict"


# ---------------------------------------------------------------------------
# CLI mode for interactive debugging
# ---------------------------------------------------------------------------


def main():
    user_msg = (
        sys.argv[1] if len(sys.argv) > 1 else "你帮我看下 我今晚出去跑步是不是合适"
    )

    print("=" * 70)
    print(f"OpenClaw Simulation: '{user_msg}'")
    print(f"Tools: {len(TOOLS)}, Max rounds: {MAX_ROUNDS}")
    print("=" * 70)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Thu 2026-02-26 20:40 PST] {user_msg}"},
    ]

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n--- Round {round_num}: msgs={len(messages)} ---")

        content, tool_calls, raw_chunks, elapsed = stream_request(messages)

        # Analyze chunks
        chunk_types = []
        for c in raw_chunks:
            if isinstance(c, str):
                chunk_types.append("DONE")
                continue
            if "choices" not in c or not c["choices"]:
                chunk_types.append("?")
                continue
            ch = c["choices"][0]
            d = ch.get("delta", {})
            fr = ch.get("finish_reason")
            if "role" in d:
                chunk_types.append("role")
            elif "tool_calls" in d:
                chunk_types.append("tc")
            elif "content" in d and d["content"]:
                chunk_types.append("txt")
            elif fr:
                chunk_types.append(f"fin:{fr}")
            else:
                chunk_types.append("?")

        print(
            f"  {len(raw_chunks)} chunks [{', '.join(chunk_types[:15])}] {elapsed:.1f}s"
        )

        if tool_calls:
            tc = tool_calls[0]
            fn = tc["function"]
            print(f"  TOOL: {fn['name']}({fn['arguments'][:120]})")

            result = execute_tool(fn["name"], fn["arguments"])
            print(f"  RESULT: {result[:150]}")

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": fn}
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc["id"],
                }
            )
            continue

        if content:
            print(f"  TEXT ({len(content)} chars): {content[:300]}")
            print(f"\n  SUCCESS in {round_num} rounds")
            return

        print("  EMPTY — no content, no tool_calls")
        for i, c in enumerate(raw_chunks[:5]):
            if isinstance(c, str):
                print(f"    [{i}] {c}")
            else:
                print(f"    [{i}] {json.dumps(c)[:200]}")
        print("\n  FAIL")
        return

    print(f"\n  FAIL — exceeded {MAX_ROUNDS} rounds")


if __name__ == "__main__":
    main()
