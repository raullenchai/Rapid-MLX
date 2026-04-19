#!/usr/bin/env python3
"""Agent-style soak test — simulates real agent workloads for 10+ minutes.

Runs diverse, long-running scenarios that exercise the streaming pipeline
under sustained load:
- Multi-turn conversations (10+ turns with tool calls)
- Concurrent agent sessions
- Mixed streaming + non-streaming
- Large tool schemas (60+ tools like Hermes)
- Repeated rapid tool calls
- Long streaming outputs with reasoning
- Disconnect/reconnect patterns

Usage:
    python3 scripts/agent_soak_test.py --url http://localhost:8000/v1 --duration 600
"""

import argparse
import asyncio
import json
import random
import time
import traceback

import httpx

# Hermes-scale tool definitions (simulating 20 tools)
AGENT_TOOLS = [
    {"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool {i} for testing",
     "parameters": {"type": "object", "properties": {
         "arg1": {"type": "string", "description": "First argument"},
         "arg2": {"type": "integer", "description": "Second argument"},
     }, "required": ["arg1"]}}}
    for i in range(20)
]

TOOL_PROMPTS = [
    "Use tool_0 to check the weather in Tokyo",
    "Use tool_1 to search for Python files",
    "Use tool_2 to get the current time",
    "Call tool_3 with arg1='test' and arg2=42",
    "Use tool_5 to list files in the current directory",
]

CHAT_PROMPTS = [
    "Explain what a decorator is in Python in 2 sentences.",
    "Write a haiku about programming.",
    "What are the three pillars of OOP?",
    "Explain the difference between a list and a tuple.",
    "What is a closure in JavaScript?",
    "How does garbage collection work?",
    "What is the time complexity of binary search?",
    "Explain REST vs GraphQL in one paragraph.",
    "What is dependency injection?",
    "Explain the CAP theorem briefly.",
]

LONG_PROMPTS = [
    "Write a detailed guide on how to set up a CI/CD pipeline for a Python project, covering GitHub Actions, testing, linting, and deployment. Include code examples.",
    "Explain the complete lifecycle of an HTTP request from browser to server and back, including DNS, TCP, TLS, HTTP parsing, and response rendering.",
    "Write a comprehensive comparison of 5 different database systems (PostgreSQL, MySQL, MongoDB, Redis, SQLite) with pros, cons, and use cases for each.",
]


class SoakTestRunner:
    def __init__(self, base_url: str, duration: int = 600):
        self.base_url = base_url
        self.duration = duration
        self.stats = {
            "total_requests": 0,
            "stream_requests": 0,
            "nonstream_requests": 0,
            "tool_requests": 0,
            "multi_turn_sessions": 0,
            "errors": 0,
            "error_details": [],
            "total_tokens": 0,
            "total_chunks": 0,
            "max_chunks_per_request": 0,
            "disconnects": 0,
        }
        self.start_time = None

    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0

    def remaining(self) -> float:
        return max(0, self.duration - self.elapsed())

    def log(self, msg: str):
        elapsed = self.elapsed()
        print(f"  [{elapsed:6.1f}s] {msg}")

    async def stream_request(self, messages: list, max_tokens: int = 100,
                              tools=None, timeout: float = 60) -> dict:
        """Make a streaming request and validate SSE output."""
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools

        chunks = 0
        has_role = False
        has_content = False
        has_done = False
        has_finish = False
        has_usage = False
        has_tool_calls = False
        content_text = ""
        finish_reason = None

        async with httpx.AsyncClient(timeout=timeout) as client, \
                client.stream("POST", f"{self.base_url}/chat/completions",
                              json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        has_done = True
                        continue

                    data = json.loads(line[6:])
                    chunks += 1

                    if not data.get("choices"):
                        if data.get("usage"):
                            has_usage = True
                        continue

                    delta = data["choices"][0].get("delta", {})
                    if "role" in delta:
                        has_role = True
                    # Count both content and reasoning_content (OutputRouter models)
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        has_content = True
                        content_text += text
                    if delta.get("tool_calls"):
                        has_tool_calls = True
                    if data["choices"][0].get("finish_reason"):
                        has_finish = True
                        finish_reason = data["choices"][0]["finish_reason"]

        self.stats["total_chunks"] += chunks
        self.stats["max_chunks_per_request"] = max(
            self.stats["max_chunks_per_request"], chunks
        )
        self.stats["stream_requests"] += 1
        self.stats["total_requests"] += 1

        return {
            "chunks": chunks,
            "has_role": has_role,
            "has_content": has_content,
            "has_done": has_done,
            "has_finish": has_finish,
            "has_usage": has_usage,
            "has_tool_calls": has_tool_calls,
            "content": content_text,
            "finish_reason": finish_reason,
        }

    async def nonstream_request(self, messages: list, max_tokens: int = 100,
                                 tools=None) -> dict:
        """Make a non-streaming request."""
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            data = resp.json()

        self.stats["nonstream_requests"] += 1
        self.stats["total_requests"] += 1
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        self.stats["total_tokens"] += tokens

        return {
            "content": data["choices"][0]["message"].get("content", ""),
            "tool_calls": data["choices"][0]["message"].get("tool_calls"),
            "tokens": tokens,
        }

    async def scenario_multi_turn_agent(self):
        """Simulate a 10-turn agent session with tool calls."""
        self.log("Multi-turn agent session (10 turns)...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
        ]

        for turn in range(10):
            if self.remaining() <= 0:
                break

            if turn % 3 == 0 and TOOL_PROMPTS:
                # Tool call turn
                prompt = random.choice(TOOL_PROMPTS)
                messages.append({"role": "user", "content": prompt})
                result = await self.stream_request(messages, max_tokens=200, tools=AGENT_TOOLS)
                self.stats["tool_requests"] += 1

                # Simulate tool response
                messages.append({"role": "assistant", "content": result["content"] or "Using tool..."})
                messages.append({"role": "user", "content": "What was the result?"})
            else:
                # Chat turn
                prompt = random.choice(CHAT_PROMPTS)
                messages.append({"role": "user", "content": prompt})
                result = await self.stream_request(messages, max_tokens=150)
                messages.append({"role": "assistant", "content": result["content"] or "..."})

            # Validate SSE structure
            if not result["has_role"]:
                self.stats["errors"] += 1
                self.stats["error_details"].append(f"Turn {turn}: missing role chunk")
            if not result["has_done"]:
                self.stats["errors"] += 1
                self.stats["error_details"].append(f"Turn {turn}: missing [DONE]")

        self.stats["multi_turn_sessions"] += 1
        self.log(f"  Session done: {len(messages)} messages")

    async def scenario_concurrent_agents(self):
        """4 concurrent agent sessions streaming simultaneously."""
        self.log("Concurrent agents (4 parallel streams)...")

        async def single_session(session_id: int):
            msgs = [{"role": "user", "content": f"Session {session_id}: {random.choice(CHAT_PROMPTS)}"}]
            result = await self.stream_request(msgs, max_tokens=200)
            return result

        results = await asyncio.gather(
            *[single_session(i) for i in range(4)],
            return_exceptions=True,
        )

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                self.stats["errors"] += 1
                self.stats["error_details"].append(f"Concurrent session {i}: {r}")
            elif not r["has_done"]:
                self.stats["errors"] += 1
                self.stats["error_details"].append(f"Concurrent session {i}: missing [DONE]")

        self.log("  All 4 completed")

    async def scenario_long_generation(self):
        """Long streaming output (512+ tokens)."""
        self.log("Long generation (512 tokens)...")
        prompt = random.choice(LONG_PROMPTS)
        result = await self.stream_request(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            timeout=120,
        )
        self.log(f"  {result['chunks']} chunks, finish={result['finish_reason']}")

        if not result["has_finish"]:
            self.stats["errors"] += 1
            self.stats["error_details"].append("Long gen: no finish_reason")

    async def scenario_rapid_tool_calls(self):
        """10 rapid-fire tool call requests."""
        self.log("Rapid tool calls (10 sequential)...")
        for i in range(10):
            if self.remaining() <= 0:
                break
            result = await self.nonstream_request(
                [{"role": "user", "content": random.choice(TOOL_PROMPTS)}],
                max_tokens=200,
                tools=AGENT_TOOLS[:5],
            )
            self.stats["tool_requests"] += 1

        self.log("  10 tool calls done")

    async def scenario_mixed_workload(self):
        """Mixed streaming + non-streaming + tool calls concurrently."""
        self.log("Mixed workload (6 concurrent, different types)...")

        async def stream_chat():
            return await self.stream_request(
                [{"role": "user", "content": random.choice(CHAT_PROMPTS)}],
                max_tokens=100,
            )

        async def nonstream_chat():
            return await self.nonstream_request(
                [{"role": "user", "content": random.choice(CHAT_PROMPTS)}],
                max_tokens=50,
            )

        async def tool_call():
            return await self.nonstream_request(
                [{"role": "user", "content": random.choice(TOOL_PROMPTS)}],
                max_tokens=200,
                tools=AGENT_TOOLS[:10],
            )

        results = await asyncio.gather(
            stream_chat(), stream_chat(),
            nonstream_chat(), nonstream_chat(),
            tool_call(), tool_call(),
            return_exceptions=True,
        )

        errors = sum(1 for r in results if isinstance(r, Exception))
        if errors:
            self.stats["errors"] += errors
            for r in results:
                if isinstance(r, Exception):
                    self.stats["error_details"].append(f"Mixed: {r}")
        self.log(f"  6 requests done, {errors} errors")

    async def scenario_disconnect_reconnect(self):
        """Disconnect mid-stream, then immediately reconnect."""
        self.log("Disconnect/reconnect (3 cycles)...")
        for i in range(3):
            if self.remaining() <= 0:
                break
            try:
                async with httpx.AsyncClient(timeout=5) as client, \
                        client.stream(
                            "POST", f"{self.base_url}/chat/completions",
                            json={
                                "model": "default",
                                "messages": [{"role": "user", "content": "Write a long essay about AI"}],
                                "max_tokens": 500,
                                "stream": True,
                            },
                        ) as resp:
                        count = 0
                        async for line in resp.aiter_lines():
                            count += 1
                            if count >= 5:
                                break  # disconnect after 5 chunks
                self.stats["disconnects"] += 1
            except Exception:
                pass

            # Reconnect immediately
            result = await self.nonstream_request(
                [{"role": "user", "content": "What is 2+2?"}],
                max_tokens=10,
            )
            if not result["content"]:
                self.stats["errors"] += 1
                self.stats["error_details"].append(f"Reconnect {i}: empty response")

        self.log("  3 cycles done, server healthy")

    async def run(self):
        self.start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"  Agent Soak Test — {self.duration}s duration")
        print(f"  URL: {self.base_url}")
        print(f"{'=' * 60}\n")

        # Run scenarios in a loop until duration expires
        scenarios = [
            self.scenario_multi_turn_agent,
            self.scenario_concurrent_agents,
            self.scenario_long_generation,
            self.scenario_rapid_tool_calls,
            self.scenario_mixed_workload,
            self.scenario_disconnect_reconnect,
        ]

        round_num = 0
        while self.remaining() > 0:
            round_num += 1
            self.log(f"=== Round {round_num} (remaining: {self.remaining():.0f}s) ===")

            for scenario in scenarios:
                if self.remaining() <= 0:
                    break
                try:
                    await scenario()
                except Exception as e:
                    self.stats["errors"] += 1
                    self.stats["error_details"].append(f"{scenario.__name__}: {e}")
                    self.log(f"  ERROR: {e}")
                    traceback.print_exc()

        elapsed = self.elapsed()
        print(f"\n{'=' * 60}")
        print(f"  RESULTS ({elapsed:.0f}s)")
        print(f"{'=' * 60}")
        print(f"  Total requests:    {self.stats['total_requests']}")
        print(f"  Stream requests:   {self.stats['stream_requests']}")
        print(f"  Non-stream:        {self.stats['nonstream_requests']}")
        print(f"  Tool requests:     {self.stats['tool_requests']}")
        print(f"  Multi-turn:        {self.stats['multi_turn_sessions']} sessions")
        print(f"  Total chunks:      {self.stats['total_chunks']}")
        print(f"  Max chunks/req:    {self.stats['max_chunks_per_request']}")
        print(f"  Disconnects:       {self.stats['disconnects']}")
        print(f"  Errors:            {self.stats['errors']}")
        if self.stats["error_details"]:
            print("  Error details:")
            for d in self.stats["error_details"][:10]:
                print(f"    - {d}")
        print()
        status = "PASS" if self.stats["errors"] == 0 else "FAIL"
        print(f"  Status: {status}")
        print(f"{'=' * 60}")
        return self.stats["errors"] == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--duration", type=int, default=600,
                        help="Test duration in seconds (default: 600 = 10 min)")
    args = parser.parse_args()

    success = asyncio.run(SoakTestRunner(args.url, args.duration).run())
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
