"""Agent integration test runner — auto-generates tests from profile declarations.

Instead of static test files per agent, this module dynamically builds a test
plan from the AgentProfile's capability declarations and runs it.

Usage:
    from vllm_mlx.agents.testing import AgentTestRunner

    runner = AgentTestRunner(profile, base_url="http://localhost:8000/v1")
    report = runner.run()
    report.print_summary()
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from ..http_auth import rapid_mlx_auth_headers
from .base import AgentProfile

logger = logging.getLogger(__name__)

_ANTHROPIC_REMOTE_ENV = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

# Prefix of the err sentinel ``_agent_query`` returns when the agent CLI exits
# non-zero. Sentinel-prefixed like the ``SKIP:`` one so classification stays a
# property of the string the subprocess boundary produced, not a guess made
# from the child's own prose.
_EXIT_ERR_PREFIX = "EXIT:"

# One separator character models actually use for thousands: comma, no-break
# space, narrow no-break space, plain space.
_DIGIT_GROUP_SEP = "[,\u00a0\u202f ]"


def _grouped_number_pattern(digits: str) -> str:
    """``777777`` → ``(?:777777|777<sep>777)`` — bare or thousands-grouped.

    Only these two shapes. An earlier version stripped every separator that
    sat between two digits before matching, which also turned malformed or
    entirely different numbers ("7777 77") into the expected one — a fresh
    way to pass without answering, which is the bug this module exists to
    prevent (codex review round 2).
    """
    head = len(digits) % 3 or 3
    groups = [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
    return f"(?:{digits}|{_DIGIT_GROUP_SEP.join(groups)})"


def _exact_number_re(digits: str, *, grouped: bool = False) -> re.Pattern[str]:
    """Match `digits` as that number, not as a run of digits inside another.

    Boundaries are numeric rather than merely non-digit: a leading sign or a
    decimal point makes it a DIFFERENT number, so "-4", "0.4", "4.5",
    "12.777777" and "777777.5" are wrong answers rather than sloppy right
    ones. Word characters are excluded on both sides for the same reason —
    the "4" in a request id like "a4b" is not an answer to anything. A
    trailing period stays fine: that is a sentence ending, which is why only
    ``.<digit>`` is rejected on the right.

    A group separator with a digit on its far side is rejected as well, or
    the tolerance for "777,777" would accept "4,000", "1,777777" and
    "777777,000" — each a different number that merely starts or ends with
    the expected digits.
    """
    body = _grouped_number_pattern(digits) if grouped else digits
    return re.compile(
        "(?<![\\w.\\-−–])"
        + f"(?<!\\d{_DIGIT_GROUP_SEP})"
        + body
        + f"(?!{_DIGIT_GROUP_SEP}\\d)"
        + "(?!\\w)(?!\\.\\d)"
    )


# `_test_plain_chat` asks for 2+2 over HTTP; see the comment at its assertion
# for why its question stays easy while the e2e sibling's does not.
_PLAIN_CHAT_EXPECTED_RE = _exact_number_re("4")


# ---------------------------------------------------------------------------
# Test result types
# ---------------------------------------------------------------------------


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration_ms: float = 0
    message: str = ""
    category: str = "api"  # "api" or "e2e"


@dataclass
class TestReport:
    agent_name: str
    model_id: str
    results: list[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIP)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    def print_summary(self):
        icons = {
            TestStatus.PASS: "✅",
            TestStatus.FAIL: "❌",
            TestStatus.SKIP: "⬜",
            TestStatus.ERROR: "💥",
        }

        print(f"\n{'=' * 60}")
        print(f"  {self.agent_name} Integration Test Report")
        print(f"  Model: {self.model_id}")
        print(f"{'=' * 60}")

        # Group by category
        base_results = [r for r in self.results if r.category in ("api", "e2e")]
        specific_results = [r for r in self.results if r.category == "specific"]

        if base_results:
            print("\n  Base Tests (API + E2E)")
            print(f"  {'─' * 50}")
            for r in base_results:
                icon = icons[r.status]
                ms = f"({r.duration_ms:.0f}ms)" if r.duration_ms else ""
                msg = (
                    f" — {r.message}"
                    if r.message and r.status != TestStatus.PASS
                    else ""
                )
                print(f"  {icon} {r.name:40s} {ms}{msg}")
            base_pass = sum(1 for r in base_results if r.status == TestStatus.PASS)
            print(f"  → {base_pass}/{len(base_results)} base tests passed")

        if specific_results:
            print("\n  Framework-Specific Tests")
            print(f"  {'─' * 50}")
            for r in specific_results:
                icon = icons[r.status]
                msg = (
                    f" — {r.message}"
                    if r.message and r.status != TestStatus.PASS
                    else ""
                )
                print(f"  {icon} {r.name:40s}{msg}")
            spec_pass = sum(1 for r in specific_results if r.status == TestStatus.PASS)
            print(f"  → {spec_pass}/{len(specific_results)} specific tests passed")

        print(f"\n{'─' * 60}")
        total = len(self.results)
        print(
            f"  Total: {self.passed}/{total} passed, "
            f"{self.failed} failed, "
            f"{self.skipped} skipped"
        )
        print(f"  Duration: {self.total_duration_ms:.0f}ms")

        return self.failed == 0 and self.errored == 0


# ---------------------------------------------------------------------------
# Core API call helper
# ---------------------------------------------------------------------------

BASIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents",
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
            "name": "terminal",
            "description": "Execute a shell command",
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
            "name": "search_files",
            "description": "Search for files by pattern",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]


def _api_call(
    base_url: str,
    model_id: str,
    messages: list,
    tools=None,
    stream=False,
    max_tokens=300,
    temperature=0.3,
    timeout=120,
) -> dict:
    """Direct API call to Rapid-MLX server."""
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    resp = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=rapid_mlx_auth_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------


def _test_plain_chat(base_url: str, model_id: str) -> TestResult:
    """Basic chat — model responds coherently."""
    t0 = time.time()
    try:
        r = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        )
        content = r["choices"][0]["message"]["content"]
        # The number 4, not the digit anywhere in the string: "1234", "0.4",
        # "-4", "4.5" and a request id ending in 4 are none of them answers to
        # "what is 2+2" (#1981, sibling of the `_test_e2e_chat` false green).
        # Same matcher as the e2e probe so the two cannot drift apart.
        #
        # The question itself stays 2+2, unlike its e2e sibling, and the
        # threat models are why. This grades ``choices[0].message.content``
        # from a response that already passed ``raise_for_status`` — a
        # parsed answer field, not the raw stdout+stderr of a CLI that may
        # never have reached the server, so there is no error prose here to
        # be mistaken for an answer. That leaves this the cheapest possible
        # probe of "the server answers coherently", which is what makes it
        # a useful control: when e2e_chat fails and plain_chat passes, the
        # fault is in the agent CLI path, not the model.
        if _PLAIN_CHAT_EXPECTED_RE.search(content):
            return TestResult(
                "plain_chat", TestStatus.PASS, duration_ms=(time.time() - t0) * 1000
            )
        return TestResult(
            "plain_chat",
            TestStatus.FAIL,
            duration_ms=(time.time() - t0) * 1000,
            message=f"Expected '4', got: {content[:80]}",
        )
    except Exception as e:
        return TestResult(
            "plain_chat",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_single_tool_call(base_url: str, model_id: str) -> TestResult:
    """Model produces a structured tool call."""
    t0 = time.time()
    try:
        r = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "Read the file /etc/hostname"}],
            tools=BASIC_TOOLS,
        )
        msg = r["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return TestResult(
                "single_tool_call",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message="No tool_calls in response",
            )
        tc = msg["tool_calls"][0]
        if tc["function"]["name"] != "read_file":
            return TestResult(
                "single_tool_call",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message=f"Wrong tool: {tc['function']['name']}",
            )
        return TestResult(
            "single_tool_call", TestStatus.PASS, duration_ms=(time.time() - t0) * 1000
        )
    except Exception as e:
        return TestResult(
            "single_tool_call",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_tool_choice(base_url: str, model_id: str) -> TestResult:
    """Model picks the right tool from multiple options."""
    t0 = time.time()
    try:
        r = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "Run the command 'echo hello'"}],
            tools=BASIC_TOOLS,
        )
        msg = r["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return TestResult(
                "tool_choice",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message="No tool_calls",
            )
        name = msg["tool_calls"][0]["function"]["name"]
        if name != "terminal":
            return TestResult(
                "tool_choice",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message=f"Wrong tool: {name}, expected terminal",
            )
        return TestResult(
            "tool_choice", TestStatus.PASS, duration_ms=(time.time() - t0) * 1000
        )
    except Exception as e:
        return TestResult(
            "tool_choice",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_multi_turn_tool(base_url: str, model_id: str) -> TestResult:
    """Multi-turn: tool call → tool result → follow-up."""
    t0 = time.time()
    try:
        r1 = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "Read /etc/hosts"}],
            tools=BASIC_TOOLS,
        )
        msg1 = r1["choices"][0]["message"]
        if not msg1.get("tool_calls"):
            return TestResult(
                "multi_turn_tool",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message="First turn should trigger tool call",
            )
        r2 = _api_call(
            base_url,
            model_id,
            [
                {"role": "user", "content": "Read /etc/hosts"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": msg1["tool_calls"],
                },
                {
                    "role": "tool",
                    "tool_call_id": msg1["tool_calls"][0]["id"],
                    "content": "127.0.0.1 localhost\n::1 localhost",
                },
                {"role": "user", "content": "What IP addresses are in that file?"},
            ],
            tools=BASIC_TOOLS,
        )
        content = r2["choices"][0]["message"]["content"]
        if "127.0.0.1" in content or "localhost" in content:
            return TestResult(
                "multi_turn_tool",
                TestStatus.PASS,
                duration_ms=(time.time() - t0) * 1000,
            )
        return TestResult(
            "multi_turn_tool",
            TestStatus.FAIL,
            duration_ms=(time.time() - t0) * 1000,
            message=f"Bad follow-up: {content[:80]}",
        )
    except Exception as e:
        return TestResult(
            "multi_turn_tool",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_no_tool_leak(base_url: str, model_id: str) -> TestResult:
    """No raw tool markup leaks into content."""
    t0 = time.time()
    try:
        r = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "Use the terminal to run 'echo test'"}],
            tools=BASIC_TOOLS,
        )
        content = r["choices"][0]["message"].get("content", "")
        leaks = []
        for marker in ["<tool_call>", "<function=", "<|im_end|>", "<|tool_call|>"]:
            if marker in content:
                leaks.append(marker)
        if leaks:
            return TestResult(
                "no_tool_leak",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message=f"Leaked: {leaks}",
            )
        return TestResult(
            "no_tool_leak", TestStatus.PASS, duration_ms=(time.time() - t0) * 1000
        )
    except Exception as e:
        return TestResult(
            "no_tool_leak",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_many_tools(base_url: str, model_id: str, num_tools: int) -> TestResult:
    """Correct tool selection with many tools injected."""
    t0 = time.time()
    try:
        many_tools = []
        for i in range(num_tools):
            many_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": f"Tool number {i}",
                        "parameters": {
                            "type": "object",
                            "properties": {"arg": {"type": "string"}},
                            "required": ["arg"],
                        },
                    },
                }
            )
        many_tools.extend(BASIC_TOOLS)

        r = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "Run the command 'echo hello'"}],
            tools=many_tools,
            max_tokens=500,
        )
        msg = r["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return TestResult(
                "many_tools",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message=f"No tool call with {len(many_tools)} tools",
            )
        name = msg["tool_calls"][0]["function"]["name"]
        if name == "terminal":
            return TestResult(
                "many_tools",
                TestStatus.PASS,
                duration_ms=(time.time() - t0) * 1000,
                message=f"Correct with {len(many_tools)} tools",
            )
        return TestResult(
            "many_tools",
            TestStatus.FAIL,
            duration_ms=(time.time() - t0) * 1000,
            message=f"Wrong tool: {name}",
        )
    except Exception as e:
        return TestResult(
            "many_tools",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_streaming_tool_call(base_url: str, model_id: str) -> TestResult:
    """Streaming mode: tool calls arrive as structured deltas."""
    t0 = time.time()
    try:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Read the file /etc/hosts"}],
            "tools": BASIC_TOOLS,
            "max_tokens": 200,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=rapid_mlx_auth_headers(),
            timeout=60,
        ) as resp:
            tool_chunks = []
            finish_reason = None
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                if "tool_calls" in delta:
                    tool_chunks.append(delta["tool_calls"])
                if data["choices"][0].get("finish_reason"):
                    finish_reason = data["choices"][0]["finish_reason"]

        if not tool_chunks:
            return TestResult(
                "streaming_tool_call",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message="No tool_call chunks in stream",
            )
        if finish_reason != "tool_calls":
            return TestResult(
                "streaming_tool_call",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message=f"finish_reason={finish_reason}",
            )
        return TestResult(
            "streaming_tool_call",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            message=f"{len(tool_chunks)} chunks",
        )
    except Exception as e:
        return TestResult(
            "streaming_tool_call",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_no_tool_needed(base_url: str, model_id: str) -> TestResult:
    """Tools provided but not needed — model answers directly."""
    t0 = time.time()
    try:
        r = _api_call(
            base_url,
            model_id,
            [{"role": "user", "content": "What is the capital of France?"}],
            tools=BASIC_TOOLS,
        )
        content = r["choices"][0]["message"].get("content", "")
        if "paris" in content.lower():
            return TestResult(
                "no_tool_needed", TestStatus.PASS, duration_ms=(time.time() - t0) * 1000
            )
        return TestResult(
            "no_tool_needed",
            TestStatus.FAIL,
            duration_ms=(time.time() - t0) * 1000,
            message=f"Expected Paris: {content[:80]}",
        )
    except Exception as e:
        return TestResult(
            "no_tool_needed",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_stress_no_leak(base_url: str, model_id: str, rounds: int = 5) -> TestResult:
    """Rapid tool calls — zero tag leaks."""
    t0 = time.time()
    leaked = 0
    try:
        for i in range(rounds):
            r = _api_call(
                base_url,
                model_id,
                [{"role": "user", "content": f"Run: echo test_{i}"}],
                tools=BASIC_TOOLS,
                temperature=0.8,
            )
            content = r["choices"][0]["message"].get("content", "")
            if "<tool_call>" in content or "<function=" in content:
                leaked += 1
        if leaked:
            return TestResult(
                "stress_no_leak",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message=f"{leaked}/{rounds} leaked",
            )
        return TestResult(
            "stress_no_leak",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            message=f"0/{rounds} leaks",
        )
    except Exception as e:
        return TestResult(
            "stress_no_leak",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_streaming_basic(base_url: str, model_id: str) -> TestResult:
    """SSE streaming produces valid chunks."""
    t0 = time.time()
    try:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "What is 2+2? Reply with just the number."}
            ],
            "max_tokens": 50,
            "stream": True,
        }
        chunks = 0
        content = ""
        with httpx.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=rapid_mlx_auth_headers(),
            timeout=30,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                # Accept content OR reasoning deltas (thinking models).
                # The chat route emits `reasoning_content`, not
                # `reasoning` (vllm_mlx/routes/chat.py:1434) — the prior
                # `delta.get("reasoning")` lookup silently turned this
                # test into a no-op for any thinking-only response.
                text = delta.get("content") or delta.get("reasoning_content")
                if text:
                    content += text
                    chunks += 1

        if chunks == 0:
            return TestResult(
                "streaming_basic",
                TestStatus.FAIL,
                duration_ms=(time.time() - t0) * 1000,
                message="No content/reasoning chunks",
            )
        return TestResult(
            "streaming_basic",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            message=f"{chunks} chunks",
        )
    except Exception as e:
        return TestResult(
            "streaming_basic",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


def _test_tag_suppression(
    base_url: str, model_id: str, extra_tags: list[tuple[str, str]]
) -> TestResult:
    """Verify that extra streaming tags from profile are suppressed.

    This is a structural test — we can't force the model to produce specific
    tags, but we verify the filter is configured correctly.
    """
    t0 = time.time()
    try:
        from vllm_mlx.api.utils import StreamingToolCallFilter

        f = StreamingToolCallFilter(extra_tags=extra_tags)
        # Simulate each tag pair
        for open_tag, close_tag in extra_tags:
            test_input = f"before{open_tag}hidden{close_tag}after"
            result = f.process(test_input)
            remaining = f.flush()
            combined = result + remaining
            if "hidden" in combined:
                return TestResult(
                    "tag_suppression",
                    TestStatus.FAIL,
                    duration_ms=(time.time() - t0) * 1000,
                    message=f"Tag {open_tag!r} not suppressed",
                )
            # Reset for next tag
            f = StreamingToolCallFilter(extra_tags=extra_tags)

        return TestResult(
            "tag_suppression",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            message=f"{len(extra_tags)} tag pairs verified",
        )
    except Exception as e:
        return TestResult(
            "tag_suppression",
            TestStatus.ERROR,
            duration_ms=(time.time() - t0) * 1000,
            message=str(e),
        )


# ---------------------------------------------------------------------------
# E2E tests (require agent binary)
# ---------------------------------------------------------------------------


# The fixture's first line, and the only thing `_test_e2e_file_read`
# accepts back. It has to be a sentinel: the previous assertion passed on
# "build" or "project" appearing anywhere in the agent's output, which any
# answer that merely mentions the project — or reads some other file —
# satisfies without ever having read the line the test asks for.
E2E_FIRST_LINE_TOKEN = "rapid-mlx-e2e-first-line-sentinel"
# Keep the sentinel as a semantic TOML assignment rather than a comment.
# Several competent agents (Kilo and DSH among them) interpret "first line"
# as "first meaningful config line" and intentionally omit comments even
# when asked not to.  A unique top-level key remains impossible to guess while
# making the expected evidence part of the document rather than trivia.
E2E_FIRST_LINE = f"{E2E_FIRST_LINE_TOKEN} = true"


# `_test_e2e_chat` grades whatever the agent CLI wrote to stdout+stderr, and
# that text is not always an answer — a CLI that never reached the server
# still prints something.  The old probe asked for 2+2 and accepted a bare
# "4" anywhere in that text, so a pure launch failure reported PASS:
#
#     dsh: request failed: connect ECONNREFUSED 127.0.0.1:8477
#                                                        ^ this "4"
#
# An HTTP 404, a timestamp, a token count or a version string does it just as
# well (#1981).  The expected answer therefore has to be a value that cannot
# plausibly fall out of a failure message — the same reasoning that made
# `_test_e2e_file_read` demand a sentinel instead of the word "build".
#
# Why an arithmetic result rather than a sentinel word: the expected token must
# NOT appear in the prompt, or every CLI that echoes its prompt (`codex exec`
# does) becomes a new false green.  The operands below are chosen so that no
# column carries — this is as easy as multi-digit arithmetic gets, so a small
# quantized model is not being asked for a new capability — while the answer is
# longer than any port number (max 65535) or status code, and is accepted only
# as a standalone number so it can never be a fragment of an epoch, a request
# id or a hash.
E2E_CHAT_QUERY = "What is 123456 + 654321? Reply with just the number."
E2E_CHAT_EXPECTED = "777777"
# Grouped as well as bare: a model that writes "777,777" has answered the
# question. `_exact_number_re` explains the boundaries.
_E2E_CHAT_EXPECTED_RE = _exact_number_re(E2E_CHAT_EXPECTED, grouped=True)


def _e2e_chat_answered(out: str | None) -> bool:
    """True only when the agent came back with the expected sum itself."""
    if not out:
        return False
    return bool(_E2E_CHAT_EXPECTED_RE.search(out))


@contextlib.contextmanager
def _e2e_workspace():
    """A throwaway directory for the agent to work in.

    The e2e tests hand a real coding agent a real working directory, and
    two things follow from that.

    First, correctness: `_test_e2e_file_read` asks the agent to read
    ``pyproject.toml``. Run from the caller's cwd that only works when
    the caller happened to be standing in a Python project — from
    anywhere else the agent hunts for a file that isn't there and burns
    the whole timeout. The fixture below makes the task the same task
    every time.

    Second, a defined starting point. Codex is launched with
    ``--skip-git-repo-check`` (it refuses to run outside a trusted repo,
    and the runner cannot assume one), so it will pick up whatever is in
    the directory it starts in — including files and agent instructions
    left lying around. Starting it in a directory we just created, holding
    one file we wrote, makes that set knowable.

    **This is not a sandbox and does not pretend to be.** The runner gives the
    child a throwaway HOME and deterministic working directory, but the agent
    still runs as the same OS user and can read anything that user can via an
    absolute path. Codex additionally runs its own commands under Seatbelt on
    macOS, which is its business, not something arranged here.
    """
    workdir = tempfile.mkdtemp(prefix="rapid-mlx-agent-e2e-")
    try:
        Path(workdir, "pyproject.toml").write_text(
            f"{E2E_FIRST_LINE}\n"
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n'
            "\n"
            "[project]\n"
            'name = "rapid-mlx-agent-e2e-fixture"\n'
            'version = "0.0.0"\n',
            encoding="utf-8",
        )
        yield workdir
    finally:
        _remove_workspace(workdir)


def _remove_workspace(workdir: str) -> None:
    """Delete the workspace, and say so out loud if it cannot be deleted.

    Deliberately no permission-repair pass. Two earlier attempts at one were
    both wrong, and the second review round explained why the whole idea is:

    * chmod'ing a path we just checked with ``islink()`` is a TOCTOU — the
      agent can swap the directory for a link in between, and the write lands
      outside the workspace.
    * defeating that properly means descriptor-relative traversal, which is a
      lot of machinery for a test harness.

    And it buys nothing. The agent runs as the SAME user with the same
    permissions we have; anything our cleanup could be tricked into chmod'ing,
    the agent can chmod itself, directly. There is no privilege boundary here
    to defend, so a repair pass only adds a symlink-following write primitive
    to our own code.

    What the original NIT actually asked for was not silence. A workspace we
    cannot remove is reported, once, with the path — and that is where it
    stops.
    """
    shutil.rmtree(workdir, ignore_errors=True)
    if os.path.exists(workdir):
        logger.warning(
            "agent e2e workspace could not be removed and is being left behind: "
            "%s — the agent most likely changed permissions inside it",
            workdir,
        )


@contextlib.contextmanager
def _workspace_or(cwd: str | None):
    """The caller's directory if it named one, otherwise a fresh workspace.

    Each e2e test gets its OWN workspace. Sharing one across the three
    invocations would let whatever the chat agent wrote — a file, a
    stray AGENTS.md, a half-applied patch — become the next test's
    starting condition, which is precisely the contamination the
    workspace exists to prevent.
    """
    if cwd is not None:
        yield cwd
    else:
        with _e2e_workspace() as workdir:
            yield workdir


def _agent_query(
    binary: str,
    query_cmd: str,
    query: str,
    timeout: int = 120,
    cwd: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Run a single agent query. Returns (output, error).

    `binary` may be an absolute / ``~``-prefixed path *or* a bare name
    (e.g. ``codex``, ``opencode``). Bare names are resolved via
    ``shutil.which`` against ``$PATH``. Mirror of
    ``AgentTestRunner._agent_binary_available`` — a previous bug here
    treated bare names as relative paths so every e2e gate silently
    skipped with "Binary not found" even when the CLI was installed.
    """
    import shlex

    if "/" not in binary and not binary.startswith("~"):
        resolved = shutil.which(binary)
        if not resolved:
            return None, f"Binary not found: {binary}"
        binary_path = resolved
    else:
        binary_path = os.path.expanduser(binary)
        if not os.path.exists(binary_path):
            return None, f"Binary not found: {binary}"

    # Parse the command template first, then substitute the query inside the
    # already-separated argv entries. Substituting before ``shlex.split`` lets
    # quotes or shell-like text in a prompt change the child argument vector.
    try:
        cmd_parts = shlex.split(query_cmd)
    except ValueError:
        # Fallback: simple split if shlex can't parse
        cmd_parts = query_cmd.split()
    cmd_parts = [
        part.replace("{query}", query).replace("{cwd}", cwd or os.getcwd())
        for part in cmd_parts
    ]
    # Replace first part with full binary path
    cmd_parts[0] = binary_path

    child_env = os.environ.copy()
    # A local Anthropic endpoint must win over every provider-selection path.
    # Otherwise a developer's normal Bedrock/Vertex/Foundry setup can route an
    # E2E prompt (and its tools) to a real remote account despite our overrides.
    if env_overrides and "ANTHROPIC_BASE_URL" in env_overrides:
        for key in _ANTHROPIC_REMOTE_ENV:
            child_env.pop(key, None)
    child_env.update(env_overrides or {})

    # DSH rc.6 imports Node's Zstd stream API during profile boot but its npm
    # manifest declares no minimum Node engine.  Node 23.6 therefore installs
    # cleanly and then crashes with an opaque ESM export stack.  Probe the
    # capability selected by the exact child PATH and report the actionable
    # runtime mismatch before launching the harness.
    if Path(binary_path).name == "dsh":
        node = shutil.which("node", path=child_env.get("PATH"))
        if node is None:
            return None, "DeepSeek Harness requires Node.js on PATH"
        probe = subprocess.run(
            [
                node,
                "-e",
                "process.exit(typeof require('node:zlib').createZstdDecompress === 'function' ? 0 : 1)",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            env=child_env,
        )
        if probe.returncode != 0:
            return None, (
                "DeepSeek Harness requires a Node.js runtime with the Zstd "
                "stream API; update Node (Node 22.15+ works)"
            )

    try:
        proc = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
            # An explicit workspace, not the caller's cwd — see
            # ``_e2e_workspace``. Falling back to os.getcwd() keeps any
            # caller that hasn't opted in behaving exactly as before.
            cwd=cwd or os.getcwd(),
            # Close stdin. Without this the child inherits ours, and a CLI
            # that reads stdin when it isn't a TTY blocks until `timeout`
            # instead of answering the query it was handed on argv. Codex
            # does exactly that — `codex exec '<query>'` prints "Reading
            # additional input from stdin..." and then waits forever, so
            # the e2e gate spent its whole budget on a process that never
            # got a chance to fail. Every agent CLI runs headless here, so
            # none of them has any business reading stdin (#1683).
            stdin=subprocess.DEVNULL,
            # Env-profile setup returns shell exports; it cannot mutate the
            # parent process. Pass those values explicitly so an E2E agent
            # never falls back to its normal remote provider or real key.
            env=child_env,
        )
        output = proc.stdout + proc.stderr
        if "error" in output.lower() and "HTTP 4" in output:
            return None, "Agent error: server issue"
        # Harness self-refuses to initialize when the served model's
        # advertised context window is below what the harness's tool-rich
        # system prompt requires (issue #655 — Hermes hard-requires 64K;
        # qwen3.5-4b/9b-4bit advertise 32K, so init always fails on those).
        # The subprocess exits 0 and writes the refusal to stdout, so
        # without this check downstream tests see "no expected substring"
        # and report FAIL — which is dishonest, it's a harness-config
        # mismatch, not a rapid-mlx regression. Propagate as SKIP via the
        # ``SKIP:``-prefixed err sentinel that each ``_test_e2e_*`` already
        # honors.
        #
        # IMPORTANT: collapse whitespace first. The hermes binary hard-
        # wraps stderr at ~100 cols, so a literal ``"context window"``
        # substring check misses when wrapping splits the phrase as
        # ``"context\nwindow"`` — which is exactly what tripped the
        # initial #655 fix in production (#659 round-1 verify-pass).
        collapsed = re.sub(r"\s+", " ", output).strip()
        if "Failed to initialize agent" in collapsed and "context window" in collapsed:
            # Extract the actual refusal sentence (everything from
            # "Failed to initialize agent" through the next period) so
            # the resulting TestResult.message carries the actual model
            # name + advertised vs minimum token counts (codex NIT #659).
            # Without this, the user sees a generic "below the harness
            # minimum" and has to dig in the server log to learn which
            # model and what numbers — the very signal that makes this
            # actionable.
            # Non-greedy through to a period FOLLOWED BY whitespace or
            # end-of-string. The plain ``[^.]*\.`` form stops at the
            # first period — which on a model name like "Qwen3.5" is
            # mid-sentence, truncating the model identifier and the
            # advertised/minimum numbers.
            m = re.search(r"Failed to initialize agent.*?\.(?=\s|$)", collapsed)
            refusal_line = (m.group(0) if m else "")[:200]
            detail = f": {refusal_line}" if refusal_line else ""
            return None, (
                f"SKIP: agent refused init — served model's context window "
                f"is below the harness minimum{detail}"
            )
        # The one signal the OS gives us that the run did not complete. A CLI
        # that never reached the server prints its complaint and exits
        # non-zero; without this the caller sees only the complaint text and
        # grades it as if it were an answer (#1981).
        #
        # Deliberately a SOFT err — carried alongside the output, not instead
        # of it — because exit status can neither pass nor fail a test on its
        # own here:
        #   * dsh exits 0 even when it fails outright (a bad provider prints
        #     NO_ADAPTER and still returns 0; see its profile known_issues),
        #     so a zero exit proves nothing;
        #   * an agent may answer correctly and then exit non-zero on its way
        #     out, which #1598 already established is not grounds for failing
        #     a test whose evidence is on stdout.
        # Capability evidence therefore still wins; this only decides what an
        # evidence-LESS run gets called, and turns a silent "wrong answer"
        # FAIL into an ERROR naming the launch failure.
        #
        # But a process that reported failure only gets credit for what it put
        # on STDOUT: the returned output narrows to ``proc.stdout`` so a
        # diagnostic that quotes the expected evidence back at us ("expected
        # 777777; request failed", printed on stderr) cannot be mistaken for
        # the answer and re-open this very bug from the other side. Every agent
        # CLI the profiles drive writes its answer to stdout and its complaints
        # to stderr; the failure text is not discarded, it is summarized into
        # the err below, which is what gets reported.
        #
        # The TIMEOUT branch below applies the same rule, for the same reason:
        # both are runs that misbehaved, and it would be incoherent for a hung
        # CLI to get credit for stderr text a failed one does not.
        if proc.returncode != 0:
            tail = " ".join(output.split())[-160:]
            detail = f" — {tail}" if tail else ""
            return proc.stdout, (
                f"{_EXIT_ERR_PREFIX}{proc.returncode} agent CLI exited non-zero{detail}"
            )
        # A CLEAN exit keeps the combined capture, and that boundary is
        # deliberate. Narrowing every run to stdout was considered (codex
        # review round 3) and declined: it would change grading on the normal
        # path for all 13 profiles to defend against a failure message that
        # contains the expected six-digit sum, which a CLI has no way to know.
        # The cost is not hypothetical — the Hermes binary writes user-facing
        # text to STDERR (see the hard-wrap note in the refusal detection
        # above), so stdout-only grading could fail a Tier-1 profile that
        # answered correctly. Suppressing evidence is only justified once the
        # run itself has reported that it went wrong.
        return output, None
    except subprocess.TimeoutExpired as exc:
        # A headless agent can finish the capability under test, print the
        # evidence, and then fail to terminate its own loop.  Keep that output
        # so capability-specific probes can grade what actually happened and
        # report non-termination separately (#1598).  ``TimeoutExpired`` may
        # carry bytes even with ``text=True`` on older Python releases.
        def _timeout_text(value: str | bytes | None) -> str:
            if value is None:
                return ""
            return value.decode(errors="replace") if isinstance(value, bytes) else value

        # STDOUT only, exactly as on the non-zero-exit path: a hung CLI's
        # diagnostics ("ERROR: expected 777777", "could not run echo <marker>")
        # are not evidence that it did the work, and this err is one of the two
        # that lose to evidence. Nothing is lost from the report — the timeout
        # err carries no child text in either form.
        partial = _timeout_text(exc.stdout)
        return partial or None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def _err_to_status(err: str | None) -> TestStatus:
    """Map an ``_agent_query`` err string to the right TestStatus.

    Four buckets, in priority order:

    - ``"EXIT:"`` prefix → ERROR. The agent CLI exited non-zero (#1981).
      Checked FIRST because this err carries a tail of the child's own
      output, and that text is not ours: a child that printed "command
      not found" or "model not found" would otherwise be misrouted to
      SKIP by the substring rule below and vanish from the report.
    - ``"not found"`` → SKIP. Harness binary isn't installed; we can't
      run the e2e gate and shouldn't pretend to.
    - ``"SKIP:"`` prefix → SKIP. ``_agent_query`` propagates this for
      harness-side init refusals where rapid-mlx has nothing to fix —
      currently the Hermes-on-32K-context case (#655).
    - anything else → ERROR. Subprocess crashed / timed out / etc.
    """
    if not err:
        return TestStatus.PASS  # caller shouldn't pass empty err but stay safe
    if err.startswith(_EXIT_ERR_PREFIX):
        return TestStatus.ERROR
    if "not found" in err:
        return TestStatus.SKIP
    if err.startswith("SKIP:"):
        return TestStatus.SKIP
    return TestStatus.ERROR


def _err_loses_to_evidence(err: str | None) -> bool:
    """True for errs that must not overrule evidence the agent already gave.

    An agent can do the work, print the proof, and only *then* misbehave:
    never terminate (#1598) or exit non-zero on its way out (#1981). Both
    describe how the process ended, not whether the capability worked, so
    a caller holding the expected evidence keeps its PASS. Every other err
    — binary missing, init refusal, server error, crash — means the output
    cannot be trusted at all and wins outright.

    What counts as evidence is narrowed at the source: on a non-zero exit or
    a timeout ``_agent_query`` returns stdout only, so a failure diagnostic
    quoting the expected token back at us on stderr cannot buy a PASS here.

    Callers may only use this where their evidence cannot be produced by
    echoing the prompt — `_test_e2e_terminal` deliberately does not.
    """
    return bool(err) and (err == "TIMEOUT" or err.startswith(_EXIT_ERR_PREFIX))


def _evidence_note(err: str | None, evidence: str) -> str:
    """Message for a PASS whose CLI misbehaved after producing the evidence."""
    if err == "TIMEOUT":
        return f"{evidence} observed; agent CLI did not terminate before timeout"
    if err and err.startswith(_EXIT_ERR_PREFIX):
        return f"{evidence} observed; {err}"
    return ""


def _test_e2e_chat(
    binary: str,
    query_cmd: str,
    timeout: int,
    cwd: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> TestResult:
    """Agent basic chat."""
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        try:
            # Only the workspace ENTRY is guarded here. Widening this
            # to cover the query too would relabel a launch failure
            # (EACCES on the binary, say) as a workspace problem and
            # hide what actually went wrong.
            workdir = stack.enter_context(_workspace_or(cwd))
        except OSError as exc:
            # A full or read-only temp filesystem is an environment
            # failure, not an agent failure. Report it as this test's
            # ERROR instead of taking the whole runner down with it.
            return TestResult(
                "e2e_chat",
                TestStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000,
                message=f"agent workspace unavailable: {exc}",
                category="e2e",
            )
        out, err = _agent_query(
            binary,
            query_cmd,
            E2E_CHAT_QUERY,
            timeout,
            workdir,
            env_overrides,
        )
    # Evidence first, exactly as `_test_e2e_file_read` does: an agent that
    # answered and then failed to terminate (#1598) or exited non-zero
    # (#1981) still demonstrated the capability under test.
    answered = _e2e_chat_answered(out)
    if err and not (_err_loses_to_evidence(err) and answered):
        status = _err_to_status(err)
        return TestResult(
            "e2e_chat",
            status,
            duration_ms=(time.time() - t0) * 1000,
            message=err,
            category="e2e",
        )
    if answered:
        return TestResult(
            "e2e_chat",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            message=_evidence_note(err, "answer"),
            category="e2e",
        )
    return TestResult(
        "e2e_chat",
        TestStatus.FAIL,
        duration_ms=(time.time() - t0) * 1000,
        message=f"Expected '{E2E_CHAT_EXPECTED}': {(out or '')[:80]}",
        category="e2e",
    )


def _test_e2e_file_read(
    binary: str,
    query_cmd: str,
    timeout: int,
    cwd: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> TestResult:
    """Agent reads a file via tool call."""
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        try:
            # Only the workspace ENTRY is guarded here. Widening this
            # to cover the query too would relabel a launch failure
            # (EACCES on the binary, say) as a workspace problem and
            # hide what actually went wrong.
            workdir = stack.enter_context(_workspace_or(cwd))
        except OSError as exc:
            # A full or read-only temp filesystem is an environment
            # failure, not an agent failure. Report it as this test's
            # ERROR instead of taking the whole runner down with it.
            return TestResult(
                "e2e_file_read",
                TestStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000,
                message=f"agent workspace unavailable: {exc}",
                category="e2e",
            )
        out, err = _agent_query(
            binary,
            query_cmd,
            (
                "Read pyproject.toml and copy its first physical line exactly, "
                "including any leading punctuation. Do not skip comment lines."
            ),
            timeout,
            workdir,
            env_overrides,
        )
    if err and not (
        _err_loses_to_evidence(err) and E2E_FIRST_LINE_TOKEN in (out or "")
    ):
        status = _err_to_status(err)
        return TestResult(
            "e2e_file_read",
            status,
            duration_ms=(time.time() - t0) * 1000,
            message=err,
            category="e2e",
        )
    # The sentinel, not "build" or "project" — those appear in any answer
    # that merely talks about the project, or reads some other file.
    if E2E_FIRST_LINE_TOKEN in (out or ""):
        return TestResult(
            "e2e_file_read",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            message=_evidence_note(err, "file-read evidence"),
            category="e2e",
        )
    return TestResult(
        "e2e_file_read",
        TestStatus.FAIL,
        duration_ms=(time.time() - t0) * 1000,
        message=f"Unexpected: {(out or '')[:80]}",
        category="e2e",
    )


def _test_e2e_terminal(
    binary: str,
    query_cmd: str,
    timeout: int,
    agent_name: str,
    cwd: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> TestResult:
    """Agent runs a shell command."""
    t0 = time.time()
    marker = f"rapidmlx_{agent_name}_test"
    with contextlib.ExitStack() as stack:
        try:
            # Only the workspace ENTRY is guarded here. Widening this
            # to cover the query too would relabel a launch failure
            # (EACCES on the binary, say) as a workspace problem and
            # hide what actually went wrong.
            workdir = stack.enter_context(_workspace_or(cwd))
        except OSError as exc:
            # A full or read-only temp filesystem is an environment
            # failure, not an agent failure. Report it as this test's
            # ERROR instead of taking the whole runner down with it.
            return TestResult(
                "e2e_terminal",
                TestStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000,
                message=f"agent workspace unavailable: {exc}",
                category="e2e",
            )
        out, err = _agent_query(
            binary,
            query_cmd,
            f"Run 'echo {marker}' and show me the output",
            timeout,
            workdir,
            env_overrides,
        )
    # NO evidence carve-out here, unlike the chat and file-read probes, and
    # the difference is the marker itself: it is handed to the agent IN THE
    # PROMPT ("Run 'echo <marker>'"), so any CLI that echoes its prompt prints
    # it without having run anything. That is fine while the run is otherwise
    # healthy — a healthy run that echoed the prompt and stopped would also
    # have to survive `err is None` — but it is exactly what a broken CLI does
    # on its way out, so letting a marker overrule a crash or a hang would
    # hand a PASS to an agent that never opened a shell (codex review round 4).
    # Chat and file-read can afford the carve-out because their evidence — a
    # derived sum, a sentinel that lives only in the workspace file — cannot
    # be produced by echoing the question.
    if err:
        status = _err_to_status(err)
        return TestResult(
            "e2e_terminal",
            status,
            duration_ms=(time.time() - t0) * 1000,
            message=err,
            category="e2e",
        )
    if marker in (out or ""):
        return TestResult(
            "e2e_terminal",
            TestStatus.PASS,
            duration_ms=(time.time() - t0) * 1000,
            category="e2e",
        )
    return TestResult(
        "e2e_terminal",
        TestStatus.FAIL,
        duration_ms=(time.time() - t0) * 1000,
        message=f"Missing marker: {(out or '')[:80]}",
        category="e2e",
    )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


class AgentTestRunner:
    """Dynamically builds and runs tests based on an agent profile."""

    def __init__(
        self,
        profile: AgentProfile,
        base_url: str = "http://localhost:8000/v1",
        model_id: str | None = None,
        agent_version: str | None = None,
    ):
        self.profile = profile
        self.base_url = base_url
        self.agent_version = agent_version

        # Auto-detect model from server if not specified
        if model_id:
            self.model_id = model_id
        else:
            try:
                resp = httpx.get(
                    f"{base_url}/models",
                    headers=rapid_mlx_auth_headers(),
                    timeout=5,
                )
                self.model_id = resp.json()["data"][0]["id"]
            except Exception:
                self.model_id = "default"

    def _server_available(self) -> bool:
        """Check if the Rapid-MLX server is running."""
        try:
            r = httpx.get(
                f"{self.base_url.rstrip('/').rsplit('/v1', 1)[0]}/health", timeout=3
            )
            return r.status_code == 200
        except Exception:
            return False

    def _agent_binary_available(self) -> bool:
        """Check if the agent binary is installed.

        Two forms of `testing.binary` are supported:
          - **Absolute / ``~``-prefixed path** (e.g. ``~/.hermes/venv/bin/hermes``)
            — checked with ``os.path.exists`` after ``expanduser``.
          - **Bare name** (e.g. ``codex``, ``opencode``) — resolved via
            ``shutil.which`` against the current ``$PATH``. Profiles that
            ship a homebrew/npm CLI use this form, and a previous bug
            here treated them as relative paths so the e2e gate never
            fired even with the binary installed.
        """
        testing = self.profile.get_testing_for_version(self.agent_version)
        if not testing.binary:
            return False
        binary = testing.binary
        # Bare name (no `/`, no leading `~`) → look up on $PATH.
        if "/" not in binary and not binary.startswith("~"):
            return shutil.which(binary) is not None
        return os.path.exists(os.path.expanduser(binary))

    def build_test_plan(self) -> list[str]:
        """Build the list of test names that will run for this profile."""
        tests = ["plain_chat"]

        if self.profile.needs_function_calling:
            tests += [
                "single_tool_call",
                "tool_choice",
                "multi_turn_tool",
                "no_tool_leak",
                "no_tool_needed",
            ]
            if self.profile.needs_streaming:
                tests.append("streaming_tool_call")

        streaming = self.profile.get_streaming_for_version(self.agent_version)
        if (
            streaming.max_tools
            and streaming.max_tools > 10
            and self.profile.needs_function_calling
        ):
            tests.append("many_tools")

        if streaming.extra_tool_tags:
            tests.append("tag_suppression")

        if self.profile.needs_streaming:
            tests.append("streaming_basic")

        if self.profile.needs_function_calling:
            tests.append("stress_no_leak")

        # E2E tests
        if self._agent_binary_available():
            tests += ["e2e_chat"]
            if self.profile.needs_function_calling:
                tests += ["e2e_file_read", "e2e_terminal"]

        return tests

    def run(self) -> TestReport:
        """Run all applicable tests and return a report."""
        report = TestReport(
            agent_name=self.profile.display_name,
            model_id=self.model_id,
        )

        if not self._server_available():
            report.results.append(
                TestResult(
                    "server_check",
                    TestStatus.ERROR,
                    message="Rapid-MLX server not running",
                )
            )
            return report

        # Every agent is driven from a throwaway home.  Limiting isolation to
        # profiles with a dedicated ``home_env`` left env-configured CLIs
        # (notably Claude Code) free to load the operator's plugins, hooks,
        # history and credentials.  It also let file-config agents without a
        # relocation variable (OpenCode/Qwen Code/Kilo) refresh the real
        # ``~/.config`` during ``--test``.  HOME is therefore the universal
        # boundary; tool-specific relocation variables are layered on top.
        active_config = self.profile.get_config_for_version(self.agent_version)
        isolated_config_home = tempfile.TemporaryDirectory(
            prefix=f"rapid-mlx-{self.profile.name}-home-"
        )
        config_home_env = active_config.home_env

        # Refresh the agent's on-disk config (e.g. ``~/.hermes/config.yaml``)
        # so e2e tests run against the CURRENT model_id + base_url instead
        # of whatever was left over from a prior bench / manual invocation.
        # v0.7.26 dogfood found this: after qwen2.5-14b ran first, the
        # hermes config retained ``Qwen2.5-14B-Instruct`` and every
        # subsequent harness sweep's ``e2e_file_read`` failed with
        # ``Failed to initialize agent: Model mlx-community/Qwen2.5-14B-Instruct``.
        # ``setup_agent_config`` is a no-op for env-var-style profiles
        # (codex, opencode, aider) since those carry config via env only.
        try:
            from .adapter import fetch_context_window, setup_agent_config

            context_length = fetch_context_window(self.base_url, self.model_id)

            redirected = {"HOME": isolated_config_home.name}
            if config_home_env:
                redirected[config_home_env] = isolated_config_home.name
            previous = {key: os.environ.get(key) for key in redirected}
            os.environ.update(redirected)
            try:
                setup_agent_config(
                    self.profile,
                    base_url=self.base_url,
                    model_id=self.model_id,
                    agent_version=self.agent_version,
                    context_length=context_length,
                )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        except Exception as exc:  # noqa: BLE001 — config refresh must never abort the sweep
            logger.warning(
                "could not refresh %s config before harness sweep: %s",
                self.profile.name,
                exc,
            )

        t0 = time.time()
        streaming = self.profile.get_streaming_for_version(self.agent_version)
        testing = self.profile.get_testing_for_version(self.agent_version)
        # Use the same server-advertised capacity for env-style profiles.
        # File-style profiles consumed it above during setup; rendering env
        # profiles with the 32K fallback here would make the test sweep and
        # `agents --setup` exercise materially different configurations.
        try:
            from .adapter import fetch_context_window

            context_length = fetch_context_window(self.base_url, self.model_id)
        except Exception:  # noqa: BLE001 — retain the documented fallback
            context_length = None
        rendered_config = self.profile.render_config(
            self.base_url,
            self.model_id,
            self.agent_version,
            context_length=context_length,
        )
        env_overrides = (
            {str(key): str(value) for key, value in rendered_config.items()}
            if active_config.type == "env" and isinstance(rendered_config, dict)
            else None
        )
        env_overrides = dict(env_overrides or {})
        env_overrides["HOME"] = isolated_config_home.name
        # Claude Code documents CLAUDE_CONFIG_DIR as its supported config
        # relocation.  Set it even though HOME is redirected: some packaged
        # launchers resolve the account home before applying the child env.
        if self.profile.name == "claude-code":
            env_overrides["CLAUDE_CONFIG_DIR"] = str(
                Path(isolated_config_home.name) / ".claude"
            )
        # DSH's pi-ai OpenAI transport currently insists on resolving a
        # credential even for a loopback endpoint.  The first-class setup
        # stores this non-secret sentinel in DSH's managed credential file;
        # the ephemeral test home gets the equivalent process-local value.
        if self.profile.name == "deepseek-harness":
            env_overrides["RAPID_MLX_API_KEY"] = "not-needed"
        if config_home_env:
            env_overrides[config_home_env] = isolated_config_home.name

        # --- API tests ---
        report.results.append(_test_plain_chat(self.base_url, self.model_id))

        if self.profile.needs_function_calling:
            report.results.append(_test_single_tool_call(self.base_url, self.model_id))
            report.results.append(_test_tool_choice(self.base_url, self.model_id))
            report.results.append(_test_multi_turn_tool(self.base_url, self.model_id))
            report.results.append(_test_no_tool_leak(self.base_url, self.model_id))
            report.results.append(_test_no_tool_needed(self.base_url, self.model_id))

            if self.profile.needs_streaming:
                report.results.append(
                    _test_streaming_tool_call(self.base_url, self.model_id)
                )

        if streaming.max_tools and streaming.max_tools > 10:
            report.results.append(
                _test_many_tools(self.base_url, self.model_id, streaming.max_tools)
            )

        if streaming.extra_tool_tags:
            report.results.append(
                _test_tag_suppression(
                    self.base_url, self.model_id, streaming.extra_tool_tags
                )
            )

        if self.profile.needs_streaming:
            report.results.append(_test_streaming_basic(self.base_url, self.model_id))

        if self.profile.needs_function_calling:
            report.results.append(_test_stress_no_leak(self.base_url, self.model_id))

        # --- E2E tests ---
        if self._agent_binary_available() and testing.binary and testing.query_cmd:
            binary = os.path.expanduser(testing.binary)
            # No shared workspace here on purpose: each _test_e2e_* opens
            # its own, so one invocation's leftovers cannot become the
            # next one's starting condition.
            report.results.append(
                _test_e2e_chat(
                    binary,
                    testing.query_cmd,
                    testing.query_timeout,
                    env_overrides=env_overrides,
                )
            )
            if self.profile.needs_function_calling:
                report.results.append(
                    _test_e2e_file_read(
                        binary,
                        testing.query_cmd,
                        testing.query_timeout,
                        env_overrides=env_overrides,
                    )
                )
                report.results.append(
                    _test_e2e_terminal(
                        binary,
                        testing.query_cmd,
                        testing.query_timeout,
                        self.profile.name,
                        env_overrides=env_overrides,
                    )
                )
        elif testing.binary:
            # Two distinct skip reasons land here:
            #   (a) binary on PATH but profile has `query_cmd: null`
            #       — interactive-only agent (opencode, openhands,
            #       openclaude). The runner has no way to drive it
            #       headlessly, but the binary is installed; saying
            #       "Binary not found" is actively misleading.
            #   (b) binary is genuinely missing from PATH / disk.
            if self._agent_binary_available() and not testing.query_cmd:
                msg = (
                    f"Interactive-only profile ({testing.binary} installed "
                    "but query_cmd=null — no headless driver)"
                )
            else:
                msg = f"Binary not found: {testing.binary}"
            report.results.append(
                TestResult(
                    "e2e_tests",
                    TestStatus.SKIP,
                    message=msg,
                    category="e2e",
                )
            )

        # --- Framework-specific tests ---
        if testing.specific_tests:
            specific_results = self._run_specific_tests(testing.specific_tests)
            report.results.extend(specific_results)

        report.total_duration_ms = (time.time() - t0) * 1000
        if isolated_config_home is not None:
            isolated_config_home.cleanup()
        return report

    def _run_specific_tests(self, test_module_name: str) -> list[TestResult]:
        """Run framework-specific tests from an external test module.

        The module should follow the pattern of tests/integrations/test_*.py:
        - Has a `results` dict at module level
        - Test code runs on import (module-level execution)
        - `results` maps test_name -> "PASS" | "FAIL: reason"

        Args:
            test_module_name: Filename like "test_pydantic_ai_full.py"

        Returns:
            List of TestResult from the specific test module.
        """
        import re
        from pathlib import Path

        # The loaded path is fed to spec_from_file_location and executed.
        # specific_tests today comes from package-shipped YAML profiles
        # (vllm_mlx/agents/profiles/*.yaml), so this is defense-in-depth
        # rather than a live exploit gate — but reject anything that
        # could traverse out of the bundled / source directories before
        # we resolve it.
        # Disallow leading hyphens — would never come from a profile YAML
        # but downstream tooling (e.g. shell wrappers around the file
        # path) treats `-foo` as an option flag.
        # re.ASCII pins \w to [A-Za-z0-9_] — without it, Unicode letters
        # like `tést.py` would slip through.
        if not re.fullmatch(
            r"[A-Za-z0-9_][\w\-]*\.(py|sh)", test_module_name, re.ASCII
        ):
            return [
                TestResult(
                    f"specific:{test_module_name}",
                    TestStatus.SKIP,
                    message=(
                        f"Invalid integration test name '{test_module_name}': "
                        "must be a bare filename ending in .py or .sh "
                        "(no path separators or .. components)."
                    ),
                    category="specific",
                )
            ]

        # Find the test file. Prefer the bundled location (ships with
        # pip/brew wheels via package_data) and fall back to the source
        # layout for editable / repo-clone installs.
        bundled_dir = Path(__file__).parent.parent / "_integration_tests"
        source_dir = Path(__file__).parent.parent.parent / "tests" / "integrations"
        test_path = None
        for candidate in (
            bundled_dir / test_module_name,
            source_dir / test_module_name,
        ):
            if candidate.exists():
                test_path = candidate
                break
        if test_path is None:
            return [
                TestResult(
                    f"specific:{test_module_name}",
                    TestStatus.SKIP,
                    message=(
                        f"Integration test '{test_module_name}' not found in "
                        f"bundled ({bundled_dir}) or source ({source_dir}) "
                        "locations. If you installed via pip/brew, this is a "
                        "packaging bug — please report it. For repo clones, "
                        "ensure tests/integrations/ exists."
                    ),
                    category="specific",
                )
            ]

        t0 = time.time()
        try:
            # Set base URL env var so specific test modules use the right server
            import sys

            os.environ["RAPID_MLX_BASE_URL"] = self.base_url

            # Suppress sys.exit (test modules call exit() at the end)
            original_exit = sys.exit
            exec_error = None
            namespace: dict = {}
            sys.exit = lambda *a: None
            try:
                # Integration files are executable test programs. Some expose
                # module-level results, while others intentionally guard the
                # suite with ``if __name__ == "__main__"``. Importing under a
                # synthetic module name silently skipped the latter (notably
                # PydanticAI) and then reported a harness error without making
                # a single SDK call. Execute with script semantics and collect
                # the resulting globals for both styles.
                namespace = {
                    "__name__": "__main__",
                    "__file__": str(test_path),
                    "__builtins__": __builtins__,
                }
                source = test_path.read_bytes()
                exec(compile(source, str(test_path), "exec"), namespace)
                run_suite = namespace.get("run_suite")
                if callable(run_suite):
                    run_suite()
            except SystemExit:
                pass  # expected — test modules call exit()
            except Exception as e:
                exec_error = e
            finally:
                sys.exit = original_exit

            # If module failed to execute, report it as an error
            if exec_error is not None:
                # ImportError / ModuleNotFoundError nearly always means
                # the harness deps weren't installed in the rapid-mlx
                # venv. The profile's ``install_cmd`` is exactly that
                # hint; surface it in the message so users have a
                # one-line copy-paste fix instead of "Module execution
                # failed: No module named 'langchain_core'" with no
                # path forward (v0.7.26 dogfood found this on every
                # langchain harness run).
                hint = ""
                if isinstance(exec_error, (ImportError, ModuleNotFoundError)):
                    try:
                        testing = self.profile.get_testing_for_version(
                            self.agent_version
                        )
                        if testing and testing.install_cmd:
                            hint = (
                                f" [hint: harness deps missing — run "
                                f"`{testing.install_cmd}` in the rapid-mlx venv]"
                            )
                    except (AttributeError, KeyError):
                        # Profile shape changed under us — fall back to
                        # the bare error rather than crash the harness.
                        pass
                return [
                    TestResult(
                        f"specific:{test_module_name}",
                        TestStatus.ERROR,
                        duration_ms=(time.time() - t0) * 1000,
                        message=f"Module execution failed: {exec_error!s:.120}{hint}",
                        category="specific",
                    )
                ]

            # Extract results dict
            results_dict = namespace.get("results", {})
            if not results_dict:
                return [
                    TestResult(
                        f"specific:{test_module_name}",
                        TestStatus.ERROR,
                        duration_ms=(time.time() - t0) * 1000,
                        message="No test results found (missing 'results' dict or all tests skipped)",
                        category="specific",
                    )
                ]

            test_results = []
            per_test_ms = (time.time() - t0) * 1000 / max(len(results_dict), 1)
            for name, status in results_dict.items():
                # Test modules report PASS / "FAIL: <reason>" / "SKIP: <reason>".
                # SKIP exists so deep agentic tests can signal "the environment
                # can't honestly run this" (e.g. Hermes's 32K-context refusal
                # on small models) without dirtying the gauntlet with a FAIL
                # that isn't actionable.
                if status == "PASS":
                    test_results.append(
                        TestResult(
                            name,
                            TestStatus.PASS,
                            duration_ms=per_test_ms,
                            category="specific",
                        )
                    )
                elif isinstance(status, str) and status.startswith("SKIP:"):
                    test_results.append(
                        TestResult(
                            name,
                            TestStatus.SKIP,
                            duration_ms=per_test_ms,
                            message=status[len("SKIP:") :].strip()[:120],
                            category="specific",
                        )
                    )
                else:
                    msg = (
                        status.replace("FAIL: ", "")
                        if status.startswith("FAIL:")
                        else status
                    )
                    test_results.append(
                        TestResult(
                            name,
                            TestStatus.FAIL,
                            duration_ms=per_test_ms,
                            message=msg[:120],
                            category="specific",
                        )
                    )
            return test_results

        except Exception as e:
            return [
                TestResult(
                    f"specific:{test_module_name}",
                    TestStatus.ERROR,
                    duration_ms=(time.time() - t0) * 1000,
                    message=str(e)[:120],
                    category="specific",
                )
            ]
