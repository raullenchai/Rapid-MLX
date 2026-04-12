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

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

from .base import AgentProfile

logger = logging.getLogger(__name__)


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

        print(f"\n{'='*60}")
        print(f"  {self.agent_name} Integration Test Report")
        print(f"  Model: {self.model_id}")
        print(f"{'='*60}")

        # Group by category
        base_results = [r for r in self.results if r.category in ("api", "e2e")]
        specific_results = [r for r in self.results if r.category == "specific"]

        if base_results:
            print("\n  Base Tests (API + E2E)")
            print(f"  {'─'*50}")
            for r in base_results:
                icon = icons[r.status]
                ms = f"({r.duration_ms:.0f}ms)" if r.duration_ms else ""
                msg = f" — {r.message}" if r.message and r.status != TestStatus.PASS else ""
                print(f"  {icon} {r.name:40s} {ms}{msg}")
            base_pass = sum(1 for r in base_results if r.status == TestStatus.PASS)
            print(f"  → {base_pass}/{len(base_results)} base tests passed")

        if specific_results:
            print("\n  Framework-Specific Tests")
            print(f"  {'─'*50}")
            for r in specific_results:
                icon = icons[r.status]
                msg = f" — {r.message}" if r.message and r.status != TestStatus.PASS else ""
                print(f"  {icon} {r.name:40s}{msg}")
            spec_pass = sum(1 for r in specific_results if r.status == TestStatus.PASS)
            print(f"  → {spec_pass}/{len(specific_results)} specific tests passed")

        print(f"\n{'─'*60}")
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


def _api_call(base_url: str, model_id: str, messages: list,
              tools=None, stream=False, max_tokens=300, temperature=0.3,
              timeout=120) -> dict:
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
        r = _api_call(base_url, model_id,
                       [{"role": "user", "content": "What is 2+2? Reply with just the number."}])
        content = r["choices"][0]["message"]["content"]
        if "4" in content:
            return TestResult("plain_chat", TestStatus.PASS,
                              duration_ms=(time.time() - t0) * 1000)
        return TestResult("plain_chat", TestStatus.FAIL,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"Expected '4', got: {content[:80]}")
    except Exception as e:
        return TestResult("plain_chat", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_single_tool_call(base_url: str, model_id: str) -> TestResult:
    """Model produces a structured tool call."""
    t0 = time.time()
    try:
        r = _api_call(base_url, model_id,
                       [{"role": "user", "content": "Read the file /etc/hostname"}],
                       tools=BASIC_TOOLS)
        msg = r["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return TestResult("single_tool_call", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message="No tool_calls in response")
        tc = msg["tool_calls"][0]
        if tc["function"]["name"] != "read_file":
            return TestResult("single_tool_call", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"Wrong tool: {tc['function']['name']}")
        return TestResult("single_tool_call", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000)
    except Exception as e:
        return TestResult("single_tool_call", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_tool_choice(base_url: str, model_id: str) -> TestResult:
    """Model picks the right tool from multiple options."""
    t0 = time.time()
    try:
        r = _api_call(base_url, model_id,
                       [{"role": "user", "content": "Run the command 'echo hello'"}],
                       tools=BASIC_TOOLS)
        msg = r["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return TestResult("tool_choice", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message="No tool_calls")
        name = msg["tool_calls"][0]["function"]["name"]
        if name != "terminal":
            return TestResult("tool_choice", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"Wrong tool: {name}, expected terminal")
        return TestResult("tool_choice", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000)
    except Exception as e:
        return TestResult("tool_choice", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_multi_turn_tool(base_url: str, model_id: str) -> TestResult:
    """Multi-turn: tool call → tool result → follow-up."""
    t0 = time.time()
    try:
        r1 = _api_call(base_url, model_id,
                        [{"role": "user", "content": "Read /etc/hosts"}],
                        tools=BASIC_TOOLS)
        msg1 = r1["choices"][0]["message"]
        if not msg1.get("tool_calls"):
            return TestResult("multi_turn_tool", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message="First turn should trigger tool call")
        r2 = _api_call(base_url, model_id, [
            {"role": "user", "content": "Read /etc/hosts"},
            {"role": "assistant", "content": None, "tool_calls": msg1["tool_calls"]},
            {"role": "tool", "tool_call_id": msg1["tool_calls"][0]["id"],
             "content": "127.0.0.1 localhost\n::1 localhost"},
            {"role": "user", "content": "What IP addresses are in that file?"},
        ], tools=BASIC_TOOLS)
        content = r2["choices"][0]["message"]["content"]
        if "127.0.0.1" in content or "localhost" in content:
            return TestResult("multi_turn_tool", TestStatus.PASS,
                              duration_ms=(time.time() - t0) * 1000)
        return TestResult("multi_turn_tool", TestStatus.FAIL,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"Bad follow-up: {content[:80]}")
    except Exception as e:
        return TestResult("multi_turn_tool", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_no_tool_leak(base_url: str, model_id: str) -> TestResult:
    """No raw tool markup leaks into content."""
    t0 = time.time()
    try:
        r = _api_call(base_url, model_id,
                       [{"role": "user", "content": "Use the terminal to run 'echo test'"}],
                       tools=BASIC_TOOLS)
        content = r["choices"][0]["message"].get("content", "")
        leaks = []
        for marker in ["<tool_call>", "<function=", "<|im_end|>", "<|tool_call|>"]:
            if marker in content:
                leaks.append(marker)
        if leaks:
            return TestResult("no_tool_leak", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"Leaked: {leaks}")
        return TestResult("no_tool_leak", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000)
    except Exception as e:
        return TestResult("no_tool_leak", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_many_tools(base_url: str, model_id: str, num_tools: int) -> TestResult:
    """Correct tool selection with many tools injected."""
    t0 = time.time()
    try:
        many_tools = []
        for i in range(num_tools):
            many_tools.append({
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
            })
        many_tools.extend(BASIC_TOOLS)

        r = _api_call(base_url, model_id,
                       [{"role": "user", "content": "Run the command 'echo hello'"}],
                       tools=many_tools, max_tokens=500)
        msg = r["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return TestResult("many_tools", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"No tool call with {len(many_tools)} tools")
        name = msg["tool_calls"][0]["function"]["name"]
        if name == "terminal":
            return TestResult("many_tools", TestStatus.PASS,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"Correct with {len(many_tools)} tools")
        return TestResult("many_tools", TestStatus.FAIL,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"Wrong tool: {name}")
    except Exception as e:
        return TestResult("many_tools", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


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
        with httpx.stream("POST", f"{base_url}/chat/completions",
                           json=payload, timeout=60) as resp:
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
            return TestResult("streaming_tool_call", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message="No tool_call chunks in stream")
        if finish_reason != "tool_calls":
            return TestResult("streaming_tool_call", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"finish_reason={finish_reason}")
        return TestResult("streaming_tool_call", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"{len(tool_chunks)} chunks")
    except Exception as e:
        return TestResult("streaming_tool_call", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_no_tool_needed(base_url: str, model_id: str) -> TestResult:
    """Tools provided but not needed — model answers directly."""
    t0 = time.time()
    try:
        r = _api_call(base_url, model_id,
                       [{"role": "user", "content": "What is the capital of France?"}],
                       tools=BASIC_TOOLS)
        content = r["choices"][0]["message"].get("content", "")
        if "paris" in content.lower():
            return TestResult("no_tool_needed", TestStatus.PASS,
                              duration_ms=(time.time() - t0) * 1000)
        return TestResult("no_tool_needed", TestStatus.FAIL,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"Expected Paris: {content[:80]}")
    except Exception as e:
        return TestResult("no_tool_needed", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_stress_no_leak(base_url: str, model_id: str, rounds: int = 5) -> TestResult:
    """Rapid tool calls — zero tag leaks."""
    t0 = time.time()
    leaked = 0
    try:
        for i in range(rounds):
            r = _api_call(base_url, model_id,
                           [{"role": "user", "content": f"Run: echo test_{i}"}],
                           tools=BASIC_TOOLS, temperature=0.8)
            content = r["choices"][0]["message"].get("content", "")
            if "<tool_call>" in content or "<function=" in content:
                leaked += 1
        if leaked:
            return TestResult("stress_no_leak", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message=f"{leaked}/{rounds} leaked")
        return TestResult("stress_no_leak", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"0/{rounds} leaks")
    except Exception as e:
        return TestResult("stress_no_leak", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_streaming_basic(base_url: str, model_id: str) -> TestResult:
    """SSE streaming produces valid chunks."""
    t0 = time.time()
    try:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_tokens": 50,
            "stream": True,
        }
        chunks = 0
        content = ""
        with httpx.stream("POST", f"{base_url}/chat/completions",
                           json=payload, timeout=30) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                # Accept content OR reasoning deltas (thinking models)
                text = delta.get("content") or delta.get("reasoning")
                if text:
                    content += text
                    chunks += 1

        if chunks == 0:
            return TestResult("streaming_basic", TestStatus.FAIL,
                              duration_ms=(time.time() - t0) * 1000,
                              message="No content/reasoning chunks")
        return TestResult("streaming_basic", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"{chunks} chunks")
    except Exception as e:
        return TestResult("streaming_basic", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


def _test_tag_suppression(base_url: str, model_id: str,
                           extra_tags: list[tuple[str, str]]) -> TestResult:
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
                return TestResult("tag_suppression", TestStatus.FAIL,
                                  duration_ms=(time.time() - t0) * 1000,
                                  message=f"Tag {open_tag!r} not suppressed")
            # Reset for next tag
            f = StreamingToolCallFilter(extra_tags=extra_tags)

        return TestResult("tag_suppression", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000,
                          message=f"{len(extra_tags)} tag pairs verified")
    except Exception as e:
        return TestResult("tag_suppression", TestStatus.ERROR,
                          duration_ms=(time.time() - t0) * 1000, message=str(e))


# ---------------------------------------------------------------------------
# E2E tests (require agent binary)
# ---------------------------------------------------------------------------

def _agent_query(binary: str, query_cmd: str, query: str,
                 timeout: int = 120) -> tuple[str | None, str | None]:
    """Run a single agent query. Returns (output, error)."""
    import shlex

    if not os.path.exists(os.path.expanduser(binary)):
        return None, f"Binary not found: {binary}"

    # Substitute {query} placeholder and parse with shlex to respect quotes
    cmd_str = query_cmd.replace("{query}", query)
    try:
        cmd_parts = shlex.split(cmd_str)
    except ValueError:
        # Fallback: simple split if shlex can't parse
        cmd_parts = cmd_str.split()
    # Replace first part with full binary path
    cmd_parts[0] = os.path.expanduser(binary)

    try:
        proc = subprocess.run(
            cmd_parts,
            capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
        output = proc.stdout + proc.stderr
        if "error" in output.lower() and "HTTP 4" in output:
            return None, "Agent error: server issue"
        return output, None
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def _test_e2e_chat(binary: str, query_cmd: str, timeout: int) -> TestResult:
    """Agent basic chat."""
    t0 = time.time()
    out, err = _agent_query(binary, query_cmd,
                             "What is 2+2? Reply with just the number.", timeout)
    if err:
        status = TestStatus.SKIP if "not found" in (err or "") else TestStatus.ERROR
        return TestResult("e2e_chat", status,
                          duration_ms=(time.time() - t0) * 1000,
                          message=err, category="e2e")
    if "4" in (out or ""):
        return TestResult("e2e_chat", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000, category="e2e")
    return TestResult("e2e_chat", TestStatus.FAIL,
                      duration_ms=(time.time() - t0) * 1000,
                      message=f"Expected '4': {(out or '')[:80]}", category="e2e")


def _test_e2e_file_read(binary: str, query_cmd: str, timeout: int) -> TestResult:
    """Agent reads a file via tool call."""
    t0 = time.time()
    out, err = _agent_query(binary, query_cmd,
                             "Read the first line of pyproject.toml", timeout)
    if err:
        status = TestStatus.SKIP if "not found" in (err or "") else TestStatus.ERROR
        return TestResult("e2e_file_read", status,
                          duration_ms=(time.time() - t0) * 1000,
                          message=err, category="e2e")
    if "build" in (out or "").lower() or "project" in (out or "").lower():
        return TestResult("e2e_file_read", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000, category="e2e")
    return TestResult("e2e_file_read", TestStatus.FAIL,
                      duration_ms=(time.time() - t0) * 1000,
                      message=f"Unexpected: {(out or '')[:80]}", category="e2e")


def _test_e2e_terminal(binary: str, query_cmd: str,
                        timeout: int, agent_name: str) -> TestResult:
    """Agent runs a shell command."""
    t0 = time.time()
    marker = f"rapidmlx_{agent_name}_test"
    out, err = _agent_query(binary, query_cmd,
                             f"Run 'echo {marker}' and show me the output", timeout)
    if err:
        status = TestStatus.SKIP if "not found" in (err or "") else TestStatus.ERROR
        return TestResult("e2e_terminal", status,
                          duration_ms=(time.time() - t0) * 1000,
                          message=err, category="e2e")
    if marker in (out or ""):
        return TestResult("e2e_terminal", TestStatus.PASS,
                          duration_ms=(time.time() - t0) * 1000, category="e2e")
    return TestResult("e2e_terminal", TestStatus.FAIL,
                      duration_ms=(time.time() - t0) * 1000,
                      message=f"Missing marker: {(out or '')[:80]}", category="e2e")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class AgentTestRunner:
    """Dynamically builds and runs tests based on an agent profile."""

    def __init__(self, profile: AgentProfile, base_url: str = "http://localhost:8000/v1",
                 model_id: str | None = None, agent_version: str | None = None):
        self.profile = profile
        self.base_url = base_url
        self.agent_version = agent_version

        # Auto-detect model from server if not specified
        if model_id:
            self.model_id = model_id
        else:
            try:
                resp = httpx.get(f"{base_url}/models", timeout=5)
                self.model_id = resp.json()["data"][0]["id"]
            except Exception:
                self.model_id = "default"

    def _server_available(self) -> bool:
        """Check if the Rapid-MLX server is running."""
        try:
            r = httpx.get(f"{self.base_url.rstrip('/').rsplit('/v1', 1)[0]}/health",
                          timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _agent_binary_available(self) -> bool:
        """Check if the agent binary is installed."""
        testing = self.profile.get_testing_for_version(self.agent_version)
        if not testing.binary:
            return False
        return os.path.exists(os.path.expanduser(testing.binary))

    def build_test_plan(self) -> list[str]:
        """Build the list of test names that will run for this profile."""
        tests = ["plain_chat"]

        if self.profile.needs_function_calling:
            tests += [
                "single_tool_call", "tool_choice", "multi_turn_tool",
                "no_tool_leak", "no_tool_needed",
            ]
            if self.profile.needs_streaming:
                tests.append("streaming_tool_call")

        streaming = self.profile.get_streaming_for_version(self.agent_version)
        if streaming.max_tools and streaming.max_tools > 10:
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
            report.results.append(TestResult(
                "server_check", TestStatus.ERROR,
                message="Rapid-MLX server not running",
            ))
            return report

        t0 = time.time()
        streaming = self.profile.get_streaming_for_version(self.agent_version)
        testing = self.profile.get_testing_for_version(self.agent_version)

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
                    _test_streaming_tool_call(self.base_url, self.model_id))

        if streaming.max_tools and streaming.max_tools > 10:
            report.results.append(
                _test_many_tools(self.base_url, self.model_id, streaming.max_tools))

        if streaming.extra_tool_tags:
            report.results.append(
                _test_tag_suppression(self.base_url, self.model_id,
                                      streaming.extra_tool_tags))

        if self.profile.needs_streaming:
            report.results.append(_test_streaming_basic(self.base_url, self.model_id))

        if self.profile.needs_function_calling:
            report.results.append(_test_stress_no_leak(self.base_url, self.model_id))

        # --- E2E tests ---
        if self._agent_binary_available() and testing.binary and testing.query_cmd:
            binary = os.path.expanduser(testing.binary)
            report.results.append(
                _test_e2e_chat(binary, testing.query_cmd, testing.query_timeout))
            if self.profile.needs_function_calling:
                report.results.append(
                    _test_e2e_file_read(binary, testing.query_cmd, testing.query_timeout))
                report.results.append(
                    _test_e2e_terminal(binary, testing.query_cmd,
                                       testing.query_timeout, self.profile.name))
        elif testing.binary:
            report.results.append(TestResult(
                "e2e_tests", TestStatus.SKIP,
                message=f"Binary not found: {testing.binary}", category="e2e"))

        # --- Framework-specific tests ---
        if testing.specific_tests:
            specific_results = self._run_specific_tests(testing.specific_tests)
            report.results.extend(specific_results)

        report.total_duration_ms = (time.time() - t0) * 1000
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
        import importlib.util
        from pathlib import Path

        # Find the test file
        test_dir = Path(__file__).parent.parent.parent / "tests" / "integrations"
        test_path = test_dir / test_module_name
        if not test_path.exists():
            return [TestResult(
                f"specific:{test_module_name}", TestStatus.SKIP,
                message=f"Test file not found: {test_path}", category="specific",
            )]

        t0 = time.time()
        try:
            # Load and execute the test module
            spec = importlib.util.spec_from_file_location(
                f"specific_test_{test_module_name}", test_path,
            )
            mod = importlib.util.module_from_spec(spec)

            # Set base URL env var so specific test modules use the right server
            import sys
            os.environ["RAPID_MLX_BASE_URL"] = self.base_url

            # Suppress sys.exit (test modules call exit() at the end)
            original_exit = sys.exit
            exec_error = None
            sys.exit = lambda *a: None
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass  # expected — test modules call exit()
            except Exception as e:
                exec_error = e
            finally:
                sys.exit = original_exit

            # If module failed to execute, report it as an error
            if exec_error is not None:
                return [TestResult(
                    f"specific:{test_module_name}", TestStatus.ERROR,
                    duration_ms=(time.time() - t0) * 1000,
                    message=f"Module execution failed: {exec_error!s:.120}",
                    category="specific",
                )]

            # Extract results dict
            results_dict = getattr(mod, "results", {})
            if not results_dict:
                return [TestResult(
                    f"specific:{test_module_name}", TestStatus.ERROR,
                    duration_ms=(time.time() - t0) * 1000,
                    message="No test results found (missing 'results' dict or all tests skipped)",
                    category="specific",
                )]

            test_results = []
            for name, status in results_dict.items():
                if status == "PASS":
                    test_results.append(TestResult(
                        name, TestStatus.PASS,
                        duration_ms=(time.time() - t0) * 1000 / max(len(results_dict), 1),
                        category="specific",
                    ))
                else:
                    msg = status.replace("FAIL: ", "") if status.startswith("FAIL:") else status
                    test_results.append(TestResult(
                        name, TestStatus.FAIL,
                        duration_ms=(time.time() - t0) * 1000 / max(len(results_dict), 1),
                        message=msg[:120], category="specific",
                    ))
            return test_results

        except Exception as e:
            return [TestResult(
                f"specific:{test_module_name}", TestStatus.ERROR,
                duration_ms=(time.time() - t0) * 1000,
                message=str(e)[:120], category="specific",
            )]
