"""Hermes Agent integration tests against local Rapid-MLX server.

Tests the full Hermes Agent → Rapid-MLX pipeline using the OpenAI-compatible
API. Covers chat, tool calling (single, multi-step, parallel), streaming,
reasoning, and edge cases with many tools (Hermes injects 60+ tools).

Requirements:
    1. Rapid-MLX server running: rapid-mlx serve <MODEL> --port 8000
    2. Hermes Agent installed: pip install hermes-agent (or from source)
    3. ~/.hermes/config.yaml pointing to localhost:8000

Tested models (Hermes community favorites):
    - mlx-community/Qwen3.5-4B-MLX-4bit  (fast, budget)
    - mlx-community/Qwen3.5-9B-4bit      (recommended)
    - mlx-community/Qwen3.5-27B-4bit     (quality)
    - mlx-community/Qwen3.5-35B-A3B-4bit (MoE, best quality/speed)
"""

import json
import os
import subprocess
import sys
import time
import httpx

BASE_URL = os.environ.get("RAPID_MLX_BASE_URL", "http://localhost:8000/v1")
# Auto-detect model from server
try:
    resp = httpx.get(f"{BASE_URL}/models", timeout=5)
    MODEL_ID = resp.json()["data"][0]["id"]
except Exception:
    MODEL_ID = "default"

HERMES_BIN = os.environ.get(
    "HERMES_BIN",
    # Common install locations
    os.path.expanduser("~/.hermes/venv/bin/hermes")
    if os.path.exists(os.path.expanduser("~/.hermes/venv/bin/hermes"))
    else "/tmp/hermes-agent/.venv/bin/hermes",
)

results = {}


def api_call(messages, tools=None, stream=False, max_tokens=300, temperature=0.3):
    """Direct API call to Rapid-MLX server."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def ensure_hermes_config():
    """Update ~/.hermes/config.yaml to point to the current server/model."""
    config_dir = os.path.expanduser("~/.hermes")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "config.yaml")
    config = (
        f"model:\n"
        f'  provider: "custom"\n'
        f'  default: "{MODEL_ID}"\n'
        f'  base_url: "{BASE_URL}"\n'
        f"  context_length: 32768\n"
        f"  max_tokens: 4096\n"
    )
    with open(config_path, "w") as f:
        f.write(config)


def hermes_query(query, timeout_sec=120):
    """Run a single Hermes query in non-interactive mode."""
    if not os.path.exists(HERMES_BIN):
        return None, "SKIP: hermes binary not found"
    try:
        proc = subprocess.run(
            [HERMES_BIN, "chat", "-q", query, "-Q"],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=os.getcwd(),
        )
        output = proc.stdout + proc.stderr
        # Detect Hermes-level errors
        if "Non-retryable error" in output or "HTTP 404" in output:
            return None, f"Hermes error: model mismatch or server down"
        return output, None
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def run_test(name, fn):
    """Run a test function and record the result."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    try:
        fn()
        results[name] = "PASS"
        print(f"  ✅ PASS")
    except AssertionError as e:
        results[name] = f"FAIL: {e}"
        print(f"  ❌ FAIL: {e}")
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"  ❌ ERROR: {e}")


# =============================================================================
# API-level tests (no Hermes binary needed)
# =============================================================================

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
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def test_api_plain_chat():
    """Basic chat without tools."""
    r = api_call([{"role": "user", "content": "What is 2+2? Reply with just the number."}])
    content = r["choices"][0]["message"]["content"]
    assert "4" in content, f"Expected '4' in: {content[:100]}"
    print(f"  Response: {content[:80]}")


def test_api_single_tool_call():
    """Single tool call with structured response."""
    r = api_call(
        [{"role": "user", "content": "Read the file /etc/hostname"}],
        tools=BASIC_TOOLS,
    )
    msg = r["choices"][0]["message"]
    assert msg.get("tool_calls"), f"No tool_calls in response: {msg}"
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "read_file", f"Wrong tool: {tc['function']['name']}"
    args = json.loads(tc["function"]["arguments"])
    assert "hostname" in args.get("path", "").lower(), f"Wrong path: {args}"
    print(f"  Tool: {tc['function']['name']}({args})")


def test_api_tool_choice():
    """Model correctly picks the right tool from multiple options."""
    r = api_call(
        [{"role": "user", "content": "Run the command 'echo hello'"}],
        tools=BASIC_TOOLS,
    )
    msg = r["choices"][0]["message"]
    assert msg.get("tool_calls"), f"No tool_calls: {msg.get('content', '')[:100]}"
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "terminal", f"Wrong tool: {tc['function']['name']}"
    print(f"  Correctly chose: terminal")


def test_api_multi_turn_tool():
    """Multi-turn: tool call → tool result → follow-up."""
    # First turn: ask to read a file
    r1 = api_call(
        [{"role": "user", "content": "Read /etc/hosts"}],
        tools=BASIC_TOOLS,
    )
    msg1 = r1["choices"][0]["message"]
    assert msg1.get("tool_calls"), "First turn should trigger tool call"

    # Second turn: provide tool result, ask follow-up
    r2 = api_call(
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
    content2 = r2["choices"][0]["message"]["content"]
    assert "127.0.0.1" in content2 or "localhost" in content2, f"Bad follow-up: {content2[:100]}"
    print(f"  Multi-turn response: {content2[:80]}")


def test_api_no_tool_leak():
    """Ensure no raw <tool_call> tags leak into content."""
    r = api_call(
        [{"role": "user", "content": "Use the terminal to run 'echo test'"}],
        tools=BASIC_TOOLS,
    )
    msg = r["choices"][0]["message"]
    content = msg.get("content", "")
    assert "<tool_call>" not in content, f"Tag leak in content: {content[:200]}"
    assert "<function=" not in content, f"Function tag leak: {content[:200]}"
    assert "<|im_end|>" not in content, f"EOS leak: {content[:200]}"
    print(f"  No tag leaks detected")


def test_api_many_tools():
    """Test with 20+ tools (simulating Hermes' 62-tool setup)."""
    # Generate 20 dummy tools
    many_tools = []
    for i in range(20):
        many_tools.append({
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool number {i} that does something",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                },
            },
        })
    # Add the real tools
    many_tools.extend(BASIC_TOOLS)

    r = api_call(
        [{"role": "user", "content": "Run the command 'echo hello_many_tools'"}],
        tools=many_tools,
    )
    msg = r["choices"][0]["message"]
    # Should still pick the right tool from 23 options
    assert msg.get("tool_calls"), f"No tool call with {len(many_tools)} tools"
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "terminal", f"Wrong tool: {tc['function']['name']}"
    prompt_tokens = r.get("usage", {}).get("prompt_tokens", 0)
    print(f"  Correct tool with {len(many_tools)} tools, prompt_tokens={prompt_tokens}")


def test_api_streaming_tool_call():
    """Streaming mode: tool calls arrive as structured deltas."""
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Read the file /etc/hosts"}],
        "tools": BASIC_TOOLS,
        "max_tokens": 200,
        "stream": True,
    }
    with httpx.stream(
        "POST", f"{BASE_URL}/chat/completions", json=payload, timeout=60
    ) as resp:
        tool_call_chunks = []
        content_chunks = []
        finish_reason = None
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            delta = data["choices"][0].get("delta", {})
            if "tool_calls" in delta:
                tool_call_chunks.append(delta["tool_calls"])
            if delta.get("content"):
                c = delta["content"]
                assert "<tool_call>" not in c, f"Tag leak in stream: {c}"
                content_chunks.append(c)
            if data["choices"][0].get("finish_reason"):
                finish_reason = data["choices"][0]["finish_reason"]

    assert tool_call_chunks, "No tool_call chunks in stream"
    assert finish_reason == "tool_calls", f"finish_reason={finish_reason}"
    print(f"  Streaming: {len(tool_call_chunks)} tool chunks, finish={finish_reason}")


def test_api_no_tool_needed():
    """When tools are provided but not needed, model should answer directly."""
    r = api_call(
        [{"role": "user", "content": "What is the capital of France?"}],
        tools=BASIC_TOOLS,
    )
    msg = r["choices"][0]["message"]
    content = msg.get("content", "")
    assert "Paris" in content or "paris" in content.lower(), f"Expected Paris: {content[:100]}"
    # Should NOT call a tool for a general knowledge question
    if msg.get("tool_calls"):
        print(f"  ⚠️ Unnecessary tool call: {msg['tool_calls'][0]['function']['name']}")
    else:
        print(f"  Correctly answered without tools: {content[:60]}")


def test_api_parallel_tool_calls():
    """Model can request multiple tool calls in one response."""
    r = api_call(
        [{"role": "user", "content": "Read both /etc/hosts and /etc/resolv.conf at the same time"}],
        tools=BASIC_TOOLS,
        max_tokens=500,
    )
    msg = r["choices"][0]["message"]
    if msg.get("tool_calls") and len(msg["tool_calls"]) >= 2:
        names = [tc["function"]["name"] for tc in msg["tool_calls"]]
        print(f"  Parallel calls: {names}")
    elif msg.get("tool_calls"):
        print(f"  Single call (model chose sequential): {msg['tool_calls'][0]['function']['name']}")
    else:
        print(f"  No tool calls (answered directly)")
    # Either way, no tag leaks
    content = msg.get("content", "")
    assert "<tool_call>" not in content, f"Tag leak: {content[:100]}"


def test_api_stress_no_leak():
    """10 rapid tool calls — zero tag leaks."""
    leaked = 0
    for i in range(10):
        r = api_call(
            [{"role": "user", "content": f"Run: echo test_{i}"}],
            tools=BASIC_TOOLS,
            temperature=0.8,
        )
        content = r["choices"][0]["message"].get("content", "")
        if "<tool_call>" in content or "<function=" in content:
            leaked += 1
    assert leaked == 0, f"{leaked}/10 requests had tag leaks"
    print(f"  0/10 tag leaks at temperature=0.8")


# =============================================================================
# Hermes E2E tests (requires hermes binary)
# =============================================================================

def test_hermes_chat():
    """Basic Hermes chat (no tool use)."""
    out, err = hermes_query("What is 2+2? Reply with just the number.")
    if err:
        assert False, err
    assert "4" in out, f"Expected 4 in: {out[:100]}"
    print(f"  Hermes output: {out.strip()[:80]}")


def test_hermes_read_file():
    """Hermes reads a file via tool call."""
    out, err = hermes_query("Read the first line of pyproject.toml")
    if err:
        assert False, err
    assert "build" in out.lower() or "project" in out.lower(), f"Unexpected: {out[:100]}"
    print(f"  Hermes read_file: {out.strip()[:80]}")


def test_hermes_terminal():
    """Hermes runs a shell command."""
    out, err = hermes_query("Run 'echo rapid_mlx_hermes_test' and show me the output")
    if err:
        assert False, err
    assert "rapid_mlx_hermes_test" in out, f"Command output missing: {out[:100]}"
    print(f"  Hermes terminal: OK")


def test_hermes_search():
    """Hermes searches for files."""
    out, err = hermes_query("Search for files named 'aliases.json' in this project")
    if err:
        assert False, err
    assert "aliases" in out.lower(), f"Search failed: {out[:100]}"
    print(f"  Hermes search: OK")


def test_hermes_multi_step():
    """Hermes does a multi-step task (search → read → analyze)."""
    out, err = hermes_query(
        "Find the file aliases.json, read it, and tell me how many entries it has",
        timeout_sec=180,
    )
    if err:
        assert False, err
    # Should mention a number (we have ~22 aliases)
    assert any(str(n) in out for n in range(15, 30)), f"No count found: {out[:200]}"
    print(f"  Hermes multi-step: OK")


# =============================================================================
# Deep agentic tests (requires hermes binary, tests real workflows)
# =============================================================================

def test_hermes_write_and_run():
    """Hermes writes a Python script and executes it (full agent loop)."""
    out, err = hermes_query(
        "Create a Python script at /tmp/hermes_test_fib.py that prints the first "
        "10 fibonacci numbers as a comma-separated list, then run it and show output",
        timeout_sec=120,
    )
    if err:
        assert False, err
    # Verify via Hermes output or by checking the file directly
    import subprocess
    result = subprocess.run(
        ["python3", "/tmp/hermes_test_fib.py"],
        capture_output=True, text=True, timeout=10,
    )
    fib_out = result.stdout + out
    assert any(str(n) in fib_out for n in [8, 13, 21, 34]), f"Fibonacci missing: {fib_out[:200]}"
    print(f"  Write+run: fibonacci script works")


def test_hermes_code_with_tests():
    """Hermes writes code + tests and runs them (complex agentic workflow)."""
    out, err = hermes_query(
        "Create /tmp/hermes_calc.py with add and multiply functions. "
        "Create /tmp/hermes_test_calc.py with pytest tests for both. "
        "Then run the tests.",
        timeout_sec=180,
    )
    if err:
        assert False, err
    # Verify the files exist and tests pass
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "pytest", "/tmp/hermes_test_calc.py", "-v"],
        capture_output=True, text=True, timeout=30,
    )
    assert "passed" in result.stdout.lower(), f"Tests failed: {result.stdout[:200]}{result.stderr[:200]}"
    print(f"  Code+tests: pytest passing")


def test_hermes_code_review():
    """Hermes reads a file and gives a code review suggestion."""
    out, err = hermes_query(
        "Read vllm_mlx/model_auto_config.py and suggest one specific improvement. Be concise.",
        timeout_sec=120,
    )
    if err:
        assert False, err
    # Should mention something about the code (patterns, config, etc.)
    assert len(out) > 50, f"Response too short for a code review: {out[:100]}"
    assert "model" in out.lower() or "pattern" in out.lower() or "config" in out.lower(), \
        f"Doesn't look like a code review: {out[:100]}"
    print(f"  Code review: {out.strip()[:80]}")


def test_hermes_git_analysis():
    """Hermes analyzes git history."""
    out, err = hermes_query(
        "Check the git log of this repo and tell me the last 3 commit messages",
        timeout_sec=120,
    )
    if err:
        assert False, err
    assert "commit" in out.lower() or "hermes" in out.lower() or "feat" in out.lower() or "fix" in out.lower(), \
        f"No git info: {out[:200]}"
    print(f"  Git analysis: OK")


def test_hermes_patch_file():
    """Hermes edits a file using the patch tool."""
    # Create a test file first
    test_file = "/tmp/hermes_patch_test.py"
    with open(test_file, "w") as f:
        f.write("def hello():\n    return 'hello'\n")

    out, err = hermes_query(
        f"Add a docstring 'Say hello.' to the hello function in {test_file}",
        timeout_sec=120,
    )
    if err:
        assert False, err
    # Verify the file was modified
    with open(test_file) as f:
        content = f.read()
    assert "docstring" in content.lower() or "say hello" in content.lower() or '"""' in content, \
        f"Patch not applied: {content[:200]}"
    print(f"  Patch: file edited successfully")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Rapid-MLX Hermes Integration Tests")
    print(f"Server: {BASE_URL}")
    print(f"Model:  {MODEL_ID}")
    print(f"Hermes: {HERMES_BIN}")
    print(f"{'='*60}")

    t0 = time.time()

    # API-level tests (always run)
    run_test("api_plain_chat", test_api_plain_chat)
    run_test("api_single_tool_call", test_api_single_tool_call)
    run_test("api_tool_choice", test_api_tool_choice)
    run_test("api_multi_turn_tool", test_api_multi_turn_tool)
    run_test("api_no_tool_leak", test_api_no_tool_leak)
    run_test("api_many_tools", test_api_many_tools)
    run_test("api_streaming_tool_call", test_api_streaming_tool_call)
    run_test("api_no_tool_needed", test_api_no_tool_needed)
    run_test("api_parallel_tool_calls", test_api_parallel_tool_calls)
    run_test("api_stress_no_leak", test_api_stress_no_leak)

    # Hermes E2E tests (require hermes binary)
    if os.path.exists(HERMES_BIN):
        ensure_hermes_config()
        run_test("hermes_chat", test_hermes_chat)
        run_test("hermes_read_file", test_hermes_read_file)
        run_test("hermes_terminal", test_hermes_terminal)
        run_test("hermes_search", test_hermes_search)
        run_test("hermes_multi_step", test_hermes_multi_step)

        # Deep agentic tests
        run_test("hermes_write_and_run", test_hermes_write_and_run)
        run_test("hermes_code_with_tests", test_hermes_code_with_tests)
        run_test("hermes_code_review", test_hermes_code_review)
        run_test("hermes_git_analysis", test_hermes_git_analysis)
        run_test("hermes_patch_file", test_hermes_patch_file)
    else:
        print(f"\n⚠️ Skipping Hermes E2E tests: {HERMES_BIN} not found")

    elapsed = time.time() - t0
    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v != "PASS")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(results)} passed ({elapsed:.1f}s)")
    print(f"Model:   {MODEL_ID}")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}: {status}")

    if failed:
        sys.exit(1)
