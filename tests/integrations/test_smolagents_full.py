"""Thorough smolagents test suite against local rapid-mlx server."""
import os

import httpx as _httpx
from smolagents import CodeAgent, OpenAIServerModel, ToolCallingAgent, tool

_BASE = os.environ.get("RAPID_MLX_BASE_URL", "http://localhost:8000/v1")
try:
    MODEL_ID = _httpx.get(f"{_BASE}/models", timeout=5).json()["data"][0]["id"]
except Exception:
    MODEL_ID = "default"

model = OpenAIServerModel(
    model_id=MODEL_ID,
    api_base=_BASE,
    api_key="not-needed",
)

results = {}

# === 1. CodeAgent simple ===
print("=== Test 1: CodeAgent arithmetic ===")
try:
    agent = CodeAgent(tools=[], model=model, max_steps=3)
    out = agent.run("Compute 12 * 8 and return the integer.")
    assert "96" in str(out), out
    print(f"PASS: {str(out)[:80]}")
    results["1_code_simple"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["1_code_simple"] = f"FAIL: {str(e)[:120]}"

# === 2. CodeAgent with tool ===
print("\n=== Test 2: CodeAgent + custom tool ===")
try:
    @tool
    def get_temp(city: str) -> str:
        """Returns the current temperature for a city.

        Args:
            city: Name of the city.
        """
        return f"The temperature in {city} is 18 degrees Celsius."

    agent = CodeAgent(tools=[get_temp], model=model, max_steps=4)
    out = agent.run("What is the temperature in Tokyo?")
    assert "18" in str(out) or "Tokyo" in str(out), out
    print(f"PASS: {str(out)[:80]}")
    results["2_code_tool"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["2_code_tool"] = f"FAIL: {str(e)[:120]}"

# === 3. ToolCallingAgent (uses tool_calls API instead of code) ===
print("\n=== Test 3: ToolCallingAgent + tool ===")
try:
    @tool
    def add_nums(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: first number
            b: second number
        """
        return a + b

    agent = ToolCallingAgent(tools=[add_nums], model=model, max_steps=4)
    out = agent.run("What is 17 + 25? Use the add_nums tool.")
    assert "42" in str(out), out
    print(f"PASS: {str(out)[:80]}")
    results["3_tool_calling_agent"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["3_tool_calling_agent"] = f"FAIL: {str(e)[:120]}"

# === 4. Multi-tool ToolCallingAgent ===
print("\n=== Test 4: ToolCallingAgent + multiple tools ===")
try:
    @tool
    def square(n: int) -> int:
        """Square a number.

        Args:
            n: number
        """
        return n * n

    @tool
    def double(n: int) -> int:
        """Double a number.

        Args:
            n: number
        """
        return n * 2

    agent = ToolCallingAgent(tools=[square, double], model=model, max_steps=6)
    out = agent.run("What is the square of 6, then doubled? Use the tools.")
    # 6^2 = 36, doubled = 72
    assert "72" in str(out), f"Expected 72, got: {out}"
    print(f"PASS: {str(out)[:80]}")
    results["4_multi_tool"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["4_multi_tool"] = f"FAIL: {str(e)[:120]}"

# === Summary ===
print("\n" + "=" * 50)
passed = sum(1 for v in results.values() if v == "PASS")
print(f"smolagents: {passed}/{len(results)} passed")
for k, v in results.items():
    print(f"  {k}: {v[:120]}")
exit(0 if passed == len(results) else 1)
