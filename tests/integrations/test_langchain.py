"""Thorough LangChain test suite against local rapid-mlx server."""
import os

import httpx as _httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_BASE = os.environ.get("RAPID_MLX_BASE_URL", "http://localhost:8000/v1")
try:
    MODEL_ID = _httpx.get(f"{_BASE}/models", timeout=5).json()["data"][0]["id"]
except Exception:
    MODEL_ID = "default"

llm = ChatOpenAI(
    model=MODEL_ID,
    base_url=_BASE,
    api_key="not-needed",
    temperature=0.0,
)

results = {}

# === 1. Plain invoke ===
print("=== Test 1: Plain invoke ===")
try:
    r = llm.invoke([HumanMessage(content="Reply with just '4' (the digit). What is 2+2?")])
    assert "4" in r.content, r.content
    print(f"PASS: {r.content[:80]}")
    results["1_plain"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["1_plain"] = f"FAIL: {str(e)[:120]}"

# === 2. System + User multi-message ===
print("\n=== Test 2: System + User ===")
try:
    r = llm.invoke([
        SystemMessage(content="You are a calculator. Output ONLY the integer result, nothing else."),
        HumanMessage(content="7 * 8"),
    ])
    assert "56" in r.content, r.content
    print(f"PASS: {r.content[:80]}")
    results["2_system"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["2_system"] = f"FAIL: {str(e)[:120]}"

# === 3. Streaming ===
print("\n=== Test 3: Streaming ===")
try:
    chunks = []
    for chunk in llm.stream([HumanMessage(content="Count 1 to 5, comma-separated.")]):
        chunks.append(chunk.content)
    full = "".join(chunks)
    assert "1" in full and "5" in full, full
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    print(f"PASS: {len(chunks)} chunks, content={full[:80]}")
    results["3_stream"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["3_stream"] = f"FAIL: {str(e)[:120]}"

# === 4. Tool calling (single tool) ===
print("\n=== Test 4: Single tool call ===")
try:
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"sunny, 22C in {city}"

    llm_with_tools = llm.bind_tools([get_weather])
    r = llm_with_tools.invoke([HumanMessage(content="What's the weather in Paris?")])
    tool_calls = r.tool_calls if hasattr(r, "tool_calls") else []
    assert len(tool_calls) > 0, f"No tool calls. content={r.content[:200]}"
    tc = tool_calls[0]
    assert tc["name"] == "get_weather", tc
    assert "city" in tc["args"], tc
    assert "paris" in tc["args"]["city"].lower(), tc
    print(f"PASS: tool={tc['name']}, args={tc['args']}")
    results["4_tool"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["4_tool"] = f"FAIL: {str(e)[:120]}"

# === 5. Multi-tool (model picks one) ===
print("\n=== Test 5: Multi-tool selection ===")
try:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    llm_multi = llm.bind_tools([add, multiply])
    r = llm_multi.invoke([HumanMessage(content="What is 6 multiplied by 7? Use a tool.")])
    tool_calls = r.tool_calls if hasattr(r, "tool_calls") else []
    assert len(tool_calls) > 0, f"No tool calls. content={r.content[:200]}"
    tc = tool_calls[0]
    assert tc["name"] == "multiply", f"Expected multiply, got {tc['name']}"
    assert tc["args"].get("a") in (6, 7), tc
    assert tc["args"].get("b") in (6, 7), tc
    print(f"PASS: tool={tc['name']}, args={tc['args']}")
    results["5_multi_tool"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["5_multi_tool"] = f"FAIL: {str(e)[:120]}"

# === 6. Structured output ===
print("\n=== Test 6: Structured output (with_structured_output) ===")
try:
    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age in years")

    structured_llm = llm.with_structured_output(Person)
    r = structured_llm.invoke([HumanMessage(content="Extract: 'Bob is 42 years old'")])
    assert isinstance(r, Person), type(r)
    assert r.name.lower() == "bob", r.name
    assert r.age == 42, r.age
    print(f"PASS: {r}")
    results["6_structured"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["6_structured"] = f"FAIL: {str(e)[:120]}"

# === Summary ===
print("\n" + "=" * 50)
passed = sum(1 for v in results.values() if v == "PASS")
print(f"LangChain: {passed}/{len(results)} passed")
for k, v in results.items():
    print(f"  {k}: {v[:120]}")
exit(0 if passed == len(results) else 1)
