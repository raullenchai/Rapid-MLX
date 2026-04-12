"""Anthropic SDK against rapid-mlx /v1/messages endpoint."""
import os

import httpx as _httpx
from anthropic import Anthropic

_BASE = os.environ.get("RAPID_MLX_BASE_URL", "http://localhost:8000/v1")
try:
    MODEL_ID = _httpx.get(f"{_BASE}/models", timeout=5).json()["data"][0]["id"]
except Exception:
    MODEL_ID = "default"

_BASE_NO_V1 = _BASE.rstrip("/").removesuffix("/v1")

client = Anthropic(
    base_url=_BASE_NO_V1,
    api_key="not-needed",
)

results = {}

# === 1. Plain message ===
print("=== Test 1: Plain message ===")
try:
    r = client.messages.create(
        model=MODEL_ID,
        max_tokens=50,
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    )
    text = r.content[0].text
    assert "4" in text, text
    print(f"PASS: {text[:80]}")
    results["1_plain"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["1_plain"] = f"FAIL: {str(e)[:120]}"

# === 2. System prompt ===
print("\n=== Test 2: System prompt ===")
try:
    r = client.messages.create(
        model=MODEL_ID,
        max_tokens=50,
        system="You are a calculator. Output only the integer result.",
        messages=[{"role": "user", "content": "9 * 8"}],
    )
    text = r.content[0].text
    assert "72" in text, text
    print(f"PASS: {text[:80]}")
    results["2_system"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["2_system"] = f"FAIL: {str(e)[:120]}"

# === 3. Multi-turn conversation ===
print("\n=== Test 3: Multi-turn ===")
try:
    msgs = [
        {"role": "user", "content": "My favorite color is blue. Remember this."},
        {"role": "assistant", "content": "Got it, your favorite color is blue."},
        {"role": "user", "content": "What is my favorite color?"},
    ]
    r = client.messages.create(model=MODEL_ID, max_tokens=80, messages=msgs)
    text = r.content[0].text
    assert "blue" in text.lower(), text
    print(f"PASS: {text[:80]}")
    results["3_multi_turn"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["3_multi_turn"] = f"FAIL: {str(e)[:120]}"

# === 4. Streaming ===
print("\n=== Test 4: Streaming ===")
try:
    chunks = []
    with client.messages.stream(
        model=MODEL_ID,
        max_tokens=80,
        messages=[{"role": "user", "content": "Count from 1 to 5, comma-separated."}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
    full = "".join(chunks)
    assert "1" in full and "5" in full, full
    print(f"PASS: {len(chunks)} chunks, content={full[:80]}")
    results["4_stream"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["4_stream"] = f"FAIL: {str(e)[:120]}"

# === 5. Tool use ===
print("\n=== Test 5: Tool use ===")
try:
    r = client.messages.create(
        model=MODEL_ID,
        max_tokens=200,
        tools=[{
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }],
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    )
    # Find tool_use blocks
    tool_uses = [b for b in r.content if b.type == "tool_use"]
    assert len(tool_uses) > 0, f"No tool_use in {r.content}"
    tu = tool_uses[0]
    assert tu.name == "get_weather", tu.name
    assert "city" in tu.input, tu.input
    assert "tokyo" in tu.input["city"].lower(), tu.input
    print(f"PASS: tool={tu.name}, input={tu.input}")
    results["5_tool"] = "PASS"
except Exception as e:
    print(f"FAIL: {e}")
    results["5_tool"] = f"FAIL: {str(e)[:120]}"

# === Summary ===
print("\n" + "=" * 50)
passed = sum(1 for v in results.values() if v == "PASS")
print(f"Anthropic SDK: {passed}/{len(results)} passed")
for k, v in results.items():
    print(f"  {k}: {v[:120]}")
exit(0 if passed == len(results) else 1)
