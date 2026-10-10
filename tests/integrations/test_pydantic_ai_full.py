"""Thorough PydanticAI test suite against local rapid-mlx server."""

import asyncio
import os

import httpx as _httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def run_tests(model, loop):
    results = {}

    # === 1. Plain completion ===
    print("=== Test 1: Plain completion ===")
    try:
        agent = Agent(model)
        r = agent.run_sync("What is 2+2? Reply with just the number.")
        assert "4" in r.output, r.output
        print(f"PASS: {r.output[:80]}")
        results["1_plain"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["1_plain"] = f"FAIL: {e}"

    # === 2. Streaming ===
    print("\n=== Test 2: Streaming ===")
    try:

        async def stream_test():
            agent = Agent(model)
            chunks = []
            async with agent.run_stream(
                "Count from 1 to 5, separated by commas."
            ) as result:
                async for delta in result.stream_text(delta=True):
                    chunks.append(delta)
            return "".join(chunks)

        out = loop.run_until_complete(stream_test())
        assert len(out) > 5, f"Too short: {out}"
        assert any(d in out for d in ["1", "2", "3"]), out
        print(f"PASS: chunks={len(out)} chars, output={out[:80]}")
        results["2_stream"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["2_stream"] = f"FAIL: {e}"

    # === 3. Structured output (BaseModel) ===
    print("\n=== Test 3: Structured output ===")
    try:

        class Person(BaseModel):
            name: str
            age: int

        # Keep the output-tool assertion deterministic and bound any retry.
        # Be explicit about every required field: this test runs after the
        # stress and two other agent suites in the PR lane, while a failure is
        # confirmed against a fresh base server.  The terse "Extract" prompt
        # made Qwen3.5 occasionally emit only ``age`` in the warmed lane even
        # at temperature=0, producing a repeatable false PR/base differential.
        # The contract under test is schema transport, not whether the model
        # can infer which details to copy from an underspecified instruction.
        agent = Agent(
            model,
            output_type=Person,
            model_settings={"temperature": 0.0, "max_tokens": 256},
        )
        r = agent.run_sync(
            "Call the final_result output tool with both required fields: "
            "name='Alice' and age=30. Do not omit either field."
        )
        assert isinstance(r.output, Person), type(r.output)
        assert r.output.name.lower() == "alice", r.output.name
        assert r.output.age == 30, r.output.age
        print(f"PASS: {r.output}")
        results["3_structured"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["3_structured"] = f"FAIL: {e}"

    # === 4. Tool calling (single) ===
    print("\n=== Test 4: Single tool call ===")
    try:
        agent = Agent(model)

        @agent.tool_plain
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"sunny, 22C in {city}"

        r = agent.run_sync("What's the weather in Paris?")
        assert "Paris" in r.output or "22" in r.output, r.output
        called = any("get_weather" in str(m) for m in r.all_messages())
        assert called, "tool not called"
        print(f"PASS: {r.output[:80]}")
        results["4_tool_single"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["4_tool_single"] = f"FAIL: {e}"

    # === 5. Multi-turn conversation ===
    print("\n=== Test 5: Multi-turn ===")
    try:
        # PydanticAI defaults max_tokens to ~1024; on verbose 4B-class models the
        # second turn can spill past that. Raise the cap so the SDK contract test
        # checks rapid-mlx behaviour, not PydanticAI's default ceiling.
        agent = Agent(model, model_settings={"max_tokens": 2048})
        r1 = agent.run_sync("My name is Bob. Remember this.")
        r2 = agent.run_sync("What is my name?", message_history=r1.all_messages())
        assert "bob" in r2.output.lower(), r2.output
        print(f"PASS: turn2 = {r2.output[:80]}")
        results["5_multi_turn"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["5_multi_turn"] = f"FAIL: {e}"

    # === 6. Multiple tools, sequential ===
    print("\n=== Test 6: Multiple tools ===")
    try:
        # Sequential tool calls accumulate output across two tool-call round trips;
        # PydanticAI's default max_tokens ceiling kicks in before the final answer
        # on small models. Same fix as test 5.
        agent = Agent(model, model_settings={"max_tokens": 2048})

        @agent.tool_plain
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @agent.tool_plain
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        r = agent.run_sync("Compute (3+4)*5 using the tools. Show the result.")
        assert "35" in r.output, r.output
        called_tools = [str(m) for m in r.all_messages()]
        add_called = any("add" in t for t in called_tools)
        mul_called = any("multiply" in t for t in called_tools)
        assert add_called and mul_called, f"add={add_called} mul={mul_called}"
        print(f"PASS: {r.output[:80]}")
        results["6_multi_tool"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        results["6_multi_tool"] = f"FAIL: {e}"

    # === Summary ===
    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"PydanticAI: {passed}/{total} passed")
    for k, v in results.items():
        print(f"  {k}: {v[:120]}")
    return 0 if passed == total else 1


def main():
    base_url = os.environ.get("RAPID_MLX_BASE_URL", "http://localhost:8000/v1")
    try:
        model_id = _httpx.get(f"{base_url}/models", timeout=5).json()["data"][0]["id"]
    except Exception:
        model_id = "default"

    # run_sync uses the current event loop. Drive streaming and close the
    # shared SDK pool on that same loop. Explicit ownership avoids the SDK's
    # cached default pool and works on Python 3.10 as well as newer versions.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        try:
            model = OpenAIChatModel(
                model_name=model_id,
                provider=OpenAIProvider(openai_client=client),
            )
            return run_tests(model, loop)
        finally:
            loop.run_until_complete(client.close())
    finally:
        loop.close()
        asyncio.set_event_loop(None)


if __name__ == "__main__":
    raise SystemExit(main())
