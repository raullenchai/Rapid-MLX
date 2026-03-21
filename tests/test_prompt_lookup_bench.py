# SPDX-License-Identifier: Apache-2.0
"""
Prompt Lookup Decoding Benchmark.

Connects to a running vllm-mlx server on localhost:8000 and measures
tokens/sec for prompts that contain repetitive patterns -- the kind
of output where prompt-lookup speculative decoding should shine.

Run baseline (no prompt lookup):
    python3.12 -m pytest tests/test_prompt_lookup_bench.py -v -s

The test prints a table of results at the end.  Save the output and
compare against the same prompts after wiring up prompt lookup.
"""

import asyncio
import json
import time
from dataclasses import dataclass

import aiohttp
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
MAX_TOKENS = 1024
TEMPERATURE = 0.0  # deterministic for reproducibility


@dataclass
class BenchResult:
    name: str
    prompt: str
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_sec: float = 0.0
    tokens_per_sec: float = 0.0
    ttft_sec: float = 0.0  # time to first token
    error: str | None = None


# ---------------------------------------------------------------------------
# Prompts designed to produce repetitive output (prompt-lookup friendly)
# ---------------------------------------------------------------------------

BENCH_PROMPTS: list[tuple[str, str]] = [
    (
        "getter_setter",
        "Generate a Python class called User with 10 getter/setter method pairs "
        "for the following fields: name, age, email, phone, address, city, state, "
        "zip_code, country, company. Each getter should return self._field and "
        "each setter should assign the value. Include type hints. "
        "Output ONLY the code, no explanation.",
    ),
    (
        "json_array",
        "Generate a JSON array with 20 user objects. Each object must have these "
        "exact fields: id (integer 1-20), name (string), email (string), "
        "role (one of admin/user/editor), created_at (ISO 8601 date string). "
        "Output ONLY valid JSON, no markdown fences, no explanation.",
    ),
    (
        "markdown_table",
        "Write a markdown table comparing 15 programming languages with columns: "
        "Name, Year Created, Paradigm, Typing (static/dynamic), Relative Speed "
        "(fast/medium/slow), Popularity Rank. Include: Python, JavaScript, Java, "
        "C++, C, Go, Rust, TypeScript, Swift, Kotlin, Ruby, PHP, Scala, Haskell, "
        "Lua. Output ONLY the markdown table, no explanation.",
    ),
    (
        "sql_inserts",
        "Generate 20 SQL INSERT statements for a table called 'products' with "
        "columns (id INT, name VARCHAR(100), price DECIMAL(10,2), category "
        "VARCHAR(50), in_stock BOOLEAN). Use realistic product data. "
        "Output ONLY the SQL statements.",
    ),
    (
        "html_list",
        "Generate an HTML unordered list (<ul>) with 25 list items (<li>). "
        "Each item should contain a link (<a href='#'>) with the name of a "
        "world capital city and its country in parentheses. "
        "Output ONLY the HTML, no explanation.",
    ),
    (
        "csv_data",
        "Generate CSV data with a header row and 20 data rows. Columns: "
        "employee_id, first_name, last_name, department, salary, start_date. "
        "Use realistic but fictional data. Output ONLY the CSV, no explanation.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def check_server_health() -> bool:
    """Return True if the server is reachable."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{BASE_URL}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp,
        ):
            return resp.status == 200
    except Exception:
        return False


async def run_streaming_bench(
    session: aiohttp.ClientSession,
    name: str,
    prompt: str,
) -> BenchResult:
    """Send a streaming chat completion and measure throughput."""
    result = BenchResult(name=name, prompt=prompt[:80] + "...")

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_first_token = None
    completion_tokens = 0
    prompt_tokens = 0
    collected_text = ""

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract usage from chunk if present
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    collected_text += content

    except asyncio.TimeoutError:
        result.error = "Timeout (300s)"
        return result
    except Exception as e:
        result.error = str(e)
        return result

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # If server didn't report usage, estimate from collected text
    # (rough: ~4 chars per token for English)
    if completion_tokens == 0 and collected_text:
        # Use a rough heuristic; not critical for relative comparison
        completion_tokens = max(1, len(collected_text) // 4)

    result.elapsed_sec = elapsed
    result.prompt_tokens = prompt_tokens
    result.completion_tokens = completion_tokens
    result.total_tokens = prompt_tokens + completion_tokens
    result.tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    result.ttft_sec = (t_first_token - t_start) if t_first_token else elapsed

    return result


# ---------------------------------------------------------------------------
# Pytest fixtures and tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_server_reachable():
    """Verify the server is running before benchmarking."""
    healthy = await check_server_health()
    if not healthy:
        pytest.skip("Server not reachable at localhost:8000 -- start the server first")


@pytest.mark.asyncio
async def test_prompt_lookup_baseline():
    """
    Baseline benchmark: measure tok/s for repetitive-pattern prompts.

    This test sends each prompt to the server and records throughput.
    Save the output for later comparison with prompt-lookup enabled.
    """
    healthy = await check_server_health()
    if not healthy:
        pytest.skip("Server not reachable at localhost:8000")

    results: list[BenchResult] = []

    async with aiohttp.ClientSession() as session:
        # Warm up with a short request
        warmup = await run_streaming_bench(session, "warmup", "Say hello in one word.")
        if warmup.error:
            pytest.fail(f"Warmup failed: {warmup.error}")

        # Run each benchmark prompt sequentially (to avoid contention)
        for name, prompt in BENCH_PROMPTS:
            print(f"\n  Running: {name} ...", end="", flush=True)
            r = await run_streaming_bench(session, name, prompt)
            results.append(r)
            if r.error:
                print(f" ERROR: {r.error}")
            else:
                print(
                    f" {r.completion_tokens} tok in {r.elapsed_sec:.1f}s "
                    f"= {r.tokens_per_sec:.1f} tok/s  (TTFT {r.ttft_sec:.2f}s)"
                )

    # ── Summary table ──
    print("\n")
    print("=" * 85)
    print("  PROMPT LOOKUP BASELINE BENCHMARK RESULTS")
    print("=" * 85)
    print(
        f"  {'Prompt':<18} {'Comp Tok':>10} {'Time (s)':>10} "
        f"{'Tok/s':>10} {'TTFT (s)':>10} {'Status':>10}"
    )
    print("-" * 85)

    total_tokens = 0
    total_time = 0.0
    ok_count = 0

    for r in results:
        if r.error:
            print(
                f"  {r.name:<18} {'--':>10} {'--':>10} "
                f"{'--':>10} {'--':>10} {'FAIL':>10}"
            )
        else:
            print(
                f"  {r.name:<18} {r.completion_tokens:>10} {r.elapsed_sec:>10.2f} "
                f"{r.tokens_per_sec:>10.1f} {r.ttft_sec:>10.2f} {'OK':>10}"
            )
            total_tokens += r.completion_tokens
            total_time += r.elapsed_sec
            ok_count += 1

    print("-" * 85)
    if ok_count > 0 and total_time > 0:
        avg_tps = total_tokens / total_time
        print(
            f"  {'AGGREGATE':<18} {total_tokens:>10} {total_time:>10.2f} "
            f"{avg_tps:>10.1f} {'--':>10} {f'{ok_count}/{len(results)}':>10}"
        )
    print("=" * 85)
    print()

    # At least one prompt should succeed
    assert ok_count > 0, "All benchmark prompts failed"
