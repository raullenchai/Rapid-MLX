#!/usr/bin/env python3
"""
Simulator test for event loop responsiveness and model benchmarking.

Tests:
1. GET /v1/models responds quickly during active generation (decode phase)
2. Client disconnect (ESC) releases lock, next request starts promptly
3. Request queuing: second request waits for first to complete, no preemption
4. Golden prompt benchmarks: run prompts from golden_prompts.py, measure speed

Requires a running Rapid-MLX server on localhost:8000.

Usage:
    python tests/test_event_loop.py                   # run event loop tests only
    python tests/test_event_loop.py --bench            # run golden prompt benchmarks
    python tests/test_event_loop.py --bench --level 2  # only level 2 prompts
    python tests/test_event_loop.py --bench --tag coding  # only coding prompts
    python tests/test_event_loop.py --all              # run everything
"""

import argparse
import asyncio
import sys
import time

import aiohttp
import pytest

# This module drives a live server (see the docstring above) — without one
# it fails on connect, and with one it sends real completions to whatever
# is bound on :8000. Keep it out of the default unit sweep.
pytestmark = pytest.mark.integration

BASE = "http://127.0.0.1:8000"


async def stream_completions(session, prompt, max_tokens=128, timeout=120):
    """Send a streaming completion request, return (tokens, elapsed, ttft)."""
    payload = {
        "model": "test",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    tokens = 0
    t0 = time.monotonic()
    ttft = None
    try:
        async with session.post(
            f"{BASE}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            async for line in resp.content:
                text = line.decode().strip()
                if text.startswith("data: ") and text != "data: [DONE]":
                    tokens += 1
                    if ttft is None:
                        ttft = time.monotonic() - t0
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        print(f"  stream error after {tokens} tokens: {e}")
    elapsed = time.monotonic() - t0
    return tokens, elapsed, ttft


# ── Event Loop Tests ──────────────────────────────────────────────


async def test_event_loop_responsiveness():
    """Test 1: GET /v1/models responds <2s during active decode."""
    print("\n=== Test 1: Event Loop Responsiveness ===")

    async with aiohttp.ClientSession() as session:
        gen_task = asyncio.create_task(
            stream_completions(
                session, "Write a very long story about a dragon. ", max_tokens=256
            )
        )
        await asyncio.sleep(3)

        t0 = time.monotonic()
        try:
            async with session.get(
                f"{BASE}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                await resp.json()
                latency = time.monotonic() - t0
                print(f"  GET /v1/models latency: {latency:.3f}s")
                if latency < 2.0:
                    print("  PASS: Event loop responsive during decode")
                else:
                    print(f"  FAIL: Event loop blocked ({latency:.1f}s)")
        except asyncio.TimeoutError:
            latency = time.monotonic() - t0
            print(f"  FAIL: GET /v1/models timed out ({latency:.1f}s)")

        tokens, elapsed, _ = await gen_task
        print(
            f"  Generation: {tokens} tokens in {elapsed:.1f}s ({tokens / elapsed:.1f} tok/s)"
        )


async def test_disconnect_recovery():
    """Test 2: After client disconnect, next request starts promptly."""
    print("\n=== Test 2: Disconnect Recovery (ESC) ===")

    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "test",
            "prompt": "Count from 1 to 1000 slowly. ",
            "max_tokens": 512,
            "stream": True,
        }
        tokens = 0
        async with session.post(
            f"{BASE}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            async for line in resp.content:
                text = line.decode().strip()
                if text.startswith("data: ") and text != "data: [DONE]":
                    tokens += 1
                    if tokens >= 10:
                        break
        print(f"  Disconnected after {tokens} tokens")

    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session:
        tokens, elapsed, _ = await stream_completions(
            session, "Say hello. ", max_tokens=16, timeout=30
        )
        print(f"  Next request: {tokens} tokens in {elapsed:.1f}s")
        if elapsed < 20:
            print("  PASS: Recovery after disconnect")
        else:
            print(f"  FAIL: Recovery took {elapsed:.1f}s")


async def test_request_queuing():
    """Test 3: Second request waits for first, no preemption."""
    print("\n=== Test 3: Request Queuing ===")

    async with aiohttp.ClientSession() as session:
        task_a = asyncio.create_task(
            stream_completions(
                session, "Tell me about quantum physics. ", max_tokens=64, timeout=60
            )
        )
        await asyncio.sleep(2)

        task_b = asyncio.create_task(
            stream_completions(
                session, "Tell me about biology. ", max_tokens=32, timeout=60
            )
        )

        tokens_a, elapsed_a, _ = await task_a
        tokens_b, elapsed_b, _ = await task_b

        print(f"  Request A: {tokens_a} tokens in {elapsed_a:.1f}s")
        print(f"  Request B: {tokens_b} tokens in {elapsed_b:.1f}s")

        if tokens_a >= 60:
            print("  PASS: A completed fully (no preemption)")
        else:
            print(f"  FAIL: A only generated {tokens_a} tokens (preempted?)")

        if tokens_b > 0:
            print("  PASS: B completed after A")
        else:
            print("  FAIL: B got no tokens")


# ── Golden Prompt Benchmarks ─────────────────────────────────────


async def run_golden_benchmarks(level=None, tag=None):
    """Run golden prompts and report speed metrics."""
    from golden_prompts import PROMPTS, get_prompts_by_level, get_prompts_by_tag

    if level is not None:
        prompts = get_prompts_by_level(level)
    elif tag is not None:
        prompts = get_prompts_by_tag(tag)
    else:
        prompts = PROMPTS

    # Skip multi-turn and tool-calling prompts (need chat API)
    skip_tags = {"tool_calling", "multi_turn", "agent"}
    prompts = [p for p in prompts if not (skip_tags & set(p["tags"]))]

    if not prompts:
        print("No matching prompts found.")
        return

    print(f"\n=== Golden Prompt Benchmarks ({len(prompts)} prompts) ===\n")
    print(f"{'ID':<20} {'Tokens':>6} {'TTFT':>7} {'Decode':>8} {'tok/s':>7}  Expect")
    print("-" * 80)

    results = []
    async with aiohttp.ClientSession() as session:
        for p in prompts:
            max_tok = p.get("max_tokens", 256)
            # Cap at 512 for benchmarking speed (no need for full 8192)
            bench_max = min(max_tok, 512)
            timeout = max(60, bench_max // 5)

            tokens, elapsed, ttft = await stream_completions(
                session, p["prompt"], max_tokens=bench_max, timeout=timeout
            )

            if tokens > 0 and elapsed > 0:
                decode_time = elapsed - (ttft or 0)
                tok_s = tokens / decode_time if decode_time > 0 else 0
            else:
                tok_s = 0

            result = {
                "id": p["id"],
                "level": p["level"],
                "tokens": tokens,
                "ttft": ttft,
                "elapsed": elapsed,
                "tok_s": tok_s,
            }
            results.append(result)

            ttft_str = f"{ttft:.2f}s" if ttft else "N/A"
            expect_short = p["expect"][:30]
            print(
                f"  {p['id']:<18} {tokens:>6} {ttft_str:>7} {elapsed:>7.1f}s {tok_s:>6.1f}  {expect_short}"
            )

    # Summary
    if results:
        avg_toks = sum(r["tok_s"] for r in results if r["tok_s"] > 0) / max(
            1, sum(1 for r in results if r["tok_s"] > 0)
        )
        avg_ttft = sum(r["ttft"] for r in results if r["ttft"]) / max(
            1, sum(1 for r in results if r["ttft"])
        )
        print("-" * 80)
        print(f"  {'AVERAGE':<18} {'':>6} {avg_ttft:>6.2f}s {'':>8} {avg_toks:>6.1f}")


# ── Main ─────────────────────────────────────────────────────────


async def main(args):
    # Health check
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{BASE}/v1/models", timeout=aiohttp.ClientTimeout(total=3)
            ) as resp,
        ):
            data = await resp.json()
            model = data["data"][0]["id"] if data.get("data") else "unknown"
            print(f"Server is up. Model: {model}")
    except Exception as e:
        print(f"Cannot connect to server at {BASE}: {e}")
        sys.exit(1)

    if args.bench or args.all:
        await run_golden_benchmarks(level=args.level, tag=args.tag)

    if not args.bench or args.all:
        await test_event_loop_responsiveness()
        await test_disconnect_recovery()
        await test_request_queuing()

    print("\n=== All tests complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event loop & benchmark tests")
    parser.add_argument(
        "--bench", action="store_true", help="Run golden prompt benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests + benchmarks")
    parser.add_argument("--level", type=int, help="Filter prompts by level (1-5)")
    parser.add_argument("--tag", type=str, help="Filter prompts by tag")
    args = parser.parse_args()
    asyncio.run(main(args))
