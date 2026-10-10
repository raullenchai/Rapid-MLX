#!/usr/bin/env python3
"""Measure concurrent SSE text throughput on an already-running vision server.

Run the same command against baseline and candidate servers, with speculative
and prefix decoding disabled. Timing is meaningful only with an idle GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

PROMPTS = (
    "Explain why the sky is blue in detail.",
    "Write a story about a lighthouse keeper.",
    "Describe how to implement a hash table.",
    "Explain the history of the printing press.",
)


async def measure(url: str, model: str, tokens: int, reps: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=240) as client:

        async def request(index: int, limit: int) -> dict:
            started = time.perf_counter()
            first = None
            text = ""
            usage = None
            done = False
            finish_reason = None
            async with client.stream(
                "POST",
                url.rstrip("/") + "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPTS[index]}],
                    "temperature": 0,
                    "max_tokens": limit,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line == "data: [DONE]":
                        done = True
                        break
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    usage = event.get("usage") or usage
                    for choice in event.get("choices", []):
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta", {})
                        content = (delta.get("content") or "") + (
                            delta.get("reasoning_content") or ""
                        )
                        if content:
                            if first is None:
                                first = time.perf_counter() - started
                            text += content
            if (
                first is None
                or usage is None
                or usage.get("completion_tokens") != limit
                or not done
                or finish_reason != "length"
            ):
                raise RuntimeError(
                    f"Incomplete stream: ttft={first}, usage={usage}, "
                    f"done={done}, finish_reason={finish_reason}"
                )
            return {
                "ttft": first,
                "tokens": usage["completion_tokens"],
                "elapsed": time.perf_counter() - started,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }

        await request(0, 8)
        rows = []
        for width in (1, 2, 4):
            for rep in range(reps):
                started = time.perf_counter()
                tasks = [asyncio.create_task(request(i, tokens)) for i in range(width)]
                try:
                    outputs = await asyncio.gather(*tasks)
                except BaseException:
                    # Drain siblings before the shared HTTP client closes, including
                    # when the caller cancels the complete measurement.
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                elapsed = time.perf_counter() - started
                row = {
                    "b": width,
                    "rep": rep,
                    "wall": elapsed,
                    "tok_s": sum(o["tokens"] for o in outputs) / elapsed,
                    "outputs": outputs,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.reps < 1:
        parser.error("--tokens and --reps must be positive")
    rows = asyncio.run(measure(args.url, args.model, args.tokens, args.reps))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
