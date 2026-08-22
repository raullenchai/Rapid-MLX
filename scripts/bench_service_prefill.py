#!/usr/bin/env python3
"""Benchmark cold, cached, and contended prefill through an OpenAI API.

The benchmark records server-reported prompt and cached-token counts instead
of inferring cache hits from latency.  It also distinguishes first visible
content/reasoning from role-only SSE frames, which is required for honest TTFT.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:
    from scripts.bench_metadata import write_bench_json
except ModuleNotFoundError:
    from bench_metadata import write_bench_json


FILLER = "Deterministic prefill benchmark sentence with stable repeated words. "


def token_count(encoded: Any) -> int:
    """Count token ids from list- or BatchEncoding-shaped tokenizer output."""
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids", ())
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, Sequence)
        and encoded
        and isinstance(encoded[0], Sequence)
        and not isinstance(encoded[0], (str, bytes))
    ):
        encoded = encoded[0]
    return len(encoded)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    ttfts = [float(row["ttft_ms"]) for row in rows]
    totals = [float(row["total_ms"]) for row in rows]
    return {
        "ttft_p50_ms": round(statistics.median(ttfts), 2),
        "ttft_p95_ms": round(percentile(ttfts, 0.95), 2),
        "total_p50_ms": round(statistics.median(totals), 2),
        "total_p95_ms": round(percentile(totals, 0.95), 2),
    }


def make_messages(tokenizer: Any, target_tokens: int, suffix: str = "") -> list[dict]:
    """Build a deterministic chat prompt close to ``target_tokens`` tokens."""
    system = "You are a concise benchmark assistant."
    content = FILLER * max(1, target_tokens // 8)

    def count(text: str) -> int:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text + suffix},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return token_count(rendered)
        except (AttributeError, TypeError, ValueError):
            return token_count(tokenizer.encode(system + "\n" + text + suffix))

    # Character-space binary search avoids assumptions about tokenizer word
    # boundaries while keeping the API payload valid text.
    low, high = 0, len(content)
    while low < high:
        mid = (low + high + 1) // 2
        if count(content[:mid]) <= target_tokens:
            low = mid
        else:
            high = mid - 1
    text = content[:low]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text + suffix},
    ]


def stream_request(
    client: httpx.Client,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
    }
    started = time.perf_counter()
    first_visible: float | None = None
    usage: dict[str, Any] = {}
    finish_reason = None
    visible_chunks = 0
    with client.stream(
        "POST", f"{base_url}/chat/completions", json=payload
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("tool_calls")
            ):
                visible_chunks += 1
                if first_visible is None:
                    first_visible = time.perf_counter()
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    finished = time.perf_counter()
    if first_visible is None:
        raise RuntimeError(
            "stream completed without a visible content, reasoning, or tool delta"
        )
    details = usage.get("prompt_tokens_details") or {}
    required_usage = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    invalid_usage = [
        name for name, value in required_usage.items() if not isinstance(value, int)
    ]
    if invalid_usage:
        raise RuntimeError(
            "stream completed without valid server usage metadata: "
            + ", ".join(invalid_usage)
        )
    return {
        "ttft_ms": round((first_visible - started) * 1000, 2),
        "total_ms": round((finished - started) * 1000, 2),
        "prompt_tokens": required_usage["prompt_tokens"],
        "cached_tokens": required_usage["cached_tokens"],
        "completion_tokens": required_usage["completion_tokens"],
        "visible_chunks": visible_chunks,
        "finish_reason": finish_reason,
    }


def clear_prefix_cache(client: httpx.Client, root_url: str) -> dict[str, Any]:
    response = client.post(f"{root_url}/v1/cache/clear")
    response.raise_for_status()
    return response.json()


def get_status(client: httpx.Client, root_url: str) -> dict[str, Any]:
    response = client.get(f"{root_url}/v1/status")
    response.raise_for_status()
    return response.json()


def wait_for_running_request(
    client: httpx.Client,
    root_url: str,
    *,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.01,
) -> dict[str, Any]:
    """Wait until the service confirms that a request is actively scheduled."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = get_status(client, root_url)
        if int(status.get("num_running") or 0) > 0:
            return status
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"no running request observed at {root_url}/v1/status "
        f"within {timeout_seconds:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 16384])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--contention-length", type=int, default=32768)
    parser.add_argument("--contention-repeat", type=int, default=3)
    parser.add_argument("--contention-delay-ms", type=float, default=25.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    root_url = args.url.removesuffix("/v1")
    timeout = httpx.Timeout(900.0, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        model = args.model or client.get(f"{args.url}/models").json()["data"][0]["id"]
        tokenizer_id = args.tokenizer or model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        client.get(f"{root_url}/health").raise_for_status()

        cold: dict[str, list[dict[str, Any]]] = {}
        for target in args.lengths:
            messages = make_messages(tokenizer, target)
            rows = []
            for repeat in range(args.repeat):
                clear_prefix_cache(client, root_url)
                row = stream_request(client, args.url, model, messages, args.max_tokens)
                row.update(target_tokens=target, repeat=repeat)
                rows.append(row)
            cold[str(target)] = rows

        cache_target = args.lengths[-1]
        base_messages = make_messages(tokenizer, cache_target)
        clear_prefix_cache(client, root_url)
        populate = stream_request(
            client, args.url, model, base_messages, args.max_tokens
        )
        exact = [
            stream_request(client, args.url, model, base_messages, args.max_tokens)
            for _ in range(args.repeat)
        ]
        partial_messages = [*base_messages]
        partial_messages[-1] = {
            "role": "user",
            "content": base_messages[-1]["content"] + "\nReturn only the word done.",
        }
        partial = [
            stream_request(client, args.url, model, partial_messages, args.max_tokens)
            for _ in range(args.repeat)
        ]

        long_messages = make_messages(tokenizer, args.contention_length)
        short_messages = make_messages(tokenizer, min(args.lengths))
        contention_runs = []
        for repeat in range(args.contention_repeat):
            clear_prefix_cache(client, root_url)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                long_future = executor.submit(
                    stream_request,
                    client,
                    args.url,
                    model,
                    long_messages,
                    args.max_tokens,
                )
                wait_for_running_request(client, root_url)
                time.sleep(args.contention_delay_ms / 1000)
                short_future = executor.submit(
                    stream_request,
                    client,
                    args.url,
                    model,
                    short_messages,
                    args.max_tokens,
                )
                contention_runs.append(
                    {
                        "repeat": repeat,
                        "long": long_future.result(),
                        "short": short_future.result(),
                    }
                )
        contention = {
            "long_target_tokens": args.contention_length,
            "short_target_tokens": min(args.lengths),
            "submit_delay_ms": args.contention_delay_ms,
            "short_summary": summarize([row["short"] for row in contention_runs]),
            "long_summary": summarize([row["long"] for row in contention_runs]),
            "runs": contention_runs,
        }

        status = get_status(client, root_url)

    payload = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_platform": platform.platform(),
        "base_url": args.url,
        "model": model,
        "tokenizer": tokenizer_id,
        "repeat": args.repeat,
        "max_tokens": args.max_tokens,
        "cold": {
            key: {"summary": summarize(rows), "runs": rows}
            for key, rows in cold.items()
        },
        "cache": {
            "target_tokens": cache_target,
            "populate": populate,
            "exact": {"summary": summarize(exact), "runs": exact},
            "partial": {"summary": summarize(partial), "runs": partial},
        },
        "contention": contention,
        "server_status": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_bench_json(args.output, payload, __file__)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
