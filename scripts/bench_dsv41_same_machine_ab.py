#!/usr/bin/env python3
"""Measure same-machine DeepSeek V4.1 SSE decode throughput (Python 3.12).

Run from the repository root with:
    python -m scripts.bench_dsv41_same_machine_ab --help

Print per-round metrics and a mean-throughput summary as JSON lines. Save the
first measured output beside the prompt with an ``.out.txt`` suffix.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

try:
    from scripts.bench_metadata import format_bench_json
except ImportError:  # direct `python scripts/bench_*.py` execution
    from bench_metadata import format_bench_json


def events(response):
    parts = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    if parts:
        yield "\n".join(parts)


def run(args, prompt):
    payload = {
        "model": args.model,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        args.base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    first = last = None
    delta_count = content_length = reasoning_length = 0
    usage = None
    text = []
    done = False
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            for data in events(response):
                now = time.perf_counter()
                if data.strip() == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    reasoning = "".join(
                        delta.get(key) or ""
                        for key in ("reasoning_content", "reasoning")
                    )
                    content = delta.get("content") or ""
                    if reasoning or content:
                        if first is None:
                            first = now
                        last = now
                        delta_count += 1
                        reasoning_length += len(reasoning)
                        content_length += len(content)
                        text.extend((reasoning, content))
    except urllib.error.HTTPError as exc:
        with exc:
            detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    elapsed = time.perf_counter() - started
    if first is None or not done:
        raise ValueError("stream must contain content or reasoning and end with [DONE]")
    tokens = (usage or {}).get("completion_tokens")
    if tokens is None:
        raise ValueError("server did not report usage.completion_tokens")
    interval = last - first
    rate = (tokens - 1) / interval if tokens > 1 and interval > 0 else None
    return {
        "ttft_s": first - started,
        "decode_tokens_per_s": rate,
        "total_wall_s": elapsed,
        "completion_tokens": tokens,
        "token_count_source": "usage.completion_tokens",
        "delta_count": delta_count,
        "decode_interval_s": interval,
        "content_length": content_length,
        "reasoning_length": reasoning_length,
        "usage": usage,
    }, "".join(text)


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive(value):
    number = nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="API base URL including /v1")
    parser.add_argument("--model", required=True, help="Served model identifier")
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="UTF-8 prompt file; first measured output is saved alongside it",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Build/depth/prompt label included in JSON output",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive,
        default=256,
        help="Maximum completion tokens (default: 256)",
    )
    parser.add_argument(
        "--rounds",
        type=positive,
        default=2,
        help="Measured requests after warm-up (default: 2)",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative,
        default=1,
        help="Unmeasured warm-up requests (default: 1)",
    )
    args = parser.parse_args()
    rates = []
    measured = 0
    try:
        with open(args.prompt_file, encoding="utf-8") as source:
            prompt = source.read()
        for _ in range(args.warmup):
            run(args, prompt)
        for round_number in range(1, args.rounds + 1):
            metrics, text = run(args, prompt)
            if metrics["decode_tokens_per_s"] is None:
                raise ValueError(
                    f"measured round {round_number} has unmeasurable decode throughput"
                )
            if round_number == 1:
                with open(
                    args.prompt_file + ".out.txt", "w", encoding="utf-8"
                ) as output:
                    output.write(text)
            measured += 1
            rates.append(metrics["decode_tokens_per_s"])
            print(
                format_bench_json(
                    {
                        "type": "round",
                        "label": args.label,
                        "model": args.model,
                        "round": round_number,
                        **metrics,
                    },
                    __file__,
                    indent=None,
                ),
                flush=True,
            )
    except Exception as exc:
        print(
            format_bench_json(
                {"type": "error", "label": args.label, "error": str(exc)},
                __file__,
                indent=None,
            ),
            flush=True,
        )
        return 1
    print(
        format_bench_json(
            {
                "type": "summary",
                "label": args.label,
                "model": args.model,
                "rounds": measured,
                "valid_decode_rounds": len(rates),
                "mean_decode_tokens_per_s": sum(rates) / len(rates) if rates else None,
            },
            __file__,
            indent=None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
