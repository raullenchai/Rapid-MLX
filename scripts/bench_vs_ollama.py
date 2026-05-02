#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Rapid-MLX against Ollama with self-managed server processes.

Manual usage:
    python scripts/bench_vs_ollama.py
    python scripts/bench_vs_ollama.py --model-pair qwen3.5-4b=qwen3.5:4b --runs 1
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("reports/benchmarks/ollama-comparison")

DECODE_MESSAGES = [
    {
        "role": "user",
        "content": "/no_think Explain how a CPU executes instructions. Be concise but specific.",
    }
]

MULTI_TURN_START = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "/no_think What is a binary search tree?"},
]

MULTI_TURN_FOLLOWUPS = [
    "/no_think Now describe insertion in one paragraph.",
    "/no_think What is the average search complexity?",
    "/no_think Give one practical use case.",
]

CONCURRENT_MESSAGES = [
    {
        "role": "user",
        "content": "/no_think List five practical tips for writing reliable Python services.",
    }
]


@dataclass(frozen=True)
class ModelPair:
    rapid_mlx: str
    ollama: str


@dataclass
class CliArgs:
    model_pairs: list[ModelPair]
    runs: int
    warmups: int
    max_tokens: int
    concurrency: list[int]
    output_dir: Path
    no_pull: bool
    no_download: bool
    startup_timeout: float
    request_timeout: float
    rapid_mlx_args: list[str]
    ollama_env: dict[str, str]


@dataclass
class ParsedStream:
    content_chunks: int = 0
    completion_tokens: int = 0
    eval_duration_ns: int | None = None


def default_model_pairs() -> list[ModelPair]:
    return [
        ModelPair("qwen3.5-4b", "qwen3.5:4b"),
        ModelPair("qwen3.5-9b", "qwen3.5:9b"),
    ]


def parse_model_pair(value: str) -> ModelPair:
    if "=" not in value:
        raise ValueError("--model-pair must use RAPID=OLLAMA format")
    rapid, ollama = value.split("=", 1)
    rapid = rapid.strip()
    ollama = ollama.strip()
    if not rapid or not ollama:
        raise ValueError("--model-pair must use RAPID=OLLAMA format")
    return ModelPair(rapid, ollama)


def parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--ollama-env must use KEY=VALUE format")
    key, env_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("--ollama-env must use KEY=VALUE format")
    return key, env_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Rapid-MLX against Ollama with managed servers."
    )
    parser.add_argument("--model-pair", action="append", default=[])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--rapid-mlx-arg", action="append", default=[])
    parser.add_argument("--ollama-env", action="append", default=[])
    return parser


def parse_args(argv: list[str] | None = None) -> CliArgs:
    ns = build_parser().parse_args(argv)
    model_pairs = (
        [parse_model_pair(value) for value in ns.model_pair]
        if ns.model_pair
        else default_model_pairs()
    )
    return CliArgs(
        model_pairs=model_pairs,
        runs=ns.runs,
        warmups=ns.warmups,
        max_tokens=ns.max_tokens,
        concurrency=ns.concurrency,
        output_dir=ns.output_dir,
        no_pull=ns.no_pull,
        no_download=ns.no_download,
        startup_timeout=ns.startup_timeout,
        request_timeout=ns.request_timeout,
        rapid_mlx_args=ns.rapid_mlx_arg,
        ollama_env=dict(parse_env_assignment(value) for value in ns.ollama_env),
    )


def throughput_speedup(
    rapid_value: float | None, ollama_value: float | None
) -> float | None:
    if rapid_value is None or ollama_value is None or ollama_value <= 0:
        return None
    return rapid_value / ollama_value


def latency_speedup(
    rapid_value: float | None, ollama_value: float | None
) -> float | None:
    if rapid_value is None or ollama_value is None or rapid_value <= 0:
        return None
    return ollama_value / rapid_value


def format_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_rapid_mlx_stream(lines: Iterable[str]) -> ParsedStream:
    parsed = ParsedStream()
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        usage = data.get("usage")
        if isinstance(usage, dict):
            completion_tokens = _safe_int(usage.get("completion_tokens"))
            if completion_tokens is not None:
                parsed.completion_tokens = completion_tokens
        choices = data.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                parsed.content_chunks += 1
    if parsed.completion_tokens <= 0:
        parsed.completion_tokens = parsed.content_chunks
    return parsed


def parse_ollama_stream(lines: Iterable[str]) -> ParsedStream:
    parsed = ParsedStream()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                parsed.content_chunks += 1
        eval_count = _safe_int(data.get("eval_count"))
        if eval_count is not None:
            parsed.completion_tokens = eval_count
        eval_duration = _safe_int(data.get("eval_duration"))
        if eval_duration is not None:
            parsed.eval_duration_ns = eval_duration
    if parsed.completion_tokens <= 0:
        parsed.completion_tokens = parsed.content_chunks
    return parsed


def build_stream_metric(
    parsed: ParsedStream,
    start_at: float,
    first_content_at: float | None,
    end_at: float,
) -> dict:
    ttft_s = (first_content_at - start_at) if first_content_at else (end_at - start_at)
    decode_s = max(end_at - (first_content_at or start_at), 0.0)
    decode_tok_s = parsed.completion_tokens / decode_s if decode_s > 0 else 0.0
    return {
        "ttft_ms": round(ttft_s * 1000, 1),
        "decode_tok_s": round(decode_tok_s, 1),
        "completion_tokens": parsed.completion_tokens,
        "total_ms": round((end_at - start_at) * 1000, 1),
    }


def summarize_stream_runs(runs: list[dict]) -> dict:
    keys = ["ttft_ms", "decode_tok_s", "completion_tokens", "total_ms"]
    if not runs:
        return {key: 0.0 for key in keys}
    return {
        key: round(sum(float(run.get(key, 0.0)) for run in runs) / len(runs), 1)
        for key in keys
    }


def build_rapid_mlx_payload(
    model: str,
    messages: list[dict],
    max_tokens: int,
    stream: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def build_ollama_payload(
    model: str,
    messages: list[dict],
    max_tokens: int,
    stream: bool,
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }


def summarize_multi_turn(latencies_ms: list[float]) -> dict:
    avg = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return {
        "avg_turn_ms": round(avg, 1),
        "turn_latencies_ms": [round(value, 1) for value in latencies_ms],
    }


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((percentile / 100) * len(ordered) + 0.999999) - 1
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def summarize_concurrent_batch(results: list[dict], batch_elapsed_s: float) -> dict:
    latencies = [float(result.get("latency_ms", 0.0)) for result in results]
    tokens = sum(int(result.get("completion_tokens", 0)) for result in results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    aggregate = tokens / batch_elapsed_s if batch_elapsed_s > 0 else 0.0
    return {
        "aggregate_tok_s": round(aggregate, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(percentile_nearest_rank(latencies, 95), 1),
        "requests": len(results),
        "completion_tokens": tokens,
    }


def _rapid_mlx_sse_has_content(line: str) -> bool:
    if not line.startswith("data: "):
        return False
    payload_text = line[6:]
    if payload_text == "[DONE]":
        return False
    try:
        data = json.loads(payload_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            return True
    return False


def _ollama_line_has_content(line: str) -> bool:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    message = data.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    return isinstance(content, str) and bool(content)


def post_json_lines(
    url: str, payload: dict, timeout: float
) -> tuple[list[str], float, float | None, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start_at = time.perf_counter()
    first_content_at = None
    lines: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            lines.append(line)
            if first_content_at is None and (
                _rapid_mlx_sse_has_content(line) or _ollama_line_has_content(line)
            ):
                first_content_at = time.perf_counter()
    end_at = time.perf_counter()
    return lines, start_at, first_content_at, end_at


def post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    return data, elapsed_ms


def run_stream_once(
    engine: str,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
) -> dict:
    if engine == "rapid-mlx":
        url = f"{base_url}/v1/chat/completions"
        payload = build_rapid_mlx_payload(model, messages, max_tokens, stream=True)
        lines, start_at, first_content_at, end_at = post_json_lines(
            url, payload, timeout
        )
        parsed = parse_rapid_mlx_stream(lines)
    elif engine == "ollama":
        url = f"{base_url}/api/chat"
        payload = build_ollama_payload(model, messages, max_tokens, stream=True)
        lines, start_at, first_content_at, end_at = post_json_lines(
            url, payload, timeout
        )
        parsed = parse_ollama_stream(lines)
        if parsed.eval_duration_ns and parsed.completion_tokens > 0:
            metric = build_stream_metric(parsed, start_at, first_content_at, end_at)
            metric["decode_tok_s"] = round(
                parsed.completion_tokens / (parsed.eval_duration_ns / 1_000_000_000),
                1,
            )
            return metric
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return build_stream_metric(parsed, start_at, first_content_at, end_at)


def run_multi_turn(
    engine: str, base_url: str, model: str, max_tokens: int, timeout: float
) -> dict:
    messages = list(MULTI_TURN_START)
    latencies: list[float] = []
    followups = list(MULTI_TURN_FOLLOWUPS)
    for turn in range(4):
        if engine == "rapid-mlx":
            url = f"{base_url}/v1/chat/completions"
            payload = build_rapid_mlx_payload(model, messages, max_tokens, stream=False)
            data, latency_ms = post_json(url, payload, timeout)
            content = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
        elif engine == "ollama":
            url = f"{base_url}/api/chat"
            payload = build_ollama_payload(model, messages, max_tokens, stream=False)
            data, latency_ms = post_json(url, payload, timeout)
            content = (data.get("message") or {}).get("content") or ""
        else:
            raise ValueError(f"Unknown engine: {engine}")
        latencies.append(latency_ms)
        messages.append({"role": "assistant", "content": content})
        if turn < len(followups):
            messages.append({"role": "user", "content": followups[turn]})
    return summarize_multi_turn(latencies)


def run_concurrent_throughput(
    engine: str,
    base_url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> dict:
    start = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_stream_once,
                engine,
                base_url,
                model,
                CONCURRENT_MESSAGES,
                max_tokens,
                timeout,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            metric = future.result()
            results.append(
                {
                    "latency_ms": metric["total_ms"],
                    "completion_tokens": metric["completion_tokens"],
                }
            )
    elapsed = time.perf_counter() - start
    return summarize_concurrent_batch(results, elapsed)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
