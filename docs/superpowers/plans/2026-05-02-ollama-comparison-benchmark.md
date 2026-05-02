# Ollama Comparison Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/bench_vs_ollama.py`, a reproducible benchmark that launches Rapid-MLX and Ollama sequentially, measures identical workloads, and emits raw JSON plus README-ready Markdown.

**Architecture:** Implement one focused script with small testable helpers for config parsing, stream parsing, metric math, reporting, server lifecycle, and orchestration. Keep tests offline by mocking subprocess and HTTP behavior; real model downloads and real servers are only exercised by manual runs.

**Tech Stack:** Python standard library (`argparse`, `json`, `subprocess`, `socket`, `urllib.request`, `threading`, `concurrent.futures`, `dataclasses`), pytest.

---

## Scope Check

The approved spec covers one subsystem: a CLI benchmark script. It does not need decomposition into multiple specs. The implementation should create one script and one unit test file.

## File Structure

- Create: `scripts/bench_vs_ollama.py`
  - Owns CLI parsing, model-pair configuration, server startup, HTTP requests, metrics, Markdown/JSON output, and cleanup.
  - Uses standard-library HTTP so contributors can run it without installing a new benchmark dependency.
- Create: `tests/test_bench_vs_ollama.py`
  - Imports `scripts/bench_vs_ollama.py` with `importlib.util`.
  - Tests pure helpers and mocked lifecycle behavior only.

---

### Task 1: Config, Model Pairs, And Speedup Math

**Files:**
- Create: `scripts/bench_vs_ollama.py`
- Create: `tests/test_bench_vs_ollama.py`

- [ ] **Step 1: Write failing tests for import, model-pair parsing, default config, and speedup math**

Add this test file:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/bench_vs_ollama.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_vs_ollama.py"


def load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_vs_ollama", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_model_pairs():
    bench = load_bench_module()

    pairs = bench.default_model_pairs()

    assert pairs == [
        bench.ModelPair("qwen3.5-4b", "qwen3.5:4b"),
        bench.ModelPair("qwen3.5-9b", "qwen3.5:9b"),
    ]


def test_parse_model_pair():
    bench = load_bench_module()

    pair = bench.parse_model_pair("rapid=qwen:9b")

    assert pair == bench.ModelPair("rapid", "qwen:9b")


def test_parse_model_pair_rejects_missing_separator():
    bench = load_bench_module()

    with pytest.raises(ValueError, match="RAPID=OLLAMA"):
        bench.parse_model_pair("qwen3.5-9b")


def test_parse_args_replaces_default_model_pairs():
    bench = load_bench_module()

    args = bench.parse_args(
        [
            "--model-pair",
            "rapid-a=ollama-a",
            "--model-pair",
            "rapid-b=ollama-b",
            "--runs",
            "2",
        ]
    )

    assert args.model_pairs == [
        bench.ModelPair("rapid-a", "ollama-a"),
        bench.ModelPair("rapid-b", "ollama-b"),
    ]
    assert args.runs == 2


def test_speedup_math_for_throughput_and_latency():
    bench = load_bench_module()

    assert bench.throughput_speedup(120.0, 40.0) == 3.0
    assert bench.latency_speedup(100.0, 250.0) == 2.5
    assert bench.throughput_speedup(120.0, 0.0) is None
    assert bench.format_speedup(2.345) == "2.35x"
    assert bench.format_speedup(None) == "-"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: FAIL because `scripts/bench_vs_ollama.py` does not exist.

- [ ] **Step 3: Add minimal config implementation**

Create `scripts/bench_vs_ollama.py` with these initial definitions:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Rapid-MLX against Ollama with self-managed server processes.

Manual usage:
    python scripts/bench_vs_ollama.py
    python scripts/bench_vs_ollama.py --model-pair qwen3.5-4b=qwen3.5:4b --runs 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("reports/benchmarks/ollama-comparison")


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


def throughput_speedup(rapid_value: float | None, ollama_value: float | None) -> float | None:
    if rapid_value is None or ollama_value is None or ollama_value <= 0:
        return None
    return rapid_value / ollama_value


def latency_speedup(rapid_value: float | None, ollama_value: float | None) -> float | None:
    if rapid_value is None or ollama_value is None or rapid_value <= 0:
        return None
    return ollama_value / rapid_value


def format_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "feat(bench): add ollama comparison config"
```

---

### Task 2: Stream Parsing And Metric Summaries

**Files:**
- Modify: `scripts/bench_vs_ollama.py`
- Modify: `tests/test_bench_vs_ollama.py`

- [ ] **Step 1: Write failing tests for stream parsing and summary metrics**

Append these tests to `tests/test_bench_vs_ollama.py`:

```python
def test_parse_rapid_mlx_sse_stream_prefers_usage_tokens():
    bench = load_bench_module()
    lines = [
        'data: {"choices":[{"delta":{"content":"hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":7}}',
        "data: [DONE]",
    ]

    parsed = bench.parse_rapid_mlx_stream(lines)

    assert parsed.content_chunks == 2
    assert parsed.completion_tokens == 7


def test_parse_ollama_stream_prefers_eval_metadata():
    bench = load_bench_module()
    lines = [
        '{"message":{"content":"hello"},"done":false}',
        '{"message":{"content":" world"},"done":false}',
        '{"done":true,"eval_count":9,"eval_duration":300000000}',
    ]

    parsed = bench.parse_ollama_stream(lines)

    assert parsed.content_chunks == 2
    assert parsed.completion_tokens == 9
    assert parsed.eval_duration_ns == 300000000


def test_build_stream_metric_uses_first_content_for_ttft_and_decode_time():
    bench = load_bench_module()

    metric = bench.build_stream_metric(
        parsed=bench.ParsedStream(content_chunks=2, completion_tokens=9),
        start_at=1.0,
        first_content_at=1.2,
        end_at=3.0,
    )

    assert metric["ttft_ms"] == 200.0
    assert metric["decode_tok_s"] == 5.0
    assert metric["completion_tokens"] == 9
    assert metric["total_ms"] == 2000.0


def test_summarize_runs_averages_numeric_fields():
    bench = load_bench_module()

    summary = bench.summarize_stream_runs(
        [
            {"ttft_ms": 100.0, "decode_tok_s": 40.0, "completion_tokens": 8, "total_ms": 300.0},
            {"ttft_ms": 300.0, "decode_tok_s": 80.0, "completion_tokens": 10, "total_ms": 500.0},
        ]
    )

    assert summary == {
        "ttft_ms": 200.0,
        "decode_tok_s": 60.0,
        "completion_tokens": 9.0,
        "total_ms": 400.0,
    }
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py::test_parse_rapid_mlx_sse_stream_prefers_usage_tokens tests/test_bench_vs_ollama.py::test_parse_ollama_stream_prefers_eval_metadata tests/test_bench_vs_ollama.py::test_build_stream_metric_uses_first_content_for_ttft_and_decode_time tests/test_bench_vs_ollama.py::test_summarize_runs_averages_numeric_fields -q
```

Expected: FAIL because parser helpers are not implemented.

- [ ] **Step 3: Implement stream parsing helpers**

Add these imports and definitions to `scripts/bench_vs_ollama.py`:

```python
import json
from collections.abc import Iterable


@dataclass
class ParsedStream:
    content_chunks: int = 0
    completion_tokens: int = 0
    eval_duration_ns: int | None = None


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
        usage = data.get("usage")
        if usage and usage.get("completion_tokens"):
            parsed.completion_tokens = int(usage["completion_tokens"])
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
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
        message = data.get("message") or {}
        if message.get("content"):
            parsed.content_chunks += 1
        if data.get("eval_count"):
            parsed.completion_tokens = int(data["eval_count"])
        if data.get("eval_duration"):
            parsed.eval_duration_ns = int(data["eval_duration"])
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "feat(bench): parse benchmark streams"
```

---

### Task 3: Request Payloads And Workload Runners

**Files:**
- Modify: `scripts/bench_vs_ollama.py`
- Modify: `tests/test_bench_vs_ollama.py`

- [ ] **Step 1: Write failing tests for payloads and workload aggregation**

Append these tests:

```python
def test_build_rapid_mlx_payload_is_deterministic_no_thinking():
    bench = load_bench_module()

    payload = bench.build_rapid_mlx_payload(
        model="qwen3.5-9b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        stream=True,
    )

    assert payload["model"] == "qwen3.5-9b"
    assert payload["temperature"] == 0
    assert payload["enable_thinking"] is False
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_build_ollama_payload_is_deterministic_no_thinking():
    bench = load_bench_module()

    payload = bench.build_ollama_payload(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=32,
        stream=True,
    )

    assert payload["model"] == "qwen3.5:9b"
    assert payload["think"] is False
    assert payload["stream"] is True
    assert payload["options"] == {"temperature": 0, "num_predict": 32}


def test_summarize_multi_turn_latency():
    bench = load_bench_module()

    summary = bench.summarize_multi_turn([100.0, 300.0, 500.0, 700.0])

    assert summary == {
        "avg_turn_ms": 400.0,
        "turn_latencies_ms": [100.0, 300.0, 500.0, 700.0],
    }


def test_summarize_concurrent_batch_computes_p95_and_aggregate_tps():
    bench = load_bench_module()

    summary = bench.summarize_concurrent_batch(
        results=[
            {"latency_ms": 100.0, "completion_tokens": 10},
            {"latency_ms": 300.0, "completion_tokens": 20},
            {"latency_ms": 200.0, "completion_tokens": 30},
        ],
        batch_elapsed_s=2.0,
    )

    assert summary == {
        "aggregate_tok_s": 30.0,
        "avg_latency_ms": 200.0,
        "p95_latency_ms": 300.0,
        "requests": 3,
        "completion_tokens": 60,
    }
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py::test_build_rapid_mlx_payload_is_deterministic_no_thinking tests/test_bench_vs_ollama.py::test_build_ollama_payload_is_deterministic_no_thinking tests/test_bench_vs_ollama.py::test_summarize_multi_turn_latency tests/test_bench_vs_ollama.py::test_summarize_concurrent_batch_computes_p95_and_aggregate_tps -q
```

Expected: FAIL because payload and workload helpers are missing.

- [ ] **Step 3: Implement payload builders and summary helpers**

Add these constants and functions:

```python
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
    index = max(0, min(len(ordered) - 1, int((percentile / 100) * len(ordered) + 0.999999) - 1))
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
```

- [ ] **Step 4: Add live HTTP workload functions**

Add these standard-library HTTP helpers:

```python
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post_json_lines(url: str, payload: dict, timeout: float) -> tuple[list[str], float, float | None, float]:
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
            if first_content_at is None:
                if line.startswith("data: "):
                    payload_text = line[6:]
                    if payload_text != "[DONE]":
                        try:
                            data = json.loads(payload_text)
                            if any((choice.get("delta") or {}).get("content") for choice in data.get("choices") or []):
                                first_content_at = time.perf_counter()
                        except json.JSONDecodeError:
                            pass
                else:
                    try:
                        data = json.loads(line)
                        if (data.get("message") or {}).get("content"):
                            first_content_at = time.perf_counter()
                    except json.JSONDecodeError:
                        pass
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


def run_stream_once(engine: str, base_url: str, model: str, messages: list[dict], max_tokens: int, timeout: float) -> dict:
    if engine == "rapid-mlx":
        url = f"{base_url}/v1/chat/completions"
        payload = build_rapid_mlx_payload(model, messages, max_tokens, stream=True)
        lines, start_at, first_content_at, end_at = post_json_lines(url, payload, timeout)
        parsed = parse_rapid_mlx_stream(lines)
    elif engine == "ollama":
        url = f"{base_url}/api/chat"
        payload = build_ollama_payload(model, messages, max_tokens, stream=True)
        lines, start_at, first_content_at, end_at = post_json_lines(url, payload, timeout)
        parsed = parse_ollama_stream(lines)
        if parsed.eval_duration_ns and parsed.completion_tokens > 0:
            metric = build_stream_metric(parsed, start_at, first_content_at, end_at)
            metric["decode_tok_s"] = round(parsed.completion_tokens / (parsed.eval_duration_ns / 1_000_000_000), 1)
            return metric
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return build_stream_metric(parsed, start_at, first_content_at, end_at)


def run_multi_turn(engine: str, base_url: str, model: str, max_tokens: int, timeout: float) -> dict:
    messages = list(MULTI_TURN_START)
    latencies: list[float] = []
    followups = list(MULTI_TURN_FOLLOWUPS)
    for turn in range(4):
        if engine == "rapid-mlx":
            url = f"{base_url}/v1/chat/completions"
            payload = build_rapid_mlx_payload(model, messages, max_tokens, stream=False)
            data, latency_ms = post_json(url, payload, timeout)
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
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


def run_concurrent_throughput(engine: str, base_url: str, model: str, concurrency: int, max_tokens: int, timeout: float) -> dict:
    start = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_stream_once, engine, base_url, model, CONCURRENT_MESSAGES, max_tokens, timeout)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "feat(bench): add ollama benchmark workloads"
```

---

### Task 4: Reporting, Metadata, And Output Files

**Files:**
- Modify: `scripts/bench_vs_ollama.py`
- Modify: `tests/test_bench_vs_ollama.py`

- [ ] **Step 1: Write failing tests for Markdown rendering and output writing**

Append these tests:

```python
def test_render_markdown_includes_model_table_and_speedups():
    bench = load_bench_module()
    result = {
        "metadata": {"timestamp": "2026-05-02T12:00:00", "git_commit": "abc123"},
        "config": {"runs": 3, "concurrency": [1, 2, 4]},
        "model_pairs": [
            {
                "rapid_mlx_model": "qwen3.5-9b",
                "ollama_model": "qwen3.5:9b",
                "rapid-mlx": {
                    "summary": {
                        "stream": {"ttft_ms": 100.0, "decode_tok_s": 120.0},
                        "multi_turn": {"avg_turn_ms": 200.0},
                        "concurrency": {"1": {"aggregate_tok_s": 110.0}},
                    }
                },
                "ollama": {
                    "summary": {
                        "stream": {"ttft_ms": 250.0, "decode_tok_s": 40.0},
                        "multi_turn": {"avg_turn_ms": 500.0},
                        "concurrency": {"1": {"aggregate_tok_s": 35.0}},
                    }
                },
            }
        ],
    }

    markdown = bench.render_markdown(result)

    assert "# Rapid-MLX vs Ollama Benchmark" in markdown
    assert "qwen3.5-9b vs qwen3.5:9b" in markdown
    assert "| Decode tok/s | 120.0 | 40.0 | 3.00x |" in markdown
    assert "| TTFT | 100.0 ms | 250.0 ms | 2.50x |" in markdown


def test_write_outputs_creates_json_and_markdown(tmp_path):
    bench = load_bench_module()
    result = {
        "metadata": {"timestamp": "2026-05-02T12:00:00", "git_commit": "abc123"},
        "config": {"runs": 1, "concurrency": [1]},
        "model_pairs": [],
    }

    paths = bench.write_outputs(result, tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["json"].name.endswith(".json")
    assert paths["markdown"].name.endswith(".md")
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py::test_render_markdown_includes_model_table_and_speedups tests/test_bench_vs_ollama.py::test_write_outputs_creates_json_and_markdown -q
```

Expected: FAIL because reporting helpers are missing.

- [ ] **Step 3: Implement metadata and reporting**

Add these imports and helpers:

```python
import platform
import subprocess
from datetime import datetime


def run_capture(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def collect_hardware_summary() -> dict:
    try:
        from vllm_mlx.optimizations import detect_hardware

        hw = detect_hardware()
        return {
            "chip_name": hw.chip_name,
            "total_memory_gb": round(hw.total_memory_gb, 1),
            "gpu_cores": hw.gpu_cores,
        }
    except Exception:
        return {
            "chip_name": None,
            "total_memory_gb": None,
            "gpu_cores": None,
        }


def collect_metadata() -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": run_capture(["git", "rev-parse", "--short", "HEAD"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": collect_hardware_summary(),
        "rapid_mlx_version": run_capture(["rapid-mlx", "--version"]),
        "ollama_version": run_capture(["ollama", "--version"]),
    }


def format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def render_model_pair_table(pair_result: dict) -> str:
    rapid = pair_result.get("rapid-mlx", {}).get("summary", {})
    ollama = pair_result.get("ollama", {}).get("summary", {})
    rapid_stream = rapid.get("stream", {})
    ollama_stream = ollama.get("stream", {})
    rapid_mt = rapid.get("multi_turn", {})
    ollama_mt = ollama.get("multi_turn", {})
    rapid_conc = rapid.get("concurrency", {})
    ollama_conc = ollama.get("concurrency", {})

    lines = [
        f"## {pair_result['rapid_mlx_model']} vs {pair_result['ollama_model']}",
        "",
        "| Metric | Rapid-MLX | Ollama | Speedup |",
        "|---|---:|---:|---:|",
    ]
    rapid_ttft = rapid_stream.get("ttft_ms")
    ollama_ttft = ollama_stream.get("ttft_ms")
    lines.append(
        f"| TTFT | {format_number(rapid_ttft, ' ms')} | {format_number(ollama_ttft, ' ms')} | {format_speedup(latency_speedup(rapid_ttft, ollama_ttft))} |"
    )
    rapid_decode = rapid_stream.get("decode_tok_s")
    ollama_decode = ollama_stream.get("decode_tok_s")
    lines.append(
        f"| Decode tok/s | {format_number(rapid_decode)} | {format_number(ollama_decode)} | {format_speedup(throughput_speedup(rapid_decode, ollama_decode))} |"
    )
    rapid_turn = rapid_mt.get("avg_turn_ms")
    ollama_turn = ollama_mt.get("avg_turn_ms")
    lines.append(
        f"| Multi-turn latency | {format_number(rapid_turn, ' ms')} | {format_number(ollama_turn, ' ms')} | {format_speedup(latency_speedup(rapid_turn, ollama_turn))} |"
    )
    for level in sorted(set(rapid_conc) | set(ollama_conc), key=lambda value: int(value)):
        rapid_tps = rapid_conc.get(level, {}).get("aggregate_tok_s")
        ollama_tps = ollama_conc.get(level, {}).get("aggregate_tok_s")
        lines.append(
            f"| Concurrent throughput ({level} users) | {format_number(rapid_tps)} | {format_number(ollama_tps)} | {format_speedup(throughput_speedup(rapid_tps, ollama_tps))} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_markdown(result: dict) -> str:
    metadata = result.get("metadata", {})
    hardware = metadata.get("hardware") or {}
    config = result.get("config", {})
    lines = [
        "# Rapid-MLX vs Ollama Benchmark",
        "",
        f"- Timestamp: `{metadata.get('timestamp', '-')}`",
        f"- Git commit: `{metadata.get('git_commit', '-')}`",
        f"- Python: `{metadata.get('python', '-')}`",
        f"- Platform: `{metadata.get('platform', '-')}`",
        f"- Hardware: `{hardware.get('chip_name') or '-'} ({hardware.get('total_memory_gb') or '-'} GB, {hardware.get('gpu_cores') or '-'} GPU cores)`",
        f"- Runs: `{config.get('runs', '-')}`",
        f"- Concurrency: `{config.get('concurrency', '-')}`",
        "",
        "Engines were launched sequentially on temporary localhost ports. Requests used deterministic no-thinking settings.",
        "",
    ]
    for pair_result in result.get("model_pairs", []):
        lines.append(render_model_pair_table(pair_result))
    return "\n".join(lines).rstrip() + "\n"


def safe_timestamp(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("T", "_")


def write_outputs(result: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = safe_timestamp(result["metadata"]["timestamp"])
    json_path = output_dir / f"{stamp}.json"
    markdown_path = output_dir / f"{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "feat(bench): render ollama comparison reports"
```

---

### Task 5: Server Lifecycle And Cleanup

**Files:**
- Modify: `scripts/bench_vs_ollama.py`
- Modify: `tests/test_bench_vs_ollama.py`

- [ ] **Step 1: Write failing tests for commands, ports, and cleanup**

Append these tests:

```python
def test_find_free_port_can_be_rebound():
    import socket

    bench = load_bench_module()

    port = bench.find_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_build_rapid_mlx_command_includes_explicit_benchmark_settings():
    bench = load_bench_module()

    cmd = bench.build_rapid_mlx_command("qwen3.5-9b", 9123, ["--prefill-step-size", "4096"])

    assert cmd == [
        "rapid-mlx",
        "serve",
        "qwen3.5-9b",
        "--host",
        "127.0.0.1",
        "--port",
        "9123",
        "--no-thinking",
        "--default-temperature",
        "0",
        "--prefill-step-size",
        "4096",
    ]


def test_build_ollama_environment_sets_host_and_custom_values(monkeypatch):
    bench = load_bench_module()
    monkeypatch.setenv("PATH", "/usr/bin")

    env = bench.build_ollama_environment(9124, {"OLLAMA_KEEP_ALIVE": "0"})

    assert env["PATH"] == "/usr/bin"
    assert env["OLLAMA_HOST"] == "127.0.0.1:9124"
    assert env["OLLAMA_KEEP_ALIVE"] == "0"


def test_managed_process_stop_terminates_then_waits():
    bench = load_bench_module()

    class FakeProc:
        def __init__(self):
            self.terminated = False
            self.waited = False
            self.returncode = None

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout):
            self.waited = True
            return 0

        def kill(self):
            raise AssertionError("kill should not be called")

    proc = FakeProc()
    managed = bench.ManagedProcess(proc, ["cmd"])
    managed.stop()

    assert proc.terminated is True
    assert proc.waited is True
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py::test_find_free_port_can_be_rebound tests/test_bench_vs_ollama.py::test_build_rapid_mlx_command_includes_explicit_benchmark_settings tests/test_bench_vs_ollama.py::test_build_ollama_environment_sets_host_and_custom_values tests/test_bench_vs_ollama.py::test_managed_process_stop_terminates_then_waits -q
```

Expected: FAIL because lifecycle helpers are missing.

- [ ] **Step 3: Implement lifecycle helpers**

Add these imports and definitions:

```python
import os
import shutil
import socket
import sys


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required executable not found on PATH: {name}")
    return path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_rapid_mlx_command(model: str, port: int, extra_args: list[str]) -> list[str]:
    return [
        "rapid-mlx",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-thinking",
        "--default-temperature",
        "0",
        *extra_args,
    ]


def build_ollama_environment(port: int, extra_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env.update(extra_env)
    return env


@dataclass
class ManagedProcess:
    proc: subprocess.Popen
    command: list[str]

    def stop(self) -> None:
        poll = getattr(self.proc, "poll", lambda: self.proc.returncode)
        if poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=20)


def start_process(command: list[str], env: dict[str, str] | None = None) -> ManagedProcess:
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return ManagedProcess(proc=proc, command=command)


def wait_for_url(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def prepare_models(args: CliArgs) -> None:
    if args.no_pull:
        return
    require_executable("ollama")
    for pair in args.model_pairs:
        print(f"Pulling Ollama model {pair.ollama}...", flush=True)
        result = subprocess.run(["ollama", "pull", pair.ollama], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ollama pull failed for {pair.ollama}")


def offline_env_if_needed(no_download: bool) -> dict[str, str] | None:
    if not no_download:
        return None
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "feat(bench): manage benchmark server lifecycle"
```

---

### Task 6: Orchestration, JSON Shape, And CLI Completion

**Files:**
- Modify: `scripts/bench_vs_ollama.py`
- Modify: `tests/test_bench_vs_ollama.py`

- [ ] **Step 1: Write failing tests for engine result shape and failure recording**

Append these tests:

```python
def test_build_engine_success_result_shape():
    bench = load_bench_module()

    result = bench.build_engine_success_result(
        engine="rapid-mlx",
        model="qwen3.5-9b",
        port=9123,
        command=["rapid-mlx", "serve", "qwen3.5-9b"],
        raw_runs={"stream": [{"ttft_ms": 100.0}]},
        summary={"stream": {"ttft_ms": 100.0}},
    )

    assert result["engine"] == "rapid-mlx"
    assert result["model"] == "qwen3.5-9b"
    assert result["port"] == 9123
    assert result["command"] == ["rapid-mlx", "serve", "qwen3.5-9b"]
    assert result["raw_runs"]["stream"] == [{"ttft_ms": 100.0}]
    assert result["summary"]["stream"] == {"ttft_ms": 100.0}
    assert "error" not in result


def test_build_engine_failure_result_shape():
    bench = load_bench_module()

    result = bench.build_engine_failure_result(
        engine="ollama",
        model="qwen3.5:9b",
        port=9124,
        command=["ollama", "serve"],
        error=RuntimeError("startup failed"),
    )

    assert result["engine"] == "ollama"
    assert result["model"] == "qwen3.5:9b"
    assert result["error"] == "startup failed"
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py::test_build_engine_success_result_shape tests/test_bench_vs_ollama.py::test_build_engine_failure_result_shape -q
```

Expected: FAIL because result helpers are missing.

- [ ] **Step 3: Implement engine suite and result builders**

Add these functions:

```python
def build_engine_success_result(
    engine: str,
    model: str,
    port: int,
    command: list[str],
    raw_runs: dict,
    summary: dict,
) -> dict:
    return {
        "engine": engine,
        "model": model,
        "port": port,
        "command": command,
        "raw_runs": raw_runs,
        "summary": summary,
    }


def build_engine_failure_result(
    engine: str,
    model: str,
    port: int | None,
    command: list[str],
    error: Exception,
) -> dict:
    return {
        "engine": engine,
        "model": model,
        "port": port,
        "command": command,
        "error": str(error),
        "raw_runs": {},
        "summary": {},
    }


def run_engine_suite(
    engine: str,
    base_url: str,
    model: str,
    args: CliArgs,
) -> tuple[dict, dict]:
    for _ in range(args.warmups):
        run_stream_once(engine, base_url, model, DECODE_MESSAGES, min(args.max_tokens, 32), args.request_timeout)

    stream_runs = [
        run_stream_once(engine, base_url, model, DECODE_MESSAGES, args.max_tokens, args.request_timeout)
        for _ in range(args.runs)
    ]
    multi_turn = run_multi_turn(engine, base_url, model, min(args.max_tokens, 128), args.request_timeout)
    concurrency = {
        str(level): run_concurrent_throughput(
            engine,
            base_url,
            model,
            level,
            args.max_tokens,
            args.request_timeout,
        )
        for level in args.concurrency
    }
    raw_runs = {"stream": stream_runs}
    summary = {
        "stream": summarize_stream_runs(stream_runs),
        "multi_turn": multi_turn,
        "concurrency": concurrency,
    }
    return raw_runs, summary
```

- [ ] **Step 4: Implement managed benchmark orchestration**

Add these functions:

```python
def benchmark_rapid_mlx(pair: ModelPair, args: CliArgs) -> dict:
    require_executable("rapid-mlx")
    port = find_free_port()
    command = build_rapid_mlx_command(pair.rapid_mlx, port, args.rapid_mlx_args)
    process: ManagedProcess | None = None
    try:
        process = start_process(command, env=offline_env_if_needed(args.no_download))
        wait_for_url(f"http://127.0.0.1:{port}/health", args.startup_timeout)
        raw_runs, summary = run_engine_suite(
            "rapid-mlx",
            f"http://127.0.0.1:{port}",
            pair.rapid_mlx,
            args,
        )
        return build_engine_success_result("rapid-mlx", pair.rapid_mlx, port, command, raw_runs, summary)
    except Exception as exc:
        return build_engine_failure_result("rapid-mlx", pair.rapid_mlx, port, command, exc)
    finally:
        if process:
            process.stop()


def benchmark_ollama(pair: ModelPair, args: CliArgs) -> dict:
    require_executable("ollama")
    port = find_free_port()
    command = ["ollama", "serve"]
    process: ManagedProcess | None = None
    try:
        process = start_process(command, env=build_ollama_environment(port, args.ollama_env))
        wait_for_url(f"http://127.0.0.1:{port}/api/tags", args.startup_timeout)
        raw_runs, summary = run_engine_suite(
            "ollama",
            f"http://127.0.0.1:{port}",
            pair.ollama,
            args,
        )
        return build_engine_success_result("ollama", pair.ollama, port, command, raw_runs, summary)
    except Exception as exc:
        return build_engine_failure_result("ollama", pair.ollama, port, command, exc)
    finally:
        if process:
            process.stop()


def run_benchmark(args: CliArgs) -> dict:
    prepare_models(args)
    result = {
        "metadata": collect_metadata(),
        "config": {
            "runs": args.runs,
            "warmups": args.warmups,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "output_dir": str(args.output_dir),
            "no_pull": args.no_pull,
            "no_download": args.no_download,
            "rapid_mlx_args": args.rapid_mlx_args,
            "ollama_env_keys": sorted(args.ollama_env),
        },
        "model_pairs": [],
    }
    for pair in args.model_pairs:
        print(f"\nBenchmarking {pair.rapid_mlx} vs {pair.ollama}", flush=True)
        rapid_result = benchmark_rapid_mlx(pair, args)
        ollama_result = benchmark_ollama(pair, args)
        result["model_pairs"].append(
            {
                "rapid_mlx_model": pair.rapid_mlx,
                "ollama_model": pair.ollama,
                "rapid-mlx": rapid_result,
                "ollama": ollama_result,
            }
        )
    return result
```

- [ ] **Step 5: Wire `main` to run benchmark and write reports**

Replace the current `main` with:

```python
def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run_benchmark(args)
        paths = write_outputs(result, args.output_dir)
        markdown = render_markdown(result)
        print("\n" + markdown)
        print(f"JSON written to: {paths['json']}")
        print(f"Markdown written to: {paths['markdown']}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
```

- [ ] **Step 6: Run unit tests**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 7: Run CLI help smoke test**

Run:

```bash
python scripts/bench_vs_ollama.py --help
```

Expected: exits 0 and lists `--model-pair`, `--runs`, `--no-pull`, and `--no-download`.

- [ ] **Step 8: Commit**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "feat(bench): orchestrate ollama comparison benchmark"
```

---

### Task 7: Final Verification And Manual Run Notes

**Files:**
- Modify: `scripts/bench_vs_ollama.py`

- [ ] **Step 1: Add concise manual run notes to the script docstring**

Expand the module docstring to:

```python
"""Benchmark Rapid-MLX against Ollama with self-managed server processes.

The script launches one engine at a time on a temporary localhost port, runs
identical no-thinking deterministic workloads, then writes raw JSON and
README-ready Markdown to reports/benchmarks/ollama-comparison/.

Manual usage:
    python scripts/bench_vs_ollama.py
    python scripts/bench_vs_ollama.py --model-pair qwen3.5-4b=qwen3.5:4b --runs 1
    python scripts/bench_vs_ollama.py --no-pull --no-download --runs 1
"""
```

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
python -m pytest tests/test_bench_vs_ollama.py -q
```

Expected: PASS.

- [ ] **Step 3: Run ruff on touched files**

Run:

```bash
python -m ruff check scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
```

Expected: PASS. If `ruff` is not installed, run:

```bash
ruff check scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
```

Expected: PASS.

- [ ] **Step 4: Run CLI help smoke test**

Run:

```bash
python scripts/bench_vs_ollama.py --help
```

Expected: PASS with help text printed.

- [ ] **Step 5: Commit final polish**

```bash
git add scripts/bench_vs_ollama.py tests/test_bench_vs_ollama.py
git commit -m "docs(bench): document ollama comparison usage"
```

---

## Final Acceptance Criteria

- `scripts/bench_vs_ollama.py` exists and can be run as a CLI.
- The CLI launches Rapid-MLX and Ollama sequentially on free temporary localhost ports.
- Default model pairs are `qwen3.5-4b=qwen3.5:4b` and `qwen3.5-9b=qwen3.5:9b`.
- Requests use deterministic no-thinking settings for both engines.
- The script writes JSON and Markdown to `reports/benchmarks/ollama-comparison/`.
- Unit tests are offline and pass with `python -m pytest tests/test_bench_vs_ollama.py -q`.
- CLI help works with `python scripts/bench_vs_ollama.py --help`.
- Real benchmark runs are manual because they download models and start inference servers.
