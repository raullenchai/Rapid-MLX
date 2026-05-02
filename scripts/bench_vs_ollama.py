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
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
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
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.runs < 1:
        parser.error("--runs must be >= 1")
    if ns.warmups < 0:
        parser.error("--warmups must be >= 0")
    if ns.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if any(level < 1 for level in ns.concurrency):
        parser.error("--concurrency values must be >= 1")
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
    _ = args


def offline_env_if_needed(no_download: bool) -> dict[str, str] | None:
    if not no_download:
        return None
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


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


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _nested_dict(value: dict, key: str) -> dict:
    return _as_dict(value.get(key))


def _engine_summary(pair_result: dict, engine: str) -> dict:
    return _nested_dict(_as_dict(pair_result.get(engine)), "summary")


def _engine_error(pair_result: dict, engine: str) -> str | None:
    error = _as_dict(pair_result.get(engine)).get("error")
    if error is None:
        return None
    return str(error)


def _concurrency_sort_key(value: object) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def render_model_pair_table(pair_result: dict) -> str:
    rapid = _engine_summary(pair_result, "rapid-mlx")
    ollama = _engine_summary(pair_result, "ollama")
    rapid_stream = _nested_dict(rapid, "stream")
    ollama_stream = _nested_dict(ollama, "stream")
    rapid_mt = _nested_dict(rapid, "multi_turn")
    ollama_mt = _nested_dict(ollama, "multi_turn")
    rapid_conc = _nested_dict(rapid, "concurrency")
    ollama_conc = _nested_dict(ollama, "concurrency")

    lines = [
        f"## {pair_result['rapid_mlx_model']} vs {pair_result['ollama_model']}",
        "",
    ]
    rapid_error = _engine_error(pair_result, "rapid-mlx")
    ollama_error = _engine_error(pair_result, "ollama")
    if rapid_error:
        lines.append(f"**Rapid-MLX error:** {rapid_error}")
    if ollama_error:
        lines.append(f"**Ollama error:** {ollama_error}")
    if rapid_error or ollama_error:
        lines.append("")
    lines.extend(
        [
        "| Metric | Rapid-MLX | Ollama | Speedup |",
        "|---|---:|---:|---:|",
        ]
    )
    rapid_ttft = rapid_stream.get("ttft_ms")
    ollama_ttft = ollama_stream.get("ttft_ms")
    lines.append(
        "| TTFT | "
        f"{format_number(rapid_ttft, ' ms')} | "
        f"{format_number(ollama_ttft, ' ms')} | "
        f"{format_speedup(latency_speedup(rapid_ttft, ollama_ttft))} |"
    )
    rapid_decode = rapid_stream.get("decode_tok_s")
    ollama_decode = ollama_stream.get("decode_tok_s")
    lines.append(
        "| Decode tok/s | "
        f"{format_number(rapid_decode)} | "
        f"{format_number(ollama_decode)} | "
        f"{format_speedup(throughput_speedup(rapid_decode, ollama_decode))} |"
    )
    rapid_turn = rapid_mt.get("avg_turn_ms")
    ollama_turn = ollama_mt.get("avg_turn_ms")
    lines.append(
        "| Multi-turn latency | "
        f"{format_number(rapid_turn, ' ms')} | "
        f"{format_number(ollama_turn, ' ms')} | "
        f"{format_speedup(latency_speedup(rapid_turn, ollama_turn))} |"
    )
    concurrency_levels = sorted(
        set(rapid_conc) | set(ollama_conc),
        key=_concurrency_sort_key,
    )
    for level in concurrency_levels:
        rapid_tps = rapid_conc.get(level, {}).get("aggregate_tok_s")
        ollama_tps = ollama_conc.get(level, {}).get("aggregate_tok_s")
        lines.append(
            f"| Concurrent throughput ({level} users) | "
            f"{format_number(rapid_tps)} | "
            f"{format_number(ollama_tps)} | "
            f"{format_speedup(throughput_speedup(rapid_tps, ollama_tps))} |"
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
        "- Hardware: `"
        f"{hardware.get('chip_name') or '-'} "
        f"({hardware.get('total_memory_gb') or '-'} GB, "
        f"{hardware.get('gpu_cores') or '-'} GPU cores)`",
        f"- Runs: `{config.get('runs', '-')}`",
        f"- Concurrency: `{config.get('concurrency', '-')}`",
        "",
        "Engines were launched sequentially on temporary localhost ports. Requests "
        "used deterministic no-thinking settings.",
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


def extract_rapid_mlx_message_content(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def extract_ollama_message_content(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


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
    url: str,
    payload: dict,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[list[str], float, float | None, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
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


def post_json(
    url: str,
    payload: dict,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
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
    headers: dict[str, str] | None = None,
) -> dict:
    if engine == "rapid-mlx":
        url = f"{base_url}/v1/chat/completions"
        payload = build_rapid_mlx_payload(model, messages, max_tokens, stream=True)
        lines, start_at, first_content_at, end_at = post_json_lines(
            url, payload, timeout, headers=headers
        )
        parsed = parse_rapid_mlx_stream(lines)
    elif engine == "ollama":
        url = f"{base_url}/api/chat"
        payload = build_ollama_payload(model, messages, max_tokens, stream=True)
        lines, start_at, first_content_at, end_at = post_json_lines(
            url, payload, timeout, headers=headers
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


def run_embedding_once(
    engine: str,
    base_url: str,
    model: str,
    inputs: list[str],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict:
    if engine == "rapid-mlx":
        url = f"{base_url}/v1/embeddings"
        payload = {"model": model, "input": inputs}
        data, latency_ms = post_json(url, payload, timeout, headers=headers)
        vectors = data.get("data") if isinstance(data, dict) else None
        usage = data.get("usage") if isinstance(data, dict) else None
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    elif engine == "ollama":
        url = f"{base_url}/api/embed"
        payload = {"model": model, "input": inputs}
        data, latency_ms = post_json(url, payload, timeout, headers=headers)
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        prompt_tokens = data.get("prompt_eval_count") if isinstance(data, dict) else None
        total_tokens = prompt_tokens
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return {
        "latency_ms": latency_ms,
        "embeddings": len(vectors) if isinstance(vectors, list) else 0,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
    }


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
            content = extract_rapid_mlx_message_content(data)
        elif engine == "ollama":
            url = f"{base_url}/api/chat"
            payload = build_ollama_payload(model, messages, max_tokens, stream=False)
            data, latency_ms = post_json(url, payload, timeout)
            content = extract_ollama_message_content(data)
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


def summarize_embedding_runs(runs: list[dict]) -> dict:
    if not runs:
        return {
            "avg_latency_ms": 0.0,
            "requests": 0,
            "embeddings": 0,
            "prompt_tokens": 0,
        }
    latency = sum(float(run.get("latency_ms", 0.0)) for run in runs) / len(runs)
    return {
        "avg_latency_ms": round(latency, 1),
        "requests": len(runs),
        "embeddings": sum(int(run.get("embeddings", 0)) for run in runs),
        "prompt_tokens": sum(int(run.get("prompt_tokens") or 0) for run in runs),
    }


def _run_concurrent_chat_batch(
    engine: str,
    base_url: str,
    workload: dict,
    concurrency: int,
    timeout: float,
    headers: dict[str, str] | None,
) -> dict:
    start = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_stream_once,
                engine,
                base_url,
                workload["chat_model"],
                workload["chat_messages"],
                workload["max_tokens"],
                timeout,
                headers,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - start
    batch = summarize_concurrent_batch(
        [
            {
                "latency_ms": result["total_ms"],
                "completion_tokens": result["completion_tokens"],
            }
            for result in results
        ],
        elapsed,
    )
    batch["runs"] = results
    return batch


def _run_concurrent_embedding_batch(
    engine: str,
    base_url: str,
    workload: dict,
    concurrency: int,
    timeout: float,
    headers: dict[str, str] | None,
) -> dict:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_embedding_once,
                engine,
                base_url,
                workload["embedding_model"],
                workload["embedding_input"],
                timeout,
                headers=headers,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    batch = summarize_embedding_runs(results)
    batch["runs"] = results
    return batch


def _average_summaries(summaries: list[dict], keys: list[str]) -> dict:
    if not summaries:
        return {}
    averaged: dict[str, float | int] = {}
    for key in keys:
        values = [
            float(summary[key])
            for summary in summaries
            if isinstance(summary.get(key), int | float)
        ]
        if values:
            averaged[key] = round(sum(values) / len(values), 1)
    return averaged


def build_workload(model: str, max_tokens: int) -> dict:
    return {
        "chat_model": model,
        "embedding_model": model,
        "chat_messages": DECODE_MESSAGES,
        "embedding_input": ["Rapid-MLX and Ollama benchmark embedding workload."],
        "max_tokens": max_tokens,
    }


def build_engine_success_result(
    engine: str,
    model: str,
    port: int,
    command: list[str],
    raw_runs: dict,
    summary: dict,
    errors: list[dict] | None = None,
    server_url: str | None = None,
    prepared: bool = True,
) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    server_url = server_url or f"http://127.0.0.1:{port}"
    return {
        "engine": engine,
        "model": model,
        "port": port,
        "command": command,
        "server": {
            "host": "127.0.0.1",
            "port": port,
            "url": server_url,
        },
        "runtime": {
            "prepared": prepared,
            "command": command,
        },
        "metadata": {
            "engine": engine,
            "model": model,
            "runtime": "managed-server",
        },
        "started_at": now,
        "finished_at": now,
        "errors": errors or [],
        "raw_runs": raw_runs,
        "summary": summary,
    }


def build_engine_failure_result(
    engine: str,
    model: str,
    port: int | None,
    command: list[str],
    error: Exception,
    server_url: str | None = None,
    prepared: bool = False,
    stage: str = "engine",
) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    error_text = str(error)
    return {
        "engine": engine,
        "model": model,
        "port": port,
        "command": command,
        "server": {
            "host": "127.0.0.1",
            "port": port,
            "url": server_url or (f"http://127.0.0.1:{port}" if port else None),
        },
        "runtime": {
            "prepared": prepared,
            "command": command,
        },
        "metadata": {
            "engine": engine,
            "model": model,
            "runtime": "managed-server",
        },
        "started_at": now,
        "finished_at": now,
        "error": error_text,
        "errors": [{"stage": stage, "error": error_text}],
        "raw_runs": {},
        "summary": {},
    }


def run_engine_suite(
    engine_name: str,
    base_url: str,
    workload: dict,
    *,
    concurrency_levels: list[int],
    runs_per_level: int,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[dict, dict, list[dict]]:
    raw_runs: dict = {"stream": [], "multi_turn": {}, "chat": {}, "embeddings": {}}
    errors: list[dict] = []
    for run_index in range(1, runs_per_level + 1):
        try:
            raw_runs["stream"].append(
                run_stream_once(
                    engine_name,
                    base_url,
                    workload["chat_model"],
                    workload["chat_messages"],
                    workload["max_tokens"],
                    timeout,
                    headers=headers,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "workload": "stream",
                    "run": run_index,
                    "error": str(exc),
                }
            )
    try:
        raw_runs["multi_turn"] = run_multi_turn(
            engine_name,
            base_url,
            workload["chat_model"],
            min(workload["max_tokens"], 128),
            timeout,
        )
    except Exception as exc:
        errors.append({"workload": "multi_turn", "error": str(exc)})
    for level in concurrency_levels:
        level_key = str(level)
        raw_runs["chat"][level_key] = []
        raw_runs["embeddings"][level_key] = []
        for run_index in range(1, runs_per_level + 1):
            try:
                raw_runs["chat"][level_key].append(
                    _run_concurrent_chat_batch(
                        engine_name, base_url, workload, level, timeout, headers
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "workload": "chat",
                        "concurrency": level,
                        "run": run_index,
                        "error": str(exc),
                    }
                )
            try:
                raw_runs["embeddings"][level_key].append(
                    _run_concurrent_embedding_batch(
                        engine_name, base_url, workload, level, timeout, headers
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "workload": "embeddings",
                        "concurrency": level,
                        "run": run_index,
                        "error": str(exc),
                    }
                )
    concurrency = {
        level: _average_summaries(
            batches,
            [
                "aggregate_tok_s",
                "avg_latency_ms",
                "p95_latency_ms",
                "completion_tokens",
                "requests",
            ],
        )
        for level, batches in raw_runs["chat"].items()
    }
    embeddings = {
        level: _average_summaries(
            batches,
            ["avg_latency_ms", "requests", "embeddings", "prompt_tokens"],
        )
        for level, batches in raw_runs["embeddings"].items()
    }
    summary = {
        "stream": summarize_stream_runs(raw_runs["stream"]),
        "multi_turn": raw_runs["multi_turn"],
        "concurrency": concurrency,
        "embeddings": embeddings,
    }
    return raw_runs, summary, errors


def prepare_rapid_mlx_model(model: str, args: CliArgs) -> bool:
    if args.no_download:
        return False
    require_executable("rapid-mlx")
    # Rapid-MLX loads/downloads models as part of server startup; there is no
    # separate stable download-only CLI, so the managed serve command below is
    # the prep step when downloads are allowed.
    _ = model
    return True


def prepare_ollama_model(model: str, args: CliArgs, env: dict[str, str]) -> bool:
    if args.no_pull:
        return False
    print(f"Pulling Ollama model {model}...", flush=True)
    result = subprocess.run(["ollama", "pull", model], env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ollama pull failed for {model}")
    return True


def benchmark_rapid_mlx(pair: ModelPair, args: CliArgs) -> dict:
    port: int | None = None
    command: list[str] = []
    process: ManagedProcess | None = None
    prepared = False
    try:
        require_executable("rapid-mlx")
        prepared = prepare_rapid_mlx_model(pair.rapid_mlx, args)
        port = find_free_port()
        command = build_rapid_mlx_command(pair.rapid_mlx, port, args.rapid_mlx_args)
        server_url = f"http://127.0.0.1:{port}"
        process = start_process(command, env=offline_env_if_needed(args.no_download))
        wait_for_url(f"{server_url}/health", args.startup_timeout)
        workload = build_workload(pair.rapid_mlx, args.max_tokens)
        for _ in range(args.warmups):
            run_stream_once(
                "rapid-mlx",
                server_url,
                pair.rapid_mlx,
                DECODE_MESSAGES,
                min(args.max_tokens, 32),
                args.request_timeout,
            )
        raw_runs, summary, errors = run_engine_suite(
            "rapid-mlx",
            server_url,
            workload,
            concurrency_levels=args.concurrency,
            runs_per_level=args.runs,
            timeout=args.request_timeout,
        )
        return build_engine_success_result(
            "rapid-mlx",
            pair.rapid_mlx,
            port,
            command,
            raw_runs,
            summary,
            errors=errors,
            server_url=server_url,
            prepared=prepared,
        )
    except Exception as exc:
        return build_engine_failure_result(
            "rapid-mlx", pair.rapid_mlx, port, command, exc, prepared=prepared
        )
    finally:
        if process:
            process.stop()


def benchmark_ollama(pair: ModelPair, args: CliArgs) -> dict:
    port: int | None = None
    command = ["ollama", "serve"]
    process: ManagedProcess | None = None
    prepared = False
    try:
        require_executable("ollama")
        port = find_free_port()
        server_url = f"http://127.0.0.1:{port}"
        env = build_ollama_environment(port, args.ollama_env)
        process = start_process(command, env=env)
        wait_for_url(f"{server_url}/api/tags", args.startup_timeout)
        prepared = prepare_ollama_model(pair.ollama, args, env)
        workload = build_workload(pair.ollama, args.max_tokens)
        for _ in range(args.warmups):
            run_stream_once(
                "ollama",
                server_url,
                pair.ollama,
                DECODE_MESSAGES,
                min(args.max_tokens, 32),
                args.request_timeout,
            )
        raw_runs, summary, errors = run_engine_suite(
            "ollama",
            server_url,
            workload,
            concurrency_levels=args.concurrency,
            runs_per_level=args.runs,
            timeout=args.request_timeout,
        )
        return build_engine_success_result(
            "ollama",
            pair.ollama,
            port,
            command,
            raw_runs,
            summary,
            errors=errors,
            server_url=server_url,
            prepared=prepared,
        )
    except Exception as exc:
        return build_engine_failure_result(
            "ollama", pair.ollama, port, command, exc, prepared=prepared
        )
    finally:
        if process:
            process.stop()


def build_comparisons(pair_result: dict) -> dict:
    rapid = _engine_summary(pair_result, "rapid-mlx")
    ollama = _engine_summary(pair_result, "ollama")
    rapid_stream = _nested_dict(rapid, "stream")
    ollama_stream = _nested_dict(ollama, "stream")
    rapid_conc = _nested_dict(rapid, "concurrency")
    ollama_conc = _nested_dict(ollama, "concurrency")
    rapid_embeddings = _nested_dict(rapid, "embeddings")
    ollama_embeddings = _nested_dict(ollama, "embeddings")
    comparisons = {
        "stream_decode_tok_s_speedup": throughput_speedup(
            rapid_stream.get("decode_tok_s"),
            ollama_stream.get("decode_tok_s"),
        ),
        "stream_ttft_latency_speedup": latency_speedup(
            rapid_stream.get("ttft_ms"),
            ollama_stream.get("ttft_ms"),
        ),
        "concurrency": {},
        "embeddings": {},
    }
    for level in sorted(set(rapid_conc) | set(ollama_conc), key=_concurrency_sort_key):
        rapid_level = _as_dict(rapid_conc.get(level))
        ollama_level = _as_dict(ollama_conc.get(level))
        comparisons["concurrency"][str(level)] = {
            "aggregate_tok_s_speedup": throughput_speedup(
                rapid_level.get("aggregate_tok_s"),
                ollama_level.get("aggregate_tok_s"),
            )
        }
    for level in sorted(
        set(rapid_embeddings) | set(ollama_embeddings),
        key=_concurrency_sort_key,
    ):
        rapid_level = _as_dict(rapid_embeddings.get(level))
        ollama_level = _as_dict(ollama_embeddings.get(level))
        comparisons["embeddings"][str(level)] = {
            "avg_latency_speedup": latency_speedup(
                rapid_level.get("avg_latency_ms"),
                ollama_level.get("avg_latency_ms"),
            )
        }
    return comparisons


def run_benchmark(args: CliArgs) -> dict:
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
            pair_result := {
                "rapid_mlx_model": pair.rapid_mlx,
                "ollama_model": pair.ollama,
                "rapid-mlx": rapid_result,
                "ollama": ollama_result,
            }
        )
        pair_result["comparisons"] = build_comparisons(pair_result)
    return result


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


if __name__ == "__main__":
    raise SystemExit(main())
