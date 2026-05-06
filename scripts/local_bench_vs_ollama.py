from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)
    
try:
    import requests
except ImportError:
    print("Error: requests is required. Install with: pip install requests")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    ttft_ms: float
    decode_tok_s: float
    memory_gen_mb: float
    memory_peak_mb: float
    completion_tokens: int


@dataclass
class ComparisonResult:
    model: str
    rapid: BenchmarkResult | None
    ollama: BenchmarkResult | None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ollama_model_name(model: str) -> str:
    """Convert Rapid-MLX model name to Ollama format."""
    if ":" in model:
        return model
    return f"{model}:4b"


def wait_for_url(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timeout waiting for {url}")


def get_memory_mb(pid: int) -> float:
    """Get memory usage in MB for a process."""
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def find_process_by_name(name: str) -> int | None:
    """Find PID of a process by name."""
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if name.lower() in proc.info["name"].lower():
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def start_rapid_mlx(model: str, port: int) -> subprocess.Popen:
    """Start Rapid-MLX server and return process."""
    cmd = [
        "rapid-mlx", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-thinking",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    wait_for_url(f"{url}/health/ready", timeout=300)
    return proc


def start_ollama(port: int) -> subprocess.Popen:
    """Start Ollama server and return process."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    wait_for_url(f"{url}/api/tags", timeout=60)
    return proc


def benchmark_rapid_mlx(url: str, model: str, max_tokens: int, warmup: bool = True) -> BenchmarkResult:
    """Benchmark Rapid-MLX server."""
    if warmup:
        requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=30,
        )

    pid = find_process_by_name("rapid-mlx")

    memory_gen = 0.0
    memory_peak = 0.0
    if pid:
        memory_peak = get_memory_mb(pid)

    start = time.perf_counter()
    first_token_at = None
    completion_tokens = max_tokens

    with requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Explain CPU in one sentence."}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        if pid:
                            memory_gen = get_memory_mb(pid)
                usage = data.get("usage", {})
                if usage:
                    completion_tokens = usage.get("completion_tokens", max_tokens)
            if line == b"data: [DONE]":
                break
        total_time = time.perf_counter() - start

    if pid:
        memory_peak = max(memory_peak, get_memory_mb(pid))

    ttft_ms = (first_token_at - start) * 1000 if first_token_at else total_time * 1000
    decode_time = total_time - (first_token_at - start) if first_token_at else total_time
    tok_s = completion_tokens / decode_time if decode_time > 0 else 0

    return BenchmarkResult(
        ttft_ms=round(ttft_ms, 1),
        decode_tok_s=round(tok_s, 1),
        memory_gen_mb=round(memory_gen, 1),
        memory_peak_mb=round(memory_peak, 1),
        completion_tokens=completion_tokens,
    )


def benchmark_ollama(url: str, model: str, max_tokens: int, warmup: bool = True) -> BenchmarkResult:
    """Benchmark Ollama server."""
    if warmup:
        requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 8, "temperature": 0},
            },
            timeout=30,
        )

    pid = find_process_by_name("ollama")

    memory_gen = 0.0
    memory_peak = 0.0
    if pid:
        memory_peak = get_memory_mb(pid)

    start = time.perf_counter()
    first_token_at = None
    completion_tokens = 0

    start = time.perf_counter()
    first_token_at = None
    completion_tokens = 0

    with requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Explain CPU in one sentence."}],
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0},
        },
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("message", {}).get("content"):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    if pid:
                        memory_gen = get_memory_mb(pid)
            if data.get("done"):
                completion_tokens = data.get("eval_count", 0)
        total_time = time.perf_counter() - start

    if pid:
        memory_peak = max(memory_peak, get_memory_mb(pid))

    if completion_tokens == 0:
        completion_tokens = max_tokens

    ttft_ms = (first_token_at - start) * 1000 if first_token_at else total_time * 1000
    decode_time = total_time - (first_token_at - start) if first_token_at else total_time
    tok_s = completion_tokens / decode_time if decode_time > 0 else 0

    return BenchmarkResult(
        ttft_ms=round(ttft_ms, 1),
        decode_tok_s=round(tok_s, 1),
        memory_gen_mb=round(memory_gen, 1),
        memory_peak_mb=round(memory_peak, 1),
        completion_tokens=completion_tokens,
    )


def speedup(a: float, b: float) -> str:
    """Calculate speedup ratio. a/b means b is X times faster than a."""
    if a <= 0 or b <= 0:
        return "-"
    return f"{a / b:.2f}x"


def render_table(result: ComparisonResult) -> str:
    rapid = result.rapid
    ollama = result.ollama

    lines = [
        f"# Benchmark Results: {result.model}",
        "",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Metric | Rapid-MLX | Ollama | Speedup |",
        "|---|---|---|---|",
    ]

    if rapid and ollama:
        lines.append(
            f"| TTFT (ms) | {rapid.ttft_ms} | {ollama.ttft_ms} | "
            f"{speedup(ollama.ttft_ms, rapid.ttft_ms)} |"
        )
        lines.append(
            f"| tok/s | {rapid.decode_tok_s} | {ollama.decode_tok_s} | "
            f"{speedup(rapid.decode_tok_s, ollama.decode_tok_s)} |"
        )
    elif rapid:
        lines.append(f"| TTFT (ms) | {rapid.ttft_ms} | - | - |")
        lines.append(f"| tok/s | {rapid.decode_tok_s} | - | - |")
    elif ollama:
        lines.append(f"| TTFT (ms) | - | {ollama.ttft_ms} | - |")
        lines.append(f"| tok/s | - | {ollama.decode_tok_s} | - |")

    for label in ["Memory - Gen (MB)", "Memory - Peak (MB)"]:
        if rapid:
            r_val = rapid.memory_gen_mb if label == "Memory - Gen (MB)" else rapid.memory_peak_mb
        else:
            r_val = 0
        if ollama:
            o_val = ollama.memory_gen_mb if label == "Memory - Gen (MB)" else ollama.memory_peak_mb
        else:
            o_val = 0
        lines.append(f"| {label} | {r_val} | {o_val} | - |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Rapid-MLX vs Ollama")
    parser.add_argument("--model", default="qwen3.5-4b", help="Model to test")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    model = args.model
    ollama_model = ollama_model_name(model)

    print(f"Testing model: {model}")
    print("=" * 50)

    print("\n[Benchmarking Rapid-MLX...]")
    rapid_proc = None
    rapid_result = None
    try:
        port = find_free_port()
        rapid_proc = start_rapid_mlx(model, port)
        url = f"http://127.0.0.1:{port}"

        runs = []
        for i in range(args.runs):
            print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
            result = benchmark_rapid_mlx(
                url, model, args.max_tokens, warmup=(i == 0 and not args.no_warmup)
            )
            runs.append(result)
            print(f"ttft={result.ttft_ms}ms, tok/s={result.decode_tok_s}")

        rapid_result = BenchmarkResult(
            ttft_ms=round(sum(r.ttft_ms for r in runs) / len(runs), 1),
            decode_tok_s=round(sum(r.decode_tok_s for r in runs) / len(runs), 1),
            memory_gen_mb=round(sum(r.memory_gen_mb for r in runs) / len(runs), 1),
            memory_peak_mb=round(sum(r.memory_peak_mb for r in runs) / len(runs), 1),
            completion_tokens=runs[0].completion_tokens,
        )
        print(f"  Average: ttft={rapid_result.ttft_ms}ms, tok/s={rapid_result.decode_tok_s}")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        if rapid_proc:
            rapid_proc.terminate()
            rapid_proc.wait(timeout=10)

    print("\n[Benchmarking Ollama...]")
    ollama_proc = None
    ollama_result = None
    try:
        port = find_free_port()
        ollama_proc = start_ollama(port)
        url = f"http://127.0.0.1:{port}"

        runs = []
        for i in range(args.runs):
            print(f"  Run {i + 1}/{args.runs}...", end=" ", flush=True)
            result = benchmark_ollama(
                url, ollama_model, args.max_tokens, warmup=(i == 0 and not args.no_warmup)
            )
            runs.append(result)
            print(f"ttft={result.ttft_ms}ms, tok/s={result.decode_tok_s}")

        ollama_result = BenchmarkResult(
            ttft_ms=round(sum(r.ttft_ms for r in runs) / len(runs), 1),
            decode_tok_s=round(sum(r.decode_tok_s for r in runs) / len(runs), 1),
            memory_gen_mb=round(sum(r.memory_gen_mb for r in runs) / len(runs), 1),
            memory_peak_mb=round(sum(r.memory_peak_mb for r in runs) / len(runs), 1),
            completion_tokens=runs[0].completion_tokens,
        )
        print(f"  Average: ttft={ollama_result.ttft_ms}ms, tok/s={ollama_result.decode_tok_s}")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        if ollama_proc:
            ollama_proc.terminate()
            ollama_proc.wait(timeout=10)

    result = ComparisonResult(model=model, rapid=rapid_result, ollama=ollama_result)

    print("\n" + "=" * 50)
    print(render_table(result))

    if args.output:
        data = {
            "model": model,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "rapid-mlx": {
                "ttft_ms": rapid_result.ttft_ms if rapid_result else None,
                "decode_tok_s": rapid_result.decode_tok_s if rapid_result else None,
                "memory_gen_mb": rapid_result.memory_gen_mb if rapid_result else None,
                "memory_peak_mb": rapid_result.memory_peak_mb if rapid_result else None,
            },
            "ollama": {
                "ttft_ms": ollama_result.ttft_ms if ollama_result else None,
                "decode_tok_s": ollama_result.decode_tok_s if ollama_result else None,
                "memory_gen_mb": ollama_result.memory_gen_mb if ollama_result else None,
                "memory_peak_mb": ollama_result.memory_peak_mb if ollama_result else None,
            },
        }
        args.output.write_text(json.dumps(data, indent=2))
        print(f"\nJSON written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())