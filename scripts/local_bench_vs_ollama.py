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

# Fix Windows console encoding for box-drawing chars
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
    """Benchmark Rapid-MLX server with improved stream parsing."""
    if warmup:
        try:
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
        except Exception as e:
            print(f"  Warning: warmup failed: {e}")

    pid = find_process_by_name("rapid-mlx")

    memory_gen = 0.0
    memory_peak = 0.0
    if pid:
        memory_peak = get_memory_mb(pid)

    start = time.perf_counter()
    first_token_at = None
    completion_tokens = max_tokens
    debug_log = []

    try:
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
            # Check for non-200 status
            if resp.status_code != 200:
                debug_log.append(f"Non-200 status: {resp.status_code}")
                # Try to read error response
                try:
                    error_body = resp.text[:500]
                    debug_log.append(f"Error body: {error_body}")
                except Exception:
                    pass
            
            for line in resp.iter_lines():
                if not line:
                    continue
                
                # Decode to string for checking
                try:
                    line_str = line.decode("utf-8")
                except Exception:
                    debug_log.append(f"Failed to decode line: {line[:50]}")
                    continue
                
                # Skip non-SSE lines (lines that dont start with "data: ")
                if not line_str.startswith("data: "):
                    # Check for error responses
                    if line_str.startswith("{") or line_str.startswith("["):
                        try:
                            data = json.loads(line_str)
                            if "error" in data:
                                debug_log.append(f"Server error: {data.get('error')}")
                                continue
                            # Handle other JSON responses
                            debug_log.append(f"Non-SSE JSON: {line_str[:100]}")
                        except json.JSONDecodeError as e:
                            debug_log.append(f"Non-SSE parse error: {e}, content: {line_str[:100]}")
                    continue
                
                # Extract the JSON data after "data: " prefix
                data_str = line_str[6:].strip()
                
                # Skip empty data markers
                if not data_str or data_str == "[DONE]":
                    continue
                
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError as e:
                    debug_log.append(f"JSON decode error: {e}, data: {data_str[:100]}")
                    continue
                
                # Process the response
                try:
                    if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            if pid:
                                memory_gen = get_memory_mb(pid)
                    
                    usage = data.get("usage", {})
                    if usage:
                        completion_tokens = usage.get("completion_tokens", max_tokens)
                except (KeyError, IndexError, TypeError) as e:
                    debug_log.append(f"Data parse error: {e}")
                    continue
            
            total_time = time.perf_counter() - start

    except requests.exceptions.RequestException as e:
        debug_log.append(f"Request exception: {e}")
        total_time = time.perf_counter() - start
    
    except Exception as e:
        debug_log.append(f"Unexpected error: {e}")
        total_time = time.perf_counter() - start

    # Log debug information if there were issues
    if debug_log and not first_token_at:
        print(f"  Debug: {debug_log[:3]}")

    if pid:
        memory_peak = max(memory_peak, get_memory_mb(pid))

    # Handle case where no tokens were generated
    if not first_token_at:
        ttft_ms = total_time * 1000
        decode_time = total_time
    else:
        ttft_ms = (first_token_at - start) * 1000
        decode_time = total_time - (first_token_at - start)
    
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
    generated_content = ""
    eval_count = 0

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
                content = data["message"]["content"]
                generated_content += content
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    if pid:
                        memory_gen = get_memory_mb(pid)
            if data.get("done"):
                eval_count = data.get("eval_count") or 0

    total_time = time.perf_counter() - start

    # Use eval_count OR generated content length as fallback
    if eval_count > 0:
        completion_tokens = eval_count
    elif len(generated_content) > 0:
        completion_tokens = len(generated_content)
    else:
        completion_tokens = max_tokens

    if pid:
        try:
            memory_peak = max(memory_peak, get_memory_mb(pid))
        except:
            pass

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


def speedup(a: float, b: float, format_text: str = " faster") -> str:
    """Calculate speedup ratio. a/b means b is X times faster than a."""
    if a <= 0 or b <= 0:
        return "-"
    ratio = a / b
    if ratio >= 1:
        return f"{ratio:.2f}x{format_text}"
    else:
        return f"{1/ratio:.2f}x slower"


def render_table(result: ComparisonResult) -> str:
    """Render benchmark results as a clean, terminal-friendly table."""
    rapid = result.rapid
    ollama = result.ollama

    def fmt_val(val):
        if val is None:
            return "         -"
        return f"{val:>10.1f}"

    sep = "├" + "─" * 20 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤"
    header = ("╔" + "═" * 20 + "╦" + "═" * 12 + "╦" + "═" * 12 + "╦" + "═" * 12 + "╗\n"
             "║" + "Metric".center(20) + "║" + "Rapid-MLX".center(12) + "║" + "Ollama".center(12) + "║" + "Speedup".center(12) + "║\n"
             + sep.replace("├", "╠").replace("┤", "╣"))
    footer = "╚" + "═" * 20 + "╩" + "═" * 12 + "╩" + "═" * 12 + "╩" + "═" * 12 + "╝"

    lines = [
        "\n╔═╗ Benchmark Results: " + result.model + " ╔═╗",
        "║  Timestamp: " + datetime.now().isoformat(timespec='seconds') + " ║",
        header,
    ]

    # TTFT (ms) - lower is better
    if rapid and ollama:
        sp = speedup(ollama.ttft_ms, rapid.ttft_ms)
        lines.append("║" + "TTFT (ms)".ljust(20) + "║" + fmt_val(rapid.ttft_ms) + "║" + fmt_val(ollama.ttft_ms) + "║" + sp.center(12) + "║")
    elif rapid:
        lines.append("║" + "TTFT (ms)".ljust(20) + "║" + fmt_val(rapid.ttft_ms) + "║" + "         -".center(12) + "║" + "         -".center(12) + "║")
    elif ollama:
        lines.append("║" + "TTFT (ms)".ljust(20) + "║" + "         -".center(12) + "║" + fmt_val(ollama.ttft_ms) + "║" + "         -".center(12) + "║")

    # tok/s - higher is better
    if rapid and ollama:
        sp = speedup(rapid.decode_tok_s, ollama.decode_tok_s)
        lines.append("║" + "tok/s".ljust(20) + "║" + fmt_val(rapid.decode_tok_s) + "║" + fmt_val(ollama.decode_tok_s) + "║" + sp.center(12) + "║")
    elif rapid:
        lines.append("║" + "tok/s".ljust(20) + "║" + fmt_val(rapid.decode_tok_s) + "║" + "         -".center(12) + "║" + "         -".center(12) + "║")
    elif ollama:
        lines.append("║" + "tok/s".ljust(20) + "║" + "         -".center(12) + "║" + fmt_val(ollama.decode_tok_s) + "║" + "         -".center(12) + "║")

    lines.append(sep)

    # Memory - Gen (MB)
    if rapid and ollama:
        lines.append("║" + "Memory - Gen (MB)".ljust(20) + "║" + fmt_val(rapid.memory_gen_mb) + "║" + fmt_val(ollama.memory_gen_mb) + "║" + "         -".center(12) + "║")
    elif rapid:
        lines.append("║" + "Memory - Gen (MB)".ljust(20) + "║" + fmt_val(rapid.memory_gen_mb) + "║" + "         -".center(12) + "║" + "         -".center(12) + "║")
    elif ollama:
        lines.append("║" + "Memory - Gen (MB)".ljust(20) + "║" + "         -".center(12) + "║" + fmt_val(ollama.memory_gen_mb) + "║" + "         -".center(12) + "║")

    # Memory - Peak (MB)
    if rapid and ollama:
        lines.append("║" + "Memory - Peak (MB)".ljust(20) + "║" + fmt_val(rapid.memory_peak_mb) + "║" + fmt_val(ollama.memory_peak_mb) + "║" + "         -".center(12) + "║")
    elif rapid:
        lines.append("║" + "Memory - Peak (MB)".ljust(20) + "║" + fmt_val(rapid.memory_peak_mb) + "║" + "         -".center(12) + "║" + "         -".center(12) + "║")
    elif ollama:
        lines.append("║" + "Memory - Peak (MB)".ljust(20) + "║" + "         -".center(12) + "║" + fmt_val(ollama.memory_peak_mb) + "║" + "         -".center(12) + "║")

    lines.append(footer)

    # Summary
    if rapid and ollama:
        if rapid.ttft_ms and ollama.ttft_ms:
            ttft_ratio = ollama.ttft_ms / rapid.ttft_ms if rapid.ttft_ms > 0 else 0
            tok_ratio = rapid.decode_tok_s / ollama.decode_tok_s if ollama.decode_tok_s > 0 else 0
            mem_diff = rapid.memory_peak_mb - ollama.memory_peak_mb
            mem_status = str(abs(int(mem_diff))) + "MB " + ("less" if mem_diff < 0 else ("more" if mem_diff > 0 else "same"))

            lines.append("")
            lines.append("╟─ SUMMARY ────────────────────────╢")
            lines.append("║ TTFT:        " + f"{ttft_ratio:>8.2f}x faster    ║".format(ttft_ratio=ttft_ratio))
            lines.append("║ Decode:      " + f"{tok_ratio:>8.2f}x faster    ║".format(tok_ratio=tok_ratio))
            lines.append("║ Peak Memory:" + mem_status.rjust(19) + "║")
            lines.append("╚═════════════════════════════════╝")

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
