from __future__ import annotations

import argparse
import json
import os
import re
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


# ── ANSI color palette ────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    WHITE = "\033[97m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"

    @staticmethod
    def strip(s: str) -> str:
        return re.sub(r"\033\[[^m]*m", "", s)

    @staticmethod
    def ljust(s: str, width: int) -> str:
        return s + " " * max(0, width - len(C.strip(s)))

    @staticmethod
    def rjust(s: str, width: int) -> str:
        return " " * max(0, width - len(C.strip(s))) + s

    @staticmethod
    def center(s: str, width: int) -> str:
        pad = max(0, width - len(C.strip(s)))
        return " " * (pad // 2) + s + " " * (pad - pad // 2)


# ── Data classes ──────────────────────────────────────────────────────────────
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
    runs: int
    max_tokens: int


# ── Utilities ─────────────────────────────────────────────────────────────────
def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def ollama_model_name(model: str) -> str:
    """Best-effort conversion of a Rapid-MLX model name to an Ollama tag."""
    if ":" in model:
        return model
    known = {
        "qwen3.5-4b-4bit": "qwen3:4b",
        "qwen3.5-8b": "qwen3:8b",
        "qwen3.5-14b": "qwen3:14b",
        "qwen3.5-32b": "qwen3:32b",
        "llama3.2-3b": "llama3.2:3b",
        "llama3.2-8b": "llama3.2:8b",
        "mistral-7b": "mistral:7b",
        "phi4-4b": "phi4:4b",
        "phi4-mini": "phi4-mini:latest",
        "gemma3-4b": "gemma3:4b",
        "gemma3-12b-4bit": "gemma3:12b",
    }
    if model in known:
        return known[model]
    m = re.match(r"^(.+?)-(\d+b)$", model, re.IGNORECASE)
    if m:
        return f"{m.group(1)}:{m.group(2).lower()}"
    return model


def wait_for_url(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timeout waiting for {url}")


# ── Memory tracking ───────────────────────────────────────────────────────────
def get_process_tree_mb(pid: int) -> float:
    """
    Total RSS (MB) for a process and all its descendants.
    Essential for Ollama which spawns Metal/CUDA runner subprocesses,
    and for rapid-mlx which may fork helpers. Without this, memory
    shows near-zero because the actual work happens in child processes.
    """
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
        total = 0.0
        for p in procs:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def find_process_by_cmdline(keywords: list[str]) -> int | None:
    """
    Find PID by scanning cmdline arguments.
    More reliable than name-matching on macOS where psutil may show
    'Python' or 'python3' instead of the script name.
    """
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if all(kw.lower() in cmdline for kw in keywords):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def find_rapid_mlx_pid() -> int | None:
    pid = find_process_by_cmdline(["rapid-mlx", "serve"])
    if pid:
        return pid
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if "rapid" in (proc.info.get("name") or "").lower():
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def find_ollama_pid() -> int | None:
    pid = find_process_by_cmdline(["ollama", "serve"])
    if pid:
        return pid
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if "ollama" in (proc.info.get("name") or "").lower():
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


# ── Server management ─────────────────────────────────────────────────────────
def start_rapid_mlx(model: str, port: int) -> subprocess.Popen:
    cmd = [
        "rapid-mlx",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-thinking",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_url(f"http://127.0.0.1:{port}/health/ready", timeout=300)
    return proc


def start_ollama(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    wait_for_url(f"http://127.0.0.1:{port}/api/tags", timeout=60)
    return proc


# ── Benchmarking ──────────────────────────────────────────────────────────────
def benchmark_rapid_mlx(
    url: str, model: str, max_tokens: int, warmup: bool = True
) -> BenchmarkResult:
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
            print(f"  {C.YELLOW}Warning: warmup failed: {e}{C.RESET}")

    pid = find_rapid_mlx_pid()
    mem_before = get_process_tree_mb(pid) if pid else 0.0
    memory_gen = 0.0
    memory_peak = mem_before
    debug_log: list[str] = []

    start = time.perf_counter()
    first_token_at = None
    completion_tokens = max_tokens

    try:
        with requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "Explain CPU in one sentence."}
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                debug_log.append(f"HTTP {resp.status_code}")

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    line_str = line.decode("utf-8")
                except Exception:
                    continue
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                try:
                    if data.get("choices") and data["choices"][0].get("delta", {}).get(
                        "content"
                    ):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            if pid:
                                memory_gen = get_process_tree_mb(pid)
                        if pid:
                            memory_peak = max(memory_peak, get_process_tree_mb(pid))
                    usage = data.get("usage") or {}
                    if usage.get("completion_tokens"):
                        completion_tokens = usage["completion_tokens"]
                except (KeyError, IndexError, TypeError):
                    pass

        total_time = time.perf_counter() - start

    except Exception as e:
        debug_log.append(str(e))
        total_time = time.perf_counter() - start

    if debug_log and not first_token_at:
        print(f"  {C.YELLOW}Debug: {debug_log[:2]}{C.RESET}")

    if pid:
        memory_peak = max(memory_peak, get_process_tree_mb(pid))

    ttft_ms = (first_token_at - start) * 1000 if first_token_at else total_time * 1000
    decode_time = (
        (total_time - (first_token_at - start)) if first_token_at else total_time
    )
    tok_s = completion_tokens / decode_time if decode_time > 0 else 0.0
    mem_growth = max(0.0, memory_gen - mem_before) if memory_gen > 0 else 0.0

    return BenchmarkResult(
        ttft_ms=round(ttft_ms, 1),
        decode_tok_s=round(tok_s, 1),
        memory_gen_mb=round(mem_growth, 1),
        memory_peak_mb=round(memory_peak, 1),
        completion_tokens=completion_tokens,
    )


def debug_ollama_stream(url: str, model: str) -> None:
    print(f"\n  {C.GRAY}[DEBUG] Raw Ollama stream (first 10 chunks):{C.RESET}")
    try:
        with requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say hi."}],
                "stream": True,
                "think": False,
                "options": {"num_predict": 10, "temperature": 0},
            },
            stream=True,
            timeout=60,
        ) as resp:
            for count, line in enumerate(resp.iter_lines()):
                if not line or count >= 10:
                    break
                try:
                    print(f"    chunk {count}: {json.dumps(json.loads(line))[:200]}")
                except Exception:
                    print(f"    raw: {line[:200]}")
    except Exception as e:
        print(f"    debug failed: {e}")
    print()


def benchmark_ollama(
    url: str, model: str, max_tokens: int, warmup: bool = True, debug: bool = False
) -> BenchmarkResult:
    if warmup:
        try:
            requests.post(
                f"{url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "think": False,
                    "options": {"num_predict": 8, "temperature": 0},
                },
                timeout=60,
            )
        except Exception as e:
            print(f"  {C.YELLOW}Warning: Ollama warmup failed: {e}{C.RESET}")

    if debug:
        debug_ollama_stream(url, model)

    pid = find_ollama_pid()
    mem_before = get_process_tree_mb(pid) if pid else 0.0
    memory_gen = 0.0
    memory_peak = mem_before

    start = time.perf_counter()
    first_token_at = None
    generated_content = ""
    completion_tokens = 0

    try:
        with requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Explain CPU in one sentence."}
                ],
                "stream": True,
                "think": False,
                "options": {"num_predict": max_tokens, "temperature": 0},
            },
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                print(
                    f"  {C.YELLOW}Warning: Ollama status {resp.status_code}: {resp.text[:200]}{C.RESET}"
                )

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = data.get("message", {})
                chunk_text = msg.get("content") or msg.get("thinking") or ""
                if chunk_text:
                    generated_content += msg.get("content", "")
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        if pid:
                            memory_gen = get_process_tree_mb(pid)
                    if pid:
                        memory_peak = max(memory_peak, get_process_tree_mb(pid))

                if data.get("done"):
                    # eval_count = generated tokens (prompt_eval_count = prompt tokens)
                    ec = data.get("eval_count") or 0
                    if ec > 0:
                        completion_tokens = ec

    except requests.exceptions.RequestException as e:
        print(f"  {C.YELLOW}Warning: Ollama request error: {e}{C.RESET}")
    except Exception as e:
        print(f"  {C.YELLOW}Warning: Ollama error: {e}{C.RESET}")

    total_time = time.perf_counter() - start

    if completion_tokens == 0:
        if generated_content:
            completion_tokens = max(1, int(len(generated_content) / 4))
        else:
            completion_tokens = max_tokens
            print(
                f"  {C.YELLOW}Warning: no tokens received — check model name and that it is pulled{C.RESET}"
            )

    if pid:
        try:
            memory_peak = max(memory_peak, get_process_tree_mb(pid))
        except Exception:
            pass

    ttft_ms = (first_token_at - start) * 1000 if first_token_at else total_time * 1000
    decode_time = (total_time - (first_token_at - start)) if first_token_at else 0.0
    tok_s = completion_tokens / decode_time if decode_time > 0 else 0.0
    mem_growth = max(0.0, memory_gen - mem_before) if memory_gen > 0 else 0.0

    return BenchmarkResult(
        ttft_ms=round(ttft_ms, 1),
        decode_tok_s=round(tok_s, 1),
        memory_gen_mb=round(mem_growth, 1),
        memory_peak_mb=round(memory_peak, 1),
        completion_tokens=completion_tokens,
    )


# ── Terminal display ──────────────────────────────────────────────────────────
BAR_W = 22
W = 74  # total box width


def make_bar(value: float, max_value: float, width: int, color: str) -> str:
    filled = int(round((value / max_value) * width)) if max_value > 0 else 0
    filled = max(0, min(width, filled))
    return f"{color}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


def box_line(inner: str) -> None:
    pad = W - 2 - len(C.strip(inner))
    print(f"{C.GRAY}│{C.RESET}{inner}{' ' * max(0, pad)}{C.GRAY}│{C.RESET}")


def blank() -> None:
    box_line("")


def rule(ch: str = "─", left: str = "├", right: str = "┤") -> None:
    print(f"{C.GRAY}{left}{ch * (W - 2)}{right}{C.RESET}")


def speedup_str(ratio: float) -> str:
    if ratio <= 0:
        return f"{C.GRAY}—{C.RESET}"
    color = C.GREEN if ratio >= 2.0 else (C.YELLOW if ratio >= 1.2 else C.RED)
    arrow = "▲" if ratio >= 1.0 else "▼"
    r = ratio if ratio >= 1.0 else 1 / ratio
    return f"{color}{arrow} {r:.2f}×{C.RESET}"


def metric_row(label: str, r_val: str, o_val: str, sp: str, note: str = "") -> None:
    lc = C.ljust(f"  {C.WHITE}{label}{C.RESET}", 24)
    rc = C.rjust(r_val, 15)
    oc = C.rjust(o_val, 15)
    sc = C.center(sp, 14)
    nc = f"  {C.GRAY}{note}{C.RESET}" if note else ""
    box_line(f"{lc}{rc}{oc}{sc}{nc}")


def bar_section(
    label: str, r_val: float, o_val: float, unit: str, r_label: str, o_label: str
) -> None:
    lbl = f"  {C.DIM}{label}{C.RESET}"
    box_line(lbl)
    max_v = max(r_val, o_val, 0.001)
    for tag, val, color in [(r_label, r_val, C.BLUE), (o_label, o_val, C.GREEN)]:
        bar = make_bar(val, max_v, BAR_W, color)
        num = f"{color}{val:>7.1f}{C.RESET} {C.DIM}{unit}{C.RESET}"
        tag_ = C.ljust(f"  {C.DIM}{tag}{C.RESET}", 16)
        box_line(f"{tag_}{bar} {num}")


def render_results(result: ComparisonResult) -> None:
    rapid = result.rapid
    ollama = result.ollama

    print()
    print(f"{C.GRAY}╭{'─' * (W - 2)}╮{C.RESET}")

    # Title
    title = f"  {C.BOLD}{C.WHITE}⚡ Benchmark Results{C.RESET}  {C.CYAN}{result.model}{C.RESET}"
    ts = f"{C.DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}  "
    pad = W - 2 - len(C.strip(title)) - len(C.strip(ts))
    print(f"{C.GRAY}│{C.RESET}{title}{' ' * max(0, pad)}{ts}{C.GRAY}│{C.RESET}")

    meta = f"  {C.DIM}{result.runs} runs · max_tokens={result.max_tokens}{C.RESET}"
    box_line(meta)
    blank()

    # Column headers
    hl = C.ljust(f"  {C.DIM}METRIC{C.RESET}", 24)
    hr = C.rjust(f"{C.BOLD}{C.BLUE}Rapid-MLX{C.RESET}", 15)
    ho = C.rjust(f"{C.BOLD}{C.GREEN}Ollama{C.RESET}", 15)
    hs = C.center(f"{C.DIM}SPEEDUP{C.RESET}", 14)
    box_line(f"{hl}{hr}{ho}{hs}")
    rule("─")

    # ── Performance ───────────────────────────────────────────────────────────
    if rapid and ollama:
        ttft_ratio = ollama.ttft_ms / rapid.ttft_ms if rapid.ttft_ms > 0 else 0.0
        tok_ratio = (
            rapid.decode_tok_s / ollama.decode_tok_s if ollama.decode_tok_s > 0 else 0.0
        )

        token_note = ""
        if abs(rapid.completion_tokens - ollama.completion_tokens) > 2:
            token_note = f"⚠ token counts differ ({rapid.completion_tokens} vs {ollama.completion_tokens})"

        def fv(v: float, unit: str, color: str) -> str:
            return f"{color}{v:>7.1f}{C.RESET} {C.DIM}{unit}{C.RESET}"

        metric_row(
            "TTFT  (lower=better)",
            fv(rapid.ttft_ms, "ms", C.BLUE),
            fv(ollama.ttft_ms, "ms", C.GREEN),
            speedup_str(ttft_ratio),
        )
        metric_row(
            "Decode  (higher=better)",
            fv(rapid.decode_tok_s, "tok/s", C.BLUE),
            fv(ollama.decode_tok_s, "tok/s", C.GREEN),
            speedup_str(tok_ratio),
            note=token_note,
        )
    elif rapid:
        metric_row("TTFT", f"{rapid.ttft_ms:.1f} ms", "—", "—")
        metric_row("Decode speed", f"{rapid.decode_tok_s:.1f} tok/s", "—", "—")
    elif ollama:
        metric_row("TTFT", "—", f"{ollama.ttft_ms:.1f} ms", "—")
        metric_row("Decode speed", "—", f"{ollama.decode_tok_s:.1f} tok/s", "—")

    # ── Bar charts ────────────────────────────────────────────────────────────
    if rapid and ollama:
        blank()
        rule("╌")
        blank()
        bar_section(
            "TTFT  (lower = better)",
            rapid.ttft_ms,
            ollama.ttft_ms,
            "ms",
            f"{C.BLUE}Rapid-MLX{C.RESET}",
            f"{C.GREEN}Ollama   {C.RESET}",
        )
        blank()
        bar_section(
            "Decode speed  (higher = better)",
            rapid.decode_tok_s,
            ollama.decode_tok_s,
            "tok/s",
            f"{C.BLUE}Rapid-MLX{C.RESET}",
            f"{C.GREEN}Ollama   {C.RESET}",
        )

    # ── Memory ────────────────────────────────────────────────────────────────
    blank()
    rule("─")

    def mem_val(v: float, color: str) -> str:
        if v <= 0:
            return f"{C.GRAY}{'N/A':>7}{C.RESET}"
        return f"{color}{v:>7.1f}{C.RESET} {C.DIM}MB{C.RESET}"

    if rapid and ollama:
        # Peak RSS
        r_peak = rapid.memory_peak_mb
        o_peak = ollama.memory_peak_mb
        diff = r_peak - o_peak
        if r_peak > 0 and o_peak > 0:
            mem_note = (
                f"{C.GREEN}Rapid {abs(diff):.0f} MB less{C.RESET}"
                if diff < 0
                else (
                    f"{C.YELLOW}Rapid {diff:.0f} MB more{C.RESET}"
                    if diff > 5
                    else "≈ same"
                )
            )
        else:
            mem_note = f"{C.GRAY}(process not found for one runner){C.RESET}"
        metric_row(
            "Peak RSS (process tree)",
            mem_val(r_peak, C.BLUE),
            mem_val(o_peak, C.GREEN),
            "—",
            note=mem_note,
        )
        metric_row(
            "RSS at gen start",
            mem_val(rapid.memory_gen_mb, C.BLUE),
            mem_val(ollama.memory_gen_mb, C.GREEN),
            "—",
        )
    elif rapid:
        metric_row("Peak RSS", mem_val(rapid.memory_peak_mb, C.BLUE), "—", "—")
    elif ollama:
        metric_row("Peak RSS", "—", mem_val(ollama.memory_peak_mb, C.GREEN), "—")

    blank()
    note_line = f"  {C.DIM}ℹ  RSS = full process tree incl. children. GPU/Metal VRAM is not counted.{C.RESET}"
    box_line(note_line)

    # ── Summary ───────────────────────────────────────────────────────────────
    if rapid and ollama:
        blank()
        rule("═", "╞", "╡")
        blank()

        ttft_ratio = ollama.ttft_ms / rapid.ttft_ms if rapid.ttft_ms > 0 else 0.0
        tok_ratio = (
            rapid.decode_tok_s / ollama.decode_tok_s if ollama.decode_tok_s > 0 else 0.0
        )

        def summary_row(metric: str, ratio: float, desc: str) -> None:
            if ratio >= 1.0:
                winner = f"{C.BOLD}{C.BLUE}Rapid-MLX{C.RESET}"
                rs = f"{C.GREEN}{ratio:.2f}×{C.RESET} {C.DIM}{desc}{C.RESET}"
            else:
                winner = f"{C.BOLD}{C.GREEN}Ollama{C.RESET}"
                rs = f"{C.YELLOW}{1 / ratio:.2f}×{C.RESET} {C.DIM}{desc}{C.RESET}"
            box_line(f"  {C.DIM}{metric:<20}{C.RESET}  {winner}   {rs}")

        summary_row("TTFT", ttft_ratio, "faster first token")
        summary_row("Decode speed", tok_ratio, "faster decode")
        blank()

    print(f"{C.GRAY}╰{'─' * (W - 2)}╯{C.RESET}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Rapid-MLX vs Ollama")
    parser.add_argument("--model", default="qwen3.5-4b-4bit", help="Rapid-MLX model name")
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model tag (e.g. qwen3:4b). Auto-derived if omitted.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens per request"
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    parser.add_argument(
        "--debug", action="store_true", help="Print raw Ollama chunks on run 1"
    )
    parser.add_argument("--output", type=Path, help="Write JSON results to file")
    args = parser.parse_args()

    model = args.model
    ollama_model = args.ollama_model or ollama_model_name(model)

    print(f"\n{C.BOLD}{C.WHITE}⚡ rapid-mlx vs ollama benchmark{C.RESET}")
    print(
        f"{C.DIM}model={model}  ollama-tag={ollama_model}  runs={args.runs}  max-tokens={args.max_tokens}{C.RESET}"
    )
    print(f"{C.DIM}(use --ollama-model to override the Ollama tag if wrong){C.RESET}\n")

    # ── Rapid-MLX ─────────────────────────────────────────────────────────────
    print(f"{C.BOLD}{C.BLUE}▶ Benchmarking Rapid-MLX...{C.RESET}")
    rapid_proc, rapid_result = None, None
    try:
        port = find_free_port()
        rapid_proc = start_rapid_mlx(model, port)
        url = f"http://127.0.0.1:{port}"
        runs = []
        for i in range(args.runs):
            print(f"  {C.DIM}run {i + 1}/{args.runs}{C.RESET}", end="  ", flush=True)
            r = benchmark_rapid_mlx(
                url, model, args.max_tokens, warmup=(i == 0 and not args.no_warmup)
            )
            runs.append(r)
            print(
                f"ttft={C.BLUE}{r.ttft_ms}{C.RESET}ms  "
                f"tok/s={C.BLUE}{r.decode_tok_s}{C.RESET}  "
                f"tokens={r.completion_tokens}  "
                f"rss={C.DIM}{r.memory_peak_mb:.0f}MB{C.RESET}"
            )
        rapid_result = BenchmarkResult(
            ttft_ms=round(sum(r.ttft_ms for r in runs) / len(runs), 1),
            decode_tok_s=round(sum(r.decode_tok_s for r in runs) / len(runs), 1),
            memory_gen_mb=round(sum(r.memory_gen_mb for r in runs) / len(runs), 1),
            memory_peak_mb=round(sum(r.memory_peak_mb for r in runs) / len(runs), 1),
            completion_tokens=runs[0].completion_tokens,
        )
        print(
            f"  {C.DIM}avg  ttft={rapid_result.ttft_ms}ms  tok/s={rapid_result.decode_tok_s}{C.RESET}\n"
        )
    except Exception as e:
        print(f"  {C.RED}Error: {e}{C.RESET}\n")
    finally:
        if rapid_proc:
            rapid_proc.terminate()
            rapid_proc.wait(timeout=10)

    # ── Ollama ────────────────────────────────────────────────────────────────
    print(f"{C.BOLD}{C.GREEN}▶ Benchmarking Ollama...{C.RESET}")
    ollama_proc, ollama_result = None, None
    try:
        port = find_free_port()
        ollama_proc = start_ollama(port)
        url = f"http://127.0.0.1:{port}"
        runs = []
        for i in range(args.runs):
            print(f"  {C.DIM}run {i + 1}/{args.runs}{C.RESET}", end="  ", flush=True)
            r = benchmark_ollama(
                url,
                ollama_model,
                args.max_tokens,
                warmup=(i == 0 and not args.no_warmup),
                debug=(i == 0 and args.debug),
            )
            runs.append(r)
            print(
                f"ttft={C.GREEN}{r.ttft_ms}{C.RESET}ms  "
                f"tok/s={C.GREEN}{r.decode_tok_s}{C.RESET}  "
                f"tokens={r.completion_tokens}  "
                f"rss={C.DIM}{r.memory_peak_mb:.0f}MB{C.RESET}"
            )
        ollama_result = BenchmarkResult(
            ttft_ms=round(sum(r.ttft_ms for r in runs) / len(runs), 1),
            decode_tok_s=round(sum(r.decode_tok_s for r in runs) / len(runs), 1),
            memory_gen_mb=round(sum(r.memory_gen_mb for r in runs) / len(runs), 1),
            memory_peak_mb=round(sum(r.memory_peak_mb for r in runs) / len(runs), 1),
            completion_tokens=runs[0].completion_tokens,
        )
        print(
            f"  {C.DIM}avg  ttft={ollama_result.ttft_ms}ms  tok/s={ollama_result.decode_tok_s}{C.RESET}\n"
        )
    except Exception as e:
        print(f"  {C.RED}Error: {e}{C.RESET}\n")
    finally:
        if ollama_proc:
            ollama_proc.terminate()
            ollama_proc.wait(timeout=10)

    # ── Results ───────────────────────────────────────────────────────────────
    render_results(
        ComparisonResult(
            model=model,
            rapid=rapid_result,
            ollama=ollama_result,
            runs=args.runs,
            max_tokens=args.max_tokens,
        )
    )

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.output:

        def r_dict(r: BenchmarkResult | None) -> dict:
            if not r:
                return {
                    k: None
                    for k in (
                        "ttft_ms",
                        "decode_tok_s",
                        "memory_gen_mb",
                        "memory_peak_mb",
                        "completion_tokens",
                    )
                }
            return {
                "ttft_ms": r.ttft_ms,
                "decode_tok_s": r.decode_tok_s,
                "memory_gen_mb": r.memory_gen_mb,
                "memory_peak_mb": r.memory_peak_mb,
                "completion_tokens": r.completion_tokens,
            }

        data = {
            "model": model,
            "ollama_model": ollama_model,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "rapid-mlx": r_dict(rapid_result),
            "ollama": r_dict(ollama_result),
        }
        args.output.write_text(json.dumps(data, indent=2))
        print(f"{C.DIM}JSON written to: {args.output}{C.RESET}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
