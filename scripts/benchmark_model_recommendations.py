#!/usr/bin/env python3
"""Measure the memory and interactive speed used by model recommendations.

Runs one cached model at a time through the real ``rapid-mlx serve`` path.
The output is append-friendly JSON; maintainers commit reviewed rows to
``docs/benchmarks/model-recommendation-measurements.json``.

macOS only: the harness uses ``footprint`` so Metal/unified memory is counted.
It never downloads weights unless ``--allow-download`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

GIB = 1 << 30
DEFAULT_MODELS = [
    "lfm2.5-1b-4bit",
    "lfm2.5-2.6b-4bit",
    "lfm2.5-8b-a1b-4bit",
    "qwen3.5-4b-4bit",
    "qwen3.5-9b-4bit",
    "gemma-4-12b-4bit",
    "bonsai-27b-2bit",
    "gemma-4-26b-4bit",
    "qwen3.5-35b-4bit",
    "qwen3.6-27b-4bit",
    "qwen3.6-35b-4bit",
]
MODEL_SERVE_ARGS = {
    "gemma-4-26b-4bit": [
        "--no-mllm",
        "--kv-cache-dtype",
        "bf16",
        "--cache-memory-mb",
        "512",
    ]
}


def command(*args: str, timeout: float = 15) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=timeout
    ).stdout.strip()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def json_request(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def footprint(pid: int) -> tuple[float, float]:
    text = command("footprint", "-p", str(pid), timeout=30)
    current = re.search(r"Footprint:\s+([0-9.]+)\s+(MB|GB)", text)
    peak = re.search(r"phys_footprint_peak:\s+([0-9.]+)\s+(MB|GB)", text)
    if not current or not peak:
        raise RuntimeError("could not parse macOS footprint output")

    def gb(match: re.Match[str]) -> float:
        value = float(match.group(1))
        return value if match.group(2) == "GB" else value / 1024

    return round(gb(current), 3), round(gb(peak), 3)


def swap_used_mb() -> float:
    text = command("sysctl", "-n", "vm.swapusage")
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if not match:
        return 0.0
    value = float(match.group(1))
    return value * 1024 if match.group(2) == "G" else value


def physical_ram_gb() -> int:
    return round(int(command("sysctl", "-n", "hw.memsize")) / GIB)


def huggingface_cache_dir(override: str | None = None) -> Path:
    """Return the concrete HF Hub cache directory.

    ``HF_HUB_CACHE`` is the direct cache path; ``HF_HOME`` contains a
    ``hub`` child.  Keeping both forms straight matters when the cache is a
    mounted NAS/Jetson volume rather than the default local disk.
    """
    if override:
        return Path(override).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def cached_repo_for(alias: str, cache_dir: Path | None = None) -> bool:
    # Resolve through Rapid-MLX so aliases and HF cache directory names cannot drift.
    from vllm_mlx.model_aliases import resolve_model

    repo = resolve_model(alias)
    root = cache_dir or huggingface_cache_dir()
    return (root / ("models--" + repo.replace("/", "--"))).is_dir()


def wait_ready(proc: subprocess.Popen[str], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with status {proc.returncode}")
        try:
            json_request(url, timeout=2)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(1)
    raise TimeoutError(f"server was not ready after {timeout:.0f}s")


def run_prompt(port: int, alias: str, content: str, max_tokens: int) -> dict:
    started = time.monotonic()
    response = json_request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {
            "model": alias,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    )
    elapsed = time.monotonic() - started
    usage = response.get("usage") or {}
    status = json_request(f"http://127.0.0.1:{port}/v1/status")
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "wall_decode_tps": round(completion / elapsed, 2) if elapsed else 0.0,
        "server_prompt_tps": float(status.get("prompt_tps") or 0),
        "server_generation_tps": float(status.get("generation_tps") or 0),
        "metal": status.get("metal") or {},
    }


def stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def measure(alias: str, args: argparse.Namespace, environment: dict) -> dict:
    cache_dir = huggingface_cache_dir(args.hf_cache)
    if not args.allow_download and not cached_repo_for(alias, cache_dir):
        return {"model": alias, "status": "skipped_not_cached"}

    log_path = Path(args.log_dir) / f"{alias}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    swap_before = swap_used_mb()
    started = time.monotonic()
    with log_path.open("w") as log:
        process_environment = os.environ.copy()
        process_environment["HF_HUB_CACHE"] = str(cache_dir)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-u",
                "-s",
                "-m",
                "vllm_mlx.cli",
                "serve",
                alias,
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                # A benchmark rerun must measure prefill, not a persisted
                # prefix-cache hit from an earlier run of the same prompt.
                "--disable-prefix-cache",
                *MODEL_SERVE_ARGS.get(alias, []),
                *args.serve_arg,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_environment,
        )
        try:
            wait_ready(proc, args.port, args.load_timeout)
            load_s = time.monotonic() - started
            idle_gb, load_peak_gb = footprint(proc.pid)
            if load_peak_gb > environment["physical_ram_gb"] * args.abort_ram_fraction:
                raise RuntimeError(
                    f"load peak {load_peak_gb} GB exceeded safety limit "
                    f"{environment['physical_ram_gb'] * args.abort_ram_fraction:.1f} GB"
                )

            short = run_prompt(
                args.port,
                alias,
                "Explain in one concise paragraph why local inference is useful.",
                args.output_tokens,
            )
            short_gb, short_peak_gb = footprint(proc.pid)
            swap_after_short = max(0.0, swap_used_mb() - swap_before)
            if swap_after_short > args.abort_swap_mb:
                return {
                    "model": alias,
                    "status": "aborted_swap",
                    "load_s": round(load_s, 2),
                    "idle_footprint_gb": idle_gb,
                    "load_peak_gb": load_peak_gb,
                    "short_footprint_gb": short_gb,
                    "short_peak_gb": short_peak_gb,
                    "short": short,
                    "swap_delta_mb": round(swap_after_short, 1),
                    "reason": f"new swap exceeded {args.abort_swap_mb:.0f} MB safety limit",
                }

            # Repeated neutral text gives every tokenizer a comparable ~8K context.
            long_text = (
                "Local models keep data private and available offline. " * 900
            ).strip()
            long = run_prompt(args.port, alias, long_text, args.output_tokens)
            long_gb, long_peak_gb = footprint(proc.pid)
            swap_after_long = max(0.0, swap_used_mb() - swap_before)

            row = {
                "model": alias,
                "status": (
                    "aborted_swap" if swap_after_long > args.abort_swap_mb else "ok"
                ),
                "load_s": round(load_s, 2),
                "idle_footprint_gb": idle_gb,
                "load_peak_gb": load_peak_gb,
                "short_footprint_gb": short_gb,
                "short_peak_gb": short_peak_gb,
                "long_footprint_gb": long_gb,
                "long_peak_gb": long_peak_gb,
                "short": short,
                "long": long,
                "swap_delta_mb": round(swap_after_long, 1),
            }
            if swap_after_long > args.abort_swap_mb:
                row["reason"] = (
                    f"new swap exceeded {args.abort_swap_mb:.0f} MB safety limit"
                )
            return row
        except Exception as exc:
            return {
                "model": alias,
                "status": "error",
                "error": str(exc),
                "log": str(log_path),
            }
        finally:
            stop(proc)
            time.sleep(args.cooldown)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--load-timeout", type=float, default=300)
    parser.add_argument("--cooldown", type=float, default=10)
    parser.add_argument("--abort-ram-fraction", type=float, default=0.80)
    parser.add_argument("--abort-swap-mb", type=float, default=256)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--hf-cache",
        help="HF Hub cache directory (for example a mounted Jetson/NAS cache)",
    )
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="extra serve argv; repeat and use --serve-arg=--flag for flags",
    )
    parser.add_argument("--log-dir", default="/tmp/rapid-model-recommendations")
    args = parser.parse_args()

    if platform.system() != "Darwin":
        parser.error("this benchmark requires macOS footprint")

    environment = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "chip_model": command("sysctl", "-n", "hw.model"),
        "physical_ram_gb": physical_ram_gb(),
        "macos": platform.mac_ver()[0],
        "rapid_mlx": package_version("rapid-mlx"),
        "mlx": package_version("mlx"),
        "mlx_lm": package_version("mlx-lm"),
        "git_sha": command("git", "rev-parse", "HEAD"),
        "python": platform.python_version(),
    }
    result = {
        "schema_version": 1,
        "environment": environment,
        "method": {
            "serve_path": "python -m vllm_mlx.cli serve --disable-prefix-cache",
            "prefix_cache": "disabled to prevent cross-run contamination",
            "output_tokens": args.output_tokens,
            "long_prompt": "~8K tokens (900 repeated neutral sentences)",
            "abort_ram_fraction": args.abort_ram_fraction,
            "abort_swap_mb": args.abort_swap_mb,
            "extra_serve_args": args.serve_arg,
            "per_model_serve_args": MODEL_SERVE_ARGS,
            "hf_cache": str(huggingface_cache_dir(args.hf_cache)),
            "models_are_sequential": True,
        },
        "measurements": [],
    }
    output = Path(args.output)
    for alias in args.models:
        print(f"Measuring {alias}...", flush=True)
        row = measure(alias, args, environment)
        result["measurements"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)
    return 1 if any(r["status"] == "error" for r in result["measurements"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
