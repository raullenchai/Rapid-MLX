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


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
