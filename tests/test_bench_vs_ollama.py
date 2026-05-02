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
