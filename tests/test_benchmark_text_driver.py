"""Unit tests for the text benchmark driver in ``rapid_mlx.benchmark``.

``benchmark_single_prompt`` and ``run_benchmark`` normally need a real model.
These tests stub the model-facing seams (``mlx_lm.stream_generate``, model
loading, hardware detection, resource monitoring) so the timing/aggregation
logic runs on every lane, including no-MLX Linux CI.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx import benchmark


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _install_fake_mlx_lm(monkeypatch: pytest.MonkeyPatch, chunks: int) -> list:
    calls: list = []

    def stream_generate(model, tokenizer, prompt, *, max_tokens, sampler):
        calls.append((prompt, max_tokens, sampler))
        yield from range(chunks)

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.stream_generate = stream_generate
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda temp: ("sampler", temp)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return calls


def test_benchmark_single_prompt_counts_streamed_tokens(monkeypatch):
    calls = _install_fake_mlx_lm(monkeypatch, chunks=5)

    result = benchmark.benchmark_single_prompt(
        object(), _Tokenizer(), "one two three", max_tokens=9, temperature=0.3
    )

    assert result is not None
    assert result.prompt == "one two three"
    assert result.prompt_tokens == 3
    assert result.generated_tokens == 5
    assert 0 <= result.ttft <= result.total_time
    assert calls == [("one two three", 9, ("sampler", 0.3))]


def test_benchmark_single_prompt_with_no_output_uses_total_time_as_ttft(
    monkeypatch,
):
    _install_fake_mlx_lm(monkeypatch, chunks=0)

    result = benchmark.benchmark_single_prompt(object(), _Tokenizer(), "hi")

    assert result is not None
    assert result.generated_tokens == 0
    assert result.ttft == result.total_time


def test_run_benchmark_runs_each_warmup_then_measures_every_prompt(monkeypatch):
    hardware = ModuleType("rapid_mlx.optimizations")
    hardware.detect_hardware = lambda: SimpleNamespace(
        chip_name="Test Chip",
        total_memory_gb=16,
        memory_bandwidth_gbs=100,
        gpu_cores=8,
    )
    tokenizer_mod = ModuleType("rapid_mlx.utils.tokenizer")
    tokenizer_mod.load_model_with_fallback = lambda name: (object(), _Tokenizer())
    monkeypatch.setitem(sys.modules, "rapid_mlx.optimizations", hardware)
    monkeypatch.setitem(sys.modules, "rapid_mlx.utils.tokenizer", tokenizer_mod)

    class _Monitor:
        def start(self):
            pass

        def sample(self):
            pass

        def get_summary(self):
            return {}

    monkeypatch.setattr(benchmark, "ResourceMonitor", _Monitor)
    monkeypatch.setattr(benchmark, "get_mlx_memory_info", lambda reset_peak: {})

    calls: list[tuple[str, int]] = []

    def fake_single(model, tokenizer, prompt, max_tokens=256, temperature=0.7):
        calls.append((prompt, max_tokens))
        return benchmark.BenchmarkResult(
            prompt=prompt[:50],
            prompt_tokens=4,
            generated_tokens=8,
            ttft=0.01,
            total_time=0.1,
        )

    monkeypatch.setattr(benchmark, "benchmark_single_prompt", fake_single)

    summary = benchmark.run_benchmark(
        "fake-model", num_prompts=2, max_tokens=8, warmup_runs=3
    )

    warmups = [c for c in calls if c == ("Hello, how are you?", 20)]
    assert len(warmups) == 3
    assert len(calls) == 3 + 2
    assert summary is not None
    assert summary.model_name == "fake-model"
    assert summary.num_runs == 2
    assert summary.total_generated_tokens == 16
