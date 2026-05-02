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


def test_parse_rapid_mlx_stream_ignores_malformed_shapes_and_bad_usage():
    bench = load_bench_module()
    lines = [
        "data: []",
        'data: {"usage":{"completion_tokens":"bad"}}',
        'data: {"choices":[null,{"delta":null},{"delta":[]}]}',
        'data: {"choices":{"delta":{"content":"ignored"}}}',
        'data: {"choices":[{"delta":{"content":123}}]}',
        'data: {"choices":[{"delta":{"content":"hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]

    parsed = bench.parse_rapid_mlx_stream(lines)

    assert parsed.content_chunks == 2
    assert parsed.completion_tokens == 2


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


def test_parse_ollama_stream_ignores_malformed_shapes_and_bad_metadata():
    bench = load_bench_module()
    lines = [
        "[]",
        '{"message":null}',
        '{"message":[]}',
        '{"message":{"content":123}}',
        '{"eval_count":"bad","eval_duration":"bad"}',
        '{"message":{"content":"hello"},"done":false}',
        '{"message":{"content":" world"},"done":false}',
    ]

    parsed = bench.parse_ollama_stream(lines)

    assert parsed.content_chunks == 2
    assert parsed.completion_tokens == 2
    assert parsed.eval_duration_ns is None


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


def test_extract_rapid_mlx_message_content_ignores_malformed_shapes():
    bench = load_bench_module()

    assert bench.extract_rapid_mlx_message_content([]) == ""
    assert bench.extract_rapid_mlx_message_content({"choices": [None]}) == ""
    assert bench.extract_rapid_mlx_message_content({"choices": {"0": {}}}) == ""


def test_extract_rapid_mlx_message_content_returns_assistant_content():
    bench = load_bench_module()

    content = bench.extract_rapid_mlx_message_content(
        {"choices": [{"message": {"content": "hello"}}]}
    )

    assert content == "hello"


def test_extract_ollama_message_content_ignores_malformed_shapes():
    bench = load_bench_module()

    assert bench.extract_ollama_message_content([]) == ""
    assert bench.extract_ollama_message_content({"message": ["bad"]}) == ""


def test_extract_ollama_message_content_returns_assistant_content():
    bench = load_bench_module()

    content = bench.extract_ollama_message_content(
        {"message": {"content": "hello"}}
    )

    assert content == "hello"


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
