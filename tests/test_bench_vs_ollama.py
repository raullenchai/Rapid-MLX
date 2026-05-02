# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/bench_vs_ollama.py."""

from __future__ import annotations

import importlib.util
import subprocess
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


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--runs", "0"], "--runs must be >= 1"),
        (["--warmups", "-1"], "--warmups must be >= 0"),
        (["--max-tokens", "0"], "--max-tokens must be >= 1"),
        (["--concurrency", "0"], "--concurrency values must be >= 1"),
        (["--startup-timeout", "0"], "--startup-timeout must be > 0"),
        (["--request-timeout", "0"], "--request-timeout must be > 0"),
        (["--startup-timeout", "inf"], "--startup-timeout must be > 0"),
        (["--startup-timeout", "nan"], "--startup-timeout must be > 0"),
        (["--request-timeout", "inf"], "--request-timeout must be > 0"),
        (["--request-timeout", "nan"], "--request-timeout must be > 0"),
    ],
)
def test_parse_args_rejects_invalid_numeric_settings(argv, message, capsys):
    bench = load_bench_module()

    with pytest.raises(SystemExit) as exc_info:
        bench.parse_args(argv)

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


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


def test_render_markdown_surfaces_engine_errors():
    bench = load_bench_module()
    result = {
        "metadata": {"timestamp": "2026-05-02T12:00:00", "git_commit": "abc123"},
        "config": {"runs": 1, "concurrency": [1]},
        "model_pairs": [
            {
                "rapid_mlx_model": "qwen3.5-9b",
                "ollama_model": "qwen3.5:9b",
                "rapid-mlx": {"error": "boom"},
                "ollama": {"summary": {"stream": {"decode_tok_s": 40.0}}},
            }
        ],
    }

    markdown = bench.render_markdown(result)

    assert "**Rapid-MLX error:** boom" in markdown


def test_render_markdown_surfaces_workload_errors():
    bench = load_bench_module()
    result = {
        "metadata": {"timestamp": "2026-05-02T12:00:00", "git_commit": "abc123"},
        "config": {"runs": 1, "concurrency": [1]},
        "model_pairs": [
            {
                "rapid_mlx_model": "qwen3.5-9b",
                "ollama_model": "qwen3.5:9b",
                "rapid-mlx": {
                    "errors": [
                        {
                            "workload": "multi_turn",
                            "error": "request timed out",
                        }
                    ],
                    "summary": {"stream": {"decode_tok_s": 40.0}},
                },
                "ollama": {
                    "errors": [
                        {
                            "workload": "chat",
                            "concurrency": 2,
                            "run": 1,
                            "error": "connection reset",
                        }
                    ],
                    "summary": {"stream": {"decode_tok_s": 20.0}},
                },
            }
        ],
    }

    markdown = bench.render_markdown(result)

    assert "**Rapid-MLX workload errors:**" in markdown
    assert "- multi_turn: request timed out" in markdown
    assert "**Ollama workload errors:**" in markdown
    assert "- chat concurrency=2 run=1: connection reset" in markdown


def test_render_markdown_tolerates_none_engine_payloads():
    bench = load_bench_module()
    result = {
        "metadata": {"timestamp": "2026-05-02T12:00:00", "git_commit": "abc123"},
        "config": {"runs": 1, "concurrency": [1]},
        "model_pairs": [
            {
                "rapid_mlx_model": "qwen3.5-9b",
                "ollama_model": "qwen3.5:9b",
                "rapid-mlx": None,
                "ollama": {"error": "ollama down"},
            }
        ],
    }

    markdown = bench.render_markdown(result)

    assert "**Ollama error:** ollama down" in markdown
    assert "| TTFT | - | - | - |" in markdown


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


def test_build_engine_success_result_shape():
    bench = load_bench_module()

    result = bench.build_engine_success_result(
        engine="rapid-mlx",
        model="qwen3.5-9b",
        port=9123,
        command=["rapid-mlx", "serve", "qwen3.5-9b"],
        raw_runs={"stream": [{"ttft_ms": 100.0}]},
        summary={"stream": {"ttft_ms": 100.0}},
        errors=[],
        server_url="http://127.0.0.1:9123",
        prepared=True,
    )

    assert result["engine"] == "rapid-mlx"
    assert result["model"] == "qwen3.5-9b"
    assert result["port"] == 9123
    assert result["command"] == ["rapid-mlx", "serve", "qwen3.5-9b"]
    assert result["server"]["url"] == "http://127.0.0.1:9123"
    assert result["runtime"]["prepared"] is True
    assert result["metadata"]["engine"] == "rapid-mlx"
    assert result["metadata"]["model"] == "qwen3.5-9b"
    assert result["raw_runs"]["stream"] == [{"ttft_ms": 100.0}]
    assert result["summary"]["stream"] == {"ttft_ms": 100.0}
    assert result["errors"] == []
    assert "started_at" in result
    assert "finished_at" in result
    assert "error" not in result


def test_build_engine_failure_result_shape():
    bench = load_bench_module()

    result = bench.build_engine_failure_result(
        engine="ollama",
        model="qwen3.5:9b",
        port=9124,
        command=["ollama", "serve"],
        error=RuntimeError("startup failed"),
        server_url="http://127.0.0.1:9124",
        prepared=False,
    )

    assert result["engine"] == "ollama"
    assert result["model"] == "qwen3.5:9b"
    assert result["port"] == 9124
    assert result["command"] == ["ollama", "serve"]
    assert result["error"] == "startup failed"
    assert result["errors"] == [{"stage": "engine", "error": "startup failed"}]
    assert result["server"]["url"] == "http://127.0.0.1:9124"
    assert result["runtime"]["prepared"] is False
    assert result["summary"] == {}


def test_run_engine_suite_records_workload_errors_and_continues(monkeypatch):
    bench = load_bench_module()
    calls = []

    def fake_chat(engine, base_url, model, messages, max_tokens, timeout, headers=None):
        calls.append(("chat", model, len(messages), max_tokens, timeout))
        chat_calls = [call for call in calls if call[0] == "chat"]
        if len(chat_calls) == 2:
            raise RuntimeError("chat failed")
        return {
            "ttft_ms": 10.0,
            "decode_tok_s": 20.0,
            "completion_tokens": 2,
            "total_ms": 50.0,
        }

    def fake_embed(engine, base_url, model, inputs, timeout, headers=None):
        calls.append(("embedding", model, len(inputs), timeout, headers))
        return {"latency_ms": 5.0, "embeddings": len(inputs)}

    monkeypatch.setattr(bench, "run_stream_once", fake_chat)
    monkeypatch.setattr(bench, "run_embedding_once", fake_embed)
    monkeypatch.setattr(
        bench,
        "run_multi_turn",
        lambda *args, **kwargs: {"avg_turn_ms": 10.0, "turn_latencies_ms": [10.0]},
    )

    raw_runs, summary, errors = bench.run_engine_suite(
        "rapid-mlx",
        "http://server",
        {
            "chat_model": "chat-model",
            "embedding_model": "embed-model",
            "chat_messages": [{"role": "user", "content": "hi"}],
            "embedding_input": ["hello"],
            "max_tokens": 16,
        },
        concurrency_levels=[1, 2],
        runs_per_level=1,
        timeout=30.0,
        headers={"Authorization": "Bearer test"},
    )

    assert len(calls) == 7
    assert raw_runs["stream"][0]["completion_tokens"] == 2
    assert raw_runs["chat"]["1"] == []
    assert raw_runs["chat"]["2"][0]["runs"][0]["completion_tokens"] == 2
    assert raw_runs["embeddings"]["1"][0]["embeddings"] == 1
    assert summary["concurrency"]["2"]["requests"] == 2.0
    assert summary["embeddings"]["1"]["avg_latency_ms"] == 5.0
    assert errors == [
        {
            "workload": "chat",
            "concurrency": 1,
            "run": 1,
            "error": "chat failed",
        }
    ]


def test_run_engine_suite_populates_multi_turn_summary(monkeypatch):
    bench = load_bench_module()

    monkeypatch.setattr(
        bench,
        "run_stream_once",
        lambda *args, **kwargs: {
            "ttft_ms": 10.0,
            "decode_tok_s": 20.0,
            "completion_tokens": 2,
            "total_ms": 50.0,
        },
    )
    monkeypatch.setattr(
        bench,
        "run_embedding_once",
        lambda *args, **kwargs: {"latency_ms": 5.0, "embeddings": 1},
    )
    monkeypatch.setattr(
        bench,
        "run_multi_turn",
        lambda *args, **kwargs: {
            "avg_turn_ms": 123.4,
            "turn_latencies_ms": [100.0, 146.8],
        },
    )

    raw_runs, summary, errors = bench.run_engine_suite(
        "rapid-mlx",
        "http://server",
        {
            "chat_model": "chat-model",
            "embedding_model": "embed-model",
            "chat_messages": [{"role": "user", "content": "hi"}],
            "embedding_input": ["hello"],
            "max_tokens": 16,
        },
        concurrency_levels=[1],
        runs_per_level=1,
        timeout=30.0,
    )

    assert raw_runs["multi_turn"]["avg_turn_ms"] == 123.4
    assert summary["multi_turn"]["avg_turn_ms"] == 123.4
    assert errors == []


def test_run_engine_suite_stream_summary_does_not_require_concurrency_one(
    monkeypatch,
):
    bench = load_bench_module()

    monkeypatch.setattr(
        bench,
        "run_stream_once",
        lambda *args, **kwargs: {
            "ttft_ms": 11.0,
            "decode_tok_s": 22.0,
            "completion_tokens": 3,
            "total_ms": 55.0,
        },
    )
    monkeypatch.setattr(
        bench,
        "run_embedding_once",
        lambda *args, **kwargs: {"latency_ms": 5.0, "embeddings": 1},
    )
    monkeypatch.setattr(
        bench,
        "run_multi_turn",
        lambda *args, **kwargs: {"avg_turn_ms": 20.0, "turn_latencies_ms": [20.0]},
    )

    raw_runs, summary, errors = bench.run_engine_suite(
        "rapid-mlx",
        "http://server",
        {
            "chat_model": "chat-model",
            "embedding_model": "embed-model",
            "chat_messages": [{"role": "user", "content": "hi"}],
            "embedding_input": ["hello"],
            "max_tokens": 16,
        },
        concurrency_levels=[2],
        runs_per_level=1,
        timeout=30.0,
    )

    assert raw_runs["stream"] == [
        {
            "ttft_ms": 11.0,
            "decode_tok_s": 22.0,
            "completion_tokens": 3,
            "total_ms": 55.0,
        }
    ]
    assert summary["stream"]["ttft_ms"] == 11.0
    assert summary["stream"]["decode_tok_s"] == 22.0
    assert errors == []


def test_run_engine_suite_passes_headers_to_multi_turn(monkeypatch):
    bench = load_bench_module()
    seen_headers = []

    monkeypatch.setattr(
        bench,
        "run_stream_once",
        lambda *args, **kwargs: {
            "ttft_ms": 10.0,
            "decode_tok_s": 20.0,
            "completion_tokens": 2,
            "total_ms": 50.0,
        },
    )
    monkeypatch.setattr(
        bench,
        "run_embedding_once",
        lambda *args, **kwargs: {"latency_ms": 5.0, "embeddings": 1},
    )

    def fake_multi_turn(engine, base_url, model, max_tokens, timeout, headers=None):
        seen_headers.append(headers)
        return {"avg_turn_ms": 20.0, "turn_latencies_ms": [20.0]}

    monkeypatch.setattr(bench, "run_multi_turn", fake_multi_turn)

    bench.run_engine_suite(
        "rapid-mlx",
        "http://server",
        {
            "chat_model": "chat-model",
            "embedding_model": "embed-model",
            "chat_messages": [{"role": "user", "content": "hi"}],
            "embedding_input": ["hello"],
            "max_tokens": 16,
        },
        concurrency_levels=[1],
        runs_per_level=1,
        timeout=30.0,
        headers={"Authorization": "Bearer test"},
    )

    assert seen_headers == [{"Authorization": "Bearer test"}]


def test_run_multi_turn_passes_headers_to_post_json(monkeypatch):
    bench = load_bench_module()
    seen_headers = []

    def fake_post_json(url, payload, timeout, headers=None):
        seen_headers.append(headers)
        return {"choices": [{"message": {"content": "ok"}}]}, 10.0

    monkeypatch.setattr(bench, "post_json", fake_post_json)

    summary = bench.run_multi_turn(
        "rapid-mlx",
        "http://server",
        "chat-model",
        16,
        30.0,
        headers={"Authorization": "Bearer test"},
    )

    assert summary["avg_turn_ms"] == 10.0
    assert seen_headers == [{"Authorization": "Bearer test"}] * 4


def test_run_benchmark_executes_engines_sequentially_and_adds_comparisons(
    monkeypatch, tmp_path
):
    bench = load_bench_module()
    calls = []

    args = bench.CliArgs(
        model_pairs=[bench.ModelPair("rapid-a", "ollama-a")],
        runs=2,
        warmups=0,
        max_tokens=32,
        concurrency=[1, 2],
        output_dir=tmp_path,
        no_pull=True,
        no_download=True,
        startup_timeout=1.0,
        request_timeout=2.0,
        rapid_mlx_args=[],
        ollama_env={},
    )

    monkeypatch.setattr(
        bench,
        "collect_metadata",
        lambda: {"timestamp": "2026-05-02T12:00:00", "git_commit": "abc123"},
    )

    def fake_rapid(pair, call_args):
        calls.append(("rapid", pair, call_args))
        return {
            "engine": "rapid-mlx",
            "model": pair.rapid_mlx,
            "summary": {
                "stream": {"ttft_ms": 100.0, "decode_tok_s": 120.0},
                "concurrency": {
                    "1": {"aggregate_tok_s": 100.0},
                    "2": {"aggregate_tok_s": 180.0},
                },
                "embeddings": {"1": {"avg_latency_ms": 10.0}},
            },
            "raw_runs": {},
            "errors": [],
        }

    def fake_ollama(pair, call_args):
        calls.append(("ollama", pair, call_args))
        return {
            "engine": "ollama",
            "model": pair.ollama,
            "summary": {
                "stream": {"ttft_ms": 250.0, "decode_tok_s": 40.0},
                "concurrency": {"1": {"aggregate_tok_s": 50.0}},
                "embeddings": {"1": {"avg_latency_ms": 20.0}},
            },
            "raw_runs": {},
            "errors": [],
        }

    monkeypatch.setattr(bench, "benchmark_rapid_mlx", fake_rapid)
    monkeypatch.setattr(bench, "benchmark_ollama", fake_ollama)

    result = bench.run_benchmark(args)

    assert calls == [
        ("rapid", args.model_pairs[0], args),
        ("ollama", args.model_pairs[0], args),
    ]
    pair_result = result["model_pairs"][0]
    assert pair_result["rapid-mlx"]["model"] == "rapid-a"
    assert pair_result["ollama"]["model"] == "ollama-a"
    assert pair_result["comparisons"]["stream_decode_tok_s_speedup"] == 3.0
    assert pair_result["comparisons"]["stream_ttft_latency_speedup"] == 2.5
    assert pair_result["comparisons"]["concurrency"]["1"]["aggregate_tok_s_speedup"] == 2.0
    assert pair_result["comparisons"]["concurrency"]["2"]["aggregate_tok_s_speedup"] is None
    assert pair_result["comparisons"]["embeddings"]["1"]["avg_latency_speedup"] == 2.0


def test_benchmark_ollama_pulls_after_managed_server_start_with_managed_env(
    monkeypatch, tmp_path
):
    bench = load_bench_module()
    calls = []

    args = bench.CliArgs(
        model_pairs=[],
        runs=1,
        warmups=0,
        max_tokens=16,
        concurrency=[1],
        output_dir=tmp_path,
        no_pull=False,
        no_download=True,
        startup_timeout=1.0,
        request_timeout=2.0,
        rapid_mlx_args=[],
        ollama_env={"OLLAMA_KEEP_ALIVE": "0"},
    )

    class FakeManagedProcess:
        def stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(bench, "require_executable", lambda name: calls.append(("require", name)))
    monkeypatch.setattr(bench, "find_free_port", lambda: 9124)

    def fake_start_process(command, env=None):
        calls.append(("start", command, env["OLLAMA_HOST"], env["OLLAMA_KEEP_ALIVE"]))
        return FakeManagedProcess()

    def fake_wait_for_url(url, timeout):
        calls.append(("wait", url, timeout))

    def fake_prepare_ollama_model(model, call_args, env):
        calls.append(("pull", model, env["OLLAMA_HOST"], env["OLLAMA_KEEP_ALIVE"]))
        return True

    monkeypatch.setattr(bench, "start_process", fake_start_process)
    monkeypatch.setattr(bench, "wait_for_url", fake_wait_for_url)
    monkeypatch.setattr(bench, "prepare_ollama_model", fake_prepare_ollama_model)
    monkeypatch.setattr(bench, "run_engine_suite", lambda *args, **kwargs: ({}, {}, []))

    result = bench.benchmark_ollama(bench.ModelPair("rapid-a", "ollama-a"), args)

    assert "error" not in result
    assert calls == [
        ("require", "ollama"),
        ("start", ["ollama", "serve"], "127.0.0.1:9124", "0"),
        ("wait", "http://127.0.0.1:9124/api/tags", 1.0),
        ("pull", "ollama-a", "127.0.0.1:9124", "0"),
        ("stop",),
    ]
    assert result["runtime"]["prepared"] is True


def test_cli_help_smoke_lists_core_options():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--model-pair" in result.stdout
    assert "--runs" in result.stdout
    assert "--no-pull" in result.stdout
    assert "--no-download" in result.stdout
