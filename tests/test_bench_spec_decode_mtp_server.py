import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

from bench.bench_spec_decode_mtp_server import (
    TurnResult,
    derived_mtp_metrics,
    metric_delta,
    metric_total,
    parse_mtp_metrics,
    split_metric_observations,
    stream_turn,
    summarize,
    write_receipt,
)


class _FakeResponse:
    def __init__(self, body: bytes):
        self._lines = io.BytesIO(body)

    def __enter__(self):
        return self._lines

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_parse_mtp_metrics_and_delta_preserve_label_series():
    before = parse_mtp_metrics(
        'rapid_mlx_spec_decode_attempts_total{method="mtp",family="qwen"} 2\n'
        'rapid_mlx_spec_decode_k_chosen_total{method="mtp",k="0"} 1\n'
        "unrelated_metric 99\n"
    )
    after = parse_mtp_metrics(
        'rapid_mlx_spec_decode_attempts_total{family="qwen",method="mtp"} 7\n'
        'rapid_mlx_spec_decode_k_chosen_total{method="mtp",k="0"} 3\n'
        'rapid_mlx_spec_decode_k_chosen_total{method="mtp",k="1"} 4\n'
    )

    delta = metric_delta(before, after)

    assert metric_total(delta, "rapid_mlx_spec_decode_attempts_total") == 5
    assert metric_total(delta, "rapid_mlx_spec_decode_k_chosen_total") == 6


def test_metric_observations_do_not_treat_cost_gauge_as_delta():
    before = parse_mtp_metrics(
        'rapid_mlx_spec_decode_attempts_total{method="mtp"} 2\n'
        'rapid_mlx_spec_decode_accept_ratio{method="mtp"} 0.5\n'
        'rapid_mlx_spec_decode_k_cost_ms{method="mtp",k="1"} 10\n'
    )
    after = parse_mtp_metrics(
        'rapid_mlx_spec_decode_attempts_total{method="mtp"} 7\n'
        'rapid_mlx_spec_decode_accept_ratio{method="mtp"} 0.7\n'
        'rapid_mlx_spec_decode_k_cost_ms{method="mtp",k="1"} 12\n'
    )

    cumulative_delta, gauges_after = split_metric_observations(before, after)

    assert metric_total(cumulative_delta, "rapid_mlx_spec_decode_attempts_total") == 5
    assert metric_total(gauges_after, "rapid_mlx_spec_decode_accept_ratio") == 0.7
    assert metric_total(gauges_after, "rapid_mlx_spec_decode_k_cost_ms") == 12


def test_derived_mtp_metrics_use_only_cell_deltas():
    delta = {
        "rapid_mlx_spec_decode_attempts_total{family=qwen,method=mtp}": 10,
        "rapid_mlx_spec_decode_accepts_total{family=qwen,method=mtp}": 7,
        "rapid_mlx_spec_decode_tokens_saved_total{family=qwen,method=mtp}": 7,
        "rapid_mlx_spec_decode_k_chosen_rounds_total{family=qwen,method=mtp}": 5,
        "rapid_mlx_spec_decode_k_chosen_total{family=qwen,k=0,method=mtp}": 1,
        "rapid_mlx_spec_decode_k_chosen_total{family=qwen,k=2,method=mtp}": 4,
    }

    derived = derived_mtp_metrics(delta)

    assert derived["accept_ratio"] == 0.7
    assert derived["k_counts"] == {"0": 1, "2": 4}
    assert derived["k_shares"] == {"0": 0.2, "2": 0.8}


def test_stream_turn_measures_first_content_and_reads_trailing_usage():
    body = b"".join(
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    with patch(
        "bench.bench_spec_decode_mtp_server.urllib.request.urlopen",
        return_value=_FakeResponse(body),
    ):
        text, result = stream_turn(
            "http://server/v1/chat/completions",
            "model",
            [{"role": "user", "content": "hi"}],
            8,
            {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            10,
        )

    assert text == "hello"
    assert result.ttft_s is not None
    assert result.prompt_tokens == 9
    assert result.completion_tokens == 2
    assert result.finish_reason == "stop"
    assert result.response_sha256 == hashlib.sha256(b"hello").hexdigest()


def test_summary_reports_pooled_rate_and_tail_latency():
    results = [
        TurnResult("a", "coding", 0, True, 0.1, 1.0, 10, 5, "stop", "a", None),
        TurnResult("b", "coding", 0, True, 0.3, 3.0, 20, 15, "stop", "b", None),
    ]

    summary = summarize(results, wall_time_s=2.0)

    assert summary["pooled_completion_tokens_per_s"] == 5.0
    assert summary["aggregate_completion_tokens_per_s"] == 10.0
    assert summary["ttft_s"]["median"] == 0.2
    assert summary["latency_s"]["p95"] == 2.9


def test_receipt_write_is_atomic_and_has_checksum(tmp_path: Path):
    output = tmp_path / "receipt.json"

    write_receipt(output, {"answer": 42})

    encoded = output.read_bytes()
    checksum = output.with_suffix(".json.sha256").read_text(encoding="utf-8")
    assert json.loads(encoded) == {"answer": 42}
    assert checksum == f"{hashlib.sha256(encoded).hexdigest()}  receipt.json\n"
    assert not list(tmp_path.glob("tmp*"))
