from unittest.mock import MagicMock

import pytest

from scripts import bench_service_prefill as bench
from scripts.bench_service_prefill import percentile, summarize, token_count


def test_token_count_handles_batch_encoding_shape():
    assert token_count({"input_ids": [1, 2, 3]}) == 3
    assert token_count({"input_ids": [[1, 2, 3]]}) == 3


def test_percentile_interpolates_small_samples():
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([10.0, 20.0], 0.95) == 19.5


def test_summary_reports_ttft_and_total_p50_p95():
    rows = [
        {"ttft_ms": 10, "total_ms": 20},
        {"ttft_ms": 30, "total_ms": 40},
        {"ttft_ms": 20, "total_ms": 30},
    ]
    assert summarize(rows) == {
        "ttft_p50_ms": 20.0,
        "ttft_p95_ms": 29.0,
        "total_p50_ms": 30.0,
        "total_p95_ms": 39.0,
    }


def test_wait_for_running_request_polls_server_state(monkeypatch):
    statuses = iter([{"num_running": 0}, {"num_running": 1}])
    monkeypatch.setattr(bench, "get_status", lambda _client, _url: next(statuses))
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)

    observed = bench.wait_for_running_request(
        MagicMock(), "http://rapid", timeout_seconds=1
    )

    assert observed["num_running"] == 1


def test_stream_request_rejects_stream_without_visible_delta(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":4}}',
        "data: [DONE]",
    ]
    client = MagicMock()
    client.stream.return_value = response
    monkeypatch.setattr(bench.time, "perf_counter", MagicMock(side_effect=[1.0, 2.0]))

    with pytest.raises(RuntimeError, match="without a visible"):
        bench.stream_request(client, "http://rapid/v1", "model", [], 1)


def test_stream_request_rejects_missing_server_usage(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_lines.return_value = [
        'data: {"choices":[{"delta":{"content":"done"}}]}',
        "data: [DONE]",
    ]
    client = MagicMock()
    client.stream.return_value = response
    monkeypatch.setattr(
        bench.time, "perf_counter", MagicMock(side_effect=[1.0, 1.5, 2.0])
    )

    with pytest.raises(RuntimeError, match="server usage metadata"):
        bench.stream_request(client, "http://rapid/v1", "model", [], 1)
