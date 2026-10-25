"""The same-inventory benchmark must not count incomplete streams as wins."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_prefix_selection",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_prefix_selection.py",
)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


@pytest.mark.parametrize(
    "fault", [None, "no_done", "no_usage", "truncated", "role_only", "error"]
)
def test_measure_requires_complete_content_and_usage(monkeypatch, fault):
    events = [{"choices": [{"delta": {"role": "assistant"}}]}]
    if fault != "role_only":
        events.append({"choices": [{"delta": {"content": "ok"}}]})
    events.append(
        {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "length" if fault == "truncated" else "stop",
                }
            ]
        }
    )
    if fault != "no_usage":
        events.append({"usage": {"prompt_tokens": 100, "completion_tokens": 1}})
    if fault == "error":
        events.append({"error": "generation failed"})
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if fault != "no_done":
        wire += "data: [DONE]\n\n"
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=wire))
    )
    monkeypatch.setattr(bench.httpx, "Client", lambda **kwargs: client)
    if fault:
        with pytest.raises(RuntimeError):
            bench.measure("http://test", {})
    else:
        row = bench.measure("http://test", {})
        assert row["text"] == "ok"
        assert 0 <= row["ttft_s"] <= row["elapsed_s"]
        assert row["usage"]["completion_tokens"] == 1
