"""Validate the measurement harness rejects incomplete streaming responses."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench_multiturn_prefix.py"
_SPEC = importlib.util.spec_from_file_location("bench_multiturn_prefix", _SCRIPT)
assert _SPEC and _SPEC.loader
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def _client(events: list[dict | str]) -> httpx.Client:
    wire = "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n"
        for event in events
    )
    return httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=wire)),
    )


@pytest.mark.parametrize("terminal", [{"type": "message_stop"}, "[DONE]"])
def test_messages_requires_final_usage_and_its_own_terminal_event(terminal):
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 100, "output_tokens": 0}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        },
        {"type": "message_delta", "usage": {}},
        terminal,
    ]
    with _client(events) as client, pytest.raises(RuntimeError):
        bench._stream(client, "/v1/messages", {})


def test_chat_requires_final_token_counts():
    events = [
        {"choices": [{"delta": {"content": "ok"}}]},
        {"choices": [], "usage": {}},
        "[DONE]",
    ]
    with _client(events) as client, pytest.raises(RuntimeError, match="terminal usage"):
        bench._stream(client, "/v1/chat/completions", {})
