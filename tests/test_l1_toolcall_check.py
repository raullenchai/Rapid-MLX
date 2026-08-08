from __future__ import annotations

import sys

from scripts import l1_toolcall_check as gate


class _Response:
    def __init__(self, payload=None, lines=None):
        self._payload = payload
        self._lines = lines or []

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _forced_payload():
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_probe",
                            "type": "function",
                            "function": {
                                "name": "release_probe",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ]
    }


def test_gate_replays_tool_result_in_both_modes(monkeypatch):
    posts = iter([_Response(_forced_payload()), _Response({"choices": [{}]})])
    streams = iter(
        [
            _Response(
                lines=[
                    'data: {"choices":[{"delta":{"tool_calls":[{}]}}]}',
                    "data: [DONE]",
                ]
            ),
            _Response(lines=["data: {}", "data: [DONE]"]),
        ]
    )
    monkeypatch.setattr(gate.httpx, "post", lambda *args, **kwargs: next(posts))
    monkeypatch.setattr(gate.httpx, "stream", lambda *args, **kwargs: next(streams))
    monkeypatch.setattr(sys, "argv", ["l1_toolcall_check.py"])

    assert gate.main() == 0


def test_gate_fails_closed_when_stream_replay_has_no_done(monkeypatch):
    posts = iter([_Response(_forced_payload()), _Response({"choices": [{}]})])
    streams = iter(
        [
            _Response(
                lines=[
                    'data: {"choices":[{"delta":{"tool_calls":[{}]}}]}',
                    "data: [DONE]",
                ]
            ),
            _Response(lines=["data: {}"]),
        ]
    )
    monkeypatch.setattr(gate.httpx, "post", lambda *args, **kwargs: next(posts))
    monkeypatch.setattr(gate.httpx, "stream", lambda *args, **kwargs: next(streams))
    monkeypatch.setattr(sys, "argv", ["l1_toolcall_check.py"])

    assert gate.main() == 1
