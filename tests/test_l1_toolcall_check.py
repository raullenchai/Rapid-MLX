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
    requests = []
    posts = iter(
        [
            _Response(_forced_payload()),
            _Response({"choices": [{"message": {"content": "ack"}}]}),
        ]
    )
    streams = iter(
        [
            _Response(
                lines=[
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"release_","arguments":"{\\"prompt\\":"}}]}}]}',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"probe","arguments":"\\"inspect\\"}"}}]}}]}',
                    "data: [DONE]",
                ]
            ),
            _Response(
                lines=[
                    'data: {"choices":[{"delta":{"content":"ack"}}]}',
                    "data: [DONE]",
                ]
            ),
        ]
    )

    def fake_post(*args, **kwargs):
        requests.append(("post", kwargs["json"]))
        return next(posts)

    def fake_stream(*args, **kwargs):
        requests.append(("stream", kwargs["json"]))
        return next(streams)

    monkeypatch.setattr(gate.httpx, "post", fake_post)
    monkeypatch.setattr(gate.httpx, "stream", fake_stream)
    monkeypatch.setattr(sys, "argv", ["l1_toolcall_check.py"])

    assert gate.main() == 0
    assert [request[1]["stream"] for request in requests] == [False, True, False, True]
    for _, replay in requests[2:]:
        messages = replay["messages"]
        assistant = messages[1]
        result = messages[2]
        assert (
            assistant["tool_calls"]
            == _forced_payload()["choices"][0]["message"]["tool_calls"]
        )
        assert result == {
            "role": "tool",
            "tool_call_id": "call_probe",
            "name": "release_probe",
            "content": "RELEASE_PROBE_OK",
        }
        assert messages[3] == {
            "role": "user",
            "content": "Acknowledge the completed probe.",
        }


def test_gate_fails_closed_when_stream_replay_has_no_done(monkeypatch):
    posts = iter(
        [
            _Response(_forced_payload()),
            _Response({"choices": [{"message": {"content": "ack"}}]}),
        ]
    )
    streams = iter(
        [
            _Response(
                lines=[
                    'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"release_probe","arguments":"{}"}}]}}]}',
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


def test_gate_fails_closed_when_nonstream_replay_has_no_message(monkeypatch):
    posts = iter([_Response(_forced_payload()), _Response({"choices": [{}]})])
    streams = iter(
        [
            _Response(
                lines=[
                    'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"release_probe","arguments":"{}"}}]}}]}',
                    "data: [DONE]",
                ]
            )
        ]
    )
    monkeypatch.setattr(gate.httpx, "post", lambda *args, **kwargs: next(posts))
    monkeypatch.setattr(gate.httpx, "stream", lambda *args, **kwargs: next(streams))
    monkeypatch.setattr(sys, "argv", ["l1_toolcall_check.py"])

    assert gate.main() == 1
