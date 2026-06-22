# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``scripts/stress_test.py`` streaming accounting."""

from __future__ import annotations

import json
from unittest.mock import patch

import scripts.stress_test as stress_test


class _FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        events = [
            {"choices": [{"delta": {"content": "large fused diffusion chunk"}}]},
            {"choices": [{"delta": {"content": "second fused chunk"}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 1024,
                    "total_tokens": 1031,
                },
            },
        ]
        for event in events:
            yield "data: " + json.dumps(event)
        yield "data: [DONE]"


def test_stream_chat_prefers_usage_completion_tokens_over_chunk_count():
    captured = {}

    def fake_stream(method, url, json, timeout):  # noqa: A002
        captured.update(json)
        return _FakeStream()

    with patch.object(stress_test.httpx, "stream", side_effect=fake_stream):
        ms, tokens, content = stress_test.chat("long", max_tokens=1024, stream=True)

    assert ms >= 0
    assert tokens == 1024
    assert "large fused diffusion chunk" in content
    assert captured["stream_options"] == {"include_usage": True}


def test_stream_chat_falls_back_to_chunk_count_without_usage():
    class NoUsageStream(_FakeStream):
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"b"}}]}'
            yield "data: [DONE]"

    with patch.object(stress_test.httpx, "stream", return_value=NoUsageStream()):
        _, tokens, content = stress_test.chat("short", stream=True)

    assert tokens == 2
    assert content == "ab"
