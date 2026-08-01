# SPDX-License-Identifier: Apache-2.0
"""Focused non-stream and incremental coverage for Cohere action envelopes."""

from __future__ import annotations

import json

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.cohere_tool_parser import CohereToolParser


def test_action_envelope_extracts_multiple_calls_and_preserves_content() -> None:
    parser = CohereToolParser(None)
    text = (
        "I will inspect that."
        "<|START_ACTION|>"
        '[{"tool_call_id":"north-a","tool_name":"read_file",'
        '"parameters":{"path":"/etc/hostname"}},'
        '{"tool_call_id":"north-b","name":"list_dir",'
        '"arguments":{"path":"/tmp"}}]'
        "<|END_ACTION|>"
        "Done."
    )

    result = parser.extract_tool_calls(text)

    assert result.tools_called is True
    assert result.content == "I will inspect that.Done."
    assert result.tool_calls == [
        {
            "id": "north-a",
            "name": "read_file",
            "arguments": json.dumps({"path": "/etc/hostname"}),
        },
        {
            "id": "north-b",
            "name": "list_dir",
            "arguments": json.dumps({"path": "/tmp"}),
        },
    ]


def test_streaming_waits_for_complete_action_then_emits_native_delta() -> None:
    parser = CohereToolParser(None)
    previous = "Checking."
    action = (
        '<|START_ACTION|>[{"tool_call_id":"north-a",'
        '"tool_name":"read_file","parameters":{"path":"/etc/hostname"}}]'
    )

    assert parser.extract_tool_calls_streaming("", previous, previous) == {
        "content": previous
    }
    assert (
        parser.extract_tool_calls_streaming(previous, previous + action, action) is None
    )
    delta = parser.extract_tool_calls_streaming(
        previous + action,
        previous + action + "<|END_ACTION|>",
        "<|END_ACTION|>",
    )

    assert delta == {
        "tool_calls": [
            {
                "index": 0,
                "id": "north-a",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "/etc/hostname"}),
                },
            }
        ]
    }


def test_incomplete_action_remains_plain_content_until_closed() -> None:
    parser = CohereToolParser(None)
    text = '<|START_ACTION|>{"tool_name":"read_file"}'

    result = parser.extract_tool_calls(text)

    assert result.tools_called is False
    assert result.content == text
    assert parser.has_pending_tool_call(text) is True
    assert parser.flush_held_content(text) == text


def test_marker_stripped_action_payload_uses_checkpoint_compatible_fallback() -> None:
    parser = CohereToolParser(None)

    result = parser.extract_tool_calls(
        '[{"tool_call_id":"north-a","function":"read_file",'
        '"arguments":{"path":"/etc/hostname"}}]'
    )

    assert result.tools_called is True
    assert result.content is None
    assert result.tool_calls == [
        {
            "id": "north-a",
            "name": "read_file",
            "arguments": json.dumps({"path": "/etc/hostname"}),
        }
    ]


def test_cohere_aliases_share_the_same_rapid_native_parser() -> None:
    assert ToolParserManager.get_tool_parser("cohere") is CohereToolParser
    assert ToolParserManager.get_tool_parser("cohere2_moe") is CohereToolParser
