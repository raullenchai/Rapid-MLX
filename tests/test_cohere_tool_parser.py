# SPDX-License-Identifier: Apache-2.0
"""Focused non-stream and incremental coverage for Cohere action envelopes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import vllm_mlx.service.helpers as helpers
from vllm_mlx.config import get_config
from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.cohere_tool_parser import NorthToolParser


def _request(*names: str, tool_choice=None) -> dict:
    request = {
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in names
        ]
    }
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return request


def test_action_envelope_extracts_multiple_calls_and_preserves_content() -> None:
    parser = NorthToolParser(None)
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
    parser = NorthToolParser(None)
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
        ],
        "preserve_post_tool_content": True,
    }


def test_incomplete_action_remains_plain_content_until_closed() -> None:
    parser = NorthToolParser(None)
    text = '<|START_ACTION|>{"tool_name":"read_file"}'

    result = parser.extract_tool_calls(text)

    assert result.tools_called is False
    assert result.content == text
    assert result.rejection_authoritative is True
    assert parser.has_pending_tool_call(text) is True
    assert parser.flush_held_content(text) == text


def test_unframed_action_payload_stays_content_without_wire_provenance() -> None:
    parser = NorthToolParser(None)
    wire = (
        '[{"tool_call_id":"north-a","function":"read_file",'
        '"arguments":{"path":"/etc/hostname"}}]'
    )

    result = parser.extract_tool_calls(wire, _request("read_file"))

    assert result.tools_called is False
    assert result.content == wire
    assert result.tool_calls == []
    assert result.rejection_authoritative is True


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "required",
        {"type": "function", "function": {"name": "delete_file"}},
    ],
)
def test_nonstream_service_never_repromotes_unframed_sensitive_call(
    tool_choice,
) -> None:
    cfg = get_config()
    saved = (cfg.enable_auto_tool_choice, cfg.tool_call_parser)
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "north"
    request_dict = _request("delete_file", tool_choice=tool_choice)
    request = SimpleNamespace(model_dump=lambda: request_dict)
    wire = '[{"tool_name":"delete_file","parameters":{"path":"/tmp/example"}}]'
    try:
        content, calls = helpers._run_tool_parser(wire, request)
    finally:
        cfg.enable_auto_tool_choice, cfg.tool_call_parser = saved

    assert calls is None
    assert content == wire


def test_north_specific_aliases_share_the_same_rapid_native_parser() -> None:
    assert ToolParserManager.get_tool_parser("north") is NorthToolParser
    assert ToolParserManager.get_tool_parser("cohere_north") is NorthToolParser
    assert "cohere" not in ToolParserManager.list_registered()


def test_declared_tool_is_promoted() -> None:
    parser = NorthToolParser(None)
    wire = (
        '<|START_ACTION|>{"tool_name":"read_file",'
        '"parameters":{"path":"/tmp/a"}}<|END_ACTION|>'
    )

    result = parser.extract_tool_calls(wire, _request("read_file"))

    assert result.tools_called is True
    assert [call["name"] for call in result.tool_calls] == ["read_file"]


def test_json_encoded_object_arguments_are_normalized() -> None:
    parser = NorthToolParser(None)
    wire = (
        "<|START_ACTION|>"
        + json.dumps(
            {
                "tool_name": "read_file",
                "arguments": json.dumps({"path": "/tmp/a"}),
            }
        )
        + "<|END_ACTION|>"
    )

    result = parser.extract_tool_calls(wire, _request("read_file"))

    assert result.tools_called is True
    assert json.loads(result.tool_calls[0]["arguments"]) == {"path": "/tmp/a"}


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        "required",
        {"type": "function", "function": {"name": "danger"}},
    ],
)
@pytest.mark.parametrize(
    "invalid_arguments",
    [
        "not-json",
        json.dumps("json-string-root"),
        ["list-root"],
        7,
        None,
    ],
)
def test_non_object_arguments_fail_closed_with_streaming_parity(
    tool_choice,
    invalid_arguments,
) -> None:
    request = _request("danger", tool_choice=tool_choice)
    wire = (
        "<|START_ACTION|>"
        + json.dumps({"tool_name": "danger", "arguments": invalid_arguments})
        + "<|END_ACTION|>"
    )

    nonstream = NorthToolParser(None).extract_tool_calls(wire, request)
    streaming = NorthToolParser(None).extract_tool_calls_streaming(
        "", wire, wire, request=request
    )

    assert nonstream.tools_called is False
    assert nonstream.tool_calls == []
    assert nonstream.content == wire
    assert nonstream.rejection_authoritative is True
    assert streaming == {"content": wire}


def test_mixed_valid_and_invalid_calls_fail_entire_envelope_closed() -> None:
    request = _request("read_file", "danger")
    wire = (
        "<|START_ACTION|>"
        + json.dumps(
            [
                {"tool_name": "read_file", "arguments": {"path": "/tmp/a"}},
                {"tool_name": "danger", "arguments": ["not", "an", "object"]},
            ]
        )
        + "<|END_ACTION|>"
    )

    nonstream = NorthToolParser(None).extract_tool_calls(wire, request)
    streaming = NorthToolParser(None).extract_tool_calls_streaming(
        "", wire, wire, request=request
    )

    assert nonstream.tools_called is False
    assert nonstream.content == wire
    assert streaming == {"content": wire}


def test_undeclared_tool_stays_text() -> None:
    parser = NorthToolParser(None)
    wire = (
        '<|START_ACTION|>{"tool_name":"delete_everything",'
        '"parameters":{}}<|END_ACTION|>'
    )

    result = parser.extract_tool_calls(wire, _request("read_file"))

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == wire
    assert result.rejection_authoritative is True


def test_mixed_declared_envelopes_have_streaming_parity() -> None:
    request = _request("read_file")
    undeclared = (
        '<|START_ACTION|>{"tool_name":"delete_everything",'
        '"parameters":{}}<|END_ACTION|>'
    )
    wire = (
        "Before."
        '<|START_ACTION|>{"tool_name":"read_file",'
        '"parameters":{}}<|END_ACTION|>' + undeclared + "After."
    )

    nonstream = NorthToolParser(None).extract_tool_calls(wire, request)
    streaming = NorthToolParser(None).extract_tool_calls_streaming(
        "", wire, wire, request=request
    )

    assert nonstream.tools_called is True
    assert [call["name"] for call in nonstream.tool_calls] == ["read_file"]
    assert nonstream.content == "Before." + undeclared + "After."
    assert nonstream.rejection_authoritative is True
    assert streaming is not None
    assert [call["function"]["name"] for call in streaming.get("tool_calls", [])] == [
        "read_file"
    ]
    assert streaming.get("content") == nonstream.content


def test_nonstream_does_not_synthesize_envelope_across_excision_seam() -> None:
    request = _request("read_file")
    wire = (
        "<|START_ACT"
        "<|START_ACTION|>x<|END_ACTION|>"
        'ION|>{"tool_name":"read_file","parameters":{}}<|END_ACTION|>'
    )

    nonstream = NorthToolParser(None).extract_tool_calls(wire, request)
    streaming = NorthToolParser(None).extract_tool_calls_streaming(
        "", wire, wire, request=request
    )

    assert nonstream.tools_called is False
    assert nonstream.tool_calls == []
    assert nonstream.content == wire
    assert nonstream.rejection_authoritative is True
    assert streaming == {"content": wire}


def test_nonstream_service_does_not_repromote_rejected_north_envelope() -> None:
    cfg = get_config()
    saved = (cfg.enable_auto_tool_choice, cfg.tool_call_parser)
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "north"
    request_dict = _request("read_file")
    request = SimpleNamespace(model_dump=lambda: request_dict)
    wire = '<|START_ACTION|>{"name":"delete_everything","arguments":{}}<|END_ACTION|>'
    try:
        content, calls = helpers._run_tool_parser(wire, request)
    finally:
        cfg.enable_auto_tool_choice, cfg.tool_call_parser = saved

    assert calls is None
    assert content == wire


def test_tools_null_does_not_promote_action() -> None:
    parser = NorthToolParser(None)
    wire = (
        '<|START_ACTION|>{"tool_name":"read_file",'
        '"parameters":{"path":"/tmp/a"}}<|END_ACTION|>'
    )

    result = parser.extract_tool_calls(wire, {"tools": None})

    assert result.tools_called is False
    assert result.content == wire


def test_tool_choice_none_does_not_promote_action() -> None:
    parser = NorthToolParser(None)
    wire = (
        '<|START_ACTION|>{"tool_name":"read_file",'
        '"parameters":{"path":"/tmp/a"}}<|END_ACTION|>'
    )

    result = parser.extract_tool_calls(wire, _request("read_file", tool_choice="none"))

    assert result.tools_called is False
    assert result.content == wire


def test_streaming_undeclared_tool_is_released_as_text() -> None:
    parser = NorthToolParser(None)
    request = _request("read_file")
    prefix = "Checking."
    action = (
        '<|START_ACTION|>{"tool_name":"delete_everything",'
        '"parameters":{}}<|END_ACTION|>'
    )

    assert parser.extract_tool_calls_streaming("", prefix, prefix, request=request) == {
        "content": prefix
    }
    current = prefix + action
    delta = parser.extract_tool_calls_streaming(
        prefix, current, action, request=request
    )

    assert delta == {"content": action}


@pytest.mark.parametrize(
    "value",
    [
        "def f():\n    return 1\n",
        "literal <|END_ACTION|> and </tool_call> markers",
        'quotes "and" backslashes \\ survive',
        r"literal backslash-pipe \\| survives",
    ],
)
def test_north_json_argument_values_round_trip_exactly(value: str) -> None:
    parser = NorthToolParser(None)
    wire = (
        "<|START_ACTION|>"
        + json.dumps([{"tool_name": "note_write", "parameters": {"body": value}}])
        + "<|END_ACTION|>"
    )

    result = parser.extract_tool_calls(wire, _request("note_write"))

    assert result.tools_called is True
    assert json.loads(result.tool_calls[0]["arguments"])["body"] == value


def _stream_characterwise(text: str, request: dict) -> tuple[str, list[dict]]:
    parser = NorthToolParser(None)
    previous = ""
    content: list[str] = []
    calls: list[dict] = []
    for char in text:
        current = previous + char
        delta = parser.extract_tool_calls_streaming(
            previous, current, char, request=request
        )
        if delta:
            content.append(delta.get("content", ""))
            calls.extend(delta.get("tool_calls", []))
        previous = current
    return "".join(content), calls


def test_streaming_holds_every_character_split_of_action_opener() -> None:
    wire = (
        "Before."
        '<|START_ACTION|>{"tool_call_id":"north-a",'
        '"tool_name":"read_file","parameters":{"path":"/tmp/a"}}'
        "<|END_ACTION|>After."
    )

    content, calls = _stream_characterwise(wire, _request("read_file"))

    assert content == "Before.After."
    assert [call["function"]["name"] for call in calls] == ["read_file"]


def test_streaming_ignores_end_marker_inside_json_string() -> None:
    value = "literal <|END_ACTION|> marker"
    wire = (
        "Before."
        + "<|START_ACTION|>"
        + json.dumps(
            {
                "tool_call_id": "north-a",
                "tool_name": "note_write",
                "parameters": {"body": value},
            }
        )
        + "<|END_ACTION|>After."
    )

    content, calls = _stream_characterwise(wire, _request("note_write"))

    assert content == "Before.After."
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"])["body"] == value


def test_streaming_emits_post_action_content_on_later_delta() -> None:
    parser = NorthToolParser(None)
    action = (
        '<|START_ACTION|>{"tool_call_id":"north-a",'
        '"tool_name":"read_file","parameters":{}}<|END_ACTION|>'
    )

    first = parser.extract_tool_calls_streaming(
        "", action, action, request=_request("read_file")
    )
    second = parser.extract_tool_calls_streaming(
        action, action + "Done.", "Done.", request=_request("read_file")
    )

    assert first and [tc["function"]["name"] for tc in first["tool_calls"]] == [
        "read_file"
    ]
    assert second == {"content": "Done."}
