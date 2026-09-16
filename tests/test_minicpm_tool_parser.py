# SPDX-License-Identifier: Apache-2.0
"""Focused contract tests for MiniCPM5's native XML tool-call format."""

from __future__ import annotations

import json

from rapid_mlx.tool_parsers import MiniCPMToolParser, ToolParserManager


def _request() -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                            "alerts": {"type": "boolean"},
                            "filters": {"type": "object"},
                        },
                    },
                },
            }
        ]
    }


def _arguments(call: dict) -> dict:
    return json.loads(call["arguments"])


def test_registry_and_wire_format_declaration() -> None:
    assert ToolParserManager.get_tool_parser("minicpm") is MiniCPMToolParser
    assert MiniCPMToolParser.EXPECTED_WIRE_FORMATS == ("minicpm_native",)
    assert MiniCPMToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True


def test_extracts_documented_xml_and_normalizes_schema_types() -> None:
    parser = MiniCPMToolParser()
    result = parser.extract_tool_calls(
        '<function name="weather">\n'
        '<param name="city"><![CDATA[San Francisco & Bay]]></param>\n'
        '<param name="days">3</param>\n'
        '<param name="alerts">true</param>\n'
        '<param name="filters">{"uv":true}</param>\n'
        "</function>",
        _request(),
    )

    assert result.tools_called is True
    assert result.content is None
    assert result.tool_calls[0]["name"] == "weather"
    assert _arguments(result.tool_calls[0]) == {
        "city": "San Francisco & Bay",
        "days": 3,
        "alerts": True,
        "filters": {"uv": True},
    }


def test_cdata_value_can_contain_a_literal_function_close_tag() -> None:
    result = MiniCPMToolParser().extract_tool_calls(
        '<function name="weather"><param name="city"><![CDATA[before '
        "</function> after]]></param></function>"
    )

    assert result.tools_called is True
    assert _arguments(result.tool_calls[0]) == {"city": "before </function> after"}


def test_preserves_non_tool_content_and_wire_order() -> None:
    parser = MiniCPMToolParser()
    result = parser.extract_tool_calls(
        'Before <function name="first"><param name="x">one</param></function>'
        ' between <function name="second"></function> after'
    )

    assert result.tools_called is True
    assert [call["name"] for call in result.tool_calls] == ["first", "second"]
    assert _arguments(result.tool_calls[0]) == {"x": "one"}
    assert _arguments(result.tool_calls[1]) == {}
    assert result.content == "Before  between  after"


def test_rejects_invalid_or_incomplete_markup_without_losing_content() -> None:
    parser = MiniCPMToolParser()
    for text in (
        "No tool call here.",
        '<functionality name="not-a-tool"></functionality>',
        '<function name="bad"><param name="x">1</param>',
        '<function name="bad"><param name="x"><![CDATA[1</function>',
        '<function name="bad"><param name="x">a & b</param></function>',
        '<function name="bad" extra="nope"><param name="x">1</param></function>',
        '<function name="bad"><param name="x">1</param><param name="x">2</param></function>',
        '<function name="bad"><param name="x"><nested /></param></function>',
    ):
        result = parser.extract_tool_calls(text)
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == text


def test_invalid_self_closed_function_does_not_hide_next_tool_call() -> None:
    result = MiniCPMToolParser().extract_tool_calls(
        '<function/><function name="weather"><param name="city">Paris</param>'
        "</function>",
        _request(),
    )

    assert result.tools_called is True
    assert result.content == "<function/>"
    assert result.tool_calls[0]["name"] == "weather"
    assert _arguments(result.tool_calls[0]) == {"city": "Paris"}


def test_malformed_outer_opener_does_not_swallow_a_nested_valid_call() -> None:
    """An unclosed ``<function>`` must not consume a later call's close tag.

    ``_function_end`` binds the single ``</function>`` to the outer opener,
    ``ET.fromstring`` then fails on the malformed span; the scanner must
    resynchronize at the next opener instead of skipping past the nested
    well-formed call.
    """
    result = MiniCPMToolParser().extract_tool_calls(
        '<function name="outer"><function name="inner">'
        '<param name="city">Paris</param></function>',
        _request(),
    )

    assert result.tools_called is True
    assert [call["name"] for call in result.tool_calls] == ["inner"]
    assert _arguments(result.tool_calls[0]) == {"city": "Paris"}


def test_valid_call_after_an_unterminated_opener_is_recovered() -> None:
    """A dangling opener with no close must not abandon later valid calls."""
    result = MiniCPMToolParser().extract_tool_calls(
        '<function name="broken"<function name="ok">'
        '<param name="city">Rome</param></function>',
        _request(),
    )

    assert result.tools_called is True
    assert [call["name"] for call in result.tool_calls] == ["ok"]
    assert _arguments(result.tool_calls[0]) == {"city": "Rome"}


def test_function_tag_inside_unterminated_cdata_is_not_a_tool_call() -> None:
    """A complete ``<function>`` literal in a still-open CDATA stays data.

    Resynchronization after the outer opener (whose CDATA has no ``]]>``)
    must not scan into that CDATA and fabricate an executable call from a
    string literal — everything after the unterminated CDATA is held.
    """
    parser = MiniCPMToolParser()
    text = (
        '<function name="outer"><param name="x">'
        '<![CDATA[<function name="evil"></function>'
    )

    result = parser.extract_tool_calls(text)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == text
    # Streaming holds the whole in-flight call; no markup leaks as content.
    assert parser.extract_tool_calls_streaming("", text, text) is None


def test_function_tag_inside_terminated_cdata_is_a_param_value_not_a_call() -> None:
    """A closed CDATA carrying a ``<function>`` literal is the param value."""
    result = MiniCPMToolParser().extract_tool_calls(
        '<function name="a"><param name="x">'
        '<![CDATA[<function name="evil"></function>]]></param></function>'
    )
    assert [call["name"] for call in result.tool_calls] == ["a"]
    assert _arguments(result.tool_calls[0]) == {
        "x": '<function name="evil"></function>'
    }


def test_whitespace_padded_param_name_is_stripped_to_the_schema_key() -> None:
    """Padded ``name`` attributes normalize like the stripped function name."""
    result = MiniCPMToolParser().extract_tool_calls(
        '<function name="weather"><param name=" days ">3</param></function>',
        _request(),
    )

    assert result.tools_called is True
    assert _arguments(result.tool_calls[0]) == {"days": 3}


def test_streaming_holds_partial_xml_and_emits_each_completed_call_once() -> None:
    parser = MiniCPMToolParser()
    first = 'Intro <function name="weather"><param name="city">San'
    second = first + " Francisco</param></function>"
    third = (
        second + '<function name="weather"><param name="city">Paris</param></function>'
    )

    assert parser.extract_tool_calls_streaming(
        "", first, first, request=_request()
    ) == {"content": "Intro "}
    first_event = parser.extract_tool_calls_streaming(
        first, second, second[len(first) :], request=_request()
    )
    assert first_event is not None
    assert _arguments(first_event["tool_calls"][0]["function"]) == {
        "city": "San Francisco"
    }
    second_event = parser.extract_tool_calls_streaming(
        second, third, third[len(second) :], request=_request()
    )
    assert second_event is not None
    assert second_event["tool_calls"][0]["index"] == 1
    assert second_event["tool_calls"][0]["function"]["name"] == "weather"
    assert _arguments(second_event["tool_calls"][0]["function"]) == {"city": "Paris"}


def test_streaming_keeps_text_adjacent_to_valid_and_invalid_markup() -> None:
    parser = MiniCPMToolParser()
    complete = (
        '<function name="weather"><param name="city">Paris</param></function> after'
    )
    event = parser.extract_tool_calls_streaming(
        "", complete, complete, request=_request()
    )
    assert event is not None
    assert event["content"] == " after"

    partial = "Text <fun"
    assert parser.extract_tool_calls_streaming("", partial, partial) == {
        "content": "Text "
    }
    assert parser.flush_held_content(partial) == "<fun"
    malformed = partial + 'ction name="weather" extra="bad"></function>'
    assert parser.extract_tool_calls_streaming(
        partial, malformed, malformed[len(partial) :]
    ) == {"content": '<function name="weather" extra="bad"></function>'}


def test_streaming_flushes_partial_opener_after_completed_tool_call() -> None:
    parser = MiniCPMToolParser()
    text = '<function name="weather"></function><fun'

    event = parser.extract_tool_calls_streaming("", text, text, request=_request())

    assert event is not None
    assert event["tool_calls"][0]["function"]["name"] == "weather"
    assert parser.flush_held_content(text) == "<fun"


def test_streaming_does_not_leak_outer_markup_when_cdata_holds_a_function_tag() -> None:
    """A ``<function`` inside an unterminated CDATA must not become the hold point.

    ``rfind`` would latch onto the CDATA occurrence and emit the outer
    opener as content mid-call; the forward CDATA-aware scan holds from the
    real outer opener so nothing leaks until the call completes.
    """
    parser = MiniCPMToolParser()
    partial = '<function name="run"><param name="cmd"><![CDATA[echo <function'

    assert parser.has_pending_tool_call(partial) is True
    # Held from the outer opener at index 0 -> no content leaks.
    assert parser.extract_tool_calls_streaming("", partial, partial) is None

    full = partial + "]]></param></function>"
    event = parser.extract_tool_calls_streaming(
        partial, full, full[len(partial) :], request=_request()
    )
    assert event is not None
    assert event["tool_calls"][0]["function"]["name"] == "run"
    assert _arguments(event["tool_calls"][0]["function"]) == {"cmd": "echo <function"}


def test_streaming_recovers_call_after_malformed_opener_matching_batch() -> None:
    """Malformed-opener recovery must be identical streaming vs. one-shot."""
    parser = MiniCPMToolParser()
    text = (
        '<function name="broken"<function name="ok">'
        '<param name="city">Rome</param></function>'
    )
    batch = MiniCPMToolParser().extract_tool_calls(text, _request())

    event = parser.extract_tool_calls_streaming("", text, text, request=_request())
    assert event is not None
    assert (
        [tc["function"]["name"] for tc in event["tool_calls"]]
        == [c["name"] for c in batch.tool_calls]
        == ["ok"]
    )
    assert _arguments(event["tool_calls"][0]["function"]) == {"city": "Rome"}


def test_flush_matches_streaming_residual_not_raw_text() -> None:
    """Final flush is computed from the same call-stripped residual as streaming."""
    parser = MiniCPMToolParser()
    # Completed call + trailing partial opener: flush releases only the tail.
    text = '<function name="weather"><param name="city">Rome</param></function><fun'
    assert parser.flush_held_content(text) == "<fun"
    # No pending opener after a fully-formed call: nothing to release.
    done = '<function name="weather"><param name="city">Rome</param></function> done'
    assert parser.flush_held_content(done) == ""


def test_streaming_content_and_final_flush_behave_like_plain_text() -> None:
    parser = MiniCPMToolParser()
    assert parser.extract_tool_calls_streaming("", "Hello", "Hello") == {
        "content": "Hello"
    }
    partial = 'Hello <function name="weather"'
    assert parser.extract_tool_calls_streaming("Hello", partial, partial[5:]) == {
        "content": " "
    }
    assert parser.has_pending_tool_call(partial) is True
    assert parser.flush_held_content(partial) == '<function name="weather"'
    assert parser.extract_tool_calls_streaming("wrong", "revised", "revised") == {
        "content": "revised"
    }
    assert parser.has_pending_tool_call("<functionality>") is False
    assert (
        parser.has_pending_tool_call('<function name="complete"></function>') is False
    )
    assert parser.flush_held_content("complete text") == ""
    assert parser.extract_tool_calls_streaming(
        "stale", 'Fresh <function name="weather"', 'Fresh <function name="weather"'
    ) == {"content": "Fresh "}
    revised = (
        '<function name="weather"><param name="city">Rome</param></function> after'
    )
    event = parser.extract_tool_calls_streaming(
        "stale", revised, revised, request=_request()
    )
    assert event is not None
    assert event["content"] == " after"
