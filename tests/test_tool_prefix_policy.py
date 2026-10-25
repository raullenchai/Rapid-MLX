"""A possible call is held; disproved or unfinished candidates stay visible."""

import json

import pytest

from rapid_mlx.api.utils import StreamingToolCallFilter
from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
from rapid_mlx.tool_parsers.json_prefix import tool_json_prefix
from tests.test_finalization_tool_contract import TOOLS, invoke

LITERALS = [
    '<tool_call>{"name":"other","arguments":{}}',
    '<tool_call>{"example":"literal"}</tool_call>',
    '<tool_call>{"name":"other","arguments":{}}</tool_call>',
    '<tool_call>{"name":"write_file"}</tool_call>',
    '<tool_call>{"name":"write_file","arguments":BROKEN}</tool_call>',
    '<tool_call>{"name":"write_file","arguments":{"x":"unfinished',
    '<tool_call>{"name":"write_file","arguments":{}} junk</tool_call>',
    '<tool_call>{"name":12,"arguments":{}}</tool_call>',
]


@pytest.mark.parametrize("raw", LITERALS)
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_literal_or_unfinished_envelope(surface, stream, raw):
    visible, calls, errors, response = invoke(surface, stream, raw, parser=None)
    assert visible == raw and not calls and not errors, response.text


@pytest.mark.parametrize("raw", LITERALS)
def test_every_split_preserves_literals(raw):
    for split in range(len(raw) + 1):
        filt = StreamingToolCallFilter(parser_name="hermes", request={"tools": TOOLS})
        actual = filt.process(raw[:split]) + filt.process(raw[split:]) + filt.flush()
        assert actual == raw, (split, actual)
        parser = HermesToolParser()
        parts = []
        previous = ""
        for delta in (raw[:split], raw[split:]):
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous, current, delta, request={"tools": TOOLS}
            )
            if result:
                assert not result.get("tool_calls")
                parts.append(result.get("content", ""))
            previous = current
        parts.append(parser.flush_held_content(raw))
        assert "".join(parts) == raw, (split, parts)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("suffix", ["", "<", "</tool_", "</tool_call"])
def test_valid_unclosed_call(surface, stream, suffix):
    raw = '<tool_call>{"name":"write_file","arguments":{"content":"hello"}}' + suffix
    visible, calls, errors, response = invoke(surface, stream, raw, parser=None)
    assert not visible and calls and not errors, response.text


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"x": [1, -2.5e-3, True, False, None, {}, []]},
        {"content": 'escaped "\\ \n \t </tool_call> \u1234'},
    ],
)
def test_json_prefix_every_boundary(value):
    raw = json.dumps({"name": "write_file", "arguments": value})
    for end in range(len(raw)):
        assert tool_json_prefix(raw[:end], {"write_file"}) == (True, None)
    assert tool_json_prefix(raw, {"write_file"}) == (True, len(raw))


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"name":false}',
        '{"name":"other","arguments":{}}',
        '{"name":"","arguments":{}}',
        '{"name":"write_file","arguments":[]}',
        '{"name":"write_file","arguments":{"x":01}}',
        '{"name":"write_file","arguments":{"x":truX',
        '{"name":"write_file","arguments":{"x":"\\q',
        '{"name":"write_file","arguments":{"x":"\\u12x',
        '{"name":"write_file","arguments":{"x":"\n',
        '{"name":"write_file","arguments":{"x":1,}}',
        '{"name":"write_file","arguments":{"x":[1,]}}',
        '{"name":"write_file","arguments":{"x":1 2}}',
        '{"name" 1',
        "{x",
        '{"example":1}',
    ],
)
def test_invalid_prefix(raw):
    assert tool_json_prefix(raw, {"write_file"})[0] is False


def test_nonviable_prefix_releases_before_eof():
    filt = StreamingToolCallFilter(parser_name="hermes", request={"tools": TOOLS})
    start = '<tool_call>{"name":"write_file","arguments":{"x":"'
    assert filt.process(start + "x" * 1048577) == ""
    assert filt.process("\\q") == start + "x" * 1048577 + "\\q"
    assert filt.process(" tail") == " tail"
    assert filt.flush() == ""


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("literal", [False, True])
def test_api_marker_split_boundaries(surface, literal, monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput
    from tests.test_finalization_tool_contract import Engine

    raw = (
        '<tool_call>{"example":"literal"}</tool_call>'
        if literal
        else '<tool_call>{"name":"write_file","arguments":{"content":"</tool_call>"}}</tool_call>'
    )
    boundaries = set(range(len("<tool_call>") + 1))
    boundaries.update(range(len(raw) - len("</tool_call>"), len(raw) + 1))
    for split in sorted(boundaries):

        async def chunks(self, split=split, **kwargs):
            for start, end in ((0, split), (split, len(raw))):
                yield GenerationOutput(
                    text=raw[:end], new_text=raw[start:end], finished=False
                )
            yield GenerationOutput(
                text=raw, new_text="", finished=True, finish_reason="stop"
            )

        monkeypatch.setattr(Engine, "stream_chat", chunks)
        visible, calls, errors, response = invoke(surface, True, raw, parser=None)
        assert not errors, response.text
        if literal:
            assert visible == raw and not calls, (split, response.text)
        else:
            assert not visible and calls, (split, response.text)


@pytest.mark.parametrize(
    "raw",
    [
        '{"name":"write_file","arguments":{"x":-',
        '{"name":"write_file","arguments":{"x":1.',
        '{"name":"write_file","arguments":{"x":1e',
        '{"name":"write_file","arguments":{"x":1e-',
        '{"name":"write_file","arguments":{"x":"\\',
        '{"name":"write_file","arguments":{"x":"\\u1',
        '{"name":"write_file","arguments":{"x":"\\u1234',
    ],
)
def test_incomplete_token_is_viable(raw):
    assert tool_json_prefix(raw, {"write_file"}) == (True, None)


def test_undecided_whitespace_cutoff_releases_exact_bytes():
    filt = StreamingToolCallFilter(parser_name="hermes")
    assert filt.process("<tool_call>" + " " * 32768) == ""
    assert filt.process(" " * 32768) == ""
    assert filt.process(" ") == "<tool_call>" + " " * 65537
    assert filt.flush() == ""


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_unclosed_candidate_after_completed_call(surface, stream, complete):
    first = '<tool_call>{"name":"write_file","arguments":{}}</tool_call>'
    tail = '<tool_call>{"name":"write_file","arguments":' + (
        "{}}" if complete else '{"x":'
    )
    visible, calls, errors, response = invoke(
        surface, stream, first + tail, parser=None, step=len(first)
    )
    assert not errors, response.text
    assert visible == ("" if complete else tail), response.text
    if surface == "messages" and stream:
        calls = [c for c in calls if c.get("type") == "tool_use"]
    assert len(calls) == (2 if complete else 1), response.text


@pytest.mark.parametrize(
    "arguments", ["<malformed_json_arguments>broken", "<malformed_function_body>broken"]
)
def test_forced_repair_never_turns_parser_diagnostic_into_empty_call(arguments):
    from rapid_mlx.api.models import FunctionCall, ToolCall
    from rapid_mlx.routes.chat import _repair_forced_call_arguments

    call = ToolCall(
        id="call_invalid",
        type="function",
        function=FunctionCall(name="write_file", arguments=arguments),
    )
    error = _repair_forced_call_arguments([call], "", "write_file", TOOLS)
    assert error and "valid JSON" in error
    assert call.function.arguments == arguments


def test_argument_validation_rejects_malformed_parser_output():
    from rapid_mlx.api.models import FunctionCall, ToolCall
    from rapid_mlx.service.helpers import (
        _InvalidToolArgumentsError,
        _validate_tool_call_params,
    )

    call = ToolCall(
        id="call_invalid",
        type="function",
        function=FunctionCall(name="write_file", arguments="BROKEN"),
    )
    with pytest.raises(_InvalidToolArgumentsError, match="valid JSON"):
        _validate_tool_call_params([call], TOOLS)


def test_eof_probe_read_failure_is_safe_and_closes_handle():
    from rapid_mlx.api.utils import ToolProbeStorageError

    class FailedRead:
        closed = False

        def seek(self, offset):
            pass

        def read(self):
            raise OSError("private filesystem detail")

        def close(self):
            self.closed = True

    filt = StreamingToolCallFilter(parser_name="hermes")
    pending = FailedRead()
    filt._prefix_whitespace = pending
    with pytest.raises(ToolProbeStorageError) as error:
        filt.flush()
    assert "private" not in str(error.value)
    assert pending.closed
    filt.close()


@pytest.mark.parametrize("arguments", ["BROKEN", "1", "[]", "null"])
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_encoded_arguments_are_literal(surface, stream, arguments):
    raw = (
        "<tool_call>"
        + json.dumps({"name": "write_file", "arguments": arguments})
        + "</tool_call>"
    )
    visible, calls, errors, response = invoke(surface, stream, raw, parser=None)
    assert visible == raw and not calls and not errors, response.text


def test_encoded_object_arguments_are_valid():
    raw = json.dumps({"name": "write_file", "arguments": '{"content":"hello"}'})
    assert tool_json_prefix(raw, {"write_file"}) == (True, len(raw))


def test_two_closed_streamed_calls_do_not_replay_first():
    parser = HermesToolParser()
    raw = '<tool_call>{"name":"write_file","arguments":{}}</tool_call>'
    first = parser.extract_tool_calls_streaming("", raw, raw, request={"tools": TOOLS})
    second = parser.extract_tool_calls_streaming(
        raw, raw + raw, raw, request={"tools": TOOLS}
    )
    assert len(first["tool_calls"]) == len(second["tool_calls"]) == 1
    assert second["tool_calls"][0]["index"] == 1
    assert not second.get("content")


@pytest.mark.parametrize("arguments", [123, [], "123", "BROKEN"])
@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_forced_choice_never_synthesizes_rejected_shape(surface, stream, arguments):
    raw = (
        "<tool_call>"
        + json.dumps({"name": "write_file", "arguments": arguments})
        + "</tool_call>"
    )
    _, calls, errors, response = invoke(
        surface, stream, raw, parser=None, choice="required"
    )
    assert not calls and errors, response.text
    assert "valid JSON" in json.dumps(errors), response.text


@pytest.mark.parametrize(
    "body",
    [
        '{"name":"write_file","arguments":{"a":[0,-1,1.2,-0.3,1e2,1E+2,1e-2,true,false,null,{},[]],"s":"a\\n\\u0041\\""},"extra":"ok"}',
        '{"arguments":"{\\"x\\":1}","name":"write_file"}',
        '{"name":"other","arguments":{}}',
        '{"name":"","arguments":{}}',
        '{"name":1,"arguments":{}}',
        '{"name":"write_file","arguments":[]}',
        '{"name":"write_file","arguments":"1"}',
        '{"name":"write_file","arguments":"BROKEN"}',
        '{"example":"literal"}',
        "{}",
        "[]",
        "{x",
        '{"name"!',
        '{"name":"write_file"!',
        '{"name":"write_file","arguments":{"x":truth}}',
        '{"name":"write_file","arguments":{"x":01}}',
        '{"name":"write_file","arguments":{"x":1.}}',
        '{"name":"write_file","arguments":{"x":-}}',
        '{"name":"write_file","arguments":{"x":1e+}}',
        '{"name":"write_file","arguments":{"x":?}}',
        '{"name":"write_file","arguments":{"x":"\\q"}}',
        '{"name":"write_file","arguments":{"x":"\\uX"}}',
        '{"name":"write_file","arguments":{"x":"\x01"}}',
    ],
)
def test_incremental_json_matches_prefix_policy(body):
    from rapid_mlx.tool_parsers.json_prefix import ToolJSONPrefix

    for split in range(len(body) + 1):
        checker = ToolJSONPrefix({"write_file"})
        checker.feed(body[:split])
        viable, end = tool_json_prefix(body[:split], {"write_file"})
        assert checker.viable == viable, (body, split)
        if viable:
            assert checker.complete == (end is not None)
        checker.feed(body[split:])
        assert checker.viable == tool_json_prefix(body, {"write_file"})[0]


def test_large_json_deltas_are_examined_once(monkeypatch):
    from rapid_mlx.tool_parsers.json_prefix import ToolJSONPrefix

    examined = 0
    original = ToolJSONPrefix.feed

    def counted(self, text):
        nonlocal examined
        examined += len(text)
        return original(self, text)

    monkeypatch.setattr(ToolJSONPrefix, "feed", counted)
    filt = StreamingToolCallFilter(parser_name="hermes", request={"tools": TOOLS})
    assert filt.process('<tool_call>{"name":"write_file","arguments":{"x":"') == ""
    for _ in range(20000):
        assert filt.process("a") == ""
        assert not filt._buffer
    assert examined < 20100
    assert filt.process('"}}') == ""
    assert filt.flush() == ""
    assert not filt._json_parts


def test_json_probe_cleanup_releases_chunks():
    filt = StreamingToolCallFilter(parser_name="hermes")
    assert not filt.process('<tool_call>{"name":"write_file","arguments":{"x":"')
    assert filt._json_parts
    filt.close()
    assert not filt._json_parts and filt._json_prefix is None and not filt._buffer


def test_legacy_classifier_callback_preserves_json_delimiters():
    raw = (
        '<tool_call>{"name":"write_file","arguments":{"x":"</tool_call>"}}</tool_call>'
    )
    for split in range(len(raw) + 1):
        filt = StreamingToolCallFilter(
            classify_prefix=lambda opener, body: (
                HermesToolParser.classify_tool_call_prefix(opener, body)
            )
        )
        assert (
            filt.process(raw[:split]) + filt.process(raw[split:]) + filt.flush() == ""
        )


def test_incremental_multi_digit_numbers():
    from rapid_mlx.tool_parsers.json_prefix import ToolJSONPrefix

    checker = ToolJSONPrefix()
    raw = '{"name":"write_file","arguments":{"x":123.456e789}}'
    for char in raw:
        checker.feed(char)
        assert checker.viable
    assert checker.complete


@pytest.mark.parametrize("name", ["hermes", "nous", "qwen3_coder", "qwen3_coder_xml"])
def test_registry_classifier_request_binding(name):
    from rapid_mlx.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser(name)
    is_json = name != "qwen3_coder_xml"
    assert (parser is HermesToolParser) == is_json
    raw = (
        '<tool_call>{"name":"write_file","arguments":{}}</tool_call>'
        if is_json
        else "<tool_call><function=write_file></function></tool_call>"
    )
    for split in range(len(raw) + 1):
        filt = StreamingToolCallFilter(parser_name=name, request={"tools": TOOLS})
        assert filt._json_enabled == is_json
        assert (
            filt.process(raw[:split]) + filt.process(raw[split:]) + filt.flush() == ""
        )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_large_call_after_literal_uses_pending_prefix(surface, stream):
    literal = '<tool_call>{"example":"literal"}</tool_call>'
    raw = (
        literal
        + "<tool_call>"
        + json.dumps({"name": "write_file", "arguments": {"content": "x" * 140000}})
        + "</tool_call>"
    )
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, step=32768
    )
    assert visible == literal and calls and not errors, response.text[:1000]


def test_eof_only_suffix_uses_stream_authorization():
    parser = HermesToolParser()
    prefix = '<tool_call>{"name":"'
    assert (
        parser.extract_tool_calls_streaming(
            "", prefix, prefix, request={"tools": TOOLS}
        )
        is None
    )
    final = '<tool_call>{"name":"other","arguments":{}}'
    assert parser.flush_held_content(final) == final
    parser.reset()
    assert parser._stream_request is None


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_forced_undeclared_name_is_unfulfilled_not_invalid_json(surface):
    raw = '<tool_call>{"name":"other","arguments":{}}</tool_call>'
    visible, calls, errors, response = invoke(
        surface, False, raw, parser=None, choice="required"
    )
    assert response.status_code == 422 and errors and not calls and not visible
    assert "required declared tool call" in json.dumps(errors)
    assert "valid JSON" not in json.dumps(errors)


def test_complete_json_with_visible_suffix_is_released_at_every_split():
    raw = '<tool_call>{"name":"write_file","arguments":{}}Final answer'
    for split in range(len(raw) + 1):
        filt = StreamingToolCallFilter(parser_name="hermes", request={"tools": TOOLS})
        assert (
            filt.process(raw[:split]) + filt.process(raw[split:]) + filt.flush() == raw
        )


@pytest.mark.parametrize(
    "choice",
    ["auto", "required", {"type": "function", "function": {"name": "write_file"}}],
)
def test_typed_request_eof_choice_guard(choice):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.service.helpers import _InvalidToolArgumentsError
    from tests.test_postprocessor import StreamingPostProcessor, _make_cfg, _make_output

    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "test"}],
        tools=TOOLS,
        tool_choice=choice,
    )
    pp = StreamingPostProcessor(
        _make_cfg(enable_auto_tool_choice=True, tool_call_parser="hermes"),
        request=request,
    )
    raw = '<tool_call>{"name":'
    pp.process_chunk(_make_output(raw))
    if choice == "auto":
        assert "".join(e.content or "" for e in pp.finalize()) == raw
    else:
        with pytest.raises(_InvalidToolArgumentsError):
            pp.finalize()


def test_large_call_after_prose_at_each_opener_split():
    from tests.test_postprocessor import StreamingPostProcessor, _make_cfg, _make_output

    opener = "<tool_call>"
    body = (
        json.dumps({"name": "write_file", "arguments": {"content": "x" * 140000}})
        + "</tool_call>"
    )
    for split in range(len(opener) + 1):
        pp = StreamingPostProcessor(
            _make_cfg(enable_auto_tool_choice=True, tool_call_parser="hermes"),
            request={"tools": TOOLS},
        )
        events = pp.process_chunk(_make_output("Intro. " + opener[:split]))
        rest = opener[split:] + body
        for start in range(0, len(rest), 32768):
            events.extend(pp.process_chunk(_make_output(rest[start : start + 32768])))
        events.extend(pp.finalize())
        assert not pp._tool_suppression_released
        assert "x" * 100 not in "".join(e.content or "" for e in events)
        calls = [call for event in events for call in event.tool_calls or []]
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"])["content"] == "x" * 140000


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_malformed_native_call_never_becomes_executable(surface, stream):
    raw = "<tool_call><function=write_file>oops</tool_call>"
    _, calls, errors, response = invoke(surface, stream, raw, parser=None, step=1)
    assert errors and not calls, response.text
    assert "valid JSON" in json.dumps(errors), response.text
