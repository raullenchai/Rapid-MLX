# SPDX-License-Identifier: Apache-2.0
"""Declared tools authorize parsed calls across buffered and streaming paths."""

import json
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.routes.chat import router
from rapid_mlx.service import helpers
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers.abstract_tool_parser import ExtractedToolCallInformation
from tests.test_empty_rejected_tool_reply import (
    NOTICE,
    REJECTED,
    TOOLS,
    _Engine,
    _Request,
)
from tests.test_postprocessor import _make_cfg, _make_output

GEMMA = '<|tool_call>call:send_email{body:<|"|>hello<|"|>}<tool_call|>'
VALID = '<tool_call><function=write_file>{"path":"x"}</function></tool_call>'


@pytest.mark.parametrize("closer", ["</function>", "</tool_call>"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_nemotron_buffered_chat_preserves_post_call_literal_closer(wrapped, closer):
    """Exercise the fresh-parser helper used by non-stream chat/completions."""
    cfg = reset_config()
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "nemotron"
    function = "<function=write_file><parameter=path>x</parameter></function>"
    prefix = "<tool_call>" + function + "</tool_call>" if wrapped else function
    tail = f"Explain {closer} literally."
    try:
        content, calls = helpers._parse_tool_calls_with_parser(
            prefix + " " + tail, _Request()
        )
        assert content == tail
        assert len(calls) == 1
        assert calls[0].function.name == "write_file"
        assert json.loads(calls[0].function.arguments) == {"path": "x"}
    finally:
        reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "parser,text,expected,valid",
    [
        ("nemotron", REJECTED, NOTICE, False),
        ("nemotron", "Before. " + REJECTED + " After.", "Before.  After.", False),
        ("nemotron", REJECTED + VALID, "", True),
        ("gemma4", GEMMA, NOTICE, False),
    ],
)
def test_chat_all_formats(stream, parser, text, expected, valid):
    cfg = reset_config()
    cfg.engine = _Engine(text, deltas=text.splitlines(keepends=True))
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = parser
    app = FastAPI()
    app.include_router(router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": TOOLS,
            },
        )
        assert response.status_code == 200, response.text
        if stream:
            choices = [
                c
                for line in response.text.splitlines()
                if line.startswith("data: {")
                for c in json.loads(line[6:])["choices"]
            ]
            content = "".join(c["delta"].get("content") or "" for c in choices)
            calls = [tc for c in choices for tc in c["delta"].get("tool_calls", [])]
            finish = choices[-1]["finish_reason"]
        else:
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            calls = choice["message"].get("tool_calls") or []
            finish = choice["finish_reason"]
        assert content.strip() == expected.strip()
        assert bool(calls) == valid
        if valid:
            assert len(calls) == 1
            call = calls[0]
            assert isinstance(call["id"], str) and call["id"]
            assert call == {
                **({"index": 0} if stream else {}),
                "id": call["id"],
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path":"x"}'},
            }
        else:
            assert calls == []
        assert finish == ("tool_calls" if valid else "stop")
    finally:
        reset_config()


@pytest.mark.parametrize(
    "parser,text",
    [
        (
            "hermes",
            '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>',
        ),
        (
            "qwen",
            '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>',
        ),
        (
            "qwen3_xml",
            '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>',
        ),
        ("gemma4", GEMMA),
        ("nemotron", REJECTED),
        ("llama", '<function=send_email>{"x":1}</function>'),
        ("mistral", '[TOOL_CALLS] [{"name":"send_email","arguments":{"x":1}}]'),
        (
            "harmony",
            '<|start|>assistant to=functions.send_email<|channel|>commentary json<|message|>{"x":1}<|call|>',
        ),
    ],
)
def test_buffered_parser_authorization(monkeypatch, parser, text):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", parser)
    content, calls = helpers._parse_tool_calls_with_parser(text, _Request())
    assert not calls
    if parser == "hermes":
        assert content == text
    else:
        assert helpers._was_undeclared_tool_dropped()
        assert not content.strip()


@pytest.mark.parametrize(
    "parser,text",
    [
        pytest.param(
            "hy_v3",
            '<tool_call:opensource>send_email<tool_sep:opensource>{"x":1}<end_of_tool_call:opensource>',
            id="hy_v3-parser-refusal",
        ),
        pytest.param(
            "k2_horizon",
            '<ifm|tool_calls><ifm|tool_call>{"name":"send_email","arguments":{"x":1}}</ifm|tool_call></ifm|tool_calls>',
            id="k2_horizon-parser-refusal",
        ),
        pytest.param(
            "glm47",
            "<tool_call>send_email<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            id="glm47-parser-refusal",
        ),
    ],
)
def test_parser_level_undeclared_refusal(parser, text):
    """These parsers gate names internally; this does not test shared filtering."""
    cls = helpers.ToolParserManager.get_tool_parser(parser)
    parser_instance = cls(None)
    allowed = {
        "chat_template_kwargs": {"tool_call_format": "json"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            }
        ],
    }
    accepted = parser_instance.extract_tool_calls(text, allowed)
    assert accepted.tools_called
    assert len(accepted.tool_calls) == 1
    assert accepted.tool_calls[0]["name"] == "send_email"
    parser_instance.reset()
    refused = parser_instance.extract_tool_calls(
        text, {"tools": TOOLS, "chat_template_kwargs": {"tool_call_format": "json"}}
    )
    assert not refused.tools_called
    assert refused.tool_calls == []


@pytest.mark.parametrize(
    "parser_name", ["hy_v3", "k2_horizon", "glm47", "arbitrary_stub"]
)
def test_shared_buffered_authorization_of_parser_output(monkeypatch, parser_name):
    """Any parser returning an undeclared call must pass the shared filter."""
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", parser_name)
    parser = Mock()
    parser._undeclared_dropped = 0
    parser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=[
            {"id": "bad", "name": "send_email", "arguments": "{}"},
            {"id": "good", "name": "write_file", "arguments": "{}"},
        ],
        content="Kept narration",
    )
    monkeypatch.setattr(
        helpers.ToolParserManager, "get_tool_parser", lambda _: lambda _: parser
    )
    content, calls = helpers._parse_tool_calls_with_parser("wire", _Request())
    parser.extract_tool_calls.assert_called_once_with("wire", {"tools": TOOLS})
    assert content == "Kept narration"
    assert [call.model_dump() for call in calls] == [
        {
            "id": "good",
            "type": "function",
            "function": {"name": "write_file", "arguments": "{}"},
        }
    ]
    assert helpers._was_undeclared_tool_dropped()


def processor(parser="hermes"):
    return StreamingPostProcessor(
        _make_cfg(enable_auto_tool_choice=True, tool_call_parser=parser),
        request={"tools": TOOLS},
    )


def anchor(name, index=0):
    return {
        "index": index,
        "id": f"call_{index}",
        "function": {"name": name, "arguments": "{"},
    }


def test_incremental_drop_and_renumber():
    pp = processor()
    pp.tool_parser = Mock()
    pp.tool_parser.EXPECTED_WIRE_FORMATS = ("tool_call_json",)
    pp.tool_parser._undeclared_dropped = 0
    pp.tool_parser.extract_tool_calls_streaming.side_effect = [
        {"tool_calls": [anchor("send_email")], "content": "Before"},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"x":1}'}}]},
        {"tool_calls": [{"function": {"arguments": "ignored"}}]},
        {"tool_calls": [anchor("write_file", 1)]},
        {"tool_calls": [{"index": 1, "function": {"arguments": '"path":"x"}'}}]},
    ]
    assert pp._detect_tool_calls("<tool_call>") == {"content": "Before"}
    assert not pp.tool_calls_detected
    assert pp._detect_tool_calls("x") == {"content": ""}
    assert pp._detect_tool_calls("x") == {"content": ""}
    assert pp._detect_tool_calls("x")["tool_calls"][0]["index"] == 0
    assert pp._detect_tool_calls("x")["tool_calls"][0]["index"] == 0
    assert pp.undeclared_tool_dropped
    pp.reset()
    assert not pp.undeclared_tool_dropped


def test_channel_undeclared_finish():
    pp = processor()
    events = pp.process_chunk(
        _make_output(
            "",
            channel="tool_call",
            tool_calls=[{"name": "send_email", "arguments": "{}"}],
            finished=True,
            finish_reason="tool_calls",
        )
    )
    assert not any(e.tool_calls for e in events)
    assert not pp.tool_calls_detected
    assert events[-1].finish_reason == "stop"


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("name", ["write_file", "send_email"])
@pytest.mark.parametrize("channel", ["content", "tool_call"])
def test_channel_structured_call_preserves_narration(channel, name, finished):
    pp = processor()
    text = "I will write the file." if channel == "content" else '{"path":"x"}'
    events = pp.process_chunk(
        _make_output(
            text,
            channel=channel,
            tool_calls=[{"name": name, "arguments": '{"path":"x"}'}],
            finished=finished,
            finish_reason="tool_calls" if finished else None,
        )
    )
    assert "".join(e.content or "" for e in events) == (
        text if channel == "content" else ""
    )
    calls = [call for e in events for call in (e.tool_calls or [])]
    assert [call["function"] for call in calls] == (
        [{"name": "write_file", "arguments": '{"path":"x"}'}]
        if name == "write_file"
        else []
    )
    if channel == "content" and calls:
        assert [e.type for e in events] == ["content", "tool_call"]
        assert pp.accumulated_text == text
    if finished:
        assert events[-1].finish_reason == (
            "tool_calls" if name == "write_file" else "stop"
        )


@pytest.mark.parametrize("finished", [False, True])
def test_channel_structured_none_preserves_narration(finished):
    pp = StreamingPostProcessor(
        _make_cfg(enable_auto_tool_choice=True, tool_call_parser="hermes"),
        request={"tools": TOOLS, "tool_choice": "none"},
    )
    events = pp.process_chunk(
        _make_output(
            "Narration.",
            channel="content",
            tool_calls=[{"name": "write_file", "arguments": "{}"}],
            finished=finished,
            finish_reason="tool_calls" if finished else None,
        )
    )
    assert "".join(e.content or "" for e in events) == "Narration."
    assert not any(e.tool_calls for e in events)
    if finished:
        assert events[-1].finish_reason == "stop"


@pytest.mark.parametrize("finished", [False, True])
def test_channel_structured_cap_preserves_narration(finished):
    pp = StreamingPostProcessor(
        _make_cfg(enable_auto_tool_choice=True, tool_call_parser="hermes"),
        request={"tools": TOOLS, "parallel_tool_calls": False},
    )
    call = {"name": "write_file", "arguments": "{}"}
    pp.process_chunk(_make_output(channel="tool_call", tool_calls=[call]))
    events = pp.process_chunk(
        _make_output(
            "More narration.",
            channel="content",
            tool_calls=[call],
            finished=finished,
        )
    )
    assert "".join(e.content or "" for e in events) == "More narration."
    assert not any(e.tool_calls for e in events)


@pytest.mark.parametrize("mode", ["extract", "fallback", "raw", "reported"])
def test_finalize_authorizes_every_recovery(mode):
    pp = processor()
    pp.tool_parser = Mock()
    pp.tool_parser.EXPECTED_WIRE_FORMATS = ("tool_call_json",)
    pp.tool_parser._undeclared_dropped = int(mode == "reported")
    pp.tool_parser.finalize_legacy_raw_stream = None
    pp.tool_parser.flush_pending = Mock(return_value=None)
    pp.tool_parser.finish = Mock(return_value=None)
    pp.tool_parser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=mode == "extract",
        tool_calls=[{"id": "bad", "name": "send_email", "arguments": "{}"}]
        if mode == "extract"
        else [],
        content=None,
    )
    if mode == "raw":
        pp.tool_parser.finalize_legacy_raw_stream = Mock(
            return_value={"tool_calls": [anchor("send_email")]}
        )
    pp.tool_accumulated_text = (
        '<tool_call>{"name":"send_email","arguments":{"x":1}}</tool_call>'
    )
    events = pp.finalize()
    assert not any(e.tool_calls for e in events)
    assert not pp.tool_calls_detected


@pytest.mark.parametrize(
    "req", [None, {}, {"tools": []}, {"tools": [{"name": "write_file"}]}, _Request()]
)
def test_buffered_structured_shapes(monkeypatch, req):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "gemma4")
    content, calls = helpers._parse_tool_calls_with_parser(
        "Narration",
        req,
        structured_tool_calls=[
            {"name": "send_email", "arguments": "{}"},
            {"name": "write_file", "arguments": "{}"},
        ],
    )
    assert content == "Narration"
    if req and req != {"tools": []}:
        assert [c.function.name for c in calls] == ["write_file"]
    else:
        assert calls is None
        assert not helpers._was_undeclared_tool_dropped()


def test_nemotron_boundaries_and_logging(caplog):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    pp = NemotronToolParser()
    pp.reset()
    prefix = REJECTED.split("</tool_call>")[0]
    assert pp.extract_tool_calls_streaming(
        "", prefix, prefix, request={"tools": TOOLS}
    ) in (None, {"content": ""})
    delta = "</tool_call>"
    assert pp.extract_tool_calls_streaming(
        prefix, prefix + delta, delta, request={"tools": TOOLS}
    ) == {"content": ""}
    assert pp._undeclared_dropped == 1
    assert sum("dropped undeclared" in r.message for r in caplog.records) == 1
    assert not any("unhandled variant" in r.message for r in caplog.records)
    for text in [
        "```xml\n" + REJECTED + "\n```",
        REJECTED.replace("<tool_call>", "").replace("</tool_call>", ""),
        REJECTED.replace("today's price", "x" * 32768),
    ]:
        pp.reset()
        assert pp.extract_tool_calls(text, {"tools": TOOLS}).content == text
        assert pp._undeclared_dropped == 0


def test_nemotron_every_byte_split():
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    wire = "Before. " + REJECTED + " After."
    for split in range(1, len(wire)):
        parser = NemotronToolParser()
        parser.reset()
        previous = ""
        visible = ""
        for delta in [wire[:split], wire[split:]]:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous, current, delta, request={"tools": TOOLS}
            )
            if result:
                visible += result.get("content", "")
            previous = current
        visible += parser.flush_held_content(wire)
        assert visible == "Before.  After.", (split, visible)


def test_stream_filter_no_declared_tools():
    pp = processor()
    pp.request = None
    calls = [anchor("send_email")]
    assert pp._filter_declared_calls(calls, incremental=True) == calls


def test_nemotron_partial_prefix_flush():
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    parser.reset()
    assert parser.extract_tool_calls_streaming("", "Hello <too", "Hello <too") == {
        "content": "Hello "
    }
    assert parser.flush_held_content("Hello <too") == ""
    assert parser.flush_held_content("Hello <too") == ""
    parser.reset()
    assert parser.flush_held_content(REJECTED) == ""


def test_declared_pydantic_tool_in_dict():
    from rapid_mlx.api.models import ToolDefinition
    from rapid_mlx.tool_call_authorization import declared_names

    assert declared_names(
        {"tools": [ToolDefinition(**TOOLS[0]), None, {}, {"name": None}]}
    ) == {"write_file"}


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("authorized", [set(), {"write_file", "send_email"}])
def test_request_authorization_survives_copy_and_streaming_dump(deep, authorized):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.tool_call_authorization import declared_names

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Use a tool"}], tools=TOOLS
    )
    assert declared_names(request) == {"write_file"}
    request._tool_authorization_names = authorized
    copied = request.model_copy(deep=deep)
    assert copied._tool_authorization_names == authorized
    assert declared_names(copied) == authorized

    dumped = copied.model_dump(exclude_none=True)
    assert "_tool_authorization_names" not in dumped
    assert "_tool_authorization_names" not in copied.model_dump_json()
    assert (
        "_tool_authorization_names"
        not in ChatCompletionRequest.model_json_schema()["properties"]
    )
    # The route transfers the snapshot separately from the narrowed prompt tools.
    pp = StreamingPostProcessor(
        _make_cfg(),
        request=dumped,
        tool_authorization_names=copied._tool_authorization_names,
    )
    call = {"function": {"name": "send_email", "arguments": "{}"}}
    assert pp._filter_declared_calls([call]) == [call]


@pytest.mark.parametrize("called", [False, True])
def test_inferred_parser_reports_drops(monkeypatch, called):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", False)
    monkeypatch.setattr(cfg, "reasoning_parser_name", "minimax")
    parser = Mock()
    parser._undeclared_dropped = 1
    parser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=called,
        content="Kept",
        tool_calls=[{"id": "ok", "name": "write_file", "arguments": "{}"}]
        if called
        else [],
    )
    monkeypatch.setattr(
        helpers.ToolParserManager, "get_tool_parser", lambda _: lambda _: parser
    )
    content, calls = helpers._parse_tool_calls_with_parser(REJECTED, _Request())
    assert content == "Kept"
    assert bool(calls) == called
    assert helpers._was_undeclared_tool_dropped()


@pytest.mark.parametrize("channel", [None, "content"])
def test_dropped_stream_keeps_prose_and_honest_finish(channel):
    pp = processor()
    pp.tool_parser = Mock()
    pp.tool_parser.EXPECTED_WIRE_FORMATS = ("tool_call_json",)
    pp.tool_parser._undeclared_dropped = 0
    pp.tool_parser.extract_tool_calls_streaming.side_effect = [
        {"tool_calls": [anchor("send_email")], "content": "Before. "},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"x":1}'}}]},
        {"content": "After."},
    ]
    events = []
    for i, text in enumerate(["<attempt>", "arguments", "tail"]):
        events += pp.process_chunk(
            _make_output(
                text,
                channel=channel,
                finished=i == 2,
                finish_reason="tool_calls" if i == 2 else None,
            )
        )
    assert not any(e.tool_calls for e in events)
    assert "".join(e.content or "" for e in events) == "Before. After."
    assert events[-1].finish_reason == "stop"
    assert not pp.tool_calls_detected


def test_qwen_structured_forced_undeclared_reports_drop(monkeypatch):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    content, calls = helpers._parse_tool_calls_with_parser(
        "Kept",
        {
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "write_file"}},
        },
        structured_tool_calls=[{"name": "send_email", "arguments": "{}"}],
    )
    assert content == "Kept"
    assert calls is None
    assert helpers._was_undeclared_tool_dropped()


def test_buffered_authorization_ignores_forced_choice(monkeypatch):
    cfg = helpers.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", "gemma4")
    _, calls = helpers._parse_tool_calls_with_parser(
        "",
        {
            "tools": [*TOOLS, {"name": "send_email"}],
            "tool_choice": {"type": "function", "function": {"name": "write_file"}},
        },
        structured_tool_calls=[{"name": "send_email", "arguments": "{}"}],
    )
    assert calls[0].function.name == "send_email"
    assert not helpers._was_undeclared_tool_dropped()


LITERAL_OPEN = (
    '<tool_call><function=send_email><parameter=body>print("<tool_call>")'
    "</parameter></function></tool_call>"
)
FENCED_EXAMPLE = "```xml\n<tool_call>example</tool_call>\n```"


def _route_reply(
    text,
    *,
    stream,
    deltas=None,
    tools=TOOLS,
    tool_choice=None,
    expected_status=200,
    expected_sse_error=None,
):
    cfg = reset_config()
    cfg.engine = _Engine(text, deltas=deltas)
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "nemotron" if tool_choice is None else "hermes"
    app = FastAPI()
    app.include_router(router)
    try:
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": stream,
                "messages": [{"role": "user", "content": "Use a tool"}],
                "tools": tools,
                "tool_choice": tool_choice,
            },
        )
        assert response.status_code == expected_status, response.text
        if expected_status != 200:
            return response.json()
        if stream and expected_sse_error:
            packets = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            errors = [packet["error"] for packet in packets if "error" in packet]
            assert len(errors) == 1 and errors[0]["code"] == expected_sse_error
            assert not any(
                choice.get("delta", {}).get("tool_calls")
                for packet in packets
                for choice in packet.get("choices", [])
            )
            return errors[0]
        if stream:
            choices = [
                c
                for line in response.text.splitlines()
                if line.startswith("data: {")
                for c in json.loads(line[6:])["choices"]
            ]
            content = "".join(c["delta"].get("content") or "" for c in choices)
            names = [
                tc["function"]["name"]
                for c in choices
                for tc in c["delta"].get("tool_calls", [])
                if tc.get("function", {}).get("name")
            ]
            finish = choices[-1]["finish_reason"]
        else:
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            names = [
                tc["function"]["name"]
                for tc in choice["message"].get("tool_calls") or []
            ]
            finish = choice["finish_reason"]
        return content.strip(), names, finish
    finally:
        reset_config()


@pytest.mark.parametrize(
    "text,expected",
    [
        (LITERAL_OPEN, NOTICE),
        ("Before. " + LITERAL_OPEN + " After.", "Before.  After."),
        (LITERAL_OPEN + REJECTED, NOTICE),
        (REJECTED + "\n" + FENCED_EXAMPLE, FENCED_EXAMPLE),
        (FENCED_EXAMPLE + "\n" + REJECTED, FENCED_EXAMPLE),
    ],
)
def test_nemotron_rejected_route_every_byte_boundary(text, expected):
    expected_result = (expected, [], "stop")
    assert _route_reply(text, stream=False) == expected_result
    for split in range(1, len(text)):
        assert (
            _route_reply(text, stream=True, deltas=[text[:split], text[split:]])
            == expected_result
        ), split
    assert _route_reply(text, stream=True, deltas=list(text)) == expected_result


@pytest.mark.parametrize("stream", [False, True])
def test_forced_choice_uses_original_authorization(stream, caplog):
    text = '<tool_call>{"name":"send_email","arguments":{"body":"hi"}}</tool_call>'
    result = _route_reply(
        text,
        stream=stream,
        expected_status=200 if stream else 422,
        expected_sse_error="invalid_tool_arguments" if stream else None,
        tools=[*TOOLS, {"type": "function", "function": {"name": "send_email"}}],
        tool_choice={"type": "function", "function": {"name": "write_file"}},
    )
    # Authorization agrees; existing forced-choice enforcement still differs:
    # Buffered rejects a wrong choice; streaming rejects the malformed
    # forced-prefix envelope rather than inventing a replacement call.
    if stream:
        assert result["code"] == "invalid_tool_arguments"
    else:
        assert "emitted calls to ['send_email']" in result["detail"]
    assert not any("undeclared" in record.message for record in caplog.records)


@pytest.mark.parametrize("with_valid", [False, True])
def test_nemotron_long_narration_after_rejection(with_valid):
    narration = "x" * 71000
    text = REJECTED + narration + (VALID if with_valid else "")
    expected = (
        narration,
        ["write_file"] if with_valid else [],
        "tool_calls" if with_valid else "stop",
    )
    assert _route_reply(text, stream=False) == expected
    assert (
        _route_reply(
            text,
            stream=True,
            deltas=[text[i : i + 1024] for i in range(0, len(text), 1024)],
        )
        == expected
    )


@pytest.mark.parametrize("deltas", [[REJECTED, VALID], list(REJECTED + VALID)])
def test_nemotron_valid_call_after_rejection_has_index_zero(deltas):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    parser.reset()
    previous = ""
    calls = []
    for delta in deltas:
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous, current, delta, request={"tools": TOOLS}
        )
        if result:
            calls.extend(result.get("tool_calls", []))
        previous = current
    assert len(calls) == 1
    assert calls[0]["index"] == 0
    assert calls[0]["function"]["name"] == "write_file"


def test_nemotron_consumed_content_range_stays_empty():
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    # A range already accounted for must not replay rejected markup or prose.
    wire = REJECTED + " Narration"
    assert (
        parser._visible_content_between(wire, len(wire), len(wire), {"tools": TOOLS})
        == ""
    )


@pytest.mark.parametrize(
    "finish_reason,expected", [("tool_calls", "tool_calls"), ("length", "length")]
)
def test_mixed_stream_finish_reason(finish_reason, expected):
    pp = processor()
    events = pp.process_chunk(
        _make_output(
            "",
            channel="tool_call",
            tool_calls=[
                {"name": "send_email", "arguments": "{}"},
                {"name": "write_file", "arguments": "{}"},
            ],
            finished=True,
            finish_reason=finish_reason,
        )
    )
    assert pp.undeclared_tool_dropped
    assert pp._tool_calls_emitted_to_wire == 1
    assert events[-1].finish_reason == expected


@pytest.mark.parametrize("prefix", [VALID, REJECTED])
@pytest.mark.parametrize(
    "suffix",
    ["<too", "<function=write_file>{", "<tool_call><function=send_email>{", ""],
)
def test_nemotron_flush_releases_only_safe_held_prose(prefix, suffix):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    parser.reset()
    parser.extract_tool_calls_streaming("", prefix, prefix, request={"tools": TOOLS})
    tail = "Visible trailing prose. " + suffix
    wire = prefix + tail
    result = parser.extract_tool_calls_streaming(
        prefix, wire, tail, request={"tools": TOOLS}
    )
    visible = (result or {}).get("content", "") + parser.flush_held_content(wire)
    assert visible == "Visible trailing prose. "
    assert parser._content_upto == len(prefix + "Visible trailing prose. ")
    assert parser.flush_held_content(wire) == ""


@pytest.mark.parametrize("closer", ["</tool_call>", "</function>"])
def test_nemotron_round4_orphan_closer_is_prose(closer):
    text = f"Write {closer} to close the block."
    expected = (text, [], "stop")
    assert _route_reply(text, stream=False) == expected
    assert _route_reply(text, stream=True, deltas=list(text)) == expected


@pytest.mark.parametrize(
    "before,after",
    [("Calling now: ", ""), ("", " Call finished."), ("Before: ", " After.")],
)
def test_nemotron_round4_rejected_wrapper_narration(before, after):
    text = f'<tool_call>{before}<function=send_email>{{"x":1}}</function>{after}</tool_call>'
    expected = ((before + after).strip(), [], "stop")
    assert _route_reply(text, stream=False) == expected
    assert _route_reply(text, stream=True, deltas=list(text)) == expected
    for split in range(1, len(text)):
        assert (
            _route_reply(text, stream=True, deltas=[text[:split], text[split:]])
            == expected
        )


def test_nemotron_round4_unclosed_openers_linear_time():
    from time import perf_counter

    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    text = "<tool_call><function=bad>" * 4000  # 100,000 bytes
    # Warm imports/caches; take the best of three to tolerate scheduler pauses.
    parser.extract_tool_calls("warmup", {"tools": TOOLS})
    times = []
    for _ in range(3):
        start = perf_counter()
        result = parser.extract_tool_calls(text, {"tools": TOOLS})
        times.append(perf_counter() - start)
        assert not result.tools_called
        assert result.content == text
    assert min(times) < 0.2, times


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>example </function></tool_call>",
        '<tool_call>example</tool_call> <function=send_email>{"x":1}</function>',
        '<tool_call><function=send_email>{"x":1}</function>'
        + "x" * 32768
        + "</tool_call>",
    ],
)
def test_nemotron_round4_rejection_requires_bounded_wrapped_function(text):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    result = parser.extract_tool_calls(text, {"tools": TOOLS})
    assert not result.tool_calls
    assert result.content == text
    assert parser._undeclared_dropped == 0


def test_nemotron_rejection_preserves_declared_non_json_arguments():
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    text = (
        REJECTED + "<tool_call><function=write_file>{raw text}</function></tool_call>"
    )
    result = parser.extract_tool_calls(text, {"tools": TOOLS})
    assert result.tools_called
    assert result.content is None
    assert parser._undeclared_dropped == 1
    assert [(call["name"], call["arguments"]) for call in result.tool_calls] == [
        ("write_file", "{raw text}")
    ]


@pytest.mark.parametrize("name", ["send_email", "write_file"])
@pytest.mark.parametrize("byte_stream", [False, True])
def test_nemotron_round5_resolved_wrapper_suffix(name, byte_stream):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    chunks = [
        f'<tool_call><function={name}>{{"path":"x"}}</function></t',
        "able>",
        "</tool_call>",
    ]
    text = "".join(chunks)
    deltas = list(text) if byte_stream else chunks
    expected_names = ["write_file"] if name == "write_file" else []
    expected = ("</table>", expected_names, "tool_calls" if expected_names else "stop")
    assert _route_reply(text, stream=False) == expected
    assert _route_reply(text, stream=True, deltas=deltas) == expected

    parser = NemotronToolParser()
    parser.reset()
    buffered = NemotronToolParser().extract_tool_calls(text, {"tools": TOOLS})
    previous = ""
    content = ""
    calls = []
    dropped = False
    for delta in deltas:
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous, current, delta, request={"tools": TOOLS}
        )
        dropped |= bool(parser._undeclared_dropped)
        if result:
            content += result.get("content", "")
            calls.extend(result.get("tool_calls", []))
        previous = current
    content += parser.flush_held_content(text)
    assert content == buffered.content == "</table>"
    assert "tool_call" not in content and "function" not in content
    assert [call["function"]["name"] for call in calls] == expected_names
    assert dropped == (name == "send_email")


@pytest.mark.parametrize("name", ["send_email", "write_file"])
@pytest.mark.parametrize("suffix", ["</t", " \n", " \n</t"])
def test_nemotron_round5_accounts_function_before_pending_suffix(name, suffix):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    parser.reset()
    function = f'<tool_call><function={name}>{{"path":"x"}}</function>'
    result = parser.extract_tool_calls_streaming(
        "", function + suffix, function + suffix, request={"tools": TOOLS}
    )
    assert parser._content_upto == len(function)
    assert result is not None
    assert not result.get("content")
    assert len(result.get("tool_calls", [])) == (name == "write_file")
    assert bool(parser._undeclared_dropped) == (name == "send_email")


@pytest.mark.parametrize(
    "names",
    [
        ("send_email", "send_email"),
        ("send_email", "write_file"),
        ("send_email", "send_email", "write_file"),
    ],
)
@pytest.mark.parametrize("narration", [False, True, "formatting"])
def test_nemotron_round6_wrapper_siblings(names, narration):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    prefix, gap, suffix = (
        ("Before. ", " Between. ", " After.") if narration else ("", "", "")
    )
    if narration == "formatting":
        gap = "\n  "
    functions = [
        f'<function={name}>{{"x":{i}}}</function>' for i, name in enumerate(names, 1)
    ]
    text = "<tool_call>" + prefix + gap.join(functions) + suffix + "</tool_call>"
    content = (
        prefix + ("" if gap.isspace() else gap * (len(names) - 1)) + suffix
    ).strip()
    valid = [name for name in names if name == "write_file"]
    expected = (
        content or ("" if valid else NOTICE),
        valid,
        "tool_calls" if valid else "stop",
    )
    assert _route_reply(text, stream=False) == expected
    for deltas in (
        [text],
        [text[i : i + 7] for i in range(0, len(text), 7)],
        list(text),
    ):
        assert _route_reply(text, stream=True, deltas=deltas) == expected
        parser = NemotronToolParser()
        parser.reset()
        previous = ""
        visible = ""
        calls = []
        for delta in deltas:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous, current, delta, request={"tools": TOOLS}
            )
            if result:
                visible += result.get("content", "")
                calls.extend(result.get("tool_calls", []))
            previous = current
        visible += parser.flush_held_content(text)
        assert visible.strip() == content
        assert [call["function"]["name"] for call in calls] == valid
        assert parser._undeclared_dropped == names.count("send_email")
    buffered = NemotronToolParser()
    buffered.extract_tool_calls(text, {"tools": TOOLS})
    assert buffered._undeclared_dropped == names.count("send_email")


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_authorization_round6_mixed_index_resume(cap, batch):
    pp = processor()
    pp.request["parallel_tool_calls"] = not cap
    good = anchor("write_file", 0)
    good["function"]["arguments"] = '{"path":"'
    fragments = [
        good,
        anchor("send_email", 1),
        {"function": {"arguments": "rejected"}},
        {"index": 1, "function": {"arguments": "rejected"}},
        {"index": 0, "function": {"arguments": "x"}},
        {"function": {"arguments": '"}'}},
    ]
    pp.tool_parser = Mock()
    pp.tool_parser.EXPECTED_WIRE_FORMATS = ("tool_call_json",)
    pp.tool_parser._undeclared_dropped = 0
    pp.tool_parser.extract_tool_calls_streaming.side_effect = (
        [{"tool_calls": fragments}]
        if batch
        else [{"tool_calls": [call]} for call in fragments]
    )
    calls = []
    for _ in range(1 if batch else len(fragments)):
        calls.extend(pp._detect_tool_calls("<tool_call>").get("tool_calls", []))
    assert (
        "".join(call["function"].get("arguments", "") for call in calls)
        == '{"path":"x"}'
    )
    assert pp.undeclared_tool_dropped


def test_nemotron_round6_literal_wrapper_close_inside_rejection():
    text = '<tool_call><function=send_email>{"x":"</tool_call>"}</function></tool_call>'
    expected = (NOTICE, [], "stop")
    assert _route_reply(text, stream=False) == expected


def test_authorization_round6_rejected_index_resumes_suppression():
    pp = processor()
    pp._filter_declared_calls(
        [anchor("write_file", 0), anchor("send_email", 1)], incremental=True
    )
    kept = pp._filter_declared_calls(
        [
            {"index": 0, "function": {"arguments": "good"}},
            {"index": 1, "function": {"arguments": "bad"}},
            {"function": {"arguments": "also bad"}},
            {"index": 0, "function": {"arguments": "good again"}},
            {"function": {"arguments": "good tail"}},
        ],
        incremental=True,
    )
    assert [call["function"]["arguments"] for call in kept] == [
        "good",
        "good again",
        "good tail",
    ]


@pytest.mark.parametrize("name", ["send_email", "write_file"])
def test_nemotron_round5_consumes_resolved_wrapper_suffix(name):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    parser = NemotronToolParser()
    parser.reset()
    chunks = [
        f'<tool_call><function={name}>{{"path":"x"}}</function> \n</t',
        "ool_call>",
    ]
    previous = ""
    content = ""
    calls = []
    for delta in chunks:
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous, current, delta, request={"tools": TOOLS}
        )
        assert result is not None
        content += result.get("content", "")
        calls.extend(result.get("tool_calls", []))
        previous = current
    content += parser.flush_held_content(previous)
    assert content == ""
    assert parser._content_upto == len(previous)
    assert len(calls) == (name == "write_file")
    assert bool(parser._undeclared_dropped) == (name == "send_email")


@pytest.mark.parametrize("separate_wrappers", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_nemotron_rejected_sibling_before_declared_has_no_markup(
    separate_wrappers, stream
):
    rejected = '<function=send_email>{"x":1}</function>'
    declared = '<function=write_file>{"path":"x"}</function>'
    if separate_wrappers:
        text = f"<tool_call>{rejected}</tool_call><tool_call>{declared}</tool_call>"
    else:
        text = f"<tool_call>{rejected}\n  {declared}</tool_call>"
    expected = ("", ["write_file"], "tool_calls")
    assert _route_reply(text, stream=stream, deltas=[text]) == expected
    if stream:
        assert _route_reply(text, stream=True, deltas=list(text)) == expected


@pytest.mark.parametrize("separate_chunks", [False, True])
def test_channel_complete_router_calls_filter_before_assigning_wire_indices(
    separate_chunks,
):
    pp = processor()
    # Routers emit complete flat calls at call boundaries, without source indices.
    router_calls = [
        {"name": "send_email", "arguments": "{}"},
        {"name": "write_file", "arguments": '{"path":"x"}'},
    ]
    batches = [[call] for call in router_calls] if separate_chunks else [router_calls]
    events = []
    for batch in batches:
        events.extend(
            pp.process_chunk(_make_output(channel="tool_call", tool_calls=batch))
        )
    events.extend(
        pp.process_chunk(
            _make_output(channel="tool_call", finished=True, finish_reason="tool_calls")
        )
    )
    calls = [tc for event in events for tc in (event.tool_calls or [])]
    assert len(calls) == 1
    assert calls[0]["index"] == 0
    assert calls[0]["function"] == {
        "name": "write_file",
        "arguments": '{"path":"x"}',
    }
    assert events[-1].finish_reason == "tool_calls"
    assert pp.undeclared_tool_dropped


@pytest.mark.parametrize("text", ["", "<tool_call><function=send_email>{"])
def test_nemotron_incomplete_call_has_no_safe_accounting_boundary(text):
    from rapid_mlx.tool_parsers.nemotron_tool_parser import NemotronToolParser

    assert NemotronToolParser()._safe_accounting_boundary(text) == 0


def test_rejected_channel_call_preserves_held_reasoning_on_finalize():
    from rapid_mlx.reasoning.k2_horizon_parser import K2HorizonReasoningParser

    pp = StreamingPostProcessor(
        _make_cfg(reasoning_parser=K2HorizonReasoningParser()),
        request={"tools": TOOLS},
        enable_thinking=True,
    )
    text = "plan</ifm|thi"
    events = pp.process_chunk(_make_output(text))
    events.extend(
        pp.process_chunk(
            _make_output(channel="tool_call", tool_calls=[anchor("send_email")])
        )
    )
    events.extend(pp.finalize())
    assert "".join(event.reasoning or "" for event in events) == text
    assert all(not event.content and not event.tool_calls for event in events)
    assert pp.undeclared_tool_dropped
