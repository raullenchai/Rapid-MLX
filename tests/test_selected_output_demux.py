"""Selected grammars preserve ownership across every split and reject bad calls."""

import json
from types import SimpleNamespace

import pytest

from rapid_mlx.parsing.output_demux import (
    EventReducer,
    OutputDemultiplexer,
    ParseContext,
)
from rapid_mlx.parsing.selected_grammar import SelectedGrammar, ToolPolicy, terminals
from rapid_mlx.parsing.session import make_session, parse_buffered
from rapid_mlx.prompt_parse_context import PromptParseContext


def run(raw, parts, *, format_name="hermes", channel="content", **limits):
    policy = ToolPolicy(
        format_name, {"f": {"properties": {"x": {"type": "string"}}}}, True
    )
    context = ParseContext(
        default_channel=channel,
        authorized_tools=frozenset({"f"}),
        terminals=terminals(policy),
        **limits,
    )
    parser = OutputDemultiplexer(context, SelectedGrammar(context, policy))
    events = []
    for part in parts:
        events.extend(parser.feed(part))
    events.extend(parser.finish())
    assert parser.finish() == []
    assert parser.commit_cursor == len(raw)
    reducer = EventReducer()
    reducer.feed(events)
    return (
        "".join(reducer.content),
        "".join(reducer.reasoning),
        reducer.calls,
        reducer.failures,
    )


@pytest.mark.parametrize(
    "format_name,raw,content,reasoning,args",
    [
        (
            "hermes",
            '<think>secret<tool_call>{"name":"f","arguments":{"x":"</tool_call>"}}</tool_call></think>Answer',
            "Answer",
            "secret",
            {"x": "</tool_call>"},
        ),
        (
            "hermes",
            "<tool_call><function=f><parameter=x>a</tool_call>b</parameter></function></tool_call>Answer",
            "Answer",
            "",
            {"x": "a</tool_call>b"},
        ),
        (
            "glm47",
            "<tool_call>f<arg_key>x</arg_key><arg_value>a</tool_call>b</arg_value></tool_call>Answer",
            "Answer",
            "",
            {"x": "a</tool_call>b"},
        ),
        (
            "hermes",
            '<tool_call>{"arguments":{"x":"v"},"name":"f"}</tool_call>Answer',
            "Answer",
            "",
            {"x": "v"},
        ),
        (
            "hermes",
            '<tool_call>{"function":{"name":"f","arguments":{"x":"v"}}}</tool_call>Answer',
            "Answer",
            "",
            {"x": "v"},
        ),
        (
            "seed_oss",
            "<seed:tool_call><function=f><parameter=x>v</parameter></function></seed:tool_call>Answer",
            "Answer",
            "",
            {"x": "v"},
        ),
    ],
)
def test_valid_call_every_split(format_name, raw, content, reasoning, args):
    for parts in ([raw], list(raw), *([raw[:i], raw[i:]] for i in range(len(raw) + 1))):
        actual, private, calls, failures = run(raw, parts, format_name=format_name)
        assert (actual, private) == (content, reasoning)
        assert failures == []
        assert len(calls) == 1
        assert calls[0]["name"] == "f"
        assert json.loads(calls[0]["arguments"]) == args


@pytest.mark.parametrize(
    "body",
    [
        '{"name":"f","arguments":{"x":1},"arguments":{"x":2}}',
        '{"name":"f","arguments":{"x":1},"function":{"name":"f","arguments":{"x":2}}}',
        '{"name":"f","arguments":{},"parameters":{"x":2}}',
        '{"name":"f"}',
        '{"name":"other","arguments":{}}',
        '{"name":"f","arguments":{"x":NaN}}',
    ],
)
def test_invalid_envelope_never_completes(body):
    raw = "<tool_call>" + body + "</tool_call>Answer"
    for parts in ([raw], list(raw)):
        content, reasoning, calls, failures = run(raw, parts)
        assert calls == []
        assert failures
        assert content.endswith("Answer")
        assert reasoning == ""


@pytest.mark.parametrize(
    "raw",
    [
        "Example <tool_call>foo",
        '```<tool_call>{"name":"f","arguments":{}}</tool_call>```',
    ],
)
def test_literal_policy_keeps_source(raw):
    for parts in ([raw], list(raw)):
        content, reasoning, calls, _ = run(raw, parts, format_name="glm47")
        assert (content, reasoning, calls) == (raw, "", [])


def config():
    return SimpleNamespace(
        reasoning_parser_name="qwen3",
        reasoning_parser=None,
        tool_call_parser="hermes",
        enable_auto_tool_choice=True,
    )


def test_empty_chunk_does_not_claim_native_ownership():
    session = make_session(config(), {})
    assert session.feed(SimpleNamespace(new_text="")) == []
    assert session.feed(SimpleNamespace(new_text="answer", channel="content")) is None
    assert session.finish() == []


def test_buffered_namespace_and_prompt_state_match_stream():
    prompt = PromptParseContext("", (), "content", False, "chatml")
    output = SimpleNamespace(
        text="Hello world.", new_text="", prompt_parse_context=prompt
    )
    parsed = parse_buffered(config(), output, {})
    assert parsed.content == output.text
    assert parsed.reasoning == ""
    session = make_session(config(), {})
    session.feed(SimpleNamespace(new_text="", prompt_parse_context=prompt))
    events = session.feed(SimpleNamespace(new_text=output.text))
    assert "".join(event.content or "" for event in events) == output.text
    assert session.finish() == session.finish() == []
    assert session.result.content == parsed.content


def test_literal_parameter_closer_does_not_truncate_call():
    raw = "<function=f><parameter=x>before </parameter> literal </function> after</parameter></function>"
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts)
        assert content == ""
        assert failures == []
        assert json.loads(calls[0]["arguments"]) == {
            "x": "before </parameter> literal </function> after"
        }


def test_header_limit_is_partition_independent():
    raw = (
        '<tool_call>{"name":"f","padding":"'
        + "x" * 40
        + '","arguments":"{}"}</tool_call>'
    )
    results = []
    for parts in ([raw], list(raw)):
        content, reasoning, calls, failures = run(raw, parts, max_header_chars=20)
        results.append(
            (content, reasoning, calls, [failure.code for failure in failures])
        )
    assert results[0] == results[1]
    assert results[0][2] == []
    assert "header_limit" in results[0][3]


def test_deep_json_is_failed_without_executable_call():
    raw = (
        '<tool_call>{"name":"f","arguments":{"x":'
        + "[" * 1100
        + "0"
        + "]" * 1100
        + "}}</tool_call>"
    )
    for parts in ([raw], list(raw)):
        _, _, calls, failures = run(raw, parts)
        assert calls == []
        assert failures[-1].code == "invalid_tool"


def tool_session(**options):
    request = {
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in ("f", "g")
        ],
        **options,
    }
    return make_session(config(), request)


def output(text="", **kwargs):
    return SimpleNamespace(new_text=text, **kwargs)


@pytest.mark.parametrize("prefix", ["answer <", "answer <tool_call>unfinished"])
def test_native_metadata_drains_pending_source_and_preserves_id(prefix):
    session = tool_session()
    session.feed(
        output(
            prefix,
            prompt_parse_context=PromptParseContext("", (), "content", False, "chatml"),
        )
    )
    arguments = '{ "x" : "value" }'
    events = session.feed(
        output(
            "", tool_calls=[{"id": "native-call", "name": "f", "arguments": arguments}]
        )
    )
    assert [
        call["id"]
        for event in events
        for call in event.tool_calls or []
        if "id" in call
    ] == ["native-call"]
    session.feed(output(" next"))
    session.finish("length")
    assert session.result.content == prefix + " next"
    assert session.result.calls == [
        {"id": "native-call", "name": "f", "arguments": arguments}
    ]
    assert session.reducer.finish_reason == "length"


def test_metadata_cannot_complete_an_incomplete_text_call():
    session = tool_session()
    session.feed(output('<tool_call>{"name":"f","arguments":{"x":'))
    events = session.feed(output(tool_calls=[{"name": "g", "arguments": "{}"}]))
    assert any(event.type == "parse_failure" for event in events)
    assert not any(
        call.get("function", {}).get("name") == "g"
        for event in events
        for call in event.tool_calls or []
    )
    assert session.result.calls == []
    with pytest.raises(ValueError, match="failed"):
        session.feed(output("later"))


def test_text_and_metadata_share_indices_without_collapsing_equal_calls():
    session = tool_session(parallel_tool_calls=True)
    session.feed(output('<tool_call>{"name":"f","arguments":{}}</tool_call>'))
    session.feed(
        output(
            tool_calls=[
                {"id": "native-1", "name": "f", "arguments": "{}"},
                {"id": "native-2", "name": "f", "arguments": "{}"},
            ]
        )
    )
    session.finish()
    assert len(session.result.calls) == 3
    assert [call["id"] for call in session.result.calls[1:]] == ["native-1", "native-2"]
    assert list(session._public_indices) == [0, 1, 2]


def test_pinned_name_and_single_call_policy_apply_before_publication():
    from rapid_mlx.parsing.session import OutputParseError, protocol_calls

    session = tool_session(tool_choice={"type": "function", "function": {"name": "f"}})
    wrong = '<tool_call>{"name":"g","arguments":{}}</tool_call>'
    assert not any(event.tool_calls for event in session.feed(output(wrong)))
    session.finish()
    with pytest.raises(OutputParseError):
        protocol_calls(session.result)
    session = tool_session(tool_choice={"type": "function", "function": {"name": "f"}})
    right = wrong.replace('"g"', '"f"')
    events = session.feed(output(right + right))
    session.finish()
    assert (
        len(
            [
                call
                for event in events
                for call in event.tool_calls or []
                if "id" in call
            ]
        )
        == 1
    )
    assert len(session.result.calls) == 1


@pytest.mark.parametrize(
    "format_name", ["qwen", "qwen3", "qwen3_xml", "auto", "generic"]
)
@pytest.mark.parametrize("space", ["", " ", "\n"])
def test_bracket_calls_are_owned_and_string_closers_are_literal(format_name, space):
    raw = "[Calling tool:" + space + 'f({"x": ")]"})]Answer'
    for parts in ([raw], list(raw), [raw[:20], raw[20:]]):
        content, _, calls, failures = run(raw, parts, format_name=format_name)
        assert content == "Answer"
        assert not failures
        assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [
            ("f", {"x": ")]"})
        ]


@pytest.mark.parametrize("format_name", ["nemotron", "nemotron3"])
def test_optional_outer_wrapper_finishes_on_function_end(format_name):
    raw = "<tool_call><function=f><parameter=x>value</parameter></function>"
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts, format_name=format_name)
        assert content == ""
        assert not failures
        assert json.loads(calls[0]["arguments"]) == {"x": "value"}


def test_routed_prose_establishes_literal_context_without_relexing():
    session = tool_session()
    session.feed(
        output(
            " ",
            prompt_parse_context=PromptParseContext("", (), "content", False, "chatml"),
        )
    )
    session.feed(output("Example: ", channel="content"))
    raw = '{"name":"f","arguments":{}}'
    session.feed(output(raw))
    session.finish()
    assert session.result.content == " Example: " + raw
    assert session.result.calls == []


def test_unauthorized_outer_call_does_not_execute_nested_argument_markup():
    raw = '<tool_call>{"name":"unknown","arguments":{"x":"<tool_call>{\\"name\\":\\"f\\",\\"arguments\\":{}}</tool_call>"}}</tool_call>'
    for parts in ([raw], list(raw)):
        content, _, calls, _ = run(raw, parts)
        assert content == raw
        assert calls == []


@pytest.mark.parametrize(
    "body,fmt",
    [
        ("<function=f></parameter></function>", "hermes"),
        ("<function=f>junk<parameter=x>1</parameter></function>", "hermes"),
        ("<function=f><parameter=x>1</function>", "hermes"),
        ("<function=f>junk</function>", "hermes"),
        (
            "<function=f><parameter=x>1</parameter><parameter=z>2</parameter></function>",
            "hermes",
        ),
        ("<atem:invoke bad></atem:invoke>", "muse"),
        ('{"arguments":{}}', "hermes"),
        ("<arg_key>x</arg_key>", "glm47"),
        ("f junk", "glm47"),
        ("f<arg_key>x", "glm47"),
        ("f<arg_key>x</arg_key>bad", "glm47"),
        ("f<arg_key>x</arg_key><arg_value>bad", "glm47"),
        ("unsupported", "hermes"),
    ],
)
def test_complete_body_validation_rejects_ambiguous_or_incomplete_arguments(body, fmt):
    from rapid_mlx.parsing.tool_values import decode_body

    with pytest.raises(ValueError):
        decode_body(body, fmt, {"f": {"properties": {"x": {"type": "string"}}}})


@pytest.mark.parametrize(
    "body,fmt,expected",
    [
        ('{"name":"f","arguments":"{\\"x\\":1}"}', "hermes", {"x": 1}),
        (
            '<function name="f"><parameter name="x">1</parameter><parameter name="y">true</parameter></function>',
            "hermes",
            {"x": 1, "y": True},
        ),
        (
            "<function=f><parameter=x>a<parameter=literal>b</parameter></function>",
            "hermes",
            {"x": "a<parameter=literal>b"},
        ),
        ('<function=f>{"x":"</function>"}</function>', "hermes", {"x": "</function>"}),
        (
            '<function><name>f</name><arguments>{"x":1}</arguments></function>',
            "hermes",
            {"x": 1},
        ),
        (
            "f </think> <arg_key>x</arg_key> <arg_value>a</arg_value> literal </arg_value> \n <arg_key>y</arg_key><arg_value>2</arg_value>",
            "glm47",
            {"x": "a</arg_value> literal", "y": 2},
        ),
        (
            '<atem:invoke name="f"><atem:parameter name="x">1</atem:parameter></atem:invoke>',
            "muse",
            {"x": "1"},
        ),
    ],
)
def test_complete_body_conversion_preserves_values(body, fmt, expected):
    from rapid_mlx.parsing.tool_values import decode_body

    assert decode_body(body, fmt, {}) == ("f", expected)


@pytest.mark.parametrize(
    "fmt,raw",
    [
        (
            "muse",
            '<atem:function_calls><atem:invoke name="f"><atem:parameter name="x">v</atem:parameter></atem:invoke></atem:function_calls>',
        ),
        ("hermes", '<function=f>{"x":"</function>"}</function>'),
        (
            "qwen",
            "<tool_call><function=f><parameter=x>v</parameter></function> \n </tool_call>Answer",
        ),
    ],
)
def test_additional_native_shapes_are_partition_invariant(fmt, raw):
    whole = run(raw, [raw], format_name=fmt)
    assert not whole[3]
    assert len(whole[2]) == 1
    assert run(raw, list(raw), format_name=fmt) == whole


@pytest.mark.parametrize(
    "raw",
    [
        '<tool_call><function=unknown>{"x":"</tool_call>"}</function></tool_call>',
        '<tool_call>{"arguments":{},"name":"unknown"}</tool_call>',
        '{"name":"unknown","arguments":{}}',
        "<seed:tool_call>bad</seed:tool_call>",
        "[Calling tool: invalid)]",
    ],
)
def test_invalid_uncommitted_candidates_remain_literal(raw):
    fmt = "seed_oss" if raw.startswith("<seed:") else "qwen"
    for parts in ([raw], list(raw)):
        content, _, calls, _ = run(raw, parts, format_name=fmt)
        assert content == raw
        assert not calls


def test_argument_limits_fail_committed_calls_and_preserve_uncommitted_text():
    raw = '<tool_call>{"name":"f","arguments":{"x":"long value"}}</tool_call>Answer'
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts, max_argument_chars=30)
        assert not calls
        assert any(f.index is not None and f.code == "argument_limit" for f in failures)
        assert content == "Answer"
    raw = '<tool_call>{"arguments":{"x":"long value"},"name":"f"}</tool_call>'
    content, _, calls, failures = run(raw, list(raw), max_argument_chars=20)
    assert content == raw
    assert not calls
    assert failures[0].code == "argument_limit"


def test_metadata_rejects_invalid_ingress_and_failed_adapter_is_terminal():
    from rapid_mlx.parsing.output_demux import ChannelGrammar, ContentDelta, Span

    parser = OutputDemultiplexer(ParseContext())
    with pytest.raises(ValueError):
        parser.accept_metadata([])
    policy = ToolPolicy("hermes", {}, True)
    ctx = ParseContext(authorized_tools=frozenset({"f"}), terminals=terminals(policy))

    def fresh():
        return OutputDemultiplexer(ctx, SelectedGrammar(ctx, policy))

    for kwargs in ({"text": "x"}, {"channel": "invalid"}):
        with pytest.raises(ValueError):
            fresh().accept_metadata([], **kwargs)
    for terminal in ("finish", "native"):
        p = fresh()
        if terminal == "finish":
            p.finish()
        else:
            p.accept_native([ContentDelta(Span(0, 1), "x")], end_offset=1)
        with pytest.raises(ValueError):
            p.accept_metadata([])

    class BrokenBoundary(ChannelGrammar):
        def consume(self, lexeme):
            return []

        def boundary(self, offset):
            return []

        def metadata(self, calls, offset, **kwargs):
            return []

    p = OutputDemultiplexer(ParseContext(), BrokenBoundary(ParseContext()))
    p.feed("lost")
    with pytest.raises(ValueError, match="unowned"):
        p.accept_metadata([])
    with pytest.raises(ValueError):
        p.feed("retry")
    p = fresh()
    with pytest.raises(ValueError):
        p.accept_metadata([{"name": "f", "arguments": "bad"}])
    with pytest.raises(ValueError):
        p.accept_metadata([])


def test_session_metadata_validation_and_private_budget():
    from rapid_mlx.parsing.session import OutputParseError

    session = make_session(config(), {}, reasoning_max_tokens=1)
    events = session.feed(output("secret private text"))
    assert "".join(e.reasoning or "" for e in events) == "secr"
    session.feed(output("more private"))
    session.finish()
    assert session.result.reasoning == "secr"
    assert session.result.content == ""
    with pytest.raises(ValueError):
        session.feed(output("late"))
    empty = make_session(config(), {})
    assert empty.finish() == []
    assert empty.result.content == ""
    assert parse_buffered(config(), SimpleNamespace(channel="content"), {}) is None
    for kwargs in (
        {"channel": "unknown"},
        {"tool_calls": [{"name": "f", "arguments": []}]},
    ):
        session = tool_session()
        session.feed(output(" "))
        with pytest.raises(OutputParseError):
            session.feed(output("", **kwargs))
    session = tool_session()
    session.feed(output(" "))
    session.feed(
        output(
            "",
            tool_calls=[
                {"name": "unknown", "arguments": {}},
                {"name": "f", "arguments": {}},
            ],
        )
    )
    session.finish()
    assert [c["name"] for c in session.result.calls] == ["f"]


@pytest.mark.parametrize(
    "raw,fmt",
    [
        ('<tool_call>{"na\\qme":"f","arguments":{}}</tool_call>', "hermes"),
        ('<tool_call>{"name":"f","arguments":{}}}</tool_call>', "hermes"),
        ("[Calling tool:unknown({})]", "qwen"),
        (
            "<tool_call>unknown <arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            "glm47",
        ),
        ('<tool_call>{"name":"f","arguments":[]}</tool_call>', "hermes"),
        ("<tool_call>" + "a" * 1030, "hermes"),
    ],
)
def test_rejected_shapes_never_complete_calls(raw, fmt):
    for parts in ([raw], list(raw)):
        _, _, calls, failures = run(raw, parts, format_name=fmt)
        assert not calls
        assert failures


@pytest.mark.parametrize("suffix", ["Answer", "  Answer"])
def test_optional_wrapper_followed_by_answer(suffix):
    raw = "<tool_call><function=f></function>" + suffix
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts, format_name="qwen")
        assert content == suffix
        assert len(calls) == 1
        assert not failures


def test_json_scanner_rejects_invalid_initial_tokens():
    from rapid_mlx.parsing.selected_grammar import JsonEnvelope

    scanner = JsonEnvelope()
    for i, char in enumerate(" x"):
        scanner.feed(char, i)
    assert scanner.invalid


def test_converter_cannot_change_a_committed_name(monkeypatch):
    import rapid_mlx.parsing.selected_grammar as grammar

    monkeypatch.setattr(grammar, "decode_body", lambda *args: ("g", {}))
    policy = ToolPolicy("hermes", {}, True)
    ctx = ParseContext(
        authorized_tools=frozenset({"f", "g"}), terminals=terminals(policy)
    )
    p = OutputDemultiplexer(ctx, SelectedGrammar(ctx, policy))
    reducer = EventReducer()
    reducer.feed(p.feed('<tool_call>{"name":"f","arguments":{}}</tool_call>'))
    reducer.feed(p.finish())
    assert not reducer.calls
    assert reducer.failures[0].code == "invalid_tool"


@pytest.mark.parametrize(
    "text,calls", [("hello", []), ("", [{"name": "f", "arguments": {}}])]
)
def test_metadata_admission_cannot_switch_to_native_coordinates(text, calls):
    from rapid_mlx.parsing.output_demux import ContentDelta, Span

    policy = ToolPolicy("hermes", {}, True)
    context = ParseContext(
        authorized_tools=frozenset({"f"}), terminals=terminals(policy)
    )
    p = OutputDemultiplexer(context, SelectedGrammar(context, policy))
    assert p.accept_metadata([]) == []
    p.accept_metadata(calls, text=text, channel="content")
    with pytest.raises(ValueError):
        p.accept_native(
            [ContentDelta(Span(len(text), len(text) + 1), "!")],
            end_offset=len(text) + 1,
        )


def test_reasoning_and_structured_call_in_one_engine_event_keep_both_lanes():
    session = tool_session()
    session.feed(output("private "))
    events = session.feed(
        output(
            "continued",
            channel="reasoning",
            tool_calls=[{"name": "f", "arguments": {}}],
        )
    )
    session.finish()
    assert session.result.reasoning == "private continued"
    assert session.result.content == ""
    assert [c["name"] for c in session.result.calls] == ["f"]
    assert [e.type for e in events] == ["reasoning", "tool_call", "tool_call"]


def test_unauthorized_engine_metadata_is_an_explicit_failure():
    from rapid_mlx.parsing.session import OutputParseError, protocol_calls

    session = tool_session()
    session.feed(output(" "))
    events = session.feed(output("", tool_calls=[{"name": "unknown", "arguments": {}}]))
    session.finish()
    assert any(e.type == "parse_failure" for e in events)
    with pytest.raises(OutputParseError):
        protocol_calls(session.result)


def test_terminal_engine_chunk_drains_before_finish_event():
    from rapid_mlx.config.server_config import ServerConfig
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    cfg = ServerConfig(reasoning_parser_name="qwen3", tool_call_parser="hermes")
    processor = StreamingPostProcessor(cfg, tools_requested=False)
    events = processor.process_chunk(
        GenerationOutput(
            text="literal <",
            new_text="literal <",
            finished=True,
            prompt_parse_context=PromptParseContext("", (), "content", False, "chatml"),
        )
    )
    assert events[-1].type == "finish"
    assert "".join(e.content or "" for e in events[:-1]) == "literal <"
    assert processor.finalize() == []


@pytest.mark.parametrize(
    "fmt,raw",
    [
        (
            "hermes",
            "<tool_call><function=f><parameter=x>first</parameter><parameter=x>second</parameter></function></tool_call>",
        ),
        (
            "glm47",
            "<tool_call>f<arg_key>x</arg_key><arg_value>first</arg_value><arg_key>x</arg_key><arg_value>second</arg_value></tool_call>",
        ),
        (
            "muse",
            '<atem:invoke name="f"><atem:parameter name="x">first</atem:parameter><atem:parameter name="x">second</atem:parameter></atem:invoke>',
        ),
    ],
)
def test_duplicate_native_argument_names_fail_instead_of_overwriting(fmt, raw):
    for parts in ([raw], list(raw)):
        _, _, calls, failures = run(raw, parts, format_name=fmt)
        assert calls == []
        assert any(f.index is not None and f.code == "invalid_tool" for f in failures)


def test_selected_content_holdback_properties_track_only_pending_marker():
    from rapid_mlx.config.server_config import ServerConfig
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    p = StreamingPostProcessor(ServerConfig(reasoning_parser_name="qwen3"))
    assert not p.content_held
    assert p.content_consumed == 0
    p.process_chunk(GenerationOutput(text="<thi", new_text="<thi", finished=False))
    assert p.content_held
    p.process_chunk(GenerationOutput(text="<thix", new_text="x", finished=False))
    assert not p.content_held
    assert p.content_consumed == 5


@pytest.mark.parametrize(
    "fmt,raw",
    [
        ("glm47", "<tool_call>f<arg_key>x</arg_key><arg_value>value</arg_value>"),
        ("hermes", '<tool_call>{"name":"f","arguments":{"x":"value"}}'),
        ("hermes", "<tool_call><function=f><parameter=x>value</parameter></function>"),
    ],
)
@pytest.mark.parametrize("partial", ["", "<", "</tool_", "</tool_call"])
def test_interrupted_outer_envelope_can_finish_only_validated_complete_arguments(
    fmt, raw, partial
):
    raw += partial
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts, format_name=fmt)
        assert not content and not failures
        assert json.loads(calls[0]["arguments"]) == {"x": "value"}


@pytest.mark.parametrize("fence", ["```", "~~~", "````", "~~~~"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_fenced_calls_are_literal_and_next_outside_call_is_owned(fence, newline):
    call = '<tool_call>{"name":"f","arguments":{"x":"value"}}</tool_call>'
    example = fence + "json" + newline + call + newline + fence + newline
    raw = example + call
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts)
        assert content == example
        assert len(calls) == 1 and not failures


@pytest.mark.parametrize("ticks", ["`", "``"])
def test_inline_code_call_is_literal(ticks):
    call = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
    raw = ticks + call + ticks + " text"
    for parts in ([raw], list(raw)):
        content, _, calls, failures = run(raw, parts)
        assert content == raw
        assert not calls and not failures


@pytest.mark.parametrize("slashes", [1, 3])
def test_escaped_backtick_does_not_open_inline_code(slashes):
    raw = "\\" * slashes + '` then `<tool_call>{"name":"f","arguments":{}}</tool_call>`'
    for split in range(len(raw) + 1):
        content, reasoning, calls, _ = run(raw, [raw[:split], raw[split:]])
        assert content == raw
        assert not reasoning and not calls


@pytest.mark.parametrize(
    "body",
    [
        "I should use <tool_call><function=f> to look this up.",
        "A literal <tool_call>" + " " * 1100,
    ],
)
def test_uncommitted_reasoning_example_releases_at_channel_boundary(body):
    raw = "<think>" + body + "</think>Answer"
    for parts in ([raw], list(raw)):
        content, reasoning, calls, _ = run(raw, parts)
        assert content == "Answer" and reasoning == body and not calls


@pytest.mark.parametrize(
    "body",
    [
        "<function=f></function>",
        "<function=f><parameter=x>value</parameter></function>",
    ],
)
def test_private_xml_requires_parameter_grammar_or_complete_empty_call(body):
    raw = "<think><tool_call>" + body + "</tool_call></think>Answer"
    for parts in ([raw], list(raw)):
        content, reasoning, calls, failures = run(raw, parts)
        assert content == "Answer" and not reasoning and not failures
        assert len(calls) == 1


@pytest.mark.parametrize("marker", ["<think>"])
def test_unquoted_reasoning_marker_retains_private_ownership_after_prose(marker):
    raw = "Use the " + marker + " tag in the example."
    for split in range(len(raw) + 1):
        content, reasoning, calls, failures = run(raw, [raw[:split], raw[split:]])
        assert content == "Use the " and reasoning == " tag in the example."
        assert not calls and not failures


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            "other<arg_key>x</arg_key><arg_value>private</arg_value>",
            "<arg_key>x</arg_key><arg_value>private</arg_value>",
        ),
        ("other", ""),
        ("other prose", None),
        ("<function=other></function>", None),
    ],
)
def test_completed_glm_rejection_diagnostic_is_non_executable(body, expected):
    raw = "<tool_call>" + body + "</tool_call>"
    for parts in ([raw], list(raw)):
        content, reasoning, calls, failures = run(raw, parts, format_name="glm47")
        assert content == raw and not reasoning and not calls
        diagnostics = [failure for failure in failures if failure.tool_name is not None]
        assert [(event.tool_name, event.raw_arguments) for event in diagnostics] == (
            [] if expected is None else [("other", expected)]
        )


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("value", ["</think>secret", "x" * 2000])
def test_private_xml_object_arguments_own_marker_strings_and_large_values(outer, value):
    body = "<function=f>" + json.dumps({"x": value}) + "</function>"
    raw = "<tool_call>" + body + "</tool_call>" if outer else body
    for parts in ([raw], list(raw)):
        content, reasoning, calls, failures = run(raw, parts, channel="reasoning")
        assert not content and not reasoning and not failures
        assert len(calls) == 1 and json.loads(calls[0]["arguments"]) == {"x": value}
