# SPDX-License-Identifier: Apache-2.0
"""Release-format fixtures, including the vendor's non-Qwen edge semantics."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from rapid_mlx.reasoning import get_parser
from rapid_mlx.reasoning.raon_1_parser import RaonReasoningParser
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.tool_parsers.raon_1_tool_parser import RaonToolParser

PROPERTIES = {
    "text": {"type": "string"},
    "count": {"type": "integer"},
    "enabled": {"type": "boolean"},
    "ratio": {"type": "number"},
    "items": {"type": "array", "items": {"type": "integer"}},
    "options": {"type": "object", "properties": {"n": {"type": "integer"}}},
    "maybe": {"type": ["null", "integer"]},
    "choice": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
}
REQUEST = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "run",
                "parameters": {"type": "object", "properties": PROPERTIES},
            },
        }
    ]
}
VALUES = {
    "text": "first line\n  second line\n",
    "count": "42",
    "enabled": "1",
    "ratio": "2.5",
    "items": '["1", 2]',
    "options": '{"n":"3"}',
    "maybe": "null",
    "choice": "7",
}
EXPECTED = {
    "text": VALUES["text"],
    "count": 42,
    "enabled": True,
    "ratio": 2.5,
    "items": ["1", 2],
    "options": {"n": "3"},
    "maybe": None,
    "choice": 7,
}


def call(values=VALUES, name="run"):
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in values.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def stream_tools(text, chunks, request=REQUEST):
    parser = RaonToolParser()
    previous = ""
    calls = {}
    content = ""
    for part in chunks:
        delta = parser.extract_tool_calls_streaming(
            previous, previous + part, part, request=request
        )
        previous += part
        if delta:
            content += delta.get("content", "")
            for item in delta.get("tool_calls", []):
                current = calls.setdefault(item["index"], {"name": "", "arguments": ""})
                fn = item.get("function", {})
                current["name"] += fn.get("name", "")
                current["arguments"] += fn.get("arguments", "")
    assert previous == text
    content += parser.flush_held_content(text)
    return content, [(v["name"], json.loads(v["arguments"])) for v in calls.values()]


def test_registered():
    assert get_parser("raon_1") is RaonReasoningParser
    assert ToolParserManager.get_tool_parser("raon_1") is RaonToolParser
    assert RaonToolParser.supports_native_format()


@pytest.mark.parametrize(
    "value",
    [
        "text with a literal </parameter> inside",
        "text with a literal </function> inside",
        "text with a literal <parameter=fake> inside",
        "has <parameter=q> and </parameter> both",
    ],
)
def test_complete_literal_markers_preserve_values_and_call_boundaries(value):
    text = "Before." + call({"text": value}) + call({"count": "42"}) + "After."
    result = RaonToolParser().extract_tool_calls(text, REQUEST)
    assert result.content == "Before.After."
    assert [
        (item["name"], json.loads(item["arguments"])) for item in result.tool_calls
    ] == [("run", {"text": value}), ("run", {"count": 42})]


@pytest.mark.parametrize("marker", ["</parameter>", "</function>"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_postprocessor_literal_marker_value_parity_at_every_split(marker, wrapped):
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    value = f"a literal {marker} inside"
    text = call({"text": value})
    if not wrapped:
        text = text.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    complete = RaonToolParser().extract_tool_calls(text, REQUEST)
    expected = [(tc["name"], json.loads(tc["arguments"])) for tc in complete.tool_calls]
    assert expected == [("run", {"text": value})]

    for chunks in [list(text)] + [
        [text[:offset], text[offset:]] for offset in range(len(text) + 1)
    ]:
        cfg = MagicMock()
        cfg.engine = None
        cfg.reasoning_parser = None
        cfg.reasoning_parser_name = None
        cfg.tool_call_parser = "raon_1"
        cfg.tool_parser_instance = None
        cfg.enable_auto_tool_choice = True
        pp = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
        events = []
        for chunk in chunks:
            events.extend(
                pp.process_chunk(
                    SimpleNamespace(
                        new_text=chunk,
                        finished=False,
                        token=0,
                        finish_reason=None,
                        prompt_tokens=0,
                        generation_tokens=0,
                        channel=None,
                        tool_calls=None,
                        completion_tokens=0,
                        tokens=[],
                        logprobs=None,
                    )
                )
            )
        events.extend(pp.finalize())
        calls = {}
        for event in events:
            if event.type != "tool_call":
                continue
            for tc in event.tool_calls:
                current = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                for key in current:
                    current[key] += tc.get("function", {}).get(key, "")
        assert [
            (tc["name"], json.loads(tc["arguments"])) for tc in calls.values()
        ] == expected, chunks


@pytest.mark.parametrize("step", [1, 2, 7, 31, 10000])
def test_multiple_calls_typed_multiline_and_streaming(step):
    text = "Working.\n" + call() + "\n" + call({"text": "끝", "count": "0"}) + "\nDone."
    result = RaonToolParser().extract_tool_calls(text, REQUEST)
    expected = [("run", EXPECTED), ("run", {"text": "끝", "count": 0})]
    assert [
        (v["name"], json.loads(v["arguments"])) for v in result.tool_calls
    ] == expected
    content, streamed = stream_tools(
        text, [text[i : i + step] for i in range(0, len(text), step)]
    )
    assert streamed == expected
    assert content.strip() == result.content


@pytest.mark.parametrize("boundary", range(len(call({"count": "42"})) + 1))
def test_every_two_chunk_boundary(boundary):
    text = call({"count": "42"})
    assert stream_tools(text, [text[:boundary], text[boundary:]])[1] == [
        ("run", {"count": 42})
    ]


@pytest.mark.parametrize(
    "text,reasoning,content",
    [
        ("<think>check\n</think>\n\nAnswer", "check", "\n\nAnswer"),
        ("check\n</think>\n\nAnswer", "check", "\n\nAnswer"),
        ("<think></think>\n\nAnswer", None, "\n\nAnswer"),
        ("check " + call({}), "check", call({})),
        ("still checking", "still checking", None),
        ("<think>check</think>Answer</think>", "check", "Answer"),
    ],
)
@pytest.mark.parametrize("step", [1, 3, 10000])
def test_reasoning_stream_parity(text, reasoning, content, step):
    parser = RaonReasoningParser(None)
    assert parser.extract_reasoning(text) == (reasoning, content)
    parser.reset_state()
    parts = []
    for i in range(0, len(text), step):
        parts.append(
            parser.extract_reasoning_streaming(
                text[:i], text[: i + step], text[i : i + step]
            )
        )
    parts.append(parser.finish_stream())
    assert "".join(v.reasoning or "" for v in parts if v) == (reasoning or "")
    assert "".join(v.content or "" for v in parts if v) == (content or "")


def test_thinking_off_and_request_reset():
    parser = RaonReasoningParser(None)
    assert parser.extract_reasoning("Direct answer", enable_thinking=False) == (
        None,
        "Direct answer",
    )
    parser.set_enable_thinking(False)
    msg = parser.extract_reasoning_streaming("", "Direct", "Direct")
    assert msg.content == "Direct" and msg.reasoning is None
    parser.set_enable_thinking(True)
    assert parser.extract_reasoning_streaming("", "Check", "Check").reasoning == "Check"


@pytest.mark.parametrize(
    "text", ["ordinary prose", "angle <", "partial <tool_ca", "x > y"]
)
def test_prose_and_partial_tags_flush(text):
    result = RaonToolParser().extract_tool_calls(text)
    assert not result.tools_called and result.content == text
    assert stream_tools(text, list(text), request=None) == (text, [])


def test_schema_and_reasoning_differ_from_existing_parsers():
    # Same XML spelling does not imply identical coercion. The vendor accepts
    # boolean 1 while retaining JSON collection members as written.
    existing = ToolParserManager.get_tool_parser("qwen3_coder_xml")()
    other = existing.extract_tool_calls(call(), REQUEST)
    assert json.loads(other.tool_calls[0]["arguments"]) != EXPECTED
    # Raon's template starts in reasoning even if generation truncates before
    # the close tag. Existing Qwen defaults classify untagged output as content.
    assert get_parser("qwen3")().extract_reasoning("still checking") != (
        "still checking",
        None,
    )


def test_token_ids_preserve_specials_and_drop_eos():
    class Tokenizer:
        all_special_tokens = [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<eos>",
        ]
        all_special_ids = [10, 11, 12, 13, 14]

        def get_vocab(self):
            return dict(zip(self.all_special_tokens, self.all_special_ids))

        def decode(self, ids):
            return "".join(
                dict(zip(self.all_special_ids, self.all_special_tokens)).get(i, "")
                for i in ids
            )

    parser = RaonToolParser(Tokenizer())
    text = call({})
    delta = parser.extract_tool_calls_streaming(
        "", text, text, delta_token_ids=[12, 13]
    )
    assert delta["tool_calls"][0]["function"]["name"] == "run"
    assert not parser.extract_tool_calls_streaming(
        text, text + "<eos>", "<eos>", delta_token_ids=[14]
    )


def test_release_template_default_and_no_think():
    path = Path(__file__).parent / "fixtures/raon_1/chat_template.jinja"
    template = _compile_jinja_template(path.read_text())
    kwargs = dict(
        messages=[{"role": "user", "content": "Hi"}], add_generation_prompt=True
    )
    assert template.render(**kwargs).endswith("<think>\n")
    assert template.render(**kwargs, enable_thinking=False).endswith(
        "<think></think>\n\n"
    )


def test_postprocessor_routes_reasoning_and_tools():
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = RaonReasoningParser(None)
    cfg.reasoning_parser_name = "raon_1"
    cfg.tool_call_parser = "raon_1"
    cfg.tool_parser_instance = None
    cfg.enable_auto_tool_choice = True
    pp = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    text = "<think>check</think>\n\n" + call({"count": "42"})
    events = []
    for char in text:
        events.extend(
            pp.process_chunk(
                SimpleNamespace(
                    new_text=char,
                    finished=False,
                    token=0,
                    finish_reason=None,
                    prompt_tokens=0,
                    generation_tokens=0,
                    channel=None,
                    tool_calls=None,
                    completion_tokens=0,
                    tokens=[],
                    logprobs=None,
                )
            )
        )
    events.extend(pp.finalize())
    assert "check" in "".join(
        v.reasoning or "" for v in events if v.type == "reasoning"
    )
    calls = [v for v in events if v.type == "tool_call"]
    assert calls
    name = "".join(
        tc.get("function", {}).get("name", "")
        for event in calls
        for tc in event.tool_calls
    )
    arguments = "".join(
        tc.get("function", {}).get("arguments", "")
        for event in calls
        for tc in event.tool_calls
    )
    assert name == "run"
    assert json.loads(arguments) == {"count": 42}


def test_invalid_typed_values_preserve_vendor_fallback():
    values = {
        "count": "forty-two",
        "enabled": "maybe",
        "ratio": "NaN",
        "items": "not-json",
    }
    text = call(values)
    result = RaonToolParser().extract_tool_calls(text, REQUEST)
    assert json.loads(result.tool_calls[0]["arguments"]) == values
    assert stream_tools(text, list(text))[1] == [("run", values)]


def test_empty_malformed_call_does_not_invent_a_function():
    text = "Before.<tool_call><function=></function></tool_call>After."
    result = RaonToolParser().extract_tool_calls(text, REQUEST)
    assert not result.tools_called and result.content == "Before.After."
    assert stream_tools(text, list(text)) == ("Before.After.", [])


class TokenFixture:
    """Small detokenizer with real marker IDs and controllable decoded gaps."""

    all_special_tokens = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<eos>"]
    all_special_ids = [10, 11, 12, 13, 14]
    texts = {
        10: "<think>",
        11: "</think>",
        12: "<tool_call>",
        13: "</tool_call>",
        14: "<eos>",
        20: "why",
        21: "answer",
        22: " ",
        23: "",
        24: "<",
    }

    def get_vocab(self):
        return dict(zip(self.all_special_tokens, self.all_special_ids))

    def decode(self, ids):
        return "".join(self.texts[i] for i in ids)


def test_reasoning_engine_lifecycle_and_token_accounting():
    parser = RaonReasoningParser(TokenFixture())
    assert (parser.reasoning_start_str, parser.reasoning_end_str) == (
        "<think>",
        "</think>",
    )
    request = SimpleNamespace(chat_template_kwargs={"force_nonempty_content": True})
    assert parser.adjust_request(request) is request
    assert request.skip_special_tokens is False
    assert parser.count_reasoning_tokens([]) == 0
    assert parser.count_reasoning_tokens([10, 20, 22, 11, 21]) == 2
    assert parser.count_reasoning_tokens([20, 11]) == 1
    assert parser.extract_content_ids([20, 11, 21]) == [21]
    assert parser.extract_content_ids([20]) == [20]
    for ids, ended in [
        ([], False),
        ([10], False),
        ([11], True),
        ([12], True),
        ([12, 13], False),
        ([20], False),
    ]:
        assert parser.is_reasoning_end(ids) is ended
    assert parser.is_open_in_think("why")
    assert not parser.is_open_in_think("why</think>")
    assert not parser.has_engine_confirmed_reasoning_end()
    assert parser._extract_reasoning("why", request) == (None, "why")
    parser.reset_state()
    assert parser.get_streaming_fallback_content("", request) is None
    delta = parser._extract_reasoning_streaming(
        "", "why ", "why ", [], [20, 22], [20, 22]
    )
    assert delta.reasoning == "why"
    assert parser.count_reasoning_tokens([]) == 2
    assert parser.get_streaming_fallback_content("", request) == "why "
    assert parser.get_streaming_fallback_content("", SimpleNamespace()) is None
    parser._extract_reasoning_streaming(
        "why ", "why </think>", "</think>", [20, 22], [20, 22, 11], [11]
    )
    assert parser.has_engine_confirmed_reasoning_end()
    parser.set_enable_thinking(False)
    assert parser.is_reasoning_end([])
    assert not parser.is_open_in_think("why")


def test_markup_token_validation_and_minimal_tokenizers():
    from rapid_mlx.reasoning.raon_1_parser import RaonMarkupTokens

    with pytest.raises(ValueError, match="missing from the tokenizer"):
        RaonMarkupTokens(SimpleNamespace(get_vocab=lambda: {}))
    tokens = RaonMarkupTokens(
        SimpleNamespace(get_vocab=TokenFixture().get_vocab, decode=lambda _: None)
    )
    assert tokens.text(20) == ""
    assert tokens.text(20) == ""  # cached non-string decode


def test_delayed_special_tokens_and_eof_preserve_text_and_counts():
    from rapid_mlx.reasoning.raon_1_parser import RaonMarkupStream, RaonMarkupTokens

    stream = RaonMarkupStream(RaonMarkupTokens(TokenFixture()))
    assert stream.feed("", [20, 11, 21]) == []
    assert stream.feed("partial", [22]) == []
    pieces = stream.feed("</think>answer", [21])
    assert [(p.kind, p.value) for p in pieces] == [
        (None, "partial"),
        ("</think>", "</think>"),
        (None, "answer"),
    ]
    stream.reset()
    assert stream.feed("", [20, 11, 21]) == []
    pieces = stream.finish()
    assert [(p.kind, p.value, p.count) for p in pieces] == [
        ("</think>", "</think>", 0),
        (None, "", 1),
    ]
    stream.reset()
    assert stream.feed("prefix", [11]) == []
    assert [(p.kind, p.value) for p in stream.finish()] == [
        (None, "prefix"),
        ("</think>", "</think>"),
    ]


@pytest.mark.parametrize(
    "text,ids,expected",
    [
        (
            "why</think>answer",
            [20, 11, 21],
            [(None, "why"), ("</think>", "</think>"), (None, "answer")],
        ),
        ("prefix</think>", [11], [(None, "prefix"), ("</think>", "</think>")]),
        (
            "WHY</think>ANSWER",
            [20, 11, 21],
            [(None, "WHY"), ("</think>", "</think>"), (None, "ANSWER")],
        ),
        ("</think>", [10, 11], [("</think>", "</think>")]),
        ("<think>", [24], [(None, "<think>")]),
    ],
)
def test_token_alignment_uses_detokenized_text(text, ids, expected):
    from rapid_mlx.reasoning.raon_1_parser import RaonMarkupStream, RaonMarkupTokens

    stream = RaonMarkupStream(RaonMarkupTokens(TokenFixture()))
    assert [(p.kind, p.value) for p in stream.feed(text, ids)] == expected


@pytest.mark.parametrize(
    "value,schema,expected",
    [
        ({"n": "2"}, {"properties": {"n": {"type": "integer"}}}, {"n": 2}),
        ({"n": 2}, {}, {"n": 2}),
        (["2", "3"], {"items": {"type": "integer"}}, [2, 3]),
        ([1], {}, [1]),
        (True, {"type": "integer"}, True),
        ("0", {"type": "boolean"}, False),
    ],
)
def test_recursive_schema_conformance(value, schema, expected):
    from rapid_mlx.tool_parsers.raon_1_tool_parser import _conform

    result, _ = _conform(value, schema)
    assert result == expected


def test_schema_enum_inventory_and_unknown_properties():
    from rapid_mlx.tool_parsers.raon_1_tool_parser import _conform_object, _schema_types

    assert _schema_types(None) == {"string"}
    assert _schema_types({"enum": [None, True, 1, 1.5, "x", [], {}]}) == {
        "null",
        "boolean",
        "integer",
        "number",
        "string",
        "array",
        "object",
    }
    values = {"unknown": "2"}
    assert not _conform_object(values, {"unknown": None})
    assert values == {"unknown": "2"}


@pytest.mark.parametrize(
    "body,expected",
    [
        ('{"name":"run","arguments":{"x":2}}', ("run", '{"x": 2}')),
        ('{"name":"run","parameters":"raw"}', ("run", "raw")),
        ('{"name":"run","x":2}', ("run", '{"x": 2}')),
        ("[]", ("", "[]")),
        ("broken", ("", "broken")),
    ],
)
def test_json_call_body_fallback(body, expected):
    assert RaonToolParser._json_call(body) == expected


@pytest.mark.parametrize(
    "text,count,content",
    [
        ("<tool_call></tool_call>", 0, ""),
        ("<function=run</function>", 1, ""),
        ("<function=run>", 1, ""),
        ("<tool_call><function=run>", 1, ""),
        ("<function=run></function><function=run></function>", 2, ""),
        ("<function=run></function><tool_call><function=run></function>", 2, ""),
        ("<tool_call>ignored</tool_call>", 0, ""),
        ("<eos>", 0, ""),
        ("   ", 0, "   "),
    ],
)
def test_incomplete_and_adjacent_calls_flush_exactly_once(text, count, content):
    from rapid_mlx.tool_parsers.raon_1_tool_parser import _wire

    parser = RaonToolParser(TokenFixture())
    deltas = [parser.extract_tool_calls_streaming("", text, text)]
    deltas.append(_wire(parser.finish_streaming()))
    assert parser.finish_streaming() is None
    calls = [tc for delta in deltas if delta for tc in delta.get("tool_calls", [])]
    assert [tc["index"] for tc in calls] == list(range(count))
    assert len({tc["id"] for tc in calls}) == count
    assert [tc["function"] for tc in calls] == [
        {"name": "run", "arguments": "{}"}
    ] * count
    assert "".join(delta.get("content", "") for delta in deltas if delta) == content
    parser.reset()
    assert parser.extract_tool_calls("plain").content == "plain"


def test_tool_suppression_and_internal_reasoning_state():
    parser = RaonToolParser(TokenFixture())
    result = parser.extract_tool_calls_streaming(
        "", call({}), call({}), request={**REQUEST, "tool_choice": "none"}
    )
    assert result is None or "tool_calls" not in result
    parser.reset()
    # The shared engine may start in the grammar's pre-content state.
    pieces = parser._stream.feed("why</think><function=run></function>", [])
    parser._reset("before")
    result = parser._delta(parser._events(pieces), finished=True)
    assert result.reasoning == "why"
    assert result.tool_calls[0].function.name == "run"
    parser._reset("before")
    assert parser._finish_events() == [("reasoning_end", "", -1)]


def test_tool_schema_objects_and_invalid_json_are_preserved():
    tool = SimpleNamespace(model_dump=lambda: REQUEST["tools"][0])
    parser = RaonToolParser(tools=[None, tool])
    assert parser._properties("run") == PROPERTIES
    assert parser._typed("bad", "run") == "bad"
    assert parser._typed("[]", "run") == "[]"
    assert parser._typed("{}", "missing") == "{}"
    assert parser._string_keys("missing") is None
    assert parser._string_keys("run") == {"text"}


def test_markup_rejects_ambiguous_literal_prefixes(monkeypatch):
    from rapid_mlx.reasoning import raon_1_parser as module

    monkeypatch.setattr(module, "TEXT_TAGS", (*module.TEXT_TAGS, "<think>x"))
    with pytest.raises(ValueError, match="must not prefix"):
        module.RaonMarkupTokens(TokenFixture())


def test_alignment_empty_decodes_and_unmatched_token_text():
    from rapid_mlx.reasoning.raon_1_parser import (
        RaonMarkupStream,
        RaonMarkupTokens,
        _Text,
    )

    stream = RaonMarkupStream(RaonMarkupTokens(TokenFixture()))
    assert [p.text for p in stream._align("text", [])] == ["text"]
    assert [p.text for p in stream._align("text", [_Text("", (), 1)])] == ["text", ""]
    assert [p.text for p in stream._align("text", [_Text("different")])] == ["text"]
    assert stream._char_counts("", (), 2) == []
    assert stream._char_counts("text", ("", "missing"), 2) == [2, 0, 0, 0]
    parser = RaonReasoningParser()
    from rapid_mlx.reasoning.raon_1_parser import Piece

    assert parser._events([Piece(None, "")]) == []


def test_waiting_tokens_resolve_at_start_and_before_another_special():
    from rapid_mlx.reasoning.raon_1_parser import RaonMarkupStream, RaonMarkupTokens

    stream = RaonMarkupStream(RaonMarkupTokens(TokenFixture()))
    assert stream.feed("", [11, 21]) == []
    pieces = stream.feed("</think><eos>", [14])
    assert [(p.kind, p.value, p.count) for p in pieces] == [
        ("</think>", "</think>", 0),
        (None, "", 1),
        ("<discard>", "<eos>", 0),
    ]


def test_stable_prefix_retains_unterminated_string_with_escapes():
    from rapid_mlx.tool_parsers.raon_1_tool_parser import _stable_prefix

    raw = '{"text":"a\\"b'
    assert _stable_prefix(raw, {"text"}) == raw


def test_delta_merge_fills_missing_metadata_and_appends_arguments():
    from rapid_mlx.tool_parsers.raon_1_tool_parser import (
        DeltaFunctionCall,
        DeltaToolCall,
    )

    parts = [
        DeltaToolCall(0),
        DeltaToolCall(0, id="call_1", type="function"),
        DeltaToolCall(0, function=DeltaFunctionCall(arguments="{")),
        DeltaToolCall(0, function=DeltaFunctionCall(name="run", arguments="}")),
    ]
    merged = RaonToolParser._merge(parts)
    assert len(merged) == 1
    assert merged[0].id == "call_1" and merged[0].type == "function"
    assert merged[0].function.name == "run"
    assert merged[0].function.arguments == "{}"


def test_before_content_tool_transition_and_out_of_range_close():
    parser = RaonToolParser()
    pieces = parser._stream.feed("<tool_call><function=run></function>", [])
    result = parser._delta(parser._events(pieces))
    assert result.tool_calls[0].function.name == "run"
    deltas = []
    parser._on_close(100, deltas)
    assert deltas == []


def test_json_body_and_empty_named_call_complete_extraction():
    from rapid_mlx.tool_parsers.raon_1_tool_parser import _Call

    parser = RaonToolParser()
    # Bare JSON is a fallback when the grammar has accumulated a call body
    # without a function header (the supported low-level engine event path).
    delta = parser._delta(
        [("open", "", 0), ("args", '{"name":"run","arguments":{}}', 0)]
    )
    assert delta.tool_calls[0].function.name == "run"
    parser._calls[0].name = ""
    calls = parser._calls_found()
    assert calls[0].function.name == "run"
    assert json.loads(calls[0].function.arguments) == {}
    empty = _Call()
    empty.name = "run"
    parser._calls = [empty]
    calls = parser._calls_found()
    assert calls[0].function.arguments == "{}"
