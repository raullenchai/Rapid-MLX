# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 parser fixtures, adapted to Rapid's dictionary protocol."""

import json
from pathlib import Path

import pytest
import regex as re

from rapid_mlx.reasoning.hy4_parser import HyV4ReasoningParser
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.tool_parsers.hy_v4_tool_parser import HyV4ToolParser


class Tokenizer:
    def __init__(self, suffix=":opensource"):
        self.vocab = {
            f"<{close}{name}{suffix}>": i
            for i, (name, close) in enumerate(
                (name, close)
                for name in ("think", "tool_calls", "tool_call", "arg_key", "arg_value")
                for close in ("", "/")
            )
        }
        self.init_kwargs = {}

    def get_vocab(self):
        return self.vocab


@pytest.mark.parametrize("module", ["reasoning", "tools"])
@pytest.mark.parametrize("token", ["<think:other>", "</arg_value:other>"])
def test_ambiguous_structural_suffix_is_rejected(token, module):
    from rapid_mlx.reasoning import hy4_parser
    from rapid_mlx.tool_parsers import hy_v4_tool_parser

    detect_token_suffix = (
        hy4_parser if module == "reasoning" else hy_v4_tool_parser
    ).detect_token_suffix

    tokenizer = Tokenizer()
    tokenizer.vocab[token] = 99
    with pytest.raises(RuntimeError, match="suffix"):
        detect_token_suffix(tokenizer)


@pytest.mark.parametrize("module", ["reasoning", "tools"])
@pytest.mark.parametrize("missing", list(Tokenizer().vocab))
def test_incomplete_structural_tokens_are_rejected(module, missing):
    from rapid_mlx.reasoning import hy4_parser
    from rapid_mlx.tool_parsers import hy_v4_tool_parser

    detect_token_suffix = (
        hy4_parser if module == "reasoning" else hy_v4_tool_parser
    ).detect_token_suffix

    tokenizer = Tokenizer()
    del tokenizer.vocab[missing]
    with pytest.raises(RuntimeError, match="suffix"):
        detect_token_suffix(tokenizer)


def call(pairs, suffix):
    args = "".join(
        f"<arg_key{suffix}>{k}</arg_key{suffix}><arg_value{suffix}>{v}</arg_value{suffix}>"
        for k, v in pairs
    )
    return f"<tool_call{suffix}>record{args}</tool_call{suffix}>"


def stream(parser, chunks, request):
    previous, content, calls, names = "", "", {}, []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, current, chunk, request=request
        )
        previous = current
        if not delta:
            continue
        content += delta.get("content") or ""
        for tc in delta.get("tool_calls", []):
            if "name" in tc["function"]:
                assert tc["id"].startswith("call_")
                names.append((tc["index"], tc["function"]["name"]))
            calls[tc["index"]] = calls.get(tc["index"], "") + (
                tc["function"].get("arguments") or ""
            )
    return content, [json.loads(calls[i]) for i in sorted(calls)], names


@pytest.mark.parametrize("suffix", ["", ":opensource", ":6124c78e"])
@pytest.mark.parametrize("key", [" count ", "\tcount\n", " \t "])
@pytest.mark.parametrize(
    "kind,raw,expected",
    [
        ("string", '  a"b\\c  ', '  a"b\\c  '),
        ("integer", "7", 7),
        ("number", "1.5", 1.5),
        ("boolean", "true", True),
        ("array", "[1, 2]", [1, 2]),
        ("object", '{"x":1}', {"x": 1}),
        ("null", "null", None),
    ],
)
def test_typed_whitespace_keys_all_chunk_boundaries(suffix, key, kind, raw, expected):
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "record",
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": kind}},
                    },
                },
            }
        ]
    }
    wire = (
        f"<tool_calls{suffix}>" + call([(key, raw)], suffix) + f"</tool_calls{suffix}>"
    )
    result = HyV4ToolParser(Tokenizer(suffix)).extract_tool_calls(wire, request)
    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {key: expected}
    # Every split includes ones inside closing tags, escapes and Unicode text.
    for split in range(len(wire) + 1):
        parser = HyV4ToolParser(Tokenizer(suffix))
        assert stream(parser, [wire[:split], wire[split:]], request) == (
            "",
            [{key: expected}],
            [(0, "record")],
        )
    assert stream(HyV4ToolParser(Tokenizer(suffix)), list(wire), request)[1] == [
        {key: expected}
    ]


@pytest.mark.parametrize("suffix", ["", ":opensource"])
def test_parallel_duplicate_keys_nullable_refs_and_bare_calls(suffix):
    params = {
        "type": "object",
        "$defs": {"n": {"type": ["integer", "null"]}},
        "properties": {
            " n ": {"$ref": "#/$defs/n"},
            "s": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }
    request = {"tools": [{"function": {"name": "record", "parameters": params}}]}
    bare = (
        call([(" n ", "null"), ("s", "null")], suffix)
        + call([(" n ", "2"), (" n ", "3")], suffix)
        + call([], suffix)
    )
    wire = f"prefix<tool_calls{suffix}>" + bare + f"</tool_calls{suffix}>"
    expected = [{" n ": None, "s": None}, {" n ": 3}, {}]
    for text in (wire, "prefix" + bare):
        result = HyV4ToolParser(Tokenizer(suffix)).extract_tool_calls(text, request)
        assert [json.loads(c["arguments"]) for c in result.tool_calls] == expected
        content, calls, names = stream(
            HyV4ToolParser(Tokenizer(suffix)), list(text), request
        )
        assert content == "prefix"
        assert calls == expected
        assert names == [(i, "record") for i in range(3)]


@pytest.mark.parametrize("suffix", ["", ":opensource"])
def test_multiple_outer_envelopes_preserve_all_calls(suffix):
    wire = "prefix" + "".join(
        f"<tool_calls{suffix}>"
        + call([("x", value)], suffix)
        + f"</tool_calls{suffix}>"
        for value in ("first", "second")
    )
    result = HyV4ToolParser(Tokenizer(suffix)).extract_tool_calls(wire)
    assert result.tools_called
    assert result.content == "prefix"
    expected = [{"x": "first"}, {"x": "second"}]
    assert [json.loads(c["arguments"]) for c in result.tool_calls] == expected
    for split in range(len(wire) + 1):
        assert stream(
            HyV4ToolParser(Tokenizer(suffix)), [wire[:split], wire[split:]], None
        ) == ("prefix", expected, [(0, "record"), (1, "record")])
    assert stream(HyV4ToolParser(Tokenizer(suffix)), list(wire), None)[1] == expected


@pytest.mark.parametrize(
    "tail", ["garbage", "<tool_calls:opensource>garbage", "<tool_calls:broken>"]
)
def test_trailing_malformed_payload_falls_back_to_content(tail):
    wire = (
        "prefix<tool_calls:opensource>"
        + call([], ":opensource")
        + "</tool_calls:opensource>"
        + tail
    )
    result = HyV4ToolParser().extract_tool_calls(wire)
    assert not result.tools_called
    assert result.content == wire
    assert result.tool_calls == []


@pytest.mark.parametrize("tail", ["<tool_ca", "<tool_calls:broken>"])
def test_streaming_flush_preserves_malformed_tail(tail):
    wire = (
        "<tool_calls:opensource>"
        + call([], ":opensource")
        + "</tool_calls:opensource>"
        + tail
    )
    for chunks in ([wire], list(wire)):
        parser = HyV4ToolParser()
        content, calls, _ = stream(parser, chunks, None)
        assert calls == [{}]
        assert content + parser.flush_held_content(wire) == tail
        assert parser.flush_held_content(wire) == ""


def test_streaming_atomic_multiple_outer_envelopes():
    from rapid_mlx.tool_parsers.hy_v4_tool_parser import HYV4ToolExtractor

    tokenizer = Tokenizer("")
    wire = (
        "prefix"
        + "".join(
            "<tool_calls>" + call([("x", value)], "") + "</tool_calls>"
            for value in ("first", "second")
        )
        + "garbage"
    )
    ids = list(tokenizer.vocab.values())
    ex = HYV4ToolExtractor(tokenizer.vocab, "", True)
    result = ex.extract_tool_calls_streaming(
        "", wire, wire, [], ids, ids, None, guided=False
    )
    assert result["content"] == "prefixgarbage"
    assert [json.loads(c["arguments"]) for c in result["tool_calls"]] == [
        {"x": "first"},
        {"x": "second"},
    ]


def test_streaming_trailing_garbage_is_content():
    wire = (
        "prefix<tool_calls:opensource>"
        + call([], ":opensource")
        + "</tool_calls:opensource>garbage"
    )
    for split in range(len(wire) + 1):
        assert stream(HyV4ToolParser(), [wire[:split], wire[split:]], None) == (
            "prefixgarbage",
            [{}],
            [(0, "record")],
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "ordinary < prose",
        "<tool_calls:opensource>garbage</tool_calls:opensource>",
        "<tool_calls:opensource><tool_call:opensource></tool_call:opensource></tool_calls:opensource>",
        "<tool_calls:opensource><tool_call:opensource>name<arg_key:opensource>x</arg_key:opensource>bad</tool_call:opensource></tool_calls:opensource>",
    ],
)
def test_malformed_and_plain_content(text):
    result = HyV4ToolParser().extract_tool_calls(text)
    assert not result.tools_called
    assert result.content == text
    if "tool_calls" not in text:
        assert stream(HyV4ToolParser(), list(text), None)[0] == text


@pytest.mark.parametrize("suffix", ["", ":opensource"])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("thought{end}answer", ("thought", "answer")),
        ("{start}thought{end}answer", ("thought", "answer")),
        ("unfinished thought", ("unfinished thought", None)),
        ("{end}answer", ("", "answer")),
    ],
)
def test_reasoning_prompt_injection_and_every_boundary(suffix, text, expected):
    tokenizer = Tokenizer(suffix)
    text = text.format(start=f"<think{suffix}>", end=f"</think{suffix}>")
    parser = HyV4ReasoningParser(tokenizer)
    assert parser.extract_reasoning(text) == expected
    for split in range(len(text) + 1):
        parser.reset_state()
        previous, r, c = "", "", ""
        for piece in (text[:split], text[split:]):
            delta = parser.extract_reasoning_streaming(
                previous, previous + piece, piece
            )
            previous += piece
            if delta:
                r += delta.reasoning or ""
                c += delta.content or ""
        assert (r, c or None) == expected
    parser.set_enable_thinking(False)
    assert parser.extract_reasoning(text) == (None, text)
    assert parser.extract_reasoning_streaming("", text, text).content == text
    assert parser.extract_reasoning_streaming("", "", "") is None


def test_reasoning_finish_literal_partial_and_registration():
    parser = HyV4ReasoningParser()
    for prefix, lane in [("", "reasoning"), ("</think:opensource>answer", "content")]:
        parser.reset_state()
        text = prefix + "<thi"
        parser.extract_reasoning_streaming("", text, text)
        assert getattr(parser.finish_stream(), lane) == "<thi"
        assert parser.finish_stream() is None
    assert ToolParserManager.get_tool_parser("hy_v4") is HyV4ToolParser
    from rapid_mlx.reasoning import get_parser

    assert get_parser("hy_v4") is HyV4ReasoningParser
    parser = HyV4ToolParser()
    parser.reset()
    assert parser.current_tool_id == -1
    assert not parser.prev_tool_call_arr


def test_template_tools_and_reasoning_effort(tmp_path):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    from rapid_mlx.model_auto_config import detect_model_config

    fixture = Path(__file__).parent / "fixtures/hy_v4_flash"
    template = (fixture / "chat_template.jinja").read_text()
    compiled = _compile_jinja_template(template)
    messages = [{"role": "user", "content": "Weather?"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "record", "parameters": {"type": "object"}},
        }
    ]
    for effort in ("high", "no_think", "max"):
        rendered = compiled.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
        assert "<tool_calls:opensource>" in rendered
        assert '"name": "record"' in rendered
        assert rendered.endswith(
            "<think:opensource></think:opensource>"
            if effort == "no_think"
            else "<think:opensource>"
        )
    (tmp_path / "config.json").write_text((fixture / "config.json").read_text())
    cfg = detect_model_config(str(tmp_path))
    assert cfg.tool_call_parser == "hy_v4"
    assert cfg.reasoning_parser == "hy_v4"
    assert cfg.supports_spec_decode is False


def test_native_no_think_stays_in_sync_with_template_and_request():
    from types import SimpleNamespace

    from transformers.utils.chat_template_utils import _compile_jinja_template

    from rapid_mlx.service.helpers import _extract_thinking_from_request
    from rapid_mlx.utils.chat_template import apply_chat_template

    source = (
        Path(__file__).parent / "fixtures/hy_v4_flash/chat_template.jinja"
    ).read_text()
    compiled = _compile_jinja_template(source)

    class TemplateTokenizer:
        chat_template = source

        def apply_chat_template(self, messages, **kwargs):
            return compiled.render(messages=messages, **kwargs)

    for enable in (True, False):
        result = apply_chat_template(
            TemplateTokenizer(),
            [{"role": "user", "content": "hi"}],
            enable_thinking=enable,
        )
        assert result.endswith(
            "<think:opensource>" if enable else "<think:opensource></think:opensource>"
        )
    for effort in ("high", "no_think"):
        result = apply_chat_template(
            TemplateTokenizer(),
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": effort},
        )
        assert result.endswith(
            "<think:opensource>"
            if effort == "high"
            else "<think:opensource></think:opensource>"
        )
        request = SimpleNamespace(
            enable_thinking=None, chat_template_kwargs={"reasoning_effort": effort}
        )
        assert _extract_thinking_from_request(request) is (
            False if effort == "no_think" else None
        )


def test_schema_resolution_limits_and_value_fallbacks(monkeypatch):
    from rapid_mlx.tool_parsers import hy_v4_tool_parser as p

    for raw, normalized in [
        ("str", "string"),
        ("uint64", "integer"),
        ("FLOAT32", "number"),
        ("custom", "custom"),
    ]:
        assert p._normalize_type(raw) == normalized
    root = {
        "a/b": {"~c": {"type": "integer"}},
        "list": [{"type": "number"}],
        "scalar": 3,
    }
    assert p._resolve_json_pointer(root, "") is root
    assert p._resolve_json_pointer(root, "/a~1b/~0c") == {"type": "integer"}
    assert p._resolve_json_pointer(root, "/list/0") == {"type": "number"}
    for pointer in ("not/pointer", "/missing", "/list/9", "/list/x", "/scalar/x"):
        assert p._resolve_json_pointer(root, pointer) is None
    anchor = {"items": [{"$anchor": "x", "type": "boolean"}]}
    assert p._resolve_local_schema_ref(anchor, "#x")["type"] == "boolean"
    assert p._resolve_local_schema_ref(anchor, "external#x") is None
    assert p._resolve_local_schema_ref(root, "#/scalar") is None
    anchor["cycle"] = anchor
    assert p._find_schema_anchor(anchor, "absent") is None
    nested = current = {}
    for _ in range(10):
        current["child"] = {}
        current = current["child"]
    current["$anchor"] = "too-deep"
    assert p._find_schema_anchor(nested, "too-deep") is None
    assert p._resolve_ref_type({}, {}) == {}
    assert p._resolve_ref_type({"$ref": "#a"}, {"$anchor": "a", "$ref": "#a"}) == {}
    assert p._resolve_ref_type({"$ref": "#a"}, {}, ref_hops=3) == {}
    assert p._resolve_ref_type({"$ref": "#missing"}, {}) == {}
    assert p._resolve_ref_type({"$ref": "#"}, {}) == {}
    refs = {"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"type": "integer"}}}
    assert p._resolve_ref_type({"$ref": "#/$defs/a"}, refs) == {"type": "integer"}
    schema = {"oneOf": [{"$ref": "#/$defs/b"}, {"type": "string"}]}
    assert p._resolve_union_ref_types(schema, refs) == {
        "oneOf": [{"type": "integer"}, {"type": "string"}]
    }
    assert "$ref" in schema["oneOf"][0]
    assert p._resolve_ref_type({"$ref": "#"}, schema) == {"oneOf": schema["oneOf"]}
    for params in (None, {}, {"properties": []}, {"properties": {"a": 3}}):
        assert p._get_arg_schema("f", "a", [{"name": "f", "parameters": params}]) == {}
    assert p._get_arg_schema("f", "a", [{"name": "g"}]) == {}
    assert p._get_arg_schema("f", "a", None) == {}
    # Malformed union branches are rejected locally rather than crashing a response.
    malformed = [{"name": "f", "parameters": {"properties": {"a": {"oneOf": [3]}}}}]
    assert p._get_arg_schema("f", "a", malformed) == {"oneOf": [3]}
    with monkeypatch.context() as fault:

        def fail(*args, **kwargs):
            raise ValueError("malformed schema")

        fault.setattr(p, "_resolve_union_ref_types", fail)
        assert p._get_arg_schema("f", "a", malformed) == {}
    assert p._get_schema_options({}) == [{"type": "string"}]
    for kind, raw, expected in [
        ("boolean", "FALSE", False),
        ("boolean", "x", "x"),
        ("integer", "x", "x"),
        ("number", "7", 7),
        ("number", "bad", "bad"),
        ("object", "{'a': 1}", {"a": 1}),
        ("object", "bad", "bad"),
    ]:
        tools = [{"name": "f", "parameters": {"properties": {"a": {"type": kind}}}}]
        assert p._parse_value(raw, "f", "a", tools) == expected
    assert p._deserialize("[1, true]") == [1, True]


def test_extractor_atomic_paths_strict_rejections_and_suffix_errors(monkeypatch):
    from rapid_mlx.reasoning import hy4_parser as r
    from rapid_mlx.tool_parsers import hy_v4_tool_parser as p

    tokenizer = Tokenizer("")
    for module in (p, r):
        tokenizer.init_kwargs = {
            "model_specific_special_tokens": {"think_begin_token": "<think>"}
        }
        with pytest.raises(RuntimeError, match="declares"):
            module.detect_token_suffix(tokenizer)
        tokenizer.init_kwargs = {}
        empty = Tokenizer("")
        empty.vocab = {"ordinary": 99}
        assert module.detect_token_suffix(empty) == ""
    with pytest.raises(RuntimeError, match="could not locate"):
        p.HYV4ToolExtractor({}, "", True)
    with pytest.raises(RuntimeError, match="could not locate"):
        r.HYV4ReasoningExtractor({}, "")
    wire = "<tool_calls>" + call([("x", "abc")], "") + "</tool_calls>"
    for text in (wire, call([("x", "abc")], "")):
        extractor = p.HYV4ToolExtractor(tokenizer.vocab, "", False)
        assert extractor.extract_tool_calls(text, None)["tools_called"]
        pattern = "|".join(map(re.escape, tokenizer.vocab)) + "|[\\s\\S]"
        pieces = re.findall(pattern, text)
        previous, previous_ids, args = "", [], ""
        for piece in pieces:
            ids = [tokenizer.vocab.get(piece, -1)]
            result = extractor.extract_tool_calls_streaming(
                previous,
                previous + piece,
                piece,
                previous_ids,
                previous_ids + ids,
                ids,
                None,
                guided=False,
            )
            previous, previous_ids = previous + piece, previous_ids + ids
            if result:
                args += "".join(tc["arguments"] or "" for tc in result["tool_calls"])
        assert json.loads(args) == {"x": "abc"}
    extractor = p.HYV4ToolExtractor(tokenizer.vocab, "", True)
    delta = extractor.extract_tool_calls_streaming(
        "", "a < b", "a < b", [], [-1], [-1], None, guided=False
    )
    assert delta["content"] == "a < b"
    delta = extractor.extract_tool_calls_streaming(
        "",
        wire,
        wire,
        [],
        list(tokenizer.vocab.values()),
        list(tokenizer.vocab.values()),
        None,
        guided=False,
    )
    assert delta["tool_calls"]
    malformed = [
        "</tool_calls><tool_calls>",
        "<tool_calls><tool_call>x</tool_calls>",
        "<tool_calls><tool_call>x</tool_calls><tool_calls></tool_call></tool_calls>",
        "<tool_calls><tool_call><tool_call>x</tool_call></tool_call></tool_calls>",
        "<tool_calls><tool_call>x<arg_key>k</arg_key><arg_value>v</tool_call></tool_calls>",
    ]
    for text in malformed:
        with pytest.raises(ValueError):
            extractor._extract_tool_calls_strict(text, None)

    class Broken:
        def findall(self, _):
            raise ValueError("bad regex input")

    extractor.tool_call_regex = Broken()
    assert extractor._extract_tool_calls(wire, None) == []


def test_reasoning_reference_token_api():
    from rapid_mlx.reasoning.hy4_parser import HYV4ReasoningExtractor

    r = HYV4ReasoningExtractor({"<think>": 0, "</think>": 1}, "")
    assert r.has_reasoning_ended([1, 0]) is False
    assert r.has_reasoning_ended([0, 1]) is True
    assert r.reasoning_ended_in_delta([], [1]) is True
    assert r.extract_content_ids([3]) == []
    assert r.extract_content_ids([3, 1, 4]) == [4]
    assert r.count_reasoning_tokens([0, 3, 1, 4]) == 1
    assert r.count_reasoning_tokens([3, 4]) == 2
    cases = [
        ([0], [1, 4], "thought</think>answer", ("thought", "answer")),
        ([0], [1], "tail", (None, None)),
        ([0], [1, 4], "</think>answer", ("", "answer")),
        ([], [0, 3, 1, 4], "<think>thought</think>answer", ("thought", "answer")),
        ([], [0, 3], "<think>thought", ("<think>thought", None)),
        ([], [1, 4], "thought</think>answer", ("thought", "answer")),
        ([1], [4], "answer", (None, "answer")),
    ]
    for previous_ids, delta_ids, text, expected in cases:
        result = r.extract_reasoning_streaming(
            "", "", text, previous_ids, previous_ids + delta_ids, delta_ids
        )
        if result is None:
            assert expected == (None, None)
        else:
            assert (result["reasoning"], result["content"]) == expected
    r.thinking = False
    assert r.has_reasoning_ended([])
    assert r.reasoning_ended_in_delta([], [])
    assert r.extract_content_ids([3]) == [3]
    assert r.count_reasoning_tokens([3]) == 0
    assert r.extract_reasoning_streaming("", "", "", [], [], []) is None
    assert r.extract_reasoning_streaming("", "x", "x", [], [], []) == {
        "reasoning": None,
        "content": "x",
    }
    r.thinking, r.end_token_id = True, None
    assert r.reasoning_ended_in_delta([], []) is False
    parser = HyV4ReasoningParser()
    assert parser.reasoning_start_str == "<think:opensource>"
    assert parser.reasoning_end_str == "</think:opensource>"
    assert parser.extract_reasoning("x", enable_thinking=False) == (None, "x")


def test_batched_atomic_tool_deltas_and_empty_calls():
    from rapid_mlx.tool_parsers import hy_v4_tool_parser as p

    tokenizer = Tokenizer("")
    vocab = tokenizer.vocab
    ids = list(vocab.values())
    bare = call([("x", "雪")], "") + call([], "") + "<tool_calls></tool_calls>"
    ex = p.HYV4ToolExtractor(vocab, "", False)
    assert ex.extract_tool_calls(call([], ""), None)["tools_called"]
    assert not ex.extract_tool_calls("<tool_calls></tool_calls>", None)["tools_called"]
    for guided in (True, False):
        ex = p.HYV4ToolExtractor(vocab, "", True)
        result = ex.extract_tool_calls_streaming(
            "", bare, bare, [], ids, ids, None, guided=guided
        )
        assert len(result["tool_calls"]) == 2
        assert [json.loads(c["arguments"]) for c in result["tool_calls"]] == [
            {"x": "雪"},
            {},
        ]
    ex = p.HYV4ToolExtractor(vocab, "", True)
    # Atomic outer wrapper in one delta; inner name and arguments in the next.
    first = "lead<tool_calls>"
    assert (
        ex.extract_tool_calls_streaming(
            "",
            first,
            first,
            [],
            [vocab["<tool_calls>"]],
            [vocab["<tool_calls>"]],
            None,
            guided=False,
        )["content"]
        == "lead"
    )
    rest = call([], "") + "</tool_calls>"
    result = ex.extract_tool_calls_streaming(
        first, first + rest, rest, [vocab["<tool_calls>"]], ids, ids, None, guided=False
    )
    assert json.loads(result["tool_calls"][0]["arguments"]) == {}
    ex = p.HYV4ToolExtractor(vocab, "", True)
    # A saved argument prefix may be restored before the function-name header.
    ex._streamed_json_len = 1
    ex._streaming_tool_name = "record"
    result = ex._extract_streaming_incremental(True, "record", None)
    assert result["tool_calls"][0]["name"] == "record"
    assert result["tool_calls"][0]["arguments"] is None


def test_atomic_bare_marker_after_visible_prefix_and_wrapper_resume():
    from rapid_mlx.tool_parsers.hy_v4_tool_parser import HYV4ToolExtractor

    t = Tokenizer("")
    ex = HYV4ToolExtractor(t.vocab, "", True)
    wire = "prefix" + call([], "")
    ids = list(t.vocab.values())
    result = ex.extract_tool_calls_streaming(
        "", wire, wire, [], ids, ids, None, guided=False
    )
    assert result["content"] == "prefix"
    # Resume a marker accumulated by an earlier detokenizer update.
    ex = HYV4ToolExtractor(t.vocab, "", True)
    ex._buffer = "<tool_call>"
    ex._bare_tool_call = False
    ex.extract_tool_calls_streaming(
        "<tool_calls><tool_call>",
        "<tool_calls><tool_call>record",
        "record",
        [t.vocab["<tool_calls>"], t.vocab["<tool_call>"]],
        ids,
        [-1],
        None,
        guided=False,
    )


def test_whole_prefix_and_literal_partial_marker_flush():
    suffix = ":opensource"
    parser = HyV4ToolParser()
    wire = (
        "prefix<tool_calls:opensource>" + call([], suffix) + "</tool_calls:opensource>"
    )
    assert stream(parser, [wire], None)[:2] == ("prefix", [{}])
    assert parser.flush_held_content(wire) == ""
    for text in ("a <", "a <tool_ca", "a </tool_ca", "a <tool_call:opensou"):
        parser = HyV4ToolParser()
        content, calls, _ = stream(parser, list(text), None)
        assert not calls
        assert content + parser.flush_held_content(text) == text
        assert parser.flush_held_content(text) == ""


def test_boolean_schema_branches_do_not_break_type_inference():
    from rapid_mlx.tool_parsers.hy_v4_tool_parser import (
        _allows_null,
        _get_schema_options,
        _get_types,
    )

    schema = {"anyOf": [True, {"type": "integer"}, False, {"type": "null"}]}
    assert _get_types(schema) == {"string", "integer"}
    assert _allows_null(schema)
    assert _get_schema_options(True) == [{"type": "string"}]
