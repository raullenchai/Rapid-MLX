"""Inkling typed blocks retain their channels across arbitrary stream chunks."""

import json

import pytest

from rapid_mlx.reasoning import get_parser
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.utils.decode import IncrementalDecoder

M = "<|message_model|>"
R = "<|content_thinking|>"
T = "<|content_text|>"
C = "<|content_invoke_tool_json|>"
E = "<|end_message|>"
S = "<|content_model_end_sampling|>"
CALL = M + "read_file" + C + '{"name":"read_file","args":{"path":"sample.txt"}}' + E


def test_registered():
    assert get_parser("inkling")
    assert ToolParserManager.get_tool_parser("inkling")


@pytest.mark.parametrize(
    "raw,expected",
    [
        (M + R + "Secret." + E, ("Secret.", None)),
        (M + T + "Answer" + E + S, (None, "Answer")),
        (M + R + "Secret." + E + M + T + "Answer" + E, ("Secret.", "Answer")),
        ("Plain prose.", (None, "Plain prose.")),
        (R + "Unfinished", ("Unfinished", None)),
    ],
)
@pytest.mark.parametrize("width", [1, 7, 31, 1000])
def test_reasoning(raw, expected, width):
    parser = get_parser("inkling")()
    assert parser.extract_reasoning(raw) == expected
    reasoning = content = previous = ""
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        msg = parser.extract_reasoning_streaming(previous, previous + chunk, chunk)
        previous += chunk
        if msg:
            reasoning += msg.reasoning or ""
            content += msg.content or ""
    msg = parser.finish_stream()
    if msg:
        reasoning += msg.reasoning or ""
        content += msg.content or ""
    assert (reasoning or None, content or None) == expected


@pytest.mark.parametrize("width", [1, 7, 31, 1000])
@pytest.mark.parametrize("count", [1, 2])
def test_tools(width, count):
    raw = M + R + "Secret." + E + CALL * count + M + T + "Done" + E + S
    reasoning = get_parser("inkling")()
    tool = ToolParserManager.get_tool_parser("inkling")()
    r, c = reasoning.extract_reasoning(raw)
    assert r == "Secret."
    result = tool.extract_tool_calls(c)
    assert result.content == "Done"
    assert [(v["name"], json.loads(v["arguments"])) for v in result.tool_calls] == [
        ("read_file", {"path": "sample.txt"})
    ] * count
    previous = tool_text = content = secret = ""
    calls = []
    for pos in range(0, len(raw), width):
        chunk = raw[pos : pos + width]
        msg = reasoning.extract_reasoning_streaming(previous, previous + chunk, chunk)
        previous += chunk
        if not msg:
            continue
        secret += msg.reasoning or ""
        delta = msg.content or ""
        out = tool.extract_tool_calls_streaming(tool_text, tool_text + delta, delta)
        tool_text += delta
        if out:
            content += out.get("content") or ""
            calls.extend(out.get("tool_calls") or [])
    assert secret == "Secret."
    assert content == "Done"
    merged = {}
    for call in calls:
        fn = merged.setdefault(call["index"], {"name": "", "arguments": ""})
        for key, value in call["function"].items():
            fn[key] += value
    assert len(merged) == count
    assert all(
        v["name"] == "read_file"
        and json.loads(v["arguments"]) == {"path": "sample.txt"}
        for v in merged.values()
    )


@pytest.mark.parametrize(
    "payload", ["oops", '{"name":"f","args":', "[]", '{"args":{}}']
)
def test_malformed(payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + C + payload + E + M + T + "Answer" + E
    out = parser.extract_tool_calls(raw)
    assert not out.tools_called
    assert out.content == payload + "Answer"


def test_partial_delimiter_and_reset():
    parser = get_parser("inkling")()
    assert (
        parser.extract_reasoning_streaming("", "<|content_think", "<|content_think")
        is None
    )
    assert parser.finish_stream().content == "<|content_think"
    parser.reset_state()
    assert parser.extract_reasoning_streaming("", R + "x", R + "x").reasoning == "x"


def test_special_tokens_reach_parser():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens=True):
            assert skip_special_tokens is False
            return "".join([M, R, "Secret", E, M, T, "Answer", E][i] for i in ids)

    decoder = IncrementalDecoder(Tokenizer())
    text = "".join(decoder.add_token(i) for i in range(8))
    assert get_parser("inkling")().extract_reasoning(text) == ("Secret", "Answer")


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_routes(surface, stream):
    from tests.parser_migration.harness import route

    result = route(
        "inkling",
        "inkling",
        M + R + "Secret." + E + CALL + M + T + "Done" + E,
        surface,
        stream,
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
    )
    assert result["reasoning"] == "Secret."
    assert result["content"] == "Done"
    assert result["calls"] == [["read_file", {"path": "sample.txt"}]]


@pytest.mark.parametrize(
    "payload",
    [
        '{"name":"f","args":{"nested":{"name":"other","args":{}},"x":[1,true,null],"s":"é\\"}"}}',
        '{"args":{"x":1},"name":"f"}',
        '{"name":"f"}',
    ],
)
def test_argument_deltas(payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + "header_name" + C + payload + E
    deltas = []
    for i, char in enumerate(raw):
        result = parser.extract_tool_calls_streaming(raw[:i], raw[: i + 1], char)
        if result:
            deltas.extend(result.get("tool_calls", []))
    assert sum("id" in d for d in deltas) == 1
    assert deltas[0]["function"]["name"] == "f"
    args = "".join(d["function"].get("arguments", "") for d in deltas)
    assert json.loads(args) == json.loads(payload).get("args", {})
    assert all(d["index"] == 0 for d in deltas)
    if "nested" in payload:
        assert len(deltas) > 4


def test_token_stream_and_repeated_blocks():
    parser = get_parser("inkling")()
    previous = reasoning = content = ""
    for token in [M, R, "first", E, M, T, "one", E, M, R, "second", E, M, T, "two", S]:
        msg = parser.extract_reasoning_streaming(previous, previous + token, token)
        previous += token
        if msg:
            reasoning += msg.reasoning or ""
            content += msg.content or ""
    assert (reasoning, content) == ("firstsecond", "onetwo")


def test_tool_finalize_and_reset():
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = CALL[: -len(E)]
    parser.extract_tool_calls_streaming("", raw, raw)
    assert parser.finalize_legacy_raw_stream(raw) is None
    assert parser.payload is None
    parser.reset()
    result = parser.extract_tool_calls_streaming("", CALL, CALL)
    assert result["tool_calls"][0]["index"] == 0
    assert parser.extract_tool_calls("Plain prose.").content == "Plain prose."


def test_reasoning_whitespace():
    parser = get_parser("inkling")()
    assert parser.extract_reasoning(R + "one two \n" + E + T + "Answer") == (
        "one two",
        "Answer",
    )
    previous = reasoning = ""
    for char in R + "one two \n" + E:
        msg = parser.extract_reasoning_streaming(previous, previous + char, char)
        previous += char
        if msg:
            reasoning += msg.reasoning or ""
    assert reasoning == "one two"


@pytest.mark.parametrize(
    "payload", ["oops", "[]", '{"args":{}}', '{"name":"f","args":']
)
def test_malformed_streaming(payload):
    parser = ToolParserManager.get_tool_parser("inkling")()
    raw = M + C + payload + E
    content = ""
    for i, char in enumerate(raw):
        out = parser.extract_tool_calls_streaming(raw[:i], raw[: i + 1], char)
        if out:
            assert not out.get("tool_calls")
            content += out.get("content", "")
    assert content == payload
