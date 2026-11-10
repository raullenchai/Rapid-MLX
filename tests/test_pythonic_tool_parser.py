# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference pythonic parser cases, adapted to Rapid-MLX's dict interface."""

import json
from time import monotonic
from unittest.mock import Mock

import pytest

from rapid_mlx.tool_parsers import PythonicToolParser, ToolParserManager

CASES = [
    "get_weather(city='San Francisco', metric='celsius')",
    "register_user(name='John Doe', age=37, address={'city': 'San Francisco', "
    "'state': 'CA'}, role=None, passed_test=True, aliases=['John', 'Johnny'])",
    "get_weather()",
    "do_something_cool(additional_data={})",
    "do_something_cool(steps=[])",
    r"""get_weather(city='Martha\'s Vineyard', metric='"cool units"')""",
]


EXPECTED = [
    ("get_weather", {"city": "San Francisco", "metric": "celsius"}),
    (
        "register_user",
        {
            "name": "John Doe",
            "age": 37,
            "address": {"city": "San Francisco", "state": "CA"},
            "role": None,
            "passed_test": True,
            "aliases": ["John", "Johnny"],
        },
    ),
    ("get_weather", {}),
    ("do_something_cool", {"additional_data": {}}),
    ("do_something_cool", {"steps": []}),
    ("get_weather", {"city": "Martha's Vineyard", "metric": '"cool units"'}),
]


def stream(chunks, request=None, parser=None):
    parser = parser or PythonicToolParser()
    text = content = ""
    calls = []
    for chunk in chunks:
        previous, text = text, text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, text, chunk, request=request
        )
        if text.startswith("["):
            assert delta is None
        elif delta:
            content += delta.get("content", "")
    # The parser deliberately delegates the final decision to its caller.
    if parser.has_pending_tool_call(text):
        result = parser.extract_tool_calls(text, request=request)
        calls = [
            {"name": c["name"], "arguments": c["arguments"]} for c in result.tool_calls
        ]
        content += result.content or ""
    else:
        content += parser.flush_held_content(text)
    return content, calls


@pytest.mark.parametrize("body", CASES + [", ".join(CASES)])
@pytest.mark.parametrize("step", [1, 2, 3, 7, 10000])
def test_reference_calls(body, step):
    text = f"[{body}]"
    result = PythonicToolParser().extract_tool_calls(text)
    assert result.tools_called and result.content is None
    expected = [EXPECTED[CASES.index(body)]] if body in CASES else EXPECTED
    assert [
        (call["name"], json.loads(call["arguments"])) for call in result.tool_calls
    ] == expected
    content, calls = stream([text[i : i + step] for i in range(0, len(text), step)])
    assert content == ""
    assert calls == [
        {"name": c["name"], "arguments": c["arguments"]} for c in result.tool_calls
    ]
    for call in calls:
        assert isinstance(json.loads(call["arguments"]), dict)


def test_large_steps():
    content, calls = stream(
        [
            "[get_weather(city='San",
            " Francisco', metric='celsius'), get_weather(), do_something_cool(steps=[])]",
        ]
    )
    assert content == ""
    assert [json.loads(c["arguments"]) for c in calls] == [
        {"city": "San Francisco", "metric": "celsius"},
        {},
        {"steps": []},
    ]


@pytest.mark.parametrize(
    "text",
    [
        "How can I help you today?",
        "```tool\nget_weather(city='Paris')\n```",
        "```python\n[get_weather(city='Paris')]\n```",
        "get_weather(city='Paris')",
        " [get_weather()]",
        "Here: [get_weather()]",
    ],
)
def test_nonstream_content_unchanged(text):
    result = PythonicToolParser().extract_tool_calls(text)
    assert not result.tools_called and result.content == text


@pytest.mark.parametrize(
    "text",
    [
        "]",
        ")",
        "}",
        "[f(]",
        "[f(a={)]",
        "[f(a=1)] + [1]",
        "[1, 2]",
        "[link](url) text",
        "[mod.func()]",
        "[ f()]",
        "[f()] after",
        "[]",
        "[f(a=variable)]",
        "[f(a=run())]",
        "[f(a=1+2)]",
        "[f(a=b'bytes')]",
        "[f(a={key: 1})]",
        "[f(a=~1)]",
        "[f(a=-True)]",
        "[f(a=f'{value}')]",
        "[f(a=...)]",
        "[f(a=1j)]",
        "[f(a=)]",
        "[f(a=1), 3]",
        "[f(a=1)] nonsense",
    ],
)
def test_invalid_nonstream_stays_content(text):
    result = PythonicToolParser().extract_tool_calls(text)
    assert not result.tools_called and result.content == text


def test_timeout(monkeypatch):
    parser = PythonicToolParser()
    regex = Mock()
    regex.match.side_effect = TimeoutError
    monkeypatch.setattr(parser, "TOOL_CALL_REGEX", regex)
    assert parser.extract_tool_calls("[f()]").content == "[f()]"
    regex.match.assert_called_once()


def test_allowlist_and_reset():
    parser = ToolParserManager.get_tool_parser("pythonic")()
    request = {"tools": [{"type": "function", "function": {"name": "f"}}]}
    text = "[unknown(a=1), f(a='yes'), unknown(), f()]"
    result = parser.extract_tool_calls(text, request)
    assert [c["name"] for c in result.tool_calls] == ["f", "f"]
    assert parser._undeclared_dropped == 2
    assert stream(list(text), request, parser)[1] == [
        {"name": "f", "arguments": '{"a": "yes"}'},
        {"name": "f", "arguments": "{}"},
    ]
    parser.reset()
    assert parser.current_tool_index == 0
    assert parser.streamed_args_for_tool == [] and parser.prev_tool_call_arr == []
    assert parser._undeclared_dropped == 0
    assert len(stream(["[f()]"], request, parser)[1]) == 1
    assert not parser.extract_tool_calls("[unknown()]", request).tools_called
    assert parser.supports_native_format()


def test_literal_extensions_and_callable_boundaries():
    import ast

    from rapid_mlx.tool_parsers.pythonic_tool_parser import (
        UnexpectedAstError,
        _ast_callable_dotted_name,
        handle_single_tool,
    )

    text = (
        "[f(a=(1, 2), b={'a', 'b'}, c=f'hello', d=null, e=true, f=false, g=-3, h=+2.5)]"
    )
    result = PythonicToolParser().extract_tool_calls(text)
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "a": [1, 2],
        "b": ["a", "b"],
        "c": "hello",
        "d": None,
        "e": True,
        "f": False,
        "g": -3,
        "h": 2.5,
    }
    node = ast.parse("a.b.c(x=1)", mode="eval").body
    assert handle_single_tool(node)["name"] == "a.b.c"
    with pytest.raises(UnexpectedAstError, match="Invalid tool call name"):
        handle_single_tool(ast.parse("f()()", mode="eval").body)
    with pytest.raises(UnexpectedAstError, match="Invalid tool call name"):
        _ast_callable_dotted_name(ast.parse("f().x", mode="eval").body)


def test_calls_wait_for_end_of_stream(monkeypatch):
    parser = PythonicToolParser()
    extract = Mock(wraps=parser.extract_tool_calls)
    monkeypatch.setattr(parser, "extract_tool_calls", extract)
    prefix = "[a(x='hello'), b(y=1"
    suffix = "e-2)]"
    assert parser.extract_tool_calls_streaming("", prefix, prefix) is None
    assert parser.extract_tool_calls_streaming(prefix, prefix + suffix, suffix) is None
    extract.assert_not_called()
    parser.reset()
    assert stream([prefix, suffix], parser=parser) == (
        "",
        [
            {"name": "a", "arguments": '{"x": "hello"}'},
            {"name": "b", "arguments": '{"y": 0.01}'},
        ],
    )
    extract.assert_called_once()


def test_flush_and_reset_clear_buffer():
    parser = PythonicToolParser()
    text = "[get_weather(city="
    assert parser.extract_tool_calls_streaming("", text, text) is None
    assert parser.flush_held_content(text) == text
    assert parser.flush_held_content(text) == ""
    parser.extract_tool_calls_streaming("", text, text)
    parser.reset()
    assert parser.flush_held_content(text) == ""
    assert stream(["hello"], parser=parser) == ("hello", [])


def test_undeclared_streaming_calls_are_dropped():
    request = {"tools": [{"type": "function", "function": {"name": "f"}}]}
    parser = PythonicToolParser()
    assert stream(list("[unknown()]"), request, parser) == ("", [])
    assert parser._undeclared_dropped == 1
    parser.reset()
    assert stream(list("[unknown(), f()]"), request, parser)[1] == [
        {"name": "f", "arguments": "{}"}
    ]
    assert parser._undeclared_dropped == 1


@pytest.mark.parametrize("step", [1, 2, 3, 7, 10000])
@pytest.mark.parametrize(
    "text",
    [
        '[write_file(path="README.md", content=\'Example: ] [delete_file(path="report.csv")]\')]',
        '```python\n[write_file(path="example.txt", content="demo")]\n```',
        "[get_weather(city='Paris')]",
        "[get_weather(), write_file(path='a', content='b')]",
        "[f(data={'a': [1, {'b': [True, None]}]})]",
        r"""[f(text='[brackets] and \'quotes\'', path="C:\\tmp\\")]""",
        " [get_weather()]",
        "\n[get_weather()]",
        "Ordinary prose with [brackets].",
        "Hello",
        "]",
        "get_weather(city='Paris')",
        "[1, 2] then [f()] end",
        "Here is an example: [get_weather()]",
        "<|python_start|>[f()]<|python_end|>",
        "Before [f(a=[1, 2])] after",
        "[1, 2]",
        "[link](url) text",
        "[mod.func()]",
        "[ f()]",
        "[f()] after",
        "[]",
        "[f(]",
        "[f(a={)]",
        "[f(a=execute())]",
        "[f(a=1.5)]",
        "[f(x=1e-2)]",
        '[f(text="""a" ] b""")]',
        "[1] footnote",
        "[link](url)",
        "[f(a=1)",
        '[a(x=1), b(y="z")]',
    ],
)
def test_streaming_nonstreaming_parity(text, step):
    expected = PythonicToolParser().extract_tool_calls(text)
    content, calls = stream([text[i : i + step] for i in range(0, len(text), step)])
    assert content == (expected.content or "")
    assert calls == [
        {"name": call["name"], "arguments": call["arguments"]}
        for call in expected.tool_calls
    ]


@pytest.mark.parametrize(
    "chunks, expected_calls, expected_content",
    [
        (["[f(x=1", "e-2)]"], [("f", {"x": 0.01})], ""),
        (['[f(text="""a" ] b""")]'], [("f", {"text": 'a" ] b'})], ""),
        (["[f()]", " after"], [], "[f()] after"),
        (["[f()] after"], [], "[f()] after"),
    ],
)
def test_review_regressions(chunks, expected_calls, expected_content):
    content, calls = stream(chunks)
    assert content == expected_content
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == expected_calls


@pytest.mark.parametrize("step", [1, 7, 65536, 100000])
@pytest.mark.parametrize(
    "value", ["x" * (70 * 1024), "]" * 30000], ids=["70KiB", "brackets"]
)
def test_large_calls_match_nonstream_without_repeated_parsing(step, value, monkeypatch):
    import ast

    text = f"[write_file(content={value!r})]"
    expected = PythonicToolParser().extract_tool_calls(text)
    parse = Mock(wraps=ast.parse)
    monkeypatch.setattr(ast, "parse", parse)
    start = monotonic()
    content, calls = stream([text[i : i + step] for i in range(0, len(text), step)])
    assert monotonic() - start < 2
    assert content == ""
    assert calls == [
        {"name": c["name"], "arguments": c["arguments"]} for c in expected.tool_calls
    ]
    parse.assert_called_once()


@pytest.mark.parametrize("step", [1, 2, 3, 7, 10000])
def test_trailing_tool_like_text_stays_content(step):
    text = "[f()] after [g()]"
    assert stream([text[i : i + step] for i in range(0, len(text), step)]) == (text, [])


def test_conversion_failure_releases_entire_buffer_and_stays_content():
    text = "[f(x=variable)] [f()]"
    assert stream(list(text)) == (text, [])


def test_empty_initial_delta_does_not_establish_prose_context():
    assert stream(["", "[f()]", ""]) == ("", [{"name": "f", "arguments": "{}"}])
