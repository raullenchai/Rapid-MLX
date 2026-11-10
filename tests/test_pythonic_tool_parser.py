# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference pythonic parser cases, adapted to Rapid-MLX's dict interface."""

import json
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
    calls = {}
    for chunk in chunks:
        previous, text = text, text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous, text, chunk, request=request
        )
        if delta is None:
            continue
        content += delta.get("content", "")
        for call in delta.get("tool_calls", []):
            index = call["index"]
            fn = call["function"]
            if index not in calls:
                assert call["id"].startswith("call_")
                assert call["type"] == "function"
                calls[index] = {"name": fn["name"], "arguments": ""}
            else:
                assert "name" not in fn
                assert "id" not in call
            calls[index]["arguments"] += fn["arguments"]
    content += parser.flush_held_content(text)
    return content, list(calls.values())


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


def test_calls_wait_for_complete_list():
    parser = PythonicToolParser()
    prefix = "[a(x='hello'), b(y=1"
    assert parser.extract_tool_calls_streaming("", prefix, prefix) is None
    suffix = "e-2)]"
    last = parser.extract_tool_calls_streaming(prefix, prefix + suffix, suffix)
    calls = last["tool_calls"]
    assert [c["index"] for c in calls] == [0, 1]
    assert len({c["id"] for c in calls}) == 2
    assert all(c["type"] == "function" for c in calls)
    assert [c["function"] for c in calls] == [
        {"name": "a", "arguments": '{"x": "hello"}'},
        {"name": "b", "arguments": '{"y": 0.01}'},
    ]
    assert parser.flush_held_content(prefix + suffix) == ""


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
        (["[f()]", " after"], [("f", {})], " after"),
        (["[f()] after"], [("f", {})], " after"),
    ],
)
def test_review_regressions(chunks, expected_calls, expected_content):
    content, calls = stream(chunks)
    assert content == expected_content
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == expected_calls


def test_closing_brackets_are_tried_once(monkeypatch):
    import ast

    parse = Mock(wraps=ast.parse)
    monkeypatch.setattr(ast, "parse", parse)
    parser = PythonicToolParser()
    prefix = "[f(x=[1]"
    assert parser.extract_tool_calls_streaming("", prefix, prefix) is None
    assert parse.call_count == 1
    middle = ", y="
    assert parser.extract_tool_calls_streaming(prefix, prefix + middle, middle) is None
    assert parse.call_count == 1
    final = "2)]"
    delta = parser.extract_tool_calls_streaming(
        prefix + middle, prefix + middle + final, final
    )
    assert parse.call_count == 2
    assert json.loads(delta["tool_calls"][0]["function"]["arguments"]) == {
        "x": [1],
        "y": 2,
    }


@pytest.mark.parametrize("character", ["x", "🧪"])
def test_buffer_byte_limit_releases_once_and_stays_content(character):
    parser = PythonicToolParser()
    prefix = "[f(text='" + character * (
        (64 * 1024 - 9) // len(character.encode("utf-8"))
    )
    assert len(prefix.encode("utf-8")) <= 64 * 1024
    assert parser.extract_tool_calls_streaming("", prefix, prefix) is None
    extra = "xxxx"
    assert len((prefix + extra).encode("utf-8")) > 64 * 1024
    assert parser.extract_tool_calls_streaming(prefix, prefix + extra, extra) == {
        "content": prefix + extra
    }
    assert parser.flush_held_content(prefix + extra) == ""
    suffix = "')] [f()]"
    assert parser.extract_tool_calls_streaming(
        prefix + extra, prefix + extra + suffix, suffix
    ) == {"content": suffix}
    parser.reset()
    assert stream(["[f()]"], parser=parser)[1] == [{"name": "f", "arguments": "{}"}]


@pytest.mark.parametrize("step", [1, 2, 3, 7, 10000])
def test_trailing_tool_like_text_stays_content(step):
    text = "[f()] after [g()]"
    content, calls = stream([text[i : i + step] for i in range(0, len(text), step)])
    assert content == " after [g()]"
    assert calls == [{"name": "f", "arguments": "{}"}]


def test_conversion_failure_releases_entire_buffer_and_stays_content():
    text = "[f(x=variable)] [f()]"
    assert stream(list(text)) == (text, [])


def test_empty_initial_delta_does_not_establish_prose_context():
    assert stream(["", "[f()]", ""]) == ("", [{"name": "f", "arguments": "{}"}])
