"""Pythonic calls arrive whole; preceding prose stays literal."""

import pytest

from .harness import route


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("step", [1, 2, 3, 7, 10000])
@pytest.mark.parametrize(
    "raw, content, calls",
    [
        ("[f(a=1.5)]", "", [["f", {"a": 1.5}]]),
        ("[f(x=1e-2)]", "", [["f", {"x": 0.01}]]),
        ('[f(text="""a" ] b""")]', "", [["f", {"text": 'a" ] b'}]]),
        ("[f()] after", "[f()] after", []),
        (
            "[f(data={'a': [1, {'b': [True, None]}]})]",
            "",
            [["f", {"data": {"a": [1, {"b": [True, None]}]}}]],
        ),
        ("[1] footnote", "[1] footnote", []),
        ("[link](url)", "[link](url)", []),
        ("[f(a=1)", "[f(a=1)", []),
        ("Ordinary prose.", "Ordinary prose.", []),
        (
            '[write_file(path="README.md", content=\'Example: ] [delete_file(path="report.csv")]\')]',
            "",
            [
                [
                    "write_file",
                    {
                        "path": "README.md",
                        "content": 'Example: ] [delete_file(path="report.csv")]',
                    },
                ]
            ],
        ),
        (
            '```python\n[write_file(path="example.txt", content="demo")]\n```',
            '```python\n[write_file(path="example.txt", content="demo")]\n```',
            [],
        ),
        ("[1, 2]", "[1, 2]", []),
        ("[link](url) text", "[link](url) text", []),
        ("[]", "[]", []),
        ("[mod.func()]", "[mod.func()]", []),
        ("[ f()]", "[ f()]", []),
        ("[unknown()]", "", []),
        ("[unknown(), f()]", "", [["f", {}]]),
        ("[a(x=1), b(y='z')]", "", [["a", {"x": 1}], ["b", {"y": "z"}]]),
    ],
)
def test_pythonic_streaming_segments(surface, step, raw, content, calls):
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ["f", "get_weather", "a", "b", "write_file", "delete_file"]
    ]
    actual = route(
        None,
        "pythonic",
        raw,
        surface,
        True,
        chunks=[raw[i : i + step] for i in range(0, len(raw), step)],
        tools=tools,
    )
    # Chat already surfaces the request allowlist rejection as a user message.
    if raw == "[unknown()]" and surface == "chat":
        content = (
            "I couldn't complete the request because a needed tool isn't available."
        )
    assert actual.get("content") == content, actual
    assert actual.get("calls") == calls, actual


@pytest.mark.parametrize("step", [1, 2, 3, 7, 10000])
@pytest.mark.parametrize(
    "raw, content, calls",
    [
        (
            "Sure: [get_weather(city='Paris')]",
            "Sure: [get_weather(city='Paris')]",
            [],
        ),
        ("Before [f()] after", "Before [f()] after", []),
        ("Before [f()] [unterminated", "Before [f()] [unterminated", []),
    ],
)
def test_chat_pythonic_streaming_mixed_content(step, raw, content, calls):
    test_pythonic_streaming_segments("chat", step, raw, content, calls)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("step", [1, 4096, 100000])
@pytest.mark.parametrize(
    "value", ["x" * (70 * 1024), "]" * 30000], ids=["70KiB", "brackets"]
)
def test_large_pythonic_calls_through_streaming_adapters(surface, step, value):
    import json

    from rapid_mlx.tool_parsers import PythonicToolParser

    raw = f"[write_file(content={value!r})]"
    expected = PythonicToolParser().extract_tool_calls(raw)
    actual = route(
        None,
        "pythonic",
        raw,
        surface,
        True,
        chunks=[raw[i : i + step] for i in range(0, len(raw), step)],
        tools=[
            {
                "type": "function",
                "function": {"name": "write_file", "parameters": {"type": "object"}},
            }
        ],
    )
    assert actual["content"] == (expected.content or "")
    assert actual["calls"] == [
        [c["name"], json.loads(c["arguments"])] for c in expected.tool_calls
    ]
