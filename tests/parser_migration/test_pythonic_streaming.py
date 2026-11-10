"""Pythonic calls stream incrementally; preceding prose stays literal."""

import pytest

from .harness import route


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("step", [1, 3, 7, 10000])
@pytest.mark.parametrize(
    "raw, content, calls",
    [
        ("[f(a=1.5)]", "", [["f", {"a": 1.5}]]),
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
    assert actual.get("content") == content, actual
    assert actual.get("calls") == calls, actual


@pytest.mark.parametrize("step", [1, 3, 7, 10000])
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
