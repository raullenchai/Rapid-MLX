# SPDX-License-Identifier: Apache-2.0
"""Declared XML calls inside Markdown are examples, including at stream EOF."""

import pytest

from rapid_mlx.config import get_config
from rapid_mlx.service import helpers
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from tests.qwen3coder_stream_harness import non_stream, stream
from tests.test_tool_choice_enforcement import _make_client, _RecordingEngine

CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
    "</parameter>\n</function>\n</tool_call>"
)
BARE = CALL.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
REQUEST = {
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ],
}
EXAMPLE = f"Here is the format:\n```xml\n{CALL}\n```\n"


def test_first_visible_opener_without_markup():
    parser = Qwen3CoderToolParser(None)
    assert parser._first_opener_pos("") == 0
    assert parser._first_opener_pos("plain") == len("plain")


@pytest.mark.parametrize("size", [None, 0, 1, 2, 7, 13, 10000])
@pytest.mark.parametrize(
    "text, content, calls",
    [
        (EXAMPLE, EXAMPLE, 0),
        (f"```xml\n{CALL}\n```", f"```xml\n{CALL}\n```", 0),
        (f"Example:\n~~~xml\n{CALL}\n~~~\n", f"Example:\n~~~xml\n{CALL}\n~~~\n", 0),
        (f"Example:\n```xml\n{BARE}\n```\n", f"Example:\n```xml\n{BARE}\n```\n", 0),
        (f"Example: `{BARE}`", f"Example: `{BARE}`", 0),
        (f"Example:\n```xml\n{CALL}", f"Example:\n```xml\n{CALL}", 0),
        (
            "Example:\n```xml\n<tool_call>\n<function=get_weather>",
            "Example:\n```xml\n<tool_call>\n<function=get_weather>",
            0,
        ),
        (f"```xml\nx\n```\n{CALL}", "```xml\nx\n```\n", 1),
        (EXAMPLE + CALL, EXAMPLE, 1),
        (EXAMPLE + CALL + CALL, EXAMPLE, 2),
        (CALL, "", 1),
    ],
)
def test_markdown_boundary(text, content, calls, size):
    got_calls, got_content = (
        non_stream(text, REQUEST) if size is None else stream(text, REQUEST, size)
    )
    assert got_calls == [("get_weather", '{"city": "Paris"}')] * calls
    assert got_content == content


def test_service_does_not_reextract_example(monkeypatch):
    class Request:
        tools = REQUEST["tools"]

        def model_dump(self):
            return REQUEST

    cfg = get_config()
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", True)
    monkeypatch.setattr(cfg, "tool_call_parser", "qwen3_coder_xml")
    monkeypatch.setattr(
        helpers,
        "parse_tool_calls",
        lambda *a, **kw: pytest.fail("generic fallback re-extracted the example"),
    )
    content, calls = helpers._parse_tool_calls_with_parser(EXAMPLE, Request())
    assert content == EXAMPLE
    assert not calls


@pytest.mark.parametrize("real_call", [False, True])
def test_chat_completions_preserves_example(real_call):
    # The route strips outer whitespace from all assistant responses.
    example = EXAMPLE.rstrip()
    client = _make_client(
        _RecordingEngine(text=example + ("\n" + CALL if real_call else "")),
        tool_call_parser="qwen3_coder_xml",
    )
    get_config().enable_auto_tool_choice = True
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Show the XML format."}],
            "enable_thinking": False,
            **REQUEST,
        },
    )
    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert message["content"] == example
    assert len(message.get("tool_calls") or []) == int(real_call)
