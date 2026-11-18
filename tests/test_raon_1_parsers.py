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
