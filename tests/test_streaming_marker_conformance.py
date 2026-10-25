# SPDX-License-Identifier: Apache-2.0
"""Decoded marker boundaries must not change visible text or structured calls."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser
from rapid_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from rapid_mlx.routes import anthropic, chat, responses
from tests.test_rejected_tool_diagnostics import Engine as BaseEngine
from tests.test_tool_call_streaming_parity import PARITY_FIXTURES


class Engine(BaseEngine):
    def __init__(self, raw, chunks):
        super().__init__(raw)
        self.chunks = chunks

    async def stream_chat(self, **kwargs):
        text = ""
        for chunk in self.chunks:
            text += chunk
            yield GenerationOutput(
                text=text,
                new_text=chunk,
                finished=False,
                prompt_tokens=4,
                completion_tokens=len(text),
            )
        yield GenerationOutput(
            text=text,
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=len(text),
        )


def partitions(raw):
    yield [raw]
    yield list(raw)
    for i in range(1, len(raw)):
        yield [raw[:i], raw[i:]]


def request_result(
    surface, raw, chunks, parser=None, reasoning=None, names=(), body_updates=None
):
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.no_thinking = reasoning is None
    cfg.reasoning_parser_name = reasoning
    cfg.reasoning_parser = (
        {"gemma4": Gemma4ReasoningParser, "harmony": HarmonyReasoningParser}[
            reasoning
        ]()
        if reasoning
        else None
    )
    cfg.tool_call_parser = parser
    cfg.enable_auto_tool_choice = bool(parser)
    cfg.engine = Engine(raw, chunks)
    app = FastAPI()
    app.include_router(
        {"chat": chat, "responses": responses, "messages": anthropic}[surface].router
    )
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in names
    ]
    body = dict(model="test-model", stream=True, max_tokens=2048)
    if tools:
        body["tools"] = tools
    if surface == "responses":
        body["input"] = "Test"
        if tools:
            body["tools"] = [{"type": "function", **t["function"]} for t in tools]
        url = "/v1/responses"
    else:
        body["messages"] = [{"role": "user", "content": "Test"}]
        url = "/v1/chat/completions" if surface == "chat" else "/v1/messages"
        if surface == "messages" and tools:
            body["tools"] = [
                {"name": n, "input_schema": {"type": "object"}} for n in names
            ]
    if reasoning:
        body["chat_template_kwargs"] = {"enable_thinking": True}
        if surface == "messages":
            body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    if body_updates:
        body.update(body_updates)
    try:
        reply = TestClient(app).post(url, json=body)
        assert reply.status_code == 200, reply.text
        packets = [
            json.loads(line[6:])
            for line in reply.text.splitlines()
            if line.startswith("data: {")
        ]
        text, thought, calls = "", "", {}
        for packet in packets:
            if surface == "chat":
                if not packet.get("choices"):
                    continue
                delta = packet["choices"][0].get("delta", {})
                text += delta.get("content") or ""
                thought += delta.get("reasoning_content") or ""
                for call in delta.get("tool_calls", []):
                    item = calls.setdefault(call["index"], ["", ""])
                    item[0] += call.get("function", {}).get("name") or ""
                    item[1] += call.get("function", {}).get("arguments") or ""
            elif surface == "responses":
                if packet["type"] == "response.output_text.delta":
                    text += packet["delta"]
                elif packet["type"] == "response.reasoning_summary_text.delta":
                    thought += packet["delta"]
                elif (
                    packet["type"] == "response.output_item.done"
                    and packet["item"]["type"] == "function_call"
                ):
                    item = packet["item"]
                    calls[item["id"]] = [item["name"], item["arguments"]]
            elif packet["type"] == "content_block_start":
                block = packet["content_block"]
                if block["type"] == "tool_use":
                    calls[packet["index"]] = [block["name"], ""]
            elif packet["type"] == "content_block_delta":
                delta = packet["delta"]
                text += delta.get("text", "")
                thought += delta.get("thinking", "")
                if "partial_json" in delta:
                    calls[packet["index"]][1] += delta["partial_json"]
        return (
            text,
            thought,
            [(name, json.loads(args or "{}")) for name, args in calls.values()],
        )
    finally:
        reset_config()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "marker",
    [
        "<|im_end|>",
        "<|assistant|>",
        "<|endoftext|>",
        "<|channel|>",
        "<|call|>",
        "<|eot_id|>",
    ],
)
def test_special_tokens(surface, marker):
    raw = "Answer" + marker
    for chunks in partitions(raw):
        assert request_result(surface, raw, chunks) == ("Answer", "", []), chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "reasoning,raw",
    [
        ("gemma4", "<|channel>thought\nSecret.<channel|>Answer"),
        (
            "harmony",
            "<|start|>assistant<|channel|>analysis<|message|>Secret.<|end|><|start|>assistant<|channel|>final<|message|>Answer<|return|>",
        ),
    ],
)
def test_channels(surface, reasoning, raw):
    for chunks in partitions(raw):
        assert request_result(surface, raw, chunks, reasoning=reasoning) == (
            "Answer",
            "Secret.",
            [],
        ), chunks


NATIVE = [entry for entry in PARITY_FIXTURES if isinstance(entry, tuple)] + [
    (
        "granite",
        "granite",
        '<|tool_call|>[{"name":"read_file","arguments":{"path":"sample.txt"}}]',
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "functionary",
        "functionary",
        '<|from|>assistant\n<|recipient|>read_file\n<|content|>{"path":"sample.txt"}',
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "deepseek_v3",
        "deepseek",
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\n```json\n{"path":"sample.txt"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "kimi",
        "kimi",
        '<|tool_calls_section_begin|><|tool_call_begin|>read_file:0<|tool_call_argument_begin|>{"path":"sample.txt"}<|tool_call_end|><|tool_calls_section_end|>',
        [("read_file", {"path": "sample.txt"})],
    ),
]


NATIVE = [
    (
        p,
        label,
        ("Action: type(content='hello')" if p == "ui_tars" else r),
        (
            [("computer", {"action": "type", "content": "hello"})]
            if p == "ui_tars"
            else e
        ),
    )
    for p, label, r, e in NATIVE
]

NATIVE += [
    (
        "gemma4",
        "gemma4",
        '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>',
        [("get_weather", {"city": "Paris"})],
    ),
    (
        "harmony",
        "harmony",
        '<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city":"Paris"}<|call|>',
        [("get_weather", {"city": "Paris"})],
    ),
    (
        "mistral",
        "mistral",
        '[TOOL_CALLS][{"name":"read_file","arguments":{"path":"sample.txt"}}]',
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "llama",
        "llama",
        '<|python_tag|>{"name":"read_file","parameters":{"path":"sample.txt"}}',
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "seed_oss",
        "seed",
        "<seed:tool_call><function=read_file><parameter=path>sample.txt</parameter></function></seed:tool_call>",
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "k2_horizon",
        "k2",
        "<ifm|tool_calls><ifm|tool_call>ping</ifm|tool_call></ifm|tool_calls>",
        [("ping", {})],
    ),
    (
        "minimax",
        "minimax",
        '<minimax:tool_call><invoke name="read_file"><parameter name="path">sample.txt</parameter></invoke></minimax:tool_call>',
        [("read_file", {"path": "sample.txt"})],
    ),
    (
        "deepseek_v31",
        "v31",
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file<｜tool▁sep｜>{"path":"sample.txt"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        [("read_file", {"path": "sample.txt"})],
    ),
]
NATIVE += [
    ("deepseek", label, raw, expected)
    for parser, label, raw, expected in NATIVE
    if parser == "deepseek_v3"
]
NATIVE += [
    ("auto", label, raw, expected)
    for parser, label, raw, expected in NATIVE
    if parser == "hermes"
]


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("parser,label,raw,expected", NATIVE)
def test_native_tools(surface, parser, label, raw, expected):
    for chunks in partitions(raw):
        result = request_result(
            surface,
            raw,
            chunks,
            parser=parser,
            reasoning="harmony" if parser == "harmony" else None,
            names=[name for name, _ in expected],
        )
        assert result == ("Checking. " if parser == "lfm" else "", "", expected), (
            chunks,
            result,
        )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "raw",
    [
        "A < B",
        "plain <",
        "ordinary [text]",
        "literal <tool_call> example",
        "Answer<|im_en",
    ],
)
def test_literal_and_truncated_text(surface, raw):
    for chunks in partitions(raw):
        assert request_result(surface, raw, chunks)[0] == raw


@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize(
    "parser", ["gemma4", "hy_v3", "muse", "kimi", "deepseek_v3", "minicpm", "seed_oss"]
)
def test_native_prose(surface, parser):
    _, _, wire, expected = next(row for row in NATIVE if row[0] == parser)
    raw = "Before. " + wire + " After."
    result = request_result(
        surface, raw, list(raw), parser=parser, names=[name for name, _ in expected]
    )
    assert result[0] == "Before.  After.", result
    assert result[2] == expected


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_closer_inside_native_arguments(surface):
    args = {"path": "literal <end_of_tool_call:opensource> remains data"}
    raw = (
        "<tool_call:opensource>read_file<tool_sep:opensource>"
        + json.dumps(args)
        + "<end_of_tool_call:opensource>"
    )
    for chunks in partitions(raw):
        result = request_result(
            surface, raw, chunks, parser="hy_v3", names=["read_file"]
        )
        assert result == ("", "", [("read_file", args)]), chunks


def test_wire_recognition_contract_complete():
    from rapid_mlx.tool_parsers.abstract_tool_parser import WIRE_FORMAT_LABELS
    from rapid_mlx.tool_parsers.streaming import WIRE_OPENERS

    assert set(WIRE_OPENERS) == WIRE_FORMAT_LABELS


@pytest.mark.parametrize(
    "parser_type,header",
    [
        (Gemma4ReasoningParser, "<|channel>thought\n"),
        (Gemma4ReasoningParser, "<|channel>final\n"),
        (HarmonyReasoningParser, "<|channel|>analysis<|message|>"),
        (HarmonyReasoningParser, "<|channel|>final<|message|>"),
        (HarmonyReasoningParser, "<|channel|>commentary<|message|>"),
    ],
)
@pytest.mark.parametrize("suffix", ["<", "<|cha", ""])
def test_channel_eof(parser_type, header, suffix):
    parser = parser_type()
    previous = ""
    for char in header + "Body" + suffix:
        parser.extract_reasoning_streaming(previous, previous + char, char)
        previous += char
    final = parser.finalize_streaming(previous, finish_reason="length")
    if suffix == "<":
        assert final and (final.reasoning or final.content) == "<"
    else:
        assert final is None
    assert parser.finalize_streaming(previous) is None
    parser.reset_state()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("parser", ["gemma4", "hy_v3", "muse"])
def test_repeated_native_calls(surface, parser):
    _, _, wire, expected = next(row for row in NATIVE if row[0] == parser)
    raw = wire + wire
    for chunks in ([raw], list(raw)):
        result = request_result(
            surface, raw, chunks, parser=parser, names=[name for name, _ in expected]
        )
        assert result == ("", "", expected + expected)


@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize(
    "marker", ["<s>", "</s>", "<pad>", "[PAD]", "[SEP]", "[CLS]", "[e~[", "]~!b["]
)
def test_legacy_special_tokens(surface, marker):
    raw = "Answer" + marker
    for chunks in partitions(raw):
        assert request_result(surface, raw, chunks)[0] == "Answer"


@pytest.mark.parametrize("surface", ["responses", "messages"])
def test_gemma_closer_inside_string(surface):
    value = "literal <tool_call|> remains data"
    raw = '<|tool_call>call:get_weather{city:<|"|>' + value + '<|"|>}<tool_call|>'
    for chunks in partitions(raw):
        result = request_result(
            surface, raw, chunks, parser="gemma4", names=["get_weather"]
        )
        assert result == ("", "", [("get_weather", {"city": value})]), chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_coalesced_gemma_channel_transitions(surface):
    raw = "<|channel>thought\nSecret.<|channel>final\nAnswer"
    for chunks in partitions(raw):
        assert request_result(surface, raw, chunks, reasoning="gemma4") == (
            "Answer",
            "Secret.",
            [],
        )


def test_delayed_literal_content_keeps_order_when_reasoning_resumes(monkeypatch):
    from rapid_mlx.api.utils import StreamingReasoningSanitizer

    class WithPendingContent(StreamingReasoningSanitizer):
        def __init__(self):
            super().__init__()
            # The sanitizer owns text until its destination and marker status
            # resolve. Exercise its supported content-release contract.
            assert super().process("[Calli", "reasoning_overflow") == []

    monkeypatch.setattr(responses, "StreamingReasoningSanitizer", WithPendingContent)
    raw = "<|channel>thought\nXSecret.<channel|>Answer"
    assert (
        request_result("responses", raw, [raw], reasoning="gemma4")[0] == "[CalliAnswer"
    )


def test_messages_terminal_parser_release_uses_stream_sanitizer(monkeypatch):
    from rapid_mlx import reasoning
    from rapid_mlx.reasoning.base import DeltaMessage

    class TerminalParser(Gemma4ReasoningParser):
        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            if delta_text == "</think>":
                return DeltaMessage(content="Answer<|im_end|>")
            return DeltaMessage(reasoning=delta_text)

    monkeypatch.setattr(reasoning, "get_parser", lambda name: TerminalParser)
    result = request_result(
        "messages",
        "abcd",
        ["abcd"],
        reasoning="gemma4",
        body_updates={"thinking": {"type": "enabled", "budget_tokens": 1}},
    )
    assert result[0] == "Answer"
    assert result[1] == "abcd"


@pytest.mark.parametrize("parser", ["hermes", "gemma4", "auto"])
def test_eof_marker_tail_follows_older_sanitizer_tail(parser):
    raw = "[Calli<"
    for chunks in partitions(raw):
        assert request_result(
            "chat", raw, chunks, parser=parser, names=["get_weather"]
        ) == (raw, "", [])


def test_post_call_marker_suffix_uses_parser_eof_content_policy():
    from tests.test_postprocessor import StreamingPostProcessor, _make_cfg, _make_output

    literal = '<tool_call>{"example":"literal"}</tool_call>Tail <'
    wire = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>' + literal
    request = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
    for chunks in partitions(wire):
        processor = StreamingPostProcessor(
            _make_cfg(enable_auto_tool_choice=True, tool_call_parser="hermes"),
            request=request,
        )
        events = []
        for chunk in chunks:
            events.extend(processor.process_chunk(_make_output(chunk)))
        events.extend(processor.finalize())
        assert "".join(event.content or "" for event in events) == literal
        calls = [call for event in events for call in event.tool_calls or []]
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == "{}"


@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize(
    "parser,suffix", [("granite", "<|too"), ("granite", "<"), ("mistral", "[TOOL_")]
)
def test_post_call_partial_native_opener_stays_private(surface, parser, suffix):
    _, _, wire, expected = next(row for row in NATIVE if row[0] == parser)
    raw = wire + suffix
    for chunks in partitions(raw):
        assert request_result(
            surface, raw, chunks, parser=parser, names=[name for name, _ in expected]
        ) == ("", "", expected)


def test_tool_content_filter_preserves_parser_owned_eof_suffix():
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
    from rapid_mlx.tool_parsers.streaming import StreamingToolContentFilter

    request = {"tools": [{"type": "function", "function": {"name": "get_weather"}}]}
    literal = '<tool_call>{"example":"literal"}</tool_call>Tail <'
    wire = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>' + literal
    for chunks in partitions(wire):
        content_filter = StreamingToolContentFilter(HermesToolParser(), request)
        visible = "".join(content_filter.process(chunk) for chunk in chunks)
        visible += content_filter.flush()
        assert visible == literal
        assert content_filter.flush() == ""


@pytest.mark.parametrize(
    "overflow,expected", [("Answer<|im_end|>", "Answer"), ("[Calli", "[Calli")]
)
def test_responses_channel_overflow_and_eof_use_shared_filter(
    monkeypatch, overflow, expected
):
    original = Engine.stream_chat

    async def reasoning_channel(self, **kwargs):
        async for output in original(self, **kwargs):
            output.channel = "reasoning"
            yield output

    monkeypatch.setattr(Engine, "stream_chat", reasoning_channel)
    raw = "abcd" + overflow
    for chunks in partitions(raw):
        visible, _, calls = request_result(
            "responses", raw, chunks, body_updates={"reasoning_max_tokens": 1}
        )
        assert visible == expected
        assert calls == []


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_harmony_plain_commentary_header_is_not_content(surface):
    raw = "<|start|>assistant<|channel|>commentary<|message|>Working.<|end|>"
    for chunks in partitions(raw):
        assert request_result(surface, raw, chunks, reasoning="harmony") == (
            "Working.",
            "",
            [],
        )


@pytest.mark.parametrize("recipient", ["", " to=functions.get_weather"])
def test_harmony_incomplete_commentary_header_stays_held(recipient):
    raw = "<|channel|>commentary" + recipient
    for chunks in partitions(raw):
        parser = HarmonyReasoningParser()
        previous = ""
        for chunk in chunks:
            assert (
                parser.extract_reasoning_streaming(previous, previous + chunk, chunk)
                is None
            )
            previous += chunk
        assert parser.finalize_streaming(raw) is None
