"""Finalization keeps reasoning and tool wire out of the answer lane."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning import get_parser, list_parsers
from rapid_mlx.routes import anthropic, chat, responses
from tests.test_rejected_tool_diagnostics import Engine as BaseEngine

TOOLS = [
    {
        "type": "function",
        "function": {"name": "write_file", "parameters": {"type": "object"}},
    }
]


class Engine(BaseEngine):
    def __init__(self, raw, step, finish, matched_stop):
        super().__init__(raw)
        self.step, self.finish, self.matched_stop = step, finish, matched_stop
        self.tokenizer = SimpleNamespace(
            chat_template="{% if add_generation_prompt %}<think>{% endif %}",
            get_vocab=lambda: {},
        )

    async def chat(self, **kwargs):
        return GenerationOutput(
            text="" if getattr(self, "engine_routed", False) else self.raw,
            raw_text="" if getattr(self, "engine_routed", False) else self.raw,
            reasoning_text=self.raw if getattr(self, "engine_routed", False) else "",
            prompt_tokens=4,
            completion_tokens=len(self.raw),
            finished=True,
            finish_reason=self.finish,
            matched_stop=self.matched_stop,
        )

    async def stream_chat(self, **kwargs):
        for i in range(0, len(self.raw), self.step):
            yield GenerationOutput(
                text=self.raw[: i + self.step],
                new_text=self.raw[i : i + self.step],
                channel="reasoning" if getattr(self, "engine_routed", False) else None,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=False,
            )
        yield GenerationOutput(
            text=self.raw,
            new_text="",
            prompt_tokens=4,
            completion_tokens=len(self.raw),
            finished=True,
            finish_reason=self.finish,
            matched_stop=self.matched_stop,
        )


def invoke(
    surface,
    stream,
    raw,
    *,
    parser="qwen3",
    tool="hermes",
    step=1,
    finish="stop",
    matched_stop=None,
    tools_requested=True,
    engine_routed=False,
    repeat=False,
    choice=None,
    template=None,
    thinking=None,
):
    cfg = reset_config()
    cfg.model_name = "contract-test"
    cfg.no_thinking = parser is None
    cfg.reasoning_parser_name = parser
    cfg.reasoning_parser = get_parser(parser)() if parser else None
    cfg.tool_call_parser = tool
    cfg.enable_auto_tool_choice = True
    cfg.engine = Engine(raw, step, finish, matched_stop)
    cfg.engine.engine_routed = engine_routed
    if template is not None:
        cfg.engine.tokenizer.chat_template = template
    app = FastAPI()
    for router in (chat.router, responses.router, anthropic.router):
        app.include_router(router)
    body = dict(
        model=cfg.model_name,
        stream=stream,
        temperature=0,
        tools=TOOLS,
        messages=[{"role": "user", "content": "Write a file."}],
        chat_template_kwargs={
            "enable_thinking": bool(parser) if thinking is None else thinking
        },
    )
    url = "/v1/chat/completions"
    if surface == "responses":
        url = "/v1/responses"
        body["input"] = body.pop("messages")
        body["tools"] = [{"type": "function", **TOOLS[0]["function"]}]
    elif surface == "messages":
        url = "/v1/messages"
        body["max_tokens"] = 2000000
        body["tools"] = [{"name": "write_file", "input_schema": {"type": "object"}}]
        if parser and thinking is not False:
            body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    if choice:
        body["tool_choice"] = {"type": "any"} if surface == "messages" else choice
    if not tools_requested:
        body.pop("tools")
    try:
        client = TestClient(app)
        response = client.post(url, json=body)
        if repeat:
            replay = client.post(url, json=body)
            assert replay.status_code == response.status_code
            first, second = response.json(), replay.json()
            for payload in (first, second):
                payload.pop("id", None)
                payload.pop("created", None)
            assert second == first
            response = replay
        packets = (
            [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            if stream
            else ([response.json()] if response.content else [])
        )
        visible, calls, errors = [], [], []
        for p in packets:
            if (
                p.get("error")
                or p.get("detail")
                or p.get("type") == "response.failed"
                or p.get("status") == "failed"
            ):
                errors.append(p)
            if surface == "chat":
                for choice in p.get("choices", []):
                    message = choice.get("delta", choice.get("message", {}))
                    visible.append(message.get("content") or "")
                    calls.extend(message.get("tool_calls") or [])
            elif surface == "responses":
                if stream:
                    if p.get("type") == "response.output_text.delta":
                        visible.append(p["delta"])
                    if (
                        p.get("type") == "response.output_item.done"
                        and p["item"]["type"] == "function_call"
                    ):
                        calls.append(p["item"])
                else:
                    for item in p.get("output", []):
                        if item["type"] == "message":
                            visible.extend(c["text"] for c in item["content"])
                        if item["type"] == "function_call":
                            calls.append(item)
            elif stream:
                delta = p.get("delta", {})
                if delta.get("type") == "text_delta":
                    visible.append(delta["text"])
                block = p.get("content_block", {})
                if block.get("type") == "tool_use":
                    calls.append(block)
                if delta.get("type") == "input_json_delta":
                    calls.append(delta)
            else:
                for block in p.get("content", []):
                    if block["type"] == "text":
                        visible.append(block["text"])
                    if block["type"] == "tool_use":
                        calls.append(block)
        return "".join(visible), calls, errors, response
    finally:
        reset_config()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "finish,matched_stop",
    [
        ("stop", None),
        ("length", None),
        ("stop", "STOP"),
        (None, None),
    ],
)
@pytest.mark.parametrize("raw", ["<think>Private secret.", "Private secret."])
def test_reasoning_only(surface, stream, finish, matched_stop, raw):
    visible, calls, errors, response = invoke(
        surface, stream, raw, finish=finish, matched_stop=matched_stop
    )
    assert "Private" not in visible, response.text
    assert not calls
    if finish != "length":
        assert not visible and errors, response.text
        assert "model_no_final_answer" in json.dumps(errors), response.text
        assert response.status_code == (
            422 if not stream and surface != "responses" else 200
        )
    else:
        from rapid_mlx.service.helpers import REASONING_CUTOFF_SENTINEL

        assert visible == REASONING_CUTOFF_SENTINEL and not errors, response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "raw", ["<think>Private.</think>Answer", "Private.</think>Answer"]
)
def test_final_answer_preserved(surface, stream, raw):
    visible, calls, errors, response = invoke(surface, stream, raw)
    assert visible == "Answer", response.text
    assert not calls and not errors


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        '{"name":"write_file","arguments":BROKEN}',
        '{"name":"write_file","arguments":{"content":',
        '{"name":"write_file","argu":BROKEN}',
        '{"name":"write_file","arguments":{"content":"truncated',
    ],
)
def test_invalid_arguments(surface, stream, body):
    raw = "<tool_call>" + body + "</tool_call>"
    visible, calls, errors, response = invoke(surface, stream, raw, parser=None)
    assert visible == raw and not calls and not errors, response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("size", [65535, 65536, 65537])
def test_large_valid_call(surface, stream, size):
    raw = (
        "<tool_call>"
        + json.dumps({"name": "write_file", "arguments": {"content": "x" * size}})
        + "</tool_call>"
    )
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, step=32768
    )
    assert not visible, visible[:100]
    assert calls and not errors, response.text[:1000]
    assert "x" * size in response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parser", list_parsers())
def test_engine_routed_reasoning_remains_private(surface, stream, parser):
    visible, calls, errors, response = invoke(
        surface,
        stream,
        "Private engine trace.",
        parser=parser,
        engine_routed=True,
    )
    assert not visible and not calls, response.text
    assert errors, response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_only_without_tools(surface, stream):
    visible, calls, errors, response = invoke(
        surface,
        stream,
        "<think>Private.",
        tools_requested=False,
    )
    assert not visible and not calls, response.text
    assert errors, response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_plain_nonreasoning_answer_preserved(surface, stream):
    visible, calls, errors, response = invoke(
        surface,
        stream,
        "Answer",
        parser=None,
        tools_requested=False,
    )
    assert visible == "Answer" and not errors, response.text


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_reasoning_only_cache_replay(finish):
    from rapid_mlx.response_cache import (
        configure_response_cache,
        get_response_cache,
        reset_response_cache_for_tests,
    )

    reset_response_cache_for_tests()
    configure_response_cache(8)
    try:
        visible, calls, errors, response = invoke(
            "chat",
            False,
            "<think>Private cached trace.",
            finish=finish,
            repeat=True,
        )
        assert "Private" not in visible and not calls
        stats = get_response_cache().snapshot()
        if finish == "stop":
            assert errors and stats["entries"] == 0
        else:
            assert stats["hits"] == 1, stats
    finally:
        reset_response_cache_for_tests()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool", ["hermes", "glm47", "qwen3_coder_xml"])
@pytest.mark.parametrize("size", [65536, 65537, 1048577])
def test_large_native_arguments_exact(surface, stream, tool, size):
    content = "x" * size
    if tool == "hermes":
        raw = (
            "<tool_call>"
            + json.dumps({"name": "write_file", "arguments": {"content": content}})
            + "</tool_call>"
        )
    elif tool == "glm47":
        raw = (
            "<tool_call>write_file<arg_key>content</arg_key><arg_value>"
            + content
            + "</arg_value></tool_call>"
        )
    else:
        raw = (
            "<tool_call><function=write_file><parameter=content>"
            + content
            + "</parameter></function></tool_call>"
        )
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, tool=tool, step=32768
    )
    assert not visible and not errors and calls, response.text[:500]
    if surface == "chat":
        arguments = json.loads(
            "".join(call.get("function", {}).get("arguments", "") for call in calls)
        )
    elif surface == "responses":
        arguments = json.loads(calls[0]["arguments"])
    elif stream:
        arguments = json.loads("".join(call.get("partial_json", "") for call in calls))
    else:
        arguments = calls[0]["input"]
    assert arguments == {"content": content}


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("step", [1, 10000])
@pytest.mark.parametrize("reverse", [False, True])
def test_malformed_member_of_multiple_calls_fails(surface, stream, step, reverse):
    valid = '<tool_call>{"name":"write_file","arguments":{"content":"ok"}}</tool_call>'
    invalid = '<tool_call>{"name":"write_file","arguments":BROKEN}</tool_call>'
    raw = invalid + valid if reverse else valid + invalid
    visible, calls, errors, response = invoke(
        surface, stream, raw, step=step, parser=None
    )
    assert not errors and calls, response.text
    assert visible == invalid, response.text
    assert "malformed_" not in response.text


def test_all_tool_filter_dialects_keep_bounded_state_until_split_close():
    from rapid_mlx.api.utils import _TOOL_CALL_TAGS, StreamingToolCallFilter

    for opener, closer in _TOOL_CALL_TAGS:
        for split in range(1, len(closer)):
            filt = StreamingToolCallFilter()
            assert filt.process(opener + "x" * 1048577 + closer[:split]) == ""
            assert len(filt._buffer) < len(closer)
            assert filt.process(closer[split:] + "Answer") == "Answer"
            assert filt.flush() == ""


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_forced_choice_cannot_repair_malformed_arguments_to_empty_object(
    surface, stream
):
    raw = '<tool_call>{"name":"write_file","arguments":BROKEN}</tool_call>'
    visible, calls, errors, response = invoke(
        surface, stream, raw, choice="required", parser=None
    )
    assert not calls and errors, response.text
    assert "malformed_" not in response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_bare_json_call_preserves_literal_marker_argument(surface, stream):
    raw = json.dumps(
        {
            "name": "write_file",
            "arguments": {"content": "<malformed_json_arguments> is literal text"},
        }
    )
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, step=10000
    )
    assert calls and not errors, response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("suffix", ["<", "<th", "</th"])
def test_answer_keeps_literal_partial_marker_at_eof(surface, stream, suffix):
    visible, calls, errors, response = invoke(
        surface, stream, "</think>Plain " + suffix
    )
    assert visible == "Plain " + suffix, response.text
    assert not calls and not errors


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parser", ["deepseek_r1_distill", "deepseek_r1", "qwen3"])
@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("thinking", [False, True])
def test_nonpriming_template_preserves_genuine_answer(
    surface, stream, parser, finish, thinking
):
    answer = "Unified memory is shared by the CPU and GPU."
    visible, calls, errors, response = invoke(
        surface,
        stream,
        answer,
        parser=parser,
        finish=finish,
        template="{% if add_generation_prompt %}assistant:{% endif %}",
        thinking=thinking,
    )
    assert visible == answer and not calls and not errors, response.text


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_large_unfinished_call_is_content_at_eof(surface, stream):
    raw = '<tool_call>{"name":"write_file","arguments":{"content":"' + "x" * 1048577
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, step=65536, finish="length"
    )
    assert visible == raw and not calls and not errors, response.text[:1000]


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_nonpriming_template_never_reclassifies_engine_reasoning(surface, stream):
    visible, calls, errors, response = invoke(
        surface,
        stream,
        "Private secret.",
        parser="deepseek_r1_distill",
        template="{% if add_generation_prompt %}assistant:{% endif %}",
        thinking=False,
        engine_routed=True,
    )
    assert not visible and not calls and errors, response.text
    assert "model_no_final_answer" in json.dumps(errors), response.text


@pytest.mark.parametrize("step", [1, 7, 32768, 200000])
def test_hermes_releases_confirmed_literal_prose_without_byte_budget(step):
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    parser = HermesToolParser()
    raw = (
        "The literal <tool_call> marker is used in examples. "
        + "answer " * (10000 if step >= 32768 else 100)
        + "End."
    )
    previous = ""
    visible = []
    for offset in range(0, len(raw), step):
        delta = raw[offset : offset + step]
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous, current, delta, request={"tools": TOOLS}
        )
        assert not (result or {}).get("tool_calls")
        visible.append((result or {}).get("content", ""))
        previous = current
    visible.append(parser.flush_held_content(raw))
    assert "".join(visible) == raw


@pytest.mark.parametrize("step", [1, 7, 32768, 200000])
def test_filter_releases_confirmed_literal_prose_without_byte_budget(step):
    from rapid_mlx.api.utils import StreamingToolCallFilter
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    filt = StreamingToolCallFilter(
        classify_prefix=HermesToolParser.classify_tool_call_prefix
    )
    raw = (
        "The literal <tool_call> marker is used in examples. "
        + "answer " * (10000 if step >= 32768 else 100)
        + "End."
    )
    visible = (
        "".join(filt.process(raw[i : i + step]) for i in range(0, len(raw), step))
        + filt.flush()
    )
    assert visible == raw


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("step", [1, 7, 32768])
@pytest.mark.parametrize("tool", ["hermes", "qwen3_coder_xml"])
def test_literal_tool_opener_keeps_genuine_answer_on_every_surface(
    surface, stream, step, tool
):
    raw = (
        "The literal <tool_call> marker is used in examples. "
        + "answer " * (10000 if step == 32768 else 10)
        + "End."
    )
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, step=step, tool=tool
    )
    assert visible == raw and not calls and not errors, response.text[:1000]


@pytest.mark.parametrize("parser_name", ["hermes", "qwen3_coder_xml"])
def test_classifier_does_not_release_another_envelope(parser_name):
    from rapid_mlx.api.utils import StreamingToolCallFilter

    filt = StreamingToolCallFilter(parser_name=parser_name)
    assert filt.process("<function=write_file><parameter=content>hello") == ""
    assert filt.process("</parameter></function>Answer") == "Answer"


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("literal", [False, True])
def test_large_undecided_whitespace_preserves_calls_and_literal_answers(
    surface, stream, literal
):
    prefix = "<tool_call>" + " " * 65537
    raw = prefix + (
        "literal example"
        if literal
        else '{"name":"write_file","arguments":{"content":"hello"}}</tool_call>'
    )
    visible, calls, errors, response = invoke(
        surface, stream, raw, parser=None, step=8192
    )
    assert not errors, response.text[:1000]
    assert visible == raw and not calls, response.text[:1000]


def test_filter_spills_undecided_whitespace_and_closes_temporary_file():
    from rapid_mlx.api.utils import StreamingToolCallFilter

    filt = StreamingToolCallFilter(parser_name="hermes")
    assert filt.process("<tool_call>" + " " * 9000) == ""
    assert len(filt._buffer) <= 8192
    pending = filt._prefix_whitespace
    assert pending is not None and not pending.closed
    assert filt.flush() == "<tool_call>" + " " * 9000
    assert pending.closed


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_parser_rejection_is_not_repaired(surface, stream, monkeypatch):
    from rapid_mlx.service.helpers import _InvalidToolArgumentsError
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    def reject(*args, **kwargs):
        raise _InvalidToolArgumentsError(status_code=400, detail="Arguments rejected")

    monkeypatch.setattr(HermesToolParser, "extract_tool_calls", reject)
    raw = '<tool_call>{"name":"write_file","arguments":{"content":"hello"}}</tool_call>'
    visible, calls, errors, response = invoke(surface, stream, raw, parser=None)
    assert not visible and not calls and errors, response.text
    if stream:
        assert "invalid_tool_arguments" in json.dumps(errors), response.text
    else:
        assert (
            response.status_code == 400
            and response.json()["detail"] == "Arguments rejected"
        )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_nonpriming_template_does_not_rescue_scratchpad_labels(surface, stream):
    visible, calls, errors, response = invoke(
        surface,
        stream,
        "Here's a thinking process: Private secret.",
        template="{% if add_generation_prompt %}assistant:{% endif %}",
        thinking=True,
    )
    assert not visible and not calls and errors, response.text
    assert "model_no_final_answer" in json.dumps(errors), response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("engine_routed", [False, True])
def test_cancelled_reasoning_is_not_a_model_failure(stream, engine_routed):
    visible, calls, _, response = invoke(
        "responses",
        stream,
        "PRIVATE_CANCELLED_REASONING",
        finish="cancelled",
        engine_routed=engine_routed,
    )
    assert not visible
    assert not calls
    assert "model_no_final_answer" not in response.text
    if stream:
        assert response.status_code == 200
        assert "request_cancelled" in response.text
    else:
        assert response.status_code == 499
        assert not response.content


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_content_filter_keeps_protocol_terminal_state(surface, stream):
    visible, calls, errors, response = invoke(
        surface,
        stream,
        "<think>PRIVATE_FILTERED_REASONING",
        finish="content_filter",
        tools_requested=False,
    )
    assert response.status_code == 200
    assert not visible and not calls and not errors, response.text
    assert "model_no_final_answer" not in response.text
    if surface == "chat":
        assert '"content_filter"' in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize(
    "failure", ["cancel", "engine", "disk", "cancel-close", "engine-close"]
)
async def test_probe_storage_closes_on_stream_failures(surface, failure, monkeypatch):
    import asyncio

    from rapid_mlx.api import utils
    from rapid_mlx.api.models import ChatCompletionRequest

    opened = []
    real_tempfile = utils.tempfile.TemporaryFile

    def open_probe(*args, **kwargs):
        handle = real_tempfile(*args, **kwargs)
        opened.append(handle)
        if failure.endswith("-close"):
            original_close = handle.close

            def fail_close():
                original_close()
                raise OSError("PRIVATE_CLEANUP_DETAIL")

            monkeypatch.setattr(handle, "close", fail_close)
        if failure == "disk":

            def fail_write(value):
                raise OSError("probe storage unavailable")

            monkeypatch.setattr(handle, "write", fail_write)
        return handle

    monkeypatch.setattr(utils.tempfile, "TemporaryFile", open_probe)
    raw = "<tool_call>" + " " * 9000

    class InterruptedEngine(Engine):
        async def stream_chat(self, **kwargs):
            yield GenerationOutput(text=raw, new_text=raw, finished=False)
            if failure.startswith("cancel"):
                raise asyncio.CancelledError()
            raise RuntimeError("engine interrupted")

    cfg = reset_config()
    cfg.model_name = "contract-test"
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_call_parser = "hermes"
    cfg.enable_auto_tool_choice = True
    engine = cfg.engine = InterruptedEngine(raw, 9000, "stop", None)
    request = ChatCompletionRequest(
        model="contract-test",
        messages=[{"role": "user", "content": "write"}],
        tools=TOOLS,
    )
    if surface == "responses":
        stream = responses._stream_responses(
            engine,
            request,
            responses.ResponsesRequest(model="contract-test", input="write"),
        )
    else:
        stream = anthropic._stream_anthropic_messages(
            engine,
            request,
            anthropic.AnthropicRequest(
                model="contract-test",
                max_tokens=100,
                messages=[{"role": "user", "content": "write"}],
            ),
        )
    try:
        try:
            packets = []
            async for packet in stream:
                packets.append(packet)
            if failure == "engine-close":
                assert "engine interrupted" in "".join(packets)
                assert "tool_parse_storage_error" not in "".join(packets)
        except (asyncio.CancelledError, RuntimeError, OSError) as error:
            if failure.endswith("-close"):
                if failure == "cancel-close":
                    assert isinstance(error, asyncio.CancelledError)
                else:
                    assert (
                        isinstance(error, RuntimeError)
                        and str(error) == "engine interrupted"
                    )
                assert "PRIVATE_CLEANUP_DETAIL" not in str(error)
        finally:
            await stream.aclose()
        assert opened, "the route must exercise the spilled whitespace probe"
        assert all(handle.closed for handle in opened)
    finally:
        reset_config()


@pytest.mark.parametrize("reset", [False, True])
def test_hermes_stream_cursor_starts_fresh_on_reuse(reset):
    from rapid_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

    parser = HermesToolParser()
    old = "Old answer <tool_call> prose"
    first = parser.extract_tool_calls_streaming("", old, old)
    assert first["content"] == old
    if reset:
        parser.reset()
        previous = "Already consumed "
    else:
        previous = ""
    answer = "New answer."
    result = parser.extract_tool_calls_streaming(previous, previous + answer, answer)
    assert result["content"] == answer


@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.parametrize("failure", ["create", "write", "read", "close"])
def test_probe_storage_failure_has_safe_protocol_error(surface, failure, monkeypatch):
    from rapid_mlx.api import utils

    opened = []
    real_tempfile = utils.tempfile.TemporaryFile

    def fail(*args, **kwargs):
        raise OSError("PRIVATE_STORAGE_PATH")

    def open_probe(*args, **kwargs):
        if failure == "create":
            fail()
        handle = real_tempfile(*args, **kwargs)
        opened.append(handle)
        if failure == "close":
            original_close = handle.close

            def fail_close():
                original_close()
                fail()

            monkeypatch.setattr(handle, "close", fail_close)
        else:
            monkeypatch.setattr(handle, failure, fail)
        return handle

    monkeypatch.setattr(utils.tempfile, "TemporaryFile", open_probe)
    raw = "<tool_call>" + " " * 9000 + "literal example"
    visible, calls, errors, response = invoke(
        surface, True, raw, parser=None, step=9000
    )
    assert response.status_code == 200
    assert not visible and not calls and errors, response.text
    assert "tool_parse_storage_error" in json.dumps(errors)
    assert utils.ToolProbeStorageError.public_message in response.text
    assert "PRIVATE_STORAGE_PATH" not in response.text
    assert all(handle.closed for handle in opened)


def test_probe_close_error_is_safe_and_idempotent(monkeypatch):
    from rapid_mlx.api.utils import StreamingToolCallFilter, ToolProbeStorageError

    filt = StreamingToolCallFilter(parser_name="hermes")
    assert filt.process("<tool_call>" + " " * 9000) == ""
    handle = filt._prefix_whitespace
    original_close = handle.close

    def fail_close():
        original_close()
        raise OSError("PRIVATE_STORAGE_PATH")

    monkeypatch.setattr(handle, "close", fail_close)
    with pytest.raises(ToolProbeStorageError) as caught:
        filt.close()
    assert str(caught.value) == ToolProbeStorageError.public_message
    assert handle.closed
    filt.close()
