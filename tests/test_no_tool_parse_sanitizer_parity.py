# SPDX-License-Identifier: Apache-2.0
"""Buffered tool-like text matches streaming when parsing is disabled."""

import pytest

from rapid_mlx.api.utils import sanitize_output
from rapid_mlx.config import get_config, reset_config
from tests.test_tool_call_finish_reason import ReplayEngine, client_for, sse

GEMMA_CALL = '<|tool_call>call:get_weather{"city":"Paris"}<tool_call|>'
TEXT_CALL = '[Calling tool: get_weather({"city":"Paris"})]'
GEMMA_PROSE = "Use <|tool_call>call:x{}<tool_call|> to call."
TEXT_CASES = [GEMMA_CALL, TEXT_CALL, GEMMA_PROSE]

RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.fixture(autouse=True)
def _restore_config():
    # client_for and the parser override mutate the process-wide config.
    yield
    reset_config()


MESSAGES_TOOL = {
    "name": "get_weather",
    "input_schema": RESPONSES_TOOL["parameters"],
}


def _payload(route, mode):
    if route == "responses":
        payload = {
            "model": "test-model",
            "input": "Weather?",
            "max_output_tokens": 128,
        }
        tool = RESPONSES_TOOL
        choice = mode
    else:
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Weather?"}],
            "max_tokens": 128,
        }
        tool = MESSAGES_TOOL
        choice = {"type": mode}

    if mode == "empty":
        payload["tools"] = []
    elif mode != "missing":
        payload.update(tools=[tool], tool_choice=choice)
    return payload


def _visible_and_calls(route, body, events):
    if route == "responses":
        buffered = "".join(
            part["text"]
            for item in body["output"]
            for part in item.get("content", [])
            if part["type"] == "output_text"
        )
        streamed = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        buffered_calls = [
            item for item in body["output"] if item["type"] == "function_call"
        ]
        streamed_calls = [
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        ]
        return buffered, streamed, buffered_calls, streamed_calls

    buffered = "".join(
        item["text"] for item in body["content"] if item["type"] == "text"
    )
    streamed = "".join(
        event["delta"]["text"]
        for event in events
        if event["type"] == "content_block_delta"
        and event["delta"]["type"] == "text_delta"
    )
    buffered_calls = [item for item in body["content"] if item["type"] == "tool_use"]
    streamed_calls = [
        event["content_block"]
        for event in events
        if event["type"] == "content_block_start"
        and event["content_block"]["type"] == "tool_use"
    ]
    return buffered, streamed, buffered_calls, streamed_calls


@pytest.mark.parametrize("route", ["responses", "messages"])
@pytest.mark.parametrize("mode", ["missing", "empty", "none", "auto"])
@pytest.mark.parametrize("text", TEXT_CASES)
@pytest.mark.parametrize("chunking", ["whole", "characters"])
def test_buffered_and_streaming_tool_markup_parity(route, mode, text, chunking):
    deltas = [text] if chunking == "whole" else list(text)
    client = client_for(ReplayEngine(deltas, "stop"))
    get_config().tool_call_parser = "hermes"
    payload = _payload(route, mode)

    buffered_response = client.post(f"/v1/{route}", json=payload)
    assert buffered_response.status_code == 200, buffered_response.text
    body = buffered_response.json()
    streaming_response = client.post(f"/v1/{route}", json={**payload, "stream": True})
    assert streaming_response.status_code == 200, streaming_response.text
    events = sse(streaming_response.text)

    buffered, streamed, buffered_calls, streamed_calls = _visible_and_calls(
        route, body, events
    )
    assert buffered == streamed

    parsing_enabled = mode == "auto"
    if not parsing_enabled:
        assert not any(
            "error" in event or event.get("type") == "error" for event in events
        ), streaming_response.text
        assert buffered == text
        assert not buffered_calls
        assert not streamed_calls
        if route == "responses":
            assert body["status"] == "completed"
            assert events[-1]["type"] == "response.completed"
            assert events[-1]["response"]["status"] == "completed"
        else:
            assert body["stop_reason"] == "end_turn"
            assert events[-1]["type"] == "message_stop"
            stop = next(
                event["delta"]["stop_reason"]
                for event in events
                if event["type"] == "message_delta"
            )
            assert stop == "end_turn"
        return

    assert "<|tool_call>" not in buffered
    assert "<tool_call|>" not in buffered
    assert "[Calling tool" not in buffered
    expect_call = text == TEXT_CALL
    assert bool(buffered_calls) is expect_call
    assert bool(streamed_calls) is expect_call
    if expect_call:
        assert buffered_calls[0]["name"] == "get_weather"
        assert streamed_calls[0]["name"] == "get_weather"
    if route == "responses":
        expected_status = "failed" if text == GEMMA_CALL else "completed"
        assert body["status"] == expected_status
        assert events[-1]["type"] == f"response.{expected_status}"
    else:
        expected_stop = "tool_use" if expect_call else "end_turn"
        assert body["stop_reason"] == expected_stop
        stop = next(
            event["delta"]["stop_reason"]
            for event in events
            if event["type"] == "message_delta"
        )
        assert stop == expected_stop


@pytest.mark.parametrize(
    "text",
    [
        GEMMA_CALL,
        "<|tool_call>",
        "<tool_call|>",
        TEXT_CALL,
        '[Calling tool="get_weather" city="Paris"]',
    ],
)
def test_sanitize_output_can_preserve_tool_markup(text):
    assert sanitize_output(text, strip_tool_markup=False) == text


def test_no_tool_markup_mode_still_strips_other_content_markers():
    text = "answer<|im_end|><|channel></think>"
    assert sanitize_output(text, strip_tool_markup=False) == "answer"


@pytest.mark.parametrize(
    "text", [GEMMA_CALL, "<|tool_call>", "<tool_call|>", TEXT_CALL]
)
def test_sanitize_output_default_still_strips_tool_markup(text):
    assert sanitize_output(text) is None
