# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible Responses API."""

import json
import platform
import sys
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture()
def client():
    from vllm_mlx.server import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def server_state():
    import vllm_mlx.server as srv

    original_engine = srv._engine
    original_model_name = srv._model_name
    original_store = srv._responses_store
    original_store_max_size = srv._RESPONSES_STORE_MAX_SIZE
    original_api_key = srv._api_key
    original_default_chat_template_kwargs = getattr(
        srv, "_default_chat_template_kwargs", None
    )

    srv._engine = None
    srv._model_name = "test-model"
    srv._responses_store = OrderedDict()
    srv._RESPONSES_STORE_MAX_SIZE = 1000
    srv._api_key = None
    srv._default_chat_template_kwargs = None

    try:
        yield
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name
        srv._responses_store = original_store
        srv._RESPONSES_STORE_MAX_SIZE = original_store_max_size
        srv._api_key = original_api_key
        srv._default_chat_template_kwargs = original_default_chat_template_kwargs


def _mock_engine(*outputs):
    engine = MagicMock()
    engine.model_name = "test-model"
    engine.preserve_native_tool_format = False
    engine.chat = AsyncMock(side_effect=list(outputs))
    stream_calls = []

    async def _stream_chat(**kwargs):
        stream_calls.append(kwargs)
        for output in getattr(engine, "_stream_outputs", []):
            yield output

    engine._stream_calls = stream_calls
    engine._stream_outputs = []
    engine.stream_chat = _stream_chat
    return engine


def _output(text: str, prompt_tokens: int = 7, completion_tokens: int = 3):
    return SimpleNamespace(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )


def _stream_output(
    new_text: str,
    prompt_tokens: int = 7,
    completion_tokens: int = 1,
    finish_reason: str | None = None,
):
    return SimpleNamespace(
        new_text=new_text,
        text=new_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        finished=finish_reason is not None,
    )


def _parse_sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for chunk in body.strip().split("\n\n"):
        if not chunk.strip():
            continue
        event_type = None
        payload = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: ").strip())
        if event_type is not None and payload is not None:
            events.append((event_type, payload))
    return events


class TestResponsesEndpoint:
    def test_basic_response(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello there"))

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Say hello"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response"
        assert body["output_text"] == "Hello there"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["type"] == "output_text"
        assert body["usage"]["input_tokens"] == 7
        assert body["usage"]["output_tokens"] == 3

    def test_responses_applies_server_default_chat_template_kwargs(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello there"))
        srv._engine = engine
        srv._default_chat_template_kwargs = {"enable_thinking": False}

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Say hello"},
        )

        assert resp.status_code == 200
        assert engine.chat.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_streaming_responses_validates_remote_media_once(self, client, monkeypatch):
        import vllm_mlx.server as srv

        engine = _mock_engine()
        engine._stream_outputs = [_stream_output("Hello there", finish_reason="stop")]
        srv._engine = engine

        validate_calls = []

        def fake_validate(messages):
            validate_calls.append(messages)

        def fake_responses_to_chat(_request):
            return srv.ChatCompletionRequest(
                model="test-model",
                messages=[
                    srv.Message(
                        role="user",
                        content=[
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            },
                        ],
                    )
                ],
                max_tokens=8,
                stream=True,
            )

        monkeypatch.setattr(srv, "_validate_remote_media_urls", fake_validate)
        monkeypatch.setattr(
            srv, "_responses_request_to_chat_request", fake_responses_to_chat
        )

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "describe this", "stream": True},
        )

        assert resp.status_code == 200
        assert len(validate_calls) == 1

    def test_responses_request_kwargs_override_server_defaults(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello there"))
        srv._engine = engine
        srv._default_chat_template_kwargs = {
            "enable_thinking": False,
            "server_default_only": "yes",
        }

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Say hello",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "request_only": 1,
                },
            },
        )

        assert resp.status_code == 200
        assert engine.chat.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": True,
            "server_default_only": "yes",
            "request_only": 1,
        }

    def test_previous_response_id_reuses_prior_context(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "user"
        assert second_messages[0]["content"] == "First prompt"
        assert second_messages[1]["role"] == "assistant"
        assert second_messages[1]["content"] == "First answer"
        assert second_messages[2]["role"] == "user"
        assert second_messages[2]["content"] == "Follow-up prompt"

    def test_previous_response_id_chains_across_multiple_follow_ups(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(
            _output("First answer"),
            _output("Second answer"),
            _output("Third answer"),
        )
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Second prompt",
            },
        )
        second_id = second.json()["id"]

        third = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": second_id,
                "input": "Third prompt",
            },
        )

        assert third.status_code == 200
        third_messages = engine.chat.call_args_list[2].kwargs["messages"]
        assert third_messages[0]["role"] == "user"
        assert third_messages[0]["content"] == "First prompt"
        assert third_messages[1]["role"] == "assistant"
        assert third_messages[1]["content"] == "First answer"
        assert third_messages[2]["role"] == "user"
        assert third_messages[2]["content"] == "Second prompt"
        assert third_messages[3]["role"] == "assistant"
        assert third_messages[3]["content"] == "Second answer"
        assert third_messages[4]["role"] == "user"
        assert third_messages[4]["content"] == "Third prompt"

    def test_previous_response_id_does_not_carry_prior_instructions(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "First system instruction",
                "input": "First prompt",
            },
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "Second system instruction",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[0]["content"] == "Second system instruction"
        assert "First system instruction" not in second_messages[0]["content"]
        assert second_messages[1]["role"] == "user"
        assert second_messages[1]["content"] == "First prompt"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "user"

    def test_previous_response_id_preserves_prior_system_message_items(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "system", "content": "Persist me"},
                    {"type": "message", "role": "user", "content": "First prompt"},
                ],
            },
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[0]["content"] == "Persist me"
        assert second_messages[1]["role"] == "user"
        assert second_messages[1]["content"] == "First prompt"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "user"

    def test_previous_response_id_missing_returns_404(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("unused"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": "resp_missing",
                "input": "Follow-up prompt",
            },
        )

        assert resp.status_code == 404
        assert "resp_missing" in resp.json()["detail"]

    def test_developer_role_is_normalized_to_system(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Hi"},
                    {"type": "message", "role": "developer", "content": "Be terse"},
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be terse"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"

    def test_system_prompt_canonicalization_strips_billing_header(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": (
                    "Be terse.\n"
                    "x-anthropic-billing-header: account=abc; cch=rotating-hash\n"
                    "Answer directly."
                ),
                "input": "Say hello",
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be terse.\nAnswer directly."

    def test_instructions_and_developer_message_are_merged(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "System instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": "Developer note",
                    },
                    {"type": "message", "role": "user", "content": "Hi"},
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert len([m for m in messages if m["role"] == "system"]) == 1
        assert messages[0]["role"] == "system"
        assert "System instructions" in messages[0]["content"]
        assert "Developer note" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_function_call_output_input_is_mapped_cleanly(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Done"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Run it"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "arguments": '{"cmd":"pwd"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "/tmp/work",
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[1]["role"] == "assistant"
        assert "[Calling tool: shell(" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "[Tool Result (call_1)]" in messages[2]["content"]
        assert "/tmp/work" in messages[2]["content"]

    def test_unsupported_tools_and_items_do_not_fail(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Fallback answer"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Answer directly"},
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search", "query": "ignored"},
                    },
                ],
                "tools": [
                    {"type": "web_search_preview"},
                    {"type": "file_search", "vector_store_ids": ["vs_123"]},
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "not available on this backend" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert engine.chat.call_args.kwargs["tools"][0]["type"] == "function"

    def test_function_call_response_item(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(
            _output('<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>')
        )

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Use a tool",
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["output"][0]["type"] == "function_call"
        assert body["output"][0]["name"] == "shell"
        assert body["output_text"] == ""

    def test_store_false_skips_persistence(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Ephemeral answer"))

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Do not store this",
                "store": False,
            },
        )

        assert first.status_code == 200
        assert first.json()["store"] is False
        assert first.json()["id"] not in srv._responses_store

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first.json()["id"],
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 404

    def test_responses_store_is_lru_bounded(self, client):
        import vllm_mlx.server as srv

        srv._RESPONSES_STORE_MAX_SIZE = 2
        srv._engine = _mock_engine(
            _output("First answer"),
            _output("Second answer"),
            _output("Third answer"),
        )

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        second = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Second prompt"},
        )
        third = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Third prompt"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 200
        assert list(srv._responses_store) == [second.json()["id"], third.json()["id"]]
        assert first.json()["id"] not in srv._responses_store

    def test_streaming_response_returns_sse_events(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output("Hello ", completion_tokens=1),
            _stream_output("stream", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        assert "event: response.created" in body
        assert "event: response.output_text.delta" in body
        assert "Hello stream" in body
        assert "event: response.completed" in body
        assert len(engine._stream_calls) == 1
        engine.chat.assert_not_awaited()

    def test_streaming_response_sequence_metadata_is_monotonic(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output("Hello ", completion_tokens=1),
            _stream_output("stream", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        events = _parse_sse_events(body)
        assert [event_type for event_type, _ in events[:2]] == [
            "response.created",
            "response.in_progress",
        ]
        sequence_numbers = [payload["sequence_number"] for _, payload in events]
        assert sequence_numbers == sorted(sequence_numbers)
        created_payload = events[0][1]
        completed_payload = next(
            payload
            for event_type, payload in events
            if event_type == "response.completed"
        )
        assert created_payload["response"]["id"] == completed_payload["response"]["id"]
        assert completed_payload["response"]["output_text"] == "Hello stream"

    def test_streaming_response_bracket_tool_call_does_not_leak_text(
        self, client, monkeypatch
    ):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output('[Calling tool: add({"a": 1, "b": 2})'),
            _stream_output("]", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(srv, "_tool_call_parser", "qwen3")
        monkeypatch.setattr(srv, "_tool_parser_instance", None)
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Add two numbers",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                ],
            },
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        events = _parse_sse_events(body)
        output_text_deltas = [
            payload["delta"]
            for event_type, payload in events
            if event_type == "response.output_text.delta"
        ]
        function_call_deltas = [
            payload
            for event_type, payload in events
            if event_type == "response.function_call_arguments.delta"
        ]

        assert not any("[Calling tool:" in delta for delta in output_text_deltas)
        assert len(function_call_deltas) == 1
        assert function_call_deltas[0]["delta"] == '{"a": 1, "b": 2}'

    def test_json_object_response_format_is_rejected(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "text": {"format": {"type": "json_object"}},
            },
        )

        assert resp.status_code == 400
        assert "json_object" in resp.json()["detail"]

    def test_reasoning_configuration_is_ignored(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "reasoning": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 200
        assert engine.chat.await_count == 1

    def test_reasoning_input_item_is_accepted(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Hello"},
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "x"}],
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "x"

    def test_length_finish_reason_marks_response_incomplete(self, client):
        import vllm_mlx.server as srv

        output = _output("Cut off", completion_tokens=5)
        output.finish_reason = "length"
        srv._engine = _mock_engine(output)

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "max_output_tokens": 5},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "max_output_tokens"}
