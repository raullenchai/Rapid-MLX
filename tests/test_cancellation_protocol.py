# SPDX-License-Identifier: Apache-2.0
"""Cancellation terminal-state mapping across public serving protocols."""

import json
import threading
from types import SimpleNamespace

import pytest

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest
from vllm_mlx.api.responses_adapter import _convert_status
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.request import Request, RequestStatus, SamplingParams


class _CancelledChatEngine:
    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "test-model"
    tokenizer = None

    def __init__(self, *, before_first_token: bool):
        self.before_first_token = before_first_token

    async def stream_chat(self, messages, **kwargs):
        del messages
        if "request_admitted_event" in kwargs:
            kwargs["request_admitted_event"].set()
        if not self.before_first_token:
            yield GenerationOutput(
                text="partial",
                new_text="partial",
                finished=False,
                finish_reason=None,
                prompt_tokens=1,
                completion_tokens=1,
            )
        yield GenerationOutput(
            text="partial" if not self.before_first_token else "",
            new_text="" if self.before_first_token else "partial",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="cancelled",
        )

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        return GenerationOutput(
            text="cancelled",
            finish_reason="cancelled",
            prompt_tokens=1,
            completion_tokens=1,
        )


class _CancelledCompletionEngine:
    supports_completion_logprobs = True
    tokenizer = SimpleNamespace(encode=lambda _text: [1], decode=lambda _ids: "x")

    async def stream_generate(self, **kwargs):
        del kwargs
        yield GenerationOutput(
            text="partial",
            new_text="partial",
            tokens=[0],
            logprobs=[0.8],
        )
        yield GenerationOutput(
            text="partial",
            new_text="partial",
            finished=True,
            finish_reason="cancelled",
        )

    async def generate(self, **kwargs):
        del kwargs
        return GenerationOutput(
            text="cancelled",
            finish_reason="cancelled",
            prompt_tokens=1,
            completion_tokens=1,
        )


class _RawRequest:
    headers = {}

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


async def _await_direct(coro, *_args, **_kwargs):
    return await coro


async def _async_json_body():
    return {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _test_config():
    cfg = reset_config()
    cfg.model_name = "test-model"
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize("before_first_token", [True, False])
async def test_chat_stream_cancellation_has_no_successful_finish_frame(
    before_first_token,
):
    from vllm_mlx.routes.chat import stream_chat_completion

    cfg = reset_config()
    cfg.model_name = "test-model"
    request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )

    chunks = [
        chunk
        async for chunk in stream_chat_completion(
            _CancelledChatEngine(before_first_token=before_first_token),
            request.messages,
            request,
        )
    ]
    events = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]

    if not before_first_token:
        assert any(
            choice.get("delta", {}).get("content") == "partial"
            for event in events
            for choice in event.get("choices", [])
        )
    assert all(
        event.get("finish_reason") not in ("cancelled", "length") for event in events
    )


@pytest.mark.asyncio
async def test_completion_stream_cancellation_has_no_successful_finish_frame():
    from vllm_mlx.routes.completions import stream_completion

    pytest.importorskip("mlx")
    cfg = reset_config()
    cfg.model_name = "test-model"
    request = CompletionRequest(model="test-model", prompt="hi", stream=True)

    logprobs_request = request.model_copy(update={"logprobs": 0})

    chunks = [
        chunk
        async for chunk in stream_completion(
            _CancelledCompletionEngine(), "hi", logprobs_request
        )
    ]
    events = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]

    assert any(event["choices"][0]["text"] for event in events)
    assert any("logprobs" in event["choices"][0] for event in events)
    assert all(
        event["choices"][0]["finish_reason"] not in ("cancelled", "length")
        for event in events
    )


def test_responses_status_does_not_treat_cancellation_as_truncation():
    assert _convert_status("cancelled") == "failed"
    assert _convert_status("length") == "incomplete"


@pytest.mark.asyncio
async def test_chat_non_stream_cancellation_returns_client_closed(monkeypatch):
    from vllm_mlx.routes import chat

    monkeypatch.setattr(chat, "_check_admission_or_503", lambda *_: None)
    monkeypatch.setattr(chat, "_wait_with_disconnect", _await_direct)

    _test_config()
    request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )

    response = await chat._create_chat_completion_impl(
        request,
        _RawRequest(),
        _CancelledChatEngine(before_first_token=False),
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert response.status_code == 499


@pytest.mark.asyncio
async def test_completion_non_stream_cancellation_returns_client_closed(monkeypatch):
    from vllm_mlx.routes import completions

    engine = _CancelledCompletionEngine()
    monkeypatch.setattr(completions, "_resolve_max_tokens", lambda *_: 32)
    monkeypatch.setattr(completions, "get_engine", lambda *_: engine)
    monkeypatch.setattr(completions, "_validate_model_name", lambda *_: None)
    monkeypatch.setattr(completions, "_check_admission_or_503", lambda *_: None)
    monkeypatch.setattr(
        completions, "_release_admission_unless_committed", lambda *_: None
    )
    monkeypatch.setattr(completions, "_wait_with_disconnect", _await_direct)
    monkeypatch.setattr(
        completions, "enforce_context_length_for_prompt", lambda *_, **__: None
    )

    response = await completions.create_completion(
        CompletionRequest(model="test-model", prompt="hi"), _RawRequest()
    )

    assert response.status_code == 499


@pytest.mark.asyncio
async def test_anthropic_non_stream_cancellation_returns_client_closed(monkeypatch):
    from vllm_mlx.routes import anthropic

    engine = _CancelledChatEngine(before_first_token=False)
    monkeypatch.setattr(anthropic, "get_engine", lambda *_: engine)
    monkeypatch.setattr(anthropic, "_validate_model_name", lambda *_: None)
    monkeypatch.setattr(anthropic, "_check_admission_or_503", lambda *_: None)
    monkeypatch.setattr(
        anthropic, "_release_admission_unless_committed", lambda *_: None
    )
    monkeypatch.setattr(anthropic, "_wait_with_disconnect", _await_direct)
    monkeypatch.setattr(
        anthropic, "enforce_context_length_for_messages", lambda *_, **__: None
    )
    monkeypatch.setattr(anthropic, "_resolve_max_tokens", lambda *_: 32)

    response = await anthropic.create_anthropic_message(
        SimpleNamespace(json=_async_json_body)
    )

    assert response.status_code == 499


@pytest.mark.asyncio
async def test_responses_non_stream_cancellation_returns_client_closed(monkeypatch):
    from vllm_mlx.api.responses_models import ResponsesRequest
    from vllm_mlx.routes import responses

    monkeypatch.setattr(responses, "_wait_with_disconnect", _await_direct)

    _test_config()
    openai_request = ChatCompletionRequest(
        model="test-model", messages=[{"role": "user", "content": "hi"}]
    )
    responses_request = ResponsesRequest(
        model="test-model",
        input=[{"type": "message", "role": "user", "content": "hi"}],
    )

    response = await responses._non_stream(
        _CancelledChatEngine(before_first_token=False),
        openai_request,
        responses_request,
        _RawRequest(),
    )

    assert response.status_code == 499


@pytest.mark.asyncio
async def test_anthropic_stream_cancellation_uses_protocol_error():
    from vllm_mlx.routes.anthropic import _stream_anthropic_messages

    cfg = reset_config()
    cfg.model_name = "test-model"
    openai_request = ChatCompletionRequest(
        model="test-model",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    anthropic_request = AnthropicRequest(
        model="test-model",
        max_tokens=32,
        messages=[{"role": "user", "content": "hi"}],
    )

    chunks = [
        chunk
        async for chunk in _stream_anthropic_messages(
            _CancelledChatEngine(before_first_token=False),
            openai_request,
            anthropic_request,
        )
    ]

    events = [
        json.loads(chunk.split("data: ", 1)[1])
        for chunk in chunks
        if chunk.startswith("event:") and "data: " in chunk
    ]
    assert any(event["type"] == "error" for event in events)
    assert all(event["type"] != "message_delta" for event in events)


@pytest.mark.asyncio
async def test_responses_stream_cancellation_emits_response_failed():
    from vllm_mlx.api.responses_models import ResponsesRequest
    from vllm_mlx.routes.responses import _stream_responses

    _test_config()
    openai_request = ChatCompletionRequest(
        model="test-model", stream=True, messages=[{"role": "user", "content": "hi"}]
    )
    responses_request = ResponsesRequest(
        model="test-model",
        input=[{"type": "message", "role": "user", "content": "hi"}],
    )

    events = []
    async for chunk in _stream_responses(
        _CancelledChatEngine(before_first_token=False),
        openai_request,
        responses_request,
    ):
        if chunk.startswith("event:"):
            event_type = chunk.split("\n", 1)[0].removeprefix("event: ")
            data = json.loads(chunk.split("data: ", 1)[1])
            events.append((event_type, data))

    assert events[-1][0] == "response.failed"
    assert events[-1][1]["response"]["status"] == "failed"
    assert events[-1][1]["response"]["error"]["code"] == "request_cancelled"


@pytest.mark.asyncio
async def test_completion_json_stream_cancellation_has_no_terminal_chunk():
    from vllm_mlx.routes.completions import stream_completion

    _test_config()
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        stream=True,
        response_format={"type": "json_object"},
    )

    chunks = [
        chunk
        async for chunk in stream_completion(
            _CancelledCompletionEngine(), "hi", request
        )
    ]

    assert chunks == []


def test_text_scheduler_abort_marks_terminal_cancelled():
    pytest.importorskip("mlx")
    from vllm_mlx.scheduler import Scheduler

    tokenizer = SimpleNamespace(
        eos_token_id=2, encode=lambda _text: [1, 2], decode=lambda _ids: ""
    )
    scheduler = Scheduler(SimpleNamespace(), tokenizer)
    request = Request("req-1", "hello", SamplingParams(max_tokens=4))
    scheduler.add_request(request)

    assert scheduler._do_abort_request("req-1") is True
    assert request.status is RequestStatus.FINISHED_CANCELLED
    assert request.get_finish_reason() == "cancelled"


def test_mllm_scheduler_abort_marks_terminal_cancelled():
    pytest.importorskip("mlx")
    from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.requests = {}
    scheduler.waiting = []
    scheduler.running = {}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler.batch_generator = None
    scheduler.finished_req_ids = set()
    scheduler._detokenizer_pool = {}
    scheduler._aborted_queue_ids = set()
    scheduler._pending_abort_ids = set()
    scheduler._cancelled_request_ids = set()
    scheduler._cancel_counter_lock = threading.Lock()
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_cancelled = 0
    request = MLLMRequest(request_id="req-1", prompt="hello")
    scheduler.requests["req-1"] = request

    assert scheduler.abort_request("req-1") is True
    scheduler._do_abort_request("req-1")

    assert request.status is RequestStatus.FINISHED_CANCELLED
