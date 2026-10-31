# SPDX-License-Identifier: Apache-2.0
"""Memory failures retain their category and disable futile SDK retries."""

from collections import deque
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler  # noqa: E402
from rapid_mlx.request import ClientRequestError, InferenceAbortedError  # noqa: E402
from tests.test_chat_route_vlm_image import (  # noqa: E402
    _make_client_with_envelope,
    _StubMLLMEngine,
    _StubOOMFailureEngine,
    _StubStreamingOOMFailureEngine,
)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (
            RuntimeError(
                "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
            ),
            "insufficient_memory",
        ),
        (RuntimeError("boom"), "engine_aborted"),
        (ClientRequestError("invalid input"), "invalid_request"),
    ],
)
def test_scheduler_classifies_batch_failure(error, kind):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="failure", prompt="hello")
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = deque()
    scheduler.running = {request.request_id: request}
    scheduler.request_id_to_uid = {request.request_id: 42}
    scheduler.uid_to_request_id = {42: request.request_id}
    scheduler.finished_req_ids = set()
    scheduler._detokenizer_pool = {}
    scheduler._pending_abort_ids = set()
    scheduler._aborted_queue_ids = set()
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.next.side_effect = error
    scheduler._process_pending_aborts = MagicMock()
    scheduler._schedule_waiting = MagicMock(return_value=[])
    result = scheduler._step_no_queue().outputs[0]
    assert result.error_kind == kind
    if kind == "insufficient_memory":
        assert "memory" in result.error.lower()
        assert "METAL" not in result.error


@pytest.mark.parametrize("stream", [False, True])
def test_chat_memory_abort(stream):
    from rapid_mlx.config import reset_config

    engine = _StubStreamingOOMFailureEngine() if stream else _StubOOMFailureEngine()
    try:
        with _make_client_with_envelope(engine) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-vl-8b-4bit",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                    "stream": stream,
                },
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "insufficient_memory"
        assert "memory" in response.json()["error"]["message"].lower()
        assert response.headers["x-should-retry"] == "false"
    finally:
        reset_config()


def test_openai_sdk_does_not_retry_memory_response():
    openai = pytest.importorskip("openai")
    from rapid_mlx.config import reset_config

    attempts = []
    app_client = _make_client_with_envelope(_StubOOMFailureEngine())

    def respond(request):
        attempts.append(request)
        response = app_client.post(
            request.url.path, content=request.content, headers=dict(request.headers)
        )
        return httpx.Response(
            response.status_code, headers=response.headers, content=response.content
        )

    try:
        with (
            app_client,
            httpx.Client(transport=httpx.MockTransport(respond)) as http_client,
        ):
            client = openai.OpenAI(
                api_key="test",
                base_url="http://test/v1",
                http_client=http_client,
                max_retries=2,
            )
            with pytest.raises(openai.APIStatusError):
                client.chat.completions.create(
                    model="qwen3-vl-8b-4bit",
                    messages=[{"role": "user", "content": "hello"}],
                )
        assert len(attempts) == 1
    finally:
        reset_config()


class _AbortEngine(_StubMLLMEngine):
    def __init__(self, error_kind):
        super().__init__()
        self.error_kind = error_kind
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        raise InferenceAbortedError("inference aborted", error_kind=self.error_kind)

    async def chat(self, **kwargs):
        self.calls += 1
        raise InferenceAbortedError("inference aborted", error_kind=self.error_kind)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/completions", {"prompt": "hello"}),
        (
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8},
        ),
        ("/v1/responses", {"input": "hello"}),
    ],
)
def test_memory_abort_mapping_preserves_header(path, payload, monkeypatch):
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import anthropic, completions, responses

    engine = _AbortEngine("insufficient_memory")
    client = _make_client_with_envelope(engine)
    for route in (anthropic, completions, responses):
        client.app.include_router(route.router)
        monkeypatch.setattr(route, "_check_admission_or_503", lambda _engine: None)

    async def wait_for_engine(coro, _request, timeout):
        return await coro

    for route in (anthropic, completions, responses):
        monkeypatch.setattr(route, "_wait_with_disconnect", wait_for_engine)

    try:
        with client:
            response = client.post(
                path,
                json={"model": "qwen3-vl-8b-4bit", "stream": False, **payload},
            )
        assert engine.calls == 1
        assert response.status_code == 503
        assert response.headers["x-should-retry"] == "false"
        body = response.json()
        if path == "/v1/messages":
            assert body["type"] == "error"
        assert body["error"]["code"] == "insufficient_memory"
        assert "memory" in body["error"]["message"].lower()
    finally:
        reset_config()


@pytest.mark.parametrize(
    ("path", "payload", "route_name"),
    [
        ("/v1/completions", {"prompt": "hello"}, "completions"),
        (
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8},
            "anthropic",
        ),
        ("/v1/responses", {"input": "hello"}, "responses"),
    ],
)
@pytest.mark.parametrize("error_kind", ["engine_aborted", "lifecycle"])
def test_completions_engine_abort_does_not_disable_retries(
    path, payload, route_name, error_kind, monkeypatch
):
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes import anthropic, completions, responses

    route = {
        "anthropic": anthropic,
        "completions": completions,
        "responses": responses,
    }[route_name]
    engine = _AbortEngine(error_kind)
    client = _make_client_with_envelope(engine)
    client.app.include_router(route.router)
    client = TestClient(client.app, raise_server_exceptions=False)
    monkeypatch.setattr(route, "_check_admission_or_503", lambda _engine: None)

    async def wait_for_engine(coro, _request, timeout):
        return await coro

    monkeypatch.setattr(route, "_wait_with_disconnect", wait_for_engine)
    try:
        with client:
            response = client.post(
                path,
                json={"model": "qwen3-vl-8b-4bit", "stream": False, **payload},
            )
        assert engine.calls == 1
        assert response.status_code == 503
        assert "x-should-retry" not in response.headers
    finally:
        reset_config()


class _StreamingAbortEngine(_StubMLLMEngine):
    async def stream_chat(self, messages, **kwargs):
        async for output in super().stream_chat(messages, **kwargs):
            output.finished = False
            yield output
        raise InferenceAbortedError("private detail", error_kind="insufficient_memory")


@pytest.mark.parametrize(
    ("path", "payload", "route_name", "event_name"),
    [
        (
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8},
            "anthropic",
            "error",
        ),
        ("/v1/responses", {"input": "hello"}, "responses", "response.failed"),
    ],
)
def test_streaming_memory_abort(path, payload, route_name, event_name, monkeypatch):
    import json

    from rapid_mlx.config import reset_config
    from rapid_mlx.request import inference_aborted_error_payload
    from rapid_mlx.routes import anthropic, responses

    route = {"anthropic": anthropic, "responses": responses}[route_name]
    client = _make_client_with_envelope(_StreamingAbortEngine())
    client.app.include_router(route.router)
    monkeypatch.setattr(route, "_check_admission_or_503", lambda _engine: None)
    try:
        with client:
            response = client.post(
                path, json={"model": "qwen3-vl-8b-4bit", "stream": True, **payload}
            )
        assert response.status_code == 200
        frames = response.text.split("\n\n")
        frame = next(
            frame for frame in frames if frame.startswith(f"event: {event_name}\n")
        )
        event = json.loads(
            next(line[6:] for line in frame.splitlines() if line.startswith("data: "))
        )
        expected = inference_aborted_error_payload(
            InferenceAbortedError("private detail", error_kind="insufficient_memory")
        )
        assert "memory" in expected["message"].lower()
        assert "private detail" not in response.text
        if path == "/v1/messages":
            assert "event: message_start" in response.text
            assert event == {
                "type": "error",
                "error": {"type": "overloaded_error", "message": expected["message"]},
            }
            assert "[DONE]" not in response.text
        else:
            assert "event: response.created" in response.text
            assert event["response"]["error"] == {
                "code": "insufficient_memory",
                "message": expected["message"],
            }
    finally:
        reset_config()


def test_disconnect_guard_openai_memory_abort():
    import asyncio
    import json
    from types import SimpleNamespace

    from rapid_mlx.request import inference_aborted_error_payload
    from rapid_mlx.service.helpers import _disconnect_guard

    async def source():
        yield "data: started\n\n"
        raise InferenceAbortedError("private detail", error_kind="insufficient_memory")

    class Request:
        url = SimpleNamespace(path="/v1/chat/completions")

        async def is_disconnected(self):
            return False

    async def collect():
        return [chunk async for chunk in _disconnect_guard(source(), Request())]

    chunks = asyncio.run(collect())
    expected = inference_aborted_error_payload(
        InferenceAbortedError("private detail", error_kind="insufficient_memory")
    )
    assert json.loads(chunks[-2][6:]) == {
        "error": {key: expected[key] for key in ("message", "type", "code")}
    }
    assert chunks[-1] == "data: [DONE]\n\n"


def _collect_guard_abort(path, error_kind, response_state=None):
    import asyncio
    from types import SimpleNamespace

    from rapid_mlx.service.helpers import _disconnect_guard

    async def source():
        yield "data: started\n\n"
        raise InferenceAbortedError("private detail", error_kind=error_kind)

    async def is_disconnected():
        return False

    request = SimpleNamespace(
        url=SimpleNamespace(path=path), is_disconnected=is_disconnected
    )

    async def collect():
        return [
            frame
            async for frame in _disconnect_guard(
                source(), request, keepalive_seconds=0, response_state=response_state
            )
        ]

    return asyncio.run(collect())


@pytest.mark.parametrize(
    ("kind", "event_type"),
    [
        ("insufficient_memory", "overloaded_error"),
        ("engine_aborted", "api_error"),
        ("lifecycle", "api_error"),
    ],
)
def test_disconnect_guard_anthropic_abort(kind, event_type):
    import json

    from rapid_mlx.request import inference_aborted_error_payload

    frames = _collect_guard_abort("/v1/messages", kind)
    assert frames[0] == "data: started\n\n"
    assert len(frames) == 2
    assert frames[1].startswith("event: error\ndata: ")
    event = json.loads(frames[1].split("data: ", 1)[1])
    message = (
        "Request cancelled by model replacement"
        if kind == "lifecycle"
        else inference_aborted_error_payload(
            InferenceAbortedError("private detail", error_kind=kind)
        )["message"]
    )
    assert event == {"type": "error", "error": {"type": event_type, "message": message}}
    assert "[DONE]" not in "".join(frames)


@pytest.mark.parametrize("with_state", [True, False])
def test_disconnect_guard_responses_abort(with_state):
    import json

    state = (
        {"response": {"id": "resp_x", "model": "m"}, "sequence_number": [5]}
        if with_state
        else None
    )
    frames = _collect_guard_abort("/v1/responses", "insufficient_memory", state)
    assert frames[0] == "data: started\n\n"
    assert len(frames) == 2
    assert frames[1].startswith("event: response.failed\ndata: ")
    event = json.loads(frames[1].split("data: ", 1)[1])
    assert event["type"] == "response.failed"
    assert event["sequence_number"] == (5 if with_state else 0)
    assert event["response"]["status"] == "failed"
    assert event["response"]["error"]["code"] == "insufficient_memory"
    if with_state:
        assert event["response"]["id"] == "resp_x"
        assert event["response"]["model"] == "m"
        assert state["sequence_number"] == [6]
    assert "[DONE]" not in "".join(frames)


def test_disconnect_guard_responses_lifecycle_abort():
    import json

    frames = _collect_guard_abort("/v1/responses", "lifecycle")
    assert len(frames) == 2
    assert frames[1].startswith("event: response.failed\ndata: ")
    event = json.loads(frames[1].split("data: ", 1)[1])
    assert event["type"] == "response.failed"
    assert event["response"]["error"]["code"] == "model_replacement"
