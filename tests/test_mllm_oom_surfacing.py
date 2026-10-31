# SPDX-License-Identifier: Apache-2.0
"""Memory failures retain their category and disable futile SDK retries."""

from collections import deque
from unittest.mock import MagicMock

import httpx
import pytest

from rapid_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler
from rapid_mlx.request import ClientRequestError, InferenceAbortedError
from tests.test_chat_route_vlm_image import (
    _make_client_with_envelope,
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


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses", "/v1/completions"])
def test_memory_abort_mapping_preserves_header(path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.request import inference_aborted_http_exception

    app = FastAPI()
    install_exception_handlers(app)

    @app.get(path)
    def fail():
        raise inference_aborted_http_exception(
            InferenceAbortedError("sanitized", error_kind="insufficient_memory")
        )

    response = TestClient(app).get(path)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "insufficient_memory"
    assert response.headers["x-should-retry"] == "false"
