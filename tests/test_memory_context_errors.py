# SPDX-License-Identifier: Apache-2.0
"""Permanent memory overflow preserves the context protocol at runtime boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from rapid_mlx.errors import BackpressureError, MemoryContextExceededError
from rapid_mlx.request import InferenceAbortedError, inference_aborted_http_exception
from rapid_mlx.service import helpers


@pytest.fixture
def memory_ceiling(monkeypatch):
    monkeypatch.setattr(helpers, "memory_context_ceiling", lambda engine, **kwargs: 80)
    emit = Mock()
    monkeypatch.setattr("rapid_mlx.telemetry.inference.emit_capability_rejected", emit)
    return emit


@pytest.mark.parametrize("kind", ["admission", "oom"])
def test_memory_overflow_classifier(kind, memory_ceiling):
    exc = (
        MemoryContextExceededError(prompt_tokens=101, memory_ceiling=80)
        if kind == "admission"
        else InferenceAbortedError(
            "private",
            error_kind="insufficient_memory",
            prompt_tokens=91,
            completion_tokens=10,
        )
    )
    error = helpers.context_overflow_from_memory_error(None, exc)
    assert error.status_code == 400
    assert error.prompt_tokens == 101
    assert error.limit == 80
    assert error.error_payload["message"].startswith(
        "This model's maximum context length is 80 tokens."
    )
    assert error.error_payload["code"] == "context_length_exceeded"
    assert memory_ceiling.call_args.kwargs["reject_reason"] == "memory_window"


@pytest.mark.parametrize(
    "kind,tokens",
    [
        ("insufficient_memory", 79),
        ("insufficient_memory", 80),
        ("engine_aborted", 101),
        ("lifecycle", 101),
    ],
)
def test_other_aborts_retain_503(kind, tokens, memory_ceiling):
    exc = InferenceAbortedError("private", error_kind=kind, prompt_tokens=tokens)
    assert helpers.context_overflow_from_memory_error(None, exc) is None
    response = inference_aborted_http_exception(exc)
    assert response.status_code == 503
    if kind == "insufficient_memory":
        assert response.detail["error"]["code"] == "insufficient_memory"
        assert response.headers == {"x-should-retry": "false"}
    memory_ceiling.assert_not_called()


def test_unknown_memory_ceiling_preserves_abort(monkeypatch):
    monkeypatch.setattr(
        helpers, "memory_context_ceiling", lambda engine, **kwargs: None
    )
    exc = InferenceAbortedError(
        "private", error_kind="insufficient_memory", prompt_tokens=101
    )
    assert helpers.context_overflow_from_memory_error(None, exc) is None
    assert (
        helpers.context_overflow_from_memory_error(None, ValueError("private")) is None
    )


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages"]
)
@pytest.mark.parametrize("kind", ["admission", "oom"])
def test_stream_memory_overflow_uses_existing_protocol(path, kind, memory_ceiling):
    async def collect():
        async def broken():
            yield "data: first\n\n"
            if kind == "admission":
                raise MemoryContextExceededError(prompt_tokens=101, memory_ceiling=80)
            raise InferenceAbortedError(
                "private", error_kind="insufficient_memory", prompt_tokens=101
            )

        async def connected():
            return False

        request = SimpleNamespace(
            url=SimpleNamespace(path=path), is_disconnected=connected
        )
        return [
            frame
            async for frame in helpers._disconnect_guard(
                broken(), request, keepalive_seconds=0
            )
        ]

    frames = asyncio.run(collect())
    assert frames[0] == "data: first\n\n"
    wire = "".join(frames[1:])
    assert "private" not in wire
    if path == "/v1/messages":
        assert "event: error" in wire
        assert "prompt is too long: 101 tokens > 80 maximum" in wire
        assert "overloaded_error" not in wire
    else:
        assert "context_length_exceeded" in wire
        if path == "/v1/responses":
            assert "event: response.failed" in wire
        else:
            assert frames[-1] == "data: [DONE]\n\n"


def test_front_admission_maps_only_permanent_size_failure(memory_ceiling):
    def refuse():
        raise MemoryContextExceededError(prompt_tokens=101, memory_ceiling=80)

    with pytest.raises(helpers.ContextLengthExceeded):
        helpers._check_admission_or_503(SimpleNamespace(check_admission=refuse))
    assert not issubclass(MemoryContextExceededError, BackpressureError)


def test_front_admission_keeps_transient_retry():
    def refuse():
        raise BackpressureError("busy")

    with pytest.raises(HTTPException) as caught:
        helpers._check_admission_or_503(SimpleNamespace(check_admission=refuse))
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "1"}


@pytest.mark.parametrize("surface", ["chat", "completions", "responses", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["admission", "oom", "pressure", "small_oom"])
def test_runtime_route_memory_classification(
    surface, stream, kind, monkeypatch, memory_ceiling
):
    from rapid_mlx.errors import MetalMemoryBackpressureError
    from rapid_mlx.routes import anthropic, chat, completions, responses
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    def failure():
        if kind == "admission":
            return MemoryContextExceededError(prompt_tokens=101, memory_ceiling=80)
        if kind == "pressure":
            return MetalMemoryBackpressureError("other requests hold memory")
        return InferenceAbortedError(
            "private",
            error_kind="insufficient_memory",
            prompt_tokens=101 if kind == "oom" else 79,
        )

    class RuntimeEngine(_StubEngine):
        def __init__(self):
            super().__init__()
            self.reservations = 0
            self.releases = 0

        def check_admission(self):
            self.reservations += 1

        def release_admission_reservation(self):
            self.reservations -= 1
            self.releases += 1

        async def chat(self, **kwargs):
            raise failure()

        async def generate(self, **kwargs):
            raise failure()

        async def stream_chat(self, *args, **kwargs):
            raise failure()
            yield  # pragma: no cover

        async def stream_generate(self, *args, **kwargs):
            raise failure()
            yield  # pragma: no cover

    cases = {
        "chat": (
            chat.router,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        ),
        "completions": (completions.router, "/v1/completions", {"prompt": "hi"}),
        "responses": (responses.router, "/v1/responses", {"input": "hi"}),
        "anthropic": (
            anthropic.router,
            "/v1/messages",
            {"messages": [{"role": "user", "content": "hi"}]},
        ),
    }
    inference_emit = Mock()
    monkeypatch.setattr(
        "rapid_mlx.telemetry.inference.emit_completed_request", inference_emit
    )
    router, path, body = cases[surface]
    engine = RuntimeEngine()
    client = _make_app([router], engine=engine)
    response = client.post(
        path,
        json={"model": "qwen3-0.6b-8bit", "max_tokens": 2, "stream": stream, **body},
    )
    if stream and surface == "responses" and kind in {"admission", "oom"}:
        assert response.status_code == 200
        if kind in {"admission", "oom"}:
            if surface == "anthropic":
                assert "prompt is too long: 101 tokens > 80 maximum" in response.text
            else:
                assert "context_length_exceeded" in response.text
        elif kind == "small_oom":
            assert (
                "overloaded_error" if surface == "anthropic" else "insufficient_memory"
            ) in response.text
    elif kind in {"admission", "oom"}:
        assert response.status_code == 400, response.text
        if surface == "anthropic":
            assert (
                response.json()["error"]["message"]
                == "prompt is too long: 101 tokens > 80 maximum"
            )
        else:
            assert response.json()["error"]["code"] == "context_length_exceeded"
    else:
        assert response.status_code == 503, response.text
        if kind == "pressure":
            assert response.headers["retry-after"] == "1"
        else:
            assert response.headers["x-should-retry"] == "false"

    if stream:
        failed = [
            call
            for call in inference_emit.call_args_list
            if call.kwargs["result"] == "failed"
        ]
        assert len(failed) == 1
    assert memory_ceiling.call_count == (1 if kind in {"admission", "oom"} else 0)
    assert engine.reservations == 0
    assert engine.releases == 1


@pytest.mark.parametrize("tokens,status", [(101, 400), (79, 503)])
def test_mllm_oom_before_stream_headers_is_classified(tokens, status, memory_ceiling):
    from rapid_mlx.routes.chat import router
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    class MllmEngine(_StubEngine):
        is_mllm = True

        async def stream_chat(self, **kwargs):
            raise InferenceAbortedError(
                "private", error_kind="insufficient_memory", prompt_tokens=tokens
            )
            yield  # pragma: no cover

    response = _make_app([router], engine=MllmEngine()).post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
        },
    )
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == (
        "context_length_exceeded" if status == 400 else "insufficient_memory"
    )


def test_already_translated_overflow_preserves_context_protocol():
    error = helpers.ContextLengthExceeded(
        prompt_tokens=101, limit=80, message="overflow"
    )
    assert helpers.context_overflow_from_memory_error(None, error) is error


def test_admission_overflow_reports_total_requested_tokens():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.middleware.exception_handlers import install_exception_handlers

    error = MemoryContextExceededError(
        prompt_tokens=10, memory_ceiling=50, requested_tokens=100
    )
    assert error.prompt_tokens == 10
    overflow = helpers.context_overflow_from_memory_error(None, error)
    assert overflow.prompt_tokens == 100
    assert "you requested 100 tokens" in overflow.error_payload["message"]
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/v1/messages")
    def reject():
        raise overflow

    response = TestClient(app).get("/v1/messages")
    assert response.status_code == 400
    assert (
        response.json()["error"]["message"]
        == "prompt is too long: 100 tokens > 50 maximum"
    )


def test_real_oom_uses_advisory_ceiling_even_when_admission_cap_disabled():
    from rapid_mlx.config import reset_config

    reset_config()
    scheduler = SimpleNamespace(
        memory_context_ceiling=lambda native: None,
        projected_memory_max_context=lambda native: 80,
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        _model=SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262144)),
    )
    assert helpers.enforce_context_length(engine, 100, max_tokens=2) == 2
    error = helpers.context_overflow_from_memory_error(
        engine,
        InferenceAbortedError(
            "private", error_kind="insufficient_memory", prompt_tokens=100
        ),
    )
    assert error.status_code == 400
    assert error.limit == 80


@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"]
)
def test_midstream_backpressure_is_retryable_protocol_error(path):
    async def collect():
        async def broken():
            yield "data: first\n\n"
            raise BackpressureError("server busy; retry")

        async def connected():
            return False

        request = SimpleNamespace(
            url=SimpleNamespace(path=path), is_disconnected=connected
        )
        return "".join(
            [
                frame
                async for frame in helpers._disconnect_guard(
                    broken(), request, keepalive_seconds=0
                )
            ]
        )

    wire = asyncio.run(collect())
    assert "Internal error" not in wire
    assert ("overloaded_error" if path == "/v1/messages" else "server_busy") in wire
    if path == "/v1/responses":
        assert "response.failed" in wire


@pytest.mark.parametrize("mode", ["guided", "postgen"])
@pytest.mark.parametrize("kind", ["context", "pressure"])
def test_strict_chat_admission_precedes_headers(
    mode, kind, memory_ceiling, monkeypatch
):
    from rapid_mlx.routes import chat
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    monkeypatch.setattr(helpers, "memory_context_ceiling", lambda *a, **kw: 1000)

    class Engine(_StubEngine):
        supports_guided_generation = mode == "guided"

        async def generate_with_schema(self, **kwargs):
            raise failure()

        async def stream_chat(self, **kwargs):
            raise failure()
            yield  # pragma: no cover

    def failure():
        if kind == "context":
            return MemoryContextExceededError(prompt_tokens=101, memory_ceiling=80)
        return BackpressureError("busy")

    response = _make_app([chat.router], engine=Engine()).post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {"type": "object", "properties": {}},
                },
            },
        },
    )
    assert response.status_code == (400 if kind == "context" else 503), response.text
    if kind == "context":
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert memory_ceiling.call_count == 1
    else:
        assert response.headers["retry-after"] == "1"


@pytest.mark.parametrize(
    "terminal",
    [
        "empty",
        "output",
        "admitted",
        "admitted_wait",
        "admitted_empty",
        "disconnect",
        "cancel",
        "fault",
    ],
)
def test_stream_preflight_owns_pending_work(terminal, monkeypatch):
    async def run():
        closed = asyncio.Event()
        release = asyncio.Event()
        holder = [None]
        aborted = []
        monkeypatch.setattr(
            helpers, "_force_abort_request", lambda engine, ids: aborted.append(ids[0])
        )

        async def connected():
            return terminal == "disconnect"

        request = SimpleNamespace(is_disconnected=connected)

        async def stream():
            try:
                if terminal == "empty":
                    return
                if terminal == "fault":
                    raise ValueError("test fault")
                if terminal == "output":
                    async for _ in helpers._admission_aware_stream(output()):
                        yield "data: output\n\n"
                    return
                holder[0] = "admitted-id"
                if terminal.startswith("admitted"):
                    helpers._stream_admission_event().set()
                if terminal in {"admitted_wait", "admitted_empty"}:
                    await release.wait()
                    if terminal == "admitted_empty":
                        return
                yield "data: lifecycle\n\n"
                await release.wait()
                yield "data: content\n\n"
            finally:
                closed.set()

        async def output():
            yield object()

        task = asyncio.create_task(
            helpers._preflight_stream(
                stream(), request, engine=object(), request_id_holder=holder
            )
        )
        if terminal == "cancel":
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif terminal == "disconnect":
            with pytest.raises(HTTPException) as exc:
                await task
            assert exc.value.status_code == 499
        elif terminal == "fault":
            replay = await task
            # Unrelated failures retain the guard's established SSE contract;
            # only typed admission failures move before HTTP headers.
            with pytest.raises(ValueError, match="test fault"):
                await anext(replay)
        else:
            replay = await asyncio.wait_for(task, 5)
            if terminal.startswith("admitted"):
                # Admission must finish while the first model token is blocked.
                assert not release.is_set()
                if terminal == "admitted_empty":
                    release.set()
                    assert [chunk async for chunk in replay] == []
                    assert closed.is_set()
                    return
                if terminal == "admitted_wait":
                    release.set()
                assert await anext(replay) == "data: lifecycle\n\n"
                release.set()
                assert await anext(replay) == "data: content\n\n"
                await replay.aclose()
            else:
                frames = [chunk async for chunk in replay]
                assert frames == ([] if terminal == "empty" else ["data: output\n\n"])
        assert closed.is_set()
        assert aborted == (
            ["admitted-id"] if terminal in {"cancel", "disconnect"} else []
        )

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["pressure", "small_oom"])
def test_responses_midstream_failure_remains_structured(kind, memory_ceiling):
    from rapid_mlx.engine import GenerationOutput
    from rapid_mlx.routes import responses
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    class Engine(_StubEngine):
        async def stream_chat(self, **kwargs):
            yield GenerationOutput(
                text="hello",
                new_text="hello",
                finished=False,
                prompt_tokens=1,
                completion_tokens=1,
            )
            if kind == "pressure":
                raise BackpressureError("server busy; retry")
            raise InferenceAbortedError(
                "private", error_kind="insufficient_memory", prompt_tokens=1
            )

    response = _make_app([responses.router], engine=Engine()).post(
        "/v1/responses",
        json={
            "model": "qwen3-0.6b-8bit",
            "input": "hi",
            "stream": True,
            "max_output_tokens": 2,
        },
    )
    assert response.status_code == 200
    assert "response.failed" in response.text
    assert (
        "server_busy" if kind == "pressure" else "insufficient_memory"
    ) in response.text
    assert "internal_error" not in response.text


def test_completion_preflight_replays_successful_stream():
    from rapid_mlx.engine import GenerationOutput
    from rapid_mlx.routes import completions
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    class Engine(_StubEngine):
        async def stream_generate(self, **kwargs):
            yield GenerationOutput(
                text="hello",
                new_text="hello",
                finished=True,
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

    response = _make_app([completions.router], engine=Engine()).post(
        "/v1/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "prompt": "hi",
            "stream": True,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200
    assert response.text.count('"text": "hello"') == 1
    assert response.text.count("data: [DONE]") == 1


def test_anthropic_preflight_replays_successful_stream():
    from rapid_mlx.engine import GenerationOutput
    from rapid_mlx.routes import anthropic
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    class Engine(_StubEngine):
        async def stream_chat(self, **kwargs):
            yield GenerationOutput(
                text="hello",
                new_text="hello",
                finished=True,
                prompt_tokens=1,
                completion_tokens=1,
                finish_reason="stop",
            )

    response = _make_app([anthropic.router], engine=Engine()).post(
        "/v1/messages",
        json={
            "model": "qwen3-0.6b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 200
    assert response.text.count("event: message_start") == 1
    assert response.text.count("event: message_stop") == 1


def test_streaming_invalid_media_retains_client_error():
    from rapid_mlx.request import ClientRequestError
    from rapid_mlx.routes import chat
    from tests.test_context_overflow_400 import _make_app, _StubEngine

    class Engine(_StubEngine):
        is_mllm = True

        async def stream_chat(self, **kwargs):
            raise ClientRequestError("invalid image data")
            yield  # pragma: no cover

    response = _make_app([chat.router], engine=Engine()).post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-0.6b-8bit",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 2,
        },
    )
    assert response.status_code == 400
    assert "invalid image data" in response.text
