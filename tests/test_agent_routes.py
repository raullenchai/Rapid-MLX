# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from vllm_mlx.agent_runtime import AgentRunStatus, ToolRisk, ToolSpec
from vllm_mlx.agent_runtime.server import (
    AgentEventsView,
    AgentRunCapacityError,
    AgentRunConflictError,
    AgentRunCreateRequest,
    AgentRunNotFoundError,
    AgentRunView,
    AgentServerError,
    AgentToolSelectionError,
    generate_chat_turn,
)
from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
)
from vllm_mlx.config import get_config, reset_config
from vllm_mlx.middleware.auth import check_rate_limit, verify_api_key
from vllm_mlx.routes import agents as agent_routes


@pytest.mark.asyncio
async def test_chat_driver_reuses_non_stream_route_and_decodes_tool_call(monkeypatch):
    captured = []
    response = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(
                                name="files__read_file",
                                arguments=json.dumps({"path": "notes.txt"}),
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    async def fake_chat(request, raw_request):
        captured.append((request, raw_request))
        return Response(
            content=response.model_dump_json(exclude_none=True),
            media_type="application/json",
        )

    from vllm_mlx.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "create_chat_completion", fake_chat)
    settings = AgentRunCreateRequest(goal="Read notes")
    tool = ToolSpec(
        name="files__read_file",
        risk=ToolRisk.READ_ONLY,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    )

    turn = await generate_chat_turn(
        "served", [{"role": "user", "content": "Read notes"}], [tool], settings
    )

    request = captured[0][0]
    assert request.stream is False
    assert request.parallel_tool_calls is False
    assert request.max_tokens == 900
    assert request.timeout == 300.0
    assert request.temperature == 0.7
    assert request.top_p == 0.95
    assert request.enable_thinking is False
    assert turn.tool_calls[0].arguments == {"path": "notes.txt"}


@pytest.mark.asyncio
async def test_chat_driver_rejects_output_limit_truncation(monkeypatch):
    response = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content="truncated"),
                finish_reason="length",
            )
        ],
    )

    async def fake_chat(*_args):
        return Response(content=response.model_dump_json(exclude_none=True))

    from vllm_mlx.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "create_chat_completion", fake_chat)

    with pytest.raises(AgentServerError, match="output limit"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["not json", "[]"])
async def test_chat_driver_fails_closed_on_malformed_tool_arguments(
    monkeypatch, arguments
):
    response = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(name="tool", arguments=arguments),
                        )
                    ]
                )
            )
        ],
    )

    async def fake_chat(*_args):
        return Response(content=response.model_dump_json(exclude_none=True))

    from vllm_mlx.routes import chat as chat_routes

    monkeypatch.setattr(chat_routes, "create_chat_completion", fake_chat)

    with pytest.raises(AgentServerError, match="tool arguments"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )


@pytest.mark.asyncio
async def test_chat_driver_rejects_unsuccessful_or_ambiguous_response(monkeypatch):
    from vllm_mlx.routes import chat as chat_routes

    async def unavailable(*_args):
        return Response(status_code=503)

    monkeypatch.setattr(chat_routes, "create_chat_completion", unavailable)
    with pytest.raises(AgentServerError, match="successful response"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )

    ambiguous = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(message=AssistantMessage(content="one")),
            ChatCompletionChoice(message=AssistantMessage(content="two")),
        ],
    )

    async def two_choices(*_args):
        return Response(content=ambiguous.model_dump_json(exclude_none=True))

    monkeypatch.setattr(chat_routes, "create_chat_completion", two_choices)
    with pytest.raises(AgentServerError, match="choice count"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )

    filtered = ChatCompletionResponse(
        model="served",
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content="filtered"),
                finish_reason="content_filter",
            )
        ],
    )

    async def invalid_finish(*_args):
        return Response(content=filtered.model_dump_json(exclude_none=True))

    monkeypatch.setattr(chat_routes, "create_chat_completion", invalid_finish)
    with pytest.raises(AgentServerError, match="invalid finish reason"):
        await generate_chat_turn(
            "served",
            [{"role": "user", "content": "x"}],
            [],
            AgentRunCreateRequest(goal="x"),
        )


class _RouteService:
    def __init__(self):
        self.created = []

    async def create(
        self,
        request,
        *,
        model,
        request_model=None,
        profile_model_config=None,
        profile_tool_call_parser=None,
        model_generation=None,
    ):
        if request.goal == "capacity":
            raise AgentRunCapacityError("full")
        if request.goal == "bad tools":
            raise AgentToolSelectionError("bad selection")
        self.created.append(
            (
                request,
                model,
                request_model,
                profile_model_config,
                profile_tool_call_parser,
                model_generation,
            )
        )
        return self.view()

    @staticmethod
    def view():
        return AgentRunView(
            id="run-1",
            model="model",
            profile="minicpm5-2b",
            status=AgentRunStatus.READY,
            model_turns=0,
            tool_rounds=0,
            final_synthesis=False,
        )

    async def get(self, run_id):
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return self.view()

    async def events(self, run_id, *, after):
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return AgentEventsView(
            run_id=run_id,
            status=AgentRunStatus.READY,
            events=[],
            next_after=after,
        )

    async def approve(self, run_id, request):
        if run_id == "conflict":
            raise AgentRunConflictError("conflict")
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return self.view()

    async def submit_result(self, run_id, request):
        return await self.approve(run_id, request)

    async def cancel(self, run_id):
        if run_id == "conflict":
            raise AgentRunConflictError("conflict")
        if run_id == "missing":
            raise AgentRunNotFoundError("missing")
        return self.view()


def test_agent_routes_require_bearer_and_bind_profile_to_real_model(monkeypatch):
    cfg = reset_config()
    cfg.api_key = "secret"
    cfg.model_name = "pretty-served-name"
    cfg.model_alias = "minicpm5-2b-4bit"
    cfg.model_path = "openbmb/MiniCPM5-2B-MLX"
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        assert client.post("/v1/agent/runs", json={"goal": "x"}).status_code == 401
        response = client.post(
            "/v1/agent/runs",
            headers={"Authorization": "Bearer secret"},
            json={"goal": "x", "model": "pretty-served-name"},
        )

    assert response.status_code == 202
    assert service.created[0][1:3] == (
        "openbmb/MiniCPM5-2B-MLX",
        "pretty-served-name",
    )
    reset_config()


def test_agent_create_rejects_unknown_model_before_starting_background_work(
    monkeypatch,
):
    cfg = reset_config()
    cfg.model_name = "known"
    cfg.model_path = "openbmb/MiniCPM5-2B-MLX"
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x", "model": "unknown"})

    assert response.status_code == 404
    assert service.created == []
    reset_config()


def test_agent_create_requires_a_configured_model(monkeypatch):
    reset_config()
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 503
    assert service.created == []
    reset_config()


def test_agent_create_resolves_registry_model_identity(monkeypatch):
    from types import SimpleNamespace

    class Registry:
        def __bool__(self):
            return True

        def __contains__(self, name):
            return name == "served"

        def get_entry(self, name):
            assert name == "served"
            return SimpleNamespace(
                model_path="openbmb/MiniCPM5-2B-MLX",
                model_name="canonical",
                tool_call_parser="minicpm",
            )

    cfg = reset_config()
    cfg.model_name = "served"
    cfg.model_registry = Registry()
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 202
    assert service.created[0][1:3] == (
        "openbmb/MiniCPM5-2B-MLX",
        "served",
    )
    reset_config()


def test_agent_create_qualifies_custom_local_minicpm_from_metadata(monkeypatch):
    from types import SimpleNamespace

    minicpm_config = {
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 130560,
    }

    class Registry:
        def __bool__(self):
            return True

        def __contains__(self, name):
            return name == "friendly-name"

        def get_entry(self, _name):
            return SimpleNamespace(
                model_path="/models/arbitrary-folder",
                model_name="friendly-name",
                tool_call_parser="minicpm",
            )

    cfg = reset_config()
    cfg.model_name = "friendly-name"
    cfg.model_registry = Registry()
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    monkeypatch.setattr(
        agent_routes,
        "read_model_metadata",
        lambda _path: SimpleNamespace(config=minicpm_config),
    )
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 202
    assert service.created[0][1] == "/models/arbitrary-folder"
    assert service.created[0][3] == minicpm_config
    assert service.created[0][4] == "minicpm"
    reset_config()


def test_agent_create_maps_model_metadata_failure_to_503(monkeypatch):
    from types import SimpleNamespace

    class Registry:
        def __contains__(self, name):
            return name == "known"

        def get_entry(self, _name):
            return SimpleNamespace(
                model_path="/models/unreadable",
                model_name="known",
                tool_call_parser=None,
            )

    cfg = reset_config()
    cfg.model_name = "known"
    cfg.model_registry = Registry()

    def fail_read(_path):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(agent_routes, "read_model_metadata", fail_read)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        response = client.post("/v1/agent/runs", json={"goal": "x"})

    assert response.status_code == 503
    assert response.json() == {"detail": "model metadata is temporarily unavailable"}
    assert "private filesystem detail" not in response.text
    reset_config()


@pytest.mark.asyncio
async def test_single_model_metadata_is_cached_per_engine_generation(monkeypatch):
    from types import SimpleNamespace

    reads = []

    def read(path):
        reads.append(path)
        return SimpleNamespace(config={"model_type": "llama"})

    monkeypatch.setattr(agent_routes, "read_model_metadata", read)
    monkeypatch.setattr(agent_routes, "_single_model_metadata_cache", None)
    first_engine = object()
    second_engine = object()

    assert await agent_routes._single_model_config(first_engine, "/model") == {
        "model_type": "llama"
    }
    assert await agent_routes._single_model_config(first_engine, "/model") == {
        "model_type": "llama"
    }
    assert await agent_routes._single_model_config(second_engine, "/model") == {
        "model_type": "llama"
    }
    assert reads == ["/model", "/model"]


@pytest.mark.asyncio
async def test_registry_entry_metadata_is_cached(monkeypatch):
    from types import SimpleNamespace

    reads = []
    entry = SimpleNamespace(model_path="/model")

    def read(path):
        reads.append(path)
        return SimpleNamespace(config={"model_type": "llama"})

    monkeypatch.setattr(agent_routes, "read_model_metadata", read)
    first = await agent_routes._entry_model_config(entry)
    second = await agent_routes._entry_model_config(entry)
    assert first == second == {"model_type": "llama"}
    assert reads == ["/model"]


def test_agent_create_maps_registry_lookup_and_single_metadata_failures(monkeypatch):
    class MissingRegistry:
        def __contains__(self, _name):
            return True

        def get_entry(self, _name):
            raise KeyError("missing generation")

    cfg = reset_config()
    cfg.model_name = "known"
    cfg.model_registry = MissingRegistry()
    app = FastAPI()
    app.include_router(agent_routes.router)
    with TestClient(app) as client:
        missing = client.post("/v1/agent/runs", json={"goal": "x"})
    assert missing.status_code == 404

    cfg.model_registry = None
    cfg.model_path = "/models/unreadable"

    def fail_read(_path):
        raise ValueError("private parse detail")

    monkeypatch.setattr(agent_routes, "read_model_metadata", fail_read)
    with TestClient(app) as client:
        unavailable = client.post("/v1/agent/runs", json={"goal": "x"})
    assert unavailable.status_code == 503
    assert "private parse detail" not in unavailable.text
    reset_config()


def test_http_error_maps_registry_unavailable_to_503():
    from vllm_mlx.agent_runtime.server import AgentToolRegistryUnavailableError

    error = agent_routes._http_error(
        AgentToolRegistryUnavailableError("registry unavailable")
    )
    assert error.status_code == 503


def test_agent_http_surface_maps_success_and_stable_failures(monkeypatch):
    cfg = reset_config()
    cfg.model_name = "known"
    service = _RouteService()
    monkeypatch.setattr(agent_routes, "get_agent_service", lambda: service)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        assert client.get("/v1/agent/runs/run-1").status_code == 200
        assert client.get("/v1/agent/runs/missing").status_code == 404
        events = client.get("/v1/agent/runs/run-1/events?after=7")
        assert events.status_code == 200
        assert events.json()["next_after"] == 7
        assert client.get("/v1/agent/runs/missing/events").status_code == 404

        approval = {"call_id": "call", "approved": True}
        assert (
            client.post("/v1/agent/runs/run-1/approval", json=approval).status_code
            == 200
        )
        assert (
            client.post("/v1/agent/runs/conflict/approval", json=approval).status_code
            == 409
        )
        assert (
            client.post("/v1/agent/runs/missing/approval", json=approval).status_code
            == 404
        )

        result = {"call_id": "call", "content": "ok", "executed": True}
        assert (
            client.post("/v1/agent/runs/run-1/tool-result", json=result).status_code
            == 200
        )
        assert (
            client.post("/v1/agent/runs/conflict/tool-result", json=result).status_code
            == 409
        )
        assert client.post("/v1/agent/runs/run-1/cancel").status_code == 200
        assert client.post("/v1/agent/runs/conflict/cancel").status_code == 409
        assert client.post("/v1/agent/runs/missing/cancel").status_code == 404

        assert (
            client.post("/v1/agent/runs", json={"goal": "capacity"}).status_code == 503
        )
        assert (
            client.post("/v1/agent/runs", json={"goal": "bad tools"}).status_code == 422
        )

    reset_config()


@pytest.mark.asyncio
async def test_agent_route_singleton_closes_and_resets(monkeypatch):
    closed = []

    class Service:
        async def close(self):
            closed.append(True)

    monkeypatch.setattr(agent_routes, "_service", Service())
    monkeypatch.setattr(agent_routes, "_service_shutting_down", False)
    await agent_routes.close_agent_service()
    await agent_routes.close_agent_service()

    assert closed == [True]
    assert agent_routes._service is None
    with pytest.raises(AgentRunCapacityError, match="shutting down"):
        agent_routes.get_agent_service()

    agent_routes.start_agent_service_lifecycle()
    assert agent_routes._service_shutting_down is False


@pytest.mark.asyncio
async def test_agent_route_singleton_rejects_replacement_during_shutdown(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    class Service:
        async def close(self):
            entered.set()
            await release.wait()

    original = Service()
    monkeypatch.setattr(agent_routes, "_service", original)
    monkeypatch.setattr(agent_routes, "_service_shutting_down", False)
    closing = asyncio.create_task(agent_routes.close_agent_service())
    await entered.wait()

    with pytest.raises(AgentRunCapacityError, match="shutting down"):
        agent_routes.get_agent_service()
    assert agent_routes._service is original

    release.set()
    await closing
    assert agent_routes._service is None


@pytest.mark.asyncio
async def test_agent_route_singleton_retains_service_after_failed_shutdown(monkeypatch):
    class Service:
        async def close(self):
            raise AgentRunCapacityError("live work remains")

    original = Service()
    monkeypatch.setattr(agent_routes, "_service", original)
    monkeypatch.setattr(agent_routes, "_service_shutting_down", False)

    with pytest.raises(AgentRunCapacityError, match="live work remains"):
        await agent_routes.close_agent_service()

    assert agent_routes._service is original
    assert agent_routes._service_shutting_down is True
    with pytest.raises(AgentRunCapacityError, match="shutting down"):
        agent_routes.get_agent_service()
    with pytest.raises(RuntimeError, match="survived"):
        agent_routes.start_agent_service_lifecycle()


def test_agent_route_singleton_is_lazy(monkeypatch):
    instance = object()
    monkeypatch.setattr(agent_routes, "_service", None)
    monkeypatch.setattr(agent_routes, "AgentServerService", lambda: instance)

    assert agent_routes.get_agent_service() is instance
    assert agent_routes.get_agent_service() is instance


def test_all_agent_routes_share_auth_and_rate_limit_dependencies():
    route_dependencies = [
        route.dependencies
        for route in agent_routes.router.routes
        if hasattr(route, "dependencies")
    ]

    assert route_dependencies
    expected = {verify_api_key, check_rate_limit}
    assert all(
        {dependency.dependency for dependency in dependencies} == expected
        for dependencies in route_dependencies
    )


def test_all_agent_routes_map_shutdown_to_503(monkeypatch):
    cfg = reset_config()
    cfg.model_name = "known"
    monkeypatch.setattr(agent_routes, "_service", None)
    monkeypatch.setattr(agent_routes, "_service_shutting_down", True)
    app = FastAPI()
    app.include_router(agent_routes.router)

    with TestClient(app) as client:
        responses = [
            client.post("/v1/agent/runs", json={"goal": "x"}),
            client.get("/v1/agent/runs/run"),
            client.get("/v1/agent/runs/run/events"),
            client.post(
                "/v1/agent/runs/run/approval",
                json={"call_id": "call", "approved": False},
            ),
            client.post(
                "/v1/agent/runs/run/tool-result",
                json={"call_id": "call", "content": "x", "executed": False},
            ),
            client.post("/v1/agent/runs/run/cancel"),
        ]

    assert [response.status_code for response in responses] == [503] * 6
    reset_config()
    assert get_config() is not None
