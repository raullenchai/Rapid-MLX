# SPDX-License-Identifier: Apache-2.0
"""Malformed tool history fails before conversion or engine dispatch (#5195)."""

from copy import deepcopy

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
from rapid_mlx.api.anthropic_models import AnthropicRequest
from rapid_mlx.api.models import ChatCompletionRequest, Message
from rapid_mlx.api.responses_adapter import responses_to_openai
from rapid_mlx.api.responses_models import ResponsesRequest
from rapid_mlx.middleware.exception_handlers import install_exception_handlers

CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": "{}"},
}
MISSING = object()


@pytest.fixture
def client():
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/v1/chat/completions")
    def chat(body: ChatCompletionRequest):
        return body.model_dump()

    @app.post("/v1/messages")
    async def messages(request: Request):
        return anthropic_to_openai(
            AnthropicRequest(**await request.json())
        ).model_dump()

    @app.post("/v1/responses")
    async def responses(request: Request):
        return responses_to_openai(
            ResponsesRequest(**await request.json())
        ).model_dump()

    with TestClient(app) as session:
        yield session


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in ("id", "type", "function", "function.name", "function.arguments")
        for value in (MISSING, None, 7, True, [], {})
        if not (field == "function" and value == {})
    ]
    + [("function", "bad")],
)
def test_malformed_call_returns_field_named_400(client, field, value):
    call = deepcopy(CALL)
    parent = call["function"] if field.startswith("function.") else call
    key = field.split(".")[-1]
    if value is MISSING:
        del parent[key]
    else:
        parent[key] = value
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [call]},
            ],
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "messages.1.tool_calls" in error["message"]
    assert f"tool_calls[0].{field}" in error["message"]


@pytest.mark.parametrize("parameters", ["bad", None, 1, True, [], ["object"]])
def test_tool_parameters_must_be_object(client, parameters):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": parameters},
                }
            ],
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "tools.0" in error["message"]
    assert "function.parameters" in error["message"]


@pytest.mark.parametrize(
    "parameters", [MISSING, {}, {"type": "object", "properties": {}}]
)
def test_valid_history_and_schemas_preserve_wire_data(client, parameters):
    call = deepcopy(CALL)
    call["extension"] = {"opaque": True}
    function = {"name": "lookup"}
    if parameters is not MISSING:
        function["parameters"] = parameters
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "assistant", "tool_calls": [call]}],
            "tools": [{"type": "function", "function": function}],
        },
    )
    assert response.status_code == 200
    assert response.json()["messages"][0]["tool_calls"] == [call]
    assert response.json()["tools"][0]["function"] == function
    assert Message(role="assistant", content="", tool_calls=[]).tool_calls == []
    assert Message(role="user", content="hi", tool_calls=None).tool_calls is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (path, value)
        for path in ("/v1/messages", "/v1/responses")
        for value in (None, 123, [], {})
        if not (path == "/v1/responses" and value is None)
    ],
)
def test_shared_conversion_rejects_non_string_names(client, path, value):
    if path == "/v1/messages":
        body = {
            "model": "test",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_1", "name": value, "input": {}}
                    ],
                }
            ],
        }
    else:
        # Responses historically supplies an empty name for omitted/null names;
        # this path cannot trigger the missing dictionary-key crash.
        body = {
            "model": "test",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": value,
                    "arguments": "{}",
                }
            ],
        }
    response = client.post(path, json=body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "name" in response.json()["error"]["message"]


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
def test_valid_shared_tool_history_conversion(client, path):
    if path == "/v1/messages":
        body = {
            "model": "test",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                }
            ],
        }
    else:
        body = {
            "model": "test",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
        }
    response = client.post(path, json=body)
    assert response.status_code == 200
    assert response.json()["messages"][0]["tool_calls"] == [CALL]
