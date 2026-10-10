# SPDX-License-Identifier: Apache-2.0
"""Reject missing custom-tool schemas before inference (#4455)."""

import pytest
from pydantic import ValidationError

from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
from rapid_mlx.api.anthropic_models import AnthropicRequest

pytest_plugins = ("tests.test_anthropic_output_config",)


SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
INVALID_TOOLS = [
    {"name": "get_weather"},
    {"type": "function", "name": "get_weather", "parameters": SCHEMA},
    {"type": "function", "function": {"name": "get_weather", "parameters": SCHEMA}},
    *[
        {"name": "get_weather", "input_schema": value}
        for value in (None, [], "{}", False)
    ],
]


def _payload(tools, **extra):
    return {
        "model": "test-model",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Weather in Paris? Use the tool."}],
        "tools": tools,
        **extra,
    }


@pytest.mark.parametrize("tool", INVALID_TOOLS)
def test_tool_schema_required_at_parse_time(tool):
    with pytest.raises(ValidationError) as exc:
        AnthropicRequest(**_payload([tool]))
    assert ("tools", 0, "input_schema") in [e["loc"] for e in exc.value.errors()]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_choice", [None, {"type": "auto"}, {"type": "any"}])
@pytest.mark.parametrize("tool", INVALID_TOOLS)
def test_invalid_tool_schema_returns_400_before_inference(
    anthropic_client, tool, stream, tool_choice
):
    extra = {"stream": stream}
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    response = anthropic_client.client.post(
        "/v1/messages", json=_payload([tool], **extra)
    )
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "tools.0.input_schema" in error["message"]
    assert anthropic_client.engine.calls == []


@pytest.mark.parametrize("schema", [SCHEMA, {"type": "object", "properties": {}}, {}])
def test_native_tool_schema_preserved(schema):
    request = AnthropicRequest(
        **_payload([{"name": "get_weather", "input_schema": schema}])
    )
    converted = anthropic_to_openai(request)
    assert converted.tools[0].function["parameters"] == schema


def test_second_tool_missing_schema_is_identified(anthropic_client):
    response = anthropic_client.client.post(
        "/v1/messages",
        json=_payload(
            [
                {"name": "valid", "input_schema": SCHEMA},
                {"name": "invalid", "parameters": SCHEMA},
            ]
        ),
    )
    assert response.status_code == 400, response.text
    assert "tools.1.input_schema" in response.json()["error"]["message"]
    assert anthropic_client.engine.calls == []
