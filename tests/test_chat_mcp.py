# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the MCP runtime owned by ``rapid-mlx chat``."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from vllm_mlx.chat_mcp import ChatMCPRuntime, _server_parameters
from vllm_mlx.mcp.types import MCPServerConfig, MCPTransport


class _FakeResult:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments

    def model_dump(self, **_kwargs):
        return {
            "content": [{"type": "text", "text": f"{self.name}:{self.arguments}"}],
            "isError": False,
        }


class _FakeSessionGroup:
    instances: list[_FakeSessionGroup] = []
    calls: list[tuple[str, dict]] = []
    fail_enter = False
    call_started = threading.Event()
    call_cancelled = threading.Event()
    connect_thread_names: list[str] = []

    def __init__(self, component_name_hook):
        self._name_hook = component_name_hook
        self.tools = {}
        self.exited = False
        self.instances.append(self)

    async def __aenter__(self):
        if self.fail_enter:
            raise RuntimeError("group enter failed")
        return self

    async def __aexit__(self, *_args):
        self.exited = True

    async def connect_to_server(self, params):
        self.connect_thread_names.append(threading.current_thread().name)
        label = params.args[0]
        if label == "hang":
            await asyncio.Event().wait()
        if label == "fail":
            raise RuntimeError("connection refused")
        if label == "empty":
            return
        tool_name = params.args[1]
        full_name = self._name_hook(
            tool_name,
            SimpleNamespace(name="same-name-from-every-server"),
        )
        schema = (
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
            if tool_name == "lookup"
            else {"type": "object"}
        )
        self.tools[full_name] = SimpleNamespace(
            name=tool_name,
            description=f"{label} tool",
            inputSchema=schema,
        )

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if arguments.get("hang"):
            self.call_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.call_cancelled.set()
        if arguments.get("raise"):
            raise RuntimeError("tool exploded")
        return _FakeResult(name, arguments)


@pytest.fixture(autouse=True)
def _fake_sdk_group(monkeypatch):
    import mcp.client.session_group

    _FakeSessionGroup.instances = []
    _FakeSessionGroup.calls = []
    _FakeSessionGroup.fail_enter = False
    _FakeSessionGroup.call_started = threading.Event()
    _FakeSessionGroup.call_cancelled = threading.Event()
    _FakeSessionGroup.connect_thread_names = []
    monkeypatch.setattr(
        mcp.client.session_group,
        "ClientSessionGroup",
        _FakeSessionGroup,
    )


def _write_config(tmp_path, servers, **extra):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": servers, **extra}))
    return path


def test_runtime_uses_sdk_groups_for_multiple_servers(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "alpha": {"command": "python3", "args": ["alpha", "lookup"]},
            "beta": {"command": "python3", "args": ["beta", "lookup"]},
        },
    )

    runtime = ChatMCPRuntime(str(path))
    try:
        assert runtime.server_count == 2
        assert runtime.connection_errors == {}
        assert {tool["function"]["name"] for tool in runtime.tools} == {
            "alpha__lookup",
            "beta__lookup",
        }

        messages = runtime.execute_tool_calls(
            [
                {
                    "id": "call-a",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": '{"value":"A"}',
                    },
                },
                {
                    "id": "call-b",
                    "function": {
                        "name": "beta__lookup",
                        "arguments": {"value": "B"},
                    },
                },
            ]
        )
        assert [message["tool_call_id"] for message in messages] == [
            "call-a",
            "call-b",
        ]
        assert _FakeSessionGroup.calls == [
            ("alpha__lookup", {"value": "A"}),
            ("beta__lookup", {"value": "B"}),
        ]
        assert '"isError": false' in messages[0]["content"]
    finally:
        runtime.close()

    assert all(group.exited for group in _FakeSessionGroup.instances)
    runtime.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        runtime.execute_tool_calls([])


def test_runtime_keeps_healthy_server_when_another_fails(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "broken": {"command": "python3", "args": ["fail", "lookup"]},
            "working": {"command": "python3", "args": ["working", "lookup"]},
        },
    )

    runtime = ChatMCPRuntime(str(path))
    try:
        assert runtime.server_count == 1
        assert runtime.connection_errors == {"broken": "connection refused"}
        assert runtime.tools[0]["function"]["name"] == "working__lookup"
    finally:
        runtime.close()


def test_runtime_bounds_connection_startup(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "stuck": {
                "command": "python3",
                "args": ["hang", "lookup"],
                "timeout": 0.01,
            }
        },
    )

    with pytest.raises(RuntimeError, match="No MCP tools available.*timed out"):
        ChatMCPRuntime(str(path))


def test_runtime_rejects_qualified_tool_name_collisions(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "a__b": {"command": "python3", "args": ["first", "c"]},
            "a": {"command": "python3", "args": ["second", "b__c"]},
        },
    )

    with pytest.raises(RuntimeError, match="tool name collision.*a__b__c"):
        ChatMCPRuntime(str(path))


def test_runtime_rejects_non_openai_tool_names(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "bad.name"]}},
    )

    with pytest.raises(RuntimeError, match="not a valid OpenAI function name"):
        ChatMCPRuntime(str(path))


def test_runtime_rejects_configs_without_usable_tools(tmp_path):
    disabled = _write_config(
        tmp_path,
        {"off": {"command": "python3", "enabled": False}},
    )
    with pytest.raises(ValueError, match="no enabled servers"):
        ChatMCPRuntime(str(disabled))

    empty = _write_config(
        tmp_path,
        {"empty": {"command": "python3", "args": ["empty", "ignored"]}},
    )
    with pytest.raises(RuntimeError, match="No MCP tools available"):
        ChatMCPRuntime(str(empty))


def test_runtime_returns_validation_and_execution_errors_to_model(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )
    runtime = ChatMCPRuntime(str(path))
    try:
        messages = runtime.execute_tool_calls(
            [
                {
                    "id": "bad-json",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": "{",
                    },
                },
                {
                    "id": "bad-shape",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": "[]",
                    },
                },
                {
                    "id": "bad-schema",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": '{"value":1}',
                    },
                },
                {
                    "id": "unknown",
                    "function": {
                        "name": "alpha__missing",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "tool-error",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": '{"value":"ok","raise":true}',
                    },
                },
            ]
        )
    finally:
        runtime.close()

    errors = [json.loads(message["content"])["error"] for message in messages]
    assert "Expecting property name" in errors[0]
    assert "JSON object" in errors[1]
    assert "is not of type 'string'" in errors[2]
    assert "Unknown MCP tool" in errors[3]
    assert errors[4] == "tool exploded"


def test_runtime_close_cancels_inflight_tool(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )

    runtime = ChatMCPRuntime(str(path))
    errors = []

    def _execute():
        try:
            runtime.execute_tool_calls(
                [
                    {
                        "id": "cancel-me",
                        "function": {
                            "name": "alpha__lookup",
                            "arguments": '{"value":"ok","hang":true}',
                        },
                    }
                ]
            )
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=_execute)
    caller.start()
    assert _FakeSessionGroup.call_started.wait(timeout=1)
    runtime.close()
    caller.join(timeout=1)

    assert not caller.is_alive()
    assert errors
    assert _FakeSessionGroup.call_cancelled.is_set()


def test_runtime_blocks_high_risk_tool_without_explicit_opt_in(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "ops": {
                "command": "python3",
                "args": ["ops", "run_command"],
            }
        },
    )
    runtime = ChatMCPRuntime(str(path))
    try:
        [message] = runtime.execute_tool_calls(
            [
                {
                    "id": "danger",
                    "function": {
                        "name": "ops__run_command",
                        "arguments": "{}",
                    },
                }
            ]
        )
    finally:
        runtime.close()
    assert "blocked by default" in json.loads(message["content"])["error"]
    assert _FakeSessionGroup.calls == []


def test_runtime_surfaces_sdk_startup_failure(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )
    _FakeSessionGroup.fail_enter = True
    with pytest.raises(RuntimeError, match="group enter failed"):
        ChatMCPRuntime(str(path))


def test_server_parameters_use_official_sdk_types():
    stdio = MCPServerConfig(
        name="stdio",
        command="python3",
        args=["server.py"],
        env={"TOKEN": "secret"},
    )
    stdio_params = _server_parameters(stdio)
    assert stdio_params.command == "python3"
    assert stdio_params.args == ["server.py"]
    assert stdio_params.env["TOKEN"] == "secret"

    sse = MCPServerConfig(
        name="remote",
        transport=MCPTransport.SSE,
        url="http://localhost:9999/sse",
        timeout=12,
    )
    sse_params = _server_parameters(sse)
    assert sse_params.url == "http://localhost:9999/sse"
    assert sse_params.timeout == 12
    assert sse_params.sse_read_timeout == 300


def test_runtime_thread_is_dedicated_to_chat(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )
    runtime = ChatMCPRuntime(str(path))
    try:
        assert _FakeSessionGroup.connect_thread_names == ["rapid-mlx-chat-mcp"]
    finally:
        runtime.close()
