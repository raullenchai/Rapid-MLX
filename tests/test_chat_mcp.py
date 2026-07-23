# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the MCP runtime owned by ``rapid-mlx chat``."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
import threading
from types import SimpleNamespace

import pytest

from vllm_mlx.chat_mcp import (
    ChatMCPRuntime,
    ChatToolEvent,
    _mcp_server_stderr_to,
    _quiet_optional_component_warnings,
    _server_parameters,
)
from vllm_mlx.mcp.types import MCPServerConfig, MCPTransport


class _FakeResult:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments
        self.isError = bool(arguments.get("is_error"))

    def model_dump(self, **_kwargs):
        return {
            "content": [{"type": "text", "text": f"{self.name}:{self.arguments}"}],
            "isError": self.isError,
        }


class _LaneAbort(BaseException):
    pass


class _FakeSessionGroup:
    instances: list[_FakeSessionGroup] = []
    calls: list[tuple[str, dict]] = []
    fail_enter = False
    call_started = threading.Event()
    call_cancelled = threading.Event()
    connect_thread_names: list[str] = []
    active_calls = 0
    max_active_calls = 0
    active_calls_by_server: dict[str, int] = {}
    max_active_calls_by_server: dict[str, int] = {}

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
        server_name = name.split("__", 1)[0]
        type(self).active_calls += 1
        type(self).active_calls_by_server[server_name] = (
            type(self).active_calls_by_server.get(server_name, 0) + 1
        )
        type(self).max_active_calls = max(
            type(self).max_active_calls,
            type(self).active_calls,
        )
        type(self).max_active_calls_by_server[server_name] = max(
            type(self).max_active_calls_by_server.get(server_name, 0),
            type(self).active_calls_by_server[server_name],
        )
        try:
            if arguments.get("delay"):
                await asyncio.sleep(arguments["delay"])
            if arguments.get("hang"):
                self.call_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.call_cancelled.set()
            if arguments.get("raise"):
                raise RuntimeError("tool exploded")
            return _FakeResult(name, arguments)
        finally:
            type(self).active_calls -= 1
            type(self).active_calls_by_server[server_name] -= 1


@pytest.fixture(autouse=True)
def _fake_sdk_group(monkeypatch):
    import mcp.client.session_group

    _FakeSessionGroup.instances = []
    _FakeSessionGroup.calls = []
    _FakeSessionGroup.fail_enter = False
    _FakeSessionGroup.call_started = threading.Event()
    _FakeSessionGroup.call_cancelled = threading.Event()
    _FakeSessionGroup.connect_thread_names = []
    _FakeSessionGroup.active_calls = 0
    _FakeSessionGroup.max_active_calls = 0
    _FakeSessionGroup.active_calls_by_server = {}
    _FakeSessionGroup.max_active_calls_by_server = {}
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
        assert sorted(_FakeSessionGroup.calls) == [
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


def test_runtime_parallelizes_servers_but_serializes_each_server(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "alpha": {"command": "python3", "args": ["alpha", "lookup"]},
            "beta": {"command": "python3", "args": ["beta", "lookup"]},
        },
    )
    runtime = ChatMCPRuntime(str(path))
    try:
        messages = runtime.execute_tool_calls(
            [
                {
                    "id": "alpha-first",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": {"value": "first", "delay": 0.02},
                    },
                },
                {
                    "id": "beta",
                    "function": {
                        "name": "beta__lookup",
                        "arguments": {"value": "beta", "delay": 0.02},
                    },
                },
                {
                    "id": "alpha-second",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": {"value": "second", "delay": 0.02},
                    },
                },
            ]
        )
    finally:
        runtime.close()

    assert _FakeSessionGroup.max_active_calls == 2
    assert _FakeSessionGroup.max_active_calls_by_server == {
        "alpha": 1,
        "beta": 1,
    }
    assert [message["tool_call_id"] for message in messages] == [
        "alpha-first",
        "beta",
        "alpha-second",
    ]
    alpha_calls = [
        arguments["value"]
        for name, arguments in _FakeSessionGroup.calls
        if name == "alpha__lookup"
    ]
    assert alpha_calls == ["first", "second"]


@pytest.mark.asyncio
async def test_runtime_cancels_sibling_lanes_when_one_aborts():
    runtime = object.__new__(ChatMCPRuntime)
    runtime._tools_by_name = {
        "alpha__lookup": SimpleNamespace(server_name="alpha"),
        "beta__lookup": SimpleNamespace(server_name="beta"),
    }
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()

    async def _execute_one(position, _tool_call, _on_event):
        if position == 0:
            first_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                first_cancelled.set()
        await first_started.wait()
        raise _LaneAbort

    runtime._execute_one = _execute_one
    calls = [
        {"function": {"name": "alpha__lookup"}},
        {"function": {"name": "beta__lookup"}},
    ]

    with pytest.raises(_LaneAbort):
        await runtime._execute(calls, None)

    assert first_cancelled.is_set()


def test_runtime_emits_tool_start_and_finish_events(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )
    events: list[ChatToolEvent] = []
    runtime = ChatMCPRuntime(str(path))
    try:
        [message] = runtime.execute_tool_calls(
            [
                {
                    "id": "call-a",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": '{"value":"A"}',
                    },
                }
            ],
            on_event=events.append,
        )
    finally:
        runtime.close()

    assert [(event.phase, event.call_id, event.name) for event in events] == [
        ("start", "call-a", "alpha__lookup"),
        ("finish", "call-a", "alpha__lookup"),
    ]
    assert events[0].elapsed_seconds is None
    assert events[0].message is None
    assert events[1].elapsed_seconds is not None
    assert events[1].elapsed_seconds >= 0
    assert events[1].is_error is False
    assert events[1].message == message


def test_runtime_finish_event_marks_mcp_errors(tmp_path):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )
    events: list[ChatToolEvent] = []
    runtime = ChatMCPRuntime(str(path))
    try:
        [message] = runtime.execute_tool_calls(
            [
                {
                    "id": "call-a",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": '{"value":"A","is_error":true}',
                    },
                }
            ],
            on_event=events.append,
        )
    finally:
        runtime.close()

    assert events[-1].phase == "finish"
    assert events[-1].is_error is True
    assert events[-1].message == message


def test_runtime_tool_event_callback_failure_does_not_fail_tool(tmp_path, caplog):
    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )

    def _fail(_event):
        raise RuntimeError("renderer failed")

    runtime = ChatMCPRuntime(str(path))
    try:
        [message] = runtime.execute_tool_calls(
            [
                {
                    "id": "call-a",
                    "function": {
                        "name": "alpha__lookup",
                        "arguments": '{"value":"A"}',
                    },
                }
            ],
            on_event=_fail,
        )
    finally:
        runtime.close()

    assert message["tool_call_id"] == "call-a"
    assert "renderer failed" in caplog.text


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


def test_server_parameters_keep_npx_bootstrap_output_off_protocol():
    quiet = _server_parameters(
        MCPServerConfig(
            name="filesystem",
            command="/opt/homebrew/bin/npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    )
    assert quiet.env["npm_config_loglevel"] == "silent"

    explicit = _server_parameters(
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["server"],
            env={"NPM_CONFIG_LOGLEVEL": "warn"},
        )
    )
    assert explicit.env["NPM_CONFIG_LOGLEVEL"] == "warn"
    assert "npm_config_loglevel" not in explicit.env


def test_optional_component_warning_filter_keeps_actionable_warnings(caplog):
    sdk_logger = logging.getLogger("mcp.client.session_group")
    with (
        caplog.at_level(logging.WARNING),
        _quiet_optional_component_warnings(),
    ):
        # MCP 1.28 uses logging.warning directly; keep the named-logger path
        # covered too so a future SDK cleanup cannot reintroduce the noise.
        logging.warning("Could not fetch prompts: Method not found")
        sdk_logger.warning("Could not fetch resources: Method not found")
        sdk_logger.warning("Could not fetch prompts: permission denied")

    assert [record.getMessage() for record in caplog.records] == [
        "Could not fetch prompts: permission denied"
    ]


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


def test_mcp_server_stderr_to_injects_errlog_and_restores():
    import mcp

    original = mcp.stdio_client
    logfile = io.StringIO()
    with _mcp_server_stderr_to(logfile):
        patched = mcp.stdio_client
        assert patched is not original
        assert isinstance(patched, functools.partial)
        assert patched.func is original
        assert patched.keywords.get("errlog") is logfile
    assert mcp.stdio_client is original


def test_mcp_server_stderr_to_is_noop_without_logfile():
    import mcp

    original = mcp.stdio_client
    with _mcp_server_stderr_to(None):
        assert mcp.stdio_client is original
    assert mcp.stdio_client is original


def test_runtime_exposes_server_log_path(tmp_path):
    import os

    path = _write_config(
        tmp_path,
        {"alpha": {"command": "python3", "args": ["alpha", "lookup"]}},
    )
    runtime = ChatMCPRuntime(str(path))
    try:
        log_path = runtime.server_log_path
        assert log_path is not None
        assert os.path.exists(log_path)
    finally:
        runtime.close()
    # The log survives close so a user can read what the servers printed.
    assert os.path.exists(log_path)
