# SPDX-License-Identifier: Apache-2.0
"""MCP tools for the built-in ``rapid-mlx chat`` agent.

The official MCP SDK owns transports and sessions. AnyIO's BlockingPortal
bridges the synchronous REPL to that async SDK without a custom worker queue
or event-loop lifecycle.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import sys
import threading
from contextlib import AsyncExitStack
from typing import Any

from anyio.abc import TaskStatus
from anyio.from_thread import start_blocking_portal

from .mcp.config import load_mcp_config
from .mcp.executor import validate_tool_arguments
from .mcp.security import ToolSandbox
from .mcp.tools import mcp_tools_to_openai
from .mcp.types import MCPServerConfig, MCPTool, MCPTransport

_OPENAI_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_SHUTDOWN_SECONDS = 5.0


class ChatMCPRuntime:
    """Synchronous facade over SDK sessions running in an AnyIO portal."""

    def __init__(self, config_path: str):
        self._config = load_mcp_config(config_path)
        self._enabled_servers = [
            server for server in self._config.servers.values() if server.enabled
        ]
        if not self._enabled_servers:
            raise ValueError("MCP config has no enabled servers")

        self._sandbox = ToolSandbox(
            allowed_high_risk_tools=set(self._config.allowed_high_risk_tools)
        )
        self._groups: dict[str, Any] = {}
        self._tools_by_name: dict[str, MCPTool] = {}
        self._connection_errors: dict[str, str] = {}
        self._closed = False
        self._state_lock = threading.RLock()
        self._active_future = None
        self._active_done: threading.Event | None = None
        self._stop_event: asyncio.Event | None = None
        self.tools: list[dict[str, Any]] = []

        self._portal_context = start_blocking_portal(
            backend="asyncio",
            name="rapid-mlx-chat-mcp",
        )
        self._portal = self._portal_context.__enter__()
        try:
            self._lifecycle_future, _ = self._portal.start_task(self._lifecycle)
        except BaseException:
            exc_info = sys.exc_info()
            self._portal_context.__exit__(*exc_info)
            self._closed = True
            raise

        if not self.tools:
            details = "; ".join(
                f"{name}: {error}"
                for name, error in sorted(self._connection_errors.items())
            )
            self.close()
            suffix = f" ({details})" if details else ""
            raise RuntimeError(f"No MCP tools available{suffix}")

    @property
    def connection_errors(self) -> dict[str, str]:
        """Servers that failed while other configured servers stayed usable."""

        return dict(self._connection_errors)

    @property
    def server_count(self) -> int:
        """Number of connected MCP servers."""

        return len(self._groups)

    def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Execute calls sequentially and return OpenAI ``tool`` messages."""

        done = threading.Event()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("MCP runtime is closed")
            future = self._portal.start_task_soon(
                self._execute_with_done,
                tool_calls,
                done,
            )
            self._active_future = future
            self._active_done = done
        try:
            return future.result()
        except BaseException:
            future.cancel()
            cancel_timeout = max(server.timeout for server in self._enabled_servers) + 1
            if not done.wait(timeout=cancel_timeout):
                raise RuntimeError(
                    f"MCP tool cancellation timed out after {cancel_timeout:g} seconds"
                )
            raise
        finally:
            with self._state_lock:
                if self._active_future is future:
                    self._active_future = None
                    self._active_done = None

    def close(self) -> None:
        """Close SDK sessions and stop the AnyIO portal."""

        with self._state_lock:
            if self._closed:
                return
            cancellation_error = None
            if self._active_future is not None:
                self._active_future.cancel()
                cancel_timeout = (
                    max(server.timeout for server in self._enabled_servers) + 1
                )
                if self._active_done is not None and not self._active_done.wait(
                    timeout=cancel_timeout
                ):
                    cancellation_error = RuntimeError(
                        "MCP tool cancellation timed out after "
                        f"{cancel_timeout:g} seconds"
                    )
            try:
                self._portal.call(self._request_stop)
                shutdown_timeout = (
                    sum(
                        min(server.timeout, _MAX_SHUTDOWN_SECONDS)
                        for server in self._enabled_servers
                    )
                    + 1
                )
                try:
                    self._lifecycle_future.result(timeout=shutdown_timeout)
                except concurrent.futures.TimeoutError as exc:
                    self._lifecycle_future.cancel()
                    raise RuntimeError(
                        "MCP lifecycle shutdown timed out after "
                        f"{shutdown_timeout:g} seconds"
                    ) from exc
            finally:
                self._portal_context.__exit__(None, None, None)
                self._closed = True
            if cancellation_error is not None:
                raise cancellation_error

    async def _lifecycle(self, *, task_status: TaskStatus[None]) -> None:
        from mcp.client.session_group import ClientSessionGroup

        self._stop_event = asyncio.Event()
        async with AsyncExitStack() as stack:
            for server in self._enabled_servers:
                group = ClientSessionGroup(
                    component_name_hook=lambda name, _info, label=server.name: (
                        f"{label}__{name}"
                    )
                )
                try:
                    await group.__aenter__()
                    try:
                        await asyncio.wait_for(
                            group.connect_to_server(_server_parameters(server)),
                            timeout=server.timeout,
                        )
                    except BaseException:
                        await _close_group(
                            group,
                            server.name,
                            server.timeout,
                            sys.exc_info(),
                        )
                        raise
                except (TimeoutError, asyncio.TimeoutError):
                    self._connection_errors[server.name] = (
                        f"connection timed out after {server.timeout:g} seconds"
                    )
                    continue
                except Exception as exc:
                    self._connection_errors[server.name] = str(exc)
                    continue

                stack.push_async_callback(
                    _close_group,
                    group,
                    server.name,
                    server.timeout,
                    (None, None, None),
                )
                self._groups[server.name] = group
                for full_name, sdk_tool in group.tools.items():
                    if not _OPENAI_FUNCTION_NAME_RE.fullmatch(full_name):
                        raise RuntimeError(
                            f"MCP tool name {full_name!r} is not a valid OpenAI "
                            "function name (1-64 letters, digits, '_' or '-')"
                        )
                    if full_name in self._tools_by_name:
                        previous = self._tools_by_name[full_name]
                        raise RuntimeError(
                            f"MCP tool name collision: {full_name!r} is exposed by "
                            f"both {previous.server_name!r} and {server.name!r}"
                        )
                    self._tools_by_name[full_name] = MCPTool(
                        server_name=server.name,
                        name=sdk_tool.name,
                        description=sdk_tool.description or "",
                        input_schema=sdk_tool.inputSchema or {},
                    )

            self.tools = mcp_tools_to_openai(list(self._tools_by_name.values()))
            task_status.started()
            await self._stop_event.wait()

    def _request_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    async def _execute_with_done(
        self,
        tool_calls: list[dict[str, Any]],
        done: threading.Event,
    ) -> list[dict[str, Any]]:
        try:
            return await self._execute(tool_calls)
        finally:
            done.set()

    async def _execute(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for position, tool_call in enumerate(tool_calls):
            call_id = str(tool_call.get("id") or f"call_{position}")
            function = tool_call.get("function") or {}
            full_name = str(function.get("name") or "")
            tool = self._tools_by_name.get(full_name)
            arguments: dict[str, Any] = {}

            try:
                raw_arguments = function.get("arguments", "{}")
                parsed_arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
                if not isinstance(parsed_arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                arguments = parsed_arguments
                if tool is None:
                    raise ValueError(f"Unknown MCP tool: {full_name or '<empty>'}")

                validate_tool_arguments(tool, arguments, strict=True)
                self._sandbox.validate_tool_execution(
                    tool.name,
                    tool.server_name,
                    arguments,
                )

                group = self._groups[tool.server_name]
                timeout = self._server_timeout(tool.server_name)
                try:
                    result = await asyncio.wait_for(
                        group.call_tool(full_name, arguments),
                        timeout=timeout,
                    )
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    raise RuntimeError(
                        f"MCP tool {full_name!r} timed out after {timeout:g} seconds"
                    ) from exc

                content = json.dumps(
                    result.model_dump(mode="json", by_alias=True, exclude_none=True),
                    ensure_ascii=False,
                )
                is_error = bool(getattr(result, "isError", False))
                self._sandbox.record_execution(
                    tool.name,
                    tool.server_name,
                    arguments,
                    success=not is_error,
                    error_message=content if is_error else None,
                )
            except Exception as exc:
                content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                if tool is not None:
                    self._sandbox.record_execution(
                        tool.name,
                        tool.server_name,
                        arguments,
                        success=False,
                        error_message=str(exc),
                    )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        return messages

    def _server_timeout(self, server_name: str) -> float:
        return self._config.servers[server_name].timeout


async def _close_group(
    group: Any,
    server_name: str,
    timeout: float,
    exc_info: tuple,
) -> None:
    shutdown_timeout = min(timeout, _MAX_SHUTDOWN_SECONDS)
    try:
        await asyncio.wait_for(
            group.__aexit__(*exc_info),
            timeout=shutdown_timeout,
        )
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise RuntimeError(
            f"MCP server {server_name!r} shutdown timed out after "
            f"{shutdown_timeout:g} seconds"
        ) from exc


def _server_parameters(server: MCPServerConfig):
    """Translate Rapid's existing MCP config into official SDK parameters."""

    from mcp.client.session_group import SseServerParameters
    from mcp.client.stdio import StdioServerParameters, get_default_environment

    if server.transport == MCPTransport.STDIO:
        env = get_default_environment()
        env.update(server.env or {})
        return StdioServerParameters(
            command=server.command or "",
            args=server.args or [],
            env=env,
        )
    if server.transport == MCPTransport.SSE:
        return SseServerParameters(
            url=server.url or "",
            timeout=server.timeout,
        )

    raise ValueError(f"Unsupported MCP transport: {server.transport}")
