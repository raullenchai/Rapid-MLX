# SPDX-License-Identifier: Apache-2.0
"""MCP tools for the built-in ``rapid-mlx chat`` agent.

The official MCP SDK owns transports and sessions. AnyIO's BlockingPortal
bridges the synchronous REPL to that async SDK without a custom worker queue
or event-loop lifecycle.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import logging
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from anyio.abc import TaskStatus
from anyio.from_thread import start_blocking_portal

from .mcp.config import load_mcp_config
from .mcp.executor import validate_tool_arguments
from .mcp.security import ToolSandbox
from .mcp.tools import mcp_tools_to_openai
from .mcp.types import MCPServerConfig, MCPTool, MCPTransport

_OPENAI_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_SHUTDOWN_SECONDS = 5.0
_OPTIONAL_COMPONENT_WARNINGS = {
    "Could not fetch prompts: Method not found",
    "Could not fetch resources: Method not found",
}


@dataclass(frozen=True)
class ChatToolEvent:
    """One lifecycle update for a tool call owned by ``rapid-mlx chat``."""

    phase: Literal["start", "finish"]
    call_id: str
    name: str
    elapsed_seconds: float | None = None
    is_error: bool | None = None
    message: dict[str, Any] | None = None


class _OptionalComponentWarningFilter(logging.Filter):
    """Hide SDK noise for optional MCP capabilities chat does not consume."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() not in _OPTIONAL_COMPONENT_WARNINGS


@contextmanager
def _quiet_optional_component_warnings():
    """Temporarily filter known ClientSessionGroup capability-probe noise."""

    loggers = (
        logging.getLogger(),
        logging.getLogger("mcp.client.session_group"),
    )
    warning_filter = _OptionalComponentWarningFilter()
    for logger in loggers:
        logger.addFilter(warning_filter)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(warning_filter)


@contextmanager
def _mcp_server_stderr_to(logfile):
    """Route spawned MCP servers' stderr to *logfile* instead of the chat REPL.

    The SDK's ``ClientSessionGroup`` builds the stdio transport internally and
    exposes no way to pass ``errlog`` (it defaults to ``sys.stderr``), so a
    server that logs to stderr — the official reference servers print a startup
    banner, FastMCP-based ones log every request — would scribble over the clean
    tool-activity display. We wrap the ``stdio_client`` the SDK resolves to
    inject ``errlog`` for the duration of connection setup only. A child's
    stderr fd is bound at exec, so it keeps writing to *logfile* for its whole
    life, including per-request logging after setup.

    The installed SDK calls ``mcp.stdio_client(...)`` by attribute, but to stay
    robust if a future version binds ``stdio_client`` locally in the
    ``session_group`` module instead, we patch every reachable reference. If the
    SDK stops routing through any of them this degrades to the previous (noisy)
    behaviour rather than breaking.
    """

    if logfile is None:
        yield
        return
    import mcp
    import mcp.client.session_group as _session_group

    targets = [(mcp, "stdio_client")]
    if hasattr(_session_group, "stdio_client"):
        targets.append((_session_group, "stdio_client"))
    saved = []
    try:
        for module, name in targets:
            original = getattr(module, name)
            saved.append((module, name, original))
            setattr(module, name, functools.partial(original, errlog=logfile))
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


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

        # Per-session log sink for MCP servers' stderr, so their banners and
        # request logging land in a file instead of the interactive chat.
        # Best-effort: if it can't be opened we simply keep the old behaviour.
        self._server_log = None
        self.server_log_path: str | None = None
        try:
            handle = tempfile.NamedTemporaryFile(
                prefix="rapid-mlx-mcp-", suffix=".log", mode="a", delete=False
            )
            self._server_log = handle
            self.server_log_path = handle.name
        except OSError:
            pass

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
        on_event: Callable[[ChatToolEvent], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute independent server lanes and return ordered tool messages."""

        done = threading.Event()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("MCP runtime is closed")
            future = self._portal.start_task_soon(
                self._execute_with_done,
                tool_calls,
                done,
                on_event,
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
                if self._server_log is not None:
                    try:
                        self._server_log.close()
                    except OSError:
                        pass
                    self._server_log = None
                self._closed = True
            if cancellation_error is not None:
                raise cancellation_error

    async def _lifecycle(self, *, task_status: TaskStatus[None]) -> None:
        from mcp.client.session_group import ClientSessionGroup

        self._stop_event = asyncio.Event()
        async with AsyncExitStack() as stack:
            with _mcp_server_stderr_to(self._server_log):
                for server in self._enabled_servers:
                    group = ClientSessionGroup(
                        component_name_hook=lambda name, _info, label=server.name: (
                            f"{label}__{name}"
                        )
                    )
                    try:
                        await group.__aenter__()
                        try:
                            with _quiet_optional_component_warnings():
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
                                f"MCP tool name collision: {full_name!r} is exposed "
                                f"by both {previous.server_name!r} and "
                                f"{server.name!r}"
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
        on_event: Callable[[ChatToolEvent], None] | None,
    ) -> list[dict[str, Any]]:
        try:
            return await self._execute(tool_calls, on_event)
        finally:
            done.set()

    async def _execute(
        self,
        tool_calls: list[dict[str, Any]],
        on_event: Callable[[ChatToolEvent], None] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any] | None] = [None] * len(tool_calls)
        lanes: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for position, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            tool = self._tools_by_name.get(str(function.get("name") or ""))
            lane = (
                f"server:{tool.server_name}"
                if tool is not None
                else f"invalid:{position}"
            )
            lanes.setdefault(lane, []).append((position, tool_call))

        async def _run_lane(
            calls: list[tuple[int, dict[str, Any]]],
        ) -> None:
            for position, tool_call in calls:
                messages[position] = await self._execute_one(
                    position,
                    tool_call,
                    on_event,
                )

        tasks = [asyncio.create_task(_run_lane(calls)) for calls in lanes.values()]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [message for message in messages if message is not None]

    async def _execute_one(
        self,
        position: int,
        tool_call: dict[str, Any],
        on_event: Callable[[ChatToolEvent], None] | None,
    ) -> dict[str, Any]:
        call_id = str(tool_call.get("id") or f"call_{position}")
        function = tool_call.get("function") or {}
        full_name = str(function.get("name") or "")
        tool = self._tools_by_name.get(full_name)
        arguments: dict[str, Any] = {}
        started = time.monotonic()
        _emit_tool_event(
            on_event,
            ChatToolEvent("start", call_id, full_name),
        )
        content: str | None = None
        message: dict[str, Any] | None = None
        is_error = True

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
        finally:
            if content is not None:
                message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            _emit_tool_event(
                on_event,
                ChatToolEvent(
                    "finish",
                    call_id,
                    full_name,
                    elapsed_seconds=time.monotonic() - started,
                    is_error=is_error,
                    message=message,
                ),
            )

        assert message is not None
        return message

    def _server_timeout(self, server_name: str) -> float:
        return self._config.servers[server_name].timeout


def _emit_tool_event(
    on_event: Callable[[ChatToolEvent], None] | None,
    event: ChatToolEvent,
) -> None:
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        logging.getLogger(__name__).exception("MCP tool event callback failed")


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
        if Path(server.command or "").name == "npx" and not any(
            key.lower() == "npm_config_loglevel" for key in env
        ):
            # npm may print cold-install progress to stdout, which is the
            # JSON-RPC transport for stdio MCP servers.
            env["npm_config_loglevel"] = "silent"
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
