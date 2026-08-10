# SPDX-License-Identifier: Apache-2.0
"""
MCP client for connecting to individual MCP servers.
"""

import asyncio
import logging
import tempfile
import time
from typing import Any

from .types import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)

logger = logging.getLogger(__name__)


def _sdk_attr(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read an mcp SDK model attribute across the SDK's camelCase→snake_case rename.

    mcp 1.x exposed model fields in camelCase (``protocolVersion``,
    ``inputSchema``, ``isError``); mcp 2.0 renamed the Python attributes to
    snake_case (``protocol_version``, ``input_schema``, ``is_error``) and kept
    camelCase only as a serialization alias. Reading the wrong name raises
    ``AttributeError`` — which previously aborted the initialize handshake and
    left every configured server with 0 tools (rapid-desktop#604), because the
    dev/CI env pinned mcp 1.x while a fresh sidecar build resolved mcp 2.0.
    Try snake_case first (mcp>=2.0), then camelCase (mcp<2.0), then ``default``.
    """
    sentinel = object()
    value = getattr(obj, snake, sentinel)
    if value is sentinel:
        value = getattr(obj, camel, sentinel)
    return default if value is sentinel else value


class MCPClient:
    """
    Client for connecting to a single MCP server.

    Supports both stdio and SSE transports.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._tools: list[MCPTool] = []
        self._state = MCPServerState.DISCONNECTED
        self._error: str | None = None
        self._last_connected: float | None = None
        self._lock = asyncio.Lock()
        # Issue #1716: the stdio child's stderr. The MCP SDK's ``stdio_client``
        # defaults ``errlog`` to our own stderr, so a server that dies on
        # startup wrote its traceback somewhere the user never sees and the
        # row reported only "Connection closed" — true, and useless. Capturing
        # it lets the failure name its own cause.
        #
        # A real temp FILE, not ``io.StringIO``: the SDK hands ``errlog``
        # to ``anyio.open_process(stderr=...)``, which needs an actual file
        # descriptor. A StringIO gets as far as ``fileno()`` and raises —
        # turning every connection error into the word "fileno".
        self._stderr_file: tempfile.TemporaryFile | None = None  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def state(self) -> MCPServerState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == MCPServerState.CONNECTED

    @property
    def tools(self) -> list[MCPTool]:
        """Get discovered tools."""
        return self._tools

    def get_status(self) -> MCPServerStatus:
        """Get server status."""
        return MCPServerStatus(
            name=self.name,
            state=self._state,
            transport=self.config.transport,
            tools_count=len(self._tools),
            error=self._error,
            last_connected=self._last_connected,
        )

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        async with self._lock:
            if self._state == MCPServerState.CONNECTED:
                return True

            if not self.config.enabled:
                logger.info(f"MCP server '{self.name}' is disabled")
                return False

            self._state = MCPServerState.CONNECTING
            self._error = None

            try:
                if self.config.transport == MCPTransport.STDIO:
                    await self._connect_stdio()
                elif self.config.transport == MCPTransport.SSE:
                    await self._connect_sse()
                else:
                    raise ValueError(f"Unknown transport: {self.config.transport}")

                # Initialize session
                await self._initialize_session()

                # Discover tools
                await self._discover_tools()

                self._state = MCPServerState.CONNECTED
                self._last_connected = time.time()
                logger.info(
                    f"Connected to MCP server '{self.name}' "
                    f"({len(self._tools)} tools available)"
                )
                return True

            except Exception as e:
                self._state = MCPServerState.ERROR
                # ``_describe_failure`` reads the captured stderr, so close it
                # only after. A failed connect never reaches ``disconnect`` on
                # its own (the manager just sees ``False``), so without this the
                # temp file's descriptor leaks on every startup failure.
                self._error = self._describe_failure(e)
                self._close_stderr_file()
                logger.error(
                    f"Failed to connect to MCP server '{self.name}': {self._error}"
                )
                return False

    #: How much of the child's stderr to append to a connection error. Enough
    #: for a Python traceback's final line, short enough for a settings row.
    _STDERR_TAIL_CHARS = 400

    def _describe_failure(self, exc: Exception) -> str:
        """Build the error string the UI shows for a failed connection.

        Issue #1716: the exception alone is frequently content-free. A stdio
        server that dies during import raises ``ClosedResourceError`` here,
        which renders as "Connection closed" — accurate, and useless to
        someone trying to fix it. The reason is in the child's stderr, which
        ``_connect_stdio`` now captures. Appending its tail turns the row from
        "it didn't work" into the actual ImportError / missing-module /
        bad-argument line the user needs.
        """
        base = str(exc) or type(exc).__name__
        tail = ""
        if self._stderr_file is not None:
            try:
                self._stderr_file.seek(0)
                tail = self._stderr_file.read().strip()
            except Exception:  # pragma: no cover - defensive
                tail = ""
        if not tail:
            return base
        # Last non-empty line: for a traceback that's the exception line, which
        # is the one that names the cause.
        last = tail.splitlines()[-1].strip()
        if len(last) > self._STDERR_TAIL_CHARS:
            last = last[: self._STDERR_TAIL_CHARS] + "…"
        # Don't repeat ourselves when the exception already said it.
        return last if last in base else f"{base} — {last}"

    async def _connect_stdio(self):
        """Connect via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Security: Log the command being executed for audit trail
        logger.info(
            f"MCP SECURITY AUDIT: Server '{self.name}' executing command: "
            f"{self.config.command} {' '.join(self.config.args or [])}"
        )

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )

        # Create stdio client context. ``errlog`` captures the child's stderr
        # instead of letting it escape to ours — see ``_stderr_file``. Close any
        # prior handle first: a reconnect on the same client would otherwise
        # leak the previous temp file's descriptor.
        self._close_stderr_file()
        self._stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._stdio_client = stdio_client(server_params, errlog=self._stderr_file)
        self._read, self._write = await self._stdio_client.__aenter__()

        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _connect_sse(self):
        """Connect via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Create SSE client context
        self._sse_client = sse_client(self.config.url)
        self._read, self._write = await self._sse_client.__aenter__()

        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()

    async def _initialize_session(self):
        """Initialize the MCP session."""
        if self._session is None:
            raise RuntimeError("Session not created")

        # Initialize with capabilities
        result = await self._session.initialize()
        server_info = _sdk_attr(result, "server_info", "serverInfo")
        logger.debug(
            f"MCP server '{self.name}' initialized: "
            f"protocol={_sdk_attr(result, 'protocol_version', 'protocolVersion')}, "
            f"server={server_info.name if server_info else 'unknown'}"
        )

    async def _discover_tools(self):
        """Discover available tools from the server."""
        if self._session is None:
            raise RuntimeError("Session not initialized")

        try:
            result = await self._session.list_tools()
            self._tools = []

            for tool in result.tools:
                mcp_tool = MCPTool(
                    server_name=self.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=_sdk_attr(tool, "input_schema", "inputSchema", {}),
                )
                self._tools.append(mcp_tool)
                logger.debug(f"Discovered tool: {mcp_tool.full_name}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from '{self.name}': {e}")
            self._tools = []

    async def disconnect(self):
        """Disconnect from the MCP server."""
        async with self._lock:
            if self._state == MCPServerState.DISCONNECTED:
                return

            try:
                if self._session:
                    await self._session.__aexit__(None, None, None)
                    self._session = None

                if hasattr(self, "_stdio_client") and self._stdio_client:
                    await self._stdio_client.__aexit__(None, None, None)
                    self._stdio_client = None

                if hasattr(self, "_sse_client") and self._sse_client:
                    await self._sse_client.__aexit__(None, None, None)
                    self._sse_client = None

            except Exception as e:
                logger.warning(f"Error disconnecting from '{self.name}': {e}")

            finally:
                # Close the captured-stderr temp file, or each reload
                # (disconnect + reconnect every client) leaks one fd until GC.
                self._close_stderr_file()
                self._state = MCPServerState.DISCONNECTED
                self._tools = []
                logger.info(f"Disconnected from MCP server '{self.name}'")

    def _close_stderr_file(self) -> None:
        """Close and drop the captured-stderr temp file if one is open."""
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._stderr_file = None

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        if not self.is_connected:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Not connected to server '{self.name}'",
            )

        if self._session is None:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message="Session not initialized",
            )

        try:
            # Call with timeout
            timeout = timeout or self.config.timeout

            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Extract content from result
            content = self._extract_content(result)

            return MCPToolResult(
                tool_name=tool_name,
                content=content,
                is_error=bool(_sdk_attr(result, "is_error", "isError", False)),
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Tool call timed out after {timeout}s",
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(e),
            )

    def _extract_content(self, result) -> Any:
        """Extract content from MCP tool result."""
        if not hasattr(result, "content") or not result.content:
            return None

        # Handle list of content items
        contents = []
        for item in result.content:
            if hasattr(item, "text"):
                contents.append(item.text)
            elif hasattr(item, "data"):
                contents.append(item.data)
            else:
                contents.append(str(item))

        # Return single item or list
        if len(contents) == 1:
            return contents[0]
        return contents

    async def refresh_tools(self):
        """Refresh the list of available tools."""
        if not self.is_connected:
            return

        await self._discover_tools()
