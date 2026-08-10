# SPDX-License-Identifier: Apache-2.0
"""
MCP Client Manager for handling multiple MCP server connections.
"""

import asyncio
import logging
from typing import Any

from .client import MCPClient
from .tools import mcp_tools_to_openai, merge_tools, openai_call_to_mcp
from .types import (
    MCPConfig,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
)

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Manages multiple MCP server connections.

    Provides a unified interface for:
    - Connecting to multiple MCP servers
    - Discovering and aggregating tools
    - Executing tool calls
    - Managing connection lifecycle
    """

    def __init__(self, config: MCPConfig):
        """
        Initialize MCP Client Manager.

        Args:
            config: MCP configuration with server definitions
        """
        self.config = config
        self._clients: dict[str, MCPClient] = {}
        self._started = False
        self._lock = asyncio.Lock()

        # Create clients for each server
        for name, server_config in config.servers.items():
            self._clients[name] = MCPClient(server_config)

    @property
    def is_started(self) -> bool:
        """Check if manager has been started."""
        return self._started

    async def start(self):
        """
        Start the manager and connect to all enabled servers.

        Connections are made in parallel for faster startup.
        """
        async with self._lock:
            if self._started:
                return

            logger.info(
                f"Starting MCP client manager with {len(self._clients)} servers"
            )

            # Connect to all servers in parallel
            tasks = [
                client.connect()
                for client in self._clients.values()
                if client.config.enabled
            ]

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log results
                for client, result in zip(
                    [c for c in self._clients.values() if c.config.enabled],
                    results,
                ):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to connect to '{client.name}': {result}")
                    elif result:
                        logger.info(f"Connected to '{client.name}'")

            self._started = True

            # Log summary
            connected = sum(1 for c in self._clients.values() if c.is_connected)
            total_tools = sum(len(c.tools) for c in self._clients.values())
            logger.info(
                f"MCP manager started: {connected}/{len(self._clients)} servers, "
                f"{total_tools} tools available"
            )

    async def stop(self):
        """Stop the manager and disconnect from all servers."""
        async with self._lock:
            if not self._started:
                return

            logger.info("Stopping MCP client manager")

            # Disconnect from all servers in parallel
            tasks = [client.disconnect() for client in self._clients.values()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self._started = False
            logger.info("MCP client manager stopped")

    def get_all_tools(self) -> list[MCPTool]:
        """
        Get all tools from all connected servers.

        Returns:
            List of MCPTool instances
        """
        tools = []
        for client in self._clients.values():
            if client.is_connected:
                tools.extend(client.tools)
        return tools

    def get_all_tools_openai(self) -> list[dict[str, Any]]:
        """
        Get all tools in OpenAI function calling format.

        Returns:
            List of OpenAI-compatible tool definitions
        """
        return mcp_tools_to_openai(self.get_all_tools())

    def get_merged_tools(
        self,
        user_tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get MCP tools merged with user-provided tools.

        User tools take precedence on name conflicts.

        Args:
            user_tools: Optional user-provided tools in OpenAI format

        Returns:
            Combined list of tools in OpenAI format
        """
        return merge_tools(self.get_all_tools(), user_tools)

    def get_server_status(self) -> list[MCPServerStatus]:
        """
        Get status of all servers.

        Returns:
            List of MCPServerStatus for each server
        """
        return [client.get_status() for client in self._clients.values()]

    def get_client(self, server_name: str) -> MCPClient | None:
        """
        Get client for a specific server.

        Args:
            server_name: Name of the server

        Returns:
            MCPClient instance or None if not found
        """
        return self._clients.get(server_name)

    def resolve_tool_target(self, full_name: str) -> tuple[str | None, str]:
        """Resolve ``(server_name, bare_tool_name)`` for a namespaced name.

        Returns ``(None, full_name)`` when no connected server owns the tool.
        Public so the HTTP execute route can gate a call through the sandbox
        against the SAME (server, tool) split :meth:`execute_tool` dispatches
        on — otherwise the route would validate one name and run another.
        """
        server_name, tool_name, _ = openai_call_to_mcp(
            {"function": {"name": full_name, "arguments": "{}"}}
        )
        # If no server prefix, try to find the tool by bare name.
        if not server_name:
            server_name = self._find_tool_server(full_name)
            tool_name = full_name
        return server_name, tool_name

    async def execute_tool(
        self,
        full_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> MCPToolResult:
        """
        Execute a tool by its full name (server__tool).

        Args:
            full_name: Full tool name with server prefix
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        server_name, tool_name = self.resolve_tool_target(full_name)

        if not server_name:
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Tool '{full_name}' not found in any connected server",
            )

        # Get client
        client = self._clients.get(server_name)
        if not client:
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Server '{server_name}' not found",
            )

        if not client.is_connected:
            return MCPToolResult(
                tool_name=full_name,
                content=None,
                is_error=True,
                error_message=f"Server '{server_name}' is not connected",
            )

        # Execute tool
        return await client.call_tool(
            tool_name,
            arguments,
            timeout=timeout or self.config.default_timeout,
        )

    async def execute_tool_call(
        self,
        tool_call: dict[str, Any],
        timeout: float | None = None,
    ) -> MCPToolResult:
        """
        Execute a tool call from OpenAI format.

        Args:
            tool_call: OpenAI tool call object
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        server_name, tool_name, arguments = openai_call_to_mcp(tool_call)

        if server_name:
            full_name = f"{server_name}__{tool_name}"
        else:
            full_name = tool_name

        return await self.execute_tool(full_name, arguments, timeout)

    def _find_tool_server(self, tool_name: str) -> str | None:
        """
        Find which server has a tool by name.

        Args:
            tool_name: Tool name (without server prefix)

        Returns:
            Server name or None if not found
        """
        for client in self._clients.values():
            if client.is_connected:
                for tool in client.tools:
                    if tool.name == tool_name:
                        return client.name
        return None

    async def refresh_tools(self):
        """Refresh tools from all connected servers."""
        tasks = [
            client.refresh_tools()
            for client in self._clients.values()
            if client.is_connected
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def reconnect(self, server_name: str | None = None):
        """
        Reconnect to server(s).

        Args:
            server_name: Specific server to reconnect, or None for all
        """
        if server_name:
            client = self._clients.get(server_name)
            if client:
                await client.disconnect()
                await client.connect()
        else:
            # Reconnect all
            for client in self._clients.values():
                await client.disconnect()
                await client.connect()
