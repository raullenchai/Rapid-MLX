# SPDX-License-Identifier: Apache-2.0
"""
Type definitions for MCP client support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Set


class MCPTransport(str, Enum):
    """Supported MCP transport types."""

    STDIO = "stdio"
    SSE = "sse"


class MCPServerState(str, Enum):
    """MCP server connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: MCPTransport = MCPTransport.STDIO

    # For stdio transport
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None

    # For SSE transport
    url: str | None = None

    # Common options
    enabled: bool = True
    timeout: float = 30.0

    # Security options
    skip_security_validation: bool = False  # WARNING: Only for development!

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.transport, str):
            self.transport = MCPTransport(self.transport)

        if self.transport == MCPTransport.STDIO:
            if not self.command:
                raise ValueError(
                    f"MCP server '{self.name}': stdio transport requires 'command'"
                )
        elif self.transport == MCPTransport.SSE:
            if not self.url:
                raise ValueError(
                    f"MCP server '{self.name}': sse transport requires 'url'"
                )

        # Security validation
        self._validate_security()

    def _validate_security(self) -> None:
        """Validate security of the configuration."""
        from .security import MCPSecurityError, validate_mcp_server_config

        if self.skip_security_validation:
            import logging

            logging.getLogger(__name__).warning(
                f"MCP server '{self.name}': Security validation SKIPPED. "
                f"This is dangerous and should only be used in development!"
            )
            return

        try:
            validate_mcp_server_config(
                server_name=self.name,
                command=self.command,
                args=self.args,
                env=self.env,
                url=self.url,
            )
        except MCPSecurityError as e:
            raise ValueError(str(e)) from e


@dataclass
class MCPConfig:
    """Root configuration for MCP client."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    max_tool_calls: int = 10
    default_timeout: float = 30.0
    allowed_high_risk_tools: Set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPConfig":
        """Create config from dictionary."""
        servers = {}
        for name, server_data in data.get("servers", {}).items():
            server_data["name"] = name
            servers[name] = MCPServerConfig(**server_data)

        return cls(
            servers=servers,
            max_tool_calls=data.get("max_tool_calls", 10),
            default_timeout=data.get("default_timeout", 30.0),
            allowed_high_risk_tools=set(data.get("allowed_high_risk_tools", [])),
        )


@dataclass
class MCPTool:
    """Normalized tool representation from MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get namespaced tool name (server__tool)."""
        return f"{self.server_name}__{self.name}"

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.full_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPToolResult:
    """Result from a tool execution."""

    tool_name: str
    content: Any
    is_error: bool = False
    error_message: str | None = None

    def to_message(self, tool_call_id: str) -> dict[str, Any]:
        """Convert to OpenAI tool result message format."""
        if self.is_error:
            content = f"Error: {self.error_message}"
        elif isinstance(self.content, str):
            content = self.content
        else:
            import json

            content = json.dumps(self.content)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }


@dataclass
class MCPServerStatus:
    """Status of an MCP server connection."""

    name: str
    state: MCPServerState
    transport: MCPTransport
    tools_count: int = 0
    error: str | None = None
    last_connected: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "state": self.state.value,
            "transport": self.transport.value,
            "tools_count": self.tools_count,
            "error": self.error,
            "last_connected": self.last_connected,
        }
