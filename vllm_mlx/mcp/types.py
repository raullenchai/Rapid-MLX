# SPDX-License-Identifier: Apache-2.0
"""
Type definitions for MCP client support.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Recognized non-server top-level config keys. A config with only these (and
# no server map) is an intentional globals-only config, not a mistyped
# server-key footgun — see select_server_map.
_KNOWN_SETTING_KEYS = frozenset({"default_timeout", "allowed_high_risk_tools"})


def select_server_map(data: dict[str, Any]) -> dict[str, Any]:
    """Return the server-map sub-dict from a raw MCP config.

    Accepts BOTH the ecosystem-standard ``mcpServers`` key (Claude Desktop,
    ``.mcp.json``, VS Code — and our own docs) and the legacy ``servers``
    key, keyed on PRESENCE so a present-but-invalid standard key (e.g.
    ``{"mcpServers": null}``) raises rather than silently falling back.
    ``mcpServers`` wins when both are present.

    Single source of truth for key selection, shared by
    :func:`vllm_mlx.mcp.config.validate_config` and
    :meth:`MCPConfig.from_dict` so the two load paths can never diverge.
    Warns (never raises) on the both-keys and typo'd-key cases so a misconfig
    is visible instead of silently yielding zero servers.
    """
    has_standard = "mcpServers" in data
    has_legacy = "servers" in data
    servers_key = "mcpServers" if has_standard else "servers"
    servers_data = data.get(servers_key, {})
    if not isinstance(servers_data, dict):
        raise ValueError(f"'{servers_key}' must be a dictionary")
    if has_standard and has_legacy:
        logger.warning(
            "MCP config has both 'mcpServers' and 'servers'; using the "
            "standard 'mcpServers' key and ignoring 'servers'."
        )
    elif not has_standard and not has_legacy:
        # Warn only when there's an unrecognized top-level key — a likely
        # mistyped server map (e.g. "mcp_servers", "Servers") that would
        # otherwise silently load nothing. A config with only recognized
        # global settings and no servers is intentional, not a typo.
        unexpected = set(data) - _KNOWN_SETTING_KEYS
        if unexpected:
            logger.warning(
                "MCP config has neither 'mcpServers' nor 'servers' (found "
                "unrecognized key(s): %s) — no MCP servers will be loaded. "
                "The standard key is 'mcpServers'.",
                sorted(unexpected),
            )
    return servers_data


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
class MCPRejectedServer:
    """A server entry that failed to parse or failed security validation.

    Issue #1716: one bad entry used to abort the whole config load, which
    read to the user as "all my connectors vanished". Tolerant loading keeps
    the good entries and records the bad ones here so the reason survives all
    the way to the UI instead of only reaching the server log.
    """

    name: str
    error: str
    #: Transport as DECLARED in the config, not as validated — the entry never
    #: became an ``MCPServerConfig``, so this is the raw string the user wrote.
    #: Reported as-is so the error row doesn't claim a transport the user
    #: didn't choose; unparseable/absent falls back to the stdio default the
    #: loader would itself have applied.
    transport: str = "stdio"


@dataclass
class MCPConfig:
    """Root configuration for MCP client."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    default_timeout: float = 30.0
    # Tools whose names match HIGH_RISK_TOOL_PATTERNS (execute, shell, eval,
    # exec, system, run_command, subprocess) are blocked by default. Add the
    # full namespaced tool name (e.g. "filesystem__execute") here to opt-in.
    allowed_high_risk_tools: list[str] = field(default_factory=list)
    # Entries dropped by a tolerant load. Empty under strict loading, which
    # raises instead.
    rejected: list[MCPRejectedServer] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPConfig":
        """Create config from dictionary.

        Accepts the ecosystem-standard ``mcpServers`` key as well as the
        historical ``servers`` key (``mcpServers`` wins when both present),
        via the shared :func:`select_server_map` used by
        :func:`vllm_mlx.mcp.config.validate_config` — so the two load paths
        can never diverge on key handling.
        """
        servers = {}
        for name, server_data in select_server_map(data).items():
            server_data["name"] = name
            servers[name] = MCPServerConfig(**server_data)

        return cls(
            servers=servers,
            default_timeout=data.get("default_timeout", 30.0),
            allowed_high_risk_tools=data.get("allowed_high_risk_tools", []),
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
