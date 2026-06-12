# SPDX-License-Identifier: Apache-2.0
"""
MCP (Model Context Protocol) client support for rapid-mlx.

This module provides integration with MCP servers, allowing the rapid-mlx server
to discover and execute tools from external MCP servers.

Example usage:
    from vllm_mlx.mcp import MCPClientManager, load_mcp_config

    config = load_mcp_config("./mcp.json")
    manager = MCPClientManager(config)
    await manager.start()

    # Get all available tools in OpenAI format
    tools = manager.get_all_tools()

    # Execute a tool call
    result = await manager.execute_tool("filesystem__read_file", {"path": "/tmp/test.txt"})
"""

from .client import MCPClient
from .config import load_mcp_config, validate_config
from .executor import ToolArgumentValidationError, ToolExecutor, validate_tool_arguments
from .manager import MCPClientManager
from .security import (
    ALLOWED_COMMANDS,
    DANGEROUS_TOOL_ARG_PATTERNS,
    HIGH_RISK_TOOL_PATTERNS,
    MCPCommandValidator,
    MCPSecurityError,
    ToolExecutionAudit,
    # Sandboxing
    ToolSandbox,
    get_sandbox,
    get_validator,
    set_sandbox,
    set_validator,
    validate_mcp_server_config,
)
from .tools import format_tool_result, mcp_tool_to_openai, openai_call_to_mcp
from .types import (
    MCPConfig,
    MCPServerConfig,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
)

__all__ = [
    # Types
    "MCPServerConfig",
    "MCPConfig",
    "MCPTool",
    "MCPToolResult",
    "MCPServerStatus",
    # Config
    "load_mcp_config",
    "validate_config",
    # Client
    "MCPClient",
    "MCPClientManager",
    # Tools
    "mcp_tool_to_openai",
    "openai_call_to_mcp",
    "format_tool_result",
    # Executor
    "ToolExecutor",
    "ToolArgumentValidationError",
    "validate_tool_arguments",
    # Security
    "MCPSecurityError",
    "MCPCommandValidator",
    "ALLOWED_COMMANDS",
    "validate_mcp_server_config",
    "get_validator",
    "set_validator",
    # Sandboxing
    "ToolSandbox",
    "ToolExecutionAudit",
    "get_sandbox",
    "set_sandbox",
    "DANGEROUS_TOOL_ARG_PATTERNS",
    "HIGH_RISK_TOOL_PATTERNS",
]
