# SPDX-License-Identifier: Apache-2.0
"""
MCP security module for command validation and sandboxing.

This module provides security controls to prevent command injection
and other attacks via MCP server configurations.
"""

import logging
import os
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# Whitelist of allowed MCP server commands
# These are well-known, trusted MCP server executables
ALLOWED_COMMANDS: set[str] = {
    # Node.js package runners (for official MCP servers)
    "npx",
    "npm",
    "node",
    # Python package runners
    "uvx",
    "uv",
    "python",
    "python3",
    "pip",
    "pipx",
    # Official MCP servers (when installed globally)
    "mcp-server-filesystem",
    "mcp-server-sqlite",
    "mcp-server-postgres",
    "mcp-server-github",
    "mcp-server-slack",
    "mcp-server-memory",
    "mcp-server-puppeteer",
    "mcp-server-brave-search",
    "mcp-server-google-maps",
    "mcp-server-fetch",
    # Docker (for containerized MCP servers)
    "docker",
}

# Patterns that indicate dangerous commands
DANGEROUS_PATTERNS: list[re.Pattern] = [
    re.compile(r";\s*"),  # Command chaining with ;
    re.compile(r"\|\s*"),  # Piping
    re.compile(r"&&\s*"),  # Command chaining with &&
    re.compile(r"\|\|\s*"),  # Command chaining with ||
    re.compile(r"`"),  # Backtick command substitution
    re.compile(r"\$\("),  # $() command substitution
    re.compile(r">\s*"),  # Output redirection
    re.compile(r"<\s*"),  # Input redirection
    re.compile(r"\.\./"),  # Path traversal
    re.compile(r"~"),  # Home directory expansion (can be abused)
]

# Dangerous argument patterns
DANGEROUS_ARG_PATTERNS: list[re.Pattern] = [
    re.compile(r";\s*"),
    re.compile(r"\|\s*"),
    re.compile(r"&&\s*"),
    re.compile(r"\|\|\s*"),
    re.compile(r"`"),
    re.compile(r"\$\("),
    re.compile(r"\$\{"),
    re.compile(r">\s*/"),  # Redirect to absolute path
    re.compile(r"<\s*/"),  # Read from absolute path
]


class MCPSecurityError(Exception):
    """Raised when MCP security validation fails."""

    pass


class MCPCommandValidator:
    """
    Validates MCP server commands for security.

    This class provides methods to validate commands and arguments
    before they are executed, preventing command injection attacks.
    """

    def __init__(
        self,
        allowed_commands: set[str] | None = None,
        allow_unsafe: bool = False,
        custom_whitelist: set[str] | None = None,
        check_path_exists: bool = True,
    ):
        """
        Initialize the command validator.

        Args:
            allowed_commands: Set of allowed command names. If None, uses default whitelist.
            allow_unsafe: If True, allows any command (for development only).
                         WARNING: This disables security checks!
            custom_whitelist: Additional commands to allow beyond the default whitelist.
            check_path_exists: If True, verify command exists in PATH. Set to False for testing.
        """
        self.allow_unsafe = allow_unsafe
        self.allowed_commands = allowed_commands or ALLOWED_COMMANDS.copy()
        self.check_path_exists = check_path_exists

        if custom_whitelist:
            self.allowed_commands.update(custom_whitelist)

        if allow_unsafe:
            logger.warning(
                "MCP SECURITY WARNING: Unsafe mode enabled. "
                "All commands will be allowed without validation. "
                "This should NEVER be used in production!"
            )

    def validate_command(self, command: str, server_name: str) -> None:
        """
        Validate that a command is safe to execute.

        Args:
            command: The command to validate
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If the command is not allowed
        """
        if self.allow_unsafe:
            logger.warning(
                f"MCP security bypassed for server '{server_name}': "
                f"allowing command '{command}' (unsafe mode)"
            )
            return

        # Check for dangerous patterns in command
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(command):
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Command contains dangerous pattern: "
                    f"'{command}'. Command injection attempt blocked."
                )

        # Extract base command name (without path)
        base_command = Path(command).name

        # Check if command is in whitelist
        if base_command not in self.allowed_commands:
            # Check if it's an absolute path to an allowed command
            if os.path.isabs(command):
                resolved_name = Path(command).name
                if resolved_name in self.allowed_commands:
                    # Verify the path actually exists and is executable
                    if os.path.isfile(command) and os.access(command, os.X_OK):
                        logger.info(
                            f"MCP server '{server_name}': Allowing absolute path "
                            f"to whitelisted command: {command}"
                        )
                        return

            raise MCPSecurityError(
                f"MCP server '{server_name}': Command '{base_command}' is not in the "
                f"allowed commands whitelist. Allowed commands: {sorted(self.allowed_commands)}"
            )

        # Verify command exists in PATH (for non-absolute paths)
        if self.check_path_exists and not os.path.isabs(command):
            resolved_path = shutil.which(command)
            if resolved_path is None:
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Command '{command}' not found in PATH. "
                    f"Ensure the command is installed and accessible."
                )

        logger.debug(
            f"MCP server '{server_name}': Command '{command}' validated successfully"
        )

    def validate_args(self, args: list[str], server_name: str) -> None:
        """
        Validate command arguments for dangerous patterns.

        Args:
            args: List of command arguments
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If any argument contains dangerous patterns
        """
        if self.allow_unsafe:
            return

        for i, arg in enumerate(args):
            for pattern in DANGEROUS_ARG_PATTERNS:
                if pattern.search(arg):
                    raise MCPSecurityError(
                        f"MCP server '{server_name}': Argument {i} contains dangerous "
                        f"pattern: '{arg}'. Potential command injection blocked."
                    )

        logger.debug(
            f"MCP server '{server_name}': {len(args)} arguments validated successfully"
        )

    def validate_env(self, env: dict[str, str] | None, server_name: str) -> None:
        """
        Validate environment variables for dangerous values.

        Args:
            env: Dictionary of environment variables
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If any env var contains dangerous patterns
        """
        if self.allow_unsafe or not env:
            return

        # Dangerous environment variables that could affect execution
        dangerous_env_vars = {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "PATH",  # Modifying PATH could redirect commands
            "PYTHONPATH",
            "NODE_PATH",
        }

        for key, value in env.items():
            # Check for dangerous env var names
            if key.upper() in dangerous_env_vars:
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Setting '{key}' environment variable "
                    f"is not allowed for security reasons."
                )

            # Check for dangerous patterns in values
            for pattern in DANGEROUS_ARG_PATTERNS:
                if pattern.search(value):
                    raise MCPSecurityError(
                        f"MCP server '{server_name}': Environment variable '{key}' "
                        f"contains dangerous pattern. Potential injection blocked."
                    )

        logger.debug(
            f"MCP server '{server_name}': {len(env)} environment variables validated"
        )

    def validate_url(self, url: str, server_name: str) -> None:
        """
        Validate SSE URL for security.

        Args:
            url: The SSE URL to validate
            server_name: Name of the MCP server (for logging)

        Raises:
            MCPSecurityError: If the URL is not safe
        """
        if self.allow_unsafe:
            return

        # Must be http or https
        if not url.startswith(("http://", "https://")):
            raise MCPSecurityError(
                f"MCP server '{server_name}': URL must use http:// or https:// scheme. "
                f"Got: {url}"
            )

        # Warn about non-HTTPS URLs
        if url.startswith("http://") and not url.startswith("http://localhost"):
            logger.warning(
                f"MCP server '{server_name}': Using insecure HTTP connection to {url}. "
                f"Consider using HTTPS for production environments."
            )

        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(url):
                raise MCPSecurityError(
                    f"MCP server '{server_name}': URL contains dangerous pattern: {url}"
                )

        logger.debug(f"MCP server '{server_name}': URL '{url}' validated successfully")


# Global validator instance (can be reconfigured)
_validator: MCPCommandValidator | None = None


def get_validator() -> MCPCommandValidator:
    """Get the global command validator instance."""
    global _validator
    if _validator is None:
        _validator = MCPCommandValidator()
    return _validator


def set_validator(validator: MCPCommandValidator) -> None:
    """Set a custom global validator."""
    global _validator
    _validator = validator


def validate_mcp_server_config(
    server_name: str,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    url: str | None = None,
) -> None:
    """
    Validate MCP server configuration for security.

    This is a convenience function that uses the global validator.

    Args:
        server_name: Name of the MCP server
        command: Command to execute (for stdio transport)
        args: Command arguments
        env: Environment variables
        url: SSE URL (for sse transport)

    Raises:
        MCPSecurityError: If validation fails
    """
    validator = get_validator()

    if command:
        validator.validate_command(command, server_name)

    if args:
        validator.validate_args(args, server_name)

    if env:
        validator.validate_env(env, server_name)

    if url:
        validator.validate_url(url, server_name)


# ============================================================================
# Tool Execution Sandboxing
# ============================================================================

# Dangerous tool argument patterns (for tool execution, not command args)
DANGEROUS_TOOL_ARG_PATTERNS: list[re.Pattern] = [
    re.compile(r"\.\./"),  # Path traversal
    re.compile(r"/etc/"),  # System config access
    re.compile(r"/proc/"),  # Process info
    re.compile(r"/sys/"),  # System info
    re.compile(r"~root"),  # Root home
    re.compile(r"/root/"),  # Root home directory
]

# Tools that are considered high-risk and require explicit allowlisting
HIGH_RISK_TOOL_PATTERNS: list[str] = [
    "execute",
    "run_command",
    "shell",
    "eval",
    "exec",
    "system",
    "subprocess",
]


@dataclass
class ToolExecutionAudit:
    """Record of a tool execution for audit purposes."""

    timestamp: float
    tool_name: str
    server_name: str
    arguments: dict[str, Any]
    success: bool
    error_message: str | None = None
    execution_time_ms: float | None = None


class ToolSandbox:
    """
    Sandboxing controls for MCP tool execution.

    Provides:
    - Tool allowlisting/blocklisting
    - Argument sanitization
    - Audit logging
    - Rate limiting
    """

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        blocked_tools: set[str] | None = None,
        allowed_high_risk_tools: set[str] | None = None,
        blocked_arg_patterns: list[re.Pattern] | None = None,
        max_calls_per_minute: int = 60,
        audit_callback: Callable[[ToolExecutionAudit], None] | None = None,
        enabled: bool = True,
    ):
        """
        Initialize tool sandbox.

        Args:
            allowed_tools: If set, only these tools can be executed (whitelist mode).
            blocked_tools: Tools that are always blocked (blacklist mode).
            allowed_high_risk_tools: High-risk tools that are explicitly allowed.
            blocked_arg_patterns: Patterns to block in tool arguments.
            max_calls_per_minute: Rate limit for tool calls (0 = unlimited).
            audit_callback: Optional callback for audit events.
            enabled: If False, sandbox checks are bypassed (dev mode only).
        """
        self.allowed_tools = allowed_tools
        self.blocked_tools = blocked_tools or set()
        self.allowed_high_risk_tools = {
            tool.lower() for tool in (allowed_high_risk_tools or set())
        }
        self.blocked_arg_patterns = (
            blocked_arg_patterns or DANGEROUS_TOOL_ARG_PATTERNS.copy()
        )
        self.max_calls_per_minute = max_calls_per_minute
        self.audit_callback = audit_callback
        self.enabled = enabled

        # Rate limiting state
        self._call_times: dict[str, list[float]] = defaultdict(list)
        self._rate_limit_lock = Lock()

        # Audit log (in-memory, bounded)
        self._audit_log: list[ToolExecutionAudit] = []
        self._audit_log_max_size = 1000
        self._audit_lock = Lock()

        if not enabled:
            logger.warning(
                "SECURITY WARNING: Tool sandbox is DISABLED. "
                "All tool executions will be allowed without checks."
            )

    def validate_tool_execution(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """
        Validate that a tool execution is allowed.

        Args:
            tool_name: Name of the tool to execute
            server_name: MCP server providing the tool
            arguments: Tool arguments

        Raises:
            MCPSecurityError: If execution is not allowed
        """
        if not self.enabled:
            logger.debug(f"Sandbox disabled, allowing tool '{tool_name}'")
            return

        full_name = f"{server_name}__{tool_name}"

        # Check blocklist first
        if self._is_blocked(tool_name, full_name):
            raise MCPSecurityError(f"Tool '{tool_name}' is blocked by security policy")

        # Check allowlist if configured
        if self.allowed_tools is not None:
            if (
                tool_name not in self.allowed_tools
                and full_name not in self.allowed_tools
            ):
                raise MCPSecurityError(
                    f"Tool '{tool_name}' is not in the allowed tools list"
                )

        # Check for high-risk tool patterns
        self._check_high_risk_tool(tool_name, full_name)

        # Validate arguments
        self._validate_arguments(tool_name, arguments)

        # Check rate limit
        self._check_rate_limit(full_name)

        logger.debug(f"Tool execution validated: {full_name}")

    def _is_blocked(self, tool_name: str, full_name: str) -> bool:
        """Check if tool is in blocklist."""
        return (
            tool_name in self.blocked_tools
            or full_name in self.blocked_tools
            or tool_name.lower() in self.blocked_tools
        )

    def _check_high_risk_tool(self, tool_name: str, full_name: str) -> None:
        """Check if tool matches high-risk patterns."""
        tool_lower = tool_name.lower()
        full_lower = full_name.lower()
        for pattern in HIGH_RISK_TOOL_PATTERNS:
            if pattern in tool_lower:
                if (
                    tool_lower in self.allowed_high_risk_tools
                    or full_lower in self.allowed_high_risk_tools
                ):
                    logger.warning(
                        "Allowing high-risk tool '%s' due to explicit allowlist entry",
                        full_name,
                    )
                    return
                raise MCPSecurityError(
                    f"High-risk tool '{tool_name}' is blocked by security policy. "
                    f"Add '{full_name}' or '{tool_name}' to allowed_high_risk_tools "
                    f"to allow it explicitly."
                )

    def _validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validate tool arguments for dangerous patterns."""

        def check_value(key: str, value: Any, path: str = "") -> None:
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                for pattern in self.blocked_arg_patterns:
                    if pattern.search(value):
                        raise MCPSecurityError(
                            f"Tool '{tool_name}' argument '{current_path}' contains "
                            f"blocked pattern: {pattern.pattern}"
                        )
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(k, v, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_value(f"[{i}]", item, current_path)

        for key, value in arguments.items():
            check_value(key, value)

    def _check_rate_limit(self, full_name: str) -> None:
        """Check and enforce rate limit for tool calls."""
        if self.max_calls_per_minute <= 0:
            return

        now = time.time()
        window_start = now - 60  # 1 minute window

        with self._rate_limit_lock:
            # Clean old entries
            self._call_times[full_name] = [
                t for t in self._call_times[full_name] if t > window_start
            ]

            # Check limit
            if len(self._call_times[full_name]) >= self.max_calls_per_minute:
                raise MCPSecurityError(
                    f"Rate limit exceeded for tool '{full_name}': "
                    f"max {self.max_calls_per_minute} calls per minute"
                )

            # Record this call
            self._call_times[full_name].append(now)

    def record_execution(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any],
        success: bool,
        error_message: str | None = None,
        execution_time_ms: float | None = None,
    ) -> ToolExecutionAudit:
        """
        Record a tool execution for audit purposes.

        Args:
            tool_name: Name of the executed tool
            server_name: MCP server that executed the tool
            arguments: Arguments passed to the tool
            success: Whether execution succeeded
            error_message: Error message if failed
            execution_time_ms: Execution time in milliseconds

        Returns:
            The audit record
        """
        audit = ToolExecutionAudit(
            timestamp=time.time(),
            tool_name=tool_name,
            server_name=server_name,
            arguments=self._sanitize_arguments_for_log(arguments),
            success=success,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
        )

        # Store in audit log
        with self._audit_lock:
            self._audit_log.append(audit)
            # Trim if over max size
            if len(self._audit_log) > self._audit_log_max_size:
                self._audit_log = self._audit_log[-self._audit_log_max_size :]

        # Log the execution
        if success:
            logger.info(
                f"AUDIT: Tool executed - {server_name}__{tool_name} "
                f"(took {execution_time_ms:.1f}ms)"
                if execution_time_ms
                else f"AUDIT: Tool executed - {server_name}__{tool_name}"
            )
        else:
            logger.warning(
                f"AUDIT: Tool failed - {server_name}__{tool_name}: {error_message}"
            )

        # Call callback if configured
        if self.audit_callback:
            try:
                self.audit_callback(audit)
            except Exception as e:
                logger.error(f"Audit callback failed: {e}")

        return audit

    def _sanitize_arguments_for_log(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Sanitize arguments for logging (redact sensitive data)."""
        sensitive_keys = {"password", "token", "secret", "key", "credential", "auth"}

        def sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: (
                        "[REDACTED]"
                        if any(s in k.lower() for s in sensitive_keys)
                        else sanitize(v)
                    )
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            elif isinstance(obj, str) and len(obj) > 1000:
                return obj[:100] + f"... [truncated, {len(obj)} chars total]"
            return obj

        return sanitize(arguments)

    def get_audit_log(
        self,
        limit: int = 100,
        tool_filter: str | None = None,
        server_filter: str | None = None,
        errors_only: bool = False,
    ) -> list[ToolExecutionAudit]:
        """
        Get audit log entries.

        Args:
            limit: Maximum entries to return
            tool_filter: Filter by tool name (substring match)
            server_filter: Filter by server name
            errors_only: Only return failed executions

        Returns:
            List of audit entries
        """
        with self._audit_lock:
            entries = self._audit_log.copy()

        # Apply filters
        if tool_filter:
            entries = [e for e in entries if tool_filter in e.tool_name]
        if server_filter:
            entries = [e for e in entries if server_filter in e.server_name]
        if errors_only:
            entries = [e for e in entries if not e.success]

        # Return most recent entries
        return entries[-limit:]

    def clear_audit_log(self) -> int:
        """Clear audit log and return number of entries cleared."""
        with self._audit_lock:
            count = len(self._audit_log)
            self._audit_log.clear()
        return count


# Global sandbox instance
_sandbox: ToolSandbox | None = None


def get_sandbox() -> ToolSandbox:
    """Get the global tool sandbox instance."""
    global _sandbox
    if _sandbox is None:
        _sandbox = ToolSandbox()
    return _sandbox


def set_sandbox(sandbox: ToolSandbox) -> None:
    """Set a custom global sandbox."""
    global _sandbox
    _sandbox = sandbox
