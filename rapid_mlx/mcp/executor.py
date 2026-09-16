# SPDX-License-Identifier: Apache-2.0
"""
Tool executor for handling tool calls from model responses.
"""

import asyncio
import logging
import time
from typing import Any

import jsonschema
from jsonschema import ValidationError

from .manager import MCPClientManager
from .security import MCPSecurityError, ToolSandbox, get_sandbox
from .tools import extract_tool_calls, format_tool_result
from .types import MCPTool, MCPToolResult

logger = logging.getLogger(__name__)


class ToolArgumentValidationError(Exception):
    """Raised when tool arguments fail validation against schema."""

    pass


def validate_tool_arguments(
    tool: MCPTool,
    arguments: dict[str, Any],
    strict: bool = True,
) -> None:
    """
    Validate tool arguments against the tool's input schema.

    Args:
        tool: The MCP tool with input_schema
        arguments: Arguments to validate
        strict: If True, raise exception on validation failure

    Raises:
        ToolArgumentValidationError: If validation fails and strict=True
    """
    schema = tool.input_schema
    if not schema:
        logger.debug(
            f"Tool '{tool.full_name}' has no input schema, skipping validation"
        )
        return

    try:
        jsonschema.validate(instance=arguments, schema=schema)
        logger.debug(f"Tool '{tool.full_name}' arguments validated successfully")
    except ValidationError as e:
        error_msg = (
            f"Tool '{tool.full_name}' argument validation failed: {e.message}. "
            f"Path: {'.'.join(str(p) for p in e.path) or 'root'}"
        )
        logger.warning(error_msg)
        if strict:
            raise ToolArgumentValidationError(error_msg) from e


class ToolExecutor:
    """
    Handles execution of tool calls from model responses.

    Provides utilities for:
    - Extracting tool calls from responses
    - Executing multiple tool calls (parallel or sequential)
    - Formatting results for conversation
    - Validating tool arguments against schemas
    """

    def __init__(
        self,
        manager: MCPClientManager,
        max_parallel: int = 5,
        default_timeout: float | None = None,
        validate_arguments: bool = True,
        sandbox: ToolSandbox | None = None,
    ):
        """
        Initialize tool executor.

        Args:
            manager: MCP client manager
            max_parallel: Maximum parallel tool executions
            default_timeout: Default timeout for tool calls
            validate_arguments: If True, validate arguments against tool schemas
            sandbox: Optional tool sandbox for security controls. Uses global if None.
        """
        self.manager = manager
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout or manager.config.default_timeout
        self.validate_arguments = validate_arguments
        self.sandbox = sandbox or get_sandbox()

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        parallel: bool = True,
    ) -> list[tuple[MCPToolResult, str]]:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of OpenAI tool call objects
            parallel: Execute in parallel (True) or sequential (False)

        Returns:
            List of (MCPToolResult, tool_call_id) tuples
        """
        if not tool_calls:
            return []

        if parallel:
            return await self._execute_parallel(tool_calls)
        else:
            return await self._execute_sequential(tool_calls)

    def _get_tool_by_name(self, full_name: str) -> MCPTool | None:
        """Get a tool by its full name (server__tool or just tool)."""
        for tool in self.manager.get_all_tools():
            if tool.full_name == full_name:
                return tool
        # Try without server prefix
        if "__" not in full_name:
            for tool in self.manager.get_all_tools():
                if tool.name == full_name:
                    return tool
        return None

    def _validate_tool_call(self, tool_call: dict[str, Any]) -> str | None:
        """
        Validate a tool call's arguments against the tool's schema.

        Returns:
            Error message if validation fails, None if valid
        """
        if not self.validate_arguments:
            return None

        func = tool_call.get("function", {})
        name = func.get("name", "")
        arguments = func.get("arguments", {})

        # Parse arguments if string
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Invalid JSON in arguments for tool '{name}'"

        tool = self._get_tool_by_name(name)
        if not tool:
            return None  # Let execution handle missing tool

        try:
            validate_tool_arguments(tool, arguments, strict=True)
            return None
        except ToolArgumentValidationError as e:
            return str(e)

    def _validate_sandbox(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """
        Validate tool execution against sandbox policy.

        Returns:
            Error message if blocked, None if allowed
        """
        try:
            self.sandbox.validate_tool_execution(tool_name, server_name, arguments)
            return None
        except MCPSecurityError as e:
            return str(e)

    def _get_server_for_tool(self, full_name: str) -> str:
        """Extract server name from full tool name or find it."""
        if "__" in full_name:
            return full_name.split("__")[0]
        # Find which server has this tool
        for tool in self.manager.get_all_tools():
            if tool.name == full_name:
                return tool.server_name
        return "unknown"

    async def _execute_parallel(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[tuple[MCPToolResult, str]]:
        """Execute tool calls in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(tool_call: dict[str, Any]):
            async with semaphore:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                call_id = tool_call.get("id", "")

                # Parse arguments if string
                if isinstance(arguments, str):
                    import json

                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                server_name = self._get_server_for_tool(name)
                tool_name = name.split("__")[-1] if "__" in name else name

                # Validate arguments before execution
                validation_error = self._validate_tool_call(tool_call)
                if validation_error:
                    self.sandbox.record_execution(
                        tool_name,
                        server_name,
                        arguments,
                        success=False,
                        error_message=validation_error,
                    )
                    return (
                        MCPToolResult(
                            tool_name=name,
                            content=None,
                            is_error=True,
                            error_message=validation_error,
                        ),
                        call_id,
                    )

                # Validate sandbox policy
                sandbox_error = self._validate_sandbox(
                    tool_name, server_name, arguments
                )
                if sandbox_error:
                    self.sandbox.record_execution(
                        tool_name,
                        server_name,
                        arguments,
                        success=False,
                        error_message=sandbox_error,
                    )
                    return (
                        MCPToolResult(
                            tool_name=name,
                            content=None,
                            is_error=True,
                            error_message=sandbox_error,
                        ),
                        call_id,
                    )

                # Execute with timing for audit
                start_time = time.time()
                result = await self.manager.execute_tool_call(
                    tool_call,
                    timeout=self.default_timeout,
                )
                execution_time_ms = (time.time() - start_time) * 1000

                # Record execution in audit log
                self.sandbox.record_execution(
                    tool_name,
                    server_name,
                    arguments,
                    success=not result.is_error,
                    error_message=result.error_message,
                    execution_time_ms=execution_time_ms,
                )

                return (result, call_id)

        tasks = [execute_with_semaphore(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            call_id = tool_calls[i].get("id", "")
            if isinstance(result, Exception):
                processed.append(
                    (
                        MCPToolResult(
                            tool_name=tool_calls[i].get("function", {}).get("name", ""),
                            content=None,
                            is_error=True,
                            error_message=str(result),
                        ),
                        call_id,
                    )
                )
            else:
                processed.append(result)

        return processed

    async def _execute_sequential(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[tuple[MCPToolResult, str]]:
        """Execute tool calls sequentially."""
        results = []
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", {})
            call_id = tool_call.get("id", "")

            # Parse arguments if string
            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            server_name = self._get_server_for_tool(name)
            tool_name = name.split("__")[-1] if "__" in name else name

            # Validate arguments before execution
            validation_error = self._validate_tool_call(tool_call)
            if validation_error:
                self.sandbox.record_execution(
                    tool_name,
                    server_name,
                    arguments,
                    success=False,
                    error_message=validation_error,
                )
                results.append(
                    (
                        MCPToolResult(
                            tool_name=name,
                            content=None,
                            is_error=True,
                            error_message=validation_error,
                        ),
                        call_id,
                    )
                )
                continue

            # Validate sandbox policy
            sandbox_error = self._validate_sandbox(tool_name, server_name, arguments)
            if sandbox_error:
                self.sandbox.record_execution(
                    tool_name,
                    server_name,
                    arguments,
                    success=False,
                    error_message=sandbox_error,
                )
                results.append(
                    (
                        MCPToolResult(
                            tool_name=name,
                            content=None,
                            is_error=True,
                            error_message=sandbox_error,
                        ),
                        call_id,
                    )
                )
                continue

            try:
                # Execute with timing for audit
                start_time = time.time()
                result = await self.manager.execute_tool_call(
                    tool_call,
                    timeout=self.default_timeout,
                )
                execution_time_ms = (time.time() - start_time) * 1000

                # Record execution in audit log
                self.sandbox.record_execution(
                    tool_name,
                    server_name,
                    arguments,
                    success=not result.is_error,
                    error_message=result.error_message,
                    execution_time_ms=execution_time_ms,
                )
                results.append((result, call_id))
            except Exception as e:
                self.sandbox.record_execution(
                    tool_name,
                    server_name,
                    arguments,
                    success=False,
                    error_message=str(e),
                )
                results.append(
                    (
                        MCPToolResult(
                            tool_name=name,
                            content=None,
                            is_error=True,
                            error_message=str(e),
                        ),
                        call_id,
                    )
                )
        return results

    async def execute_and_format(
        self,
        tool_calls: list[dict[str, Any]],
        parallel: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Execute tool calls and format results as messages.

        Args:
            tool_calls: List of OpenAI tool call objects
            parallel: Execute in parallel

        Returns:
            List of tool result messages ready for conversation
        """
        results = await self.execute_tool_calls(tool_calls, parallel)
        return [format_tool_result(result, call_id) for result, call_id in results]

    def extract_and_validate(
        self,
        response: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Extract tool calls from response and validate them.

        Args:
            response: Model response in OpenAI format

        Returns:
            Tuple of (tool_calls, all_valid)
        """
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            return [], True

        # Validate each tool call
        all_valid = True
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")

            # Check if tool exists
            if not self._tool_exists(name):
                logger.warning(f"Tool '{name}' not found in any MCP server")
                all_valid = False

        return tool_calls, all_valid

    def _tool_exists(self, full_name: str) -> bool:
        """Check if a tool exists in any connected server."""
        # Check by full name
        for tool in self.manager.get_all_tools():
            if tool.full_name == full_name:
                return True

        # Check by just tool name (without server prefix)
        if "__" not in full_name:
            for tool in self.manager.get_all_tools():
                if tool.name == full_name:
                    return True

        return False


async def execute_single_tool(
    manager: MCPClientManager,
    tool_name: str,
    arguments: dict[str, Any],
    timeout: float | None = None,
) -> MCPToolResult:
    """
    Convenience function to execute a single tool.

    Args:
        manager: MCP client manager
        tool_name: Full tool name (server__tool)
        arguments: Tool arguments
        timeout: Optional timeout

    Returns:
        MCPToolResult
    """
    return await manager.execute_tool(tool_name, arguments, timeout)
