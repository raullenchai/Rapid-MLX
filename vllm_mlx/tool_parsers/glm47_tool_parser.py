# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7 tool call parser for vllm-mlx.

Handles GLM-4.7-Flash style tool calling format.
Based on vLLM's glm47_moe_tool_parser.py
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["glm47", "glm4"])
class Glm47ToolParser(ToolParser):
    """
    Tool call parser for GLM-4.7 and GLM-4.7-Flash models.

    Supports GLM-4.7 tool call format:
    <tool_call>function_name
    <arg_key>param1</arg_key><arg_value>value1</arg_value>
    <arg_key>param2</arg_key><arg_value>value2</arg_value>
    </tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser glm47 are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Match entire tool call block
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # Match function name and optional arguments
    # GLM47 format: <tool_call>func_name\n<arg_key>...</arg_key>...
    FUNC_DETAIL_PATTERN = re.compile(
        r"<tool_call>\s*([^\n<]+?)(?:\n|\s*)(<arg_key>.*?)?</tool_call>", re.DOTALL
    )

    # Match individual argument key-value pairs
    ARG_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
    )

    def _deserialize(self, value: str) -> Any:
        """Convert string value to appropriate Python type.

        Uses json.loads for type coercion, falls back to raw string.
        """
        value = value.strip()

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request."""
        if not request or "tools" not in request:
            return set()
        return {
            t.get("function", {}).get("name", "")
            for t in request.get("tools", [])
            if isinstance(t, dict)
        }

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete GLM-4.7 model response.
        """
        tool_calls = []

        # Strip think tags using the base class method (handles both
        # full <think>...</think> and implicit ...</think> patterns)
        cleaned_text = self.strip_think_tags(model_output)

        # Get valid tool names for validation
        valid_names = self._get_tool_names(request)

        # Find all tool call blocks
        matches = self.FUNC_DETAIL_PATTERN.findall(cleaned_text)

        for match in matches:
            func_name = match[0].strip() if match[0] else ""
            args_section = match[1] if len(match) > 1 and match[1] else ""

            if not func_name:
                continue

            # Validate tool name against available tools if provided
            if valid_names and func_name not in valid_names:
                continue

            # Parse arguments
            arguments = {}
            if args_section:
                arg_matches = self.ARG_PATTERN.findall(args_section)
                for arg_key, arg_value in arg_matches:
                    key = arg_key.strip()
                    value = self._deserialize(arg_value)
                    if key:
                        arguments[key] = value

            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        # When tool calls are found, don't return reasoning text as content
        # GLM often outputs thinking/reasoning before tool calls without <think> tags
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming GLM-4.7 model output.
        """
        # Skip thinking content in streaming
        if "<think>" in current_text and "</think>" not in current_text:
            return None

        # Once <tool_call> is detected, buffer everything until it closes.
        # Do NOT emit content deltas here, because if tool calls are found
        # the non-streaming path sets content=None (reasoning before the
        # tag should not leak as regular content).
        if "<tool_call>" in current_text:
            if "</tool_call>" in delta_text:
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    return {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(result.tool_calls)
                        ]
                    }
            return None

        # No tool call detected yet; emit content delta.
        # Only strip think tags if they're actually present (avoid .strip()
        # on normal deltas which would eat inter-word spaces).
        if "</think>" in delta_text:
            clean_delta = self.strip_think_tags(delta_text)
            if clean_delta:
                return {"content": clean_delta}
            return None
        if delta_text:
            return {"content": delta_text}
        return None
