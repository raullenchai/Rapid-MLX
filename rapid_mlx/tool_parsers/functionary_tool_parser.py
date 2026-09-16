# SPDX-License-Identifier: Apache-2.0
"""
Functionary tool call parser for rapid-mlx.

Handles MeetKai Functionary models' tool calling format.
Similar to OpenAI function calling with JSON arguments.
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


@ToolParserManager.register_module(["functionary", "meetkai"])
class FunctionaryToolParser(ToolParser):
    """
    Tool call parser for MeetKai Functionary models.

    Supports Functionary's tool call format similar to OpenAI:
    - Uses special tokens to mark tool calls
    - Arguments are JSON strings

    Formats supported:
    - <|from|>assistant\n<|recipient|>func_name\n<|content|>{"args": ...}
    - <function=name>{"args": ...}</function>

    Used when --enable-auto-tool-choice --tool-call-parser functionary are set.
    """

    # Functionary chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("functionary_native", "function_bare")

    # Functionary v3 format
    RECIPIENT_PATTERN = re.compile(
        r"<\|recipient\|>\s*(\w+)\s*\n<\|content\|>\s*(\{.*?\})(?=<\||$)",
        re.DOTALL,
    )

    # Alternative function format
    FUNCTION_PATTERN = re.compile(
        r"<function=([^>]+)>(\{.*?\})</function>",
        re.DOTALL,
    )

    # OpenAI-style JSON array
    JSON_ARRAY_PATTERN = re.compile(r"^\s*\[.*\]\s*$", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Functionary model output.
        """
        tool_calls = []
        cleaned_text = model_output

        # Try recipient pattern (Functionary v3)
        recipient_matches = self.RECIPIENT_PATTERN.findall(model_output)
        for func_name, args_str in recipient_matches:
            if func_name.lower() in ["all", "user"]:
                continue  # Skip non-function recipients
            try:
                json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": args_str,
                    }
                )
            except json.JSONDecodeError:
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": args_str,
                    }
                )

        if recipient_matches:
            cleaned_text = self.RECIPIENT_PATTERN.sub("", cleaned_text)
            cleaned_text = re.sub(r"<\|from\|>assistant\s*", "", cleaned_text).strip()

        # Try function pattern
        function_matches = self.FUNCTION_PATTERN.findall(cleaned_text)
        for func_name, args_str in function_matches:
            try:
                json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": args_str,
                    }
                )
            except json.JSONDecodeError:
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": args_str,
                    }
                )

        if function_matches:
            cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text).strip()

        # Try JSON array format
        if not tool_calls and self.JSON_ARRAY_PATTERN.match(model_output.strip()):
            try:
                parsed = json.loads(model_output.strip())
                if isinstance(parsed, list):
                    for call in parsed:
                        if isinstance(call, dict) and "name" in call:
                            args = call.get("arguments", {})
                            tool_calls.append(
                                {
                                    "id": generate_tool_id(),
                                    "name": call["name"],
                                    "arguments": (
                                        json.dumps(args, ensure_ascii=False)
                                        if isinstance(args, dict)
                                        else str(args)
                                    ),
                                }
                            )
                    cleaned_text = None
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
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
        Extract tool calls from streaming Functionary model output.
        """
        markers = ["<|recipient|>", "<function=", "["]
        has_marker = any(m in current_text for m in markers)

        if not has_marker:
            return {"content": delta_text}

        end_markers = ["<|content|>", "</function>", "]"]
        if any(m in delta_text for m in end_markers):
            result = self.extract_tool_calls(current_text)
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
