# SPDX-License-Identifier: Apache-2.0
"""
Granite tool call parser for vllm-mlx.

Handles IBM Granite models' tool calling format:
- <|tool_call|> or <tool_call> followed by JSON array
- [{"name": "func", "arguments": {...}}]
"""

import json
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


@ToolParserManager.register_module(["granite", "granite3"])
class GraniteToolParser(ToolParser):
    """
    Tool call parser for IBM Granite models.

    Supports Granite's tool call format:
    <|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]

    Or Granite 3.1:
    <tool_call>[{"name": "get_weather", "arguments": {"city": "Paris"}}]

    Used when --enable-auto-tool-choice --tool-call-parser granite are set.
    """

    # Granite 3.1 chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("granite_native",)

    BOT_TOKEN = "<|tool_call|>"
    BOT_STRING = "<tool_call>"

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Granite model output.
        """
        stripped = model_output.strip()

        # Remove tool call markers
        if stripped.startswith(self.BOT_TOKEN):
            stripped = stripped[len(self.BOT_TOKEN) :].lstrip()
        elif stripped.startswith(self.BOT_STRING):
            stripped = stripped[len(self.BOT_STRING) :].lstrip()

        # Check if it starts with JSON array
        if not stripped or stripped[0] != "[":
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            raw_calls = json.loads(stripped)
            if not isinstance(raw_calls, list):
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = []
            for call in raw_calls:
                if isinstance(call, dict):
                    # Granite uses "name" or "type" for function name
                    func_name = call.get("name") or call.get("type")
                    if func_name:
                        args = call.get("arguments", {})
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": func_name,
                                "arguments": (
                                    json.dumps(args, ensure_ascii=False)
                                    if isinstance(args, dict)
                                    else str(args)
                                ),
                            }
                        )

            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=None,
                )

        except json.JSONDecodeError:
            pass

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
        Extract tool calls from streaming Granite model output.
        """
        stripped = current_text.strip()
        has_marker = stripped.startswith(self.BOT_TOKEN) or stripped.startswith(
            self.BOT_STRING
        )

        if not has_marker:
            return {"content": delta_text}

        # Try to parse when we have a complete JSON array
        if "]" in delta_text:
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
