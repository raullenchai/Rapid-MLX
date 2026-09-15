# SPDX-License-Identifier: Apache-2.0
"""
Kimi/Moonshot tool call parser for rapid-mlx.

Handles Kimi K2 and related models' tool calling format:
- <|tool_calls_section_begin|>...<|tool_calls_section_end|>
- <|tool_call_begin|>func_name:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
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


@ToolParserManager.register_module(["kimi", "kimi_k2", "moonshot"])
class KimiToolParser(ToolParser):
    """
    Tool call parser for Kimi K2 and Moonshot models.

    Supports Kimi's tool call format:
    <|tool_calls_section_begin|>
    <|tool_call_begin|>func:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
    <|tool_calls_section_end|>

    Used when --enable-auto-tool-choice --tool-call-parser kimi are set.
    """

    # Kimi chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("kimi_native",)

    # Kimi tokens
    TOOL_CALLS_START = "<|tool_calls_section_begin|>"
    TOOL_CALLS_START_ALT = "<|tool_call_section_begin|>"  # Singular variant
    TOOL_CALLS_END = "<|tool_calls_section_end|>"
    TOOL_CALLS_END_ALT = "<|tool_call_section_end|>"
    TOOL_CALL_START = "<|tool_call_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_ARG_START = "<|tool_call_argument_begin|>"

    # Pattern to match individual tool calls
    TOOL_CALL_PATTERN = re.compile(
        r"<\|tool_call_begin\|>\s*(?P<func_id>[^<]+?)(?::\d+)?\s*<\|tool_call_argument_begin\|>\s*(?P<args>.*?)\s*<\|tool_call_end\|>",
        re.DOTALL,
    )

    def _has_tool_section(self, text: str) -> bool:
        """Check if text contains tool section markers."""
        return (
            self.TOOL_CALLS_START in text
            or self.TOOL_CALLS_START_ALT in text
            or self.TOOL_CALL_START in text
        )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Kimi model output.
        """
        if not self._has_tool_section(model_output):
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []

        # Extract content before tool calls
        content = None
        for marker in [self.TOOL_CALLS_START, self.TOOL_CALLS_START_ALT]:
            if marker in model_output:
                idx = model_output.find(marker)
                content = model_output[:idx].strip() if idx > 0 else None
                break

        # Find all tool calls
        matches = self.TOOL_CALL_PATTERN.findall(model_output)
        for match in matches:
            func_id, func_args = match
            # func_id format: functions.get_weather:0 or get_weather:0
            func_name = func_id.split(":")[-2] if ":" in func_id else func_id
            func_name = func_name.split(".")[-1]  # Remove 'functions.' prefix

            try:
                # Validate JSON
                json.loads(func_args)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )
            except json.JSONDecodeError:
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )
        else:
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
        Extract tool calls from streaming Kimi model output.
        """
        if not self._has_tool_section(current_text):
            return {"content": delta_text}

        if self.TOOL_CALL_END in delta_text:
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
