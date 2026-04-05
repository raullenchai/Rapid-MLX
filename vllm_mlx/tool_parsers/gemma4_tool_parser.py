# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's native tool calling format:
  <|tool_call>call:FUNC_NAME{key:<|"|>value<|"|>,...}<tool_call|>
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

# Match: <|tool_call>call:name{...}<tool_call|>
GEMMA4_TOOL_PATTERN = re.compile(
    r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL
)

# Match key-value pairs inside {}: key:<|"|>value<|"|>
GEMMA4_KV_PATTERN = re.compile(
    r'(\w+):<\|"\|>(.*?)<\|"\|>', re.DOTALL
)


def _parse_gemma4_args(args_str: str) -> dict[str, Any]:
    """Parse Gemma 4's key:<|"|>value<|"|> format into a dict."""
    result = {}
    for match in GEMMA4_KV_PATTERN.finditer(args_str):
        key = match.group(1)
        value = match.group(2)
        # Try to parse as JSON value (number, bool, etc.)
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            result[key] = value
    return result


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["gemma4", "gemma_4"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Format: <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_tool_count = 0

    def reset(self):
        """Reset state for a new request."""
        self._emitted_tool_count = 0

    def extract_tool_calls(
        self, model_output: str, request: Any = None
    ) -> ExtractedToolCallInformation:
        matches = list(GEMMA4_TOOL_PATTERN.finditer(model_output))

        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)
            args = _parse_gemma4_args(args_str)

            tool_calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(args),
                }
            )

        # Content is everything outside the tool calls
        content = GEMMA4_TOOL_PATTERN.sub("", model_output).strip() or None

        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence = (),
        current_token_ids: Sequence = (),
        delta_token_ids: Sequence = (),
    ) -> dict | None:
        # Check if we're inside a tool call
        if "<|tool_call>" in current_text:
            # Count completed tool calls so far
            completed = current_text.count("<tool_call|>")
            open_count = current_text.count("<|tool_call>")

            # Still accumulating an incomplete tool call
            if completed < open_count:
                return None  # suppress output while inside tool markup

            # Only emit newly completed tool calls (dedup)
            if completed <= self._emitted_tool_count:
                return None

            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                # Only emit tool calls we haven't sent yet
                new_calls = result.tool_calls[self._emitted_tool_count:]
                self._emitted_tool_count = len(result.tool_calls)

                if new_calls:
                    return {
                        "tool_calls": [
                            {
                                "index": self._emitted_tool_count
                                - len(new_calls)
                                + i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(new_calls)
                        ]
                    }

        # No tool call markup — pass through as content
        return {"content": delta_text}
