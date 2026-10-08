# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek tool call parser for rapid-mlx.

Handles DeepSeek V3 and R1 tool calling formats:
- <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> wrapper
- <｜tool▁call▁begin｜>function<｜tool▁sep｜>name
  ```json
  {...}
  ```<｜tool▁call▁end｜>
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


# Note: ``deepseek_v3`` was previously aliased here. As of R12-5 it is
# owned by ``DeepSeekV3ToolParser`` (deepseek_v3_tool_parser.py) — the
# dedicated, block-wise V3 parser. This legacy parser retains the
# ``deepseek`` / ``deepseek_r1`` names for backward compatibility with
# pre-R12-5 configurations (DeepSeek V2 distill / R1 non-0528 family).
@ToolParserManager.register_module(["deepseek", "deepseek_r1"])
class DeepSeekToolParser(ToolParser):
    """
    Tool call parser for DeepSeek V2 / R1 (non-0528) models.

    Supports DeepSeek's tool call format with special unicode tokens:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"city": "Paris"}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

    Used when --enable-auto-tool-choice --tool-call-parser deepseek are set.
    """

    EXPECTED_WIRE_FORMATS = ("deepseek_native",)

    # DeepSeek V3 chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Special DeepSeek tokens (unicode)
    TOOL_CALLS_START = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_START = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"

    # Pattern to match individual tool calls
    TOOL_CALL_PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    # Alternative pattern without type
    TOOL_CALL_SIMPLE_PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from DeepSeek model output.
        """
        # Check for tool calls marker
        if self.TOOL_CALLS_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []

        # Extract content before tool calls
        content_end = model_output.find(self.TOOL_CALLS_START)
        content = model_output[:content_end].strip() if content_end > 0 else None

        # Try full pattern with type first
        matches = self.TOOL_CALL_PATTERN.findall(model_output)
        for match in matches:
            tool_type, func_name, func_args = match
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
                # Keep raw arguments
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )

        # Try simple pattern if no matches
        if not tool_calls:
            simple_matches = self.TOOL_CALL_SIMPLE_PATTERN.findall(model_output)
            for match in simple_matches:
                func_name, func_args = match
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
        Extract tool calls from streaming DeepSeek model output.
        """
        if self.TOOL_CALLS_START not in current_text:
            marker = self.TOOL_CALLS_START
            current_hold = max(
                (n for n in range(1, len(marker)) if current_text.endswith(marker[:n])),
                default=0,
            )
            previous_hold = max(
                (
                    n
                    for n in range(1, len(marker))
                    if previous_text.endswith(marker[:n])
                ),
                default=0,
            )
            current_safe = len(current_text) - current_hold
            previous_safe = len(previous_text) - previous_hold
            content = current_text[previous_safe:current_safe]
            return {"content": content} if content else None

        # Compare complete parsed blocks, not close markers in this delta:
        # the outer close often arrives after the block close and must not
        # replay the same call (or miss a split close token).
        if current_text.count(self.TOOL_CALL_END) > previous_text.count(
            self.TOOL_CALL_END
        ):
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                previous = self.extract_tool_calls(previous_text)
                already = len(previous.tool_calls) if previous.tools_called else 0
                new_calls = result.tool_calls[already:]
                if not new_calls:
                    return None
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
                        for i, tc in enumerate(new_calls, start=already)
                    ]
                }

        return None
