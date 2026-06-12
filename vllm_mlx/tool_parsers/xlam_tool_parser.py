# SPDX-License-Identifier: Apache-2.0
"""
xLAM tool call parser for rapid-mlx.

Handles Salesforce xLAM models' tool calling format which supports:
- JSON arrays of tool calls
- Tool calls in markdown code blocks
- Tool calls after </think> reasoning blocks
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


@ToolParserManager.register_module("xlam")
class xLAMToolParser(ToolParser):
    """
    Tool call parser for Salesforce xLAM models.

    Supports multiple formats:
    - JSON array: [{"name": "func", "arguments": {...}}]
    - Markdown code blocks: ```json [...] ```
    - After thinking: </think>[...]

    Used when --enable-auto-tool-choice --tool-call-parser xlam are set.
    """

    EXPECTED_WIRE_FORMATS = ("raw_json",)

    # Patterns for extracting JSON
    CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
    THINKING_PATTERN = re.compile(r"</think>\s*([\s\S]*)")
    TOOL_CALLS_TAG_PATTERN = re.compile(r"\[TOOL_CALLS\]([\s\S]*?)(?:\n|$)")

    def _try_extract_json(self, text: str) -> tuple[str | None, list | None]:
        """
        Try to extract JSON tool calls from text.

        Returns:
            Tuple of (content, tool_calls_list)
        """
        # Try markdown code blocks
        for pattern in [
            self.CODE_BLOCK_PATTERN,
            self.TOOL_CALLS_TAG_PATTERN,
        ]:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    if isinstance(parsed, list):
                        content = pattern.sub("", text).strip()
                        return content if content else None, parsed
                except json.JSONDecodeError:
                    continue

        # Try after </think> tag
        thinking_match = self.THINKING_PATTERN.search(text)
        if thinking_match:
            after_think = thinking_match.group(1).strip()
            try:
                parsed = json.loads(after_think)
                if isinstance(parsed, list):
                    content = text[: thinking_match.start() + len("</think>")].strip()
                    return content if content else None, parsed
            except json.JSONDecodeError:
                pass

        # Try entire text as JSON array
        text = text.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return None, parsed
            except json.JSONDecodeError:
                pass

        return text, None

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from xLAM model output.
        """
        content, tool_calls_data = self._try_extract_json(model_output)

        if not tool_calls_data:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=content or model_output
            )

        tool_calls = []
        for call in tool_calls_data:
            if isinstance(call, dict) and "name" in call:
                args = call.get("arguments", call.get("parameters", {}))
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

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
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
        Extract tool calls from streaming xLAM model output.
        """
        # Check for any indicators of tool calls
        markers = ["```", "[TOOL_CALLS]", "</think>"]
        has_marker = any(m in current_text for m in markers)

        # Also check for JSON array start
        stripped = current_text.strip()
        if stripped.startswith("[") and "{" in stripped:
            has_marker = True

        if not has_marker:
            return {"content": delta_text}

        # Try to parse when we see completion markers
        if "]" in delta_text or "```" in delta_text:
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
