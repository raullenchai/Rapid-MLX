# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for vllm-mlx.

Handles Qwen's tool calling formats:
- XML style: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket style: [Calling tool: func_name({"arg": "value"})]
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


@ToolParserManager.register_module(["qwen", "qwen3", "qwen3_xml"])
class QwenToolParser(ToolParser):
    """
    Tool call parser for Qwen models.

    Supports multiple Qwen tool call formats:
    - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Bracket: [Calling tool: func_name({"arg": "value"})]

    Used when --enable-auto-tool-choice --tool-call-parser qwen are set.
    """

    # Pattern for XML-style: <tool_call>{"json"}</tool_call>
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # Pattern for bracket-style: [Calling tool: func_name({...})]
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Qwen model response.
        """
        tool_calls = []

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(model_output)

        # Try bracket pattern first (Qwen3 style)
        bracket_matches = self.BRACKET_PATTERN.findall(cleaned_text)
        for name, args_str in bracket_matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                continue

        if bracket_matches:
            cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text).strip()

        # Try XML pattern (traditional Qwen style)
        xml_matches = self.XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if xml_matches:
            cleaned_text = self.XML_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
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
        Extract tool calls from streaming Qwen model output.

        Counts closing markers in current vs previous text to dedup already-
        emitted tool calls (same pattern as HermesToolParser). Without this
        dedup, every closing marker re-emitted ALL tool calls found so far,
        which the OpenAI streaming protocol then merges by `index` →
        `name="readread"` and `arguments="{}{}"` for two-tool turns.
        """
        # Check for tool call markers
        has_tool_marker = (
            "<tool_call>" in current_text or "[Calling tool:" in current_text
        )

        if not has_tool_marker:
            return {"content": delta_text}

        # Count closes seen in current vs previous text. A "close" is either
        # </tool_call> (XML form) or )] (bracket form). Both styles consume
        # exactly one close per tool call, so a single counter works for the
        # mixed case too.
        prev_closes = previous_text.count("</tool_call>") + previous_text.count(")]")
        cur_closes = current_text.count("</tool_call>") + current_text.count(")]")

        if cur_closes > prev_closes:
            # New tool call(s) completed in this delta — parse and emit only
            # the newly-completed ones, indexed past what we've already sent.
            result = self.extract_tool_calls(current_text, request)
            if result.tools_called:
                new_calls = result.tool_calls[prev_closes:]
                if new_calls:
                    return {
                        "tool_calls": [
                            {
                                "index": prev_closes + i,
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

        return None
