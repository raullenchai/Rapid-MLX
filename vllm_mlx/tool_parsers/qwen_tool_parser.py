# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for rapid-mlx.

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

    Note on naming: "qwen3_xml" in the registration list refers to the XML
    *wrapper* (<tool_call>...</tool_call> tags), NOT to XML-body parameters.
    The JSON body inside the wrapper is what this parser extracts. Vanilla
    Qwen3.6 non-reasoning models emit XML-body parameters, which this parser
    does NOT handle on its own — the cross-format fallback at
    service/postprocessor.py::finalize() (PR #426, fixes #425) routes those
    through api.tool_calling.parse_tool_calls.
    """

    EXPECTED_WIRE_FORMATS = ("tool_call_json", "calling_tool_text")

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

        # Use the count of *successfully parsed* tool calls in previous_text
        # as the dedup offset, not raw close-marker count. If a malformed
        # tool call slips past the close-marker counter but fails JSON parse,
        # raw counts desync from emitted-call indices and later valid calls
        # get wrong indices or get dropped. Re-parsing previous_text costs
        # one extra extract per delta but stays correct under malformed input.
        prev_close_count = previous_text.count("</tool_call>") + previous_text.count(
            ")]"
        )
        cur_close_count = current_text.count("</tool_call>") + current_text.count(")]")

        if cur_close_count <= prev_close_count:
            return None

        result = self.extract_tool_calls(current_text, request)
        if not result.tools_called:
            return None

        prev_result = self.extract_tool_calls(previous_text, request)
        prev_emitted = len(prev_result.tool_calls) if prev_result.tools_called else 0

        new_calls = result.tool_calls[prev_emitted:]
        if not new_calls:
            return None

        return {
            "tool_calls": [
                {
                    "index": prev_emitted + i,
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
