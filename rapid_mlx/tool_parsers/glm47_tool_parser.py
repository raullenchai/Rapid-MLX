# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7 tool call parser for rapid-mlx.

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


class Glm47ContentFilter:
    """Hide confirmed GLM envelopes and leading orphan arguments, not prose."""

    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"
    _ORPHANS = ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", _CLOSE)
    _HEADER = re.compile(r"<tool_call>\s*[\w.:-]*\s*")

    def __init__(self):
        self.enabled = True
        self._buffer = ""
        self._in_wire = False
        self._wire_end = self._CLOSE
        self._at_start = True

    def process(self, delta: str) -> str:
        if not self.enabled:
            return delta
        self._buffer += delta
        visible = []
        while self._buffer:
            if self._in_wire:
                end = self._buffer.find(self._wire_end)
                if end < 0:
                    # Only the possible closing-marker suffix needs retaining.
                    self._buffer = self._buffer[-len(self._wire_end) + 1 :]
                    break
                self._buffer = self._buffer[end + len(self._wire_end) :]
                self._in_wire = False
                continue
            markers = (self._OPEN,) + (self._ORPHANS if self._at_start else ())
            if any(
                len(self._buffer) < len(marker) and marker.startswith(self._buffer)
                for marker in markers
            ):
                break
            if self._at_start and self._buffer.startswith(self._ORPHANS):
                marker = next(m for m in self._ORPHANS if self._buffer.startswith(m))
                self._buffer = self._buffer[len(marker) :]
                if not marker.startswith("</"):
                    self._in_wire = True
                    self._wire_end = marker.replace("<", "</", 1)
                continue
            if self._buffer.startswith(self._OPEN):
                header = self._HEADER.match(self._buffer)
                assert header is not None  # The prefix check guarantees a match.
                tail = self._buffer[header.end() :]
                boundaries = ("<arg_key>", self._CLOSE)
                if tail.startswith(boundaries):
                    self._in_wire = True
                    self._wire_end = self._CLOSE
                    continue
                if not tail or any(marker.startswith(tail) for marker in boundaries):
                    break
            next_marker = self._buffer.find("<", 1)
            end = next_marker if next_marker >= 0 else len(self._buffer)
            text, self._buffer = self._buffer[:end], self._buffer[end:]
            visible.append(text)
            if text.strip():
                self._at_start = False
        return "".join(visible)

    def flush(self) -> str:
        incomplete_header = (
            self.enabled
            and self._buffer.startswith(self._OPEN)
            and (
                self._at_start
                or re.fullmatch(r"<tool_call>\s*[\w.:-]+\s*", self._buffer) is not None
            )
        )
        visible = "" if self._in_wire or incomplete_header else self._buffer
        self._buffer = ""
        self._in_wire = False
        return visible


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

    # GLM-4.7 chat templates render `tool_calls` natively via the same
    # <tool_call>…</tool_call> markup the parser emits, so feed previous-turn
    # tool calls back to the model in their native form instead of converting
    # them to a synthetic <function=…> string.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("glm_named_tool_call",)

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

    # A delayed reasoning close may interrupt the name, before arguments begin.
    _INTERRUPTED_NAME = re.compile(
        r"(<tool_call>\s*[\w.:-]+)\s*</think>\s*(?=<arg_key>)"
    )

    def reset(self) -> None:
        super().reset()
        self._content_filter = Glm47ContentFilter()

    def flush_held_content(self, full_text: str) -> str:
        """Release an incomplete literal prefix, but discard open wire blocks."""
        content_filter = getattr(self, "_content_filter", None)
        return content_filter.flush() if content_filter is not None else ""

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
        if not request:
            return set()
        tools = request.get("tools") or []
        return {
            t.get("function", {}).get("name", "") for t in tools if isinstance(t, dict)
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
        model_output = self._INTERRUPTED_NAME.sub(r"</think>\1", model_output)
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
                from ..tool_call_diagnostics import record_rejected_tool_call

                record_rejected_tool_call(func_name, args_section)
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
            content_filter = Glm47ContentFilter()
            content_filter.enabled = bool(
                request
                and request.get("tools")
                and request.get("tool_choice") != "none"
            )
            content = content_filter.process(cleaned_text) + content_filter.flush()
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=content
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
        # Reasoning bytes must not change the content filter's leading-orphan
        # state. Reset at the first close, including a split closing marker.
        if "<think>" in current_text and "</think>" not in current_text:
            return None
        filter_delta = delta_text
        if "</think>" in current_text and "</think>" not in previous_text:
            self._content_filter = Glm47ContentFilter()
            filter_delta = current_text.partition("</think>")[2]
        if not hasattr(self, "_content_filter"):
            self.reset()
        self._content_filter.enabled = bool(
            request and request.get("tools") and request.get("tool_choice") != "none"
        )
        # Include complete envelopes so a previously held prefix cannot survive
        # after its block is consumed.
        filtered_delta = self._content_filter.process(filter_delta)

        # Once <tool_call> is detected, buffer everything until it closes.
        # Do NOT emit content deltas here, because if tool calls are found
        # the non-streaming path sets content=None (reasoning before the
        # tag should not leak as regular content).
        if "<tool_call>" in current_text:
            if current_text.count("</tool_call>") > previous_text.count("</tool_call>"):
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    previous = self.extract_tool_calls(previous_text, request)
                    already = len(previous.tool_calls) if previous.tools_called else 0
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
                            for i, tc in enumerate(
                                result.tool_calls[already:], start=already
                            )
                        ]
                    }

        delta_text = filtered_delta

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
