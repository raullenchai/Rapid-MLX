# SPDX-License-Identifier: Apache-2.0
"""
Mistral tool call parser for rapid-mlx.

Handles Mistral's tool calling format:
- Format: [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
- Or newer: [TOOL_CALLS]func_name{"arg": "value"}

Used with models like Mistral-7B-Instruct, Devstral, etc.
"""

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

ALPHANUMERIC = ascii_letters + digits


def generate_mistral_tool_id() -> str:
    """
    Generate a random Mistral-compatible tool call ID.

    Mistral Tool Call IDs must be alphanumeric with a length of 9.
    """
    return "".join(choices(ALPHANUMERIC, k=9))


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral models.

    Supports both old and new Mistral tool call formats:
    - Old (< v11): [TOOL_CALLS] [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    - New (>= v11): [TOOL_CALLS]add{"a": 1, "b": 2}

    Used when --enable-auto-tool-choice --tool-call-parser mistral are set.
    """

    # Mistral chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("mistral_tool_calls",)

    BOT_TOKEN = "[TOOL_CALLS]"
    TOOL_CALL_REGEX = re.compile(r"\[{.*}\]", re.DOTALL)

    def has_pending_tool_call(self, text: str) -> bool:
        return "[TOOL_CALLS]" in text

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.bot_token_id = self.vocab.get(self.BOT_TOKEN) if self.vocab else None
        # Number of complete tool calls already emitted on the streaming path.
        self.streamed_tool_call_count: int = 0

    def reset(self) -> None:
        """Reset parser state for a new request (streaming counters too)."""
        super().reset()
        self.streamed_tool_call_count = 0

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Mistral model response.

        Args:
            model_output: The complete model output string
            request: Optional request context

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        # If the tool call token is not present, return as text response
        if self.BOT_TOKEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_and_raw_tool_calls = model_output.split(self.BOT_TOKEN)
        content = content_and_raw_tool_calls[0].strip()
        raw_tool_calls = content_and_raw_tool_calls[1:]

        tool_calls = []

        for raw_tool_call in raw_tool_calls:
            raw_tool_call = raw_tool_call.strip()
            if not raw_tool_call:
                continue

            # Try new format first: func_name{"arg": "value"}
            # Devstral may emit func_name[ARGS]{"arg": "value"} — strip [ARGS].
            if not raw_tool_call.startswith("[") and "{" in raw_tool_call:
                end_name = raw_tool_call.find("{")
                tool_name = raw_tool_call[:end_name].replace("[ARGS]", "").strip()
                args_str = raw_tool_call[end_name:]

                if tool_name:
                    tool_calls.append(
                        {
                            "id": generate_mistral_tool_id(),
                            "name": tool_name,
                            "arguments": args_str,
                        }
                    )
                continue

            # Try old format: [{"name": "func", "arguments": {...}}]
            try:
                parsed = json.loads(raw_tool_call)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            args = item.get("arguments", {})
                            tool_calls.append(
                                {
                                    "id": generate_mistral_tool_id(),
                                    "name": item["name"],
                                    "arguments": (
                                        json.dumps(args, ensure_ascii=False)
                                        if isinstance(args, dict)
                                        else str(args)
                                    ),
                                }
                            )
                continue
            except json.JSONDecodeError:
                pass

            # Fallback: try regex to extract JSON array
            try:
                match = self.TOOL_CALL_REGEX.search(raw_tool_call)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                args = item.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_mistral_tool_id(),
                                        "name": item["name"],
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
            except (json.JSONDecodeError, AttributeError):
                # If all parsing fails, treat as content
                if raw_tool_call:
                    content = (
                        (content + " " + raw_tool_call).strip()
                        if content
                        else raw_tool_call
                    )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
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
        Extract tool calls from streaming Mistral model output.

        Rather than parse each token delta in isolation — which can't
        reliably reconstruct ``[TOOL_CALLS]name[ARGS]{json}`` once the
        ``[ARGS]`` separator or the JSON body is split across arbitrary
        token boundaries (issue #579) — we re-parse the full accumulated
        text with the non-streaming :meth:`extract_tool_calls` (the single
        source of truth) and emit each tool call exactly once, the moment
        its arguments form complete JSON. This guarantees stream ↔
        non-stream parity by construction.
        """
        # No tool call marker yet → plain content delta.
        if self.BOT_TOKEN not in current_text:
            return {"content": delta_text}

        result: dict[str, Any] = {}

        # Emit any content that preceded the marker, once, on the delta
        # where the marker first appears.
        if self.BOT_TOKEN not in previous_text:
            leading = delta_text.split(self.BOT_TOKEN, 1)[0]
            if leading:
                result["content"] = leading

        # Re-parse everything accumulated so far and keep only the tool
        # calls whose arguments are complete, parseable JSON. A call still
        # mid-stream has a truncated body and is skipped until it closes.
        parsed = self.extract_tool_calls(current_text)
        complete: list[dict[str, Any]] = []
        for tool_call in parsed.tool_calls:
            args = tool_call.get("arguments")
            try:
                json.loads(args)
            except (TypeError, json.JSONDecodeError):
                continue
            complete.append(tool_call)

        # Emit only newly completed calls (those past what we've streamed).
        new_calls = complete[self.streamed_tool_call_count :]
        if new_calls:
            deltas = []
            for offset, tool_call in enumerate(new_calls):
                deltas.append(
                    {
                        "index": self.streamed_tool_call_count + offset,
                        "id": tool_call.get("id") or generate_mistral_tool_id(),
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                        },
                    }
                )
            self.streamed_tool_call_count += len(new_calls)
            result["tool_calls"] = deltas

        return result or None
