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

# Match a quoted-string value: <|"|>...<|"|>
GEMMA4_QUOTED_VAL_PATTERN = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
# Match a bare key:value pair (key, then anything up to , or end-of-string)
GEMMA4_KV_BARE_PATTERN = re.compile(r"(\w+)\s*:\s*([^,]+?)(?=\s*,|\s*$)")


def _parse_gemma4_args(args_str: str) -> dict[str, Any]:
    """Parse Gemma 4's argument format into a dict.

    Gemma 4 uses two value styles inside the {...} block:
      - String values are wrapped in quote tokens:  key:<|"|>value<|"|>
      - Numeric / bool / null values are bare:      key:3   key:true   key:null

    Strategy: replace each quoted string with a placeholder, run a generic
    bare-KV parser over the result, then restore placeholders before
    returning. This lets a single pass handle mixed-type arg dicts.
    """
    # Step 1: stash quoted string values so they can't confuse the bare parser
    stashed: list[str] = []

    def _stash(m: re.Match) -> str:
        stashed.append(m.group(1))
        return f"__Q{len(stashed) - 1}__"

    cleaned = GEMMA4_QUOTED_VAL_PATTERN.sub(_stash, args_str)

    # Step 2: bare KV parse
    result: dict[str, Any] = {}
    for kv in GEMMA4_KV_BARE_PATTERN.finditer(cleaned):
        key = kv.group(1)
        raw_val = kv.group(2).strip()
        # Restore stashed string
        if raw_val.startswith("__Q") and raw_val.endswith("__"):
            try:
                idx = int(raw_val[3:-2])
                result[key] = stashed[idx]
                continue
            except (ValueError, IndexError):
                pass
        # Try to parse as JSON literal (int, float, bool, null)
        try:
            result[key] = json.loads(raw_val)
        except (json.JSONDecodeError, ValueError):
            result[key] = raw_val
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
        super().reset()
        self._emitted_tool_count = 0

    def has_pending_tool_call(self, text: str) -> bool:
        """Gemma 4 uses <|tool_call> (with pipe), not <tool_call>."""
        return "<|tool_call>" in text or self.has_text_format_tool_call(text)

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

        # Text-format tool call recovery: catch [Calling tool: name({...})]
        # Models degrade to this format after multiple tool rounds at low quant
        from .abstract_tool_parser import TEXT_TOOL_CALL_FN_PATTERN, TEXT_TOOL_CALL_ANY

        if TEXT_TOOL_CALL_ANY.search(current_text):
            # Check if we have a complete text tool call
            matches = list(TEXT_TOOL_CALL_FN_PATTERN.finditer(current_text))
            new_matches = matches[self._emitted_tool_count:]
            if new_matches:
                self._emitted_tool_count = len(matches)
                return {
                    "tool_calls": [
                        {
                            "index": self._emitted_tool_count - len(new_matches) + i,
                            "id": _generate_tool_id(),
                            "type": "function",
                            "function": {
                                "name": m.group(1),
                                "arguments": m.group(2),
                            },
                        }
                        for i, m in enumerate(new_matches)
                    ]
                }
            # Already emitted or partial — suppress
            return None

        # No tool call markup — pass through as content
        return {"content": delta_text}
