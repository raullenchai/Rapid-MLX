# SPDX-License-Identifier: Apache-2.0
"""Rapid-native parser for Cohere North action envelopes.

North-Mini-Code emits ``<|START_ACTION|>`` followed by a JSON object or array
and ``<|END_ACTION|>``.  Unlike mlx-lm's lightweight parser hook, this class
returns Rapid's ``ExtractedToolCallInformation`` and streaming delta shapes.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


@ToolParserManager.register_module(["cohere", "cohere2_moe"])
class CohereToolParser(ToolParser):
    """Parse North's JSON action envelope without consuming thinking/text lanes."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("cohere_action_envelope",)

    START = "<|START_ACTION|>"
    END = "<|END_ACTION|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_tool_count = 0
        self._content_emitted_len = 0

    def reset(self) -> None:
        super().reset()
        self._emitted_tool_count = 0
        self._content_emitted_len = 0

    @staticmethod
    def _tool_id(call: dict[str, Any]) -> str:
        supplied = call.get("tool_call_id")
        return (
            supplied
            if isinstance(supplied, str) and supplied
            else f"call_{uuid.uuid4().hex[:8]}"
        )

    @staticmethod
    def _arguments(call: dict[str, Any]) -> str:
        arguments = call.get("parameters", call.get("arguments", {}))
        if isinstance(arguments, str):
            return arguments
        return json.dumps(
            arguments if arguments is not None else {}, ensure_ascii=False
        )

    @classmethod
    def _parse_action(cls, body: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(body.replace("\\|", "|"))
        except json.JSONDecodeError:
            return []
        entries = [payload] if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            return []
        calls = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("tool_name") or entry.get("name") or entry.get("function")
            if not isinstance(name, str) or not name:
                continue
            calls.append(
                {
                    "id": cls._tool_id(entry),
                    "name": name,
                    "arguments": cls._arguments(entry),
                }
            )
        return calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.START not in model_output:
            # Some decode paths have already consumed the action control
            # tokens. Keep mlx-lm's bare-payload fallback, but only accept
            # entries that still carry Cohere's tool-name fields.
            calls = self._parse_action(model_output.strip())
            if calls:
                return ExtractedToolCallInformation(True, calls, None)

        calls: list[dict[str, Any]] = []
        remainder = model_output
        while self.START in remainder:
            prefix, _, after_start = remainder.partition(self.START)
            body, separator, after_end = after_start.partition(self.END)
            if not separator:
                break
            calls.extend(self._parse_action(body.strip()))
            remainder = prefix + after_end
        if not calls:
            return ExtractedToolCallInformation(False, [], model_output)
        content = remainder.strip()
        return ExtractedToolCallInformation(True, calls, content or None)

    def has_pending_tool_call(self, text: str) -> bool:
        return text.rfind(self.START) > text.rfind(
            self.END
        ) or self.has_text_format_tool_call(text)

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
        if self.START not in current_text:
            self._content_emitted_len = len(current_text)
            return {"content": delta_text}
        if current_text.count(self.END) <= previous_text.count(self.END):
            return None
        parsed = self.extract_tool_calls(current_text, request)
        if (
            not parsed.tools_called
            or len(parsed.tool_calls) <= self._emitted_tool_count
        ):
            return None
        start = self._emitted_tool_count
        new_calls = parsed.tool_calls[start:]
        self._emitted_tool_count = len(parsed.tool_calls)
        return {
            "tool_calls": [
                {
                    "index": start + index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for index, call in enumerate(new_calls)
            ]
        }

    def flush_held_content(self, full_text: str) -> str:
        return full_text[self._content_emitted_len :]
