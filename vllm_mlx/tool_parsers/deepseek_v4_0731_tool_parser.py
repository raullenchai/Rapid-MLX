# SPDX-License-Identifier: MIT
"""Tool-call parser for DeepSeek-V4-Flash-0731's DSML format."""

from __future__ import annotations

import json
import re
import shlex
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


@ToolParserManager.register_module(["deepseek_v4_0731"])
class DeepSeekV40731ToolParser(ToolParser):
    EXPECTED_WIRE_FORMATS = ("deepseek_v4_dsml",)
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    START = "<｜DSML｜tool_calls>"
    END = "</｜DSML｜tool_calls>"
    INVOKE = re.compile(
        r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)">(?P<body>.*?)</｜DSML｜invoke>',
        re.DOTALL,
    )
    PARAM = re.compile(
        r'<｜DSML｜parameter\s+name="(?P<name>[^"]+)"\s+string="(?P<string>true|false)">(?P<value>.*?)</｜DSML｜parameter>',
        re.DOTALL,
    )

    def reset(self) -> None:
        super().reset()
        self._stream_calls_emitted = False

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Hold any suffix that could grow into the DSML opener."""
        start = text.find(cls.START)
        if start >= 0:
            return text[:start]
        max_prefix = min(len(text), len(cls.START) - 1)
        for size in range(max_prefix, 0, -1):
            if cls.START.startswith(text[-size:]):
                return text[:-size]
        return text

    def has_pending_tool_call(self, text: str) -> bool:
        return self.START in text or self._safe_content_prefix(text) != text

    def flush_held_content(self, full_text: str) -> str:
        safe = self._safe_content_prefix(full_text)
        return full_text[len(safe) :] if self.START not in full_text else ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ):
        if self.START not in model_output:
            return ExtractedToolCallInformation(False, [], model_output)
        content = model_output.split(self.START, 1)[0].strip() or None
        calls = []
        for match in self.INVOKE.finditer(model_output):
            arguments: dict[str, Any] = {}
            for param in self.PARAM.finditer(match.group("body")):
                raw = param.group("value")
                if param.group("string") == "true":
                    value: Any = raw
                else:
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        value = raw
                arguments[param.group("name")] = value
            # DeepSeek occasionally serializes Codex's reusable approval
            # prefix as a shell-like scalar even though it is an argv prefix.
            # Preserve argument boundaries while normalizing it for strict
            # schema validation. Leave malformed quoting unchanged so the
            # validator rejects it instead of silently changing its meaning.
            if match.group("name") == "exec_command" and isinstance(
                arguments.get("prefix_rule"), str
            ):
                try:
                    prefix_rule = shlex.split(arguments["prefix_rule"])
                    if prefix_rule:
                        arguments["prefix_rule"] = prefix_rule
                except ValueError:
                    pass
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": match.group("name"),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )
        return ExtractedToolCallInformation(
            bool(calls), calls, content if calls else model_output
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
        if not hasattr(self, "_stream_calls_emitted"):
            self.reset()
        if self._stream_calls_emitted:
            return None
        if self.START not in current_text:
            previous_safe = self._safe_content_prefix(previous_text)
            current_safe = self._safe_content_prefix(current_text)
            newly_safe = current_safe[len(previous_safe) :]
            return {"content": newly_safe} if newly_safe else None
        if self.END not in current_text:
            return None
        result = self.extract_tool_calls(current_text, request)
        if not result.tools_called:
            return None
        self._stream_calls_emitted = True
        return {
            "tool_calls": [
                {
                    "index": i,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for i, call in enumerate(result.tool_calls)
            ]
        }
