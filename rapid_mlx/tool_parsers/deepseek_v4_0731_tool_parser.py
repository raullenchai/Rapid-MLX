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

_ESCAPED_CONTROL_LITERAL = re.compile(
    r"<(?P<esc>\u200b+)(?=(?:/?(?:think|reasoning)>|/?｜DSML｜))"
)


def _restore_string_parameter(value: str) -> str:
    """Decode DSML's U+200B escape for literal control-looking text."""

    def _restore(match: re.Match) -> str:
        original_count = len(match.group("esc")) // 2
        return "<" + ("\u200b" * original_count)

    return _ESCAPED_CONTROL_LITERAL.sub(_restore, value)


def _restore_argument_value(value: Any) -> Any:
    """Recursively reverse framing in string leaves and JSON object keys."""
    if isinstance(value, str):
        return _restore_string_parameter(value)
    if isinstance(value, list):
        return [_restore_argument_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _restore_string_parameter(key) if isinstance(key, str) else key: (
                _restore_argument_value(item)
            )
            for key, item in value.items()
        }
    return value


@ToolParserManager.register_module(["deepseek_v4_0731"])
class DeepSeekV40731ToolParser(ToolParser):
    EXPECTED_WIRE_FORMATS = ("deepseek_v4_dsml",)
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    START = "<｜DSML｜tool_calls>"
    START_R = "<｜DSML｜r:tool_calls>"
    END = "</｜DSML｜tool_calls>"
    INVOKE = re.compile(
        r'<｜DSML｜(?:r:)?invoke\s+name="(?P<name>[^"]+)">'
        r"(?P<body>.*?)</｜DSML｜(?:r:)?invoke>",
        re.DOTALL,
    )
    PARAM = re.compile(
        r'<｜DSML｜(?:r:)?parameter\s+name="(?P<name>[^"]+)"\s+'
        r'string="(?P<string>true|false)">(?P<value>.*?)'
        r"</｜DSML｜(?:r:)?parameter>",
        re.DOTALL,
    )

    @classmethod
    def _first_start(cls, text: str) -> tuple[int, str] | None:
        matches = ((text.find(tag), tag) for tag in (cls.START, cls.START_R))
        present = [(index, tag) for index, tag in matches if index >= 0]
        return min(present, default=None, key=lambda match: match[0])

    def reset(self) -> None:
        super().reset()
        self._stream_calls_emitted = False

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Hold any suffix that could grow into the DSML opener."""
        start = cls._first_start(text)
        if start is not None:
            return text[: start[0]]
        for tag in (cls.START, cls.START_R):
            max_prefix = min(len(text), len(tag) - 1)
            for size in range(max_prefix, 0, -1):
                if tag.startswith(text[-size:]):
                    return text[:-size]
        return text

    def has_pending_tool_call(self, text: str) -> bool:
        return (
            self._first_start(text) is not None
            or self._safe_content_prefix(text) != text
        )

    def flush_held_content(self, full_text: str) -> str:
        safe = self._safe_content_prefix(full_text)
        return full_text[len(safe) :] if self._first_start(full_text) is None else ""

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ):
        start = self._first_start(model_output)
        if start is None:
            return ExtractedToolCallInformation(False, [], model_output)
        content = model_output[: start[0]].strip() or None
        calls = []
        for match in self.INVOKE.finditer(model_output):
            arguments: dict[str, Any] = {}
            for param in self.PARAM.finditer(match.group("body")):
                raw = param.group("value")
                if param.group("string") == "true":
                    value: Any = _restore_argument_value(raw)
                else:
                    try:
                        value = _restore_argument_value(json.loads(raw))
                    except json.JSONDecodeError:
                        value = _restore_argument_value(raw)
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
            # Sampled 0731 output occasionally renders an omitted optional
            # enum as JSON null or an empty collection. Omission is the only
            # schema-valid representation; preserve non-empty values so the
            # normal strict validator remains authoritative.
            if (
                match.group("name") == "exec_command"
                and "sandbox_permissions" in arguments
                and arguments.get("sandbox_permissions") in (None, [], {})
            ):
                arguments.pop("sandbox_permissions")
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
        if self._first_start(current_text) is None:
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
