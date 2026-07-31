# SPDX-License-Identifier: MIT
"""Tool-call parser for DeepSeek-V4-Flash-0731's DSML format."""

from __future__ import annotations

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
        if self.START not in current_text:
            return {"content": delta_text}
        if self.END not in current_text:
            return None
        result = self.extract_tool_calls(current_text, request)
        if not result.tools_called:
            return None
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
