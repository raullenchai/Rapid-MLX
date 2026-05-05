# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek tool call parser for vllm-mlx.

Handles DeepSeek V3 and R1 tool calling formats:
- <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> wrapper
- <｜tool▁call▁begin｜>function<｜tool▁sep｜>name
  ```json
  {...}
  ```<｜tool▁call▁end｜>
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


@ToolParserManager.register_module(["deepseek", "deepseek_v3", "deepseek_r1"])
class DeepSeekToolParser(ToolParser):
    """
    Tool call parser for DeepSeek V3 and R1 models.

    Supports DeepSeek's tool call format with special unicode tokens:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
    ```json
    {"city": "Paris"}
    ```<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

    Used when --enable-auto-tool-choice --tool-call-parser deepseek are set.
    """

    # DeepSeek V3 chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Special DeepSeek tokens (unicode)
    TOOL_CALLS_START = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_START = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"
    DSML_TOOL_CALLS_START = "<｜DSML｜tool_calls>"
    DSML_TOOL_CALLS_END = "</｜DSML｜tool_calls>"

    # Pattern to match individual tool calls
    TOOL_CALL_PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    # Alternative pattern without type
    TOOL_CALL_SIMPLE_PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<name>.*?)\n```json\n(?P<args>.*?)\n```<｜tool▁call▁end｜>",
        re.DOTALL,
    )
    DSML_BLOCK_PATTERN = re.compile(
        r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>", re.DOTALL
    )
    DSML_INVOKE_PATTERN = re.compile(
        r'<｜DSML｜invoke\s+name="([^"]+)">(.*?)</｜DSML｜invoke>',
        re.DOTALL,
    )
    DSML_PARAMETER_PATTERN = re.compile(
        r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</｜DSML｜parameter>',
        re.DOTALL,
    )

    def has_pending_tool_call(self, text: str) -> bool:
        return (
            self.TOOL_CALLS_START in text
            or self.DSML_TOOL_CALLS_START in text
            or self.has_text_format_tool_call(text)
        )

    def _extract_dsml_tool_calls(self, model_output: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        blocks = self.DSML_BLOCK_PATTERN.findall(model_output)
        for block in blocks:
            for func_name, params_block in self.DSML_INVOKE_PATTERN.findall(block):
                arguments: dict[str, Any] = {}
                for p_name, is_string, p_value in self.DSML_PARAMETER_PATTERN.findall(
                    params_block
                ):
                    value = p_value.strip()
                    if is_string == "true":
                        arguments[p_name] = value
                        continue
                    try:
                        arguments[p_name] = json.loads(value)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        arguments[p_name] = value
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
        return tool_calls

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from DeepSeek model output.
        """
        if self.DSML_TOOL_CALLS_START in model_output:
            tool_calls = self._extract_dsml_tool_calls(model_output)
            if tool_calls:
                content = self.DSML_BLOCK_PATTERN.sub("", model_output).strip()
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

        # Check for tool calls marker
        if self.TOOL_CALLS_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []

        # Extract content before tool calls
        content_end = model_output.find(self.TOOL_CALLS_START)
        content = model_output[:content_end].strip() if content_end > 0 else None

        # Try full pattern with type first
        matches = self.TOOL_CALL_PATTERN.findall(model_output)
        for match in matches:
            tool_type, func_name, func_args = match
            try:
                # Validate JSON
                json.loads(func_args)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )
            except json.JSONDecodeError:
                # Keep raw arguments
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )

        # Try simple pattern if no matches
        if not tool_calls:
            simple_matches = self.TOOL_CALL_SIMPLE_PATTERN.findall(model_output)
            for match in simple_matches:
                func_name, func_args = match
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
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
        Extract tool calls from streaming DeepSeek model output.
        """
        if self.DSML_TOOL_CALLS_START in current_text:
            if current_text.count(self.DSML_TOOL_CALLS_END) > previous_text.count(
                self.DSML_TOOL_CALLS_END
            ):
                result = self.extract_tool_calls(current_text)
                if result.tools_called:
                    prev_complete = previous_text.count(self.DSML_TOOL_CALLS_END)
                    new_calls = result.tool_calls[prev_complete:]
                    return {
                        "tool_calls": [
                            {
                                "index": prev_complete + i,
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

        if self.TOOL_CALLS_START not in current_text:
            return {"content": delta_text}

        # If we see the end marker, parse the complete output
        if self.TOOL_CALL_END in delta_text or self.TOOL_CALLS_END in delta_text:
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
