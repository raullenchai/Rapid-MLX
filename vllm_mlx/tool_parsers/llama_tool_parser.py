# SPDX-License-Identifier: Apache-2.0
"""
Llama tool call parser for vllm-mlx.

Handles Llama's tool calling format:
- XML style: <function=name>{"arg": "value"}</function>
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


@ToolParserManager.register_module(["llama", "llama3", "llama4"])
class LlamaToolParser(ToolParser):
    """
    Tool call parser for Llama models.

    Supports Llama tool call format:
    - <function=name>{"arg": "value"}</function>

    Used when --enable-auto-tool-choice --tool-call-parser llama are set.
    """

    # Llama 3+ chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Pattern for Llama-style: <function=name>{"json"}</function>
    FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(\{.*?\})</function>", re.DOTALL)

    def has_pending_tool_call(self, text: str) -> bool:
        return "<function=" in text

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Llama model response.
        """
        tool_calls = []
        cleaned_text = model_output

        matches = self.FUNCTION_PATTERN.findall(model_output)
        for name, args_str in matches:
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
                # Keep the raw arguments string
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": args_str,
                    }
                )

        if matches:
            cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text).strip()

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
        Extract tool calls from streaming Llama model output.
        """
        # Check for tool call markers
        if "<function=" not in current_text:
            return {"content": delta_text}

        # If we detect end of function, parse
        if "</function>" in delta_text:
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
