# SPDX-License-Identifier: Apache-2.0
"""
MiniMax tool call parser for vllm-mlx.

Parses the MiniMax-M2 native XML tool call format:
<minimax:tool_call>
<invoke name="tool-name">
<parameter name="param-key">param-value</parameter>
</invoke>
</minimax:tool_call>
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
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["minimax", "minimax_m2"])
class MiniMaxToolParser(ToolParser):
    """
    Parser for MiniMax-M2 tool call format.

    Format:
        <minimax:tool_call>
        <invoke name="func_name">
        <parameter name="key">value</parameter>
        </invoke>
        </minimax:tool_call>
    """

    TOOL_CALL_BLOCK = re.compile(
        r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
    )
    INVOKE_PATTERN = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
    PARAM_PATTERN = re.compile(
        r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL
    )
    THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

    def _extract_invokes(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from invoke elements, with or without wrapper."""
        tool_calls: list[dict[str, Any]] = []
        invokes = self.INVOKE_PATTERN.findall(text)
        for func_name, params_block in invokes:
            params = self.PARAM_PATTERN.findall(params_block)
            # Skip bare <invoke> tags without parameters (hallucinated junk)
            if not params:
                continue
            arguments = {}
            for p_name, p_value in params:
                p_value = p_value.strip()
                try:
                    arguments[p_name] = json.loads(p_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[p_name] = p_value

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
        # Try wrapped format first: <minimax:tool_call>...<invoke>...</minimax:tool_call>
        blocks = self.TOOL_CALL_BLOCK.findall(model_output)
        if blocks:
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                tool_calls.extend(self._extract_invokes(block))

            cleaned = self.TOOL_CALL_BLOCK.sub("", model_output).strip()
            cleaned = self.THINK_PATTERN.sub("", cleaned).strip()
            cleaned = re.sub(r"\[e~\[.*$", "", cleaned).strip()

            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=cleaned if cleaned else None,
            )

        # Fallback: bare <invoke> without <minimax:tool_call> wrapper
        # (model sometimes emits tool calls inside <think> without wrapper)
        tool_calls = self._extract_invokes(model_output)
        if tool_calls:
            # Strip matched invoke blocks and thinking from content
            cleaned = self.INVOKE_PATTERN.sub("", model_output).strip()
            cleaned = self.THINK_PATTERN.sub("", cleaned).strip()
            cleaned = re.sub(r"\[e~\[.*$", "", cleaned).strip()
            # Remove leftover closing tags
            cleaned = cleaned.replace("</invoke>", "").strip()

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned if cleaned else None,
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def _has_tool_start(self, text: str) -> bool:
        """Check if text contains the start of a tool call block."""
        return "<minimax:tool_call>" in text or (
            '<invoke name="' in text and self.INVOKE_PATTERN.search(text) is not None
        )

    def _has_tool_end(self, current: str, previous: str) -> bool:
        """Check if a tool call block just completed."""
        # If wrapped format is used, only trigger on the wrapper closing tag
        if "<minimax:tool_call>" in current:
            return (
                "</minimax:tool_call>" in current
                and "</minimax:tool_call>" not in previous
            )
        # Bare invoke: </invoke> just appeared
        if "</invoke>" in current and "</invoke>" not in previous:
            return True
        return False

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
        # Not inside a tool call block yet — pass content through
        if not self._has_tool_start(current_text):
            return {"content": delta_text}

        # Tool call block just completed
        if self._has_tool_end(current_text, previous_text):
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

        # Inside tool call block but not yet complete — suppress output
        return None
