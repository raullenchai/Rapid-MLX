# SPDX-License-Identifier: Apache-2.0
"""
Harmony tool call parser for GPT-OSS models.

Harmony uses control tokens and channels for tool calling:

    <|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location": "SF"}<|call|>

The final response is in the 'final' channel:

    <|start|>assistant<|channel|>final<|message|>The weather is 72F.<|end|>
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


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Tool call pattern — supports both formats from the harmony spec:
#   Model-generated: <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>ARGS<|call|>
#   Template-encoded (history): to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>
_COMMENTARY_BLOCK_PATTERN = re.compile(
    r"(?:"
    # Real format: to=functions.NAME<|channel|>commentary [content_type]<|message|>
    r"to=functions\.(\w+)<\|channel\|>commentary(?:\s+\w+)?<\|message\|>(.*?)<\|call\|>"
    r"|"
    # Legacy format: <|channel|>commentary to=functions.NAME ... <|message|>
    r"<\|channel\|>commentary\s+to=functions\.(\w+)(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>(.*?)<\|call\|>"
    r")",
    re.DOTALL,
)

# Final channel — both <|end|> and <|return|> terminators
_FINAL_BLOCK_PATTERN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)",
    re.DOTALL,
)


@ToolParserManager.register_module(["harmony", "gpt-oss"])
class HarmonyToolParser(ToolParser):
    """
    Tool call parser for GPT-OSS models using Harmony format.

    Harmony uses control tokens and 3 channels:
    - analysis: internal reasoning (handled by reasoning parser)
    - commentary: tool calls addressed with to=functions.{name}
    - final: user-facing response

    Used when --enable-auto-tool-choice --tool-call-parser harmony are set.
    """

    # GPT-OSS chat template natively handles tool_calls and role="tool"
    # messages using harmony channel tokens (to=functions.NAME, <|call|>).
    # Without this, tool history is converted to "[Calling tool: ...]" text
    # which breaks the model's understanding of the tool flow.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Harmony model response.

        Parses commentary channel blocks for tool calls and the final
        channel for the user-facing content.
        """
        tool_calls = []

        # Extract tool calls from commentary channel blocks
        # Regex has 4 groups: (1,2) for real format, (3,4) for legacy format
        for match in _COMMENTARY_BLOCK_PATTERN.finditer(model_output):
            tool_name = match.group(1) or match.group(3)
            args_str = (match.group(2) or match.group(4) or "").strip()

            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
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
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": args_str,
                    }
                )

        # Extract final channel content
        final_match = _FINAL_BLOCK_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        # No tool calls: return all text as content
        # If there's a final channel, use that; otherwise return the raw output
        # stripped of control tokens
        if content is None:
            content = _strip_control_tokens(model_output)

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content,
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
        Extract tool calls from streaming Harmony model output.

        Waits for <|call|> to complete a tool call, and emits final
        channel content as regular content deltas.
        """
        # If we see a tool call completion marker in the delta
        if "<|call|>" in delta_text:
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

        # If we're in the final channel, emit content token by token.
        # Track emitted length to only send new content each delta.
        if "<|channel|>final" in current_text:
            final_start = current_text.rfind("<|channel|>final")
            msg_start = current_text.find("<|message|>", final_start)
            if msg_start >= 0:
                raw = current_text[msg_start + len("<|message|>"):]
                # Strip control tokens from the extracted content
                clean = _strip_control_tokens(raw).strip()
                # Calculate what's new since previous extraction
                prev_final = previous_text.rfind("<|channel|>final")
                prev_clean = ""
                if prev_final >= 0:
                    prev_msg = previous_text.find("<|message|>", prev_final)
                    if prev_msg >= 0:
                        prev_raw = previous_text[prev_msg + len("<|message|>"):]
                        prev_clean = _strip_control_tokens(prev_raw).strip()
                new_content = clean[len(prev_clean):]
                if new_content:
                    return {"content": new_content}
            # In final channel but no new content yet (control token)
            return {"content": ""}

        # If no tool markers at all, pass through as content
        if "<|channel|>" not in current_text:
            return {"content": delta_text}

        # Building tool call or in analysis channel, suppress output
        return None

    def has_pending_tool_call(self, text: str) -> bool:
        """Check if text contains incomplete Harmony tool call markup."""
        return "to=functions." in text


def _strip_control_tokens(text: str) -> str:
    """Remove Harmony control tokens from text."""
    tokens = [
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|return|>",
        "<|call|>",
    ]
    result = text
    for token in tokens:
        result = result.replace(token, "")
    # Clean up channel names and constrain values
    result = re.sub(r"(?:analysis|commentary|final)\s*", "", result)
    result = re.sub(r"to=functions\.\w+\s*", "", result)
    result = re.sub(r"json\s*", "", result)
    return result.strip()


def _is_control_token(text: str) -> bool:
    """Check if text is a Harmony control token."""
    return text.strip() in {
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|return|>",
        "<|call|>",
    }
