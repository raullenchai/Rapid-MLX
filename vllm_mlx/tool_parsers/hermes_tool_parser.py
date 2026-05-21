# SPDX-License-Identifier: Apache-2.0
"""
Hermes/Nous tool call parser for vllm-mlx.

Handles Hermes-style tool calling format used by NousResearch models.
"""

import ast
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


def _parse_param_value(val: str) -> Any:
    """Parse a tool call parameter value, handling both JSON and Python literals.

    Tries json.loads first. If that fails, falls back to ast.literal_eval
    for Python literal syntax (single quotes, True/False, None). Converts
    sets to lists and rejects types that are not JSON-serializable (complex,
    bytes) to avoid crashes during json.dumps later.
    """
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        python_val = ast.literal_eval(val)
        if isinstance(python_val, set):
            python_val = sorted(python_val, key=str)
        if isinstance(python_val, (complex, bytes)):
            return val
        json.dumps(python_val)
        return python_val
    except (ValueError, SyntaxError, TypeError):
        return val


@ToolParserManager.register_module(["hermes", "nous", "qwen3_coder"])
class HermesToolParser(ToolParser):
    """
    Tool call parser for Hermes/Nous models.

    Supports Hermes tool call format:
    - <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Sometimes with additional reasoning in <tool_call_reasoning>
    - Fallback: raw JSON {"name": "func", "arguments": {...}} (for models that omit tags)

    Used when --enable-auto-tool-choice --tool-call-parser hermes are set.
    """

    # Qwen3 / Hermes chat templates handle role="tool" and tool_calls natively.
    # Without this, tool history is converted to "[Calling tool: ...]" text,
    # which causes the model to mimic that text format instead of producing
    # proper <tool_call> XML after a few rounds of tool use.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Hermes is the most flexible parser — handles JSON body inside
    # <tool_call>, the Nemotron-style XML body fallback (covers vanilla
    # Qwen3.6), bare <function=...> blocks, raw JSON tool calls, and the
    # [Calling tool:] text-fallback for low-quant degradation.
    EXPECTED_WIRE_FORMATS = (
        "tool_call_json",
        "tool_call_xml_body",
        "function_bare",
        "raw_json",
        "calling_tool_text",
    )

    # Standard format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # Lenient format: <tool_call or <tool_call> followed by JSON (handles malformed tags)
    TOOL_CALL_LENIENT_PATTERN = re.compile(
        r'<tool_call[^{]*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\})', re.DOTALL
    )
    # Nemotron XML: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    NEMOTRON_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
    )
    PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    REASONING_PATTERN = re.compile(
        r"<tool_call_reasoning>(.*?)</tool_call_reasoning>", re.DOTALL
    )
    # Fallback pattern for raw JSON tool calls (without tags)
    RAW_JSON_TOOL_PATTERN = re.compile(
        r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}', re.DOTALL
    )
    # Bare Nemotron XML: <function=name>...</function> without <tool_call> wrapper
    BARE_FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Hermes model response.
        """
        tool_calls = []
        cleaned_text = model_output

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(cleaned_text)

        # Remove reasoning tags first (keep for content)
        reasoning_matches = self.REASONING_PATTERN.findall(cleaned_text)
        cleaned_text = self.REASONING_PATTERN.sub("", cleaned_text)

        # Parse tool calls with <tool_call> tags (primary format)
        matches = self.TOOL_CALL_PATTERN.findall(cleaned_text)
        for match in matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        # Try Nemotron XML format if no JSON tool calls found
        if not tool_calls:
            nemotron_matches = self.NEMOTRON_PATTERN.findall(cleaned_text)
            for name, params_block in nemotron_matches:
                params = self.PARAM_PATTERN.findall(params_block)
                arguments = {}
                for p_name, p_value in params:
                    arguments[p_name.strip()] = _parse_param_value(p_value.strip())
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            if nemotron_matches:
                cleaned_text = self.NEMOTRON_PATTERN.sub("", cleaned_text).strip()

        # Try bare Nemotron XML: <function=name>...</function> without <tool_call> wrapper
        # This happens when the chat template provides <tool_call> as generation prompt
        # and the model generates <function=...> directly.
        if not tool_calls:
            bare_matches = self.BARE_FUNCTION_PATTERN.findall(cleaned_text)
            for name, params_block in bare_matches:
                params = self.PARAM_PATTERN.findall(params_block)
                arguments = {}
                for p_name, p_value in params:
                    arguments[p_name.strip()] = _parse_param_value(p_value.strip())
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            if bare_matches:
                cleaned_text = self.BARE_FUNCTION_PATTERN.sub("", cleaned_text).strip()

        # Fallback: try lenient pattern for malformed tags like <tool_call without >
        if not tool_calls:
            lenient_matches = self.TOOL_CALL_LENIENT_PATTERN.findall(cleaned_text)
            for match in lenient_matches[:1]:  # Only first to avoid hallucinations
                try:
                    data = json.loads(match)
                    name = data.get("name", "")
                    arguments = data.get("arguments", {})
                    if name:
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": name,
                                "arguments": (
                                    json.dumps(arguments, ensure_ascii=False)
                                    if isinstance(arguments, dict)
                                    else str(arguments)
                                ),
                            }
                        )
                        cleaned_text = self.TOOL_CALL_LENIENT_PATTERN.sub(
                            "", cleaned_text, count=1
                        ).strip()
                except json.JSONDecodeError:
                    continue

        # Fallback: try raw JSON format if no tagged tool calls found
        # Only parse the FIRST valid tool call to avoid hallucinated multiple calls
        if not tool_calls:
            raw_matches = self.RAW_JSON_TOOL_PATTERN.findall(cleaned_text)
            if raw_matches:
                name, args_str = raw_matches[0]
                try:
                    arguments = json.loads(args_str)
                    valid_tool = True
                    if request and "tools" in request:
                        tool_names = [
                            t.get("function", {}).get("name", "")
                            for t in request.get("tools", [])
                            if isinstance(t, dict)
                        ]
                        valid_tool = name in tool_names

                    if valid_tool and name:
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            }
                        )
                        cleaned_text = self.RAW_JSON_TOOL_PATTERN.sub(
                            "", cleaned_text, count=1
                        ).strip()
                except json.JSONDecodeError:
                    pass

        # Include reasoning in content if present
        if reasoning_matches:
            reasoning_text = " ".join(reasoning_matches)
            if cleaned_text:
                cleaned_text = f"{cleaned_text}\n\n(Reasoning: {reasoning_text})"
            else:
                cleaned_text = f"(Reasoning: {reasoning_text})"

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

    @staticmethod
    def _format_streaming_tool_calls(
        tool_calls: list[dict], start_index: int = 0
    ) -> dict[str, Any]:
        """Format tool calls for streaming response."""
        return {
            "tool_calls": [
                {
                    "index": start_index + i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
        }

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
        Extract tool calls from streaming Hermes model output.

        Uses tag counting to correctly handle multiple sequential tool calls.
        """
        # Count <tool_call> / </tool_call> tags for multi-tool support
        open_count = current_text.count("<tool_call>")
        close_count = current_text.count("</tool_call>")
        prev_close_count = previous_text.count("</tool_call>")

        if open_count > 0:
            if open_count > close_count:
                # Inside an incomplete tool call block, suppress output
                return None

            if close_count > prev_close_count:
                # New tool call(s) completed in this delta
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    # Only emit newly completed tool calls (skip already emitted)
                    new_calls = result.tool_calls[prev_close_count:]
                    if new_calls:
                        return self._format_streaming_tool_calls(
                            new_calls, start_index=prev_close_count
                        )

            # All current tool calls already emitted, pass content through
            return {"content": delta_text}

        # Bare Nemotron XML: <function=name>...</function> without <tool_call> wrapper
        # This happens when the chat template provides <tool_call> as generation prompt.
        if "<function=" in current_text:
            func_close_count = current_text.count("</function>")
            prev_func_close = previous_text.count("</function>")

            if current_text.count("<function=") > func_close_count:
                # Inside an incomplete function block, suppress output
                return None

            if func_close_count > prev_func_close:
                # New function block(s) completed
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    new_calls = result.tool_calls[prev_func_close:]
                    if new_calls:
                        return self._format_streaming_tool_calls(
                            new_calls, start_index=prev_func_close
                        )

            return {"content": delta_text}

        # Fallback: check for raw JSON tool calls (detect closing brace pattern)
        if '{"name":' in current_text and '"arguments":' in current_text:
            if delta_text.rstrip().endswith("}"):
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    return self._format_streaming_tool_calls(result.tool_calls)
            return None

        return {"content": delta_text}
