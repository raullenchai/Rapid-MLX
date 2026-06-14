# SPDX-License-Identifier: Apache-2.0
"""
Auto-detecting tool call parser for rapid-mlx.

Automatically detects and parses tool calls from various model formats.
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
from .lfm_tool_parser import LFM_CALL_START, parse_lfm_tool_calls


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["auto", "generic"])
class AutoToolParser(ToolParser):
    """
    Auto-detecting tool call parser.

    Tries multiple formats in order:
    1. Mistral: [TOOL_CALLS] ...
    2. Qwen bracket: [Calling tool: func_name({...})]
    3. Qwen/Hermes XML: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    4. Llama: <function=name>{"arg": "value"}</function>
    5. Nemotron: <tool_call><function=name>...</function></tool_call>
    6. LFM pythonic: [func_name(arg="value")]
    7. Raw JSON: {"name": "...", "arguments": {...}}

    This is the default parser when no specific parser is selected.
    """

    # Patterns for different formats
    MISTRAL_TOKEN = "[TOOL_CALLS]"

    QWEN_BRACKET_PATTERN = re.compile(
        r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL
    )
    QWEN_XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    LLAMA_PATTERN = re.compile(r"<function=([^>]+)>(\{.*?\})</function>", re.DOTALL)
    NEMOTRON_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
    )
    NEMOTRON_PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
    )

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Guard against re-emitting the same tool calls when a later
        # delta contains another end marker (e.g. a ``]`` in trailing
        # prose) — re-running ``extract_tool_calls`` produces the same
        # call with a fresh id at the same index, which OpenAI-delta
        # clients concatenate into corrupt arguments.
        self._streaming_tools_emitted = False
        # Length of ``current_text`` already emitted as content. When a
        # marker appears, content is held (None) for the rest of the
        # stream; if no tool call ever completes, ``flush_held_content``
        # releases everything past this boundary so prose that merely
        # looked like markup isn't silently dropped.
        self._content_emitted_len = 0

    def reset(self) -> None:
        super().reset()
        self._streaming_tools_emitted = False
        self._content_emitted_len = 0

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls by trying all known formats.
        """
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        # 1. Try Mistral format
        if self.MISTRAL_TOKEN in model_output:
            parts = model_output.split(self.MISTRAL_TOKEN)
            content = parts[0].strip()
            raw_tool_calls = parts[1:]

            for raw in raw_tool_calls:
                raw = raw.strip()
                if not raw:
                    continue

                # New Mistral format: func_name{"args"}
                if not raw.startswith("[") and "{" in raw:
                    end_name = raw.find("{")
                    name = raw[:end_name].strip()
                    args = raw[end_name:]
                    if name:
                        tool_calls.append(
                            {"id": generate_tool_id(), "name": name, "arguments": args}
                        )
                    continue

                # Old Mistral format: [{"name": "...", "arguments": {...}}]
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                args = item.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_tool_id(),
                                        "name": item["name"],
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
                except json.JSONDecodeError:
                    pass

            if tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

        # 2. Try Qwen bracket pattern
        bracket_matches = self.QWEN_BRACKET_PATTERN.findall(model_output)
        for name, args_str in bracket_matches:
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
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": args_str,
                    }
                )

        if bracket_matches:
            cleaned_text = self.QWEN_BRACKET_PATTERN.sub("", cleaned_text).strip()

        # 3. Try Nemotron pattern (before Qwen XML as it's more specific)
        nemotron_matches = self.NEMOTRON_PATTERN.findall(cleaned_text)
        for name, params_block in nemotron_matches:
            params = self.NEMOTRON_PARAM_PATTERN.findall(params_block)
            arguments = {p_name.strip(): p_value.strip() for p_name, p_value in params}
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name.strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        if nemotron_matches:
            cleaned_text = self.NEMOTRON_PATTERN.sub("", cleaned_text).strip()

        # 4. Try Qwen/Hermes XML pattern
        xml_matches = self.QWEN_XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
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

        if xml_matches:
            cleaned_text = self.QWEN_XML_PATTERN.sub("", cleaned_text).strip()

        # 5. Try Llama pattern
        llama_matches = self.LLAMA_PATTERN.findall(cleaned_text)
        for name, args_str in llama_matches:
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
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": args_str,
                    }
                )

        if llama_matches:
            cleaned_text = self.LLAMA_PATTERN.sub("", cleaned_text).strip()

        # 6. Try LFM bracket pythonic calls format
        if not tool_calls and LFM_CALL_START.search(cleaned_text):
            lfm_tool_calls, lfm_cleaned = parse_lfm_tool_calls(cleaned_text)
            if lfm_tool_calls:
                tool_calls.extend(lfm_tool_calls)
                cleaned_text = lfm_cleaned

        # 7. Fallback: Try raw JSON
        if not tool_calls:
            raw_calls = self._parse_raw_json_tool_calls(cleaned_text)
            if raw_calls:
                tool_calls.extend(raw_calls)
                cleaned_text = ""

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

    def _parse_raw_json_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        Parse raw JSON tool calls from text.

        Handles:
        - Single JSON object: {"name": "func", "arguments": {...}}
        - JSON array: [{...}, {...}]
        """
        if not text:
            return []

        text = text.strip()
        tool_calls = []

        # Try JSON array first
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            # Support "name" and "type" fields (Granite)
                            func_name = item.get("name") or item.get("type")
                            if func_name:
                                args = item.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_tool_id(),
                                        "name": func_name,
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Find JSON objects with balanced braces
        depth = 0
        start = None

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    # Reset on unbalanced braces
                    depth = 0
                    start = None
                    continue
                if depth == 0 and start is not None:
                    json_str = text[start : i + 1]
                    try:
                        obj = json.loads(json_str)
                        if isinstance(obj, dict):
                            # Support both "name" and "type" fields
                            func_name = obj.get("name") or obj.get("type")
                            if func_name:
                                args = obj.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_tool_id(),
                                        "name": func_name,
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
                    except json.JSONDecodeError:
                        pass
                    start = None

        return tool_calls

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
        Extract tool calls from streaming model output.

        Uses simple heuristics to detect when a tool call might be complete.
        """
        # Check for any tool call markers
        markers = [
            self.MISTRAL_TOKEN,
            "[Calling tool:",
            "<tool_call>",
            "<function=",
        ]

        has_marker = any(m in current_text for m in markers) or bool(
            LFM_CALL_START.search(current_text)
        )

        if not has_marker:
            self._content_emitted_len = len(current_text)
            return {"content": delta_text}

        # Check for completion markers. A bare ``]`` (rather than ``)]``)
        # covers LFM pythonic calls whose closing ``)`` and ``]`` arrive
        # in separate deltas; extraction below only fires if a complete
        # call actually parses, so the looser trigger is safe.
        end_markers = ["</tool_call>", "</function>", "]"]
        if not self._streaming_tools_emitted and any(
            m in delta_text for m in end_markers
        ):
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                self._streaming_tools_emitted = True
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

    def flush_held_content(self, full_text: str) -> str:
        """Release content held behind a marker that never became a tool call.

        The postprocessor only calls this when no tool calls fired, so
        everything past the last emitted boundary is ordinary content.
        """
        return full_text[self._content_emitted_len :]
