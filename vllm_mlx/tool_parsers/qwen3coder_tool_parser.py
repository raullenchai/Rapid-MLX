# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Coder XML tool call parser for vllm-mlx.

Ported from vLLM upstream (vllm/tool_parsers/qwen3coder_tool_parser.py).

Format:
  <tool_call>
  <function=NAME>
  <parameter=KEY>VALUE</parameter>
  </function>
  </tool_call>

Similar to Seed-OSS but without the seed: namespace prefix.
"""

import ast
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _get_arguments_config(func_name: str, tools: list[dict] | None) -> dict:
    """Extract argument config from tools list for type conversion."""
    if tools is None:
        return {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", {})
        if func.get("name") == func_name:
            params = func.get("parameters", {})
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            elif isinstance(params, dict):
                return params
            return {}
    return {}


def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    """Convert parameter value based on its type in the schema."""
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        return param_value

    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        param_type = str(cfg["type"]).strip().lower()
    else:
        param_type = "string"

    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    elif param_type.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            return param_value
    elif param_type.startswith(("num", "float", "double")):
        try:
            return float(param_value)
        except (ValueError, TypeError):
            return param_value
    elif param_type in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    else:
        if param_type in ("object", "array", "arr") or param_type.startswith(
            ("dict", "list")
        ):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            return param_value


@ToolParserManager.register_module(["qwen3_coder_xml", "qwen3_xml"])
class Qwen3CoderToolParser(ToolParser):
    """
    Tool call parser for Qwen3-Coder models using XML format.

    Supports the XML-based tool call format with <tool_call>/<function=...>
    tags and type conversion from tool schema.

    Used when --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_prefix = "<function="
        self.function_end_token = "</function>"
        self.parameter_prefix = "<parameter="
        self.parameter_end_token = "</parameter>"

        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Token IDs for streaming (graceful fallback if tokenizer absent)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self._current_tool_id = None
        self.current_function_name = None
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self._streaming_request = None
        self.prev_tool_call_arr = []

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[dict] | None
    ) -> dict | None:
        """Parse a single function call from XML and return a tool call dict."""
        try:
            end_index = function_call_str.index(">")
        except ValueError:
            return None
        function_name = function_call_str[:end_index]
        param_config = _get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1:]
        param_dict = {}
        for match_text in self.tool_call_parameter_regex.findall(parameters):
            try:
                idx = match_text.index(">")
            except ValueError:
                continue
            p_name = match_text[:idx]
            p_value = str(match_text[idx + 1:])
            if p_value.startswith("\n"):
                p_value = p_value[1:]
            if p_value.endswith("\n"):
                p_value = p_value[:-1]
            param_dict[p_name] = _convert_param_value(
                p_value, p_name, param_config, function_name
            )
        return {
            "id": _generate_tool_id(),
            "name": function_name,
            "arguments": json.dumps(param_dict, ensure_ascii=False),
        }

    def _get_function_calls(self, model_output: str) -> list[str]:
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            m[0] if m[0] else m[1] for m in matched_ranges
        ]
        if not raw_tool_calls:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tc in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tc))
        return [m[0] if m[0] else m[1] for m in raw_function_calls]

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if not function_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tools = None
            if request and isinstance(request, dict):
                tools = request.get("tools")

            tool_calls = []
            for fc_str in function_calls:
                tc = self._parse_xml_function_call(fc_str, tools)
                if tc:
                    tool_calls.append(tc)

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
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
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_request = request

        if not delta_text:
            return None

        delta_token_ids = delta_token_ids or []
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                if self.current_tool_index >= current_text.count(
                    self.tool_call_start_token
                ):
                    self.is_tool_call_started = False
                return None

        # Handle content before tool calls
        if not self.is_tool_call_started:
            if (
                self.tool_call_start_token_id is not None
                and self.tool_call_start_token_id in delta_token_ids
            ) or self.tool_call_start_token in delta_text:
                self.is_tool_call_started = True
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[
                        : delta_text.index(self.tool_call_start_token)
                    ]
                    if content_before:
                        return {"content": content_before}
                # Fall through to header parsing below instead of returning
                # None — the function header may already be in current_text.
            else:
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    return None
                return {"content": delta_text}

        # Find current tool call portion
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            return None

        tool_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_start_positions):
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx: tool_end_idx + len(self.tool_call_end_token)
            ]

        # Parse function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    self.current_function_name = tool_text[func_start:func_end]
                    self._current_tool_id = _generate_tool_id()
                    self.header_sent = True
                    self.in_function = True

                    # If the function body is already complete, emit the full
                    # tool call in one chunk to prevent header-only output
                    # when coarse deltas or max_tokens truncation leave no
                    # further parser calls.
                    if self.function_end_token in tool_text:
                        tools = None
                        if request and isinstance(request, dict):
                            tools = request.get("tools")
                        fc = tool_text[func_start:tool_text.find(
                            self.function_end_token, func_start
                        )]
                        parsed = self._parse_xml_function_call(fc, tools)
                        args = parsed["arguments"] if parsed else "{}"
                        self.json_started = True
                        self.json_closed = True
                        self.in_function = False
                        self.accumulated_params = {}
                        self.prev_tool_call_arr.append(
                            {"name": self.current_function_name, "arguments": args}
                        )
                        return {
                            "tool_calls": [
                                {
                                    "index": self.current_tool_index,
                                    "id": self._current_tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": self.current_function_name,
                                        "arguments": args,
                                    },
                                }
                            ]
                        }

                    self.prev_tool_call_arr.append(
                        {"name": self.current_function_name, "arguments": "{}"}
                    )
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_index,
                                "id": self._current_tool_id,
                                "type": "function",
                                "function": {
                                    "name": self.current_function_name,
                                    "arguments": "",
                                },
                            }
                        ]
                    }
            return None

        # Handle function body
        if self.in_function:
            if not self.json_started:
                self.json_started = True
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": "{"},
                        }
                    ]
                }

            # Find all parameter start positions
            param_starts = []
            si = 0
            while True:
                si = tool_text.find(self.parameter_prefix, si)
                if si == -1:
                    break
                param_starts.append(si)
                si += len(self.parameter_prefix)

            # Process complete parameters
            json_fragments = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]
                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                param_end_idx = value_text.find(self.parameter_end_token)
                if param_end_idx == -1:
                    # Try next parameter or function end as delimiter
                    next_param = value_text.find(self.parameter_prefix)
                    func_end = value_text.find(self.function_end_token)
                    if next_param != -1 and (func_end == -1 or next_param < func_end):
                        param_end_idx = next_param
                    elif func_end != -1:
                        param_end_idx = func_end
                    else:
                        tool_end_in_val = value_text.find(self.tool_call_end_token)
                        if tool_end_in_val != -1:
                            param_end_idx = tool_end_in_val
                        else:
                            break

                if param_end_idx == -1:
                    break

                pv = value_text[:param_end_idx]
                if pv.endswith("\n"):
                    pv = pv[:-1]

                self.accumulated_params[current_param_name] = pv

                # Type conversion
                tools = None
                if self._streaming_request:
                    tools = self._streaming_request.get("tools") if isinstance(
                        self._streaming_request, dict
                    ) else None
                param_config = _get_arguments_config(
                    self.current_function_name or "", tools
                )
                converted = _convert_param_value(
                    pv, current_param_name, param_config, self.current_function_name or ""
                )
                serialized = json.dumps(converted, ensure_ascii=False)

                if self.param_count == 0:
                    frag = f'"{current_param_name}": {serialized}'
                else:
                    frag = f', "{current_param_name}": {serialized}'
                self.param_count += 1
                json_fragments.append(frag)

            if json_fragments:
                combined = "".join(json_fragments)
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": combined},
                        }
                    ]
                }

            # Check for function end
            if not self.json_closed and self.function_end_token in tool_text:
                self.json_closed = True

                # Update prev_tool_call_arr with final arguments
                tools = None
                if self._streaming_request:
                    tools = self._streaming_request.get("tools") if isinstance(
                        self._streaming_request, dict
                    ) else None
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    fc = tool_text[func_start:func_content_end]
                    try:
                        parsed = self._parse_xml_function_call(fc, tools)
                        if parsed and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = parsed["arguments"]
                    except Exception:
                        pass

                self.in_function = False
                self.accumulated_params = {}
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": "}"},
                        }
                    ]
                }

        return None
