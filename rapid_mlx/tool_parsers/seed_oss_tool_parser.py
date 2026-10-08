# SPDX-License-Identifier: Apache-2.0
"""
Seed-OSS tool call parser for rapid-mlx.

Ported from vLLM upstream (vllm/tool_parsers/seed_oss_tool_parser.py).

Format:
  <seed:tool_call>
  <function=NAME>
  <parameter=KEY>VALUE</parameter>
  </function>
  </seed:tool_call>

Thinking:
  <seed:think>...</seed:think>
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


class _ObjectPairs(list):
    """Keep repeated JSON object keys distinct from array elements."""


def _restore_json_value(value):
    if isinstance(value, _ObjectPairs):
        return {key: _restore_json_value(item) for key, item in value}
    if isinstance(value, list):
        return [_restore_json_value(item) for item in value]
    return value


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
        if param_type == "object" or param_type.startswith("dict"):
            try:
                return json.loads(param_value)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            return param_value


@ToolParserManager.register_module(["seed_oss", "seed", "gpt_oss"])
class SeedOssToolParser(ToolParser):
    """
    Tool call parser for Seed-OSS / GPT-OSS models.

    Supports the XML-based tool call format with <seed:tool_call> wrapper
    and <seed:think> thinking blocks.

    Used when --enable-auto-tool-choice --tool-call-parser seed_oss are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("seed_oss_native",)

    TOOL_CALL_START = "<seed:tool_call>"
    TOOL_CALL_END = "</seed:tool_call>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.tool_call_start_token = self.TOOL_CALL_START
        self.tool_call_end_token = self.TOOL_CALL_END
        self.tool_call_prefix = "<function="
        self.function_end_token = "</function>"
        self.parameter_prefix = "<parameter="
        self.parameter_end_token = "</parameter>"
        self.think_start_token = "<seed:think>"
        self.think_end_token = "</seed:think>"

        tool_start_re = re.escape(self.tool_call_start_token)
        tool_end_re = re.escape(self.tool_call_end_token)

        self.tool_call_complete_regex = re.compile(
            rf"{tool_start_re}(.*?){tool_end_re}", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            rf"{tool_start_re}(.*?){tool_end_re}|{tool_start_re}(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

        # Token IDs for streaming (graceful fallback if tokenizer absent)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)

        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.is_thinking_end = False
        self.header_sent = False
        self.current_tool_id = -1
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
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
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        param_pairs = []
        for match in self.tool_call_parameter_regex.findall(parameters):
            match_text = match[0] if match[0] else match[1]
            try:
                idx = match_text.index(">")
            except ValueError:
                continue
            p_name = match_text[:idx]
            p_value = str(match_text[idx + 1 :])
            if p_value.startswith("\n"):
                p_value = p_value[1:]
            if p_value.endswith("\n"):
                p_value = p_value[:-1]
            converted = _convert_param_value(
                p_value, p_name, param_config, function_name
            )
            param_dict[p_name] = converted
            param_pairs.append((p_name, converted))
        arguments = (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                for key, value in param_pairs
            )
            + "}"
            if len(param_pairs) != len(param_dict)
            else json.dumps(param_dict, ensure_ascii=False)
        )
        return {
            "id": _generate_tool_id(),
            "name": function_name,
            "arguments": arguments,
        }

    def _wrapper_start_positions(self, text: str) -> list[int]:
        """Find outer wrappers, ignoring marker text inside a parameter."""
        starts = []
        cursor = 0
        while (start := text.find(self.tool_call_start_token, cursor)) >= 0:
            cursor = start + len(self.tool_call_start_token)
            param_start = text.rfind(self.parameter_prefix, 0, start)
            param_end = text.rfind(self.parameter_end_token, 0, start)
            if param_start > param_end:
                function_end = text.rfind(self.function_end_token, param_start, start)
                next_body = text[cursor:].lstrip()
                if function_end < 0 or not next_body.startswith(self.tool_call_prefix):
                    continue
            starts.append(start)
        return starts

    def _get_function_calls(self, model_output: str) -> list[str]:
        starts = self._wrapper_start_positions(model_output)
        raw_tool_calls = []
        for index, start in enumerate(starts):
            limit = starts[index + 1] if index + 1 < len(starts) else len(model_output)
            end = model_output.rfind(self.tool_call_end_token, start, limit)
            if end >= 0 and self.function_end_token in model_output[end:limit]:
                end = limit
            raw_tool_calls.append(
                model_output[
                    start + len(self.tool_call_start_token) : end if end >= 0 else limit
                ]
            )
        if not starts:
            raw_tool_calls = [model_output]

        function_calls = []
        for tc in raw_tool_calls:
            for start, close in self._function_spans(tc):
                body_start = start + len(self.tool_call_prefix)
                function_calls.append(tc[body_start : close if close >= 0 else len(tc)])
        return function_calls

    def _function_spans(self, text: str) -> list[tuple[int, int]]:
        """Locate function closes outside parameter values."""
        spans = []
        cursor = 0
        while (start := text.find(self.tool_call_prefix, cursor)) >= 0:
            header_end = text.find(">", start + len(self.tool_call_prefix))
            if header_end < 0:
                break
            scan = header_end + 1
            close = -1
            while True:
                param_start = text.find(self.parameter_prefix, scan)
                function_end = text.find(self.function_end_token, scan)
                if function_end < 0:
                    break
                if param_start >= 0 and param_start < function_end:
                    param_end = text.find(self.parameter_end_token, param_start)
                    if param_end < 0:
                        break
                    next_function = text.find(
                        self.tool_call_prefix,
                        function_end + len(self.function_end_token),
                    )
                    if 0 <= next_function < param_end:
                        close = function_end
                        break
                    scan = param_end + len(self.parameter_end_token)
                    continue
                close = function_end
                break
            spans.append((start, close))
            cursor = close + len(self.function_end_token) if close >= 0 else len(text)
        return spans

    def _function_close_positions(self, text: str) -> list[int]:
        return [close for _, close in self._function_spans(text) if close >= 0]

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Handle <seed:think>...</seed:think>
        if (
            self.think_start_token in model_output
            and self.think_end_token in model_output
        ):
            think_end_index = model_output.find(self.think_end_token) + len(
                self.think_end_token
            )
            result_content = model_output[think_end_index:]
            thinking_content = model_output[:think_end_index]
        else:
            thinking_content = ""
            result_content = model_output

        try:
            function_calls = self._get_function_calls(result_content)
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
            tc_start = result_content.find(self.tool_call_start_token)
            if tc_start < 0:
                tc_start = result_content.find(self.tool_call_prefix)
            content = thinking_content + result_content[:tc_start]

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

    def has_pending_tool_call(self, text: str) -> bool:
        return (
            self.TOOL_CALL_START in text
            or self.tool_call_prefix in text
            or self.has_text_format_tool_call(text)
            or self._safe_content_prefix(text) != text
        )

    def _safe_content_prefix(self, text: str) -> str:
        marker = self.tool_call_start_token
        hold = max(
            (n for n in range(1, len(marker)) if text.endswith(marker[:n])),
            default=0,
        )
        return text[:-hold] if hold else text

    def flush_held_content(self, full_text: str) -> str:
        if self.tool_call_start_token in full_text:
            return ""
        return full_text[len(self._safe_content_prefix(full_text)) :]

    def finalize_legacy_raw_stream(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Finish an active call and append calls missed after malformed XML."""
        complete = self.extract_tool_calls(model_output, request)
        already = len(self.prev_tool_call_arr)
        if not complete.tools_called or len(complete.tool_calls) <= already:
            return None
        fresh = complete.tool_calls[already:]
        calls = []
        if self.in_function and self.header_sent:
            current = fresh.pop(0)
            if self.json_started:
                pairs = json.loads(current["arguments"], object_pairs_hook=_ObjectPairs)
                remaining = pairs[self.param_count :]
                prefix = ", " if self.param_count and remaining else ""
                suffix = (
                    prefix
                    + ", ".join(
                        f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(_restore_json_value(value), ensure_ascii=False)}"
                        for key, value in remaining
                    )
                    + "}"
                )
            else:
                suffix = current["arguments"]
            calls.append({"index": already, "function": {"arguments": suffix}})
            self.prev_tool_call_arr.append(current)
            already += 1
        calls.extend(
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for index, call in enumerate(fresh, start=already)
        )
        self.prev_tool_call_arr.extend(fresh)
        self.in_function = False
        self.json_closed = True
        return {"tool_calls": calls}

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

        # If an in-flight call and another complete call arrive together,
        # finish the active call first, then feed the remainder through the
        # completed-call reconciliation below. Otherwise the single-call
        # state machine returns after the first function close.
        previous_closes = len(self._function_close_positions(previous_text))
        close_positions = self._function_close_positions(current_text)
        current_closes = len(close_positions)
        if self.in_function and current_closes - previous_closes > 1:
            split = close_positions[previous_closes] + len(self.function_end_token)
            if split > len(previous_text):
                first = self.extract_tool_calls_streaming(
                    previous_text,
                    current_text[:split],
                    current_text[len(previous_text) : split],
                    request=request,
                )
                second = self.extract_tool_calls_streaming(
                    current_text[:split],
                    current_text,
                    current_text[split:],
                    request=request,
                )
                if first and second:
                    return {
                        "tool_calls": first.get("tool_calls", [])
                        + second.get("tool_calls", []),
                        "content": first.get("content", "") + second.get("content", ""),
                    }
                return first or second

        # One model delta may contain several finished calls. Reconcile all
        # complete calls before the single-call state machine advances once.
        if not self.in_function and current_closes > previous_closes:
            complete = self.extract_tool_calls(current_text, request)
            closed_count = current_closes
            already = len(self.prev_tool_call_arr)
            if min(len(complete.tool_calls), closed_count) > already:
                fresh = complete.tool_calls[already:closed_count]
                self.prev_tool_call_arr.extend(fresh)
                completed_starts = [
                    start
                    for start, close in self._function_spans(current_text)
                    if close >= 0
                ]
                last_start = completed_starts[closed_count - 1]
                wrapper_index = (
                    sum(
                        start <= last_start
                        for start in self._wrapper_start_positions(current_text)
                    )
                    - 1
                )
                self.current_tool_index = max(already + len(fresh) - 1, wrapper_index)
                self.json_closed = True
                self.header_sent = True
                self.is_tool_call_started = True
                output: dict[str, Any] = {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for index, call in enumerate(fresh, start=already)
                    ]
                }
                if already == 0:
                    first_start = current_text.find(self.tool_call_start_token)
                    prefix = current_text[len(previous_text) : first_start]
                    if prefix:
                        output["content"] = prefix
                return output

        if not delta_text:
            return None

        self.accumulated_text = current_text
        delta_token_ids = delta_token_ids or []

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                if self.current_tool_index >= len(
                    self._wrapper_start_positions(current_text)
                ):
                    self.is_tool_call_started = False
                return None

        # Check if thinking ended (or never started)
        if not self.is_thinking_end:
            # If there's no <seed:think> in the text at all, skip thinking gate
            if (
                self.think_start_token not in current_text
                or (
                    self.think_end_token_id is not None
                    and self.think_end_token_id in delta_token_ids
                )
                or self.think_end_token in delta_text
            ):
                self.is_thinking_end = True

        if not self.is_thinking_end:
            return {"content": delta_text}

        # Handle content before tool calls
        if not self.is_tool_call_started:
            if (
                self.tool_call_start_token_id is not None
                and self.tool_call_start_token_id in delta_token_ids
            ) or (
                self.tool_call_start_token in current_text
                and self.tool_call_start_token not in previous_text
            ):
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
                current_safe = self._safe_content_prefix(current_text)
                previous_safe = self._safe_content_prefix(previous_text)
                content = current_safe[len(previous_safe) :]
                return {"content": content} if content else None

        # Find current tool call portion
        # Locate tool text
        think_end_idx = 0
        if self.think_end_token in current_text:
            think_end_idx = current_text.find(self.think_end_token) + len(
                self.think_end_token
            )
        tool_starts = [
            start
            for start in self._wrapper_start_positions(current_text)
            if start >= think_end_idx
        ]

        if self.current_tool_index >= len(tool_starts):
            return None

        tool_start_idx = tool_starts[self.current_tool_index]
        next_start = (
            tool_starts[self.current_tool_index + 1]
            if self.current_tool_index + 1 < len(tool_starts)
            else -1
        )
        search_end = next_start if next_start >= 0 else len(current_text)
        # A literal wrapper closer can occur inside a parameter value. The
        # last closer before the next wrapper belongs to this call.
        tool_end_idx = current_text.rfind(
            self.tool_call_end_token, tool_start_idx, search_end
        )
        if (
            tool_end_idx >= 0
            and self.function_end_token in current_text[tool_end_idx:search_end]
        ):
            # The only wrapper closer seen so far was inside a parameter;
            # the function closes later, before its real wrapper closer.
            tool_end_idx = -1
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]
        function_closes = self._function_close_positions(tool_text)
        function_close = function_closes[0] if function_closes else -1

        # Parse function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = _generate_tool_id()
                    self.header_sent = True
                    self.in_function = True

                    # If the function body is already complete, emit the full
                    # tool call in one chunk.  This prevents header-only output
                    # when coarse deltas (or max_tokens truncation) leave no
                    # further parser calls to emit the arguments.
                    if function_close >= 0:
                        tools = None
                        if request and isinstance(request, dict):
                            tools = request.get("tools")
                        fc = tool_text[func_start:function_close]
                        parsed = self._parse_xml_function_call(fc, tools)
                        args = parsed["arguments"] if parsed else "{}"
                        self.json_started = True
                        self.json_closed = True
                        self.in_function = False
                        self.prev_tool_call_arr.append(
                            {"name": self.current_function_name, "arguments": args}
                        )
                        return {
                            "tool_calls": [
                                {
                                    "index": self.current_tool_index,
                                    "id": self.current_tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": self.current_function_name,
                                        "arguments": args,
                                    },
                                }
                            ]
                        }

                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_index,
                                "id": self.current_tool_id,
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
                if function_close >= 0:
                    tools = request.get("tools") if isinstance(request, dict) else None
                    start = tool_text.find(self.tool_call_prefix) + len(
                        self.tool_call_prefix
                    )
                    parsed = self._parse_xml_function_call(
                        tool_text[start:function_close], tools
                    )
                    arguments = parsed["arguments"] if parsed else "{}"
                    self.json_closed = True
                    self.in_function = False
                    if parsed:
                        self.prev_tool_call_arr.append(parsed)
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_index,
                                "function": {"arguments": arguments},
                            }
                        ]
                    }
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": "{"},
                        }
                    ]
                }

            # Check for function end
            if not self.json_closed and function_close >= 0:
                self.json_closed = True
                self.in_function = False

                # Extract complete params for prev_tool_call_arr
                tools = None
                if request and isinstance(request, dict):
                    tools = request.get("tools")
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = function_close
                closing_arguments = "}"
                if func_content_end != -1:
                    fc = tool_text[func_start:func_content_end]
                    parsed = self._parse_xml_function_call(fc, tools)
                    if parsed:
                        self.prev_tool_call_arr.append(
                            {"name": parsed["name"], "arguments": parsed["arguments"]}
                        )
                        # A chunk can contain the remaining parameter tags
                        # and the function close together. The older branch
                        # emitted only `}` here, dropping every parameter
                        # not seen on a previous chunk.
                        pairs = json.loads(
                            parsed["arguments"], object_pairs_hook=_ObjectPairs
                        )
                        remaining = pairs[self.param_count :]
                        if remaining:
                            prefix = ", " if self.param_count else ""
                            closing_arguments = (
                                prefix
                                + ", ".join(
                                    f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(_restore_json_value(value), ensure_ascii=False)}"
                                    for key, value in remaining
                                )
                                + "}"
                            )

                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": closing_arguments},
                        }
                    ]
                }

            # Look for complete parameters
            complete_params = tool_text.count(self.parameter_end_token)
            if not self.in_param and self.param_count < complete_params:
                param_starts = []
                si = 0
                while True:
                    si = tool_text.find(self.parameter_prefix, si)
                    if si == -1:
                        break
                    param_starts.append(si)
                    si += len(self.parameter_prefix)

                if len(param_starts) > self.param_count:
                    param_idx = param_starts[self.param_count]
                    param_start = param_idx + len(self.parameter_prefix)
                    remaining = tool_text[param_start:]
                    if ">" in remaining:
                        name_end = remaining.find(">")
                        param_name = remaining[:name_end]
                        value_start = param_start + name_end + 1
                        value_text = tool_text[value_start:]
                        if value_text.startswith("\n"):
                            value_text = value_text[1:]
                        param_end_idx = value_text.find(self.parameter_end_token)
                        if param_end_idx != -1:
                            pv = value_text[:param_end_idx]
                            if pv.endswith("\n"):
                                pv = pv[:-1]
                            # Type conversion using tool schema
                            tools = None
                            if request and isinstance(request, dict):
                                tools = request.get("tools")
                            param_config = _get_arguments_config(
                                self.current_function_name or "", tools
                            )
                            converted = _convert_param_value(
                                pv,
                                param_name,
                                param_config,
                                self.current_function_name or "",
                            )
                            serialized = json.dumps(converted, ensure_ascii=False)
                            if self.param_count == 0:
                                frag = f"{json.dumps(param_name, ensure_ascii=False)}: {serialized}"
                            else:
                                frag = f", {json.dumps(param_name, ensure_ascii=False)}: {serialized}"
                            self.param_count += 1
                            return {
                                "tool_calls": [
                                    {
                                        "index": self.current_tool_index,
                                        "function": {"arguments": frag},
                                    }
                                ]
                            }

        return None
