# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.1 tool call parser for rapid-mlx.

Ported from vLLM upstream (vllm/tool_parsers/deepseekv31_tool_parser.py).

Format (different from V3 — no ```json``` code fence, no "function" type tag):
  <｜tool▁calls▁begin｜>
  <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGS<｜tool▁call▁end｜>
  <｜tool▁calls▁end｜>
"""

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


@ToolParserManager.register_module(["deepseek_v31", "deepseek_r1_0528"])
class DeepSeekV31ToolParser(ToolParser):
    """
    Tool call parser for DeepSeek V3.1 and R1-0528 models.

    Uses the same unicode special tokens as V3 but with a simpler format:
    <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGS<｜tool▁call▁end｜>
    (no "function" type prefix, no ```json``` fencing)

    Used when --enable-auto-tool-choice --tool-call-parser deepseek_v31 are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("deepseek_v31_native",)

    TOOL_CALLS_START = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_START = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>"
            r"(?P<function_arguments>.*?)<｜tool▁call▁end｜>",
            re.DOTALL,
        )
        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>(?P<function_arguments>.*)",
            re.DOTALL,
        )
        self.stream_tool_call_name_regex = re.compile(
            r"(?P<function_name>.*)<｜tool▁sep｜>"
        )

        # Token IDs for streaming (graceful fallback if absent)
        self.tool_calls_start_token_id = self.vocab.get(self.TOOL_CALLS_START)
        self.tool_calls_end_token_id = self.vocab.get(self.TOOL_CALLS_END)
        self.tool_call_start_token_id = self.vocab.get(self.TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(self.TOOL_CALL_END)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.TOOL_CALLS_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = self.tool_call_regex.findall(model_output)
            tool_calls = []
            for func_name, func_args in matches:
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": func_name.strip(),
                        "arguments": func_args.strip(),
                    }
                )

            content = model_output[: model_output.find(self.TOOL_CALLS_START)]
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
            self.TOOL_CALLS_START in text
            or self.TOOL_CALL_START in text
            or self.has_text_format_tool_call(text)
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
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self.current_tool_id = -1
            self.prev_tool_call_arr = []

        current_token_ids = current_token_ids or []
        previous_token_ids = previous_token_ids or []
        delta_token_ids = delta_token_ids or []

        # Use token IDs if available, fall back to string matching
        has_tool_start = (
            self.tool_calls_start_token_id is not None
            and self.tool_calls_start_token_id in current_token_ids
        ) or self.TOOL_CALLS_START in current_text

        if not has_tool_start:
            return {"content": delta_text}

        delta_text = delta_text.replace(self.TOOL_CALLS_START, "").replace(
            self.TOOL_CALLS_END, ""
        )

        try:
            # Count tool call tokens (string-based fallback)
            prev_tool_start_count = previous_text.count(self.TOOL_CALL_START)
            prev_tool_end_count = previous_text.count(self.TOOL_CALL_END)
            cur_tool_start_count = current_text.count(self.TOOL_CALL_START)
            cur_tool_end_count = current_text.count(self.TOOL_CALL_END)

            tool_call_portion = None

            # Generating text (no open tool calls)
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.TOOL_CALL_END not in delta_text
            ):
                return {"content": delta_text}

            if self.TOOL_CALL_END in delta_text:
                full_text = current_text
                tool_call_portion = (
                    full_text.split(self.TOOL_CALL_START)[-1]
                    .split(self.TOOL_CALL_END)[0]
                    .rstrip()
                )
                delta_text = delta_text.split(self.TOOL_CALL_END)[0].rstrip()

            # Starting new tool call
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                if len(delta_text) > 1:
                    tool_call_portion = current_text.split(self.TOOL_CALL_START)[-1]
                else:
                    tool_call_portion = None

                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            # Updating existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                tool_call_portion = current_text.split(self.TOOL_CALL_START)[-1]

            # Closing tool call
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                if not self.prev_tool_call_arr or self.current_tool_id >= len(
                    self.prev_tool_call_arr
                ):
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff and '"}' in delta_text:
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_id,
                                "function": {"arguments": diff},
                            }
                        ]
                    }
                return None
            else:
                text = delta_text.replace(self.TOOL_CALL_START, "").replace(
                    self.TOOL_CALL_END, ""
                )
                return {"content": text} if text else None

            # Parse tool call portion
            current_tool_call: dict = {}
            if tool_call_portion:
                m = self.stream_tool_call_portion_regex.match(tool_call_portion)
                if m:
                    current_tool_call["name"] = m.group("function_name")
                    current_tool_call["arguments"] = m.group("function_arguments")
                else:
                    m2 = self.stream_tool_call_name_regex.match(tool_call_portion)
                    if m2:
                        current_tool_call["name"] = m2.group("function_name")
                        current_tool_call["arguments"] = ""
                    else:
                        return None

            # Send tool name
            if not self.current_tool_name_sent:
                if not current_tool_call:
                    return None
                func_name = current_tool_call.get("name")
                if func_name:
                    self.current_tool_name_sent = True
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_id,
                                "id": _generate_tool_id(),
                                "type": "function",
                                "function": {"name": func_name, "arguments": ""},
                            }
                        ]
                    }
                return None

            if tool_call_portion is None:
                return None

            # Ensure prev_tool_call_arr has entry
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            delta = None
            if not cur_arguments and not prev_arguments:
                delta = None
            elif cur_arguments and not prev_arguments:
                delta = {
                    "tool_calls": [
                        {
                            "index": self.current_tool_id,
                            "function": {"arguments": cur_arguments},
                        }
                    ]
                }
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
            elif cur_arguments and prev_arguments:
                if len(cur_arguments) > len(
                    prev_arguments
                ) and cur_arguments.startswith(prev_arguments):
                    diff = cur_arguments[len(prev_arguments) :]
                    delta = {
                        "tool_calls": [
                            {
                                "index": self.current_tool_id,
                                "function": {"arguments": diff},
                            }
                        ]
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None
