# SPDX-License-Identifier: Apache-2.0
"""
Nemotron tool call parser for rapid-mlx.

Handles NVIDIA Nemotron models' tool calling format:
- <tool_call><function=name><parameter=p>v</parameter></function></tool_call>

Supports Nemotron-3-Nano-30B-A3B and similar models.
"""

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


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["nemotron", "nemotron3"])
class NemotronToolParser(ToolParser):
    """
    Tool call parser for NVIDIA Nemotron models.

    Supports Nemotron's tool call format:
    <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>

    Also supports JSON arguments:
    <tool_call><function=get_weather>{"city": "Paris"}</function></tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser nemotron are set.
    """

    EXPECTED_WIRE_FORMATS = ("tool_call_xml_body",)

    # Pattern for Nemotron-style with parameters.
    #
    # The load-bearing signature of a call is ``<function=NAME>...</function>``;
    # the ``<tool_call>``/``</tool_call>`` wrapper is treated as optional /
    # decorative so that observed degradations still parse:
    #   (a) a missing/truncated ``</tool_call>``,
    #   (b) a bare ``<function=..>..</function>`` with no wrapper at all,
    #   (d) stray text between ``</function>`` and ``</tool_call>``,
    #   (e) prose between ``<tool_call>`` and ``<function=``.
    # Prose without ``<function=..>..</function>`` still never matches, so
    # this cannot manufacture a tool call out of plain text.
    TOOL_CALL_PATTERN = re.compile(
        r"<function=([^>]+)>(.*?)</function>",
        re.DOTALL,
    )

    # Residual bare wrapper tags left behind after the function bodies have
    # been stripped from ``content``; removed so they don't leak as text.
    RESIDUAL_WRAPPER_PATTERN = re.compile(r"</?tool_call>")

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Nemotron model output.
        """
        if "<tool_call>" not in model_output and "<function=" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        cleaned_text = model_output

        matches = self.TOOL_CALL_PATTERN.findall(model_output)
        for func_name, content in matches:
            func_name = func_name.strip()

            # Try to parse content as JSON first
            content = content.strip()
            if content.startswith("{"):
                try:
                    json.loads(content)
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": content,
                        }
                    )
                    continue
                except json.JSONDecodeError:
                    pass

            # Parse parameter tags
            params = self.PARAM_PATTERN.findall(content)
            if params:
                arguments = {}
                for param_name, param_value in params:
                    # Try to parse value as JSON (for nested objects)
                    try:
                        arguments[param_name.strip()] = json.loads(param_value.strip())
                    except json.JSONDecodeError:
                        arguments[param_name.strip()] = param_value.strip()

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            elif content:
                # Raw content without parameter tags
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": content,
                    }
                )

        # Clean the text: drop the function bodies and any residual bare
        # <tool_call>/</tool_call> wrapper tags so they don't leak as content.
        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text)
            cleaned_text = self.RESIDUAL_WRAPPER_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            # Diagnostic: a tool-call marker was present but nothing parsed
            # out — i.e. an as-yet-unhandled wire variant. Log the raw output
            # (truncated) so the real shape can be captured next time it hits.
            logger.warning(
                "nemotron tool parser: tool-call marker present but no tool "
                "call extracted (possible unhandled variant); raw model_output"
                "[:500]=%r",
                model_output[:500],
            )
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
        Extract tool calls from streaming Nemotron model output.
        """
        if "<tool_call>" not in current_text and "<function=" not in current_text:
            return {"content": delta_text}

        # Fire on either close tag: a truncated variant may only ever emit
        # </function> (no </tool_call>). extract_tool_calls re-parses the full
        # current_text and returns ALL calls, so we de-dupe against the number
        # already streamed (tracked in current_tool_id, which reset() zeroes
        # per request) to avoid re-emitting when </function> and </tool_call>
        # arrive in separate deltas.
        if "</tool_call>" in delta_text or "</function>" in delta_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                already_emitted = self.current_tool_id + 1
                total = len(result.tool_calls)
                if total > already_emitted:
                    new_calls = result.tool_calls[already_emitted:]
                    self.current_tool_id = total - 1
                    return {
                        "tool_calls": [
                            {
                                "index": already_emitted + i,
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
