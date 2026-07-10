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
            # out — i.e. an as-yet-unhandled wire variant. Emit only a
            # STRUCTURAL SUMMARY of the shape, never the raw payload: this is
            # the normal degraded-wire path, and model_output can carry user
            # prompts, tool arguments, or credentials. The counts below are
            # enough to triage the unhandled variant without leaking content.
            has_tool_call_marker = "<tool_call>" in model_output
            function_tag_count = model_output.count("<function=")
            logger.warning(
                "nemotron tool parser: tool-call marker present but no tool "
                "call extracted (possible unhandled variant); "
                "<tool_call> present=%s, %d <function= tags, 0 parseable",
                has_tool_call_marker,
                function_tag_count,
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    @staticmethod
    def _close_tag_count(text: str) -> int:
        """Number of completed tool-call close tags present in ``text``.

        A call may legitimately close on either ``</function>`` (a truncated
        variant that never emits the wrapper) or ``</tool_call>``. Used by the
        streaming path to detect that a NEW call finished in the latest delta
        (count went up) rather than re-parsing on every chunk.
        """
        return text.count("</function>") + text.count("</tool_call>")

    @staticmethod
    def _clean_trailing_content(current_text: str) -> str | None:
        """The plain-content tail of ``current_text``, or ``None`` if none.

        "Tail" = everything after the last COMPLETE tool-call close tag
        (``</function>`` / ``</tool_call>``). It is content-safe only if it
        contains no ``<`` at all — the moment a ``<`` appears we are (possibly)
        building the next call and must suppress, so no tag (complete or a
        partial fragment like ``"<fun"`` / ``"</fun"``) can ever leak into
        user-visible content.

        Returns:
          * ``None``  — still inside markup (no call closed yet, or a new
            ``<`` has started after the last close): suppress.
          * ``""``    — a call has closed and nothing (yet) follows it.
          * ``str``   — the safe trailing content after the last close.
        """
        end = 0
        for tag in ("</function>", "</tool_call>"):
            idx = current_text.rfind(tag)
            if idx != -1:
                end = max(end, idx + len(tag))
        if end == 0:
            # No close tag yet → we are still inside the (first) call's markup.
            return None
        tail = current_text[end:]
        return None if "<" in tail else tail

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

        # Trigger from the COMPLETION STATE of current_text, NOT from a close
        # tag appearing inside a single delta_text. We fire only when a NEW
        # close tag finished in this delta — i.e. the close-tag count in
        # current_text ticked up versus previous_text.
        #
        # Counting the delta (rather than testing membership in delta_text) is
        # what makes a close tag split across chunks work: the tokenizer can
        # emit "</fun" then "ction>", so no single delta ever contains the
        # whole "</function>" — but the accumulated current_text does once both
        # fragments arrive, and only then does the count go up. Gating on the
        # *increase* also means we never re-parse current_text on the many
        # trailing deltas after a call has closed (avoiding O(n^2) re-parsing
        # and repeated fail-open WARNINGs on a trailing unparseable marker).
        #
        # extract_tool_calls re-parses current_text and returns ALL complete
        # calls; we de-dupe against the number already streamed (tracked in
        # current_tool_id, which reset() zeroes per request) so each completed
        # call is emitted exactly once even when </function> and </tool_call>
        # arrive in separate deltas (each bumps the count → one re-parse each,
        # but the second finds nothing new to emit).
        if self._close_tag_count(current_text) > self._close_tag_count(previous_text):
            result = self.extract_tool_calls(current_text)
            # Trailing assistant text that arrived in THIS SAME delta, after the
            # close tag (e.g. the tokenizer emits "</function> done" as one
            # chunk). It is new (everything past the just-closed tag) and, being
            # content-safe per _clean_trailing_content, must not be dropped — we
            # ride it out on the same delta via the combined content+tool_calls
            # return the postprocessor already supports.
            tail = self._clean_trailing_content(current_text)
            if result.tools_called:
                already_emitted = self.current_tool_id + 1
                total = len(result.tool_calls)
                if total > already_emitted:
                    new_calls = result.tool_calls[already_emitted:]
                    self.current_tool_id = total - 1
                    out: dict[str, Any] = {
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
                    if tail:
                        out["content"] = tail
                    return out
            # Close tag but no NEW call to emit (e.g. the second of </function>
            # + </tool_call> for a call already streamed). Still surface any
            # trailing content that rode in on this delta.
            if tail:
                return {"content": tail}
            return None

        # No new call closed in this delta. If we are past all tool-call markup
        # (a call has closed and no new "<" has started since), the delta is
        # trailing assistant content and must pass through instead of being
        # silently dropped. _clean_trailing_content being non-None guarantees no
        # partial or complete tag can leak, so we never emit "<function=",
        # "</function>", or a fragment like "</fun" as user-visible content. We
        # emit only delta_text (the new chars), never the whole tail, so
        # already-streamed trailing content is not re-sent.
        if self._clean_trailing_content(current_text) is not None:
            return {"content": delta_text}

        return None
