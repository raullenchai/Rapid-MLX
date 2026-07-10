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

    # Tokens that OPEN tool-call markup. A ``<`` in the content stream is only
    # suppressed when it could be forming one of these — an ordinary ``<`` in
    # prose (e.g. ``"2 < 3"``) is content and must stream through.
    _STREAM_OPENERS = ("<tool_call>", "<function=")

    @classmethod
    def _safe_content_len(cls, text: str) -> int:
        """Length of the leading run of ``text`` that is safe to emit as content.

        Content is everything up to the first ``<`` that could begin (or has
        begun) a tool-call opener — either a complete ``<function=``/
        ``<tool_call>`` or a partial trailing prefix of one (``"<fun"``,
        ``"<tool_cal"``, a lone ``"<"``). Any other ``<`` (``"2 < 3"``) is
        ordinary prose and does not truncate.
        """
        start = 0
        while True:
            j = text.find("<", start)
            if j == -1:
                return len(text)
            rest = text[j:]
            if any(
                op.startswith(rest) or rest.startswith(op) for op in cls._STREAM_OPENERS
            ):
                return j
            start = j + 1

    def _streamed_content(self, text: str) -> str:
        """The assistant-content channel that should have streamed for ``text``.

        Complete tool-call bodies and residual bare wrapper tags are removed
        (they belong to the ``tool_calls`` channel, never content); then any
        trailing in-flight opener is held back via :meth:`_safe_content_len`, so
        a partial/complete tag can never leak. This is cumulative and monotone
        in ``text``, so the per-delta content is just the new suffix versus the
        previous cumulative value — no re-emission of already-streamed prose.
        """
        cleaned = self.TOOL_CALL_PATTERN.sub("", text)
        cleaned = self.RESIDUAL_WRAPPER_PATTERN.sub("", cleaned)
        return cleaned[: self._safe_content_len(cleaned)]

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

        # --- tool_calls channel ------------------------------------------
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
        tool_calls_payload: list[dict[str, Any]] | None = None
        if self._close_tag_count(current_text) > self._close_tag_count(previous_text):
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                already_emitted = self.current_tool_id + 1
                total = len(result.tool_calls)
                if total > already_emitted:
                    new_calls = result.tool_calls[already_emitted:]
                    self.current_tool_id = total - 1
                    tool_calls_payload = [
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

        # --- content channel ---------------------------------------------
        # Assistant prose outside tool markup — before, between, and after
        # calls — must stream through instead of being dropped. We diff the
        # cumulative safe-content of current vs previous, so trailing prose
        # after a call (including in the SAME delta as the close tag, via the
        # combined content+tool_calls return the postprocessor supports) is
        # preserved, an in-flight opener is held until it resolves, and a
        # genuine "<" in prose ("2 < 3") is never suppressed.
        prev_content = self._streamed_content(previous_text)
        cur_content = self._streamed_content(current_text)
        new_content = (
            cur_content[len(prev_content) :]
            if cur_content.startswith(prev_content)
            else ""
        )

        out: dict[str, Any] = {}
        if tool_calls_payload is not None:
            out["tool_calls"] = tool_calls_payload
        if new_content:
            out["content"] = new_content
        return out or None
