# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.1 tool call parser for rapid-mlx.

Targets the V3.1 "thinking-channel" tool-call body shape:

    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{ARGS_JSON}<｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

All envelope characters are the fullwidth pipe ``｜`` (U+FF5C).

Why this is V3.1-only as of R12-5
---------------------------------
Originally this parser was extended (D-DSV31 hotfix, PR #795) to also
recognise the DeepSeek-V3 "function-typed, JSON-fenced" body shape,
because the DeepSeek-R1-0528-Qwen3-8B chat template inherits V3 and
the alias entry pointed here. That worked but it left this module
carrying two unrelated wire shapes plus a streaming gate that
suppressed incremental ``arguments`` deltas any time a V3 marker was
seen — a footgun for anyone reading the V3.1 parser expecting V3.1
semantics.

R12-5 split the V3 path into its own ``DeepSeekV3ToolParser`` (see
``deepseek_v3_tool_parser.py``) and restored this parser to V3.1-only.
``aliases.json`` and ``model_auto_config.py`` now route R1-0528 to the
V3 parser; this parser only handles checkpoints whose chat template
actually emits the V3.1 shape (gpt-oss style thinking channel).

Block-wise scanning hardening preserved from D-DSV31
----------------------------------------------------
The block-wise envelope scanner is preserved here even though the body
shape is single-format: it's what makes truncated trailing blocks,
parallel calls, and literal-marker content in plain text all behave
correctly (codex r8 BLOCKING-1 on D-DSV31). The original V3.1 parser
used a single greedy regex that produced over-greedy matches on
parallel calls.
"""

import logging
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


@ToolParserManager.register_module("deepseek_v31")
class DeepSeekV31ToolParser(ToolParser):
    """
    Tool call parser for DeepSeek V3.1 thinking-channel format.

    V3-shaped checkpoints (R1-0528, vanilla V3) use
    ``DeepSeekV3ToolParser`` instead. See module docstring.

    Used when ``--enable-auto-tool-choice --tool-call-parser deepseek_v31``
    is set.
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
        self._streamed_call_count = 0
        self._streamed_header_count = 0

    # -----------------------------------------------------------------
    # Block-wise scanner.
    # -----------------------------------------------------------------
    @classmethod
    def _envelope_bounds(cls, model_output: str) -> tuple[int, int] | None:
        """Locate the outer ``<tool_calls_begin>...<tool_calls_end>``
        envelope. Returns ``(inner_start, inner_end)`` pointing at the
        substring strictly between the markers, or ``None`` if the
        outer ``<tool_calls_begin>`` is absent.

        Scanning MUST be bounded to the outer envelope; otherwise a
        response that quotes ``<｜tool▁call▁begin｜>`` as literal content
        could have that content treated as a tool call.
        """
        outer_start = model_output.find(cls.TOOL_CALLS_START)
        if outer_start == -1:
            return None
        inner_start = outer_start + len(cls.TOOL_CALLS_START)
        outer_end = model_output.find(cls.TOOL_CALLS_END, inner_start)
        inner_end = outer_end if outer_end != -1 else len(model_output)
        return (inner_start, inner_end)

    @classmethod
    def _iter_block_bodies(cls, model_output: str) -> list[str]:
        """Yield body text between each ``<call_begin>`` / ``<call_end>``
        pair inside the outer envelope."""
        bounds = cls._envelope_bounds(model_output)
        if bounds is None:
            return []
        inner_start, inner_end = bounds
        bodies: list[str] = []
        pos = inner_start
        start_len = len(cls.TOOL_CALL_START)
        end_len = len(cls.TOOL_CALL_END)
        while pos < inner_end:
            start = model_output.find(cls.TOOL_CALL_START, pos, inner_end)
            if start == -1:
                break
            body_start = start + start_len
            end = model_output.find(cls.TOOL_CALL_END, body_start, inner_end)
            if end == -1:
                # Truncated trailing block — drop and stop.
                break
            bodies.append(model_output[body_start:end])
            pos = end + end_len
        return bodies

    @classmethod
    def _has_open_or_unparsed_block(cls, model_output: str) -> bool:
        bounds = cls._envelope_bounds(model_output)
        if bounds is None:
            return False
        inner_start, inner_end = bounds
        inner = model_output[inner_start:inner_end]
        return inner.count(cls.TOOL_CALL_START) > inner.count(cls.TOOL_CALL_END)

    @classmethod
    def _parse_block_body(cls, body: str) -> tuple[str, str] | None:
        """Parse a V3.1 block body ``NAME<sep>ARGS`` into ``(name, args)``.

        Returns ``None`` if the body has no separator or empty name —
        the caller drops the block. Note: this parser is V3.1-only;
        V3-shaped bodies (``function<sep>NAME\\n\\`\\`\\`json...``) will
        produce ``name="function"`` here, which is the wrong tool — to
        avoid that, route V3-shape checkpoints to
        ``DeepSeekV3ToolParser`` via ``aliases.json``. This parser
        does NOT auto-detect V3 (that fallback is what made D-DSV31 a
        P0 — see module docstring).
        """
        body = body.strip("\n")
        sep_idx = body.find(cls.TOOL_SEP)
        if sep_idx == -1:
            return None
        name = body[:sep_idx].strip()
        args = body[sep_idx + len(cls.TOOL_SEP) :].strip()
        if not name:
            return None
        return name, args

    # -----------------------------------------------------------------
    # Non-streaming extraction.
    # -----------------------------------------------------------------
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.TOOL_CALLS_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            bodies = self._iter_block_bodies(model_output)
            tool_calls: list[dict[str, Any]] = []
            had_malformed = False
            for body in bodies:
                parsed = self._parse_block_body(body)
                if parsed is None:
                    had_malformed = True
                    continue
                name, args = parsed
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": name,
                        "arguments": args,
                    }
                )

            has_truncated = self._has_open_or_unparsed_block(model_output)

            if tool_calls:
                prefix_content = model_output[
                    : model_output.find(self.TOOL_CALLS_START)
                ]
                if had_malformed or has_truncated:
                    content = model_output
                else:
                    content = prefix_content
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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

    # -----------------------------------------------------------------
    # Streaming. Completed blocks are emitted with their final argument
    # bytes; a split control token is never copied into arguments.
    # -----------------------------------------------------------------
    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        marker = cls.TOOL_CALLS_START
        hold = max(
            (n for n in range(1, len(marker)) if text.endswith(marker[:n])),
            default=0,
        )
        return text[: len(text) - hold] if hold else text

    def flush_held_content(self, full_text: str) -> str:
        return full_text[len(self._safe_content_prefix(full_text)) :]

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
            self._streamed_call_count = 0
            self._streamed_header_count = 0

        if self.TOOL_CALLS_START not in current_text:
            current_safe = self._safe_content_prefix(current_text)
            previous_safe = self._safe_content_prefix(previous_text)
            content = current_safe[len(previous_safe) :]
            return {"content": content} if content else None

        result = self.extract_tool_calls(current_text, request)
        count = len(result.tool_calls) if result.tools_called else 0
        already = getattr(self, "_streamed_call_count", 0)
        calls = []
        for index, call in enumerate(result.tool_calls[already:], start=already):
            if index < self._streamed_header_count:
                calls.append(
                    {"index": index, "function": {"arguments": call["arguments"]}}
                )
            else:
                calls.append(
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                )
        self._streamed_call_count = count

        # Anchor an unfinished call as soon as its name is known. The service
        # has a 64 KiB limit for text held before the first tool delta; a
        # complete but larger JSON body must not be released as prose.
        bounds = self._envelope_bounds(current_text)
        if bounds is not None and self.TOOL_CALLS_END not in current_text[bounds[0] :]:
            position, inner_end = bounds
            while position < inner_end:
                opener = current_text.find(self.TOOL_CALL_START, position, inner_end)
                if opener < 0:
                    break
                name_start = opener + len(self.TOOL_CALL_START)
                closer = current_text.find(self.TOOL_CALL_END, name_start, inner_end)
                if closer >= 0:
                    position = closer + len(self.TOOL_CALL_END)
                    continue
                # This is the first unfinished structural block, so any
                # opener inside its JSON body is argument text, not a name.
                if self._streamed_header_count <= count:
                    separator = current_text.find(self.TOOL_SEP, name_start, inner_end)
                    if separator >= 0:
                        name = current_text[name_start:separator].strip()
                        if name:
                            calls.append(
                                {
                                    "index": count,
                                    "id": _generate_tool_id(),
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }
                            )
                            self._streamed_header_count = count + 1
                break
        if not calls:
            return None
        output: dict[str, Any] = {"tool_calls": calls}
        if already == 0:
            prefix = current_text[: current_text.find(self.TOOL_CALLS_START)]
            previous_safe = self._safe_content_prefix(previous_text)
            content = prefix[len(previous_safe) :]
            if content:
                output["content"] = content
        return output
