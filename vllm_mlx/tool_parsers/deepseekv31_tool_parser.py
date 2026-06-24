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

        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        # V3.1 streaming regexes — capture the ``NAME<sep>ARGS`` skeleton
        # in the open block as the model streams in.
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
    # Streaming (V3.1 delta machine — unchanged from original).
    # -----------------------------------------------------------------
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
            prev_tool_start_count = previous_text.count(self.TOOL_CALL_START)
            prev_tool_end_count = previous_text.count(self.TOOL_CALL_END)
            cur_tool_start_count = current_text.count(self.TOOL_CALL_START)
            cur_tool_end_count = current_text.count(self.TOOL_CALL_END)

            tool_call_portion = None

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

            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                tool_call_portion = current_text.split(self.TOOL_CALL_START)[-1]

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
