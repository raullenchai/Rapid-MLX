# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's native tool calling format:
  <|tool_call>call:FUNC_NAME{key:<|"|>value<|"|>,...}<tool_call|>
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

# Match the gemma4 tool-call wire form. The model trains on
#   <|tool_call>call:name{...}<tool_call|>
# but those outer markers are special tokens that HuggingFace's
# ``tokenizer.decode(..., skip_special_tokens=True)`` (the default
# the mlx-vlm / mlx-lm streaming detokenizer invokes) silently strips
# at decode time even when we kept them in ``skip_special_token_ids``.
# Empirically (PR #558 share probe 2026-06-11 on DiffusionGemma 4-bit):
#   prompt:  weather in palo alto
#   output:  call:weather{location:<|"|>Palo Alto<|"|>}
# i.e. the model emits id=48/49 for the outer wrappers (gets stripped),
# but emits the inner ``<|"|>`` (id=52) as raw BPE bytes that survive
# the same decode call. So in practice we see only the inner body.
#
# Make the outer wrappers OPTIONAL so the parser recognises both the
# pristine wire form AND the post-decode stripped form. The body
# ``call:NAME{...}`` is itself a learned wire token unique to tool
# calling — Gemma 4 does not emit ``call:NAME{...}`` in natural prose,
# so allowing the wrappers to be absent does not introduce false
# positives on regular chat turns.
GEMMA4_TOOL_PATTERN = re.compile(
    r"(?:<\|tool_call>)?call:(\w+)\{(.*?)\}(?:<tool_call\|>)?", re.DOTALL
)

# Match a quoted-string value: <|"|>...<|"|>
GEMMA4_QUOTED_VAL_PATTERN = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)
# Match a bare key:value pair (key, then anything up to , or end-of-string)
GEMMA4_KV_BARE_PATTERN = re.compile(r"(\w+)\s*:\s*([^,]+?)(?=\s*,|\s*$)")


def _parse_gemma4_args(args_str: str) -> dict[str, Any]:
    """Parse Gemma 4's argument format into a dict.

    Gemma 4 uses two value styles inside the {...} block:
      - String values are wrapped in quote tokens:  key:<|"|>value<|"|>
      - Numeric / bool / null values are bare:      key:3   key:true   key:null

    Strategy: replace each quoted string with a placeholder, run a generic
    bare-KV parser over the result, then restore placeholders before
    returning. This lets a single pass handle mixed-type arg dicts.
    """
    # Step 1: stash quoted string values so they can't confuse the bare parser
    stashed: list[str] = []

    def _stash(m: re.Match) -> str:
        stashed.append(m.group(1))
        return f"__Q{len(stashed) - 1}__"

    cleaned = GEMMA4_QUOTED_VAL_PATTERN.sub(_stash, args_str)

    # Step 2: bare KV parse
    result: dict[str, Any] = {}
    for kv in GEMMA4_KV_BARE_PATTERN.finditer(cleaned):
        key = kv.group(1)
        raw_val = kv.group(2).strip()
        # Restore stashed string
        if raw_val.startswith("__Q") and raw_val.endswith("__"):
            try:
                idx = int(raw_val[3:-2])
                result[key] = stashed[idx]
                continue
            except (ValueError, IndexError):
                pass
        # Try to parse as JSON literal (int, float, bool, null)
        try:
            result[key] = json.loads(raw_val)
        except (json.JSONDecodeError, ValueError):
            result[key] = raw_val
    return result


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["gemma4", "gemma_4"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Gemma 4 models.

    Format: <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>
    """

    EXPECTED_WIRE_FORMATS = ("gemma4_native", "calling_tool_text")

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_tool_count = 0

    def reset(self):
        """Reset state for a new request."""
        super().reset()
        self._emitted_tool_count = 0

    def has_pending_tool_call(self, text: str) -> bool:
        """A tool call is in flight as soon as we see the body opener
        ``call:NAME{`` — works for both the pristine wire form
        (``<|tool_call>call:NAME{...}<tool_call|>``) AND the
        post-HF-decode stripped form (``call:NAME{...}``). See the
        comment above ``GEMMA4_TOOL_PATTERN`` for why the wrappers can
        be absent.
        """
        if "<|tool_call>" in text:
            return True
        if re.search(r"call:\w+\{", text):
            return True
        return self.has_text_format_tool_call(text)

    def extract_tool_calls(
        self, model_output: str, request: Any = None
    ) -> ExtractedToolCallInformation:
        matches = list(GEMMA4_TOOL_PATTERN.finditer(model_output))

        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)
            args = _parse_gemma4_args(args_str)

            tool_calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(args),
                }
            )

        # Content is everything outside the tool calls
        content = GEMMA4_TOOL_PATTERN.sub("", model_output).strip() or None

        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence = (),
        current_token_ids: Sequence = (),
        delta_token_ids: Sequence = (),
        request: dict[str, Any] | None = None,
    ) -> dict | None:
        # Check if we're inside a tool call. Either the pristine wire
        # form (``<|tool_call>...<tool_call|>``) or the post-HF-decode
        # stripped form (``call:NAME{...}``) triggers parsing — see the
        # comment above ``GEMMA4_TOOL_PATTERN`` for the empirical
        # justification.
        if "<|tool_call>" in current_text or re.search(r"call:\w+\{", current_text):
            # ``GEMMA4_TOOL_PATTERN`` matches completed bodies (it
            # requires the closing ``}`` and optionally the
            # ``<tool_call|>`` trailer). Count those as completed; if
            # the body opener appears more often than completed bodies,
            # we're still mid-stream and should suppress emission.
            completed_matches = list(GEMMA4_TOOL_PATTERN.finditer(current_text))
            completed = len(completed_matches)
            opener_re = re.compile(r"call:\w+\{")
            open_count = len(list(opener_re.finditer(current_text)))

            # Still accumulating an incomplete tool call
            if completed < open_count:
                return None  # suppress output while inside tool markup

            # Only emit newly completed tool calls (dedup)
            if completed <= self._emitted_tool_count:
                return None

            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                # Only emit tool calls we haven't sent yet
                new_calls = result.tool_calls[self._emitted_tool_count :]
                self._emitted_tool_count = len(result.tool_calls)

                if new_calls:
                    return {
                        "tool_calls": [
                            {
                                "index": self._emitted_tool_count - len(new_calls) + i,
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

        # Text-format tool call recovery: catch [Calling tool: name({...})]
        # Models degrade to this format after multiple tool rounds at low quant
        from .abstract_tool_parser import TEXT_TOOL_CALL_ANY, TEXT_TOOL_CALL_FN_PATTERN

        if TEXT_TOOL_CALL_ANY.search(current_text):
            # Check if we have a complete text tool call
            matches = list(TEXT_TOOL_CALL_FN_PATTERN.finditer(current_text))
            new_matches = matches[self._emitted_tool_count :]
            if new_matches:
                self._emitted_tool_count = len(matches)
                return {
                    "tool_calls": [
                        {
                            "index": self._emitted_tool_count - len(new_matches) + i,
                            "id": _generate_tool_id(),
                            "type": "function",
                            "function": {
                                "name": m.group(1),
                                "arguments": m.group(2),
                            },
                        }
                        for i, m in enumerate(new_matches)
                    ]
                }
            # Already emitted or partial — suppress
            return None

        # No tool call markup — pass through as content
        return {"content": delta_text}
