# SPDX-License-Identifier: Apache-2.0
"""
Harmony tool call parser for GPT-OSS models.

Harmony uses control tokens and channels for tool calling:

    <|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"location": "SF"}<|call|>

The final response is in the 'final' channel:

    <|start|>assistant<|channel|>final<|message|>The weather is 72F.<|end|>
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


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Tool call pattern — supports both formats from the harmony spec:
#   Model-generated: <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>ARGS<|call|>
#   Template-encoded (history): to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>
# Terminator: ``<|call|>`` is the in-output token, but the engine stops
# generation when it emits it (``<|call|>`` is part of the harmony EOS
# set), so the token is consumed and never appears in ``output_text``.
# Empirically (gpt-oss-20b-mxfp4-q8 via /v1/chat/completions, 2026-05-22) the
# commentary block ends with the JSON args and no terminator. Accept
# end-of-string OR the next channel marker as alternative terminators
# so a complete-but-unterminated tool call still parses. Same regression
# class as PR #436's hermes unclosed-``<tool_call>`` fix.
#
# Tool names follow the OpenAI/Anthropic spec (letters, digits,
# underscores, hyphens) — ``[\w-]+`` covers them. ``\w+`` alone would
# silently drop hyphenated names (``get-weather``, ``my-tool``).
_COMMENTARY_BLOCK_PATTERN = re.compile(
    r"(?:"
    # Real format: to=functions.NAME<|channel|>commentary [content_type]<|message|>
    r"to=functions\.([\w-]+)<\|channel\|>commentary(?:\s+\w+)?<\|message\|>"
    r"(.*?)"
    r"(?:<\|call\|>|<\|channel\|>|<\|start\|>|<\|end\|>|<\|return\|>|$)"
    r"|"
    # Legacy format: <|channel|>commentary to=functions.NAME ... <|message|>
    r"<\|channel\|>commentary\s+to=functions\.([\w-]+)(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>"
    r"(.*?)"
    r"(?:<\|call\|>|<\|channel\|>|<\|start\|>|<\|end\|>|<\|return\|>|$)"
    r")",
    re.DOTALL,
)

# Final channel — both <|end|> and <|return|> terminators
_FINAL_BLOCK_PATTERN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)",
    re.DOTALL,
)


@ToolParserManager.register_module(["harmony", "gpt-oss"])
class HarmonyToolParser(ToolParser):
    """
    Tool call parser for GPT-OSS models using Harmony format.

    Harmony uses control tokens and 3 channels:
    - analysis: internal reasoning (handled by reasoning parser)
    - commentary: tool calls addressed with to=functions.{name}
    - final: user-facing response

    Used when --enable-auto-tool-choice --tool-call-parser harmony are set.
    """

    # GPT-OSS chat template natively handles tool_calls and role="tool"
    # messages using harmony channel tokens (to=functions.NAME, <|call|>).
    # Without this, tool history is converted to "[Calling tool: ...]" text
    # which breaks the model's understanding of the tool flow.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    EXPECTED_WIRE_FORMATS = ("harmony_commentary",)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Harmony model response.

        Parses commentary channel blocks for tool calls and the final
        channel for the user-facing content.
        """
        tool_calls = []

        # Extract tool calls from commentary channel blocks
        # Regex has 4 groups: (1,2) for real format, (3,4) for legacy format
        for match in _COMMENTARY_BLOCK_PATTERN.finditer(model_output):
            tool_name = match.group(1) or match.group(3)
            args_str = (match.group(2) or match.group(4) or "").strip()

            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                # Keep the raw arguments string
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": args_str,
                    }
                )

        # Extract final channel content
        final_match = _FINAL_BLOCK_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        # No tool calls: return all text as content
        # If there's a final channel, use that; otherwise return the raw output
        # stripped of control tokens
        if content is None:
            content = _strip_control_tokens(model_output)

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content,
        )

    # Harmony control-token sentinels. ``extract_tool_calls_streaming``
    # must hold back partial-prefix suffixes (``<``, ``<|``, ``<|ch``…)
    # before the full opener arrives, otherwise per-char streaming
    # leaks them as content deltas (issue #444 / #480).
    _STREAMING_SENTINELS: tuple[str, ...] = (
        "<|channel|>",
        "<|message|>",
        "<|call|>",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|constrain|>",
    )

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Strip the longest harmony-sentinel prefix off ``text``'s tail.

        Mirrors ``HermesToolParser._safe_content_prefix`` — see that
        docstring for the algorithm. Distinct copy because the
        sentinel set is harmony-specific.
        """
        max_hold = 0
        for sentinel in cls._STREAMING_SENTINELS:
            for length in range(min(len(text), len(sentinel) - 1), 0, -1):
                if text.endswith(sentinel[:length]):
                    if length > max_hold:
                        max_hold = length
                    break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    @classmethod
    def _emit_safe_content(
        cls, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        """Emit the new content delta with sentinel prefixes held back."""
        safe_current = cls._safe_content_prefix(current_text)
        safe_previous = cls._safe_content_prefix(previous_text)
        if len(safe_current) <= len(safe_previous):
            return None
        return {"content": safe_current[len(safe_previous) :]}

    def flush_held_content(self, full_text: str) -> str:
        """Release the prefix-held suffix at stream end.

        Mirror of ``HermesToolParser.flush_held_content`` — see that
        docstring. Handles harmony sentinels (``<|...``); avoids
        end-of-stream truncation under char-level streaming.
        """
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
        """
        Extract tool calls from streaming Harmony model output.

        Waits for <|call|> to complete a tool call, and emits final
        channel content as regular content deltas. Partial harmony
        sentinels (``<|chan``, ``<|c``…) are held back via
        ``_emit_safe_content`` so per-char streaming doesn't leak the
        leading ``<`` chars as content deltas before the full opener
        arrives (issues #444 / #480 — family-wide leak across every
        harmony streaming entry point).
        """
        # If a tool-call completion marker just FINISHED arriving — i.e.
        # ``<|call|>`` is in ``current_text`` but was absent from
        # ``previous_text`` — emit the structured tool call. Using
        # ``in delta_text`` would only fire when a single delta carries
        # the entire sentinel atomically (whole-token delivery); under
        # char-level streaming the sentinel spans 8 deltas and the
        # check would never fire.
        prev_call_count = previous_text.count("<|call|>")
        curr_call_count = current_text.count("<|call|>")
        if curr_call_count > prev_call_count:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
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

        # If we're in the final channel, emit content token by token.
        # Track emitted length to only send new content each delta.
        if "<|channel|>final" in current_text:
            final_start = current_text.rfind("<|channel|>final")
            msg_start = current_text.find("<|message|>", final_start)
            if msg_start >= 0:
                raw = current_text[msg_start + len("<|message|>") :]
                # Strip COMPLETE control tokens, then apply prefix-hold
                # so a partial trailing sentinel (``<``, ``<|``, ``<|e``
                # …) doesn't leak before the full ``<|end|>`` arrives
                # under char-level streaming (codex round-2 CRITICAL).
                # DO NOT ``.strip()`` here — a trailing space immediately
                # before a held sentinel (``hello <``) would be silently
                # dropped, surfacing as ``hello<`` (codex round-5
                # CRITICAL). Leading whitespace can appear in legit
                # model output (e.g. final-channel content starting with
                # a space) so trimming it would also be lossy. Diff the
                # raw stripped+held text as-is.
                clean = self._safe_content_prefix(_strip_control_tokens_inner(raw))
                # Calculate what's new since previous extraction
                prev_final = previous_text.rfind("<|channel|>final")
                prev_clean = ""
                if prev_final >= 0:
                    prev_msg = previous_text.find("<|message|>", prev_final)
                    if prev_msg >= 0:
                        prev_raw = previous_text[prev_msg + len("<|message|>") :]
                        prev_clean = self._safe_content_prefix(
                            _strip_control_tokens_inner(prev_raw)
                        )
                new_content = clean[len(prev_clean) :]
                if new_content:
                    return {"content": new_content}
            # In final channel but no new content yet (control token)
            return {"content": ""}

        # If no full sentinel present yet, emit safe content (partial
        # sentinel prefixes held back).
        if "<|channel|>" not in current_text:
            return self._emit_safe_content(previous_text, current_text)

        # Building tool call or in analysis channel, suppress output
        return None

    def has_pending_tool_call(self, text: str) -> bool:
        """Check if text contains incomplete Harmony tool call markup."""
        return "to=functions." in text


# Module-level constants exposed for cross-checking by regression-test
# infrastructure (tests/parsers/_harmony_markers.py). Keep these in sync
# with what ``_strip_control_tokens_inner`` actually removes — the smoke
# test ``test_harmony_markers_match_source`` asserts set equality, so a
# new control token added here without updating the regression allowlist
# (or vice versa) fails loudly instead of silently masking a leak.
HARMONY_STRIPPED_CONTROL_TOKENS: tuple[str, ...] = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)


def _strip_control_tokens_inner(text: str) -> str:
    """Remove Harmony control tokens from ``text`` WITHOUT trimming
    surrounding whitespace.

    Streaming callers must preserve whitespace fidelity (a trailing
    space in user-visible content must reach the client). The
    public ``_strip_control_tokens`` wraps this and trims for the
    non-stream extract_tool_calls path that historically expected
    a trimmed return.
    """
    result = text
    for token in HARMONY_STRIPPED_CONTROL_TOKENS:
        result = result.replace(token, "")
    # Clean up channel names and constrain values
    result = re.sub(r"(?:analysis|commentary|final)\s*", "", result)
    result = re.sub(r"to=functions\.\w+\s*", "", result)
    result = re.sub(r"json\s*", "", result)
    return result


def _strip_control_tokens(text: str) -> str:
    """Remove Harmony control tokens from text (non-stream / convenience).

    Trims surrounding whitespace — used by the non-stream
    ``extract_tool_calls`` path. Streaming code that must preserve
    whitespace fidelity should call ``_strip_control_tokens_inner``
    directly.
    """
    return _strip_control_tokens_inner(text).strip()


def _is_control_token(text: str) -> bool:
    """Check if text is a Harmony control token."""
    return text.strip() in {
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|return|>",
        "<|call|>",
    }
