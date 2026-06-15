# SPDX-License-Identifier: Apache-2.0
"""
Mistral tool call parser for rapid-mlx.

Handles Mistral's tool calling format:
- Format: [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
- Or newer: [TOOL_CALLS]func_name{"arg": "value"}

Used with models like Mistral-7B-Instruct, Devstral, etc.
"""

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

ALPHANUMERIC = ascii_letters + digits


def generate_mistral_tool_id() -> str:
    """
    Generate a random Mistral-compatible tool call ID.

    Mistral Tool Call IDs must be alphanumeric with a length of 9.
    """
    return "".join(choices(ALPHANUMERIC, k=9))


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral models.

    Supports both old and new Mistral tool call formats:
    - Old (< v11): [TOOL_CALLS] [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    - New (>= v11): [TOOL_CALLS]add{"a": 1, "b": 2}

    Used when --enable-auto-tool-choice --tool-call-parser mistral are set.
    """

    # Mistral chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("mistral_tool_calls",)

    BOT_TOKEN = "[TOOL_CALLS]"
    TOOL_CALL_REGEX = re.compile(r"\[{.*}\]", re.DOTALL)

    def has_pending_tool_call(self, text: str) -> bool:
        return "[TOOL_CALLS]" in text

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.bot_token_id = self.vocab.get(self.BOT_TOKEN) if self.vocab else None
        self._reset_stream_state()

    # Held-back partial sentinels in the pre-[TOOL_CALLS] content phase
    # and inside new-format args (where a partial subsequent [TOOL_CALLS]
    # could be heading). Mirrors the hermes_tool_parser pattern that
    # closed the same class of leak for ``<tool_call>`` openers.
    _STREAMING_SENTINELS: tuple[str, ...] = ("[TOOL_CALLS]",)

    def _reset_stream_state(self) -> None:
        """Reset per-request streaming state machine (see #579)."""
        # None until first non-whitespace byte after [TOOL_CALLS]:
        #   "new" → Devstral / Mistral v11 ``name[ARGS]{json}`` or ``name{json}``
        #   "old" → Mistral v10- ``[{"name":..., "arguments":...}]`` array form
        self._stream_format: str | None = None
        self._stream_old_emitted: bool = False  # old-format is emit-once
        # Per-tool state for new-format multi-tool streams. Each entry:
        #   {"name_emitted": bool, "args_emitted": int, "id": str}
        # The 1-tool case (overwhelmingly the common one) just keeps one
        # entry here. Codex #581 flagged that a single shared counter
        # silently swallowed second + N-th calls in
        # ``[TOOL_CALLS]a{}[TOOL_CALLS]b{}`` streams.
        self._tool_states: list[dict[str, Any]] = []

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Strip the longest sentinel-prefix suffix off ``text``.

        Mirror of ``HermesToolParser._safe_content_prefix`` — when the
        model emits ``[``, ``[T``, ``[TO``... ahead of the full
        ``[TOOL_CALLS]`` opener, those partial bytes must NOT fall
        through as content (codex #581 BLOCKING-1). Returns the portion
        of ``text`` safe to ship right now.
        """
        max_hold = 0
        for sentinel in cls._STREAMING_SENTINELS:
            for length in range(min(len(text), len(sentinel) - 1), 0, -1):
                if text.endswith(sentinel[:length]):
                    if length > max_hold:
                        max_hold = length
                    break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    def flush_held_content(self, full_text: str) -> str:
        """Release any prefix-held suffix at stream end.

        If a stream ends with bytes that look like a partial
        ``[TOOL_CALLS]`` opener (e.g. ``"abc["``), those bytes are
        ordinary content and must surface — otherwise the response
        ``"abc["`` would arrive at the client as ``"abc"``.
        """
        if self.BOT_TOKEN in full_text:
            # The tool-call branch already claimed everything from the
            # opener onward; nothing to flush from the pre-opener phase.
            return ""
        return full_text[len(self._safe_content_prefix(full_text)) :]

    def reset(self) -> None:
        super().reset()
        self._reset_stream_state()

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Mistral model response.

        Args:
            model_output: The complete model output string
            request: Optional request context

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        # If the tool call token is not present, return as text response
        if self.BOT_TOKEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_and_raw_tool_calls = model_output.split(self.BOT_TOKEN)
        content = content_and_raw_tool_calls[0].strip()
        raw_tool_calls = content_and_raw_tool_calls[1:]

        tool_calls = []

        for raw_tool_call in raw_tool_calls:
            raw_tool_call = raw_tool_call.strip()
            if not raw_tool_call:
                continue

            # Try new format first: func_name{"arg": "value"}
            # Devstral may emit func_name[ARGS]{"arg": "value"} — strip [ARGS].
            if not raw_tool_call.startswith("[") and "{" in raw_tool_call:
                end_name = raw_tool_call.find("{")
                tool_name = raw_tool_call[:end_name].replace("[ARGS]", "").strip()
                args_str = raw_tool_call[end_name:]

                if tool_name:
                    tool_calls.append(
                        {
                            "id": generate_mistral_tool_id(),
                            "name": tool_name,
                            "arguments": args_str,
                        }
                    )
                continue

            # Try old format: [{"name": "func", "arguments": {...}}]
            try:
                parsed = json.loads(raw_tool_call)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            args = item.get("arguments", {})
                            tool_calls.append(
                                {
                                    "id": generate_mistral_tool_id(),
                                    "name": item["name"],
                                    "arguments": (
                                        json.dumps(args, ensure_ascii=False)
                                        if isinstance(args, dict)
                                        else str(args)
                                    ),
                                }
                            )
                continue
            except json.JSONDecodeError:
                pass

            # Fallback: try regex to extract JSON array
            try:
                match = self.TOOL_CALL_REGEX.search(raw_tool_call)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                args = item.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_mistral_tool_id(),
                                        "name": item["name"],
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
            except (json.JSONDecodeError, AttributeError):
                # If all parsing fails, treat as content
                if raw_tool_call:
                    content = (
                        (content + " " + raw_tool_call).strip()
                        if content
                        else raw_tool_call
                    )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
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
        """Stream tool calls from cumulative Mistral / Devstral output (#579).

        The pre-#579 implementation worked off ``delta_text`` alone and only
        handled the Devstral ``[ARGS]`` separator when it landed in the same
        chunk as ``{``. Real token streams split ``[ARGS]`` across deltas,
        which leaked the literal separator into ``arguments`` and (worse)
        clobbered the name with ``""`` whenever ``[ARGS]{`` arrived fused.

        This implementation drives a tiny state machine off ``current_text``
        so token boundaries are irrelevant — the name is only emitted once
        the boundary character (``[ARGS]`` or ``{``) has been observed in
        full, and ``arguments`` is diffed against per-tool ``args_emitted``
        offsets so each char ships exactly once.

        Three correctness invariants enforced by the structure (and pinned
        by codex review on PR #581):

        1. **Partial-sentinel prefix-hold (BLOCKING-1).** While we're still
           in the pre-``[TOOL_CALLS]`` content phase, a delta of just ``[``
           or ``[T`` must NOT ship as content — it could be the start of
           the opener. ``_safe_content_prefix`` mirrors the hermes pattern
           that closed this leak for ``<tool_call>``.
        2. **Multi-tool new-format (BLOCKING-2).** Bodies like
           ``a{}[TOOL_CALLS]b{}`` are split on subsequent ``[TOOL_CALLS]``
           markers and each segment carries its own name/args offsets.
           Previously the second tool got swallowed into the first call's
           arguments.
        3. **Boundary buffering.** ``read[`` could be heading toward
           ``read[ARGS]`` or could be a tool call ``read`` with arg
           ``[...]``. The state machine waits until either the full
           ``[ARGS]`` separator or the first ``{`` appears in the segment
           before deciding — and prefix-holds any sentinel-suffix at the
           end of the args window so a second-tool boundary still en route
           doesn't leak into the first tool's args.

        Branches on the body's first non-whitespace byte:

        - ``[`` → old Mistral v10- ``[{"name":..., "arguments":...}]`` form
          (buffered until the closing ``]`` then emitted whole).
        - anything else → new Devstral / v11+ ``name[ARGS]?{json}`` form.
        """
        # ----- Phase 1: pre-[TOOL_CALLS] content (prefix-held) -----
        if self.BOT_TOKEN not in current_text:
            return self._emit_safe_content(previous_text, current_text)

        result: dict[str, Any] = {}

        # On the boundary delta, release any pre-opener content that was
        # held back as a partial sentinel in earlier deltas. The pre-
        # opener portion of current_text is now provably plain content
        # (since the opener has fully arrived), so anything safe-but-
        # unshipped from previous_text plus the new head bytes flows out
        # as the final content event.
        if self.BOT_TOKEN not in previous_text:
            head, _, _ = current_text.partition(self.BOT_TOKEN)
            already_shipped = self._safe_content_prefix(previous_text)
            if len(head) > len(already_shipped):
                new_content = head[len(already_shipped) :]
                if new_content:
                    result["content"] = new_content

        # ----- Phase 2: classify the body (one-shot, latches) -----
        _, _, body = current_text.partition(self.BOT_TOKEN)
        if self._stream_format is None:
            stripped = body.lstrip()
            if not stripped:
                return result or None
            self._stream_format = "old" if stripped.startswith("[") else "new"

        if self._stream_format == "old":
            return self._stream_old_format(body, result)
        return self._stream_new_format(body, result)

    def _emit_safe_content(
        self, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        """Return the new-content delta with sentinel prefixes held back."""
        safe_prev = self._safe_content_prefix(previous_text)
        safe_cur = self._safe_content_prefix(current_text)
        if len(safe_cur) <= len(safe_prev):
            return None
        return {"content": safe_cur[len(safe_prev) :]}

    def _stream_old_format(
        self, body: str, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Old ``[{...}]`` array form: buffer until ``]``, then emit whole."""
        if self._stream_old_emitted:
            return result or None
        if "]" not in body:
            return result or None

        info = self.extract_tool_calls(self.BOT_TOKEN + body)
        if not info.tools_called:
            # Malformed array — let downstream finalize handle it.
            return result or None

        tool_calls_out: list[dict[str, Any]] = []
        for i, tc in enumerate(info.tool_calls):
            tool_calls_out.append(
                {
                    "index": i,
                    "id": tc.get("id") or generate_mistral_tool_id(),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
            )
        self._stream_old_emitted = True
        self.current_tool_id = max(self.current_tool_id, len(info.tool_calls) - 1)
        result["tool_calls"] = tool_calls_out
        return result

    def _stream_new_format(
        self, body: str, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """New ``name[ARGS]?{json}`` form, possibly with multiple ``[TOOL_CALLS]``
        delimited calls in the same response.

        Walks the body segment-by-segment (split on ``[TOOL_CALLS]``) and
        maintains per-segment ``{name_emitted, args_emitted, id}`` state so
        the second + N-th tool calls get their own index instead of being
        absorbed into the first call's args (codex #581 BLOCKING-2).
        """
        args_tag = "[ARGS]"
        segments = body.split(self.BOT_TOKEN)

        # The LAST segment is the only one whose tail might still be
        # mid-arrival — anything earlier had its tail fixed when the
        # next ``[TOOL_CALLS]`` boundary landed. Hold back any suffix
        # that could be a prefix of the next ``[TOOL_CALLS]`` opener so
        # it doesn't leak into the current tool's args.
        if segments:
            tail = segments[-1]
            safe_tail = self._safe_content_prefix(tail)
            segments[-1] = safe_tail

        tool_calls_out: list[dict[str, Any]] = []

        for i, seg in enumerate(segments):
            while len(self._tool_states) <= i:
                self._tool_states.append(
                    {"name_emitted": False, "args_emitted": 0, "id": ""}
                )
            state = self._tool_states[i]

            args_idx = seg.find(args_tag)
            brace_idx = seg.find("{")
            if args_idx != -1 and (brace_idx == -1 or args_idx < brace_idx):
                sep_idx = args_idx
                args_start_off = sep_idx + len(args_tag)
            elif brace_idx != -1:
                sep_idx = brace_idx
                args_start_off = sep_idx  # ``{`` is part of the JSON args
            else:
                # Boundary not in this segment yet — name still streaming.
                # For non-last segments this would be malformed (no body
                # between two ``[TOOL_CALLS]`` markers); skip silently.
                continue

            name = seg[:sep_idx].strip()
            if not name:
                continue
            args = seg[args_start_off:]

            entry: dict[str, Any] | None = None
            if not state["name_emitted"]:
                state["id"] = generate_mistral_tool_id()
                state["name_emitted"] = True
                self.current_tool_id = max(self.current_tool_id, i)
                entry = {
                    "index": i,
                    "id": state["id"],
                    "type": "function",
                    "function": {"name": name},
                }
                tool_calls_out.append(entry)

            if len(args) > state["args_emitted"]:
                args_delta = args[state["args_emitted"] :]
                state["args_emitted"] = len(args)
                if entry is not None:
                    entry["function"]["arguments"] = args_delta
                else:
                    tool_calls_out.append(
                        {
                            "index": i,
                            "type": "function",
                            "function": {"arguments": args_delta},
                        }
                    )

        if tool_calls_out:
            result["tool_calls"] = tool_calls_out
        return result or None
