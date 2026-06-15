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
        """Reset per-request streaming state machine (see #579).

        The new-format machine processes body bytes incrementally:
        ``_parsed_body_len`` records how many cumulative body bytes we
        have already consumed; each call only walks the new suffix
        (``body[parsed_body_len:]``) so total work is O(n) over the
        stream rather than O(n²) (codex #581 round-2 BLOCKING-3).
        Per-tool state lives in ``_tool_segments`` — each segment owns
        its name buffer, JSON depth / string state, and args-emitted
        offset, so a second ``[TOOL_CALLS]`` is detected only when the
        current tool's args have actually closed (codex #581 round-2
        BLOCKING-1 — a literal ``[TOOL_CALLS]`` inside a string value
        must not split tools).
        """
        # None until first non-whitespace byte after [TOOL_CALLS]:
        #   "new" → Devstral / Mistral v11 ``name[ARGS]{json}`` or ``name{json}``
        #   "old" → Mistral v10- ``[{"name":..., "arguments":...}]`` array form
        self._stream_format: str | None = None
        self._stream_old_emitted: bool = False  # old-format is emit-once
        self._parsed_body_len: int = 0
        self._tool_segments: list[dict[str, Any]] = []
        self._cur_seg_idx: int = 0
        # Buffer for bytes between a closed tool and the next ``[TOOL_CALLS]``
        # opener. Trimmed to longest sentinel-prefix suffix on every
        # non-match so a long junk run between tools doesn't grow it
        # without bound.
        self._inter_buffer: str = ""

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

    _ARGS_TAG: str = "[ARGS]"

    def _new_segment_state(self) -> dict[str, Any]:
        return {
            "id": "",
            "name": "",
            "name_buffer": "",
            "name_emitted": False,
            "needs_name_emit": False,
            "saw_separator": False,
            "args_buffer": "",
            "args_emitted": 0,
            "args_closed": False,
            # JSON tokenizer for boundary detection inside args:
            "json_depth": 0,
            "in_string": False,
            "escape_next": False,
        }

    @staticmethod
    def _json_advance(seg: dict[str, Any], ch: str) -> None:
        """Advance the per-segment JSON tokenizer one char.

        Tracks ``{``/``}`` and ``[``/``]`` nesting plus string + escape
        state so a literal ``[TOOL_CALLS]`` inside a string value can't
        masquerade as a tool-call boundary (codex #581 round-2
        BLOCKING-1). When ``json_depth`` returns to 0 outside a string,
        the segment's args are complete.
        """
        if seg["escape_next"]:
            seg["escape_next"] = False
            return
        if seg["in_string"]:
            if ch == "\\":
                seg["escape_next"] = True
            elif ch == '"':
                seg["in_string"] = False
            return
        if ch == '"':
            seg["in_string"] = True
        elif ch in ("{", "["):
            seg["json_depth"] += 1
        elif ch in ("}", "]"):
            seg["json_depth"] -= 1

    def _stream_new_format(
        self, body: str, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Incremental, JSON-aware new-format processor.

        Walks only the new suffix ``body[parsed_body_len:]`` so total
        work over a stream is O(n) (round-2 BLOCKING-3). Each char is
        classified by the current segment's phase:

        - **name** — accumulating into ``name_buffer`` until either
          ``[ARGS]`` or the first ``{`` is seen; that locks in the name
          and queues a ``needs_name_emit``.
        - **args** — appended to ``args_buffer``; ``_json_advance``
          updates depth/string state; when depth returns to 0 outside a
          string the segment is marked ``args_closed``.
        - **between tools** — bytes go into ``_inter_buffer`` until
          ``[TOOL_CALLS]`` lands (start a new segment) or the buffer
          can no longer be a prefix of the opener (discard).

        At the end of the walk, each segment that has new emit-able
        state (just-locked name and/or new args bytes) contributes one
        entry to ``tool_calls_out``. Args bytes are streamed verbatim
        as they arrive — no prefix-hold inside args, since while
        ``json_depth > 0`` or ``in_string`` is True the chars are
        unambiguously part of the JSON value (round-2 BLOCKING-2: the
        previous design held a partial sentinel suffix inside args and
        could drop trailing bytes like ``"["`` at EOS).
        """
        new_bytes = body[self._parsed_body_len :]
        self._parsed_body_len = len(body)
        if not new_bytes:
            return result or None

        for ch in new_bytes:
            while len(self._tool_segments) <= self._cur_seg_idx:
                self._tool_segments.append(self._new_segment_state())
            seg = self._tool_segments[self._cur_seg_idx]

            if seg["args_closed"]:
                # Between-tool phase: looking for the next ``[TOOL_CALLS]``.
                self._inter_buffer += ch
                if self.BOT_TOKEN in self._inter_buffer:
                    self._cur_seg_idx += 1
                    self._inter_buffer = ""
                else:
                    self._inter_buffer = self._trim_to_sentinel_prefix(
                        self._inter_buffer
                    )
                continue

            if not seg["saw_separator"]:
                seg["name_buffer"] += ch
                if seg["name_buffer"].endswith(self._ARGS_TAG):
                    name = seg["name_buffer"][: -len(self._ARGS_TAG)].strip()
                    if name:
                        seg["name"] = name
                        seg["saw_separator"] = True
                        if not seg["name_emitted"]:
                            seg["id"] = generate_mistral_tool_id()
                            seg["needs_name_emit"] = True
                    continue
                if ch == "{":
                    name = seg["name_buffer"][:-1].strip()
                    if name:
                        seg["name"] = name
                        seg["saw_separator"] = True
                        if not seg["name_emitted"]:
                            seg["id"] = generate_mistral_tool_id()
                            seg["needs_name_emit"] = True
                        seg["args_buffer"] = "{"
                        self._json_advance(seg, "{")
                    continue
                # Still streaming the name; nothing else to do.
                continue

            # Args phase.
            seg["args_buffer"] += ch
            self._json_advance(seg, ch)
            if seg["json_depth"] == 0 and not seg["in_string"]:
                seg["args_closed"] = True

        # Build emission from any segment with new emit-able state.
        tool_calls_out: list[dict[str, Any]] = []
        for i, seg in enumerate(self._tool_segments):
            entry: dict[str, Any] | None = None
            if seg["needs_name_emit"]:
                seg["needs_name_emit"] = False
                seg["name_emitted"] = True
                self.current_tool_id = max(self.current_tool_id, i)
                entry = {
                    "index": i,
                    "id": seg["id"],
                    "type": "function",
                    "function": {"name": seg["name"]},
                }
                tool_calls_out.append(entry)
            if seg["name_emitted"] and len(seg["args_buffer"]) > seg["args_emitted"]:
                args_delta = seg["args_buffer"][seg["args_emitted"] :]
                seg["args_emitted"] = len(seg["args_buffer"])
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

    @classmethod
    def _trim_to_sentinel_prefix(cls, buf: str) -> str:
        """Trim ``buf`` to its longest suffix that's a prefix of BOT_TOKEN.

        Used in the between-tools phase: a 3-char run of junk like
        ``XYZ`` between closing ``}`` and the next ``[TOOL_CALLS]``
        would otherwise grow the buffer unbounded. The longest suffix
        that could still complete a sentinel is the only part worth
        keeping.
        """
        for length in range(min(len(buf), len(cls.BOT_TOKEN) - 1), 0, -1):
            if cls.BOT_TOKEN.startswith(buf[-length:]):
                return buf[-length:]
        return ""
