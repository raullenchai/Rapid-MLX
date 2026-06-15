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

    def _reset_stream_state(self) -> None:
        """Reset per-request streaming state machine (see #579)."""
        # None until first non-whitespace byte after [TOOL_CALLS]:
        #   "new" → Devstral / Mistral v11 ``name[ARGS]{json}`` or ``name{json}``
        #   "old" → Mistral v10- ``[{"name":..., "arguments":...}]`` array form
        self._stream_format: str | None = None
        self._stream_name_emitted: bool = False
        self._stream_args_emitted: int = 0  # chars of args already streamed
        self._stream_id_value: str = ""
        self._stream_old_emitted: bool = False  # old-format is emit-once

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
        full, and ``arguments`` is diffed against ``_stream_args_emitted``
        so each char ships exactly once. The state machine also branches on
        the body's first non-whitespace byte:

        - ``[`` → old Mistral v10- ``[{"name":..., "arguments":...}]`` form
          (buffered until the closing ``]`` then emitted whole). Old format
          streaming was previously broken by the same naive delta logic.
        - anything else → new Devstral / v11+ ``name[ARGS]{json}`` form.

        Multi-tool new-format streams (``[TOOL_CALLS]a{}[TOOL_CALLS]b{}``)
        currently emit only the first call; tracked separately.
        """
        # ----- Phase 1: pre-[TOOL_CALLS] content streams unchanged -----
        if self.BOT_TOKEN not in current_text:
            return {"content": delta_text} if delta_text else None

        result: dict[str, Any] = {}

        # If [TOOL_CALLS] crossed in *this* delta, emit any preceding
        # plain-text portion of the delta as content. (Earlier deltas
        # already shipped their pre-BOT_TOKEN bytes via the phase-1
        # branch above, so we only need to handle the boundary delta.)
        if self.BOT_TOKEN not in previous_text and self.BOT_TOKEN in delta_text:
            head_delta, _, _ = delta_text.partition(self.BOT_TOKEN)
            if head_delta:
                result["content"] = head_delta

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
        """New ``name[ARGS]{json}`` / ``name{json}`` form: state-machine stream."""
        # Locate the FIRST name→args boundary. ``[ARGS]`` (Devstral) wins
        # over the first ``{`` only when it appears earlier in the stream;
        # the v11+ format without ``[ARGS]`` uses ``{`` directly.
        args_tag = "[ARGS]"
        args_idx = body.find(args_tag)
        brace_idx = body.find("{")

        if args_idx != -1 and (brace_idx == -1 or args_idx < brace_idx):
            sep_idx = args_idx
            args_start = sep_idx + len(args_tag)
        elif brace_idx != -1:
            sep_idx = brace_idx
            args_start = sep_idx  # ``{`` is part of the JSON args
        else:
            # No boundary yet — could still be ``re`` (name) or ``re[`` (en
            # route to ``[ARGS]``). Buffer; do not emit a partial name (the
            # whole point of the fix is to never ship the separator as
            # name/args by accident).
            return result or None

        name = body[:sep_idx].strip()
        if not name:
            return result or None
        args = body[args_start:]

        tool_calls_out: list[dict[str, Any]] = []

        if not self._stream_name_emitted:
            self.current_tool_id += 1
            self._stream_id_value = generate_mistral_tool_id()
            self._stream_name_emitted = True
            tool_calls_out.append(
                {
                    "index": self.current_tool_id,
                    "id": self._stream_id_value,
                    "type": "function",
                    "function": {"name": name},
                }
            )

        if len(args) > self._stream_args_emitted:
            args_delta = args[self._stream_args_emitted :]
            self._stream_args_emitted = len(args)
            if tool_calls_out:
                tool_calls_out[0]["function"]["arguments"] = args_delta
            else:
                tool_calls_out.append(
                    {
                        "index": self.current_tool_id,
                        "type": "function",
                        "function": {"arguments": args_delta},
                    }
                )

        if tool_calls_out:
            result["tool_calls"] = tool_calls_out
        return result or None
