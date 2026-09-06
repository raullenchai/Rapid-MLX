# SPDX-License-Identifier: Apache-2.0
"""Rapid-native parser for Cohere North action envelopes.

North-Mini-Code emits ``<|START_ACTION|>`` followed by a JSON object or array
and ``<|END_ACTION|>``.  Unlike mlx-lm's lightweight parser hook, this class
returns Rapid's ``ExtractedToolCallInformation`` and streaming delta shapes.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


@ToolParserManager.register_module(["north", "cohere_north"])
class NorthToolParser(ToolParser):
    """Parse North's JSON action envelope without consuming thinking/text lanes."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("cohere_action_envelope",)

    START = "<|START_ACTION|>"
    END = "<|END_ACTION|>"
    END_THINKING = "<|END_THINKING|>"
    REASONING_TOOL_MARKERS = (START,)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._emitted_tool_count = 0
        self._content_emitted_len = 0
        self._streaming_engaged = False

    def reset(self) -> None:
        super().reset()
        self._emitted_tool_count = 0
        self._content_emitted_len = 0
        self._streaming_engaged = False

    @staticmethod
    def _tool_id(call: dict[str, Any]) -> str:
        supplied = call.get("tool_call_id")
        return (
            supplied
            if isinstance(supplied, str) and supplied
            else f"call_{uuid.uuid4().hex[:8]}"
        )

    @staticmethod
    def _arguments(call: dict[str, Any]) -> str | None:
        """Return canonical object-root arguments, or reject the call.

        OpenAI tool-call arguments must encode a JSON object.  North usually
        emits that object directly, but some checkpoints serialize it once
        before placing it in the action envelope.  Accept either form while
        rejecting malformed JSON and scalar/list/null roots.
        """
        arguments = call.get("parameters", call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        return json.dumps(arguments, ensure_ascii=False)

    @classmethod
    def _parse_action(cls, body: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return []
        entries = [payload] if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            return []
        calls = []
        for entry in entries:
            if not isinstance(entry, dict):
                return []
            name = entry.get("tool_name") or entry.get("name") or entry.get("function")
            if not isinstance(name, str) or not name:
                return []
            arguments = cls._arguments(entry)
            if arguments is None:
                # Treat an action envelope atomically.  Executing the valid
                # subset would hide the rejected bytes and can turn a partly
                # malformed model response into a different operation list.
                return []
            calls.append(
                {
                    "id": cls._tool_id(entry),
                    "name": name,
                    "arguments": arguments,
                }
            )
        return calls

    @classmethod
    def _find_action_end(cls, text: str) -> int | None:
        """Find an END marker outside JSON string literals."""
        in_string = False
        escaped = False
        index = 0
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif text.startswith(cls.END, index):
                    return index
            index += 1
        return None

    @classmethod
    def _pending_action_start(cls, text: str) -> int | None:
        """Return the first action opener without a matching JSON-aware end."""
        cursor = 0
        while cursor < len(text):
            start = text.find(cls.START, cursor)
            if start < 0:
                return None
            body_start = start + len(cls.START)
            end_index = cls._find_action_end(text[body_start:])
            if end_index is None:
                return start
            cursor = body_start + end_index + len(cls.END)
        return None

    @staticmethod
    def _declared_tool_names(
        request: dict[str, Any] | None,
    ) -> frozenset[str] | None:
        """Return the executable names for this request.

        ``None`` means the parser was called without request context (unit
        parsing and internal parity checks).  An empty set means the request
        explicitly exposed no executable tools, including ``tools: null`` and
        ``tool_choice: none``.  Keeping those cases distinct preserves the
        parser's context-free API without promoting a model-hallucinated name
        on an actual chat request.
        """
        if not isinstance(request, dict):
            return None
        if request.get("tool_choice") == "none":
            return frozenset()
        tools = request.get("tools")
        if not isinstance(tools, list):
            return frozenset()
        names = set()
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
        choice = request.get("tool_choice")
        if isinstance(choice, dict):
            function = choice.get("function")
            selected = (
                function.get("name")
                if isinstance(function, dict)
                else choice.get("name")
            )
            if isinstance(selected, str) and selected:
                names.intersection_update({selected})
        return frozenset(names)

    @classmethod
    def _calls_are_declared(
        cls,
        calls: list[dict[str, Any]],
        request: dict[str, Any] | None,
    ) -> bool:
        declared = cls._declared_tool_names(request)
        return (
            declared is None
            or bool(calls)
            and all(call["name"] in declared for call in calls)
        )

    @classmethod
    def _clean_content_lane(cls, text: str) -> str | None:
        """Remove North's private reasoning lane from parser-owned content.

        North's template opens thinking in the prompt, so generated output
        commonly starts with bare reasoning and only emits the closing token.
        The tool parser removes the action envelope; stripping the preceding
        reasoning lane here prevents duplication without a shared-path
        "content equals reasoning" heuristic that could erase legitimate
        output from unrelated parser families.
        """
        if cls.END_THINKING in text:
            text = text.partition(cls.END_THINKING)[2]
        content = text.strip()
        return content or None

    @classmethod
    def split_reasoning_tool_markup(
        cls,
        text: str,
        *,
        pending: bool = False,
        pending_text: str = "",
    ) -> tuple[str | None, str | None]:
        """Separate native action envelopes from surrounding reasoning.

        ``pending`` means a prior streaming delta already routed an unclosed
        action opener to the tool parser, so bytes through the matching closer
        continue on the promoted lane. The JSON-aware closer scan prevents a
        quoted ``<|END_ACTION|>`` argument value from ending the envelope.
        """
        reasoning_parts: list[str] = []
        promoted_parts: list[str] = []
        cursor = 0

        if pending:
            pending_start = cls._pending_action_start(pending_text)
            pending_body = (
                pending_text[pending_start + len(cls.START) :]
                if pending_start is not None
                else ""
            )
            end_index = cls._find_action_end(pending_body + text)
            if end_index is None:
                return None, text
            envelope_end = end_index - len(pending_body) + len(cls.END)
            promoted_parts.append(text[:envelope_end])
            cursor = envelope_end

        while cursor < len(text):
            start = text.find(cls.START, cursor)
            if start < 0:
                reasoning_parts.append(text[cursor:])
                break
            reasoning_parts.append(text[cursor:start])
            body_start = start + len(cls.START)
            end_index = cls._find_action_end(text[body_start:])
            if end_index is None:
                promoted_parts.append(text[start:])
                break
            envelope_end = body_start + end_index + len(cls.END)
            promoted_parts.append(text[start:envelope_end])
            cursor = envelope_end

        reasoning = "".join(reasoning_parts) or None
        promoted = "".join(promoted_parts) or None
        return reasoning, promoted

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if self.START not in model_output:
            # A JSON-shaped assistant answer is not authenticated North wire
            # evidence.  Never promote it merely because it names a declared
            # tool, and make the rejection authoritative so the shared raw-
            # JSON fallback cannot resurrect it after this parser declines.
            return ExtractedToolCallInformation(
                False,
                [],
                model_output,
                rejection_authoritative=True,
            )

        calls: list[dict[str, Any]] = []
        content_parts: list[str] = []
        cursor = 0
        rejected_envelope = False
        while cursor < len(model_output):
            start = model_output.find(self.START, cursor)
            if start < 0:
                content_parts.append(model_output[cursor:])
                break
            content_parts.append(model_output[cursor:start])
            body_start = start + len(self.START)
            end_index = self._find_action_end(model_output[body_start:])
            if end_index is None:
                content_parts.append(model_output[start:])
                break
            body_end = body_start + end_index
            envelope_end = body_end + len(self.END)
            envelope_calls = self._parse_action(
                model_output[body_start:body_end].strip()
            )
            if envelope_calls and self._calls_are_declared(envelope_calls, request):
                calls.extend(envelope_calls)
            else:
                # Rejected and malformed envelopes stay visible as text.  Do
                # not excise and rejoin around them: marker fragments on the
                # two sides were never contiguous on the model wire and must
                # not be rescanned as a synthetic action envelope.
                content_parts.append(model_output[start:envelope_end])
                rejected_envelope = True
            cursor = envelope_end

        if not calls:
            # A complete North envelope is a positive wire-format match.  Its
            # policy or syntax rejection is authoritative: feeding the same
            # bytes to the generic raw-JSON scanner can resurrect an
            # undeclared or malformed call from inside the envelope.
            return ExtractedToolCallInformation(
                False,
                [],
                model_output,
                rejection_authoritative=self.START in model_output,
            )
        content = self._clean_content_lane("".join(content_parts))
        return ExtractedToolCallInformation(
            True,
            calls,
            content,
            rejection_authoritative=rejected_envelope,
        )

    def has_pending_tool_call(self, text: str) -> bool:
        native_pending = self._pending_action_start(text) is not None
        return native_pending or self.has_text_format_tool_call(text)

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
        del delta_text

        if not self._streaming_engaged:
            self._streaming_engaged = True
            if previous_text:
                # First engagement can happen mid-stream: earlier marker-free
                # deltas took the postprocessor's fast path and are already on
                # the wire.  Never re-scan (and re-emit) those bytes.
                self._content_emitted_len = len(previous_text)

        content_parts: list[str] = []
        new_calls: list[dict[str, Any]] = []
        cursor = self._content_emitted_len
        while cursor < len(current_text):
            start = current_text.find(self.START, cursor)
            if start < 0:
                # Hold the longest suffix that is a strict prefix of the
                # opener.  This prevents a character-split ``<|START...``
                # from leaking before the parser knows what it is.
                safe_end = len(current_text)
                max_prefix = min(len(self.START) - 1, len(current_text) - cursor)
                for size in range(max_prefix, 0, -1):
                    if current_text.endswith(self.START[:size]):
                        safe_end = len(current_text) - size
                        break
                if safe_end > cursor:
                    content_parts.append(current_text[cursor:safe_end])
                    cursor = safe_end
                break

            if start > cursor:
                content_parts.append(current_text[cursor:start])
                cursor = start

            body_start = start + len(self.START)
            end_index = self._find_action_end(current_text[body_start:])
            if end_index is None:
                break
            body_end = body_start + end_index
            envelope_end = body_end + len(self.END)
            calls = self._parse_action(current_text[body_start:body_end].strip())

            if calls and self._calls_are_declared(calls, request):
                new_calls.extend(calls)
            else:
                # Syntax failures and undeclared calls remain visible as
                # ordinary text, matching non-stream behavior.
                content_parts.append(current_text[start:envelope_end])
            cursor = envelope_end

        self._content_emitted_len = cursor
        result: dict[str, Any] = {}
        content = "".join(content_parts)
        if content:
            result["content"] = content
        if new_calls:
            start_index = self._emitted_tool_count
            result["tool_calls"] = [
                {
                    "index": start_index + index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for index, call in enumerate(new_calls)
            ]
            self._emitted_tool_count += len(new_calls)
            # North's protocol allows prose after a completed action envelope
            # (e.g. a text block in the same turn), so ask the postprocessor
            # to keep emitting later content deltas instead of applying the
            # emit-tool-then-suppress default.
            result["preserve_post_tool_content"] = True
        return result or None

    def flush_held_content(self, full_text: str) -> str:
        # Structural, not cursor-based: deltas without markers can bypass this
        # parser entirely (the postprocessor's fast path), so a cursor replay
        # here would duplicate content already on the wire.  The only bytes
        # the streaming branch ever withholds are a trailing unclosed action
        # envelope or a partial opener suffix.
        pending_start = self._pending_action_start(full_text)
        if pending_start is not None:
            return full_text[pending_start:]
        max_prefix = min(len(self.START) - 1, len(full_text))
        for size in range(max_prefix, 0, -1):
            if full_text.endswith(self.START[:size]):
                return full_text[-size:]
        return ""
