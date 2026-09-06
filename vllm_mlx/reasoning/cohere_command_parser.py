# SPDX-License-Identifier: Apache-2.0
"""Reasoning detector for the Cohere Command typed-channel protocol.

The generation prompt normally opens the thinking channel, so generated text
starts with reasoning bytes rather than the opening marker. A complete turn is
one of::

    reasoning<|END_THINKING|><|START_TEXT|>answer<|END_TEXT|>
    reasoning<|END_THINKING|><|START_ACTION|>...<|END_ACTION|>

When reasoning is disabled, generation begins directly with a text or action
block. JSON response formats are the one protocol exception: the template asks
the model to emit bare JSON without block markers.

The incremental detector owns marker-prefix buffering and exposes an explicit
end-of-stream drain. Routes never inspect its private phase or opt it into a
model-specific EOF path.
"""

from __future__ import annotations

from .base import DeltaMessage, ReasoningParser

THINK_START = "<|START_THINKING|>"
THINK_END = "<|END_THINKING|>"
TEXT_START = "<|START_TEXT|>"
TEXT_END = "<|END_TEXT|>"
ACTION_START = "<|START_ACTION|>"
ACTION_END = "<|END_ACTION|>"

_REASONING_TRANSITIONS = (THINK_END, TEXT_START, ACTION_START)
_OUTPUT_TRANSITIONS = (TEXT_START, ACTION_START)
_FORCED_CONTENT_MARKERS = (THINK_END, TEXT_START, TEXT_END, ACTION_START)


def _partial_marker_suffix_length(text: str, markers: tuple[str, ...]) -> int:
    """Return the longest suffix that is a strict prefix of a marker."""
    if not text:
        return 0
    limit = min(len(text), max(len(marker) for marker in markers) - 1)
    for size in range(limit, 0, -1):
        suffix = text[-size:]
        if any(marker.startswith(suffix) for marker in markers):
            return size
    return 0


def _first_marker(text: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
    matches = [
        (index, marker) for marker in markers if (index := text.find(marker)) >= 0
    ]
    return min(matches) if matches else None


def _first_marker_outside_json_strings(
    text: str, markers: tuple[str, ...]
) -> tuple[int, str] | None:
    """Return the first marker that is not quoted as JSON string data."""
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        for marker in markers:
            if text.startswith(marker, index):
                return index, marker
    return None


def _json_container_end(text: str) -> int | None:
    """Return the end offset of a leading JSON object/array, if complete."""
    start = len(text) - len(text.lstrip())
    if start == len(text) or text[start] not in "[{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _first_reasoning_transition(
    text: str, *, protect_leading_json: bool
) -> tuple[int, str] | None:
    """Find a typed-channel transition without inspecting JSON string data.

    In structured mode a leading container is ambiguous until it closes: it
    can be the final bare document or JSON-shaped scratch reasoning. Only a
    transition after that top-level container is protocol evidence.
    """
    if protect_leading_json and text.lstrip().startswith(("{", "[")):
        # Scratch JSON may itself be malformed or truncated. Container
        # completeness therefore cannot gate protocol recovery; quote state is
        # the only distinction required to keep marker-looking string values
        # private while honoring real channel transitions.
        return _first_marker_outside_json_strings(text, _REASONING_TRANSITIONS)
    return _first_marker(text, _REASONING_TRANSITIONS)


class CohereCommand4ReasoningParser(ReasoningParser):
    """Parse Command-style thinking, text, and action blocks.

    Streaming starts in the reasoning phase because the chat template puts the
    opening thinking marker in the assistant prefix. The parser then moves to
    exactly one public-output phase: text (markers removed) or action (markers
    preserved for a downstream tool-call parser).
    """

    # The model template can prime reasoning even when the generic route has
    # resolved ``enable_thinking`` false, so bypassing this parser could expose
    # scratch text and structural markers as public content.
    sanitize_when_thinking_disabled = True
    implicit_reasoning_until_close = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._json_mode = False
        self.reset_state()

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    # Compatibility with reasoning parsers that historically exposed these
    # names to the shared truncation and budget helpers.
    @property
    def start_token(self) -> str:
        return THINK_START

    @property
    def end_token(self) -> str:
        return THINK_END

    def reset_state(self) -> None:
        self._buffer = ""
        self._phase = "reasoning"
        self._reasoning_started = False
        self._action_resumes_reasoning = False
        self._forced_end_pending = False
        self._json_protocol_undecided = self._json_mode
        self._json_depth = 0
        self._json_in_string = False
        self._json_escape = False

    def prepare_forced_reasoning_end(self) -> None:
        self._forced_end_pending = True

    def configure_request(
        self,
        *,
        enable_thinking: bool | None = None,
        prompt_thinking_active: bool = False,
        json_mode: bool = False,
    ) -> None:
        # The protocol template, not the generic enable_thinking flag, decides
        # whether the generated stream begins inside thinking. JSON mode is
        # explicit request metadata and must never be inferred from prose.
        del enable_thinking, prompt_thinking_active
        self.reset_state()
        self._json_mode = bool(json_mode)
        self._json_protocol_undecided = self._json_mode

    @staticmethod
    def _strip_leading_think_start(text: str) -> str:
        return text[len(THINK_START) :] if text.startswith(THINK_START) else text

    @staticmethod
    def _extract_text_block(text: str) -> str | None:
        start = text.find(TEXT_START)
        if start < 0:
            return None
        body = text[start + len(TEXT_START) :]
        end = body.find(TEXT_END)
        return body if end < 0 else body[:end]

    def is_open_in_think(self, accumulated_text: str) -> bool:
        if self._json_mode or not accumulated_text:
            return False
        return not any(marker in accumulated_text for marker in _REASONING_TRANSITIONS)

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
        json_mode: bool | None = None,
    ) -> tuple[str | None, str | None]:
        del enable_thinking
        if not model_output:
            return None, None

        request_json_mode = self._json_mode if json_mode is None else bool(json_mode)
        # Non-streaming orchestration performs extraction and then asks the
        # same parser whether a length-truncated buffer is still inside the
        # reasoning channel. Keep the explicit request mode available to that
        # lifecycle probe; otherwise a bare structured document is mistaken
        # for implicit reasoning after it was already parsed as content.
        self._json_mode = request_json_mode
        transition = _first_reasoning_transition(
            model_output, protect_leading_json=request_json_mode
        )
        if (
            request_json_mode
            and model_output.lstrip().startswith(("{", "["))
            and transition is None
        ):
            end = _json_container_end(model_output)
            return None, model_output if end is None else model_output[:end]
        has_protocol_marker = THINK_START in model_output or transition is not None
        if request_json_mode and not has_protocol_marker:
            return None, model_output

        if transition is None:
            transition = _first_marker(model_output, _REASONING_TRANSITIONS)
        if transition is None:
            implicit_reasoning = self._strip_leading_think_start(model_output)
            return implicit_reasoning or None, None

        index, marker = transition
        parsed_reasoning = self._strip_leading_think_start(
            model_output[:index]
        ).lstrip()
        reasoning: str | None = parsed_reasoning or None
        if marker == THINK_END:
            output = model_output[index + len(marker) :]
        else:
            output = model_output[index:]

        output_transition = _first_marker(output, _OUTPUT_TRANSITIONS)
        if output_transition is None:
            return reasoning, None
        output_index, output_marker = output_transition
        output = output[output_index:]
        if output_marker == ACTION_START:
            if marker == ACTION_START:
                # The action opened inside the thinking lane (no closing
                # thinking marker preceded it). Bytes after its envelope are
                # still private reasoning until the protocol closes thinking,
                # so split them off and parse the tail as a fresh turn.
                end = _first_marker_outside_json_strings(
                    output[len(ACTION_START) :], (ACTION_END,)
                )
                if end is not None:
                    cut = len(ACTION_START) + end[0] + len(ACTION_END)
                    envelope, tail = output[:cut], output[cut:]
                    if tail:
                        tail_reasoning, tail_content = self.extract_reasoning(
                            tail, json_mode=False
                        )
                        reasoning = ((reasoning or "") + (tail_reasoning or "")) or None
                        return reasoning, envelope + (tail_content or "")
                    return reasoning, envelope
            return reasoning, output
        return reasoning, self._extract_text_block(output)

    @staticmethod
    def _emit_prefix(text: str, held: int) -> tuple[str, str]:
        if not held:
            return text, ""
        return text[:-held], text[-held:]

    def _drain(self, *, flush: bool) -> DeltaMessage | None:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        while self._buffer:
            if self._phase == "reasoning":
                if self._json_protocol_undecided:
                    transition = _first_reasoning_transition(
                        self._buffer, protect_leading_json=True
                    )
                    think_start = _first_marker_outside_json_strings(
                        self._buffer, (THINK_START,)
                    )
                    if transition is not None or think_start is not None:
                        self._json_protocol_undecided = False
                    elif flush:
                        self._phase = (
                            "json_document"
                            if self._buffer.lstrip().startswith(("{", "["))
                            else "bare"
                        )
                        continue
                    else:
                        # A JSON-shaped prefix is not sufficient to publish:
                        # scratch reasoning may itself be valid JSON and then
                        # close through the typed-channel protocol. Since SSE
                        # cannot retract bytes, wait for protocol evidence or
                        # EOF before choosing the public channel.
                        break

                if not self._reasoning_started:
                    if self._buffer.startswith(THINK_START):
                        self._buffer = self._buffer[len(THINK_START) :]
                        continue
                    if not flush and THINK_START.startswith(self._buffer):
                        break

                transition = _first_reasoning_transition(
                    self._buffer, protect_leading_json=self._json_mode
                )
                if transition is not None:
                    index, marker = transition
                    prefix = self._buffer[:index]
                    if prefix:
                        if self._forced_end_pending:
                            # Incremental marker detection may own a suffix
                            # that the reasoning budget has not seen yet. Once
                            # the orchestrator declares a synthetic boundary,
                            # those held bytes are post-cap public content.
                            content_parts.append(prefix)
                        else:
                            if not self._reasoning_started:
                                prefix = prefix.lstrip()
                            if prefix:
                                reasoning_parts.append(prefix)
                                self._reasoning_started = True
                    if marker == ACTION_START:
                        self._buffer = self._buffer[index:]
                        self._phase = "action"
                        # No closing thinking marker yet: the action opened
                        # inside the thinking lane, so once its envelope
                        # closes the stream is still private reasoning.
                        self._action_resumes_reasoning = True
                    elif marker == TEXT_START:
                        self._buffer = self._buffer[index + len(marker) :]
                        self._phase = "text"
                    else:
                        self._buffer = self._buffer[index + len(marker) :]
                        self._phase = (
                            "forced_content"
                            if self._forced_end_pending
                            else "awaiting_output"
                        )
                        self._forced_end_pending = False
                    continue

                if flush:
                    emitted, self._buffer = self._buffer, ""
                else:
                    held = _partial_marker_suffix_length(
                        self._buffer, (THINK_START,) + _REASONING_TRANSITIONS
                    )
                    emitted, self._buffer = self._emit_prefix(self._buffer, held)
                if emitted:
                    if not self._reasoning_started:
                        emitted = emitted.lstrip()
                    if not emitted:
                        break
                    reasoning_parts.append(emitted)
                    self._reasoning_started = True
                break

            if self._phase == "awaiting_output":
                transition = _first_marker(self._buffer, _OUTPUT_TRANSITIONS)
                if transition is None:
                    if flush:
                        self._buffer = ""
                    else:
                        held = _partial_marker_suffix_length(
                            self._buffer, _OUTPUT_TRANSITIONS
                        )
                        self._buffer = self._buffer[-held:] if held else ""
                    break
                index, marker = transition
                if marker == ACTION_START:
                    self._buffer = self._buffer[index:]
                    self._phase = "action"
                    self._action_resumes_reasoning = False
                else:
                    self._buffer = self._buffer[index + len(marker) :]
                    self._phase = "text"
                continue

            if self._phase == "text":
                end = self._buffer.find(TEXT_END)
                if end >= 0:
                    content_parts.append(self._buffer[:end])
                    self._buffer = ""
                    self._phase = "done"
                    break
                if flush:
                    emitted, self._buffer = self._buffer, ""
                else:
                    held = _partial_marker_suffix_length(self._buffer, (TEXT_END,))
                    emitted, self._buffer = self._emit_prefix(self._buffer, held)
                if emitted:
                    content_parts.append(emitted)
                break

            if self._phase == "action":
                if self._action_resumes_reasoning:
                    # Hold the envelope until it closes; a quoted marker in
                    # JSON string data does not close it. When the action
                    # opened mid-thinking, the bytes after the envelope are
                    # still private reasoning, not public content.
                    end = _first_marker_outside_json_strings(
                        self._buffer[len(ACTION_START) :], (ACTION_END,)
                    )
                    if end is None:
                        if flush:
                            content_parts.append(self._buffer)
                            self._buffer = ""
                        break
                    cut = len(ACTION_START) + end[0] + len(ACTION_END)
                    content_parts.append(self._buffer[:cut])
                    self._buffer = self._buffer[cut:]
                    self._phase = "reasoning"
                    self._action_resumes_reasoning = False
                    continue
                content_parts.append(self._buffer)
                self._buffer = ""
                break

            if self._phase == "bare":
                content_parts.append(self._buffer)
                self._buffer = ""
                break

            if self._phase == "json_document":
                json_end: int | None = None
                for index, char in enumerate(self._buffer):
                    if self._json_in_string:
                        if self._json_escape:
                            self._json_escape = False
                        elif char == "\\":
                            self._json_escape = True
                        elif char == '"':
                            self._json_in_string = False
                        continue
                    if char == '"':
                        self._json_in_string = True
                    elif char in "[{":
                        self._json_depth += 1
                    elif char in "]}":
                        self._json_depth -= 1
                        if self._json_depth == 0:
                            json_end = index + 1
                            break

                if json_end is None:
                    content_parts.append(self._buffer)
                    self._buffer = ""
                    break

                content_parts.append(self._buffer[:json_end])
                self._buffer = ""
                self._phase = "done"
                break

            if self._phase == "forced_content":
                transition = _first_marker(self._buffer, _FORCED_CONTENT_MARKERS)
                if transition is not None:
                    index, marker = transition
                    if index:
                        content_parts.append(self._buffer[:index])
                    if marker == ACTION_START:
                        self._buffer = self._buffer[index:]
                        self._phase = "action"
                    else:
                        self._buffer = self._buffer[index + len(marker) :]
                    continue
                if flush:
                    emitted, self._buffer = self._buffer, ""
                else:
                    held = _partial_marker_suffix_length(
                        self._buffer, _FORCED_CONTENT_MARKERS
                    )
                    emitted, self._buffer = self._emit_prefix(self._buffer, held)
                if emitted:
                    content_parts.append(emitted)
                break

            self._buffer = ""
            break

        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        del previous_text, current_text
        self._buffer += delta_text
        return self._drain(flush=False)

    def finish_stream(self) -> DeltaMessage | None:
        return self._drain(flush=True)

    def finalize_streaming(
        self, accumulated_text: str, **kwargs
    ) -> DeltaMessage | None:
        del accumulated_text, kwargs
        return self.finish_stream()
