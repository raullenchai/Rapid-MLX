# SPDX-License-Identifier: Apache-2.0
"""Typed-content state machine adapted to Rapid parser lifecycle interfaces."""

from .base import DeltaMessage, ReasoningParser

MESSAGE = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
SAMPLING_END = "<|content_model_end_sampling|>"
MARKERS = (
    MESSAGE,
    THINKING,
    TEXT,
    TOOL,
    END,
    SAMPLING_END,
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
)


class InklingBlocks:
    """Incremental typed-block transitions, shared by both parser passes.

    Header text is metadata. Thinking, text and JSON blocks may repeat in
    any order; either end marker closes a block. The text grammar also works
    on reconstructed content passed from reasoning to the tool parser.
    """

    def __init__(self):
        self.state = "content"
        self.buffer = ""

    def feed(self, text, final=False):
        self.buffer += text
        events = []
        while self.buffer:
            marker = next((m for m in MARKERS if self.buffer.startswith(m)), None)
            if marker:
                self.buffer = self.buffer[len(marker) :]
                if marker == MESSAGE:
                    self.state = "header"
                elif marker == THINKING:
                    self.state = "reasoning"
                elif marker == TOOL:
                    self.state = "tool"
                elif marker in (END, SAMPLING_END):
                    self.state = "content"
                else:
                    self.state = "content"
                events.append(("marker", marker))
                continue
            if not final and any(m.startswith(self.buffer) for m in MARKERS):
                break
            end = 1
            while end < len(self.buffer) and self.buffer[end] != "<":
                end += 1
            events.append((self.state, self.buffer[:end]))
            self.buffer = self.buffer[end:]
        return events


class InklingReasoningParser(ReasoningParser):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer)
        self.reset_state()

    @property
    def reasoning_start_str(self):
        return THINKING

    @property
    def reasoning_end_str(self):
        return END

    def reset_state(self):
        self.blocks = InklingBlocks()
        self.tool = False
        self.reasoning_whitespace = ""

    def adjust_request(self, request):
        request.skip_special_tokens = False
        return request

    def _delta(self, events):
        segments = []
        for kind, value in events:
            if kind == "marker":
                if value != THINKING:
                    self.reasoning_whitespace = ""
                if value == TOOL:
                    self.tool = True
                    segments.append(("content", value))
                elif self.tool and value in (END, SAMPLING_END):
                    segments.append(("content", value))
                    self.tool = False
                continue
            if kind == "reasoning":
                value = self.reasoning_whitespace + value
                trimmed = value.rstrip()
                self.reasoning_whitespace = value[len(trimmed) :]
                if trimmed:
                    segments.append(("reasoning", trimmed))
            elif kind in ("content", "tool"):
                segments.append(("content", value))
        reasoning = "".join(v for k, v in segments if k == "reasoning")
        content = "".join(v for k, v in segments if k == "content")
        return (
            DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
                source_segments=tuple(segments),
            )
            if segments
            else None
        )

    def extract_reasoning(self, model_output, enable_thinking=None):
        parser = type(self)(self.tokenizer)
        msg = parser._delta(parser.blocks.feed(model_output, final=True))
        return (msg.reasoning, msg.content) if msg else (None, None)

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
        return self._delta(self.blocks.feed(delta_text))

    def finish_stream(self):
        return self._delta(self.blocks.feed("", final=True))

    def is_open_in_think(self, accumulated_text):
        blocks = InklingBlocks()
        blocks.feed(accumulated_text, final=True)
        return blocks.state == "reasoning"
