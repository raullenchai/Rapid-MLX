# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for the Cohere North family (North-Mini-Code).

North chat templates pre-open the reasoning channel by ending the
generation prompt with ``<|START_THINKING|>``. The model then emits::

    ...chain of thought...<|END_THINKING|><|START_TEXT|>answer<|END_TEXT|>

so the output contains the *closing* thinking marker but usually not the
opening one (the implicit-think shape ``BaseThinkingReasoningParser``
already supports), plus ``<|START_TEXT|>`` / ``<|END_TEXT|>`` wrappers
around the user-visible answer that must be stripped from ``content``.

Found via release dogfood (2026-08-20): without this parser the alias
profile fell back to ``reasoning_parser=None`` and the raw chain of
thought — policy deliberation included — shipped verbatim inside
``message.content`` together with the literal channel markers.
"""

import re

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_START_THINKING = "<|START_THINKING|>"
_END_THINKING = "<|END_THINKING|>"
_START_TEXT = "<|START_TEXT|>"
_END_TEXT = "<|END_TEXT|>"
_TEXT_MARKERS = (_START_TEXT, _END_TEXT)
_ALL_MARKERS = (_START_THINKING, _END_THINKING, _START_TEXT, _END_TEXT)
_TEXT_MARKER_RE = re.compile(r"<\|(?:START|END)_TEXT\|>")
# Content-phase strip: TEXT wrappers plus a genuine ``<|END_THINKING|>``.
# After a reasoning-cap forced close flips the machine to "content", the
# model (which never saw the forged closer) still emits its own
# ``<|END_THINKING|>`` later; in content phase that marker is a
# structural no-op and must never ship as visible bytes (codex round-4
# MAJOR on #2171).
_CONTENT_MARKER_RE = re.compile(r"<\|(?:(?:START|END)_TEXT|END_THINKING)\|>")
_CONTENT_MARKERS = (_START_TEXT, _END_TEXT, _END_THINKING)


def _strip_text_markers(text: str) -> str:
    return _TEXT_MARKER_RE.sub("", text)


def _strip_content_phase_markers(text: str) -> str:
    return _CONTENT_MARKER_RE.sub("", text)


def _partial_marker_suffix_len(
    text: str, markers: tuple[str, ...] = _TEXT_MARKERS
) -> int:
    """Length of the longest trailing run of ``text`` that is a proper
    prefix of one of ``markers`` — i.e. a marker possibly split across
    streaming deltas that must be withheld until the next delta decides.
    """
    longest = max(len(m) for m in markers) - 1
    for n in range(min(len(text), longest), 0, -1):
        suffix = text[-n:]
        if any(marker.startswith(suffix) for marker in markers):
            return n
    return 0


class NorthReasoningParser(BaseThinkingReasoningParser):
    """Parser for North's ``<|START_THINKING|>``/``<|END_THINKING|>``
    reasoning markers with ``<|START_TEXT|>``/``<|END_TEXT|>`` content
    wrappers.

    Reuses the hardened think-tag state machine for the reasoning split
    and layers the TEXT-wrapper stripping on top of the content channel
    (both full-output and streaming, with cross-delta marker buffering).
    """

    # North chat templates do not consult ``enable_thinking`` — they
    # unconditionally end the generation prompt inside
    # ``<|START_THINKING|>``, so the model emits a thought trace even
    # when the route resolved thinking to False (e.g. the casual-chat
    # auto-disable). Keep the parser engaged in that case or the raw
    # chain of thought and the literal channel markers stream into
    # ``delta.content`` (2026-08-20 dogfood repro). Same pair of flags
    # as DeepSeek-R1-Distill, whose templates also prime thinking
    # unconditionally: ``sanitize_when_thinking_disabled`` keeps the
    # postprocessor from bypassing the parser, and
    # ``implicit_reasoning_until_close`` tells
    # ``_should_start_in_thinking``/rescue that generation starts inside
    # the reasoning channel even when thinking resolved False (codex r2
    # #3 — without it a marker-free truncated thought gets promoted back
    # into user-visible content by terminal rescue).
    sanitize_when_thinking_disabled = True
    implicit_reasoning_until_close = True
    # Opt in to the StreamingPostProcessor EOF flush (the UI-TARS
    # hold-back pattern): the machine withholds marker-like suffixes
    # across deltas, so the route must call ``finalize_streaming`` at
    # end of stream or a trailing ``<|END_TE``-style run is dropped
    # (codex r3 BLOCKING).
    stream_eof_flush = True

    @property
    def start_token(self) -> str:
        return "<|START_THINKING|>"

    @property
    def end_token(self) -> str:
        return "<|END_THINKING|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Self-contained streaming state machine (the base think-tag
        # streamer does not withhold split ``<|END_THINKING|>`` bytes in
        # implicit mode — codex r2 #1 — so North owns its streaming):
        # ``_sm_buf`` holds bytes not yet classified; ``_sm_phase`` is
        # "thinking" (initial — North templates pre-open the channel) or
        # "content" (after ``<|END_THINKING|>`` or ``<|START_TEXT|>``).
        self._sm_buf = ""
        self._sm_phase = "thinking"
        # Whether any non-whitespace reasoning byte has been emitted —
        # the first emission is lstripped to mirror the non-streaming
        # ``.strip()`` (the template's pre-opened channel often starts
        # with cosmetic whitespace).
        self._sm_reasoning_started = False

    def reset_state(self):
        super().reset_state()
        self._sm_buf = ""
        self._sm_phase = "thinking"
        self._sm_reasoning_started = False

    def is_open_in_think(self, accumulated_text: str) -> bool:
        """North templates always pre-open the thinking channel, so a
        truncated stream with no ``<|END_THINKING|>`` and no
        ``<|START_TEXT|>`` is an unclosed thought, even though the opener
        never appears in the output."""
        if not accumulated_text:
            return False
        if self.end_token in accumulated_text:
            return False
        if _START_TEXT in accumulated_text:
            return False
        return True

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        no_thinking_markers = (
            self.start_token not in model_output and self.end_token not in model_output
        )
        if no_thinking_markers and _START_TEXT in model_output:
            # Direct-answer shape: ``<|START_TEXT|>answer<|END_TEXT|>``
            # with no thinking block (e.g. thinking disabled). Any prefix
            # before the wrapper is stray channel spill — route it to
            # reasoning rather than the user-visible answer.
            before, _, after = model_output.partition(_START_TEXT)
            reasoning = before.strip() or None
            content = _strip_text_markers(after).strip() or None
            return reasoning, content
        if no_thinking_markers:
            # No markers at all. The North chat template ends the prompt
            # inside ``<|START_THINKING|>``, so marker-free output is a
            # thought trace truncated before ``<|END_THINKING|>`` — the
            # same routing the streaming path applies. ``content`` stays
            # None so the truncated meta-cognition never ships as the
            # user-visible answer.
            return model_output.strip() or None, None
        reasoning, content = super().extract_reasoning(model_output, enable_thinking)
        if reasoning:
            reasoning = _strip_text_markers(reasoning).strip() or None
        if content:
            content = _strip_text_markers(content).strip() or None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Self-contained streaming split.

        The machine starts in the "thinking" phase (North templates
        pre-open the channel in the prompt) and flips to "content" at the
        FIRST structural transition — ``<|END_THINKING|>`` (normal shape)
        or ``<|START_TEXT|>`` (direct-answer / channel-spill shape, codex
        r2 #2). Bytes that could be the head of a marker split across
        deltas are withheld in ``_sm_buf`` until decidable (bounded by
        one marker length, codex r2 #1); ``finalize_streaming`` flushes
        whatever remains so no model bytes are silently dropped.

        Known limitation (same class as the think-tag parsers' literal-
        tag caveat): a literal marker string INSIDE the chain of thought
        is indistinguishable from the structural transition and flips
        the phase early.
        """
        self._sm_buf += delta_text
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        def push_reasoning(text: str) -> None:
            if not self._sm_reasoning_started:
                text = text.lstrip()
            if text:
                self._sm_reasoning_started = True
                reasoning_parts.append(text)

        while True:
            if self._sm_phase == "thinking":
                transitions = [
                    (idx, marker)
                    for marker in (_END_THINKING, _START_TEXT)
                    if (idx := self._sm_buf.find(marker)) != -1
                ]
                if transitions:
                    idx, marker = min(transitions)
                    push_reasoning(self._sm_buf[:idx].replace(_START_THINKING, ""))
                    self._sm_buf = self._sm_buf[idx + len(marker) :]
                    self._sm_phase = "content"
                    continue  # classify the remainder as content
                cleaned = self._sm_buf.replace(_START_THINKING, "")
                held = _partial_marker_suffix_len(cleaned, _ALL_MARKERS)
                emit = cleaned[: len(cleaned) - held] if held else cleaned
                self._sm_buf = cleaned[len(emit) :]
                push_reasoning(emit)
                break
            # content phase: strip complete TEXT markers and any genuine
            # ``<|END_THINKING|>`` arriving after a forced close, withhold
            # a trailing partial-marker prefix.
            stripped = _strip_content_phase_markers(self._sm_buf)
            held = _partial_marker_suffix_len(stripped, _CONTENT_MARKERS)
            emit = stripped[: len(stripped) - held] if held else stripped
            self._sm_buf = stripped[len(emit) :]
            if emit:
                content_parts.append(emit)
            break
        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def finalize_streaming(self, accumulated_text: str, **kwargs):
        """Flush withheld bytes at end of stream.

        A trailing run like ``<|END_TE`` is withheld by the streaming
        machine because it could be a marker split across deltas; when
        the stream ends without completing the marker, those bytes are
        model output and must not be silently dropped (codex r1 #2; same
        never-drop contract as #569). Phase decides the channel.
        """
        del accumulated_text, kwargs  # the machine owns the full state
        carry = self._sm_buf
        self._sm_buf = ""
        if not carry:
            return None
        if self._sm_phase == "content":
            carry = _strip_text_markers(carry)
            return DeltaMessage(content=carry) if carry else None
        carry = carry.replace(_START_THINKING, "")
        return DeltaMessage(reasoning=carry) if carry else None
