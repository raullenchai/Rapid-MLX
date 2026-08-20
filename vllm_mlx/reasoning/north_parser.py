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

_START_TEXT = "<|START_TEXT|>"
_END_TEXT = "<|END_TEXT|>"
_TEXT_MARKERS = (_START_TEXT, _END_TEXT)
_TEXT_MARKER_RE = re.compile(r"<\|(?:START|END)_TEXT\|>")


def _strip_text_markers(text: str) -> str:
    return _TEXT_MARKER_RE.sub("", text)


def _partial_marker_suffix_len(text: str) -> int:
    """Length of the longest trailing run of ``text`` that is a proper
    prefix of a TEXT marker — i.e. a marker possibly split across
    streaming deltas that must be withheld until the next delta decides.
    """
    longest = max(len(m) for m in _TEXT_MARKERS) - 1
    for n in range(min(len(text), longest), 0, -1):
        suffix = text[-n:]
        if any(marker.startswith(suffix) for marker in _TEXT_MARKERS):
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
    # ``delta.content`` (2026-08-20 dogfood repro). Same contract as
    # DeepSeek-V4 / DeepSeek-R1 / Muse.
    sanitize_when_thinking_disabled = True

    @property
    def start_token(self) -> str:
        return "<|START_THINKING|>"

    @property
    def end_token(self) -> str:
        return "<|END_THINKING|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # TEXT-marker bytes withheld from the last streamed content delta
        # because they could be the head of a marker split across deltas.
        self._content_marker_carry = ""
        # Streaming phase: "undecided" until the head of the stream shows
        # whether this is a direct answer (``<|START_TEXT|>`` first — no
        # thinking block) or the normal implicit-think shape; then
        # "direct" (own content lane) or "thinking" (delegate to the
        # think-tag state machine).
        self._north_phase = "undecided"

    def reset_state(self):
        super().reset_state()
        self._content_marker_carry = ""
        self._north_phase = "undecided"

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

    def _emit_content_delta(self, text: str) -> DeltaMessage | None:
        """Emit ``text`` on the content lane, stripping TEXT markers and
        withholding a trailing partial-marker prefix until the next delta
        (or the finalize flush) decides what it is."""
        merged = self._content_marker_carry + text
        self._content_marker_carry = ""
        stripped = _strip_text_markers(merged)
        held = _partial_marker_suffix_len(stripped)
        if held:
            self._content_marker_carry = stripped[-held:]
            stripped = stripped[:-held]
        if not stripped:
            return None
        return DeltaMessage(content=stripped)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if self._north_phase == "direct":
            return self._emit_content_delta(delta_text)
        if self._north_phase == "undecided":
            head = current_text.lstrip()
            if not head or (
                len(head) < len(_START_TEXT) and _START_TEXT.startswith(head)
            ):
                # Still ambiguous: everything so far could be the head of a
                # direct-answer ``<|START_TEXT|>``. Withhold until decidable
                # (bounded by the marker length plus leading whitespace).
                return None
            if head.startswith(_START_TEXT):
                # Direct-answer shape: no thinking block — mirror the
                # non-streaming branch and route the wrapped bytes to
                # content (codex r1 BLOCKING #1).
                self._north_phase = "direct"
                return self._emit_content_delta(head[len(_START_TEXT) :])
            # Normal implicit-think shape. Replay the (tiny, bounded)
            # withheld head to the think-tag state machine as one delta so
            # its own bookkeeping starts consistent.
            self._north_phase = "thinking"
            msg = super().extract_reasoning_streaming("", current_text, current_text)
        else:
            msg = super().extract_reasoning_streaming(
                previous_text, current_text, delta_text
            )
        if msg is None or not msg.content:
            return msg
        content_msg = self._emit_content_delta(msg.content)
        stripped = content_msg.content if content_msg is not None else None
        if not stripped and not msg.reasoning:
            return None
        msg.content = stripped
        return msg

    def finalize_streaming(self, accumulated_text: str, **kwargs):
        """Flush any withheld partial-marker bytes at end of stream.

        A trailing run like ``<|END_TE`` is withheld by the streaming
        stripper because it could be a marker split across deltas; when
        the stream ends without completing the marker, those bytes are
        model output and must not be silently dropped (codex r1 BLOCKING
        #2; same never-drop contract as #569).
        """
        try:
            msg = super().finalize_streaming(accumulated_text, **kwargs)
        except TypeError:
            msg = super().finalize_streaming(accumulated_text)
        carry = self._content_marker_carry
        self._content_marker_carry = ""
        if not carry:
            return msg
        if msg is None:
            return DeltaMessage(content=carry)
        msg.content = (msg.content or "") + carry
        return msg
