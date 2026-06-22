# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for UI-TARS (ByteDance) GUI-agent VLMs.

UI-TARS emits an unambiguous ``Thought: <chain-of-thought>\n\nAction: ...``
shape per the upstream ``codes/ui_tars/prompt.py`` (COMPUTER_USE_DOUBAO,
MOBILE_USE_DOUBAO). The 1.5 line also occasionally opens with
``Reflection: ... Action_Summary: ...`` as a richer scratchpad — both
preludes are reasoning, not user-facing prose.

This parser splits the prelude into ``reasoning_content`` and leaves the
``Action:`` lines for the matching ``UiTarsToolParser`` to consume.
Symmetric with the tool parser: the action lines are the tool channel,
the ``Thought:`` / ``Reflection:`` line is the reasoning channel.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Anchor at start-of-output so a mid-response mention of "Thought:" inside
# a user-supplied screenshot caption doesn't get misclassified.
#
# Three preamble shapes are recognized (one match per response):
#   1. "Thought: ...\nAction: ..."         (default UI-TARS prompt)
#   2. "Reflection: ...\nAction_Summary: ...\nAction: ..."  (1.5 reflective shape)
#   3. "Action_Summary: ...\nAction: ..."  (1.5 minimal-reflection shape)
#
# Non-greedy up to the first ``Action:`` literal so a runaway thought
# trace doesn't swallow the entire response.
_PREAMBLE_RE = re.compile(
    r"^\s*"
    r"(?P<thought>"
    r"(?:Thought:\s*(?:.*?))"
    r"|(?:Reflection:\s*(?:.*?)\s*Action_Summary:\s*(?:.*?))"
    r"|(?:Action_Summary:\s*(?:.*?))"
    r")"
    r"(?=\s*Action:)",
    re.DOTALL,
)


class UiTarsReasoningParser(ReasoningParser):
    """Reasoning parser for UI-TARS Thought:/Reflection: preambles.

    Returns the ``Thought:`` / ``Reflection: ... Action_Summary: ...``
    prelude as ``reasoning`` and everything from the first ``Action:`` line
    onward as ``content``. The tool parser then strips the ``Action:``
    lines from that content and emits them as ``tool_calls`` — the
    reasoning parser stays out of that concern.

    Streaming: greedy in the early bytes. If we've seen ``Thought:`` /
    ``Reflection:`` / ``Action_Summary:`` at offset 0 but no ``Action:``
    yet, every delta streams to reasoning. Once the first ``Action:``
    arrives we flip channels and the rest goes to content. This matches
    DeepSeek-R1's pattern but with literal-label boundaries instead of
    ``<think>`` tags.
    """

    _PREAMBLE_OPENERS: tuple[str, ...] = (
        "Thought:",
        "Reflection:",
        "Action_Summary:",
    )

    # Maximum trailing-byte prefix that could be the start of the
    # ``Action:`` boundary token. ``Action:`` is 7 chars; we hold back up
    # to 6 trailing bytes from a reasoning delta when they might be the
    # partial opener so the tool parser sees the complete ``Action:``
    # token in its content stream, not pre-truncated to ``ction:`` etc.
    _MAX_BOUNDARY_PREFIX = len("Action:") - 1

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Streaming state: True once we've routed at least one byte to
        # reasoning so the channel is sticky until ``Action:`` shows up.
        self._in_reasoning = False
        self._in_content = False

    # ------------------------------------------------------------------
    # Complete-response API
    # ------------------------------------------------------------------
    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """Split a complete UI-TARS response into (reasoning, content).

        ``enable_thinking`` is accepted for protocol compatibility with the
        base class but ignored — the UI-TARS prompt always elicits the
        Thought: preamble unless the request explicitly used the
        Grounding template (no preamble, action-only). In that case the
        regex simply fails to match and we return ``(None, model_output)``.
        """
        if not model_output:
            return None, model_output

        m = _PREAMBLE_RE.match(model_output)
        if m is None:
            # No preamble — pure-action response (Grounding template) or
            # a malformed output. Surface as content; tool parser will
            # still extract the actions.
            return None, model_output

        thought = m.group("thought").strip() or None
        rest = model_output[m.end() :].lstrip()
        return thought, (rest if rest else None)

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Route each streaming delta to ``reasoning`` or ``content``.

        State machine:

        * Initial: route delta bytes to ``content`` until we've seen
          either an unambiguous preamble opener (``Thought:`` /
          ``Reflection:`` / ``Action_Summary:``) anchored at offset 0, OR
          an ``Action:`` token (which means there's no preamble at all and
          the entire response is content).
        * Reasoning: route delta bytes to ``reasoning`` until the literal
          ``Action:`` arrives. On the transition delta, split bytes
          around the ``Action:`` boundary.
        * Content: route delta bytes to ``content`` until end-of-stream.

        Returns ``None`` on a no-op delta (e.g. partial ``Thoug`` /
        ``Reflec`` opener that hasn't matched yet — bytes are held by
        the postprocessor through ``flush_held_content``).
        """
        if not delta_text:
            return None

        # Already past the preamble — all remaining bytes are content.
        if self._in_content:
            return DeltaMessage(content=delta_text)

        # Discriminate the very first delta(s). We're "in reasoning" as
        # soon as we've seen a preamble opener at offset 0 of the buffer.
        if not self._in_reasoning:
            stripped = current_text.lstrip()
            if not stripped:
                # All whitespace so far — defer decision until we have
                # a real token.
                return None
            if any(stripped.startswith(p) for p in self._PREAMBLE_OPENERS):
                self._in_reasoning = True
                # FALLTHROUGH to the reasoning branch below — first delta
                # of the preamble emits as reasoning.
            elif stripped.startswith("Action:"):
                # Action-only response (Grounding template) — no preamble.
                # Flip to content immediately so a ``wait()`` / ``finished()``
                # ack lands promptly instead of being held until the buffer
                # reaches ``Action_Summary:`` length (codex r1 BLOCKING #2).
                # ``previous_text`` is whatever we previously held back
                # (returned ``None`` for); prepend it so the tool parser
                # sees the complete ``Action:`` sentinel including any
                # leading whitespace.
                self._in_content = True
                if previous_text:
                    return DeltaMessage(content=previous_text + delta_text)
                return DeltaMessage(content=delta_text)
            else:
                # Buffer doesn't yet match any opener. Use the SHORTEST
                # disambiguating prefix length — once ``len(stripped) >=
                # len("Action:")`` (7), we know it can't be a preamble
                # whose opener starts with ``T``/``R``/``A`` because
                # those would have matched above. Flip to content and
                # flush any previously held bytes alongside this delta.
                if len(stripped) >= len("Action:"):
                    self._in_content = True
                    if previous_text:
                        return DeltaMessage(content=previous_text + delta_text)
                    return DeltaMessage(content=delta_text)
                # Hold off — the next delta might complete a preamble.
                return None

        # In reasoning channel. Look for the ``Action:`` boundary inside
        # this delta to decide whether to split.
        action_pos = current_text.find("Action:")
        if action_pos == -1:
            # No full ``Action:`` yet. The trailing bytes of current_text
            # MIGHT be the partial leading edge of ``Action:`` (e.g.
            # delta ``Action`` then later ``:``). Hold any tail bytes
            # back so the tool parser receives the complete ``Action:``
            # token once the boundary forms, instead of seeing the
            # leading ``Action`` token leak into the reasoning channel.
            held = self._compute_partial_action_hold(current_text)
            if held == 0:
                return DeltaMessage(reasoning=delta_text)
            # ``held`` bytes are the trailing partial-opener candidate.
            # Emit only the prefix bytes of this delta that fall BEFORE
            # the partial-opener zone. Bytes already in previous_text
            # were emitted on a prior delta — those can't be unsent —
            # but we can still avoid streaming the boundary candidate
            # bytes that arrived in THIS delta.
            held_window_start = len(current_text) - held
            delta_start_in_current = len(previous_text)
            if held_window_start <= delta_start_in_current:
                # Every byte of this delta is in the held window — emit
                # nothing this round (postprocessor's per-event buffer
                # holds the bytes until the next delta resolves them).
                return None
            split = held_window_start - delta_start_in_current
            return DeltaMessage(reasoning=delta_text[:split])

        # ``Action:`` is in current_text but might be straddling this
        # delta or might be entirely in previous_text. Compute the offset
        # of the boundary within the delta.
        prev_action_pos = previous_text.find("Action:") if previous_text else -1
        if prev_action_pos != -1:
            # Boundary was already crossed on a prior delta — should not
            # happen in practice (we flip ``_in_content=True`` below the
            # first time it does), but defend against double-fire.
            self._in_content = True
            return DeltaMessage(content=delta_text)

        # First time we see ``Action:``. Split delta around the boundary:
        # bytes before it are reasoning, bytes from it onward are content.
        # NOTE: if any partial-opener bytes were held back on a prior
        # delta (e.g. trailing ``Action`` while the colon hadn't arrived
        # yet), they're now part of the content half — prepend them so
        # the tool parser sees the full ``Action:`` token.
        delta_start_in_current = len(previous_text)
        boundary_in_delta = action_pos - delta_start_in_current
        self._in_content = True
        if boundary_in_delta <= 0:
            # Boundary at the start of this delta — but the prefix
            # bytes of ``Action:`` might already be in previous_text
            # that we previously held off emitting. Prepend the held
            # window so the tool parser receives the complete token.
            held_prefix = current_text[action_pos:delta_start_in_current]
            return DeltaMessage(content=held_prefix + delta_text)
        if boundary_in_delta >= len(delta_text):
            # Boundary past end of this delta (defensive).
            return DeltaMessage(reasoning=delta_text)

        reasoning_part = delta_text[:boundary_in_delta]
        content_part = delta_text[boundary_in_delta:]
        return DeltaMessage(reasoning=reasoning_part, content=content_part)

    def _compute_partial_action_hold(self, current_text: str) -> int:
        """Return the number of trailing bytes that might be a partial ``Action:``.

        Specifically: the longest non-empty suffix ``current_text[-k:]``
        (1 ≤ k ≤ 6) that matches a strict prefix of ``"Action:"``. Returns
        0 if no such partial overlap exists.

        Example: current_text ends with ``...thing.\nAction`` — the
        trailing 6 bytes ``Action`` match the 6-char prefix of
        ``Action:``, so we hold those 6 bytes back. The next delta brings
        ``:``, the full ``Action:`` resolves, and the held bytes are
        released to content alongside.
        """
        for k in range(self._MAX_BOUNDARY_PREFIX, 0, -1):
            if k > len(current_text):
                continue
            if "Action:".startswith(current_text[-k:]):
                return k
        return 0

    # ------------------------------------------------------------------
    # Lifecycle hook used by the streaming dispatcher between requests.
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        self._in_reasoning = False
        self._in_content = False
