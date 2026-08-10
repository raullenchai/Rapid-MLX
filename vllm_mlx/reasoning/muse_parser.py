# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Meta Muse Glimmer's recipient-routed channels.

Muse Glimmer thinks on a ``to=self`` channel instead of ``<think>`` tags::

    <|start|>assistant to=self<|message|>[reasoning]<|eom|>
    <|start|>assistant to=user<|message|>[answer]<|eot|>

The generation prompt ends with ``<|start|>assistant``, so the FIRST
segment of model output has an implicit header — it begins directly
with `` to=self<|message|>`` / `` to=user<|message|>`` (or a bare
``<|message|>``, which the template's ``or 'user'`` default makes
equivalent to ``to=user``).

Segments addressed to a tool (``to=<tool_name>``) carry the ATEM
function-calls block. Those are CONTENT here, not reasoning — they pass
through so the downstream ``MuseToolParser`` (which the postprocessor
feeds with this parser's content channel) can extract the calls; its
prefix-hold machinery keeps the markup off the wire. Mirrors the
harmony split, where the reasoning parser owns analysis/final and the
tool parser owns commentary.
"""

from __future__ import annotations

import re

from .base import DeltaMessage, ReasoningParser

# Explicit segment header. The recipient is optional (history renders
# ``to=user`` explicitly, but a bare ``<|start|>assistant<|message|>``
# is representable and means user).
_HEADER_RE = re.compile(r"<\|start\|>assistant(?:\s+to=(?P<to>\S+?))?\s*<\|message\|>")
# Implicit first-segment header (prompt ends with ``<|start|>assistant``).
_IMPLICIT_RE = re.compile(r"^\s?(?:to=(?P<to>\S+?))?\s*<\|message\|>")
_TERMINATORS = ("<|eot|>", "<|eom|>")
_CONTROL_TOKENS = ("<|start|>", "<|message|>", "<|eot|>", "<|eom|>")

# Partial-prefix hold set for streaming (bug class #444/#480: per-char
# streaming must not leak ``<``, ``<|``, ``<|eo``… as visible content).
_SENTINELS: tuple[str, ...] = ("<|start|>", "<|message|>", "<|eot|>", "<|eom|>")


def _segments(text: str) -> list[tuple[str, str]]:
    """Split model output into ``(recipient, body)`` segments.

    The first segment may use the implicit header; later segments use
    the explicit ``<|start|>assistant`` form. Text with no channel
    markers at all yields a single ``("user", text)`` segment so a
    model that skips the plumbing degrades to plain content.
    """
    segs: list[tuple[str, str]] = []
    first = _IMPLICIT_RE.match(text)
    matches = list(_HEADER_RE.finditer(text))
    if first is None and not matches:
        return [("user", text)] if text else []

    # Region before the first explicit header belongs to the implicit
    # segment (when present). Without an implicit header, plain text
    # before the first explicit header is still model output — it is
    # user-facing content, not discardable plumbing (codex r4 #4).
    if first is not None:
        start = first.end()
        end = matches[0].start() if matches else len(text)
        segs.append((first.group("to") or "user", text[start:end]))
    elif matches and matches[0].start() > 0:
        segs.append(("user", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segs.append((m.group("to") or "user", text[start:end]))

    cleaned: list[tuple[str, str]] = []
    for recipient, body in segs:
        for term in _TERMINATORS:
            body = body.replace(term, "")
        cleaned.append((recipient, body))
    return cleaned


class MuseReasoningParser(ReasoningParser):
    """Routes ``to=self`` segments to reasoning, the rest to content."""

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        # Channel routing is unambiguous — the flag is informational
        # (same signature-parity rationale as GptOssReasoningParser).
        del enable_thinking
        if not model_output:
            return None, None

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for recipient, body in _segments(model_output):
            if recipient == "self":
                reasoning_parts.append(body)
            else:
                # Tool-addressed segments pass through as content so the
                # tool parser can extract the ATEM block.
                content_parts.append(body)

        # Plain concatenation, no per-segment strip — the streaming path
        # emits body bytes verbatim, and the two modes must produce the
        # same API output (codex r1 BLOCKING #3). Only protocol tokens
        # are removed; the model's own whitespace is content.
        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        return reasoning, content

    # ------------------------------------------------------------------
    # Streaming — stateless phase detection from accumulated text, the
    # GptOssReasoningParser pattern.
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        prev_reasoning, prev_content = self._visible(previous_text)
        curr_reasoning, curr_content = self._visible(current_text)

        new_reasoning = curr_reasoning[len(prev_reasoning) :]
        new_content = curr_content[len(prev_content) :]
        if not new_reasoning and not new_content:
            return None
        return DeltaMessage(
            reasoning=new_reasoning or None,
            content=new_content or None,
        )

    @classmethod
    def _visible(cls, text: str) -> tuple[str, str]:
        """(reasoning_bytes, content_bytes) safely emittable for ``text``.

        Recomputed per delta from the full accumulated text — O(n) per
        chunk, same order as the other stateless streaming parsers here.
        The last segment's tail is truncated by the longest partial
        sentinel so a half-arrived ``<|eom|>`` never leaks (#444/#480).
        """
        reasoning: list[str] = []
        content: list[str] = []
        # A ``<|start|>`` not yet followed by ``<|message|>`` is a header
        # still arriving — everything from it on is unclassifiable (the
        # recipient decides reasoning vs content), so hold it. Without
        # this, the tail of ``…<|eom|><|start|>assistant to=u`` would
        # survive token-stripping as literal "assistant to=u" bytes.
        pending = text.rfind("<|start|>")
        if pending >= 0 and "<|message|>" not in text[pending:]:
            text = text[:pending]
        segs = _segments(text)
        # Header-in-progress: the tail may be a partial header for a
        # segment we cannot classify yet. ``_segments`` only recognises
        # COMPLETE headers (through ``<|message|>``), so bytes belonging
        # to a partial header at the tail must not be emitted. Detect:
        # anything after the last complete segment's start that has not
        # passed ``<|message|>`` yet is either terminator plumbing or a
        # growing header — both unemittable. That is exactly the text
        # AFTER the region ``_segments`` assigned, and since each
        # segment runs to the next header (or end), only the no-marker
        # fallback can misfire; guard it separately.
        if segs and segs == [("user", text)] and cls._could_be_header(text):
            return "", ""
        for i, (recipient, body) in enumerate(segs):
            if i == len(segs) - 1:
                body = cls._hold_partial_sentinel(body)
            for token in _CONTROL_TOKENS:
                body = body.replace(token, "")
            if recipient == "self":
                reasoning.append(body)
            else:
                content.append(body)
        return "".join(reasoning), "".join(content)

    @staticmethod
    def _could_be_header(text: str) -> bool:
        """True while ``text`` could still grow into an implicit header.

        The implicit header is `` to=recipient<|message|>``. Until a
        ``<|message|>`` arrives, output like `` to=se`` is unclassifiable;
        emitting it as content would leak plumbing bytes that belong to
        the header. Bounded: headers are short, so only inspect a text
        that is still plausibly all-header (no whitespace beyond the
        leading one, or an incomplete ``<|message|>`` tail).
        """
        candidate = text[1:] if text.startswith(" ") else text
        if "<|message|>" in candidate:
            return False
        # Includes the prefixes of ``to=`` itself — at char granularity
        # the very first deltas are `` t``, `` to`` and are just as
        # unclassifiable as `` to=se``.
        return bool(re.fullmatch(r"|t|to|to=\S*", candidate))

    @staticmethod
    def _hold_partial_sentinel(body: str) -> str:
        max_hold = 0
        for sentinel in _SENTINELS:
            for length in range(min(len(body), len(sentinel) - 1), 0, -1):
                if body.endswith(sentinel[:length]):
                    max_hold = max(max_hold, length)
                    break
        return body if max_hold == 0 else body[: len(body) - max_hold]
