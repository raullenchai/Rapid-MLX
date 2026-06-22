# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 reasoning parser.

Gemma 4 uses channel tokens for thinking:
  <|channel>thought\n...reasoning...<channel|>
  <|channel>content\n...answer...<channel|>

The parser separates thinking from content by tracking the active channel.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Match full thought blocks in complete text
# Codex round-13 BLOCKING (PR #799): the inner non-greedy
# ``[\s\S]*?`` previously allowed a stray ``<|channel>`` opener
# inside the thought body — so the malformed shape
# ``<|channel>thought\nsecret<|channel>content\nanswer<channel|>``
# would match as ONE thought block from the opener to the FIRST
# ``<channel|>`` (the content channel's closer), and the post-
# match ``replace("<channel|>", "")`` step would leak
# ``secret<|channel>content\nanswer`` into reasoning while the
# content branch saw an empty residual.
#
# Fix: disallow a nested ``<|channel>`` opener inside the body
# via a negative lookahead at every position. The match now stops
# at the FIRST closer ``<channel|>`` OR aborts as soon as
# another channel opener appears, so the malformed shape falls
# through to the no-blocks "unterminated thought" branch where
# the round-13 unterminated-thought split routes the body to
# reasoning and the downstream content channel to content.
_THOUGHT_BLOCK = re.compile(
    r"<\|channel>thought\n(?:(?!<\|channel>)[\s\S])*?<channel\|>\s*",
    re.DOTALL,
)
# Match content channel markers
_CONTENT_START = re.compile(r"<\|channel>(?:content|final)\n?")
_CHANNEL_END = re.compile(r"<channel\|>")
_TURN_END = re.compile(r"<turn\|>")

# Codex round-14 BLOCKING (PR #799): match ANY channel opener so the
# unterminated-thought split can route downstream channels by TYPE.
# Captures the channel type into the named group ``type`` so a nested
# ``thought`` channel that follows the unterminated one routes to
# reasoning instead of leaking into content.
_CHANNEL_SEGMENT = re.compile(r"<\|channel>(?P<type>thought|content|final)\n?")


class Gemma4ReasoningParser(ReasoningParser):
    """Parser for Gemma 4's channel-based thinking format."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._in_thought = False
        self._in_content = False
        self._saw_any_channel = False

    def reset_state(self):
        super().reset_state()
        self._in_thought = False
        self._in_content = False
        self._saw_any_channel = False

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete output.

        ``enable_thinking`` accepted for cross-parser signature parity
        (#575); Gemma 4 uses unambiguous ``<|channel|>`` tokens so the
        flag is informational only.
        """
        del enable_thinking  # noqa: F841 — channel parser ignores the flag
        if not model_output:
            return None, model_output

        # r5-D finalize-on-truncation (F-DGF-V080-B-7): when the model
        # is cut mid-thought by ``finish_reason="length"`` the closing
        # ``<channel|>`` sentinel never arrives, so ``_THOUGHT_BLOCK``
        # (which only matches CLOSED blocks) misses the in-progress
        # buffer entirely. Pre-fix the "no thinking tags — all content"
        # fall-through returned the raw scratchpad as ``content`` AND
        # — when the engine's token-level OutputRouter ALSO populated
        # ``engine_reasoning_text`` from the same bytes — the route
        # then duplicated those bytes into ``reasoning_content`` for
        # the desktop client (identical 132/128/512-char repro).
        #
        # Detect via ``is_open_in_think`` (shared finalize-on-truncation
        # contract — see ``base.finalize_truncation``) and route the
        # post-opener body as reasoning instead. Strip the opener
        # marker bytes so the reasoning surface is clean.
        if self.is_open_in_think(model_output):
            last_open = model_output.rfind("<|channel>thought")
            before = model_output[:last_open]
            after = model_output[last_open + len("<|channel>thought") :]
            # Strip the leading newline that follows the opener.
            after = after.lstrip("\n")
            # When ``before`` contains a prior CLOSED ``thought`` block
            # (multi-block truncation shape), surface that as reasoning
            # too so we don't lose the earlier reasoning round; then
            # concatenate the trailing unclosed buffer. When ``before``
            # is empty (the common single-block truncation shape) this
            # collapses to ``(after, None)``. ``content`` is None on
            # truncation — the buffer was inside the think tag at EOS,
            # never the user-visible answer.
            prior_reasoning: str | None = None
            if before:
                prior_thought_blocks = _THOUGHT_BLOCK.findall(before)
                if prior_thought_blocks:
                    parts = []
                    for block in prior_thought_blocks:
                        inner = (
                            block.replace("<|channel>thought\n", "")
                            .replace("<channel|>", "")
                            .strip()
                        )
                        if inner:
                            parts.append(inner)
                    if parts:
                        prior_reasoning = "".join(parts)
            trailing_reasoning = after.strip() or None
            merged_reasoning = (
                "\n".join(p for p in (prior_reasoning, trailing_reasoning) if p) or None
            )
            return merged_reasoning, None

        # Extract thought blocks as reasoning
        thought_blocks = _THOUGHT_BLOCK.findall(model_output)
        if not thought_blocks:
            # D-STOP-THINK (cycle-6 F-CORR-2, gemma-4-26b/12b):
            # when ``stop`` matches inside ``<|channel>thought\n...``
            # before the closing ``<channel|>``, the regex above
            # (which requires the closer to match) returns no
            # blocks — and the no-blocks branch then leaks the
            # entire thought trace into ``content`` (only stripping
            # the literal ``<|channel>thought\n`` opener token, not
            # the body). Same shape as the qwen3 / deepseek_r1
            # ``</think>``-never-crossed bug, just expressed in
            # Gemma 4's channel grammar.
            #
            # Detection: ``<|channel>thought`` opener present AND
            # no matching ``<channel|>`` closer downstream. The
            # body from the opener to end-of-text is an unterminated
            # thought trace — route to ``reasoning``, leave content
            # as the pre-opener prefix (typically empty) so the
            # client doesn't see the thought trace in the user-
            # visible answer channel.
            thought_open_idx = model_output.find("<|channel>thought")
            if thought_open_idx >= 0:
                after_opener_idx = thought_open_idx + len("<|channel>thought")
                # Skip the optional newline directly after the opener.
                if (
                    after_opener_idx < len(model_output)
                    and model_output[after_opener_idx] == "\n"
                ):
                    after_opener_idx += 1
                trailing = model_output[after_opener_idx:]
                # Codex round-13 BLOCKING (PR #799): the prior heuristic
                # ``if "<channel|>" not in trailing`` treated ANY later
                # channel closer as closing the thought block, so a
                # malformed-but-plausible
                # ``<|channel>thought\nsecret<|channel>content\nanswer<channel|>``
                # (unterminated thought followed by a content channel)
                # fell through to the "no thinking tags — all content"
                # branch and leaked the thought ``secret`` into
                # ``content``. Fix: locate the NEXT ``<channel|>`` closer
                # AND the NEXT ``<|channel>`` opener — if the next
                # opener arrives BEFORE any closer, the thought block
                # is genuinely unterminated (a new channel started
                # without closing the previous one). Route the bytes
                # between the thought opener and the next opener (or
                # end-of-text) to reasoning.
                next_closer = trailing.find("<channel|>")
                next_opener = trailing.find("<|channel>")
                # Unterminated when either no closer at all, OR the
                # next opener arrives before any closer.
                unterminated = next_closer < 0 or (
                    next_opener >= 0 and next_opener < next_closer
                )
                if unterminated:
                    pre_opener = model_output[:thought_open_idx]
                    # Strip any leading content-channel markers from the
                    # pre-opener prefix so the user-visible content
                    # surface stays clean.
                    pre_cleaned = _CONTENT_START.sub("", pre_opener)
                    pre_cleaned = _CHANNEL_END.sub("", pre_cleaned)
                    pre_cleaned = _TURN_END.sub("", pre_cleaned).strip()
                    # Codex round-13 BLOCKING (PR #799): when the
                    # malformed shape includes a content channel
                    # AFTER the unterminated thought, the bytes
                    # belonging to the content channel must surface
                    # as content (not reasoning). Split ``trailing``
                    # at the next opener if any; everything before
                    # the opener is the thought body (reasoning),
                    # everything from the opener onward is the
                    # downstream channel(s) which we parse with the
                    # standard sub-pattern strippers so the user
                    # still sees the answer.
                    if next_opener >= 0 and (
                        next_closer < 0 or next_opener < next_closer
                    ):
                        reasoning_body = trailing[:next_opener].strip()
                        downstream = trailing[next_opener:]
                        # Codex round-14 BLOCKING (PR #799): parse
                        # downstream channels BY TYPE so a nested
                        # ``<|channel>thought…`` after the unterminated
                        # one is routed to reasoning instead of leaking
                        # into content. Round-13's ``_CONTENT_START.sub``
                        # only stripped ``content|final`` markers — any
                        # body that came under a ``thought`` channel
                        # was passed straight through to ``content_out``
                        # with only the marker removed.
                        downstream_reasoning_parts: list[str] = []
                        downstream_content_parts: list[str] = []
                        # Split downstream into (channel_type, body)
                        # segments. Each segment starts at a
                        # ``<|channel>X`` marker and ends at the next
                        # marker OR end-of-text.
                        for m in _CHANNEL_SEGMENT.finditer(downstream):
                            ch_type = m.group("type")
                            body_start = m.end()
                            # Locate end of this segment: the next
                            # ``<|channel>`` opener (any type), else EOT.
                            next_marker = downstream.find("<|channel>", body_start)
                            seg_end = (
                                next_marker if next_marker >= 0 else len(downstream)
                            )
                            body = downstream[body_start:seg_end]
                            # Strip the segment's own ``<channel|>``
                            # closer + any stray ``<turn|>`` end tokens.
                            body = _CHANNEL_END.sub("", body)
                            body = _TURN_END.sub("", body).strip()
                            if not body:
                                continue
                            if ch_type == "thought":
                                downstream_reasoning_parts.append(body)
                            else:
                                # ``content`` / ``final`` → content surface.
                                downstream_content_parts.append(body)
                        reasoning_full = (
                            (reasoning_body or "")
                            + (
                                ("\n" if reasoning_body else "")
                                + "\n".join(downstream_reasoning_parts)
                                if downstream_reasoning_parts
                                else ""
                            )
                        ).strip() or None
                        downstream_content = (
                            " ".join(downstream_content_parts).strip()
                            if downstream_content_parts
                            else ""
                        )
                        content_out = (
                            (pre_cleaned + " " + downstream_content).strip()
                            if pre_cleaned and downstream_content
                            else (pre_cleaned or downstream_content)
                        )
                        return reasoning_full, content_out or None
                    reasoning_body = trailing.strip()
                    return reasoning_body or None, pre_cleaned or None
            # No thinking tags — all content
            cleaned = _CONTENT_START.sub("", model_output)
            cleaned = _CHANNEL_END.sub("", cleaned)
            cleaned = _TURN_END.sub("", cleaned).strip()
            return None, cleaned

        # Reasoning = thought block contents (strip markers)
        reasoning = ""
        for block in thought_blocks:
            inner = (
                block.replace("<|channel>thought\n", "")
                .replace("<channel|>", "")
                .strip()
            )
            reasoning += inner

        # Content = everything after thought blocks, strip markers
        content = _THOUGHT_BLOCK.sub("", model_output)
        content = _CONTENT_START.sub("", content)
        content = _CHANNEL_END.sub("", content)
        content = _TURN_END.sub("", content).strip()

        return reasoning or None, content or None

    def extract_reasoning_streaming(
        self, previous_text: str, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming delta."""
        if not delta_text:
            return None

        # Snapshot pre-update state so we can detect a thought-to-content
        # flip that happened DURING this delta (issue #219).
        was_in_thought = self._in_thought

        # Track channel state based on accumulated text
        # Check if we just entered thought channel
        if "<|channel>thought" in current_text and not self._in_content:
            self._in_thought = True
            self._saw_any_channel = True

        # Check if we just entered content channel
        if "<|channel>content" in current_text or "<|channel>final" in current_text:
            self._in_thought = False
            self._in_content = True

        # Check if thought ended (first <channel|> after thought start)
        if self._in_thought and "<channel|>" in current_text:
            thought_starts = current_text.count("<|channel>thought")
            channel_ends = current_text.count("<channel|>")
            if channel_ends >= thought_starts:
                self._in_thought = False
                # If no explicit content channel follows, switch to content mode
                if (
                    "<|channel>content" not in current_text
                    and "<|channel>final" not in current_text
                ):
                    self._in_content = True

        # If a thought-to-content flip happened DURING this delta and a
        # state-flipping marker is fully visible in delta_text, split the
        # delta so reasoning bytes that arrived before the marker stay in
        # delta.reasoning instead of being misrouted into delta.content.
        # Pre-fix (#219), the whole-delta classifier below tagged the entire
        # delta as content whenever should_send() flushed a buffered delta
        # straddling the channel transition.
        if was_in_thought and not self._in_thought:
            flip_pos = -1
            for marker in ("<channel|>", "<|channel>content", "<|channel>final"):
                idx = delta_text.find(marker)
                if idx >= 0 and (flip_pos < 0 or idx < flip_pos):
                    flip_pos = idx
            if flip_pos >= 0:
                pre = delta_text[:flip_pos]
                post = delta_text[flip_pos:]
                for m in (
                    "<|channel>",
                    "<channel|>",
                    "<|turn>",
                    "<turn|>",
                    "thought\n",
                    "content\n",
                    "final\n",
                ):
                    pre = pre.replace(m, "")
                    post = post.replace(m, "")
                if pre or post:
                    return DeltaMessage(
                        reasoning=pre if pre else None,
                        content=post if post else None,
                    )

        # Filter out channel markers from delta
        clean = delta_text
        for marker in [
            "<|channel>",
            "<channel|>",
            "<|turn>",
            "<turn|>",
            "thought\n",
            "content\n",
            "final\n",
        ]:
            clean = clean.replace(marker, "")

        if not clean:
            return None  # pure marker token, skip

        if self._in_thought:
            return DeltaMessage(reasoning=clean)
        elif self._in_content:
            return DeltaMessage(content=clean)
        elif not self._saw_any_channel:
            # No channel tokens seen — plain content (no thinking)
            return DeltaMessage(content=clean)
        else:
            # Between channels — treat as reasoning
            return DeltaMessage(reasoning=clean)

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> DeltaMessage | None:
        """Handle end of stream — emit any remaining content.

        ``matched_stop``, ``prompt_thinking_active`` and
        ``finish_reason`` are accepted for API symmetry with the
        ``<think>``-family parsers (PR #799 D-STOP-THINK). Gemma4's
        channel-grammar variant has a separate plug in
        ``extract_reasoning`` so this finalize is a no-op.
        """
        return None

    def is_open_in_think(self, accumulated_text: str) -> bool:
        """r5-D — gemma4 unclosed-thought-channel detection.

        Gemma 4's channel grammar opens with ``<|channel>thought\\n``
        and closes the channel with ``<channel|>``. When
        ``finish_reason="length"`` truncates before the closer arrives
        (max_tokens cut mid-thought), the non-streaming first-pass
        ``extract_reasoning`` runs ``_THOUGHT_BLOCK.findall`` which
        only matches CLOSED thought blocks, finds nothing, and falls
        through to the "all content" branch — leaking the raw
        scratchpad bytes into ``content`` and (when the engine's
        token-level OutputRouter has ALSO populated
        ``engine_reasoning_text``) duplicating the same bytes into
        both ``content`` and ``reasoning_content`` for the desktop
        client (the F-DGF-V080-B-7 132/128/512-char identical-dup
        repro).

        Open-in-think signal:

        * Saw an opener (``<|channel>thought``) AND
        * No closer (``<channel|>``) appears AFTER the latest opener.

        Multi-block resilience: ``rfind`` on the opener and ``find``
        past that position handles both single-block and multi-block
        truncation shapes (closed thought → content channel → second
        thought opener with no closer still reads open-in-think).
        """
        if not accumulated_text:
            return False
        # Codex r1 BLOCKING on PR #825: the pre-fix ``rfind`` gate
        # mis-fired on a legitimate answer that mentions the literal
        # ``<|channel>thought`` substring inside an already-opened
        # content/final channel — ``rfind`` lands on the literal
        # substring inside the answer, the tail is short, and the
        # closer-not-found check spuriously returns True. The
        # extracted ``cleaned_text`` path also tripped the gate
        # because the upstream content opener had already been
        # stripped by ``extract_reasoning`` before the route probed
        # the buffer.
        #
        # Systematic fix — three conservative guards:
        # 1. Buffer must START (modulo whitespace) with
        #    ``<|channel>thought``. This is the only shape where the
        #    EOS-was-mid-think interpretation is unambiguous,
        #    matching the minimax ``lstrip().startswith("<think>")``
        #    guard and the ``_sweep_residual_think_tags`` scope from
        #    PR #722 codex r3.
        # 2. No ``<|channel>content`` / ``<|channel>final`` opener
        #    appears — once content opens, any further
        #    ``<|channel>thought`` bytes are literal answer text in
        #    Gemma 4's single-round grammar.
        # 3. No ``<channel|>`` close after the (only) opener.
        if not accumulated_text.lstrip().startswith("<|channel>thought"):
            return False
        if (
            "<|channel>content" in accumulated_text
            or "<|channel>final" in accumulated_text
        ):
            return False
        last_open = accumulated_text.rfind("<|channel>thought")
        if last_open < 0:
            return False
        return "<channel|>" not in accumulated_text[last_open:]
