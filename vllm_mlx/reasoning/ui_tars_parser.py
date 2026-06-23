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
# Five preamble shapes are recognized (one match per response):
#   1. "Thought: ...\nAction: ..."         (default UI-TARS Computer-Use prompt)
#   2. "Reflection: ...\nAction_Summary: ...\nAction: ..."  (1.5 reflective shape)
#   3. "Action_Summary: ...\nAction: ..."  (1.5 minimal-reflection shape)
#   4. "Thought: ...\n\n<plain-chat answer>" (plain chat lane, blank-line boundary — R6-M1)
#   4b. "Thought: <single-line>$"          (plain chat lane, EOS-only — R6-M1)
#   5. "<think>...</think><plain-chat answer>"  (generic think-tag fallback — R6-M1)
#
# R6-M1 fix (Aki R1, 2026-06-21): pre-fix the regex required ``(?=\s*Action:)``
# so shapes #4/#5 silently failed to match. On the plain chat lane (no
# Computer-Use tool declared, so r5-B's tool-coupled gate skipped the
# action-API sysprompt injection) the model still emitted ``Thought: ...``
# preambles — because the UI-TARS checkpoint is post-trained on the format
# — but the reasoning channel returned ``None``. The decoupling: the
# reasoning emission gate triggers on the parser-detected preamble shape,
# NOT on the presence of any auto-injected sysprompt. The Action: lookahead
# was a coincidence of the C-05 dogfood scenario, not a structural
# requirement.
#
# Boundary semantics:
# - For shape #1/#2/#3 (Action lane): the rest-after-preamble is consumed
#   by the tool parser as ``content``.
# - For shape #4 (plain Thought, blank-line boundary): the boundary is the
#   first ``\n\s*\n`` — bytes before are reasoning, bytes after are content.
# - For shape #4b (plain Thought, EOS only): when the model emitted a
#   single-line ``Thought:`` and nothing else (no follow-up answer), the
#   entire buffer is reasoning. Codex r4 MEDIUM — pre-fix this branch
#   was ``\s*\Z`` with ``re.DOTALL``, which lazy-matched up to EOS even
#   when the body spanned multiple lines of plain prose (e.g.
#   ``"Thought: I should answer directly.\nThe answer is 4."`` was
#   classified as 100 % reasoning, dropping the model's answer). The
#   restricted body ``[^\n]*?`` bans embedded newlines so the EOS branch
#   only fires for genuinely single-line truncated thoughts; multi-line
#   plain prose without a blank-line boundary falls through to "no
#   preamble" and the whole text is routed to content.
# - For shape #5 (think-tag): the boundary is the literal ``</think>``.
#
# Non-greedy bodies prevent a runaway thought trace from swallowing
# legitimate downstream content / action lines.
_PREAMBLE_RE = re.compile(
    r"^\s*"
    r"(?P<thought>"
    # 1+2+3 — Action-lane preambles. Body up to the ``Action:`` lookahead.
    r"(?:"
    r"(?:Thought:\s*(?:.*?))"
    r"|(?:Reflection:\s*(?:.*?)\s*Action_Summary:\s*(?:.*?))"
    r"|(?:Action_Summary:\s*(?:.*?))"
    r")(?=\s*Action:)"
    # 4 — plain-chat ``Thought:`` with blank-line boundary. The bytes
    # before the blank line are reasoning; bytes after are content.
    r"|(?:Thought:\s*(?:.*?))(?=\s*\n\s*\n)"
    # 4b — plain-chat ``Thought:`` ending at EOS with NO embedded
    # newline. Single-line truncated thoughts (e.g. ``"Thought: I'm
    # uncertain."``) surface as reasoning. Multi-line plain prose
    # WITHOUT a blank-line boundary falls through to no-match —
    # ``extract_reasoning`` then routes the entire buffer to content.
    r"|(?:Thought:[^\n]*?)(?=\s*\Z)"
    # 5 — generic ``<think>...</think>`` tag. The tag itself is part of
    # the matched preamble; the post-match content (``model_output[m.end():]``)
    # is the user-facing answer.
    r"|(?:<think>\s*(?:.*?)</think>)"
    r")",
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

    # R8-M6 (2026-06-22): the ``<think>`` tag wrapper is a 5th preamble
    # opener (non-stream shape #5). The streaming state machine treats
    # it like any other opener — the bytes after the tag stream as
    # reasoning until ``</think>`` arrives.
    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"

    # R8-M6: full set of exit predicates that flip the streaming state
    # machine from ``reasoning`` to ``content``. Mirrors the non-stream
    # boundary semantics in ``extract_reasoning`` (shapes #1-#5 and the
    # ``a16d8c8`` plain-chat follow-up):
    #   * ``Action:``  — action-lane boundary (shapes #1/#2/#3)
    #   * ``</think>`` — think-tag wrapper close (shape #5)
    #   * ``Answer:``  — UI-TARS-native plain chat (``Thought:\n\nAnswer:``
    #     observed on UI-TARS plain-chat lane; Sven r8 evidence)
    #   * ``\n\n``     — plain-chat blank-line boundary (shape #4)
    # Order matters only for tie-breaking: the FIRST predicate to appear
    # in the buffer wins, so a delta with both ``\n\n`` and ``Answer:``
    # splits at ``\n\n`` (the blank line is part of the reasoning
    # trailing whitespace; the ``Answer:`` heading goes to content).
    _EXIT_PREDICATES: tuple[str, ...] = (
        "Action:",
        "</think>",
        "Answer:",
        "\n\n",
    )

    # Maximum trailing-byte prefix that could be the start of any exit
    # predicate. We hold back up to ``max_len - 1`` trailing bytes from
    # a reasoning delta when they might be the partial opener so a tag
    # that straddles an SSE chunk boundary is recognized whole instead
    # of pre-leaking the leading bytes (e.g. ``</thi`` then ``nk>``).
    # R8-M6 broadened this from ``Action:`` only to all exit predicates;
    # ``</think>`` is 8 chars so the new max hold is 7.
    _MAX_BOUNDARY_PREFIX = max(len(t) - 1 for t in _EXIT_PREDICATES)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Streaming state: True once we've routed at least one byte to
        # reasoning so the channel is sticky until any exit predicate
        # shows up (``Action:`` / ``</think>`` / ``Answer:`` / ``\n\n``
        # — see ``_EXIT_PREDICATES``).
        self._in_reasoning = False
        self._in_content = False
        # R10-M1 (2026-06-23): per-request ``enable_thinking`` override.
        # When set to ``False`` via ``set_enable_thinking`` the streaming
        # path treats the whole buffer as plain content and skips the
        # preamble-opener detection. Defense in depth so a future
        # postprocessor refactor that calls ``extract_reasoning_streaming``
        # outside the ``_should_route_through_reasoning`` gate still
        # honours the off-flag — and so unit tests can exercise the
        # parser directly without depending on the dispatcher.
        self._enable_thinking: bool | None = None

    def set_enable_thinking(self, enable_thinking: bool | None) -> None:
        """Record the request-level ``enable_thinking`` override.

        R10-M1: called by the streaming dispatcher (postprocessor) at
        request-construction time. ``None`` and ``True`` keep the
        historic preamble-splitting behaviour; ``False`` makes
        ``extract_reasoning_streaming`` route every byte to
        ``delta.content`` until end-of-stream.
        """
        self._enable_thinking = enable_thinking

    # ------------------------------------------------------------------
    # Complete-response API
    # ------------------------------------------------------------------
    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """Split a complete UI-TARS response into (reasoning, content).

        ``enable_thinking`` is honoured: when the client explicitly sets
        ``enable_thinking=False`` (chat_template_kwargs) the parser
        treats the whole buffer as plain content and surfaces it via
        the ``content`` channel — the user asked the model to skip
        thinking and answer directly, so any ``Thought:`` / ``<think>``
        preamble the checkpoint emits anyway (UI-TARS is post-trained
        on the format and frequently ignores the off-flag) MUST NOT be
        moved to ``reasoning_content``. R10-M1 (Mira r10-R1, 2026-06-23):
        the pre-fix path documented the flag as "accepted for protocol
        compatibility but ignored", which made the response shape
        identical with and without the flag — defeating the purpose of
        the override.

        When ``enable_thinking is None`` or ``True`` the parser keeps
        the historic behaviour (run the preamble regex, split into
        reasoning / content). The UI-TARS prompt usually elicits the
        ``Thought:`` preamble; if the request used the Grounding
        template (no preamble, action-only) the regex simply fails to
        match and we return ``(None, model_output)``.
        """
        if not model_output:
            return None, model_output

        # R10-M1: enable_thinking=False bypass — entire buffer is
        # content, no reasoning split. The Action-lane tool parser
        # downstream still extracts ``Action: ...`` lines from the
        # content; this bypass only suppresses the reasoning-channel
        # surfacing of the ``Thought:`` / ``Reflection:`` / ``<think>``
        # prelude.
        if enable_thinking is False:
            return None, model_output

        m = _PREAMBLE_RE.match(model_output)
        if m is None:
            # No preamble — pure-action response (Grounding template) or
            # a malformed output. Surface as content; tool parser will
            # still extract the actions.
            return None, model_output

        raw_thought = m.group("thought")
        # R6-M1: shape #5 captures the ``<think>...</think>`` tag wrapper
        # inside the named group. Strip the structural tags so the
        # reasoning channel surfaces only the human-readable thought,
        # matching how every other reasoning parser in the codebase
        # (qwen3, think_parser, deepseek_r1) handles their own opener
        # tokens.
        thought_text = raw_thought
        if thought_text.lstrip().startswith("<think>"):
            inner = thought_text.lstrip()[len("<think>") :]
            if inner.endswith("</think>"):
                inner = inner[: -len("</think>")]
            thought_text = inner
        thought = thought_text.strip() or None
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

        # R10-M1 (2026-06-23): honour ``enable_thinking=False``. The
        # client opted the request out of reasoning surfacing; route
        # every delta byte to ``content`` and latch ``_in_content`` so
        # the finalize hook doesn't re-flush. ``set_enable_thinking``
        # is called by the dispatcher at request-construction time.
        if self._enable_thinking is False:
            self._in_content = True
            return DeltaMessage(content=delta_text)

        # Already past the preamble — all remaining bytes are content.
        if self._in_content:
            return DeltaMessage(content=delta_text)

        # Discriminate the very first delta(s). We're "in reasoning" as
        # soon as we've seen a preamble opener at offset 0 of the buffer.
        # R8-M6: the ``<think>`` tag wrapper (shape #5) is recognised as
        # a 5th opener here so its body streams through ``delta.reasoning``
        # like every other preamble — pre-fix it leaked verbatim into
        # ``delta.content`` because the streaming state machine ignored
        # the wrapper entirely.
        if not self._in_reasoning:
            stripped = current_text.lstrip()
            if not stripped:
                # All whitespace so far — defer decision until we have
                # a real token.
                return None
            if any(stripped.startswith(p) for p in self._PREAMBLE_OPENERS):
                self._in_reasoning = True
                # Dogfood F-R1-04 follow-up: bytes from PRIOR deltas
                # that we held back (returned ``None`` for) live in
                # ``previous_text``. Now that the opener has resolved,
                # release them through the reasoning channel alongside
                # this delta — otherwise the held prefix bytes would
                # be silently dropped from the SSE stream.
                if previous_text:
                    # Use the slice of ``current_text`` that contains
                    # the full opener-bearing prefix (i.e. everything
                    # the parser has seen) so the emitted reasoning
                    # text starts from the actual model-output start.
                    return DeltaMessage(reasoning=previous_text + delta_text)
                # FALLTHROUGH to the reasoning branch below — first delta
                # of the preamble emits as reasoning.
            elif stripped.startswith(self._THINK_OPEN):
                # R8-M6 shape #5: ``<think>...</think><answer>`` wrapper.
                # Flip to reasoning; the opening tag itself is structural
                # and gets stripped (the model didn't intend to surface
                # the literal ``<think>`` token in either channel). The
                # body bytes between the open and close tag stream as
                # reasoning; the close tag below acts as an exit predicate
                # that flips to content.
                self._in_reasoning = True
                # Find where the opener ends in current_text and slice
                # everything after as reasoning. ``previous_text`` and
                # ``delta_text`` together make up ``current_text``; the
                # delta may carry the whole opener, the tail of the
                # opener, or post-opener bytes only.
                think_idx = current_text.find(self._THINK_OPEN)
                after_open = think_idx + len(self._THINK_OPEN)
                delta_start_in_current = len(previous_text)
                # If the post-opener slice extends into this delta, emit
                # it (and check for the close tag exit on this same
                # delta). If the delta is entirely inside the opener
                # tag, emit nothing this round.
                if after_open >= len(current_text):
                    # All bytes seen so far are still within the opener
                    # tag (or just up to its end with no body yet).
                    return None
                # Drop tag bytes; reasoning slice starts at the higher
                # of ``after_open`` (post-tag start) and
                # ``delta_start_in_current`` (this delta's start in the
                # buffer).
                reason_start_in_current = max(after_open, delta_start_in_current)
                reason_in_current = current_text[reason_start_in_current:]
                # Now check whether the close-tag exit also lives in
                # this buffer slice — if so split at the close tag and
                # flip to content; else emit the whole slice as reasoning
                # (the close-tag exit will fire on a later delta) modulo
                # the trailing partial-tag hold-back.
                exit_pos, exit_tok = self._find_first_exit(current_text)
                if exit_pos != -1 and exit_pos >= reason_start_in_current:
                    # Close-tag (or other exit) lands inside the slice
                    # we're about to emit. Split.
                    reasoning_part = current_text[reason_start_in_current:exit_pos]
                    content_part = self._content_after_exit(
                        current_text, exit_pos, exit_tok
                    )
                    self._in_content = True
                    return DeltaMessage(
                        reasoning=reasoning_part or None,
                        content=content_part or None,
                    )
                # No exit yet; reasoning slice survives. Apply trailing
                # partial-exit hold-back so we don't pre-leak a
                # straddling ``</thi`` etc.
                held = self._compute_partial_exit_hold(current_text)
                safe_end_in_current = len(current_text) - held
                if safe_end_in_current <= reason_start_in_current:
                    return None
                emit = current_text[reason_start_in_current:safe_end_in_current]
                if not emit:
                    return None
                return DeltaMessage(reasoning=emit)
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
                # Buffer doesn't yet match any opener. We need to wait
                # until ``stripped`` is either long enough that NO
                # opener prefix can match anymore, OR ``stripped`` is
                # not even a prefix of any opener — only then can we
                # safely flip to content.
                #
                # Dogfood F-R1-04 fix (Ana 2026-06-21): the previous
                # heuristic ``len(stripped) >= len("Action:")`` was 7
                # chars, but ``"Thought"`` is ALSO 7 chars — and at
                # exactly that length the buffer is still the strict
                # prefix of ``"Thought:"`` (waiting for the colon on
                # the next delta). The old code flipped to content
                # there and leaked ``Thought: ...`` into
                # ``delta.content`` instead of routing it to the
                # reasoning channel. Same hazard for ``"Reflection"``
                # (10 chars, prefix of ``Reflection:``) and
                # ``"Action_Summa"`` (12 chars, prefix of
                # ``Action_Summary:``).
                #
                # Correct gate: flip to content ONLY when ``stripped``
                # is NOT a prefix of any known opener AND not a prefix
                # of ``"Action:"`` (which means it's regular content).
                # Otherwise keep buffering — the next delta will land
                # the colon and the opener-match branch above fires.
                # R8-M6: include ``<think>`` in the prefix set so a
                # split-opener (e.g. ``<thi`` then ``nk>``) is held
                # back instead of leaking to content.
                _OPENERS_PLUS_BOUNDARIES = self._PREAMBLE_OPENERS + (
                    "Action:",
                    self._THINK_OPEN,
                )
                still_could_be_opener = any(
                    op.startswith(stripped) for op in _OPENERS_PLUS_BOUNDARIES
                )
                if still_could_be_opener:
                    # Hold off — the next delta might complete the
                    # opener (e.g. ``"Thought"`` → ``"Thought:"``).
                    return None
                # ``stripped`` is no longer a prefix of any opener —
                # this is regular content. Flip to content and flush
                # any previously held bytes alongside this delta.
                self._in_content = True
                if previous_text:
                    return DeltaMessage(content=previous_text + delta_text)
                return DeltaMessage(content=delta_text)

        # In reasoning channel. Look for the FIRST exit predicate
        # (``Action:`` / ``</think>`` / ``Answer:`` / ``\n\n``) inside
        # this delta to decide whether to split. R8-M6: pre-fix only
        # ``Action:`` was honoured; the plain-chat and ``<think>``
        # boundary forms leaked the post-boundary answer into
        # ``delta.reasoning``.
        exit_pos, exit_tok = self._find_first_exit(current_text)
        if exit_pos == -1:
            # No full exit predicate yet. The trailing bytes of
            # current_text MIGHT be the partial leading edge of one
            # (e.g. delta ``Action`` then later ``:``, or ``</thi`` then
            # ``nk>``, or a single ``\n`` that might become ``\n\n``).
            # Hold any tail bytes back so the tool parser / content
            # channel receives the complete sentinel once the boundary
            # forms, instead of seeing the leading bytes leak into the
            # reasoning channel.
            #
            # Bookkeeping: track which bytes have already been emitted
            # on prior deltas. We've emitted everything up to
            # ``len(previous_text) - prev_held``. New emit window is
            # ``[emitted_so_far, safe_end)`` where ``safe_end`` is
            # ``len(current_text) - held``.
            prev_held = (
                self._compute_partial_exit_hold(previous_text) if previous_text else 0
            )
            held = self._compute_partial_exit_hold(current_text)
            emitted_so_far = len(previous_text) - prev_held
            safe_end = len(current_text) - held
            if safe_end <= emitted_so_far:
                return None
            return DeltaMessage(reasoning=current_text[emitted_so_far:safe_end])

        # Exit predicate is in current_text. Split around it.
        #
        # Bookkeeping: on prior deltas we may have HELD back trailing
        # bytes of ``previous_text`` because they looked like a partial
        # exit predicate (e.g. a single ``\n`` that might have been the
        # start of ``\n\n``, or ``Acti`` that might have been the start
        # of ``Action:``). Those bytes were not emitted on the prior
        # delta. Now that we know what the real exit predicate is, we
        # need to:
        #   * RECOVER held bytes that fall BEFORE the exit position
        #     and emit them as reasoning (they were reasoning content
        #     that got optimistically held).
        #   * PREPEND held bytes that fall INSIDE the exit token to the
        #     content side for in-place sentinels (``Action:``,
        #     ``Answer:``), so the tool parser sees the complete token.
        prev_held = (
            self._compute_partial_exit_hold(previous_text) if previous_text else 0
        )
        already_emitted_reasoning_end = len(previous_text) - prev_held
        delta_start_in_current = len(previous_text)
        token_start = exit_pos
        token_end = exit_pos + len(exit_tok)

        # Reasoning side: bytes from the last-emitted position up to the
        # exit token. May span across the previous_text held region AND
        # the head of this delta.
        if token_start > already_emitted_reasoning_end:
            reasoning_part = current_text[already_emitted_reasoning_end:token_start]
        else:
            # Exit token starts inside or before the already-emitted
            # region — no extra reasoning bytes to recover.
            reasoning_part = ""

        # Content side: for structural exits (``</think>``, ``\n\n``)
        # drop the token bytes entirely. For in-place sentinels
        # (``Action:`` / ``Answer:``) keep them so the tool parser sees
        # the full token.
        if exit_tok in ("</think>", "\n\n"):
            content_part = current_text[token_end:]
        else:
            content_part = current_text[token_start:]

        # ``content_part`` is the full post-boundary tail of the
        # current_text buffer. Bytes from prior deltas in that tail
        # have NOT been emitted (we were in reasoning channel until
        # this delta), so we emit them whole — no slicing needed.

        self._in_content = True
        return DeltaMessage(
            reasoning=reasoning_part or None,
            content=content_part or None,
        )

    def _find_first_exit(self, buffer: str) -> tuple[int, str]:
        """Find the position of the FIRST exit predicate in ``buffer``.

        Returns ``(position, token)`` of the earliest match across
        ``_EXIT_PREDICATES``. Returns ``(-1, "")`` if no exit predicate
        is present. Tie-breaking on equal positions favours the LONGER
        token (so ``Answer:`` wins over a degenerate empty match — the
        actual predicates don't overlap so ties are rare in practice).
        """
        best_pos = -1
        best_tok = ""
        for tok in self._EXIT_PREDICATES:
            pos = buffer.find(tok)
            if pos == -1:
                continue
            if best_pos == -1 or pos < best_pos:
                best_pos = pos
                best_tok = tok
            elif pos == best_pos and len(tok) > len(best_tok):
                best_tok = tok
        return best_pos, best_tok

    def _content_after_exit(self, buffer: str, exit_pos: int, exit_tok: str) -> str:
        """Return the bytes of ``buffer`` that belong on the content side
        after the exit predicate at ``exit_pos`` of token ``exit_tok``.

        For structural separators (``</think>`` / ``\n\n``) the token
        bytes are dropped. For in-place sentinels (``Action:`` /
        ``Answer:``) the token bytes are kept so downstream parsers see
        the full sentinel.
        """
        if exit_tok in ("</think>", "\n\n"):
            return buffer[exit_pos + len(exit_tok) :]
        return buffer[exit_pos:]

    def _compute_partial_exit_hold(self, current_text: str) -> int:
        """Return the number of trailing bytes that might be a partial
        exit predicate.

        Specifically: the longest non-empty suffix ``current_text[-k:]``
        (1 ≤ k ≤ ``_MAX_BOUNDARY_PREFIX``) that matches a strict prefix
        of ANY token in ``_EXIT_PREDICATES``. Returns 0 if no such
        partial overlap exists.

        Example: current_text ends with ``...thing.\nAction`` — the
        trailing 6 bytes ``Action`` match the 6-char prefix of
        ``Action:``, so we hold those 6 bytes back. The next delta brings
        ``:``, the full ``Action:`` resolves, and the held bytes are
        released to content alongside.

        R8-M6 broadened this from ``Action:`` only to all exit
        predicates so a ``</thi`` straddle for shape #5 (or a single
        ``\n`` straddle for shape #4's ``\n\n`` boundary) is held
        instead of leaking the leading bytes into ``delta.reasoning``.
        """
        for k in range(self._MAX_BOUNDARY_PREFIX, 0, -1):
            if k > len(current_text):
                continue
            suffix = current_text[-k:]
            for tok in self._EXIT_PREDICATES:
                if tok.startswith(suffix):
                    return k
        return 0

    # ------------------------------------------------------------------
    # Lifecycle hook used by the streaming dispatcher between requests.
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        self._in_reasoning = False
        self._in_content = False
        # R10-M1: clear the per-request override so a re-used parser
        # instance doesn't carry the prior request's flag into a fresh
        # stream. The dispatcher calls ``set_enable_thinking`` after
        # this reset, so a no-op default (None) is the correct
        # post-reset state.
        self._enable_thinking = None

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> DeltaMessage | None:
        """Flush held opener-prefix bytes at end-of-stream.

        Codex r5 BLOCKING: the opener-prefix hold-back loop returns
        ``None`` when the buffer is a strict prefix of any known
        opener (``"Thought"``, ``"Reflection"``, etc.) — waiting for
        the disambiguating colon on a future delta. But if the
        stream ENDS while a prefix is still held (model produced
        plain text ``"Thought"`` with no colon, or got cut off
        mid-token), those bytes are silently dropped.

        At end-of-stream:
        - If we never flipped channels (``_in_reasoning ==
          _in_content == False``), every byte the model emitted is
          still held. Flush as ``content`` — by definition the
          buffer is NOT a complete opener (else the live loop
          would have flipped to reasoning), so the held bytes are
          plain text content.
        - If we're already in a channel, the live loop has already
          emitted everything; nothing to flush.

        The D-STOP-THINK keyword arguments are accepted for the shared
        parser interface. UI-TARS does not use those signals because its
        channel evidence comes from literal ``Thought:`` / ``Action:``
        boundaries.
        """
        if self._in_reasoning or self._in_content:
            return None
        if not accumulated_text:
            return None
        # All emitted as content — the in-flight loop never
        # disambiguated to an opener, so by end-of-stream this is
        # plain content. Flip the latch so a subsequent finalize
        # call is a no-op.
        self._in_content = True
        return DeltaMessage(content=accumulated_text)
