# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

This module provides BaseThinkingReasoningParser, a concrete implementation
for extracting reasoning content from models that use thinking tags.

Supports three scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. No tags: pure content
"""

from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base parser for models using <think>...</think> style tags.

    This parser handles the common pattern where reasoning content is wrapped
    in special tags. Subclasses define the specific start and end tokens.

    Supports "implicit reasoning mode" where <think> is injected in the prompt
    and only </think> appears in the model output. This is common with AI agents
    like OpenCode that force models to reason by injecting thinking tags.

    The parser tracks state during streaming to correctly separate reasoning
    from content as tokens arrive incrementally.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._saw_any_tag = False
        # SSE-boundary withhold (PR #715 bundle, fuzz finding C): the
        # number of trailing bytes of ``previous_text`` that we held back
        # on the prior delta because they looked like a partial tag
        # prefix. Used to flush them on the next delta when the prefix
        # turned out NOT to be a tag.
        self._held_tag_suffix_len = 0

    def reset_state(self):
        """Reset state for a new streaming request."""
        super().reset_state()
        self._saw_any_tag = False
        self._held_tag_suffix_len = 0

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Handles four cases:
        1. Both tags present: <think>reasoning</think>content
        2. Only closing tag: reasoning</think>content (think in prompt)
        3. Only start tag: <think>reasoning... (incomplete reasoning, no end yet)
        4. No tags at all: the routing depends on ``enable_thinking``.

        Case 4 — the implicit-thinking path — is the load-bearing
        addition for #575. Qwen3 chat templates that pre-inject
        ``<think>\\n`` into the prompt itself (see
        ``vllm_mlx/utils/chat_templates`` for the family list)
        emit only the **closing** ``</think>`` in the model output;
        when the response is truncated mid-thought (``finish_reason
        == "length"``) it emits *neither* tag — and the entire
        thought trace would leak to ``content`` if Case 4 stayed
        unconditional. Round 2 of the 2026-06-14 autoresearch sweep
        observed this on qwen3.5-4b and qwen3.6-35b at every budget
        from 2 K to 16 K tokens.

        Fix: when the request set ``enable_thinking=True`` AND
        neither tag is present, treat the whole output as reasoning
        — symmetric with the streaming path
        (``extract_reasoning_streaming``) which already uses Case-3
        "haven't seen </think> yet → reasoning" semantics. When
        ``enable_thinking`` is None / False, behaviour is unchanged
        and the output flows to ``content`` exactly as before.

        Args:
            model_output: Complete model output text.
            enable_thinking: Whether the request set
                ``chat_template_kwargs.enable_thinking=True``. ``None``
                preserves pre-#575 behaviour (Case 4 → content); this
                lets callers that don't know the thinking state opt
                out of the symmetric-with-streaming path. Threaded
                through ``_finalize_content_and_reasoning``.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        text = model_output

        # Case 1: Both tags present (normal case)
        if self.start_token in text and self.end_token in text:
            # Get everything after start token
            _, _, after_start = text.partition(self.start_token)
            # Split on end token
            reasoning, _, content = after_start.partition(self.end_token)
            # Sweep any residual think blocks the naive first-pair
            # partition left in ``content``. The first-pair split
            # consumes ONLY the first ``<think>…</think>`` block, so
            # multi-block outputs (phi-4-mini-reasoning emits a second
            # block after the answer when ``reasoning_max_tokens``
            # truncates mid-thought — 2026-06-19 round-1 fuzz repro)
            # would otherwise ship the trailing ``<think>`` or
            # ``</think>`` literal bytes through to ``message.content``.
            reasoning, content = self._sweep_residual_think_tags(reasoning, content)
            return reasoning.strip() or None, content.strip() or None

        # Case 2: Only closing tag (think was injected in prompt)
        # Everything before </think> is reasoning
        if self.end_token in text:
            reasoning, _, content = text.partition(self.end_token)
            # Same multi-block sweep as Case 1 — Case 2 hits when the
            # chat template pre-injected ``<think>`` and the model
            # emitted ``</think>answer<think>more</think>`` (the
            # implicit-think analogue of the phi-4-mini-reasoning
            # repro).
            reasoning, content = self._sweep_residual_think_tags(reasoning, content)
            return reasoning.strip() or None, content.strip() or None

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            return reasoning.strip() or None, None

        # Case 4: No tags at all. With ``enable_thinking=True`` the
        # chat template already injected ``<think>`` into the
        # prompt — anything we see is the model's continuation of
        # the thought trace, NOT user-visible content. Route to
        # reasoning; ``content`` stays None so the empty assistant
        # bubble doesn't ship a wall of meta-cognition to the UI.
        # See #575.
        if enable_thinking is True:
            return model_output.strip() or None, None
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta using text-based detection.

        Handles implicit reasoning mode where <think> was in the prompt
        and only </think> appears in the output.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Skip if delta is just the special tokens themselves
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            return None
        if stripped_delta == self.end_token:
            return None

        # Check token positions in text (stateless text-based detection)
        start_in_prev = self.start_token in previous_text
        start_in_current = self.start_token in current_text
        end_in_prev = self.end_token in previous_text
        end_in_delta = self.end_token in delta_text

        # Case 1: Explicit <think> found in text - standard behavior
        if start_in_current:
            self._saw_any_tag = True
            return self._handle_explicit_think(
                previous_text,
                current_text,
                delta_text,
                start_in_prev,
                end_in_prev,
                end_in_delta,
            )

        # Case 2: No <think> but </think> found - implicit reasoning mode
        # This handles when <think> was injected in the prompt
        if self.end_token in current_text:
            self._saw_any_tag = True
            return self._handle_implicit_think(delta_text, end_in_prev, end_in_delta)

        # Case 3: No think tags seen yet
        # We can't know if <think> was in the prompt, so we must make a choice:
        # - Treat as content (safe, but loses reasoning if think was in prompt)
        # - Treat as reasoning (risky, wrong if no thinking at all)
        # We choose to treat as reasoning IF we haven't seen </think> yet,
        # because if think was in prompt, we want to capture the reasoning.
        # This will be corrected once </think> is seen.
        #
        # SSE-boundary withhold (PR #715 bundle, fuzz finding C): when the
        # model emits the literal ``<think>`` open tag autonomously (phi-4-
        # mini-reasoning / nanbeige4.1 family), the tag can be split across
        # SSE chunk boundaries (e.g. delta=``<thi`` then ``nk>``). Without
        # the withhold, the partial ``<thi`` would land in
        # ``reasoning_content`` and the trailing ``nk>`` would fall through
        # the next-tick ``_handle_explicit_think`` fallback into
        # ``content``, leaving the client with a visibly mangled response
        # (live-fuzz repro: ``content=">\n", reasoning="<thinkOkay..."``).
        #
        # Strategy: ``self._held_tag_suffix_len`` records how many trailing
        # bytes of ``previous_text`` we withheld on the prior delta. The
        # bytes already emitted from ``current_text`` are everything
        # except the last ``self._held_tag_suffix_len`` bytes of
        # ``previous_text``. On this delta, we compute the new partial-tag
        # suffix in ``current_text`` and emit the difference (i.e. the
        # bytes that have moved out of the partial-tag region).
        prev_held = self._held_tag_suffix_len
        held = self._held_partial_tag_len(current_text)
        # Position in current_text up to which we've already emitted.
        emitted_so_far = len(previous_text) - prev_held
        # Position in current_text up to which we can safely emit now.
        safe_end = len(current_text) - held
        self._held_tag_suffix_len = held
        if safe_end <= emitted_so_far:
            # Nothing new safe to emit yet — the whole delta (and possibly
            # some of previous_text's held bytes) is still in the partial-
            # tag region. Wait for the next delta.
            return None
        emit = current_text[emitted_so_far:safe_end]
        if not emit:
            return None
        return DeltaMessage(reasoning=emit)

    def _sweep_residual_think_tags(
        self, reasoning: str, content: str
    ) -> tuple[str, str]:
        """Strip any residual ``<think>…</think>`` blocks left in
        ``content`` after the first-pair partition, and reroute
        unclosed trailing thoughts into ``reasoning``.

        2026-06-19 round-1 fuzz repro (phi-4-mini-reasoning-4bit):
        when the model emits two ``<think>`` blocks (an inner
        closed one before the answer and a SECOND opener after the
        answer that ``reasoning_max_tokens`` / ``max_tokens``
        truncates before its closing tag), the naive
        ``partition(<think>)`` + ``partition(</think>)`` in
        ``extract_reasoning`` Case 1 consumes ONLY the first pair.
        The trailing ``<think>thought2…`` opener was leaking
        verbatim into ``message.content`` — neither
        ``strip_thinking_tags`` (matches closed blocks only) nor
        ``sanitize_output`` (matches a stray ``</think>`` only)
        catch an orphan ``<think>`` opener, so the tag bytes
        survived all the way to the wire.

        Sweep algorithm (operates on ``content`` only — ``reasoning``
        is already authoritative from the first partition):

        * Iteratively strip any complete ``<think>…</think>`` block
          and accumulate its body into ``reasoning`` (preserves
          ordering: appears AFTER the first thought, which matches
          the model's emission order).
        * If a trailing unclosed ``<think>…`` remains, append its
          body to ``reasoning`` and drop everything from the
          opener onward from ``content`` (matches the Case 3
          "no end tag → reasoning" semantics).
        * Strip any orphan ``</think>`` left after step 1 — these
          appear when the model emits a stray closer with no
          matching opener (or when downstream regex passes have
          already eaten the opener but left the closer).

        The fix lives in the base class so every thinking-tag
        parser (DeepSeek-R1, Qwen3, VibeThinker, …) benefits
        without per-subclass duplication — the user's directive
        was explicit on this. ``GLM4`` / ``Minimax`` / ``Gemma4``
        parsers do not subclass ``BaseThinkingReasoningParser`` so
        their wire formats are unaffected; they have their own
        sweepers where the wire grammar requires them.
        """
        if not content:
            return reasoning, content
        # Step 1: strip closed blocks left-to-right, accumulating
        # their bodies into ``reasoning`` in emission order. Cap the
        # loop at the actual occurrence count so a pathological
        # input (no further tags) doesn't loop. ``find`` returns -1
        # on miss so the ``while`` exits cleanly.
        max_passes = content.count(self.start_token) + 1
        passes = 0
        while passes < max_passes:
            start_idx = content.find(self.start_token)
            if start_idx < 0:
                break
            after_start = content[start_idx + len(self.start_token) :]
            end_idx = after_start.find(self.end_token)
            if end_idx < 0:
                # Unclosed trailing ``<think>…`` — append the body
                # to reasoning and truncate content at the opener.
                trailing_reasoning = after_start.rstrip()
                if trailing_reasoning:
                    reasoning = (
                        (reasoning.rstrip() + "\n" + trailing_reasoning)
                        if reasoning
                        else trailing_reasoning
                    )
                content = content[:start_idx].rstrip()
                break
            # Closed block: pull body into reasoning, splice the
            # block out of content.
            block_body = after_start[:end_idx]
            if block_body:
                reasoning = (
                    (reasoning.rstrip() + "\n" + block_body)
                    if reasoning
                    else block_body
                )
            block_end = (
                start_idx + len(self.start_token) + end_idx + len(self.end_token)
            )
            content = content[:start_idx] + content[block_end:]
            passes += 1
        # Step 2: strip any orphan ``</think>`` closers left in
        # content. These would otherwise survive
        # ``sanitize_output``'s stray-closer strip ONLY if the input
        # is empty after stripping (the helper collapses empty
        # results to ``None``), so belt-and-braces here keeps the
        # bytes off the wire when content remains non-empty.
        if self.end_token in content:
            content = content.replace(self.end_token, "")
        return reasoning, content

    def _held_partial_tag_len(self, current_text: str) -> int:
        """Length of the suffix of ``current_text`` that could be a strict
        prefix of ``start_token`` or ``end_token``.

        Used by the Case-3 SSE-boundary withhold (see
        ``extract_reasoning_streaming``). Returns the LONGEST matching
        prefix length so a ``<thin`` suffix holds back all 4 chars (the
        next delta might be ``k>`` completing ``<think>``).

        Excludes the full-match case — if ``current_text`` already ends
        with the complete ``start_token`` / ``end_token`` we don't need
        to withhold, the regular Case-1 / Case-2 branches will pick it
        up on the next pass.
        """
        for tag in (self.start_token, self.end_token):
            # Search from longest possible prefix down to 1 char so the
            # LONGEST partial-tag suffix wins.
            max_len = min(len(tag) - 1, len(current_text))
            for n in range(max_len, 0, -1):
                if current_text.endswith(tag[:n]):
                    return n
        return 0

    def _handle_explicit_think(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        start_in_prev: bool,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> tag is explicitly in the output."""
        start_in_delta = self.start_token in delta_text

        if start_in_prev:
            # We're after the start token. Use emit-by-position
            # bookkeeping uniformly so held partial-tag bytes from
            # previous deltas (PR #715 bundle, fuzz finding C — codex
            # r1 P2 follow-up) are correctly flushed whether the end
            # tag arrives in this delta, straddles it, or is still
            # pending.
            #
            # ``emitted_so_far`` is the position in ``current_text``
            # up through which we've already emitted reasoning bytes
            # on prior deltas. We compute it from the position right
            # after the start tag (so the ``<think>`` opener bytes
            # are never counted) PLUS however many bytes after the
            # opener were already emitted on prior deltas (i.e.
            # previous_text minus the held-suffix region). When the
            # opener completed on the previous delta, previous_text
            # ends exactly at or before the end of ``<think>`` and the
            # ``max()`` keeps emitted_so_far at the post-opener start.
            start_idx_cur = current_text.find(self.start_token)
            after_start_in_current = start_idx_cur + len(self.start_token)
            prev_held = self._held_tag_suffix_len
            already_emitted_after_opener = max(
                0, len(previous_text) - prev_held - after_start_in_current
            )
            emitted_so_far = after_start_in_current + already_emitted_after_opener
            if end_in_prev:
                # We're past the FIRST ``</think>`` — but the model
                # may have re-entered reasoning by emitting another
                # ``<think>`` after the answer (the 2026-06-19
                # round-1 fuzz repro on phi-4-mini-reasoning-4bit
                # emits 6–7 think blocks across a single 2 K-token
                # response when ``reasoning_max_tokens`` truncates
                # the first one). Delegate to a multi-block-aware
                # router so subsequent ``<think>…</think>`` pairs
                # are correctly split instead of leaking the
                # literal tag bytes into ``content`` via the prior
                # "all delta = content" fallback.
                #
                # Codex r1 BLOCKING on PR #722: do NOT clear
                # ``_held_tag_suffix_len`` here. The held suffix
                # from the prior chunk encodes a partial tag that
                # may STRADDLE into this delta (e.g. prev ended
                # with ``A<thi``, delta is ``nk>R``) — the router
                # needs to see the held value to back its scan
                # window up by that many bytes and recognise the
                # completed straddle. The router resets the held
                # value itself based on this delta's trailing
                # partial-tag suffix.
                return self._handle_multi_block_after_close(
                    previous_text, current_text, delta_text
                )
            # End tag may be in current_text (delta or straddle) or
            # still pending.
            end_idx_cur = current_text.find(self.end_token)
            if end_idx_cur >= 0:
                # End tag is complete in current_text. Emit all
                # un-emitted reasoning bytes up to the end tag, then
                # any post-tag bytes as content.
                self._held_tag_suffix_len = 0
                # Reasoning portion: everything from emitted_so_far up
                # to the start of the end tag, but clipped so we don't
                # re-emit prefix bytes that were part of the
                # ``<think>`` opener (when start_in_prev is True the
                # opener has already been consumed; if held bytes
                # were ALSO consumed as part of the now-complete end
                # tag, those bytes must be excluded from reasoning).
                reasoning_part = current_text[emitted_so_far:end_idx_cur]
                content_part = current_text[end_idx_cur + len(self.end_token) :]
                # ``content_part`` includes everything after the end
                # tag in current_text — but only the portion in
                # ``delta_text`` is new. Anything from ``previous_text``
                # past the end tag was already emitted on a prior
                # delta. Slice to keep only the new content bytes.
                prev_len = len(current_text) - len(delta_text)
                content_start_in_current = end_idx_cur + len(self.end_token)
                if content_start_in_current < prev_len:
                    content_part = content_part[prev_len - content_start_in_current :]
                # ``reasoning_part`` may be empty if the held bytes
                # turned out to be the start of the end tag (e.g. we
                # held ``</thi`` and now see ``nk>``); in that case
                # ``emitted_so_far`` already passes ``end_idx_cur``
                # and reasoning_part is empty — OK.
                if end_idx_cur < emitted_so_far:
                    reasoning_part = ""
                return DeltaMessage(
                    reasoning=reasoning_part or None,
                    content=content_part or None,
                )
            # End tag not yet in current_text. Withhold any trailing
            # partial-tag suffix so the next delta can complete it.
            held = self._held_partial_tag_len(current_text)
            safe_end = len(current_text) - held
            self._held_tag_suffix_len = held
            if safe_end <= emitted_so_far:
                return None
            emit = current_text[emitted_so_far:safe_end]
            if not emit:
                return None
            return DeltaMessage(reasoning=emit)

        elif start_in_delta:
            # Start token is in this delta
            start_idx = delta_text.find(self.start_token)

            if end_in_delta:
                # Both tokens in this delta. Use the multi-block router
                # so a delta carrying ``<think>R1</think>A<think>R2</think>B…``
                # (the model emitted multiple ``<think>`` blocks in a
                # single SSE chunk — observed on small thinking models
                # with low ``stream_interval`` settings or fast-batched
                # output) splits cleanly without leaking the literal
                # tag bytes of the second-and-later blocks into
                # ``content``. 2026-06-19 round-1 fuzz follow-on.
                # Pre-fix the single-pair partition below consumed only
                # the FIRST opener/closer and emitted the rest as
                # content verbatim.
                return self._handle_multi_block_after_close(
                    previous_text, current_text, delta_text
                )
            else:
                # Only start token - beginning of reasoning
                reasoning_part = delta_text[start_idx + len(self.start_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None
                )

        # SSE-boundary recovery (PR #715 bundle, fuzz finding C): the
        # start_token straddles ``previous_text`` and ``delta_text`` —
        # ``start_in_current=True`` but neither ``start_in_prev`` nor
        # ``start_in_delta`` is True. The Case-3 withhold in
        # ``extract_reasoning_streaming`` already held the matching
        # suffix of the previous delta, so we ONLY need to emit the
        # portion of ``delta_text`` that lands AFTER the now-complete
        # start_token. Pre-withhold (without this branch) the trailing
        # bytes of the tag (e.g. ``nk>``) would fall through to the
        # old ``return DeltaMessage(content=delta_text)`` fallback and
        # leak literally into ``content`` — the original live-fuzz
        # bug shape on phi-4-mini-reasoning / nanbeige4.1.
        #
        # Codex r2 P2 follow-up: clear ``_held_tag_suffix_len`` after
        # consuming the straddle. The held bytes were part of the
        # start tag (not pending reasoning) and on the NEXT delta the
        # ``start_in_prev`` branch's emit-by-position bookkeeping uses
        # the held value to compute ``already_emitted_after_opener``.
        # Leaving it non-zero would cause the bookkeeping to treat
        # the just-emitted reasoning bytes as "still un-emitted" and
        # re-emit them, duplicating the streamed reasoning (codex
        # caught this with the ``['<thi', 'nk>Okay', ' more']`` repro).
        start_idx_cur = current_text.find(self.start_token)
        prev_len = len(current_text) - len(delta_text)
        # Bytes of delta_text BEFORE the start_token's last char —
        # these are the tail of an in-progress tag whose head sits in
        # previous_text. They're tag bytes, not user-visible text;
        # drop them.
        after_start_in_current = start_idx_cur + len(self.start_token)
        # Number of delta_text chars that fall before the end of the
        # start tag (these are tag chars, drop them).
        tag_overlap = max(0, after_start_in_current - prev_len)
        reasoning_part = delta_text[tag_overlap:] if delta_text else ""
        if end_in_delta:
            # End tag also lands in this delta — split.
            end_idx = reasoning_part.find(self.end_token)
            if end_idx >= 0:
                content_part = reasoning_part[end_idx + len(self.end_token) :]
                reasoning_part = reasoning_part[:end_idx]
                self._held_tag_suffix_len = 0
                return DeltaMessage(
                    reasoning=reasoning_part or None,
                    content=content_part or None,
                )
        # Codex r4 P2 follow-up: when the same chunk that completes a
        # split opener ALSO ends with a partial ``</think>`` (e.g.
        # chunks ``"<thi"``, ``"nk>OK</thi"``, ``"nk>ans"``), we must
        # withhold the trailing partial-end-tag bytes so they don't
        # leak literally into ``reasoning_content``. Reuse the same
        # ``_held_partial_tag_len`` machinery as the Case-3 / start_in_prev
        # branches so the next delta's ``start_in_prev`` path can
        # complete the close cleanly.
        held = self._held_partial_tag_len(current_text)
        self._held_tag_suffix_len = held
        if held > 0 and reasoning_part:
            # Strip the trailing partial-tag bytes from this emit.
            # The withhold is measured on ``current_text``; convert to
            # a slice on ``reasoning_part`` (which is the suffix of
            # delta_text after the opener overlap).
            safe_end_in_current = len(current_text) - held
            # Position in current_text where reasoning_part starts:
            reasoning_start_in_current = prev_len + tag_overlap
            keep = max(0, safe_end_in_current - reasoning_start_in_current)
            reasoning_part = reasoning_part[:keep]
        return DeltaMessage(reasoning=reasoning_part or None)

    def _handle_multi_block_after_close(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Route a delta that arrives AFTER the first ``</think>``.

        2026-06-19 round-1 fuzz repro (phi-4-mini-reasoning-4bit):
        the model re-enters reasoning after the answer by emitting a
        SECOND ``<think>`` block (and may do this many times before
        ``max_tokens`` hits). The pre-fix streaming path emitted the
        whole post-close delta as ``content``, leaking every
        subsequent ``<think>`` / ``</think>`` literal to the wire.

        Multi-block streaming rule:

        * Determine the current phase from ``<think>`` / ``</think>``
          counts in ``current_text``. Equal counts → CONTENT. One
          excess opener → REASONING (inside an unclosed block).
        * Compute the position in ``current_text`` where the LAST
          phase transition happened (last ``<think>`` for a
          REASONING phase, last ``</think>`` for a CONTENT phase).
        * Emit only the bytes in ``delta_text`` that lie in the
          current phase. Bytes from ``delta_text`` that span a
          phase boundary are split — pre-boundary goes to the prior
          phase, post-boundary to the new one.
        * Withhold any trailing partial-tag suffix so a tag that
          straddles SSE chunks gets recovered on the next delta
          (same machinery as the Case-3 / start_in_prev branches).
        * Strip the tag bytes themselves — they're structural, not
          user-visible.

        The simpler single-block streaming path is unchanged; this
        helper only fires when ``start_in_prev AND end_in_prev``
        (i.e. at least one full ``<think>…</think>`` pair has
        already been streamed).

        Known limitation (codex r2 finding on PR #722): a literal
        ``<think>`` substring inside the model's answer text
        (e.g. ``"The user said: <think> is a tag"``) is
        reclassified as a structural opener and subsequent bytes
        flow to reasoning. The non-streaming
        ``extract_reasoning`` ``partition`` path has the SAME
        behaviour — there's no out-of-band signal in the tag-based
        protocol to distinguish a structural tag from a literal
        substring. The router preserves the existing semantic
        (streaming ↔ non-streaming parity, pinned by
        ``test_literal_think_in_answer_text_is_known_limitation``)
        so any future fix (e.g. tokenizer-id-level structural-tag
        detection) lands once and benefits both paths.
        """
        prev_len = len(current_text) - len(delta_text)
        # Phase at the END OF PREVIOUS DELTA (start of this delta).
        # REASONING if there's an unclosed opener up to that point,
        # CONTENT otherwise. The walk below shifts phase as it
        # crosses tag boundaries inside the delta.
        prev_n_open = previous_text.count(self.start_token)
        prev_n_close = previous_text.count(self.end_token)
        in_reasoning_prev = prev_n_open > prev_n_close
        # Walk through ``current_text`` from ``prev_len`` to the end,
        # splitting at each tag boundary. Emit the segments in the
        # appropriate phase, stripping the tag bytes themselves.
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        cursor = prev_len
        # Codex r1 BLOCKING fix: backtrack the scan window by the
        # ``_held_tag_suffix_len`` so a tag that STRADDLES the SSE
        # boundary (e.g. ``previous_text`` ends with ``<thi``,
        # ``delta_text`` opens with ``nk>R``) is recognised as a
        # complete tag at position ``prev_len - prev_held``. Without
        # this backtrack, the scan from ``prev_len`` misses the
        # straddle and the ``nk>R`` bytes fall through to the
        # trailing-emit path as raw content, leaking the closing tag
        # bytes onto the wire — symmetric to the
        # ``start_in_prev`` straddle case the single-block path
        # already handles. The held suffix bytes were withheld on
        # the prior delta and have NOT been emitted, so including
        # them in the scan is safe (they're not double-counted).
        prev_held = self._held_tag_suffix_len
        scan_from = max(0, prev_len - prev_held)
        # When a straddle is recognised, ``cursor`` also needs to
        # back up to ``scan_from`` so the tag bytes (including the
        # held prefix from previous_text) are dropped. The
        # emit-by-position bookkeeping naturally handles this
        # because the inter-tag segments start from ``cursor`` and
        # we advance ``cursor`` past each tag.
        cursor = scan_from
        # Build a sorted list of all tag occurrences in current_text
        # from ``scan_from`` onwards.
        tags: list[tuple[int, int, str]] = []
        idx = current_text.find(self.start_token, scan_from)
        while idx != -1:
            tags.append((idx, len(self.start_token), "open"))
            idx = current_text.find(self.start_token, idx + 1)
        idx = current_text.find(self.end_token, scan_from)
        while idx != -1:
            tags.append((idx, len(self.end_token), "close"))
            idx = current_text.find(self.end_token, idx + 1)
        tags.sort(key=lambda t: t[0])
        # Phase at cursor — start with the phase that was active at
        # the end of previous_text MINUS the held suffix region
        # (which we've now folded back into the scan, so its
        # contribution to the prev_n_open/prev_n_close counts
        # — always zero because a partial-tag prefix is strictly
        # shorter than the tag — is unchanged).
        phase = "reasoning" if in_reasoning_prev else "content"
        for tag_start, tag_len, tag_kind in tags:
            # Emit bytes between cursor and tag_start in the current
            # phase.
            if tag_start > cursor:
                segment = current_text[cursor:tag_start]
                if phase == "reasoning":
                    reasoning_parts.append(segment)
                else:
                    content_parts.append(segment)
            # Drop the tag bytes (structural) and flip phase.
            cursor = tag_start + tag_len
            phase = "reasoning" if tag_kind == "open" else "content"
        # Trailing bytes after the last tag. Withhold any partial-tag
        # suffix so an in-progress tag at the SSE boundary gets
        # recovered on the next delta.
        held = self._held_partial_tag_len(current_text)
        self._held_tag_suffix_len = held
        safe_end = len(current_text) - held
        if safe_end > cursor:
            segment = current_text[cursor:safe_end]
            if phase == "reasoning":
                reasoning_parts.append(segment)
            else:
                content_parts.append(segment)
        # Phase invariant: after walking all tags in the delta,
        # ``phase`` matches ``current_text``'s overall open/close
        # parity (REASONING if open > close, CONTENT otherwise) —
        # modulo the held partial-tag suffix.
        reasoning = "".join(reasoning_parts) or None
        content = "".join(content_parts) or None
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def _handle_implicit_think(
        self,
        delta_text: str,
        end_in_prev: bool,
        end_in_delta: bool,
    ) -> DeltaMessage | None:
        """Handle case where <think> was in prompt (only </think> in output)."""
        if end_in_delta:
            # Transition: end token in this delta
            idx = delta_text.find(self.end_token)
            reasoning_part = delta_text[:idx]
            content_part = delta_text[idx + len(self.end_token) :]
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )
        elif end_in_prev:
            # Already past reasoning phase - pure content
            return DeltaMessage(content=delta_text)
        else:
            # Still in implicit reasoning phase
            return DeltaMessage(reasoning=delta_text)
