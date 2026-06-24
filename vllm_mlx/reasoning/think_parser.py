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

import logging
import re
from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser

logger = logging.getLogger(__name__)

# ── Tool-call promotion (port of waybarrios#433 / closes #344) ──────
#
# Thinking models (Qwen 3.5/3.6, Hermes-distill, …) sometimes emit
# ``<tool_call>`` XML/JSON blocks INSIDE the ``<think>`` block. The
# default parser pipeline classifies everything between ``<think>``
# and ``</think>`` as reasoning, so the tool parser downstream never
# sees the call. Promotion re-routes the tool_call block from
# ``reasoning`` to ``content`` so the tool parser can pick it up.
#
# The constants and regex live at module scope so the regex objects
# are compiled once. The promotion logic is centralised in
# ``_promote_tool_calls`` (non-streaming) and a streaming buffer
# state machine; both pipelines use the SAME tag literals so a single
# definition keeps streaming/non-streaming parity by construction.
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
# Closed block: ``<tool_call>…</tool_call>`` (non-greedy across newlines).
_TOOL_CALL_CLOSED_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Unclosed block: ``<tool_call>`` followed by ``{`` or ``<`` (JSON or XML
# structural opener) plus arbitrary tail. The structural guard prevents
# prose mentions like ``"I would use <tool_call> to call functions"``
# from being promoted as a faux tool call — a bare ``<tool_call>``
# followed by free text never matches.
_TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call>\s*[\{<].*$", re.DOTALL)


def _split_unclosed_at_prose_boundary(unclosed_block: str) -> tuple[str, str]:
    """Trim trailing prose from an unclosed ``<tool_call>`` body.

    The upstream regex ``<tool_call>\\s*[\\{<].*$`` (DOTALL) eats
    everything up to end-of-string. That's fine for a TRULY truncated
    tool_call — every line is ``<…>`` / ``</…>`` / JSON — but the
    model occasionally emits a malformed call and then resumes plain
    reasoning (e.g. ``<tool_call>{json}</function>\\nDone thinking.``).
    The trailing prose ``Done thinking.`` belongs in reasoning, not
    in the promoted block.

    Walk the body line-by-line starting AFTER ``<tool_call>``. The
    FIRST line that is purely prose (no ``<`` and no ``{``) marks
    the end of the tool_call body. Returns
    ``(promoted_block, trailing_prose)`` where ``trailing_prose``
    begins with the prose line (and is empty when the entire body
    is tool-call-shaped).

    JSON-aware (codex round-4 finding #8): once an unmatched ``{``
    opens a JSON body, track brace depth and DO NOT treat
    pretty-printed JSON content lines (e.g. ``"name": "get_weather",``)
    as prose even when they contain none of ``<``, ``{``, ``}``.
    Only after the brace depth returns to zero — i.e. the JSON object
    has closed — can a subsequent bare-text line be classified as
    trailing prose.
    """
    # Strip the ``<tool_call>`` opener for inspection — the opener
    # itself must always go into the promoted block.
    if not unclosed_block.startswith(_TOOL_CALL_START):
        # Defensive: the caller passed in a match that doesn't start
        # at the opener. Leave it intact.
        return unclosed_block, ""
    head = unclosed_block[: len(_TOOL_CALL_START)]
    body = unclosed_block[len(_TOOL_CALL_START) :]
    lines = body.split("\n")
    promoted_lines: list[str] = []
    trailing_lines: list[str] = []
    boundary_hit = False
    json_depth = 0  # net {…} depth across processed lines
    for line in lines:
        if boundary_hit:
            trailing_lines.append(line)
            continue
        stripped = line.strip()
        if stripped == "":
            # Blank lines are ambiguous — keep with promoted block.
            promoted_lines.append(line)
            continue
        if "<" in stripped or "{" in stripped or "}" in stripped:
            # XML/JSON-ish — still inside the tool_call body. Update
            # net brace depth so subsequent pretty-printed JSON value
            # lines are recognised as still-inside-body even though
            # they have no structural chars.
            json_depth += stripped.count("{") - stripped.count("}")
            promoted_lines.append(line)
            continue
        if json_depth > 0:
            # Pretty-printed JSON value line inside an open ``{…}`` —
            # NOT prose. Keep it in the promoted block.
            promoted_lines.append(line)
            continue
        # Pure prose line — boundary.
        boundary_hit = True
        trailing_lines.append(line)
    promoted_block = head + "\n".join(promoted_lines)
    trailing_prose = "\n".join(trailing_lines)
    if trailing_prose:
        trailing_prose = "\n" + trailing_prose
    return promoted_block, trailing_prose


def _partial_tool_call_open_suffix(text: str) -> int:
    """Length of the suffix of ``text`` that could be a strict prefix of
    ``<tool_call>``.

    Used by the streaming promotion filter to withhold trailing bytes
    that look like a partial ``<tool_call>`` opener so a tag straddling
    SSE chunks reassembles correctly on the next delta. The full match
    case is excluded — if ``text`` already ends in the complete opener
    the buffer branch handles it.
    """
    tag = _TOOL_CALL_START
    max_len = min(len(tag) - 1, len(text))
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


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
        # Tool-call promotion state (port of waybarrios#433 / #344).
        # When the model emits ``<tool_call>`` INSIDE a ``<think>`` block,
        # the bytes must be promoted from the reasoning channel into the
        # content channel so the downstream tool parser can find them.
        # ``_in_tool_call`` flips to True the moment we observe the
        # ``<tool_call>`` opener while routing a delta as reasoning;
        # subsequent reasoning deltas are buffered into
        # ``_tool_call_buffer`` until ``</tool_call>`` arrives (closed
        # promotion) OR ``</think>`` arrives first (unclosed flush) OR
        # the stream ends (finalize flush).
        self._in_tool_call: bool = False
        self._tool_call_buffer: str = ""
        # SSE-boundary carry for the ``<tool_call>`` opener. When the
        # tag straddles a chunk boundary (e.g. delta ends with ``<too``,
        # next starts with ``l_call>``), the trailing partial-tag bytes
        # are stashed here so they reassemble with the next chunk
        # instead of leaking into the reasoning channel.
        self._reasoning_carry: str = ""
        # Codex r3 BLOCKING on PR #722: track streaming phase
        # explicitly instead of recomputing it from whole-history
        # ``previous_text.count(...)``. Counting historical tag
        # occurrences misclassifies literal ``<think>`` or
        # ``</think>`` substrings inside ALREADY-EMITTED CONTENT
        # (e.g. a closed ``<think>...</think>`` pair that survived
        # the conservative sweep) as structural tags and flips the
        # phase. ``None`` means we have not yet entered the
        # multi-block router; the router seeds it the first time it
        # fires (always "content" at that point, because the FIRST
        # ``</think>`` has just been consumed). Subsequent deltas
        # read this field and update it as they cross structural
        # tags inside the delta.
        self._streaming_phase: str | None = None

    def reset_state(self):
        """Reset state for a new streaming request."""
        super().reset_state()
        self._saw_any_tag = False
        self._held_tag_suffix_len = 0
        self._streaming_phase = None
        self._in_tool_call = False
        self._tool_call_buffer = ""
        self._reasoning_carry = ""

    def is_open_in_think(self, accumulated_text: str) -> bool:
        """Return True iff the accumulated text shows an opened but
        not-yet-closed ``<think>``-style block (the truncation-mid-think
        shape).

        r5-D shared finalize-on-truncation contract: the route invokes
        this on ``finish_reason="length"`` to decide whether the
        unclosed buffer should be routed to ``reasoning_content``
        (open-in-think) or kept as ``content`` (think already closed
        OR never opened). The classic shape is
        ``<think>…``-without-``</think>`` — either the model emitted
        the start token autonomously (``<think>`` at the head) OR the
        chat template pre-injected it and we are observing an implicit
        unclosed thought.

        Note: explicit-start callers — qwen3 / deepseek-r1 / glm4 /
        vibethinker — share this implementation. Subclasses that only
        need a different policy for the no-tag autonomous-think shape
        (Qwen3 ``enable_thinking=True`` Case-4) can override and
        consult their own out-of-band signal.
        """
        if not accumulated_text:
            return False
        if self.start_token in accumulated_text and (
            self.end_token not in accumulated_text
        ):
            # Classic mid-think: opener arrived but closer never did.
            return True
        return False

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
            r = reasoning.strip() or None
            c = content.strip() or None
            return self._promote_tool_calls(r, c)

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
            r = reasoning.strip() or None
            c = content.strip() or None
            return self._promote_tool_calls(r, c)

        # Case 3: Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            r = reasoning.strip() or None
            return self._promote_tool_calls(r, None)

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
        """Public streaming entry point.

        Composes the existing think-tag state machine
        (``_extract_reasoning_streaming_inner``) with the tool-call
        promotion filter (``_apply_tool_call_promotion``). The filter
        watches the reasoning channel for ``<tool_call>`` blocks and
        re-routes them to the content channel — port of waybarrios#433
        (closes #344). Centralising the promotion at this seam keeps
        the inner state machine untouched (no per-emit-point
        modifications) and ensures the same filter fires for every
        ``<think>``-tag subclass (Qwen3 / DeepSeek-R1 / Glm4 /
        VibeThinker / …) without per-subclass duplication.
        """
        msg = self._extract_reasoning_streaming_inner(
            previous_text, current_text, delta_text
        )
        return self._apply_tool_call_promotion(msg)

    def _extract_reasoning_streaming_inner(
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
        # Skip if delta is just the special tokens themselves.
        # Codex r3 BLOCKING follow-up on PR #722: when the multi-block
        # router has already taken over phase tracking
        # (``_streaming_phase is not None``), flip the phase here so
        # the NEXT delta enters the router with the correct phase
        # state. The router's intra-delta walk only sees tags
        # scanned from ``prev_len - held``, so a tag that arrived as
        # a stand-alone delta would otherwise be missed and the
        # phase would lie on the following delta.
        #
        # Codex r4 BLOCKING follow-up on PR #722: ALSO seed the phase
        # when ``_streaming_phase is None``. The single-block path
        # (``_handle_explicit_think`` before ``end_in_prev``) never
        # writes ``_streaming_phase`` — it stays None right up until
        # the multi-block router fires for the first time. If the
        # SECOND ``<think>`` block arrives as a stand-alone delta
        # IMMEDIATELY after the first ``</think>`` (no intermediate
        # content delta to trigger the router), the early-skip below
        # would leave the phase at None — and the next reasoning
        # delta would enter the router with ``in_reasoning_prev =
        # False`` (the documented fall-through), causing the second
        # reasoning block to leak into ``content``. Seeding the
        # phase here when ``previous_text`` already contains a
        # structural closer (resp. opener) closes that hole without
        # breaking the literal-tag-in-text known-limitation contract
        # — the seed only fires when the delta IS the bare tag
        # (overwhelmingly structural in practice; the whole-history
        # count drift codex r3 documented is bounded here to this
        # single-delta event).
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            if self._streaming_phase is not None:
                self._streaming_phase = "reasoning"
            elif self.end_token in previous_text:
                # Standalone opener after the first ``</think>`` —
                # this is the second-block opener; seed the phase so
                # the next delta's router call routes reasoning bytes
                # correctly.
                self._streaming_phase = "reasoning"
            return None
        if stripped_delta == self.end_token:
            if self._streaming_phase is not None:
                self._streaming_phase = "content"
            elif self.start_token in previous_text:
                # Standalone closer after an opener — seed to content
                # so trailing answer bytes don't get misrouted.
                self._streaming_phase = "content"
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
            # F-100 (2026-06-19): in implicit mode, once we've already
            # crossed the first ``</think>`` (``end_in_prev=True``) the
            # model can re-enter reasoning by autonomously emitting a
            # SECOND ``<think>`` opener after the answer (phi-4-mini-
            # reasoning streams ``…JSON…<think>more thought</think>tail``
            # under ``response_format=json_schema`` + ``stream=true``).
            # The naive ``end_in_prev → content=delta_text`` fallback in
            # ``_handle_implicit_think`` leaks the literal ``<think>``
            # bytes (including SSE-split fragments like ``<th``, ``ink``,
            # ``>\n``) into ``delta.content`` and routes the second
            # block's body to content instead of reasoning. Delegate to
            # the multi-block router — it already handles partial-tag
            # withhold AND ``_streaming_phase`` flip on subsequent
            # deltas, so the second block streams as reasoning and the
            # following ``</think>`` flips cleanly back to content.
            # Symmetric with the explicit-mode multi-block dispatch in
            # ``_handle_explicit_think`` (``start_in_prev`` AND
            # ``end_in_prev``).
            if end_in_prev:
                return self._handle_multi_block_after_close(
                    previous_text, current_text, delta_text
                )
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

    # ------------------------------------------------------------------
    # D-STOP-THINK shared finalize contract (cross-cycle bundle):
    #
    # When ``stop`` matches inside an unterminated ``<think>`` block (OR
    # ``max_tokens`` cuts mid-thought before ``</think>``), the streaming
    # loop has already shipped every reasoning byte as
    # ``reasoning_content`` (the per-delta route — Anthropic
    # ``thinking`` block, OpenAI ``delta.reasoning_content``, Responses
    # API thinking event). Anthropic/Responses streams ALSO call
    # ``finalize_streaming(accumulated_raw)`` at end-of-stream and emit
    # ``final_msg.content`` as a NEW text block. Pre-fix Qwen3 and
    # DeepSeek-R1 ``finalize_streaming`` returned
    # ``DeltaMessage(content=<full trace>)`` on the unterminated-think
    # path — which made the client see the EXACT SAME bytes in BOTH
    # ``reasoning_content`` AND ``content``. Cross-cycle repro across
    # qwen3 / deepseek_r1 / glm4 / gemma4 / hermes / VibeThinker
    # families: bug_report.md:128, cycle-3 F-3, cycle-5 F-1 (hermes-
    # qwen3.5-27b-8bit), cycle-6 F-CORR-2 (gemma-4), cycle-7 F-1
    # (nemotron-30b), cycle-8 F-801 (glm4.7-9b-4bit), cycle-11 F-11-7
    # (phi-4-mini-reasoning).
    #
    # The shared invariant lives here in the base class so every
    # ``<think>``-tag parser inherits the same gate. Codex round-8
    # NIT (PR #799): the gate applies ONLY to streams that gave
    # EXPLICIT or PROMPT-INJECTED thinking evidence — NOT to bare
    # no-tag casual answers.
    #
    #     If ``</think>`` was NEVER crossed AND the stream gave
    #     explicit thinking evidence (a literal ``<think>`` opener
    #     reached the parser) AND a real truncation fired
    #     (``finish_reason=="length"`` OR ``matched_stop is not
    #     None``), ``finalize_streaming`` MUST NOT emit ``content``.
    #     The route already shipped these bytes as
    #     ``reasoning_content`` during the stream loop, so a content
    #     emission here is a pure duplicate. The same suppression
    #     applies to PROMPT-INJECTED thinking evidence (chat template
    #     injected ``<think>``, ``enable_thinking`` non-False) under
    #     the same truncation signals.
    #
    #     CASUAL no-tag answers (no opener, no truncation signal,
    #     no bare-preamble label) are EXEMPT from this rule — Qwen3
    #     and DeepSeek-R1 INTENTIONALLY emit
    #     ``DeltaMessage(content=...)`` for that path (#570/#572)
    #     because the streaming Case-3 default routed the bytes to
    #     ``reasoning_content`` but the answer was never actually
    #     chain-of-thought. Without the content correction, the
    #     casual-answer wire would be a silently empty assistant
    #     turn. NATURAL-EOS with an explicit ``<think>`` opener but
    #     no truncation signal is ALSO exempt — see qwen3_parser.py
    #     saw_think_prefix branch (#569 silent-drop rescue).
    #
    # Subclasses that want to emit a reasoning ``DeltaMessage`` at
    # finalize (e.g. Qwen3's bare-text-preamble surfacing for the
    # non-streaming envelope) MAY do so — Anthropic / Responses routes
    # only act on ``final_msg.content``, so reasoning-channel finalize
    # output is harmless.
    def _missing_think_close(self, accumulated_text: str) -> bool:
        """Return True when the accumulated text has no closing
        ``</think>`` (or subclass-specific ``end_token``).

        NARROW SEMANTICS (codex round-10 NIT, PR #799): this
        predicate's name now EXPLICITLY says what it tests —
        "missing think close". It does NOT itself imply that the
        stream carried thinking evidence, nor that any D-STOP-THINK
        suppression should fire — those are SEPARATE signals (a
        literal ``<think>`` opener in the buffer, OR
        ``prompt_thinking_active`` under a truncation signal). The
        predicate's job is to be the FIRST gate: callers AND it
        with their evidence signals before suppressing a content
        emission.

        Misuse warning: do NOT use this predicate alone to decide
        whether to suppress a ``DeltaMessage(content=...)`` emission
        — a plain no-tag casual answer (no opener, no truncation
        signal) would trip it and you'd silently drop the assistant
        turn (the #569/#570/#572 regression this PR closes from the
        OTHER direction). Always combine with evidence:

            if (self._missing_think_close(text)
                and (saw_think_opener OR
                     (prompt_thinking_active AND
                      (matched_stop OR finish_reason == "length"))))
            then suppress content (route to reasoning)
            else allow content (casual-answer rescue per #569)

        See ``qwen3_parser.py`` saw-prefix and no-prefix branches for
        the canonical caller pattern. The base class default
        ``finalize_streaming`` returns ``None`` for ANY accumulated
        text and is safe to inherit — the AND-of-signals logic only
        lives in subclasses that emit corrections.

        Subclasses with different closer tokens override
        ``end_token``; this helper reads ``self.end_token`` so the
        predicate adapts uniformly.
        """
        return bool(accumulated_text) and self.end_token not in accumulated_text

    def _finalize_in_think_block(self, accumulated_text: str) -> bool:
        """Backward-compatible evidence-aware mid-think predicate.

        Unlike ``_missing_think_close``, this keeps the old "currently
        inside a think block" semantics: a real opener must be present
        and unmatched. Plain no-tag text such as ``"hello"`` is missing
        a closer but is not itself proof of an open think block.
        """
        return self.is_open_in_think(accumulated_text)

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> "DeltaMessage | None":
        """Default base-class finalize: no correction.

        D-STOP-THINK invariant: when ``</think>`` was never crossed,
        finalize MUST NOT emit content. The default ``return None``
        upholds that. Subclasses that want to inject a reasoning-channel
        correction (Qwen3's bare-text preamble surfacing, etc.) MUST
        gate their content emission on ``not self._finalize_in_think_block``.

        The base class default ignores the ``matched_stop``,
        ``prompt_thinking_active`` and ``finish_reason`` signals because
        the no-correction return is safe either way. Subclasses (Qwen3 /
        DeepSeek-R1) use the AND of those signals to discriminate
        prompt-injected mid-think (stop OR max_tokens) from casual
        no-tag answers.
        """
        return None

    def _sweep_residual_think_tags(
        self, reasoning: str, content: str
    ) -> tuple[str, str]:
        """Strip any residual ``<think>…</think>`` blocks left in
        ``content`` after the first-pair partition, and reroute
        the TRAILING unclosed thought into ``reasoning``.

        2026-06-19 round-1 fuzz repro (phi-4-mini-reasoning-4bit):
        when the model emits multiple ``<think>`` blocks (closed
        inner ones before the answer and a TRAILING opener after
        the answer that ``reasoning_max_tokens`` / ``max_tokens``
        truncates before its closing tag), the naive first-pair
        ``partition`` in ``extract_reasoning`` Case 1 consumes
        ONLY the first pair. The trailing ``<think>thought_N…``
        opener was leaking verbatim into ``message.content`` —
        neither ``strip_thinking_tags`` (matches closed blocks
        only) nor ``sanitize_output`` (matches a stray
        ``</think>`` only) catch an orphan ``<think>`` opener,
        so the tag bytes survived all the way to the wire.

        Codex r3 BLOCKING on PR #722: an earlier draft of this
        sweep stripped EVERY closed ``<think>…</think>`` block
        from content too. That broke a documented but rare case
        — answers that legitimately contain a literal ``<think>``
        substring in the text (e.g. ``"The user said: <think>
        is literal"``) — by silently reclassifying the literal
        text as a structural reasoning block. Pre-PR behaviour
        preserved that text in content (the first-pair partition
        only ate one pair and ``strip_thinking_tags`` only
        matches CLOSED blocks, which DO get cleaned but the
        unclosed literal opener stays in content as the user
        sees it). The conservative scope below restores that
        pre-PR behaviour for closed blocks while STILL closing
        the actual round-1 leak (trailing unclosed opener).

        Sweep scope (codex r3-final, conservative):

        * Closed ``<think>…</think>`` blocks left in content are
          UNTOUCHED here — downstream ``strip_thinking_tags`` /
          ``sanitize_output`` handles them, AND if any remain
          they were legitimate literal text the user intended.
        * Only the TRAILING unclosed ``<think>…`` (the round-1
          leak shape) gets routed into ``reasoning``. The opener
          must have NO matching ``</think>`` AFTER it for this
          branch to fire.
        * Stray ``</think>`` closers with no preceding opener
          in content are LEFT for ``sanitize_output`` to strip.

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
        # Locate the LAST ``<think>`` opener in content. If it has
        # NO matching ``</think>`` after it, it's the trailing
        # unclosed block from the round-1 leak shape — route to
        # reasoning. Otherwise leave content untouched so any
        # closed ``<think>…</think>`` blocks (potentially literal
        # text) survive the sweep and reach the downstream
        # ``strip_thinking_tags`` / ``sanitize_output`` stages
        # unaltered.
        last_open = content.rfind(self.start_token)
        if last_open < 0:
            return reasoning, content
        after_last_open = content[last_open + len(self.start_token) :]
        if self.end_token in after_last_open:
            # The last opener has a matching closer — it's a fully
            # formed block (closed OR contains a closer). Leave
            # content alone; the downstream regex strippers will
            # remove the structural form, and a literal-text form
            # stays as-is.
            return reasoning, content
        # Trailing unclosed opener — route its body to reasoning,
        # truncate content at the opener. Matches Case 3
        # "no end tag → reasoning" semantics.
        trailing_reasoning = after_last_open.rstrip()
        if trailing_reasoning:
            reasoning = (
                (reasoning.rstrip() + "\n" + trailing_reasoning)
                if reasoning
                else trailing_reasoning
            )
        content = content[:last_open].rstrip()
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
                # F-100: synchronise ``_streaming_phase`` with the
                # post-close state so a SUBSEQUENT delta routed through
                # the multi-block dispatch starts from the correct phase
                # rather than inheriting whatever the SSE-boundary
                # recovery branch left behind (a split-opener earlier in
                # the same request would otherwise keep the phase at
                # ``"reasoning"`` and leak post-``</think>`` bytes into
                # ``reasoning_content``).
                self._streaming_phase = "content"
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
                # F-100: synchronise ``_streaming_phase`` with the
                # post-close state so a SUBSEQUENT delta routed through
                # the multi-block dispatch (``start_in_prev AND
                # end_in_prev`` after this) starts from the correct
                # phase instead of defaulting to ``in_reasoning_prev =
                # False``. We just emitted the close inside this delta;
                # post-close state is ``content``.
                self._streaming_phase = "content"
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
        # F-100: synchronise ``_streaming_phase`` with the just-opened
        # block so a SUBSEQUENT delta routed through the multi-block
        # dispatch (``start_in_prev AND end_in_prev`` after this) starts
        # from the correct phase. We just emitted the opener inside this
        # delta; post-opener state is ``reasoning``. Without this seed,
        # ``_handle_multi_block_after_close`` falls back to
        # ``in_reasoning_prev = False`` and the next-delta body of a
        # second ``<think>`` block leaks into ``content`` (F-100 repro
        # tail: ``Okay`` / ``second think`` after a split second opener).
        self._streaming_phase = "reasoning"
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
        #
        # Codex r3 BLOCKING fix on PR #722: do NOT recompute phase
        # from whole-history ``previous_text.count(<think>)`` vs
        # ``count(</think>)``. A literal ``<think>`` or ``</think>``
        # substring in already-emitted content (e.g. a closed
        # ``<think>...</think>`` pair that survived the conservative
        # non-streaming sweep, or a user-facing answer that mentions
        # the tag) inflates the historical count and makes the
        # phase decision lie — subsequent structural reasoning
        # chunks would then leak into ``content`` (or vice versa).
        # Instead, persist the phase as instance state and update
        # it ONLY when this router crosses structural tag
        # boundaries. ``_streaming_phase is None`` means this is
        # the first time the multi-block router fires — by
        # construction (we dispatched on ``end_in_prev`` or on a
        # full pair landing in this delta) the phase at that
        # entry point is "content" (we've just exited the first
        # ``<think>...</think>`` block). For subsequent calls we
        # read the field directly.
        if self._streaming_phase is None:
            in_reasoning_prev = False
        else:
            in_reasoning_prev = self._streaming_phase == "reasoning"
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
        # ``phase`` reflects the end-of-delta structural state
        # (modulo the held partial-tag suffix). Persist it so the
        # NEXT delta starts from the correct phase without having
        # to recount historical tags (codex r3 BLOCKING fix).
        self._streaming_phase = phase
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

    # ── Tool-call promotion (port of waybarrios#433 / #344) ─────────
    #
    # Thinking models occasionally emit ``<tool_call>`` blocks INSIDE
    # ``<think>``. The base think state machine routes those bytes to
    # the reasoning channel, so the downstream tool parser never sees
    # them. The promotion logic below re-routes the bytes to the
    # content channel for both the non-streaming
    # (``_promote_tool_calls``) and streaming
    # (``_apply_tool_call_promotion`` filter + ``finalize_streaming``
    # flush hook) paths. The shared filter lives on the base class so
    # every ``<think>``-tag subclass benefits without per-parser
    # duplication.

    @classmethod
    def _promote_tool_calls(
        cls, reasoning: str | None, content: str | None
    ) -> tuple[str | None, str | None]:
        """Move ``<tool_call>`` blocks from ``reasoning`` into ``content``.

        Non-streaming half of the promotion port. Applied AFTER the
        normal think-tag split so the structural separation of
        reasoning vs content is preserved — only the tool-call XML/JSON
        blocks move.

        Algorithm:

        * Pull every CLOSED ``<tool_call>…</tool_call>`` block out of
          ``reasoning`` and append it to ``content`` (chronological
          order — the thinking happened before the answer).
        * Pull the LAST UNCLOSED ``<tool_call>`` block out (if present)
          and PREPEND it to ``content`` — the spanning shape is
          ``<think>R<tool_call>partial</think>rest</tool_call>C``
          where the closer and remainder land in ``content``; the
          buffered head must reassemble with that remainder, hence
          prepend.
        * Structural guard via ``_TOOL_CALL_UNCLOSED_RE``: requires
          ``<tool_call>`` to be followed by ``{`` or ``<`` (JSON or
          XML opener). Prose mentions like ``"use <tool_call> to
          invoke functions"`` never match — preventing false
          positives that would clobber legitimate reasoning text.
        * Logs a warning summarising the number of blocks moved, so
          operators monitoring reasoning-channel health can see the
          promotion firing.
        """
        if not reasoning or _TOOL_CALL_START not in reasoning:
            return reasoning, content

        # Closed regex first: extract complete <tool_call>...</tool_call> blocks.
        # Then unclosed regex on the already-stripped reasoning.
        closed: list[str] = []

        def _collect_closed(match: re.Match[str]) -> str:
            closed.append(match.group(0))
            return ""

        cleaned = _TOOL_CALL_CLOSED_RE.sub(_collect_closed, reasoning)

        unclosed_match = _TOOL_CALL_UNCLOSED_RE.search(cleaned)
        unclosed_block: str | None = None
        if unclosed_match:
            unclosed_block = unclosed_match.group(0)
            unclosed_start = unclosed_match.start()
            # Trim trailing prose: the upstream regex eats everything
            # up to end-of-string with ``.*$`` DOTALL, but the model
            # sometimes emits a MALFORMED tool_call and then returns
            # to plain reasoning (e.g. ``<tool_call>{json}</function>
            # \nDone thinking.``). The trailing prose belongs in
            # reasoning, not in the promoted block. Detect the
            # boundary by walking the unclosed body line-by-line:
            # the first line that is purely prose (no ``<`` and no
            # ``{``) marks the END of the tool_call body. This keeps
            # the upstream behaviour for well-formed truncated calls
            # (every line is ``<…>`` / ``</…>`` / JSON) while letting
            # the existing wire-scrub pipeline handle malformed-wire +
            # returned-to-reasoning shapes — closes the over-promotion
            # gap exposed by ``test_t3_chat_route_scrubs_wire_leak_from_reasoning_content``.
            trimmed_block, trailing_prose = _split_unclosed_at_prose_boundary(
                unclosed_block
            )
            unclosed_block = trimmed_block
            cleaned = cleaned[:unclosed_start] + trailing_prose

        cleaned = cleaned.strip() or None
        promoted_count = len(closed) + (1 if unclosed_block else 0)

        if promoted_count == 0:
            return reasoning, content

        result_content = content or ""

        if unclosed_block:
            result_content = (
                unclosed_block + "\n" + result_content
                if result_content
                else unclosed_block
            )

        if closed:
            closed_text = "\n".join(closed)
            result_content = (
                result_content + "\n" + closed_text if result_content else closed_text
            )

        result_content = result_content.strip() or None

        logger.warning(
            "Promoted %d tool_call block(s) from reasoning to content "
            "(%d closed, %d unclosed)",
            promoted_count,
            len(closed),
            1 if unclosed_block else 0,
        )

        return cleaned, result_content

    def _apply_tool_call_promotion(
        self, msg: DeltaMessage | None
    ) -> DeltaMessage | None:
        """Streaming half of the promotion port — per-delta filter.

        Watches the reasoning channel for ``<tool_call>`` and buffers
        bytes between ``<tool_call>`` and ``</tool_call>`` so they can
        be flushed into the content channel instead. The filter is
        composed at the public ``extract_reasoning_streaming`` seam so
        every ``<think>``-tag subclass picks it up uniformly.

        Six cases the filter handles:

        1. ``msg is None`` (skip) — pass through.
        2. Not buffering, no ``<tool_call>`` in reasoning — pass through.
        3. Not buffering, ``<tool_call>`` appears mid-reasoning — emit
           the prefix as reasoning, start buffering the rest (which may
           ALSO contain ``</tool_call>`` if the whole call lands in one
           delta; the recursive flush handles closure correctly).
        4. Buffering, more reasoning arrives — append to buffer. If the
           buffer now contains ``</tool_call>``, flush the closed
           portion as content; the post-close remainder may start
           another buffer or continue as reasoning.
        5. Buffering, ``msg.content`` arrives (think ended while
           buffering) — flush the buffered prefix as content (unclosed
           promotion), then concatenate ``msg.content`` after.
        6. Always carry through ``msg.content`` unchanged when not
           buffering — the content channel is not subject to promotion.

        Streaming/non-streaming parity invariant: when the entire
        output lands in a SINGLE delta (e.g. tests that feed the
        whole text at once), the inner state machine returns one
        ``DeltaMessage(reasoning=R, content=C)`` and the filter
        delegates to ``_promote_tool_calls(R, C)`` — guaranteeing
        the same wire shape as the non-streaming path.
        """
        if msg is None:
            return None

        r_in = msg.reasoning
        c_in = msg.content

        # Single-delta catch-all: the inner ran the WHOLE state machine
        # in this delta (the SSE-boundary tests feed
        # chunk_size=len(text)) and emitted reasoning + content
        # together. Delegate to the non-streaming promoter so the
        # streaming wire shape matches the non-streaming wire shape
        # exactly. This is upstream's ``_transition_to_content``
        # catch-all in spirit, recast in our wrapper.
        #
        # The shortcut is ONLY safe when no prior chunk left a partial
        # ``<tool_call>`` opener carry — otherwise delegating to the
        # non-streaming promoter would see only ``r_in`` and silently
        # drop ``self._reasoning_carry``. When a carry exists, fall
        # through to the per-channel state machine so
        # ``_absorb_reasoning_chunk`` re-prepends the carry first.
        if (
            not self._in_tool_call
            and not self._reasoning_carry
            and r_in
            and c_in is not None
            and _TOOL_CALL_START in r_in
        ):
            new_r, new_c = self._promote_tool_calls(r_in, c_in)
            if new_r is None and new_c is None:
                return None
            return DeltaMessage(
                role=msg.role,
                reasoning=new_r,
                content=new_c,
            )

        # Otherwise walk the per-channel state machine.
        out_reasoning_parts: list[str] = []
        out_content_parts: list[str] = []

        if r_in:
            self._absorb_reasoning_chunk(r_in, out_reasoning_parts, out_content_parts)

        # Content channel: if we were buffering when content arrived,
        # the think block ended mid-tool-call. Flush the buffered head
        # as content first, then append the inner's content.
        if c_in is not None:
            # The carry held a partial ``<tool_call>`` opener that
            # straddled across deltas. Now that the think block is
            # ending, the carry is unresolved — flush it back as
            # reasoning so no bytes are silently dropped.
            if self._reasoning_carry:
                out_reasoning_parts.append(self._reasoning_carry)
                self._reasoning_carry = ""
            if self._in_tool_call and self._tool_call_buffer:
                flushed = self._tool_call_buffer
                self._tool_call_buffer = ""
                self._in_tool_call = False
                logger.warning(
                    "Promoted unclosed streaming tool_call "
                    "(think ended before tool_call closed)"
                )
                out_content_parts.append(flushed)
            out_content_parts.append(c_in)

        new_r = "".join(out_reasoning_parts) or None
        new_c = "".join(out_content_parts) or None
        if new_r is None and new_c is None:
            return None
        return DeltaMessage(role=msg.role, reasoning=new_r, content=new_c)

    def _absorb_reasoning_chunk(
        self,
        text: str,
        out_reasoning: list[str],
        out_content: list[str],
    ) -> None:
        """Walk one reasoning-channel chunk through the tool-call buffer.

        Loops until the chunk is fully consumed, alternating between
        buffer (inside ``<tool_call>``) and direct reasoning emit
        (outside). Mutates ``out_reasoning`` and ``out_content`` in
        place — each emit goes to the appropriate output channel.

        SSE-boundary safety: when the ``<tool_call>`` opener straddles
        chunk boundaries (e.g. ``<too`` then ``l_call>X``), the filter
        carries the unresolved tail in
        ``self._reasoning_carry`` so a partial opener at the end of
        one chunk reassembles with the head of the next. Same idea as
        the inner state machine's ``_held_tag_suffix_len`` but scoped
        to the ``<tool_call>`` / ``</tool_call>`` tag set instead of
        ``<think>`` / ``</think>``.

        Promotion is symmetric across nested deltas: if the same
        chunk contains multiple ``<tool_call>…</tool_call>`` blocks
        the loop promotes each one individually.
        """
        # Prepend any unresolved partial-opener carry from prior deltas
        # so a straddled ``<tool_call>`` opener reassembles.
        remaining = self._reasoning_carry + text
        self._reasoning_carry = ""
        while remaining:
            if self._in_tool_call:
                # Buffering. Look for </tool_call> close.
                self._tool_call_buffer += remaining
                end_idx = self._tool_call_buffer.find(_TOOL_CALL_END)
                if end_idx < 0:
                    # Whole chunk consumed by buffer — still open.
                    return
                # Closed: split buffer at the closer.
                promoted = self._tool_call_buffer[: end_idx + len(_TOOL_CALL_END)]
                tail = self._tool_call_buffer[end_idx + len(_TOOL_CALL_END) :]
                out_content.append(promoted)
                self._tool_call_buffer = ""
                self._in_tool_call = False
                logger.warning("Promoted streaming tool_call block from reasoning")
                remaining = tail
                continue
            # Not buffering — look for <tool_call> in remaining.
            start_idx = remaining.find(_TOOL_CALL_START)
            if start_idx < 0:
                # No full opener — but the trailing bytes may be a
                # PARTIAL opener (``<``, ``<t``, ``<tool_call`` …).
                # Withhold the longest matching suffix so a straddled
                # tag reassembles on the next call.
                carry_len = _partial_tool_call_open_suffix(remaining)
                if carry_len > 0:
                    safe = remaining[:-carry_len]
                    self._reasoning_carry = remaining[-carry_len:]
                    if safe:
                        out_reasoning.append(safe)
                else:
                    out_reasoning.append(remaining)
                return
            # Structural guard — parity with non-streaming
            # ``_TOOL_CALL_UNCLOSED_RE`` (``<tool_call>\s*[\{<]``).
            # A real tool_call body starts with ``{`` (JSON) or ``<``
            # (XML opener) after the tag and optional whitespace.
            # A prose mention like ``use <tool_call> to invoke ...``
            # has no structural opener after the tag and must NOT be
            # promoted — otherwise the prose tail leaks into the
            # content channel via the buffer.
            after_start = start_idx + len(_TOOL_CALL_START)
            probe = remaining[after_start:]
            # Skip whitespace looking for the first non-ws byte.
            j = 0
            while j < len(probe) and probe[j] in (" ", "\t", "\n", "\r"):
                j += 1
            if j == len(probe):
                # All whitespace (or empty) after the opener — the
                # discriminating byte hasn't arrived yet. Emit the
                # prefix as reasoning and carry the tag + trailing
                # whitespace so the next chunk reveals whether this
                # is a real tool_call or a prose mention.
                if start_idx > 0:
                    out_reasoning.append(remaining[:start_idx])
                self._reasoning_carry = remaining[start_idx:]
                return
            if probe[j] not in ("{", "<"):
                # Prose mention — emit up to AND INCLUDING the bare
                # ``<tool_call>`` tag as reasoning and keep scanning
                # ``remaining`` past the opener for another candidate.
                out_reasoning.append(remaining[:after_start])
                remaining = remaining[after_start:]
                continue
            # Real tool_call: emit prefix as reasoning, start buffering
            # from the opener.
            if start_idx > 0:
                out_reasoning.append(remaining[:start_idx])
            self._in_tool_call = True
            self._tool_call_buffer = ""
            remaining = remaining[start_idx:]
            # Loop continues — the buffer branch will consume
            # ``remaining`` (which starts with ``<tool_call>``).

    def _flush_pending_tool_call(self) -> DeltaMessage | None:
        """Flush any buffered ``<tool_call>`` bytes at end of stream.

        Called from ``finalize_streaming`` (this class and Qwen3
        override below). The streaming filter buffers bytes after a
        ``<tool_call>`` opener so they can be promoted to ``content``
        once ``</tool_call>`` arrives. If the stream ends before the
        closer (truncation or natural EOS mid-call), the buffered
        bytes would otherwise be silently dropped — exactly the
        wire-shape gap waybarrios#433 closed for the closed-block
        case. Flushing as ``content`` preserves the tool-call XML/JSON
        for the downstream tool parser.

        Also flushes any leftover ``_reasoning_carry`` as reasoning —
        a partial-tag opener withheld at the SSE boundary that turned
        out NOT to be a tool_call must NOT be silently dropped at
        end-of-stream.
        """
        reasoning_flush: str | None = None
        if self._reasoning_carry:
            reasoning_flush = self._reasoning_carry
            self._reasoning_carry = ""
        content_flush: str | None = None
        if self._in_tool_call and self._tool_call_buffer:
            content_flush = self._tool_call_buffer
            self._tool_call_buffer = ""
            self._in_tool_call = False
            logger.warning("Promoted unclosed streaming tool_call at stream end")
        if reasoning_flush is None and content_flush is None:
            return None
        return DeltaMessage(reasoning=reasoning_flush, content=content_flush)
