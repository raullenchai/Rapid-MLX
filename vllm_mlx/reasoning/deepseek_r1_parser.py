# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for DeepSeek-R1 models.

DeepSeek-R1 uses <think>...</think> tags for reasoning content.
The model may sometimes start outputting reasoning without the explicit
<think> tag, so this parser is more lenient than Qwen3.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for DeepSeek-R1 model.

    DeepSeek-R1 uses <think>...</think> tokens to denote reasoning text.
    This parser is more lenient than Qwen3:
    - The <think> tag may not be explicitly generated (model assumes it)
    - If only </think> is found, everything before it is reasoning

    Example:
        Input: "<think>Step 1: analyze...\nStep 2: solve...</think>The answer is 42."
        Output: reasoning="Step 1: analyze...\nStep 2: solve...", content="The answer is 42."

        Input: "reasoning content</think>final answer"  # No opening tag
        Output: reasoning="reasoning content", content="final answer"
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from DeepSeek-R1 output.

        More lenient than Qwen3 - handles cases where start tag is implicit.

        Args:
            model_output: Complete model output text.
            enable_thinking: Threaded through to ``BaseThinkingReasoningParser``
                Case 4 — when True, no-tag output is routed to reasoning
                (#575 symmetric-with-streaming fallback). DeepSeek-R1
                callers rarely set this explicitly; the no-tag branch
                below short-circuits before the base call, so the flag
                only matters if a future caller wires it on.

        Returns:
            (reasoning, content) tuple.
        """
        # If we have end token but no start token, treat beginning as reasoning
        if self.end_token in model_output and self.start_token not in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            reasoning = reasoning.strip() or None
            content = content.strip() or None
            # Promote any ``<tool_call>`` blocks from the implicit
            # reasoning span to content (waybarrios#433 port / #344).
            return self._promote_tool_calls(reasoning, content)

        # If neither token, return as pure content — UNLESS the caller
        # explicitly set enable_thinking=True, in which case the chat
        # template injected ``<think>`` into the prompt and a truncated
        # response with no tags is the model's continued thought trace.
        # See ``BaseThinkingReasoningParser.extract_reasoning`` for the
        # full rationale (#575).
        if self.end_token not in model_output and self.start_token not in model_output:
            if enable_thinking is True:
                return model_output.strip() or None, None
            return None, model_output

        # Use base class for standard case
        return super().extract_reasoning(model_output, enable_thinking=enable_thinking)

    # Character threshold for no-tag content detection.
    # If no think tags are seen after this many characters, treat output as
    # content rather than reasoning. Real reasoning models emit <think> within
    # the first few tokens; 64 chars (~15-20 tokens) is a safe threshold for
    # DeepSeek-R1, which always opens with the ``<think>`` token.
    #
    # Subclasses can override this when their model emits a preamble before
    # the ``<think>`` opener — see ``VibeThinkerReasoningParser`` for the
    # Qwen2-derived VibeThinker family (2026-06-17 live test) which needs a
    # larger window. Codex r2 P2: keeping the base threshold at 64 avoids
    # globally widening the reasoning-buffer window for all DeepSeek-R1-family
    # callers (the parent class is still wired to ``deepseek-r1`` and several
    # distilled-on-Qwen aliases that DO open with ``<think>`` immediately).
    NO_TAG_CONTENT_THRESHOLD = 64

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Handles DeepSeek-R1's pattern where <think> may be implicit.
        If no think tags are seen after NO_TAG_CONTENT_THRESHOLD characters,
        treats output as content to avoid misclassifying non-reasoning output.

        Args:
            previous_text: Text accumulated before this delta.
            current_text: Text including this delta.
            delta_text: Just the new text.

        Returns:
            DeltaMessage with reasoning/content, or None to skip.
        """
        # Check if any tags are in the current text
        has_tags = self.start_token in current_text or self.end_token in current_text

        # No tags seen yet and past threshold → treat as content.
        # Codex round-4 BLOCKING finding #1: if the under-threshold
        # phase already opened the tool_call buffer (a structural
        # ``<tool_call>`` arrived while we were routing to reasoning),
        # we must flush that buffer FIRST so the buffered prefix
        # reaches the wire as content before the new content-channel
        # delta. Bare-returning here would strand the buffered prefix
        # and the post-threshold bytes would skip promotion entirely.
        if not has_tags and not self._saw_any_tag:
            if len(current_text) >= self.NO_TAG_CONTENT_THRESHOLD:
                flushed_prefix: str | None = None
                if self._in_tool_call and self._tool_call_buffer:
                    flushed_prefix = self._tool_call_buffer
                    self._tool_call_buffer = ""
                    self._in_tool_call = False
                merged_content = (flushed_prefix or "") + delta_text
                return DeltaMessage(content=merged_content)
            # Under threshold: delegate to base (defaults to reasoning
            # for early implicit mode, will be corrected by finalize)

        # First try base class logic. Use the UNFILTERED inner method so
        # the tool-call promotion filter only runs ONCE at the end of
        # this dispatcher (after our DeepSeek-R1 special case below has
        # had a chance to override). Otherwise a buffered ``<tool_call>``
        # would be partially flushed by the inner filter and then
        # overwritten — losing the buffered bytes.
        result = super()._extract_reasoning_streaming_inner(
            previous_text, current_text, delta_text
        )

        # Handle DeepSeek-R1 special case: no start token seen but end token appears
        if result is not None:
            start_in_prev = self.start_token in previous_text
            start_in_delta = self.start_token in delta_text
            end_in_delta = self.end_token in delta_text

            # If end token in delta but we never saw start token
            if not start_in_prev and not start_in_delta and end_in_delta:
                # Everything before end token is reasoning
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :]
                result = DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )

        # Apply the shared tool-call promotion filter (waybarrios#433
        # port / #344) to whichever DeltaMessage we settled on so a
        # ``<tool_call>`` inside the reasoning channel gets re-routed
        # to ``content`` for the downstream tool parser.
        return self._apply_tool_call_promotion(result)

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> DeltaMessage | None:
        """
        Finalize streaming output.

        Codex round-4 BLOCKING fix (PR #799 review): ``matched_stop``
        alone is NOT enough to identify prompt-injected mid-think.
        A casual answer like ``"The answer is STOP"`` under
        ``stop=["STOP"]`` ALSO has matched_stop set but is not
        chain-of-thought.

        Codex round-6 BLOCKING fix (PR #799 review): ``max_tokens`` cuts
        mid-think share the same accumulator state as stop-mid-think,
        so ``finish_reason="length" AND prompt_thinking_active`` is the
        third D-STOP-THINK signal — without it, the no-tag short answer
        arm would flip to content even though the model was thinking via
        the injected template.

        Discriminator (AND of route-supplied signals):

        * ``finish_reason="length"`` AND prompt_thinking_active →
          D-STOP-THINK max_tokens-cut shape → route to reasoning.
        * ``matched_stop`` set AND prompt_thinking_active → D-STOP-THINK
          stop-cut shape → route to reasoning.
        * Otherwise → casual answer (or no-evidence path) → flip to
          content per #570/#572 so the route consumer surfaces a
          text block.

        The D-STOP-THINK explicit-opener path is still handled by the
        base class default ``finalize_streaming`` which returns None:
        the short-no-tag arm below does NOT fire there because
        ``_saw_any_tag`` becomes True after the literal ``<think>``.

        Args:
            accumulated_text: Complete accumulated text from stream.
            matched_stop: User-supplied stop string that fired, or
                None for natural EOS / max_tokens.
            prompt_thinking_active: True when the chat template
                injected ``<think>`` AND ``enable_thinking`` is non-
                False — i.e. the model was actually in thinking mode.
            finish_reason: Engine finish reason for this turn. Used to
                disambiguate max_tokens cuts (``"length"``) from natural
                EOS (``"stop"``) when ``matched_stop`` is None.

        Returns:
            DeltaMessage correction, or None if no correction needed.
        """
        if not self._saw_any_tag and accumulated_text:
            if finish_reason == "length" and prompt_thinking_active:
                # Prompt-injected mid-think + max_tokens cut — route to
                # reasoning to suppress D-STOP-THINK duplication. This
                # must run before the no-tag content threshold: long
                # prompt-injected thoughts are still thoughts when a real
                # truncation signal arrives.
                return DeltaMessage(reasoning=accumulated_text)
            if matched_stop is not None and prompt_thinking_active:
                # Prompt-injected mid-think shape — route to reasoning
                # to suppress D-STOP-THINK duplication. Same
                # above-threshold rationale as the length arm.
                return DeltaMessage(reasoning=accumulated_text)
            if len(accumulated_text) >= self.NO_TAG_CONTENT_THRESHOLD:
                return None
            # Casual no-tag answer (or max_tokens cut, or stop without
            # active thinking) — flip to content per #570/#572.
            # Without this the casual answer would be silently empty
            # on message.content.
            return DeltaMessage(content=accumulated_text)
        return None


class VibeThinkerReasoningParser(DeepSeekR1ReasoningParser):
    """DeepSeek-R1 variant for the VibeThinker (Weibo AI) family.

    VibeThinker is Qwen2-derived (1.5B base = Qwen2.5-Math-1.5B, 3B base
    = Qwen2.5-Coder-3B) and emits a chatty multi-sentence preamble BEFORE
    its ``<think>`` opener — observed in the 2026-06-17 live test:

        "Okay, let me think about this carefully and step by step.\n\n"
        "<think>Step 1: scan the intervals..."

    The 80-char preamble (~13 tokens) blows past the parent class's
    64-char ``NO_TAG_CONTENT_THRESHOLD``, so streaming routing flipped
    from reasoning → content mid-preamble; by the time the literal
    ``<think>`` arrived, the reasoning trace was already leaking into
    ``content`` deltas (live-test merge_intervals row).

    A 1024-char (~250-300 token) window gives the model room to produce
    a multi-sentence preamble before ``<think>`` while ``finalize_streaming``
    still issues the reasoning → content correction for genuinely no-tag
    short responses that stay under the new threshold for the entire
    stream.

    Scoped narrowly to the VibeThinker family (codex r2 P2): widening the
    parent class's threshold globally would push every DeepSeek-R1-family
    no-tag answer under 1024 chars into the reasoning channel and delay
    visible ``content`` until completion. This subclass localises the
    larger window to the only model that actually needs it.
    """

    NO_TAG_CONTENT_THRESHOLD = 1024
