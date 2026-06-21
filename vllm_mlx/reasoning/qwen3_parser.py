# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Qwen3 models.

Qwen3 uses <think>...</think> tags for reasoning content and supports
a strict switch via 'enable_thinking=False' in chat template kwargs.

Supports implicit reasoning mode where <think> is injected in the prompt
by AI agents (e.g., OpenCode) and only </think> appears in the output.
"""

import re

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

# Bare-text "thinking process" prefix patterns.
#
# Qwen3 chat templates inject ``<think>\n`` after the assistant generation
# marker when ``enable_thinking=True`` — putting the model in implicit-think
# mode. The model is supposed to emit its chain-of-thought followed by
# ``</think>`` and then the user-facing answer. In practice the model
# sometimes restates the channel boundary inline as a bare-text prefix
# like ``Here's a thinking process:\n\n1. **Analyze...`` (the same shape
# Gemini / older Anthropic models use). When that happens and the model
# also runs out of ``max_tokens`` before producing ``</think>``, the
# entire output is reasoning preamble but neither tag is in the output
# string — so the default "no end token, no start token" branch routes
# the whole thing into ``content`` and ``reasoning_content`` stays empty.
#
# Scoped narrowly to **unambiguous scratchpad labels** — phrases that
# overwhelmingly signal chain-of-thought, identified by (a) being
# known scratchpad nouns and (b) ending with label punctuation (``:``).
#
# Excluded — common direct-answer phrasings (would clobber valid
# answers if the model said them with ``enable_thinking=False`` or
# without ``enable_thinking`` set):
#   * ``Let me think…`` / ``I need to analyze…`` (codex r1 BLOCKING)
#   * Bare ``Step by step:`` / ``Step-by-step:`` (codex r2 BLOCKING)
#   * Bare ``thinking:`` (``Here's my thinking: …``) — the broader
#     ``thinking(?:\s+process)?`` form generated false positives on
#     direct answers (codex r3 BLOCKING)
#   * Bare ``reasoning:`` / ``reasoning process:`` (``Here's my
#     reasoning: …``) — ``reasoning`` alone is a very common direct-
#     answer opener and many legacy callers default to
#     ``enable_thinking=None``; firing on this label clobbers valid
#     responses on the most common code path (codex r4 BLOCKING).
#     ``thinking process``, ``thought process``, ``chain-of-thought``,
#     and ``scratchpad`` survive because they are scratchpad-shaped
#     in a way ``reasoning`` is not.
#   * Verb-form ``Thinking step by step:`` / ``Thinking out loud:`` /
#     ``Thinking through this:`` / ``Thinking carefully:`` /
#     ``Thinking aloud:`` — these are conversational answer openers
#     ("Thinking carefully: Portland is the safest option") and would
#     misclassify when the caller defaults to ``enable_thinking=None``
#     (codex r5 BLOCKING). The unambiguous scratchpad form is always
#     ``Here's [my/a/the] <noun>:`` — the noun-led shape is what makes
#     it scratchpad-shaped, the verb-led form is too conversational.
#
# Match anchored at ``^\s*`` so a normal answer that merely mentions
# a scratchpad noun mid-response is not reclassified.
_BARE_THINK_PREFIX_RE = re.compile(
    r"^(?:\s*)"  # leading whitespace from the injected ``<think>\n``
    r"(?:"
    # "Here's a thinking process:" / "Here's the thought process:" /
    # "Here is the chain-of-thought:" / "Here's the scratchpad:".
    # Must end with ``:`` to ensure it's a scratchpad label, not a
    # casual answer like "Here is the answer: ...". ``reasoning``
    # and ``reasoning process`` are deliberately NOT in this
    # alternation — see the header comment for why.
    r"(?:Here(?:'s|\s+is)\s+(?:my\s+|a\s+|the\s+)?"
    r"(?:thinking\s+process|chain[-\s]of[-\s]thought|"
    r"scratchpad|thought\s+process)"
    r"\s*:)"
    # "My thought process:" — scratchpad label that requires ``:``
    # (e.g. NOT "My thought is that ..."). ``My reasoning process:``
    # is excluded because the same ``reasoning`` over-broadening that
    # bit the ``Here's`` alternation also bites here on the legacy
    # ``enable_thinking=None`` path (codex r4 BLOCKING).
    r"|(?:My\s+thought\s+process\s*:)"
    r")",
    re.IGNORECASE,
)

# Tool-call markup detector used to suppress the bare-text fallback
# when the model embedded a tool call inside what looks like a thinking
# preamble. The fallback would otherwise echo the raw output (including
# ``<tool_call>{...}`` markup) into ``reasoning_content`` — leaking the
# tool tag the route's tool parser already stripped from ``content``.
# Defer to the tool parser by skipping the bare-text branch instead.
# (Codex r2 BLOCKING.)
_TOOL_CALL_MARKUP_RE = re.compile(
    r"<tool_call>|<function=|<\|tool_call\|>|<invoke\s|<minimax:tool_call>",
    re.IGNORECASE,
)


def _looks_like_bare_think_preamble(text: str) -> bool:
    """Return True when ``text`` starts with a known bare-text thinking marker.

    Used as a fallback signal when ``<think>`` was injected by the chat
    template into the prompt (so it is absent from the model output) and
    the model never emitted ``</think>`` before being truncated.

    Returns False when ``text`` contains any tool-call markup so the
    raw tool tags are not echoed into ``reasoning_content`` by the
    fallback (codex r2 BLOCKING — the tool parser already stripped
    them from ``content`` but the reasoning parser otherwise sees the
    raw output unmodified).
    """
    if not text:
        return False
    if _BARE_THINK_PREFIX_RE.match(text) is None:
        return False
    if _TOOL_CALL_MARKUP_RE.search(text):
        return False
    return True


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Qwen3 models.

    Qwen3 uses <think>...</think> tokens to denote reasoning text.

    Supports three scenarios:
    1. Both tags in output: <think>reasoning</think>content
    2. Only closing tag (think in prompt): reasoning</think>content
    3. No tags: pure content

    Example (normal):
        Input: "<think>Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (think in prompt):
        Input: "Let me analyze this...</think>The answer is 42."
        Output: reasoning="Let me analyze this...", content="The answer is 42."

    Example (bare-text thinking preamble, truncated before ``</think>``):
        Input: "Here's a thinking process:\n\n1. Analyze the request..."
        Output: reasoning="Here's a thinking process:\n\n1. Analyze...",
                content=""  # empty-string sentinel, not None — the
                            # upstream ``_finalize_content_and_reasoning``
                            # only blanks ``cleaned_text`` when content
                            # is explicitly ``""``; ``None`` would let
                            # the raw preamble fall through to the client.
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
        Extract reasoning from Qwen3 output.

        Handles both explicit <think>...</think> tags and implicit mode
        where <think> was in the prompt (only </think> in output).

        Args:
            model_output: Complete model output text.
            enable_thinking: Whether the request set
                ``chat_template_kwargs.enable_thinking=True``. When True
                AND neither tag is present, the whole output is treated
                as reasoning (#575 — Qwen3 chat template pre-injects
                ``<think>\\n``; truncation leaves zero tags; pre-#575
                the entire thought trace leaked to ``content``). When
                None / False the no-tag case is treated as plain
                content as before.

        Returns:
            (reasoning, content) tuple.
        """
        # If no end token at all:
        if self.end_token not in model_output:
            # If start token is present, model started thinking but never finished
            # (truncated by max_tokens or garbled by high temperature).
            # Treat everything after <think> as reasoning, content is None.
            if self.start_token in model_output:
                _, _, reasoning = model_output.partition(self.start_token)
                return reasoning.strip() or None, None
            # #575 — when ``enable_thinking=True`` the chat template
            # pre-injected ``<think>\n`` into the prompt, so a no-tag
            # response is a truncated thought trace (the streaming
            # path's Case-3 ``haven't seen </think> yet → reasoning``
            # already routes it correctly; this branch is the
            # non-streaming symmetry). The whole output is reasoning;
            # ``content`` is None so the empty assistant bubble
            # doesn't leak a wall of meta-cognition to the client.
            if enable_thinking is True:
                return model_output.strip() or None, None
            # #570 bare-text fallback: even without the explicit
            # ``enable_thinking=True`` signal (e.g. when callers leave
            # the kwarg defaulted), the chat template still injects
            # ``<think>\n`` for Qwen3 thinking models. If the output
            # opens with a recognizable bare-text thinking marker
            # (``Here's a thinking process:`` and a few close variants
            # — see ``_BARE_THINK_PREFIX_RE``), treat the whole output
            # as reasoning so it surfaces in ``reasoning_content``
            # instead of leaking into ``content``.
            #
            # Gated on ``enable_thinking is not False`` so an explicit
            # ``enable_thinking=False`` from the caller wins: a
            # non-thinking answer that happens to start with
            # ``Here's my reasoning:`` must NOT be reclassified — the
            # caller has affirmatively told us thinking is disabled
            # and clobbering a valid answer would leave the client
            # with empty ``message.content`` (codex r3 BLOCKING).
            # ``None`` (legacy callers that don't thread the flag) and
            # ``True`` (explicit thinking-on) both let the fallback
            # fire defensively.
            #
            # Return ``""`` (not ``None``) for content so the upstream
            # ``_finalize_content_and_reasoning`` overwrites
            # ``cleaned_text`` — the explicit ``<think>...</think>``
            # path relies on ``strip_thinking_tags`` to collapse tagged
            # reasoning to empty downstream, but bare-text reasoning
            # has no tag to strip, so the parser must signal "no
            # content" explicitly here or the original raw output
            # would leak through to the client.
            if enable_thinking is not False and _looks_like_bare_think_preamble(
                model_output
            ):
                return model_output.strip() or None, ""
            # No think tags at all — pure content
            return None, model_output

        # Use base class implementation (handles both explicit and implicit)
        return super().extract_reasoning(model_output, enable_thinking=enable_thinking)

    def finalize_streaming(self, accumulated_text: str) -> DeltaMessage | None:
        """
        Finalize streaming output.

        Three cases:

        1. ``</think>`` seen — reasoning was properly completed. Either the
           model produced content after ``</think>`` (already emitted as
           text_delta), or the stream ended right at ``</think>``. No
           correction needed.

        2. No ``</think>`` AND no leading ``<think>`` prefix — the model
           emitted a bare-text response with NO evidence that thinking
           mode was active. Pre-fix this returned a content correction
           so the Anthropic streaming adapter could ship the buffered
           text as a text_delta; we keep that behaviour here because
           the streaming Case-3 default initially routed the bytes to
           ``reasoning`` and the Anthropic ``finalize_streaming`` path
           treats ``content`` as a flip-to-text directive. (No-evidence
           branch — see ``test_finalize_streaming_bare_preamble_without_think_prefix``.)

        3. ``</think>`` absent BUT a leading ``<think>`` prefix is present
           (the template-injected or model-emitted opener) AND ``stop``
           or ``max_tokens`` cut the stream before the closer arrived.
           This is the D-STOP-THINK leak shape (cross-cycle, six
           parser families): pre-fix we emitted the buffered trace as
           ``content``, which the Anthropic / Responses routes
           appended as a NEW text block — but the streaming loop had
           already shipped every byte as ``reasoning_content``. Clients
           saw the EXACT SAME bytes in both channels.

           Fix: in this case, surface the trace as ``reasoning`` (the
           bare-text preamble fallback already does this — extend the
           same routing to non-preamble trailing thoughts).
           ``final_msg.reasoning`` is silently dropped by the
           Anthropic / Responses routes (they only act on
           ``final_msg.content``), so no extra bytes hit the wire.
           The reasoning emission keeps the parser contract honest
           for callers that inspect ``DeltaMessage`` directly
           (tests, custom routes) without leaking duplicates.
        """
        if self.end_token in accumulated_text:
            # Case 1: proper close tag seen — no correction
            return None
        if accumulated_text:
            # No close tag — strip the template-injected ``<think>``
            # prefix if present so the correction text matches what was
            # streamed.
            saw_think_prefix = accumulated_text.startswith(self.start_token)
            cleaned = (
                accumulated_text[len(self.start_token) :]
                if saw_think_prefix
                else accumulated_text
            )
            if not cleaned:
                return None
            # D-STOP-THINK (cycle-3 F-3 / cycle-5 hermes-qwen3.5-27b-
            # 8bit / cycle-7 nemotron-30b / cycle-11 phi-4-mini-
            # reasoning):
            #
            # The streaming loop has already shipped every reasoning
            # byte as ``reasoning_content`` (base class Case-1
            # ``start_in_prev`` OR Case-3 "no tags yet → reasoning").
            # Stop OR max_tokens cut before ``</think>`` arrived. Pre-
            # fix we returned ``DeltaMessage(content=cleaned)`` here;
            # the Anthropic / Responses ``finalize_streaming`` path
            # then emitted ``cleaned`` as a NEW text block — duplicating
            # the entire thought trace into BOTH the thinking block
            # AND the text block. Cross-cycle repro on qwen3.5-4b-4bit
            # (this parser) and qwen3.5-27b-8bit + hermes (this parser
            # via the qwen3 reasoning route + hermes tool route).
            #
            # Fix (shared base-class invariant via
            # ``_finalize_in_think_block``): emit ``reasoning`` instead
            # of ``content``. The Anthropic / Responses routes only
            # act on ``final_msg.content`` (lines anthropic.py:1715,
            # responses.py:907) so the reasoning emission is silently
            # dropped from the wire — exactly the right outcome
            # because the bytes ALREADY shipped as reasoning during
            # the stream loop. The reasoning emission keeps the
            # parser contract honest for callers that inspect
            # ``DeltaMessage`` directly (tests, custom routes).
            #
            # Branch detail: bare-text preamble surfaces as
            # ``reasoning`` (pre-existing #570 behaviour); non-preamble
            # text ALSO surfaces as ``reasoning`` (D-STOP-THINK fix —
            # the casual-answer "flip to content" protocol assumed the
            # consumer would undo the streamed reasoning, which no
            # route actually implements).
            return DeltaMessage(reasoning=cleaned)
        return None
