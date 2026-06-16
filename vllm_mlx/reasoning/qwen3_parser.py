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
# Scoped narrowly to **explicit scratchpad labels** — phrases that are
# overwhelmingly unambiguous chain-of-thought preambles, identified by
# (a) being known scratchpad nouns (``thinking process``, ``reasoning``,
# ``chain-of-thought``, ``scratchpad``, ``thought process``,
# ``reasoning process``) and (b) ending with a label punctuation
# (``:``).  Conversational phrases like ``Let me think...`` or ``I need
# to analyze...`` are NOT matched — they are common openers for direct
# answers and would be misclassified when ``enable_thinking=False``
# (codex r1 BLOCKING).  Bare ``Step by step:`` / ``Step-by-step:`` is
# also NOT matched — that form is the canonical heading for a direct
# answer to "explain step by step" and would clobber legitimate
# tutorials / how-tos (codex r2 BLOCKING).  The bare ``thinking:``
# label (without ``process``) is also NOT matched — ``Here's my
# thinking:`` is normal user-facing phrasing and the broader form
# generated false positives on direct answers (codex r3 BLOCKING).
# Match anchored at ``^\s*`` so a normal answer that merely mentions
# a scratchpad noun mid-response is not reclassified.
_BARE_THINK_PREFIX_RE = re.compile(
    r"^(?:\s*)"  # leading whitespace from the injected ``<think>\n``
    r"(?:"
    # "Here's a thinking process:" / "Here is the reasoning:" / etc.
    # Must end with ``:`` to ensure it's a scratchpad label, not a
    # casual answer like "Here is the answer: ...". ``thinking``
    # alone is excluded because ``Here's my thinking:`` is normal
    # phrasing — only ``thinking process`` is the scratchpad form
    # (codex r3 BLOCKING).
    r"(?:Here(?:'s|\s+is)\s+(?:my\s+|a\s+|the\s+)?"
    r"(?:thinking\s+process|reasoning|chain[-\s]of[-\s]thought|"
    r"scratchpad|thought\s+process|reasoning\s+process)"
    r"\s*:)"
    # "Thinking step by step:" / "Thinking out loud:" / etc. — only
    # the labelled scratchpad form (terminating ``:`` required).
    # The bare ``Step by step:`` / ``Step-by-step:`` form is excluded
    # because it is the canonical heading for direct "explain
    # step-by-step" answers (tutorials, how-tos).
    r"|(?:Thinking\s+(?:step\s+by\s+step|out\s+loud|through(?:\s+this)?|"
    r"carefully|aloud)\s*:)"
    # "My thought process:" / "My reasoning process:" — scratchpad
    # label that requires ``:`` (e.g. NOT "My thought is that ...").
    r"|(?:My\s+(?:thought|reasoning)\s+process\s*:)"
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
                content=None
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

        1. No tags seen at all — base class classified everything as reasoning
           (to support implicit think). Emit correction with full text.

        2. <think> seen (template injected or model generated) but </think>
           never appeared — model never produced the closing tag. The base
           class classified everything as reasoning. Emit correction with
           full text (stripping the template-injected <think> prefix).

        3. </think> seen — reasoning was properly completed. Either the model
           produced content after </think> (already emitted as text_delta), or
           the stream ended right at </think>. No correction needed.

        Cases 1 and 2 fix a regression in the Anthropic streaming adapter
        (#185 follow-on): when the chat template injects <think> as a prefix,
        _saw_any_tag is set True from the first delta, preventing the original
        no-tags correction. Checking for </think> presence directly handles
        both the template-injected and genuinely-no-thought scenarios.
        """
        if self.end_token in accumulated_text:
            # Case 3: proper close tag seen — no correction
            return None
        if accumulated_text:
            # Cases 1 & 2: no close tag — emit full text as content,
            # stripping the template-injected ``<think>`` prefix if present.
            saw_think_prefix = accumulated_text.startswith(self.start_token)
            cleaned = (
                accumulated_text[len(self.start_token) :]
                if saw_think_prefix
                else accumulated_text
            )
            if not cleaned:
                return None
            # Bare-text thinking fallback (mirrors ``extract_reasoning``):
            # when the chat template injects ``<think>`` and the model is
            # truncated mid-thought before producing ``</think>``, the
            # accumulated text opens with a bare-text "thinking process"
            # preamble. The streaming Case-3 default would surface that
            # preamble as ``content``; keep it in ``reasoning`` instead so
            # OpenAI-compatible clients can distinguish chain-of-thought
            # leakage from the final answer. (Issue #570.)
            #
            # Gated on ``saw_think_prefix`` because ``finalize_streaming``
            # has no ``enable_thinking`` kwarg — the leading
            # ``<think>`` token (template-injected or model-generated)
            # is our only evidence that thinking mode was active for
            # this stream. Without that evidence, a bare-text
            # preamble in the output is more likely a casual answer
            # opener (``Here's a thinking process I followed: ...``)
            # than an actual thought trace, so falling through to the
            # content path matches the pre-fix Anthropic-streaming
            # correction protocol and avoids clobbering valid
            # non-thinking answers (codex r3 BLOCKING symmetry).
            if saw_think_prefix and _looks_like_bare_think_preamble(cleaned):
                return DeltaMessage(reasoning=cleaned)
            return DeltaMessage(content=cleaned)
        return None
