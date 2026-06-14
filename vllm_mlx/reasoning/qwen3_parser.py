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
# Match conservatively at the very start of the output so a normal answer
# that merely mentions "let me think" mid-response is not reclassified.
_BARE_THINK_PREFIX_RE = re.compile(
    r"^(?:\s*)"  # leading whitespace from the injected ``<think>\n``
    r"(?:"
    # English bare-text "thinking process" / "scratchpad" preambles
    r"(?:Here(?:'s|\s+is)\s+(?:my\s+|a\s+|the\s+)?"
    r"(?:thinking(?:\s+process)?|reasoning|chain[-\s]of[-\s]thought|scratchpad|analysis)"
    r"\s*[:.\-])"
    r"|(?:Let\s+me\s+(?:think|analyze|reason|work\s+through|break\s+(?:this|it)\s+down)"
    r"\b)"
    r"|(?:Thinking\s+(?:step\s+by\s+step|out\s+loud|through(?:\s+this)?|carefully|aloud)"
    r"\s*[:.\-]?)"
    r"|(?:(?:Step\s+by\s+step|Step-by-step)[:.\-])"
    r"|(?:I(?:'ll|\s+will|\s+need\s+to|\s+should)\s+(?:think|analyze|reason|"
    r"work\s+through|consider|break\s+(?:this|it)\s+down)\b)"
    r"|(?:Analyzing\s+the\s+(?:user(?:'s)?\s+)?(?:request|question|input|prompt)\b)"
    r"|(?:My\s+(?:thought|reasoning)\s+process\s*[:.\-])"
    r")",
    re.IGNORECASE,
)


def _looks_like_bare_think_preamble(text: str) -> bool:
    """Return True when ``text`` starts with a known bare-text thinking marker.

    Used as a fallback signal when ``<think>`` was injected by the chat
    template into the prompt (so it is absent from the model output) and
    the model never emitted ``</think>`` before being truncated.
    """
    if not text:
        return False
    return _BARE_THINK_PREFIX_RE.match(text) is not None


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
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from Qwen3 output.

        Handles both explicit <think>...</think> tags and implicit mode
        where <think> was in the prompt (only </think> in output).

        Args:
            model_output: Complete model output text.

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
            # Bare-text fallback: the chat template injects ``<think>\n``
            # into the prompt (implicit-think mode), so neither tag appears
            # in the model output when the model is truncated mid-thought.
            # If the output opens with a recognizable bare-text thinking
            # marker, treat the whole output as reasoning so it surfaces
            # in ``reasoning_content`` instead of leaking into ``content``.
            # This mirrors the streaming path which already routes pre-tag
            # text to reasoning while ``</think>`` has not yet been seen.
            #
            # Return ``""`` (not ``None``) for content so the upstream
            # ``_finalize_content_and_reasoning`` overwrites ``cleaned_text``
            # — the explicit ``<think>...</think>`` path relies on
            # ``strip_thinking_tags`` to collapse tagged reasoning to empty
            # downstream, but bare-text reasoning has no tag to strip, so
            # the parser must signal "no content" explicitly here or the
            # original raw output would leak through to the client.
            if _looks_like_bare_think_preamble(model_output):
                return model_output.strip() or None, ""
            # No think tags at all — pure content
            return None, model_output

        # Use base class implementation (handles both explicit and implicit)
        return super().extract_reasoning(model_output)

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
            cleaned = accumulated_text
            if cleaned.startswith(self.start_token):
                cleaned = cleaned[len(self.start_token) :]
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
            if _looks_like_bare_think_preamble(cleaned):
                return DeltaMessage(reasoning=cleaned)
            return DeltaMessage(content=cleaned)
        return None
