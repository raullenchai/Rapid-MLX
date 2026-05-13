# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for the GLM-4 family (GLM-4.5-Air, GLM-4.6V, GLM-4.7).

GLM-4 uses ``<think>...</think>`` tags for reasoning content. Unlike
Qwen3 and DeepSeek-R1, GLM-4's chat template does NOT inject ``<think>``
in the prompt — the model decides autonomously whether to reason. So:

- Output with both tags: reasoning + content (standard case).
- Output with no tags at all: pure content, NOT implicit reasoning.
  The base class' "no tags yet → treat as reasoning" default is correct
  for Qwen3 (where ``<think>`` is injected in the prompt and missing
  tags mean truncated reasoning) but wrong for GLM-4. This parser
  overrides that one branch.

GLM-4.6V additionally wraps content in
``<|begin_of_box|>...<|end_of_box|>`` container markers; we strip them
so the user-facing content stays clean.

Adapted from upstream waybarrios/vllm-mlx#295 (``Glm4ReasoningParser``).
Upstream uses a ``_phase`` enum state machine; our base class uses a
``_saw_any_tag`` flag (see ``think_parser.py``). The behavioural
divergence is the same — only the override surface differs.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_BOX_START = "<|begin_of_box|>"
_BOX_END = "<|end_of_box|>"


class Glm4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for the GLM-4 family.

    Diverges from the base class on exactly one streaming path: when
    neither ``<think>`` nor ``</think>`` has appeared yet, GLM-4 output
    is pure content. The base class defaults to reasoning for that
    branch (correct for Qwen3-style prompt injection, wrong here).

    Also strips GLM-4.6V ``<|begin_of_box|>`` / ``<|end_of_box|>``
    container markers in both the streaming and non-streaming paths.
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    @staticmethod
    def _strip_box(text: str) -> str:
        return text.replace(_BOX_START, "").replace(_BOX_END, "")

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        # Strip 4.6V box markers before tag inspection. They're whole
        # special tokens, never embedded inside actual reasoning prose,
        # so a literal replace is safe.
        return super().extract_reasoning(self._strip_box(model_output))

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        # 4.6V box tags are whole special tokens, never split across
        # deltas. Strip from delta + accumulated text so downstream tag
        # detection sees the clean string.
        if _BOX_START in delta_text or _BOX_END in delta_text:
            delta_text = self._strip_box(delta_text)
            if not delta_text:
                return None
            previous_text = self._strip_box(previous_text)
            current_text = self._strip_box(current_text)

        # Diverge from the base class here, and only here. GLM-4 doesn't
        # inject ``<think>``, so an early token before any tag appears
        # is genuine content, not implicit reasoning. The base class
        # default (``reasoning=delta_text``) would misclassify it.
        has_tags = self.start_token in current_text or self.end_token in current_text
        if not has_tags and not self._saw_any_tag:
            return DeltaMessage(content=delta_text)

        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )
