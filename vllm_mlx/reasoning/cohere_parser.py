# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Cohere North thinking envelopes."""

from .think_parser import BaseThinkingReasoningParser


class CohereReasoningParser(BaseThinkingReasoningParser):
    """Split North's private reasoning from its final text/action lane.

    North's chat template places ``<|START_THINKING|>`` in the prompt, so
    normal generation begins with reasoning text and emits only the closing
    marker before the final text or action.  ``BaseThinkingReasoningParser``
    already implements that implicit-start shape for both complete and
    streaming output; this class supplies the checkpoint's native sentinels.
    """

    # North's checked-in chat template always opens the thinking lane and
    # does not implement the Qwen-style ``enable_thinking`` switch.  The
    # streaming dispatcher must therefore keep this parser engaged even when
    # the generic request resolver defaults that optional flag to ``False``.
    always_route = True

    @property
    def start_token(self) -> str:
        return "<|START_THINKING|>"

    @property
    def end_token(self) -> str:
        return "<|END_THINKING|>"
