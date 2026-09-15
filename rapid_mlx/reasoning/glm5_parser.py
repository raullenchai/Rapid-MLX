# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for GLM-5 models with prompt-primed thinking.

GLM-5.3's shipped chat template ends every generation prompt with
``<think>``.  Generated text therefore starts inside the reasoning lane and
only ``</think>`` transitions to public content.  This differs from GLM-4,
whose model output decides autonomously whether to emit an opener.

The wire protocol uses the shared thinking-tag state machine, but deliberately
does not inherit a model family's no-tag fallback threshold: GLM-5 is known to
be inside reasoning until the closing token arrives.
"""

from .think_parser import BaseThinkingReasoningParser


class Glm5ReasoningParser(BaseThinkingReasoningParser):
    """Parse GLM-5's implicit ``<think>``-until-``</think>`` protocol."""

    implicit_reasoning_until_close = True
    sanitize_when_thinking_disabled = True

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def configure_request(
        self,
        *,
        enable_thinking: bool | None = None,
        prompt_thinking_active: bool | None = None,
    ) -> None:
        del enable_thinking
        self.reset_state()
        self._prompt_primed_thinking = prompt_thinking_active is True

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
        prompt_thinking_active: bool | None = None,
    ) -> tuple[str | None, str | None]:
        del enable_thinking
        if self.start_token not in model_output and self.end_token not in model_output:
            if prompt_thinking_active is True:
                return self._promote_tool_calls(model_output.strip() or None, None)
            return None, model_output
        return super().extract_reasoning(model_output)
