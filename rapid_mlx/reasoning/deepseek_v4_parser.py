# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 reasoning parser.

A bare ``</think>`` is a control token even when thinking is disabled,
and a DSML tool-call opener implicitly closes an unfinished reasoning block.
Inside a DSML tool-call block, reasoning markers are literal argument text.
"""

from __future__ import annotations

from .base import DeltaMessage, ReasoningParser

_THINK_START = "<think>"
_THINK_END = "</think>"
_DSML_TOOL_START = "<｜DSML｜tool_calls>"
_DSML_TOOL_START_R = "<｜DSML｜r:tool_calls>"
_DSML_TOOL_END = "</｜DSML｜tool_calls>"
_CONTROL_TOKENS = (
    _THINK_START,
    _THINK_END,
    _DSML_TOOL_START,
    _DSML_TOOL_START_R,
)


class DeepSeekV4ReasoningParser(ReasoningParser):
    """Incremental DeepSeek V4 ``<think>``/DSML state machine."""

    # Unlike generic reasoning parsers, this parser must remain active in chat
    # mode to absorb the protocol's bare ``</think>`` control token.
    sanitize_when_thinking_disabled = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset_state()

    def reset_state(
        self,
        *,
        start_in_thinking: bool = False,
        thinking_enabled: bool | None = None,
    ):
        super().reset_state()
        self._in_reasoning = start_in_thinking
        self._in_tool_call = False
        self._thinking_enabled = (
            start_in_thinking if thinking_enabled is None else thinking_enabled
        )
        self._suppress_reasoning = False
        self._buffer = ""

    def configure_request(self, *, enable_thinking: bool | None = None) -> None:
        # DeepSeek V4 emits implicit scratch reasoning without an opening
        # marker.  Treat an unspecified request like the model's default
        # thinking mode; only an explicit False starts in visible content.
        self.reset_state(
            start_in_thinking=enable_thinking is not False,
            thinking_enabled=enable_thinking is not False,
        )

    @staticmethod
    def _partial_control_suffix(text: str, tokens: tuple[str, ...]) -> int:
        best = 0
        for token in tokens:
            for size in range(1, min(len(text), len(token) - 1) + 1):
                if text.endswith(token[:size]):
                    best = max(best, size)
        return best

    def _consume(self, text: str, *, final: bool = False) -> DeltaMessage | None:
        pending = self._buffer + text
        self._buffer = ""
        content: list[str] = []
        reasoning: list[str] = []

        def emit(value: str) -> None:
            if not value or self._suppress_reasoning:
                return
            (reasoning if self._in_reasoning else content).append(value)

        while pending:
            # Tool arguments are opaque to reasoning extraction. Hold split
            # closers using the same carry buffer as other control markers.
            tokens = (_DSML_TOOL_END,) if self._in_tool_call else _CONTROL_TOKENS
            matches = [
                (idx, token) for token in tokens if (idx := pending.find(token)) >= 0
            ]
            if not matches:
                held = 0 if final else self._partial_control_suffix(pending, tokens)
                emit(pending[:-held] if held else pending)
                if held:
                    self._buffer = pending[-held:]
                break

            index, token = min(matches, key=lambda match: match[0])
            emit(pending[:index])
            pending = pending[index + len(token) :]
            if token == _DSML_TOOL_END:
                content.append(token)
                self._in_tool_call = False
            elif token == _THINK_START:
                # With thinking explicitly disabled both markers are only
                # sanitizers; never create a reasoning channel the request
                # opted out of. Otherwise absorb duplicate openers while
                # remaining in reasoning mode.
                self._suppress_reasoning = not self._thinking_enabled
                self._in_reasoning = self._thinking_enabled
            elif token == _THINK_END:
                # Absorb a bare closer even in content state.
                self._suppress_reasoning = False
                self._in_reasoning = False
            else:
                # A DSML tool call is always outside reasoning. Preserve its
                # opener for the downstream DeepSeek V4 tool parser.
                self._suppress_reasoning = False
                self._in_reasoning = False
                self._in_tool_call = True
                content.append(token)

        normal = "".join(content) or None
        thought = "".join(reasoning) or None
        if normal is None and thought is None:
            return None
        return DeltaMessage(content=normal, reasoning=thought)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        del previous_text, current_text
        return self._consume(delta_text)

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
    ) -> tuple[str | None, str | None]:
        self.reset_state(
            start_in_thinking=enable_thinking is not False,
            thinking_enabled=enable_thinking is not False,
        )
        parsed = self._consume(model_output, final=True)
        if parsed is None:
            return None, None
        return parsed.reasoning, parsed.content

    def finalize_streaming(
        self,
        accumulated_text: str,
        **_kwargs,
    ) -> DeltaMessage | None:
        del accumulated_text
        return self._consume("", final=True)

    def is_open_in_think(self, _accumulated_text: str) -> bool:
        return self._in_reasoning
