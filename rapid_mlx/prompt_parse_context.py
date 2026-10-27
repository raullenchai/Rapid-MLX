# SPDX-License-Identifier: Apache-2.0
"""Immutable current-turn metadata for the output-parser migration.

This module does not classify generated text. Unknown template contracts stay
unknown; a request's thinking flag is not proof of its rendered initial state.
"""

import re
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class PromptParseContext:
    assistant_suffix: str
    assistant_suffix_ids: tuple[int, ...] | None
    initial_channel: Literal["reasoning", "content"] | None
    requested_thinking: bool | None
    template_contract: Literal["chatml", "flint", "unknown"]
    forced_assistant_prefix: str = ""


_CHATML_ROLE = re.compile(r"<\|im_start\|>(system|developer|user|assistant|tool)\n")
_FLINT_ROLE = re.compile(r"<\|(system|developer|user|assistant|observation)\|>")


def _initial_channel(suffix: str, *, flint: bool):
    # Only a leading control block establishes state. Tags in already-visible
    # prose or JSON are literals; ambiguous continuations await the shared lexer.
    body = suffix.lstrip()
    if body.startswith("<think>"):
        body = body[len("<think>") :]
        close = body.find("</think>")
        tool = body.find("<tool_call>")
        if tool >= 0 and (close < 0 or tool < close):
            return None
        return "content" if close >= 0 else "reasoning"
    if body.startswith("</think>"):
        return "content"
    return "content" if flint and not body else None


def prompt_parse_context(
    prompt: str,
    *,
    thinking: bool | None,
    assistant_start: int | None = None,
    flint_template: bool = False,
    forced_prefix: str = "",
    prompt_ids: list[int] | None = None,
    tokenizer=None,
) -> PromptParseContext:
    """Resolve the renderer-owned assistant boundary, never a marker in history.

    ``assistant_start`` must come from the added generation prefix, or an
    explicitly supplied continuation boundary. Missing evidence stays unknown.

    Suffix IDs are retained only when they exactly match the actual encoded
    prompt tail. This rejects boundary-dependent encodings instead of silently
    presenting independently encoded text as generation's real prompt IDs.
    """
    role = (
        (_FLINT_ROLE if flint_template else _CHATML_ROLE).match(prompt, assistant_start)
        if assistant_start is not None and 0 <= assistant_start <= len(prompt)
        else None
    )
    contract: Literal["chatml", "flint", "unknown"] = (
        "flint" if flint_template else "chatml"
    )
    suffix = ""
    channel: Literal["reasoning", "content"] | None = None
    suffix_ids = None
    if role is not None and role[1] == "assistant":
        suffix = prompt[role.end() :]
        # An ended assistant row is history, not a continued assistant turn.
        if not flint_template and "<|im_end|>" in suffix:
            contract = "unknown"
            suffix = ""
        else:
            channel = _initial_channel(suffix, flint=flint_template)
            if prompt_ids is not None and tokenizer is not None:
                suffix_ids = _verified_suffix_ids(suffix, prompt_ids, tokenizer)
    else:
        contract = "unknown"
    return PromptParseContext(
        suffix, suffix_ids, channel, thinking, contract, forced_prefix
    )


def _verified_suffix_ids(suffix, prompt_ids, tokenizer):
    try:
        encoded = tuple(tokenizer.encode(suffix, add_special_tokens=False))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if not encoded or tuple(prompt_ids[-len(encoded) :]) == encoded:
        return encoded
    return None


class PromptContextCapture:
    """One-shot refinement from scheduler admission; never encodes history."""

    def __init__(self, context: PromptParseContext | None, tokenizer):
        self.context = context
        self.tokenizer = tokenizer

    def __call__(self, prompt_ids):
        if self.context is not None and prompt_ids is not None:
            self.context = replace(
                self.context,
                assistant_suffix_ids=_verified_suffix_ids(
                    self.context.assistant_suffix, prompt_ids, self.tokenizer
                ),
            )
