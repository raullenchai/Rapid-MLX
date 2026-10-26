# SPDX-License-Identifier: Apache-2.0
"""Immutable current-turn metadata for the output-parser migration.

This module does not classify generated text. Unknown template contracts stay
unknown; a request's thinking flag is not proof of its rendered initial state.
"""

import re
from dataclasses import dataclass
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
    return "reasoning" if flint and not body else None


def prompt_parse_context(
    prompt: str,
    *,
    thinking: bool | None,
    flint_template: bool = False,
    forced_prefix: str = "",
    prompt_ids: list[int] | None = None,
    tokenizer=None,
) -> PromptParseContext:
    """Resolve only a recognized current assistant turn, never history.

    Suffix IDs are retained only when they exactly match the actual encoded
    prompt tail. This rejects boundary-dependent encodings instead of silently
    presenting independently encoded text as generation's real prompt IDs.
    """
    roles = list((_FLINT_ROLE if flint_template else _CHATML_ROLE).finditer(prompt))
    contract: Literal["chatml", "flint", "unknown"] = (
        "flint" if flint_template else "chatml"
    )
    suffix = ""
    channel: Literal["reasoning", "content"] | None = None
    suffix_ids = None
    if roles and roles[-1][1] == "assistant":
        suffix = prompt[roles[-1].end() :]
        # An ended assistant row is history, not a continued assistant turn.
        if not flint_template and "<|im_end|>" in suffix:
            contract = "unknown"
            suffix = ""
        else:
            channel = _initial_channel(suffix, flint=flint_template)
            if prompt_ids is not None and tokenizer is not None:
                try:
                    encoded = tuple(tokenizer.encode(suffix, add_special_tokens=False))
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    # Metadata must not make a previously valid prompt fail.
                    encoded = None
                if encoded is not None and (
                    not encoded or tuple(prompt_ids[-len(encoded) :]) == encoded
                ):
                    suffix_ids = encoded
    else:
        contract = "unknown"
    return PromptParseContext(
        suffix, suffix_ids, channel, thinking, contract, forced_prefix
    )
