# SPDX-License-Identifier: Apache-2.0
"""
Token-level output router for LLM generation.

Routes model output tokens into semantic channels (thinking, content, tool_calls)
based on special token IDs read from the tokenizer. No regex, no text matching.

Architecture:
  1. Read special token IDs from tokenizer vocabulary (config-driven)
  2. As tokens stream in, a state machine routes each to the correct channel
  3. Text is decoded only AFTER routing, so partial-token issues are impossible

Usage:
    router = OutputRouter.from_tokenizer(tokenizer)
    for token_id in generation:
        event = router.feed(token_id)
        if event.channel == "content":
            yield event.text
        elif event.channel == "reasoning":
            yield_reasoning(event.text)
        elif event.channel == "tool_call":
            accumulate_tool_call(event.text)

Designed to replace the fragile regex-based strip_special_tokens +
reasoning_parser + tool_call_parser chain with a single unified router.

Currently implements Gemma 4 format. Other models can be added by defining
their token mappings in MODEL_TOKEN_MAPS.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class Channel(Enum):
    """Output channel for a token."""
    CONTENT = auto()
    REASONING = auto()
    TOOL_CALL = auto()
    CONTROL = auto()  # special tokens that should be suppressed


@dataclass
class RouterEvent:
    """A single routed token."""
    channel: Channel
    token_id: int
    text: str  # decoded text for this token


@dataclass
class TokenMap:
    """Special token ID mappings for a model family."""
    # Channel control (Gemma 4 style)
    channel_start: int | None = None      # <|channel> = 100
    channel_end: int | None = None        # <channel|> = 101
    thought_word: int | None = None       # "thought" = 45518
    content_word: int | None = None       # "content" = 3955
    final_word: int | None = None         # "final" = 10218

    # Turn control
    turn_start: int | None = None         # <|turn> = 105
    turn_end: int | None = None           # <turn|> = 106

    # Tool call (Gemma 4 style)
    tool_call_start: int | None = None    # <|tool_call> = 48
    tool_call_end: int | None = None      # <tool_call|> = 49
    tool_quote: int | None = None         # <|"|> = 52

    # Think tags (Qwen/DeepSeek style) — for future migration
    think_start: int | None = None        # <think> token ID
    think_end: int | None = None          # </think> token ID

    # Standard control
    bos: int | None = None
    eos: int | None = None
    pad: int | None = None

    # Set of all control token IDs (suppress from output)
    @property
    def control_ids(self) -> set[int]:
        ids = set()
        for fld in [self.channel_start, self.channel_end,
                     self.turn_start, self.turn_end,
                     self.tool_call_start, self.tool_call_end,
                     self.bos, self.eos, self.pad]:
            if fld is not None:
                ids.add(fld)
        return ids


class RouterState(Enum):
    """State machine states."""
    INIT = auto()
    THINKING = auto()           # inside thought channel
    CONTENT = auto()            # inside content/final channel
    TOOL_CALL = auto()          # inside tool call
    AWAITING_CHANNEL_TYPE = auto()  # saw <|channel>, waiting for thought/content/final


class OutputRouter:
    """
    Token-level output router with state machine.

    Processes token IDs one at a time, routing each to the appropriate
    semantic channel without any text-level regex matching.
    """

    def __init__(self, token_map: TokenMap, tokenizer: Any):
        self.map = token_map
        self.tokenizer = tokenizer
        self.state = RouterState.INIT
        self._tool_tokens: list[int] = []  # accumulated tool call token IDs

    def reset(self):
        """Reset state for a new request."""
        self.state = RouterState.INIT
        self._tool_tokens = []

    def feed(self, token_id: int) -> RouterEvent | None:
        """
        Feed a single token and get the routing decision.

        Returns RouterEvent with the channel assignment, or None if the
        token should be suppressed entirely (control tokens).
        """
        text = self.tokenizer.decode([token_id])
        m = self.map

        # === Control tokens: always suppress ===
        if token_id in (m.bos, m.eos, m.pad):
            return None
        if token_id == m.turn_start or token_id == m.turn_end:
            return None

        # === Channel start: transition to AWAITING_CHANNEL_TYPE ===
        if token_id == m.channel_start:
            self.state = RouterState.AWAITING_CHANNEL_TYPE
            return None  # suppress <|channel>

        # === Channel type word: set state based on which channel ===
        if self.state == RouterState.AWAITING_CHANNEL_TYPE:
            if token_id == m.thought_word:
                self.state = RouterState.THINKING
                return None  # suppress "thought"
            elif token_id == m.content_word or token_id == m.final_word:
                self.state = RouterState.CONTENT
                return None  # suppress "content" / "final"
            else:
                # Unknown channel type — treat as content
                self.state = RouterState.CONTENT
                return RouterEvent(Channel.CONTENT, token_id, text)

        # === Channel end: transition back ===
        if token_id == m.channel_end:
            if self.state == RouterState.THINKING:
                # After thought ends, next content is actual content
                self.state = RouterState.CONTENT
            return None  # suppress <channel|>

        # === Tool call start ===
        if token_id == m.tool_call_start:
            self.state = RouterState.TOOL_CALL
            self._tool_tokens = [token_id]
            return None  # suppress, accumulate

        # === Inside tool call: accumulate ===
        if self.state == RouterState.TOOL_CALL:
            self._tool_tokens.append(token_id)
            if token_id == m.tool_call_end:
                # Tool call complete — return the full tool call text
                full_text = self.tokenizer.decode(self._tool_tokens)
                self.state = RouterState.CONTENT
                self._tool_tokens = []
                return RouterEvent(Channel.TOOL_CALL, token_id, full_text)
            return None  # accumulating, suppress individual tokens

        # === Newline right after channel type word: suppress ===
        if token_id == 107 and self.state in (RouterState.THINKING, RouterState.CONTENT):
            # First \n after "thought\n" or "content\n" — skip
            # (only if it's the very first token in the channel)
            pass  # let it through, it's just whitespace

        # === Default: route based on current state ===
        if self.state == RouterState.THINKING:
            return RouterEvent(Channel.REASONING, token_id, text)
        else:
            # CONTENT, INIT, or any other state → content
            return RouterEvent(Channel.CONTENT, token_id, text)

    def feed_sequence(self, token_ids: list[int]) -> dict[str, str]:
        """
        Feed a complete token sequence and return separated channels.

        Returns:
            {"content": "...", "reasoning": "...", "tool_calls": [...]}
        """
        content = ""
        reasoning = ""
        tool_calls = []

        for tid in token_ids:
            event = self.feed(tid)
            if event is None:
                continue
            if event.channel == Channel.CONTENT:
                content += event.text
            elif event.channel == Channel.REASONING:
                reasoning += event.text
            elif event.channel == Channel.TOOL_CALL:
                tool_calls.append(event.text)

        return {
            "content": content.strip() or None,
            "reasoning": reasoning.strip() or None,
            "tool_calls": tool_calls or None,
        }

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "OutputRouter | None":
        """
        Create an OutputRouter from a tokenizer by reading its vocabulary.

        Returns None if the tokenizer doesn't have the expected special tokens
        (i.e., the model doesn't use a supported format).
        """
        vocab = tokenizer.get_vocab()

        # Gemma 4 detection: look for <|channel> and <|tool_call>
        if "<|channel>" in vocab and "<|tool_call>" in vocab:
            token_map = TokenMap(
                channel_start=vocab["<|channel>"],
                channel_end=vocab["<channel|>"],
                thought_word=vocab.get("thought"),
                content_word=vocab.get("content"),
                final_word=vocab.get("final"),
                turn_start=vocab.get("<|turn>"),
                turn_end=vocab.get("<turn|>"),
                tool_call_start=vocab["<|tool_call>"],
                tool_call_end=vocab.get("<tool_call|>"),
                tool_quote=vocab.get('<|"|>'),
                bos=vocab.get("<bos>"),
                eos=vocab.get("<eos>"),
                pad=vocab.get("<pad>"),
            )
            logger.info(
                "[OutputRouter] Gemma 4 format detected: "
                "channel=%d/%d, tool=%d/%d",
                token_map.channel_start, token_map.channel_end,
                token_map.tool_call_start, token_map.tool_call_end,
            )
            return cls(token_map, tokenizer)

        # Qwen/DeepSeek detection: look for <think> and </think>
        # TODO: implement when migrating existing parsers
        # if "<think>" in vocab and "</think>" in vocab:
        #     ...

        return None  # unsupported model format
