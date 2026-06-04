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

Currently implements Gemma 4, Qwen3 / DeepSeek R1 `<think>`-tag, and
GPT-OSS Harmony `<|channel|>` formats. Other models can be added by
defining their token mappings in MODEL_TOKEN_MAPS.
"""

import logging
from dataclasses import dataclass
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

    format_tag: str = ""

    # Channel control (Gemma 4 style)
    channel_start: int | None = None  # <|channel> = 100
    channel_end: int | None = None  # <channel|> = 101
    thought_word: int | None = None  # "thought" = 45518
    content_word: int | None = None  # "content" = 3955
    final_word: int | None = None  # "final" = 10218

    # Turn control
    turn_start: int | None = None  # <|turn> = 105
    turn_end: int | None = None  # <turn|> = 106

    # Tool call (Gemma 4 style)
    tool_call_start: int | None = None  # <|tool_call> = 48
    tool_call_end: int | None = None  # <tool_call|> = 49
    tool_quote: int | None = None  # <|"|> = 52
    tool_start: int | None = None  # <|tool> = 46
    tool_end: int | None = None  # <tool|> = 47
    tool_response_start: int | None = None  # <|tool_response> = 50
    tool_response_end: int | None = None  # <tool_response|> = 51

    # Think tags (Qwen/DeepSeek style)
    think_start: int | None = None  # <think> token ID
    think_end: int | None = None  # </think> token ID

    # Channel control (GPT-OSS/Harmony style)
    harmony_channel: int | None = None  # <|channel|>
    harmony_message: int | None = None  # <|message|>
    harmony_start: int | None = None  # <|start|>
    harmony_end: int | None = None  # <|end|>
    harmony_return: int | None = None  # <|return|>
    harmony_call: int | None = None  # <|call|>
    harmony_constrain: int | None = None  # <|constrain|>
    harmony_analysis_word: int | None = None  # "analysis"
    harmony_final_word: int | None = None  # "final"

    # Standard control
    bos: int | None = None
    eos: int | None = None
    pad: int | None = None


class RouterState(Enum):
    """State machine states."""

    INIT = auto()
    THINKING = auto()  # inside thought channel
    CONTENT = auto()  # inside content/final channel
    TOOL_CALL = auto()  # inside tool call
    AWAITING_CHANNEL_TYPE = auto()  # saw <|channel>, waiting for thought/content/final
    AWAITING_MESSAGE = auto()  # saw Harmony channel name, waiting for <|message|>
    # Saw ``<|start|>`` (harmony header). The next tokens are the role
    # name + optional recipient info (e.g. ``assistant``, ``user``,
    # ``tool to=functions.get_weather``) — all metadata that must be
    # swallowed before the payload starts. We stay here until
    # ``<|channel|>`` (transition to AWAITING_CHANNEL_TYPE) or
    # ``<|message|>`` (transition straight to CONTENT — used by
    # system/user/tool turns that have no channel marker).
    AFTER_START = auto()


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
        self._pending_channel_style: str | None = None
        self._pending_message_channel: Channel | None = None
        self._pending_control_tokens: list[int] = []

    def reset(self):
        """Reset state for a new request."""
        self.state = RouterState.INIT
        self._tool_tokens = []
        self._pending_channel_style = None
        self._pending_message_channel = None
        self._pending_control_tokens = []

    def feed(self, token_id: int) -> RouterEvent | None:
        """
        Feed a single token and get the routing decision.

        Returns RouterEvent with the channel assignment, or None if the
        token should be suppressed entirely (control tokens).
        """
        m = self.map

        # === Control tokens: always suppress (no decode needed) ===
        if token_id in (m.bos, m.eos, m.pad):
            return None
        if token_id == m.turn_start or token_id == m.turn_end:
            return None

        if token_id == m.harmony_channel:
            self.state = RouterState.AWAITING_CHANNEL_TYPE
            self._pending_channel_style = "harmony"
            self._pending_message_channel = None
            self._pending_control_tokens = []
            return None

        if token_id == m.harmony_end or token_id == m.harmony_return:
            self.state = RouterState.CONTENT
            self._pending_channel_style = None
            self._pending_message_channel = None
            self._pending_control_tokens = []
            return None

        # ``<|start|>`` opens a header block: ``<|start|>{role}[ recipient
        # info]<|channel|>{channel}<|message|>...`` (assistant turns) or
        # ``<|start|>{role}<|message|>...`` (system/user/tool turns).
        # Swallow header tokens until the next channel/message marker —
        # otherwise the role name (``assistant``) leaks into ``content``
        # at the start of every second-and-later assistant turn, since
        # without a state transition the role token falls through to the
        # default CONTENT path.
        if token_id == m.harmony_start:
            self.state = RouterState.AFTER_START
            self._pending_channel_style = None
            self._pending_message_channel = None
            self._pending_control_tokens = []
            return None

        if token_id in (m.harmony_call, m.harmony_constrain):
            return None

        # In AFTER_START we swallow every non-special token until the
        # payload boundary is hit. ``<|channel|>`` is handled above (it
        # already transitions to AWAITING_CHANNEL_TYPE). ``<|message|>``
        # opens a payload directly (no channel) — go straight to CONTENT
        # so the body is emitted as content text.
        if self.state == RouterState.AFTER_START:
            if token_id == m.harmony_message:
                self.state = RouterState.CONTENT
                return None
            return None

        # Suppress tool-related markers that may appear without proper nesting
        if token_id in (
            m.tool_response_start,
            m.tool_response_end,
            m.tool_start,
            m.tool_end,
        ):
            return None

        # === Channel start: transition to AWAITING_CHANNEL_TYPE ===
        if token_id == m.channel_start:
            self.state = RouterState.AWAITING_CHANNEL_TYPE
            self._pending_control_tokens = []
            return None  # suppress <|channel>

        # === Channel type word: set state based on which channel ===
        if self.state == RouterState.AWAITING_CHANNEL_TYPE:
            if self._pending_channel_style == "harmony":
                if token_id == m.harmony_analysis_word:
                    self._pending_message_channel = Channel.REASONING
                    self.state = RouterState.AWAITING_MESSAGE
                    return None
                if token_id == m.harmony_final_word:
                    self._pending_message_channel = Channel.CONTENT
                    self.state = RouterState.AWAITING_MESSAGE
                    return None

                self.state = RouterState.CONTENT
                self._pending_channel_style = None
                self._pending_control_tokens = []
                text = self.tokenizer.decode([token_id])
                return RouterEvent(Channel.CONTENT, token_id, text)

            if token_id == m.thought_word:
                self.state = RouterState.THINKING
                return None  # suppress "thought"
            elif token_id == m.content_word or token_id == m.final_word:
                self.state = RouterState.CONTENT
                return None  # suppress "content" / "final"
            else:
                # Unknown channel type — treat as content
                self.state = RouterState.CONTENT
                self._pending_control_tokens = []
                text = self.tokenizer.decode([token_id])
                return RouterEvent(Channel.CONTENT, token_id, text)

        # === Harmony message boundary: suppress metadata before payload ===
        if self.state == RouterState.AWAITING_MESSAGE:
            if token_id == m.harmony_message:
                self.state = (
                    RouterState.THINKING
                    if self._pending_message_channel == Channel.REASONING
                    else RouterState.CONTENT
                )
                self._pending_channel_style = None
                self._pending_message_channel = None
                self._pending_control_tokens = []
            else:
                self._pending_control_tokens.append(token_id)
            return None

        if token_id == m.harmony_message:
            return None

        # === Channel end: transition back ===
        if token_id == m.channel_end:
            if self.state == RouterState.THINKING:
                self.state = RouterState.CONTENT
            return None  # suppress <channel|>

        # === Orphan tool call end (no matching start): suppress ===
        if token_id == m.tool_call_end and self.state != RouterState.TOOL_CALL:
            return None

        # === Tool call start ===
        if token_id == m.tool_call_start:
            self.state = RouterState.TOOL_CALL
            self._tool_tokens = [token_id]
            return None

        # === Inside tool call: accumulate (no per-token decode) ===
        if self.state == RouterState.TOOL_CALL:
            self._tool_tokens.append(token_id)
            if token_id == m.tool_call_end:
                full_text = self.tokenizer.decode(self._tool_tokens)
                self.state = RouterState.CONTENT
                self._tool_tokens = []
                return RouterEvent(Channel.TOOL_CALL, token_id, full_text)
            return None

        # === Think tags (Qwen / DeepSeek style) ===
        if token_id == m.think_start:
            self.state = RouterState.THINKING
            return None

        if token_id == m.think_end:
            self.state = RouterState.CONTENT
            return None

        # === Bare Gemma 4 channel-type words (issue #447) ===
        #
        # The Gemma 4 channel-marker words (``thought``/``content``/
        # ``final``) have dedicated single-token IDs in the vocab. The
        # standard chat-template wraps each in ``<|channel>...<channel|>``,
        # but some models on the unconstrained generation path emit
        # the bare word without the ``<|channel>`` opener (see issue
        # #447). The AWAITING_CHANNEL_TYPE branch above only triggers
        # on ``<|channel>``, so bare words used to fall through to the
        # default emit and leak as literal CONTENT text (e.g.
        # ``content="thought\nanalysis_body\nfinal\nmessage_body"``).
        #
        # Treat bare channel-type words as transitions from any state
        # except AWAITING_CHANNEL_TYPE (already handled above) and
        # TOOL_CALL (handled above). The IDs are ordinary vocab entries
        # (verified against the real Gemma 4 tokenizer: thought=45518,
        # content=3955, final=10218 — distinct from the structural
        # markers like ``<|channel>`` which are added_tokens), but a
        # natural-language ``final`` inside content tokenizes to a
        # different vocab entry (e.g. ``▁final`` with a leading
        # space-marker), so direct ID matching is unambiguous in
        # practice.
        if m.thought_word is not None and token_id == m.thought_word:
            self.state = RouterState.THINKING
            return None
        if m.content_word is not None and token_id == m.content_word:
            self.state = RouterState.CONTENT
            return None
        if m.final_word is not None and token_id == m.final_word:
            self.state = RouterState.CONTENT
            return None

        # === Default: decode and route based on current state ===
        text = self.tokenizer.decode([token_id])
        if self.state == RouterState.THINKING:
            return RouterEvent(Channel.REASONING, token_id, text)
        else:
            return RouterEvent(Channel.CONTENT, token_id, text)

    def finalize(self) -> RouterEvent | None:
        """Drain any buffered state at stream end.

        This is best-effort: complete channel transitions are still handled by
        feed(), while finalize() preserves buffered tool calls or pending
        Harmony pre-message text that would otherwise be dropped.
        """
        if self.state == RouterState.TOOL_CALL and self._tool_tokens:
            token_id = self._tool_tokens[-1]
            text = self.tokenizer.decode(self._tool_tokens)
            self.state = RouterState.CONTENT
            self._tool_tokens = []
            return RouterEvent(Channel.TOOL_CALL, token_id, text)

        if self.state in (
            RouterState.AWAITING_CHANNEL_TYPE,
            RouterState.AWAITING_MESSAGE,
        ):
            if self._pending_control_tokens:
                channel = self._pending_message_channel or Channel.CONTENT
                token_id = self._pending_control_tokens[-1]
                text = self.tokenizer.decode(self._pending_control_tokens)
                self.state = RouterState.CONTENT
                self._pending_channel_style = None
                self._pending_message_channel = None
                self._pending_control_tokens = []
                if text.strip():
                    return RouterEvent(channel, token_id, text)
            else:
                self.state = RouterState.CONTENT
                self._pending_channel_style = None
                self._pending_message_channel = None

        return None

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

        event = self.finalize()
        if event is not None:
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
                format_tag="gemma4",
                channel_start=vocab.get("<|channel>"),
                channel_end=vocab.get("<channel|>"),
                thought_word=vocab.get("thought"),
                content_word=vocab.get("content"),
                final_word=vocab.get("final"),
                turn_start=vocab.get("<|turn>"),
                turn_end=vocab.get("<turn|>"),
                tool_call_start=vocab.get("<|tool_call>"),
                tool_call_end=vocab.get("<tool_call|>"),
                tool_quote=vocab.get('<|"|>'),
                tool_start=vocab.get("<|tool>"),
                tool_end=vocab.get("<tool|>"),
                tool_response_start=vocab.get("<|tool_response>"),
                tool_response_end=vocab.get("<tool_response|>"),
                bos=vocab.get("<bos>"),
                eos=vocab.get("<eos>"),
                pad=vocab.get("<pad>"),
            )
            logger.info(
                "[OutputRouter] Gemma 4 format detected: channel=%d/%d, tool=%d/%d",
                token_map.channel_start,
                token_map.channel_end,
                token_map.tool_call_start,
                token_map.tool_call_end,
            )
            return cls(token_map, tokenizer)

        # GPT-OSS/Harmony detection: channel/message special tokens.
        if "<|channel|>" in vocab and "<|message|>" in vocab:
            token_map = TokenMap(
                format_tag="harmony",
                harmony_channel=vocab.get("<|channel|>"),
                harmony_message=vocab.get("<|message|>"),
                harmony_start=vocab.get("<|start|>"),
                harmony_end=vocab.get("<|end|>"),
                harmony_return=vocab.get("<|return|>"),
                harmony_call=vocab.get("<|call|>"),
                harmony_constrain=vocab.get("<|constrain|>"),
                harmony_analysis_word=vocab.get("analysis"),
                harmony_final_word=vocab.get("final"),
                bos=vocab.get("<|endoftext|>"),
                eos=vocab.get("<|endoftext|>"),
                pad=vocab.get("<|endoftext|>"),
            )
            logger.info(
                "[OutputRouter] Harmony format detected: channel=%d, message=%d",
                token_map.harmony_channel,
                token_map.harmony_message,
            )
            return cls(token_map, tokenizer)

        # Qwen3 / DeepSeek R1 detection: <think>...</think> reasoning tags.
        # DeepSeek's BOS/EOS use unicode brackets; fall back to plain <bos>/<eos>
        # for Qwen3 (which has neither in its vocab).
        if "<think>" in vocab and "</think>" in vocab:
            token_map = TokenMap(
                format_tag="think",
                think_start=vocab.get("<think>"),
                think_end=vocab.get("</think>"),
                bos=vocab.get("<｜begin▁of▁sentence｜>") or vocab.get("<bos>"),
                eos=vocab.get("<｜end▁of▁sentence｜>") or vocab.get("<eos>"),
                pad=vocab.get("<pad>"),
            )
            logger.info(
                "[OutputRouter] Think tag format detected: think=%d/%d",
                token_map.think_start,
                token_map.think_end,
            )
            return cls(token_map, tokenizer)

        return None  # unsupported model format
