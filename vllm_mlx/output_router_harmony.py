# SPDX-License-Identifier: Apache-2.0
"""
Harmony-format streaming router backed by ``openai-harmony.StreamableParser``.

Issue #513 / cluster #444/#455/#468/#480: the custom Gemma 4–style
``OutputRouter`` state machine cannot model the harmony protocol's
tool-call channel reliably. Production ``commentary`` is two tokens
(``comment`` + ``ary``) so a single-token channel-type match never
fires; the recipient string (``functions.<name>``) and constrain
directive (``<|constrain|>json``) are multi-token literals that the
naive router state machine swallows or leaks. The marker-preserving
redesign discussed in PR #514 / #513 is exactly the behavior
``openai-harmony``'s ``StreamableParser`` already implements — and is
the same library vLLM and SGLang delegate to for gpt-oss tool calls.

This module exposes ``HarmonyStreamingRouter``, a shim that exposes the
same ``feed`` / ``finalize`` / ``reset`` / ``feed_sequence`` / ``map``
surface as ``OutputRouter`` so the engine streaming path
(``BatchedEngine._stream_with_output_router``) and the non-stream
sequence path (``_finalize_with_router_sequence``) can pick it up
without changes. The underlying state, channel transitions, recipient
parsing, and constraint-type detection all come from
``StreamableParser`` — we do not maintain a parallel state machine.

Token ID compatibility: ``mlx-community/gpt-oss-20b-MXFP4-Q8`` (and the
upstream gpt-oss family) use exactly the harmony encoding's IDs for
the structural markers and for body tokens — verified at PR-time by
encoding ``<|channel|>``, ``<|message|>``, ``<|call|>``, ``<|end|>``,
``<|return|>``, ``<|start|>``, ``<|constrain|>`` and a multi-token body
string through both the model's HF tokenizer and the harmony encoding
and asserting set equality. So we can feed model-emitted token IDs
directly to ``StreamableParser`` without re-encoding.
"""

from __future__ import annotations

import logging
from typing import Any

from .output_router import Channel, RouterEvent, TokenMap

logger = logging.getLogger(__name__)

# Channel name strings emitted by openai-harmony StreamableParser.
_HARMONY_CHANNEL_ANALYSIS = "analysis"
_HARMONY_CHANNEL_FINAL = "final"
_HARMONY_CHANNEL_COMMENTARY = "commentary"


class HarmonyStreamingRouter:
    """Duck-typed replacement for ``OutputRouter`` on harmony-format
    models. Delegates state tracking to ``openai-harmony.StreamableParser``.

    The class deliberately mirrors ``OutputRouter``'s public API:
    ``feed(tid) -> RouterEvent | None``,
    ``finalize() -> RouterEvent | None``,
    ``reset() -> None``,
    ``feed_sequence(token_ids) -> dict``, and a ``.map`` attribute
    containing a ``TokenMap`` so callers that read
    ``router.map.format_tag`` (e.g. allowlist filtering in
    ``BatchedEngine._create_output_router``) work unchanged.
    """

    def __init__(self, token_map: TokenMap, tokenizer: Any):
        # Import inside __init__ so module import is cheap even when
        # the optional dep is missing — discovery code can decide whether
        # to construct this class or fall back to a different router.
        from openai_harmony import (
            HarmonyEncodingName,
            Role,
            StreamableParser,
            load_harmony_encoding,
        )

        self.map = token_map
        self.tokenizer = tokenizer
        self._enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._role = Role.ASSISTANT
        self._StreamableParser = StreamableParser
        self._parser = StreamableParser(self._enc, role=self._role)
        # Index of the last message we already surfaced as a TOOL_CALL
        # event — used to detect freshly-closed commentary messages.
        self._emitted_msg_count = 0

    def reset(self) -> None:
        """Reset state for a new request — re-create the parser."""
        self._parser = self._StreamableParser(self._enc, role=self._role)
        self._emitted_msg_count = 0

    @staticmethod
    def _reconstruct_tool_call_text(message: Any) -> str:
        """Render a commentary ``Message`` back into the text form the
        downstream ``HarmonyToolParser`` expects.

        ``StreamableParser.messages[i]`` gives us a structured
        ``Message`` with channel / recipient / content_type / content,
        but the rest of the route pipeline (postprocessor + chat route
        + ``HarmonyToolParser.extract_tool_calls``) is text-based. To
        avoid changing that contract for one router, we reconstruct
        the canonical wire format:
        ``<|channel|>commentary to=functions.<name>
        [<|constrain|>json]<|message|>{body}<|call|>``
        and hand it through as a single TOOL_CALL channel event. The
        text parser then extracts the structured call.
        """
        body = ""
        for c in message.content:
            t = getattr(c, "text", None)
            if t:
                body += t
        parts: list[str] = ["<|channel|>commentary"]
        recipient = getattr(message, "recipient", None)
        if recipient:
            parts.append(f" to={recipient}")
        ctype = getattr(message, "content_type", None)
        if ctype:
            # ``content_type`` from StreamableParser is e.g.
            # ``"<|constrain|>json"``; emit verbatim.
            parts.append(f" {ctype}")
        parts.append("<|message|>")
        parts.append(body)
        parts.append("<|call|>")
        return "".join(parts)

    def feed(self, token_id: int) -> RouterEvent | None:
        """Feed one token and emit the routed event, if any.

        Routing rules:
          * Channel ``analysis`` → Channel.REASONING with the
            parser's ``last_content_delta`` for this token.
          * Channel ``final`` → Channel.CONTENT with the
            ``last_content_delta``.
          * Channel ``commentary`` (tool call): suppress per-token
            deltas during the body. When the message closes (parser
            transitions out of CONTENT to EXPECT_START and adds an
            entry to ``messages``), emit a single Channel.TOOL_CALL
            event carrying the reconstructed wire-format text so the
            downstream HarmonyToolParser can extract the structured
            call.
          * Anything else (control tokens, headers, transitions) →
            None.
        """
        try:
            self._parser.process(token_id)
        except Exception as e:
            # The model emitted a token sequence the harmony parser
            # can't follow (e.g. corrupted output, mid-stream
            # truncation). Surface as a router failure so the engine
            # falls back to the legacy text-based parsers — see
            # ``BatchedEngine._stream_with_output_router``'s
            # ``except Exception`` handler at the call site.
            raise RuntimeError(
                f"HarmonyStreamableParser rejected token_id={token_id}: {e}"
            ) from e

        # Did a message just close? StreamableParser appends to
        # ``messages`` when it sees ``<|end|>`` / ``<|return|>`` /
        # ``<|call|>``. A freshly-closed commentary message with a
        # recipient is a tool call ready to surface.
        new_msg_count = len(self._parser.messages)
        if new_msg_count > self._emitted_msg_count:
            closed = self._parser.messages[-1]
            self._emitted_msg_count = new_msg_count
            if getattr(
                closed, "channel", None
            ) == _HARMONY_CHANNEL_COMMENTARY and getattr(closed, "recipient", None):
                text = self._reconstruct_tool_call_text(closed)
                return RouterEvent(Channel.TOOL_CALL, token_id, text)

        # Per-token body delta routing for analysis / final.
        ch = self._parser.current_channel
        delta = self._parser.last_content_delta
        if delta is None or delta == "":
            return None
        if ch == _HARMONY_CHANNEL_ANALYSIS:
            return RouterEvent(Channel.REASONING, token_id, delta)
        if ch == _HARMONY_CHANNEL_FINAL:
            return RouterEvent(Channel.CONTENT, token_id, delta)
        # commentary body deltas are buffered; emission happens on
        # message close above. Any other channel ID (None during
        # headers, unknown future channels) → no emission.
        return None

    def finalize(self) -> RouterEvent | None:
        """End-of-stream drain.

        ``StreamableParser`` may have an in-progress message that never
        reached ``<|end|>`` / ``<|return|>`` / ``<|call|>`` (truncated
        generation). Force-close via ``process_eos`` so any buffered
        commentary message surfaces as a TOOL_CALL event.
        """
        try:
            self._parser.process_eos()
        except Exception as e:  # noqa: BLE001
            logger.debug("StreamableParser.process_eos failed: %s", e)
            return None

        new_msg_count = len(self._parser.messages)
        if new_msg_count > self._emitted_msg_count:
            closed = self._parser.messages[-1]
            self._emitted_msg_count = new_msg_count
            if getattr(
                closed, "channel", None
            ) == _HARMONY_CHANNEL_COMMENTARY and getattr(closed, "recipient", None):
                text = self._reconstruct_tool_call_text(closed)
                # No "current" token id at finalize — pick the parser's
                # last processed token if available, else 0.
                token_id = self._parser.tokens[-1] if self._parser.tokens else 0
                return RouterEvent(Channel.TOOL_CALL, token_id, text)
        return None

    def feed_sequence(self, token_ids: list[int]) -> dict[str, Any]:
        """Batch path: route a complete token sequence and return
        the same separated-channels dict as ``OutputRouter.feed_sequence``.

        Returns:
            ``{"content": str|None, "reasoning": str|None,
               "tool_calls": list[str]|None}``
        """
        content = ""
        reasoning = ""
        tool_calls: list[str] = []

        def _accumulate(event: RouterEvent | None) -> None:
            nonlocal content, reasoning
            if event is None:
                return
            if event.channel == Channel.CONTENT:
                content += event.text
            elif event.channel == Channel.REASONING:
                reasoning += event.text
            elif event.channel == Channel.TOOL_CALL:
                tool_calls.append(event.text)

        for tid in token_ids:
            _accumulate(self.feed(tid))
        # Drain end-of-stream (truncated messages, etc.).
        _accumulate(self.finalize())

        return {
            "content": content.strip() or None,
            "reasoning": reasoning.strip() or None,
            "tool_calls": tool_calls or None,
        }


def is_openai_harmony_available() -> bool:
    """Return True iff the optional ``openai-harmony`` dep can be
    imported. The detection caller (``OutputRouter.from_tokenizer``)
    uses this to decide whether to construct the new harmony router
    or fall back to the legacy custom state machine.
    """
    try:
        import openai_harmony  # noqa: F401

        return True
    except ImportError:
        return False


def is_openai_harmony_compatible(token_map: TokenMap) -> bool:
    """Return True iff the model's harmony special-token IDs match the
    ``openai-harmony`` encoding's IDs AND the optional dep is present.

    Why the gate exists: ``StreamableParser`` consumes integer token
    IDs from its own encoding's vocabulary. Production upstream
    ``mlx-community/gpt-oss-20b-MXFP4-Q8`` (and the gpt-oss family in
    general) emit the SAME IDs the harmony encoding uses for
    ``<|channel|>`` / ``<|message|>`` / ``<|call|>`` / ``<|end|>`` /
    ``<|return|>`` / ``<|start|>`` / ``<|constrain|>``, so we can
    forward model token IDs directly without re-encoding (verified
    PR-time). Synthetic test vocabs that map ``<|channel|>`` to a
    different ID would feed ``StreamableParser`` IDs it doesn't
    recognize as control tokens, producing garbled output. This
    gate short-circuits to the legacy state machine in that case.
    """
    if not is_openai_harmony_available():
        return False
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except Exception:  # noqa: BLE001
        return False
    # Verify each known harmony marker the TokenMap recorded matches
    # what the harmony encoder produces. Any miss → not compatible.
    pairs = (
        (token_map.harmony_channel, "<|channel|>"),
        (token_map.harmony_message, "<|message|>"),
        (token_map.harmony_call, "<|call|>"),
        (token_map.harmony_end, "<|end|>"),
        (token_map.harmony_return, "<|return|>"),
        (token_map.harmony_start, "<|start|>"),
        (token_map.harmony_constrain, "<|constrain|>"),
    )
    for model_id, marker in pairs:
        if model_id is None:
            continue
        try:
            harmony_ids = enc.encode(marker, allowed_special="all")
        except Exception:  # noqa: BLE001
            return False
        if harmony_ids != [model_id]:
            return False
    return True
