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
import re
from typing import Any

from .output_router import Channel, RouterEvent, TokenMap

logger = logging.getLogger(__name__)

# Channel name strings emitted by openai-harmony StreamableParser.
_HARMONY_CHANNEL_ANALYSIS = "analysis"
_HARMONY_CHANNEL_FINAL = "final"
_HARMONY_CHANNEL_COMMENTARY = "commentary"

# Tool-call recipient shape: ``functions.<safe-name>``. The downstream
# ``HarmonyToolParser`` does its own parsing of this field but reads
# it verbatim from the reconstructed wire text — a recipient with
# whitespace, marker-like characters, or newlines could break the
# parser's regex or leak content. Codex round-1 BLOCKING (PR #515).
#
# Codex round-2 BLOCKING: original ``^functions\.[A-Za-z_]...`` rejected
# OpenAI-spec valid tool names that start with a digit
# (e.g. ``functions.2fa_lookup``). OpenAI's documented function-name
# regex is ``[a-zA-Z0-9_-]{1,64}`` — match that exactly so any tool the
# upstream API accepts also round-trips through the router.
_RECIPIENT_SHAPE = re.compile(r"^functions\.[A-Za-z0-9_\-]{1,64}$")

# Tokenizer-identity allowlist — known-compatible HF / mlx-community
# names whose vocab is the harmony encoding. Codex round-2 BLOCKING:
# matching a 3-string probe set against the harmony encoding is not
# enough to prove full-vocab parity — a tokenizer with the right
# markers and the right probes but a remapped uncommon token could
# silently corrupt later content. The cleanest defense is to ALSO
# require the tokenizer's reported identity to be a known gpt-oss
# family member. ``in`` substring match keeps the gate robust to MLX
# quant-suffix renames (``-MXFP4-Q8`` etc.).
_KNOWN_HARMONY_TOKENIZERS = (
    "gpt-oss",  # openai/gpt-oss-* and any mlx-community/gpt-oss-*-MXFP4-Q8 variant
)

# Probe strings used by ``is_openai_harmony_compatible`` to verify
# that the model's body-token vocabulary matches the openai-harmony
# encoding's vocabulary. If any probe round-trips to a different ID
# list, the gate falls back to the legacy router — feeding mismatched
# IDs to ``StreamableParser`` decodes bodies through the wrong vocab
# and corrupts content / tool-call arguments (codex round-1 BLOCKING).
# Pick short strings that exercise common body-vocab regions: plain
# English, JSON-shaped text, and the smoking-gun multi-token word
# ``commentary`` from PR #514 (``comment``+``ary`` on gpt-oss-20b).
_BODY_VOCAB_PROBES = (
    "Hello world",
    'functions.get_weather {"a":1}',
    "commentary",
)


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
        from openai_harmony import Role, StreamableParser

        self.map = token_map
        self.tokenizer = tokenizer
        # Codex round-3 NIT: reuse the module-level cached encoding so
        # the relatively expensive ``load_harmony_encoding`` only runs
        # once per process instead of once per request.
        self._enc = _get_harmony_encoding()
        if self._enc is None:
            raise RuntimeError(
                "HarmonyStreamingRouter: openai_harmony.load_harmony_encoding "
                "is unavailable; the gate is_openai_harmony_compatible should "
                "have rejected this tokenizer before construction."
            )
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
        # Codex round-1 BLOCKING: validate recipient against the
        # canonical ``functions.<name>`` shape before interpolating —
        # whitespace, marker-like text, or a name containing
        # ``<|...|>`` literals in the recipient would produce wire
        # text the downstream HarmonyToolParser cannot parse.
        if recipient:
            if not isinstance(recipient, str) or not _RECIPIENT_SHAPE.match(recipient):
                raise ValueError(
                    f"HarmonyStreamingRouter: refusing to reconstruct "
                    f"tool-call wire text for malformed recipient "
                    f"{recipient!r}; expected 'functions.<name>'"
                )
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
        """End-of-stream flush.

        Matches vLLM / SGLang safer-default: only flush the parser
        state via ``process_eos`` so its internal buffers are released.
        Do NOT synthesize a tool call from a truncated commentary
        message — a ``max_tokens`` cutoff mid-body must not be executed
        as if the model had emitted ``<|call|>`` — and do NOT re-emit
        any post-EOS ``last_content_delta``; per-token ``feed()`` has
        already streamed every body byte the model produced.
        """
        try:
            self._parser.process_eos()
        except Exception as e:  # noqa: BLE001
            logger.debug("StreamableParser.process_eos failed: %s", e)
        return None

    def feed_sequence(self, token_ids: list[int]) -> dict[str, Any]:
        """Batch path: route a complete token sequence and return
        the same separated-channels dict as ``OutputRouter.feed_sequence``.

        Returns:
            ``{"content": str|None, "reasoning": str|None,
               "tool_calls": list[str]|None}``

        Codex round-2 NIT: accumulate per-token deltas in lists and
        ``"".join`` once at return — Python string ``+=`` is O(n²) for
        long non-stream generations and a 4k-token reply would spend
        most of its time copying the accumulator.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[str] = []

        def _accumulate(event: RouterEvent | None) -> None:
            if event is None:
                return
            if event.channel == Channel.CONTENT:
                content_parts.append(event.text)
            elif event.channel == Channel.REASONING:
                reasoning_parts.append(event.text)
            elif event.channel == Channel.TOOL_CALL:
                tool_calls.append(event.text)

        for tid in token_ids:
            _accumulate(self.feed(tid))
        # Drain end-of-stream (truncated messages, etc.).
        _accumulate(self.finalize())

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        # Codex round-1 BLOCKING: do NOT ``.strip()`` accumulated
        # content / reasoning — harmony bodies can legitimately begin
        # or end with whitespace / newlines (e.g. markdown blocks,
        # code fences) and stripping silently mutates them. Only
        # convert the exact empty string to ``None`` so non-emitting
        # channels still surface as missing.
        return {
            "content": content if content != "" else None,
            "reasoning": reasoning if reasoning != "" else None,
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


# Codex round-3 NIT: cache by tokenizer identity so the compatibility
# probe + harmony encoding load only happen once per model identity,
# not once per request. The streaming factory runs per
# ``/v1/chat/completions`` and the marker / probe checks call into
# ``enc.encode`` six times — measurable when serving high QPS.
#
# Codex round-4 NIT: the key must include the marker-ID tuple too —
# two tokenizer instances with the same ``name_or_path`` but distinct
# marker IDs (e.g. mock tokenizers in tests, or a model loaded with
# a custom vocab override) would otherwise share a stale entry. The
# marker IDs uniquely identify a (model, harmony-format) pair.
_COMPAT_RESULT_CACHE: dict[tuple, bool] = {}
_HARMONY_ENCODING_CACHE: dict[str, Any] = {}


def _get_harmony_encoding() -> Any | None:
    """Load (and cache) the ``HARMONY_GPT_OSS`` encoding.

    The harmony encoding load is relatively expensive on first call
    (loads tiktoken-style merges); cache the instance so every
    compatibility probe and ``HarmonyStreamingRouter.__init__`` reuses
    the same object instead of re-loading.
    """
    cached = _HARMONY_ENCODING_CACHE.get("gpt_oss")
    if cached is not None:
        return cached
    try:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    except Exception:  # noqa: BLE001
        return None
    _HARMONY_ENCODING_CACHE["gpt_oss"] = enc
    return enc


def is_openai_harmony_compatible(token_map: TokenMap, tokenizer: Any) -> bool:
    """Return True iff the model's vocabulary matches the
    ``openai-harmony`` encoding's vocabulary AND the optional dep is
    present.

    Why the gate exists: ``StreamableParser`` consumes integer token
    IDs from its own encoding's vocabulary. Production upstream
    ``mlx-community/gpt-oss-20b-MXFP4-Q8`` (and the gpt-oss family in
    general) share the harmony encoding's vocabulary for BOTH special
    markers AND body tokens — so we can forward model token IDs
    directly without re-encoding. A model whose markers happen to
    match but whose body-token IDs differ would feed
    ``StreamableParser`` IDs that decode through the wrong vocabulary,
    corrupting streamed content and tool-call arguments (e.g.
    synthetic test vocabs in ``tests/test_engine_router_non_stream.py``
    where ``"Reason"=1`` and ``"ing"=2`` decode to garbage under
    harmony's encoding).

    Three layers of defense:

    1. Tokenizer-identity allowlist (codex round-2 BLOCKING). A
       tokenizer whose ``name_or_path`` does not name a known gpt-oss
       family member is rejected even if markers and probes pass —
       this guards against the case where matching markers and three
       probe strings coincide but uncommon body tokens are remapped.
    2. Marker-ID parity. Each known harmony marker the ``TokenMap``
       recorded must encode to the same ID under the harmony encoding.
    3. Body-vocab probe set (codex round-1 BLOCKING). A representative
       set of plain / JSON / multi-token strings must round-trip
       through both encoders to identical IDs.

    All three must pass. Result is cached per tokenizer identity
    (codex round-3 NIT) so the probe runs at most once per model.
    """
    if not is_openai_harmony_available():
        return False

    # Cache lookup (codex round-3 NIT). Codex round-4 NIT: include the
    # marker-ID tuple in the key — two tokenizer instances with the
    # same ``name_or_path`` but different marker IDs (e.g. test mocks,
    # or a model loaded with a vocab override) must NOT share a
    # cached compatibility decision.
    name_or_path = getattr(tokenizer, "name_or_path", "") or ""
    marker_ids = (
        token_map.harmony_channel,
        token_map.harmony_message,
        token_map.harmony_call,
        token_map.harmony_end,
        token_map.harmony_return,
        token_map.harmony_start,
        token_map.harmony_constrain,
    )
    cache_key = (str(name_or_path).lower(), marker_ids)
    cached = _COMPAT_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    enc = _get_harmony_encoding()
    if enc is None:
        _COMPAT_RESULT_CACHE[cache_key] = False
        return False

    result = _compute_compat(token_map, tokenizer, enc, name_or_path)
    _COMPAT_RESULT_CACHE[cache_key] = result
    return result


def _compute_compat(
    token_map: TokenMap, tokenizer: Any, enc: Any, name_or_path: str
) -> bool:
    """Helper that runs the three-layer compatibility check. Split
    out from ``is_openai_harmony_compatible`` so the cache-write logic
    lives in one place around a single return value (codex round-3 NIT).
    """
    # (1) Tokenizer-identity allowlist. ``name_or_path`` is set by HF /
    # mlx-lm tokenizers to the model id passed at load time. We do a
    # case-insensitive substring match so quant-suffix renames
    # (``-MXFP4-Q8``) and org prefixes (``mlx-community/`` vs
    # ``openai/``) all resolve to the same family. Missing or empty
    # name_or_path → can't prove identity → reject.
    name_lc = str(name_or_path).lower()
    if not any(known in name_lc for known in _KNOWN_HARMONY_TOKENIZERS):
        return False

    # (2) Marker-ID parity. Codex round-4 BLOCKING: ALL seven harmony
    # markers must be present in the tokenizer's vocab AND match the
    # harmony encoding's IDs — a tokenizer with only ``<|channel|>``
    # and ``<|message|>`` (but missing ``<|call|>`` / ``<|end|>`` etc.)
    # could otherwise be upgraded and then crash inside
    # ``StreamableParser`` when the model emits a marker the parser
    # expects but the gate didn't verify. Requiring all seven is the
    # documented invariant of the harmony encoding; any production
    # gpt-oss tokenizer has them all.
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
            return False
        try:
            harmony_ids = enc.encode(marker, allowed_special="all")
        except Exception:  # noqa: BLE001
            return False
        if harmony_ids != [model_id]:
            return False

    # (3) Body-vocab probe set. A tokenizer that lacks ``.encode`` is
    # likewise rejected (e.g. ``_FakeTokenizer`` in the legacy harmony
    # test suite — only exposes ``decode`` + ``get_vocab``).
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return False
    for probe in _BODY_VOCAB_PROBES:
        try:
            harmony_ids = enc.encode(probe, allowed_special="none")
        except Exception:  # noqa: BLE001
            return False
        try:
            # ``add_special_tokens=False`` keeps the probe pure-body —
            # HF tokenizers wrap with BOS/EOS otherwise. Pass it as a
            # kwarg the tokenizer may or may not accept; some
            # tokenizers raise TypeError on unknown kwargs, in which
            # case we fall back to a positional call. Either way, a
            # raise means we can't prove compatibility → return False.
            try:
                model_ids = encode(probe, add_special_tokens=False)
            except TypeError:
                model_ids = encode(probe)
        except Exception:  # noqa: BLE001
            return False
        # ``encode`` returns ``list[int]`` for HF / mlx_lm tokenizers;
        # convert to list for comparison robustness.
        if list(model_ids) != list(harmony_ids):
            return False
    return True
