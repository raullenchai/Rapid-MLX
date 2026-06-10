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

Codex round-12/14 BLOCKING (PR #515): earlier revisions reconstructed
the parsed ``Message`` back into harmony wire text
(``<|channel|>commentary to=functions.X<|message|>{body}<|call|>``)
so the downstream text-based ``HarmonyToolParser`` could re-parse it.
That round-trip was lossy — any tool-call body containing a literal
harmony sentinel substring (e.g. ``{"text":"<|call|>"}``) corrupted
the downstream regex parse and silently dropped the call. The current
design eliminates the round-trip: TOOL_CALL events surface a
structured ``{"name", "arguments"}`` payload on
``RouterEvent.tool_call``, the engine plumbs it through
``GenerationOutput.tool_calls``, and the route layer bypasses the
text-based parser entirely when the structured field is populated.
This matches vLLM and SGLang exactly.

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
import weakref
from typing import Any

from .output_router import Channel, RouterEvent, TokenMap

logger = logging.getLogger(__name__)

# Channel name strings emitted by openai-harmony StreamableParser.
_HARMONY_CHANNEL_ANALYSIS = "analysis"
_HARMONY_CHANNEL_FINAL = "final"
_HARMONY_CHANNEL_COMMENTARY = "commentary"

# Tool-call recipient shape: ``functions.<safe-name>``. OpenAI's
# documented function-name regex is ``[a-zA-Z0-9_-]{1,64}`` — match
# that exactly so any tool the upstream API accepts also round-trips
# through the router. A recipient that doesn't match (whitespace,
# marker-like characters, newlines, leading digit beyond the allowed
# set) is treated as structural corruption: the router abstains rather
# than surfacing a half-parsed call.
_RECIPIENT_SHAPE = re.compile(r"^functions\.[A-Za-z0-9_\-]{1,64}$")

# HuggingFace hub cache snapshot dir pattern. Path components have the
# form ``models--<owner>--<name>`` so ``/.../models--openai--gpt-oss-20b
# /snapshots/<sha>/`` resolves to the identity ``openai/gpt-oss-20b-mxfp4-q8``
# (the basename is the snapshot SHA, which on its own gives no hint
# that this is a gpt-oss tokenizer). Codex round-14 BLOCKING — the
# previous basename-only check rejected this path shape and the gate
# fell back to the leaking legacy router for any model loaded straight
# from the HF cache. Pattern is tolerant of either dir separator since
# the path may be normalised before reaching us.
_HF_CACHE_MODEL_DIR_RE = re.compile(r"models--([^-/\\]+(?:-[^-/\\]+)*)--([^/\\]+)")

# Tokenizer-identity allowlist — known-compatible HF / mlx-community
# names whose vocab is the harmony encoding. Codex round-2 BLOCKING:
# matching a 3-string probe set against the harmony encoding is not
# enough to prove full-vocab parity — a tokenizer with the right
# markers and the right probes but a remapped uncommon token could
# silently corrupt later content. The cleanest defense is to ALSO
# require the tokenizer's reported identity to be a known gpt-oss
# family member.
#
# Codex round-11 BLOCKING / round-12 BLOCKING / round-13 BLOCKING /
# round-14 BLOCKING — the tokenizer-identity gate has had four rounds
# of tightening:
#   * round-11: naive ``known in name_lc`` substring → anchored basename
#     regex (rejected tail-substring fakes).
#   * round-12: anchored basename still let arbitrary owners through
#     (``some-user/gpt-oss-remapped``) → restrict to known owners.
#   * round-13: pure remote-id-prefix matching rejected legitimate
#     LOCAL paths (``/models/gpt-oss-20b-mxfp4-q8``, ``~/.cache/.../gpt-oss-20b-mxfp4-q8``)
#     and made production fall back to the leaking legacy router.
#   * round-14: HF cache snapshot dir ``models--openai--gpt-oss-20b
#     /snapshots/<sha>`` has SHA basename → recognise the ``models--
#     owner--name`` segment anywhere in the path and treat it as a
#     remote id for the owner check.
# Final shape (three-tier):
#   * Local filesystem paths (``/``, ``~``, ``./``, ``../`` prefixes)
#     that DO NOT carry an HF cache marker trust the basename — the
#     user explicitly loaded that artifact, and the body-probe set is
#     the authoritative vocab check.
#   * Paths that contain ``models--<owner>--<name>`` (HF hub cache
#     layout) are treated as the remote id ``<owner>/<name>`` and run
#     through the remote-owner allowlist below.
#   * Remote HF IDs (``owner/name`` shape with no leading separator)
#     require an allowlisted owner; this defense-in-depth catches the
#     ``some-user/gpt-oss-remapped`` shape codex flagged.
# The basename regex itself is the same anchored
# ``^gpt-oss(?:-|$)`` shape from round-11.
_BASENAME_GPT_OSS_RE = re.compile(r"^gpt-oss(?:-|$)")
_KNOWN_HARMONY_REMOTE_OWNERS = frozenset(
    {
        "openai",
        "mlx-community",
        "unsloth",
    }
)


def _is_known_harmony_identity(name_or_path: str) -> bool:
    """Three-tier allowlist: HF cache snapshot paths resolve to their
    ``owner/name`` segment, plain local paths trust the basename, and
    remote HF IDs require an allowlisted owner. Returns True iff the
    identity names a known gpt-oss family member.
    """
    name_lc = name_or_path.lower().rstrip("/")
    if not name_lc:
        return False

    # HF cache snapshot layout. ``~/.cache/huggingface/hub/models--<owner>
    # --<name>/snapshots/<sha>/`` — the basename is opaque, so search the
    # full path for the canonical ``models--<owner>--<name>`` directory
    # segment and use the embedded owner/name pair. Tolerant of either
    # ``/`` or ``\`` separators so HF hub paths on either OS work.
    cache_match = _HF_CACHE_MODEL_DIR_RE.search(name_lc)
    if cache_match is not None:
        owner, name = cache_match.group(1), cache_match.group(2)
        if not _BASENAME_GPT_OSS_RE.match(name):
            return False
        return owner in _KNOWN_HARMONY_REMOTE_OWNERS

    basename = name_lc.rsplit("/", 1)[-1]
    if not _BASENAME_GPT_OSS_RE.match(basename):
        return False
    # Bare basename (no path separator at all) — accept.
    if "/" not in name_lc:
        return True
    # Local filesystem path — accept (user-controlled artifact).
    if (
        name_lc.startswith("/")
        or name_lc.startswith("~")
        or name_lc.startswith("./")
        or name_lc.startswith("../")
    ):
        return True
    # Remote HF ID (``owner/name`` shape) — owner must be allowlisted.
    owner = name_lc.split("/", 1)[0]
    return owner in _KNOWN_HARMONY_REMOTE_OWNERS


# Probe strings used by ``is_openai_harmony_compatible`` to verify
# that the model's body-token vocabulary matches the openai-harmony
# encoding's vocabulary. If any probe round-trips to a different ID
# list, the gate falls back to the legacy router — feeding mismatched
# IDs to ``StreamableParser`` decodes bodies through the wrong vocab
# and corrupts content / tool-call arguments (codex round-1 BLOCKING).
# Pick short strings that exercise common body-vocab regions: plain
# English, JSON-shaped text, and the smoking-gun multi-token word
# ``commentary`` from PR #514 (``comment``+``ary`` on gpt-oss-20b-mxfp4-q8).
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

    TOOL_CALL events carry the parsed ``{"name", "arguments"}``
    payload on ``RouterEvent.tool_call`` (set on closed commentary
    messages with a valid ``functions.<name>`` recipient). The engine
    plumbs this through ``GenerationOutput.tool_calls`` and routes
    consume it directly — bypassing the legacy wire-text → regex
    round-trip that lost calls whose JSON arguments contained literal
    harmony sentinel substrings (PR #515 codex round-12/14 BLOCKING).
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
    def _extract_structured_tool_call(message: Any) -> dict | None:
        """Pull a structured ``{"name", "arguments"}`` payload out of a
        closed harmony commentary ``Message``.

        Returns ``None`` if the recipient shape is malformed — the
        upstream pipeline treats that as "no tool call surfaced here"
        and the corrupt frame is dropped. This is a strictly stronger
        gate than the legacy wire-text path which would have produced
        a garbled call.

        ``arguments`` is the verbatim concatenated body bytes — no
        normalisation, no sentinel scrubbing, no JSON re-encoding.
        That bytes-faithful surface is the entire point of the
        round-trip elimination: a tool call whose JSON arguments
        contain a literal ``<|call|>`` substring (the documented PR
        #515 codex round-12/14 BLOCKING scenario) is now preserved
        intact instead of being corrupted by sentinel-anchored regex
        parsing.
        """
        recipient = getattr(message, "recipient", None)
        if not recipient:
            return None
        if not isinstance(recipient, str) or not _RECIPIENT_SHAPE.match(recipient):
            logger.debug(
                "HarmonyStreamingRouter: dropping tool call with "
                "malformed recipient %r (expected functions.<name>)",
                recipient,
            )
            return None
        # ``functions.get_weather`` → ``get_weather``. The OpenAI
        # tool-call schema's ``function.name`` field carries the bare
        # name; the ``functions.`` namespace prefix is harmony-specific
        # transport metadata.
        name = recipient.split(".", 1)[1]
        # Codex round-13 NIT: list-build + join (avoid quadratic ``+=``).
        body_parts: list[str] = []
        for c in message.content:
            t = getattr(c, "text", None)
            if t:
                body_parts.append(t)
        arguments = "".join(body_parts)
        return {"name": name, "arguments": arguments}

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
            event with the structured ``{"name", "arguments"}``
            payload on ``RouterEvent.tool_call``.
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
                structured = self._extract_structured_tool_call(closed)
                if structured is not None:
                    return RouterEvent(
                        channel=Channel.TOOL_CALL,
                        token_id=token_id,
                        # ``text`` carries the JSON args as a human-
                        # readable summary for legacy consumers that
                        # may peek at the field. The structured payload
                        # below is the authoritative source for
                        # downstream processing.
                        text=structured["arguments"],
                        tool_call=structured,
                    )

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
        the separated-channels dict.

        Returns:
            ``{"content": str|None, "reasoning": str|None,
               "tool_calls": list[dict]|None}``

        ``tool_calls`` entries are structured ``{"name", "arguments"}``
        dicts — the engine plumbs them through
        ``GenerationOutput.tool_calls`` and routes bypass text-based
        extraction when this field is populated. The shape matches
        ``HarmonyToolParser.extract_tool_calls`` so downstream
        consumers can treat both sources uniformly.

        Codex round-2 NIT: accumulate per-token deltas in lists and
        ``"".join`` once at return — Python string ``+=`` is O(n²) for
        long non-stream generations and a 4k-token reply would spend
        most of its time copying the accumulator.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict] = []

        def _accumulate(event: RouterEvent | None) -> None:
            if event is None:
                return
            if event.channel == Channel.CONTENT:
                content_parts.append(event.text)
            elif event.channel == Channel.REASONING:
                reasoning_parts.append(event.text)
            elif event.channel == Channel.TOOL_CALL and event.tool_call is not None:
                tool_calls.append(event.tool_call)

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
#
# Codex round-12 BLOCKING / NIT: the previous key
# ``(name_or_path_lc, marker_ids, id(tokenizer))`` was vulnerable to
# Python's ``id()`` reuse after garbage collection — a tokenizer freed
# between requests could have its memory address reused by a brand-new
# (potentially incompatible) tokenizer, which then hit the stale True
# entry and was upgraded incorrectly. Switch to a
# ``WeakKeyDictionary`` keyed on the tokenizer object itself; when the
# tokenizer is garbage-collected the entry is automatically reaped,
# eliminating the id-reuse hazard and bounding cache growth across
# model reloads. The inner per-tokenizer dict is keyed on
# ``(name_or_path_lc, marker_ids)`` so distinct ``TokenMap`` overrides
# on a single tokenizer still segregate.
#
# Fallback: a few tokenizer mocks (and any tokenizer whose class
# disables ``__weakref__``) are not weak-referenceable. For those we
# fall back to recomputing on every call — cheap, no correctness risk.
_COMPAT_RESULT_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple, bool]] = (
    weakref.WeakKeyDictionary()
)
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
    # Codex round-8 NIT: skip the redundant ``is_openai_harmony_available``
    # call — ``_get_harmony_encoding`` is the single dependency probe
    # and returns ``None`` if the import fails. The pre-cache return
    # path below collapses cleanly to ``False`` in that case.

    # Cache lookup (codex round-3 NIT / round-4 NIT / round-11 BLOCKING
    # / round-12 BLOCKING+NIT). The cache is now a
    # ``WeakKeyDictionary[tokenizer] -> dict[(name_lc, marker_ids), bool]``
    # — distinct tokenizer instances naturally segregate, and entries
    # auto-clear when the tokenizer is garbage-collected (no ``id()``
    # reuse hazard, bounded growth across model reloads).
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
    inner_key = (str(name_or_path).lower(), marker_ids)
    try:
        tk_cache = _COMPAT_RESULT_CACHE.get(tokenizer)
    except TypeError:
        # Tokenizer class is not weak-referenceable (e.g. some test
        # mocks). Skip caching for this call; correctness is preserved
        # at the cost of a recompute per request.
        tk_cache = None
        cacheable = False
    else:
        cacheable = True
    if tk_cache is not None:
        cached = tk_cache.get(inner_key)
        if cached is not None:
            return cached

    enc = _get_harmony_encoding()
    if enc is None:
        result = False
    else:
        result = _compute_compat(token_map, tokenizer, enc, name_or_path)

    if cacheable:
        try:
            slot = _COMPAT_RESULT_CACHE.setdefault(tokenizer, {})
        except TypeError:
            pass
        else:
            slot[inner_key] = result
    return result


def _compute_compat(
    token_map: TokenMap, tokenizer: Any, enc: Any, name_or_path: str
) -> bool:
    """Helper that runs the three-layer compatibility check. Split
    out from ``is_openai_harmony_compatible`` so the cache-write logic
    lives in one place around a single return value (codex round-3 NIT).
    """
    # (1) Tokenizer-identity allowlist (codex round-11 / round-12 /
    # round-13 / round-14). Three-tier: HF cache snapshot dirs resolve
    # via ``models--<owner>--<name>`` segment, plain local paths trust
    # the basename, remote HF IDs require an allowlisted owner. See
    # ``_is_known_harmony_identity`` docstring for the full algorithm.
    if not _is_known_harmony_identity(str(name_or_path)):
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
        # Codex round-8 BLOCKING: ``encode`` is expected to return a
        # flat list[int] for HF / mlx_lm tokenizers, but some HF
        # tokenizer configurations (e.g. ``return_tensors`` defaults
        # or wrapped Fast tokenizers) yield ``BatchEncoding`` /
        # tensor-like objects whose ``list(...)`` is either a column
        # vector or raises. Defensively coerce to a flat int sequence
        # and fall back to False on anything we cannot interpret.
        try:
            model_ids_list = list(model_ids)
        except TypeError:
            return False
        if not all(isinstance(x, int) for x in model_ids_list):
            return False
        if model_ids_list != list(harmony_ids):
            return False
    return True
