# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for text processing and model detection.
"""

import json
import logging
import re

from ..model_aliases import resolve_profile
from ..model_metadata import (
    checkpoint_has_multimodal_weights,
    config_indicates_multimodal,
    read_cached_model_metadata,
    read_local_model_metadata,
    read_model_metadata,
)
from .models import Message

logger = logging.getLogger(__name__)

# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
# Keeps <think>...</think> blocks intact for reasoning models
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|eom_id\|>|<\|python_tag\|>|"
    r"<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|channel\|>|<\|message\|>|<\|start\|>|<\|return\|>|<\|call\|>|<\|constrain\|>|"
    r"<\|turn>|<turn\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]|"
    r"\[e~\[|\]~b\][a-z]*|\]~!b\["
)

# Fast-path characters that MUST be present for any special token to match.
# If none of these appear in the text, regex can be skipped entirely.
_SPECIAL_TOKEN_CHARS = frozenset("<[]")

# Muse Glimmer wire detection for ``clean_output_text``. The output is
# muse-shaped when a recipient header accompanies ``<|message|>``:
# either the implicit first-segment form (`` to=recipient<|message|>``
# at the start) or an explicit ``<|start|>assistant`` header. A bare
# leading ``<|message|>`` (implicit user segment) also counts. Ordinary
# prose that merely MENTIONS ``<|message|>`` mid-text matches none of
# these anchored shapes. The explicit form requires the FULL header
# through ``<|message|>`` — a bare ``<|start|>assistant`` substring in
# prose must not trigger the branch (codex r5 #2); prose spelling a
# complete header is indistinguishable from wire by construction, the
# same residual the harmony branch carries.
_MUSE_WIRE_PROBE = re.compile(
    r"^\s?(?:to=\S+\s*)?<\|message\|>"
    r"|<\|start\|>assistant(?:\s+to=\S+)?\s*<\|message\|>"
)


def strip_special_tokens(text: str) -> str:
    """Remove special tokens from text with a fast-path bypass.

    Most per-token deltas are plain text without special token markers.
    Checking for marker characters first avoids regex overhead on ~99% of tokens.
    """
    # Fast path: no marker characters → no special tokens possible
    for ch in text:
        if ch in _SPECIAL_TOKEN_CHARS:
            return SPECIAL_TOKENS_PATTERN.sub("", text)
    return text


# =============================================================================
# Final sanitizer — last-mile catch-all before content reaches the client.
# Catches ANY remaining markup that earlier layers missed, including:
# - All <|..> and <..|> asymmetric tokens (Gemma 4 style)
# - All <|..|> symmetric tokens (Qwen, GPT-OSS style)
# - [Calling tool:...] text-format tool calls
# - Stray </think>, </tool_call>, etc.
#
# IMPORTANT — what is NOT stripped here:
# The bare ``<think>`` OPENER is intentionally NOT in this regex. The
# streaming postprocessor legitimately prepends ``<think>`` to the first
# content chunk on Nemotron-family models, and the standard / chat path
# routes that chunk through ``sanitize_output`` — stripping the opener
# here would erase the prefix injection (broke
# ``TestStreamingPostProcessorNemotron::test_thinking_prefix_injected``
# during the R12-M1b first pass). Models that legitimately mention the
# literal ``<think>`` tag in prose (e.g. "use the <think> tag in HTML")
# must also pass through unchanged.
#
# Reasoning-channel leaks of the opener (Mira r12 R-3 bonus regression:
# ``reasoning_text="<think>"`` at ``max_tokens=1`` because the prompt
# template's pre-injected opener was the only token the parser saw) are
# handled at the REASONING-channel layer instead — see
# ``api.anthropic_adapter._thinking_block_content`` and
# ``service.helpers._build_reasoning_rescue_payload``, both of which
# call the dedicated ``strip_reasoning_channel_markup`` helper below.
# =============================================================================

_FINAL_SANITIZER_ALTERNATIVES = (
    # Full Gemma 4 tool call (greedy body): <|tool_call>call:name{...}<tool_call|>
    # MUST be listed BEFORE the bare-token strippers, otherwise the inner
    # `call:name{...}` body would be left orphaned in content.
    r"<\|tool_call>.*?<tool_call\|>"
    # Any <|...> or <...|> token (Gemma 4 asymmetric: <|channel>, <tool_call|>, etc.)
    r"|<\|[A-Za-z0-9_\"]+>|<[A-Za-z0-9_\"]+\|>"
    # Any <|...|> token (symmetric: <|im_end|>, <|channel|>, etc.)
    # The alphabet is intentionally case-complete: Cohere North uses
    # uppercase sentinels such as ``<|END_THINKING|>`` and declares them
    # as ordinary added tokens (``special=false``), so tokenizer-level
    # ``skip_special_tokens=True`` does not remove them.
    r"|<\|[A-Za-z0-9_]+\|>"
    # [Calling tool:...] or [Calling tool="..."] or bare "[Calling tool" (Gemma 4 mimicry)
    r"|\[Calling\s+tool[^\]]*\]?"
    # Stray closing tags
    #
    # Codex R4 [P2] on R12-FIX-V2 was considered and rejected: adding
    # plain ``<tool_call>`` opener stripping to this global sanitizer
    # breaks the existing T1/T2/T3 ``tool_choice="required"`` test
    # suite (``test_tool_choice_enforcement.py`` r7/r8/r9 BLOCKING
    # codex rounds), which intentionally pin that legitimate prose
    # mentioning ``<tool_call>`` as text MUST survive. The
    # route-level ``_scrub_visible_tool_wire_leaks`` already
    # discriminates structural wire vs literal-token-mention via
    # ``_contains_structural_tool_wire_leak`` and runs on
    # ``reasoning_text`` for the forced/required path
    # (``routes/chat.py:~3245``). Defense-in-depth at the global
    # sanitizer would over-strip; the existing layered gate is the
    # correct architecture.
    #
    # The CLOSER ``</tool_call>`` used to be stripped here — that was
    # the same over-strip the paragraph above rejects for the opener,
    # and the argument applies verbatim to the closer. It deleted the
    # token out of ordinary assistant prose on EVERY surface
    # (``/v1/chat/completions``, ``/v1/messages``, ``/v1/responses``),
    # so a coding agent asked to write a file documenting the tool
    # protocol silently got a file with the tag missing. Reproduced
    # end-to-end against real Claude Code and real Codex: both wrote
    # ``"A tool call block ends with  on its own."`` — marker excised,
    # double space left behind — and both then burned turns trying to
    # work around their own apparently-broken output. Structural
    # residue is the layered gate's job (it is payload-aware); prose is
    # not residue.
    r"|</think>"
)

#: CONTENT channel. ``</tool_call>`` is deliberately absent — see the
#: block above.
_FINAL_SANITIZER = re.compile(_FINAL_SANITIZER_ALTERNATIVES, re.DOTALL)

#: REASONING channel. Same set PLUS the ``</tool_call>`` closer.
#:
#: The channel split mirrors the one already drawn for ``<think>``
#: (``_REASONING_CHANNEL_TAG_RE`` below): inside a reasoning trace a
#: bare wire marker is a structural parser artifact, never text the
#: user asked the model to produce, so the aggressive strip is correct
#: there. In ``content`` the same token can be exactly what was asked
#: for — a coding agent writing documentation about the tool protocol.
_REASONING_FINAL_SANITIZER = re.compile(
    _FINAL_SANITIZER_ALTERNATIVES + r"|</tool_call>", re.DOTALL
)


#: Reasoning-channel sanitizer — strips ``<think>`` opener + closer
#: BOTH. Distinct from ``_FINAL_SANITIZER`` (which leaves the opener
#: alone so legit Nemotron prefix injection and literal-tag prose
#: survive). Current consumers (R12-M1b):
#:
#: * the Anthropic ``thinking`` content block (via
#:   ``_thinking_block_content`` in ``api.anthropic_adapter``)
#: * the rescue-tail copy of the reasoning trace that surfaces in
#:   ``content`` (via ``_build_reasoning_rescue_payload`` in
#:   ``service.helpers``)
#:
#: OpenAI ``message.reasoning_content`` and ``/v1/responses`` reasoning
#: items intentionally DO NOT route through this regex in this PR; if
#: the same ``<think>`` opener leakage is observed on those surfaces,
#: wire the helper through ``strip_reasoning_channel_markup`` at the
#: matching emit site rather than expanding the regex itself. In every
#: reasoning-channel context the ``<think>`` opener is a structural
#: parser artifact, never legit user-visible text — so the channel-
#: aware strip is always safe where it is wired in.
_REASONING_CHANNEL_TAG_RE = re.compile(r"</?think>")


def strip_reasoning_channel_markup(text: str) -> str:
    """Strip ``<think>`` / ``</think>`` tags that the reasoning parser
    may have left in the reasoning channel.

    Current call sites (R12-M1b):

    * ``api.anthropic_adapter._thinking_block_content`` — sanitizes
      bytes destined for the Anthropic ``thinking`` content block.
    * ``service.helpers._build_reasoning_rescue_payload`` — sanitizes
      the rescue tail that surfaces a slice of the reasoning trace
      into the user-visible ``content`` channel.

    The OpenAI ``reasoning_content`` field and the ``/v1/responses``
    reasoning item DO NOT currently route through this helper — they
    surface the raw parser output. Wire them through here too if a
    future report shows the same ``<think>`` opener leakage on those
    surfaces.

    Why this isn't in ``sanitize_output``: the canonical sanitizer is
    applied to ``content``-channel bytes too, where the bare ``<think>``
    opener is sometimes legitimate (Nemotron prefix injection, literal-
    tag prose). Splitting the strip rule by channel keeps both
    invariants intact:

    * ``content`` channel — opener passes through, closer is stripped
      (existing pre-R12-M1b behaviour, no regression risk).
    * ``reasoning_content`` / ``thinking`` channel — both opener and
      closer are stripped, because the channel itself MEANS "this is
      the model's thought trace" so wrapping tags are redundant /
      structural noise.

    Empty / None input returns the input unchanged.
    """
    if not text:
        return text
    return _REASONING_CHANNEL_TAG_RE.sub("", text)


def sanitize_output(text: str) -> str:
    """Final catch-all sanitizer for client-facing content.

    This is the LAST defense against markup leakage. Runs after all
    parsers and filters. Strips anything that looks like a special token
    or internal markup pattern.

    Designed to be aggressive — better to over-strip than to leak.
    """
    return _sanitize_with(text, _FINAL_SANITIZER)


def _sanitize_with(text: str, pattern: re.Pattern[str]) -> str:
    """Shared body for the content / reasoning sanitizers.

    Only the alternation differs between the two channels; the
    fast-path bypass and the collapse-to-None semantics are identical.
    """
    if not text:
        return text
    for ch in text:
        if ch in _SPECIAL_TOKEN_CHARS:
            cleaned = pattern.sub("", text).strip()
            return cleaned or None  # collapse empty to None
    return text


def sanitize_reasoning_content(text: str | None) -> str | None:
    """Sanitize ``reasoning_content`` so chat-template special tokens never
    reach the wire.

    Vlad r12 dogfood (0.8.15) MED-2: ``<|im_start|>`` leaked verbatim into
    ``message.reasoning_content`` on the ``tool_choice="required"`` branch
    for ``qwen3-0.6b-4bit``. The non-stream chat route ran ``sanitize_output``
    on the visible ``content`` only — the ``reasoning_content`` companion
    field was passed through to ``AssistantMessage`` untouched. Streaming
    deltas had the same gap. The systematic fix: **every** user-visible
    string field that originated from a raw token decode (``content``,
    ``reasoning_content``, Anthropic ``thinking`` blocks, Responses
    ``output_text``) must flow through the same final sanitizer.

    Mirrors ``sanitize_output`` semantics:

    - Empty / ``None`` input → returned as-is (no rewrite cost on the
      hot path; reasoning_content is frequently absent).
    - Plain text with no special-token marker chars → returned unchanged
      (fast-path bypass via the ``_SPECIAL_TOKEN_CHARS`` membership
      check).
    - Text containing markup → stripped via the same ``_FINAL_SANITIZER``
      regex; collapses to ``None`` if the entire string was markup
      (so callers writing the field through pydantic + ``exclude_none``
      drop it cleanly rather than emitting an empty string).

    Use ``sanitize_reasoning_for_stream`` (defined below) when the call
    site can't tolerate a ``None`` return (per-delta streaming where a
    None would change the field's type contract).

    Uses ``_REASONING_FINAL_SANITIZER``, which differs from the content
    channel by ALSO stripping the ``</tool_call>`` closer — see the
    comment on that pattern for why the two channels diverge.
    """
    return _sanitize_with(text, _REASONING_FINAL_SANITIZER)


def sanitize_reasoning_for_stream(text: str | None) -> str:
    """Streaming variant of :func:`sanitize_reasoning_content`.

    Per-delta streaming emits chunks via ``_fast_sse_chunk`` and must
    write a STRING value into the JSON envelope — a ``None`` here would
    serialize as JSON ``null`` and change the field's type contract on
    the wire (clients consume ``delta.reasoning_content`` as a string).

    **Whitespace preservation contract**: streaming clients concatenate
    deltas verbatim, so ``.strip()``-ing an individual delta corrupts
    cross-delta boundaries — e.g. a prior delta ``"foo"`` followed by
    ``" bar <|im_start|>"`` would arrive as ``"foobar"`` instead of
    ``"foo bar"`` if the sanitizer trimmed leading whitespace after
    removing the marker. This variant therefore removes ONLY the marker
    bytes via :data:`_REASONING_FINAL_SANITIZER` (this is the reasoning
    channel, so the ``</tool_call>`` closer is stripped here — see that
    pattern's comment) and leaves all surrounding
    whitespace intact. (The non-stream :func:`sanitize_output` strips
    because it operates on a fully-assembled final string where leading/
    trailing whitespace is cosmetic.)

    Codex R2 [P2] on R12-FIX-V2.

    Returns:
        - ``""`` for ``None`` / empty input so callers can use the
          return as the JSON value directly.
        - The marker-stripped text otherwise (whitespace preserved).
          May still be ``""`` if the input was entirely markup — the
          caller decides whether to suppress the empty delta.
    """
    return _sanitize_for_stream(text, _REASONING_FINAL_SANITIZER)


class StreamingReasoningSanitizer:
    """Preserve canonical wire-marker carry across reasoning fragments.

    ``sanitize_reasoning_for_stream`` is intentionally stateless.  Routes
    that enforce the reasoning sanitizer must additionally retain suffixes
    that can begin canonical wire markers; otherwise a token split can bypass
    the per-fragment regex.  The finite set below is the literal-token portion
    of :data:`_REASONING_FINAL_SANITIZER`; payload-shaped regex matches remain
    handled by the stateless sanitizer once their complete fragment arrives.
    """

    _MARKERS = (
        "</tool_call>",
        "</think>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|channel|>",
        "<|message|>",
        "<|channel>",
        "<|constrain|>",
        "<|tool_call>",
        "<tool_call|>",
    )

    def __init__(self) -> None:
        self._pending: list[tuple[str, str]] = []

    def process(self, text: str | None, destination: str) -> list[tuple[str, str]]:
        items = self._pending + [(destination, ch) for ch in (text or "")]
        self._pending = []
        if not items:
            return []

        deferred_payload: list[tuple[str, str]] = []
        raw_combined = "".join(ch for _, ch in items)
        calling_at = raw_combined.rfind("[Calling")
        calling_candidate = raw_combined[calling_at:] if calling_at >= 0 else ""
        if (
            calling_at >= 0
            and len(calling_candidate) <= 128
            and re.fullmatch(r"\[Calling\s+tool[^\]]*", calling_candidate)
        ):
            deferred_payload = items[calling_at:]
            items = items[:calling_at]
        items = self._remove_regex_matches(items)
        kept: list[tuple[str, str]] = []
        for item in items:
            kept.append(item)
            for marker in self._MARKERS:
                marker_size = len(marker)
                if len(kept) >= marker_size and all(
                    kept[-marker_size + offset][1] == marker_char
                    for offset, marker_char in enumerate(marker)
                ):
                    del kept[-marker_size:]
                    break

        kept_text = "".join(ch for _, ch in kept)
        pending_at = self._pending_suffix_start(kept_text)
        if pending_at >= 0:
            self._pending = kept[pending_at:]
            kept = kept[:pending_at]
        if deferred_payload:
            self._pending.extend(deferred_payload)
        return self._group_and_sanitize(kept)

    @classmethod
    def _pending_suffix_start(cls, text: str) -> int:
        max_carry = min(len(text), 128)
        calling_prefix = "[Calling tool"
        for size in range(max_carry, 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in cls._MARKERS):
                return len(text) - size
            if calling_prefix.startswith(suffix):
                return len(text) - size
            if re.fullmatch(r"\[Calling\s+tool[^\]]*", suffix):
                return len(text) - size
        return -1

    def flush(self) -> list[tuple[str, str]]:
        pending = self._pending
        self._pending = []
        return self._group_and_sanitize(pending)

    def transition_to_content(self, text: str | None) -> list[tuple[str, str]]:
        """Resolve a reasoning-side prefix without sanitizing content prose."""
        content = text or ""
        if not self._pending:
            return [("content", content)] if content else []
        pending_text = "".join(ch for _, ch in self._pending)
        if pending_text.startswith("[Calling"):
            pending = self._pending
            self._pending = []
            parts = self._group_and_sanitize(pending)
            if content:
                parts.append(("content", content))
            return parts
        items = self._pending + [("content", ch) for ch in content]
        self._pending = []
        items = self._remove_regex_matches(
            items,
            require_reasoning_origin=True,
        )
        combined = "".join(ch for _, ch in items)
        pending_at = self._pending_suffix_start(combined)
        if pending_at >= 0 and any(
            destination != "content" for destination, _ in items[pending_at:]
        ):
            self._pending = items[pending_at:]
            items = items[:pending_at]
        return self._group_and_sanitize(items, sanitize_content=False)

    @staticmethod
    def _remove_regex_matches(
        items: list[tuple[str, str]],
        *,
        require_reasoning_origin: bool = False,
    ) -> list[tuple[str, str]]:
        while items:
            combined = "".join(ch for _, ch in items)
            spans = []
            for match in _REASONING_FINAL_SANITIZER.finditer(combined):
                if require_reasoning_origin and not any(
                    destination != "content"
                    for destination, _ in items[match.start() : match.end()]
                ):
                    continue
                spans.append((match.start(), match.end()))
            if not spans:
                return items
            removed = [False] * len(items)
            for start, end in spans:
                removed[start:end] = [True] * (end - start)
            items = [item for index, item in enumerate(items) if not removed[index]]
        return items

    @staticmethod
    def _group_and_sanitize(
        items: list[tuple[str, str]], *, sanitize_content: bool = True
    ) -> list[tuple[str, str]]:
        grouped: list[tuple[str, str]] = []
        current_destination: str | None = None
        current_chars: list[str] = []
        for destination, char in items:
            if current_destination is not None and destination != current_destination:
                grouped.append((current_destination, "".join(current_chars)))
                current_chars = []
            current_destination = destination
            current_chars.append(char)
        if current_destination is not None:
            grouped.append((current_destination, "".join(current_chars)))
        result = []
        for destination, text in grouped:
            cleaned = (
                sanitize_reasoning_for_stream(text)
                if sanitize_content or destination != "content"
                else text
            )
            if cleaned:
                result.append((destination, cleaned))
        return result


def sanitize_content_for_stream(text: str | None) -> str:
    """Streaming variant of :func:`sanitize_output` for ``content``.

    Same whitespace-preservation contract as
    :func:`sanitize_reasoning_for_stream`, but on the CONTENT channel —
    so the ``</tool_call>`` closer is NOT stripped. Streaming is the
    path every coding agent actually uses (Claude Code, Codex, OpenClaw
    and Hermes all stream), so routing content deltas through the
    reasoning sanitizer is what let the marker get eaten out of files
    the agent was asked to write.
    """
    return _sanitize_for_stream(text, _FINAL_SANITIZER)


def _sanitize_for_stream(text: str | None, pattern: re.Pattern[str]) -> str:
    """Shared body for the per-delta streaming sanitizers."""
    if not text:
        return ""
    # Fast-path bypass: no marker characters → no rewrite at all
    # (preserves identity, no whitespace touched).
    for ch in text:
        if ch in _SPECIAL_TOKEN_CHARS:
            # Strip markers ONLY — do NOT ``.strip()`` the whitespace
            # around them, because cross-delta whitespace is
            # load-bearing in the streaming concatenation contract.
            return pattern.sub("", text)
    return text


# Regex for matching final channel marker with optional constrain token:
#   <|channel|>final<|message|>
#   <|channel|>final <|constrain|>JSON<|message|>
_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>"
)

# Commentary-channel tool-call markers (both legacy and current forms).
# If ANY of these are present, the output carries tool-call structure
# that the harmony tool parser needs to see intact — bail out of
# stripping. Matches:
#   <|channel|>commentary to=functions.NAME ... <|message|>...<|call|>
#   to=functions.NAME<|channel|>commentary ... <|message|>...<|call|>
# Tool names follow the OpenAI/Anthropic naming spec (letters, digits,
# underscores, hyphens) — ``[\w-]+`` covers all of those. ``\w+`` alone
# would silently drop ``get-weather`` and any hyphenated builtin.
_COMMENTARY_TOOL_CALL_RE = re.compile(
    r"<\|channel\|>commentary\s+to=functions\.[\w-]+"
    r"|"
    r"to=functions\.[\w-]+<\|channel\|>commentary"
)


def _clean_gpt_oss_output(text: str) -> str:
    """
    Extract final channel content from GPT-OSS channel-based output.

    When reasoning parser is not enabled, this provides a fallback that
    extracts the 'final' channel content so the API response is usable.

    Handles both standard and extended format with constrain token:
        <|channel|>final<|message|>...
        <|channel|>final <|constrain|>JSON<|message|>...

    Args:
        text: Raw model output containing channel tokens.

    Returns:
        Extracted final content, or text with channel tokens stripped.
    """
    # Tool-call structure must survive to the harmony tool parser:
    # if the model emitted ``<|channel|>commentary to=functions.X...<|call|>``
    # (which gpt-oss-20b-mxfp4-q8 does for every tool invocation), the parser needs
    # those structural tokens intact to extract the call. Stripping them
    # here drops the args into plain text and the parser returns 0 calls.
    # Same regression class as PR #436 but for the tool parser. Final
    # channel is unaffected because the route runs ``clean_output_text``
    # again after parsers run (chat.py / anthropic.py).
    #
    # Reasoning-channel context is also preserved here: HarmonyReasoningParser
    # needs the analysis-channel markers intact to extract reasoning_content.
    # A previous "defense in depth" version stripped non-commentary tokens
    # before re-emitting commentary, which dropped the analysis channel and
    # broke pydantic_ai multi-tool turn loops (model lost its prior-call
    # context because reasoning_content came back empty, then called the
    # same tool repeatedly). Keep the bail-out simple: hand the entire
    # text to downstream parsers untouched.
    if _COMMENTARY_TOOL_CALL_RE.search(text):
        return text

    match = _FINAL_CHANNEL_RE.search(text)
    if match:
        content = text[match.end() :]
        # Strip trailing structural tokens (including <|constrain|>)
        content = re.sub(
            r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|message\|>|<\|constrain\|>",
            "",
            content,
        )
        return content.strip()

    # No final channel — strip all channel/structural tokens (including constrain)
    cleaned = re.sub(
        r"<\|channel\|>[^<]*(?:<\|constrain\|>[^<]*)?<\|message\|>|<\|start\|>[^<]*|<\|return\|>|<\|call\|>|<\|constrain\|>[^<]*",
        "",
        text,
    )
    return cleaned.strip()


def clean_output_text(text: str, *, muse_wire: bool = False) -> str:
    """
    Clean model output by removing special tokens.

    Keeps <think>...</think> blocks intact for reasoning models.
    Adds opening <think> tag if missing (happens when thinking is enabled
    in the prompt template but the tag is part of the prompt, not output).
    Handles GPT-OSS channel-based format as fallback when reasoning parser
    is not enabled.

    Args:
        text: Raw model output
        muse_wire: True when the SERVING MODEL is muse_glimmer (resolved
            from the checkpoint's model_type by the caller, never from
            output bytes). Gates the ATEM channel demux below so a
            non-muse model emitting literal wire-shaped text can never
            have its content misclassified and erased (codex r6 #1).

    Returns:
        Cleaned text with special tokens removed
    """
    if not text:
        return text

    # GPT-OSS channel format — extract final content before general stripping
    if "<|channel|>" in text and "<|message|>" in text:
        text = _clean_gpt_oss_output(text)
        return text

    # Muse Glimmer ATEM recipient-routed wire — extract the content
    # channels before generic stripping, mirroring the harmony branch
    # above. The generic SPECIAL_TOKENS_PATTERN would eat the
    # ``<|start|>/<|message|>/<|eot|>`` markers while leaving the
    # textual `` to=self`` header bytes and the reasoning bytes behind
    # as content mush (real-weights mini smoke, 2026-08-10). Muse has
    # ``<|message|>`` but no ``<|channel|>``, so this branch can only
    # be reached by non-harmony wire; the recipient-header probe keeps
    # ordinary prose that merely mentions ``<|message|>`` out.
    if muse_wire and "<|message|>" in text and _MUSE_WIRE_PROBE.search(text):
        from ..reasoning.muse_parser import MuseReasoningParser

        _, content = MuseReasoningParser().extract_reasoning(text)
        return (content or "").strip()

    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    text = text.strip()

    # Add opening <think> tag if response has closing but not opening
    # This happens when enable_thinking=True in the chat template
    if "</think>" in text and not text.lstrip().startswith("<think>"):
        text = "<think>" + text

    return text


# Pattern to match thinking blocks:
# - <think>...</think> (Qwen, DeepSeek, etc.)
# - <|channel>thought\n...<channel|> (Gemma 4)
THINK_PATTERN = re.compile(
    r"<think>[\s\S]*?</think>\s*"
    r"|<\|channel>thought\n[\s\S]*?<channel\|>\s*",
    re.DOTALL,
)


def strip_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Used when the client expects pure content (e.g., JSON) without
    reasoning blocks that would break parsing.

    Args:
        text: Model output that may contain thinking blocks

    Returns:
        Text with thinking blocks removed
    """
    if not text:
        return text
    return THINK_PATTERN.sub("", text).strip()


def extract_json_from_response(text: str) -> str:
    """
    Extract JSON object/array from model response, handling common wrapping.

    Models often wrap JSON in various ways:
    - Reasoning prefix: "Let me think... {json}"  (Qwen3)
    - Markdown code block: ```json\n{json}\n```    (Gemma 4, Llama)
    - Mixed: "Here's the result:\n```json\n{}\n```\nDone."
    - Plain JSON: {json}

    This is part of the output compensation layer — normalizes model
    output variations so downstream frameworks (PydanticAI, LangChain)
    see clean JSON regardless of model quirks.

    Args:
        text: Model output that may contain text before/after JSON

    Returns:
        Extracted JSON string if found, otherwise original text
    """
    if not text:
        return text

    text = text.strip()

    # If already valid JSON, return as-is
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return text

    # Strip markdown code blocks: ```json\n{...}\n``` or ```\n{...}\n```
    # This is the most common wrapping pattern (Gemma 4, Llama 3.x)
    stripped = _strip_markdown_code_block(text)
    if stripped != text:
        return stripped

    # Try to find JSON object at the end of the response
    # Find the last { and match to the end
    last_brace = text.rfind("{")
    if last_brace != -1 and text.endswith("}"):
        potential_json = text[last_brace:]
        if _is_balanced(potential_json, "{", "}"):
            return potential_json

    # Try to find JSON array at the end
    last_bracket = text.rfind("[")
    if last_bracket != -1 and text.endswith("]"):
        potential_json = text[last_bracket:]
        if _is_balanced(potential_json, "[", "]"):
            return potential_json

    # No JSON found, return original
    return text


def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block wrapping from text.

    Handles:
        ```json\n{...}\n```
        ```\n{...}\n```
        Text before ```json\n{...}\n``` text after
    """
    import re

    # Match ```json or ``` followed by content and closing ```
    pattern = re.compile(
        r"```(?:json|JSON)?\s*\n([\s\S]*?)\n\s*```",
    )
    match = pattern.search(text)
    if match:
        inner = match.group(1).strip()
        # Verify it looks like JSON
        if inner and (inner[0] in "{["):
            return inner
    return text


def _is_balanced(text: str, open_char: str, close_char: str) -> bool:
    """Check if brackets/braces are balanced."""
    depth = 0
    for char in text:
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
    return depth == 0


# =============================================================================
# Streaming Tool Call Filter
# =============================================================================

# Safety cap for tool call buffer (bytes). If a tool call block never closes,
# the buffer is capped to prevent unbounded memory growth. In practice, the
# buffer is bounded by max_tokens (~100KB at 32768 tokens), but this cap
# protects against pathological cases.
_MAX_TOOL_BUFFER_BYTES = 1_048_576  # 1 MB

# Tags that delimit tool call blocks in streaming output.
# Content inside these tags should be suppressed during streaming because
# it will be re-emitted as structured tool_use blocks after parsing.
#
# This list is extensible — call register_tool_call_tag() to add a pattern
# globally, or pass extra_tags to StreamingToolCallFilter for one request.
# Note the server never learns which agent is on the other end of a request,
# so a profile's ``streaming.extra_tool_tags`` cannot reach this list; any
# genuinely needed tag belongs here or in a per-request extra_tags argument.
_TOOL_CALL_TAGS: list[tuple[str, str]] = [
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"),  # DeepSeek V4 0731
    ("<｜DSML｜r:tool_calls>", "</｜DSML｜tool_calls>"),  # 0731 sampled alias
    ("<tool_call>", "</tool_call>"),  # hermes, qwen3
    ("<function=", "</function>"),
    ("[TOOL_CALL]", "[/TOOL_CALL]"),
    # Gemma 4 native wire-format markers (asymmetric: opener has no closing
    # ``|>`` and closer has no leading ``<|``). The mlx-vlm / mlx-lm streaming
    # detokenizer USUALLY strips these as special tokens (ids 48/49), but on
    # the ~40% of runs where the BPE byte form survives decode (issue #686
    # gemma-4-12b-4bit + Codex CLI), the raw markup leaks into the user-
    # visible ``response.output_text.delta`` channel. Suppressing the envelope
    # here also removes the inner ``<|"|>...<|"|>`` string-quoting markers,
    # because those only appear INSIDE the envelope (verified against the
    # gemma4_tool_parser pattern at line 41 + tokenizer_config.json
    # ``stc_token`` / ``etc_token`` fields). Confirmed in all three sources:
    #   - vllm_mlx/tool_parsers/gemma4_tool_parser.py (GEMMA4_TOOL_PATTERN)
    #   - tokenizer_config.json (stc_token / etc_token)
    #   - tests/test_output_router.py (special-token ids 48/49)
    ("<|tool_call>", "<tool_call|>"),
    (
        "[Calling tool",
        "\n",
    ),  # Bracket-style tool calls: suppress until newline (covers both ]\n and bare \n)
]


def register_tool_call_tag(open_tag: str, close_tag: str) -> bool:
    """Register an additional tool call tag pair for streaming suppression.

    Use this to extend the filter with agent-specific or model-specific
    markup patterns that should be suppressed during streaming.

    Args:
        open_tag: Opening tag (e.g. "<my_tool>")
        close_tag: Closing tag (e.g. "</my_tool>")

    Returns:
        True if the tag was added, False if it was already registered.
    """
    pair = (open_tag, close_tag)
    if pair not in _TOOL_CALL_TAGS:
        _TOOL_CALL_TAGS.append(pair)
        return True
    return False


def get_tool_call_tags() -> list[tuple[str, str]]:
    """Get the current list of tool call tag pairs (read-only copy)."""
    return list(_TOOL_CALL_TAGS)


class StreamingToolCallFilter:
    """Buffer streaming text to suppress tool call markup.

    Tool call XML (e.g. <minimax:tool_call>...</minimax:tool_call>) arrives
    split across multiple streaming deltas. This filter detects entry into a
    tool call block, suppresses all output until the block closes, and emits
    only non-tool-call text.

    The full unfiltered text is still accumulated separately for tool call
    parsing at stream end.

    Args:
        extra_tags: Additional (open, close) tag pairs to suppress, beyond
                    the global _TOOL_CALL_TAGS. Useful for per-request or
                    per-agent customization without mutating global state.
    """

    def __init__(self, extra_tags: list[tuple[str, str]] | None = None):
        self._buffer = ""
        self._in_block = False
        self._close_tag = ""
        # Merge global tags with per-instance extras
        self._tags = _TOOL_CALL_TAGS
        if extra_tags:
            self._tags = _TOOL_CALL_TAGS + [
                t for t in extra_tags if t not in _TOOL_CALL_TAGS
            ]
        # Longest open tag - used to determine how much buffer to hold back
        self._max_open_len = max(len(t[0]) for t in self._tags)

    def process(self, delta: str) -> str:
        """Process a streaming delta. Returns text to emit (may be empty)."""
        self._buffer += delta

        if self._in_block:
            return self._consume_block()
        else:
            return self._scan_for_open()

    def _scan_for_open(self) -> str:
        """Scan buffer for tool call open tags. Emit safe text."""
        # Check for complete open tags
        for open_tag, close_tag in self._tags:
            idx = self._buffer.find(open_tag)
            if idx >= 0:
                # Found an open tag - emit text before it, enter block mode
                emit = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(open_tag) :]
                self._in_block = True
                self._close_tag = close_tag
                # Process remainder in case close tag is already in buffer
                after = self._consume_block()
                return emit + after

        # No complete open tag found. Check if buffer ends with a partial
        # match of any open tag - hold that back to avoid emitting a fragment.
        hold_back = 0
        for open_tag, _ in self._tags:
            for prefix_len in range(min(len(open_tag), len(self._buffer)), 0, -1):
                if self._buffer.endswith(open_tag[:prefix_len]):
                    hold_back = max(hold_back, prefix_len)
                    break

        if hold_back > 0:
            emit = self._buffer[:-hold_back]
            self._buffer = self._buffer[-hold_back:]
            return emit

        # No partial match - safe to emit everything
        emit = self._buffer
        self._buffer = ""
        return emit

    def _consume_block(self) -> str:
        """Consume content inside a tool call block. Returns empty string
        unless the block closes and there's text after it."""
        idx = self._buffer.find(self._close_tag)
        if idx >= 0:
            # Block closed - discard content up to and including close tag
            self._buffer = self._buffer[idx + len(self._close_tag) :]
            self._in_block = False
            self._close_tag = ""
            # Process remainder - might have more text or another tool call
            if self._buffer:
                return self._scan_for_open()
            return ""
        # Still inside block - suppress everything but cap buffer size
        if len(self._buffer) > _MAX_TOOL_BUFFER_BYTES:
            logger.warning(
                f"Tool call buffer exceeded {_MAX_TOOL_BUFFER_BYTES} bytes, "
                f"discarding and exiting block"
            )
            self._buffer = ""
            self._in_block = False
            self._close_tag = ""
        return ""

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        if self._in_block:
            # Unterminated tool call block - discard
            self._buffer = ""
            self._in_block = False
            return ""
        emit = self._buffer
        self._buffer = ""
        return emit


# =============================================================================
# Streaming Think Block Router
# =============================================================================


class StreamingThinkRouter:
    """Route <think>...</think> content to separate Anthropic thinking blocks.

    Instead of emitting thinking content as plain text (where it's
    indistinguishable from the response), this router yields tagged
    pieces that the streaming handler can emit as proper Anthropic
    content block types.

    Each call to process() returns a list of (block_type, text) tuples:
    - ("thinking", text) for content inside <think>...</think>
    - ("text", text) for content outside think blocks

    Args:
        start_in_thinking: If True, assume the model starts in thinking
            mode (e.g. MiniMax adds <think> to the generation prompt,
            so the tag never appears in the output stream).
    """

    def __init__(self, start_in_thinking: bool = False):
        self._buffer = ""
        self._in_think = start_in_thinking

    def process(self, delta: str) -> list[tuple[str, str]]:
        """Process a delta. Returns list of (block_type, text) pieces."""
        self._buffer += delta
        pieces = []
        self._extract_pieces(pieces)
        return pieces

    def _extract_pieces(self, pieces: list[tuple[str, str]]) -> None:
        """Extract all complete pieces from the buffer."""
        while True:
            if self._in_think:
                idx = self._buffer.find("</think>")
                if idx >= 0:
                    # Emit thinking content, exit think mode
                    thinking = self._buffer[:idx]
                    self._buffer = self._buffer[idx + len("</think>") :]
                    self._in_think = False
                    if thinking:
                        pieces.append(("thinking", thinking))
                    continue  # Process remainder
                else:
                    # Check for partial close tag at end
                    for plen in range(min(len("</think>"), len(self._buffer)), 0, -1):
                        if self._buffer.endswith("</think>"[:plen]):
                            # Hold back partial match
                            emit = self._buffer[:-plen]
                            self._buffer = self._buffer[-plen:]
                            if emit:
                                pieces.append(("thinking", emit))
                            return
                    # No partial match - emit all as thinking
                    if self._buffer:
                        pieces.append(("thinking", self._buffer))
                        self._buffer = ""
                    return
            else:
                idx = self._buffer.find("<think>")
                if idx >= 0:
                    # Emit text before tag, enter think mode
                    before = self._buffer[:idx]
                    self._buffer = self._buffer[idx + len("<think>") :]
                    self._in_think = True
                    if before:
                        pieces.append(("text", before))
                    continue  # Process remainder
                else:
                    # Check for partial open tag at end
                    for plen in range(min(len("<think>"), len(self._buffer)), 0, -1):
                        if self._buffer.endswith("<think>"[:plen]):
                            emit = self._buffer[:-plen]
                            self._buffer = self._buffer[-plen:]
                            if emit:
                                pieces.append(("text", emit))
                            return
                    # No partial match - emit all as text
                    if self._buffer:
                        pieces.append(("text", self._buffer))
                        self._buffer = ""
                    return

    def flush(self) -> list[tuple[str, str]]:
        """Flush remaining buffer at end of stream."""
        pieces = []
        if self._buffer:
            block_type = "thinking" if self._in_think else "text"
            pieces.append((block_type, self._buffer))
            self._buffer = ""
        self._in_think = False
        return pieces


# =============================================================================
# Model Detection
# =============================================================================

# Patterns that indicate a multimodal language model (MLLM/VLM)
MLLM_PATTERNS = [
    "-VL-",
    "-VL/",
    "VL-",  # Qwen-VL, Qwen2-VL, Qwen3-VL, etc.
    "llava",
    "LLaVA",  # LLaVA models
    "idefics",
    "Idefics",  # Idefics models
    "paligemma",
    "PaliGemma",  # PaliGemma
    "gemma-3",
    "gemma3",  # Gemma 3 (multimodal)
    "medgemma",
    "MedGemma",  # MedGemma (medical multimodal with SigLIP vision encoder)
    "pixtral",
    "Pixtral",  # Pixtral
    "molmo",
    "Molmo",  # Molmo
    "phi3-vision",
    "phi-3-vision",  # Phi-3 Vision
    "cogvlm",
    "CogVLM",  # CogVLM
    "internvl",
    "InternVL",  # InternVL
    "deepseek-vl",
    "DeepSeek-VL",  # DeepSeek-VL
    # UI-TARS (ByteDance) — Qwen2-VL / Qwen2.5-VL based GUI-agent VLM.
    # The model id ``UI-TARS-…`` does not match the generic ``-VL-`` pattern
    # (the VL part is in the underlying architecture, not the public name),
    # so list it explicitly. Without this entry, ``is_mllm_model`` returns
    # False on full HF paths like ``mlx-community/UI-TARS-1.5-7B-4bit``
    # and the engine boots the text-only path, breaking the screenshot+
    # instruction contract every UI-TARS deployment needs.
    "UI-TARS",
    "ui-tars",
    "UI_TARS",
    "ui_tars",
]


def _try_read_config_json(name_or_path: str) -> dict | None:
    """Compatibility wrapper for local config metadata lookup."""
    metadata = read_local_model_metadata(name_or_path)
    return metadata.config if metadata is not None else None


def _try_read_hub_config_json(model_name: str) -> dict | None:
    """Compatibility wrapper for cached HF config metadata lookup."""
    metadata = read_cached_model_metadata(model_name)
    return metadata.config if metadata is not None else None


def _config_indicates_vlm(config: dict) -> bool:
    """Compatibility wrapper for shared multimodal config inspection."""
    return config_indicates_multimodal(config)


def _local_checkpoint_has_multimodal_weights(model_dir) -> bool | None:
    """Compatibility wrapper for sharded-checkpoint modality inspection."""
    return checkpoint_has_multimodal_weights(model_dir)


def _check_legacy_string_patterns(model_name: str) -> bool:
    """Validation 1: substring match of MLLM_PATTERNS against the input string.

    Kept for HF repo IDs (where no local config.json is reachable) and as
    a fallback when config.json cannot be read.
    """
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in MLLM_PATTERNS)


def is_mllm_model(model_name: str) -> bool:
    """Check if a model name or path indicates a multimodal language model.

    A curated alias that POSITIVELY declares text-only serving
    (``is_text_only`` — the #393 state-pin) is authoritative and stays on the
    text lane even against real vision weights: it is an operator decision, with
    text-serve coverage, to run a vision-config checkpoint through the AR text
    lane.  A bare alias that merely defaults to the ``text`` modality does NOT
    override real vision weights — such a repackaged VLM still routes to the
    MLLM lane via the checkpoint-evidence path (#1121).  Otherwise, local
    metadata is authoritative for unaliased paths.  A cached HF snapshot is
    promoted only after it supplies positive modality evidence; this preserves
    text routing for a partial cache that contains an inherited vision config
    but no vision weights, and an inconclusive verdict is NEVER promoted on the
    bare existence of checkpoint files.  The shared probe never sends a network
    request.  It applies two checks in order:

    1. Config inspection: ``architectures`` / ``vision_config`` /
       ``audio_config`` declare whether the checkpoint is multimodal.

    2. Weights-presence override: when the config says "VLM" but a
       ``model.safetensors.index.json`` has NO
       multimodal tensors (``vision_tower``, ``visual.``, ``mm_projector``,
       …), the checkpoint is a text-only fork of a multimodal
       architecture. Flip the answer to False so the model loads
       through the text path instead of crashing in the MLLM batched
       engine on a missing vision tower. Fixes #393 (Qwen3.6-35B-A3B
       text-only fork — config.json declares ``vision_config`` because
       the base ``Qwen3_5MoeForConditionalGeneration`` architecture is
       multimodal-capable, but the user's safetensors only contain
       language tensors).

    If cached metadata has no index/header evidence, the legacy name matcher
    remains the compatibility fallback.  During ``serve``, model download has
    already completed before this function runs, so a re-packaged VLM such as
    Agents-A1 has its actual checkpoint evidence available rather than relying
    on an arbitrary repository name.

    Args:
        model_name: HuggingFace repo ID or local filesystem path.

    Returns:
        True if the model is detected as multimodal (MLLM/VLM).
    """
    profile = resolve_profile(model_name)
    # A curated alias that declares text-only serving (``is_text_only`` — the
    # #393 state-pin) is authoritative and outranks the checkpoint's raw vision
    # weights: it is a deliberate operator decision, with text-serve coverage,
    # to run a vision-config checkpoint through the AR text lane.  Some upstream
    # repackages of a Qwen3.5 text model still ship a ``vision_tower`` in the
    # safetensors index, so the weight-evidence path below would (correctly, on
    # the raw bytes) return True; the pin short-circuits BEFORE that path so the
    # curated text modality wins.  A bare alias that merely defaults to the
    # ``text`` modality is NOT short-circuited here — real vision weights still
    # route it to the MLLM lane via the evidence path (#1121), so a repackaged
    # VLM served under a text-family alias name is not misrouted to text.
    if profile is not None and profile.is_text_only:
        return False

    metadata = read_model_metadata(model_name)
    config = metadata.config if metadata is not None else None
    if config is not None:
        if not _config_indicates_vlm(config):
            return False
        verdict = checkpoint_has_multimodal_weights(metadata.snapshot_dir, config)
        if verdict is False:
            return False
        # Positive checkpoint evidence is authoritative and must win over any
        # repository-name pattern.  A registered alias only short-circuits when
        # it POSITIVELY declares text-only (handled above via
        # ``profile.is_text_only``); a bare alias entry must never override real
        # vision weights, or a renamed/repackaged VLM served under an aliased
        # text-family name would be misrouted to the text engine (#1121).
        if verdict is True:
            return True
        # An inconclusive verdict (``None`` — the checkpoint was inspected but
        # its tensor layout is unrecognised, or there is no index/header
        # evidence at all) routes to the NAME/LOCALITY fallback below, NOT to a
        # forced-multimodal verdict.
        #
        # DELIBERATE OVERRIDE of codex round-3 #1, which asked to PRESERVE the
        # multimodal route on an inspected-but-unknown layout.  That reverses
        # codex round-2 #1, which demanded we "never promote an inconclusive
        # result based solely on file existence" — the author complied and
        # removed exactly that promotion.  Honouring round-3 #1 would
        # re-introduce what round-2 #1 rejected (codex oscillated between the two
        # rounds).  Both misroute directions are harmful: force-VLM crashes a
        # text-only fork in the MLLM batched engine on a missing vision tower
        # (#393), while force-text starves a genuine VLM of its vision path.
        # With genuinely ambiguous evidence there is no evidentiary basis to
        # prefer either, so the name/locality heuristic — the established
        # rapid-mlx default — is used as the NEUTRAL tie-breaker: it decides on
        # the repo-name signal rather than on the bare presence of files.  This
        # note is left so a future reviewer sees the round-2↔round-3 oscillation
        # and the reasoned choice rather than re-litigating it.
        #
        # The fallback itself: a registered alias or a cached remote config can
        # be a partial metadata download, so a name without legacy MLLM evidence
        # stays on text until positive proof.  A bare local directory with a VLM
        # config and no index is trusted as multimodal.
        if profile is not None:
            return _check_legacy_string_patterns(model_name)
        if metadata.is_local:
            return True
        return _check_legacy_string_patterns(model_name)

    return _check_legacy_string_patterns(model_name)


# Backwards compatibility alias
is_vlm_model = is_mllm_model


# Multimodal model_types for which we ship a vendored TEXT backbone
# (``vllm_mlx/models/<model_type>.py``) while the installed mlx-vlm has no
# model package for the arch. ``resolve_serving_lane`` auto-downgrades these
# to the text lane instead of letting the MLLM engine crash at load on an
# unknown architecture. The ``find_spec`` probe in
# :func:`mllm_arch_unsupported_but_text_vendored` flips a type back to
# multimodal routing automatically once a dependency bump ships support —
# no code change needed here (remove the entry when the vendored backbone
# itself is retired).
_VENDORED_TEXT_FALLBACK_MODEL_TYPES = ("muse_glimmer",)


def mllm_arch_unsupported_but_text_vendored(model_name: str) -> bool:
    """True iff the checkpoint's arch needs the text-lane vendored fallback.

    Offline (cached config only, never loads weights). Returns True when
    BOTH hold:

    1. ``config.model_type`` is in ``_VENDORED_TEXT_FALLBACK_MODEL_TYPES``
       (we vendor its text backbone), and
    2. the installed mlx-vlm has no ``mlx_vlm.models.<model_type>``
       package (including mlx-vlm not installed at all) — so the MLLM
       lane would crash at load.

    Once mlx-vlm ships the arch (e.g. Blaizzy/mlx-vlm#1838 for
    ``muse_glimmer``) the ``find_spec`` probe finds the package and this
    returns False, restoring normal multimodal routing without a code
    change.
    """
    metadata = read_model_metadata(model_name)
    config = metadata.config if metadata is not None else None
    if not isinstance(config, dict):
        return False
    if config.get("model_type") not in _VENDORED_TEXT_FALLBACK_MODEL_TYPES:
        return False
    import importlib.util as _importlib_util

    try:
        spec = _importlib_util.find_spec(f"mlx_vlm.models.{config['model_type']}")
    except (ImportError, ValueError):
        # mlx-vlm absent entirely (no ``[vision]`` extra) — the MLLM lane
        # is unavailable regardless, so the vendored text fallback applies.
        spec = None
    return spec is None


def mllm_backbone_is_hybrid(model_name: str) -> bool:
    """True when a checkpoint's *language* backbone is hybrid/linear-attention.

    A hybrid backbone (Qwen3.5/3.6 GatedDeltaNet ``linear_attention`` layers,
    Mamba/recurrent state-space blocks, …) produces ``ArraysCache`` layers that
    the MLLM continuous-batching engine cannot assemble into a ``BatchKVCache``
    — it needs standard ``KVCache`` / ``RotatingKVCache`` (GitHub #352). A
    genuine VLM whose backbone is plain / sliding attention (Gemma-4, Qwen3-VL)
    returns False and keeps its multimodal routing.

    The signal is the checkpoint config alone — the ``text_config.layer_types``
    list and recurrent ``model_type`` markers — so this is offline, never loads
    weights, and never hits the network. A missing/unreadable config returns
    False (unknown → no fallback), matching the conservative contract of
    :func:`is_mllm_model`: this probe only ever *adds* a text-lane fallback for a
    demonstrably-incompatible backbone, never removes multimodal routing from a
    checkpoint we cannot positively classify as hybrid.

    Args:
        model_name: HuggingFace repo ID or local filesystem path.

    Returns:
        True if the language backbone uses linear-attention / recurrent layers.
    """
    metadata = read_model_metadata(model_name)
    config = metadata.config if metadata is not None else None
    if not isinstance(config, dict):
        return False
    text_cfg = config.get("text_config")
    if not isinstance(text_cfg, dict):
        text_cfg = config

    # Per-layer attention markers: GatedDeltaNet declares ``linear_attention``
    # entries interleaved with ``full_attention``; Mamba/recurrent blocks label
    # their own layers. Sliding-window attention (``sliding_attention``) is NOT
    # hybrid — it still uses a RotatingKVCache the MLLM engine handles.
    layer_types = text_cfg.get("layer_types")
    if isinstance(layer_types, list) and any(
        isinstance(lt, str)
        and any(tok in lt.lower() for tok in ("linear", "mamba", "recurrent"))
        for lt in layer_types
    ):
        return True

    # Whole-model recurrent / state-space architectures that don't enumerate
    # per-layer types (pure Mamba, RecurrentGemma, Qwen3-Next linear stack).
    for mt in (text_cfg.get("model_type"), config.get("model_type")):
        if isinstance(mt, str) and any(
            tok in mt.lower() for tok in ("mamba", "recurrent", "qwen3_next")
        ):
            return True
    return False


def resolve_serving_lane(
    model_name: str, *, force_mllm: bool = False, force_text: bool = False
) -> tuple[bool, bool]:
    """Decide the FINAL serving lane for a model, resolving the automatic
    hybrid→text-only fallback up front so every consumer (PFlash defaulting,
    ``validate_model_support``, engine selection, diagnostics) agrees on one
    answer.

    Returns ``(is_mllm_lane, auto_text_fallback)``:

    * ``is_mllm_lane`` — True iff the model will actually be served on the
      MLLM/VLM continuous-batching lane. This is the verdict PFlash defaulting
      and ``validate_model_support`` must use — NOT the raw ``is_mllm_model``
      checkpoint classification — so a checkpoint that auto-downgrades to the
      text lane is treated as text (PFlash-capable) everywhere, exactly as an
      explicit ``--text-only`` run would be.
    * ``auto_text_fallback`` — True iff a multimodal checkpoint was
      *automatically* routed to the text-only lane because its language
      backbone is hybrid/linear-attention (ArraysCache, incompatible with MLLM
      continuous batching — GitHub #352). Kept DISTINCT from an explicit
      ``--no-mllm``/``force_text`` so diagnostics can say "auto-downgraded"
      rather than falsely claim the user passed ``--no-mllm``.

    Explicit flags win: ``force_text`` → text lane (no auto-fallback marker),
    ``force_mllm`` → MLLM lane (the engine may still reject a hybrid backbone
    with its own #352 error — the operator asked for it, so they get the flag
    they set named in the message).

    The two probes are offline and read the checkpoint config from the local
    cache. Callers MUST materialize that config first (the model download must
    have completed — see ``_ensure_model_downloaded``) so the probes have real
    evidence instead of a name-pattern guess; otherwise a first-time uncached
    hybrid VLM would probe "not hybrid" and be routed into the crashing MLLM
    engine (#352 dogfood P1-②).
    """
    if force_text:
        return (False, False)
    if force_mllm:
        return (True, False)
    if not is_mllm_model(model_name):
        return (False, False)
    # Auto-routed to the MLLM lane by its vision weights — but the lane
    # cannot serve every backbone. Two auto-downgrade causes, both marked
    # with the same ``auto_text_fallback`` flag:
    #  * the installed mlx-vlm has no model package for the arch and we
    #    vendor a text backbone for it (Muse Glimmer until
    #    Blaizzy/mlx-vlm#1838 ships in a release we pin);
    #  * a hybrid/linear-attention backbone produces ArraysCache layers the
    #    MLLM continuous-batching engine cannot assemble (GitHub #352).
    if mllm_arch_unsupported_but_text_vendored(model_name):
        return (False, True)
    if mllm_backbone_is_hybrid(model_name):
        return (False, True)
    return (True, False)


def decode_inline_tool_call_arguments(messages: list[dict]) -> None:
    """Decode `tool_calls[].function.arguments` from JSON string to dict in-place.

    The OpenAI API serializes tool-call arguments as a JSON-encoded string.
    Some chat templates (GLM-4.6V, Qwen3 MLLM variants) iterate the arguments
    dict via `.items()`/`|items` and crash on a string. The non-MLLM path
    handles this inside `extract_multimodal_content()`; the MLLM branch
    bypasses that helper, so callers in the MLLM path call this directly.

    Mutates `messages` in-place. Malformed JSON is left untouched.
    """
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function") or {}
            args = func.get("arguments")
            if isinstance(args, str):
                try:
                    func["arguments"] = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass


# =============================================================================
# Multimodal Content Extraction
# =============================================================================

TEXT_CONTENT_TYPES = {"text", "input_text", "output_text"}
IMAGE_CONTENT_TYPES = {"image_url", "image", "input_image"}
VIDEO_CONTENT_TYPES = {"video", "video_url"}
AUDIO_CONTENT_TYPES = {"audio_url", "audio", "input_audio"}
MEDIA_CONTENT_TYPES = IMAGE_CONTENT_TYPES | VIDEO_CONTENT_TYPES | AUDIO_CONTENT_TYPES
KNOWN_CONTENT_TYPES = TEXT_CONTENT_TYPES | MEDIA_CONTENT_TYPES
SUPPORTED_INPUT_AUDIO_FORMATS = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "pcm",
    "m4a",
    "webm",
}


def _content_part_to_dict(item) -> dict:
    """Return a plain dict for a content part, or raise on malformed shape."""
    if hasattr(item, "model_dump"):
        item = item.model_dump(exclude_none=True)
    elif hasattr(item, "dict"):
        item = {k: v for k, v in item.dict().items() if v is not None}

    if not isinstance(item, dict):
        raise ValueError(f"content blocks must be objects (got {type(item).__name__})")
    item_type = item.get("type")
    if not isinstance(item_type, str) or not item_type:
        raise ValueError("content block is missing required string field 'type'")
    if item_type not in KNOWN_CONTENT_TYPES:
        raise ValueError(f"Unsupported content block type: {item_type!r}")
    return item


def _require_string(value, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(
            f"{field_name} must be a non-empty string (got {type(value).__name__})"
        )
    return value


def _extract_object_url(item: dict, field_name: str) -> str:
    value = item.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be an object with required field 'url' "
            f"(got {type(value).__name__})"
        )
    return _require_string(value.get("url"), f"{field_name}.url")


def _validate_content_part_payload(item: dict) -> None:
    item_type = item["type"]
    if item_type in TEXT_CONTENT_TYPES:
        if "text" not in item:
            raise ValueError(f"{item_type}.text is required")
        text = item.get("text")
        if not isinstance(text, str):
            if item_type in {"input_text", "output_text"}:
                raise ValueError(
                    f"{item_type}.text must be a string (got {type(text).__name__})"
                )
            raise ValueError(
                f"content[].text must be a non-empty string (got {type(text).__name__})"
            )
        if text == "" and item_type in {"input_text", "output_text"}:
            raise ValueError(f"{item_type}.text must be a non-empty string")
    elif item_type == "image_url":
        _extract_object_url(item, "image_url")
    elif item_type == "input_image":
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            _require_string(image_url.get("url"), "input_image.image_url.url")
        else:
            _require_string(image_url, "input_image.image_url")
    elif item_type == "image":
        _require_string(item.get("image", item.get("url")), "image")
    elif item_type == "video":
        _require_string(item.get("video", item.get("url")), "video")
    elif item_type == "video_url":
        _extract_object_url(item, "video_url")
    elif item_type == "audio_url":
        _extract_object_url(item, "audio_url")
    elif item_type == "audio":
        _require_string(item.get("audio", item.get("url")), "audio")
    elif item_type == "input_audio":
        value = item.get("input_audio")
        if not isinstance(value, dict):
            raise ValueError(
                "input_audio must be an object with required fields 'data' "
                "and 'format'; input_audio.format is required"
            )
        _require_string(value.get("data"), "input_audio.data")
        if "format" not in value:
            raise ValueError("input_audio.format is required")
        # Validation is intentionally non-mutating; downstream code that
        # consumes audio blocks should normalize casing at its own boundary.
        audio_format = _require_string(
            value.get("format"), "input_audio.format"
        ).lower()
        if audio_format not in SUPPORTED_INPUT_AUDIO_FORMATS:
            raise ValueError(
                "input_audio.format must be one of "
                f"{sorted(SUPPORTED_INPUT_AUDIO_FORMATS)}"
            )


def validate_content_blocks_for_capabilities(
    messages: list,
    *,
    model_name: str,
    allow_image: bool,
    allow_video: bool,
    allow_audio: bool = False,
) -> None:
    """Reject content blocks the active model/path cannot preserve."""
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for raw_item in content:
            item = _content_part_to_dict(raw_item)
            item_type = item["type"]
            _validate_content_part_payload(item)
            if item_type in TEXT_CONTENT_TYPES:
                continue
            if item_type in IMAGE_CONTENT_TYPES and allow_image:
                continue
            if item_type in VIDEO_CONTENT_TYPES and allow_video:
                continue
            if item_type == "input_audio" and allow_audio:
                continue

            if item_type in AUDIO_CONTENT_TYPES:
                detail = (
                    "audio inputs in this shape; only input_audio is supported"
                    if allow_audio
                    else "audio inputs"
                )
            elif item_type in IMAGE_CONTENT_TYPES:
                detail = "image inputs"
            elif item_type in VIDEO_CONTENT_TYPES:
                detail = "video inputs"
            else:
                detail = f"{item_type!r} content blocks"
            raise ValueError(f"Model '{model_name}' does not support {detail}.")


def normalize_responses_content_part(item) -> dict:
    """Convert a Responses input content item into Chat content-part shape."""
    data = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
    if not isinstance(data, dict):
        raise ValueError(
            f"Responses content blocks must be objects (got {type(data).__name__})"
        )
    item_type = data.get("type")
    if item_type in ("input_text", "output_text"):
        if "text" not in data:
            raise ValueError(f"{item_type}.text is required")
        text = data.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"{item_type}.text must be a string (got {type(text).__name__})"
            )
        if text == "":
            raise ValueError(f"{item_type}.text must be a non-empty string")
        return {"type": "text", "text": text}
    if item_type == "input_image":
        image_url = data.get("image_url")
        if isinstance(image_url, dict):
            normalized_image_url = {
                key: value
                for key, value in image_url.items()
                if key in {"url", "detail"}
            }
            _require_string(
                normalized_image_url.get("url"), "input_image.image_url.url"
            )
        else:
            url = _require_string(image_url, "input_image.image_url")
            normalized_image_url = {"url": url}
        return {"type": "image_url", "image_url": normalized_image_url}
    if item_type == "input_audio":
        raise ValueError("Responses input_audio content blocks are not supported")
    raise ValueError(f"Unsupported Responses content block type: {item_type!r}")


def _content_to_text(content) -> str:
    """Extract text from content that can be str, list[ContentPart], or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            data = (
                item.model_dump(exclude_none=True)
                if hasattr(item, "model_dump")
                else item
            )
            if isinstance(data, dict) and data.get("type") in (
                "text",
                "input_text",
                "output_text",
            ):
                text = data.get("text", "")
                parts.append(text if isinstance(text, str) else "")
        return "\n".join(parts)
    return str(content)


def extract_multimodal_content(
    messages: list[Message],
    preserve_native_format: bool = False,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Extract text content, images, and videos from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Multimodal messages with images/videos
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        preserve_native_format: If True, preserve native tool message format
            (role="tool", tool_calls field) instead of converting to text.
            Required for models with native tool support in chat templates
            (e.g., Mistral, Llama 3+, DeepSeek V3).

    Returns:
        Tuple of (processed_messages, images, videos)
        - processed_messages: List of {"role": str, "content": str}
        - images: List of image URLs/paths/base64
        - videos: List of video URLs/paths/base64
    """
    processed_messages = []
    images = []
    videos = []

    for msg in messages:
        # Handle both dict and Pydantic model messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content")
        else:
            role = msg.role
            content = msg.content

        # Handle tool response messages (role="tool")
        if role == "tool":
            if isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id", "") or ""
            else:
                tool_call_id = getattr(msg, "tool_call_id", None) or ""
            # F-111: tool replies routinely arrive as
            # ``content: [{"type":"text","text":"X"}]`` (OpenAI o1/o3
            # SDK default). Downstream the message is run through
            # ``_normalize_tool_call_arguments_for_template`` which
            # serialises everything with ``json.dumps(..., default=str)``;
            # a pydantic ``ContentPart`` instance there is coerced to its
            # ``repr()`` string and the chat template renders garbage.
            # Flatten text-only content arrays to a plain string at the
            # API boundary so every downstream stage sees the same shape
            # as the legacy ``content: "X"`` string form. ``_content_to_text``
            # already does the right thing for text parts and is what the
            # assistant branch uses too — single source of truth. The
            # F-111 route-level validator has already rejected non-text
            # parts on a ``tool`` role before we get here, so the flatten
            # is loss-free in production (the only non-text path here is
            # the tests that bypass the route validator).
            tool_content = _content_to_text(content) if content else ""

            if preserve_native_format:
                # Preserve native tool format for models that support it
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                # Convert to user role for models without native support
                processed_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                    }
                )
            continue

        # Handle assistant messages with tool_calls
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "assistant" and tool_calls:
            if preserve_native_format:
                # Preserve native tool_calls format
                tool_calls_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_copy = tc
                    elif hasattr(tc, "model_dump"):
                        tc_copy = tc.model_dump()
                    elif hasattr(tc, "dict"):
                        tc_copy = tc.dict()
                    else:
                        continue

                    # Chat templates (e.g. Qwen3) iterate arguments|items,
                    # but OpenAI API sends arguments as a JSON string.
                    # Parse it into a dict so the template can iterate it.
                    func = tc_copy.get("function") or {}
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            import json

                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass

                    tool_calls_list.append(tc_copy)

                msg_dict = {"role": role, "content": _content_to_text(content)}
                if tool_calls_list:
                    msg_dict["tool_calls"] = tool_calls_list
                processed_messages.append(msg_dict)
            else:
                # Convert tool calls to text for models without native support
                tool_calls_text = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")

                text = _content_to_text(content)
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)

                processed_messages.append({"role": role, "content": text})
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multimodal message - extract text and media
            text_parts = []
            for item in content:
                item = _content_part_to_dict(item)
                _validate_content_part_payload(item)
                item_type = item.get("type", "")

                if item_type in TEXT_CONTENT_TYPES:
                    text_parts.append(item.get("text", ""))

                elif item_type == "image_url":
                    images.append(_extract_object_url(item, "image_url"))

                elif item_type == "image":
                    images.append(
                        _require_string(item.get("image", item.get("url")), "image")
                    )

                elif item_type == "input_image":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        images.append(
                            _require_string(
                                image_url.get("url"), "input_image.image_url.url"
                            )
                        )
                    else:
                        images.append(
                            _require_string(image_url, "input_image.image_url")
                        )

                elif item_type == "video":
                    videos.append(
                        _require_string(item.get("video", item.get("url")), "video")
                    )

                elif item_type == "video_url":
                    videos.append(_extract_object_url(item, "video_url"))

                elif item_type in AUDIO_CONTENT_TYPES:
                    raise ValueError(
                        "Audio content blocks are not supported on this path."
                    )

            # Combine text parts
            combined_text = "\n".join(text_parts) if text_parts else ""
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages, images, videos
