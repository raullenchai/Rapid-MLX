# SPDX-License-Identifier: Apache-2.0
"""
Adapter for converting between Anthropic Messages API and OpenAI Chat Completions API.

Handles translation of:
- Requests: Anthropic → OpenAI format
- Responses: OpenAI → Anthropic format
- Messages: Content blocks, tool calls, tool results
"""

import json
import re
import secrets
import uuid

from .anthropic_models import (
    ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS,
    AnthropicMessage,
    AnthropicOutputConfig,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicToolDef,
    AnthropicUsage,
)
from .constants import (
    RESCUE_TAIL_LENGTH,
    is_rescue_payload,
)
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolDefinition,
)
from .utils import sanitize_output, strip_reasoning_channel_markup

# F9: Anthropic's public spec uses ``id="toolu_<hex>"`` on every
# ``tool_use`` block (and every matching ``tool_result.tool_use_id``).
# Our underlying tool parsers all mint OpenAI-style ``call_<hex>`` IDs
# (see the ~20 tool_parsers/*.py call sites that share the
# ``f"call_{uuid.uuid4().hex[:8]}"`` shape) — which is the right thing
# for the OpenAI ``/v1/chat/completions`` surface but leaks the
# underlying conversion through the Anthropic ``/v1/messages`` envelope.
#
# Single source of truth: the Anthropic adapter rewrites IDs as they
# cross the boundary. ``to_anthropic_tool_use_id`` preserves the hex
# tail when the input already follows the ``call_<hex>`` shape so
# call-id correlation across logs still works; otherwise it mints a
# fresh ``toolu_<24 hex>`` id matching Anthropic's public examples.
#
# Cross-route consistency: ``/v1/chat/completions`` and ``/v1/responses``
# keep returning ``call_<hex>`` (OpenAI parity). Only the Anthropic
# adapter and route apply this rewrite.
_TOOLU_TAIL_RE = re.compile(r"^[0-9a-fA-F]+$")


def to_anthropic_tool_use_id(openai_id: str | None) -> str:
    """Convert an OpenAI-style ``call_<hex>`` id to Anthropic's
    ``toolu_<hex>`` id (or mint a fresh one when the input is missing
    or unusable).

    Preserves the hex tail when present so an operator can correlate
    the same call across the OpenAI-side parser log and the
    Anthropic-side wire response — matching the F9 single-source-of-
    truth requirement. Codex r2 BLOCKING #3: require a non-empty tail
    on the ``call_`` rewrite branch so a degenerate input like
    ``"call_"`` (empty tail) doesn't produce the invalid
    ``"toolu_"`` id; mint a fresh ``toolu_<hex>`` in that case
    instead. Same guard applies to a bare ``"toolu_"`` pass-through
    so a future caller can't accidentally re-emit an empty-tail id.

    Codex r4 BLOCKING #1: the public contract is ``toolu_<hex>`` —
    only preserve the tail when it actually matches that shape
    (``[0-9a-fA-F]+``). Malformed upstream ids like
    ``"call_unknown_prefix_!!!"`` (caller bug, attacker probe, or a
    third-party tool parser that didn't follow our convention) now
    fall through to a fresh ``toolu_{secrets.token_hex(12)}`` rather
    than emitting a non-hex tail on the Anthropic wire.
    """
    if isinstance(openai_id, str) and openai_id.startswith("call_"):
        tail = openai_id[len("call_") :]
        if tail and _TOOLU_TAIL_RE.match(tail):
            return "toolu_" + tail
    if isinstance(openai_id, str) and openai_id.startswith("toolu_"):
        tail = openai_id[len("toolu_") :]
        if tail and _TOOLU_TAIL_RE.match(tail):
            return openai_id
    # Anthropic's public examples use ~24 hex chars after ``toolu_``;
    # ``secrets.token_hex(12)`` gives 24 hex chars from a CSPRNG so
    # we don't rely on uuid4's structure leaking into the id.
    return f"toolu_{secrets.token_hex(12)}"


class AnthropicOutputConfigError(ValueError):
    """Raised when ``output_config`` on a /v1/messages request is malformed.

    Adapter-layer error type — the route layer (``routes/anthropic.py``)
    converts this into ``HTTPException(400)``. Kept distinct from a plain
    ``ValueError`` so the route can match on type without sniffing the
    message string, and to make grep-for-callers trivial. Codex review
    flagged the message string as the validation surface; subclassing
    here gives both ergonomic typing AND a stable string identity.
    """


def anthropic_to_openai(request: AnthropicRequest) -> ChatCompletionRequest:
    """
    Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    Handles:
    - system field → system message
    - Content blocks → OpenAI message format
    - tool_use/tool_result → OpenAI tool_calls/tool messages
    - Anthropic tools → OpenAI tools

    Args:
        request: Anthropic Messages API request

    Returns:
        OpenAI ChatCompletionRequest
    """
    messages = []

    # Convert system to system message
    if request.system:
        if isinstance(request.system, str):
            system_text = request.system
        elif isinstance(request.system, list):
            # System can be a list of content blocks
            parts = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n".join(parts)
        else:
            system_text = str(request.system)
        # Strip per-request billing/tracking headers injected by some
        # clients (e.g. Claude Code).  These contain a per-request hash
        # that prevents prefix-cache reuse across turn boundaries.
        system_text = re.sub(r"x-anthropic-billing-header:[^\n]*\n?", "", system_text)
        messages.append(Message(role="system", content=system_text))

    # Convert each message
    for msg in request.messages:
        converted = _convert_message(msg)
        messages.extend(converted)

    # Convert tools
    tools = None
    if request.tools:
        tools = [_convert_tool(t) for t in request.tools]

    # Convert tool_choice
    tool_choice = None
    if request.tool_choice:
        tool_choice = _convert_tool_choice(request.tool_choice)

    # F7: ``tool_choice={"type":"none"}`` means the model must NOT call
    # a tool. Anthropic's real backend strips tool definitions from the
    # prompt entirely when the client sets ``none``; without parity
    # here, the chat template still injects the tools into the system
    # prompt, the model decides to call anyway, and the partial
    # ``<tool_call>{...}`` text leaks through to the response (Sergei
    # repro F7 — leaked text="<tool_call>\n{\"name\": \"get_weather\"...").
    # Drop tools at the adapter so the OpenAI-side request goes
    # downstream with no tool definitions. This is the single source of
    # truth — the downstream chat-route mirror (``routes/chat.py:861``)
    # does the same on ``"none"`` for OpenAI clients, but the Anthropic
    # route never delegates to it. Applied AFTER ``tool_choice`` is
    # converted so the ``"none"`` signal still propagates to the
    # OpenAI-side field for any downstream consumer that branches on
    # it.
    if tool_choice == "none":
        tools = None

    # Translate ``output_config.format = json_schema`` (Anthropic shape,
    # upstream vLLM PR #42396) into the OpenAI ``response_format`` shape
    # the chat-completions guided-decode pipeline already understands.
    # Adapter-layer validation: invalid shapes raise
    # ``AnthropicOutputConfigError``; the route converts that to HTTP 400.
    response_format = _convert_output_config(request.output_config)

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        # Forward None when the Anthropic client omits the field so the
        # server-side sampling cascade (request > CLI > alias overlay >
        # generation_config.json > fallback) can fire. Hard-coding 0.7
        # / 0.9 here would short-circuit the cascade at layer 1 and rob
        # Anthropic-compat clients of the model author's curated defaults.
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream,
        stop=request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        # Pick 1 (this PR) — upstream vLLM PR #20859 + #42396 backport.
        # Translates ``output_config.effort`` (or legacy
        # ``thinking.budget_tokens``) into a per-request reasoning cap
        # on the OpenAI surface.
        reasoning_max_tokens=_resolve_reasoning_max_tokens(request),
    )


def _resolve_reasoning_max_tokens(request: AnthropicRequest) -> int | None:
    """Pick the reasoning cap from the Anthropic-side fields.

    Precedence (first wins):
      1. ``output_config.effort`` — newer Anthropic SDK shape (v0.22,
         upstream vLLM PR #42396). ``max`` and unset both mean "no cap".
      2. ``thinking.budget_tokens`` — legacy v0.20 shape (upstream vLLM
         PR #20859). Verbatim integer budget.
      3. ``None`` — no cap, model decides.

    Returning ``None`` keeps the OpenAI-side request unchanged so the
    existing global ``cfg.thinking_token_budget`` semantic (additive
    max_tokens headroom for reasoning models) keeps applying — these
    two budgets are independent dials.
    """
    if request.output_config is not None and request.output_config.effort is not None:
        # ``max`` → None (no cap) via the canonical mapping; other
        # values resolve to a concrete integer cap.
        return ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS.get(
            request.output_config.effort
        )
    if isinstance(request.thinking, dict):
        budget = request.thinking.get("budget_tokens")
        if isinstance(budget, int) and budget >= 1:
            return budget
    return None


def _sanitize_reasoning_channel(text: str | None) -> str | None:
    """Run the reasoning-channel two-stage sanitizer on ``text``.

    Stage 1 — :func:`strip_reasoning_channel_markup` strips the
    ``<think>`` opener + closer (the canonical reasoning-channel
    parser artifact that the catch-all :func:`sanitize_output`
    intentionally leaves alone — see ``api.utils`` docstring).

    Stage 2 — :func:`sanitize_output` strips the rest of the special-
    token catch-all (``<|im_end|>``, harmony channel markers,
    ``</tool_call>``, …).

    Returns ``None`` when the result is empty / whitespace-only so the
    caller can suppress the surrounding channel emission.
    """
    if not text:
        return None
    stripped = strip_reasoning_channel_markup(text)
    sanitized = sanitize_output(stripped)
    if not sanitized or not sanitized.strip():
        return None
    return sanitized


def _thinking_block_content(
    reasoning_text: str | None,
    text: str | None,
    finish_reason: str | None = None,
) -> str | None:
    """Compute the body of the Anthropic ``thinking`` content block.

    Two systematic invariants on top of the raw ``reasoning_text``:

    1. **Sanitization (R12-M1b fix #1)** — strips the ``<think>`` /
       ``</think>`` tags that the reasoning parser may have left in
       the trace, then runs the canonical :func:`sanitize_output` for
       the rest of the special-token catch-all. Mira r12 R-3 bonus
       regression: at ``max_tokens=1`` on a thinking model, the prompt
       template's pre-injected ``<think>`` is the only token seen by
       the parser and it ended up as the literal ``thinking`` block
       content; routing through the channel-aware sanitizer closes
       that without disturbing the ``content`` channel (where the
       opener can be legit Nemotron prefix injection or literal-tag
       prose).

    2. **Rescue-tail dedupe (R12-M1b fix #2)** — when ``content``
       carries the R12-8 / H-01 rescue payload
       (``sentinel + "\\n\\n" + tail-of-reasoning``), the LAST
       ``RESCUE_TAIL_LENGTH`` chars of the reasoning trace have already
       been published to the user via the ``text`` block. Re-emitting
       the same slice as the suffix of the ``thinking`` block makes
       the same partial sentence render twice (Mira r12 R-3 dupe arm).
       Trim that suffix off the ``thinking`` body so each byte of the
       reasoning trace surfaces on EXACTLY one Anthropic content block.

       The full original ``reasoning_content`` is unchanged on the
       OpenAI-side ``AssistantMessage`` field (the Anthropic envelope
       does not surface it, but the OpenAI-side response — which the
       adapter is being called on — keeps the full trace for
       cross-route observability). PR #802's intentional design
       (rescue surfaces to Anthropic) is preserved: the sentinel +
       tail remain in the ``text`` block; only the duplication is
       fixed.

       Gated on ``finish_reason == "length"`` (codex r4 P3 R12-M1b):
       the R12-8 rescue helper only fires on length-cut truncation
       (see ``_apply_reasoning_cutoff_notice``), so a content payload
       that looks like a rescue payload on a non-length finish (e.g.
       a model legitimately echoing the literal sentinel followed by
       ``\\n\\n`` and more text) is NOT a rescue artifact and the
       dedupe must not fire. ``finish_reason`` defaults to ``None``
       for back-compat with external callers that don't pass the
       field; in that case the dedupe is SKIPPED entirely (the full
       sanitized reasoning trace is preserved in the thinking block).
       This is the safe default: when the caller can't tell us
       whether the truncation was a length-cut, we MUST NOT silently
       drop a legitimate suffix of the reasoning trace. Direct
       callers that intend to trigger the dedupe MUST pass
       ``finish_reason="length"``.

    Returns ``None`` when sanitization + trimming leave nothing —
    callers MUST treat ``None`` as "do not emit a thinking block",
    matching the existing whitespace-only guard.
    """
    if not reasoning_text:
        return None
    # Dedupe rescue tail: strip channel markup FIRST so the trim
    # boundary aligns with the rescue builder's slice boundary (which
    # ALSO strips first — see
    # ``service.helpers._build_reasoning_rescue_payload``). Codex r3
    # P2 (R12-M1b): if we slice the raw reasoning bytes, a
    # ``<think>`` tag that straddles the slice boundary leaves an
    # orphaned ``<th`` / ``ink>`` fragment in the thinking block (the
    # regex stripper no longer matches the orphaned partial). Stripping
    # first means the slice operates on the clean, in-channel byte
    # stream and can never bisect a tag because there are no tags
    # left to bisect.
    #
    # Codex r4 P3 (R12-M1b): gate the dedupe on
    # ``finish_reason == "length"`` AND the rescue-payload shape. The
    # rescue helper only fires on length-cut truncation, so a content
    # body that happens to look like a rescue payload on a clean
    # ``stop`` finish (e.g. a model echoing the literal sentinel
    # followed by ``\n\n`` and more text) is NOT a rescue artifact.
    # The shape gate alone would still false-positive in that case
    # and silently drop a legit thinking block.
    is_length_cut = finish_reason == "length"
    if is_length_cut and is_rescue_payload(text):
        stripped = strip_reasoning_channel_markup(reasoning_text.rstrip())
        prefix = (
            stripped[:-RESCUE_TAIL_LENGTH] if len(stripped) > RESCUE_TAIL_LENGTH else ""
        )
        if not prefix:
            # Entire reasoning trace already surfaces in the rescue
            # ``text`` block — suppress the ``thinking`` block so the
            # tail is not duplicated. The text block alone carries the
            # truncated reasoning excerpt; clients still see the full
            # structural truncation signal via ``stop_reason="max_tokens"``.
            return None
        # Channel markup already stripped above; only the general
        # special-token catch-all remains.
        sanitized = sanitize_output(prefix)
        if not sanitized or not sanitized.strip():
            return None
        return sanitized
    return _sanitize_reasoning_channel(reasoning_text)


def openai_to_anthropic(
    response: ChatCompletionResponse,
    model: str,
    *,
    reasoning_enabled: bool = True,
    matched_stop: str | None = None,
) -> AnthropicResponse:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages API format.

    Args:
        response: OpenAI ChatCompletionResponse
        model: Model name for the response
        reasoning_enabled: Whether the served alias is configured with a
            ``reasoning_parser`` (i.e. structurally capable of producing
            reasoning text). When False, the ``thinking`` block is never
            emitted regardless of what ``reasoning_content`` carries —
            matches Anthropic's public API where non-extended-thinking
            models never emit a ``thinking`` block. Defaults to True so
            external callers that don't pass the flag keep their existing
            behavior (pre-issue #702).
        matched_stop: H-03 — when a user-supplied ``stop_sequences`` entry
            fired, the engine surfaces the matched string via
            ``GenerationOutput.matched_stop``. The route passes it through
            here so the response carries
            ``stop_reason="stop_sequence"`` + ``stop_sequence: <str>`` per
            Anthropic's public spec. ``None`` (the default) means EOS /
            length / no-stop, and the legacy ``_convert_stop_reason``
            mapping (``stop`` → ``end_turn``) applies.

    Returns:
        Anthropic Messages API response
    """
    content = []
    choice = response.choices[0] if response.choices else None

    if choice:
        # Issue #702: emit a ``thinking`` block iff the alias is
        # reasoning-capable AND the reasoning text is genuinely distinct
        # from the answer text.
        #
        # Without this gate, two failure modes leak into Anthropic clients:
        #   (1) An alias with ``reasoning_parser: null`` whose OpenAI-side
        #       response happens to carry ``reasoning_content`` would
        #       still get a ``thinking`` block. Anthropic's public API
        #       never emits one for non-extended-thinking models, so any
        #       client branching on ``content[0].type == "thinking"``
        #       mis-detects capability.
        #   (2) The ``_rescue_silent_drop_from_reasoning`` helper (#569)
        #       deliberately promotes a stuck reasoning trace into
        #       ``content`` so the OpenAI-side message isn't silently
        #       empty. The adapter has no other way to know
        #       ``reasoning_content == content`` is a rescue artifact, so
        #       it would dutifully emit BOTH blocks carrying the same
        #       string — Claude Code / claude-cli / langchain-anthropic
        #       render the same paragraph twice.
        #
        # Both cases collapse to "emit text only" under the same
        # predicate: the reasoning channel must be enabled AND the
        # reasoning bytes must differ from the content bytes (and be
        # non-empty AND non-whitespace). When the predicate fails we
        # still surface the answer as ``text`` — silent drop is the
        # worse failure mode (#569). The whitespace-only guard mirrors
        # ``_rescue_silent_drop_from_reasoning`` which treats
        # ``"   \n"`` as semantically empty — without this gate the
        # adapter would emit a leading ``thinking`` block of pure
        # whitespace that Claude Code surfaces as a blank thought.
        # Codex r1 NIT on PR #705.
        reasoning_text = choice.message.reasoning_content
        text = choice.message.content
        # R12-M1b: route the thinking block content through the
        # canonical sanitizer + rescue-tail dedupe helper. This:
        #   * strips ``<think>`` / ``</think>`` / ``<|...|>`` leaks
        #     that the reasoning parser may have left in the trace
        #     (Mira r12 R-3 ``<think>`` literal at ``max_tokens=1``)
        #   * trims the trailing slice that already surfaces in the
        #     rescue ``text`` block, so the same partial sentence is
        #     never rendered on BOTH content blocks (Mira r12 R-3
        #     duped-rescue-tail arm)
        # Returns ``None`` when sanitization + trimming leave nothing,
        # which signals "do not emit a thinking block" (same shape as
        # the previous whitespace-only guard).
        thinking_body = _thinking_block_content(
            reasoning_text, text, finish_reason=choice.finish_reason
        )
        # Compare BOTH sides post-sanitize so visible-byte equality
        # decides the duplicate-block gate. Codex r1 P2 + r3 BLOCKING
        # on R12-M1b: the prior implementation compared the sanitized
        # ``thinking_body`` against the raw ``text``. That left a
        # second markup-only-delta hole — when ``text`` itself carries
        # removable markup that the canonical content-channel
        # sanitizer collapses to the same visible bytes as
        # ``thinking_body`` (e.g. ``text="done</think>"`` which the
        # Anthropic route's ``sanitize_output`` pass on ``final_content``
        # already strips to ``"done"`` upstream, OR a non-route caller
        # of ``openai_to_anthropic`` that hasn't sanitized ``text``
        # before passing it in), the raw comparison would report
        # "different" and both blocks would render the same VISIBLE
        # bytes to the client. Normalising ``text`` with the canonical
        # content-channel ``sanitize_output`` here makes the gate
        # robust regardless of how much upstream sanitization has
        # already run. The rescue path is unaffected:
        # ``_thinking_block_content`` already trims the rescue-tail
        # suffix from ``thinking_body`` before this gate runs, so the
        # comparison stays True there.
        sanitized_text = sanitize_output(text) if text else text
        emit_thinking = (
            reasoning_enabled
            and thinking_body is not None
            and thinking_body != sanitized_text
        )
        # Add thinking block FIRST so it appears before the answer text,
        # matching Anthropic's extended-thinking SDK convention. Without
        # this block ``<think>...</think>`` reasoning would silently
        # disappear from the non-streaming response — issue #413.
        if emit_thinking:
            content.append(
                AnthropicResponseContentBlock(
                    type="thinking",
                    thinking=thinking_body,
                )
            )

        # Add text content. Emit the SAME ``sanitized_text`` used by
        # the dedupe gate above — keeping a single normalised payload
        # avoids the asymmetry codex r4 flagged: the prior
        # implementation gated on ``sanitize_output(text)`` but emitted
        # raw ``text``, so a non-route caller passing ``text="done</think>"``
        # could still leak ``</think>`` into the Anthropic text block
        # even after the gate correctly suppressed the duplicate
        # thinking block. The in-route path's upstream
        # ``sanitize_output(final_content)`` in ``routes/anthropic.py``
        # remains in place; this is defense-in-depth at the adapter
        # boundary so non-route callers (test helpers, future SSE
        # finalize paths, hypothetical internal routes) also see
        # sanitised bytes on the Anthropic wire. ``sanitized_text`` is
        # ``None`` only when ``text`` itself collapsed to empty after
        # sanitization (all-markup input); in that case the text block
        # is suppressed.
        if sanitized_text:
            content.append(
                AnthropicResponseContentBlock(
                    type="text",
                    text=sanitized_text,
                )
            )

        # Add tool use blocks
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}

                # R6-M2: when the upstream parser is UI-TARS (the only
                # parser whose ``computer`` tool emits the canonical
                # ``point`` / ``start_point`` / ``end_point`` keys) the
                # Anthropic ``tool_use.input`` MUST surface the spec
                # ``coordinate`` / ``start_coordinate`` keys per
                # Anthropic's Computer-Use docs (single-point verbs use
                # ``coordinate``; drag uses ``start_coordinate`` +
                # ``coordinate`` for the end). Anthropic-strict consumers
                # (claude-agent-sdk, Computer-Use harnesses) reject the
                # ``point`` shape. Translation is gated on ``name ==
                # "computer"`` so vanilla function tools whose arguments
                # happen to carry a key named ``point`` are untouched.
                if tc.function.name == "computer" and isinstance(tool_input, dict):
                    from ..tool_parsers.ui_tars_tool_parser import (
                        translate_to_anthropic_spec_keys,
                    )

                    tool_input = translate_to_anthropic_spec_keys(tool_input)

                content.append(
                    AnthropicResponseContentBlock(
                        type="tool_use",
                        # F9: rewrite OpenAI-style ``call_<hex>`` ids to
                        # Anthropic's ``toolu_<hex>`` prefix. See
                        # ``to_anthropic_tool_use_id`` for the prefix
                        # contract; cross-route audit lives in tests
                        # ``test_anthropic_adapter::test_tool_use_id_uses_toolu_prefix``
                        # and the route-level streaming variant.
                        id=to_anthropic_tool_use_id(tc.id),
                        name=tc.function.name,
                        input=tool_input,
                    )
                )

        stop_reason = _convert_stop_reason(choice.finish_reason)
        # H-03: when a user-supplied stop fired, override the generic
        # "stop"→"end_turn" mapping with Anthropic's dedicated
        # ``stop_sequence`` value. Tool/length finishes still win — a
        # tool_calls finish_reason should not be reclassified just
        # because the engine happened to also see a stop string in
        # auxiliary text. Matches Anthropic's public spec where
        # ``stop_sequence`` is mutually exclusive with the other reasons.
        if matched_stop is not None and stop_reason == "end_turn":
            stop_reason = "stop_sequence"
    else:
        stop_reason = "end_turn"

    # If no content blocks, add empty text
    if not content:
        content.append(AnthropicResponseContentBlock(type="text", text=""))

    # Map the OpenAI prefix-cache field onto Anthropic's usage shape.
    # Per Anthropic's prompt-caching docs the three input fields are
    # mutually exclusive and satisfy
    #     total_input_tokens
    #         = input_tokens
    #         + cache_read_input_tokens
    #         + cache_creation_input_tokens
    # so ``input_tokens`` is "the non-cached share", NOT the whole
    # prompt. We only populate ``cache_read_input_tokens`` — the prefix
    # served from the local KV cache — and leave
    # ``cache_creation_input_tokens`` unset: Anthropic's "creation"
    # specifically means tokens being written between explicit
    # ``cache_control`` breakpoints (billed 1.25x), which has no
    # analog on a local engine that auto-caches every prefix without
    # a billing dimension. Cache fields stay ``None`` when the engine
    # didn't report a hit so clients can keep distinguishing "engine
    # doesn't report" from "engine reported a hit".
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    cached_tokens = 0
    if response.usage and response.usage.prompt_tokens_details is not None:
        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
    # Clamp once so cache_read + input_tokens cannot exceed prompt_tokens —
    # a defensive guard against an upstream over-report (e.g. prefix-cache
    # bookkeeping bug) that would otherwise emit an impossible Anthropic
    # usage block where cache_read_input_tokens > total prompt tokens.
    cached_tokens = min(cached_tokens, prompt_tokens)
    cache_read = cached_tokens if cached_tokens else None
    input_tokens = prompt_tokens - cached_tokens
    return AnthropicResponse(
        model=model,
        content=content,
        stop_reason=stop_reason,
        # H-03: only surface ``stop_sequence`` when the matched-stop
        # rewrite actually took effect (``stop_reason == "stop_sequence"``).
        # Carrying the matched bytes alongside an ``end_turn`` /
        # ``max_tokens`` / ``tool_use`` reason would violate Anthropic's
        # spec ("stop_sequence is set iff stop_reason == 'stop_sequence'").
        stop_sequence=matched_stop if stop_reason == "stop_sequence" else None,
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
    )


def _convert_message(msg: AnthropicMessage) -> list[Message]:
    """
    Convert an Anthropic message to one or more OpenAI messages.

    Anthropic tool_result blocks (sent as user messages) need to be
    split into separate OpenAI tool messages.

    Args:
        msg: Anthropic message

    Returns:
        List of OpenAI messages
    """
    # Simple string content
    if isinstance(msg.content, str):
        return [Message(role=msg.role, content=msg.content)]

    # Content is a list of blocks
    messages = []
    text_parts = []
    tool_calls_for_assistant = []
    tool_results = []

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text or "")

        elif block.type == "tool_use":
            # Assistant message with tool calls
            tool_input = block.input or {}
            tool_calls_for_assistant.append(
                {
                    "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": json.dumps(tool_input),
                    },
                }
            )

        elif block.type == "tool_result":
            # Tool result → OpenAI tool message
            result_content = block.content
            if isinstance(result_content, list):
                # Extract text from content blocks
                parts = []
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                result_content = "\n".join(parts)
            elif result_content is None:
                result_content = ""

            tool_results.append(
                Message(
                    role="tool",
                    content=str(result_content),
                    tool_call_id=block.tool_use_id or "",
                )
            )

    # Build the messages
    if msg.role == "assistant":
        combined_text = "\n".join(text_parts) if text_parts else None
        if tool_calls_for_assistant:
            messages.append(
                Message(
                    role="assistant",
                    content=combined_text or "",
                    tool_calls=tool_calls_for_assistant,
                )
            )
        elif combined_text is not None:
            messages.append(Message(role="assistant", content=combined_text))
        else:
            messages.append(Message(role="assistant", content=""))
    elif msg.role == "user":
        # User messages: collect text parts, then add tool results separately
        if text_parts:
            combined_text = "\n".join(text_parts)
            messages.append(Message(role="user", content=combined_text))

        # Tool results become separate tool messages
        messages.extend(tool_results)

        # If no text and no tool results, add empty user message
        if not text_parts and not tool_results:
            messages.append(Message(role="user", content=""))
    else:
        # Other roles
        combined_text = "\n".join(text_parts) if text_parts else ""
        messages.append(Message(role=msg.role, content=combined_text))

    return messages


def _convert_tool(tool: AnthropicToolDef) -> ToolDefinition:
    """
    Convert an Anthropic tool definition to OpenAI format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    return ToolDefinition(
        type="function",
        function={
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    )


def _convert_tool_choice(tool_choice: dict) -> str | dict | None:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
    OpenAI: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
    """
    choice_type = tool_choice.get("type", "auto")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    elif choice_type == "none":
        return "none"

    return "auto"


def _convert_output_config(
    output_config: AnthropicOutputConfig | None,
) -> ResponseFormat | None:
    """Translate Anthropic ``output_config`` → OpenAI ``response_format``.

    Backport of upstream vLLM PR #42396. Only ``format.type == "json_schema"``
    is supported on this surface today; downstream of this call the existing
    chat-completions guided-decode pipeline (``api/guided.py`` + outlines)
    runs unchanged.

    ``output_config.effort`` is intentionally NOT translated here — see the
    docstring on ``AnthropicOutputConfig``. The field is accepted by the
    Pydantic model but Pick 1 (a concurrent PR) owns wiring it through.

    Raises:
        AnthropicOutputConfigError: when ``format.type`` is not
            ``"json_schema"`` or when the ``schema`` field is missing /
            not a JSON object. The route layer converts this to HTTP 400.
    """
    if output_config is None or output_config.format is None:
        return None

    fmt = output_config.format
    fmt_type = fmt.type
    if fmt_type != "json_schema":
        # Mirror the message style of routes/chat.py's 400 responses so
        # error strings on the two surfaces look like siblings.
        raise AnthropicOutputConfigError(
            f"output_config.format.type={fmt_type!r} is not supported on "
            "/v1/messages; only 'json_schema' is accepted. See upstream "
            "vLLM PR #42396 for the backport contract."
        )

    schema = fmt.schema_
    if schema is None:
        raise AnthropicOutputConfigError(
            "output_config.format.schema is required when "
            "output_config.format.type == 'json_schema' on /v1/messages."
        )
    if not isinstance(schema, dict):
        # Pydantic would have already coerced strings/lists away here for
        # the dict-typed field, but guard explicitly so the message stays
        # informative if a future schema type widens.
        raise AnthropicOutputConfigError(
            "output_config.format.schema must be a JSON object "
            f"(got {type(schema).__name__})."
        )

    # ResponseFormatJsonSchema requires ``name`` — default to "response"
    # to match the existing OpenAI surface's behavior when the field is
    # absent (see api/tool_calling.build_json_system_prompt fallback).
    return ResponseFormat(
        type="json_schema",
        json_schema=ResponseFormatJsonSchema(
            name=fmt.name or "response",
            description=fmt.description,
            schema=schema,
            strict=fmt.strict if fmt.strict is not None else False,
        ),
    )


def _convert_stop_reason(openai_reason: str | None) -> str:
    """
    Convert OpenAI finish_reason to Anthropic stop_reason.

    OpenAI: "stop" | "tool_calls" | "length" | "content_filter"
    Anthropic: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
    """
    if openai_reason is None:
        return "end_turn"

    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason, "end_turn")
