# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for route handlers.

These functions were extracted from server.py to enable route modules
(chat, completions, anthropic) to share common logic without importing
from the monolithic server module.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException
from starlette.requests import Request

from ..api.models import (
    CompletionTokensDetails,
    FunctionCall,
    PromptTokensDetails,
    TokenLogProb,
    ToolCall,
    TopLogProb,
    Usage,
)
from ..api.tool_calling import parse_tool_calls
from ..config import get_config
from ..engine import BaseEngine, GenerationOutput
from ..tool_parsers import ToolParserManager

logger = logging.getLogger(__name__)

# ── Fallback defaults ──────────────────────────────────────────────
_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


def _check_admission_or_503(engine) -> None:
    """Atomic admission gate for route handlers — reserves a slot.

    Calls ``engine.check_admission()`` which, under
    ``_admission_lock``, checks the cap and increments the engine's
    reservation counter. If the cap is reached, raises HTTP 503 with
    Retry-After before any response body is sent. This is necessary
    for streaming routes — once ``StreamingResponse`` starts yielding,
    headers are flushed and the only way to signal backpressure would
    be an SSE error chunk on a 200 response.

    The reservation is released by ``_disconnect_guard`` (streaming)
    or ``_wait_with_disconnect`` (non-streaming) via their ``finally``
    clauses when the caller passes ``engine=engine`` to them. Routes
    that bypass both (e.g. the chat ``want_logprobs`` branch) must
    call ``engine.release_admission_reservation()`` themselves.

    Engines without a ``check_admission`` attribute (test stubs)
    silently no-op.
    """
    from ..scheduler import BackpressureError

    check = getattr(engine, "check_admission", None)
    if check is None:
        # Engine doesn't implement admission control (e.g. test stub) —
        # fall through to the runtime catch in ``_wait_with_disconnect``.
        return
    try:
        check()
    except BackpressureError as exc:
        _raise_backpressure_503(exc)


def _release_admission_unless_committed(engine, committed: bool) -> None:
    """Release a slot reserved by ``_check_admission_or_503`` unless
    release responsibility has been handed off to a streaming helper.

    Pair with a route-handler ``try/finally``: set a local
    ``_admission_committed_to_helper = False`` right after the
    reservation, flip to ``True`` immediately before returning a
    ``StreamingResponse(_disconnect_guard(..., engine=engine))``
    (the helper releases when the SSE generator closes), and call
    this from the ``finally``. Closes the codex R3 leak — validation
    errors (``messages=[]``, invalid ``max_tokens``, unsupported
    image-on-text, ``response_format`` schema errors,
    chat-template errors, …) that previously pinned a slot until
    restart now drop the slot via this finally.

    ``release_admission_reservation`` is idempotent below zero so a
    stray double release (defensive callers, helper fires just as
    the route handler also releases) cannot corrupt the accounting.
    """
    if committed:
        return
    release = getattr(engine, "release_admission_reservation", None)
    if release is None:
        return
    try:
        release()
    except Exception:
        logger.warning(
            "release_admission_reservation raised on route finally",
            exc_info=True,
        )


def _raise_backpressure_503(exc: Exception) -> None:
    """Convert ``BackpressureError`` from the scheduler into HTTP 503
    with a Retry-After header (RFC 9110 §10.2.4).

    Backpressure is a normal load-shedding outcome, not a bug — clients
    that respect Retry-After can simply re-queue. Without this catch,
    the error reaches FastAPI's generic 500 handler and the client
    sees an opaque ``Internal server error`` body, defeating the
    point of admission control.
    """
    raise HTTPException(
        status_code=503,
        # 1s is a sensible default — the cap usually clears within
        # a few tokens of decode on the saturated batch.
        headers={"Retry-After": "1"},
        detail=(
            "Server is busy (max concurrent requests reached). "
            f"Retry after the Retry-After delay. ({exc})"
        ),
    )


def _finalize_content_and_reasoning(
    raw_text: str,
    cleaned_text: str,
    tool_calls: list,
    reasoning_parser,
    engine_reasoning_text: str = "",
    enable_thinking: bool | None = None,
    reasoning_max_tokens: int | None = None,
) -> tuple[str, str | None]:
    """Compute final ``content`` + ``reasoning_text`` after tool parsing.

    Shared between the OpenAI ``/v1/chat/completions`` and Anthropic
    ``/v1/messages`` non-streaming paths so both surfaces extract
    reasoning identically — bypassing this on one route was the
    silent-divergence bug filed as issue #413.

    Rule (drives the unclosed-`<tool_call>` leak fix in PR #208): when
    the tool parser successfully extracted ``tool_calls`` its
    ``cleaned_text`` is authoritative — both ``<think>`` and tool tags
    are already stripped. Run the reasoning parser on the raw output
    only to recover ``reasoning_text``, never to overwrite
    ``cleaned_text`` (that path would re-introduce the tool tags the
    parser stripped, since the reasoning parser only knows about
    ``<think>``).

    When no tool_calls fire, the reasoning parser is the only thing
    that can pull ``<think>`` out — run it on cleaned_text (or raw
    output if cleaning produced an empty string). If the parser
    returns ``(None, None)`` it means the input has no reasoning
    markers it understands — keep the original ``cleaned_text``
    instead of clobbering it with ``None``. This is critical for
    harmony models, where ``clean_output_text`` (called in
    ``engine.generate``) has already extracted the final-channel
    content and stripped channel markup before the parser ever sees
    the text: a ``HarmonyReasoningParser`` searching for
    ``<|channel|>final`` on the already-cleaned string returns
    ``(None, None)`` and without this guard would silently turn a
    fully-formed answer into an empty ``TextBlock`` for clients
    (anthropic_sdk / langchain / pydantic_ai non-streaming
    integrations, v0.6.64 pr_validate baseline).
    """
    reasoning_text = None
    # Engine-level token routing is authoritative when present. The
    # ``OutputRouter`` state machine tracks channel boundaries at the
    # token level (same code path the streaming route already trusts),
    # so the text-based retry below is redundant — and would in fact
    # be wrong for truncated harmony output, where the engine cleaner
    # leaks analysis content into ``cleaned_text`` and the parser's
    # regex misses it without an ``<|end|>`` terminator. When the
    # engine populated ``reasoning_text``, use it directly and skip
    # the parser. (No reasoning_parser is still a short-circuit
    # below.) Issue #442.
    if engine_reasoning_text:
        # 2026-06-17 VibeThinker live test: when the engine routes via
        # ``OutputRouter`` (Qwen ``<think>`` token IDs) and the response
        # is truncated mid-thought (``finish_reason=length``), the
        # router emits ``reasoning_text`` (the post-``<think>`` trace)
        # but ``cleaned_text`` — passed through from
        # ``clean_output_text`` which preserves ``<think>`` blocks for
        # the parser stack — still carries the full raw text including
        # the unclosed ``<think>`` opener and the trace. The result is
        # ``content`` and ``reasoning_content`` carrying the same
        # bytes (the live-test math row showed content_len=4974,
        # reasoning_content_len=4967, identical except for the
        # ``<think>`` opener).
        #
        # ``strip_thinking_tags`` (the downstream sanitiser) only
        # matches **closed** ``<think>…</think>`` blocks, so the
        # unclosed opener falls through. Trim everything from the
        # ``<think>`` opener onward — preserves any pre-think preamble
        # ("Okay, let me think...\n<think>...") as legitimate
        # ``content`` while dropping the leaked thought trace.
        # Codex r1 P2: the previous ``startswith`` check missed the
        # documented VibeThinker preamble shape (model emits a chatty
        # intro BEFORE ``<think>``, then truncates mid-thought) —
        # ``partition`` handles both the start-aligned (math row) and
        # preamble (live-test merge_intervals streaming) cases.
        truncated_think = (
            cleaned_text
            and "<think>" in cleaned_text
            and "</think>" not in cleaned_text
        )
        if truncated_think:
            cleaned_text = cleaned_text.partition("<think>")[0].rstrip()
            # Codex r3 P2: ``_apply_reasoning_cap`` prepends the
            # over-cap reasoning suffix back into ``cleaned_text`` so
            # the wire ordering matches the model's emission order —
            # but for a truncated thought the overflow IS the leaked
            # thought, which is exactly what we just trimmed. Use the
            # reasoning-only cap so cleaned_text stays blanked / preamble-
            # only and the overflow does NOT re-leak into ``content``.
            return cleaned_text, _truncate_reasoning_only(
                engine_reasoning_text, reasoning_max_tokens
            )
        # F-041 (2026-06-19): the ``cleaned_text``-gated check above misses
        # the case where the OutputRouter consumed the structural
        # ``<think>`` token before it ever reached ``cleaned_text`` — the
        # router emits ``content=None`` AND sets ``text=""`` (engine
        # ``_route_tokens_for_channels`` lines 1588-1589), so the engine-
        # routed branch falls through to ``_apply_reasoning_cap`` with
        # ``cleaned_text=""``. With ``reasoning_max_tokens`` set, the cap
        # then prepends the over-cap reasoning suffix back into
        # ``cleaned_text`` (the empty-content fallback), shipping the
        # truncated-thought trace into ``content``. Mirror the
        # cleaned-text-gated truncated_think plug: when the engine routed
        # reasoning AND ``raw_text`` shows an unclosed ``<think>`` (model
        # was still mid-thought at ``finish_reason=length``), use the
        # reasoning-only cap so the over-cap suffix is dropped rather
        # than leaking into the user-visible answer channel.
        if (
            raw_text
            and "<think>" in raw_text
            and "</think>" not in raw_text
            and not (cleaned_text and cleaned_text.strip())
        ):
            return cleaned_text or "", _truncate_reasoning_only(
                engine_reasoning_text, reasoning_max_tokens
            )
        return _apply_reasoning_cap(
            cleaned_text,
            engine_reasoning_text,
            reasoning_max_tokens,
            has_tool_calls=bool(tool_calls),
        )
    if reasoning_parser is None:
        return _apply_reasoning_cap(
            cleaned_text,
            reasoning_text,
            reasoning_max_tokens,
            has_tool_calls=bool(tool_calls),
        )
    # #575 — thread the request-level ``enable_thinking`` so the
    # underlying ``BaseThinkingReasoningParser.extract_reasoning``
    # can apply its symmetric-with-streaming Case-4 fallback when
    # the chat template pre-injected ``<think>`` and the model was
    # truncated mid-thought (``finish_reason="length"`` with no
    # ``</think>`` ever emitted). Older / third-party reasoning
    # parsers that don't accept the kwarg fall back to a 1-arg call
    # so we don't break their contract — detected via
    # ``inspect.signature`` (no side-effecting probe call, codex
    # R1 NIT: an ``extract("")`` probe could hide a real ``TypeError``
    # raised inside the parser body OR trigger third-party parser
    # side effects on the empty-string input).
    if _parser_accepts_enable_thinking(reasoning_parser):
        extract = lambda text: reasoning_parser.extract_reasoning(
            text, enable_thinking=enable_thinking
        )
    else:
        extract = lambda text: reasoning_parser.extract_reasoning(text)
    if tool_calls:
        reasoning_text, _ = extract(raw_text)
    else:
        text_to_parse = cleaned_text or raw_text
        new_reasoning, new_cleaned = extract(text_to_parse)
        # Capture the FIRST-parse Case-4 signal BEFORE the harmony
        # retry overwrites ``new_reasoning``. The leak plug below
        # MUST gate on what the parser routed when it saw the
        # already-cleaned text, not on what the retry-on-raw-text
        # later produced — otherwise harmony's analysis-channel
        # recovery looks like a Case-4 fallback to the guard and
        # spuriously clears legitimate final-channel content
        # (codex R2 BLOCKING). The signal is: first parse routed
        # the whole input to reasoning AND the input had no
        # ``<think>`` tags (so it really was the no-tag Case-4
        # fallback firing, not Case 3's ``…</think>answer`` split
        # nor a harmony channel-strip outcome).
        first_parse_was_case4 = (
            new_reasoning is not None
            and new_cleaned is None
            and bool(text_to_parse)
            and "<think>" not in text_to_parse
            and "</think>" not in text_to_parse
        )
        # 2026-06-17 VibeThinker live test: Case-3 (truncated
        # ``<think>`` with no ``</think>``) leaks identically to Case-4
        # but the #575 plug above doesn't catch it because ``<think>``
        # IS in ``text_to_parse``. Parser returned ``(reasoning, None)``
        # — i.e. the reasoning parser found a ``<think>`` opener and
        # routed everything after it into reasoning — but the original
        # ``cleaned_text`` still carries the full raw text including
        # ``<think>…<the trace>``. ``strip_thinking_tags`` (the
        # downstream sanitiser) only matches CLOSED ``<think>…</think>``
        # blocks, so a truncated ``finish_reason=length`` response
        # with an unclosed ``<think>`` opener falls straight through
        # and the client sees identical bytes in ``content`` and
        # ``reasoning_content`` (live-test math row: content_len ==
        # reasoning_len == 5449, byte-identical).
        #
        # Signal: parser returned reasoning-only AND ``text_to_parse``
        # contains an unclosed ``<think>`` opener (so the parser's
        # ``(reasoning, None)`` was Case-3, not Case-4, not the
        # harmony-style ``(None, None)`` rescued by the retry above).
        # Unlike the #575 plug, this branch is NOT gated on
        # ``enable_thinking`` — the literal ``<think>`` token in the
        # output is the model's own evidence that thinking was active
        # for this turn, irrespective of what the caller passed.
        #
        # Codex r1 P2: the previous ``lstrip().startswith("<think>")``
        # gate missed the documented VibeThinker preamble shape (the
        # model emits a chatty intro BEFORE ``<think>``, then truncates
        # mid-thought). The fix below uses ``partition("<think>")[0]``
        # so a preamble like ``"Okay, let me think...\n<think>..."``
        # has the preamble preserved as ``content`` while the unclosed
        # thought trace is dropped (the trace is already carried in
        # ``reasoning_text``). Catches BOTH the live-test math row
        # (``<think>`` at lstrip-start) and the merge_intervals
        # streaming row (preamble before ``<think>``).
        first_parse_was_truncated_think = (
            new_reasoning is not None
            and new_cleaned is None
            and bool(text_to_parse)
            and "<think>" in text_to_parse
            and "</think>" not in text_to_parse
        )
        # Harmony retry: the engine's ``clean_output_text`` strips
        # ``<|channel|>analysis<|message|>…`` markers before the route
        # ever sees the output, so a ``HarmonyReasoningParser`` running
        # on ``cleaned_text`` finds no channels and returns ``None``.
        # When the engine populated ``raw_text`` with the pre-clean
        # output, re-run the parser on it to recover the analysis-channel
        # content. Only triggers when (a) first parse found no reasoning
        # AND (b) raw_text actually differs from the text we just parsed
        # — non-harmony parsers (``<think>``) are unaffected because
        # their first parse succeeds on cleaned_text. (Counterpart to
        # PR #436's empty-TextBlock fix: that PR rescued ``content``
        # from being clobbered to None; this rescues ``reasoning``.)
        if new_reasoning is None and raw_text and raw_text != text_to_parse:
            retry_reasoning, _ = extract(raw_text)
            if retry_reasoning is not None:
                new_reasoning = retry_reasoning
        reasoning_text = new_reasoning
        # Only overwrite cleaned_text when the parser explicitly
        # produced new content. ``new_cleaned is None`` means the
        # parser had nothing concrete to say about content — either
        # it found no markers at all (harmony pre-cleaned case) or
        # it found only reasoning (qwen3 ``<think>``-only case). In
        # both cases the original cleaned_text is the right thing to
        # keep; downstream ``strip_thinking_tags`` + sanitization
        # will collapse think-only inputs to empty further along the
        # pipeline. (Originally widened with ``or new_reasoning is
        # not None`` after the harmony empty-TextBlock fix, but
        # DeepSeek review on PR #436 pointed out that branch still
        # clobbered cleaned_text whenever the parser returned
        # ``(reasoning, None)`` — same regression by a different
        # route.)
        if new_cleaned is not None:
            cleaned_text = new_cleaned
        # #575 leak-plug: when ``enable_thinking=True`` AND the
        # parser's FIRST parse routed the whole no-tag output to
        # reasoning (Case-4 fallback path), the original
        # ``cleaned_text`` is the same raw thought trace —
        # ``strip_thinking_tags`` only matches **closed**
        # ``<think>…</think>`` blocks, so a no-tag truncated thought
        # would pass straight through to ``final_content`` and the
        # client would see the exact same prose in BOTH
        # ``reasoning_content`` AND ``content`` (codex R1 BLOCKING
        # finding). Clear ``cleaned_text`` so the route renders
        # ``content=None`` for that case. Streaming-symmetry
        # invariant — the streaming Case-3 path never emits content
        # for truncated thoughts either. Note the gate on the
        # FIRST-parse outcome rather than the post-retry value: a
        # harmony reasoning-from-raw-text retry can produce the
        # same ``(reasoning, None)`` shape but the cleaned_text in
        # that case is the legitimate final-channel answer that
        # MUST survive (codex R2 BLOCKING).
        if enable_thinking is True and first_parse_was_case4:
            cleaned_text = ""
            # F-041 (2026-06-19): same rationale as the codex r3 P2 plug
            # below for ``first_parse_was_truncated_think`` — when the
            # chat template pre-injected ``<think>`` and the model was
            # truncated mid-thought emitting NO tags at all, the
            # accumulated text IS the thought trace. Letting it fall
            # through to ``_apply_reasoning_cap`` would prepend the
            # over-cap reasoning suffix back into the (now-blank)
            # ``cleaned_text`` and ship the leaked thought trace as
            # ``content``. Live VibeThinker repro at
            # ``reasoning_max_tokens=30`` (max_tokens=200, finish=length):
            # the Case-4 fallback routed the no-tag output to reasoning,
            # the cap truncated to 120 chars, and the remaining
            # ~500 chars of thought trace surfaced verbatim as
            # ``content`` — the leak shape F-041 was filed against.
            # Use the reasoning-only cap so the overflow is dropped.
            return cleaned_text, _truncate_reasoning_only(
                reasoning_text, reasoning_max_tokens
            )
        # Truncated-``<think>`` plug (2026-06-17). Mirrors the #575
        # Case-4 plug above but fires on the explicit-start-no-end
        # signal independent of ``enable_thinking`` — see the
        # ``first_parse_was_truncated_think`` definition for the
        # full rationale and the live-test repro.
        #
        # ``partition`` keeps any pre-think preamble (legitimate
        # content) and drops the unclosed thought trace (already
        # carried in ``reasoning_text``). For the live-test math
        # row the preamble is empty so this collapses to ``""``;
        # for the merge_intervals streaming row it preserves the
        # ~80-char chatty intro the model emitted before ``<think>``.
        if first_parse_was_truncated_think:
            cleaned_text = (cleaned_text or "").partition("<think>")[0].rstrip()
            # Codex r3 P2: bypass the cleaned_text overflow prepend
            # path of ``_apply_reasoning_cap`` for truncated thoughts
            # — see the engine-routed branch above for the rationale.
            return cleaned_text, _truncate_reasoning_only(
                reasoning_text, reasoning_max_tokens
            )
    return _apply_reasoning_cap(
        cleaned_text,
        reasoning_text,
        reasoning_max_tokens,
        has_tool_calls=bool(tool_calls),
    )


def _truncate_reasoning_only(
    reasoning_text: str | None,
    reasoning_max_tokens: int | None,
) -> str | None:
    """Cap ``reasoning_text`` to the per-request budget WITHOUT
    rerouting the overflow into ``content``.

    Used by the truncated-``<think>`` plug paths
    (``first_parse_was_truncated_think`` and the engine-routed
    branch) where the reasoning trace is an in-progress thought,
    not the final answer. ``_apply_reasoning_cap``'s default
    behaviour of prepending overflow into ``cleaned_text`` would
    re-introduce exactly the leak the plug is trying to prevent —
    codex r3 P2.

    Uses the same chars-÷4 heuristic as ``_apply_reasoning_cap``
    so the OpenAI usage block stays consistent across both paths.
    """
    if (
        reasoning_max_tokens is None
        or not reasoning_text
        or not isinstance(reasoning_text, str)
    ):
        return reasoning_text
    max_chars = reasoning_max_tokens * 4
    if len(reasoning_text) <= max_chars:
        return reasoning_text
    return reasoning_text[:max_chars]


def _apply_reasoning_cap(
    cleaned_text: str,
    reasoning_text: str | None,
    reasoning_max_tokens: int | None,
    *,
    has_tool_calls: bool = False,
) -> tuple[str, str | None]:
    """Truncate ``reasoning_text`` to the per-request cap and reroute
    the overflow into ``cleaned_text`` (upstream vLLM PR #20859
    backport).

    Non-stream equivalent of
    ``StreamingPostProcessor._consume_reasoning_budget``:
    ``None`` short-circuits to a no-op so back-compat callers behave
    exactly as before. Uses the same chars-÷4 heuristic as
    ``_build_usage`` and the streaming postprocessor — single source
    of truth for "how many tokens does this text approximate" so the
    OpenAI usage block, the streaming SSE deltas, and this non-stream
    finalize all agree on a single token count.

    F-041 (2026-06-19): the overflow-into-``cleaned_text`` reroute is
    only meaningful when there is NO real visible payload — i.e. the
    parser found no closed ``</think>`` AND no tool calls fired, so
    the model never produced a user-visible answer (we'd be silently
    dropping the whole response otherwise). When the response DOES
    have a real payload (closed ``<think>…</think>answer`` split, OR
    structured ``tool_calls`` — codex r1 follow-up: tool-only responses
    legitimately ship ``content=""`` per the OpenAI spec, so an empty
    ``cleaned_text`` alone isn't proof that the response is empty),
    prepending the over-cap reasoning bytes pollutes the visible
    payload with the truncated thought trace. The vibethinker repro at
    ``reasoning_max_tokens=30`` shipped the entire post-cap reasoning
    suffix + the model's training-time system prompt into ``content``
    BEFORE the actual answer ``"The capital of Japan is **Tokyo**."``
    — exactly the leak shape PR #722 closed for the multi-block
    ``<think>`` case. The user opting into a small reasoning cap
    explicitly asked us to drop the reasoning past the cap; they did
    not ask us to reclassify those bytes as content. Drop the
    overflow when a real payload exists; preserve the prepend-into-
    content fallback only when both ``cleaned_text`` is empty AND no
    tool calls fired (the model emitted nothing visible).
    """
    if (
        reasoning_max_tokens is None
        or not reasoning_text
        or not isinstance(reasoning_text, str)
    ):
        return cleaned_text, reasoning_text
    max_chars = reasoning_max_tokens * 4
    if len(reasoning_text) <= max_chars:
        return cleaned_text, reasoning_text
    overflow = reasoning_text[max_chars:]
    truncated = reasoning_text[:max_chars]
    # F-041 plug: when the response already carries a real visible
    # payload (parser routed the post-``</think>`` final content into
    # ``cleaned_text``, OR the tool parser surfaced structured
    # ``tool_calls`` — the OpenAI-compat ``tool_choice`` paths
    # legitimately ship ``content=""`` alongside ``tool_calls``),
    # the model gave us its visible answer — the user-requested cap
    # is the contract, not a "best-effort don't-drop-bytes" hint, so
    # drop the over-cap reasoning suffix rather than letting it bleed
    # into ``content`` ahead of the answer / alongside the tool call.
    has_visible_content = bool(cleaned_text and cleaned_text.strip())
    if has_visible_content or has_tool_calls:
        return cleaned_text, truncated
    # Codex round-11 BLOCKING: prepend overflow rather than appending
    # it. In the source ordering, the overflow bytes were emitted by
    # the model BEFORE any post-``</think>`` final content. Appending
    # ``cleaned_text + overflow`` would reorder the response as
    # ``final-answer + dropped-reasoning``, which:
    #   1. mis-represents the model's actual token order on the wire,
    #   2. breaks the streaming-vs-non-streaming parity (the streaming
    #      pipeline emits overflow on the cap-crossing chunk, BEFORE
    #      any subsequent content delta — same as putting overflow
    #      first here),
    #   3. confuses downstream consumers that pattern-match the start
    #      of the response (e.g. JSON-schema validators that scan for
    #      the opening ``{``).
    # Prepend so the time-ordered emission is preserved.
    cleaned_text = overflow + (cleaned_text or "")
    return cleaned_text, truncated


def _rescue_silent_drop_from_reasoning(
    final_content: str | None,
    reasoning_text: str | None,
    tool_calls: list | None,
    finish_reason: str | None = None,
    raw_text: str | None = None,
    *,
    reasoning_is_case4: bool = False,
) -> str | None:
    """Issue #569: never silently drop an assistant turn.

    The route layer's normal ``content`` extraction can legitimately
    produce an empty ``final_content`` when the model emits ONLY
    reasoning tokens and never closes the reasoning channel into a
    ``content``/``final`` channel or a tool call. The exact production
    failure mode: ``gemma-4-26b-4bit`` multi-turn tool flows where the
    model gets stuck inside ``<|channel>thought\\n...`` and runs out of
    its token budget before emitting any ``<|tool_call>`` or
    ``<|channel>content`` marker. The engine's token-level
    ``OutputRouter`` correctly routes every token to ``reasoning`` —
    but the route then emits an OpenAI-compat message with
    ``content=null`` and ``tool_calls=null`` while
    ``reasoning_content`` carries the entire stuck thought. Agentic
    clients (Cline, Cursor, Codex CLI) read ``content`` and
    ``tool_calls`` only, see an empty message, and either retry into
    the same trap or stall.

    Rescue rule: when ``final_content`` is empty/None AND no
    ``tool_calls`` fired AND ``reasoning_text`` is non-empty, surface
    ``reasoning_text`` as ``content``. ``reasoning_content`` stays
    populated unchanged — duplication between the two fields is the
    lesser evil vs. a silently empty response.

    Cases that fall through unchanged:

    * Happy path: ``final_content`` is non-empty AND has at least one
      non-whitespace char → return as-is. Whitespace-only
      ``final_content`` (``"   \n"``) is treated as semantically
      absent for rescue purposes (codex round-3 NIT on #676): an
      OpenAI-compat client still sees an empty assistant turn, so
      the rescue must be allowed to fire when reasoning is present.
      The strip is on the predicate only — the original
      ``final_content`` propagates back on the happy path so callers
      that DO want the whitespace preserved still see it as-is.
    * Tool-call path: ``tool_calls`` non-empty → the spec already
      requires ``content`` to be ``None`` (the tool call IS the
      response); rescue does NOT fire.
    * Truly empty: ``reasoning_text`` empty OR whitespace-only →
      nothing semantically rescue-worthy; ``None`` propagates. The
      whitespace-only check (codex round-1 NIT on #676) closes a
      gap where ``"   \n"`` would surface as non-empty ``content``
      while still being semantically empty to clients. The
      ORIGINAL ``reasoning_text`` is returned untouched (no
      ``.strip()`` on the assignment) so callers that DO want the
      whitespace preserved still see it as-is — the strip is on
      the predicate only.

    The rescue lives at the route layer (not the engine) because the
    engine's ``_route_tokens_for_channels`` has a tested contract
    (issue #442's harmony fix pins ``content == ""`` when only the
    analysis channel fires) — flipping that at the engine level
    would re-leak analysis text into ``content`` for the original
    #442 case. The rescue runs AFTER tool-call parsing and AFTER the
    reasoning/content split, as a final route-level safety net so
    silent drops never escape to clients regardless of which model
    family produced them.

    Codex round-3 BLOCKING on #676: this helper is now the SINGLE
    predicate for both the non-streaming AND streaming rescue paths
    (chat.py:~1285 and chat.py:~1605). The streaming path used to
    promote ``processor.accumulated_reasoning`` directly into
    ``delta.content`` without the whitespace guard, so a
    reasoning-only stream of ``"   \n"`` would emit a semantically
    empty ``delta.content`` while non-streaming correctly suppressed
    it. Routing both call sites through this helper closes that
    asymmetry — the predicate cannot drift between the two paths
    because there's only one of it.
    """
    if final_content and final_content.strip():
        return final_content
    if tool_calls:
        return final_content
    if not reasoning_text or not reasoning_text.strip():
        return final_content
    # 2026-06-17 VibeThinker live test: when the model was truncated
    # mid-thought (``finish_reason="length"``) with an unclosed
    # ``<think>`` opener in ``raw_text``, the reasoning trace is NOT
    # the final answer — it's an interrupted chain of thought. Surfacing
    # it as ``content`` per the #569 rescue would feed the client the
    # SAME bytes as ``reasoning_content`` and break the "content is the
    # final answer" contract. Skip the rescue and let the client see
    # ``content=null`` so they can detect "model ran out of budget
    # before producing an answer" via the ``finish_reason="length"``
    # signal — symmetric with how OpenAI's o1 / o3 behave on truncated
    # reasoning.
    #
    # Gate on BOTH ``finish_reason="length"`` AND raw_text opening with
    # an unclosed ``<think>``. Other ``finish_reason="length"`` cases
    # (e.g. a non-thinking model truncated mid-answer where reasoning
    # was empty but content was building) still get rescued — the
    # opener check is the discriminator.
    #
    # Also gate on the helper-Case-4 signal (``reasoning_is_case4``)
    # passed from the route — covers the PR #715-bundle live-test
    # repro where VibeThinker is asked a no-tool no-think prompt:
    # the chat template doesn't pre-inject ``<think>``, the model
    # answers in plain prose (no ``<think>`` token emitted), but the
    # route still defaults ``enable_thinking=True`` for the family.
    # The parser's Case-4 fallback routes the WHOLE output to
    # reasoning AND the helper blanks ``cleaned_text=""``. The
    # ``raw_text`` opener check then misses (raw_text doesn't start
    # with ``<think>``), and the rescue without the Case-4 signal
    # mistakes the no-content state for a #569 silent drop and
    # surfaces the reasoning as content — duplicating the trace
    # byte-identically. The Case-4 signal stops that.
    if (
        finish_reason == "length"
        and raw_text
        and raw_text.lstrip().startswith("<think>")
        and "</think>" not in raw_text
    ):
        return final_content
    if finish_reason == "length" and reasoning_is_case4:
        return final_content
    return reasoning_text


# OpenAI-spec closed enum for ``response_format.type``. Any value outside
# this set used to be silently accepted (defaulted to "text" by
# ``build_json_system_prompt``) so a client typo like ``"xml"`` or an
# empty string returned HTTP 200 with no structure enforcement — the
# client received plain prose instead of the JSON they asked for and had
# no signal anything was wrong (F-013 silent-accept arm). The validator
# below pins the enum + the ``json_schema``-requires-schema invariant so
# malformed requests get a clean 400 BEFORE
# ``build_json_system_prompt`` is reached (which used to leak the raw
# Python ``AttributeError: 'NoneType' object has no attribute 'get'`` in
# the 400 body — F-013 raw-leak arm).
_VALID_RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")


def _validate_response_format(response_format) -> None:
    """Raise a clean HTTP 400 for malformed ``response_format`` payloads.

    Pins three invariants the previous ``try/except`` in routes/chat.py
    failed to enforce:

    1. ``response_format.type`` must be one of ``text``,
       ``json_object``, ``json_schema``. Any other value (``"xml"``,
       ``""``, etc.) used to slip through to ``build_json_system_prompt``
       which fell back to ``"text"`` semantics — client received no
       structure enforcement and a misleading 200. Now → 400.
    2. ``type:"json_schema"`` requires a non-empty ``json_schema``
       field. Previously the missing field raised
       ``AttributeError: 'NoneType' object has no attribute 'get'``
       deep inside ``build_json_system_prompt`` which surfaced
       verbatim in the 400 body (F-013 raw-leak arm). Now → clean
       400 message naming the missing field.
    3. Empty ``response_format={}`` used to fall through with no
       ``type`` so ``rf_dict.get("type", "text")`` produced silent
       "text" semantics — fine in isolation but indistinguishable from
       a client bug that meant to send a real format. Now → clean
       400 requiring ``type``.

    Accepts either a Pydantic ``ResponseFormat`` instance or a raw
    ``dict`` (the request field is declared as
    ``ResponseFormat | dict | None`` so both shapes reach the route).
    """
    if response_format is None:
        return

    # Normalize to the shape we actually validate against.
    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
        # ``{}`` — no ``type`` key at all. Pydantic-typed path has a
        # default of "text" so this branch is only the raw-dict shape.
        if "type" not in response_format:
            raise HTTPException(
                status_code=400,
                detail="response_format.type is required",
            )
        json_schema_field = response_format.get("json_schema")
    else:
        rf_type = getattr(response_format, "type", None)
        json_schema_field = getattr(response_format, "json_schema", None)

    if rf_type not in _VALID_RESPONSE_FORMAT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "response_format.type must be 'text', 'json_object', "
                "or 'json_schema'"
            ),
        )

    if rf_type == "json_schema":
        # Treat None AND empty dict the same — both fail the "non-empty
        # json_schema spec" contract. The raw-leak path was specifically
        # ``json_schema=None`` (omitted entirely); the silent-200 path
        # was ``json_schema={}`` (present but empty so
        # ``json_schema_spec.get("schema", {})`` returned ``{}`` and
        # ``extract_json_schema_for_guided`` then bailed out at the
        # ``if not schema: return None`` guard — request proceeded with
        # no constraint).
        if not json_schema_field:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format.type='json_schema' requires "
                    "non-empty 'json_schema' field"
                ),
            )


def _is_structured_output_requested(response_format) -> bool:
    """Codex round-2 BLOCKING on #676: shared predicate for "client
    asked for structured output" — used by BOTH the non-streaming
    and streaming silent-drop rescue gates in
    ``vllm_mlx/routes/chat.py`` to decide whether to suppress the
    reasoning→content rescue.

    Returns ``True`` iff ``response_format.type`` is ``json_object``
    or ``json_schema`` — the two OpenAI-compat shapes where surfacing
    reasoning prose as ``content`` would feed the client unstructured
    text instead of validated JSON (or the existing empty/error path
    they can retry on). ``text`` (the default) and ``None`` return
    ``False`` so agentic clients still get the rescue.

    Accepts either a Pydantic ``ResponseFormat`` object (real route
    use) or a raw ``dict`` (tests / inbound JSON). Round 1 inlined
    this same check at the non-streaming call site only; round 2
    pulled it into a helper after codex caught the streaming path
    drifting — one definition, two call sites, no chance for the
    two predicates to disagree again.
    """
    if response_format is None:
        return False
    rf_type = getattr(response_format, "type", None)
    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
    return rf_type in ("json_object", "json_schema")


def _parser_accepts_enable_thinking(reasoning_parser) -> bool:
    """Return True iff ``reasoning_parser.extract_reasoning`` declares
    an ``enable_thinking`` parameter (or ``**kwargs`` catch-all).

    Static signature check avoids the side-effecting ``extract("")``
    probe an earlier draft used — that probe could hide an unrelated
    ``TypeError`` raised inside a third-party parser body and could
    trigger empty-input side effects on parsers with stateful
    accumulators. The result is cacheable per parser class but the
    function is called once per non-tool-call non-stream finalize so
    the introspection cost is negligible vs. a real LLM call.
    """
    extract = getattr(reasoning_parser, "extract_reasoning", None)
    if extract is None:
        return False
    try:
        sig = inspect.signature(extract)
    except (TypeError, ValueError):
        # Builtins / C-extensions with no introspectable signature —
        # fall back to the 1-arg call so we don't blow up here.
        return False
    params = sig.parameters
    if "enable_thinking" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _cascade(cli_value, alias_key: str, gen_key: str | None = None):
    """Layers 3+4 of the sampling resolve chain.

    Returns the first set value among:
      * ``cli_value`` — already-resolved CLI default (layer 2)
      * ``cfg.alias_recommended_sampling[alias_key]`` (layer 3)
      * ``cfg.generation_config_sampling[gen_key or alias_key]`` (layer 4)

    Returns ``None`` when nothing is set; the caller decides whether to
    apply a hard-coded fallback (temperature / top_p) or forward
    ``None`` to the engine (top_k / min_p / penalties).
    """
    if cli_value is not None:
        return cli_value
    cfg = get_config()
    alias = cfg.alias_recommended_sampling or {}
    if alias_key in alias:
        return alias[alias_key]
    gen = cfg.generation_config_sampling or {}
    key2 = gen_key or alias_key
    if key2 in gen:
        return gen[key2]
    return None


# Tool-use system prompt (auto-injected when tools are provided and parser is active)
_TOOL_USE_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: When the user's request can be answered using the provided tools, "
    "you MUST use the appropriate tool immediately. Do NOT ask for clarification when "
    "a reasonable default exists. Do NOT explain what you will do — just do it. "
    "Be direct and concise in your responses. "
    "Do NOT think out loud or show your reasoning process. "
    "Give direct answers only — no preamble like 'The user asks...' or 'Let me think...'."
)

# Tool-use system prompt for ``tool_choice="required"`` (#468). Strict
# variant of the default suffix — the OpenAI spec guarantees a tool_call
# will be present in the response when ``required`` is set, but local
# inference has no decoder-level enforcement (no FSM constraint yet,
# tracked under PR #132). Prompt injection is the strongest tool we
# have until then; the route also applies a post-parse 422 on the
# non-stream path to surface failures clearly.
_TOOL_USE_REQUIRED_SUFFIX = (
    "\n\nCRITICAL: You MUST call one of the provided tools to answer this request. "
    "Do NOT respond with text content. Do NOT explain. Do NOT ask for clarification. "
    "Pick the most appropriate tool and call it immediately. If no tool fits the "
    "user's request exactly, pick the closest match and call it with your best guess "
    "of the arguments. A text-only response is INVALID for this request."
)


def _tool_use_required_named_suffix(name: str) -> str:
    """Variant used when ``tool_choice={'type':'function','function':{'name':X}}``."""
    return (
        f"\n\nCRITICAL: You MUST call the tool named {name!r} to answer this "
        "request. Do NOT respond with text content. Do NOT explain. Do NOT call "
        "any other tool. Call this exact tool immediately with your best guess "
        "of the arguments. A text-only response is INVALID for this request."
    )


# ── Resolution helpers ─────────────────────────────────────────────


def _resolve_model_name(request_model: str | None) -> str:
    """Resolve the model name for responses — never return literal 'default'."""
    cfg = get_config()
    if not request_model or request_model == "default":
        return cfg.model_name or "default"
    return request_model


def _resolve_max_tokens(
    request_value: int | None, enable_thinking: bool | None = None
) -> int:
    """Resolve max_tokens with thinking budget for reasoning models.

    OpenAI semantics: ``max_tokens`` from the client is a hard upper
    bound on completion tokens (including reasoning). Three independent
    onboarding agents flagged the prior behavior (silently adding the
    thinking budget on top of the client's explicit cap) as
    spec-violating — clients send ``max_tokens=40`` for a short reply
    and the server scheduled ``max_tokens=2088``. v0.6.63 onboarding
    sweep finding #2.

    The thinking budget still applies when the client did NOT specify
    ``max_tokens`` (server default in effect): reasoning models need
    headroom to think *and* respond, and the server-side default is
    the right place to bake that in.
    """
    if request_value is not None:
        # Hard cap per client contract.
        return request_value
    cfg = get_config()
    base = cfg.default_max_tokens
    if enable_thinking is False:
        return base
    if cfg.reasoning_parser_name and base > 0 and base < 4096:
        return base + cfg.thinking_token_budget
    return base


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI > alias > generation_config > fallback."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_temperature, "temperature")
    if value is not None:
        return float(value)
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI > alias > generation_config > fallback."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_top_p, "top_p")
    if value is not None:
        return float(value)
    return _FALLBACK_TOP_P


def _resolve_top_k(request_value: int | None) -> int | None:
    """Resolve top_k: request > CLI > alias > generation_config > None.

    Unlike temperature/top_p, top_k has no application-level fallback —
    returning None signals "do not forward" so the engine's own
    SamplingParams default applies (matching the existing behavior of
    the extended-sampling forwarding loop).
    """
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_top_k, "top_k")
    return int(value) if value is not None else None


def _resolve_min_p(request_value: float | None) -> float | None:
    """Resolve min_p: request > CLI > alias > generation_config > None."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_min_p, "min_p")
    return float(value) if value is not None else None


def _resolve_repetition_penalty(request_value: float | None) -> float | None:
    """Resolve repetition_penalty: request > CLI > alias > generation_config > None."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_repetition_penalty, "repetition_penalty")
    return float(value) if value is not None else None


def _resolve_presence_penalty(request_value: float | None) -> float | None:
    """Resolve presence_penalty: request > CLI > alias > generation_config > None."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_presence_penalty, "presence_penalty")
    return float(value) if value is not None else None


def _resolve_frequency_penalty(request_value: float | None) -> float | None:
    """Resolve frequency_penalty: request > CLI > alias > generation_config > None."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    value = _cascade(cfg.default_frequency_penalty, "frequency_penalty")
    return float(value) if value is not None else None


def _extract_thinking_from_request(request) -> bool | None:
    """Read enable_thinking from a request without consulting global config.

    Order (first wins):
      1. ``request.chat_template_kwargs["enable_thinking"]`` (OpenAI ext spec)
      2. ``request.enable_thinking`` (top-level field, our extension)
      3. ``None`` (caller decides — usually means "template default")

    Pulled out so the dflash route can share the request-side precedence
    without inheriting the OpenAI/anthropic ``cfg.no_thinking`` consult
    (dflash's "no_thinking" lives in a closure, not the singleton).
    Single source of truth for the string-bool tolerance below.
    """
    ctk = getattr(request, "chat_template_kwargs", None)
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        v = ctk["enable_thinking"]
        if isinstance(v, bool):
            return v
        # Tolerate JSON string forms ("true"/"false") for client friendliness.
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
    return getattr(request, "enable_thinking", None)


def _resolve_enable_thinking(request) -> bool | None:
    """Resolve enable_thinking precedence for OpenAI/anthropic routes.

    Order (first wins):
      1. server ``--no-thinking`` (cfg.no_thinking) → ``False``
      2. ``request.chat_template_kwargs["enable_thinking"]`` (OpenAI ext spec)
      3. ``request.enable_thinking`` (top-level field, our extension)
      4. ``None`` (template default)

    Reported as #387: passing ``chat_template_kwargs={"enable_thinking":false}``
    used to be silently dropped because the request model didn't declare the
    field. Both this helper and the model field were added together.

    The dflash route does NOT call this helper — it has its own
    closure-scoped ``no_thinking`` and skips the cfg consult. See
    ``vllm_mlx/speculative/dflash/server.py`` for that path.
    """
    cfg = get_config()
    if cfg.no_thinking:
        return False
    return _extract_thinking_from_request(request)


def _effective_enable_thinking(
    resolved: bool | None, model_name: str | None
) -> bool | None:
    """Apply the same ``None`` → True/False fallback that
    ``vllm_mlx.utils.chat_template.apply_chat_template`` uses when
    rendering the prompt: when the request did not pin the flag,
    a non-"coder" model defaults to ``enable_thinking=True``.

    Needed by the #575 Case-4 fallback. ``_resolve_enable_thinking``
    leaves the value as ``None`` for the template-default path, but
    the Qwen3 / DeepSeek-R1 chat templates then convert that to
    ``True`` and pre-inject ``<think>`` into the prompt. The
    non-streaming finalize site must mirror that resolution or the
    parser's Case-4 fallback never fires for default-on requests
    (codex R1 BLOCKING — the bug user reproduced on every
    qwen3.5-4b / qwen3.6-35b request without an explicit flag).

    Returns the resolved bool when concrete, otherwise the same
    ``None`` to preserve pre-#575 behaviour for callers that don't
    pass a model name.
    """
    if resolved is not None:
        return resolved
    if not model_name:
        return None
    return "coder" not in model_name.lower()


def build_extended_sampling_kwargs(request) -> dict:
    """Resolve top_k / min_p / penalties through the 4-layer cascade.

    Shared by chat / completions / anthropic routes. Only forwards values
    the cascade actually produced — leaving a key absent lets the engine
    apply its own SamplingParams default, whereas forwarding ``None``
    would override it with garbage.

    ``request`` is a pydantic model; missing attributes are tolerated
    so the helper can be reused from request shapes that don't expose
    every extended param.
    """
    kwargs: dict = {}
    for name, resolver in (
        ("top_k", _resolve_top_k),
        ("min_p", _resolve_min_p),
        ("repetition_penalty", _resolve_repetition_penalty),
        ("presence_penalty", _resolve_presence_penalty),
        ("frequency_penalty", _resolve_frequency_penalty),
    ):
        value = resolver(getattr(request, name, None))
        if value is not None:
            kwargs[name] = value
    return kwargs


# ── Usage / logprobs ───────────────────────────────────────────────


def _build_usage(output: GenerationOutput, reasoning_text: str | None) -> Usage:
    """Build Usage with reasoning token breakdown when applicable.

    Per OpenAI spec, ``completion_tokens_details.reasoning_tokens`` is a
    SUBSET of ``completion_tokens`` — the remainder is content tokens.
    When both reasoning and content are present, we split the actual
    ``completion_tokens`` budget proportionally between them based on
    character ratio (chars-÷4 heuristic on each half is unreliable when
    one half exceeds the budget). The earlier ``min(reasoning, total)``
    clamp silently attributed ALL completion tokens to reasoning
    whenever ``len(reasoning_text)//4 >= total_completion``, leaving
    derived ``content_tokens == 0`` even when ``output.text`` was
    non-empty — surfaced by the v0.6.66 hybrid onboarding sweep on
    qwen3.6-27b-8bit (300/300 split with non-empty content).
    """
    cfg = get_config()
    total_completion = output.completion_tokens
    # ``output`` is normally ``GenerationOutput``, but the streaming
    # path builds an ad-hoc ``_UsageOutput`` namespace and the dflash
    # speculative server passes its own result type. ``getattr`` keeps
    # those alternative shapes working — they just report 0 cache hits
    # (semantically: "this path doesn't go through the prefix cache").
    cached_tokens = getattr(output, "cached_tokens", 0) or 0
    prompt_details = (
        PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens else None
    )
    if reasoning_text and cfg.reasoning_parser_name:
        reasoning_chars = len(reasoning_text)
        # ``output`` is normally ``GenerationOutput`` but the streaming
        # path synthesizes a ``_UsageOutput`` namespace and must pass
        # ``text`` explicitly. ``getattr`` keeps any other ad-hoc
        # callers from raising ``AttributeError`` here — they just lose
        # content-aware splitting and fall back to "all tokens are
        # reasoning" (the prior pre-fix shape) for that one path.
        content_chars = len(getattr(output, "text", "") or "")
        total_chars = reasoning_chars + content_chars
        if total_chars > 0:
            reasoning_tokens = round(total_completion * reasoning_chars / total_chars)
            # If reasoning is non-empty, attribute at least 1 token to it
            # so the field reflects that reasoning happened.
            if reasoning_chars > 0:
                reasoning_tokens = max(1, reasoning_tokens)
            # If content is also non-empty, reasoning_tokens MUST be
            # strictly less than total — leave at least 1 token for
            # content so the OpenAI-spec invariant (content_tokens =
            # completion_tokens - reasoning_tokens >= 0) reflects
            # what actually got generated.
            if content_chars > 0:
                reasoning_tokens = min(reasoning_tokens, max(0, total_completion - 1))
            else:
                reasoning_tokens = min(reasoning_tokens, total_completion)
        else:
            reasoning_tokens = 0
        return Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=total_completion,
            total_tokens=output.prompt_tokens + total_completion,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
            ),
            prompt_tokens_details=prompt_details,
        )
    return Usage(
        prompt_tokens=output.prompt_tokens,
        completion_tokens=total_completion,
        total_tokens=output.prompt_tokens + total_completion,
        prompt_tokens_details=prompt_details,
    )


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    cached_tokens = getattr(output, "cached_tokens", 0) or 0
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        prompt_tokens_details=(
            PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens else None
        ),
    )


def _extract_streaming_token_logprobs(
    chunk, tokenizer, top_k: int
) -> list[TokenLogProb]:
    """Yield one TokenLogProb per generated token in a streaming chunk.

    ``chunk.logprobs`` may be either a single per-step ``mx.array``
    (under ``stream_interval=1``) or a ``list[mx.array]`` of merged
    per-step distributions accumulated across skipped ``should_send()``
    steps (under ``stream_interval > 1``, after PR #210). The downstream
    SSE consumer expects one entry per *generated token*, not per flush
    — so we must iterate, pairing each per-step distribution with the
    corresponding token id. Without this iteration the list-form gets
    passed to ``_extract_token_logprob`` as one giant flattened array,
    and ``argmax`` reads from concatenated unrelated vocab dims (#220).

    The token-id source is ``chunk.tokens`` (the delta-token list per
    ``GenerationOutput`` — populated by ``BatchedEngine.stream_chat`` as
    ``tokens=output.new_token_ids``). The pre-fix code reached for
    ``chunk.new_token_ids`` directly, but that attribute exists on
    ``RequestOutput`` (engine internal) and was never added to
    ``GenerationOutput`` (engine public surface) — so every real
    streaming chunk raised ``AttributeError`` and the route returned
    HTTP 500 on any ``logprobs=true`` request. The earlier
    ``SimpleNamespace``-based tests masked this because they fabricated
    a ``new_token_ids`` attribute on the chunk stub — pinned now by
    ``test_logprobs_works_with_real_generation_output`` against the
    actual dataclass.
    """
    if chunk.logprobs is None or not getattr(chunk, "new_text", None):
        return []
    lps = chunk.logprobs if isinstance(chunk.logprobs, list) else [chunk.logprobs]
    tids = getattr(chunk, "new_token_ids", None) or chunk.tokens or [0]
    return [
        _extract_token_logprob(lp, tid, tokenizer, top_k) for lp, tid in zip(lps, tids)
    ]


def _extract_token_logprob(
    logprobs_array, token_id: int, tokenizer, top_k: int
) -> TokenLogProb:
    """Convert an mx.array of log-probabilities to a TokenLogProb with top-k alternatives."""
    import mlx.core as mx
    import numpy as np

    if hasattr(logprobs_array, "astype"):
        logprobs_array = logprobs_array.astype(mx.float32)
    probs = np.array(logprobs_array).flatten()
    top_k = min(top_k, len(probs))
    top_indices = np.argpartition(probs, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(probs[top_indices])][::-1]

    top_logprobs = []
    for idx in top_indices:
        idx = int(idx)
        tok_text = tokenizer.decode([idx])
        tok_bytes = list(tok_text.encode("utf-8", errors="replace"))
        top_logprobs.append(
            TopLogProb(
                token=tok_text,
                logprob=float(probs[idx]),
                bytes=tok_bytes,
            )
        )

    sampled_text = tokenizer.decode([token_id])
    sampled_bytes = list(sampled_text.encode("utf-8", errors="replace"))

    return TokenLogProb(
        token=sampled_text,
        logprob=float(probs[token_id]) if token_id < len(probs) else 0.0,
        bytes=sampled_bytes,
        top_logprobs=top_logprobs,
    )


# ── Engine / validation ────────────────────────────────────────────


def get_engine(model_name: str | None = None) -> BaseEngine:
    """Get the engine for a model, routing by name in multi-model mode."""
    cfg = get_config()
    if cfg.model_registry:
        try:
            return cfg.model_registry.get_engine(model_name)
        except KeyError:
            pass
    if cfg.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return cfg.engine


def _resolve_reasoning_enabled(model_name: str | None) -> bool:
    """Return whether the selected alias is reasoning-capable.

    Issue #702: the Anthropic-compat route gates the ``thinking``
    content block on this predicate so that a non-thinking alias (i.e.
    one whose ``aliases.json`` entry declares ``reasoning_parser:
    null``) never emits one regardless of what the OpenAI-side
    response carries.

    In multi-model mode (``cfg.model_registry`` set) the served alias
    can be a per-request choice rather than the process-wide default,
    so consult the registry entry first. Fall back to the global
    ``cfg.reasoning_parser`` / ``cfg.reasoning_parser_name`` pair
    (single-model mode) when registry lookup fails — both fields are
    populated together by ``server.load_model`` so either being set
    means "this serve has a reasoning parser configured". Accept
    either to keep test fixtures that only set
    ``cfg.reasoning_parser_name`` working unchanged. Codex r1
    BLOCKING on PR #705.
    """
    cfg = get_config()
    if cfg.model_registry:
        try:
            entry = cfg.model_registry.get_entry(model_name)
        except KeyError:
            entry = None
        if entry is not None:
            return bool(getattr(entry, "reasoning_parser", None))
    return cfg.reasoning_parser is not None or bool(cfg.reasoning_parser_name)


# ── Unicode validation (F-130 / F-131) ─────────────────────────────


def _find_lone_surrogate(s: str) -> int | None:
    """Return the offset of the first lone surrogate codepoint in ``s``,
    or ``None`` when the string is encodable as UTF-8.

    A Python ``str`` is a sequence of Unicode codepoints; ``json.loads``
    happily decodes ``"\\uD800"`` into a single-code-unit ``str``
    carrying codepoint U+D800. That codepoint is RESERVED for the
    high half of a UTF-16 surrogate pair and is not valid UTF-8 on its
    own — every downstream consumer (HuggingFace ``tokenizers`` /
    chat-template renderers / ``str.encode("utf-8")``) raises when
    handed one. F-130 (non-stream 500) and F-131 (stream 200 + raw
    Python error leak via SSE) are the same crash class surfacing on
    different lanes; rejecting the payload at the JSON-input boundary
    closes both at once.

    Properly-paired surrogates from JSON ``\\uD83D\\uDE00`` are
    coalesced by ``json.loads`` into a single astral codepoint
    (U+1F600 😀, ``len(s)==1``) before the ``str`` reaches Python, so
    valid emoji never hit this branch — we only catch the unpaired
    case the spec leaves ambiguous.

    Returning the offset (instead of just a bool) lets the caller
    surface a precise location in the error message, matching the
    diagnostic surface of the sibling ``max_tokens`` / ``top_p``
    validators in the chat route.
    """
    for i, ch in enumerate(s):
        cp = ord(ch)
        # Surrogate range per Unicode 15.1 §3.8: high surrogates
        # U+D800–U+DBFF, low surrogates U+DC00–U+DFFF. Any codepoint
        # in the combined range that survived ``json.loads`` is by
        # definition unpaired (paired surrogates from JSON are
        # coalesced into the astral codepoint they encode).
        if 0xD800 <= cp <= 0xDFFF:
            return i
    return None


def _scan_messages_for_lone_surrogates(messages: list) -> None:
    """Raise ``HTTPException(400)`` if any message slot carries a lone
    surrogate codepoint (F-130 / F-131).

    Covered slots — every string surface a client can populate that
    eventually flows into the chat template / tokenizer:

      * ``messages[i].content`` — plain string AND every ``text`` /
        ``image_url.url`` / ``video_url.url`` / ``audio_url.url`` slot
        of the multimodal ``list[ContentPart|dict]`` form
      * ``messages[i].tool_call_id`` — tool-response messages
      * ``messages[i].tool_calls[].function.name`` /
        ``messages[i].tool_calls[].function.arguments`` /
        ``messages[i].tool_calls[].id`` — assistant turns replaying
        prior tool calls
      * ``messages[i].name`` — OpenAI optional message author name

    Running at the route layer (sibling to the ``_valid_roles`` /
    ``max_tokens`` / ``top_p`` block) means the gate fires BEFORE the
    streaming branch opens an SSE response, so F-131's
    ``200 + data: chunk-with-Python-error`` leak cannot happen — the
    client sees a clean 400 with the precise offset before any byte
    of SSE is flushed.
    """

    def _check(value, path: str) -> None:
        if isinstance(value, str):
            offset = _find_lone_surrogate(value)
            if offset is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid unicode in {path}: lone surrogate "
                        f"codepoint U+{ord(value[offset]):04X} at offset "
                        f"{offset} (surrogates must appear as paired "
                        "high/low to encode an astral codepoint)."
                    ),
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                _check(v, f"{path}.{k}" if isinstance(k, str) else path)
        elif isinstance(value, list):
            for j, item in enumerate(value):
                _check(item, f"{path}[{j}]")
        elif hasattr(value, "model_dump"):
            # Pydantic ``BaseModel`` instance — e.g. ``ContentPart`` for
            # multimodal messages, or ``ImageUrl`` / ``VideoUrl`` /
            # ``AudioUrl`` nested URLs. Recurse via ``model_dump`` so the
            # scan covers every string field without enumerating each
            # pydantic class by hand (declared on ``api/models.py``).
            # Without this branch, ``content=[ContentPart(text="\\uD801")]``
            # bypasses the scan (the value is neither str/dict/list) and
            # the lone surrogate falls through to the tokenizer crash —
            # exactly the F-130 surface in the "multimodal text part"
            # slot.
            _check(value.model_dump(), path)

    for i, msg in enumerate(messages):
        # Pydantic Message or raw dict — normalize via attribute lookup.
        # ``content`` may be str | list[ContentPart] | list[dict] | None;
        # the recursive ``_check`` walks all three shapes uniformly.
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        if content is not None:
            _check(content, f"messages[{i}].content")

        tcid = (
            msg.tool_call_id
            if hasattr(msg, "tool_call_id")
            else (msg.get("tool_call_id") if isinstance(msg, dict) else None)
        )
        if tcid is not None:
            _check(tcid, f"messages[{i}].tool_call_id")

        # ``name`` is an OpenAI-spec optional message-author field. Not
        # declared on our ``Message`` pydantic model today (silently
        # dropped on parse), but client SDKs still send it and a
        # future-proof scanner shouldn't depend on whether the field
        # makes it past pydantic — check the raw dict form too.
        name = (
            msg.name
            if hasattr(msg, "name") and getattr(msg, "name", None) is not None
            else (msg.get("name") if isinstance(msg, dict) else None)
        )
        if name is not None:
            _check(name, f"messages[{i}].name")

        tcs = (
            msg.tool_calls
            if hasattr(msg, "tool_calls")
            else (msg.get("tool_calls") if isinstance(msg, dict) else None)
        )
        if tcs:
            _check(tcs, f"messages[{i}].tool_calls")


def _validate_model_name(request_model: str) -> None:
    """Validate that the request model name matches a served model."""
    if request_model is None:
        return
    # Empty string used to short-circuit to the default model silently,
    # masking client bugs (a typo or unset env var would still get a 200).
    # OpenAI returns 400 for empty model fields; do the same.
    if request_model == "":
        raise HTTPException(
            status_code=400,
            detail="model must not be empty",
        )

    cfg = get_config()
    if cfg.model_registry and request_model in cfg.model_registry:
        return
    if cfg.model_registry and request_model == "default":
        return

    if not cfg.model_name:
        return
    accepted = {cfg.model_name}
    if cfg.model_alias:
        accepted.add(cfg.model_alias)
    if cfg.model_path:
        accepted.add(cfg.model_path)
    if request_model not in accepted:
        available = (
            ", ".join(cfg.model_registry.list_model_names())
            if cfg.model_registry
            else cfg.model_name
        )
        raise HTTPException(
            status_code=404,
            detail=f"The model `{request_model}` does not exist. "
            f"Available: {available}",
        )


# ── Tool call parsing ──────────────────────────────────────────────


def _parse_tool_calls_with_parser(
    output_text: str,
    request=None,
    *,
    structured_tool_calls: list[dict] | None = None,
) -> tuple[str, list | None]:
    """Parse tool calls from model output using the configured parser.

    Creates a per-call parser instance to avoid state corruption under
    concurrent BatchedEngine requests.

    ``structured_tool_calls`` is the engine-surfaced ``[{"name",
    "arguments"}]`` list (populated by ``HarmonyStreamingRouter`` via
    openai-harmony's ``StreamableParser``). When present, the text-
    based parser is bypassed entirely — the router has already done
    the structural parse and returning to a regex pass would re-
    introduce the wire-text round-trip that lost tool calls whose
    JSON arguments contained literal harmony sentinel substrings (PR
    #515 codex round-12 / round-14 BLOCKING). ``output_text`` becomes
    the user-facing content directly in that case.
    """
    if structured_tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                type="function",
                function=FunctionCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in structured_tool_calls
        ]
        return output_text or "", tool_calls

    cfg = get_config()
    request_dict = request.model_dump() if request else None

    tokenizer = None
    if cfg.engine is not None and hasattr(cfg.engine, "_tokenizer"):
        tokenizer = cfg.engine._tokenizer

    if not cfg.enable_auto_tool_choice or not cfg.tool_call_parser:
        if cfg.reasoning_parser_name and request and request.tools:
            _PARSER_MAP = {"minimax": "minimax"}
            inferred = _PARSER_MAP.get(cfg.reasoning_parser_name)
            if inferred:
                try:
                    parser_cls = ToolParserManager.get_tool_parser(inferred)
                    parser = parser_cls(tokenizer)
                    parser.reset()
                    result = parser.extract_tool_calls(output_text, request_dict)
                    if result.tools_called:
                        tool_calls = [
                            ToolCall(
                                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                type="function",
                                function=FunctionCall(
                                    name=tc["name"],
                                    arguments=tc["arguments"],
                                ),
                            )
                            for tc in result.tool_calls
                        ]
                        return result.content or "", tool_calls
                except Exception as e:
                    logger.debug(f"Auto-infer tool parser failed: {e}")
        return parse_tool_calls(output_text, request_dict)

    # Per-call parser instance (not cfg.tool_parser_instance singleton)
    try:
        parser_cls = ToolParserManager.get_tool_parser(cfg.tool_call_parser)
        parser = parser_cls(tokenizer)
    except Exception as e:
        logger.warning(f"Failed to create tool parser '{cfg.tool_call_parser}': {e}")
        return parse_tool_calls(output_text, request_dict)

    try:
        parser.reset()
        result = parser.extract_tool_calls(output_text, request_dict)
        if result.tools_called:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in result.tool_calls
            ]
            return result.content or "", tool_calls
        else:
            return parse_tool_calls(output_text, request_dict)
    except Exception as e:
        logger.warning(f"Tool parser error: {e}")
        return parse_tool_calls(output_text, request_dict)


def _validate_tool_call_params(tool_calls: list, tools: list) -> None:
    """Validate tool call parameter values against their schemas (post-generation)."""
    from ..api.tool_logits import _extract_param_schemas, validate_param_value

    tool_defs = [t.model_dump() if hasattr(t, "model_dump") else t for t in tools]
    schemas = _extract_param_schemas(tool_defs)

    for tc in tool_calls:
        func = tc.function if hasattr(tc, "function") else tc.get("function", {})
        func_name = func.name if hasattr(func, "name") else func.get("name", "")
        args_str = (
            func.arguments
            if hasattr(func, "arguments")
            else func.get("arguments", "{}")
        )

        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"Tool call '{func_name}': arguments is not valid JSON: {args_str!r}"
            )
            continue

        if not isinstance(args, dict):
            continue

        for param_name, param_value in args.items():
            schema_key = f"{func_name}.{param_name}"
            schema = schemas.get(schema_key)
            if not schema:
                continue
            is_valid, error = validate_param_value(json.dumps(param_value), schema)
            if not is_valid:
                logger.warning(f"Tool call '{func_name}' param '{param_name}': {error}")


# ── Message helpers ────────────────────────────────────────────────


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """Inject JSON instruction into messages (prepend to system message)."""
    messages = list(messages)

    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{instruction}\n\n{existing}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{instruction}\n\n{existing}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


def _maybe_pin_system_prompt(messages: list) -> None:
    """Auto-pin system prompt prefix cache blocks on first request."""
    cfg = get_config()

    if not cfg.pin_system_prompt or cfg.engine is None:
        return

    system_content = None
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )
            if isinstance(content, str):
                system_content = content
                break

    if not system_content:
        return

    prompt_hash = hashlib.sha256(system_content.encode()).hexdigest()[:16]
    if prompt_hash == cfg.pinned_system_prompt_hash:
        return

    try:
        tokenizer = None
        if hasattr(cfg.engine, "_tokenizer"):
            tokenizer = cfg.engine._tokenizer
        elif hasattr(cfg.engine, "_model") and hasattr(cfg.engine._model, "tokenizer"):
            tokenizer = cfg.engine._model.tokenizer

        if tokenizer is None:
            return

        system_tokens = tokenizer.encode(system_content)
        if not system_tokens or len(system_tokens) < 16:
            return

        if (
            hasattr(cfg.engine, "_prefix_cache")
            and cfg.engine._prefix_cache is not None
        ):
            cache = cfg.engine._prefix_cache
            if hasattr(cache, "pin_prefix"):
                if cache.pin_prefix(system_tokens):
                    cfg.pinned_system_prompt_hash = prompt_hash
                    logger.info(
                        f"Auto-pinned system prompt: {len(system_tokens)} tokens, "
                        f"hash={prompt_hash}"
                    )
                    return

        if (
            hasattr(cfg.engine, "_cache_manager")
            and cfg.engine._cache_manager is not None
        ):
            cache = cfg.engine._cache_manager
            if hasattr(cache, "pin_prefix"):
                if cache.pin_prefix(system_tokens):
                    cfg.pinned_system_prompt_hash = prompt_hash
                    logger.info(
                        f"Auto-pinned system prompt (trie): {len(system_tokens)} tokens, "
                        f"hash={prompt_hash}"
                    )
                    return

    except Exception as e:
        logger.debug(f"System prompt pinning failed: {e}")


# ── Disconnect detection ───────────────────────────────────────────


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
    engine=None,
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect.

    When ``engine`` is provided, releases its admission reservation in
    the ``finally`` clause so the slot acquired by
    ``_check_admission_or_503`` is returned to the pool once the
    streaming response finishes (or the client disconnects, or the
    generator raises). The release is the safety net for the
    streaming path; non-streaming routes mirror it via
    ``_wait_with_disconnect``.
    """
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    logger.info(f"[disconnect_guard] START poll_interval={poll_interval}s")

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_elapsed()}"
                )
            if is_disc:
                return

    chunk_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        while True:
            anext_task = asyncio.ensure_future(aiter.__anext__())
            done, _ = await asyncio.wait(
                [anext_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks, elapsed={_elapsed()}"
                )
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break
            try:
                chunk = anext_task.result()
            except StopAsyncIteration:
                logger.info(
                    f"[disconnect_guard] generator exhausted normally, "
                    f"{chunk_count} chunks, elapsed={_elapsed()}"
                )
                break
            except Exception as exc:
                logger.error(
                    f"[disconnect_guard] generator raised {type(exc).__name__}: "
                    f"{exc}, {chunk_count} chunks, elapsed={_elapsed()}",
                    exc_info=True,
                )
                import json as _json

                # F-131 belt-and-suspenders: never leak the raw Python
                # exception message or class name through the SSE
                # ``data:`` payload. Pre-fix, a tokenizer crash (e.g.
                # the lone-surrogate ``TypeError``) surfaced inline in
                # the stream as
                # ``{"error":{"message":"Internal error during
                # streaming: TextEncodeInput must be …","type":
                # "TypeError"}}`` — useful for HuggingFace-library
                # fingerprinting and breaking the OpenAI SSE contract
                # (error payloads should not carry Python type names).
                # The route-level ``_scan_messages_for_lone_surrogates``
                # gate closes the primary path; this sanitization
                # remains for any OTHER mid-stream exception so the
                # ``200 + raw Python traceback in SSE`` shape can never
                # surface from this entry point. The full exception is
                # logged above with ``exc_info`` so operators retain
                # the diagnostic detail server-side.
                error_data = _json.dumps(
                    {
                        "error": {
                            "message": "Internal error during streaming",
                            "type": "internal_error",
                        }
                    }
                )
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
                break
            chunk_count += 1
            if chunk_count == 1:
                logger.info(
                    f"[disconnect_guard] first chunk arrived, elapsed={_elapsed()}"
                )
            yield chunk
    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        try:
            await generator.aclose()
        except Exception:
            pass
        if engine is not None:
            release = getattr(engine, "release_admission_reservation", None)
            if release is not None:
                try:
                    release()
                except Exception:
                    logger.warning(
                        "[disconnect_guard] release_admission_reservation raised",
                        exc_info=True,
                    )
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks total, elapsed={_elapsed()}"
        )


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
):
    """Run a coroutine with both timeout and client disconnect detection.

    Also catches ``BackpressureError`` from admission control and
    re-raises as HTTP 503 with Retry-After (RFC 9110 §10.2.4). Doing
    the conversion here means every route that goes through this
    helper (chat, completions, anthropic) gets correct 503 semantics
    without each one wiring its own try/except.

    Admission release is the caller's responsibility — wrap the route
    handler in ``with _admission_slot(engine):`` so the slot is
    released on ``with`` exit (covering normal completion, validation
    errors, timeouts, and disconnects). Releasing inside this helper
    would drop the slot *before* the route handler's post-processing
    finishes, briefly under-counting in-flight requests.
    """
    import time as _time

    from ..scheduler import BackpressureError

    _t0 = _time.monotonic()

    task = asyncio.ensure_future(coro)

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = await raw_request.is_disconnected()
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_time.monotonic() - _t0:.1f}s"
                )
            if is_disc:
                return

    disconnect_task = asyncio.create_task(_wait_disconnect())

    try:
        done, _ = await asyncio.wait(
            [task, disconnect_task],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {timeout:.1f} seconds",
            )

        if disconnect_task in done:
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return None

        try:
            return task.result()
        except BackpressureError as exc:
            _raise_backpressure_503(exc)

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()


# ─── Context-length pre-check (DoS defense, rapid-desktop#273 / #463) ──
#
# A 8 MiB body of plain ASCII is still ~2M tokens — well past any model's
# context window. The body-size middleware ``vllm_mlx/middleware/body_size.py``
# stops the worst of the DoS, but a request that fits inside the byte
# cap can still drag a small-context model into pointless prefill.
#
# These helpers surface the model's max context length and raise a
# structured OpenAI ``context_length_exceeded`` error when a prompt is
# too long, so the rejection lands BEFORE the engine starts prefill.

# Sentinel-large fallback used when the model exposes no useful context
# field. We do NOT silently bypass the gate (returning ``None`` would
# accept any prompt); instead we use a value so large that legitimate
# requests pass while the DoS pattern (≈ millions of tokens in one body)
# still trips. Sized for 8 MiB body cap × ~3.5 chars/token worst-case.
_FALLBACK_MAX_CONTEXT_TOKENS = 4_194_304


def get_model_max_context(engine) -> int:
    """Return the model's max prompt-token context window for ``engine``.

    Resolution order (first hit wins):
      1. ``engine._model.args.max_position_embeddings`` — mlx-lm dense
         LLMs and most MLX models expose the HF config there.
      2. ``engine._model.args.text_config.max_position_embeddings`` —
         multimodal Qwen3.5 / Gemma 4 nest the text-config inside.
      3. ``engine._model.config.max_position_embeddings`` — older
         attribute style.
      4. ``engine.tokenizer.model_max_length`` if not the HuggingFace
         "VERY_LARGE_INTEGER" sentinel (``1e30``). Some tokenizers
         report a useful cap here even when the model object doesn't.
      5. ``_FALLBACK_MAX_CONTEXT_TOKENS`` — see module-level comment.

    The function is intentionally permissive about missing fields: we'd
    rather pass through a request the model can handle than refuse a
    legitimate one on metadata absence. The byte cap stays as the
    last-resort DoS gate even if every probe falls through.
    """

    def _maybe_int(value) -> int | None:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        if ivalue <= 0:
            return None
        return ivalue

    model = getattr(engine, "_model", None) or getattr(engine, "model", None)

    if model is not None:
        args = getattr(model, "args", None)
        if args is not None:
            direct = _maybe_int(getattr(args, "max_position_embeddings", None))
            if direct is not None:
                return direct
            text_cfg = getattr(args, "text_config", None)
            if text_cfg is not None:
                nested = _maybe_int(getattr(text_cfg, "max_position_embeddings", None))
                if nested is not None:
                    return nested
        config = getattr(model, "config", None)
        if config is not None:
            cfg_direct = _maybe_int(getattr(config, "max_position_embeddings", None))
            if cfg_direct is not None:
                return cfg_direct
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                nested = _maybe_int(getattr(text_cfg, "max_position_embeddings", None))
                if nested is not None:
                    return nested

    tokenizer = getattr(engine, "tokenizer", None) or getattr(
        engine, "_tokenizer", None
    )
    if tokenizer is not None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        if tok_max is not None:
            # HuggingFace tokenizers report 1e30 ("no cap known") which
            # is useless as a guard. Treat anything above 10M as the
            # sentinel since no real model has that context yet.
            if isinstance(tok_max, int | float) and 0 < tok_max < 10_000_000:
                return int(tok_max)

    return _FALLBACK_MAX_CONTEXT_TOKENS


def count_prompt_tokens(engine, prompt) -> int:
    """Return the integer prompt-token count under ``engine``'s tokenizer.

    Accepts both string prompts (chat-template output, raw completions)
    and pre-tokenised forms (list[int] / list[list[int]]). The
    completions API contract today is ``str | list[str]`` (token-id
    prompts would be an OpenAI feature flag), but the helper is the
    one DoS-gate boundary and codex round-2 BLOCKING #3 flagged that a
    list arriving there should not silently bypass the cap. So we
    handle both shapes explicitly: token-id lists skip tokenization
    entirely and use ``len()``; strings flow through ``tokenizer.encode``
    with BOS-aware ``add_special_tokens`` handling that mirrors
    ``BatchedEngine.estimate_new_tokens``.

    Returns 0 on tokenizer failure / unknown shape so the caller
    falls through to engine-side validation rather than 500-ing on a
    metadata edge case. The wire-level body cap stays as the last
    line of DoS defense.
    """
    # Pre-tokenised forms — pure-arithmetic answer, no tokenizer needed.
    if isinstance(prompt, list):
        if not prompt:
            return 0
        first = prompt[0]
        if isinstance(first, int):
            # list[int] — a single tokenised prompt.
            return len(prompt)
        if isinstance(first, list):
            # list[list[int]] — multi-prompt batch; conservatively
            # return the longest so the cap fires on the worst entry.
            try:
                return max((len(p) for p in prompt if isinstance(p, list)), default=0)
            except TypeError:
                return 0
        # Fall through for list[str] — caller should have unpacked it,
        # but defensively handle the single-string case.
        if isinstance(first, str) and len(prompt) == 1:
            prompt = first
        else:
            return 0
    if not isinstance(prompt, str):
        return 0

    tokenizer = getattr(engine, "tokenizer", None) or getattr(
        engine, "_tokenizer", None
    )
    if tokenizer is None:
        return 0
    try:
        bos = getattr(tokenizer, "bos_token", None)
        add_special_tokens = bos is None or not prompt.startswith(bos)
        token_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return len(token_ids)
    except Exception:
        logger.debug("count_prompt_tokens: tokenizer.encode failed", exc_info=True)
        return 0


def enforce_context_length(
    engine,
    prompt_tokens: int,
    *,
    max_tokens: int | None = None,
) -> None:
    """Raise HTTP 400 ``context_length_exceeded`` if ``prompt_tokens`` is
    over the model's max context window.

    The check also includes ``max_tokens`` (the requested completion
    budget) so a borderline prompt that would force the decoder past
    the cap is rejected up-front rather than mid-generation. OpenAI's
    own error is shaped the same way — ``context_length_exceeded``
    fires when ``prompt + completion > model max``.
    """
    max_context = get_model_max_context(engine)
    completion = int(max_tokens) if max_tokens else 0
    requested_total = int(prompt_tokens) + max(0, completion)
    if requested_total <= max_context:
        return

    # Format the message in the OpenAI shape so SDKs can branch on the
    # ``code`` field. The exception handler in ``vllm_mlx/server.py``
    # wraps the ``detail`` payload back into the OpenAI envelope.
    detail = (
        f"This model's maximum context length is {max_context} tokens. "
        f"However, you requested {requested_total} tokens "
        f"({int(prompt_tokens)} prompt + {max(0, completion)} completion). "
        "Please reduce the length of the messages or completion."
    )
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": detail,
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "param": "messages",
            }
        },
    )


def enforce_context_length_for_messages(
    engine,
    messages: list,
    *,
    tools: list | None = None,
    max_tokens: int | None = None,
) -> None:
    """Run the context-length gate for a chat-style request.

    Renders the prompt through the engine's chat template (same path
    used by ``BatchedEngine.build_prompt``), counts the tokens, then
    delegates to :func:`enforce_context_length`. Wraps the template /
    tokenization step in a permissive try-except so a metadata edge
    case (e.g. unloaded engine on a route stub) doesn't 500 — the
    downstream scheduler still has its own validation.

    Scoped to text-only engines: MLLM models accept image / video /
    audio inputs whose token cost is computed by the multimodal
    processor and tracked separately by ``MLLMScheduler``. The
    body-size middleware still bounds the wire-level payload for
    those routes.

    Used by chat, anthropic, and responses routes so the same DoS gate
    applies regardless of which compatibility surface the client uses.
    """
    if getattr(engine, "is_mllm", False):
        return
    build_prompt = getattr(engine, "build_prompt", None)
    if build_prompt is None:
        return
    try:
        prompt = build_prompt(messages, tools=tools)
    except HTTPException:
        raise
    except Exception as exc:
        # Chat-template / malformed-tools-schema failures are user-
        # facing config errors. Fail fast with a clean 400 here so the
        # route doesn't waste cycles re-rendering the same template
        # downstream just to surface the same diagnosis (codex r3 F7).
        # Other exception shapes (tokenizer 500s, engine half-loaded
        # races) keep their original silent-fallthrough so the
        # scheduler's own validation has a chance to run — the
        # body-size middleware is still the last DoS line.
        err_msg = str(exc)
        err_type = type(exc).__name__
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Chat template error: {err_msg}",
            )
        return
    if not prompt:
        return
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return
    enforce_context_length(engine, prompt_tokens, max_tokens=max_tokens)


def enforce_context_length_for_prompt(
    engine,
    prompt,
    *,
    max_tokens: int | None = None,
) -> None:
    """Run the context-length gate for a raw-prompt completion request.

    Same shape as :func:`enforce_context_length_for_messages` but for
    routes that already hold a raw text prompt (``/v1/completions``).
    No chat template applied — the client provided the string (or
    list-of-ints token sequence) verbatim. ``count_prompt_tokens``
    handles both shapes; see its docstring for the codex round-2
    BLOCKING #3 rationale on non-string prompts.
    """
    if getattr(engine, "is_mllm", False):
        return
    if not prompt:
        return
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return
    enforce_context_length(engine, prompt_tokens, max_tokens=max_tokens)
