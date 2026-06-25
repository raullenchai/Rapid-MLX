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
import os
import threading
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException
from starlette.requests import Request

# Re-export of the wire-level sentinel literal + rescue-tail length.
# Single source of truth lives in :mod:`vllm_mlx.api.constants` to
# preserve the layering rule (api is the lower layer that service
# depends on, not the reverse) so the Responses adapter and the
# Anthropic adapter can both consume them without dragging the engine
# into the adapter's import graph. The names are re-exported through
# this module so existing callers that import
# ``REASONING_CUTOFF_SENTINEL`` / ``RESCUE_TAIL_LENGTH`` from
# ``vllm_mlx.service.helpers`` (route helpers + their tests) continue
# to work unchanged.
from ..api.constants import (  # noqa: F401
    REASONING_CUTOFF_SENTINEL,
    RESCUE_TAIL_LENGTH,
)
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
from ..api.utils import sanitize_output, strip_reasoning_channel_markup
from ..config import get_config
from ..engine import BaseEngine, GenerationOutput
from ..tool_parsers import ToolParserManager

logger = logging.getLogger(__name__)

# ── Fallback defaults ──────────────────────────────────────────────
_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


# ── SSE response headers (F-070 / F-073) ───────────────────────────
# Anti-buffering headers shared by every ``StreamingResponse`` that
# yields ``text/event-stream`` chunks. Without them, an intermediate
# nginx / Cloudflare / haproxy / Vercel-style reverse proxy will
# buffer the entire SSE response and emit it as one blob at end of
# generation, defeating streaming entirely. Both headers are
# documented anti-buffering knobs:
#
#   - ``Cache-Control: no-cache, no-transform`` tells caches not to
#     store the response AND tells transforming proxies (gzip, etc.)
#     to leave the byte stream alone. Mirrors what OpenAI's API
#     emits on its own SSE responses.
#   - ``X-Accel-Buffering: no`` is the nginx-specific opt-out
#     (consulted by ``ngx_http_proxy_module`` to disable
#     ``proxy_buffering`` per-response). Cloudflare, Vercel, and
#     several SaaS gateways also honour the same header.
#
# Use ``SSE_RESPONSE_HEADERS`` in every ``StreamingResponse(...)``
# that returns ``media_type="text/event-stream"``.
SSE_RESPONSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


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
    finish_reason: str | None = None,
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
        # r5-D autonomous-mode plug (F-DGF-V080-B-9, 2026-06-21):
        # ``engine_reasoning_text`` populated + ``finish_reason="length"``
        # + parser sees the buffer as ``is_open_in_think`` OR the parser
        # decided the buffer is autonomous-think (no tag in
        # ``cleaned_text`` but engine token-router classified the bytes
        # as reasoning) → route ``cleaned_text`` to nothing; the engine
        # reasoning is already in ``engine_reasoning_text``. Pre-fix
        # the cleaned_text was passed through verbatim and the same
        # bytes shipped as BOTH ``content`` and ``reasoning_content``
        # (the B-9 leak repro for glm4 autonomous mode).
        #
        # Gate carefully so we don't blank legitimate post-think
        # answers: when the engine routes reasoning AND ``cleaned_text``
        # still carries reasoning-shaped bytes (parser indicates
        # ``is_open_in_think`` OR engine reasoning literally appears
        # as a prefix of cleaned_text), it's the autonomous-mode
        # leak.
        if (
            finish_reason == "length"
            and cleaned_text
            and cleaned_text.strip()
            and reasoning_parser is not None
        ):
            is_open_in_think_attr = getattr(reasoning_parser, "is_open_in_think", None)
            parser_open = False
            if callable(is_open_in_think_attr):
                try:
                    parser_open = bool(is_open_in_think_attr(cleaned_text))
                except Exception:
                    parser_open = False
            engine_prefix_match = bool(
                engine_reasoning_text
                and isinstance(engine_reasoning_text, str)
                and cleaned_text.strip() == engine_reasoning_text.strip()
            )
            if parser_open or engine_prefix_match:
                return "", _truncate_reasoning_only(
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
        # r5-D shared finalize-on-truncation plug (F-DGF-V080-B-7 /
        # F-DGF-V080-B-9, 2026-06-21). When the model was cut by
        # ``finish_reason="length"`` AND the parser's first pass
        # routed the buffer as ``(None, buffer)`` (the leak shape)
        # AND the parser's family-specific ``is_open_in_think`` says
        # the buffer ended inside an unclosed reasoning span, route
        # the buffer to ``reasoning_content`` via the shared
        # ``finalize_truncation`` helper. This catches the
        # parser-side gaps the per-parser plugs above (#575 /
        # truncated-``<think>``) miss because their gates differ:
        #
        # * gemma4 (B-7): channel-token format. Pre-fix
        #   ``extract_reasoning`` fell through to "no thinking tags
        #   — all content" and the route's downstream rescue
        #   duplicated the same bytes into both fields (the
        #   132/128/512-char identical-dup repro). gemma4's own
        #   ``extract_reasoning`` now routes correctly on the
        #   parser-side (see ``gemma4_parser.py``); this branch is
        #   the route-side safety net.
        # * glm4 autonomous mode (B-9): glm4's chat template does
        #   NOT pre-inject ``<think>``, so a model that decided not
        #   to emit the tag and got truncated mid-thought leaves no
        #   tag in ``cleaned_text``. The parser returns ``(None,
        #   buffer)`` and ``first_parse_was_truncated_think``
        #   (which requires ``<think>`` in text_to_parse) does NOT
        #   fire. The engine's token-level ``OutputRouter`` may
        #   have populated ``engine_reasoning_text`` from the
        #   structural think tokens though — that's the
        #   out-of-band signal we use here.
        # * minimax (cross-sweep): same explicit-``<think>``-opener
        #   gap as gemma4; the parser-side fix handles the common
        #   case and this branch is the safety net.
        #
        # The plug is gated to fire ONLY on ``finish_reason ==
        # "length"`` so happy-path ``finish_reason="stop"`` flows
        # are byte-identical pre/post. The
        # ``first_parse_was_truncated_think`` plug below still
        # owns its explicit-``<think>``-in-cleaned-text case.
        if (
            finish_reason == "length"
            and cleaned_text
            and not first_parse_was_truncated_think
        ):
            is_open_in_think = getattr(reasoning_parser, "is_open_in_think", None)
            open_in_think = False
            if callable(is_open_in_think):
                try:
                    # Probe the FRESH cleaned_text. When the parser
                    # returned ``(reasoning, None)`` (e.g. gemma4 mid-
                    # thought, fixed parser-side), ``cleaned_text`` is
                    # still the raw text including the unclosed
                    # opener — exactly the buffer ``is_open_in_think``
                    # is designed to inspect. When the parser returned
                    # ``(None, raw)`` (glm4 autonomous mode), the
                    # ``cleaned_text`` is also the raw buffer (the
                    # ``new_cleaned`` assignment above doesn't
                    # alter it).
                    open_in_think = bool(is_open_in_think(cleaned_text))
                except Exception:
                    # Third-party parser threw on the probe — fall
                    # back to "not open-in-think" so we don't
                    # introduce a regression vs. the old leak shape
                    # (which clients are at least used to seeing).
                    open_in_think = False
            # Out-of-band engine-router evidence: a populated
            # ``engine_reasoning_text`` here is impossible because
            # the engine-routed branch returned early at the top of
            # the function — but we DEFENSIVELY honour the signal
            # so future re-orderings of this function don't silently
            # regress the glm4 autonomous-mode rescue.
            if not open_in_think and engine_reasoning_text:
                open_in_think = True
            if open_in_think:
                from ..reasoning import finalize_truncation

                # When the parser already extracted reasoning (e.g.
                # gemma4 mid-thought parser-side fix), it stripped the
                # marker bytes and placed the clean buffer in
                # ``reasoning_text``. Prefer that over rerouting the
                # raw ``cleaned_text`` (which still has the marker
                # bytes). Only when ``reasoning_text`` is empty do we
                # fall back to rerouting the raw buffer through the
                # helper.
                if reasoning_text:
                    cleaned_text = ""
                else:
                    routed_reasoning, routed_content = finalize_truncation(
                        True, cleaned_text
                    )
                    cleaned_text = routed_content or ""
                    reasoning_text = routed_reasoning or reasoning_text
                # Drop overflow rather than re-leak into content —
                # see ``_truncate_reasoning_only`` for the rationale
                # mirroring the explicit truncated-think plug.
                return cleaned_text, _truncate_reasoning_only(
                    reasoning_text, reasoning_max_tokens
                )
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


def _should_start_in_thinking(chat_template: str, enable_thinking: bool | None) -> bool:
    """Shared predicate: does this chat template start the assistant
    response inside an implicit ``<think>`` block?

    Some thinking-capable chat templates include ``<think>`` in the
    generated assistant prefix instead of emitting it as a normal
    output token. In that case the streaming router needs to start in
    thinking mode so tokens before ``</think>`` are emitted as
    reasoning deltas (Anthropic thinking_delta, Responses thinking
    event, OpenAI delta.reasoning_content).

    When thinking is explicitly disabled, the template marker is only
    stale capability metadata for routing purposes: direct answer
    tokens should be emitted as text. Otherwise the client receives a
    message with only a thinking block and no text result.

    Codex round-9 BLOCKING (PR #799): this helper used to live in
    ``routes/anthropic.py`` and ``routes/responses.py`` as duplicate
    private functions, plus an inline reimplementation in
    ``routes/chat.py`` that hard-coded the same ``"<think>"`` +
    ``"add_generation_prompt"`` substring check. The three copies
    could drift apart silently — chat completions could misclassify
    prompt-injected thinking templates that Anthropic / Responses
    correctly detect. Hoist to the shared service layer so every
    route uses the same predicate and the contract has a single
    source of truth.
    """
    if enable_thinking is False:
        return False
    return "<think>" in chat_template and "add_generation_prompt" in chat_template


def _rescue_silent_drop_from_reasoning(
    final_content: str | None,
    reasoning_text: str | None,
    tool_calls: list | None,
    finish_reason: str | None = None,
    raw_text: str | None = None,
    *,
    reasoning_is_case4: bool = False,
    matched_stop: str | None = None,
    prompt_thinking_active: bool = False,
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
    # D-STOP-THINK rescue gate. The #569 rescue normally copies
    # reasoning-only output into ``content`` so clients do not see an
    # empty assistant turn. Do not run that rescue when the empty content
    # is a known interrupted-thought shape; otherwise the same thought
    # bytes appear in both ``content`` and ``reasoning_content``.
    #
    # Current suppression matrix:
    #
    # finish | raw starts unclosed <think> | case4 | matched_stop | prompt think | suppress
    # length | yes                         | *     | *            | *            | yes
    # length | no                          | yes   | *            | true         | yes
    # length | no                          | yes   | *            | false        | no
    # stop   | yes                         | *     | set          | *            | yes
    # stop   | no                          | yes   | set          | true         | yes
    # stop   | no                          | yes   | set/none     | false        | no
    # stop   | no                          | no    | *            | *            | no
    #
    # ``case4`` means the parser routed a no-tag output wholly to
    # reasoning. For case4 we require the route's
    # ``prompt_thinking_active`` signal so a direct answer truncated by
    # ``max_tokens`` or by a user stop string still rescues to content.
    truncated_mid_think = (
        # Explicit-opener under length OR stop: raw_text proves
        # an in-progress ``<think>`` was truncated.
        #
        # Codex round-12 BLOCKING (PR #799): REVERTS round-11's
        # unconditional ``stop`` arm. Round-11 widened the
        # suppression on the ``stop`` arm to fire regardless of
        # ``matched_stop`` because the streaming chat path can lose
        # ``matched_stop`` between sampler and helper — but that
        # widening dropped the #569 silent-drop rescue for a model
        # that voluntarily ends after emitting ``<think>just a
        # thought`` (no closing ``</think>``, no user stop fired).
        # The helper's contract is "never silently drop an
        # assistant turn"; a natural-EOS in-progress thought must
        # still rescue. Fix: ``length`` stays unconditional (length
        # is unambiguously truncation); ``stop`` requires
        # ``matched_stop is not None`` so only engine-initiated
        # trims suppress, natural EOS falls through to the rescue
        # path. The matched_stop-propagation concern from r11 is
        # addressed at the call sites (chat.py streaming now
        # accumulates ``output.matched_stop`` per chunk, mirroring
        # responses.py / anthropic.py).
        (
            finish_reason == "length"
            and raw_text
            and raw_text.lstrip().startswith("<think>")
            and "</think>" not in raw_text
        )
        or (
            finish_reason == "stop"
            and matched_stop is not None
            and raw_text
            and raw_text.lstrip().startswith("<think>")
            and "</think>" not in raw_text
        )
        # Case-4 + length + prompt_thinking_active: parser routed the
        # whole body to reasoning (helper-Case-4 signal) and the route
        # says the chat template injected thinking, so ``length`` means
        # max_tokens cut an implicit thought. Without the template signal
        # this is just a non-thinking answer truncated mid-content, and
        # the #569 rescue must surface it instead of silently dropping it.
        or (finish_reason == "length" and reasoning_is_case4 and prompt_thinking_active)
        # Case-4 + stop + matched_stop + prompt_thinking_active:
        # codex round-5 BLOCKING — matched_stop alone with
        # Case-4 under finish=stop is NOT enough to identify the
        # D-STOP-THINK shape. A casual answer like ``"The answer
        # is STOP"`` under ``stop=["STOP"]`` ALSO has matched_stop
        # set but is NOT chain-of-thought. Require the route-
        # supplied ``prompt_thinking_active`` boolean (chat
        # template injected ``<think>`` AND ``enable_thinking`` is
        # non-False) as the secondary discriminator. Symmetric
        # with the parser-level ``finalize_streaming`` AND-of-
        # signals contract.
        or (
            finish_reason == "stop"
            and reasoning_is_case4
            and matched_stop is not None
            and prompt_thinking_active
        )
    )
    if truncated_mid_think:
        return final_content
    # r5-D (F-DGF-V080-B-7, 2026-06-21): gemma4 channel-token analog
    # of the truncated-``<think>`` gate above. When generation is cut
    # mid-thought-channel (``<|channel>thought\n…``-without-
    # ``<channel|>``), the reasoning trace is NOT the final answer —
    # surfacing it as ``content`` per the #569 rescue would feed the
    # desktop client the SAME bytes as ``reasoning_content`` (the
    # 132/128/512-char identical-dup repro that this PR closes).
    # Skip the rescue and let the client see ``content=null`` so it
    # can detect "model ran out of budget mid-thought" via
    # ``finish_reason="length"`` — symmetric with the
    # truncated-``<think>`` and D-HARMONY-LEAK gates.
    if (
        finish_reason == "length"
        and raw_text
        and "<|channel>thought" in raw_text
        and "<channel|>" not in raw_text[raw_text.rfind("<|channel>thought") :]
    ):
        return final_content
    # D-HARMONY-LEAK (2026-06-21): harmony-channel analog of the
    # truncated-``<think>`` gate above. The gpt-oss family (and any
    # Harmony-encoding tokenizer) emits ``<|channel|>analysis<|message|>
    # …<|end|><|channel|>final<|message|>…<|return|>`` as the wire
    # contract for a complete reasoning-then-answer turn — the
    # ``<|channel|>analysis<|message|>`` opener is the analysis-channel
    # start marker and ``<|channel|>final<|message|>`` is the
    # final-channel start marker (the user-visible answer). When
    # generation is cut short BEFORE the final-channel opener appears
    # (max_tokens cut mid-analysis OR ``stop:["X"]`` matched a stop-
    # string that happens to land in the analysis body), the engine
    # has correctly routed the analysis bytes into ``reasoning_text``
    # and left ``content`` empty — exactly the silent-drop shape this
    # rescue was designed to fix. But the analysis body is NOT the
    # model's final answer, so promoting it to ``content`` ships the
    # SAME bytes in both fields (``content == reasoning_content``
    # mojibake) — the bug filed as D-HARMONY-LEAK. Gate the rescue
    # on the harmony state machine: when raw_text shows an analysis-
    # channel opener but NO final-channel opener, we are structurally
    # mid-state-machine and the rescue must NOT fire. The gate is
    # finish_reason-AGNOSTIC because both repros (max_tokens=length
    # AND stop-string match=stop) produce the same broken shape —
    # letting the rescue fire on either would re-leak the analysis
    # body into ``content``.
    #
    # Codex r1 BLOCKING #1: an earlier revision exempted any raw_text
    # carrying ``<|call|>`` (commentary tool-call terminator) under
    # the assumption the upstream ``tool_calls`` branch would catch
    # it. That assumption is unsafe — if the tool-call parser failed
    # to extract a structured call (malformed args, downstream filter
    # dropping the entry, ``tool_calls`` not threaded into this
    # helper at all by a third-party caller), the analysis body
    # would still leak as user-visible content. Suppress the rescue
    # for ANY harmony "analysis without final" state and rely on the
    # earlier ``if tool_calls:`` branch to preserve parsed tool
    # calls — that branch already returns the original
    # ``final_content`` so a populated ``tool_calls`` list never
    # reaches this point. When the model DID reach the final channel
    # (final-marker present), control already returned through the
    # happy-path early-exit above, so no override is needed for that
    # case.
    if (
        raw_text
        and "<|channel|>analysis<|message|>" in raw_text
        and "<|channel|>final<|message|>" not in raw_text
    ):
        return final_content
    return reasoning_text


# ---------------------------------------------------------------------------
# Reasoning-cutoff sentinel (default ON; opt out via
# RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled). H-01 / R-01 / issue #858.
# ---------------------------------------------------------------------------
#
# When a reasoning model (qwen3, deepseek_r1, phi-4-mini-reasoning, glm4,
# gemma4, vibethinker, …) is called with a low ``max_tokens`` budget and
# generation is cut short BEFORE ``</think>`` (or the harmony final-channel
# marker), the parser-wide rule pinned by D-STOP-THINK + D-HARMONY-LEAK is
# "route everything to ``reasoning_content`` and leave ``content``
# null/empty" — the in-progress thought trace is NOT the final answer, so
# promoting it to ``content`` would ship byte-identical bytes in both
# fields (the leak shape those PRs explicitly closed).
#
# History:
#
# * H-01 (PR #802, 2026-06-21, v0.8.3): introduced an opt-OUT sentinel
#   that was injected into ``content`` by default to give SDK consumers
#   a literal "truncated, raise max_tokens" cue instead of an empty
#   bubble.
# * R-01 (PR #815, v0.8.5): flipped the default to opt-IN on
#   structured-purity rationale (every transport already carries an
#   unambiguous ``finish_reason="length"`` / ``status="incomplete"`` /
#   ``stop_reason="max_tokens"``, so synthesizing a literal text block
#   the model never produced was deemed harmful injection).
# * Issue #858 (this commit, v0.8.12): reverts R-01. Every GUI client
#   (rapid-desktop, vanilla OpenAI SDK consumers, OpenWebUI compat
#   layers) renders only ``message.content`` and ignores the structured
#   ``finish_reason`` field — under R-01's default-off, they showed an
#   empty bubble whenever a reasoning model hit ``max_tokens`` mid-think.
#   The literal sentinel is the user-visible cue that ``max_tokens`` was
#   too low, and restoring it as the default outweighs the
#   structured-purity gain. Power callers that want strict-null behaviour
#   set ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` (or ``0`` /
#   ``false`` / ``no`` / ``off``).
#
# Structured truncation signals — also present on every transport,
# regardless of the env var setting — for callers that DO want to gate
# on them:
#
#   * /v1/chat/completions  → ``finish_reason="length"``
#   * /v1/responses         → ``status="incomplete"`` +
#                              ``output_tokens_details.reasoning_tokens``
#   * /v1/messages          → ``stop_reason="max_tokens"`` +
#                              ``thinking`` content block
#
# Scope (unchanged across the R-01 ↔ issue #858 flip):
#
# * Fires ONLY when the env var has NOT been set to a disable value.
# * Fires ONLY on ``finish_reason="length"`` (NOT on ``"stop"`` —
#   stop-string mid-think is D-STOP-THINK's exact case, where the strict
#   null contract must hold and the caller can re-request to drive the
#   model past the stop string).
# * Fires ONLY when ``content`` is empty/None AND ``reasoning_text`` is
#   non-empty — the silent-drop shape clients actually trip on. Happy-
#   path (closed ``</think>answer`` split) flows untouched.
# * Fires ONLY when no ``tool_calls`` were extracted — tool-only responses
#   legitimately ship ``content=None`` per the OpenAI spec.
#
# Single source of truth — both the OpenAI ``/v1/chat/completions``
# non-stream + stream paths AND the Anthropic ``/v1/messages`` adapter
# AND the ``/v1/responses`` adapter call this helper, so the user-visible
# behaviour cannot drift between surfaces. (The streaming path emits the
# sentinel as one final-chunk ``delta.content`` event, not per-token, so
# no token-by-token leak of the sentinel string itself.)


#: ``RESCUE_TAIL_LENGTH`` is re-exported above from
#: :mod:`vllm_mlx.api.constants` — kept at module scope so existing
#: callers (route helpers + tests) continue to import it from
#: ``vllm_mlx.service.helpers``. The Anthropic adapter consumes the
#: SAME constant from the lower-layer ``api.constants`` to compute the
#: matching suffix it must trim from the ``thinking`` content block
#: (R12-M1b dedupe fix).

#: Env var values that EXPLICITLY DISABLE the rescue notice. The
#: primary knob is ``RAPID_MLX_REASONING_RESCUE`` (R12-8 / issue #259);
#: the legacy ``RAPID_MLX_REASONING_CUTOFF_NOTICE`` (PR #802 / #815 /
#: #860) is still honoured as a back-compat alias so existing
#: rapid-desktop / agent deployments don't break on upgrade.
#:
#: GUI clients (rapid-desktop ChatView, OpenAI-SDK consumers, etc.)
#: render blank message bubbles when ``content`` is ``None`` even
#: though ``finish_reason="length"`` is set — the rescue string is the
#: user-facing signal that ``max_tokens`` was too low PLUS a glimpse of
#: the truncated thought trace. Power callers that prefer the strict-
#: null shape can opt out with any of the listed spellings on either
#: env var.
_CUTOFF_NOTICE_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled"})

#: Primary R12-8 env var. ``RAPID_MLX_REASONING_RESCUE=off`` disables
#: the rescue; default is ``on``. Both this name and the legacy alias
#: below are read; if either is set to a disable value, the rescue is
#: off (operator intent: "I do not want the literal text injection").
_RESCUE_ENV_PRIMARY = "RAPID_MLX_REASONING_RESCUE"
#: Legacy alias from PR #802 / #860 (issue #858). Kept for back-compat
#: so callers that already set ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``
#: don't need to re-deploy when they upgrade to the R12-8 build.
_RESCUE_ENV_LEGACY = "RAPID_MLX_REASONING_CUTOFF_NOTICE"


def _cutoff_notice_enabled() -> bool:
    """Whether the cutoff rescue is enabled for this process.

    R12-8 / issue #259 expands the rescue from a bare sentinel string
    to ``sentinel + tail-of-reasoning`` (the 8-round carry kept getting
    reopened because the bare sentinel still felt like "the model
    didn't answer" to six independent reviewers). The default-on /
    opt-out policy is unchanged from PR #860 — only the rescue payload
    grew. Issue #858 revert (PR #860) settled the default-on question:
    GUI clients that only render ``content`` showed empty bubbles
    under R-01's default-off, and that ships as a user-visible bug.

    Reads the env vars on each call so test harnesses can flip the
    gate per-request via ``monkeypatch.setenv`` without restarting the
    process. The cost is negligible (``os.environ.get`` is a dict
    lookup) and matches how every other ``RAPID_MLX_*`` env-gated knob
    is read in this module.

    Two env vars are honoured (in this priority order):

    * ``RAPID_MLX_REASONING_RESCUE`` — R12-8 primary name. Easier to
      discover than the legacy alias (``RESCUE`` is what the operator
      thinks of it as) and aligns with the task spec.
    * ``RAPID_MLX_REASONING_CUTOFF_NOTICE`` — legacy alias from PR
      #802 / #860 (issue #858). Still honoured so existing
      rapid-desktop deployments and operator runbooks that already
      reference this name keep working without a rebuild.

    The rescue is DISABLED when EITHER env var is set to a disable
    spelling (``"0"`` / ``"false"`` / ``"no"`` / ``"off"`` /
    ``"disabled"``, case-insensitive, whitespace-stripped). Operator
    intent: "I do not want the rescue", regardless of which name was
    used. Anything else — including unset, the empty string,
    ``"1"`` / ``"true"`` / ``"on"`` / ``"yes"`` / ``"enabled"``, or
    any arbitrary unrecognised value — leaves the rescue enabled.
    """
    for env_name in (_RESCUE_ENV_PRIMARY, _RESCUE_ENV_LEGACY):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if raw.strip().lower() in _CUTOFF_NOTICE_DISABLED_VALUES:
            return False
    return True


def _build_reasoning_rescue_payload(reasoning_text: str) -> str:
    """Build the rescue ``content`` string for R12-8.

    Layout: ``"<sentinel>\\n\\n<tail-of-reasoning>"``. The sentinel
    anchors the opening (agentic auto-retry clients pattern-match the
    prefix), then a blank line, then the LAST ``RESCUE_TAIL_LENGTH``
    chars of the reasoning trace so a human sees the partial
    conclusion. The tail is taken from the END of ``reasoning_text``
    because that's where the partial answer lives in every reasoning
    parser dialect we ship (qwen3, deepseek_r1, glm4, vibethinker,
    harmony — all build the answer at the tail of the thought).

    Trailing whitespace on the reasoning trace is stripped before the
    slice so the rescue doesn't end on a partial newline. The caller
    has already verified ``reasoning_text`` is non-empty + non-
    whitespace (see ``_apply_reasoning_cutoff_notice``), so an empty
    tail is impossible by construction.

    The reasoning trace is stripped of channel markup BEFORE the
    tail slice is chosen, then the slice runs through
    :func:`sanitize_output` (general special-token catch-all:
    ``<|...|>``, harmony channel markers, ``</tool_call>``, etc.).
    The strip-before-slice order matters: a naive slice of the raw
    reasoning bytes can bisect a structural ``<think>`` /
    ``</think>`` tag (codex r3 P2: when reasoning is just slightly
    longer than ``RESCUE_TAIL_LENGTH`` and starts with ``<think>``,
    the raw slice begins ``hink>...`` and the regex stripper no
    longer matches the orphaned fragment — so the literal ``ink>``
    bytes leak into ``content``). Stripping channel markup first
    means the slice operates on the clean, in-channel byte stream
    and can never bisect a tag because there are no tags left to
    bisect.
    ``reasoning_content`` keeps the full original trace addressable
    for clients that walk both fields. Codex r1 (R12-8): the rescue
    path previously bypassed the sanitizer that ``content``
    consumers rely on. R12-M1b (Mira r12 R-3 bonus regression):
    also strip the ``<think>`` OPENER — at ``max_tokens=1`` the
    reasoning trace IS the literal opener, and ``sanitize_output``
    deliberately leaves the opener alone on the ``content`` channel
    (where it can be legit Nemotron prefix injection or literal-tag
    prose). The rescue tail is from the reasoning channel, so the
    channel-aware strip applies.
    """
    # Strip BEFORE slicing — see docstring for the bisection
    # regression this order closes.
    stripped = strip_reasoning_channel_markup(reasoning_text.rstrip())
    tail = stripped[-RESCUE_TAIL_LENGTH:]
    sanitized = sanitize_output(tail)
    if not sanitized:
        return REASONING_CUTOFF_SENTINEL
    return f"{REASONING_CUTOFF_SENTINEL}\n\n{sanitized}"


def _apply_reasoning_cutoff_notice(
    final_content: str | None,
    reasoning_text: str | None,
    tool_calls: list | None,
    finish_reason: str | None,
) -> str | None:
    """R12-8 / H-01: rescue ``content`` when generation was cut short
    mid-think and the strict rescue path left it empty.

    Runs AFTER ``_rescue_silent_drop_from_reasoning`` — its job is the
    UX rescue for the cases the silent-drop rescue deliberately
    SUPPRESSED (truncated ``<think>``, harmony analysis-without-final,
    Case-4 no-tag fallback). All those cases share the same observable
    shape: ``finish_reason="length"`` + empty content + non-empty
    reasoning + no tool calls.

    R12-8 (issue #259, 8-round D-carry) extends PR #802 / #860: the
    rescue is now ``sentinel + tail-of-reasoning`` rather than the bare
    sentinel, because six independent reviewers kept reopening the H-01
    carry — the bare sentinel still felt like "the model didn't
    answer". The tail is the LAST ``RESCUE_TAIL_LENGTH`` chars of the
    reasoning trace — that gives a human reader a glimpse of the
    partial conclusion without dumping the whole trace into
    ``content`` (which would re-introduce the D-STOP-THINK /
    D-HARMONY-LEAK byte-identical leak shape the rescue is allowed to
    exist alongside). ``reasoning_content`` is NEVER touched — the
    full original trace stays addressable to clients that walk both
    fields.

    Returns ``final_content`` unchanged when:
    * the env var disables the rescue (``RAPID_MLX_REASONING_RESCUE=off``
      or the legacy ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``)
    * ``finish_reason`` is anything other than ``"length"`` (stop-string
      cut mid-think hits the D-STOP-THINK regression guard — strict
      null wins; a clean ``"stop"`` finish on an empty answer is a
      legitimate model decision and gets no "raise max_tokens" hint)
    * ``final_content`` already carries a non-whitespace payload
    * ``tool_calls`` were extracted (OpenAI-spec ``content=None`` path)
    * ``reasoning_text`` is empty / whitespace (nothing to signal — the
      model produced nothing semantically, which is a different bug
      class and shouldn't get a "raise max_tokens" hint)

    Otherwise returns the rescue payload produced by
    :func:`_build_reasoning_rescue_payload` — the canonical shape is
    ``sentinel + "\\n\\n" + sanitized_tail``. The builder applies
    operations in this order (codex r3 P2 strip-before-slice):

    1. :func:`strip_reasoning_channel_markup` on
       ``reasoning_text.rstrip()`` — strips ``<think>`` / ``</think>``
       on the FULL trace so the next slice operates on a clean,
       in-channel byte stream. Doing it before the slice means a tag
       straddling the ``L - RESCUE_TAIL_LENGTH`` boundary can never
       leave an orphan ``<th`` / ``ink>`` fragment in the tail.
    2. ``[-RESCUE_TAIL_LENGTH:]`` slice on the stripped trace.
    3. :func:`sanitize_output` on the slice — general special-token
       catch-all (``<|...|>``, harmony markers, ``</tool_call>``, …).

    When sanitization collapses the tail to empty (e.g.
    ``reasoning_text="<think>"`` at ``max_tokens=1``), the rescue
    builder returns the bare sentinel — clients still see the
    structural truncation signal without a stray markup byte in
    ``content``. The caller writes the returned string into
    ``message.content`` (non-stream) or the final SSE ``delta.content``
    chunk (stream).
    """
    if not _cutoff_notice_enabled():
        return final_content
    if finish_reason != "length":
        return final_content
    if final_content and final_content.strip():
        return final_content
    if tool_calls:
        return final_content
    if not reasoning_text or not reasoning_text.strip():
        return final_content
    return _build_reasoning_rescue_payload(reasoning_text)


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
                "response_format.type must be 'text', 'json_object', or 'json_schema'"
            ),
        )

    if rf_type == "json_schema":
        # Treat None / empty dict / missing-``schema``-member all the
        # same — each fails the "non-empty json_schema spec" contract.
        # The raw-leak path was specifically ``json_schema=None``
        # (omitted entirely); the silent-200 path was either
        # ``json_schema={}`` or ``json_schema={"name":"r"}`` (present
        # but with no ``schema`` member — codex r1 BLOCKING) because
        # ``extract_json_schema_for_guided`` then bails out at
        # ``if not schema: return None`` and the request proceeds
        # unconstrained. The Pydantic-typed ``ResponseFormatJsonSchema``
        # branch declares ``schema_`` as a required field so the typed
        # path already rejects this shape — the explicit check here
        # closes the raw-dict arm (the ``ResponseFormat | dict`` union
        # on the request field).
        if not json_schema_field:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format.type='json_schema' requires "
                    "non-empty 'json_schema' field"
                ),
            )
        # Extract the inner ``schema`` member through both shapes:
        # raw dict (json_schema_field is a dict) and Pydantic
        # ``ResponseFormatJsonSchema`` (the field is aliased to
        # ``schema_`` to dodge the BaseModel.schema collision).
        if isinstance(json_schema_field, dict):
            inner_schema = json_schema_field.get("schema")
        else:
            inner_schema = getattr(json_schema_field, "schema_", None)
        if not inner_schema:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format.type='json_schema' requires "
                    "'json_schema.schema' to be a non-empty object"
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
    "Give direct answers only — no preamble like 'The user asks...' or 'Let me think...'. "
    # D-TOOLCHOICE-R1 T1: DeepSeek-R1 distills (and other reasoning
    # models) under ``tool_choice="auto"`` will happily HALLUCINATE
    # the result of a tool they were never told existed — emit
    # ``"The current temperature in Tokyo is 24°C"`` while the only
    # weather data they have is whatever the user typed. The
    # earlier "use a tool immediately" clause does not cover this:
    # the model can interpret "use a tool" as "include a tool-shaped
    # answer". This clause is a HARD floor: if you didn't actually
    # call a tool, you must not claim a tool result.
    "If you do NOT call a tool, do NOT fabricate the contents of any tool's response — "
    "answer only from what you actually know. Do NOT print fake JSON, fake API responses, "
    "or sentences that begin with 'Tool returned:' / 'Tool output:' / 'The API returned'."
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


def _aliases_match(a: str, b: str) -> bool:
    """Return True when ``a`` and ``b`` refer to the same model.

    Both sides run through the shared ``resolve_model()`` alias registry
    so that the short alias (``embeddinggemma-300m-6bit``) and the full
    HF id (``mlx-community/embeddinggemma-300m-6bit``) compare equal —
    same one-source-of-truth rule used at boot time.

    Used by ``/v1/embeddings`` + ``/v1/audio/*`` request handlers to
    accept either form the client happens to send. Pre-fix the routes
    did literal string equality against ``cfg.embedding_model_locked``
    (set to the resolved HF path at boot), so a client that legitimately
    sent the short alias listed in ``/v1/models`` ate a 400.
    """
    if a == b:
        return True
    if not a or not b:
        return False
    from ..model_aliases import resolve_model

    try:
        return resolve_model(a) == resolve_model(b)
    except Exception:  # noqa: BLE001 — registry I/O must never 500 the route
        return False


def _resolve_request_alias_or_default(
    request_model: str | None, locked: str | None
) -> str | None:
    """Map a request-supplied ``model`` field to the server-locked id.

    Single source of truth for the OpenAI-canonical ``"default"``
    placeholder + alias-aware comparison used by every
    request-time route (``/v1/embeddings``, ``/v1/audio/*``,
    ``/v1/chat/completions``-style routes that don't run the full
    registry probe).

    Resolution rules:

    * ``request_model`` is ``None`` / ``""`` / ``"default"`` → return
      ``locked`` verbatim. The OpenAI SDK + LangChain + LlamaIndex
      all default to ``"default"`` when the caller hasn't picked a
      specific model id; rejecting it breaks drop-in compatibility.
    * ``request_model`` resolves (via ``resolve_model``) to the same
      id as ``locked`` → return ``locked``. Accepts both the short
      alias and the full HF path so the user-facing ``--<flag>``
      value, the ``/v1/models`` listing, and the request body don't
      have to be byte-for-byte identical to match.
    * Otherwise → return ``None``. Caller decides the rejection
      envelope (404 vs 400) because the canonical shape differs
      between embeddings and audio routes (#805 envelope rules).
    """
    if locked is None:
        return None
    if not request_model or request_model == "default":
        return locked
    if _aliases_match(request_model, locked):
        return locked
    return None


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

    The thinking budget applies only when neither the client nor the
    operator specified a cap. If the default came from
    ``serve --max-tokens`` (or another explicit operator setting), that
    value is also a hard upper bound and must not receive additive
    headroom.
    """
    if request_value is not None:
        # Hard cap per client contract.
        return request_value
    cfg = get_config()
    base = cfg.default_max_tokens
    if cfg.default_max_tokens_is_explicit:
        return base
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


def _resolve_seed(request_value: int | None) -> int | None:
    """Resolve per-request seed (H-11).

    Unlike the other extended sampling params, seed has no CLI / alias /
    generation_config cascade — it is purely a runtime knob the client
    flips per request when they want deterministic output. Returning
    ``None`` signals "do not forward" so the scheduler keeps the
    fast-path interned sampler (cached by ``(temp, top_p, min_p,
    top_k)``) instead of building a fresh per-request sampler closure
    on every call. That matters: the seeded sampler MUST be uncached
    because it carries mutable per-call key state.
    """
    return request_value


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


def maybe_auto_disable_thinking_for_tools(request) -> bool:
    """R12-T1F: auto-disable ``enable_thinking`` when tools are declared
    and the client did not pin a preference. Mirrors the M-2 strict-
    json_schema auto-disable pattern (PR #877) — same root cause, same
    shape, single source of truth so chat / responses / anthropic /
    future surfaces share one contract.

    Trigger (all must hold):
      * ``request.tools`` is non-empty (caller wants the model to
        emit a tool_call).
      * ``request.tool_choice`` is NOT the string ``"none"``. The
        OpenAI ``tool_choice="none"`` contract explicitly tells the
        model to ignore the supplied tool list and answer in prose
        — auto-disabling thinking there would turn a prose request
        into thinking-off behavior solely because tool DEFINITIONS
        were attached, contradicting the contract (codex r1 BLOCKING).
      * Neither ``chat_template_kwargs["enable_thinking"]`` nor the
        top-level ``enable_thinking`` field is set on the request.

    Effect: merge ``{"enable_thinking": False}`` onto
    ``request.chat_template_kwargs`` so every downstream consult
    (``_resolve_enable_thinking``, ``_effective_enable_thinking``,
    ``engine.chat`` / ``stream_chat`` / ``generate_with_schema``)
    sees the resolved choice. The merge is non-destructive: any
    forward-compat keys the client passed survive untouched.

    Rationale (operator dogfood on Qwen3-0.6B-bf16, 0.8.16):
    thinking-on by default is the right answer for prose, but for
    tool-calling the model spends its entire ``max_tokens`` budget
    inside ``<think>...</think>`` before emitting the
    ``<tool_call>`` envelope. With the typical agent-SDK budget
    (``max_tokens=50..100``) the request finishes with
    ``finish_reason="length"`` and ``tool_calls=None`` — the tool
    never fires. Default-off thinking restores the "tool calling
    just works" contract; clients who explicitly opt back in
    (``chat_template_kwargs={"enable_thinking": true}`` or top-
    level ``enable_thinking: true``) keep the chain-of-thought and
    accept the budget risk.

    Returns ``True`` if the auto-disable injection fired, ``False``
    if it was skipped (no tools, or client preference already
    set). The returned bool is the load-bearing signal for the
    route's structured log line; callers that do not need it can
    discard the return value.
    """
    tools = getattr(request, "tools", None)
    if not tools:
        return False
    # tool_choice="none" tells the model to ignore the tool list
    # entirely and answer in prose — the budget-burn rationale does
    # not apply (no tool_call is expected), and forcing thinking off
    # would change a prose request's behavior solely because the
    # client attached tool DEFINITIONS. Skip the auto-disable so
    # default-on thinking is preserved for this prose path. Codex
    # r1 BLOCKING (R12-T1F follow-up): symmetric with the chat /
    # Anthropic adapters' downstream ``tool_choice="none"`` handling
    # (no system-prompt injection, no FSM enforcement) — keep the
    # auto-disable gate aligned with the rest of the no-tool path.
    tool_choice = getattr(request, "tool_choice", None)
    if isinstance(tool_choice, str) and tool_choice == "none":
        return False
    if _extract_thinking_from_request(request) is not None:
        return False
    existing_ctk = getattr(request, "chat_template_kwargs", None) or {}
    # Merge rather than replace so any non-thinking keys the client
    # passed (forward-compat, e.g. future kwargs the chat template
    # honors) survive untouched. Codex-r3 BLOCKING contract from M-2.
    merged_ctk = dict(existing_ctk)
    merged_ctk["enable_thinking"] = False
    request.chat_template_kwargs = merged_ctk
    # Codex r1 MEDIUM #2 (R12-T2F-276): tag the request so the L-05
    # ``enable_thinking_warning_header`` does NOT fire spuriously on
    # non-qwen3 parsers — the server injected the flag, not the client.
    _mark_thinking_auto_disabled(request)
    return True


def maybe_auto_disable_thinking_for_casual_chat(request, *, extra_signals=None) -> bool:
    """R12-T2F-276: auto-disable ``enable_thinking`` on a casual chat
    completion to a thinking-capable model when the caller did not
    pin a thinking preference or otherwise express explicit reasoning
    intent. Third member of the auto-disable family — mirrors the
    M-2 strict-json_schema gate (PR #877) and the R12-T1F tools gate
    (PR #891), and shares the same merge contract / single source
    of truth so chat / responses (and any future surface that adds
    a thinking-capable path) inherit the fix for free.

    Trigger (all must hold):

      * The server has a reasoning parser configured
        (``cfg.reasoning_parser_name is not None``). This is the
        only server-side signal that the active model is in fact
        thinking-capable — when no parser is registered the model
        does not emit ``<think>`` at all, so the budget-burn failure
        mode does not apply and the auto-disable would be inert
        (worst case: a silent template flip on a no-op flag).
      * The client did NOT pin ``enable_thinking`` via either
        ``chat_template_kwargs["enable_thinking"]`` or the top-level
        ``enable_thinking`` field (i.e.
        ``_extract_thinking_from_request`` is ``None``).
      * The client did NOT signal explicit reasoning intent through
        any of:
          - ``reasoning_max_tokens`` (chat / responses): an explicit
            per-request cap is itself proof the caller wants
            reasoning ON (just bounded). Default-disabling here
            would silently collapse the contract to "no reasoning".
          - ``reasoning_effort`` (chat / responses top-level): the
            OpenAI-spec knob that says "yes, I want reasoning at
            level X". Same rationale.
          - ``reasoning`` dict (responses native shape, e.g.
            ``{"effort":"low"}``): the canonical /v1/responses
            opt-in. Consulted via ``extra_signals`` because the
            ResponsesRequest adapter does NOT forward the field
            onto the materialized ``ChatCompletionRequest`` (the
            engine reads the already-translated
            ``reasoning_max_tokens`` / ``reasoning_effort`` fields)
            AND the ChatCompletionRequest schema sets
            ``extra="forbid"``-equivalent semantics so a stray
            ``setattr`` would 500. Pass the original
            ``ResponsesRequest`` as ``extra_signals`` to thread the
            signal in cleanly.

    ``extra_signals``: an optional secondary request-shaped object
    consulted for the SAME signal set. Used by the /v1/responses
    route to thread the Responses-native ``reasoning`` dict (and
    ``reasoning_effort`` / ``reasoning_max_tokens`` shorthands that
    live on the ``ResponsesRequest`` only) through to the helper
    without having to fork the trigger logic. The chat surface
    passes ``None`` because the materialized ``ChatCompletionRequest``
    already carries every signal it needs.

    Tools / strict-json interactions: the R12-T1F (tools) and R12-M2
    (strict json_schema) helpers run BEFORE this one in the route
    plumbing and inject ``chat_template_kwargs["enable_thinking"]=False``
    onto the request when their own triggers fire. By the time this
    helper runs ``_extract_thinking_from_request`` is no longer
    ``None`` for those paths and the gate above short-circuits to
    ``False`` — so the auto-disable family composes without double-
    firing or special-casing. Mirror of the M-2 + T1F merge contract:
    explicit signals from any of the three families win.

    Effect: merge ``{"enable_thinking": False}`` onto
    ``request.chat_template_kwargs`` so every downstream consult
    (``_resolve_enable_thinking``, ``_effective_enable_thinking``,
    ``engine.chat`` / ``stream_chat``) sees the resolved choice.
    Non-destructive merge: forward-compat keys the client passed
    survive (codex round-3 BLOCKING contract from M-2).

    Rationale (operator dogfood 0.8.16 brand-new-user simulation):
    a first-time SDK user writes::

        client.chat.completions.create(
            model="qwen3.5-4b-4bit",   # thinking-capable
            messages=[{"role":"user","content":"In 8 words, what is rapid-mlx?"}],
            max_tokens=80,
        )

    Pre-fix the model burns the entire 80-token budget inside
    ``<think>...</think>`` and never emits an answer — the response
    surfaces with ``finish_reason="length"`` and ``content`` carrying
    the rescue-sentinel header followed by raw chain-of-thought (the
    repro pinned in this task). The ``rapid-mlx chat`` REPL already
    solves this by defaulting ``--no-think`` for thinking-capable
    models — this helper is the OpenAI-SDK-surface parity for that
    same default, so a brand-new user gets a useful first request
    without having to learn ``chat_template_kwargs``.

    Returns ``True`` if the auto-disable injection fired, ``False``
    if it was skipped (no thinking parser, client pinned thinking,
    or client signalled reasoning intent). The returned bool is the
    load-bearing signal for the route's structured log line.
    """
    cfg = get_config()
    # Gate on the model actually being thinking-capable. Without a
    # registered reasoning parser the engine never produces ``<think>``
    # tokens and the budget-burn failure mode does not apply — the
    # helper must be a no-op so a non-thinking model (llama / mistral /
    # qwen3-coder / …) keeps whatever resolution the chat-template
    # default would have applied.
    if not getattr(cfg, "reasoning_parser_name", None):
        return False
    # Codex r1 NIT: when the operator pinned ``--no-thinking`` at the
    # server level, ``_resolve_enable_thinking`` already forces False
    # downstream — so the auto-disable injection here is purely
    # cosmetic noise (extra log line, mutated request, AND on non-
    # qwen3 parsers it would feed the L-05 spurious warning below).
    # Short-circuit so the operator kill switch keeps a single resolution
    # site instead of two.
    if getattr(cfg, "no_thinking", False):
        return False
    # Codex r1 MEDIUM #1: ``tool_choice="none"`` defeats the casual
    # helper. The R12-T1F tools helper at line 1821-1823 correctly
    # SKIPS for the ``tool_choice="none"`` case (the model is told to
    # answer in prose, no tool_call expected) — but pre-fix the casual
    # helper then fell through and injected ``enable_thinking=False``
    # anyway, silently turning a Qwen3 prose-on-tool-defs request into
    # no-thinking. Mirror the tools-helper gate AND also skip when
    # ``tools`` is non-empty without a ``"none"`` choice, because that
    # branch is already owned by R12-T1F TOOLS-AUTO (composition: T1F
    # either fires and injects ``False`` itself OR skips because client
    # opted out; either way the casual helper has nothing to add). Net
    # effect: the casual-chat gate has the cleanest possible boundary
    # — it ONLY governs the no-tools prose path.
    if getattr(request, "tools", None):
        return False
    # Explicit thinking preference (top-level OR nested kwarg) wins.
    # Same precedence ``_extract_thinking_from_request`` uses across
    # the rest of the codebase.
    if _extract_thinking_from_request(request) is not None:
        return False
    # Explicit reasoning intent through any of the documented
    # signals. ``getattr`` with default ``None`` keeps the helper
    # tolerant of shapes that don't declare every field (e.g.
    # SimpleNamespace test shims, or future surfaces that omit
    # ``reasoning_effort`` but still call the helper). ``extra_signals``
    # (the optional secondary request shape) is consulted for the
    # SAME field set so a /v1/responses caller that pinned
    # ``reasoning={"effort":"low"}`` on the ResponsesRequest
    # short-circuits the gate even though the field never gets
    # forwarded onto the materialized ChatCompletionRequest.
    sources = [request]
    if extra_signals is not None and extra_signals is not request:
        sources.append(extra_signals)
    for src in sources:
        if getattr(src, "reasoning_max_tokens", None) is not None:
            return False
        if getattr(src, "reasoning_effort", None) is not None:
            return False
        # ``reasoning`` is the native Responses-API shape
        # (``{"effort": "low|medium|high", ...}``). Codex r1 MEDIUM #3:
        # gate specifically on a NON-NULL ``effort`` rather than "any
        # non-empty dict". The Responses spec allows
        # ``reasoning={"effort": null}`` (the schema's
        # ``_validate_reasoning_dict_effort`` explicitly lets ``None``
        # through) and ``reasoning={"summary": "auto"}`` (summary is a
        # SDK convenience flag, NOT a reasoning-intent signal). Pre-fix
        # both shapes were treated as opt-in and skipped the auto-
        # disable, leaving the default-thinking-ON budget-burn intact.
        # Now only an explicit non-null ``effort`` counts as
        # "client wants reasoning". Defensive ``isinstance(dict)`` so a
        # malformed payload that survived schema validation does not
        # silently lose the gate.
        reasoning = getattr(src, "reasoning", None)
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            return False
    existing_ctk = getattr(request, "chat_template_kwargs", None) or {}
    # Merge rather than replace so any non-thinking keys the client
    # passed (forward-compat, e.g. future kwargs the chat template
    # honors) survive untouched. Codex-r3 BLOCKING contract from M-2.
    merged_ctk = dict(existing_ctk)
    merged_ctk["enable_thinking"] = False
    request.chat_template_kwargs = merged_ctk
    # Codex r1 MEDIUM #2: mark the request so ``enable_thinking_warning_header``
    # can distinguish a SERVER-injected ``enable_thinking=False`` from a
    # client-supplied hint. Without this marker the L-05 warning would
    # fire spuriously on non-qwen3 parsers, telling the client "your
    # enable_thinking was ignored" even though the client never sent the
    # hint. Pydantic's private-attribute escape hatch (``_`` prefix)
    # is allowed by both the chat and responses request schemas, so
    # ``setattr`` on this name is safe across surfaces.
    _mark_thinking_auto_disabled(request)
    return True


def _mark_thinking_auto_disabled(request) -> None:
    """Tag ``request`` so the downstream L-05 warning header skips a
    server-injected ``chat_template_kwargs.enable_thinking=False``.

    Used by every auto-disable helper in the family — R12-M2 strict-json
    (responses.py inline), R12-T1F tools (this module's
    ``maybe_auto_disable_thinking_for_tools``), and R12-T2F casual chat
    (``maybe_auto_disable_thinking_for_casual_chat``). Single source of
    truth so a future surface that calls one of these helpers inherits
    the warning-suppression contract for free.

    Pydantic permits ``setattr`` on names with a leading underscore even
    on models declared with ``extra="forbid"``, so this works
    transparently on the typed ``ChatCompletionRequest`` /
    ``ResponsesRequest`` and on the ``SimpleNamespace`` shapes the unit
    tests use.
    """
    try:
        request._auto_disabled_thinking = True
    except Exception:
        # Defensive: a request shape that rejects private-attr setattr
        # (e.g. a slotted dataclass) falls through silently — the
        # warning header consults the attribute via ``getattr`` with a
        # default, so the worst case is the spurious-warning shape we
        # had pre-fix.
        pass


# L-05: the set of reasoning parsers that actually honor
# ``chat_template_kwargs.enable_thinking``. Only ``qwen3`` consults the
# flag as a strict on/off switch (its chat template skips the ``<think>``
# pre-injection when ``False``, and the parser's Case-4 fallback only
# fires under ``True``). All other registered parsers either:
#
#   * accept the flag for signature parity but ``del enable_thinking``
#     immediately (gemma4, gpt_oss, harmony, minimax, glm4), or
#   * only consult ``enable_thinking=True`` for Case-4 routing and
#     ignore ``False`` entirely (deepseek_r1, vibethinker, think_parser).
#
# When a client explicitly sets ``chat_template_kwargs.enable_thinking``
# on a server running a non-honoring parser, we surface the silent-drop
# via the ``X-RapidMLX-Warning`` response header rather than the previous
# zero-signal behavior. See L-05 in 0.8TODO.md for the dogfooding repro
# (Theo on phi-4-mini-reasoning → deepseek_r1 parser).
_THINKING_FLAG_HONORING_PARSERS: frozenset[str] = frozenset({"qwen3"})


def enable_thinking_warning_header(request, parser_name: str | None) -> dict[str, str]:
    """Build the response-header dict that surfaces a silent
    ``enable_thinking`` drop. Empty dict means "no warning needed".

    Conditions to fire (all must hold):
      1. The client EXPLICITLY set ``chat_template_kwargs.enable_thinking``
         (the OpenAI-extension key — the top-level ``enable_thinking``
         field is the rapid-mlx extension and is already auth-traceable
         via per-request docs, so we skip it to keep the surface narrow).
      2. The active reasoning parser is not in
         ``_THINKING_FLAG_HONORING_PARSERS``.

    The CLI ``--no-thinking`` mode does NOT silence the warning: an
    operator who pinned thinking off server-side is still receiving
    a request from a client that thinks the flag is doing something,
    and the client-facing signal is what L-05 is about.

    Returns a single-entry dict ``{"X-RapidMLX-Warning": "..."}`` so
    callers can ``**spread`` it into ``Response(headers=...)`` /
    ``StreamingResponse(headers=...)`` without conditional wiring.
    """
    if not parser_name:
        return {}
    if parser_name in _THINKING_FLAG_HONORING_PARSERS:
        return {}
    ctk = getattr(request, "chat_template_kwargs", None)
    if not isinstance(ctk, dict) or "enable_thinking" not in ctk:
        return {}
    # Codex r1 MEDIUM #2 (R12-T2F-276): when the auto-disable family
    # (R12-M2 strict-json / R12-T1F tools / R12-T2F casual chat)
    # injected ``chat_template_kwargs.enable_thinking=False`` server-
    # side, the L-05 warning ("your enable_thinking was ignored") is
    # actively misleading — the CLIENT never sent the hint, so there's
    # nothing to warn about. The auto-disable helpers tag the request
    # via ``_mark_thinking_auto_disabled`` for exactly this consult;
    # ``getattr`` with default ``False`` is back-compat for request
    # shapes that never went through a helper (e.g. legacy callers, or
    # the L-05 sibling tests that build a SimpleNamespace directly).
    if getattr(request, "_auto_disabled_thinking", False):
        return {}
    return {"X-RapidMLX-Warning": (f"enable_thinking ignored for parser={parser_name}")}


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
        # H-11: seed flows through the same cascade-resolver pattern so
        # all four routes (chat, completions, responses, anthropic) pick
        # it up automatically without each having to opt in. ``seed=0``
        # is a legitimate request value (PRNG seeds are routinely zero in
        # eval harnesses), so the ``value is not None`` gate below must
        # NOT collapse it.
        ("seed", _resolve_seed),
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
    """Validate tool call parameter values against their schemas (post-generation).

    F-141 scoped fix: enforce JSON-schema constraints on the model's
    emitted ``tool_calls[].function.arguments`` instead of merely
    logging. When a violation is detected we raise ``HTTPException(400)``
    so the caller can decide how to recover (retry with a stricter
    prompt, fall back to text, etc.) rather than silently propagating a
    schema-violating payload as a normal success. This matches the
    OpenAI contract: when a ``tool_calls[i]`` is present, its arguments
    are expected to satisfy the declared parameter schema.

    Enforced today (intentionally narrow, see ``validate_param_value``):
    ``type``, ``enum``, ``minimum``/``maximum``, ``minLength``/``maxLength``.
    Deferred (TODO(F-141-followup)): ``pattern``, ``format``,
    ``multipleOf``, ``uniqueItems``. Non-JSON ``arguments`` and non-dict
    parsed args remain warn-only because they indicate a model/parser
    issue rather than a schema violation, and the upstream parser layer
    already surfaces them.

    H-05 scope refactor: the iteration is strictly **per emitted call**.
    For each ``tc`` we look up the tool spec by its ``function.name``
    and validate the call's arguments against THAT spec's properties
    only. Tool specs the model did not call are never consulted — this
    makes "validate the called tool, not every declared tool" a
    structural invariant of the function instead of an emergent
    property of a keyed-schemas dict (which was functionally correct,
    just less self-evident: a future change to ``_extract_param_schemas``
    keying could silently re-introduce the cross-tool leak). A model
    emitting a call to a function not in ``tools`` is treated as
    schema-unknown (no constraint), mirroring the previous keyed-lookup
    behaviour.
    """
    from ..api.tool_logits import _extract_param_schemas, validate_param_value

    tool_defs = [t.model_dump() if hasattr(t, "model_dump") else t for t in tools]

    # Per-tool index: name -> {param_name: schema}. Reuses
    # ``_extract_param_schemas`` on a single-tool list so the schema
    # normalisation logic (handles ``parameters`` being null /
    # non-dict, ``properties`` being non-dict, etc.) stays in exactly
    # one place. Strip the ``<name>.`` prefix from the keys so the
    # per-call inner loop only does a ``param_name`` lookup against
    # the tool we actually matched.
    tool_by_name: dict[str, dict] = {}
    for tool in tool_defs:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", tool)
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        if not name:
            continue
        scoped = _extract_param_schemas([tool])
        tool_by_name[name] = {k.split(".", 1)[1]: v for k, v in scoped.items()}

    for tc in tool_calls:
        func = tc.function if hasattr(tc, "function") else tc.get("function", {})
        func_name = func.name if hasattr(func, "name") else func.get("name", "")
        args_str = (
            func.arguments
            if hasattr(func, "arguments")
            else func.get("arguments", "{}")
        )

        # Find the called tool's schema. If the model called a tool not
        # in ``tools`` (parser hallucination), skip — the upstream
        # tool_choice / parser layers own that case; we have no schema
        # to validate against here.
        called_tool_schemas = tool_by_name.get(func_name)
        if called_tool_schemas is None:
            continue

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
            schema = called_tool_schemas.get(param_name)
            if not schema:
                continue
            is_valid, error = validate_param_value(json.dumps(param_value), schema)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Tool call '{func_name}' parameter '{param_name}' "
                        f"violates declared schema: {error}. The model "
                        "produced a schema-violating argument value; retry "
                        "with a more constrained prompt or relax the schema."
                    ),
                )


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


def _resolve_sync_scheduler_for_abort(engine):
    """C-01 codex r1 BLOCKING #2 helper: find the SYNC scheduler-side
    ``abort_request`` entry point for the **currently active**
    backend.

    The codex reviewer pointed out that ``engine.abort_request`` may
    be a coroutine (``BatchedEngine.abort_request`` when the LLM path
    is loaded, because ``AsyncEngineCore.abort_request`` is async).
    Fire-and-forget via ``asyncio.ensure_future`` doesn't actually
    free the GPU on the next ``step()`` — the coroutine has to run
    to reach ``scheduler.abort_request`` (which IS sync). So we walk
    the engine's backend graph to find that sync entry point
    directly.

    codex r2 BLOCKING #1: the walk MUST respect the engine's active
    path. ``BatchedEngine`` declares BOTH ``_engine`` (AsyncEngineCore
    for the text path) AND ``_mllm_scheduler`` (MLLM path) as
    instance attributes. They start at ``None`` and get populated by
    ``start()`` based on the model's modality — but only ONE is the
    active backend the live ``request_id`` was admitted into. Calling
    ``_mllm_scheduler.abort_request(rid)`` on a request that lives in
    ``_engine.scheduler`` would enqueue the abort into the wrong
    pending set and leave the real request running. The
    ``_is_mllm`` flag is the canonical active-path signal — same
    predicate ``stream_generate`` uses to pick which backend to call
    in the first place.

    Resolution order:

      * ``engine.scheduler``                          — plain engines
                                                       exposing the
                                                       scheduler
                                                       directly (eg.
                                                       AsyncEngineCore
                                                       passed in
                                                       unwrapped).
      * MLLM-active: ``engine._mllm_scheduler``       — only when
                                                       ``engine._is_mllm``
                                                       is True.
      * text-active: ``engine._engine.scheduler``     — only when
                                                       ``engine._is_mllm``
                                                       is False (or
                                                       absent).

    Returns the first callable found, or ``None`` when none of the
    paths resolve (caller falls back to the public async
    ``engine.abort_request`` as a last resort with documented
    fire-and-forget semantics).
    """
    # Plain engines that expose .scheduler directly (no active-path
    # ambiguity). Includes AsyncEngineCore and the fake engines used
    # in tests.
    direct_scheduler = getattr(engine, "scheduler", None)
    if direct_scheduler is not None:
        abort = getattr(direct_scheduler, "abort_request", None)
        if abort is not None and not asyncio.iscoroutinefunction(abort):
            return abort
    # BatchedEngine and similar wrappers — gate on the active path.
    # Default to text-active (``_is_mllm = False``) when the flag is
    # missing so older engine shapes still resolve sanely.
    is_mllm_active = bool(getattr(engine, "_is_mllm", False))
    if is_mllm_active:
        mllm = getattr(engine, "_mllm_scheduler", None)
        if mllm is not None:
            abort = getattr(mllm, "abort_request", None)
            if abort is not None and not asyncio.iscoroutinefunction(abort):
                return abort
    else:
        inner = getattr(engine, "_engine", None)
        if inner is not None:
            # ``engine._engine.scheduler`` — synthetic test-stub shape
            # where AsyncEngineCore-like is stubbed with a direct
            # ``.scheduler`` attribute. Kept for back-compat with the
            # existing pre-D-M01 test corpus.
            inner_sched = getattr(inner, "scheduler", None)
            if inner_sched is not None:
                abort = getattr(inner_sched, "abort_request", None)
                if abort is not None and not asyncio.iscoroutinefunction(abort):
                    return abort
            # D-M01-DEAD (0.8.2 dogfood): the PRODUCTION ``rapid-mlx
            # serve`` shape is ``BatchedEngine._engine`` →
            # ``AsyncEngineCore.engine`` → ``EngineCore.scheduler``.
            # ``AsyncEngineCore`` does NOT expose ``.scheduler``
            # directly — its scheduler lives behind ``self.engine``
            # which is an ``EngineCore``. Without this extra hop the
            # sync resolver returns ``None`` on every real production
            # engine and ``_force_abort_request`` falls into the
            # async fallback. That fallback awaits
            # ``EngineCore.abort_request`` which interleaves
            # ``scheduler.abort_request`` + ``_cleanup_request``
            # (which wipes the lifetime ledger), racing the
            # attribution helper and causing the 2x over-count +
            # flat-zero via_disconnect sub-counter that three
            # 0.8.2 personas independently reported.
            #
            # Mirror the same deep-path walk
            # ``_resolve_disconnect_abort_recorder`` already does so
            # the sync abort and the attribution call resolve to the
            # SAME ``Scheduler`` instance on every backend shape.
            inner_engine = getattr(inner, "engine", None)
            if inner_engine is not None:
                deep_sched = getattr(inner_engine, "scheduler", None)
                if deep_sched is not None:
                    abort = getattr(deep_sched, "abort_request", None)
                    if abort is not None and not asyncio.iscoroutinefunction(abort):
                        return abort
    return None


def _resolve_disconnect_abort_recorder(engine):
    """M-01: walk the engine backend graph and return the bound
    ``record_disconnect_abort`` method of the active-path scheduler.

    Codex r2 NIT: returns the BOUND METHOD, not the scheduler object,
    so the call site stays one expression
    (``recorder(rid)``) without having to know which attribute name
    to look up. The name is honest about that — earlier draft was
    ``_resolve_scheduler_for_cancel_attribution`` which suggested a
    scheduler object would be returned.

    The cancel-attribution sub-counter lives on the same scheduler
    where the public-API total counter lives, so the resolver must
    follow the active-path gate (``_is_mllm``). We deliberately keep
    this separate from ``_resolve_sync_scheduler_for_abort`` (which
    gates on ``not iscoroutinefunction(abort_request)``) because the
    sub-counter is a sync-only method wholly independent of the abort
    path — re-using the abort resolver would skip schedulers that
    expose a fine sync ``record_disconnect_abort`` but happen to have
    an async ``abort_request`` shim.

    Production text path walks one extra level versus the abort
    resolver: ``BatchedEngine._engine`` is ``AsyncEngineCore`` (which
    does NOT expose ``.scheduler`` directly) and the actual
    ``Scheduler`` lives at ``BatchedEngine._engine.engine.scheduler``
    (``AsyncEngineCore`` wraps ``EngineCore`` via ``self.engine``).
    The abort resolver doesn't dig that deep because
    ``AsyncEngineCore`` has its own async ``abort_request`` shim and
    falls through to the public-async fallback — but the
    attribution counter must NOT fall back because the sub-counter
    only lives on ``Scheduler``. Without this extra hop the
    via_disconnect series would stay flat through every real
    production disconnect (Mei + Yana surfaced the cancel-rate
    counter; the operator would still have no way to see whether
    the bulk of cancels came from disconnect vs. timeout vs.
    explicit /cancel).

    Returns ``None`` if no scheduler in the active backend exposes
    the method, in which case the helper-layer caller silently
    no-ops (older schedulers without M-01 simply don't surface the
    sub-counter, the total counter on the route side still defaults
    to zero).
    """
    # M-01 codex r7 BLOCKING #2: when ``_is_mllm`` is set we MUST
    # honor it before falling back to a direct ``engine.scheduler``
    # — otherwise a dual-shaped engine (one that exposes
    # ``.scheduler`` for the text path AND ``_mllm_scheduler`` for
    # the MLLM path, depending on which backend is active) will
    # mis-attribute disconnects to the text scheduler when the
    # MLLM path is the live one. The check pattern is the same as
    # the abort resolver's codex r2 BLOCKING #1 fix — active-path
    # gate first, plain-engine fallback last.
    is_mllm_flag = getattr(engine, "_is_mllm", None)
    if is_mllm_flag is True:
        mllm = getattr(engine, "_mllm_scheduler", None)
        if mllm is not None:
            recorder = getattr(mllm, "record_disconnect_abort", None)
            if recorder is not None:
                return recorder
    elif is_mllm_flag is False:
        inner = getattr(engine, "_engine", None)
        if inner is not None:
            # ``engine._engine.scheduler`` — matches the abort
            # resolver shape (used by every test stub today).
            inner_sched = getattr(inner, "scheduler", None)
            if inner_sched is not None:
                recorder = getattr(inner_sched, "record_disconnect_abort", None)
                if recorder is not None:
                    return recorder
            # ``engine._engine.engine.scheduler`` — the production
            # ``BatchedEngine`` over ``AsyncEngineCore`` shape. The
            # extra hop is where AsyncEngineCore wraps EngineCore.
            inner_engine = getattr(inner, "engine", None)
            if inner_engine is not None:
                deep_sched = getattr(inner_engine, "scheduler", None)
                if deep_sched is not None:
                    recorder = getattr(deep_sched, "record_disconnect_abort", None)
                    if recorder is not None:
                        return recorder
    # Plain-engine fallback: only used when ``_is_mllm`` is absent
    # (older / external engine shapes that pre-date BatchedEngine).
    # Distinct from ``is_mllm_flag is False`` above, which means
    # "explicitly text-active on a BatchedEngine".
    direct_scheduler = getattr(engine, "scheduler", None)
    if direct_scheduler is not None:
        recorder = getattr(direct_scheduler, "record_disconnect_abort", None)
        if recorder is not None:
            return recorder
    return None


# M-01 / D-M01-DEAD: once-per-engine-type ledger for the unresolved-
# engine warning. Without this dedupe, an operator running on an
# engine shape the resolvers don't yet recognise would see one
# warning per ABORT — at 1 cancel/second that's 86,400 warnings/day,
# enough to drown the disconnect_guard log signal entirely. The
# warning is informational ("which backend shape do the resolvers
# need to learn") so once-per-engine-type is the right cardinality.
# Bounded by the number of distinct engine classes in the process,
# typically 1.
_unresolved_engine_logged: set[tuple[str, str]] = set()
_unresolved_engine_lock = threading.Lock()


def _unresolved_engine_dedupe_key(engine_cls: type) -> tuple[str, str]:
    """Codex r11 NIT: dedupe by the fully-qualified class identity
    so two distinct unresolved engine classes that happen to share
    a leaf name (e.g. ``mod_a.BatchedEngine`` vs.
    ``mod_b.BatchedEngine``) do NOT suppress each other's warning.
    ``__qualname__`` covers nested classes; pairing with
    ``__module__`` makes the key bijective with the class identity
    for any sensibly-defined production class. Falling back to the
    class object itself for stragglers (lambda-defined / no
    qualname) would also work but a tuple is cheaper to hash and
    easier to inspect in a heap dump.
    """
    module = getattr(engine_cls, "__module__", "<unknown>") or "<unknown>"
    qualname = getattr(engine_cls, "__qualname__", engine_cls.__name__)
    return (module, qualname)


def _record_disconnect_abort_on_scheduler(engine, request_id) -> None:
    """M-01 attribution helper — bumps the disconnect sub-counter on
    whichever active-path scheduler owns this ``request_id``.

    Wraps the resolver in a try/except so the cancel-attribution path
    never leaks back into ``_force_abort_request``. The total counter
    is already incremented inside ``Scheduler.abort_request`` the
    moment the sync entry returns True, so failure of this attribution
    helper at worst leaves the (total - disconnect) gap one larger
    than reality — never breaks the abort itself.

    D-M01-DEAD (0.8.2 dogfood): the previous implementation silently
    no-op'd when ``_resolve_disconnect_abort_recorder`` returned
    ``None``. That swallow-with-no-log let PR #783 ship without
    catching that the production ``BatchedEngine`` over
    ``AsyncEngineCore`` shape exposed no recorder — three personas
    independently observed flat-zero ``via_disconnect_total`` on
    PyPI 0.8.2. The WARNING below (rate-limited to once-per-engine-
    type so it doesn't drown the log under sustained cancel traffic)
    ensures the next engine-shape change cannot silently regress the
    sub-counter again. The high-cardinality ``request_id`` is logged
    at DEBUG only — codex r10 NIT.
    """
    try:
        recorder = _resolve_disconnect_abort_recorder(engine)
        if recorder is None:
            engine_cls = type(engine)
            dedupe_key = _unresolved_engine_dedupe_key(engine_cls)
            # Display name is "module.qualname" so the operator log
            # carries the full backend identity, not a leaf name that
            # might collide across modules.
            engine_display = f"{dedupe_key[0]}.{dedupe_key[1]}"
            should_warn = False
            with _unresolved_engine_lock:
                if dedupe_key not in _unresolved_engine_logged:
                    _unresolved_engine_logged.add(dedupe_key)
                    should_warn = True
            if should_warn:
                logger.warning(
                    "[disconnect_guard] no record_disconnect_abort recorder "
                    "found on engine type=%s — via_disconnect sub-counter "
                    "WILL NOT advance for aborts routed through this engine. "
                    "This indicates an unrecognised engine shape; the "
                    "resolvers in _resolve_disconnect_abort_recorder + "
                    "_resolve_sync_scheduler_for_abort must learn the new "
                    "backend graph. Further occurrences for this engine "
                    "type will be suppressed at DEBUG.",
                    engine_display,
                )
            else:
                logger.debug(
                    "[disconnect_guard] no recorder for engine type=%s "
                    "(request_id=%s) — warning already emitted",
                    engine_display,
                    str(request_id)[:12] if request_id else request_id,
                )
            return
        recorder(request_id)
    except Exception:  # pragma: no cover - belt-and-suspenders
        logger.warning(
            "[disconnect_guard] record_disconnect_abort raised; "
            "via_disconnect sub-counter may under-count this abort",
            exc_info=True,
        )


def _force_abort_request(engine, request_id_holder) -> bool:
    """C-01: synchronously enqueue an abort for the live request id.

    Looks up ``request_id_holder[0]`` (populated by
    ``BatchedEngine.stream_generate`` once ``add_request`` returns)
    and invokes the loaded backend's SYNC
    ``scheduler.abort_request(rid)`` — a thread-safe, non-blocking
    ``set.add`` per ``Scheduler.abort_request`` docstring. The
    sync-path resolver (``_resolve_sync_scheduler_for_abort``) walks
    ``engine.scheduler`` first, then the active-backend branch on
    ``BatchedEngine`` (gated by ``engine._is_mllm`` per codex r2
    BLOCKING #1): MLLM → ``engine._mllm_scheduler``, text →
    ``engine._engine.scheduler``.

    Falls back to ``engine.abort_request(rid)`` (the public engine
    surface — may be async on engines that haven't been refactored)
    only when no sync path exists. In that case the async coroutine
    is scheduled with ``asyncio.ensure_future`` so the caller stays
    synchronous, and we log a warning so operators can see that the
    abort is NOT guaranteed to be in flight by the time the
    disconnect path returns (codex r1 BLOCKING #2). The cascade via
    ``generator.aclose()`` remains as the safety net for that case.

    Returns ``True`` when a sync abort was issued (force-abort
    contract satisfied), ``False`` for the no-op cases (holder unset,
    engine missing, no request id yet) AND for the
    async-fallback-only case where the abort was scheduled but
    didn't reach the scheduler synchronously. Never raises — log on
    the warning channel and swallow so disconnect handling never
    derails on an engine-side error.
    """
    if request_id_holder is None or engine is None:
        return False
    try:
        request_id = request_id_holder[0]
    except (IndexError, TypeError):
        return False
    if not request_id:
        return False
    try:
        sync_abort = _resolve_sync_scheduler_for_abort(engine)
        if sync_abort is not None:
            result = sync_abort(request_id)
            logger.info(
                f"[disconnect_guard] force-abort scheduler.abort_request("
                f"{str(request_id)[:12]}) -> {result}"
            )
            # M-01: attribute the abort to client disconnect so
            # ``rapid_mlx_requests_cancelled_via_disconnect_total`` ticks
            # alongside the total. We bump only when the sync entry
            # actually accepted the abort (``result == True``) — the
            # scheduler returns False for unknown ids and we must not
            # over-count those. The disconnect_guard fires
            # ``_force_abort_request`` from up to three branches per
            # disconnect (disconnect/GeneratorExit/finally), so the
            # scheduler-side ``_disconnect_abort_ids`` set provides
            # the once-per-request de-dup. Observability only — never
            # let a counter-side error mask the actual abort.
            if result:
                _record_disconnect_abort_on_scheduler(engine, request_id)
            return True
        # No sync path resolved — the engine is wrapping its scheduler
        # behind an async-only ``abort_request``. Best-effort fire-and-
        # forget; return ``False`` so the contract reflects that the
        # abort did NOT land synchronously and the cascade through
        # ``generator.aclose()`` is the operative defense (codex r1
        # BLOCKING #2). Tests that pin the "force-abort fired sync"
        # contract should fail loudly on this fallback path.
        public_abort = getattr(engine, "abort_request", None)
        if public_abort is not None:
            result = public_abort(request_id)
            if asyncio.iscoroutine(result):
                # M-01 codex r1 BLOCKING #1: attribution MUST chain on
                # the abort result, not fire-and-forget. Pre-fix this
                # path called ``_record_disconnect_abort_on_scheduler``
                # IMMEDIATELY after scheduling the coroutine, so a
                # stale / unknown ``request_id`` (which makes
                # ``Scheduler.abort_request`` return False and skip
                # the total counter) would still tick the sub-counter
                # — corrupting the (total - via_disconnect) gap
                # operators rely on for cause attribution. The fix:
                # wrap the coroutine in an awaiter that records ONLY
                # after the awaited abort returns truthy. The
                # ``ensure_future`` still keeps the disconnect path
                # synchronous (we don't await it here), but the
                # eventual coroutine resolution is now the gate for
                # the sub-counter.
                attribution_engine = engine

                async def _await_and_record(coro=result, rid=request_id):
                    try:
                        accepted = await coro
                    except Exception:  # pragma: no cover - belt-and-suspenders
                        logger.warning(
                            "[disconnect_guard] async force-abort raised; "
                            "via_disconnect sub-counter NOT advanced for "
                            f"{str(rid)[:12]}",
                            exc_info=True,
                        )
                        return
                    if accepted:
                        _record_disconnect_abort_on_scheduler(attribution_engine, rid)
                    else:
                        # Codex r2 NIT #3: surface the "abort coroutine
                        # returned False" path to operators. Without
                        # this log, a stale holder (request_id reused
                        # or already finished by the time the
                        # disconnect_guard fired) silently leaves the
                        # ``via_disconnect`` sub-counter behind the
                        # total, and the operator-facing
                        # (total - via_disconnect) gap drifts with no
                        # diagnostic trail. INFO-level — the sync
                        # path also INFO-logs its rejected aborts.
                        logger.info(
                            "[disconnect_guard] async force-abort returned "
                            "False for %s; via_disconnect sub-counter NOT "
                            "advanced (stale holder / already finished?)",
                            str(rid)[:12],
                        )

                asyncio.ensure_future(_await_and_record())
                logger.warning(
                    f"[disconnect_guard] force-abort fell back to async "
                    f"engine.abort_request({str(request_id)[:12]}); the "
                    f"scheduler abort is NOT guaranteed in flight by the "
                    f"time disconnect handling returns. The "
                    f"generator-close cascade is the remaining defense."
                )
                # ``False`` so the contract reflects async-fallback;
                # callers / tests treat that as "I tried, cascade is
                # the remaining defense".
                return False
            logger.info(
                f"[disconnect_guard] force-abort engine.abort_request("
                f"{str(request_id)[:12]}) -> {result}"
            )
            # Sync public-abort path (no coroutine to schedule) —
            # attribute disconnect symmetric with the sync-scheduler
            # branch above when the engine accepted the abort.
            if result:
                _record_disconnect_abort_on_scheduler(engine, request_id)
            return True
    except Exception:
        logger.warning(
            "[disconnect_guard] force-abort raised; falling back to "
            "generator-close cascade",
            exc_info=True,
        )
    return False


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
    engine=None,
    keepalive_seconds: float | None = None,
    request_id_holder: list | None = None,
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect.

    When ``engine`` is provided, releases its admission reservation in
    the ``finally`` clause so the slot acquired by
    ``_check_admission_or_503`` is returned to the pool once the
    streaming response finishes (or the client disconnects, or the
    generator raises). The release is the safety net for the
    streaming path; non-streaming routes mirror it via
    ``_wait_with_disconnect``.

    SSE keepalive (F-070): when ``keepalive_seconds > 0`` (default
    falls through to ``ServerConfig.sse_keepalive_seconds``), emit an
    SSE comment line (``: keepalive\\n\\n``) whenever the upstream
    generator stalls for that many seconds without producing a chunk.
    SSE comments start with ``:`` per the WHATWG spec, are dropped by
    every conforming consumer (``EventSource``, OpenAI SDK), and
    serve as TCP-level heartbeats that prevent intermediate proxies
    (nginx ``proxy_read_timeout=60``, Cloudflare 100 s, EventSource
    ~45 s) from tearing down the connection during long prefills. Set
    ``keepalive_seconds=0`` or ``RAPID_MLX_SSE_KEEPALIVE_SECONDS=0``
    to disable.

    C-01 force-abort (Astrid r3 dogfooding): when
    ``request_id_holder`` is a mutable list AND the engine publishes
    the admitted scheduler ``request_id`` into ``holder[0]``, the
    guard force-calls ``engine.scheduler.abort_request(holder[0])`` (or
    ``engine.abort_request(holder[0])`` as a thread-safe alternative)
    the moment a client disconnect is detected. Pre-C-01 the abort
    relied entirely on the generator-close cascade
    (``generator.aclose()`` in the ``finally`` → ``GeneratorExit``
    propagates upstream → ``stream_generate.finally`` calls
    ``scheduler.abort_request``). That cascade still runs as a
    belt-and-suspenders, but the EXPLICIT abort here closes Astrid's
    failure mode where a runaway no-EOS generation kept consuming GPU
    for >35 s after the client TCP-RST'd because nothing on the path
    actually reached into the scheduler with an abort signal until
    the generation hit its token cap. ``holder=None`` (default)
    preserves the pre-C-01 contract for non-streaming callers and for
    cloud-routed streams that don't have a local scheduler request
    id.
    """
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    # Resolve keepalive interval from the live ServerConfig at start
    # time. Per-call ``keepalive_seconds`` argument wins (lets
    # ``stream_chat_completion_guided`` and tests pin a value); else
    # consult the config singleton. A non-positive value disables
    # heartbeats entirely.
    if keepalive_seconds is None:
        try:
            keepalive_seconds = float(get_config().sse_keepalive_seconds)
        except Exception:
            keepalive_seconds = 20.0
    keepalive_enabled = keepalive_seconds and keepalive_seconds > 0

    logger.info(
        f"[disconnect_guard] START poll_interval={poll_interval}s "
        f"keepalive_seconds={keepalive_seconds}"
    )

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
    keepalive_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    # C-01 codex r1 NIT #3: track whether we got to a clean
    # ``StopAsyncIteration`` exhaust (normal stream end). When True,
    # the ``finally`` block skips the belt-and-suspenders force-abort —
    # otherwise every successful response would enqueue a spurious
    # abort against an already-finished request id, making the abort
    # logs/metrics indistinguishable from real disconnect cleanup.
    finished_normally = False
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        # Single in-flight ``__anext__`` future at any time. We
        # re-create it at the TOP of each iteration (NOT eagerly after
        # ``yield chunk``) so each iteration begins with the consumer
        # having pulled the previous chunk before we schedule the next
        # one. Codex r3 BLOCKING on PR #732: eagerly scheduling the
        # next anext_task right after ``yield`` lets the upstream
        # generator advance one token ahead of the response stream,
        # wasting inference work if the consumer is about to
        # disconnect. Lazy re-creation keeps the generator at most one
        # token ahead — and that token is the one whose yield we're
        # already awaiting from the wait below.
        #
        # The ``anext_task`` reference is preserved across keepalive
        # ticks: a keepalive cycle returns to the loop top without
        # consuming the still-pending anext, so the upstream's work in
        # flight (the long prefill) is not cancelled mid-step.
        while True:
            # Lazy (re)create the in-flight ``__anext__``:
            #   * first iteration: ``anext_task`` is None.
            #   * after a real chunk: ``anext_task.done()`` is True
            #     (we consumed ``.result()`` last iteration), and now
            #     the consumer has pulled the yielded chunk — safe to
            #     ask the upstream for another token.
            #   * during a keepalive cycle: ``anext_task.done()`` is
            #     False (upstream still mid-prefill), so we keep the
            #     existing future. The wait below ignores it.
            if anext_task is None or anext_task.done():
                anext_task = asyncio.ensure_future(aiter.__anext__())
            wait_kwargs: dict = {"return_when": asyncio.FIRST_COMPLETED}
            if keepalive_enabled:
                wait_kwargs["timeout"] = keepalive_seconds
            done, _pending = await asyncio.wait(
                [anext_task, disconnect_task],
                **wait_kwargs,
            )
            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks ({keepalive_count} keepalives), "
                    f"elapsed={_elapsed()}"
                )
                # C-01 force-abort: call into the scheduler directly
                # the moment the disconnect signal fires, BEFORE we
                # cancel the in-flight ``anext_task`` and close the
                # upstream generator. Pre-C-01 the abort relied solely
                # on the close cascade kicking ``stream_generate.finally``
                # → ``scheduler.abort_request``; Astrid r3 showed that
                # path can stall for ~35s under a runaway no-EOS
                # generation because the cancellation of ``anext_task``
                # only interrupts the in-flight ``__anext__``, while
                # the upstream batch step continues to chew GPU until
                # the next yield boundary. Hitting the scheduler's
                # ``_pending_abort_ids`` set NOW means the very next
                # ``step()`` drops the request and frees the slot. This
                # is purely defense in depth — the cascade still runs
                # via ``generator.aclose()`` in ``finally`` and
                # ``scheduler.abort_request`` is idempotent against
                # double-enqueue.
                _force_abort_request(engine, request_id_holder)
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break
            if anext_task not in done:
                # Neither completed within ``keepalive_seconds``: the
                # upstream generator is still working (e.g. mid-prefill
                # on a 64k-token prompt). Emit an SSE comment line so
                # the connection stays alive across proxies + browser
                # EventSource. The comment is a no-op to the parsed
                # event stream — F-070 fix.
                keepalive_count += 1
                if keepalive_count == 1 or keepalive_count % 5 == 0:
                    logger.info(
                        f"[disconnect_guard] emitting keepalive "
                        f"#{keepalive_count}, elapsed={_elapsed()}"
                    )
                yield ": keepalive\n\n"
                continue
            try:
                chunk = anext_task.result()
            except StopAsyncIteration:
                logger.info(
                    f"[disconnect_guard] generator exhausted normally, "
                    f"{chunk_count} chunks ({keepalive_count} keepalives), "
                    f"elapsed={_elapsed()}"
                )
                # C-01 codex r1 NIT #3: mark the clean-exit case so
                # ``finally`` skips the force-abort. The upstream
                # generator already drained, the scheduler already
                # marked the request as finished — a defensive abort
                # here would only pollute logs.
                finished_normally = True
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
            # Loop top will re-create the anext_task now that the
            # consumer has pulled this chunk. Do NOT eagerly schedule
            # the next ``__anext__`` here — see the docstring at loop
            # entry for the rationale (codex r3 BLOCKING).
    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
        # Codex r3 BLOCKING follow-up: ensure any in-flight
        # ``anext_task`` is cancelled the moment the consumer aborts
        # the iteration — the ``finally`` block below also cancels,
        # but doing it here makes the ordering explicit and avoids a
        # tiny window where the upstream might advance one more token
        # before the cleanup runs.
        if anext_task is not None and not anext_task.done():
            anext_task.cancel()
        # C-01: GeneratorExit means the consumer (Starlette's
        # ``StreamingResponse`` task) is tearing down — usually
        # because uvicorn detected a write failure to the closed
        # socket. Force-abort the scheduler request before we close
        # the generator so the upstream doesn't get one more
        # token-worth of GPU work between here and the cascade.
        _force_abort_request(engine, request_id_holder)
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        # Drive the cascade first: ``generator.aclose()`` propagates
        # ``GeneratorExit`` into the upstream so its ``finally`` runs
        # (calls ``stream_generate.finally`` → ``scheduler.abort_request``).
        # Then the belt-and-suspenders below covers the case where the
        # cascade itself raised or the upstream's finally didn't reach
        # the abort. Ordering ``aclose()`` before the belt-and-
        # suspenders also gives the test layer a clean observable
        # discriminator between the ``except GeneratorExit`` branch
        # (which runs BEFORE the cascade) and this finally fallback
        # (which runs AFTER the cascade).
        try:
            await generator.aclose()
        except Exception:
            pass
        # C-01 belt-and-suspenders: cover the abnormal-exit paths the
        # explicit ``if disconnect_task in done`` and ``except
        # GeneratorExit`` branches don't see — e.g. an exception
        # raised mid-stream that escaped the ``except Exception`` inline
        # handler. Codex r1 NIT #3 fix: skip when the stream
        # exhausted cleanly via ``StopAsyncIteration`` — the upstream
        # generator already finished and the scheduler already marked
        # the request finished, so a defensive abort here would just
        # pollute logs without freeing any GPU work. The
        # ``Scheduler.abort_request`` idempotency contract still
        # protects against the case where this fires concurrently
        # with the cascade.
        if not finished_normally:
            _force_abort_request(engine, request_id_holder)
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


def _build_prompt_with_thinking_compat(
    build_prompt,
    messages: list,
    *,
    tools: list | None,
    enable_thinking: bool | None,
):
    """Call ``engine.build_prompt`` with ``enable_thinking=...``,
    falling back to the pre-#280 two-argument shape when the callable
    doesn't accept the new kwarg.

    Rationale (codex r1 BLOCKING on PR #906): adding the
    ``enable_thinking`` parameter to the helper's forwarded call
    breaks backward compatibility for any third-party engine or test
    double still on the documented pre-#280 shape
    ``build_prompt(messages, tools=None)``. Without this compat
    shim two failure modes leak in:

      * In :func:`enforce_context_length_for_messages` the resulting
        ``TypeError`` is caught by the broad fallthrough at the
        bottom of the try-except and silently returns ``None`` —
        disabling the DoS gate for that engine.
      * In :func:`repair_messages_fit_context` the broad
        ``except Exception`` turns the same ``TypeError`` into
        ``return True``, skipping the actual fit check.

    Both shapes are silent regressions, so we prefer the call shape
    that matches the new contract but transparently fall back to
    the legacy shape if the callable raises ``TypeError`` on the
    unexpected kwarg. The fallback drops ``enable_thinking`` (the
    legacy engine never honoured it anyway, so the rendered prompt
    matches the legacy behaviour exactly — same prompt the engine
    would have produced pre-fix). Other ``TypeError``s propagate so
    they continue to surface as user-facing errors via the helper's
    existing exception sniff.

    The probe uses ``inspect.signature`` lazily (only on TypeError)
    so the hot path stays a single direct call; the fallback path
    only fires on the first call for an engine with the legacy
    signature.
    """
    try:
        return build_prompt(messages, tools=tools, enable_thinking=enable_thinking)
    except TypeError as exc:
        # ``TypeError`` from kwargs mismatch carries the offending
        # kwarg name in its message under CPython. Be conservative:
        # only fall back when the message clearly says the kwarg is
        # the problem, otherwise re-raise so a real bug surfaces.
        msg = str(exc)
        if "enable_thinking" not in msg or "unexpected keyword" not in msg.lower():
            raise
        # Drop the new kwarg and call the legacy shape. The legacy
        # engine renders with its template default (typically ``True``
        # on Qwen3 / DeepSeek-R1), which matches the pre-fix gate
        # behaviour exactly — so callers that hit this fallback see
        # no regression vs the world before PR #906.
        return build_prompt(messages, tools=tools)


def enforce_context_length_for_messages(
    engine,
    messages: list,
    *,
    tools: list | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> int | None:
    """Run the context-length gate for a chat-style request and return
    the rendered prompt's token count (``None`` on permissive-skip paths).

    Renders the prompt through the engine's chat template (same path
    used by ``BatchedEngine.build_prompt``), counts the tokens, then
    delegates to :func:`enforce_context_length`. Wraps the template /
    tokenization step in a permissive try-except so a metadata edge
    case (e.g. unloaded engine on a route stub) doesn't 500 — the
    downstream scheduler still has its own validation.

    Returns the integer prompt-token count so callers that already pay
    the build_prompt + tokenize cost can reuse it (e.g. Anthropic
    streaming usage's ``message_start.input_tokens`` plumbing). Returns
    ``None`` when the render or tokenization was skipped — MLLM engine,
    no ``build_prompt`` attribute, empty rendered prompt, or
    tokenizer-returned-zero — so callers can distinguish "no estimate
    available" from "real zero count" without re-parsing a sentinel
    overload (codex r4 NIT on PR #807). Existing call sites that
    discard the return value keep working unchanged — this is an
    additive contract change.

    Scoped to text-only engines: MLLM models accept image / video /
    audio inputs whose token cost is computed by the multimodal
    processor and tracked separately by ``MLLMScheduler``. The
    body-size middleware still bounds the wire-level payload for
    those routes.

    ``enable_thinking`` is forwarded to the engine's ``build_prompt``
    so the rendered template matches what the engine will actually
    generate against. Required for correct accounting on the
    R12-T1F / R12-T2F / R12-M2 auto-disable paths (PR #891 / #877 /
    #895): the chat / responses routes inject
    ``chat_template_kwargs.enable_thinking=False`` BEFORE this gate
    runs, but pre-fix the gate rendered with the template default
    (typically ``True`` on Qwen3 / DeepSeek-R1) — inflating the
    prompt-token estimate vs the prompt the engine actually emits.
    Two visible symptoms before the fix:

      * Requests that DO fit the model's context window were rejected
        with ``context_length_exceeded`` because the inflated estimate
        crossed the cap.
      * The H-06 #267b strict-json-schema repair gate
        (:func:`repair_messages_fit_context`) skipped retries that
        WOULD have fit if rendered with the resolved
        ``enable_thinking`` value.

    Default ``None`` preserves the legacy "let the template choose"
    behaviour for call sites that haven't been audited yet (e.g.
    ``routes/anthropic.py``, which does not run the auto-disable
    injection and so never had the divergence). Always thread the
    resolved value from sites that compute it.

    Used by chat, anthropic, and responses routes so the same DoS gate
    applies regardless of which compatibility surface the client uses.
    """
    if getattr(engine, "is_mllm", False):
        return None
    build_prompt = getattr(engine, "build_prompt", None)
    if build_prompt is None:
        return None
    try:
        prompt = _build_prompt_with_thinking_compat(
            build_prompt,
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
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
        return None
    if not prompt:
        return None
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return None
    enforce_context_length(engine, prompt_tokens, max_tokens=max_tokens)
    return prompt_tokens


def repair_messages_fit_context(
    engine,
    repair_messages: list,
    *,
    tools: list | None = None,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> bool:
    """Re-check the context-length gate for the R12-4 strict-mode
    repair-retry prompt (H-06 #267b).

    Codex review on PR #878 surfaced that ``build_repair_messages``
    builds a strictly LARGER prompt than the initial request — it
    prepends repair instructions, repeats the schema, and includes
    up to 4 KiB of the failed output. A request that passed the
    initial ``enforce_context_length_for_messages`` gate can therefore
    blow the context window only on the repair attempt and surface
    as ``502 strict_repair_engine_failure`` instead of a deterministic
    422 validation outcome.

    This helper mirrors :func:`enforce_context_length_for_messages`
    but RETURNS a boolean (``True`` = fits, ``False`` = does not
    fit) rather than raising. The caller skips the retry when this
    returns ``False`` and returns the ORIGINAL 422 envelope it would
    have returned without the retry — so the client always sees a
    consistent json-schema-violation outcome.

    On permissive-skip paths (MLLM engine, missing ``build_prompt``,
    empty rendered prompt, tokenizer-returned-zero) this returns
    ``True`` to preserve the existing behavior — the initial-request
    gate also skips those paths so the repair gate should not be
    stricter than the initial one. The strict-mode + tools combo is
    already rejected upstream by ``strict_with_tools_unsupported``,
    so for repair-prompt accounting the ``tools`` argument is
    effectively always ``None``; we still thread it through for
    contract symmetry with the initial gate.

    ``enable_thinking`` mirrors the same parameter on
    :func:`enforce_context_length_for_messages` — forward the
    resolved value so the repair-fit check renders the prompt the
    way the engine actually will. Pre-fix the helper rendered with
    ``enable_thinking=None`` (template default = ``True`` on
    thinking-capable models) while the repair turn ran with the
    resolved value (typically ``False`` under R12-T1F / R12-T2F /
    R12-M2 auto-disable), so the gate could SKIP a repair retry
    that would actually have fit. Default ``None`` preserves the
    legacy behaviour for unaudited call sites.

    Used by ``routes/chat.py`` and ``routes/responses.py`` so the
    same gate logic is applied at both call sites and cannot drift.
    """
    if getattr(engine, "is_mllm", False):
        return True
    build_prompt = getattr(engine, "build_prompt", None)
    if build_prompt is None:
        return True
    try:
        prompt = _build_prompt_with_thinking_compat(
            build_prompt,
            repair_messages,
            tools=tools,
            enable_thinking=enable_thinking,
        )
    except Exception:
        # If the repair prompt can't even be rendered, we can't make
        # a useful "fits" judgement — defer to the engine error path
        # (the existing 502 ``strict_repair_engine_failure`` handler
        # already covers an unrenderable repair turn). Return ``True``
        # so the existing surface is preserved; do NOT raise here.
        return True
    if not prompt:
        return True
    prompt_tokens = count_prompt_tokens(engine, prompt)
    if prompt_tokens <= 0:
        return True
    max_context = get_model_max_context(engine)
    completion = int(max_tokens) if max_tokens else 0
    requested_total = int(prompt_tokens) + max(0, completion)
    return requested_total <= max_context


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
