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
        return cleaned_text, engine_reasoning_text
    if reasoning_parser is None:
        return cleaned_text, reasoning_text
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
        # parser routed the whole output to reasoning (its new
        # Case-4 fallback path: ``(reasoning, None)`` on input
        # that contains neither ``<think>`` nor ``</think>``),
        # the original ``cleaned_text`` is the same raw thought
        # trace — ``strip_thinking_tags`` only matches **closed**
        # ``<think>…</think>`` blocks, so a no-tag truncated thought
        # would pass straight through to ``final_content`` and the
        # client would see the exact same prose in BOTH
        # ``reasoning_content`` AND ``content`` (codex R1 BLOCKING
        # finding). Clear ``cleaned_text`` so the route renders
        # ``content=None`` for that case. Streaming-symmetry
        # invariant — the streaming Case-3 path never emits content
        # for truncated thoughts either.
        if (
            enable_thinking is True
            and new_reasoning is not None
            and new_cleaned is None
            and text_to_parse
            and "<think>" not in text_to_parse
            and "</think>" not in text_to_parse
        ):
            cleaned_text = ""
    return cleaned_text, reasoning_text


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
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


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

                error_data = _json.dumps(
                    {
                        "error": {
                            "message": f"Internal error during streaming: {exc}",
                            "type": type(exc).__name__,
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
