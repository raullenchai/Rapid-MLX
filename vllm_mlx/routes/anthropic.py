# SPDX-License-Identifier: Apache-2.0
"""Anthropic Messages API endpoints — /v1/messages."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from ..api.anthropic_adapter import (
    AnthropicOutputConfigError,
    anthropic_to_openai,
    openai_to_anthropic,
)
from ..api.anthropic_models import AnthropicRequest
from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import (
    StreamingThinkRouter,
    StreamingToolCallFilter,
    clean_output_text,
    extract_multimodal_content,
    sanitize_output,
    strip_special_tokens,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import BaseEngine
from ..middleware.auth import check_rate_limit_or_x_api_key, verify_api_key_or_x_api_key
from ..service.helpers import (
    SSE_RESPONSE_HEADERS,
    _build_usage,
    _check_admission_or_503,
    _disconnect_guard,
    _effective_enable_thinking,
    _finalize_content_and_reasoning,
    _parse_tool_calls_with_parser,
    _release_admission_unless_committed,
    _rescue_silent_drop_from_reasoning,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _resolve_reasoning_enabled,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    enforce_context_length_for_messages,
    get_engine,
)


def _resolved_sampling_kwargs(openai_request) -> dict:
    """Resolve every sampling param through the 4-layer cascade.

    Anthropic-compat receives an ``openai_request`` shape after adapter
    translation. Mirror the chat/completions routes so ``/v1/messages``
    users get the same alias / generation_config defaults.
    """
    out = {
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        # ``stop_sequences`` from the Anthropic request flows through the
        # adapter as ``openai_request.stop``. Both /v1/messages branches
        # (non-stream + stream) were dropping this, so the engine ran
        # uncapped and the model emitted past the user's stop tokens.
        # Forward via the single sampling-kwargs helper so the two
        # branches stay in sync. Note: the response stop_reason still
        # maps "stop" → "end_turn" (not "stop_sequence") because the
        # engine doesn't yet report WHICH stop fired; that's a follow-up.
        "stop": getattr(openai_request, "stop", None),
    }
    out.update(build_extended_sampling_kwargs(openai_request))
    return out


logger = logging.getLogger(__name__)

router = APIRouter()


def _should_start_in_thinking(chat_template: str, enable_thinking: bool | None) -> bool:
    """Return whether streaming should begin in an implicit thinking block.

    Some thinking-capable chat templates include ``<think>`` in the generated
    assistant prefix instead of emitting it as a normal output token.  In that
    case the stream router needs to start in thinking mode so tokens before
    ``</think>`` are emitted as Anthropic thinking deltas.

    When thinking is explicitly disabled, however, the template marker is only
    stale capability metadata for routing purposes: direct answer tokens should
    be emitted as text.  Otherwise Claude Code receives a message with only a
    thinking block and no text result.
    """
    if enable_thinking is False:
        return False
    return "<think>" in chat_template and "add_generation_prompt" in chat_template


def _filter_tool_calls_by_tool_choice(tool_calls, tool_choice) -> list:
    """Drop tool_use blocks that don't match a forced ``tool_choice``.

    H-05: Anthropic ``tool_choice={"type":"tool","name":X}`` pins WHICH
    tool the model must call. The Anthropic adapter has already
    translated that into the OpenAI form
    ``{"type":"function","function":{"name":X}}`` on
    ``openai_request.tool_choice`` by the time we get here. Local
    inference has no decoder-level constraint, so a small model can
    happily defy the pin and emit a call to a different tool (the
    Sergei repro hit qwen3.5-4b on a two-tool prompt where the model
    fired BOTH the pinned tool AND the un-pinned one). Pre-fix, the
    downstream JSON-schema validator (F-220) then 400-ed on the
    un-pinned tool's argument schema — leaking validation across a
    tool the user never asked for and silently breaking ``tool_choice``.

    Policy: when the choice pins a specific tool, KEEP only the calls
    to that tool and drop the rest with a warning. This matches the
    user-visible expectation ("you asked for X, here is X") and keeps
    the F-220 validator scoped to the pinned tool. ``"auto"`` /
    ``"required"`` / ``"none"`` / unset shapes pass through unchanged
    — only the explicit named-tool form has a defined "wrong tool"
    case to filter.

    Chat.py's named-function path 422s on the same mismatch (see
    ``routes/chat.py:1665``). The two routes intentionally diverge
    here: ``/v1/chat/completions`` mirrors OpenAI's strict contract,
    while ``/v1/messages`` mirrors Anthropic's more forgiving
    "deliver the pinned tool's call" contract — a 422 on an extra
    call would surface a confusing error to clients that pinned the
    very tool the response carries. The validator scope refactor in
    ``_validate_tool_call_params`` ensures we never validate against
    the dropped tool's schema even if a future change weakens this
    filter.
    """
    if not tool_calls or not isinstance(tool_choice, dict):
        return tool_calls or []
    if tool_choice.get("type") != "function":
        return tool_calls
    target = (tool_choice.get("function") or {}).get("name")
    if not target:
        return tool_calls

    def _name_of(tc) -> str | None:
        # Three shapes survive into this point — see
        # ``routes/chat.py:_tool_call_name`` for the catalogue. Inline
        # here to avoid a cross-route import dependency from
        # routes/chat.py into routes/anthropic.py.
        if isinstance(tc, dict):
            fn = tc.get("function")
            if isinstance(fn, dict):
                return fn.get("name")
            if fn is not None:
                return getattr(fn, "name", None)
            return tc.get("name")
        fn = getattr(tc, "function", None)
        if fn is not None:
            return getattr(fn, "name", None)
        return getattr(tc, "name", None)

    filtered = []
    dropped: list[str] = []
    for tc in tool_calls:
        if _name_of(tc) == target:
            filtered.append(tc)
        else:
            dropped.append(_name_of(tc) or "<unknown>")
    if dropped:
        logger.warning(
            "tool_choice pinned %r but model also emitted calls to %s; "
            "dropping the un-pinned calls so the response carries only the "
            "pinned tool (Anthropic /v1/messages H-05 policy).",
            target,
            dropped,
        )
    return filtered


@router.post(
    "/v1/messages",
    dependencies=[
        Depends(verify_api_key_or_x_api_key),
        Depends(check_rate_limit_or_x_api_key),
    ],
)
async def create_anthropic_message(
    request: Request,
):
    """
    Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI format, runs inference
    through the existing engine, and converts the response back.
    """
    body = await request.json()
    # ``AnthropicRequest`` is constructed manually (not as a FastAPI body
    # parameter), so Pydantic ``ValidationError`` would otherwise surface
    # as a generic 500. Catch it explicitly to give clients a 400 with
    # the actual validation detail — matches the ergonomics of the
    # ``output_config`` 400 path below (PR #42396 backport).
    try:
        anthropic_request = AnthropicRequest(**body)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not (anthropic_request.model or "").startswith(("claude-", "gpt-")):
        _validate_model_name(anthropic_request.model)
    engine = get_engine(anthropic_request.model)

    # Pre-flight admission gate (C4) — see routes/chat.py for rationale.
    # Reservation released by the route-level ``finally`` below; on the
    # streaming path ``_admission_committed`` flips to True so
    # ``_disconnect_guard`` owns the release once the SSE generator
    # closes. Closes the codex R3 leak (validation errors between the
    # reservation and the helper used to pin the slot until restart).
    _check_admission_or_503(engine)
    _admission_committed = False
    try:
        # --- Detailed request logging ---
        n_msgs = len(anthropic_request.messages)
        total_chars = 0
        last_user_preview = ""
        for m in anthropic_request.messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            total_chars += len(content)
            if m.role == "user":
                last_user_preview = content[:300]
        sys_chars = len(anthropic_request.system) if anthropic_request.system else 0
        n_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
        logger.info(
            f"[REQUEST] POST /v1/messages (anthropic) stream={anthropic_request.stream} "
            f"model={anthropic_request.model!r} max_tokens={anthropic_request.max_tokens} "
            f"msgs={n_msgs} total_chars={total_chars} system_chars={sys_chars} "
            f"tools={n_tools}"
        )
        logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

        cfg_for_log = get_config()
        if (
            anthropic_request.model
            and cfg_for_log.model_name
            and anthropic_request.model != cfg_for_log.model_name
        ):
            logger.info(
                "Anthropic /v1/messages: request model=%r served by loaded engine=%r",
                anthropic_request.model,
                cfg_for_log.model_name,
            )

        # Convert Anthropic request -> OpenAI request. The adapter raises
        # ``AnthropicOutputConfigError`` (a ``ValueError`` subclass) on
        # malformed ``output_config`` payloads — backport of upstream vLLM
        # PR #42396; map directly to HTTP 400 with the adapter's message.
        try:
            openai_request = anthropic_to_openai(anthropic_request)
        except AnthropicOutputConfigError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValidationError as e:
            # F-034 (and any future ``ChatCompletionRequest``-layer
            # validator): the adapter constructs an OpenAI-shape request
            # from the Anthropic body, and ``ChatCompletionRequest`` now
            # rejects unsatisfiable combinations (e.g. ``tool_choice="required"``
            # — Anthropic ``any`` — with no ``tools``). Surface as 400 with
            # the validator's message instead of letting Pydantic crash
            # the route into a 500.
            raise HTTPException(status_code=400, detail=str(e))

        # Context-length pre-check — same DoS gate the chat/completions/
        # responses routes enforce. Render the prompt through the engine's
        # chat template, count tokens, raise 400 ``context_length_exceeded``
        # if over the model cap. Runs BEFORE the stream/non-stream branch
        # so streaming clients can't bypass the gate by setting
        # ``stream: true``. See service/helpers.py for rationale.
        try:
            _ctx_messages, _, _ = extract_multimodal_content(
                openai_request.messages,
                preserve_native_format=engine.preserve_native_tool_format,
            )
        except Exception:
            _ctx_messages = None
        if _ctx_messages is not None:
            enforce_context_length_for_messages(
                engine,
                _ctx_messages,
                tools=openai_request.tools,
                max_tokens=_resolve_max_tokens(
                    openai_request.max_tokens,
                    _resolve_enable_thinking(openai_request),
                ),
            )

        if anthropic_request.stream:
            _admission_committed = True
            return StreamingResponse(
                _disconnect_guard(
                    _stream_anthropic_messages(
                        engine, openai_request, anthropic_request
                    ),
                    request,
                    engine=engine,
                ),
                media_type="text/event-stream",
                # ``SSE_RESPONSE_HEADERS`` (Cache-Control no-cache/no-transform +
                # X-Accel-Buffering: no) keeps anti-buffering parity with the
                # other SSE routes; ``Connection: keep-alive`` is preserved for
                # the Anthropic SDK clients that historically checked for it.
                headers={**SSE_RESPONSE_HEADERS, "Connection": "keep-alive"},
            )

        # Non-streaming: run inference through existing engine
        messages, images, videos = extract_multimodal_content(
            openai_request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

        chat_kwargs = {
            "max_tokens": _resolve_max_tokens(
                openai_request.max_tokens,
                _resolve_enable_thinking(openai_request),
            ),
            **_resolved_sampling_kwargs(openai_request),
        }

        if openai_request.tools:
            chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
        cfg = get_config()
        # Resolve enable_thinking via shared helper (#387: chat_template_kwargs
        # passthrough). Same precedence as the OpenAI route.
        resolved_thinking = _resolve_enable_thinking(openai_request)
        effective_thinking = _effective_enable_thinking(
            resolved_thinking, cfg.model_path or cfg.model_name
        )
        if effective_thinking is not None:
            chat_kwargs["enable_thinking"] = effective_thinking

        start_time = time.perf_counter()
        timeout = cfg.default_timeout

        try:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                request,
                timeout=timeout,
            )
        except HTTPException:
            raise
        except Exception as e:
            err_msg = str(e)
            err_type = type(e).__name__
            if (
                "TemplateError" in err_type
                or "template" in err_msg.lower()
                or ("user" in err_msg.lower() and "found" in err_msg.lower())
            ):
                raise HTTPException(
                    status_code=400, detail=f"Chat template error: {err_msg}"
                )
            # Multimodal fetch failures → 400 (parity with chat route, #457).
            # Per-batch-cap errors from the MLLM engine also surface as
            # client-actionable → 400 (parity with chat route, #682). The
            # MLLM scheduler classifier in ``mllm_scheduler._step_no_queue``
            # treats both as client errors; this route must map both to 400
            # or Anthropic-style clients get a 500 for what is really an
            # oversized-image / oversized-prompt user error.
            if (
                "Failed to process image" in err_msg
                or "Failed to process video" in err_msg
                or "exceeds the per-batch cap" in err_msg
            ):
                raise HTTPException(status_code=400, detail=err_msg)
            raise
        if output is None:
            return Response(status_code=499)

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Parse tool calls — prefer the engine's structured payload
        # (HarmonyStreamingRouter via openai-harmony's StreamableParser)
        # over text-based extraction when present. See routes/chat.py
        # for the rationale (PR #515 codex round-12 / round-14 BLOCKING
        # — wire-text round-trip lost calls whose JSON args contained
        # harmony sentinels).
        engine_tool_calls = getattr(output, "tool_calls", None)
        cleaned_text, tool_calls = _parse_tool_calls_with_parser(
            output.text, openai_request, structured_tool_calls=engine_tool_calls
        )

        # H-05: tool_choice={"type":"tool","name":X} pins WHICH tool the
        # model must call. Local inference can't decoder-enforce that,
        # so a defiant model can fire an extra call to a different
        # tool. Pre-fix, the F-220 validator below then 400-ed on the
        # un-pinned tool's schema — Sergei's repro: pinned
        # ``get_weather``, model fired ``get_weather`` AND
        # ``lookup_zip``, 400 came back complaining about ``lookup_zip``.
        # Drop the un-pinned calls FIRST so the validator only sees the
        # pinned tool's call(s). See ``_filter_tool_calls_by_tool_choice``
        # for the policy rationale vs chat.py's 422-on-mismatch.
        if tool_calls and openai_request.tool_choice:
            tool_calls = _filter_tool_calls_by_tool_choice(
                tool_calls, openai_request.tool_choice
            )

        # F-220: enforce the same tool_call JSON-schema validation
        # ``routes/chat.py:1651`` runs on the OpenAI ``/v1/chat/completions``
        # route. The Anthropic adapter has already translated
        # ``input_schema`` into the OpenAI ``function.parameters`` shape
        # (see ``api/anthropic_adapter._convert_tool``), so we can reuse
        # the same validator unchanged. Without this, an enum/type/range
        # violation that returns HTTP 400 on chat-completions silently
        # propagated through ``/v1/messages`` as a 200 ``tool_use`` block
        # carrying schema-violating arguments.
        if tool_calls and openai_request.tools:
            _validate_tool_call_params(tool_calls, openai_request.tools)

        # Extract reasoning content via the same orchestration the OpenAI route
        # uses (chat.py). Skipping this is what #413 fixed — the Anthropic surface
        # used to silently drop ``<think>...</think>`` content on the non-streaming
        # path while OpenAI preserved it as ``reasoning_content``.
        cleaned_text_before_helper = cleaned_text
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=output.raw_text or output.text,
            cleaned_text=cleaned_text,
            tool_calls=tool_calls,
            reasoning_parser=cfg.reasoning_parser,
            engine_reasoning_text=getattr(output, "reasoning_text", "") or "",
            # #575 — mirror chat.py so the Anthropic non-stream surface
            # gets the same Case-4 fallback (codex R1 BLOCKING: the
            # helper is shared between both routes so leaving this
            # call site on the legacy contract would let the leak
            # persist on ``/v1/messages`` while ``/v1/chat/completions``
            # was fixed). Use ``cfg.model_path`` rather than
            # ``cfg.model_name`` to avoid divergence with the
            # prompt-render path when ``--served-model-name`` is set
            # (codex R2 BLOCKING).
            enable_thinking=_effective_enable_thinking(
                resolved_thinking, cfg.model_path or cfg.model_name
            ),
            # Per-request reasoning cap (upstream vLLM PR #20859 / #42396
            # backport). The adapter translated ``output_config.effort``
            # or legacy ``thinking.budget_tokens`` into this field on
            # the OpenAI-side request, so it propagates uniformly across
            # all three API surfaces.
            reasoning_max_tokens=getattr(openai_request, "reasoning_max_tokens", None),
        )

        final_content = None
        if cleaned_text:
            final_content = strip_thinking_tags(clean_output_text(cleaned_text))
            # Final defense against special-token / markup leakage — mirrors
            # chat.py:669 so the two surfaces don't diverge on what they
            # consider "sanitized" client-facing content. Pre-existing gap
            # flagged by codex during the #413 review.
            final_content = sanitize_output(final_content)

        # Issue #569: never silently drop. Mirror the OpenAI route's
        # rescue so the Anthropic surface gets the same protection
        # against silently-empty assistant turns when the model gets
        # stuck inside reasoning (gemma-4-26b-4bit multi-turn failure
        # mode). The Anthropic adapter downstream renders the
        # rescued ``content`` into a TextBlock; without this it would
        # emit a completely empty ``content=[]`` Messages response.
        finish_reason = "tool_calls" if tool_calls else output.finish_reason
        # PR #715 bundle, fuzz finding C: detect helper Case-4 blank
        # (parser routed whole no-tag output to reasoning, helper
        # cleared cleaned_text=""). See chat.py route for the full
        # rationale.
        reasoning_is_case4 = bool(
            cleaned_text_before_helper
            and not cleaned_text
            and reasoning_text
            and cfg.reasoning_parser is not None
            and not (getattr(output, "reasoning_text", "") or "")
        )
        final_content = _rescue_silent_drop_from_reasoning(
            final_content,
            reasoning_text,
            tool_calls,
            finish_reason=finish_reason,
            raw_text=output.raw_text or output.text,
            reasoning_is_case4=reasoning_is_case4,
        )

        openai_response = ChatCompletionResponse(
            model=cfg.model_name or openai_request.model,
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=final_content,
                        reasoning_content=reasoning_text,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=_build_usage(output, reasoning_text),
        )

        # Issue #702: signal the alias's reasoning capability to the
        # adapter so it can suppress the ``thinking`` content block when
        # the served alias has ``reasoning_parser: null`` in
        # ``aliases.json``. Without this gate, an OpenAI-side response
        # that happens to carry ``reasoning_content`` (or the
        # ``_rescue_silent_drop_from_reasoning`` duplication into
        # ``content`` above) would emit a ``thinking`` block on a model
        # that Anthropic's public API would never produce one for,
        # breaking client capability detection and rendering the same
        # paragraph twice.
        #
        # Resolve via ``_resolve_reasoning_enabled`` so the predicate
        # consults the per-request registry entry first (multi-model
        # mode) and only falls back to the global ``cfg.reasoning_parser``
        # singleton. Codex r1 BLOCKING on PR #705 — global-only lookup
        # would let the duplicate leak when a non-thinking alias is
        # served alongside a thinking default.
        anthropic_response = openai_to_anthropic(
            openai_response,
            cfg.model_name or anthropic_request.model,
            reasoning_enabled=_resolve_reasoning_enabled(anthropic_request.model),
        )
        return Response(
            content=anthropic_response.model_dump_json(exclude_none=True),
            media_type="application/json",
        )
    finally:
        _release_admission_unless_committed(engine, _admission_committed)


@router.post(
    "/v1/messages/count_tokens",
    dependencies=[
        Depends(verify_api_key_or_x_api_key),
        Depends(check_rate_limit_or_x_api_key),
    ],
)
async def count_anthropic_tokens(request: Request):
    """Count tokens for an Anthropic Messages API request.

    Validation contract — mirrors ``/v1/messages``:

    * Malformed JSON body → 400 via the global ``json.JSONDecodeError``
      handler in ``server.py`` (F-161).
    * Missing or empty ``messages`` → 400 ``invalid_request_error``
      instead of the silent ``{"input_tokens": 0}`` cost-estimation
      footgun (F-160). Anthropic's real endpoint requires a non-empty
      ``messages`` array; mirror that here so clients don't ship a
      pricing-page bug into production.
    * Unknown ``model`` → 404 via ``_validate_model_name`` instead of
      silently using the loaded model's tokenizer (F-167). A fallback
      count is mathematically meaningless to a client estimating cost
      for a *different* model.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Request body must be a JSON object",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": None,
                }
            },
        )

    # Validate ``messages`` first — Anthropic's contract requires at
    # least one message. Returning 0 tokens here is worse than an error
    # because clients use this endpoint to estimate cost and a silent
    # zero looks like "free request" rather than "bad request".
    raw_messages = body.get("messages", None)
    if raw_messages is None or (
        isinstance(raw_messages, list) and len(raw_messages) == 0
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "`messages` must be a non-empty array of "
                        "Anthropic message objects"
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "messages",
                }
            },
        )
    if not isinstance(raw_messages, list):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "`messages` must be a JSON array",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "messages",
                }
            },
        )

    # Validate model name — mirror ``/v1/chat/completions`` and
    # ``/v1/responses``. Claude/Codex aliases pass through to the
    # loaded engine just like in ``create_anthropic_message`` above
    # (PR #557 contract). A *present* non-string ``model`` (or empty
    # string) is a client bug — if we silently dropped it the loaded
    # engine's tokenizer would still produce a count and a cost
    # estimator would treat it as authoritative (codex bundled review
    # on the F-167 fix, follow-up to F-160).
    if "model" in body:
        requested_model = body["model"]
        if requested_model is not None and not isinstance(requested_model, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "`model` must be a string",
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                        "param": "model",
                    }
                },
            )
        if isinstance(requested_model, str) and requested_model == "":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "`model` must not be empty",
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                        "param": "model",
                    }
                },
            )
        if (
            isinstance(requested_model, str)
            and requested_model
            and not requested_model.startswith(("claude-", "gpt-"))
        ):
            _validate_model_name(requested_model)

    engine = get_engine()
    tokenizer = engine.tokenizer

    total_tokens = 0

    # System message
    system = body.get("system", "")
    if isinstance(system, str) and system:
        total_tokens += len(tokenizer.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    total_tokens += len(tokenizer.encode(text))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                total_tokens += len(tokenizer.encode(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total_tokens += len(tokenizer.encode(text))
                    if block.get("input"):
                        total_tokens += len(
                            tokenizer.encode(json.dumps(block["input"]))
                        )
                    sub_content = block.get("content", "")
                    if isinstance(sub_content, str) and sub_content:
                        total_tokens += len(tokenizer.encode(sub_content))
                    elif isinstance(sub_content, list):
                        for item in sub_content:
                            if isinstance(item, dict):
                                item_text = item.get("text", "")
                                if item_text:
                                    total_tokens += len(tokenizer.encode(item_text))

    # Tools
    for tool in body.get("tools", []):
        name = tool.get("name", "")
        if name:
            total_tokens += len(tokenizer.encode(name))
        desc = tool.get("description", "")
        if desc:
            total_tokens += len(tokenizer.encode(desc))
        if tool.get("input_schema"):
            total_tokens += len(tokenizer.encode(json.dumps(tool["input_schema"])))

    return {"input_tokens": total_tokens}


def _emit_content_pieces(
    pieces: list[tuple[str, str]],
    current_block_type: str | None,
    block_index: int,
) -> tuple[list[str], str | None, int]:
    """Emit Anthropic SSE events for content pieces from the think router."""
    events = []
    for block_type, text in pieces:
        if block_type != current_block_type:
            if current_block_type is not None:
                events.append(
                    f"event: content_block_stop\ndata: "
                    f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                )
                block_index += 1
            current_block_type = block_type
            content_block = (
                {"type": block_type, "text": ""}
                if block_type == "text"
                else {"type": block_type, "thinking": ""}
            )
            events.append(
                f"event: content_block_start\ndata: "
                f"{json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': content_block})}\n\n"
            )
        delta_key = "thinking" if block_type == "thinking" else "text"
        delta_type = "thinking_delta" if block_type == "thinking" else "text_delta"
        delta_event = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": delta_type, delta_key: text},
        }
        events.append(
            f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
        )
    return events, current_block_type, block_index


async def _stream_anthropic_messages(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    anthropic_request: AnthropicRequest,
) -> AsyncIterator[str]:
    """Stream Anthropic Messages API SSE events."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()

    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(
            openai_request.max_tokens,
            _resolve_enable_thinking(openai_request),
        ),
        **_resolved_sampling_kwargs(openai_request),
    }

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
    cfg = get_config()
    # Resolve enable_thinking via shared helper (#387: chat_template_kwargs
    # passthrough). Same precedence as the OpenAI route.
    resolved_thinking = _resolve_enable_thinking(openai_request)
    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    # Issue #702: per-request alias-level reasoning capability gate.
    # When the served alias declares ``reasoning_parser: null`` in
    # ``aliases.json``, the streaming path must NEVER open a
    # ``thinking`` content block — Anthropic's public API doesn't
    # emit one for non-extended-thinking models, so any client that
    # branches on ``content_block.type == "thinking"`` would
    # mis-detect capability. Applied via ``_gate_thinking_pieces``
    # below to every place this function constructs
    # ``("thinking", ...)`` pieces: channel-routed (engine
    # OutputRouter), reasoning-parser delta split, and the raw
    # ``<think>`` think_router heuristic. When the gate fires the
    # reasoning bytes are demoted to a ``text`` piece so the
    # assistant turn still surfaces the model's output (silent drop
    # is the worse failure mode, #569).
    #
    # Resolution: consult the per-request registry entry first
    # (multi-model mode), fall back to the global parser pair in
    # single-model mode (codex r1 BLOCKING on PR #705). Inlined here
    # so the predicate consumes the SAME ``cfg`` object the rest of
    # this function already reads — sharing avoids a second
    # ``get_config()`` call that test fixtures patching
    # ``anthropic_route.get_config`` would miss.
    #
    # Capability is captured ONCE at request entry and frozen in the
    # ``_gate_thinking_pieces`` closure for the entire SSE response.
    # A hot-reload that mutates ``cfg.model_registry`` mid-stream
    # MUST NOT change the gating behavior partway through one
    # response — clients expect a single coherent SSE contract per
    # request (codex r3 NIT probe 4).
    _reasoning_enabled = False
    if cfg.model_registry:
        try:
            _entry = cfg.model_registry.get_entry(anthropic_request.model)
        except KeyError:
            _entry = None
        if _entry is not None:
            _reasoning_enabled = bool(getattr(_entry, "reasoning_parser", None))
        else:
            _reasoning_enabled = cfg.reasoning_parser is not None or bool(
                cfg.reasoning_parser_name
            )
    else:
        _reasoning_enabled = cfg.reasoning_parser is not None or bool(
            cfg.reasoning_parser_name
        )

    def _gate_thinking_pieces(
        pieces: list[tuple[str, str]],
        current_block_type: str | None,
    ) -> list[tuple[str, str]]:
        """Apply the #702 capability gate + non-stream parity filter.

        Two concerns, in one pass:

        1. **Non-thinking alias demotion.** When ``_reasoning_enabled``
           is False (per-request alias has ``reasoning_parser: null``
           in ``aliases.json``), every ``("thinking", text)`` piece is
           rewritten to ``("text", text)``. The rewrite preserves order
           so downstream ``_emit_content_pieces`` still merges
           consecutive same-type pieces into a single content block.

        2. **No-empty-block parity with non-stream.** The non-stream
           ``openai_to_anthropic`` predicate skips a thinking block when
           ``reasoning_text.strip() == ""``. Mirror that on the
           streaming surface so a model that emits ``<think> </think>``
           or a whitespace-only reasoning channel delta does NOT open a
           thinking ``content_block_start`` + whitespace
           ``thinking_delta`` that Claude Code surfaces as a blank
           thought bubble.

        The whitespace guard is **state-aware**: a whitespace-only
        thinking piece is only dropped when it would OPEN a blank
        thinking block — i.e. no thinking block is currently open in
        the SSE stream (``current_block_type != "thinking"``) AND no
        later piece in this batch carries non-whitespace thinking
        content that would mark the leading whitespace as an
        intra-thinking separator. This preserves the
        ``"first" + "\n\n" + "second"`` shape that the model uses to
        break thinking into paragraphs without leaking the
        ``"   " -> open empty block`` shape (codex r3 MAJOR probe 1,
        refined per codex r4 MAJOR).

        ``current_block_type`` is the block type currently OPEN at the
        downstream emitter (None / "text" / "thinking") — when it's
        "thinking" we ALWAYS keep whitespace because it's an intra-block
        continuation, never a block opener.
        """
        # Track the EFFECTIVE open block type — i.e. what the
        # downstream emitter currently has open after the gate's
        # rewrites, NOT the raw piece type the model emitted. This
        # lets the non-thinking branch route a whitespace-only
        # ``("thinking", " ")`` piece into an already-open TEXT block
        # (demoted to ("text", " ")) instead of dropping it. Codex r5
        # MAJOR.
        #
        # ``effective`` is one of None / "text" / "thinking" and
        # reflects what ``_emit_content_pieces`` will have open after
        # consuming the pieces ``out`` so far.
        effective: str | None = current_block_type
        out: list[tuple[str, str]] = []
        for block_type, text in pieces:
            if block_type == "thinking":
                if not text.strip():
                    # Whitespace-only thinking piece. Decide whether to
                    # drop, keep as thinking, or demote to text based
                    # on which (if any) block is currently open.
                    if effective == "thinking":
                        # Intra-thinking separator — keep as-is on the
                        # reasoning-enabled path. (The non-thinking
                        # branch can't see ``effective == "thinking"``
                        # because demotion below sets ``effective`` to
                        # "text" rather than "thinking".)
                        out.append(("thinking", text))
                    elif effective == "text" and not _reasoning_enabled:
                        # Non-thinking branch with an open text block:
                        # demote the whitespace so it lands inside the
                        # current text block (codex r5 MAJOR — without
                        # this, ``("thinking", "hello") + ("thinking",
                        # " ")`` would stream as ``"hello"`` instead of
                        # ``"hello "``).
                        out.append(("text", text))
                    else:
                        # No relevant open block — dropping it avoids
                        # opening a blank thinking OR blank text block.
                        # The non-stream predicate (.strip()) does the
                        # same.
                        continue
                    continue
                # Non-whitespace thinking content. Reasoning-enabled
                # keeps as thinking; non-thinking demotes to text.
                if _reasoning_enabled:
                    out.append(("thinking", text))
                    effective = "thinking"
                else:
                    out.append(("text", text))
                    effective = "text"
            else:
                out.append((block_type, text))
                # ``block_type`` is already not "thinking" in this
                # branch — track the effective open block as that type.
                effective = block_type
        return out

    # Emit message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": cfg.model_name or anthropic_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    accumulated_text = ""
    accumulated_raw = ""
    # Structured tool calls surfaced by the engine's OutputRouter
    # (currently HarmonyStreamingRouter via openai-harmony's
    # StreamableParser). When non-empty at end-of-stream the final
    # ``_parse_tool_calls_with_parser`` call uses these directly,
    # bypassing the regex round-trip — same bytes-faithful path the
    # non-streaming branch uses (PR #515 codex round-12/14 BLOCKING
    # closure).
    accumulated_structured_tool_calls: list[dict] = []
    tool_filter = StreamingToolCallFilter()
    # ``tokenizer`` is on the BaseEngine contract; the old ``hasattr``
    # guard predated the abstract declaration and is the same silent-skip
    # shape that produced #500. The inner ``chat_template`` guard stays
    # because that attribute is HF-tokenizer-specific, not part of our
    # contract.
    _tokenizer = engine.tokenizer
    _chat_template = ""
    if _tokenizer and hasattr(_tokenizer, "chat_template"):
        _chat_template = _tokenizer.chat_template or ""
    _starts_thinking = _should_start_in_thinking(
        _chat_template, chat_kwargs.get("enable_thinking")
    )
    think_router = StreamingThinkRouter(start_in_thinking=_starts_thinking)
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0

    current_block_type = None
    block_index = 0

    # Per-request reasoning parser instance (not the singleton from cfg).
    # Avoids state corruption under concurrent BatchedEngine requests.
    reasoning_parser = None
    if cfg.reasoning_parser_name:
        try:
            from ..reasoning import get_parser

            reasoning_parser = get_parser(cfg.reasoning_parser_name)()
        except Exception:
            pass
    # Closes #223: when the client explicitly opts out of thinking, bypass
    # the reasoning parser. Parsers like qwen3 use an implicit-think
    # heuristic (no <think> tag → all tokens treated as reasoning), so a
    # direct answer would otherwise be misrouted to thinking_delta blocks
    # and the text_delta block would stay empty. Mirrors the chat-route
    # bypass at postprocessor.py:217. The think_router branch below picks
    # up the work, and `_should_start_in_thinking` already returns False
    # for enable_thinking=False, so the answer streams as text.
    if chat_kwargs.get("enable_thinking") is False:
        reasoning_parser = None
    # Issue #702 codex r2 BLOCKING: when the per-request alias is NOT
    # reasoning-capable, also bypass the parser entirely. Implicit-mode
    # parsers (Qwen3 / hermes) classify ordinary chunks as reasoning
    # until ``finalize_streaming`` emits a correction at end-of-stream
    # — and the finalize correction is emitted as plain ``text``
    # without going through ``_gate_thinking_pieces``. If we only
    # gated the per-delta pieces, a non-thinking alias served beside a
    # thinking global would stream the demoted reasoning bytes as
    # text AND then the finalize correction would emit the SAME bytes
    # again — visible duplication. Dropping the parser here puts the
    # stream on the ``think_router`` path which only opens thinking
    # blocks on literal ``<think>`` tags in the raw stream (and is
    # itself gated by ``_gate_thinking_pieces`` below).
    if not _reasoning_enabled:
        reasoning_parser = None
    if reasoning_parser:
        reasoning_parser.reset_state()

    # Per-request reasoning cap (upstream vLLM PR #20859 / #42396 backport).
    # Same chars-÷4 heuristic the OpenAI route uses so the same effective
    # budget applies regardless of which API surface the client picked.
    _reasoning_cap = getattr(openai_request, "reasoning_max_tokens", None)
    _reasoning_tokens_emitted = 0
    _reasoning_cap_hit = False
    _reasoning_close_injected = False

    def _account_for_reasoning(text: str) -> tuple[str, str]:
        """``(kept_reasoning, overflow_content)``.

        Codex round-12 BLOCKING #3: cumulative-CHARACTER accounting
        against ``cap * 4`` (not per-chunk ceiling). The earlier
        ``max(1, ceil(len/4))`` made fragmented reasoning deltas
        consume more tokens than the same contiguous text, so the
        cap on ``output_config.effort`` fired at different points
        depending only on SSE chunk boundaries. Now identical model
        output hits the cap at the same character offset regardless
        of chunking — byte-for-byte consistent with the Responses
        route + postprocessor + non-stream paths.

        ``_reasoning_tokens_emitted`` now stores CHARACTERS (name kept
        for back-compat). The cap *4 limit lives in the closure.
        """
        nonlocal _reasoning_tokens_emitted, _reasoning_cap_hit
        if _reasoning_cap is None or not text:
            return text, ""
        if _reasoning_cap_hit:
            return "", text
        max_chars = _reasoning_cap * 4
        new_total_chars = _reasoning_tokens_emitted + len(text)
        if new_total_chars < max_chars:
            _reasoning_tokens_emitted = new_total_chars
            return text, ""
        if new_total_chars == max_chars:
            # Exact-boundary latch (codex round-2 BLOCKING #2).
            _reasoning_tokens_emitted = new_total_chars
            _reasoning_cap_hit = True
            return text, ""
        remaining_chars = max_chars - _reasoning_tokens_emitted
        keep_chars = max(0, remaining_chars)
        _reasoning_tokens_emitted = max_chars
        _reasoning_cap_hit = True
        return text[:keep_chars], text[keep_chars:]

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        delta_text = output.new_text

        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens
        if hasattr(output, "cached_tokens") and output.cached_tokens:
            cached_tokens = output.cached_tokens

        # Capture engine-surfaced structured tool calls (HarmonyStreamingRouter
        # via openai-harmony's StreamableParser). The delta_text on these
        # events is the JSON args summary; we DO NOT want to feed it into
        # the text-based tool_filter / accumulator because that would re-
        # introduce the round-trip lossy path the refactor exists to
        # eliminate (PR #515 codex round-12/14 BLOCKING).
        engine_tool_calls = getattr(output, "tool_calls", None) or []
        if engine_tool_calls:
            accumulated_structured_tool_calls.extend(engine_tool_calls)
            continue

        if delta_text:
            accumulated_text += delta_text

            # When the engine has already routed this delta into a
            # semantic channel (OutputRouter — harmony/gemma4
            # models), honor the channel assignment directly.
            # Skipping this branch and feeding the channel-resolved
            # text into a text-based reasoning parser silently
            # suppresses every chunk: the parser scans for
            # ``<|channel|>`` markers that the router has already
            # stripped at the token layer, so its state machine
            # never leaves the "Unknown channel, suppress" arm and
            # this loop emits no ``content_block_delta`` events. The
            # symptom (v0.6.64 pr_validate on gpt-oss-20b-mxfp4-q8: anthropic
            # stream test 4 returned 0 content chunks) is the
            # streaming counterpart of the non-streaming empty-
            # TextBlock bug fixed in
            # ``service/helpers._finalize_content_and_reasoning`` —
            # both ultimately came from the channel-routed pipeline
            # presenting already-clean text to a parser that needs
            # to see markers. The OpenAI streaming path picks up the
            # equivalent of this branch through
            # ``service/postprocessor.StreamingPostProcessor.
            # _process_channel_routed``; the Anthropic streaming
            # path lived inline here and was missed.
            # ``getattr`` keeps legacy mocks (without ``.channel``)
            # falling through to the text path below.
            output_channel = getattr(output, "channel", None)
            if output_channel is not None:
                # Explicit allowlist (mirrors ``_CHANNEL_TO_STRING``
                # in ``engine/batched.py``). An unrecognized channel
                # is suppressed and logged rather than emitted as
                # user-facing text — if a new router channel is
                # added later (e.g. ``"system"``, ``"error"``) it
                # must opt in here before reaching the client.
                pieces_routed: list[tuple[str, str]] = []
                if output_channel == "reasoning":
                    reasoning = strip_special_tokens(delta_text)
                    if reasoning:
                        # Per-request reasoning cap — split into kept
                        # (thinking) and overflow (text) so Claude-Code
                        # eventually sees a final answer instead of an
                        # endless thinking_delta stream.
                        kept, overflow = _account_for_reasoning(reasoning)
                        if kept:
                            # Don't filter whitespace here — a
                            # whitespace-only chunk may be an
                            # intra-thinking separator (e.g. "\n\n"
                            # between two thinking paragraphs). The
                            # state-aware ``_gate_thinking_pieces``
                            # below preserves separators when a thinking
                            # block is already open and only drops a
                            # piece that would otherwise OPEN a blank
                            # thinking block. Mirrors the non-stream
                            # predicate's whole-text ``.strip()`` check
                            # (codex r3 probe 1, refined per r4 MAJOR).
                            pieces_routed.append(("thinking", kept))
                        if overflow:
                            filtered = tool_filter.process(overflow)
                            if filtered:
                                pieces_routed.append(("text", filtered))
                elif output_channel in ("content", "tool_call"):
                    # ``content`` and ``tool_call`` both render as
                    # user-facing text deltas; tool detection still
                    # runs through ``tool_filter`` so an emitted tool
                    # call (model-generated commentary channel) gets
                    # suppressed from text the same way it would on
                    # the non-routed path.
                    content = strip_special_tokens(delta_text)
                    if content:
                        filtered = tool_filter.process(content)
                        if filtered:
                            pieces_routed.append(("text", filtered))
                else:
                    logger.warning(
                        "anthropic stream: dropping delta from "
                        "unknown channel %r (delta=%r)",
                        output_channel,
                        delta_text[:64],
                    )
                if pieces_routed:
                    # Issue #702: gate thinking-piece emission on the
                    # alias's reasoning capability. ``OutputRouter`` is
                    # purely token-based and would surface reasoning
                    # for ANY alias whose tokenizer carries
                    # ``<|channel>thought`` / harmony analysis tokens
                    # — including aliases that declared
                    # ``reasoning_parser: null`` (capability opt-out
                    # for a tokenizer that nominally supports
                    # channels). Demote to text so the model output
                    # still surfaces and clients don't see a
                    # ``thinking`` block on a non-extended-thinking
                    # alias.
                    events, current_block_type, block_index = _emit_content_pieces(
                        _gate_thinking_pieces(pieces_routed, current_block_type),
                        current_block_type,
                        block_index,
                    )
                    for event in events:
                        yield event
                continue

            if reasoning_parser:
                # Closes #185: when a reasoning_parser is active it ALREADY
                # splits content vs reasoning at every chunk; routing the
                # parser's content through `think_router` (which detects
                # raw `<think>` tags in the underlying stream) double-counts
                # and silently buffers the answer as thinking_delta. Symptom
                # was Anthropic stream test 4 returning 0 chunks for every
                # qwen3-family model since v0.6.4. Bypass `think_router`
                # here and emit reasoning/content as their own block types
                # directly.
                previous_raw = accumulated_raw
                # Text-parser cap force-close: splice ``</think>`` into the
                # parser's incoming bytes once the cap has fired so the
                # state machine flips to content on this chunk. Idempotent.
                #
                # Codex round-9 BLOCKING #3: earlier draft mutated
                # ``delta_text`` to ``"</think>" + delta_text`` THEN ran
                # ``accumulated_raw += delta_text``, poisoning the
                # shared Anthropic raw buffer with the forged marker.
                # The terminal injection / finalize_streaming path then
                # re-parsed the mutated buffer, potentially mis-
                # classifying the synthetic bytes. Fix: keep
                # ``accumulated_raw`` to real model output only and
                # build a LOCAL ``parser_current`` that includes the
                # synthetic marker for the parser call. Shared buffer
                # holds ``previous_raw + original_delta``; parser sees
                # ``previous_raw + "</think>" + original_delta``.
                # Codex round-10 BLOCKING #3: only flip the close-
                # injected latch AFTER the parser call succeeds. If
                # the parser raises on the injection-carrying chunk,
                # the latch stays clear and the next chunk retries
                # the forced transition.
                injected_this_chunk = False
                if _reasoning_cap_hit and not _reasoning_close_injected:
                    parser_delta_text = "</think>" + delta_text
                    parser_current = previous_raw + parser_delta_text
                    injected_this_chunk = True
                else:
                    parser_delta_text = delta_text
                    parser_current = previous_raw + delta_text
                accumulated_raw += delta_text
                delta_msg = reasoning_parser.extract_reasoning_streaming(
                    previous_raw, parser_current, parser_delta_text
                )
                if injected_this_chunk:
                    # Parser succeeded with the synthetic marker —
                    # latch so subsequent chunks don't re-inject.
                    _reasoning_close_injected = True
                if delta_msg is None:
                    continue
                pieces: list[tuple[str, str]] = []
                if delta_msg.reasoning:
                    reasoning = strip_special_tokens(delta_msg.reasoning)
                    if reasoning:
                        kept, overflow = _account_for_reasoning(reasoning)
                        if kept:
                            # See site A's note: intra-thinking
                            # whitespace separators must reach
                            # ``_gate_thinking_pieces`` so it can
                            # preserve them when a thinking block is
                            # already open (codex r4 MAJOR).
                            pieces.append(("thinking", kept))
                        if overflow:
                            # Codex round-7 BLOCKING #1: emitting
                            # overflow as a TEXT block while the parser
                            # is still logically in thinking would open
                            # an Anthropic ``content_block`` (text) that
                            # is semantically inconsistent with the
                            # parser's internal state. Force the parser
                            # flip in THIS same chunk by re-running the
                            # extractor with a synthetic ``</think>``
                            # against a LOCAL ``current`` (don't mutate
                            # ``accumulated_raw`` — round-6 invariant).
                            flip_succeeded = _reasoning_close_injected
                            if not _reasoning_close_injected:
                                # Codex round-10 BLOCKING #3: flip
                                # the latch AFTER success only — if
                                # the parser raises, next chunk
                                # retries the forced transition.
                                # Codex round-13 BLOCKING #3:
                                # position ``</think>`` at the CAP
                                # BOUNDARY using ``previous_raw +
                                # kept`` — not ``accumulated_raw``
                                # (which would put the marker AFTER
                                # the over-budget bytes). Stateful
                                # parsers must see the close at the
                                # exact kept-reasoning boundary so
                                # the overflow bytes are
                                # unambiguously past-cap content.
                                flip_previous = previous_raw + kept
                                flip_delta = "</think>"
                                flip_current = flip_previous + flip_delta
                                try:
                                    flip_msg = (
                                        reasoning_parser.extract_reasoning_streaming(
                                            flip_previous, flip_current, flip_delta
                                        )
                                    )
                                    _reasoning_close_injected = True
                                    flip_succeeded = True
                                except Exception as e:
                                    # Codex round-8 BLOCKING #3: when
                                    # the flip raises, the parser may
                                    # still be mid-think. Emitting
                                    # ``overflow`` as a TEXT
                                    # content_block would visibly mix
                                    # reasoning bytes into the
                                    # assistant message under a failed
                                    # transition. Suppress overflow on
                                    # flip failure and log; the client
                                    # may see a slightly-truncated
                                    # response — strictly better than
                                    # semantically-invalid content.
                                    logger.warning(
                                        "anthropic in-chunk close-marker flip "
                                        "raised on %r: %s — parser state may "
                                        "stay mid-think; suppressing %d-byte "
                                        "overflow on this chunk to avoid "
                                        "leaking reasoning bytes as content",
                                        type(reasoning_parser).__name__,
                                        e,
                                        len(overflow),
                                    )
                                    flip_msg = None
                                flip_content = (
                                    getattr(flip_msg, "content", None)
                                    if flip_msg is not None
                                    else None
                                )
                                if isinstance(flip_content, str) and flip_content:
                                    filtered_flip = tool_filter.process(flip_content)
                                    if filtered_flip:
                                        pieces.append(("text", filtered_flip))
                            if flip_succeeded:
                                filtered = tool_filter.process(overflow)
                                if filtered:
                                    pieces.append(("text", filtered))
                if delta_msg.content:
                    content = strip_special_tokens(delta_msg.content)
                    if content:
                        # Tool tags only appear in the content channel —
                        # filter still applies, but reasoning bypasses it.
                        filtered = tool_filter.process(content)
                        if filtered:
                            pieces.append(("text", filtered))
                if pieces:
                    events, current_block_type, block_index = _emit_content_pieces(
                        _gate_thinking_pieces(pieces, current_block_type),
                        current_block_type,
                        block_index,
                    )
                    for event in events:
                        yield event
                continue

            # No reasoning_parser path — keep the existing think_router
            # heuristic that detects `<think>` tags in the raw stream.
            content = strip_special_tokens(delta_text)
            if content:
                content = strip_special_tokens(content)

            if content:
                filtered = tool_filter.process(content)
                if not filtered:
                    continue
                pieces = think_router.process(filtered)
                events, current_block_type, block_index = _emit_content_pieces(
                    _gate_thinking_pieces(pieces, current_block_type),
                    current_block_type,
                    block_index,
                )
                for event in events:
                    yield event

    # Flush remaining from both filters
    remaining = tool_filter.flush()
    if remaining:
        # When reasoning_parser owns the split, route flushed tool-filter
        # content straight to text — `think_router` would mis-buffer it
        # for the same reason as above.
        if reasoning_parser:
            pieces_flush: list[tuple[str, str]] = [("text", remaining)]
        else:
            pieces_flush = think_router.process(remaining)
        events, current_block_type, block_index = _emit_content_pieces(
            _gate_thinking_pieces(pieces_flush, current_block_type),
            current_block_type,
            block_index,
        )
        for event in events:
            yield event

    if not reasoning_parser:
        flush_pieces = think_router.flush()
        if flush_pieces:
            events, current_block_type, block_index = _emit_content_pieces(
                _gate_thinking_pieces(flush_pieces, current_block_type),
                current_block_type,
                block_index,
            )
            for event in events:
                yield event

    # Close final content block
    if current_block_type is not None:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
        block_index += 1

    # Codex round-3 BLOCKING #2: if the reasoning cap latched on the
    # last engine chunk of the stream (terminal exact-boundary case OR
    # the model stopped immediately after overflow), the ``</think>``
    # close marker was never spliced into the parser. The thinking
    # block stays open in the Anthropic SSE shape — the
    # ``content_block_stop`` for the thinking index never gets a
    # matching text block, and any parser-held content past the cap is
    # lost. Force the injection here so a terminal cap-hit still flips
    # the parser to content and any trailing bytes are promoted to a
    # text block before stream end. Idempotent via
    # ``_reasoning_close_injected``.
    terminal_injection_attempted = False
    if (
        reasoning_parser is not None
        and _reasoning_cap_hit
        and not _reasoning_close_injected
    ):
        _reasoning_close_injected = True
        terminal_injection_attempted = True
        # Codex round-6 BLOCKING #1: build the parser's ``current``
        # argument LOCALLY rather than mutating the shared
        # ``accumulated_raw``. If the injection produces no content
        # (no held bytes / parser early-returns) and the subsequent
        # ``finalize_streaming(accumulated_raw)`` were to run, it
        # would parse a buffer that ends with the synthetic
        # ``</think>`` marker — potentially mis-classifying the forged
        # bytes as model output. Symmetric with the postprocessor and
        # responses-route fixes.
        previous_raw = accumulated_raw
        injected_delta = "</think>"
        local_current = previous_raw + injected_delta
        try:
            final_inject = reasoning_parser.extract_reasoning_streaming(
                previous_raw, local_current, injected_delta
            )
        except Exception as e:
            # Codex round-5 BLOCKING #2: an earlier draft emitted a
            # diagnostic string ``"[reasoning cap hit — parser flush
            # failed]"`` as an Anthropic text content_block, which
            # fabricates assistant content from an INTERNAL server
            # failure — clients see an "answer" that the model never
            # produced. Log the parser failure and leave the assistant
            # content empty. The route's existing 5xx / disconnect-
            # guard semantics handle truly catastrophic failures
            # upstream; a single reasoning-cap parser bug must not
            # invent text.
            logger.warning(
                "anthropic terminal close-marker injection raised on %r: %s — "
                "trailing reasoning content (if any) will not be promoted "
                "to a text block for this request",
                type(reasoning_parser).__name__,
                e,
            )
            final_inject = None
        if final_inject is not None and getattr(final_inject, "content", None):
            inject_content = strip_special_tokens(final_inject.content)
            if inject_content:
                filtered = tool_filter.process(inject_content)
                if filtered:
                    events, current_block_type, block_index = _emit_content_pieces(
                        [("text", filtered)], current_block_type, block_index
                    )
                    for event in events:
                        yield event
        # Close any block we opened above before falling through to the
        # finalize_streaming path.
        if current_block_type is not None:
            yield (
                f"event: content_block_stop\ndata: "
                f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            )
            block_index += 1
            current_block_type = None

    # Handle reasoning parser finalization
    # Codex round-4 BLOCKING #2 + round-6 BLOCKING #1: skip the
    # parser's non-stream finalize pass when the terminal injection
    # above ran at all (whether or not it produced content).
    #
    #   1. Injection emitted content — running ``finalize_streaming``
    #      next would re-emit the SAME bytes the streaming
    #      extraction just released (qwen3 / deepseek
    #      ``finalize_streaming`` is a whole-buffer re-parse).
    #   2. Injection produced no content — the parser already had
    #      its chance to flush via the forced ``</think>``. Re-running
    #      its non-stream pass on ``accumulated_raw`` (which excludes
    #      the synthetic marker per the round-5/6 local-buffer fix)
    #      could still re-classify cap-truncated reasoning as content
    #      via the non-stream parser's broader heuristics.
    #
    # When NO terminal injection was attempted (cap never fired, or
    # was already injected mid-stream), the finalize pass still runs
    # as the safety net for normal parser-held content.
    if reasoning_parser and accumulated_raw and not terminal_injection_attempted:
        final_msg = (
            reasoning_parser.finalize_streaming(accumulated_raw)
            if hasattr(reasoning_parser, "finalize_streaming")
            else None
        )
        if final_msg and final_msg.content:
            content = strip_special_tokens(final_msg.content)
            if content:
                accumulated_text = content
                yield (
                    f"event: content_block_start\ndata: "
                    f"{json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                )
                delta_event = {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                block_index += 1

    # Check for tool calls — prefer engine-surfaced structured payload
    # (HarmonyStreamingRouter via openai-harmony's StreamableParser)
    # over text-based extraction. Same fall-through contract the
    # non-streaming branch uses.
    _, tool_calls = _parse_tool_calls_with_parser(
        accumulated_text,
        openai_request,
        structured_tool_calls=accumulated_structured_tool_calls or None,
    )

    # H-05: same un-pinned-tool drop as the non-streaming branch —
    # see the non-stream call site for the policy rationale. Run
    # BEFORE validation so the F-220 enforcer never sees the dropped
    # tool's schema-violating arguments.
    if tool_calls and openai_request.tool_choice:
        tool_calls = _filter_tool_calls_by_tool_choice(
            tool_calls, openai_request.tool_choice
        )

    # F-220: enforce JSON-schema validation on the model's emitted
    # tool_call arguments. On the streaming branch, headers are already
    # sent so a mid-stream ``HTTPException`` cannot be returned as a 400
    # response. Instead, surface the validation error as an Anthropic
    # SSE ``error`` event (``invalid_request_error``) and drop the
    # offending tool_use blocks so the client can recover. Matches the
    # non-stream branch's 400 contract in spirit while staying within
    # the Anthropic streaming protocol.
    tool_validation_error: str | None = None
    if tool_calls and openai_request.tools:
        try:
            _validate_tool_call_params(tool_calls, openai_request.tools)
        except HTTPException as exc:
            tool_validation_error = (
                exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            )
            tool_calls = []

    if tool_validation_error:
        error_event = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": tool_validation_error,
            },
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_index = block_index + i
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}

            tool_block_start = {
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(tool_block_start)}\n\n"

            input_json = json.dumps(tool_input)
            input_delta = {
                "type": "content_block_delta",
                "index": tool_index,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

    stop_reason = "tool_use" if tool_calls else "end_turn"

    # Anthropic-side cache fields mirror what the non-streaming adapter
    # at ``api/anthropic_adapter.openai_to_anthropic`` produces. Per
    # Anthropic's docs the three input fields are mutually exclusive
    # (``total_input = input + cache_read + cache_creation``), so
    # ``input_tokens`` is the *non-cached* share, NOT the whole prompt.
    # ``cache_creation_input_tokens`` is intentionally omitted —
    # Anthropic uses it for tokens written between explicit
    # ``cache_control`` breakpoints (billed 1.25x), which has no
    # analog on a local engine. Cache field stays absent when the
    # engine didn't report a hit (e.g. dflash, MLLM).
    # Clamp once so cache_read + input_tokens cannot exceed prompt_tokens —
    # an over-reported cache count from the engine would otherwise emit an
    # impossible usage block where ``cache_read_input_tokens > prompt_tokens``.
    # Mirrors ``openai_to_anthropic`` in ``api/anthropic_adapter.py``.
    cached_tokens = min(cached_tokens, prompt_tokens)
    usage_payload: dict[str, int] = {
        "input_tokens": prompt_tokens - cached_tokens,
        "output_tokens": completion_tokens,
    }
    if cached_tokens:
        usage_payload["cache_read_input_tokens"] = cached_tokens
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage_payload,
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages (stream): prompt={prompt_tokens} + completion={completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
