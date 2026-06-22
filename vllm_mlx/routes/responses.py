# SPDX-License-Identifier: Apache-2.0
"""OpenAI Responses API endpoint — /v1/responses.

Stateless shim that lets Codex CLI (and any other Responses-API client)
talk to rapid-mlx as if it were OpenAI. Translates Responses → Chat,
runs inference through the existing engine, translates back into the
seven SSE events Codex CLI parses (``response.created``,
``response.output_item.added``, ``response.output_text.delta``,
``response.function_call_arguments.delta``, ``response.output_item.done``,
``response.completed``, ``response.failed``).

Statelessness: ``previous_response_id`` returns 400. Codex CLI doesn't
use that field (openai/codex#3841) — it re-sends the full conversation
history every turn in ``input``.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from ..api.response_format_metrics import (
    incr_strict_request,
    incr_strict_violation,
)
from ..api.responses_adapter import (
    openai_to_responses,
    request_uses_computer_use,
    responses_to_openai,
    validate_responses_tool_choice,
    validate_responses_tool_types,
)
from ..api.responses_models import ResponsesRequest
from ..api.tool_calling import (
    check_schema_validity,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    is_strict_json_schema,
    validate_output_against_schema,
)
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
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    SSE_RESPONSE_HEADERS,
    _apply_reasoning_cutoff_notice,
    _build_usage,
    _check_admission_or_503,
    _disconnect_guard,
    _effective_enable_thinking,
    _finalize_content_and_reasoning,
    _parse_tool_calls_with_parser,
    _release_admission_unless_committed,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    enforce_context_length_for_messages,
    get_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolved_sampling_kwargs(openai_request: ChatCompletionRequest) -> dict:
    """Resolve sampling params through the 4-layer cascade.

    Mirrors the helper in routes/anthropic.py so ``/v1/responses`` users
    get the same alias / generation_config defaults as ``/v1/messages``
    and ``/v1/chat/completions``.
    """
    out = {
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        "stop": getattr(openai_request, "stop", None),
    }
    out.update(build_extended_sampling_kwargs(openai_request))
    return out


def _should_start_in_thinking(chat_template: str, enable_thinking: bool | None) -> bool:
    """Same heuristic as routes/anthropic.py: stream that starts inside an
    implicit ``<think>`` block should be routed as reasoning until the
    closing tag. Bypass when thinking is explicitly disabled."""
    if enable_thinking is False:
        return False
    return "<think>" in chat_template and "add_generation_prompt" in chat_template


def _enforce_responses_tool_choice(
    tool_calls: list | None,
    responses_request: ResponsesRequest,
    openai_request: ChatCompletionRequest,
) -> list | None:
    """Mirror of the chat-route post-parse forced-choice synthesis.

    The chat route synthesises a stub tool_call when the model produces
    text under ``tool_choice="required"`` (single-tool case) or under
    the named-function form (target unambiguous). The /v1/responses
    lane skipped this step, so Yuki F6 saw zero ``function_call`` items
    even though the contract guarantees one.

    Synthesis rules (parity with chat.py ~L1880):
      * ``"required"`` + exactly one tool → synthesise a call to it
      * ``"required"`` + multiple tools + no model call → 422 with the
        same diagnostic chat.py uses (codex r1 BLOCKING #1 on PR #817).
        Silently degrading to ``auto`` would let a multi-tool ``required``
        request return zero tool_calls and break the contract this PR
        claims to restore.
      * ``{"type":"function","name":X}`` + X in submitted tools →
        synthesise a call to X
      * ``auto`` / ``none`` / unrecognised shapes → pass through.
    """
    from ..routes.chat import _synthesize_forced_tool_call

    tc = responses_request.tool_choice
    if tc is None or not openai_request.tools:
        return tool_calls
    # Codex r3 BLOCKING #1 (PR #817): for the named-function form,
    # validate that EVERY model-produced tool_call targets the pinned
    # name. A model that called a different tool (``ping`` when
    # ``pong`` was pinned) violates the contract just as much as a
    # text-only response — raise 422 with the same diagnostic
    # chat.py uses (~L1969-L1978).
    if tool_calls and isinstance(tc, dict) and tc.get("type") == "function":
        _named_target = tc.get("name") or (tc.get("function") or {}).get("name")
        if _named_target:
            mismatched = [
                tc_obj
                for tc_obj in tool_calls
                if (tc_obj.function.name or "") != _named_target
            ]
            if mismatched:
                _names = [m.function.name for m in mismatched]
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": {
                            "message": (
                                f"tool_choice pinned function "
                                f"{_named_target!r} but the model emitted "
                                f"calls to {_names}. Local inference "
                                "cannot decoder-enforce a specific "
                                "function; retry with a more direct "
                                "user message."
                            ),
                            "type": "invalid_request_error",
                            "code": "tool_choice_named_mismatch",
                            "param": "tool_choice.name",
                        }
                    },
                )
    # Only coerce when the model surfaced NO calls — a real model
    # response that called the right tool already satisfies the
    # contract.
    if tool_calls:
        return tool_calls
    if tc == "required":
        if len(openai_request.tools) == 1:
            name = openai_request.tools[0].function.get("name")
            if name:
                logger.info(
                    "tool_choice='required' on /v1/responses produced no "
                    "tool_calls; synthesising a call to the sole "
                    "available tool %r to honour the OpenAI tool_call-"
                    "guaranteed contract (Yuki F6).",
                    name,
                )
                return [_synthesize_forced_tool_call(name)]
        # Multi-tool ``required`` with no model call — local inference
        # cannot guess which of N tools the user intended. Chat.py
        # raises 422 in the same situation (~L1891-1902); mirror that
        # so the Responses surface does not silently violate the
        # tool_call-guaranteed contract.
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": (
                        'tool_choice="required" but the model returned a '
                        "text response with no tool_calls. Local "
                        "inference has no decoder-level constraint; the "
                        "system-prompt enforcement was insufficient for "
                        "this prompt. Retry with a more concrete user "
                        "message or use tool_choice="
                        '{"type":"function","name":...} to pin a '
                        "specific tool."
                    ),
                    "type": "invalid_request_error",
                    "code": "tool_choice_required_unfulfilled",
                    "param": "tool_choice",
                }
            },
        )
    if isinstance(tc, dict) and tc.get("type") == "function":
        target = tc.get("name") or (tc.get("function") or {}).get("name")
        if not target:
            return tool_calls
        submitted = {
            t.function.get("name") for t in openai_request.tools if t.type == "function"
        }
        if target in submitted:
            logger.info(
                "tool_choice pinned function %r on /v1/responses produced "
                "no tool_calls; synthesising a call with empty arguments "
                "to honour the OpenAI tool_call-guaranteed contract "
                "(Yuki F6).",
                target,
            )
            return [_synthesize_forced_tool_call(target)]
    return tool_calls


@router.post(
    "/v1/responses",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
    ],
)
async def create_response(request: Request):
    """OpenAI Responses API entry point.

    Codex CLI hardcodes ``stream: true`` and sends the full
    conversation history in ``input[]`` each turn, so the streaming
    path is the hot path.
    """
    body = await request.json()
    # ``ResponsesRequest`` is constructed manually (not as a FastAPI body
    # parameter). The raw :class:`pydantic.ValidationError` it can raise
    # is now caught by the global ``_pydantic_validation_handler`` in
    # ``middleware.exception_handlers`` (H-17), which routes it through
    # the same sanitized 400 envelope used by ``/v1/chat/completions``.
    # The earlier per-route ``HTTPException(detail=str(e))`` leaked the
    # model class name (``ResponsesRequest``), the pinned pydantic
    # version (``errors.pydantic.dev/2.13/...``), and any attacker-
    # supplied ``input_value`` blob — see Rhea r0.8.1 audit.
    responses_request = ResponsesRequest(**body)

    # Statelessness gate — see module docstring. Codex CLI does not set
    # this field; clients that DO use it would get silent prompt loss
    # on retries because we have no response store, so 400 loudly.
    if responses_request.previous_response_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "previous_response_id is not supported by this server — "
                "rapid-mlx is a stateless Responses API shim. Re-send the "
                "full conversation history in the `input` field each turn."
            ),
        )

    # Yuki F13 (0.8.5 dogfood): pre-engine tool-type allowlist. Anything
    # outside ``SUPPORTED_RESPONSES_TOOL_TYPES`` 400s with a clear
    # envelope BEFORE we admit a scheduler slot — pre-0.8.5 the route
    # silently accepted ``web_search`` / ``computer_20251022`` /
    # ``file_search`` and the client thought the tool was being invoked.
    validate_responses_tool_types(responses_request.tools)
    # Yuki F6 (0.8.5 dogfood): mirror the chat-completions tool_choice
    # gate so ``required`` / named-function tool_choice REJECTS shapes
    # that cannot be honoured (e.g. ``required`` with empty tools, named
    # function not in tools). The post-parse synthesis path below
    # COERCES a tool_call when the model didn't emit one — without it,
    # the named-function form silently degraded to ``auto``.
    validate_responses_tool_choice(
        responses_request.tool_choice, responses_request.tools
    )

    # Reuse the Claude-Code / Codex bypass from #557: ``claude-*``,
    # ``gpt-*`` model names pass through to the loaded engine instead of
    # 404'ing on _validate_model_name. Codex sends ``gpt-5``,
    # ``gpt-5-codex``, etc. — none of which match a local alias.
    if not (responses_request.model or "").startswith(("claude-", "gpt-")):
        _validate_model_name(responses_request.model)
    engine = get_engine(responses_request.model)

    # Pre-flight admission — same C4 reservation shape the other two
    # routes use. ``_admission_committed`` flips to True when the
    # streaming path takes over so ``_disconnect_guard`` owns release.
    _check_admission_or_503(engine)
    _admission_committed = False
    try:
        _log_request(responses_request)

        cfg_for_log = get_config()
        if (
            responses_request.model
            and cfg_for_log.model_name
            and responses_request.model != cfg_for_log.model_name
        ):
            logger.info(
                "Responses /v1/responses: request model=%r served by loaded engine=%r",
                responses_request.model,
                cfg_for_log.model_name,
            )

        # F-034 (and any future ``ChatCompletionRequest``-layer validator):
        # the adapter materializes a fresh ``ChatCompletionRequest`` from
        # the Responses body, which now rejects unsatisfiable combinations
        # (e.g. ``tool_choice="required"`` with no ``tools``). The
        # resulting :class:`pydantic.ValidationError` bubbles to the
        # global ``_pydantic_validation_handler`` (H-17) which routes
        # it through the sanitized 400 envelope — no more ``str(e)``
        # echo that leaked the model class name and pydantic version.
        openai_request = responses_to_openai(responses_request)

        # H-06: ``text.format`` with strict json_schema on /v1/responses
        # was suggestion-only — the route went straight to
        # ``engine.chat()`` and dropped the constraint. When the engine
        # cannot honor the contract (``[guided]`` extra missing), 400
        # loudly instead of silently emitting unconstrained tokens.
        # Counter tick mirrors the chat-route gate so the operator
        # dashboards see uniform traffic shape across both surfaces.
        _rf = getattr(openai_request, "response_format", None)
        if is_strict_json_schema(_rf):
            _schema = extract_json_schema_for_guided(_rf)
            incr_strict_request()
            # Codex r3 BLOCKING #3 parity: a malformed strict request
            # without an extractable schema must fail closed (400)
            # not fall through to unconstrained ``engine.chat`` —
            # mirrors the chat-route gate.
            if not _schema:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "text.format strict=true requires a "
                                "non-empty schema. The request set "
                                "strict=true but the schema field is "
                                "missing or empty — the strict contract "
                                "cannot be enforced without one."
                            ),
                            "type": "invalid_request_error",
                            "code": "strict_schema_required",
                            "param": "text.format.schema",
                        }
                    },
                )
            # Codex r4 NIT #5 parity: validate the user-supplied
            # schema BEFORE generation so an invalid JSON Schema
            # (e.g. ``"type":"objct"`` typo) surfaces as a 400
            # ``invalid_strict_schema`` pointing at the client's
            # malformed input — instead of falling into the
            # post-decode validator and surfacing as a 502
            # ``strict_schema_violation`` (server-side breach shape).
            _schema_ok, _schema_err = check_schema_validity(_schema)
            if not _schema_ok:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "text.format.schema is not a valid "
                                f"JSON Schema document: {_schema_err}. "
                                "Fix the schema and retry."
                            ),
                            "type": "invalid_request_error",
                            "code": "invalid_strict_schema",
                            "param": "text.format.schema",
                        }
                    },
                )
            if openai_request.tools:
                # Parity with the chat-route ``strict_with_tools_unsupported``
                # gate: constrained-decoding grammar and tool-call grammar
                # are mutually exclusive on this engine.
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "text.format strict=true cannot be combined "
                                "with 'tools' — the constrained-decoding "
                                "grammar is mutually exclusive with the "
                                "tool-call grammar. Drop one or the other "
                                "and retry."
                            ),
                            "type": "invalid_request_error",
                            "code": "strict_with_tools_unsupported",
                            "param": "text.format.strict",
                        }
                    },
                )
            # Codex r4 NIT #4: check the strict+stream gate BEFORE
            # the missing-extra gate. Strict streaming on
            # /v1/responses is structurally unsupported here
            # regardless of whether [guided] is installed (the
            # constrained-decoding path is buffered-only on this
            # surface), so telling a strict+stream caller to
            # ``pip install rapid-mlx[guided]`` would be
            # misleading — installing the extra still wouldn't
            # let them use strict+stream on /v1/responses. Naming
            # the actual escape hatches first (drop stream=true,
            # or switch to /v1/chat/completions) is more
            # actionable.
            if responses_request.stream:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "text.format strict=true with stream=true "
                                "is not supported on /v1/responses — "
                                "constrained decoding on this surface is "
                                "buffered-only. Either drop stream=true "
                                "(non-stream strict response is honored) "
                                "or use /v1/chat/completions which "
                                "supports strict+streaming via the "
                                "buffered-guided SSE helper."
                            ),
                            "type": "invalid_request_error",
                            "code": "strict_stream_unsupported",
                            "param": "text.format.strict",
                        }
                    },
                )
            if not engine.supports_guided_generation:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "text.format with strict=true requires "
                                "the [guided] optional extra. Install "
                                "with: pip install 'rapid-mlx[guided]'"
                            ),
                            "type": "invalid_request_error",
                            "code": "guided_extra_required",
                            "param": "text.format.strict",
                        }
                    },
                )

        # Context-length pre-check — same DoS gate the chat/completions/
        # anthropic routes enforce. Runs BEFORE the stream branch so
        # streaming clients can't bypass by setting ``stream: true``.
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

        if responses_request.stream:
            _admission_committed = True
            # C-01 force-abort: holder list the engine populates with
            # the admitted scheduler request id; the disconnect_guard
            # reads it and force-calls scheduler.abort_request on
            # client disconnect.
            _resp_rid_holder: list[str | None] = [None]
            return StreamingResponse(
                _disconnect_guard(
                    _stream_responses(
                        engine,
                        openai_request,
                        responses_request,
                        request_id_holder=_resp_rid_holder,
                    ),
                    request,
                    engine=engine,
                    request_id_holder=_resp_rid_holder,
                ),
                media_type="text/event-stream",
                # ``SSE_RESPONSE_HEADERS`` (Cache-Control no-cache/no-transform +
                # X-Accel-Buffering: no) wraps the legacy ``Connection: keep-alive``
                # already on this route. F-073 anti-buffering parity with the
                # chat / completions / anthropic streaming responses.
                headers={**SSE_RESPONSE_HEADERS, "Connection": "keep-alive"},
            )

        return await _non_stream(engine, openai_request, responses_request, request)
    finally:
        _release_admission_unless_committed(engine, _admission_committed)


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------


async def _non_stream(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    request: Request,
) -> Response:
    cfg = get_config()
    created_at = int(time.time())

    messages, _images, _videos = extract_multimodal_content(
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

    resolved_thinking = _resolve_enable_thinking(openai_request)
    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    start_time = time.perf_counter()
    timeout = cfg.default_timeout

    # H-06: when the request asks for strict json_schema, route
    # through ``engine.generate_with_schema`` for outlines-backed
    # constrained decoding. The route gate above already 400'd if
    # guided was unavailable, so reaching here under strict means
    # ``supports_guided_generation`` was True. Under strict we DO
    # NOT fall back to unconstrained ``engine.chat`` on guided
    # failure — that turns ``strict=true`` back into best-effort
    # output (codex r2 BLOCKING #1). Instead we propagate the
    # guided-coroutine failure as 502 ``strict_schema_violation``
    # so the client sees the contract breach explicitly.
    _rf_for_strict = getattr(openai_request, "response_format", None)
    _strict_schema = (
        extract_json_schema_for_guided(_rf_for_strict)
        if is_strict_json_schema(_rf_for_strict)
        else None
    )

    # Codex r3 BLOCKING #1: wrap ONLY the guided coroutine creation
    # in our exception translator, not ``_wait_with_disconnect``
    # itself. ``_wait_with_disconnect`` raises
    # ``asyncio.TimeoutError`` / client-disconnect exceptions that
    # the outer route relies on to return the canonical 408 / 499 /
    # 503 envelopes — translating those to 502
    # ``strict_schema_violation`` would mask client-disconnect /
    # timeout as a server-side contract breach.
    #
    # Strategy: build the guided coroutine OUTSIDE the
    # ``_wait_with_disconnect`` call but INSIDE a dedicated
    # ``try`` (the one starting at ``try: _guided_coro = ...``
    # below). Sync setup errors from
    # ``engine.generate_with_schema(...)`` — AttributeError,
    # NotImplementedError, outlines-import errors, kwargs
    # collisions if the sanitization at line ~450 ever regressed —
    # materialize synchronously and the tight try catches them.
    # ``_wait_with_disconnect`` then handles the actual await
    # with its own timeout/disconnect semantics intact, in a
    # SEPARATE outer try below.
    #
    # Codex r8 BLOCKING (false positive): the round-8 review
    # claimed the call was "before the surrounding try" — see
    # line 453 below, the call site IS inside the try. The
    # ``test_strict_true_responses_sync_setup_failure_returns_502``
    # test in test_response_format_json_schema_strict.py pins
    # this behavior so any future refactor that moves the call
    # outside the try is caught.
    if _strict_schema and engine.supports_guided_generation:
        # Codex r5 BLOCKING: ``chat_kwargs`` is the merged
        # ``_resolved_sampling_kwargs`` + tools/thinking flags blob.
        # If any upstream resolver ever surfaces a ``raise_on_failure``
        # key (e.g. a future ``extra_body`` passthrough, or an
        # accidental sampling-param alias), the explicit
        # ``raise_on_failure=True`` below would TypeError with
        # "got multiple values for keyword argument" BEFORE
        # constrained decoding ran — and the outer ``except Exception``
        # would translate that operator-side wiring bug into a
        # 502 ``strict_schema_violation`` (server contract-breach
        # shape), masking the root cause from the client and from
        # logs. Sanitize the kwargs dict here so the strict gate
        # OWNS the value and no caller can collide with it.
        _guided_kwargs = {
            k: v for k, v in chat_kwargs.items() if k != "raise_on_failure"
        }
        try:
            _guided_coro = engine.generate_with_schema(
                messages=messages,
                json_schema=_strict_schema,
                raise_on_failure=True,
                **_guided_kwargs,
            )
        except HTTPException:
            raise
        except Exception as guided_err:
            logger.warning(
                "Guided generation setup failed on /v1/responses strict path: %s",
                guided_err,
            )
            incr_strict_violation()
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "message": (
                            "strict response_format requested but "
                            "constrained decoding failed: "
                            f"{type(guided_err).__name__}. Investigate "
                            "the server logs and the "
                            "rapid_mlx_response_format_strict_violations_total "
                            "metric."
                        ),
                        "type": "api_error",
                        "code": "strict_schema_violation",
                        "param": "text.format.strict",
                    }
                },
            ) from guided_err
    else:
        _guided_coro = None

    try:
        if _guided_coro is not None:
            # Codex r3 BLOCKING #1: the guided await runs under the
            # same _wait_with_disconnect contract as the
            # unconstrained path — timeout/disconnect surface as
            # the route's standard 408/499/503 envelopes (handled
            # by the outer try/except). Any guided-specific
            # runtime failure (outlines grammar error during
            # await, etc.) is translated to 502 below by checking
            # the exception class explicitly so cancellation /
            # timeout aren't misclassified.
            try:
                output = await _wait_with_disconnect(
                    _guided_coro,
                    request,
                    timeout=timeout,
                )
            except HTTPException:
                raise
            except (TimeoutError, asyncio.TimeoutError):
                # _wait_with_disconnect surfaces these from its
                # own timeout machinery — they belong to the
                # outer route's standard timeout shape, NOT to
                # the strict_schema_violation contract.
                raise
            except asyncio.CancelledError:
                # Client disconnect / cancellation — same as above,
                # belongs to the route's standard cancellation
                # path, not the strict contract.
                raise
            except Exception as guided_err:
                logger.warning(
                    "Guided generation failed mid-await on /v1/responses "
                    "strict path: %s",
                    guided_err,
                )
                incr_strict_violation()
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": {
                            "message": (
                                "strict response_format requested but "
                                "constrained decoding failed: "
                                f"{type(guided_err).__name__}. Investigate "
                                "the server logs and the "
                                "rapid_mlx_response_format_strict_violations_total "
                                "metric."
                            ),
                            "type": "api_error",
                            "code": "strict_schema_violation",
                            "param": "text.format.strict",
                        }
                    },
                ) from guided_err
        else:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                request,
                timeout=timeout,
            )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — match other routes' error shape
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
        # Multimodal fetch failures + MLLM per-batch-cap errors → 400
        # (parity with chat route, #457 / #682). The MLLM scheduler
        # classifier already treats both as client-actionable; this route
        # must map both to 400 or the /v1/responses surface returns a 500
        # for what is really an oversized-image / oversized-prompt user
        # error.
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
        f"Responses: {output.completion_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )

    # H-06 (codex r2): post-decode validation under strict mode is a
    # HARD contract — a knowingly schema-invalid 200 violates
    # OpenAI's ``strict=true`` semantics. Counter ticks for ops
    # visibility, then 502 so the client sees the contract breach
    # instead of silently consuming garbage.
    if _strict_schema and output is not None:
        ok, err = validate_output_against_schema(output.text or "", _strict_schema)
        if not ok:
            incr_strict_violation()
            logger.warning(
                "Strict json_schema response failed post-decode validation "
                "on /v1/responses: %s",
                err,
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "message": (
                            "strict response_format violated: model output "
                            f"did not validate against the supplied schema ({err}). "
                            "This indicates the constrained-decoding path silently "
                            "degraded; investigate the server logs and the "
                            "rapid_mlx_response_format_strict_violations_total metric."
                        ),
                        "type": "api_error",
                        "code": "strict_schema_violation",
                        "param": "text.format",
                    }
                },
            )

    engine_tool_calls = getattr(output, "tool_calls", None)
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text, openai_request, structured_tool_calls=engine_tool_calls
    )

    # Yuki F6 (0.8.5 dogfood): mirror the chat-route ``tool_choice``
    # coercion. The local engine has no decoder-level FSM constraint, so
    # ``required`` / named-function ``tool_choice`` rely on post-parse
    # synthesis to honour the OpenAI ``tool_call guaranteed`` contract.
    # Without this, both shapes silently degraded to ``auto`` and Yuki
    # F6 saw zero tool_calls on the wire.
    tool_calls = _enforce_responses_tool_choice(
        tool_calls, responses_request, openai_request
    )

    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=output.raw_text or output.text,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=cfg.reasoning_parser,
        engine_reasoning_text=getattr(output, "reasoning_text", "") or "",
        enable_thinking=_effective_enable_thinking(
            resolved_thinking, cfg.model_path or cfg.model_name
        ),
        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # Forwarded from ``ResponsesRequest.reasoning_max_tokens`` via
        # the Responses → OpenAI adapter. None → no cap (back-compat).
        reasoning_max_tokens=getattr(openai_request, "reasoning_max_tokens", None),
        # r5-D shared finalize-on-truncation plug — see chat.py for
        # the rationale. Forwarded so the /v1/responses path picks up
        # the same gemma4 / glm4 / minimax fixes.
        finish_reason=getattr(output, "finish_reason", None),
    )

    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)

    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # R-01 (was H-01): /v1/responses mirror of the opt-in cutoff
    # sentinel. Default-off — the Responses envelope already reports
    # ``status="incomplete"`` and
    # ``usage.output_tokens_details.reasoning_tokens``, so SDK consumers
    # have an unambiguous structured truncation signal without any
    # synthetic ``output_text`` block. When the env knob
    # ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1`` is set, the helper restores
    # the legacy literal-text cue for callers who want it. The Responses
    # surface intentionally does NOT run
    # ``_rescue_silent_drop_from_reasoning`` (this endpoint never
    # carried the issue#569 silent-drop pre-history), so the helper sees
    # a broader predicate set here than on chat/anthropic — that scope
    # is fine because the helper itself owns all the gates.
    final_content = _apply_reasoning_cutoff_notice(
        final_content,
        reasoning_text,
        tool_calls,
        finish_reason,
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

    responses_response = openai_to_responses(
        openai_response,
        model=cfg.model_name or responses_request.model,
        request=responses_request,
        created_at=created_at,
    )
    return Response(
        content=responses_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Streaming path — emits the 7 SSE events Codex CLI parses
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event in Responses-API shape.

    Codex parses ``event: <name>\\ndata: <json>\\n\\n`` framing — same as
    chat-completions and Anthropic streams. No ``data: [DONE]`` here;
    that sentinel is chat-completions-only.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _emit_function_call_item(tc, output_index: int) -> AsyncIterator[str]:
    """Stream the SSE event triplet for a single ``function_call`` item.

    Sequence: ``response.output_item.added`` →
    ``response.function_call_arguments.delta`` →
    ``response.output_item.done``. Args are sent in a single delta
    because the underlying engine doesn't surface per-token tool-call
    streaming yet (Codex CLI concatenates either way).
    """
    fc_id = f"fc_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": fc_id,
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": "",
                "status": "in_progress",
            },
        },
    )
    yield _sse(
        "response.function_call_arguments.delta",
        {
            "type": "response.function_call_arguments.delta",
            "item_id": fc_id,
            "output_index": output_index,
            "delta": tc.function.arguments or "",
        },
    )
    yield _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": fc_id,
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments or "",
                "status": "completed",
            },
        },
    )


async def _emit_computer_call_item(tc, output_index: int) -> AsyncIterator[str]:
    """Stream the SSE event pair for one Computer-Use ``computer_call``
    item (Ana C-06, 0.8.5 dogfood).

    Sequence: ``response.output_item.added`` →
    ``response.output_item.done``. There is no per-token args delta
    event for ``computer_call`` in the OpenAI spec — the entire
    ``action`` envelope ships in the ``done`` payload.
    """
    # Lazy import to avoid the route module circular-importing the
    # adapter at module load time (the adapter imports types from
    # ``responses_models`` which the route also imports).
    from ..api.responses_adapter import _parse_computer_action

    cu_id = f"cu_{uuid.uuid4().hex[:24]}"
    action = _parse_computer_action(tc.function.arguments or "")
    yield _sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "computer_call",
                "id": cu_id,
                "call_id": tc.id,
                "status": "in_progress",
                "action": action,
                "pending_safety_checks": [],
            },
        },
    )
    yield _sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "type": "computer_call",
                "id": cu_id,
                "call_id": tc.id,
                "status": "completed",
                "action": action,
                "pending_safety_checks": [],
            },
        },
    )


async def _stream_responses(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    *,
    request_id_holder: list | None = None,
) -> AsyncIterator[str]:
    """Stream a Responses-API SSE event sequence Codex CLI can parse.

    Event order Codex expects:
      1. ``response.created`` — once, before any deltas
      2. ``response.output_item.added`` (message item) — when first text
         delta arrives
      3. ``response.output_text.delta`` — each chunk of assistant text
      4. ``response.output_item.done`` (message item) — when text ends,
         before any tool_calls
      5. For each tool call:
         ``response.output_item.added`` (function_call item) +
         ``response.function_call_arguments.delta`` (full JSON args) +
         ``response.output_item.done`` (function_call item)
      6. ``response.completed`` — terminal event, carries final usage

    Errors emit ``response.failed`` then close. Codex treats
    stream-close-without-``response.completed`` as a hard failure, so
    we always finalize.
    """
    cfg = get_config()
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    start_time = time.perf_counter()
    served_model = cfg.model_name or responses_request.model

    # response.created — Codex needs this before any deltas.
    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": served_model,
                "output": [],
            },
        },
    )
    try:
        messages, _images, _videos = extract_multimodal_content(
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
        resolved_thinking = _resolve_enable_thinking(openai_request)
        if resolved_thinking is not None:
            chat_kwargs["enable_thinking"] = resolved_thinking
        # C-01: thread the request_id holder so disconnect_guard can
        # force-call scheduler.abort_request on client RST.
        if request_id_holder is not None:
            chat_kwargs["request_id_holder"] = request_id_holder

        accumulated_text = ""
        accumulated_raw = ""
        accumulated_structured_tool_calls: list[dict] = []
        tool_filter = StreamingToolCallFilter()

        # Yuki F6 codex r1 BLOCKING #2 (PR #817): when the request
        # forces a tool_choice (``required`` or named-function), the
        # message item MUST NOT be emitted if synthesis fires after
        # generation — otherwise the client sees both an
        # ``output_text`` message AND a synthesised tool_call, which
        # violates the OpenAI Responses ``tool_call-guaranteed``
        # contract. Solution: buffer text deltas in ``deferred_text``
        # under forced-choice mode; at end-of-stream, decide to either
        # flush them (model produced a real tool_call so synthesis won't
        # fire) or drop them (synthesis will fire — message item is
        # suppressed entirely). For non-forced choice, the legacy
        # streaming path is preserved (lazy message-item open on first
        # delta).
        _forced_tc = responses_request.tool_choice
        _forced_choice_active = (
            openai_request.tools is not None
            and openai_request.tools
            and (
                _forced_tc == "required"
                or (
                    isinstance(_forced_tc, dict)
                    and _forced_tc.get("type") == "function"
                )
            )
        )
        deferred_text: list[str] = []

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

        # Lazy message-item state. We do NOT emit the message
        # output_item.added until we have actual user-facing text to stream
        # — a turn that is pure tool_calls should not emit a phantom empty
        # message item.
        message_item_id: str | None = None
        message_output_index: int | None = None
        message_open = False

        # Per-request reasoning parser instance (matches anthropic.py).
        reasoning_parser = None
        if cfg.reasoning_parser_name:
            try:
                from ..reasoning import get_parser

                reasoning_parser = get_parser(cfg.reasoning_parser_name)()
            except Exception:
                pass
        if chat_kwargs.get("enable_thinking") is False:
            reasoning_parser = None
        if reasoning_parser:
            reasoning_parser.reset_state()

        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # Responses SSE drops reasoning to the floor (Codex doesn't read
        # ``response.reasoning_text.delta`` in v1) so the cap's primary
        # job here is to RECLASSIFY: once the budget is exhausted, any
        # further reasoning bytes become ``response.output_text.delta``
        # so the user actually sees a reply instead of an infinite
        # silent thinking block.
        _reasoning_cap = getattr(responses_request, "reasoning_max_tokens", None)
        _reasoning_tokens_emitted = 0
        _reasoning_cap_hit = False
        _reasoning_close_injected = False

        def _account_for_reasoning(text: str) -> tuple[str, str, bool]:
            """Returns ``(kept_reasoning, overflow_content, just_hit)``.

            Codex round-12 BLOCKING #2: cumulative-CHARACTER accounting
            against ``cap * 4`` (not per-chunk ceiling). The earlier
            ``max(1, ceil(len/4))`` made fragmented reasoning deltas
            consume more tokens than the same contiguous text, so the
            cap fired at different points depending only on SSE chunk
            boundaries. Now identical model output hits the cap at the
            same character offset regardless of chunking — matches
            ``helpers._apply_reasoning_cap`` (non-stream) AND the
            postprocessor's cumulative-char path.

            The shared ``_reasoning_tokens_emitted`` counter now holds
            CHARACTERS post-round-12 (name kept for back-compat). The
            cap *4 limit lives in ``_reasoning_max_chars`` captured
            from the request via the enclosing closure.
            """
            nonlocal _reasoning_tokens_emitted, _reasoning_cap_hit
            if _reasoning_cap is None or not text:
                return text, "", False
            if _reasoning_cap_hit:
                return "", text, False
            max_chars = _reasoning_cap * 4
            new_total_chars = _reasoning_tokens_emitted + len(text)
            if new_total_chars < max_chars:
                _reasoning_tokens_emitted = new_total_chars
                return text, "", False
            if new_total_chars == max_chars:
                # Exact-boundary latch (codex round-2 BLOCKING #3).
                _reasoning_tokens_emitted = new_total_chars
                _reasoning_cap_hit = True
                return text, "", True
            remaining_chars = max_chars - _reasoning_tokens_emitted
            keep_chars = max(0, remaining_chars)
            _reasoning_tokens_emitted = max_chars
            _reasoning_cap_hit = True
            return text[:keep_chars], text[keep_chars:], True

        # Yuki F8 (0.8.5 dogfood): track whether the content_part has
        # been opened so the streaming sequence emits
        # ``response.content_part.added`` exactly once per message item
        # (before the first text delta).
        content_part_open = False

        async def _open_message_item() -> list[str]:
            """Emit response.output_item.added + response.content_part.added.

            Returns the event strings so callers can yield them in order.
            The bookkeeping for ``message_open`` / ``content_part_open``
            lives here so the open/close pair stays symmetric.

            Yuki F8: the OpenAI Responses SSE spec puts
            ``response.content_part.added`` between the message item-
            added event and the first text delta. Pre-0.8.5 this event
            was missing; clients gating UI state on it never saw the
            ``output_text`` block open.
            """
            nonlocal \
                message_item_id, \
                message_output_index, \
                message_open, \
                content_part_open
            message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
            message_output_index = 0
            message_open = True
            content_part_open = True
            return [
                _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": message_output_index,
                        "item": {
                            "type": "message",
                            "id": message_item_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                ),
                _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                ),
            ]

        async def _emit_text_delta(delta: str) -> AsyncIterator[str]:
            """Yield the message item-added event (lazily) + a text delta.

            Under forced-choice mode (Yuki F6 codex r1 BLOCKING #2), the
            delta is BUFFERED in ``deferred_text`` instead of being
            yielded — final flush decision happens after the engine
            stream completes and we know whether synthesis is needed.
            """
            nonlocal accumulated_text
            if not delta:
                return
            if _forced_choice_active:
                deferred_text.append(delta)
                return
            if not message_open:
                for ev in await _open_message_item():
                    yield ev
            accumulated_text += delta
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta,
                },
            )

        async def _flush_deferred_text_if_no_synthesis(
            will_synthesise: bool,
        ) -> AsyncIterator[str]:
            """Forced-choice deferred-text resolution (codex r1 BLOCKING #2).

            Called after the model finished and we know whether
            ``_enforce_responses_tool_choice`` will synthesise. If
            synthesis WILL fire, the deferred text is dropped (and the
            message item never opens — no spurious assistant content
            ships before the tool_call). If synthesis won't fire (the
            model returned a real tool_call, or no forced choice was
            set), the buffered deltas are emitted as a single
            ``output_text.delta`` so the client still sees the
            assistant's actual text content.
            """
            nonlocal accumulated_text
            if will_synthesise or not deferred_text:
                deferred_text.clear()
                return
            joined = "".join(deferred_text)
            deferred_text.clear()
            if not joined:
                return
            if not message_open:
                for ev in await _open_message_item():
                    yield ev
            accumulated_text += joined
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": joined,
                },
            )

        async for output in engine.stream_chat(messages=messages, **chat_kwargs):
            delta_text = output.new_text
            # Accumulate the RAW model output (pre-filter, pre-router) so the
            # post-loop tool_call parser can see `<tool_call>...</tool_call>`
            # XML that tool_filter rightly suppresses from the user-facing
            # text channel. Without this, `accumulated_text` is empty in the
            # tool-calling case and no `response.function_call` SSE event
            # gets emitted — Codex sees turn.completed with zero output
            # items and the agent loop silently ends. The chat-completions
            # route avoids this by parsing `output.text` (the full
            # non-streamed text) directly; the streaming path needs an
            # explicit raw accumulator.
            if delta_text:
                accumulated_raw += delta_text

            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            if hasattr(output, "cached_tokens") and output.cached_tokens:
                cached_tokens = output.cached_tokens

            engine_tool_calls = getattr(output, "tool_calls", None) or []
            if engine_tool_calls:
                accumulated_structured_tool_calls.extend(engine_tool_calls)
                continue

            if not delta_text:
                continue

            # Channel-routed engines (harmony / gemma4) — honor the
            # channel directly. ``reasoning`` channel drops here
            # because Responses-API streams don't have a reasoning
            # delta event Codex parses (Codex maps it from a separate
            # ``response.reasoning_text.delta`` we omit in v1).
            output_channel = getattr(output, "channel", None)
            if output_channel is not None:
                if output_channel in ("content", "tool_call"):
                    content = strip_special_tokens(delta_text)
                    if content:
                        filtered = tool_filter.process(content)
                        if filtered:
                            async for ev in _emit_text_delta(filtered):
                                yield ev
                elif output_channel == "reasoning":
                    # Reasoning-cap reclassification: once the per-request
                    # cap fires, route the overflow portion of this and
                    # every subsequent reasoning chunk to ``content`` so
                    # the user actually sees a reply instead of an
                    # unending silent reasoning stream. Without the cap
                    # the chunk drops as before (v1 Responses contract).
                    _, overflow, _ = _account_for_reasoning(delta_text)
                    if overflow:
                        content = strip_special_tokens(overflow)
                        if content:
                            filtered = tool_filter.process(content)
                            if filtered:
                                async for ev in _emit_text_delta(filtered):
                                    yield ev
                # ``reasoning`` and unknown channels are dropped for v1.
                continue

            if reasoning_parser:
                # ``accumulated_raw`` already had the ORIGINAL
                # ``delta_text`` appended above. Pass current/previous
                # to the parser's streaming extractor. Note: ``previous_raw``
                # here is computed from the buffer minus the ORIGINAL
                # delta (round-9 fix — keep the shared buffer clean of
                # synthetic markers).
                previous_raw = (
                    accumulated_raw[: -len(delta_text)]
                    if delta_text
                    else accumulated_raw
                )
                # Text-parser path: once the cap fires, splice ``</think>``
                # in front of the next chunk so the parser flips to
                # content. Idempotent — only fires once per request.
                #
                # Codex round-9 BLOCKING #2: the earlier
                # ``accumulated_raw = previous_raw + delta_text`` (where
                # ``delta_text`` had been mutated to start with
                # ``</think>``) wrote the forged marker INTO the shared
                # buffer. The terminal injection path then re-parsed
                # that mutated buffer via ``finalize_streaming``,
                # potentially mis-classifying the synthetic bytes.
                # Fix: keep ``accumulated_raw`` to real model output
                # only (the original ``delta_text`` was already
                # appended above), and build a LOCAL ``parser_current``
                # for the parser call that includes the synthetic
                # marker. The parser sees ``previous + "</think>" +
                # original``; the shared buffer holds ``previous +
                # original``.
                # Codex round-10 BLOCKING #2: only flip the close-
                # injected latch AFTER the parser call succeeds. The
                # earlier draft flipped before the call, so a parser
                # exception on the injection-carrying chunk left the
                # latch set and the next chunk would skip injection —
                # leaving the parser permanently mid-think.
                injected_this_chunk = False
                if _reasoning_cap_hit and not _reasoning_close_injected:
                    parser_delta_text = "</think>" + delta_text
                    parser_current = previous_raw + parser_delta_text
                    injected_this_chunk = True
                else:
                    parser_delta_text = delta_text
                    parser_current = accumulated_raw
                delta_msg = reasoning_parser.extract_reasoning_streaming(
                    previous_raw, parser_current, parser_delta_text
                )
                if injected_this_chunk:
                    # Parser call succeeded with the synthetic marker
                    # — latch so subsequent chunks don't re-inject.
                    _reasoning_close_injected = True
                if delta_msg is None:
                    continue
                if delta_msg.reasoning:
                    # Account for reasoning bytes against the per-request
                    # cap. Overflow is whatever crossed the budget mid-
                    # chunk; it must NOT be promoted to content until the
                    # parser has formally transitioned out of thinking,
                    # otherwise (codex round-7 BLOCKING #2) clients see
                    # ``response.output_text.delta`` while the parser
                    # state is still logically inside reasoning. Force
                    # the parser flip in THIS same chunk by re-running
                    # the streaming extractor with a synthetic
                    # ``</think>`` delta against a locally-built
                    # ``current`` (don't mutate ``accumulated_raw`` —
                    # the round-6 local-buffer invariant applies here
                    # too).
                    kept_reasoning, overflow, _ = _account_for_reasoning(
                        delta_msg.reasoning
                    )
                    flip_succeeded = _reasoning_close_injected
                    if overflow and not _reasoning_close_injected:
                        # Codex round-10 BLOCKING #2: flip the latch
                        # AFTER success only — if the parser raises,
                        # next chunk retries the forced transition.
                        # Codex round-13 BLOCKING #2: position the
                        # synthetic ``</think>`` AT THE CAP BOUNDARY
                        # (not after the full over-budget chunk).
                        # ``previous_raw`` is the buffer before THIS
                        # delta arrived; ``previous_raw +
                        # kept_reasoning`` represents the model output
                        # up to the cap firing point. Without the
                        # boundary positioning, stateful parsers
                        # would see ``</think>`` AFTER the over-budget
                        # bytes and potentially mis-classify them.
                        flip_previous = previous_raw + kept_reasoning
                        flip_delta = "</think>"
                        flip_current = flip_previous + flip_delta
                        try:
                            flip_msg = reasoning_parser.extract_reasoning_streaming(
                                flip_previous, flip_current, flip_delta
                            )
                            _reasoning_close_injected = True
                            flip_succeeded = True
                        except Exception as e:
                            # Codex round-8 BLOCKING #2: when the flip
                            # raises, the parser may still be mid-think.
                            # Emitting ``overflow`` here would leak
                            # reasoning bytes onto the wire as
                            # ``response.output_text.delta`` even though
                            # the parser hasn't transitioned. Suppress
                            # overflow on flip failure; log so operators
                            # can see the parser bug. Worst case the
                            # client sees a slightly-truncated response,
                            # strictly preferable to mixing reasoning
                            # into content under a failed transition.
                            logger.warning(
                                "responses in-chunk close-marker flip raised "
                                "on %r: %s — parser state may stay mid-think; "
                                "suppressing %d-byte overflow on this chunk "
                                "to avoid leaking reasoning bytes as content",
                                type(reasoning_parser).__name__,
                                e,
                                len(overflow),
                            )
                            flip_msg = None
                        # Whatever content the flip released stays
                        # ahead of the overflow bytes on the wire
                        # (parser-derived content first, cap-overflow
                        # bytes second).
                        flip_content = (
                            getattr(flip_msg, "content", None)
                            if flip_msg is not None
                            else None
                        )
                        if isinstance(flip_content, str) and flip_content:
                            delta_msg.content = (delta_msg.content or "") + flip_content
                    if overflow and flip_succeeded:
                        # Safe to promote overflow: either the flip
                        # this iteration succeeded, OR the parser
                        # already transitioned on a PRIOR chunk
                        # (``_reasoning_close_injected`` was already
                        # True on entry, captured in ``flip_succeeded``
                        # via the initial assignment above).
                        delta_msg.content = (delta_msg.content or "") + overflow
                if delta_msg.content:
                    content = strip_special_tokens(delta_msg.content)
                    if content:
                        filtered = tool_filter.process(content)
                        if filtered:
                            async for ev in _emit_text_delta(filtered):
                                yield ev
                # delta_msg.reasoning intentionally dropped — see above.
                continue

            # Default path: text-only stream with think_router stripping
            # ``<think>...</think>`` from the text channel.
            content = strip_special_tokens(delta_text)
            if not content:
                continue
            filtered = tool_filter.process(content)
            if not filtered:
                continue
            pieces = think_router.process(filtered)
            for block_type, piece in pieces:
                if block_type == "text" and piece:
                    async for ev in _emit_text_delta(piece):
                        yield ev
                # block_type == "thinking" intentionally dropped.

        # Flush filters
        remaining = tool_filter.flush()
        if remaining:
            if reasoning_parser:
                async for ev in _emit_text_delta(remaining):
                    yield ev
            else:
                for block_type, piece in think_router.process(remaining):
                    if block_type == "text" and piece:
                        async for ev in _emit_text_delta(piece):
                            yield ev

        if not reasoning_parser:
            for block_type, piece in think_router.flush():
                if block_type == "text" and piece:
                    async for ev in _emit_text_delta(piece):
                        yield ev

        # Codex round-3 BLOCKING #3: if the reasoning cap latched on the
        # last engine chunk of the stream (terminal exact-boundary case
        # OR the model stopped immediately after overflow), the
        # ``</think>`` close marker was never spliced into the parser —
        # so any held content past the cap stays buffered and the
        # client sees a silent reasoning-only response with no
        # ``output_text.delta`` ever emitted. Force the injection here
        # so a terminal cap-hit flips the parser to content and any
        # trailing bytes are promoted to ``response.output_text.delta``.
        # Idempotent via ``_reasoning_close_injected``.
        terminal_injection_attempted = False
        if (
            reasoning_parser is not None
            and _reasoning_cap_hit
            and not _reasoning_close_injected
        ):
            _reasoning_close_injected = True
            terminal_injection_attempted = True
            # Codex round-6 BLOCKING #2: build the parser's
            # ``current`` argument LOCALLY rather than mutating the
            # shared ``accumulated_raw``. If the injection produces no
            # content (no held bytes / parser early-returns), the
            # subsequent ``finalize_streaming(accumulated_raw)`` would
            # otherwise re-parse a buffer that ends with the synthetic
            # ``</think>`` marker and could mis-classify the forged
            # bytes as model output. Symmetric with the postprocessor
            # fix in service/postprocessor.py.
            previous_raw = accumulated_raw
            injected_delta = "</think>"
            local_current = previous_raw + injected_delta
            try:
                final_inject = reasoning_parser.extract_reasoning_streaming(
                    previous_raw, local_current, injected_delta
                )
            except Exception as e:
                # Codex round-5 BLOCKING #3: an earlier draft emitted a
                # diagnostic string ``"[reasoning cap hit — parser
                # flush failed]"`` as ``response.output_text.delta``,
                # which fabricates assistant content from an INTERNAL
                # server failure — clients see an "answer" that the
                # model never produced. Log the parser failure and
                # leave the assistant content empty. The route's
                # existing 5xx / disconnect-guard semantics handle
                # truly catastrophic failures upstream; a single
                # reasoning-cap parser bug must not invent text.
                logger.warning(
                    "responses terminal close-marker injection raised on %r: %s — "
                    "trailing reasoning content (if any) will not be "
                    "promoted to output_text.delta for this request",
                    type(reasoning_parser).__name__,
                    e,
                )
                final_inject = None
            if final_inject is not None and getattr(final_inject, "content", None):
                content = strip_special_tokens(final_inject.content)
                if content:
                    filtered = tool_filter.process(content)
                    if filtered:
                        async for ev in _emit_text_delta(filtered):
                            yield ev

        # Codex round-4 BLOCKING #1 + round-6 BLOCKING #2: when the
        # terminal injection above ran at all (whether or not it
        # produced content), skip the parser's non-stream finalize
        # pass. Two distinct hazards:
        #
        #   1. Injection emitted content — running ``finalize_streaming``
        #      next would re-emit the SAME bytes the streaming
        #      extraction just released (qwen3 / deepseek parsers'
        #      ``finalize_streaming`` re-parses the whole accumulated
        #      buffer and can't distinguish already-streamed from
        #      still-held content).
        #   2. Injection produced no content — the parser already had
        #      its chance to flush via the forced ``</think>``. Running
        #      the non-stream finalize on the original
        #      ``accumulated_raw`` (which excludes ``</think>`` per
        #      the round-5/6 local-buffer fix) might still re-classify
        #      the cap-truncated reasoning as content via the
        #      non-stream parser's broader heuristics, double-emitting
        #      bytes already routed past the cap.
        #
        # When NO terminal injection was attempted (cap never fired,
        # or it fired and was already injected mid-stream), the
        # finalize pass still runs as the safety net for normal
        # parser-held content.
        if reasoning_parser and accumulated_raw and not terminal_injection_attempted:
            final_msg = (
                reasoning_parser.finalize_streaming(accumulated_raw)
                if hasattr(reasoning_parser, "finalize_streaming")
                else None
            )
            if final_msg and final_msg.content:
                content = strip_special_tokens(final_msg.content)
                if content:
                    async for ev in _emit_text_delta(content):
                        yield ev

        # Parse tool_calls FIRST so the forced-choice deferred-text
        # resolution (Yuki F6 codex r1 BLOCKING #2) can decide whether
        # the message item should open at all.
        # Pass `accumulated_raw` (pre-filter model output) not
        # `accumulated_text` (post-filter user-visible text) — tool_filter
        # rightly suppresses `<tool_call>...</tool_call>` XML from
        # `accumulated_text`, but the post-loop parser needs that XML
        # to extract structured tool_calls. Without this swap, the
        # text-parser path returned zero tool_calls and Codex's agent
        # loop silently terminated with no items emitted.
        _, parsed_tool_calls = _parse_tool_calls_with_parser(
            accumulated_raw,
            openai_request,
            structured_tool_calls=accumulated_structured_tool_calls or None,
        )

        # Yuki F6 (0.8.5 dogfood): mirror the non-stream synthesis so
        # ``tool_choice="required"`` / named-function always produce a
        # ``response.output_item.added`` event of type ``function_call``
        # (or ``computer_call`` for Computer-Use), honouring the
        # OpenAI ``tool_call guaranteed`` contract on the streaming
        # surface too. Codex r2 BLOCKING (PR #817): the non-stream
        # path raises 422 for multi-tool ``required`` with no model
        # call, but we cannot raise mid-stream after SSE headers
        # are out — emit a ``response.failed`` event with the same
        # error envelope instead so the client sees a clean shutdown
        # signal.
        try:
            tool_calls = _enforce_responses_tool_choice(
                parsed_tool_calls, responses_request, openai_request
            )
        except HTTPException as forced_choice_err:
            # Drop any deferred buffered text — the request failed
            # under forced choice, the deferred prose has no
            # legitimate destination on the wire.
            deferred_text.clear()
            err_detail = forced_choice_err.detail
            if isinstance(err_detail, dict):
                err_envelope = err_detail.get("error", {})
                err_code = err_envelope.get("code", "tool_choice_unfulfilled")
                err_msg = err_envelope.get(
                    "message", "tool_choice could not be fulfilled"
                )
            else:
                err_code = "tool_choice_unfulfilled"
                err_msg = str(err_detail)
            yield _sse(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "status": "failed",
                        "error": {
                            "code": err_code,
                            "message": err_msg,
                        },
                    },
                },
            )
            # ``response.failed`` IS the terminal event for the
            # OpenAI Responses SSE spec — there is no ``data: [DONE]``
            # sentinel on this surface (see module docstring + the
            # ``_sse`` helper docstring). Codex r2 reviewer flagged a
            # missing ``[DONE]`` but that's chat-completions-only;
            # Responses-API clients (Codex CLI, openai-python) detect
            # stream end via the terminal event type, not a sentinel
            # data line.
            return
        # Codex r1 BLOCKING #2 (PR #817): under forced choice the
        # ``deferred_text`` buffer holds text deltas we held back. Flush
        # them ONLY if synthesis won't fire — i.e. the model produced
        # a real tool_call, so the assistant's prose is legitimate
        # context. When synthesis WILL fire (model only emitted text),
        # drop the deferred text so the client doesn't see both a
        # message AND a synthesised tool_call.
        _synthesis_fired = bool(tool_calls) and not parsed_tool_calls
        async for ev in _flush_deferred_text_if_no_synthesis(_synthesis_fired):
            yield ev

        # Close the message item if we ever opened it.
        if message_open:
            # Yuki F8 (0.8.5 dogfood): emit ``response.output_text.done``
            # AND ``response.content_part.done`` BEFORE the message item
            # ``done`` event so clients gating UI on those events
            # actually see them — pre-0.8.5 only the broader
            # ``response.output_item.done`` fired.
            if content_part_open:
                yield _sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "text": accumulated_text,
                    },
                )
                yield _sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": accumulated_text,
                            "annotations": [],
                        },
                    },
                )
                content_part_open = False
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": {
                        "type": "message",
                        "id": message_item_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": accumulated_text,
                                "annotations": [],
                            }
                        ],
                    },
                },
            )

        # Ana C-06 (0.8.5 dogfood): when the request used Computer-Use,
        # translate ``function.name == "computer"`` tool_calls into the
        # ``computer_call`` envelope so SDK consumers walking
        # ``output_item.type`` for ``computer_call`` find them.
        uses_computer_use = request_uses_computer_use(responses_request)

        tool_output_index = (message_output_index + 1) if message_open else 0
        for tc in tool_calls or []:
            if uses_computer_use and (tc.function.name or "") == "computer":
                async for ev in _emit_computer_call_item(tc, tool_output_index):
                    yield ev
            else:
                async for ev in _emit_function_call_item(tc, tool_output_index):
                    yield ev
            tool_output_index += 1

        # H-06 (codex r2): the streaming /v1/responses path is
        # unreachable for strict=true requests — the entry-point
        # gate above 400s them as ``strict_stream_unsupported``
        # because constrained decoding here is buffered-only. So no
        # post-decode validation is needed in the stream loop;
        # belt-and-braces validation runs in the non-stream path
        # where the buffered output is available.

        # response.completed — terminal event. Codex treats a missing
        # one as a hard failure (it logs "stream closed before
        # response.completed").
        cached_tokens_clamped = min(cached_tokens, prompt_tokens)
        usage_payload = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cached_tokens_clamped:
            usage_payload["input_tokens_details"] = {
                "cached_tokens": cached_tokens_clamped
            }
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "completed",
                    "model": served_model,
                    "usage": usage_payload,
                },
            },
        )

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Responses (stream): prompt={prompt_tokens} + "
            f"completion={completion_tokens} tokens in {elapsed:.2f}s "
            f"({tokens_per_sec:.1f} tok/s)"
        )

    except Exception as e:  # noqa: BLE001
        # response.failed gives Codex a clean shutdown signal instead of
        # a half-stream-then-EOF; matches how the OpenAI cloud
        # Responses API closes errored streams.
        logger.exception("Responses stream failed: %s", e)
        yield _sse(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "status": "failed",
                    "error": {
                        "code": "internal_error",
                        "message": str(e),
                    },
                },
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_request(req: ResponsesRequest) -> None:
    """One-line request log mirroring the other route surfaces."""
    if isinstance(req.input, str):
        n_items = 1
        total_chars = len(req.input)
    else:
        n_items = len(req.input)
        total_chars = 0
        for item in req.input:
            if isinstance(item.content, str):
                total_chars += len(item.content)
            elif item.content:
                for c in item.content:
                    if c.text:
                        total_chars += len(c.text)
            if item.arguments:
                total_chars += len(item.arguments)
    n_tools = len(req.tools) if req.tools else 0
    instr_chars = len(req.instructions) if req.instructions else 0
    logger.info(
        f"[REQUEST] POST /v1/responses (codex) stream={req.stream} "
        f"model={req.model!r} max_output_tokens={req.max_output_tokens} "
        f"input_items={n_items} total_chars={total_chars} "
        f"instructions_chars={instr_chars} tools={n_tools}"
    )
