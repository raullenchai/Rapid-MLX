# SPDX-License-Identifier: Apache-2.0
"""Anthropic Messages API endpoints — /v1/messages."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.anthropic_adapter import (
    AnthropicOutputConfigError,
    anthropic_to_openai,
    openai_to_anthropic,
    to_anthropic_tool_use_id,
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
    _TOOL_USE_REQUIRED_SUFFIX,
    SSE_RESPONSE_HEADERS,
    _apply_reasoning_cutoff_notice,
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
    _tool_use_required_named_suffix,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    count_prompt_tokens,
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


def _named_tool_choice_target(tool_choice) -> str | None:
    """Return the target tool name when ``tool_choice`` pins a specific
    tool, else ``None``.

    The Anthropic adapter has already translated
    ``{"type":"tool","name":X}`` into the OpenAI form
    ``{"type":"function","function":{"name":X}}`` on
    ``openai_request.tool_choice`` by the time we get here. ``"auto"`` /
    ``"required"`` / ``"none"`` / unset shapes return ``None`` — they
    have no defined "wrong tool" case to filter or enforce.
    """
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    target = (tool_choice.get("function") or {}).get("name")
    return target or None


def _tool_call_name_anthropic(tc) -> str | None:
    """Extract the function name from a tool_call entry regardless of
    shape. Three real shapes survive into this point — see
    ``routes/chat.py:_tool_call_name`` for the same catalogue. Inlined
    here to avoid a cross-route import dependency from ``routes/chat.py``
    into ``routes/anthropic.py``.
    """
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


def _filter_tool_calls_by_tool_choice(tool_calls, tool_choice) -> list:
    """Drop tool_use blocks that don't match a forced ``tool_choice``.

    H-05: Anthropic ``tool_choice={"type":"tool","name":X}`` pins WHICH
    tool the model must call. Local inference has no decoder-level
    constraint, so a small model can happily defy the pin and emit a
    call to a different tool (the Sergei repro hit qwen3.5-4b on a
    two-tool prompt where the model fired BOTH the pinned tool AND the
    un-pinned one). Pre-fix, the downstream JSON-schema validator
    (F-220) then 400-ed on the un-pinned tool's argument schema —
    leaking validation across a tool the user never asked for and
    silently breaking ``tool_choice``.

    Policy: when the choice pins a specific tool, KEEP only the calls
    to that tool and drop the rest with a warning. This matches the
    user-visible expectation ("you asked for X, here is X") and keeps
    the F-220 validator scoped to the pinned tool. ``"auto"`` /
    ``"required"`` / ``"none"`` / unset shapes pass through unchanged
    — only the explicit named-tool form has a defined "wrong tool"
    case to filter.

    Chat.py's named-function path 422s on the same mismatch (see
    ``routes/chat.py:1665``). The two routes intentionally diverge on
    the "got pinned + extras" case: ``/v1/chat/completions`` mirrors
    OpenAI's strict contract (422), while ``/v1/messages`` mirrors
    Anthropic's more forgiving "deliver the pinned tool's call"
    contract — a 422 on an extra call would surface a confusing error
    to clients that pinned the very tool the response already
    carries. They CONVERGE on the "got zero pinned calls" case, which
    is handled by ``_enforce_named_tool_choice_present`` below (PR
    #763 codex round-1 BLOCKING #1: a filter that emptied the list
    plus a downstream "no tool_calls? end_turn." branch would 200
    with no ``tool_use`` block at all, silently violating the
    forced-tool contract).

    The validator scope refactor in ``_validate_tool_call_params``
    ensures we never validate against the dropped tool's schema even
    if a future change weakens this filter.
    """
    if not tool_calls or not isinstance(tool_choice, dict):
        return tool_calls or []
    target = _named_tool_choice_target(tool_choice)
    if not target:
        return tool_calls

    filtered = []
    dropped: list[str] = []
    for tc in tool_calls:
        if _tool_call_name_anthropic(tc) == target:
            filtered.append(tc)
        else:
            dropped.append(_tool_call_name_anthropic(tc) or "<unknown>")
    if dropped:
        logger.warning(
            "tool_choice pinned %r but model also emitted calls to %s; "
            "dropping the un-pinned calls so the response carries only the "
            "pinned tool (Anthropic /v1/messages H-05 policy).",
            target,
            dropped,
        )
    return filtered


def _synthesize_pinned_tool_call(tool_name: str):
    """Build a synthetic ``ToolCall`` for the pinned tool with empty
    ``input``. F8 best-effort fallback when the model failed to comply
    with ``tool_choice={"type":"tool","name":X}``.

    Local import to avoid widening the routes/anthropic.py module-level
    import surface — these models are pulled in lazily only on the
    pinned-tool fallback path so the happy-path import time stays flat.
    """
    from ..api.models import FunctionCall, ToolCall

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=FunctionCall(name=tool_name, arguments="{}"),
    )


def _enforce_named_tool_choice_present(
    tool_calls,
    tool_choice,
    *,
    original_call_count: int,
) -> tuple[list, bool, str | None]:
    """Return ``(tool_calls, synthesized, error)``.

    The first element is unchanged when the named-tool contract is
    satisfied. When the model emits text only or only wrong-tool calls,
    the function returns the filtered empty list plus an error string so
    callers can surface 422 instead of silently returning text for a
    pinned tool request.

    ``original_call_count`` is the size of ``tool_calls`` BEFORE the
    filter ran. A warning logs the disambiguation between "model
    returned text only" (count == 0) and "model called the wrong
    tool(s) and the filter emptied the list" (count > 0) so an
    operator debugging contract failures can see WHICH case fired.
    """
    target = _named_tool_choice_target(tool_choice)
    if not target or tool_calls:
        return tool_calls, False, None
    # Log the disambiguation an operator needs to debug small-model
    # compliance issues. The wire response shape is identical either way.
    if original_call_count == 0:
        logger.warning(
            "tool_choice pinned tool %r but the model returned a text "
            "response with no tool_calls; returning a tool_choice contract "
            "error.",
            target,
        )
    else:
        logger.warning(
            "tool_choice pinned tool %r but the model emitted %d call(s), "
            "none to %r; returning a tool_choice contract error.",
            target,
            original_call_count,
            target,
        )
    return (
        [],
        False,
        f"tool_choice pinned tool {target!r} but the model did not emit that tool",
    )


def _extract_sse_error_message(body: str) -> str | None:
    for event in body.split("\n\n"):
        if "event: error" not in event:
            continue
        data = "\n".join(
            line.removeprefix("data: ").strip()
            for line in event.splitlines()
            if line.startswith("data: ")
        )
        if not data:
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str):
            return message
    return None


async def _buffer_named_tool_choice_stream(stream: AsyncIterator[str]) -> Response:
    chunks: list[str] = []
    async for chunk in stream:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode())
        else:
            chunks.append(chunk)
    body = "".join(chunks)
    error_message = _extract_sse_error_message(body)
    if error_message and "tool_choice" in error_message:
        raise HTTPException(status_code=422, detail=error_message)
    return Response(
        content=body,
        media_type="text/event-stream",
        headers={**SSE_RESPONSE_HEADERS, "Connection": "keep-alive"},
    )


def _is_required_tool_choice(tool_choice) -> bool:
    """Return True when ``tool_choice`` forces the model to call ANY tool.

    The Anthropic adapter maps ``{"type":"any"}`` → OpenAI ``"required"``
    in ``api/anthropic_adapter._convert_tool_choice``. By the time we get
    here ``tool_choice`` is the post-adapter OpenAI shape, so the
    ``"required"`` string IS the Anthropic ``any`` contract.

    D-ANTHRO-TOOL-USAGE F3: the route was previously a no-op on this
    branch — ``_named_tool_choice_target`` returned ``None`` for
    ``"required"`` and the post-parse enforcement only fired for the
    named-function form. A request with ``tool_choice={"type":"any"}``
    therefore sailed through with the model's text reply and
    ``stop_reason="end_turn"`` — a direct spec violation. Mirror the
    OpenAI-side enforcement (``routes/chat.py`` lines 973-978 +
    1878-1902) on this surface.
    """
    return tool_choice == "required"


def _synthesize_anthropic_forced_tool_call(name: str):
    """Build a single ``ToolCall`` for a forced ``tool_choice`` whose
    parser surfaced no calls — Anthropic-route mirror of chat.py's
    ``_synthesize_forced_tool_call``.

    Inlined here (instead of importing the chat.py helper) to keep the
    two routes' import surfaces independent — the Anthropic route
    deliberately avoids depending on ``routes/chat`` so a future split
    can move them into separate modules. The behaviour is identical:
    OpenAI ``tool_choice`` is parser-agnostic and forced calls MUST
    surface a ``tool_use`` block, so when the text parser found nothing
    we synthesise an empty-argument call to the unambiguous target.
    """
    from ..api.models import FunctionCall, ToolCall

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=FunctionCall(name=name, arguments="{}"),
    )


def _inject_tool_use_required_suffix(
    messages: list,
    tool_choice,
    *,
    tools: list | None,
) -> list:
    """Mutate-in-place: append ``_TOOL_USE_REQUIRED_SUFFIX`` (or the
    named variant) to the system message so a forced ``tool_choice``
    has the same prompt-level lever the OpenAI route applies.

    Returns the (possibly prepended) ``messages`` list. When no system
    message exists, a new one is prepended carrying just the suffix.
    Mirrors ``routes/chat.py`` lines 990-1009 byte-for-byte so the two
    surfaces inject the same lever for the same tool_choice value.

    No-op when ``tool_choice`` does not force a call, OR when ``tools``
    is empty — there is nothing for the model to call, so the suffix
    would produce only a confusing "you must call a tool" stanza with
    no tools defined.
    """
    if not tools:
        return messages
    suffix = None
    if _is_required_tool_choice(tool_choice):
        suffix = _TOOL_USE_REQUIRED_SUFFIX
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        named = (tool_choice.get("function") or {}).get("name")
        if named:
            suffix = _tool_use_required_named_suffix(named)
    if not suffix:
        return messages

    has_system = any(
        (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "system"
        for m in messages
    )
    if has_system:
        for i, m in enumerate(messages):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "system":
                content = (
                    m.get("content")
                    if isinstance(m, dict)
                    else getattr(m, "content", "")
                )
                # Codex r3 NIT (PR #807): when system content is a
                # list-of-blocks (multimodal / Anthropic text-block
                # array), preserve the block shape by APPENDING a new
                # text block carrying the suffix — never stringify the
                # list via ``str(...)`` (which would emit Python repr
                # like ``"[{'type': 'text', ...}]<suffix>"`` into the
                # rendered chat-template prompt). String content is
                # the common path and stays a fast concat.
                if isinstance(content, str):
                    new_content = content + suffix
                elif isinstance(content, list):
                    new_content = list(content)
                    new_content.append({"type": "text", "text": suffix})
                elif content is None:
                    new_content = suffix
                else:
                    # Unknown shape — leave the block untouched and
                    # fall through to the "prepend a system" path so
                    # the suffix still reaches the model. Better an
                    # extra system block than a silently-dropped
                    # forced-tool lever.
                    continue
                if isinstance(m, dict):
                    messages[i] = {**m, "content": new_content}
                else:
                    m.content = new_content
                break
        else:
            # No appendable system block found (every system block had
            # a non-string / non-list content shape) — fall through to
            # the prepend path so the suffix still ships.
            messages.insert(0, {"role": "system", "content": suffix.strip()})
    else:
        messages.insert(0, {"role": "system", "content": suffix.strip()})
    return messages


def _enforce_required_tool_choice_present(
    tool_calls,
    tool_choice,
    *,
    tools: list | None,
):
    """OpenAI ``tool_choice="required"`` / Anthropic ``{"type":"any"}``
    post-parse enforcement.

    Mirrors ``routes/chat.py`` lines 1878-1902. When a forced-any
    ``tool_choice`` produced no tool_calls:

      * single-tool case → synthesise a call to that tool so the
        ``tool_use`` block contract holds (the choice is unambiguous).
      * multi-tool case → return ``("error", detail)`` so the caller
        can 422 on the non-stream branch or surface an SSE error on
        the stream branch.

    Returns ``(tool_calls, error_detail_or_None)``. The caller is
    responsible for raising / emitting the SSE error event.
    """
    if not _is_required_tool_choice(tool_choice):
        return tool_calls, None
    if tool_calls:
        return tool_calls, None
    if tools and len(tools) == 1:
        # Defensive: tools entries may be Pydantic ``Tool`` models or
        # plain dicts depending on whether the request flowed through
        # the adapter as a model instance. Prefer the model attribute
        # path so we honour any future Tool-shape changes without
        # touching this branch.
        tool = tools[0]
        fn = (
            tool.function
            if hasattr(tool, "function")
            else (tool.get("function") if isinstance(tool, dict) else None)
        )
        if fn is None:
            fn = {}
        solo_name = (
            fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        )
        if solo_name:
            logger.warning(
                "tool_choice={'type':'any'} on Anthropic route produced no "
                "tool_calls; synthesising a call to the sole available tool "
                "%r with empty arguments to honour the forced-tool contract "
                "(D-ANTHRO-TOOL-USAGE F3).",
                solo_name,
            )
            return [_synthesize_anthropic_forced_tool_call(solo_name)], None
    detail = (
        'tool_choice={"type":"any"} but the model returned a text response '
        "with no tool_calls. Local inference has no decoder-level "
        "constraint; the system-prompt enforcement was insufficient for "
        "this prompt. Retry with a more concrete user message or use "
        'tool_choice={"type":"tool","name":...} to pin a specific tool.'
    )
    return tool_calls, detail


def _estimate_anthropic_prompt_tokens(engine, messages, tools) -> int:
    """Return the prompt-token count under the engine's chat template, or 0.

    D-ANTHRO-TOOL-USAGE F5: the Anthropic streaming surface previously
    hard-coded ``message_start.usage.input_tokens=0`` because the engine
    hadn't run yet — but billing dashboards parsing the SSE stream
    therefore under-reported the input share by 100%. Anthropic's
    public stream emits ``input_tokens`` in ``message_start`` (the
    server-side estimate, finalised in ``message_delta``).

    Uses the SAME ``count_prompt_tokens`` helper the DoS gate
    (``enforce_context_length_for_messages``) already calls so the
    estimate is byte-for-byte consistent with what the context-length
    pre-check used. The render goes through ``engine.build_prompt`` so
    chat-template / tools rendering matches what the engine itself
    will tokenise during generation.

    Returns ``0`` when:

      * the engine has no ``build_prompt`` (MLLM / mock test stub) —
        the streaming usage path falls back to the engine-reported
        ``output.prompt_tokens`` like before, and the public-API
        contract just degrades to the pre-fix ``input_tokens=0`` shape
        rather than fabricating a tokenizer-free count.
      * the template render or tokenizer call raises — same fallback
        rationale; surfacing a tokenizer error here would obscure the
        actual generation error from the client.
    """
    build_prompt = getattr(engine, "build_prompt", None)
    if build_prompt is None or getattr(engine, "is_mllm", False):
        return 0
    try:
        prompt = build_prompt(messages, tools=tools)
    except Exception:
        # Codex r6 NIT (PR #807): log at debug so a genuinely broken
        # chat template / malformed tools schema is visible in the
        # server log even though we return ``0`` to let the engine's
        # own validation surface a clean 400 downstream. Without the
        # log, a debug session can't distinguish "no estimate
        # available" from "the route silently swallowed a real
        # build_prompt error".
        logger.debug(
            "_estimate_anthropic_prompt_tokens: build_prompt raised; "
            "returning 0 so the streaming usage path falls back to the "
            "engine-reported prompt_tokens",
            exc_info=True,
        )
        return 0
    if not prompt:
        return 0
    try:
        return count_prompt_tokens(engine, prompt)
    except Exception:
        logger.debug(
            "_estimate_anthropic_prompt_tokens: count_prompt_tokens raised; "
            "returning 0 so the streaming usage path falls back to the "
            "engine-reported prompt_tokens",
            exc_info=True,
        )
        return 0


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
    # parameter). The raw :class:`pydantic.ValidationError` it can raise
    # is now caught by the global ``_pydantic_validation_handler`` in
    # ``middleware.exception_handlers`` (H-17), which routes it through
    # the same sanitized 400 envelope used by ``/v1/chat/completions``.
    # Letting the exception bubble keeps the model class name, the
    # pinned pydantic version (``errors.pydantic.dev/2.13/...``), and
    # the attacker-supplied ``input_value`` out of the response body.
    anthropic_request = AnthropicRequest(**body)

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

        # Reject image/document content blocks when the loaded model has
        # no multimodal head (M-16). The Anthropic adapter
        # (``anthropic_adapter._convert_message_to_openai``) only carries
        # ``text``/``tool_use``/``tool_result`` blocks forward — every other
        # block type (``image``, ``document``) is silently dropped. Without
        # this guard the caller sees HTTP 200 with a hallucinated answer
        # about the missing media (mirrors the OpenAI-route R9P1 fix in
        # ``routes/chat.py``: text-only models never silently drop media).
        cfg_pre = get_config()
        # ``getattr`` default-True: tests with minimal engine stubs that
        # never carry image blocks shouldn't trigger this guard, and the
        # production engine always defines ``is_mllm``. Only a real
        # text-only model (``is_mllm == False``) trips the rejection.
        if getattr(engine, "is_mllm", True) is False:
            for _msg in anthropic_request.messages:
                _content = _msg.content
                if not isinstance(_content, list):
                    continue
                for _block in _content:
                    _block_type = (
                        _block.type
                        if hasattr(_block, "type")
                        else (
                            _block.get("type", "") if isinstance(_block, dict) else ""
                        )
                    )
                    if _block_type in ("image", "document"):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Model '{cfg_pre.model_name}' does not support "
                                "image or document inputs."
                            ),
                        )

        # Convert Anthropic request -> OpenAI request. The adapter raises
        # ``AnthropicOutputConfigError`` (a ``ValueError`` subclass) on
        # malformed ``output_config`` payloads — backport of upstream vLLM
        # PR #42396; map directly to HTTP 400 with the adapter's message.
        # F-034: ``anthropic_to_openai`` constructs a ``ChatCompletionRequest``
        # which now rejects unsatisfiable combinations (e.g.
        # ``tool_choice="required"`` — Anthropic ``any`` — with no
        # ``tools``). The resulting :class:`pydantic.ValidationError` is
        # caught by the global ``_pydantic_validation_handler`` (H-17)
        # so we no longer wrap it in ``str(e)`` here — that wrapping was
        # the source of the H-17 leak (model class name + pydantic
        # version + attacker ``input_value`` echo).
        try:
            openai_request = anthropic_to_openai(anthropic_request)
        except AnthropicOutputConfigError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # D-ANTHRO-TOOL-USAGE F3 (codex r3 BLOCKING #1+#2): suffix
        # injection MUST happen BEFORE the context-length DoS gate and
        # BEFORE the prompt-token count is captured. Pre-r3 the suffix
        # was injected per-branch AFTER the gate, so:
        #   (a) a forced-tool request could bypass the context-length
        #       cap by piggy-backing the suffix onto an already-at-cap
        #       prompt (DoS gate measured pre-injection tokens), and
        #   (b) ``message_start.usage.input_tokens`` (streaming) under-
        #       reported the prompt by the suffix's byte cost.
        # Inject ONCE here on the rendered engine-shape messages, then
        # run the gate + capture the count on the post-injection state.
        # Both branches reuse the same ``messages`` / ``images`` /
        # ``videos`` afterwards so there is no second injection or
        # second extract downstream — codex r5 BLOCKING #1 (multimodal
        # threading) + #2 (let extract errors propagate so a malformed
        # request body 400s here instead of crossing the streaming
        # SSE boundary mid-response).
        messages, images, videos = extract_multimodal_content(
            openai_request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )
        # Dogfood C-05 / F-R2-04 / r5-B C-11 lane parity: auto-prepend the
        # canonical UI-TARS Computer-Use sysprompt on the Anthropic lane
        # too so the three surfaces produce the SAME prompt for the SAME
        # model. r5-B threads ``tools=openai_request.tools`` so the
        # injection is tool-coupled (the same gate firing on
        # ``/v1/chat/completions`` and ``/v1/responses``): NO Computer-Use
        # tool declared → no action-API contract injected → model emits
        # plain prose, matching the other two lanes. Single shared
        # helper, single canonical sysprompt, single coordinate space.
        from ..tool_parsers.ui_tars_tool_parser import (
            maybe_inject_ui_tars_system_prompt as _maybe_inject_ui_tars_sysprompt,
        )

        _cfg_for_ui_tars = get_config()
        messages = _maybe_inject_ui_tars_sysprompt(
            messages,
            tool_call_parser=_cfg_for_ui_tars.tool_call_parser,
            tool_choice=openai_request.tool_choice,
            tools=openai_request.tools,
        )

        _inject_tool_use_required_suffix(
            messages,
            openai_request.tool_choice,
            tools=openai_request.tools,
        )

        # Context-length pre-check — same DoS gate the chat/completions/
        # responses routes enforce. Render the prompt through the engine's
        # chat template, count tokens, raise 400 ``context_length_exceeded``
        # if over the model cap. Runs BEFORE the stream/non-stream branch
        # so streaming clients can't bypass the gate by setting
        # ``stream: true``. See service/helpers.py for rationale.
        #
        # D-ANTHRO-TOOL-USAGE F5 (codex r2 NIT): capture the
        # gate-computed prompt-token count so the streaming branch's
        # ``message_start.usage.input_tokens`` can reuse it instead of
        # re-rendering + re-tokenising the same messages. The helper
        # returns ``None`` on permissive-skip paths (MLLM engines, no
        # build_prompt, empty prompt) — codex r4 NIT — which means
        # "no estimate available"; the streaming branch then falls
        # back to its own estimator helper. We forward the value
        # verbatim (including ``None``) so the streaming path can
        # distinguish "skip" from "real zero count".
        _ctx_prompt_tokens = enforce_context_length_for_messages(
            engine,
            messages,
            tools=openai_request.tools,
            max_tokens=_resolve_max_tokens(
                openai_request.max_tokens,
                _resolve_enable_thinking(openai_request),
            ),
        )

        if anthropic_request.stream:
            _admission_committed = True
            # C-01 force-abort: holder list the engine populates with
            # the admitted scheduler request id; the disconnect_guard
            # reads it and force-calls scheduler.abort_request on
            # client disconnect.
            _anth_rid_holder: list[str | None] = [None]
            stream = _disconnect_guard(
                _stream_anthropic_messages(
                    engine,
                    openai_request,
                    anthropic_request,
                    request_id_holder=_anth_rid_holder,
                    prompt_tokens_estimate=_ctx_prompt_tokens,
                    prepared_messages=messages,
                    prepared_images=images,
                    prepared_videos=videos,
                ),
                request,
                engine=engine,
                request_id_holder=_anth_rid_holder,
            )
            if _named_tool_choice_target(openai_request.tool_choice):
                return await _buffer_named_tool_choice_stream(stream)
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                # ``SSE_RESPONSE_HEADERS`` (Cache-Control no-cache/no-transform +
                # X-Accel-Buffering: no) keeps anti-buffering parity with the
                # other SSE routes; ``Connection: keep-alive`` is preserved for
                # the Anthropic SDK clients that historically checked for it.
                headers={**SSE_RESPONSE_HEADERS, "Connection": "keep-alive"},
            )

        # Non-streaming: run inference through existing engine. The
        # ``extract_multimodal_content`` + ``_inject_tool_use_required_suffix``
        # pair already ran above so ``messages`` already carries the
        # forced-tool suffix and ``images`` / ``videos`` carry the
        # multimodal payload; no second extract/inject is needed
        # (codex r3 BLOCKING #1 / codex r5 BLOCKING #1).

        chat_kwargs = {
            "max_tokens": _resolve_max_tokens(
                openai_request.max_tokens,
                _resolve_enable_thinking(openai_request),
            ),
            **_resolved_sampling_kwargs(openai_request),
        }

        if openai_request.tools:
            chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
        # Codex r5/r7 BLOCKING (PR #807): forward the extracted
        # multimodal payload to the engine — mirrors ``routes/chat.py``
        # lines 1049-1050. Pre-PR the Anthropic route silently dropped
        # ``images`` / ``videos`` on every /v1/messages request, so
        # MLLM models served via Claude SDK never saw the user's
        # uploaded image even though the wire-format carried it. The
        # ``or None`` keeps the chat-route pattern: empty list → None
        # so the engine treats "no media" identically to a text-only
        # request rather than running its multimodal preprocessor on
        # an empty list.
        if images:
            chat_kwargs["images"] = images
        if videos:
            chat_kwargs["videos"] = videos
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
        # for the policy rationale vs chat.py's 422-on-mismatch, and
        # ``_enforce_named_tool_choice_present`` for the "filter dropped
        # everything" guard added in PR #763 codex round-1.
        original_call_count = len(tool_calls or [])
        synthesized_pinned_call = False
        if openai_request.tool_choice:
            tool_calls = _filter_tool_calls_by_tool_choice(
                tool_calls or [], openai_request.tool_choice
            )
            # F8: ``_enforce_named_tool_choice_present`` now returns
            # ``(tool_calls, synthesized)`` — best-effort synthesizes a
            # placeholder ``tool_use`` (rather than raising 422) when
            # the model failed to comply with a pinned ``tool_choice``.
            # The explicit ``synthesized`` signal lets us skip
            # schema validation on the synthesized empty ``input``
            # (codex r1 BLOCKING #1: pinned tools with ``required``
            # fields would otherwise 400 the best-effort path back
            # into the symptom F8 was supposed to fix).
            (
                tool_calls,
                synthesized_pinned_call,
                _named_tool_choice_err,
            ) = _enforce_named_tool_choice_present(
                tool_calls,
                openai_request.tool_choice,
                original_call_count=original_call_count,
            )
            if _named_tool_choice_err:
                raise HTTPException(status_code=422, detail=_named_tool_choice_err)
            # D-ANTHRO-TOOL-USAGE F3: Anthropic ``{"type":"any"}`` enforcement.
            # The adapter has mapped it to OpenAI ``"required"``; mirror the
            # chat-route synth+422 policy so a no-tool reply either becomes
            # a synthesised single-tool call (unambiguous case) or surfaces
            # a 422 the client can act on.
            tool_calls, _required_err = _enforce_required_tool_choice_present(
                tool_calls,
                openai_request.tool_choice,
                tools=openai_request.tools,
            )
            if _required_err:
                raise HTTPException(status_code=422, detail=_required_err)

        # F-220: enforce the same tool_call JSON-schema validation
        # ``routes/chat.py:1651`` runs on the OpenAI ``/v1/chat/completions``
        # route. The Anthropic adapter has already translated
        # ``input_schema`` into the OpenAI ``function.parameters`` shape
        # (see ``api/anthropic_adapter._convert_tool``), so we can reuse
        # the same validator unchanged. Without this, an enum/type/range
        # violation that returns HTTP 400 on chat-completions silently
        # propagated through ``/v1/messages`` as a 200 ``tool_use`` block
        # carrying schema-violating arguments.
        #
        # F8 follow-up: synthesized best-effort calls have empty
        # ``input={}`` which intentionally may not satisfy the
        # pinned tool's schema (e.g. ``required:["city"]``). Skip
        # the validator on those — failing them would re-instate
        # the 422 path F8 is meant to retire. Schema-level "the
        # synthesized input didn't satisfy `required`" complaints
        # belong on the client's downstream dispatch path.
        if tool_calls and openai_request.tools and not synthesized_pinned_call:
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
            # r5-D shared finalize-on-truncation plug — see chat.py
            # for the rationale. Forwarded so the Anthropic surface
            # gets the same gemma4 / glm4 / minimax fixes.
            finish_reason=getattr(output, "finish_reason", None),
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
        # R-01 (was H-01): Anthropic-side mirror of the chat-route opt-in
        # cutoff sentinel. Default-off — the Anthropic envelope already
        # carries ``stop_reason="max_tokens"`` + the ``thinking`` content
        # block, so SDK consumers have an unambiguous structured
        # truncation signal without any synthetic ``text`` block. When
        # the env knob ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1`` is set,
        # the helper restores the legacy literal-text cue for callers
        # who want it (e.g. chat UIs that only render text blocks). See
        # helper docstring for the full predicate set.
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
            # H-03: forward the engine-surfaced matched stop string so
            # the response carries ``stop_reason="stop_sequence"`` +
            # ``stop_sequence: <str>`` per Anthropic's public spec.
            # ``getattr`` keeps the call defensive against engines that
            # haven't been rebuilt against the new ``GenerationOutput``
            # field (None → legacy ``stop`` → ``end_turn`` mapping).
            matched_stop=getattr(output, "matched_stop", None),
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

    # F12: count_tokens must apply the SAME chat template + tools
    # rendering that ``/v1/messages`` applies before tokenizing,
    # otherwise the count under-reports by the per-turn role/turn
    # boilerplate (``<|im_start|>user`` etc.) the real prompt carries.
    # Pre-fix this endpoint tokenized each text segment in isolation
    # and consistently reported ~5 fewer tokens than the matching
    # ``/v1/messages`` ``usage.input_tokens`` (Sergei repro: delta=-5
    # across 4 unrelated prompts).
    #
    # Single source of truth: build the same ``AnthropicRequest`` →
    # ``ChatCompletionRequest`` adapter chain that ``/v1/messages``
    # uses, then run the engine's ``build_prompt`` so the chat template
    # renders the full conversation (system + messages + tools). The
    # tokenizer encode that follows uses ``count_prompt_tokens`` which
    # mirrors ``BatchedEngine.estimate_new_tokens``' BOS-aware
    # ``add_special_tokens`` handling.
    #
    # Fall through to the legacy per-segment count only when the
    # adapter rejects the body OR the engine doesn't expose
    # ``build_prompt`` (test stubs); a meaningful-but-imperfect count
    # beats a 500 on a route whose contract is "estimate this prompt".
    total_tokens: int | None = None
    # Anthropic's ``/v1/messages/count_tokens`` does NOT require
    # ``max_tokens`` (unlike ``/v1/messages``), so callers commonly
    # omit it when estimating cost. ``AnthropicRequest`` also declares
    # ``model`` as required, but this endpoint's pre-existing contract
    # (test_anthropic_route_auth) accepts requests without ``model``.
    # Inject placeholders purely so the schema parses — the
    # count_tokens path doesn't read either field. This keeps the
    # single-adapter-source-of-truth contract without tightening the
    # public count_tokens contract beyond what's already shipped.
    _body_for_parse = dict(body)
    if "max_tokens" not in _body_for_parse:
        _body_for_parse["max_tokens"] = 1
    # Both ``"model" not in body`` (missing) and ``"model": None``
    # (explicit null) are accepted by the count_tokens contract —
    # see ``test_count_tokens_accepts_explicit_null_model``. Inject
    # a placeholder in both cases so the schema parses (the count
    # path doesn't read the field).
    if _body_for_parse.get("model") is None:
        _body_for_parse["model"] = "count-tokens-placeholder"
    # Codex r2 BLOCKING #1: let real ``ValidationError`` shapes bubble
    # out — the global ``_pydantic_validation_handler`` will surface
    # them as sanitized 400s with the same envelope ``/v1/messages``
    # uses, so the two surfaces share their validation contract.
    # Swallowing them here would 200 with a "plausible" count for a
    # malformed request — same cost-estimation footgun F-160 closed
    # for empty messages.
    anthropic_request = AnthropicRequest(**_body_for_parse)
    # Codex r2 BLOCKING #1 continued: ``AnthropicOutputConfigError``
    # is a structured ``ValueError`` subclass we map to 400 with the
    # adapter's own message — mirrors the ``/v1/messages`` route's
    # behavior. Any other unexpected exception from the adapter
    # propagates as a 500, which is the right shape for a server-side
    # regression (better than a silent fallback to legacy counting).
    try:
        openai_request = anthropic_to_openai(anthropic_request)
    except AnthropicOutputConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Codex r2 BLOCKING #2: ``preserve_native_tool_format`` is an
    # optional attribute on the engine contract — guard with
    # ``getattr`` so test stubs without it (or any future engine
    # implementation that omits it) reach the documented fallback
    # path instead of 500-ing here. Mirrors the same defensive
    # pattern in ``count_anthropic_tokens``'s fallback branch.
    try:
        _ctx_messages, _, _ = extract_multimodal_content(
            openai_request.messages,
            preserve_native_format=getattr(
                engine, "preserve_native_tool_format", False
            ),
        )
    except Exception:
        _ctx_messages = None
    build_prompt = getattr(engine, "build_prompt", None)
    if _ctx_messages is not None and callable(build_prompt):
        # Match the ``enable_thinking`` precedence ``/v1/messages``
        # uses (resolved via shared helper). On qwen3-shaped templates
        # this adds an opening ``<think>\n`` prefix to the assistant
        # turn — the prompt actually tokenized at generation time.
        # Pre-fix the count under-reported by exactly this prefix.
        _cfg = get_config()
        resolved_thinking = _resolve_enable_thinking(openai_request)
        effective_thinking = _effective_enable_thinking(
            resolved_thinking, _cfg.model_path or _cfg.model_name
        )
        try:
            rendered_tools = (
                convert_tools_for_template(openai_request.tools)
                if openai_request.tools
                else None
            )
            prompt = build_prompt(
                _ctx_messages,
                tools=rendered_tools,
                enable_thinking=effective_thinking,
            )
        except Exception:
            prompt = None
        if isinstance(prompt, str) and prompt:
            total_tokens = count_prompt_tokens(engine, prompt)

    # Legacy fallback path — only fires when the adapter / template
    # render failed. Keeps the endpoint useful on test stubs that don't
    # expose ``build_prompt`` AND keeps the historical zero-floor
    # behavior on adapter rejection. The delta this path leaves on the
    # table is the chat-template overhead (~5 tokens for Qwen-shaped
    # templates) — the same gap F12 fixes when the primary path runs.
    if total_tokens is None:
        tokenizer = engine.tokenizer
        total_tokens = 0
        system = body.get("system", "")
        if isinstance(system, str) and system:
            total_tokens += len(tokenizer.encode(system))
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total_tokens += len(tokenizer.encode(text))
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


def _split_tool_input_json(tool_input: object) -> list[str]:
    """Return a list of ``input_json_delta.partial_json`` fragments
    whose concatenation equals ``json.dumps(tool_input)``.

    R-08 (r5-A bundle) — Anthropic's streaming spec emits
    ``input_json_delta`` progressively as the model generates the
    tool arguments. Local inference produces the full JSON
    structurally (the parser only surfaces tool_calls once the
    arguments object closes), so we can't stream per-token; the
    next best is to emit one fragment per top-level key-value pair
    so a consumer that parses-as-it-goes (the documented use case
    for ``input_json_delta``) gets at least the same number of cues
    an Anthropic-side stream would.

    Codex r5 NIT (PR #826): the structural braces are attached to
    their adjacent key-value fragments — the first pair carries the
    opening ``{`` and the last pair carries the closing ``}`` — so
    every emitted fragment is a complete structural addition to the
    accumulating JSON. The earlier draft emitted standalone ``{``
    and ``}`` fragments, which strictly satisfied the
    byte-concatenation contract but surfaced empty-value cues to
    progressive consumers (the first event carried no key/value
    payload, just a brace). Pairing the braces with their adjacent
    pair removes those empty cues while keeping
    ``"".join(fragments) == json.dumps(tool_input)`` exactly.

    Empty / non-string-keyed / non-dict inputs return a single
    fragment so the wire contract (at least one ``input_json_delta``
    per tool block) holds; concatenating that single fragment still
    yields the exact ``json.dumps`` bytes.

    Codex r1 NIT #3: the per-pair split path is only safe when every
    dict key is a string — ``json.dumps`` coerces ``int`` / ``bool``
    keys to their string form (``{1: "x"}`` → ``{"1": "x"}``), and
    ``json.dumps(1)`` would emit ``1`` (no quotes), breaking the
    concatenation contract. Fall back to the monolithic shard for
    those shapes so byte-equivalence holds regardless of input.
    """
    if not isinstance(tool_input, dict) or not tool_input:
        return [json.dumps(tool_input)]
    if not all(isinstance(k, str) for k in tool_input):
        # Non-string keys: ``json.dumps`` coerces them but the
        # per-pair ``json.dumps(key)`` we use below would emit the
        # raw value (no coercion), breaking byte-equivalence. The
        # whole-blob encoding is the only safe fallback for the
        # progressive contract on those shapes.
        return [json.dumps(tool_input)]
    keys = list(tool_input.keys())
    fragments: list[str] = []
    for i, key in enumerate(keys):
        value_repr = json.dumps(tool_input[key])
        key_repr = json.dumps(key)
        opener = "{" if i == 0 else ", "
        closer = "}" if i == len(keys) - 1 else ""
        fragments.append(f"{opener}{key_repr}: {value_repr}{closer}")
    return fragments


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
    *,
    request_id_holder: list | None = None,
    prompt_tokens_estimate: int | None = None,
    prepared_messages: list | None = None,
    prepared_images: list | None = None,
    prepared_videos: list | None = None,
) -> AsyncIterator[str]:
    """Stream Anthropic Messages API SSE events.

    Args:
        request_id_holder: C-01 force-abort plumbing. Forwarded to
            ``engine.stream_chat`` so the engine writes the admitted
            scheduler request id into ``holder[0]``; the route's
            ``_disconnect_guard`` reads the same holder and force-calls
            ``scheduler.abort_request`` on client disconnect. ``None``
            (default) is a no-op.
        prompt_tokens_estimate: D-ANTHRO-TOOL-USAGE F5 — pre-computed
            prompt-token count from the route entry-point's
            ``enforce_context_length_for_messages`` call. ``None``
            (sentinel — codex r4 NIT) means "the caller did not
            compute one"; ``0`` is now a real value meaning "engine
            permissive-skip path returned no count". Both fall back
            to the route-internal ``_estimate_anthropic_prompt_tokens``
            helper, which goes through the SAME
            ``build_prompt`` + ``count_prompt_tokens`` path.
        prepared_messages: D-ANTHRO-TOOL-USAGE F3 (codex r4 BLOCKING
            #1) — pre-extracted + suffix-injected messages from the
            route entry-point. When supplied, the streaming helper
            skips its own extract+inject so suffix injection happens
            in EXACTLY ONE layer and ``prompt_tokens_estimate`` is
            guaranteed to match what the engine actually sees.
            ``None`` (the default) preserves the direct-call test
            path: extract from ``openai_request.messages`` and inject
            inline, matching pre-PR-807 streaming behaviour.
        prepared_images / prepared_videos: D-ANTHRO-TOOL-USAGE F3
            (codex r5 BLOCKING #1) — multimodal payload from the same
            route-level extract, threaded alongside
            ``prepared_messages`` so /v1/messages streaming requests
            against MLLM engines keep their image / video inputs.
            Pre-r5 the streaming helper discarded these to ``[]``
            when ``prepared_messages`` was supplied, silently
            dropping every multimodal stream's media inputs.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()

    if prepared_messages is not None:
        # Caller (the route entry-point) already extracted +
        # suffix-injected, AND already paid for the
        # ``enforce_context_length_for_messages`` DoS gate against
        # this exact list. Reuse verbatim — no second injection.
        messages = prepared_messages
        images = prepared_images if prepared_images is not None else []
        videos = prepared_videos if prepared_videos is not None else []
    else:
        messages, images, videos = extract_multimodal_content(
            openai_request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

        # D-ANTHRO-TOOL-USAGE F3: forced ``tool_choice`` levers — same
        # injection the non-stream branch performs. Only runs on the
        # direct-call path (no ``prepared_messages``); the route-level
        # caller has already done this above and threaded the result.
        _inject_tool_use_required_suffix(
            messages,
            openai_request.tool_choice,
            tools=openai_request.tools,
        )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(
            openai_request.max_tokens,
            _resolve_enable_thinking(openai_request),
        ),
        **_resolved_sampling_kwargs(openai_request),
    }
    # C-01: thread the request_id holder to the engine so disconnect
    # detection can force-call scheduler.abort_request.
    if request_id_holder is not None:
        chat_kwargs["request_id_holder"] = request_id_holder

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
    # Codex r5/r7 BLOCKING (PR #807): forward the multimodal payload
    # to the engine on the streaming path too — same parity with
    # ``routes/chat.py`` lines 1049-1050 the non-stream branch now
    # follows. The ``if`` gates avoid passing empty lists so the
    # engine's multimodal preprocessor stays on the text-only fast
    # path for plain prompts.
    if images:
        chat_kwargs["images"] = images
    if videos:
        chat_kwargs["videos"] = videos
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

    # D-ANTHRO-TOOL-USAGE F5: pre-compute prompt_tokens BEFORE
    # ``message_start`` so the SSE envelope carries the real input
    # estimate instead of the hard-coded ``0`` that under-reported the
    # input share by 100%. Prefers the route-level estimate when the
    # caller threaded one (already shared with the context-length DoS
    # gate — codex r2 NIT) and falls back to a local render when the
    # caller used the direct-call test path.
    #
    # Codex r4 NIT (PR #807): treat ``None`` as the SENTINEL for "no
    # estimate available", separate from ``0`` ("estimator ran but
    # the engine returned no count" — MLLM, empty prompt, …). The
    # earlier ``or`` chain conflated the two and would re-render every
    # genuinely-empty prompt.
    #
    # Codex r8 BLOCKING #1+#2 (PR #807): defense-in-depth — coerce
    # the result to int before it flows into the SSE envelope or the
    # running-counter seed. The fallback estimator returns ``int``
    # already (``0`` on all skip paths), but a future refactor that
    # surfaces ``None`` from either source MUST NOT poison the
    # Anthropic ``usage.input_tokens`` int field — JSON-serialising
    # ``None`` would emit ``"input_tokens": null`` which violates the
    # public schema.
    if prompt_tokens_estimate is None:
        initial_prompt_tokens_estimate = _estimate_anthropic_prompt_tokens(
            engine,
            messages,
            tools=openai_request.tools,
        )
    else:
        initial_prompt_tokens_estimate = prompt_tokens_estimate
    # Coerce ``None`` or any other non-int to ``0`` before it reaches
    # the wire.
    if not isinstance(initial_prompt_tokens_estimate, int):
        initial_prompt_tokens_estimate = 0

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
                "input_tokens": initial_prompt_tokens_estimate,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # H-05 follow-up (PR #771 codex round-2 BLOCKING #1): when
    # ``tool_choice`` pins a specific tool, the model is supposed to
    # emit a ``tool_use`` for that tool. Local inference can't
    # decoder-enforce that, so a defiant model can stream a TEXT
    # response instead — and pre-fix, those text deltas were
    # yielded chunk-by-chunk before we knew to reject the response.
    # By the time the post-loop enforcement fired the SSE error event,
    # the client had already received a partial text payload that
    # violates the forced-tool contract.
    #
    # Fix: when a named ``tool_choice`` is set, BUFFER every
    # content_block / thinking event produced inside the chunk loop
    # (and the post-loop flushes) into ``pre_filter_buffer`` instead
    # of yielding it. After the loop finishes we run the same filter +
    # enforcement step the non-stream branch uses; on success we
    # replay the buffer so streaming UX is preserved, on failure we
    # drop the buffer and emit only the SSE error event so the
    # forbidden text payload never reaches the wire. ``message_start``
    # is yielded above this (clients use it to allocate the message
    # frame) and ``message_delta`` / ``message_stop`` are yielded
    # below the enforcement (they only describe the terminal state).
    _pinned_tool_target = _named_tool_choice_target(
        getattr(openai_request, "tool_choice", None)
    )
    # D-ANTHRO-TOOL-USAGE F3: extend the pre-filter buffer to the
    # ``tool_choice={"type":"any"}`` case (Anthropic ``any`` →
    # OpenAI ``"required"``). Same rationale as the named-pin branch:
    # a defiant model can stream a text reply that violates the
    # forced-call contract, and post-loop synthesis / SSE error must
    # arrive without the forbidden text bytes ever reaching the wire.
    _buffer_for_pinned_tool = _pinned_tool_target is not None or (
        _is_required_tool_choice(getattr(openai_request, "tool_choice", None))
        and bool(openai_request.tools)
    )
    pre_filter_buffer: list[str] = []

    def _capture(event: str) -> str | None:
        """Either buffer ``event`` and return ``None``, or return
        ``event`` unchanged.

        When a named ``tool_choice`` is pinned, every chunk-loop
        content event is appended to ``pre_filter_buffer`` and this
        helper returns ``None`` so the caller's ``if ev is not None:
        yield ev`` is a no-op. Otherwise the helper returns the
        event unchanged and the caller yields it immediately —
        preserving the original "one event in ⇒ one event out"
        streaming semantics. ``async for`` constructs in Python 3.12
        don't allow ``yield from`` against a sync generator helper,
        so this returns a scalar rather than an iterator.
        """
        if _buffer_for_pinned_tool:
            pre_filter_buffer.append(event)
            return None
        return event

    accumulated_text = ""
    accumulated_raw = ""
    accumulated_raw_parts: list[str] = []
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
    # D-ANTHRO-TOOL-USAGE F5: seed the running counter with the
    # pre-message_start estimate so the terminal ``message_delta`` is
    # NEVER worse than what we already announced in ``message_start``.
    # The engine's own ``output.prompt_tokens`` (when it surfaces a
    # non-zero value below) wins over the seed because the engine count
    # is authoritative — the seed is a floor, not a ceiling.
    prompt_tokens = initial_prompt_tokens_estimate
    completion_tokens = 0
    cached_tokens = 0
    # H-03: track the most-recently-surfaced ``matched_stop`` so the
    # terminal ``message_delta`` can emit Anthropic's
    # ``stop_reason="stop_sequence"`` + ``stop_sequence: <str>`` per the
    # public spec. The scheduler pins this exactly once (on the chunk
    # that fires the stop check) and downstream wrappers preserve it
    # through to the terminal sentinel, so reading the latest non-None
    # value is equivalent to "did any stop fire on this request?".
    # Stays ``None`` for EOS / length / no-stop terminations.
    stream_matched_stop: str | None = None
    # R-06 (r5-A bundle): track the engine-surfaced ``finish_reason``
    # so the terminal ``message_delta`` can emit Anthropic's correct
    # ``stop_reason`` per the public spec instead of hard-coding
    # ``end_turn``. Pre-r5-A the route ignored ``finish_reason``
    # entirely and every non-tool stream finished with ``end_turn``,
    # breaking the spec-required ``max_tokens`` continuation pattern
    # (Mei dogfood report ``mei-r1.md`` HIGH). ``length`` →
    # ``max_tokens``; everything else maps via the existing
    # tool-use / stop-sequence / end-turn ladder below.
    stream_finish_reason: str | None = None

    current_block_type = None
    block_index = 0
    # C-08 + R-07 (r5-A bundle): track whether the chunk loop ever
    # opened a thinking block AND whether any content_block was
    # emitted at all. C-08 needs the thinking flag so the
    # ``finalize_streaming`` Case-2 path (no ``</think>`` seen) does
    # NOT re-encode already-streamed thinking bytes as a phantom text
    # block (Mei dogfood report ``mei-r1.md`` CRIT). R-07 needs the
    # any-block flag so a stream that ends with ``message_start →
    # message_delta → message_stop`` (no content blocks at all — e.g.
    # ``/no_think`` + ``max_tokens`` exhausted entirely by suppressed
    # reasoning) can synthesize a zero-text ``content_block`` pair so
    # SDKs that iterate ``message.content`` see a valid empty block
    # instead of a malformed Message (Mei ``mei-r1.md`` HIGH).
    streamed_any_thinking = False
    streamed_any_content_block = False
    # C-08 (r5-A bundle, codex r2 REQUIRED): accumulate the exact bytes
    # the chunk loop shipped as ``thinking_delta`` so the terminal
    # ``finalize_streaming`` re-encoding guard can do a byte-faithful
    # "did the parser just re-classify already-streamed bytes?" check.
    # Tag-presence heuristics (``"<think>" in accumulated_raw`` etc.)
    # miss models that emit a reasoning preamble BEFORE the literal
    # ``<think>`` opener (VibeThinker, codex r2). Byte-faithful
    # comparison is parser-agnostic — the suppression fires iff the
    # finalize correction's content is bytes the wire has already
    # carried as thinking.
    streamed_thinking_text = ""

    def _emit_and_track(
        pieces: list[tuple[str, str]],
        cur_block_type: str | None,
        blk_index: int,
    ) -> tuple[list[str], str | None, int]:
        """Run ``_emit_content_pieces`` and update C-08 / R-07 flags.

        Tracks whether any thinking piece reached the wire (C-08
        guards the ``finalize_streaming`` re-encoding against this),
        whether any content_block at all was emitted (R-07 uses this
        to decide if the terminal stream needs a synthetic empty
        content_block pair), and the exact bytes shipped as thinking
        (the byte-faithful C-08 suppression discriminator). The flag
        updates happen here so every call site stays a single line
        and the ``nonlocal`` plumbing lives in exactly one place.
        """
        nonlocal streamed_any_thinking, streamed_any_content_block
        nonlocal streamed_thinking_text
        if pieces:
            streamed_any_content_block = True
            for piece_type, piece_text in pieces:
                if piece_type == "thinking":
                    streamed_any_thinking = True
                    streamed_thinking_text += piece_text
        return _emit_content_pieces(pieces, cur_block_type, blk_index)

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
        # H-03: latch the matched stop string from whichever chunk
        # carries it. ``getattr`` keeps legacy mocks without the field
        # working unchanged.
        _chunk_matched_stop = getattr(output, "matched_stop", None)
        if _chunk_matched_stop:
            stream_matched_stop = _chunk_matched_stop
        # R-06 (r5-A bundle): latch the engine's ``finish_reason`` on
        # every chunk that carries one. The scheduler typically pins
        # this on the LAST chunk (sentinel) so simply taking the
        # newest non-None reading is equivalent to "whichever
        # finish_reason fired for this request". ``getattr`` keeps
        # legacy mocks without the field working unchanged.
        _chunk_finish_reason = getattr(output, "finish_reason", None)
        if _chunk_finish_reason:
            stream_finish_reason = _chunk_finish_reason

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
                if output_channel in ("reasoning", "content", "tool_call"):
                    accumulated_raw_parts.append(delta_text)
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
                    events, current_block_type, block_index = _emit_and_track(
                        _gate_thinking_pieces(pieces_routed, current_block_type),
                        current_block_type,
                        block_index,
                    )
                    for event in events:
                        ev = _capture(event)
                        if ev is not None:
                            yield ev
                continue

            accumulated_raw_parts.append(delta_text)

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
                # Compatibility path: reasoning parsers still consume
                # the legacy ``previous + delta == current`` API. This
                # cumulative concat remains O(n^2) for active
                # reasoning_parser streams; the list buffer above only
                # fixes the no-reasoning hot path and final parse.
                accumulated_raw = previous_raw + delta_text
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
                    events, current_block_type, block_index = _emit_and_track(
                        _gate_thinking_pieces(pieces, current_block_type),
                        current_block_type,
                        block_index,
                    )
                    for event in events:
                        ev = _capture(event)
                        if ev is not None:
                            yield ev
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
                events, current_block_type, block_index = _emit_and_track(
                    _gate_thinking_pieces(pieces, current_block_type),
                    current_block_type,
                    block_index,
                )
                for event in events:
                    ev = _capture(event)
                    if ev is not None:
                        yield ev

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
        events, current_block_type, block_index = _emit_and_track(
            _gate_thinking_pieces(pieces_flush, current_block_type),
            current_block_type,
            block_index,
        )
        for event in events:
            ev = _capture(event)
            if ev is not None:
                yield ev

    if not reasoning_parser:
        flush_pieces = think_router.flush()
        if flush_pieces:
            events, current_block_type, block_index = _emit_and_track(
                _gate_thinking_pieces(flush_pieces, current_block_type),
                current_block_type,
                block_index,
            )
            for event in events:
                ev = _capture(event)
                if ev is not None:
                    yield ev

    # Close final content block
    if current_block_type is not None:
        ev = _capture(
            f"event: content_block_stop\ndata: "
            f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
        )
        if ev is not None:
            yield ev
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
    if accumulated_raw_parts and not accumulated_raw:
        accumulated_raw = "".join(accumulated_raw_parts)

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
                    events, current_block_type, block_index = _emit_and_track(
                        [("text", filtered)], current_block_type, block_index
                    )
                    for event in events:
                        ev = _capture(event)
                        if ev is not None:
                            yield ev
        # Close any block we opened above before falling through to the
        # finalize_streaming path.
        if current_block_type is not None:
            ev = _capture(
                f"event: content_block_stop\ndata: "
                f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
            )
            if ev is not None:
                yield ev
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
        # C-08 (r5-A bundle): the parser's ``finalize_streaming``
        # fallback (qwen3 / deepseek_r1 / thinking-family parsers)
        # reclassifies the whole buffer as ``content`` when the model
        # never produced ``</think>``. That fallback exists for the
        # NON-STREAMING surface, where the whole-buffer re-parse is the
        # FIRST and ONLY chance to surface those bytes — the parser's
        # implicit-think heuristic conservatively buckets them as
        # reasoning during incremental scanning and ``finalize_streaming``
        # is the corrective pass. On the streaming surface, however,
        # we already shipped those bytes as ``thinking_delta`` events
        # via the chunk loop, and the corrective pass re-encodes them
        # as a fresh text content_block — the exact "data fabrication"
        # failure mode Mei's r1 dogfood caught as CRIT (R-01 streaming
        # twin: identical bytes shipped as both thinking AND text).
        #
        # The distinguishing question is whether the streaming
        # surface's bucketing was correct — i.e. whether the bytes
        # ``finalize_streaming`` is about to surface as content were
        # ALREADY shipped as ``thinking_delta`` to the wire. Three
        # qualitatively different streams reach this branch:
        #
        #   (a) Model emitted plain content with no tags AND finished
        #       naturally (``finish_reason="stop"``). The parser was
        #       too conservative (implicit-think); the corrective pass
        #       is RIGHT — those bytes belong in a text block (test
        #       ``test_no_think_tags_yields_text_delta`` — natural
        #       stop, no length guard). The corrective pass runs in
        #       this case because the ``stream_finish_reason ==
        #       "length"`` guard below blocks suppression.
        #   (b) Model emitted plain content with no tags AND got
        #       truncated by ``max_tokens``. The streaming surface
        #       has ALREADY shipped those bytes as ``thinking_delta``
        #       events; the finalize correction would emit them again
        #       as ``text_delta``, producing a Message with
        #       ``content=[thinking, text]`` where both blocks carry
        #       byte-identical content — the same duplicate-bytes
        #       shape C-08 closes. Re-classifying mid-stream is
        #       impossible (those events are on the wire). Suppress
        #       the duplicate emit; the client sees a thinking-only
        #       Message but with no fabricated content. This is the
        #       deliberate trade-off codex r5 questioned (BLOCKING
        #       analysis); the alternative (let the duplicate emit)
        #       re-introduces the original C-08 deception. The
        #       streaming-surface fix — never route those bytes
        #       through ``thinking_delta`` in the first place — is
        #       a parser-side change tracked separately from this
        #       finalize bundle.
        #   (c) Model produced reasoning that the streaming surface
        #       correctly bucketed as thinking (template ``<think>``
        #       prefix, model-emitted ``<think>``, OR a parser-side
        #       preamble pattern like VibeThinker's chatty pre-tag
        #       sentences) AND was truncated by ``max_tokens`` before
        #       closing. The streaming bucketing was RIGHT — those
        #       bytes are reasoning and the corrective pass would dup
        #       them as text. This is the C-08 path Mei's r1 dogfood
        #       caught.
        #
        # The byte-faithful discriminator (codex r2 REQUIRED):
        # compare the finalize correction's content against the bytes
        # we already shipped as ``thinking_delta``. Tag-presence
        # heuristics would miss case (c) — VibeThinker / DeepSeek-R1
        # implicit-reasoning preambles where the parser streams
        # thinking BEFORE the ``<think>`` opener is reached have
        # ``streamed_any_thinking=True`` and ``"<think>" not in
        # accumulated_raw`` simultaneously — and the only signal that
        # discriminates (b)+(c) from (a) is the length truncation
        # guard. The byte comparison is parser-agnostic: if
        # ``finalize_streaming`` returns exactly the bytes we already
        # streamed (after Anthropic's pre-emit
        # ``strip_special_tokens`` so the comparison is on wire-shape
        # bytes), the corrective pass is duplicating; otherwise it is
        # surfacing new (parser-held) content and must run.
        # Compute ``final_msg`` ONCE and apply the byte-faithful
        # suppression in-place. The parser's ``finalize_streaming``
        # is a pure function (qwen3 / deepseek_r1 / vibethinker —
        # read ``accumulated_text`` + class attributes only), so the
        # single call is also safer if a future parser variant
        # introduces side effects: we only invoke once.
        final_msg = (
            reasoning_parser.finalize_streaming(accumulated_raw)
            if hasattr(reasoning_parser, "finalize_streaming")
            else None
        )
        if (
            final_msg is not None
            and streamed_any_thinking
            and stream_finish_reason == "length"
            and "</think>" not in accumulated_raw
        ):
            probe_content = getattr(final_msg, "content", None)
            if isinstance(probe_content, str) and probe_content:
                # Both sides of the equality are post-
                # ``strip_special_tokens`` — the streamed buffer
                # accumulates pieces that were stripped at every
                # in-loop emission site (channel-routed reasoning at
                # line 2051, parser-routed reasoning at line 2170,
                # think_router path at line 2277), so the
                # comparison is against the EXACT wire bytes.
                normalized_probe = strip_special_tokens(probe_content)
                # Strict equality only (codex r2 REQUIRED #2): mutual
                # containment risks dropping legitimate trailing
                # content the parser HELD back during streaming and
                # now releases via finalize. For the two known
                # ``finalize_streaming`` implementers (``qwen3``,
                # ``deepseek_r1`` / ``vibethinker``), the Case-1/2
                # fallback returns exactly the wire bytes —
                # ``accumulated_text`` minus a literal ``<think>``
                # prefix when present — so equality fires for the
                # documented C-08 + VibeThinker-preamble paths. A
                # future parser whose finalize emits MORE bytes than
                # were streamed would fall through to the legacy
                # correction path; that's safe (no over-suppression
                # of new bytes) and the duplicate-detection
                # regression test would catch it for follow-up.
                if (
                    streamed_thinking_text
                    and normalized_probe == streamed_thinking_text
                ):
                    # Drop the duplicate — the wire already carried
                    # these bytes as ``thinking_delta``. ``final_msg``
                    # gets cleared so the emit branch below short-
                    # circuits.
                    final_msg = None
        if final_msg and final_msg.content:
            content = strip_special_tokens(final_msg.content)
            if content:
                accumulated_text = content
                for raw_event in (
                    f"event: content_block_start\ndata: "
                    f"{json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n",
                    f"event: content_block_delta\ndata: "
                    f"{json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': content}})}\n\n",
                    f"event: content_block_stop\ndata: "
                    f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n",
                ):
                    ev = _capture(raw_event)
                    if ev is not None:
                        yield ev
                # Re-tracking: this is a manually-yielded
                # content_block triple, not routed through
                # ``_emit_and_track``. Mark the any-block flag so R-07
                # doesn't double-synthesize a second empty block.
                streamed_any_content_block = True
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
    #
    # IMPORTANT (PR #763 codex round-1 BLOCKING #2 — confirm-and-lock):
    # NO ``content_block_start`` for ``type=tool_use`` is emitted in the
    # while-loop above. The structured tool-call payload is only
    # collected into ``accumulated_structured_tool_calls`` (see the
    # ``engine_tool_calls`` extend at the stream-chunk site), and the
    # tool_use SSE events are emitted strictly below (after the
    # filter + validation). If a future refactor moves tool_use deltas
    # earlier in the stream, this filter MUST be re-applied at the
    # earlier emission point or the dropped tool's content_block_start
    # will reach the wire before we know to suppress it.
    tool_choice_error: str | None = None
    # The helper's boolean is retained for the Anthropic ``any`` fallback
    # below; named pinned-tool failures now return an error instead of a
    # synthesized placeholder so the route can surface 422 consistently.
    synthesized_pinned_call = False
    original_call_count_stream = len(tool_calls or [])
    if openai_request.tool_choice:
        tool_calls = _filter_tool_calls_by_tool_choice(
            tool_calls or [], openai_request.tool_choice
        )
        # Named pinned-tool failures are contract errors. The streaming
        # HTTP wrapper buffers named-tool streams so this becomes an HTTP
        # 422 before a successful SSE response is returned.
        (
            tool_calls,
            synthesized_pinned_call,
            _named_tool_choice_err_stream,
        ) = _enforce_named_tool_choice_present(
            tool_calls,
            openai_request.tool_choice,
            original_call_count=original_call_count_stream,
        )
        if _named_tool_choice_err_stream:
            tool_choice_error = _named_tool_choice_err_stream

        # D-ANTHRO-TOOL-USAGE F3: stream variant of
        # ``_enforce_required_tool_choice_present`` for
        # ``tool_choice={"type":"any"}`` (Anthropic ``any`` →
        # OpenAI ``"required"``). Headers are already on the wire so
        # the 422 the non-stream branch raises becomes a buffer-replay-
        # AND-error path: when synthesis is unambiguous (single tool),
        # we add the synth call so a downstream client sees a
        # ``tool_use`` block; otherwise we surface the same
        # ``invalid_request_error`` event the named-pin branch uses.
        # Mirrors the OpenAI route's stream behaviour: best-effort
        # prompt-injection upstream + post-parse synth where the
        # target is unambiguous + SSE error event when it isn't.
        if not tool_choice_error:
            tool_calls, _required_err_stream = _enforce_required_tool_choice_present(
                tool_calls,
                openai_request.tool_choice,
                tools=openai_request.tools,
            )
            if _required_err_stream:
                tool_choice_error = _required_err_stream

    # F-220: enforce JSON-schema validation on the model's emitted
    # tool_call arguments. On the streaming branch, headers are already
    # sent so a mid-stream ``HTTPException`` cannot be returned as a 400
    # response. Instead, surface the validation error as an Anthropic
    # SSE ``error`` event (``invalid_request_error``) and drop the
    # offending tool_use blocks so the client can recover. Matches the
    # non-stream branch's 400 contract in spirit while staying within
    # the Anthropic streaming protocol.
    tool_validation_error: str | None = None
    # F8 follow-up: skip the JSON-schema validator on synthesized
    # best-effort calls — their empty ``input={}`` is intentionally
    # placeholder and may not satisfy ``required`` fields. Same
    # rationale as the non-stream branch (codex r1 BLOCKING #1).
    if tool_calls and openai_request.tools and not synthesized_pinned_call:
        try:
            _validate_tool_call_params(tool_calls, openai_request.tools)
        except HTTPException as exc:
            tool_validation_error = (
                exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            )
            tool_calls = []

    # PR #771 codex round-2 BLOCKING #1: replay or drop the
    # ``pre_filter_buffer`` we accumulated during the chunk loop.
    # Until this point in the stream the only event yielded was the
    # opening ``message_start``; every content_block / thinking event
    # that the chunk loop produced sits in the buffer. We now know
    # whether the enforcement passed:
    #
    #   * pass → replay the buffer so the streaming UX is preserved
    #     (clients see the same text-delta cadence they would have
    #     seen without the buffer). Tool_use blocks emit below.
    #   * fail → drop the buffer so the would-be text response NEVER
    #     reaches the wire. Only the SSE error event + the trailing
    #     ``message_delta`` (end_turn) + ``message_stop`` follow.
    #
    # The buffer is unused when ``tool_choice`` is not a named pin —
    # in that case the chunk loop yielded directly and ``pre_filter_buffer``
    # is empty, so this block is a no-op on every non-pinned request.
    # F8 follow-up: when synthesis fired the model emitted text instead
    # of the pinned tool. Replaying that buffered text would put the
    # forbidden text payload back on the wire (same family of contract
    # violation the 422 path used to suppress); drop the buffer and let
    # the synthesized tool_use below carry the entire content list.
    if synthesized_pinned_call:
        pre_filter_buffer.clear()
    if _buffer_for_pinned_tool and not (
        tool_choice_error or tool_validation_error or synthesized_pinned_call
    ):
        for buffered_event in pre_filter_buffer:
            yield buffered_event
        pre_filter_buffer.clear()

    # Emit a single SSE error event when either the tool_choice
    # enforcement or the schema validator fired. Both classes are
    # surfaced as ``invalid_request_error`` — they describe a
    # client-actionable failure (retry / fall back / relax pin),
    # whereas a true server failure would arrive via the route-level
    # exception handler.
    if tool_choice_error or tool_validation_error:
        # On the buffered path the buffer is intentionally NOT replayed
        # — the would-be text payload that the chunk loop accumulated
        # is precisely what the named ``tool_choice`` contract forbids.
        # Drop it on the floor and surface only the error event.
        pre_filter_buffer.clear()
        error_event = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": tool_choice_error or tool_validation_error,
            },
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        # When the pinned-tool enforcement fires, the only emit-worthy
        # output is the error event — drop any surviving tool_calls so
        # the loop below doesn't ship a ``tool_use`` for a state the
        # error event already marked unrecoverable.
        if tool_choice_error:
            tool_calls = []

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_index = block_index + i
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}

            # R6-M2: Anthropic Computer-Use spec uses ``coordinate``
            # for single-point verbs and ``start_coordinate`` +
            # ``coordinate`` (the end) for drag. The UI-TARS parser
            # emits the canonical ``point`` / ``start_point`` /
            # ``end_point`` keys (PR #812 contract — chat-completions
            # OpenAI lane stays bytes-faithful to that). Translate on
            # the streaming ``/v1/messages`` boundary so a client
            # correlating non-stream + stream sees the same spec key
            # (the non-stream adapter ``openai_to_anthropic`` applies
            # the same mapping). Gated on ``name=="computer"`` so
            # vanilla function tools whose args happen to carry
            # ``point`` are untouched.
            if tc.function.name == "computer" and isinstance(tool_input, dict):
                from ..tool_parsers.ui_tars_tool_parser import (
                    translate_to_anthropic_spec_keys,
                )

                tool_input = translate_to_anthropic_spec_keys(tool_input)

            # F9: normalize the ``tool_use.id`` once per call. The
            # current loop only references ``tc.id`` inside the
            # ``content_block_start`` event, but if a future patch adds
            # another emission point (e.g. a ``content_block_delta``
            # that references the parent id), calling
            # ``to_anthropic_tool_use_id`` afresh on a ``None`` / non-
            # ``call_`` id would mint a DIFFERENT ``toolu_<hex>`` each
            # time, breaking the stable-id correlation across stream
            # events. Compute once, reference everywhere downstream
            # (codex r1 BLOCKING #3).
            anthropic_tool_id = to_anthropic_tool_use_id(tc.id)

            tool_block_start = {
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {
                    "type": "tool_use",
                    # F9: rewrite OpenAI-style ``call_<hex>`` ids to
                    # Anthropic's ``toolu_<hex>`` prefix. Streaming
                    # branch mirrors the non-stream adapter
                    # (``openai_to_anthropic``) so a client correlating
                    # ``tool_use.id`` across stream + non-stream sees
                    # the same prefix.
                    "id": anthropic_tool_id,
                    "name": tc.function.name,
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(tool_block_start)}\n\n"
            # R-07 tracking: tool_use blocks count as content_blocks
            # for the malformed-message guard below.
            streamed_any_content_block = True

            # R-08 (r5-A bundle): emit ``input_json_delta`` PROGRESSIVELY
            # instead of buffering the entire serialized JSON into a
            # single shard. The Anthropic spec streams
            # ``input_json_delta.partial_json`` fragments that the
            # client accumulates, and consumers that parse-as-they-go
            # (the documented use case for the delta format) get no
            # incremental signal when the server ships the whole JSON
            # in one event. Mei's r2 dogfood caught this on the
            # Computer-Use streaming path. ``_split_tool_input_json``
            # produces structurally-meaningful fragments (one per
            # top-level key-value pair) so concatenation by the client
            # still yields the exact bytes the engine produced.
            for fragment in _split_tool_input_json(tool_input):
                input_delta = {
                    "type": "content_block_delta",
                    "index": tool_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": fragment,
                    },
                }
                yield (
                    f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"
                )

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

    # R-07 (r5-A bundle): synthesize a zero-text ``content_block`` pair
    # ONLY for the malformed-message case the dogfood actually caught
    # — non-zero ``output_tokens`` burned by suppressed reasoning,
    # with no content blocks emitted, ``max_tokens`` truncation, and
    # no error event already on the wire. Pre-r5-A the
    # ``/no_think`` + ``max_tokens`` case (the entire budget eaten by
    # suppressed reasoning) emitted ``message_start → message_delta →
    # message_stop`` with an empty content list and non-zero
    # ``output_tokens`` — a malformed Message that the Anthropic SDK
    # would surface as an empty ``message.content=[]`` despite billing
    # the consumer for tokens that produced no visible payload (Mei
    # ``mei-r1.md`` HIGH).
    #
    # Codex r1 REQUIRED #1: tighten the predicate so a legitimately
    # empty completion (e.g. ``max_tokens=0`` returning nothing, or
    # the engine emitting zero tokens) stays empty rather than
    # silently growing a synthetic block clients did not produce.
    # The actual bug class is "billed for tokens that produced no
    # content_block" — both ``completion_tokens > 0`` AND
    # ``finish_reason="length"`` are required to enter that state, so
    # gate on both.
    if (
        not streamed_any_content_block
        and not tool_calls
        and not (tool_choice_error or tool_validation_error)
        and completion_tokens > 0
        and stream_finish_reason == "length"
    ):
        empty_block_index = block_index
        for raw_event in (
            f"event: content_block_start\ndata: "
            f"{json.dumps({'type': 'content_block_start', 'index': empty_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n",
            f"event: content_block_stop\ndata: "
            f"{json.dumps({'type': 'content_block_stop', 'index': empty_block_index})}\n\n",
        ):
            ev = _capture(raw_event)
            if ev is not None:
                yield ev
        block_index += 1
        streamed_any_content_block = True

    # R-06 (r5-A bundle): map the engine's ``finish_reason`` onto the
    # Anthropic ``stop_reason`` enum (``end_turn``, ``max_tokens``,
    # ``stop_sequence``, ``tool_use``). Tool-use wins over everything
    # else (mutually-exclusive per the public spec); ``length`` from
    # the engine becomes ``max_tokens`` for spec-compliant
    # continuation; the ``stop_sequence`` ladder below preserves H-03
    # behaviour for user-supplied ``stop_sequences`` matches.
    if tool_calls:
        stop_reason = "tool_use"
    elif stream_finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    # H-03: when a user-supplied ``stop_sequences`` entry fired (and the
    # turn would otherwise have terminated normally with ``end_turn``),
    # surface Anthropic's dedicated ``stop_sequence`` reason + populate
    # the matched bytes — mirroring the non-stream adapter. Tool-use
    # finishes still win: the model emitting a tool_call AND happening
    # to also surface a stop string in auxiliary text should not be
    # reclassified, matching Anthropic's mutually-exclusive
    # ``stop_reason`` semantics.
    stop_sequence: str | None = None
    if stream_matched_stop is not None and stop_reason == "end_turn":
        stop_reason = "stop_sequence"
        stop_sequence = stream_matched_stop

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
        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
        "usage": usage_payload,
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages (stream): prompt={prompt_tokens} + completion={completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
