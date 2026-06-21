# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import gc
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceLogProbs,
    PromptTokensDetails,
    TokenLogProb,
    Usage,
)
from ..api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    parse_json_output,
)
from ..api.utils import (
    clean_output_text,
    decode_inline_tool_call_arguments,
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_USE_REQUIRED_SUFFIX,
    _TOOL_USE_SYSTEM_SUFFIX,
    SSE_RESPONSE_HEADERS,
    _build_usage,
    _check_admission_or_503,
    _disconnect_guard,
    _effective_enable_thinking,
    _extract_streaming_token_logprobs,
    _finalize_content_and_reasoning,
    _inject_json_instruction,
    _is_structured_output_requested,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _release_admission_unless_committed,
    _rescue_silent_drop_from_reasoning,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _scan_messages_for_lone_surrogates,
    _tool_use_required_named_suffix,
    _validate_model_name,
    _validate_response_format,
    _validate_tool_call_params,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    enable_thinking_warning_header,
    enforce_context_length_for_messages,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Exceptions worth catching around the cloud call so the local engine can
# take over: provider/network/auth/quota — transient or out-of-our-control.
# Anything outside this allowlist (AttributeError, TypeError,
# NotImplementedError, …) is an engine-contract violation or programming
# bug and MUST surface as 500. The original ``except Exception`` here hid
# both #500 (missing ``build_prompt``) and the v0.6.70 hotfix (missing
# the token-estimation helper on the engine) as silent fallback warnings.
#
# ``litellm.exceptions`` is imported lazily — its presence depends on
# whether cloud routing was configured at startup. ``httpx`` and the
# stdlib timeout/connection set are always available, so we fall back
# to those when litellm isn't importable.
def _tool_call_name(tc) -> str | None:
    """Extract the function name from a tool_call entry regardless of
    shape. Three real shapes seen in production:

    1. Pydantic ``ToolCall`` — ``tc.function.name``. Text-parser path.
    2. Wrapped dict — ``{"function": {"name": ...}}``. Anthropic
       passthrough and engine structured passthrough through
       ``_parse_tool_calls_with_parser``.
    3. Flat dict — ``{"name": ..., "arguments": ...}``. Raw engine
       ``GenerationOutput.tool_calls`` shape (Harmony StreamableParser
       output before wrapping). Surfaces in tests/fixtures and any
       downstream that forwards engine output directly.

    PR #518 round-2 codex BLOCKING added shapes 1+2; round-3 BLOCKING
    added shape 3 (the round-2 widening missed it, even though the
    same PR's test fixture emits exactly that shape).
    """
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
        if fn is not None:
            return getattr(fn, "name", None)
        # Flat shape — no ``function`` wrapper.
        return tc.get("name")
    fn = getattr(tc, "function", None)
    if isinstance(fn, dict):
        return fn.get("name")
    if fn is not None:
        return getattr(fn, "name", None)
    # Flat attr-shape — no ``function`` attribute.
    return getattr(tc, "name", None)


def _forced_tool_call_prefix(parser_name: str | None, function_name: str) -> str | None:
    """Build the assistant-turn prefix string that the model continues
    when ``tool_choice`` forces a named function.

    Returns ``None`` for parsers where prefix injection isn't a known
    win (e.g. channel-routed harmony / gemma4: those publish tool calls
    via the OutputRouter's tool-call channel directly, so the model
    already produces a structured call when prompted — and pre-pending
    the wire opener would actually CONFUSE the channel state machine).
    For parser-only families (hermes / qwen3coder / llama / kimi /
    glm47 / mistral / minimax / deepseek / nemotron / xlam / functionary)
    the wire opener is unambiguous; insert ``<tool_call>\\n{"name":
    "X", "arguments":`` so the model continues with the arguments object
    and the closer.

    This is the OpenAI ``tool_choice`` forced-function lever — pure
    prefix injection, parser-shape-agnostic of the underlying alias.
    """
    if not function_name:
        return None
    # Verified parsers that explicitly recognise the hermes ``<tool_call>``
    # JSON-body wire shape in their primary path (both non-streaming
    # ``extract_tool_calls`` regex AND the streaming state-machine
    # sentinel set). Each one's source was audited against this opener
    # before being added:
    #   - ``hermes`` (vllm_mlx/tool_parsers/hermes_tool_parser.py):
    #     ``TOOL_CALL_PATTERN = <tool_call>{JSON}</tool_call>``;
    #     ``_STREAMING_SENTINELS = ("<tool_call>", "<function=")``
    #
    # Parsers EXCLUDED on purpose because their primary wire is NOT
    # the JSON ``<tool_call>`` body shape — even when the OPENER
    # matches, the body shape conflicts with the parser's
    # expectations (codex r1 + r3 P2 on this PR):
    #   - ``qwen3coder`` / ``qwen3_coder_xml`` — same ``<tool_call>``
    #     opener but the body uses XML ``<function=NAME>...`` markers,
    #     NOT JSON. ``Qwen3CoderToolParser.extract_tool_calls`` looks
    #     for ``<function=`` after the opener and would miss a JSON
    #     body entirely.
    #   - ``minimax``  → ``<minimax:tool_call>`` / ``<invoke name="...">``
    #   - ``mistral``  → ``[TOOL_CALLS]``
    #   - ``deepseek`` → ``<｜tool▁calls▁begin｜>``
    #   - ``llama``    → ``<|python_tag|>`` / bare JSON (its own opener)
    #   - ``kimi``     → ``<|tool_calls_section_begin|>``
    #   - ``glm47``    → ``<tool_call>...<arg_key>...</arg_value>``
    #     (XML body, NOT JSON — same body-shape conflict as
    #     qwen3coder above)
    #   - ``granite``, ``xlam``, ``functionary``, ``nemotron``,
    #     ``seed_oss`` — distinct wire formats; defer to the
    #     post-parse synthesis fallback rather than risk a wrong
    #     opener.
    _verified_json_tool_call_parsers = {
        "hermes",
    }
    if parser_name in _verified_json_tool_call_parsers:
        # JSON envelope opener — model continues with the arguments
        # object body. Leave a trailing space so the model picks up
        # immediately with ``{...}``.
        #
        # ``json.dumps(function_name)`` escapes any quoted / backslash /
        # control character in the function name so a hostile tool
        # spec (``{"name": "x\\\\\\", \\"arguments\\": ..."}``) cannot
        # corrupt the wire envelope or inject extra fields. Codex r4
        # BLOCKING — direct f-string interpolation was vulnerable.
        return f'<tool_call>\n{{"name": {json.dumps(function_name)}, "arguments": '
    # Channel-routed (harmony / gemma4) and parsers whose wire shape
    # we have NOT audited: no prefix injection. The post-parse
    # synthesis path remains as a fallback (``_synthesize_forced_tool_call``).
    return None


def _synthesize_forced_tool_call(name: str, arguments: str = "{}"):
    """Build a single ``ToolCall`` for a forced ``tool_choice`` whose
    text parser surfaced no calls (#571).

    Text-parser paths (hermes / qwen3_coder / minimax / glm47 / …) only
    surface a tool_call when the model emits the parser's wire markers.
    Channel-routed paths (harmony / gemma4) bypass the text parser
    entirely — the ``OutputRouter`` extracts structured tool_calls
    directly. The two surfaces therefore diverge on the same request:
    a forced ``tool_choice`` succeeds on harmony because the model
    produced the structured channel, but 422s on hermes when the model
    produced text that the parser failed to recognise.

    The OpenAI ``tool_choice`` contract is parser-agnostic: when the
    client forces a tool call, the response MUST carry one. To restore
    symmetry we synthesise a tool_call server-side when the target tool
    is unambiguous (named-function, or ``"required"`` with a single
    tool). Arguments default to ``"{}"`` because we have no signal
    about what the model intended to pass; downstream
    ``_validate_tool_call_params`` logs a warning when required
    parameters are missing, mirroring the diagnostic surface clients
    already see for model-generated calls with bad arguments. The
    contract guarantee is "a tool_call is present", not "the arguments
    are correct".
    """
    # Lazy import — ToolCall / FunctionCall live alongside the request
    # model in ``api.models``. The lazy form keeps the synthesis path
    # scoped to forced-choice requests; the common case pays nothing.
    from ..api.models import FunctionCall, ToolCall

    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


def _engine_supports_channel_routed_tool_calls(engine) -> bool:
    """Probe whether the engine's tokenizer yields a channel-routed
    streaming path that can emit structured tool calls without a text
    parser. Harmony (gpt-oss) and Gemma 4 publish tool calls via the
    OutputRouter's tool-call channel, so a stream=true tool_choice=
    required request CAN satisfy the contract for those models even
    when ``cfg.tool_call_parser`` is unset.

    PR #518 round-10 codex BLOCKING #1: the prior gate rejected every
    parser-less streaming-required request and blocked legitimate
    harmony/gemma4 traffic. The capability probe relies on the same
    detection the engine itself uses
    (``OutputRouter.from_tokenizer_for_streaming`` + the engine's
    format allowlist), so a positive answer here means the actual
    engine path WILL produce structured tool_call deltas.
    """
    # Engine-level capability bit — if an engine explicitly declares
    # it has no tool-call surface (DiffusionEngine), the tokenizer
    # probe is moot. Without this, DiffusionGemma's tokenizer would
    # trip the Gemma 4 allowlist even though DiffusionEngine never
    # runs OutputRouter — letting tool_choice="required" finish with
    # plain text and no 422 (codex round 9 [P2] on PR #551).
    if not getattr(engine, "supports_tool_calls", True):
        return False
    try:
        from ..engine.batched import _OUTPUT_ROUTER_ALLOWLIST
        from ..output_router import OutputRouter

        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is None:
            return False
        router = OutputRouter.from_tokenizer_for_streaming(tokenizer)
        if router is None:
            return False
        return router.map.format_tag in _OUTPUT_ROUTER_ALLOWLIST
    except Exception:
        # Capability probe is best-effort — any failure means we
        # cannot prove channel-routed support, so the gate falls
        # back to the parser-only path (which 422s without one).
        return False


def _cloud_call_recoverable_exceptions() -> tuple[type[BaseException], ...]:
    """Build the allowlist of exception types we treat as recoverable from
    the cloud call. Lazy so cloud routing being disabled doesn't pay the
    litellm import cost.

    Covered failure shapes (codex round-1 review on PR #502 — broaden
    beyond ``httpx.HTTPError`` to catch real production cases):
      * ``asyncio.TimeoutError`` / ``TimeoutError`` — request budget hit
      * ``ConnectionError`` — TCP/UDP transport down
      * ``ssl.SSLError`` — certificate / handshake — common w/ corp MITM
      * ``json.JSONDecodeError`` — provider returned malformed body
      * ``httpx.HTTPError`` — covers ``HTTPStatusError``, ``RequestError``,
        ``ConnectError``, ``ProxyError``, ``ReadTimeout``, etc.
      * ``litellm.exceptions.APIError`` — provider-side surface
    """
    import asyncio
    import json
    import ssl

    exc_types: list[type[BaseException]] = [
        asyncio.TimeoutError,
        ConnectionError,
        TimeoutError,
        ssl.SSLError,
        json.JSONDecodeError,
    ]
    try:
        import httpx

        exc_types.append(httpx.HTTPError)
    except ImportError:
        pass
    try:
        from litellm import exceptions as _litellm_exc

        exc_types.append(_litellm_exc.APIError)
    except (ImportError, AttributeError):
        pass
    return tuple(exc_types)


_CLOUD_CALL_RECOVERABLE_EXCEPTIONS = _cloud_call_recoverable_exceptions()


# Matches a single backslash directly followed by a non-ASCII codepoint.
# ``lm-format-enforcer``'s grammar permits ``\\`` followed by any codepoint
# as a valid JSON escape, so a model emitting JSON with CJK / emoji content
# can produce strings like ``"\\빠\\르\\게"`` — valid JSON, but the decoded
# value carries literal backslashes. Strip them so clients see clean text.
#
# Scope / known tradeoff: this is applied only on the ``response_format``
# json-output path (see line ~632 below), not to tool-call arguments or
# regular text content. The cleanup is unconditional within that path,
# matching upstream waybarrios#525. A JSON object that LEGITIMATELY
# contains a backslash before a non-ASCII codepoint (e.g. a Windows path
# ``"C:\\사용자\\file.txt"`` in a response_format=json_object reply) will
# be mutated to ``"C:사용자file.txt"``. We accept this tradeoff because:
#  (a) the lm-format-enforcer bug is the overwhelming source of these
#      sequences in JSON-output responses; the file-path case is rare,
#  (b) gating the cleanup on a heuristic ("looks like enforcer output")
#      would be fragile and only catch the obvious patterns,
#  (c) clients that need raw backslash + non-ASCII can fall back to
#      ``response_format=text`` and parse the JSON themselves.
# If a user reports the false-positive in practice, revisit by adding a
# config flag (``--no-strip-spurious-backslashes``) rather than a heuristic.
_BACKSLASH_BEFORE_UNICODE = re.compile(r"\\([^\x00-\x7F])")


def _strip_backslash_before_unicode(obj: object) -> object:
    if isinstance(obj, dict):
        # Clean both keys and values: ``lm-format-enforcer`` can produce
        # ``"\\한\\글": "value"`` (valid JSON, ugly key). Stripping only
        # values would leak the bug into client-visible object keys.
        cleaned: dict[object, object] = {}
        for k, v in obj.items():
            new_key = _strip_backslash_before_unicode(k)
            new_val = _strip_backslash_before_unicode(v)
            if new_key in cleaned:
                # Two distinct dirty keys can collapse to the same clean
                # key (e.g. ``"\\한"`` and ``"한"`` both → ``"한"``). Keep
                # the first occurrence and surface the collision rather
                # than silently dropping a field.
                logger.warning(
                    "JSON key collision after backslash strip: %r dropped "
                    "in favor of earlier value (cleaned key=%r)",
                    k,
                    new_key,
                )
                continue
            cleaned[new_key] = new_val
        return cleaned
    if isinstance(obj, list):
        return [_strip_backslash_before_unicode(v) for v in obj]
    if isinstance(obj, str):
        return _BACKSLASH_BEFORE_UNICODE.sub(r"\1", obj)
    return obj


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    _validate_model_name(request.model)
    engine = get_engine(request.model)

    # Admission reservation is acquired LATER — after cloud-routing
    # decision (codex R9: cloud-routable requests must not be 503'd
    # solely because the local engine is at cap; they bypass local
    # generation entirely) and after the cheap validation that may
    # raise HTTPException (codex R3: validation errors used to pin
    # the slot until restart, exhausting the cap via a trivial
    # malformed-JSON DoS). ``_commit_state[0] = True`` is flipped
    # right before returning a StreamingResponse so
    # ``_disconnect_guard`` owns release after the SSE generator
    # closes; the route-level ``finally`` releases for non-streaming
    # and cloud paths.
    _commit_state = [False]
    _admission_acquired = [False]
    try:
        return await _create_chat_completion_impl(
            request, raw_request, engine, _commit_state, _admission_acquired
        )
    finally:
        if _admission_acquired[0]:
            _release_admission_unless_committed(engine, _commit_state[0])


async def _create_chat_completion_impl(
    request: ChatCompletionRequest,
    raw_request: Request,
    engine,
    _commit_state: list[bool],
    _admission_acquired: list[bool],
):
    """Inner impl for ``create_chat_completion``. Admission is
    reserved inside this function — after cloud-routing decision
    and after cheap validation — to avoid (a) 503'ing
    cloud-routable requests when the local engine is full and
    (b) leaking the slot on validation HTTPException paths."""
    # Validate messages is non-empty
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    # Validate message roles
    _valid_roles = {"system", "user", "assistant", "tool", "developer"}
    for msg in request.messages:
        if msg.role not in _valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be one of: {', '.join(sorted(_valid_roles))}",
            )

    # Reject lone-surrogate codepoints in any message-content slot
    # (F-130 + F-131). ``json.loads`` accepts ``"\\uD800"`` as a valid
    # JSON string and binds it to a Python ``str`` carrying the
    # unpaired surrogate; HuggingFace ``tokenizers`` then raises
    # ``TypeError: TextEncodeInput must be …`` deep inside the
    # chat-template render, producing either a 500 (non-stream) or —
    # WORSE — an HTTP 200 with the raw Python error text leaked in an
    # SSE ``data:`` chunk (stream). The route-layer gate runs BEFORE
    # the streaming branch opens its ``StreamingResponse``, so the
    # SSE-leak path in F-131 is closed by construction (a 400 is
    # returned before any byte of SSE is flushed).
    _scan_messages_for_lone_surrogates(request.messages)

    # F-111 / F-112 / F-051: tool-message schema validation.
    #
    # The OpenAI ``chat.completions`` spec defines three invariants on
    # ``role:"tool"`` messages that we previously accepted as 200 and
    # silently mis-rendered into the model prompt:
    #
    #   * F-051: ``tool_call_id`` is REQUIRED on every tool message.
    #     Without it, the tool reply has no provenance — the previously
    #     accepted shape rendered the result into context unlinked
    #     (effectively an attacker-controlled "extra user turn").
    #
    #   * F-112: ``tool_call_id`` MUST reference the ``id`` of a
    #     ``tool_calls[*]`` entry on a PRIOR assistant message in the
    #     same ``messages[]`` array. Orphan tool turns were accepted as
    #     200 and rendered into the prompt as if they were authoritative
    #     tool replies — a direct prompt-injection vector (e.g. an
    #     attacker-controlled ``content: "the password is OMEGA"``
    #     reached the model with no anchoring assistant tool_call).
    #
    #   * F-111: ``role:"tool"`` content MUST be text-only (a string or
    #     a ``[{type:"text",text:str}]`` array). The OpenAI spec does
    #     not define multimodal tool replies, and a non-text part on a
    #     tool reply was previously silently dropped by every text-only
    #     chat template — the model received an empty
    #     ``<tool_response>`` and hallucinated. The text-only array is
    #     accepted here and flattened to a string downstream by
    #     ``utils/chat_template.py::_normalize_text_only_content_arrays``;
    #     this validator only rejects the non-text shapes the
    #     normalization can't safely flatten.
    #
    # All three checks live up-front (BEFORE engine work) so the failure
    # mode is a clean 400 with a precise pointer at the offending
    # message — not a 500 from a downstream renderer crash. F-051 is
    # closed as a freebie by the ``tool_call_id`` REQUIRED check.
    #
    # We track tool_call ids as a CONSUMABLE pending set rather than a
    # monotonically-growing seen-set: each ``assistant.tool_calls[*].id``
    # is a single-use ticket that the next matching ``role:"tool"`` reply
    # consumes. Without this, a client can submit a valid tool reply,
    # then resubmit ANOTHER ``role:"tool"`` message with the SAME
    # ``tool_call_id`` later in ``messages[]`` — both replies render as
    # authoritative tool results for the same call, an attacker-
    # controlled "second reply" prompt-injection vector. Codex round-1
    # BLOCKING on PR #731.
    #
    # We also reject DUPLICATE ``assistant.tool_calls[*].id`` values —
    # whether within a single assistant turn or across turns. The
    # OpenAI contract guarantees ids are unique across the conversation,
    # and the consumable-ticket design relies on it: a re-used id on a
    # later assistant turn would silently re-open a ticket the client
    # never asked for, weakening the F-112 single-use guarantee. Codex
    # round-2 NIT on PR #731 — implement the invariant the docstring
    # claims rather than weaken the docstring.
    _pending_tool_call_ids: set[str] = set()
    _seen_any_tool_call_id: set[str] = set()
    for _idx, _msg in enumerate(request.messages):
        if _msg.role == "assistant" and _msg.tool_calls:
            for _tc in _msg.tool_calls:
                if isinstance(_tc, dict):
                    _tc_id = _tc.get("id")
                else:
                    _tc_id = getattr(_tc, "id", None)
                if isinstance(_tc_id, str) and _tc_id:
                    if _tc_id in _seen_any_tool_call_id:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"messages[{_idx}] assistant tool_calls "
                                f"contains duplicate id {_tc_id!r}; the OpenAI "
                                "spec requires every tool_call.id to be unique "
                                "across the conversation"
                            ),
                        )
                    _seen_any_tool_call_id.add(_tc_id)
                    _pending_tool_call_ids.add(_tc_id)
            continue
        if _msg.role != "tool":
            continue
        # F-051: tool_call_id REQUIRED.
        if not _msg.tool_call_id or not isinstance(_msg.tool_call_id, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"messages[{_idx}] with role 'tool' must have a non-empty "
                    "'tool_call_id' string"
                ),
            )
        # F-112: tool_call_id must reference a prior assistant tool_call.id
        # that has NOT already been consumed by an earlier tool reply.
        if _msg.tool_call_id not in _pending_tool_call_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"messages[{_idx}] tool_call_id {_msg.tool_call_id!r} does "
                    "not reference any prior assistant tool_call"
                ),
            )
        # Consume the ticket. A later duplicate ``role:"tool"`` reply
        # with the same id will fall through to the F-112 branch above
        # and be rejected.
        _pending_tool_call_ids.discard(_msg.tool_call_id)
        # F-111: tool content shape. Accept str, None, or text-only
        # array. Anything else is a non-text part (image/video/audio)
        # which would be silently dropped by the renderer. An EMPTY
        # ``list`` (``content: []``) is also rejected — the chat-template
        # normalizer in ``utils/chat_template.py`` does not flatten
        # empty lists (an empty array isn't a "text-only" array; there's
        # no text to extract), so accepting it here would leak a
        # non-string ``content`` into the rendered prompt and crash the
        # template. Clients that intend an empty tool reply must send
        # ``content: ""`` or ``content: null`` (codex round-1 BLOCKING
        # on PR #731).
        _content = _msg.content
        if _content is None or isinstance(_content, str):
            continue
        if isinstance(_content, list):
            if not _content:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"messages[{_idx}] role 'tool' content must not be an "
                        "empty list; send an empty string '' or null for "
                        "an empty tool reply"
                    ),
                )
            _bad = False
            for _part in _content:
                if hasattr(_part, "model_dump"):
                    _part_d = _part.model_dump(exclude_none=True)
                elif isinstance(_part, dict):
                    _part_d = _part
                else:
                    _bad = True
                    break
                if _part_d.get("type") != "text" or not isinstance(
                    _part_d.get("text"), str
                ):
                    _bad = True
                    break
            if _bad:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"messages[{_idx}] role 'tool' content must be a "
                        "string or a text-only array of "
                        "{type:'text', text:str} parts"
                    ),
                )
            continue
        # Anything else (int / dict / bool / ...) is not a valid OpenAI
        # tool content shape.
        raise HTTPException(
            status_code=400,
            detail=(
                f"messages[{_idx}] role 'tool' content must be a string or a "
                f"text-only content-parts array, not {type(_content).__name__}"
            ),
        )

    # Validate n parameter (only n=1 supported)
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail="n > 1 is not supported. Rapid-MLX generates one completion per request.",
        )

    # Validate max_tokens. Lower bound: must be positive. Upper bound: a
    # hard sanity ceiling so a buggy client passing 999_999_999 cannot
    # combine with unbounded admission to OOM the Metal allocator.
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )
    if request.max_tokens is not None and request.max_tokens > 1_000_000:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at most 1000000",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_p range (OpenAI spec: (0, 1]). Without this, top_p=2.0
    # is silently accepted while sister field `temperature` is checked,
    # so clients with a bug see no signal.
    if request.top_p is not None and (request.top_p <= 0 or request.top_p > 1):
        raise HTTPException(
            status_code=400,
            detail="top_p must be in (0, 1]",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
        )

    # Reject non-empty logit_bias with a clear 400 rather than silently
    # dropping it. We accept {} so defensive clients that always include
    # the field don't break.
    if request.logit_bias:
        raise HTTPException(
            status_code=400,
            detail="logit_bias is not supported on this server",
        )

    # Validate ``response_format`` shape BEFORE
    # ``build_json_system_prompt`` is reached (F-013). Two bugs the gate
    # closes: (a) ``type:"json_schema"`` with no ``json_schema`` field
    # used to leak ``AttributeError: 'NoneType' object has no attribute
    # 'get'`` in the 400 body via the broad ``except Exception`` at
    # the call site; (b) unknown ``type`` values (``"xml"``,
    # ``""``, ``{}``, ``type:"json_schema"`` with empty
    # ``json_schema:{}``) were silently accepted as HTTP 200 with no
    # structure enforcement — client received unconstrained prose
    # without any signal.
    _validate_response_format(request.response_format)

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

    cfg = get_config()

    # Enforce ``tool_choice`` at the prompt level (#445). The OpenAI spec
    # accepts four modes: "auto", "none", "required", and
    # ``{"type":"function","function":{"name":X}}``. Local inference has no
    # native enforcement (no FSM constraint), so the only reliable lever for
    # ``"none"`` and the specific-function form is to mutate what the model
    # sees: drop ``tools`` entirely for ``"none"``, or filter to just the
    # named function for the specific case. ``"auto"`` and ``"required"``
    # leave tools untouched — ``"required"`` enforcement is tracked
    # separately under #442 (needs decoder-level constraints, PR #132).
    tc = request.tool_choice
    if tc is not None:
        # Validation runs even when ``tools`` is empty/None: the OpenAI spec
        # treats ``tool_choice`` with a specific function but no matching
        # ``tools`` entry as a malformed request (400), not a silent
        # fall-through. Codex round-1 review of #446 flagged the previous
        # guard ``if tc is not None and request.tools:`` as silently
        # accepting these requests.
        if isinstance(tc, dict) and tc.get("type") == "function":
            fn = tc.get("function") or {}
            target = fn.get("name")
            if not target:
                raise HTTPException(
                    status_code=400,
                    detail=("tool_choice with type='function' requires function.name"),
                )
            if not request.tools:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"tool_choice references function {target!r} but the "
                        "request has no 'tools' array"
                    ),
                )
            filtered = [t for t in request.tools if t.function.get("name") == target]
            if not filtered:
                # F-145: surface a case-insensitive match as a hint when
                # one exists, so clients see "did you mean 'get_Weather'?"
                # instead of having to diff their tool list character-by-
                # character. OpenAI's API is case-sensitive too, but its
                # error message is equally terse — the rapid-mlx hint is
                # additive and OpenAI-shape-compatible (clients that ignore
                # the suffix still see the canonical 400).
                hint = ""
                target_lower = target.lower() if isinstance(target, str) else ""
                if target_lower:
                    case_matches = [
                        name
                        for t in request.tools
                        if isinstance((name := t.function.get("name")), str)
                        and name.lower() == target_lower
                    ]
                    if case_matches:
                        # ``case_matches[0]`` is deterministic — Pydantic
                        # preserves request order on ``request.tools`` so
                        # the hint points at the first matching definition.
                        # We only ever surface ONE hint; if the request
                        # somehow contains multiple case-variants of the
                        # same name (duplicate tool names are silently
                        # accepted today, see F-144), the first wins —
                        # which is the same tool the prompt-time
                        # filter would have picked anyway.
                        hint = f" (did you mean {case_matches[0]!r}? tool_choice is case-sensitive)"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"tool_choice references function {target!r} which "
                        f"is not present in the 'tools' array{hint}"
                    ),
                )
            request.tools = filtered
        elif tc == "none" and request.tools:
            request.tools = None

    # Save original messages (clean dicts) for cloud routing BEFORE
    # local mutations (extract_multimodal_content, developer→system, suffix injection).
    if cfg.cloud_router:
        _cloud_original_messages = [
            (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else {k: v for k, v in dict(msg).items() if v is not None}
            )
            for msg in request.messages
        ]
    else:
        _cloud_original_messages = None

    # Reject image/video/audio content when the loaded model has no
    # multimodal head. Without this guard ``extract_multimodal_content``
    # silently drops the media parts on the text-only path and the model
    # hallucinates (R9P1: 600M text model returned "a red rose" for
    # arbitrary images; iter12 onboarding: text-only model claimed
    # "no audio attached" while silently dropping ``audio_url``).
    if not engine.is_mllm:
        for _msg in request.messages:
            _content = (
                _msg.content if hasattr(_msg, "content") else _msg.get("content", "")
            )
            if isinstance(_content, list):
                for _item in _content:
                    _item_type = (
                        _item.type
                        if hasattr(_item, "type")
                        else (_item.get("type", "") if isinstance(_item, dict) else "")
                    )
                    if _item_type in (
                        "image_url",
                        "image",
                        "video",
                        "video_url",
                        "audio_url",
                        "audio",
                        "input_audio",
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Model '{cfg.model_name}' does not support "
                                "image, video, or audio inputs."
                            ),
                        )

    # For MLLM models, keep original messages with embedded images
    if engine.is_mllm:
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []
        # The non-MLLM branch decodes tool_call.function.arguments from JSON
        # string to dict inside extract_multimodal_content() so chat templates
        # that iterate args via .items() (e.g. GLM-4.6V) don't crash. The
        # MLLM branch bypasses that helper, so call the shared decoder here.
        if engine.preserve_native_tool_format:
            decode_inline_tool_call_arguments(messages)
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Normalize "developer" role to "system"
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "developer":
            if isinstance(m, dict):
                messages[i]["role"] = "system"
            else:
                m.role = "system"

    # Auto-inject system prompt suffix for tool use and/or reasoning control.
    # ``tool_choice="required"`` (and the specific-function form) gets a
    # stricter suffix than the default tool-use one — the OpenAI spec
    # guarantees a tool_call when ``required`` is set, but local inference
    # has no decoder-level enforcement (FSM constraint tracked in #132).
    # Prompt injection + post-parse 422 are the strongest levers we have
    # (#468). Strictness shape: explicit ``required`` > named function >
    # the default ``auto``/unset suffix.
    _inject_suffix = None
    if request.tools and cfg.tool_call_parser:
        if tc == "required":
            _inject_suffix = _TOOL_USE_REQUIRED_SUFFIX
        elif isinstance(tc, dict) and tc.get("type") == "function":
            _named = (tc.get("function") or {}).get("name")
            if _named:
                _inject_suffix = _tool_use_required_named_suffix(_named)
            else:
                _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
        else:
            _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
    elif cfg.reasoning_parser_name == "minimax":
        _inject_suffix = (
            "\n\nDo NOT think out loud or show your reasoning process. "
            "Give direct answers only — no preamble like 'The user asks...' or "
            "'We should respond...' or 'Let me think...'. Be concise."
        )

    if _inject_suffix:
        has_system = any(
            (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "system"
            for m in messages
        )
        if has_system:
            for i, m in enumerate(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "system":
                    if isinstance(m, dict):
                        messages[i] = {**m, "content": m["content"] + _inject_suffix}
                    else:
                        messages[i]["content"] = m["content"] + _inject_suffix
                    break
        else:
            system_msg = {"role": "system", "content": _inject_suffix.strip()}
            messages = [system_msg] + list(messages)

    # Auto-pin system prompt prefix cache blocks
    if cfg.pin_system_prompt:
        _maybe_pin_system_prompt(messages)

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        try:
            json_instruction = build_json_system_prompt(response_format)
        except Exception as e:
            logger.warning(f"Failed to build JSON system prompt: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format schema: {e}",
            )
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Resolve enable_thinking once and reuse — drives both the
    # max_tokens default (thinking models need more headroom) and the
    # chat_template kwarg below. (#387)
    resolved_thinking = _resolve_enable_thinking(request)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_tokens, resolved_thinking),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }

    # Extended sampling params — resolve through the request → CLI →
    # alias → generation_config cascade. Only forwards values the
    # cascade actually produced.
    chat_kwargs.update(build_extended_sampling_kwargs(request))

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # OpenAI ``tool_choice`` forced-function — assistant-turn prefix
    # injection. The chat-template renderer (and the engine's
    # ``chat()``/``stream_chat()``) accept ``forced_assistant_prefix``;
    # when set, the rendered prompt is suffixed with the parser's wire-
    # envelope opener and the model continues from inside the tool
    # call. The text parser then recovers the call in the normal flow.
    # No per-model regex, no per-alias config — the prefix is derived
    # solely from ``cfg.tool_call_parser`` and the requested function
    # name. See ``_forced_tool_call_prefix`` for the parser-shape
    # taxonomy.
    _forced_prefix = None
    if request.tools and request.tool_choice is not None:
        _forced_name: str | None = None
        if (
            isinstance(request.tool_choice, dict)
            and request.tool_choice.get("type") == "function"
        ):
            _forced_name = (request.tool_choice.get("function") or {}).get("name")
        elif request.tool_choice == "required" and len(request.tools) == 1:
            # OpenAI spec: ``required`` with a single tool is unambiguous
            # — same forcing semantics as a named choice.
            _forced_name = request.tools[0].function.get("name")
        if _forced_name:
            _forced_prefix = _forced_tool_call_prefix(
                cfg.tool_call_parser, _forced_name
            )
    if _forced_prefix:
        chat_kwargs["forced_assistant_prefix"] = _forced_prefix

    # PFlash routing (#287): structured-output prompts are
    # prompt-integrity-sensitive — lossy compression would corrupt the
    # JSON schema context, and there is no user-facing opt-out for
    # structured output, so they stay hard-protected here.
    #
    # Tools used to be lumped in with structured output but have a
    # separate user-facing knob — ``PFlashConfig.skip_when_tools``
    # (default skip; CLI ``--pflash-include-tools`` inverts it). The
    # gate flows through ``has_tools`` instead. Force-setting
    # ``requires_prompt_integrity=True`` for tools here short-circuits
    # the ``skip_when_tools`` branch and made the documented CLI
    # opt-in dead (codex r6 BLOCKING).
    if response_format:
        chat_kwargs["requires_prompt_integrity"] = True

    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    # Context-length pre-check (DoS defense + UX, rapid-desktop#273 / #463).
    # See ``service/helpers.py::enforce_context_length_for_messages`` for
    # the rationale (8 MiB body still holds ~2M tokens → context window
    # blown → ~60–90 s of wasted prefill before client gives up). Same
    # gate runs in routes/completions, routes/anthropic, routes/responses.
    enforce_context_length_for_messages(
        engine,
        messages,
        tools=request.tools,
        max_tokens=chat_kwargs.get("max_tokens"),
    )

    # Cloud routing: offload large-context requests to cloud LLM.
    #
    # The token-budget computation (``build_prompt`` + ``estimate_new_
    # tokens``) is part of the BaseEngine contract — any exception there
    # is a real bug and must surface, NOT be silently swallowed as
    # "falling back to local". The two regressions this scope-narrowing
    # closes (#500 + the v0.6.70 hotfix) both hid behind a broad
    # ``except Exception`` that turned engine-contract violations into
    # warning logs while cloud routing silently never fired.
    #
    # Only the cloud call itself is wrapped, and only "expected,
    # transient" failure shapes (network, auth, provider) are caught.
    if cfg.cloud_router and not engine.is_mllm:
        prompt = engine.build_prompt(messages, tools=request.tools)
        total_tokens, new_tokens = engine.estimate_new_tokens(prompt)
        if cfg.cloud_router.should_route_to_cloud(new_tokens):
            logger.info(
                f"[CLOUD ROUTE] {new_tokens} new tokens (total {total_tokens}) "
                f"> threshold {cfg.cloud_router.threshold}, "
                f"routing to {cfg.cloud_router.cloud_model}"
            )
            cloud_messages = _cloud_original_messages
            cloud_kwargs = {
                "temperature": chat_kwargs.get("temperature"),
                "max_tokens": chat_kwargs.get("max_tokens"),
                "top_p": chat_kwargs.get("top_p"),
            }
            if request.stop:
                cloud_kwargs["stop"] = request.stop
            if request.tool_choice is not None:
                cloud_kwargs["tool_choice"] = request.tool_choice
            if request.response_format:
                rf = request.response_format
                cloud_kwargs["response_format"] = (
                    rf.model_dump() if hasattr(rf, "model_dump") else rf
                )
            if request.tools:
                cloud_kwargs["tools"] = [
                    t.model_dump() if hasattr(t, "model_dump") else t
                    for t in request.tools
                ]
            # Cloud-routed request: the local scheduler/Metal path
            # is bypassed entirely, so admission is not acquired
            # for cloud paths. The wrapper's ``finally`` checks
            # ``_admission_acquired[0]`` (still False here) and
            # skips the release. Without this ordering (admission
            # check moved BELOW the cloud routing block), a burst
            # of local requests filling the cap would 503
            # cloud-routable requests that never touch the local
            # engine (codex R9).
            try:
                if request.stream:
                    return StreamingResponse(
                        _disconnect_guard(
                            cfg.cloud_router.stream_completion(
                                cloud_messages,
                                model_name=cfg.model_name or "cloud",
                                **cloud_kwargs,
                            ),
                            raw_request,
                        ),
                        media_type="text/event-stream",
                        headers=SSE_RESPONSE_HEADERS,
                    )
                else:
                    result = await _wait_with_disconnect(
                        cfg.cloud_router.completion(cloud_messages, **cloud_kwargs),
                        raw_request,
                        timeout=request.timeout or cfg.default_timeout,
                    )
                    if result is None:
                        return Response(status_code=499, content="Client disconnected")
                    # NOTE: L-05's enable_thinking warning intentionally
                    # does NOT fire on the cloud-routed path — the local
                    # ``cfg.reasoning_parser_name`` isn't authoritative
                    # for what the cloud provider does with the ctk
                    # hint. A warning here would be misleading.
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                    )
            except _CLOUD_CALL_RECOVERABLE_EXCEPTIONS as e:
                # Provider/network failures are transient and the local
                # engine is a reasonable fallback. Engine-contract
                # violations (AttributeError, TypeError, …) are NOT in
                # this allowlist on purpose — they must surface as 500.
                logger.warning(
                    f"[CLOUD ROUTE] Cloud call failed ({type(e).__name__}: {e}), "
                    "falling back to local"
                )
        else:
            logger.info(
                f"[LOCAL] {new_tokens} new tokens (total {total_tokens}) "
                f"<= threshold {cfg.cloud_router.threshold}, using local inference"
            )

    # ``tool_choice="required"`` + ``stream=true`` is enforceable IF the
    # engine has SOME path to produce a streaming tool_call:
    #   (a) a text-parser path — ``cfg.tool_call_parser`` set; or
    #   (b) a channel-routed path — harmony (gpt-oss) / Gemma 4 emit
    #       structured tool_calls via the OutputRouter's tool channel
    #       without needing a text parser.
    # The request can satisfy the contract iff EITHER path is available.
    # When neither is available we have NO mechanism at all, so reject
    # upfront with a clear error.
    #
    # Round-7 codex BLOCKING surfaced the silent text-only finish_reason
    # case; round-8 moved the guard below cloud routing; round-9 narrowed
    # to the truly-unenforceable case (no parser); round-10 codex BLOCKING
    # #1 widened "enforceable" to include channel-routed capability so
    # harmony/gemma4 streaming requests aren't blocked by the gate.
    # Engine-level veto — even with ``--tool-call-parser`` set, an
    # engine that has explicitly opted out of tool-call surfaces
    # (``supports_tool_calls=False``) cannot emit structured tool
    # calls because its generator never produces them in the first
    # place. The text parser would only match against the engine's
    # actual ``channel="content"`` output, which has no tool call
    # markers, so streaming would finish with plain text and the
    # contract would silently break. Reject upfront with the same
    # 422 the parser-less path uses (codex round 10 [P2] on PR #551).
    # Use the same falsey predicate (``not getattr(...)``) as
    # ``_engine_supports_channel_routed_tool_calls`` so the two
    # checks treat None / 0 / False uniformly as "engine has opted
    # out" — pr_validate codex r12 NIT. Default True (existing
    # engines) preserves prior behaviour for everything that hasn't
    # opted out.
    _engine_opts_out_of_tools = not getattr(engine, "supports_tool_calls", True)
    # Engine-level veto applies REGARDLESS of stream / non-stream
    # AND for every forced tool-choice shape (codex pr_validate r8
    # NIT #2). The OpenAI ``tool_choice`` API has two forced
    # variants beyond ``"required"``:
    #   - the named-function form ``{"type":"function",
    #     "function":{"name":"foo"}}`` — caller demands a specific
    #     tool gets called
    #   - and the deprecated ``"function"`` literal string (some
    #     legacy SDKs still send it)
    # All three are contracts an opted-out engine cannot satisfy
    # because the generator never produces structured tool_calls.
    # Pre-pr_validate r6, this check was nested inside the
    # ``request.stream`` branch below, so a non-streaming forced
    # request still ran a full diffusion generation before failing
    # in the post-parse gate at line ~1101. That is wasted GPU +
    # ambiguous client UX. Reject upfront for opted-out engines no
    # matter the stream flag (codex pr_validate r6 BLOCKING #1 +
    # r8 NIT #2 on PR #551).
    _forced_tool_choice = (
        tc == "required"
        # Legacy literal — some pre-2024 OpenAI SDKs sent the bare
        # string ``"function"`` to mean "force any function call"
        # before the dict form was added. Codex pr_validate r9 NIT
        # #1 flagged the original predicate omitted this shape so
        # opted-out engines would still run a full generation
        # before failing.
        or tc == "function"
        or (isinstance(tc, dict) and tc.get("type") == "function")
    )
    if _forced_tool_choice and request.tools and _engine_opts_out_of_tools:
        raise HTTPException(
            status_code=422,
            detail=(
                "tool_choice forces a tool call, but the active engine "
                "has explicitly opted out of tool-call surfaces "
                "(supports_tool_calls=False). The generator never emits "
                "structured tool_calls, so any forced choice — "
                '``"required"`` or a named ``{"type":"function","function":'
                '{"name":...}}`` — is unenforceable. Drop tool_choice (or '
                'set it to ``"auto"``/``"none"``), retry against an engine '
                "that supports tool calls, or remove the ``tools`` array "
                "from the request."
            ),
        )
    if (
        request.stream
        and tc == "required"
        and request.tools
        and not cfg.tool_call_parser
        and not _engine_supports_channel_routed_tool_calls(engine)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                'tool_choice="required" with stream=true requires either a '
                "streaming tool-call parser (--tool-call-parser) or a "
                "channel-routed model (harmony / Gemma 4) so the server has "
                "a path to emit structured tool_calls. Neither is available "
                "for this request — the OpenAI 'tool_call guaranteed' "
                "contract cannot be met. Either set --tool-call-parser=hermes "
                "(or your model's parser), retry with stream=false "
                "(non-stream path 422s text-only output), or pin a specific "
                'function via tool_choice={"type":"function",'
                '"function":{"name":...}}.'
            ),
        )

    # Local-path admission gate: reserve a slot before kicking the
    # engine. Placed AFTER cloud routing so cloud-routable requests
    # don't 503 just because the local cap is full (codex R9), and
    # AFTER the cheap validation above so a malformed request can't
    # pin a slot until restart (codex R3). The wrapper's ``finally``
    # uses ``_admission_acquired`` to decide whether to release.
    _check_admission_or_503(engine)
    _admission_acquired[0] = True

    # Detect guided generation BEFORE the stream/non-stream split so the
    # streaming branch can also route json_schema requests through the
    # constrained path. Pre-fix, only the non-stream branch consulted
    # ``supports_guided_generation`` and stream=true silently bypassed
    # ``GuidedGenerator`` — the model would emit unconstrained tokens
    # (e.g. a ```json ... ``` markdown fence) even with a json_schema
    # response_format set. Surfaced by Gap #2 of the v0.6.60 onboarding
    # sweep; mirrors the constraint-then-emit pattern from upstream
    # waybarrios#548.
    use_guided = False
    json_schema = None
    if response_format and not request.tools:
        json_schema = extract_json_schema_for_guided(response_format)
        if json_schema:
            # ``supports_guided_generation`` and ``generate_with_schema``
            # are on the BaseEngine contract — defaults are False /
            # NotImplementedError, so engines without guided decoding
            # opt out by leaving the property at False. The previous
            # ``hasattr`` guards were artifacts of the engine API being
            # informal; they're the same silent-skip shape that produced
            # #500 and the v0.6.70 hotfix and have no role now that the
            # contract is explicit.
            use_guided = engine.supports_guided_generation
            if use_guided:
                logger.info("Using guided generation for JSON schema enforcement")
            else:
                # Surface the silent-degradation case: client asked for
                # json_schema strict mode but the engine can't enforce it
                # (most commonly: the user installed `rapid-mlx` without
                # the `[guided]` extra, so outlines is unavailable). The
                # request will still be served with unconstrained
                # decoding, but the schema contract is NOT being honored
                # — without this warning, the client sees garbage
                # (e.g. the model thinks for max_tokens and never emits
                # JSON) with no diagnostic signal. v0.6.63 onboarding
                # sweep finding #5.
                logger.warning(
                    "json_schema response_format requested but guided "
                    "generation is unavailable (engine="
                    "%s.supports_guided_generation=False). Falling back "
                    "to unconstrained decoding — schema will NOT be "
                    "enforced. Install with `pip install "
                    "'rapid-mlx[guided]'` to enable outlines-backed "
                    "schema enforcement.",
                    type(engine).__name__,
                )

    if request.stream:
        # Validate chat template eagerly so template errors return 400
        if not engine.is_mllm:
            try:
                engine.build_prompt(
                    messages,
                    tools=chat_kwargs.get("tools"),
                    enable_thinking=chat_kwargs.get("enable_thinking"),
                )
            except Exception as e:
                err_msg = str(e)
                err_type = type(e).__name__
                if (
                    "TemplateError" in err_type
                    or "template" in err_msg.lower()
                    or ("user" in err_msg.lower() and "found" in err_msg.lower())
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Chat template error: {err_msg}",
                    )
                raise
        _commit_state[0] = True
        # L-05: surface silent ``enable_thinking`` drop on non-Qwen
        # parsers via response headers. Merging here lets the SSE
        # ``Cache-Control`` / ``Connection`` headers stay intact.
        _thinking_warning = enable_thinking_warning_header(
            request, getattr(cfg, "reasoning_parser_name", None)
        )
        _sse_headers = (
            {**SSE_RESPONSE_HEADERS, **_thinking_warning}
            if _thinking_warning
            else SSE_RESPONSE_HEADERS
        )
        # C-01: holder list the engine writes the admitted scheduler
        # request id into. ``_disconnect_guard`` reads the SAME list
        # and force-calls ``scheduler.abort_request`` on client
        # disconnect, closing the Astrid r3 hang where the
        # generator-close cascade alone took ~35s to actually free
        # the GPU once the client TCP-RST'd.
        request_id_holder: list[str | None] = [None]
        chat_kwargs["request_id_holder"] = request_id_holder
        if use_guided and json_schema:
            # Constrained streaming: run guided generation buffered, then
            # synthesize an SSE stream from the buffered output. Falls
            # back to the unconstrained streaming helper on guided
            # failure (logged), matching the non-streaming fallback.
            return StreamingResponse(
                _disconnect_guard(
                    stream_chat_completion_guided(
                        engine, messages, request, json_schema, **chat_kwargs
                    ),
                    raw_request,
                    engine=engine,
                    request_id_holder=request_id_holder,
                ),
                media_type="text/event-stream",
                headers=_sse_headers,
            )
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                raw_request,
                engine=engine,
                request_id_holder=request_id_holder,
            ),
            media_type="text/event-stream",
            headers=_sse_headers,
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or cfg.default_timeout

    # Disable GC during generation to avoid latency spikes
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    # Determine if we need per-token logprobs
    want_logprobs = request.logprobs and request.top_logprobs
    top_k_logprobs = request.top_logprobs or 0
    token_logprobs_list: list[TokenLogProb] = []

    try:
        if want_logprobs and not use_guided:
            # ``logprobs`` requests need per-token data, so we route through
            # the streaming path even for a non-stream response. The streaming
            # iterator yields per-token outputs; on channel-routed models
            # (harmony/gpt-oss, gemma4) each chunk also carries the channel
            # the router assigned that token. Accumulate text by channel so
            # ``reasoning_text`` and ``text`` reach
            # ``_finalize_content_and_reasoning`` already split — without
            # this, the loop kept only the LAST chunk's text and ``output.
            # reasoning_text`` stayed empty, so the route fell back to the
            # text-regex parser which leaks analysis-channel content into
            # ``content`` on harmony (same shape as #442 but for the logprobs
            # path).
            from dataclasses import replace as _dc_replace

            output = None
            routed_content_parts: list[str] = []
            routed_reasoning_parts: list[str] = []
            saw_channel = False
            async for chunk in engine.stream_chat(messages=messages, **chat_kwargs):
                output = chunk
                token_logprobs_list.extend(
                    _extract_streaming_token_logprobs(
                        chunk, engine.tokenizer, top_k_logprobs
                    )
                )
                ch = getattr(chunk, "channel", None)
                if ch:
                    saw_channel = True
                    if ch == "reasoning":
                        routed_reasoning_parts.append(chunk.new_text or "")
                    elif ch == "content":
                        routed_content_parts.append(chunk.new_text or "")
                    # ``tool_call`` channel is parsed downstream by
                    # ``_parse_tool_calls_with_parser``; don't fold its body
                    # into either text bucket here.
            if output is None:
                return Response(status_code=499)
            if saw_channel:
                output = _dc_replace(
                    output,
                    text="".join(routed_content_parts),
                    reasoning_text="".join(routed_reasoning_parts),
                )
            # Forced-tool prefix on the logprobs path: the stream
            # yields a synthetic first chunk carrying the prefix
            # bytes for the SSE postprocessor, but THIS internal
            # consumer only retains the last chunk. Fold the prefix
            # into ``output.text`` so the downstream tool parser sees
            # the full ``<tool_call>{"name":...,"arguments":...}``
            # envelope and recovers the model-generated arguments
            # instead of falling through to ``_synthesize_forced_tool_call``'s
            # empty-args default (codex r1 P2 on this PR).
            _forced_prefix_value = chat_kwargs.get("forced_assistant_prefix")
            if (
                _forced_prefix_value
                and output is not None
                and not (output.text or "").startswith(_forced_prefix_value)
            ):
                output = _dc_replace(
                    output,
                    text=_forced_prefix_value + (output.text or ""),
                )
        elif use_guided and json_schema:
            try:
                output = await _wait_with_disconnect(
                    engine.generate_with_schema(
                        messages=messages,
                        json_schema=json_schema,
                        **chat_kwargs,
                    ),
                    raw_request,
                    timeout=timeout,
                )
            except Exception as guided_err:
                logger.warning(
                    f"Guided generation failed, falling back to standard: {guided_err}"
                )
                logger.debug(f"Problematic schema: {json_schema}")
                # Fallback runs under the outer admission reservation
                # still held by the wrapper's ``finally`` — no
                # re-acquire needed (the helper does not release on
                # its own now that release lives at the route level).
                output = await _wait_with_disconnect(
                    engine.chat(messages=messages, **chat_kwargs),
                    raw_request,
                    timeout=timeout,
                )
        else:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                raw_request,
                timeout=timeout,
            )
    except HTTPException:
        raise
    except Exception as e:
        from ..request import InferenceAbortedError

        err_msg = str(e)
        err_type = type(e).__name__
        if isinstance(e, InferenceAbortedError):
            # Engine aborted the request (e.g. Metal runtime error caught
            # in the engine loop). 503 — the server is still up and a
            # smaller request may succeed (#353).
            raise HTTPException(status_code=503, detail=err_msg)
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {err_msg}"
            )
        # Image / video fetch failures surface from multimodal_processor
        # (and models/mllm.py:_prepare_images) as ValueError with a
        # "Failed to process image|video" prefix. Convert to 400 so VLM
        # clients get a clear error instead of a 200 with empty completion
        # (#457).
        #
        # The "exceeds the per-batch cap" marker comes from
        # ``mllm_batch_generator._process_prompts`` when vision + text
        # tokens exceed the configured cap. The MLLM scheduler now flags
        # this as a client-actionable error (#682); without the explicit
        # 400 mapping the engine would still return ``HTTPException``-less
        # 500. Surface as 400 so Desktop / curl clients see the actionable
        # message ("downscale image / raise --prefill-step-size") instead
        # of a generic server error.
        if (
            "Failed to process image" in err_msg
            or "Failed to process video" in err_msg
            or "exceeds the per-batch cap" in err_msg
        ):
            raise HTTPException(status_code=400, detail=err_msg)
        raise
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()

    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls from output using configured parser.
    # ``output.tool_calls`` is non-None when the engine's
    # ``OutputRouter`` already produced structured ``[{"name",
    # "arguments"}]`` entries (currently HarmonyStreamingRouter via
    # openai-harmony's StreamableParser). In that case the text-based
    # parser is bypassed — the structured pass is bytes-faithful
    # whereas the regex round-trip lost calls whose JSON arguments
    # contained literal harmony sentinel substrings (PR #515 codex
    # round-12 / round-14 BLOCKING).
    engine_tool_calls = getattr(output, "tool_calls", None)
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text, request, structured_tool_calls=engine_tool_calls
    )

    # Honor ``parallel_tool_calls=false`` by capping the parsed list at one.
    # No decoder-level enforcement exists, so this is a post-parse trim — the
    # only reliable lever for OpenAI-compat clients that explicitly request a
    # single tool call (see PR #132 for the longer-term FSM-constrained path).
    if tool_calls and len(tool_calls) > 1 and request.parallel_tool_calls is False:
        tool_calls = tool_calls[:1]

    # ``tool_choice="required"`` post-parse enforcement (#468 / #571).
    # The system suffix injected above (``_TOOL_USE_REQUIRED_SUFFIX``)
    # makes the model overwhelmingly likely to comply, but local
    # inference has no decoder-level guarantee.
    #
    # Channel-routed engines (harmony / gemma4) bypass the text parser
    # entirely: the ``OutputRouter`` lifts structured tool_calls out of
    # a dedicated channel, so a forced ``tool_choice`` is satisfied
    # whenever the model fires the tool — the parser path is irrelevant.
    #
    # Text-parser engines (hermes / qwen3_coder / minimax / glm47 / …)
    # only surface a tool_call when the model emits the parser's wire
    # markers. The same model behaviour that produces a structured call
    # on harmony can produce text that the hermes regex fails to
    # recognise — and pre-#571 the 422 here fired for hermes while
    # harmony returned 200, breaking parser-agnostic contracts.
    #
    # The OpenAI ``tool_choice`` contract is parser-agnostic: when the
    # client forces a tool call, the response MUST carry one. To
    # restore symmetry we synthesise a tool_call server-side when the
    # target tool is unambiguous — named-function form (the name is
    # the choice), or ``"required"`` with a single tool entry (the
    # name is unique). When ``"required"`` is paired with multiple
    # tools and the parser returned nothing, we genuinely cannot pick
    # — fall back to 422 with a message that points to the
    # ``{type:"function",function:{name:X}}`` form as the escape
    # hatch, matching pre-#571 wording for that diagnostic.
    # Streaming path is best-effort prompt-injection only; once SSE
    # chunks are out we can't 422 mid-flight.
    if request.tool_choice is not None and request.tools:
        if request.tool_choice == "required" and not tool_calls:
            if len(request.tools) == 1:
                _solo_name = request.tools[0].function.get("name")
                if _solo_name:
                    logger.warning(
                        "tool_choice='required' on a parser-only path produced "
                        "no tool_calls; synthesising a call to the sole "
                        "available tool %r with empty arguments to honor the "
                        "OpenAI tool_call-guaranteed contract (#571).",
                        _solo_name,
                    )
                    tool_calls = [_synthesize_forced_tool_call(_solo_name)]
            if not tool_calls:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        'tool_choice="required" but the model returned a text response '
                        "with no tool_calls. Local inference has no decoder-level "
                        "constraint; the system-prompt enforcement was insufficient "
                        "for this prompt. Retry with a more concrete user message or "
                        'use tool_choice={"type":"function","function":{"name":...}} '
                        "to pin a specific tool."
                    ),
                )
        if (
            isinstance(request.tool_choice, dict)
            and request.tool_choice.get("type") == "function"
        ):
            _target = (request.tool_choice.get("function") or {}).get("name")

            # OpenAI spec: a named ``tool_choice`` allows ONLY the named
            # function. A response that includes the target plus any
            # other call violates the contract — refuse to forward.
            # Round-4 codex BLOCKING #2: prior ``any(...)`` accepted
            # ``[target, wrong]`` and shipped the extra call to the
            # client. Now require: at least one match AND every emitted
            # call matches.
            if _target:
                _names = [_tool_call_name(tc) for tc in tool_calls or []]
                _mismatched = [n for n in _names if n != _target]
                # Codex R1 BLOCKING (#675): defense-in-depth — never
                # synthesise a call to a function the client did not
                # submit. The early prompt-level validation (~line 488)
                # already 400s when ``_target`` is absent from
                # ``request.tools``, but a future refactor could shift
                # or bypass that gate, and the synthesis branch must
                # not trust ``_target`` blindly. Gate on the submitted
                # tool-name set; on miss, raise 422 rather than
                # fabricating a call to a tool the client never
                # defined.
                _submitted_tool_names = {
                    t.function.get("name")
                    for t in (request.tools or [])
                    if t.type == "function"
                }
                _target_is_submitted = _target in _submitted_tool_names
                # #571: when the parser returned NOTHING (``_names`` is
                # empty), the request still has a deterministic target
                # — the named-function form names it. Synthesise rather
                # than 422 so hermes matches harmony on the same input.
                # A non-empty-but-wrong list (the model called a
                # different tool) is a different failure mode: the
                # model actively defied the choice, which we still
                # surface as 422 — synthesising over a real wrong call
                # would silently drop the model's output and is a worse
                # client experience than the explicit failure.
                if not _names and _target_is_submitted:
                    logger.warning(
                        "tool_choice pinned function %r on a parser-only path "
                        "produced no tool_calls; synthesising a call with "
                        "empty arguments to honor the OpenAI tool_call-"
                        "guaranteed contract (#571).",
                        _target,
                    )
                    tool_calls = [_synthesize_forced_tool_call(_target)]
                elif not _names and not _target_is_submitted:
                    # Codex R1 BLOCKING (#675): named tool_choice points
                    # at a function that is not in ``request.tools`` —
                    # we must not fabricate a call to it. The early 400
                    # gate normally catches this; reaching here implies
                    # the gate was bypassed (e.g. cloud-fallback rewrite
                    # or future refactor). Refuse rather than synthesise.
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"tool_choice pinned function {_target!r} but it is "
                            "not present in the request's 'tools' array; refusing "
                            "to synthesise a call to an undefined tool."
                        ),
                    )
                elif _mismatched:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"tool_choice pinned function {_target!r} but the model "
                            f"emitted calls to {_mismatched}. Local "
                            "inference cannot decoder-enforce a specific function; "
                            "retry with a more direct user message."
                        ),
                    )

    # Validate tool call parameter values against schemas
    if tool_calls and request.tools:
        _validate_tool_call_params(tool_calls, request.tools)

    # Extract reasoning content. extract_reasoning() is stateless (pure regex
    # on full text), so the singleton is safe here unlike the streaming variant.
    # The tool_calls vs no-tool_calls split is encapsulated in
    # _finalize_content_and_reasoning so the regression test suite can exercise
    # the same orchestration without re-implementing it.
    cleaned_text_before_helper = cleaned_text
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=output.raw_text or output.text,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=cfg.reasoning_parser,
        engine_reasoning_text=getattr(output, "reasoning_text", "") or "",
        # #575 — chat-template-injected ``<think>`` means the model
        # never emits the start tag; pass the *effective* flag (with
        # the same ``None`` → ``"coder" not in model_name`` fallback
        # ``vllm_mlx/utils/chat_template.py:127`` uses for prompt
        # rendering) so the parser's Case 4 fallback fires on
        # default-on thinking — codex R1 BLOCKING. Use
        # ``cfg.model_path`` (the underlying HF path / alias the
        # engine actually loaded) rather than ``cfg.model_name``,
        # which can be overridden by ``--served-model-name`` and
        # would diverge from the prompt-render path's coder check
        # (codex R2 BLOCKING).
        enable_thinking=_effective_enable_thinking(
            resolved_thinking, cfg.model_path or cfg.model_name
        ),
        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # None → back-compat no-op.
        reasoning_max_tokens=getattr(request, "reasoning_max_tokens", None),
    )

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                parsed_json = _strip_backslash_before_unicode(parsed_json)
                # ``ensure_ascii=False`` keeps non-ASCII characters as
                # raw UTF-8 rather than escaping them to ``\uXXXX``. This
                # is the standard recommendation for JSON-over-HTTP with
                # international content (matches OpenAI's own response
                # encoding); FastAPI emits this body as UTF-8 anyway, so
                # the on-wire bytes are smaller and clients don't have to
                # un-escape user-visible CJK / emoji a second time.
                cleaned_text = json.dumps(parsed_json, ensure_ascii=False)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")
        except Exception as e:
            logger.warning(f"JSON output parsing failed: {e}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Clean and strip thinking tags from content
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
        if response_format and final_content:
            final_content = extract_json_from_response(final_content)

    # Issue #569: never silently drop. If the assistant turn would
    # otherwise have ``content=null`` AND ``tool_calls=null`` but the
    # engine surfaced ``reasoning_text`` (gemma-4-26b-4bit multi-turn
    # where the model got stuck inside ``<|channel>thought\n…`` and
    # ran out of tokens before emitting a closer / final / tool call),
    # surface the reasoning trace as ``content`` so OpenAI-compat
    # agentic clients reading only ``content``/``tool_calls`` don't
    # see an empty message.
    #
    # Codex round-1 BLOCKING on #676: skip the rescue when the client
    # requested structured output (``response_format`` =
    # ``json_object`` / ``json_schema``). Reasoning prose is almost
    # never valid JSON, so surfacing it as ``content`` would break
    # the OpenAI-compat structured-output contract and feed the
    # client garbage prose instead of validated JSON. The existing
    # empty/error path lets a structured-output client retry rather
    # than be surprise-fed unstructured text. Agentic (no
    # ``response_format``) clients still get the rescue.
    #
    # Codex round-2 BLOCKING on #676: the predicate is now factored
    # into ``_is_structured_output_requested`` so the streaming
    # rescue path (chat.py:~1580) can call the SAME predicate. Round
    # 1 inlined the check here only; codex round 2 caught the
    # streaming path drifting because it had no gate at all.
    if not _is_structured_output_requested(response_format):
        # PR #715 bundle, fuzz finding C / live-test repro: when the
        # parser's Case-4 fallback blanked ``cleaned_text`` (no tags +
        # ``enable_thinking=True`` → route the whole output to
        # reasoning, #575), the rescue must NOT then surface that
        # reasoning back as ``content`` — that would duplicate the
        # trace byte-identically into both fields. The helper signals
        # Case-4 by returning an empty ``cleaned_text`` when the
        # pre-helper text was non-empty AND a reasoning_parser was
        # wired AND the engine wasn't already routing (engine path
        # has its own plug). Detect by comparing the pre- and post-
        # helper ``cleaned_text``.
        reasoning_is_case4 = bool(
            cleaned_text_before_helper
            and not cleaned_text
            and reasoning_text
            and cfg.reasoning_parser is not None
            and not (getattr(output, "reasoning_text", "") or "")
        )
        # D-STOP-THINK codex round-5 BLOCKING (PR #799): compute
        # ``prompt_thinking_active`` from the chat template +
        # resolved enable_thinking — same predicate
        # ``_should_start_in_thinking`` uses in anthropic.py:91 /
        # responses.py:88. Required by the helper's
        # Case-4 + stop + matched_stop arm to discriminate
        # prompt-injected mid-think from a casual stop-terminated
        # answer.
        _chat_template_str = ""
        _tok = getattr(engine, "tokenizer", None)
        if _tok and hasattr(_tok, "chat_template"):
            _chat_template_str = _tok.chat_template or ""
        prompt_thinking_active = (
            resolved_thinking is not False
            and "<think>" in _chat_template_str
            and "add_generation_prompt" in _chat_template_str
        )
        final_content = _rescue_silent_drop_from_reasoning(
            final_content,
            reasoning_text,
            tool_calls,
            finish_reason=finish_reason,
            raw_text=output.raw_text or output.text,
            reasoning_is_case4=reasoning_is_case4,
            matched_stop=getattr(output, "matched_stop", None),
            prompt_thinking_active=prompt_thinking_active,
        )

    # Build logprobs for response if requested
    choice_logprobs = None
    if want_logprobs and token_logprobs_list:
        choice_logprobs = ChoiceLogProbs(content=token_logprobs_list)

    chat_response = ChatCompletionResponse(
        model=_resolve_model_name(request.model),
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=choice_logprobs,
            )
        ],
        usage=_build_usage(output, reasoning_text),
    )
    # L-05: surface silent ``enable_thinking`` drop on non-Qwen parsers.
    # Empty dict when the client didn't set the flag OR the active
    # parser honors it (qwen3) — kwargs spread is the right shape.
    response_headers = enable_thinking_warning_header(
        request, getattr(cfg, "reasoning_parser_name", None)
    )
    return Response(
        content=chat_response.model_dump_json(exclude_none=True),
        media_type="application/json",
        headers=response_headers or None,
    )


async def stream_chat_completion(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    *,
    response_id: str | None = None,
    created: int | None = None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Uses StreamingPostProcessor for reasoning/tool/sanitization pipeline.
    SSE formatting stays inline for performance (fast path bypasses Pydantic).

    Args:
        response_id: Optional pre-computed response id (``chatcmpl-…``).
            When provided, all SSE chunks share this id instead of one
            generated fresh here. Used by ``stream_chat_completion_guided``
            on its unconstrained fallback path so the client-visible
            stream stays self-consistent across the guided→unconstrained
            handoff (DeepSeek pr_validate round 5 finding).
        created: Optional pre-computed Unix timestamp. Same rationale.
    """
    from ..service.postprocessor import StreamingPostProcessor

    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        if response_id is None:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Check if we should include usage in the final chunk
        include_usage = request.stream_options and request.stream_options.include_usage

        # Logprobs configuration
        want_logprobs = request.logprobs and request.top_logprobs
        top_k_logprobs = request.top_logprobs or 0

        def _build_chunk_logprobs(output: GenerationOutput) -> ChoiceLogProbs | None:
            """Build ChoiceLogProbs for a streaming chunk if logprobs requested."""
            if not want_logprobs:
                return None
            entries = _extract_streaming_token_logprobs(
                output, engine.tokenizer, top_k_logprobs
            )
            return ChoiceLogProbs(content=entries) if entries else None

        # Pre-compute SSE template parts that don't change per-token.
        _sse_created = created if created is not None else int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        def _fast_sse_chunk(text: str, field: str = "content") -> str:
            """Build SSE chunk JSON directly, bypassing Pydantic serialization."""
            escaped = json.dumps(text)
            return f'{_sse_prefix}"{field}":{escaped}{_sse_suffix}'

        # First chunk with role
        _first_sse = f'{_sse_prefix}"role":"assistant"{_sse_suffix}'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[SSE-ROLE] {_first_sse.strip()[:200]}")
        yield _first_sse

        # Initialize post-processor.
        # request_dict carries `tools` so streaming parsers (qwen3_coder etc.)
        # can do schema-driven type conversion (#171).
        request_dict = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else None
        )
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=bool(request.tools),
            # `kwargs` is the **kwargs from this function's signature; the
            # route handler unpacks chat_kwargs (which sets
            # "enable_thinking" when request.enable_thinking is not None
            # or cfg.no_thinking is set). Pulled through as a name so
            # StreamingPostProcessor can short-circuit the reasoning
            # parser when the client explicitly disabled thinking
            # (closes the empty-content streaming bug from PR #208).
            enable_thinking=kwargs.get("enable_thinking"),
            json_mode=bool(
                request.response_format
                and getattr(request.response_format, "type", "text") != "text"
            ),
            request=request_dict,
            # Per-request reasoning cap (upstream vLLM PR #20859 backport).
            # When None the postprocessor is a no-op for the cap path.
            reasoning_max_tokens=getattr(request, "reasoning_max_tokens", None),
        )
        processor.set_thinking_model(request.model)
        processor.reset()

        # Forced ``tool_choice`` synthetic-prefix replay swallow (PR #716
        # codex r9 BLOCKING #1). When the upstream chat_kwargs builder
        # set ``forced_assistant_prefix`` (the route's
        # ``_forced_tool_call_prefix`` branch), the engine's
        # ``stream_chat`` yields the prefix back as a synthetic first
        # chunk so plain-text consumers see the wire envelope. Seed the
        # postprocessor with the same bytes so it can swallow that
        # synthetic chunk BEFORE the reasoning parser sees it — without
        # this, the prefix bytes route through ``Case-3 → reasoning``
        # in ``BaseThinkingReasoningParser``, polluting
        # ``accumulated_reasoning`` AND risking a raw-byte leak into
        # ``delta.reasoning_content`` on parser variants the MiniMax
        # tool-markup redirect doesn't catch (chunk-boundary splits,
        # future parsers). See ``StreamingPostProcessor.__init__`` for
        # the swallow-buffer state machine. No-op when the prefix is
        # absent.
        processor.seed_forced_assistant_prefix(kwargs.get("forced_assistant_prefix"))

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        # Buffer the terminal "finish" event so the cross-format fallback in
        # processor.finalize() (#425) gets a chance to recover a missed tool
        # call BEFORE we emit a terminal chunk. Without this buffer the route
        # emits a finish_reason="stop" chunk first and then a separate
        # finish_reason="tool_calls" chunk from the fallback path — spec-
        # compliant clients stop reading at the first finish_reason and
        # silently drop the tool call (#v0.6.63 onboarding sweep finding #3).
        buffered_finish: tuple | None = None

        # Stream content — PostProcessor handles reasoning/tool/sanitize.
        # ``is_streaming=True`` is consumed by DiffusionEngine to disable
        # the gemma4 wire-marker carve-out in ``skip_special_token_ids``:
        # this path forwards each chunk as an SSE delta without running
        # the tool parser, so any markers left in by the carve-out would
        # surface as raw ``<|tool_call>`` wire text in ``delta.content``
        # to the client (pr_validate #558 r8 BLOCKING #2). Engines whose
        # ``stream_chat`` doesn't know the kwarg swallow it via the
        # ``**kwargs`` tail on ``BaseEngine.stream_chat`` — no behavior
        # change for BatchedEngine which uses its own special-token
        # handling.
        async for output in engine.stream_chat(
            messages=messages, is_streaming=True, **kwargs
        ):
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            # ``cached_tokens`` is a single per-request value (the
            # prefix-cache hit count set once when the request is
            # scheduled), so re-reading it on every chunk just
            # re-stamps the same value; the guard mirrors the
            # ``prompt_tokens`` branch above for ad-hoc engines that
            # don't carry the field.
            if hasattr(output, "cached_tokens") and output.cached_tokens:
                cached_tokens = output.cached_tokens

            for event in processor.process_chunk(output):
                if event.type == "content":
                    if not want_logprobs:
                        _sse = _fast_sse_chunk(event.content, "content")
                        if _sse:
                            yield _sse
                    else:
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            created=_sse_created,
                            model=_resolve_model_name(request.model),
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        content=event.content,
                                    ),
                                    logprobs=_build_chunk_logprobs(output),
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                elif event.type == "reasoning":
                    yield _fast_sse_chunk(event.reasoning, "reasoning_content")

                elif event.type == "tool_call":
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        created=_sse_created,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    tool_calls=event.tool_calls,
                                ),
                                finish_reason=event.finish_reason,
                            )
                        ],
                        # Usage placement: when ``stream_options.include_usage``
                        # is True, usage MUST appear ONLY in the dedicated
                        # trailing chunk per the OpenAI streaming spec.
                        # Without ``include_usage``, legacy clients expect it
                        # on the finish chunk.
                        usage=(
                            None
                            if include_usage
                            else (get_usage(output) if output.finished else None)
                        ),
                    )
                    _tc_sse = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    logger.info(f"[SSE-TC] {_tc_sse.strip()[:300]}")
                    yield _tc_sse

                elif event.type == "finish":
                    # Defer emission: finalize() (below) may recover a
                    # missed tool call via cross-format fallback. If it
                    # does, we must merge the recovered tool_calls into
                    # this single terminal chunk and re-stamp the
                    # finish_reason as "tool_calls" — not emit two
                    # contradictory finish chunks.
                    buffered_finish = (event, output)

        # Fallback tool call detection (post-stream). Collect ALL fallback
        # tool_call events before emitting; they get merged into the
        # buffered finish chunk so the stream produces exactly one
        # terminal chunk with one finish_reason (OpenAI spec).
        # ``content`` events from finalize() carry prefix-held bytes
        # released by the parser (codex round-3 CRITICAL): the
        # streaming parser holds back partial tool-call sentinels
        # (``<``, ``<|``, ``<func``...) so per-char streaming doesn't
        # leak them as content before the full opener arrives. When
        # the stream ends with held bytes still buffered AND no tool
        # call ever fired, the postprocessor releases them — accumulate
        # here and merge into the terminal chunk's content so the user
        # doesn't see a truncated reply (codex round-4 CRITICAL).
        fallback_tool_calls: list = []
        finalize_content_parts: list[str] = []
        for event in processor.finalize():
            if event.type == "tool_call":
                fallback_tool_calls.extend(event.tool_calls or [])
            elif event.type == "content" and event.content:
                finalize_content_parts.append(event.content)
        finalize_content = "".join(finalize_content_parts)

        # Emit the terminal chunk. Three cases:
        #   (a) Streaming parser already emitted tool_calls during the
        #       loop → buffered_finish has finish_reason="tool_calls"
        #       and fallback_tool_calls is empty. Emit as-is.
        #   (b) Streaming parser missed the call but finalize() recovered
        #       it via cross-format fallback → merge tool_calls into the
        #       buffered finish and override finish_reason="tool_calls".
        #   (c) No tool calls at all → emit the buffered finish unchanged.
        if buffered_finish is not None:
            finish_event, finish_output = buffered_finish
            if fallback_tool_calls:
                logger.info(
                    "[SSE-FALLBACK-TC-MERGED] merging %d recovered tool_call(s) "
                    "into terminal chunk; overriding finish_reason -> tool_calls",
                    len(fallback_tool_calls),
                )
            # Merge any released prefix-held content into the terminal
            # chunk's delta.content (codex round-4 CRITICAL). Concatenate
            # to whatever the finish event already carries — the
            # finish_event.content path is normally None for non-tool
            # plain-text streams (deltas already drained content during
            # the loop), so this typically just adds the held suffix.
            terminal_content = (finish_event.content or "") + finalize_content

            # Issue #569 streaming rescue: if NOTHING was streamed as
            # ``content`` across the whole turn AND no ``tool_calls``
            # fired AND the model produced reasoning, surface the
            # accumulated reasoning trace as ``content`` in the
            # terminal chunk. Mirrors the non-streaming rescue in
            # ``_rescue_silent_drop_from_reasoning`` so streaming
            # clients (Cline, Cursor, Codex CLI) reading the
            # assembled ``content`` stream don't end the turn on an
            # empty buffer when gemma-4 (etc.) got stuck inside
            # ``<|channel>thought\n…`` and never emitted any closer
            # / final / tool call. Per-delta ``reasoning_content``
            # chunks have already been sent during the loop; this
            # adds a NEW ``content`` chunk at the end (duplication of
            # the same text across the two channels is the lesser
            # evil vs. a silently empty content stream).
            #
            # Codex round-2 BLOCKING on #676: gate on the SAME
            # ``_is_structured_output_requested`` predicate as the
            # non-streaming path (chat.py:~1283). Without this, a
            # ``stream=true`` request with
            # ``response_format={"type": "json_object"|"json_schema"}``
            # would still receive reasoning prose in
            # ``delta.content`` despite the non-streaming path
            # explicitly suppressing exactly that. Structured-output
            # clients expect validated JSON or the existing empty
            # path so they can retry — never surprise prose.
            #
            # Codex round-3 BLOCKING on #676: route the streaming
            # rescue through ``_rescue_silent_drop_from_reasoning``
            # instead of promoting ``processor.accumulated_reasoning``
            # directly. The previous direct-promotion branch bypassed
            # the helper's whitespace guard, so a reasoning-only
            # stream of ``"   \n"`` would emit a semantically empty
            # ``delta.content`` while non-streaming correctly
            # suppressed it. Funneling both paths through the same
            # helper means the predicate (whitespace + content
            # presence + tool-call absence) is defined ONCE and the
            # two paths cannot drift. The structured-output gate
            # stays here at the call site (parallel to non-streaming
            # at chat.py:~1285), because it depends on per-request
            # ``response_format`` which the rescue helper has no
            # access to.
            already_streamed_content = bool(processor.accumulated_text)
            has_any_tool_calls = bool(fallback_tool_calls) or (
                finish_event.finish_reason == "tool_calls"
            )
            structured_output_requested = _is_structured_output_requested(
                request.response_format
            )
            if (
                not already_streamed_content
                and not has_any_tool_calls
                and not structured_output_requested
            ):
                # Pass ``terminal_content or None`` so the helper
                # sees the same "empty vs whitespace vs real"
                # distinction the non-streaming path does. Pass
                # ``None`` for ``tool_calls`` because we've already
                # checked ``has_any_tool_calls`` above — the helper's
                # tool-call branch would never fire here regardless,
                # but keeping the call symmetric with non-streaming
                # is the point.
                # Codex r3 P1: streaming rescue must skip the
                # truncated-``<think>`` case. The streaming reasoning
                # parser consumes the literal ``<think>`` token as a
                # state transition, so ``accumulated_reasoning`` never
                # carries it — but the parser's ``_saw_any_tag`` flag
                # records that a ``<think>``/``</think>`` boundary was
                # crossed. When the model truncated mid-thought
                # (``finish_reason="length"`` AND the parser saw the
                # opener AND never saw the closer), the reasoning
                # trace is NOT the final answer and promoting it to
                # ``content`` re-introduces the leak the non-streaming
                # gate now suppresses. Synthesise a ``raw_text`` with
                # an unclosed ``<think>`` opener so the rescue's
                # existing finish=length-with-unclosed-think gate
                # fires uniformly across both paths.
                rp = processor.reasoning_parser
                saw_open_no_close = bool(
                    rp
                    and getattr(rp, "_saw_any_tag", False)
                    and "</think>"
                    not in (
                        processor.accumulated_reasoning + processor.accumulated_text
                    )
                )
                synthetic_raw = (
                    "<think>" + processor.accumulated_reasoning
                    if saw_open_no_close
                    else processor.accumulated_reasoning or ""
                )
                # PR #715 bundle, fuzz finding C: streaming Case-4
                # mirror of the non-streaming detection. When the
                # parser never saw a ``<think>``/``</think>`` token
                # (``_saw_any_tag`` False) BUT routed everything into
                # reasoning (accumulated_text empty, accumulated_reasoning
                # non-empty) AND a parser is wired, the streamer's
                # Case-3 fallback ("no tags seen yet → reasoning") IS
                # firing — analogous to the non-streaming Case-4
                # blanking. Suppress the rescue on length-truncated
                # streams in that state too.
                reasoning_is_case4_stream = bool(
                    rp
                    and not getattr(rp, "_saw_any_tag", False)
                    and processor.accumulated_reasoning
                    and not processor.accumulated_text
                )
                # D-STOP-THINK codex round-5 BLOCKING (PR #799):
                # compute ``prompt_thinking_active`` for the streaming
                # rescue too — mirrors the non-streaming gate. The
                # streaming function doesn't have ``resolved_thinking``
                # in scope (it's a separate function from the
                # non-streaming caller), so re-resolve here from the
                # request.
                _stream_resolved_thinking = _resolve_enable_thinking(request)
                _chat_template_str_stream = ""
                _tok_stream = getattr(engine, "tokenizer", None)
                if _tok_stream and hasattr(_tok_stream, "chat_template"):
                    _chat_template_str_stream = _tok_stream.chat_template or ""
                prompt_thinking_active_stream = (
                    _stream_resolved_thinking is not False
                    and "<think>" in _chat_template_str_stream
                    and "add_generation_prompt" in _chat_template_str_stream
                )
                rescued_content = _rescue_silent_drop_from_reasoning(
                    terminal_content or None,
                    processor.accumulated_reasoning,
                    None,
                    finish_reason=finish_event.finish_reason,
                    raw_text=synthetic_raw,
                    reasoning_is_case4=reasoning_is_case4_stream,
                    matched_stop=getattr(finish_event, "matched_stop", None),
                    prompt_thinking_active=prompt_thinking_active_stream,
                )
                # The helper returns the rescued reasoning ONLY when
                # all four predicates pass (empty/whitespace content,
                # no tool calls, non-empty/non-whitespace reasoning).
                # Otherwise it returns the original input — for our
                # pass it returns ``terminal_content or None``. We
                # only want to overwrite when the helper actually
                # promoted reasoning to content, i.e. the returned
                # value differs from what we passed in.
                if rescued_content and rescued_content != (terminal_content or None):
                    terminal_content = rescued_content
                    logger.info(
                        "[SSE-RESCUE-#569] terminal chunk content empty + no "
                        "tool calls; surfacing %d-char reasoning trace as "
                        "content",
                        len(terminal_content),
                    )
            final_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=terminal_content or None,
                            reasoning_content=finish_event.reasoning,
                            tool_calls=(
                                fallback_tool_calls if fallback_tool_calls else None
                            ),
                        ),
                        finish_reason=(
                            "tool_calls"
                            if fallback_tool_calls
                            else finish_event.finish_reason
                        ),
                        logprobs=_build_chunk_logprobs(finish_output),
                    )
                ],
                # See "Usage placement" note on the tool_call branch.
                usage=(
                    None
                    if include_usage
                    else (get_usage(finish_output) if finish_output.finished else None)
                ),
            )
            yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        elif fallback_tool_calls or finalize_content:
            # Defensive: stream ended without a "finish" event but
            # finalize() produced either recovered tool calls or
            # released held content (shouldn't normally happen —
            # process_chunk emits finish on output.finished).
            #
            # Only emit material that has NOT already been streamed:
            # ``finalize_content`` (released prefix-held tail) and
            # ``fallback_tool_calls`` (cross-format recovered calls).
            # Do NOT include ``processor.accumulated_text`` /
            # ``accumulated_reasoning`` — both were already written
            # to the wire as per-delta chunks during the loop, so
            # replaying them would duplicate the whole response
            # (codex re-review BLOCKING). The original round-6 fix
            # in the postprocessor makes this branch unreachable
            # in the common case, but defense-in-depth: keep this
            # synthetic chunk additive only.
            tool_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(
                            content=finalize_content or None,
                            reasoning_content=None,
                            tool_calls=fallback_tool_calls or None,
                        ),
                        finish_reason=("tool_calls" if fallback_tool_calls else "stop"),
                    )
                ],
            )
            _fb_sse = f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
            logger.info(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
            yield _fb_sse

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested. Mirror non-streaming
        # shape by populating completion_tokens_details.reasoning_tokens
        # when the postprocessor saw a reasoning split — v0.6.63
        # onboarding sweep finding #5 (streaming previously dropped this
        # field even when non-streaming had it).
        if include_usage:
            # Build a synthetic GenerationOutput-shaped namespace so
            # _build_usage can compute the reasoning_tokens breakdown.
            class _UsageOutput:
                pass

            _u = _UsageOutput()
            _u.prompt_tokens = prompt_tokens
            _u.completion_tokens = completion_tokens
            _u.cached_tokens = cached_tokens
            # ``text`` carries the accumulated content (NOT reasoning) so
            # ``_build_usage`` can split ``completion_tokens`` between
            # reasoning and content by character ratio. Without this,
            # streaming usage chunks attribute 100% of the budget to
            # reasoning when ``len(reasoning)//4 >= completion_tokens``
            # (same root cause as the non-stream bug surfaced by the
            # v0.6.66 hybrid onboarding sweep on qwen3.6-27b-8bit).
            _u.text = processor.accumulated_text or ""
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=_build_usage(
                    _u,
                    processor.accumulated_reasoning or None,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()


async def stream_chat_completion_guided(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    json_schema: dict,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion with json_schema constrained decoding.

    Runs ``engine.generate_with_schema`` (which produces a single buffered
    ``GenerationOutput`` — outlines integration has no native streaming
    interface), then synthesizes an SSE stream from the buffered text.
    Pre-fix, ``stream=true`` requests with ``response_format: json_schema``
    silently bypassed ``GuidedGenerator`` because the stream branch of
    ``_create_chat_completion_impl`` went straight to ``engine.stream_chat``
    with no constraint hookup — the model would emit unconstrained tokens
    (e.g. a ```json ... ``` markdown fence around the JSON), defeating the
    user's intent (Gap #2, v0.6.60 onboarding sweep).

    On guided failure (exception from ``generate_with_schema``), delegates
    to the unconstrained ``stream_chat_completion`` helper to preserve
    request liveness — matches the non-streaming fallback semantics in
    ``_create_chat_completion_impl``. Clients in strict-mode use cases
    should validate the response against their schema regardless.
    """
    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        include_usage = bool(
            request.stream_options and request.stream_options.include_usage
        )

        # Pre-compute SSE template parts (mirrors stream_chat_completion's
        # fast path so chunk encoding is identical for clients).
        _sse_created = int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        # Run guided generation buffered. If it raises, fall through to
        # the unconstrained streaming helper — this preserves request
        # liveness (constraints best-effort, response always emitted)
        # and matches the non-streaming fallback semantics. We DO NOT
        # emit our own role chunk before the guided call, because on
        # fallback the unconstrained helper emits its own complete
        # SSE stream (role → content → DONE); a pre-emitted role would
        # produce a duplicate role chunk in the fallback path.
        #
        # ``raise_on_failure=True`` is critical: without it,
        # ``generate_with_schema`` silently falls back to
        # ``self.chat(...)`` on guided-engine failure and returns a
        # buffered unconstrained ``GenerationOutput``. From this
        # helper's POV that looks like a successful guided result and
        # we would emit one giant content chunk at the end —
        # defeating SSE for clients/proxies that rely on early chunks
        # (codex Round 2 finding).
        try:
            output = await engine.generate_with_schema(
                messages=messages,
                json_schema=json_schema,
                raise_on_failure=True,
                **kwargs,
            )
        except Exception as guided_err:
            logger.warning(
                "Guided streaming generation failed, falling back to "
                f"unconstrained streaming: {guided_err}"
            )
            # Log only the schema's top-level shape, not the full body —
            # user-supplied schemas may embed PII (default values),
            # internal endpoint names, or be megabytes large. Keys +
            # required-list are enough to disambiguate the failure
            # without flooding ops logs or exposing payload contents.
            _schema_keys = (
                list(json_schema.keys()) if isinstance(json_schema, dict) else None
            )
            _required = (
                json_schema.get("required") if isinstance(json_schema, dict) else None
            )
            logger.debug(
                f"Problematic schema shape: keys={_schema_keys} required={_required}"
            )
            # Forward the pre-computed response_id + _sse_created so the
            # fallback stream's chunks share id/created with this outer
            # helper's would-be chunks. Without this, a client that
            # tracks the completion id across the guided→unconstrained
            # handoff sees two different ids/timestamps for what is
            # logically one request (DeepSeek pr_validate round 5).
            async for chunk in stream_chat_completion(
                engine,
                messages,
                request,
                response_id=response_id,
                created=_sse_created,
                **kwargs,
            ):
                yield chunk
            return

        # Success path: synthesize SSE stream from the buffered output.
        # First chunk with role.
        yield f'{_sse_prefix}"role":"assistant"{_sse_suffix}'

        content = output.text or ""
        if content:
            yield f'{_sse_prefix}"content":{json.dumps(content)}{_sse_suffix}'

        # ``output`` is the single buffered ``GenerationOutput`` from
        # ``engine.generate_with_schema`` (outlines integration has no
        # native streaming interface — see this function's docstring).
        # Token counts are therefore read once and final; the main
        # ``stream_chat_completion`` path re-reads inside the stream
        # loop because the engine emits a sequence of GenerationOutputs
        # and the last one carries the authoritative counts.
        prompt_tokens = getattr(output, "prompt_tokens", 0) or 0
        completion_tokens = getattr(output, "completion_tokens", 0) or 0
        cached_tokens = getattr(output, "cached_tokens", 0) or 0
        # Pass the engine's finish_reason through directly. Matches the
        # convention in ``stream_chat_completion`` (line ~925:
        # ``finish_reason=event.finish_reason``), which never coerces a
        # falsy value. ``GenerationOutput.finish_reason`` defaults to
        # "stop" anyway, so the prior ``or "stop"`` was redundant and
        # would have silently rewritten any legitimately-None value the
        # engine emits (DeepSeek pr_validate round 3 finding).
        finish_reason = getattr(output, "finish_reason", None)

        # Final chunk with finish_reason. Usage placement:
        #  - When ``stream_options.include_usage`` is True, usage MUST
        #    appear ONLY in the dedicated usage chunk below (per the
        #    OpenAI spec; emitting it in both places would have clients
        #    that aggregate usage double-count). DeepSeek review caught
        #    the duplication on first pass.
        #  - When False, attach usage to the finish chunk so a client
        #    that doesn't set ``include_usage`` still receives token
        #    counts in the final delta (matches the legacy behavior of
        #    ``stream_chat_completion`` and the non-streaming response
        #    shape).
        finish_usage = (
            None
            if include_usage
            else Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=cached_tokens)
                    if cached_tokens
                    else None
                ),
            )
        )
        # ``created`` must be passed explicitly: the SSE prefix-style
        # chunks above already share ``_sse_created`` (computed once at
        # the top of the helper). ``ChatCompletionChunk.created`` has
        # ``default_factory=lambda: int(time.time())``, so a default
        # instantiation here would stamp a fresh timestamp on the finish
        # chunk and break the OpenAI streaming-spec invariant that all
        # chunks in one completion share a single ``created`` value
        # (DeepSeek pr_validate round 2 finding).
        finish_chunk = ChatCompletionChunk(
            id=response_id,
            created=_sse_created,
            model=_resolve_model_name(request.model),
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
            usage=finish_usage,
        )
        yield f"data: {finish_chunk.model_dump_json(exclude_none=True)}\n\n"

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (guided stream): {completion_tokens} tokens "
            f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                created=_sse_created,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens_details=(
                        PromptTokensDetails(cached_tokens=cached_tokens)
                        if cached_tokens
                        else None
                    ),
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()
