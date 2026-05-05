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
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_USE_SYSTEM_SUFFIX,
    _build_usage,
    _disconnect_guard,
    _extract_token_logprob,
    _inject_json_instruction,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _should_pass_tools_to_template,
    _uses_direct_jang_generation,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _make_prompt_progress_callback():
    from ..middleware.metrics import get_current_request_id
    from ..request_metrics import get_recorder

    req_id = get_current_request_id()
    if req_id is None:
        return None

    recorder = get_recorder()

    def _callback(processed: int, total: int) -> None:
        recorder.update(req_id, prompt_tokens=max(int(processed), int(total)))

    return _callback


def _sanitize_direct_jang_textual_tools(
    messages: list[dict],
    engine,
    has_request_tools: bool = False,
    has_tool_result: bool = False,
) -> list[dict]:
    if not _uses_direct_jang_generation(engine):
        return messages

    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        content = message.get("content")
        if message.get("role") not in {"system", "developer"} or not isinstance(
            content, str
        ):
            sanitized.append(message)
            continue

        if has_request_tools or "operating inside pi" in content:
            suffix_lines = [
                line
                for line in content.splitlines()
                if line.startswith("Current date:")
                or line.startswith("Current working directory:")
            ]
            suffix = ("\n" + "\n".join(suffix_lines)) if suffix_lines else ""
            if has_tool_result:
                replacement = (
                    "You are a concise coding assistant. A tool has already run. "
                    "Give a short final status for the latest user request. Do not "
                    "print source code, code fences, examples, or repeated symbols."
                    f"{suffix}"
                )
            elif has_request_tools:
                replacement = (
                    "You are a concise coding assistant. Answer only the latest "
                    "user request directly. When the user asks you to create, edit, "
                    "inspect, run, test, or verify files, your entire response must "
                    "be tool calls using the available tools. For multi-file project "
                    "creation, prefer one bash tool call that writes the files and "
                    "runs validation. If you do not emit DSML, emit exactly the text "
                    'format [Calling tool: bash({"command":"...", "timeout":120})] '
                    'or [Calling tool: write({"path":"...", "content":"..."})]. '
                    "Never print markdown, source code fences, explanations, or "
                    "examples for file creation requests. For a greeting, reply with "
                    "one short greeting and ask how you can help. Do not emit "
                    "repeated symbols."
                    f"{suffix}"
                )
            else:
                replacement = (
                    "You are a concise coding assistant. Answer only the latest "
                    "user request directly. For a greeting, reply with one short "
                    "greeting and ask how you can help. Do not add examples unless "
                    "the user asks. Do not emit HTML, XML, tool calls, or repeated "
                    "symbols."
                    f"{suffix}"
                )
            sanitized.append(
                {
                    **message,
                    "content": replacement,
                }
            )
            continue

        if "\nAvailable tools:\n" not in content or "\nGuidelines:\n" not in content:
            sanitized.append(message)
            continue

        before, rest = content.split("\nAvailable tools:\n", 1)
        _, after = rest.split("\nGuidelines:\n", 1)
        sanitized.append(
            {
                **message,
                "content": f"{before}\nAvailable tools:\n(none)\n\nGuidelines:\n{after}",
            }
        )
    return sanitized


_TOOL_INTENT_RE = re.compile(
    r"\b("
    r"let me|now let me|i'?ll|i will|starting with|"
    r"create|write|edit|run|test|verify|fix|repair"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_deferred_tool_use(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if '"path"' in lowered:
        return True
    return bool(_TOOL_INTENT_RE.search(text))


_FILE_CREATE_RE = re.compile(
    r"\b(?:create|write|make|save)\b.*?\b(?:file\s+)?(?:named|called)?\s*[`'\"]?([^`'\"\s]+?\.[A-Za-z0-9]+)",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_CODE_RE = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*\n(.*?)(?:\n```|$)", re.DOTALL)
_FENCED_CODE_BLOCK_RE = re.compile(
    r"```(?:[A-Za-z0-9_.+/-]+)?[^\n]*\n(.*?)(?:\n```|$)", re.DOTALL
)
_ARTIFACT_PATH_RE = re.compile(
    r"(?:^|[\s:`'\"])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*"
    r"(?:\.[A-Za-z0-9]+|package\.json|tsconfig\.json|Dockerfile|Makefile))"
    r"(?=$|[\s:`'\",)])",
    re.IGNORECASE,
)


def _tool_to_dict(tool) -> dict:
    if hasattr(tool, "model_dump"):
        return tool.model_dump(exclude_none=True)
    if isinstance(tool, dict):
        return tool
    return {}


def _tool_name(tool) -> str | None:
    tool_dict = _tool_to_dict(tool)
    function = tool_dict.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return name if isinstance(name, str) else None
    name = tool_dict.get("name")
    return name if isinstance(name, str) else None


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _clean_artifact_path(path: str) -> str:
    path = path.strip().strip("`'\".,:;()[]{}")
    while path.startswith("./"):
        path = path[2:]
    return path


def _path_from_fence_context(context: str) -> str | None:
    for line in reversed(context.splitlines()[-8:]):
        matches = [
            _clean_artifact_path(match.group(1))
            for match in _ARTIFACT_PATH_RE.finditer(line)
        ]
        matches = [
            match
            for match in matches
            if match and not match.startswith("/") and ".." not in match.split("/")
        ]
        if matches:
            return matches[-1]
    return None


def _trim_repeated_artifact_tail(content: str) -> str:
    lines = content.splitlines()
    for block_size in range(1, min(8, len(lines) // 2) + 1):
        while (
            len(lines) >= block_size * 2
            and lines[-block_size:] == lines[-(block_size * 2) : -block_size]
        ):
            del lines[-block_size:]
    return "\n".join(lines).strip()


def _extract_markdown_file_artifacts(text: str) -> list[tuple[str, str]]:
    artifacts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _FENCED_CODE_BLOCK_RE.finditer(text):
        path = _path_from_fence_context(
            text[max(0, match.start() - 500) : match.start()]
        )
        content = _trim_repeated_artifact_tail(match.group(1).strip())
        if not path or not content or path in seen:
            continue
        seen.add(path)
        artifacts.append((path, content))
    return artifacts


def _synthesize_direct_jang_write_tool_calls(
    text: str, messages: list, request: ChatCompletionRequest
) -> list[dict] | None:
    tool_names = {_tool_name(tool) for tool in (request.tools or [])}
    if "write" not in tool_names:
        return None

    artifacts = _extract_markdown_file_artifacts(text)
    if artifacts:
        return [
            {
                "index": index,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps(
                        {"path": path, "content": content}, ensure_ascii=False
                    ),
                },
            }
            for index, (path, content) in enumerate(artifacts)
        ]

    user_text = _latest_user_text(messages)
    file_match = _FILE_CREATE_RE.search(user_text)
    code_match = _FENCED_CODE_RE.search(text)
    if not file_match or not code_match:
        return None

    path = file_match.group(1).strip()
    content = code_match.group(1).strip()
    if not path or not content:
        return None

    return [
        {
            "index": 0,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps(
                    {"path": path, "content": content}, ensure_ascii=False
                ),
            },
        }
    ]


def _finalize_content_and_reasoning(
    raw_text: str,
    cleaned_text: str,
    tool_calls: list,
    reasoning_parser,
) -> tuple[str, str | None]:
    """Compute final ``content`` + ``reasoning_text`` after tool parsing.

    Pulled out of the request handler so the regression suite can drive
    the EXACT same orchestration the production path uses, instead of
    maintaining a parallel reimplementation that can silently drift.

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
    output if cleaning produced an empty string).
    """
    reasoning_text = None
    if reasoning_parser is None:
        return cleaned_text, reasoning_text
    if tool_calls:
        reasoning_text, _ = reasoning_parser.extract_reasoning(raw_text)
    else:
        text_to_parse = cleaned_text or raw_text
        reasoning_text, cleaned_text = reasoning_parser.extract_reasoning(text_to_parse)
    return cleaned_text, reasoning_text


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

    # Validate n parameter (only n=1 supported)
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail="n > 1 is not supported. Rapid-MLX generates one completion per request.",
        )

    # Validate max_tokens (must be positive)
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
        )

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
    has_tools = bool(request.tools)
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

    has_tool_result = any(msg.role == "tool" for msg in request.messages)

    # Auto-inject system prompt suffix for tool use and/or reasoning control
    _inject_suffix = None
    if request.tools and cfg.tool_call_parser:
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

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(
            request.max_tokens, request.enable_thinking, engine
        ),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }
    prompt_progress_callback = _make_prompt_progress_callback()
    if prompt_progress_callback is not None:
        chat_kwargs["prompt_progress_callback"] = prompt_progress_callback
    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools and _should_pass_tools_to_template(engine) and not has_tool_result:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    # Pass through enable_thinking if explicitly set by the client
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif _uses_direct_jang_generation(engine) or cfg.no_thinking:
        chat_kwargs["enable_thinking"] = False

    messages = _sanitize_direct_jang_textual_tools(
        messages,
        engine,
        has_request_tools=bool(request.tools),
        has_tool_result=has_tool_result,
    )

    # Cloud routing: offload large-context requests to cloud LLM
    if cfg.cloud_router and not engine.is_mllm and hasattr(engine, "build_prompt"):
        try:
            prompt = engine.build_prompt(messages, tools=request.tools)
            total_tokens, new_tokens = engine.model.estimate_new_tokens(prompt)
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
                    )
                else:
                    result = await _wait_with_disconnect(
                        cfg.cloud_router.completion(cloud_messages, **cloud_kwargs),
                        raw_request,
                        timeout=request.timeout or cfg.default_timeout,
                    )
                    if result is None:
                        return Response(status_code=499, content="Client disconnected")
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                    )
            else:
                logger.info(
                    f"[LOCAL] {new_tokens} new tokens (total {total_tokens}) "
                    f"<= threshold {cfg.cloud_router.threshold}, using local inference"
                )
        except Exception as e:
            logger.warning(
                f"[CLOUD ROUTE] Error during routing check: {e}, falling back to local"
            )

    if request.stream:
        # Validate chat template eagerly so template errors return 400
        if hasattr(engine, "build_prompt") and not engine.is_mllm:
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
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                raw_request,
            ),
            media_type="text/event-stream",
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

    # Check if we should use guided generation for JSON schema
    use_guided = False
    json_schema = None
    if response_format and not request.tools:
        json_schema = extract_json_schema_for_guided(response_format)
        if json_schema and hasattr(engine, "supports_guided_generation"):
            use_guided = engine.supports_guided_generation
            if use_guided:
                logger.info("Using guided generation for JSON schema enforcement")

    try:
        if want_logprobs and not use_guided:
            output = None
            async for chunk in engine.stream_chat(messages=messages, **chat_kwargs):
                output = chunk
                if chunk.logprobs is not None and chunk.new_text:
                    token_id = chunk.tokens[-1] if chunk.tokens else 0
                    token_logprobs_list.append(
                        _extract_token_logprob(
                            chunk.logprobs, token_id, engine.tokenizer, top_k_logprobs
                        )
                    )
            if output is None:
                return Response(status_code=499)
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

    # Parse tool calls from output using configured parser
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, request)

    retry_messages = list(messages)
    for retry_index in range(2):
        if (
            not request.tools
            or tool_calls
            or not _looks_like_deferred_tool_use(cleaned_text or output.text)
        ):
            break
        logger.info(
            "Tool intent without tool call detected; retrying (%d/2)",
            retry_index + 1,
        )
        retry_messages = retry_messages + [
            {"role": "assistant", "content": cleaned_text or output.text},
            {
                "role": "user",
                "content": (
                    "Call the appropriate tool now. Do not explain, do not describe "
                    "what you will do, and do not output raw JSON as text. If the "
                    "previous tool failed, repair it by calling another tool."
                ),
            },
        ]
        retry_output = await _wait_with_disconnect(
            engine.chat(messages=retry_messages, **chat_kwargs),
            raw_request,
            timeout=timeout,
        )
        if retry_output is None:
            break
        retry_cleaned_text, retry_tool_calls = _parse_tool_calls_with_parser(
            retry_output.text, request
        )
        output = retry_output
        cleaned_text = retry_cleaned_text
        tool_calls = retry_tool_calls

    # Validate tool call parameter values against schemas
    if tool_calls and request.tools:
        _validate_tool_call_params(tool_calls, request.tools)

    # Extract reasoning content. extract_reasoning() is stateless (pure regex
    # on full text), so the singleton is safe here unlike the streaming variant.
    # The tool_calls vs no-tool_calls split is encapsulated in
    # _finalize_content_and_reasoning so the regression test suite can exercise
    # the same orchestration without re-implementing it.
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=output.text,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=cfg.reasoning_parser,
    )

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
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
    return Response(
        content=chat_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


async def stream_chat_completion(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Uses StreamingPostProcessor for reasoning/tool/sanitization pipeline.
    SSE formatting stays inline for performance (fast path bypasses Pydantic).
    """
    from ..service.postprocessor import StreamingPostProcessor

    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Check if we should include usage in the final chunk
        include_usage = request.stream_options and request.stream_options.include_usage

        # Logprobs configuration
        want_logprobs = request.logprobs and request.top_logprobs
        top_k_logprobs = request.top_logprobs or 0

        def _build_chunk_logprobs(output: GenerationOutput) -> ChoiceLogProbs | None:
            """Build ChoiceLogProbs for a streaming chunk if logprobs requested."""
            if not want_logprobs or output.logprobs is None:
                return None
            token_id = output.tokens[-1] if output.tokens else 0
            token_lp = _extract_token_logprob(
                output.logprobs, token_id, engine.tokenizer, top_k_logprobs
            )
            return ChoiceLogProbs(content=[token_lp])

        # Pre-compute SSE template parts that don't change per-token.
        _sse_created = int(time.time())
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

        if _uses_direct_jang_generation(engine) and any(
            msg.role == "tool" for msg in request.messages
        ):
            yield _fast_sse_chunk("Done.", "content")
            chunk = ChatCompletionChunk(
                id=response_id,
                model=_resolve_model_name(request.model),
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"
            return

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
        )
        processor.set_thinking_model(request.model)
        processor.reset()

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0
        has_tool_result = any(msg.role == "tool" for msg in request.messages)
        buffer_direct_jang_tools = bool(
            request.tools
            and not has_tool_result
            and _uses_direct_jang_generation(engine)
        )
        buffered_content = ""
        direct_jang_tool_calls_detected = False

        # Stream content — PostProcessor handles reasoning/tool/sanitize
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            for event in processor.process_chunk(output):
                if event.type == "content":
                    if buffer_direct_jang_tools:
                        buffered_content += event.content
                        continue
                    if not want_logprobs:
                        _sse = _fast_sse_chunk(event.content, "content")
                        if _sse:
                            yield _sse
                    else:
                        chunk = ChatCompletionChunk(
                            id=response_id,
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
                    direct_jang_tool_calls_detected = True
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    tool_calls=event.tool_calls,
                                ),
                                finish_reason=event.finish_reason,
                            )
                        ],
                        usage=get_usage(output) if output.finished else None,
                    )
                    _tc_sse = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    logger.info(f"[SSE-TC] {_tc_sse.strip()[:300]}")
                    yield _tc_sse

                elif event.type == "finish":
                    if buffer_direct_jang_tools and event.content:
                        buffered_content += event.content
                        continue
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    content=event.content,
                                    reasoning_content=event.reasoning,
                                ),
                                finish_reason=event.finish_reason,
                                logprobs=_build_chunk_logprobs(output),
                            )
                        ],
                        usage=get_usage(output) if output.finished else None,
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        # Fallback tool call detection
        for event in processor.finalize():
            if event.type == "tool_call":
                direct_jang_tool_calls_detected = True
                tool_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=event.tool_calls,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                _fb_sse = f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                logger.info(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
                yield _fb_sse

        if buffer_direct_jang_tools and not direct_jang_tool_calls_detected:
            synthetic_tool_calls = _synthesize_direct_jang_write_tool_calls(
                buffered_content, messages, request
            )
            if synthetic_tool_calls:
                tool_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=synthetic_tool_calls,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                yield f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                direct_jang_tool_calls_detected = True
            elif buffered_content:
                yield _fast_sse_chunk(buffered_content, "content")

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested
        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()
