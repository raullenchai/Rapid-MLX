# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for vllm-mlx.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports two modes:
- Simple mode (default): Maximum throughput for single-user scenarios
- Batched mode: Continuous batching for multiple concurrent users

Features:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)

Usage:
    # Simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import asyncio
import copy
from dataclasses import dataclass
import json
import logging
import os
import re
import secrets
import socket as _socket
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette.routing import Match

# Import from new modular API
# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import anthropic_to_openai
from .api.anthropic_models import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicUsage,
)
from .api.models import (
    AssistantMessage,  # noqa: F401
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,  # noqa: F401
    CompletionRequest,
    CompletionResponse,
    ContentPart,  # noqa: F401
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    ImageUrl,  # noqa: F401
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,  # noqa: F401
    MCPServersResponse,
    MCPToolInfo,  # noqa: F401
    MCPToolsResponse,
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    ModelsResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ToolCall,
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.responses_models import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseFunctionTool,
    ResponseIncompleteDetails,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningTextPart,
    ResponseTextContentPart,
    ResponsesRequest,
    ResponsesUsage,
)
from .api.tool_calling import (
    StreamingJsonFenceStripper,
    build_json_logits_processor,
    build_json_system_prompt,
    convert_tools_for_template,
    parse_json_output,
    parse_tool_calls,
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    extract_multimodal_content,
    is_mllm_model,  # noqa: F401
)
from .audio_limits import (
    DEFAULT_MAX_AUDIO_UPLOAD_BYTES,
    DEFAULT_MAX_AUDIO_UPLOAD_MB,
    DEFAULT_MAX_TTS_INPUT_CHARS,
    save_upload_with_limit,
    validate_tts_input_length,
)
from .cli_arg_types import make_json_object_arg_parser
from .engine import BaseEngine, BatchedEngine, GenerationOutput, SimpleEngine
from .endpoint_model_policies import (
    resolve_embedding_model_name,
    resolve_stt_model_name,
    resolve_tts_model_name,
)
from .engine.base import suspend_cancellation
from .lifecycle import ModelSpec, ResidencyManager
from .model_registry import (
    ModelLease,
    ModelManager,
    RegistryServeDefaults,
    load_registry_config,
)
from .metrics import metrics as _metrics
from .models.mllm import UnsafeRemoteURLError, _validate_url_safety, is_url
from .reasoning import get_parser as get_reasoning_parser
from .tool_parsers import ToolParserManager, get_parser_stop_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_IMPORTED_SIMPLE_ENGINE = SimpleEngine

# Global engine instance
_engine: BaseEngine | None = None
_model_manager: ModelManager | None = None
_model_name: str | None = None
_model_path: str | None = (
    None  # Actual model path (for cache dir, not affected by --served-model-name)
)
_warm_prompts_path: str | None = None  # Path to JSON of prompts to pre-warm at startup
_default_model_key: str | None = None
_default_max_tokens: int = 32768
_max_request_tokens: int = 32768
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_default_chat_template_kwargs: dict[str, object] | None = None
_metrics_enabled = False
_max_audio_upload_bytes: int = DEFAULT_MAX_AUDIO_UPLOAD_BYTES
_max_tts_input_chars: int = DEFAULT_MAX_TTS_INPUT_CHARS
_force_mllm_model: bool = False
_default_thinking_token_budget: int | None = (
    None  # Set via --default-thinking-token-budget
)
_auto_unload_idle_seconds: float = 0.0
_lazy_load_model: bool = False
_residency_manager: ResidencyManager | None = None
_lifecycle_task: asyncio.Task | None = None
_lifespan_active: bool = False

_FALLBACK_TEMPERATURE = 0.7
_FALLBACK_TOP_P = 0.9


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_temperature is not None:
        return _default_temperature
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    if _default_top_p is not None:
        return _default_top_p
    return _FALLBACK_TOP_P


def _resolve_request_max_tokens(requested_value: int | None) -> int:
    """Resolve and validate a request's max_tokens budget."""
    if requested_value is None:
        return _default_max_tokens
    if requested_value > _max_request_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens exceeds server limit ({_max_request_tokens})",
        )
    return requested_value


def _resolve_chat_template_kwargs(
    request_value: dict[str, object] | None,
) -> dict[str, object]:
    """Resolve chat template kwargs: request > server default > empty dict."""
    resolved: dict[str, object] = {}
    if _default_chat_template_kwargs:
        resolved.update(_default_chat_template_kwargs)
    if request_value:
        resolved.update(request_value)
    return resolved


@dataclass
class PreparedChatInvocation:
    """Fully prepared inputs for a single engine.chat/stream_chat call."""

    messages: list[dict]
    chat_kwargs: dict[str, object]
    response_format: object | None
    json_logits_processor: object | None


def _prepare_chat_messages(
    engine: BaseEngine,
    request_messages: list[Message | dict],
) -> tuple[list[dict], list, list, list, bool]:
    """Normalize messages and collect media once for both stream/non-stream paths."""
    _validate_remote_media_urls(request_messages)

    is_mllm = bool(getattr(engine, "is_mllm", False))
    preserve_native = bool(getattr(engine, "preserve_native_tool_format", False))

    if is_mllm:
        # For MLLM models, keep original messages with embedded images
        # (MLLM.chat() extracts images from message content internally)
        messages = []
        for msg in request_messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos, audios = [], [], []  # MLLM extracts these from messages
        logger.debug(f"MLLM: Processing {len(messages)} messages")
        # Convert tool_call arguments from JSON string to dict so that
        # chat templates can iterate them (e.g. GLM-4.6V calls .items()).
        # The LLM path does this inside extract_multimodal_content(), but
        # the MLLM path bypasses that function.
        if preserve_native:
            for msg_dict in messages:
                for tc in msg_dict.get("tool_calls") or []:
                    func = tc.get("function") or {}
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
        messages = _normalize_messages(messages)
    else:
        # For LLM, extract text and media separately
        messages, images, videos, audios = extract_multimodal_content(
            request_messages,
            preserve_native_format=preserve_native,
        )
        messages = _normalize_messages(messages)

    has_media = bool(images or videos or audios)
    if is_mllm and not has_media:
        # MLLM extracts media from messages directly, so images/videos are
        # always empty. Check message content for video/image types instead.
        for msg in request_messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in (
                        "image_url",
                        "image",
                        "video",
                        "video_url",
                        "audio",
                        "audio_url",
                    ):
                        has_media = True
                        break
            if has_media:
                break

    return messages, images, videos, audios, has_media


def _iter_remote_media_urls(messages: list[Message | dict]):
    """Yield remote media URLs from OpenAI-style multimodal message content."""
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        if not isinstance(content, list):
            continue
        for item in content:
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif hasattr(item, "dict"):
                item = {k: v for k, v in item.dict().items() if v is not None}
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")
            media_value = None
            if item_type == "image_url":
                media_value = item.get("image_url", {})
            elif item_type == "video_url":
                media_value = item.get("video_url", {})
            elif item_type == "audio_url":
                media_value = item.get("audio_url", {})
            elif item_type in {"image", "video", "audio"}:
                media_value = item.get(item_type, item.get("url", ""))

            if isinstance(media_value, dict):
                media_value = media_value.get("url", "")
            if isinstance(media_value, str) and is_url(media_value):
                yield media_value


def _validate_remote_media_urls(messages: list[Message | dict]) -> None:
    """Validate remote media URLs during request preparation."""
    for url in _iter_remote_media_urls(messages):
        _validate_url_safety(url)


def _raise_remote_media_http_error(exc: UnsafeRemoteURLError) -> None:
    """Log internal URL-safety detail while returning a generic client error."""
    logger.warning(
        "Blocked unsafe remote media URL: %s",
        _sanitize_log_text(exc, limit=500),
    )
    raise HTTPException(status_code=400, detail=exc.public_message) from exc


def _prepare_json_logits_processor(
    engine: BaseEngine,
    messages: list[dict],
    response_format: object | None,
    *,
    tools: list | None,
    tool_choice: object | None,
    log_context: str | None = None,
    thinking_model: bool = False,
) -> tuple[list[dict], object | None]:
    """Inject response_format instruction and build constrained decoding processor."""
    json_logits_processor = None
    if not response_format:
        return messages, json_logits_processor

    json_instruction = build_json_system_prompt(
        response_format, thinking_model=thinking_model
    )
    if json_instruction:
        messages = _inject_json_instruction(messages, json_instruction)

    # ``tools`` + ``response_format`` is undefined in OpenAI; skip constraints
    # when tools are active so tool-call markup can still be emitted.
    if tools and tool_choice != "none":
        return messages, json_logits_processor

    tokenizer_obj = _get_engine_tokenizer(engine)
    if tokenizer_obj is None:
        return messages, json_logits_processor

    try:
        json_logits_processor = build_json_logits_processor(
            response_format, tokenizer_obj
        )
    except Exception as exc:
        logger.warning("Failed to build JSON logits processor: %s", exc)
        json_logits_processor = None

    if json_logits_processor is not None:
        log_label = f" for {log_context}" if log_context else ""
        logger.info(
            "Constrained decoding enabled%s response_format.type=%s",
            log_label,
            (
                getattr(response_format, "type", None)
                if not isinstance(response_format, dict)
                else response_format.get("type")
            ),
        )

    return messages, json_logits_processor


def _build_thinking_processor(
    engine: BaseEngine,
    thinking_token_budget: int,
    *,
    inner: object | None = None,
    prompt_has_think_tag: bool = True,
) -> object | None:
    """Build a ThinkingAwareLogitsProcessor if the tokenizer has think tokens."""
    from .constrained.thinking_processor import ThinkingAwareLogitsProcessor

    tokenizer = _get_engine_tokenizer(engine)
    if tokenizer is None:
        return None

    # Resolve <think> and </think> token IDs from the tokenizer.
    try:
        start_ids = tokenizer.encode("<think>", add_special_tokens=False)
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    except Exception:
        logger.debug("Tokenizer cannot encode think tags; skipping thinking processor")
        return None

    if not start_ids or not end_ids:
        return None

    vocab_size = getattr(tokenizer, "vocab_size", 152064)

    proc = ThinkingAwareLogitsProcessor(
        start_token_ids=start_ids,
        end_token_ids=end_ids,
        thinking_token_budget=thinking_token_budget,
        inner=inner,
        vocab_size=vocab_size,
        prompt_has_think_tag=prompt_has_think_tag,
    )
    logger.info(
        "Thinking processor enabled: budget=%d, start=%s, end=%s",
        thinking_token_budget,
        start_ids,
        end_ids,
    )
    return proc


class _ThinkingAwareLogitsProcessor:
    """Wrap a ``JSONSchemaLogitsProcessor`` so JSON constraining only activates
    after the model emits ``</think>``, letting it reason freely first.

    Without this wrapper ``enable_thinking`` is forced to ``False`` when
    constrained decoding is active, which degrades output for thinking models
    (Qwen 3.5/3.6, DeepSeek-R1, etc.) — the model produces degenerated
    whitespace/brace loops instead of valid JSON because it was trained to
    think before answering.
    """

    def __init__(self, inner, prompt_has_think_tag: bool = False):
        self._inner = inner
        self._active = False
        # When the chat template already injects ``<think>\n`` into the prompt,
        # the model's output starts INSIDE the thinking block — so ``<think>``
        # will never appear in the generated tokens.  Pre-set ``_in_thinking``
        # to skip the 3-token detection and go straight to ``</think>`` scan.
        self._in_thinking: bool | None = True if prompt_has_think_tag else None
        self._waiting_for_json = False  # Waiting for { or [ before activating
        self._base_prompt_len: int | None = None
        self._json_scan_offset: int | None = None  # Absolute offset to start scanning
        self._tokenizer = inner._tokenizer

    def _scan_for_json_start(self, tokens_list, tokens, logits):
        """Scan generated tokens for the first ``{`` or ``[``.

        Scans from ``_json_scan_offset`` (set when entering the waiting
        phase) so that thinking-span tokens are never considered.  After
        50 tokens past the scan offset without a JSON start character the
        enforcer is force-activated as a safety net.
        """
        n = len(tokens_list)
        start = self._json_scan_offset
        scan_tokens = tokens_list[start:]
        for i in range(len(scan_tokens)):
            try:
                decoded = self._tokenizer.decode([scan_tokens[i]])
            except Exception:
                continue
            if any(c in decoded for c in ("{", "[")):
                self._active = True
                self._inner._prompt_len = start + i
                return self._inner(tokens, logits)
        # Safety: >50 tokens past scan offset without JSON start
        if len(scan_tokens) > 50:
            self._active = True
            self._inner._prompt_len = n
            return self._inner(tokens, logits)
        return logits

    def __call__(self, tokens, logits):
        if self._active:
            return self._inner(tokens, logits)

        tokens_list = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        if isinstance(tokens_list, int):
            tokens_list = [tokens_list]
        elif tokens_list and isinstance(tokens_list[0], list):
            tokens_list = tokens_list[0]

        n = len(tokens_list)
        if self._base_prompt_len is None:
            self._base_prompt_len = max(0, n - 1)

        # --- Phase: waiting for JSON start ({ or [) ---
        if self._waiting_for_json:
            return self._scan_for_json_start(tokens_list, tokens, logits)

        gen_tokens = tokens_list[self._base_prompt_len :]
        if not gen_tokens:
            return logits

        if self._in_thinking is None:
            # Decode the first few generated tokens to detect <think>.
            try:
                text = self._tokenizer.decode(gen_tokens[: min(3, len(gen_tokens))])
            except Exception:
                return logits
            if "<think>" in text:
                self._in_thinking = True
            elif len(gen_tokens) >= 3:
                # No <think> detected — scan for JSON start immediately.
                self._waiting_for_json = True
                self._json_scan_offset = self._base_prompt_len
                return self._scan_for_json_start(tokens_list, tokens, logits)
            else:
                return logits  # Wait for more tokens to decide.

        if self._in_thinking:
            # Check a small window of recent tokens for </think>.
            window = min(5, len(gen_tokens))
            try:
                recent = self._tokenizer.decode(gen_tokens[-window:])
            except Exception:
                return logits
            if "</think>" in recent:
                # Thinking ended — scan only tokens AFTER the thinking span.
                self._waiting_for_json = True
                self._json_scan_offset = n
                return self._scan_for_json_start(tokens_list, tokens, logits)

        return logits

    # Forward read-only attributes so callers can inspect the inner processor.
    @property
    def schema(self):
        return self._inner.schema

    @property
    def _disabled(self):
        return self._inner._disabled


def _prepare_chat_completion_invocation(
    engine: BaseEngine,
    request: ChatCompletionRequest,
    effective_max_tokens: int,
) -> PreparedChatInvocation:
    """Precompute messages, kwargs, and decoding constraints for chat completions."""
    messages, images, videos, audios, has_media = _prepare_chat_messages(
        engine, request.messages
    )
    response_format = request.response_format
    messages, json_logits_processor = _prepare_json_logits_processor(
        engine,
        messages,
        response_format,
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking_model=bool(_reasoning_parser),
    )

    rep_penalty = request.repetition_penalty
    chat_kwargs = {
        "max_tokens": effective_max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "top_k": request.top_k or 0,
        "min_p": request.min_p or 0.0,
        "presence_penalty": request.presence_penalty or 0.0,
        "repetition_penalty": request.repetition_penalty or 1.0,
    }
    if rep_penalty is not None:
        chat_kwargs["repetition_penalty"] = rep_penalty

    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        video_fps = getattr(request, "video_fps", None)
        if video_fps:
            chat_kwargs["video_fps"] = video_fps
        video_max_frames = getattr(request, "video_max_frames", None)
        if video_max_frames:
            chat_kwargs["video_max_frames"] = video_max_frames

    if request.specprefill is not None:
        chat_kwargs["specprefill"] = request.specprefill
    if request.specprefill_keep_pct is not None:
        chat_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct
    resolved_chat_template_kwargs = _resolve_chat_template_kwargs(
        request.chat_template_kwargs
    )
    if resolved_chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = resolved_chat_template_kwargs

    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking

    if request.tools and request.tool_choice != "none":
        template_tools = convert_tools_for_template(request.tools)
        template_tools, messages = _apply_forced_tool_choice(
            request.tool_choice, template_tools, messages, chat_kwargs
        )
        chat_kwargs["tools"] = template_tools

    parser_name = _tool_call_parser if _enable_auto_tool_choice else None
    merged_stop = get_parser_stop_tokens(parser_name, request.stop)
    if merged_stop:
        chat_kwargs["stop"] = merged_stop

    if json_logits_processor is not None:
        # Determine the *effective* thinking state: the request field, the
        # resolved chat_template_kwargs, or the server default can all inject
        # ``<think>`` into the rendered prompt independently.
        ctk = chat_kwargs.get("chat_template_kwargs") or {}
        effective_thinking = (
            request.enable_thinking is True or ctk.get("enable_thinking") is True
        )

        if _reasoning_parser and effective_thinking:
            # User explicitly requested thinking with constrained decoding
            # (via top-level enable_thinking or chat_template_kwargs).
            # Wrap the processor so the enforcer only activates after </think>.
            json_logits_processor = _ThinkingAwareLogitsProcessor(
                json_logits_processor, prompt_has_think_tag=True
            )
        else:
            # Suppress thinking so the model goes straight to JSON.
            # The template injects an empty <think></think> block and the
            # enforcer constrains output from the first token onward.
            # Force both top-level and chat_template_kwargs to prevent the
            # Jinja template from rendering an open <think> block.
            request.enable_thinking = False
            chat_kwargs["enable_thinking"] = False
            if "chat_template_kwargs" in chat_kwargs:
                chat_kwargs["chat_template_kwargs"]["enable_thinking"] = False
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]

    # Thinking-aware logits processor: cap reasoning tokens when a budget is set.
    # Only build when thinking is actually enabled for this request -- a CLI
    # default budget should not alter non-thinking requests.
    thinking_budget = request.thinking_token_budget or _default_thinking_token_budget
    enable_thinking = chat_kwargs.get("enable_thinking", True)
    if thinking_budget is not None and enable_thinking is not False:
        thinking_proc = _build_thinking_processor(
            engine,
            thinking_budget,
            inner=json_logits_processor,
            prompt_has_think_tag=bool(enable_thinking),
        )
        if thinking_proc is not None:
            # Replace the logits_processors list: the thinking processor wraps
            # the JSON processor as its inner delegate, so we don't double-add.
            chat_kwargs["logits_processors"] = [thinking_proc]

    return PreparedChatInvocation(
        messages=messages,
        chat_kwargs=chat_kwargs,
        response_format=response_format,
        json_logits_processor=json_logits_processor,
    )


def _prepare_anthropic_invocation(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    effective_max_tokens: int,
) -> PreparedChatInvocation:
    """Precompute messages, kwargs, and decoding constraints for Anthropic API."""
    messages, _, _, _, _ = _prepare_chat_messages(engine, openai_request.messages)
    response_format = openai_request.response_format
    messages, json_logits_processor = _prepare_json_logits_processor(
        engine,
        messages,
        response_format,
        tools=openai_request.tools,
        tool_choice=openai_request.tool_choice,
        log_context="Anthropic",
        thinking_model=bool(_reasoning_parser),
    )

    chat_kwargs = {
        "max_tokens": effective_max_tokens,
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
        "top_k": openai_request.top_k or 0,
        "min_p": openai_request.min_p or 0.0,
        "presence_penalty": openai_request.presence_penalty or 0.0,
        "repetition_penalty": openai_request.repetition_penalty or 1.0,
    }
    resolved_chat_template_kwargs = _resolve_chat_template_kwargs(
        openai_request.chat_template_kwargs
    )
    if resolved_chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = resolved_chat_template_kwargs

    if openai_request.tools and openai_request.tool_choice != "none":
        template_tools = convert_tools_for_template(openai_request.tools)
        template_tools, messages = _apply_forced_tool_choice(
            openai_request.tool_choice, template_tools, messages, chat_kwargs
        )
        chat_kwargs["tools"] = template_tools

    if json_logits_processor is not None:
        # Same logic as the OpenAI path: check both top-level and
        # chat_template_kwargs for an explicit thinking request.
        ctk = chat_kwargs.get("chat_template_kwargs") or {}
        effective_thinking = (
            openai_request.enable_thinking is True or ctk.get("enable_thinking") is True
        )

        if _reasoning_parser and effective_thinking:
            json_logits_processor = _ThinkingAwareLogitsProcessor(
                json_logits_processor, prompt_has_think_tag=True
            )
        else:
            chat_kwargs["enable_thinking"] = False
            if "chat_template_kwargs" in chat_kwargs:
                chat_kwargs["chat_template_kwargs"]["enable_thinking"] = False
        existing = chat_kwargs.get("logits_processors") or []
        chat_kwargs["logits_processors"] = list(existing) + [json_logits_processor]

    return PreparedChatInvocation(
        messages=messages,
        chat_kwargs=chat_kwargs,
        response_format=response_format,
        json_logits_processor=json_logits_processor,
    )


# Global MCP manager
_mcp_manager = None
_mcp_executor = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used

# Global reranker engine (lazy loaded)
_rerank_engine = None
_rerank_model_locked: str | None = None  # Set when --rerank-model is used

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

# Reasoning parser (for models like Qwen3, DeepSeek-R1)
_reasoning_parser = None  # ReasoningParser instance when enabled
_reasoning_parser_name: str | None = None

# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_parser_instance = None  # Instantiated parser
_responses_store: OrderedDict[str, dict] = OrderedDict()
_RESPONSES_STORE_MAX_SIZE: int = 1000

# Pattern to strip leaked tool call markup from content output.
# Safety net: the tool parser should consume these, but if it doesn't
# (e.g. malformed JSON, stray closing tags), strip them before emitting.
_TOOL_MARKUP_PATTERN = re.compile(r"</?tool_call>|</?tool_call_reasoning>")
_STREAMING_TOOL_MARKERS = (
    "<tool_call>",
    "<|tool_call>",
    "<function=",
    "[Calling tool:",
    "[TOOL_CALLS]",
    "<minimax:tool_call>",
    '<invoke name="',
)
_STREAMING_BARE_BRACKET_MARKER = re.compile(r"\[\w+\(\{")
_STREAMING_BARE_BRACKET_PARTIAL = re.compile(r"\[\w+\($")
_STREAMING_TOOL_MARKUP_SCAN_CHARS = 512


def _sanitize_log_text(value: object, limit: int | None = None) -> str:
    """Escape control characters before logging untrusted text."""
    text = str(value)
    escaped: list[str] = []
    for ch in text:
        if ch == "\n":
            escaped.append("\\n")
        elif ch == "\r":
            escaped.append("\\r")
        elif ch == "\t":
            escaped.append("\\t")
        elif ch.isprintable():
            escaped.append(ch)
        else:
            code = ord(ch)
            if code <= 0xFF:
                escaped.append(f"\\x{code:02x}")
            else:
                escaped.append(f"\\u{code:04x}")
    sanitized = "".join(escaped)
    if limit is not None and len(sanitized) > limit:
        return sanitized[:limit] + "..."
    return sanitized


def _log_and_raise_internal_error(log_prefix: str, exc: Exception, detail: str) -> None:
    """Log a sanitized exception string and raise a generic 500 response."""
    logger.error("%s: %s", log_prefix, _sanitize_log_text(exc, limit=500))
    raise HTTPException(status_code=500, detail=detail)


@dataclass
class RequestModelContext:
    """Request-scoped engine/lease context."""

    model_name: str
    engine: BaseEngine
    lease: ModelLease | None = None

    async def release(self) -> None:
        if self.lease is not None:
            lease = self.lease
            self.lease = None
            await lease.release()


def _list_available_model_names() -> list[str]:
    if _model_manager is not None:
        return _model_manager.registered_model_names
    return [_model_name] if _model_name else []


async def _acquire_request_model(request_model: str) -> RequestModelContext:
    """Acquire the model/engine that should serve this request."""
    _validate_model_name(request_model)

    if _model_manager is None:
        engine = get_engine()
        engine.preserve_native_tool_format = _detect_native_tool_support()
        return RequestModelContext(
            model_name=_model_name or request_model, engine=engine
        )

    try:
        lease = await _model_manager.acquire(request_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    lease.engine.preserve_native_tool_format = _detect_native_tool_support()
    return RequestModelContext(
        model_name=request_model,
        engine=lease.engine,
        lease=lease,
    )


async def _stream_with_model_context(
    context: RequestModelContext,
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Ensure model leases survive for the full streaming response."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await context.release()


def _build_tool_parser(engine: BaseEngine | None):
    """Create a fresh tool parser instance for a single request/stream."""
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return None

    if _tool_parser_instance is not None:
        if hasattr(_tool_parser_instance, "reset"):
            _tool_parser_instance.reset()
        return _tool_parser_instance

    parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
    tokenizer = getattr(engine, "tokenizer", None) if engine is not None else None
    return parser_cls(tokenizer)


def _build_reasoning_parser(engine: BaseEngine | None = None):
    """Create a fresh reasoning parser instance for a single request/stream."""
    tokenizer = getattr(engine, "tokenizer", None) if engine is not None else None
    if _reasoning_parser_name is not None:
        parser_cls = get_reasoning_parser(_reasoning_parser_name)
        try:
            return parser_cls(tokenizer)
        except TypeError:
            return parser_cls()
    if _reasoning_parser is None:
        return None
    try:
        return type(_reasoning_parser)(tokenizer)
    except TypeError:
        return type(_reasoning_parser)()


# Lifecycle startup coordination — an Event lets the lifecycle loop block
# efficiently instead of polling with short sleeps.  Created lazily so
# it binds to the correct event loop at runtime rather than import time.
#
# Important: the Event is bound to the loop that was running when
# _get_idle_unload_event() is first called.  In production this is always
# the single uvicorn event loop, but test fixtures must reset this to None
# between tests to avoid cross-loop contamination when pytest creates
# fresh loops per test.
_idle_unload_enabled: asyncio.Event | None = None


def _get_idle_unload_event() -> asyncio.Event:
    """Return the idle-unload gate event, creating it on first use.

    The returned Event is bound to the running loop at creation time.
    Reset ``_idle_unload_enabled`` to ``None`` when tearing down the
    server or switching event loops (e.g. in test fixtures).
    """
    global _idle_unload_enabled
    if _idle_unload_enabled is None:
        _idle_unload_enabled = asyncio.Event()
        _idle_unload_enabled.set()
    return _idle_unload_enabled


def _invalidate_tool_parser_cache(reason: str | None = None) -> None:
    """Drop cached parser state when the serving tokenizer changes."""
    global _tool_parser_instance

    if _tool_parser_instance is None:
        return

    if reason:
        logger.debug(f"Invalidating tool parser cache: {reason}")
    _tool_parser_instance = None


def _load_prefix_cache_from_disk(engine: BaseEngine | None = None) -> None:
    """Load prefix cache from disk during startup."""
    target_engine = engine or _engine
    if target_engine is None:
        return

    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = target_engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
    except Exception as e:
        logger.warning(
            "[lifespan] Failed to load cache from disk: %s",
            _sanitize_log_text(e, limit=500),
        )


def _save_prefix_cache_to_disk(engine: BaseEngine | None = None) -> None:
    """Save prefix cache to disk during shutdown."""
    target_engine = engine or _engine
    if target_engine is None:
        return

    try:
        d = _get_cache_dir()
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = target_engine.save_cache_to_disk(d)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(
            "[lifespan] Failed to save cache to disk: %s",
            _sanitize_log_text(e, limit=500),
        )


def _get_cache_dir() -> str:
    """Get cache persistence directory based on actual model path."""
    # Use _model_path (actual model path) not _model_name (which may be overridden
    # by --served-model-name). This ensures cache is shared regardless of served name.
    model_name = (
        _model_path if _model_path else (_model_name if _model_name else "default")
    )
    logger.info(
        f"[_get_cache_dir] _model_path={_model_path!r} type={type(_model_path)}"
    )
    # Sanitize model name for filesystem
    safe_name = str(model_name).replace("/", "--").replace("\\", "--")
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", safe_name
    )
    logger.info(f"[_get_cache_dir] cache_dir={cache_dir!r}")
    return cache_dir


def _build_engine(spec: ModelSpec) -> BaseEngine:
    """Construct an engine instance from a model spec without starting it."""
    if spec.use_batching:
        from .engine.batched import BatchedEngine

        logger.info(f"Preparing BatchedEngine for residency: {spec.model_name}")
        return BatchedEngine(
            model_name=spec.model_name,
            scheduler_config=spec.scheduler_config,
            stream_interval=spec.stream_interval,
            force_mllm=spec.force_mllm,
        )

    from .engine.simple import SimpleEngine

    logger.info(f"Preparing SimpleEngine for residency: {spec.model_name}")
    max_kv_size = (
        getattr(spec.scheduler_config, "max_kv_size", 0) if spec.scheduler_config else 0
    )
    return SimpleEngine(
        model_name=spec.model_name,
        force_mllm=spec.force_mllm,
        mtp=spec.mtp,
        prefill_step_size=spec.prefill_step_size,
        specprefill_enabled=spec.specprefill_enabled,
        specprefill_threshold=spec.specprefill_threshold,
        specprefill_keep_pct=spec.specprefill_keep_pct,
        specprefill_draft_model=spec.specprefill_draft_model,
        max_kv_size=max_kv_size,
    )


async def _engine_factory(spec: ModelSpec) -> BaseEngine:
    """Async engine factory used by the residency manager."""
    return _build_engine(spec)


async def _run_blocking_engine_cache_io(io_fn, engine: BaseEngine) -> None:
    """Run blocking cache persistence off the event loop.

    If the caller is canceled while waiting, finish the in-flight thread before
    propagating cancellation so engine state cannot keep mutating in the
    background after lifecycle cleanup has started.
    """
    task = asyncio.create_task(asyncio.to_thread(io_fn, engine))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with suspend_cancellation():
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        raise


async def _restore_engine_state(spec: ModelSpec, engine: BaseEngine) -> None:
    """Restore engine-local state, such as prefix cache, after a cold load."""
    if hasattr(engine, "load_cache_from_disk"):
        await _run_blocking_engine_cache_io(_load_prefix_cache_from_disk, engine)


async def _persist_engine_state(spec: ModelSpec, engine: BaseEngine) -> None:
    """Persist engine-local state before an idle unload or shutdown unload."""
    if hasattr(engine, "save_cache_to_disk"):
        await _run_blocking_engine_cache_io(_save_prefix_cache_to_disk, engine)


def _activate_engine(engine: BaseEngine | None) -> BaseEngine | None:
    """Set the global engine pointer and refresh parser-sensitive state."""
    global _engine

    if engine is not _engine:
        _invalidate_tool_parser_cache("resident engine changed")
    _engine = engine
    if _engine is not None:
        _engine.preserve_native_tool_format = _detect_native_tool_support()
    return _engine


def _sync_engine_from_residency() -> BaseEngine | None:
    """Sync the global engine pointer from the residency manager state.

    Safety: all callers run on the single-threaded asyncio event loop and do not
    yield between reading the residency state and writing ``_engine``, so no
    additional locking is required.
    """
    if _residency_manager is None or _default_model_key is None:
        return _engine

    return _activate_engine(_residency_manager.get_engine(_default_model_key))


def _get_lifecycle_status() -> dict | None:
    """Get lifecycle status for the default resident if lifecycle is enabled."""
    if _residency_manager is None or _default_model_key is None:
        return None
    return _residency_manager.get_status(_default_model_key)


def _public_lifecycle_status(lifecycle: dict | None) -> dict | None:
    """Return residency status safe for unauthenticated public endpoints."""
    if lifecycle is None:
        return None
    public = dict(lifecycle)
    if _model_name:
        public["model_name"] = _model_name
    # Surface a generic error indicator without exposing raw exception text.
    if "last_error" in public:
        public["last_error"] = (
            "model_load_failed" if public["last_error"] is not None else None
        )
    return public


async def _lifecycle_loop() -> None:
    """Background idle-unload loop for the default resident."""
    while True:
        if _residency_manager is None or _default_model_key is None:
            await asyncio.sleep(1.0)
            continue

        # Block until idle-unload is enabled instead of polling.
        await _get_idle_unload_event().wait()

        try:
            await _residency_manager.unload_if_idle(_default_model_key)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Idle unload iteration failed")
        finally:
            _sync_engine_from_residency()

        sleep_for = min(_auto_unload_idle_seconds / 2, 5.0)
        await asyncio.sleep(sleep_for)


async def _acquire_default_engine(*, count_activity: bool = True) -> BaseEngine:
    """Acquire the default engine, auto-loading via the residency manager if needed."""
    if _residency_manager is None or _default_model_key is None:
        return get_engine()

    if count_activity:
        engine = await _residency_manager.acquire(_default_model_key)
    else:
        engine = await _residency_manager.acquire(
            _default_model_key,
            count_activity=False,
        )
    activated_engine = _activate_engine(engine)
    if activated_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return activated_engine


async def _release_default_engine(*, count_activity: bool = True) -> None:
    """Release the default engine after request processing."""
    if _residency_manager is None or _default_model_key is None:
        return

    if count_activity:
        await _residency_manager.release(_default_model_key)
    else:
        await _residency_manager.release(_default_model_key, count_activity=False)
    _sync_engine_from_residency()


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager, _model_manager, _lifecycle_task, _lifespan_active
    primary_exc: BaseException | None = None
    try:
        _get_idle_unload_event().clear()

        # Startup: ensure resident is loaded on the serving event loop when lifecycle
        # management is enabled, unless lazy startup is requested.
        if _residency_manager is not None and _default_model_key is not None:
            if not _lazy_load_model:
                await _residency_manager.ensure_loaded(_default_model_key)
            _sync_engine_from_residency()
        elif (
            _engine is not None and hasattr(_engine, "_loaded") and not _engine._loaded
        ):
            await _engine.start()
        if _model_manager is not None:
            await _model_manager.preload()

        # Load persisted cache from disk (AFTER engine start — AsyncEngineCore must exist)
        if (
            _residency_manager is None
            and _engine is not None
            and hasattr(_engine, "load_cache_from_disk")
        ):
            _load_prefix_cache_from_disk()

        # Warm up prefix cache with user-provided prompts (AFTER disk cache load,
        # so any already-persisted entries are preserved and warm-up only fills
        # gaps).
        if (
            _warm_prompts_path
            and _engine is not None
            and hasattr(_engine, "stream_chat")
        ):
            try:
                from vllm_mlx.prompt_warmup import load_warmup_file, warm_prefix_cache

                prompts = load_warmup_file(_warm_prompts_path)
                logger.info(
                    "[lifespan] Warming prefix cache with %d prompts from %s",
                    len(prompts),
                    _warm_prompts_path,
                )
                result = await warm_prefix_cache(_engine, prompts)
                logger.info(
                    "[lifespan] Warm-up done (%s): %d completed, %d skipped, %d prompt tokens in %.1fs",
                    result.get("mode", "?"),
                    result["count"],
                    result["skipped"],
                    result["total_prompt_tokens"],
                    result["elapsed_ms"] / 1000,
                )
            except Exception as e:
                logger.warning(
                    "[lifespan] Warm-up failed: %s",
                    _sanitize_log_text(e, limit=500),
                )

        if _residency_manager is not None and _auto_unload_idle_seconds > 0:
            _lifecycle_task = asyncio.create_task(_lifecycle_loop())

        # Initialize MCP if config provided
        mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
        if mcp_config:
            await init_mcp(mcp_config)

        _get_idle_unload_event().set()
        _lifespan_active = True
        yield
    except BaseException as exc:
        primary_exc = exc

    cleanup_exc: BaseException | None = None
    try:
        # Shutdown: Save cache to disk BEFORE stopping engine
        if (
            _residency_manager is None
            and _engine is not None
            and hasattr(_engine, "save_cache_to_disk")
        ):
            _save_prefix_cache_to_disk()

        # Shutdown: Close MCP connections and stop engine
        if _lifecycle_task is not None:
            _lifecycle_task.cancel()
            with suppress(asyncio.CancelledError):
                await _lifecycle_task
            _lifecycle_task = None
        if _mcp_manager is not None:
            await _mcp_manager.stop()
            logger.info("MCP manager stopped")
        if _residency_manager is not None:
            await _residency_manager.shutdown()
            _sync_engine_from_residency()
            logger.info("Lifecycle manager shut down")
        elif _engine is not None:
            await _engine.stop()
            _engine = None
            logger.info("Engine stopped")
        if _model_manager is not None:
            await _model_manager.shutdown()
            logger.info("Model manager stopped")
    except BaseException as exc:
        cleanup_exc = exc
    finally:
        _get_idle_unload_event().set()
        _lifespan_active = False

    if primary_exc is not None:
        if cleanup_exc is not None:
            logger.error(
                "Lifecycle cleanup failed while preserving the original exception",
                exc_info=(
                    type(cleanup_exc),
                    cleanup_exc,
                    cleanup_exc.__traceback__,
                ),
            )
        raise primary_exc

    if cleanup_exc is not None:
        raise cleanup_exc


app = FastAPI(
    title="vllm-mlx API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.2.9",
    lifespan=lifespan,
)

security = HTTPBearer(auto_error=False)


def _metrics_result_from_status(status_code: int) -> str:
    """Map HTTP-ish status codes to low-cardinality inference results."""
    if status_code == 499:
        return "client_closed"
    if status_code == 504:
        return "timeout"
    if status_code >= 500:
        return "error"
    return "success"


def _metrics_path_for_request(request: Request) -> str:
    """Prefer route templates over raw URLs to keep metrics cardinality bounded."""
    route = request.scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if path:
            return str(path)
    for candidate in app.router.routes:
        match, _ = candidate.matches(request.scope)
        if match in (Match.FULL, Match.PARTIAL):
            path = getattr(candidate, "path", None)
            if path:
                return str(path)
    return "__unmatched__"


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Capture generic HTTP request metrics when enabled."""
    if not _metrics.enabled:
        return await call_next(request)

    method = request.method
    path = _metrics_path_for_request(request)
    if path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    _metrics.observe_http_start(method=method, path=path)
    try:
        response = await call_next(request)
    except Exception:
        _metrics.observe_http_finish(
            method=method,
            path=path,
            status_code=500,
            duration=time.perf_counter() - start_time,
        )
        raise

    _metrics.observe_http_finish(
        method=method,
        path=path,
        status_code=response.status_code,
        duration=time.perf_counter() - start_time,
    )
    return response


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, requests_per_minute: int = 60, enabled: bool = False):
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self.window_size = 60.0  # 1 minute window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed for client.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if not self.enabled:
            return True, 0

        current_time = time.time()
        window_start = current_time - self.window_size

        with self._lock:
            # Clean old requests outside window
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > window_start
            ]

            # Check rate limit
            if len(self._requests[client_id]) >= self.requests_per_minute:
                # Calculate retry-after
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_size - current_time) + 1
                return False, max(1, retry_after)

            # Record this request
            self._requests[client_id].append(current_time)
            return True, 0


# Global rate limiter (disabled by default)
_rate_limiter = RateLimiter(requests_per_minute=60, enabled=False)


async def check_rate_limit(request: Request):
    """Rate limiting dependency."""
    # Use API key as client ID if available, otherwise use IP
    client_id = request.headers.get(
        "Authorization", request.client.host if request.client else "unknown"
    )

    allowed, retry_after = _rate_limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if authentication is enabled."""
    global _auth_warning_logged

    if _api_key is None:
        # Log warning once about running without authentication
        if not _auth_warning_logged:
            logger.warning(
                "SECURITY WARNING: Server running without API key authentication. "
                "Anyone can access the API. Use --api-key to enable authentication."
            )
            _auth_warning_logged = True
        return True  # No auth required

    if credentials is None:
        raise HTTPException(status_code=401, detail="API key required")
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, _api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_engine() -> BaseEngine:
    """Get the loaded engine, raising error if not loaded."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _engine


def _coerce_tool_arguments(
    arguments_json: str, tool_name: str, tools: list[dict] | None
) -> str:
    """
    Coerce tool call arguments to match the tool schema.

    If a schema field expects "string" but the model produced an object/array,
    JSON-stringify the value. This fixes a common LLM failure mode where models
    output raw JSON objects instead of JSON strings for file content, etc.
    """
    if not tools:
        return arguments_json

    # Find the schema for this tool
    schema = None
    for tool in tools:
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == tool_name:
            schema = tool["function"].get("parameters", {})
            break

    if not schema or "properties" not in schema:
        return arguments_json

    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return arguments_json

    if not isinstance(arguments, dict):
        return arguments_json

    properties = schema.get("properties", {})
    changed = False

    for key, value in arguments.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type == "string" and isinstance(value, (dict, list)):
                arguments[key] = json.dumps(value, ensure_ascii=False, indent=2)
                changed = True

    if changed:
        return json.dumps(arguments, ensure_ascii=False)

    return arguments_json


def _validate_model_name(request_model: str) -> None:
    """Validate that the request model name matches the served model."""
    if _model_manager is not None:
        if not _model_manager.has_model(request_model):
            available = ", ".join(f"`{name}`" for name in _list_available_model_names())
            raise HTTPException(
                status_code=404,
                detail=(
                    f"The model `{request_model}` does not exist. "
                    f"Available models: {available}"
                ),
            )
        return

    if _model_name and request_model != _model_name:
        raise HTTPException(
            status_code=404,
            detail=f"The model `{request_model}` does not exist. "
            f"Available model: `{_model_name}`",
        )


def _get_engine_tokenizer(engine: BaseEngine | None) -> object | None:
    """Return tokenizer-like parser state from the active engine."""
    if engine is None:
        return None
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return getattr(engine, "_tokenizer", None)


def _get_or_init_tool_parser(engine: BaseEngine | None = None):
    """Return the cached tool parser, initializing it from the given engine."""
    global _tool_parser_instance

    if _tool_parser_instance is None:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        tokenizer = _get_engine_tokenizer(engine if engine is not None else _engine)
        _tool_parser_instance = parser_cls(tokenizer)
        logger.info(f"Initialized tool call parser: {_tool_call_parser}")

    return _tool_parser_instance


def _parse_tool_calls_with_parser(
    output_text: str,
    request: ChatCompletionRequest | None = None,
    engine: BaseEngine | None = None,
) -> tuple[str, list | None]:
    """
    Parse tool calls from model output using the configured parser.

    If --enable-auto-tool-choice is set with --tool-call-parser, uses the
    selected parser. Otherwise falls back to the generic parse_tool_calls.

    Args:
        output_text: The model output text
        request: The original request (for context)
        engine: The request-local engine to use for parser initialization

    Returns:
        Tuple of (cleaned_text, tool_calls)
    """
    global _tool_parser_instance

    request_dict = request.model_dump() if request else None

    # tool_choice="none" means never return tool calls — skip all parsing
    if request is not None:
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is None and request_dict:
            tool_choice = request_dict.get("tool_choice")
        if tool_choice == "none":
            return output_text, None

    # If auto tool choice is not enabled, use the generic parser
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return parse_tool_calls(output_text, request_dict)

    # Initialize parser if needed
    if _tool_parser_instance is None:
        try:
            _get_or_init_tool_parser(engine)
        except Exception as e:
            logger.warning(
                "Failed to initialize tool parser '%s': %s",
                _tool_call_parser,
                _sanitize_log_text(e, limit=500),
            )
            logger.warning("Falling back to generic parser")
            return parse_tool_calls(output_text, request_dict)

    # Use the configured parser
    try:
        # Reset parser state between requests
        _tool_parser_instance.reset()
        result = _tool_parser_instance.extract_tool_calls(output_text, request_dict)
        if result.tools_called:
            tools = request_dict.get("tools") if request_dict else None
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=_coerce_tool_arguments(
                            tc["arguments"], tc["name"], tools
                        ),
                    ),
                )
                for tc in result.tool_calls
            ]
            return result.content or "", tool_calls
        else:
            # Fallback: specific parser didn't find tool calls,
            # try generic parser which handles more formats (e.g. Nemotron XML)
            return parse_tool_calls(output_text, request_dict)
    except Exception as e:
        logger.warning("Tool parser error: %s", _sanitize_log_text(e, limit=500))
        return parse_tool_calls(output_text, request_dict)


def _new_response_item_id(prefix: str) -> str:
    """Generate stable OpenAI-style item ids."""
    return f"{prefix}_{uuid.uuid4().hex}"


def _response_content_to_text(content) -> str:
    """Normalize Responses API content items into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    text_parts = []
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type")
            text = part.get("text", "")
        else:
            part_type = getattr(part, "type", None)
            text = getattr(part, "text", "")
        if part_type in {"text", "input_text", "output_text"}:
            text_parts.append(text)
    return "\n".join(part for part in text_parts if part)


def _responses_tools_to_chat_tools(
    tools: list[ResponseFunctionTool | dict],
) -> tuple[list[dict] | None, list[str]]:
    """Convert supported Responses tools and report unsupported tool types."""
    if not tools:
        return None, []

    supported: list[dict] = []
    unsupported: list[str] = []

    for tool in tools:
        if isinstance(tool, ResponseFunctionTool):
            tool_type = tool.type
            tool_name = tool.name
            tool_description = tool.description or ""
            tool_parameters = tool.parameters
        elif isinstance(tool, dict):
            tool_type = tool.get("type", "unknown")
            tool_name = tool.get("name", "")
            tool_description = tool.get("description", "")
            tool_parameters = tool.get("parameters", {})
        else:
            unsupported.append(type(tool).__name__)
            continue

        if tool_type == "function":
            supported.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_parameters
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        else:
            unsupported.append(tool_type)

    return supported or None, unsupported


def _responses_input_to_chat_messages(request: ResponsesRequest) -> list[dict]:
    """Convert Responses API input items into chat-completions-style messages."""
    messages: list[dict] = []

    if request.previous_response_id:
        previous = _responses_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response `{request.previous_response_id}` not found",
            )
        messages.extend(copy.deepcopy(previous["messages"]))

    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
        return messages

    for item in request.input:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "message":
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                messages.append(
                    {
                        "role": role,
                        "content": _response_content_to_text(item.get("content")),
                    }
                )
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get(
                                    "call_id", _new_response_item_id("call")
                                ),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments", ""),
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )
            elif item_type == "reasoning":
                parts = item.get("content", [])
                reasoning_text = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                )
                if reasoning_text:
                    messages.append({"role": "assistant", "content": reasoning_text})
            else:
                logger.info(
                    "Skipping unsupported Responses input item type %r", item_type
                )
            continue

        if isinstance(item, ResponseMessageItem):
            role = item.role
            if role == "developer":
                role = "system"
            messages.append(
                {
                    "role": role,
                    "content": _response_content_to_text(item.content),
                }
            )
        elif isinstance(item, ResponseFunctionCallItem):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                    ],
                }
            )
        elif isinstance(item, ResponseFunctionCallOutputItem):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": item.output,
                }
            )
        elif isinstance(item, ResponseReasoningItem):
            reasoning_text = "\n".join(part.text for part in (item.content or []))
            if reasoning_text:
                messages.append({"role": "assistant", "content": reasoning_text})
        else:
            logger.info(
                "Skipping unsupported Responses input item type %r",
                getattr(item, "type", type(item).__name__),
            )

    return messages


def _responses_request_to_new_persisted_messages(
    request: ResponsesRequest,
) -> list[dict]:
    """Persist only the current request's replayable input items."""
    request_without_history = request.model_copy(
        update={"previous_response_id": None, "instructions": None},
        deep=True,
    )
    return _responses_input_to_chat_messages(request_without_history)


def _responses_request_to_persisted_messages(request: ResponsesRequest) -> list[dict]:
    """Persist replayable history for chained previous_response_id requests.

    Responses `instructions` are intentionally not replayed across
    `previous_response_id`, but replayable message items are.
    """
    messages: list[dict] = []
    if request.previous_response_id:
        previous = _responses_store.get(request.previous_response_id)
        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response `{request.previous_response_id}` not found",
            )
        messages.extend(copy.deepcopy(previous["messages"]))
    messages.extend(_responses_request_to_new_persisted_messages(request))
    return messages


def _responses_request_to_chat_request(
    request: ResponsesRequest,
) -> ChatCompletionRequest:
    """Build a ChatCompletionRequest from a ResponsesRequest."""
    if request.text.format.type == "json_object":
        raise HTTPException(
            status_code=400,
            detail="Responses text.format.type='json_object' is not supported on this backend",
        )
    if request.reasoning is not None:
        logger.debug("Ignoring reasoning configuration (not supported on this backend)")

    tools, unsupported_tools = _responses_tools_to_chat_tools(request.tools)
    messages = _responses_input_to_chat_messages(request)
    if unsupported_tools:
        tool_list = ", ".join(sorted(set(unsupported_tools)))
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "The following requested tool types are not available on this "
                    f"backend: {tool_list}. Do not call them."
                ),
            },
        )

    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
    merged_system_content = "\n\n".join(
        str(msg.get("content", "")).strip()
        for msg in system_messages
        if str(msg.get("content", "")).strip()
    )
    messages = (
        [{"role": "system", "content": merged_system_content}]
        if merged_system_content
        else []
    ) + non_system_messages

    return ChatCompletionRequest(
        model=request.model,
        messages=[Message(**msg) for msg in messages],
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=False,
        tools=tools,
        tool_choice=request.tool_choice,
        chat_template_kwargs=request.chat_template_kwargs,
    )


def _build_responses_output_items(
    text: str | None,
    reasoning: str | None,
    tool_calls: list[ToolCall] | None,
) -> list[ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem]:
    """Convert parsed assistant output into Responses API output items."""
    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = []

    if reasoning:
        output_items.append(
            ResponseReasoningItem(
                id=_new_response_item_id("rs"),
                content=[ResponseReasoningTextPart(text=reasoning)],
            )
        )

    if text:
        output_items.append(
            ResponseMessageItem(
                id=_new_response_item_id("msg"),
                role="assistant",
                content=[ResponseTextContentPart(type="output_text", text=text)],
            )
        )

    for tool_call in tool_calls or []:
        output_items.append(
            ResponseFunctionCallItem(
                id=_new_response_item_id("fc"),
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
        )

    return output_items


def _response_output_items_to_chat_messages(output_items: list) -> list[dict]:
    """Persist assistant output in chat-completions form for previous_response_id."""
    assistant_text_parts: list[str] = []
    assistant_tool_calls: list[dict] = []

    for item in output_items:
        if isinstance(item, ResponseMessageItem):
            assistant_text_parts.append(_response_content_to_text(item.content))
        elif isinstance(item, ResponseFunctionCallItem):
            assistant_tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )

    if not assistant_text_parts and not assistant_tool_calls:
        return []

    return [
        {
            "role": "assistant",
            "content": "".join(assistant_text_parts),
            "tool_calls": assistant_tool_calls or None,
        }
    ]


def _build_response_object(
    request: ResponsesRequest,
    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
    response_id: str | None = None,
) -> ResponseObject:
    """Build a full Responses API object."""
    response = ResponseObject(
        id=response_id or _new_response_item_id("resp"),
        model=_model_name or request.model,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        metadata=request.metadata,
        output=output_items,
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_p=_resolve_top_p(request.top_p),
        temperature=_resolve_temperature(request.temperature),
        truncation=request.truncation,
        user=request.user,
        store=request.store,
        usage=ResponsesUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    if finish_reason == "length":
        response.status = "incomplete"
        response.incomplete_details = ResponseIncompleteDetails(
            reason="max_output_tokens"
        )
    return response


def _prepare_responses_request(
    request: ResponsesRequest,
    *,
    validate_remote_media: bool = True,
) -> tuple[BaseEngine, ChatCompletionRequest, list[dict], dict]:
    """Prepare a Responses request for execution on the chat engine."""
    _validate_model_name(request.model)
    engine = get_engine()
    chat_request = _responses_request_to_chat_request(request)

    if chat_request.messages:
        logger.info(
            f"[REQUEST] POST /v1/responses stream={request.stream} "
            f"model={request.model!r} items="
            f"{len(request.input) if isinstance(request.input, list) else 1} "
            f"tools={len(request.tools)}"
        )

    if validate_remote_media:
        _validate_remote_media_urls(chat_request.messages)

    messages, images, videos, audios = extract_multimodal_content(
        chat_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": chat_request.max_tokens or _default_max_tokens,
        "temperature": _resolve_temperature(chat_request.temperature),
        "top_p": _resolve_top_p(chat_request.top_p),
    }
    resolved_chat_template_kwargs = _resolve_chat_template_kwargs(
        chat_request.chat_template_kwargs
    )
    if resolved_chat_template_kwargs:
        chat_kwargs["chat_template_kwargs"] = resolved_chat_template_kwargs
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(chat_request.tools)
    if images:
        chat_kwargs["images"] = images
    if videos:
        chat_kwargs["videos"] = videos

    return engine, chat_request, messages, chat_kwargs


def _prepare_streaming_responses_request(
    request: ResponsesRequest,
) -> tuple[BaseEngine, ChatCompletionRequest, list[dict], dict]:
    """Prepare a streaming Responses request after eager URL validation."""
    return _prepare_responses_request(request, validate_remote_media=False)


async def _run_responses_request(
    request: ResponsesRequest,
    raw_request: Request,
) -> tuple[ResponseObject | None, list[dict]]:
    """Execute a Responses API request against the backend chat engine."""
    engine, chat_request, messages, chat_kwargs = _prepare_responses_request(request)

    timeout = _default_timeout
    output = await _wait_with_disconnect(
        engine.chat(messages=messages, **chat_kwargs),
        raw_request,
        timeout=timeout,
    )
    if output is None:
        return None, []

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, chat_request)
    reasoning_text = None
    if _reasoning_parser:
        reasoning_text, remaining_text = _reasoning_parser.extract_reasoning(
            output.text
        )
        if not tool_calls:
            cleaned_text = remaining_text
        else:
            # Tool parser already stripped tool markup from cleaned_text,
            # but reasoning markers (e.g. <|channel>thought...<channel|>)
            # remain. Run reasoning parser on cleaned_text to strip them.
            _, cleaned_text = _reasoning_parser.extract_reasoning(cleaned_text or "")

    output_items = _build_responses_output_items(
        clean_output_text(cleaned_text) if cleaned_text else None,
        reasoning_text,
        tool_calls,
    )
    response_object = _build_response_object(
        request=request,
        output_items=output_items,
        prompt_tokens=output.prompt_tokens,
        completion_tokens=output.completion_tokens,
        finish_reason=output.finish_reason,
    )

    persisted_messages = _responses_request_to_persisted_messages(request)
    persisted_messages.extend(_response_output_items_to_chat_messages(output_items))
    if request.store:
        _responses_store[response_object.id] = {
            "messages": copy.deepcopy(persisted_messages),
            "response": response_object.model_copy(deep=True),
        }
        while len(_responses_store) > _RESPONSES_STORE_MAX_SIZE:
            _responses_store.popitem(last=False)

    return response_object, persisted_messages


async def _stream_responses_request(request: ResponsesRequest) -> AsyncIterator[str]:
    """Execute a Responses API request and stream SSE events incrementally."""
    engine, chat_request, messages, chat_kwargs = _prepare_streaming_responses_request(
        request
    )

    response_id = _new_response_item_id("resp")
    sequence = 1
    base_response = _build_response_object(
        request=request,
        output_items=[],
        prompt_tokens=0,
        completion_tokens=0,
        finish_reason=None,
        response_id=response_id,
    )
    base_response.status = "in_progress"
    base_response.usage = None

    yield _responses_sse_event(
        "response.created",
        ResponseCreatedEvent(sequence_number=sequence, response=base_response),
    )
    sequence += 1
    yield _responses_sse_event(
        "response.in_progress",
        ResponseInProgressEvent(sequence_number=sequence, response=base_response),
    )
    sequence += 1

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = None
    last_output = None
    raw_accumulated_text = ""
    accumulated_text = ""
    accumulated_reasoning = ""

    text_item_id: str | None = None
    text_output_index: int | None = None
    reasoning_item_id: str | None = None
    reasoning_output_index: int | None = None
    next_output_index = 0

    def _start_text_item() -> list[str]:
        nonlocal text_item_id, text_output_index, next_output_index, sequence
        events: list[str] = []
        if text_item_id is None:
            text_item_id = _new_response_item_id("msg")
            text_output_index = next_output_index
            next_output_index += 1
            events.append(
                _responses_sse_event(
                    "response.output_item.added",
                    ResponseOutputItemAddedEvent(
                        sequence_number=sequence,
                        output_index=text_output_index,
                        item=ResponseMessageItem(
                            id=text_item_id,
                            role="assistant",
                            status="in_progress",
                            content=[],
                        ),
                    ),
                )
            )
            sequence += 1
            events.append(
                _responses_sse_event(
                    "response.content_part.added",
                    ResponseContentPartAddedEvent(
                        sequence_number=sequence,
                        item_id=text_item_id,
                        output_index=text_output_index,
                        content_index=0,
                        part=ResponseTextContentPart(type="output_text", text=""),
                    ),
                )
            )
            sequence += 1
        return events

    def _start_reasoning_item() -> list[str]:
        nonlocal reasoning_item_id, reasoning_output_index, next_output_index, sequence
        events: list[str] = []
        if reasoning_item_id is None:
            reasoning_item_id = _new_response_item_id("rs")
            reasoning_output_index = next_output_index
            next_output_index += 1
            events.append(
                _responses_sse_event(
                    "response.output_item.added",
                    ResponseOutputItemAddedEvent(
                        sequence_number=sequence,
                        output_index=reasoning_output_index,
                        item=ResponseReasoningItem(
                            id=reasoning_item_id,
                            status="in_progress",
                            content=[],
                        ),
                    ),
                )
            )
            sequence += 1
            events.append(
                _responses_sse_event(
                    "response.content_part.added",
                    ResponseContentPartAddedEvent(
                        sequence_number=sequence,
                        item_id=reasoning_item_id,
                        output_index=reasoning_output_index,
                        content_index=0,
                        part=ResponseReasoningTextPart(text=""),
                    ),
                )
            )
            sequence += 1
        return events

    if _reasoning_parser:
        _reasoning_parser.reset_state()

    global _tool_parser_instance
    tool_parser = None
    tool_accumulated_text = ""
    tool_markup_possible = False
    if _enable_auto_tool_choice and _tool_call_parser:
        if _tool_parser_instance is None:
            try:
                parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
                tokenizer = None
                if _engine is not None and hasattr(_engine, "_tokenizer"):
                    tokenizer = _engine._tokenizer
                _tool_parser_instance = parser_cls(tokenizer)
                logger.info(
                    "Initialized tool call parser for responses streaming: %s",
                    _tool_call_parser,
                )
            except Exception as e:
                logger.warning(
                    "Failed to init tool parser for responses streaming: %s", e
                )
        if _tool_parser_instance is not None:
            tool_parser = _tool_parser_instance
            tool_parser.reset()

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        last_output = output
        finish_reason = output.finish_reason
        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        delta_text = output.new_text or ""
        if not delta_text:
            continue

        previous_text = raw_accumulated_text
        raw_accumulated_text += delta_text

        if _reasoning_parser:
            delta_msg = _reasoning_parser.extract_reasoning_streaming(
                previous_text, raw_accumulated_text, delta_text
            )
            if delta_msg is None:
                continue

            if delta_msg.reasoning:
                for event in _start_reasoning_item():
                    yield event
                accumulated_reasoning += delta_msg.reasoning
                yield _responses_sse_event(
                    "response.reasoning_text.delta",
                    ResponseReasoningTextDeltaEvent(
                        sequence_number=sequence,
                        item_id=reasoning_item_id,
                        output_index=reasoning_output_index,
                        content_index=0,
                        delta=delta_msg.reasoning,
                    ),
                )
                sequence += 1

            if delta_msg.content:
                for event in _start_text_item():
                    yield event
                accumulated_text += delta_msg.content
                yield _responses_sse_event(
                    "response.output_text.delta",
                    ResponseOutputTextDeltaEvent(
                        sequence_number=sequence,
                        item_id=text_item_id,
                        output_index=text_output_index,
                        content_index=0,
                        delta=delta_msg.content,
                    ),
                )
                sequence += 1
            continue

        content = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
        if tool_parser and delta_text:
            # Fast path: skip parsing until a tool-markup marker appears.
            # Use _streaming_tool_markup_possible to catch all supported
            # shapes (<tool_call>, <function=, [Calling tool:, [TOOL_CALLS],
            # bare bracket [func({...})], etc.) — the old `"<" not in` check
            # missed bracket formats and let Qwen3.6-style tool calls leak.
            if (
                not tool_markup_possible
                and not _streaming_tool_markup_possible_after_delta(
                    tool_accumulated_text, delta_text
                )
            ):
                tool_accumulated_text += delta_text
            else:
                if not tool_markup_possible:
                    tool_markup_possible = True
                tool_result = tool_parser.extract_tool_calls_streaming(
                    tool_accumulated_text,
                    tool_accumulated_text + delta_text,
                    delta_text,
                )
                tool_accumulated_text += delta_text
                if tool_result is None:
                    continue
                if "tool_calls" in tool_result:
                    continue
                content = tool_result.get("content", "")

        if not content:
            continue

        for event in _start_text_item():
            yield event
        accumulated_text += content
        yield _responses_sse_event(
            "response.output_text.delta",
            ResponseOutputTextDeltaEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                delta=content,
            ),
        )
        sequence += 1

    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        raw_accumulated_text, chat_request
    )
    final_text = accumulated_text
    if cleaned_text is not None and not final_text and not tool_calls:
        final_text = clean_output_text(cleaned_text)

    reasoning_item = None
    if reasoning_item_id is not None:
        reasoning_item = ResponseReasoningItem(
            id=reasoning_item_id,
            status="completed",
            content=[ResponseReasoningTextPart(text=accumulated_reasoning)],
        )
        yield _responses_sse_event(
            "response.reasoning_text.done",
            ResponseReasoningTextDoneEvent(
                sequence_number=sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                content_index=0,
                text=accumulated_reasoning,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                sequence_number=sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index,
                content_index=0,
                part=reasoning_item.content[0],
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=reasoning_output_index,
                item=reasoning_item,
            ),
        )
        sequence += 1

    text_item = None
    if text_item_id is not None or final_text:
        if text_item_id is None:
            for event in _start_text_item():
                yield event
        text_item = ResponseMessageItem(
            id=text_item_id,
            role="assistant",
            status="completed",
            content=[ResponseTextContentPart(type="output_text", text=final_text)],
        )
        yield _responses_sse_event(
            "response.output_text.done",
            ResponseOutputTextDoneEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                text=final_text,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                sequence_number=sequence,
                item_id=text_item_id,
                output_index=text_output_index,
                content_index=0,
                part=text_item.content[0],
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=text_output_index,
                item=text_item,
            ),
        )
        sequence += 1

    function_call_items: list[ResponseFunctionCallItem] = []
    for tool_call in tool_calls or []:
        output_index = next_output_index
        next_output_index += 1
        item = ResponseFunctionCallItem(
            id=_new_response_item_id("fc"),
            call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        function_call_items.append(item)
        yield _responses_sse_event(
            "response.output_item.added",
            ResponseOutputItemAddedEvent(
                sequence_number=sequence,
                output_index=output_index,
                item=item.model_copy(update={"status": "in_progress"}),
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.function_call_arguments.delta",
            ResponseFunctionCallArgumentsDeltaEvent(
                sequence_number=sequence,
                item_id=item.id,
                output_index=output_index,
                delta=item.arguments,
            ),
        )
        sequence += 1
        yield _responses_sse_event(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                sequence_number=sequence,
                output_index=output_index,
                item=item,
            ),
        )
        sequence += 1

    output_items: list[
        ResponseMessageItem | ResponseReasoningItem | ResponseFunctionCallItem
    ] = []
    if reasoning_item is not None:
        output_items.append(reasoning_item)
    if text_item is not None:
        output_items.append(text_item)
    output_items.extend(function_call_items)

    response_object = _build_response_object(
        request=request,
        output_items=output_items,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        response_id=response_id,
    )

    if request.store and last_output is not None:
        persisted_messages = _responses_request_to_persisted_messages(request)
        persisted_messages.extend(_response_output_items_to_chat_messages(output_items))
        _responses_store[response_object.id] = {
            "messages": copy.deepcopy(persisted_messages),
            "response": response_object.model_copy(deep=True),
        }
        while len(_responses_store) > _RESPONSES_STORE_MAX_SIZE:
            _responses_store.popitem(last=False)

    yield _responses_sse_event(
        "response.completed",
        ResponseCompletedEvent(sequence_number=sequence, response=response_object),
    )


def _responses_sse_event(event_type: str, payload: BaseModel | dict) -> str:
    """Encode a Responses API SSE event."""
    data = (
        payload.model_dump_json()
        if isinstance(payload, BaseModel)
        else json.dumps(payload)
    )
    return f"event: {event_type}\ndata: {data}\n\n"


def _extract_reasoning_and_tool_calls(
    output_text: str,
    request: ChatCompletionRequest | None = None,
    *,
    allow_reasoning: bool = True,
    engine: BaseEngine | None = None,
) -> tuple[str | None, str | None, list[ToolCall] | None]:
    """
    Extract reasoning first, then parse tool calls from the cleaned content.

    Non-streaming responses can contain both a reasoning block and structured
    tool calls in the same final output. If tool parsing runs first and the
    response contains tools, the caller can no longer reliably recover the
    reasoning segment because the usual response path skips reasoning parsing
    once tool_calls is truthy.
    """
    reasoning_text = None
    text_for_tool_parse = output_text

    if _reasoning_parser and allow_reasoning:
        reasoning_text, cleaned_reasoning_text = _reasoning_parser.extract_reasoning(
            output_text
        )
        if cleaned_reasoning_text is not None:
            text_for_tool_parse = cleaned_reasoning_text
        elif reasoning_text is not None:
            text_for_tool_parse = ""

    # Skip tool parsing when the request defines no tools — otherwise the
    # parser can misinterpret JSON output (e.g. response_format) as tool calls.
    if request is not None and getattr(request, "tools", None):
        try:
            cleaned_text, tool_calls = _parse_tool_calls_with_parser(
                text_for_tool_parse or "",
                request,
                engine=engine,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'engine'" not in str(exc):
                raise
            cleaned_text, tool_calls = _parse_tool_calls_with_parser(
                text_for_tool_parse or "",
                request,
            )
    else:
        cleaned_text, tool_calls = text_for_tool_parse, None

    return reasoning_text, cleaned_text, tool_calls


def _detect_native_tool_support() -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Returns:
        True if native format should be preserved
    """
    if not _enable_auto_tool_choice or not _tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(_tool_call_parser)
        return parser_cls.supports_native_format()
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{_tool_call_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(
            "Failed to detect native tool support: %s",
            _sanitize_log_text(e, limit=500),
        )
        return False


def _tool_choice_disabled(request: ChatCompletionRequest | None) -> bool:
    """Return True when tool_choice explicitly disables tool calling."""
    if request is None:
        return False

    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice is None:
        request_dict = request.model_dump()
        tool_choice = request_dict.get("tool_choice")
    return tool_choice == "none"


def _get_streaming_tool_parser(
    request: ChatCompletionRequest | None,
    engine: BaseEngine | None = None,
):
    """Get a streaming-capable tool parser for this request.

    Uses the configured parser when auto tool choice is enabled, otherwise falls
    back to the generic auto parser so streaming still matches the generic
    non-streaming tool parsing behavior.
    """
    global _tool_parser_instance

    if request is None:
        return None
    if _tool_choice_disabled(request):
        return None

    tokenizer = _get_engine_tokenizer(engine if engine is not None else _engine)

    if _enable_auto_tool_choice and _tool_call_parser:
        if _tool_parser_instance is None:
            try:
                _get_or_init_tool_parser(engine)
            except Exception as e:
                logger.warning(
                    "Failed to init tool parser for streaming: %s",
                    _sanitize_log_text(e, limit=500),
                )
                return None
        _tool_parser_instance.reset()
        return _tool_parser_instance

    if not getattr(request, "tools", None):
        return None

    try:
        parser_cls = ToolParserManager.get_tool_parser("auto")
        parser = parser_cls(tokenizer)
        parser.reset()
        return parser
    except Exception as e:
        logger.warning(f"Failed to init generic streaming tool parser: {e}")
        return None


def _streaming_tool_markup_possible(text: str) -> bool:
    """Heuristic marker check to avoid parser work on ordinary text chunks."""
    return (
        any(marker in text for marker in _STREAMING_TOOL_MARKERS)
        or _STREAMING_BARE_BRACKET_MARKER.search(text) is not None
        or _STREAMING_BARE_BRACKET_PARTIAL.search(text) is not None
    )


def _streaming_tool_markup_possible_after_delta(
    accumulated_text: str, delta_text: str
) -> bool:
    """
    Check only the boundary window needed to detect newly appearing tool markup.

    Streaming paths call this before any marker has been seen. Scanning the full
    accumulated text on every ordinary chunk is quadratic for long responses, so
    keep enough trailing context to catch markers split across chunk boundaries.
    Once markup is possible, callers switch to the parser path with the full
    accumulated text.
    """
    if not delta_text:
        return False
    check_text = accumulated_text[-_STREAMING_TOOL_MARKUP_SCAN_CHARS:] + delta_text
    return _streaming_tool_markup_possible(check_text)


def load_embedding_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the embedding model engine when configured."""
    global _embedding_engine, _embedding_model_locked

    if not model_name:
        return

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(model_name)
    _embedding_engine.load()


def load_reranker_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
) -> None:
    """Load or reuse the reranker model engine when configured."""
    global _rerank_engine, _rerank_model_locked

    if not model_name:
        return

    if lock:
        _rerank_model_locked = model_name

    if (
        reuse_existing
        and _rerank_engine is not None
        and _rerank_engine.model_name == model_name
    ):
        return

    from .rerank import RerankEngine

    _rerank_engine = RerankEngine(model_name)
    _rerank_engine.load()


def load_model(
    model_name: str,
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    max_request_tokens: int = 32768,
    force_mllm: bool = False,
    gpu_memory_utilization: float = 0.90,
    served_model_name: str | None = None,
    trust_remote_code: bool = False,
    mtp: bool = False,
    prefill_step_size: int = 2048,
    specprefill_enabled: bool = False,
    specprefill_threshold: int = 8192,
    specprefill_keep_pct: float = 0.3,
    specprefill_draft_model: str = None,
    warm_prompts_path: str | None = None,
    auto_unload_idle_seconds: float = 0.0,
    lazy_load_model: bool = False,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        max_request_tokens: Maximum max_tokens accepted from API clients
        force_mllm: Force loading as MLLM even if not auto-detected
        trust_remote_code: Allow HuggingFace remote code execution during model/tokenizer loading
        mtp: Enable native MTP speculative decoding (SimpleEngine only)
        prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
        specprefill_enabled: Enable SpecPrefill (SimpleEngine only)
        specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill (default: 8192)
        specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
        specprefill_draft_model: Path to small draft model for SpecPrefill scoring
        auto_unload_idle_seconds: Idle time before auto-unloading the main model.
            When non-zero, the main model is managed through lifecycle
            residency instead of being loaded immediately in this function.
        lazy_load_model: When lifecycle residency is enabled, defer the first
            resident load until the first request instead of FastAPI lifespan
            startup.
    """
    global _engine, _model_manager, _model_name, _model_path, _default_max_tokens
    global _max_request_tokens, _tool_parser_instance, _warm_prompts_path
    global _default_model_key, _auto_unload_idle_seconds, _residency_manager
    global _force_mllm_model, _lazy_load_model, _lifespan_active

    _warm_prompts_path = warm_prompts_path

    if max_tokens < 1:
        raise ValueError("Default max tokens must be at least 1")
    if max_request_tokens < 1:
        raise ValueError("Max request tokens must be at least 1")
    if max_tokens > max_request_tokens:
        raise ValueError("Default max tokens cannot exceed max request tokens")

    if _lifespan_active:
        raise RuntimeError(
            "Cannot call load_model() after FastAPI lifespan startup; "
            "restart the server to reconfigure the main model"
        )

    if _residency_manager is None and _engine is not None:
        existing_loaded_attr = getattr(_engine, "_loaded", False)
        existing_stopped_attr = getattr(_engine, "stopped", False)
        existing_loaded = (
            existing_loaded_attr if isinstance(existing_loaded_attr, bool) else False
        )
        existing_stopped = (
            existing_stopped_attr if isinstance(existing_stopped_attr, bool) else None
        )
        existing_live = existing_loaded or existing_stopped is False
        if auto_unload_idle_seconds > 0 or lazy_load_model or existing_live:
            raise RuntimeError("Cannot replace an existing engine while it is live")

    if _residency_manager is not None and _default_model_key is not None:
        existing_engine = _residency_manager.get_engine(_default_model_key)
        existing_status = _residency_manager.get_status(_default_model_key)
        existing_state = existing_status.get("state")
        if (
            existing_engine is not None
            or existing_status.get("active_requests", 0) > 0
            or existing_state in {"loading", "loaded", "unloading"}
        ):
            raise RuntimeError(
                "Cannot replace an existing residency manager while it is live"
            )

    _default_max_tokens = max_tokens
    _max_request_tokens = max_request_tokens
    _model_manager = None
    _model_path = model_name
    _model_name = served_model_name or model_name
    _default_model_key = "default"
    _force_mllm_model = force_mllm
    _auto_unload_idle_seconds = auto_unload_idle_seconds
    _lazy_load_model = lazy_load_model
    # Reset tool parser instance when model is reloaded (tokenizer may change)
    _invalidate_tool_parser_cache("model reloaded")

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    if auto_unload_idle_seconds > 0 or lazy_load_model:
        spec = ModelSpec(
            model_key=_default_model_key,
            model_name=model_name,
            use_batching=use_batching,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval if use_batching else 1,
            max_tokens=max_tokens,
            force_mllm=force_mllm,
            mtp=mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_draft_model=specprefill_draft_model,
        )
        _residency_manager = ResidencyManager(
            _engine_factory,
            on_engine_loaded=_restore_engine_state,
            on_engine_unloading=_persist_engine_state,
            auto_unload_idle_seconds=auto_unload_idle_seconds,
        )
        _residency_manager.register_model(spec)
        _engine = None
        logger.info(
            "Lifecycle manager enabled: auto_unload_idle_seconds=%.1f",
            auto_unload_idle_seconds,
        )
        return

    _residency_manager = None
    _auto_unload_idle_seconds = 0.0
    _lazy_load_model = False

    if use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
    else:
        simple_engine_cls = SimpleEngine
        if simple_engine_cls is _IMPORTED_SIMPLE_ENGINE:
            from .engine import simple as simple_mod

            simple_engine_cls = simple_mod.SimpleEngine

        logger.info(f"Loading model with SimpleEngine: {model_name}")
        _max_kv = getattr(scheduler_config, "max_kv_size", 0) if scheduler_config else 0
        _engine = simple_engine_cls(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            force_mllm=force_mllm,
            mtp=mtp,
            prefill_step_size=prefill_step_size,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_draft_model=specprefill_draft_model,
            max_kv_size=_max_kv,
        )
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        previous_loop = None
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_engine.start())
        finally:
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()
            if previous_loop is not None and not previous_loop.is_closed():
                asyncio.set_event_loop(previous_loop)
            else:
                asyncio.set_event_loop(None)
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

    # Set native tool format support on the engine (thread-safe via instance property)
    _engine.preserve_native_tool_format = _detect_native_tool_support()
    if _engine.preserve_native_tool_format:
        logger.info(f"Native tool format enabled for parser: {_tool_call_parser}")

    logger.info(f"Default max tokens: {_default_max_tokens}")
    logger.info(f"Max request tokens: {_max_request_tokens}")


def load_model_registry(
    config_path: str,
    *,
    defaults: RegistryServeDefaults,
) -> None:
    """Load a registry-backed model manager from YAML configuration."""
    global _engine, _model_manager, _model_name, _model_path, _default_max_tokens

    manager_config, registry = load_registry_config(config_path, defaults)
    _engine = None
    _model_path = None
    _model_name = None
    _default_max_tokens = defaults.max_tokens
    _model_manager = ModelManager(manager_config, registry, defaults)

    logger.info(
        "Loaded models config: %s (%d models, %.1f GB budget)",
        config_path,
        len(registry),
        manager_config.memory_budget_bytes / (1024**3),
    )


def get_usage(output: GenerationOutput) -> Usage:
    """Extract usage metrics from GenerationOutput."""
    total_prompt_tokens = (
        output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
    )
    total_completion_tokens = (
        output.completion_tokens if hasattr(output, "completion_tokens") else 0
    )
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint (disabled by default)."""
    if not _metrics.enabled:
        raise HTTPException(status_code=404, detail="Metrics endpoint is disabled")

    payload, content_type = _metrics.render_metrics(
        engine=_engine,
        mcp_manager=_mcp_manager,
    )
    return Response(content=payload, headers={"Content-Type": content_type})


@app.get("/health")
async def health():
    """Health check endpoint."""
    mcp_info = None
    if _mcp_manager is not None:
        connected = sum(
            1 for s in _mcp_manager.get_server_status() if s.state.value == "connected"
        )
        total = len(_mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(_mcp_manager.get_all_tools()),
        }

    engine_stats = _engine.get_stats() if _engine else {}
    lifecycle = _get_lifecycle_status()
    health_status = (
        "unhealthy"
        if lifecycle is not None and lifecycle.get("state") == "failed"
        else "healthy"
    )

    payload = {
        "status": health_status,
        "model_loaded": _engine is not None or _model_manager is not None,
        "model_name": _model_name,
        "available_models": _list_available_model_names(),
        "model_type": (
            "mllm"
            if (_engine and _engine.is_mllm)
            or _force_mllm_model
            or (
                _engine is None
                and (_model_path or _model_name)
                and is_mllm_model(_model_path or _model_name)
            )
            else "llm"
        ),
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }
    if lifecycle is not None:
        lifecycle_fields = {
            "residency_state": lifecycle["state"],
            "active_requests": lifecycle["active_requests"],
            "last_used_at": lifecycle["last_used_at"],
            "loaded_at": lifecycle["loaded_at"],
            "auto_unload_idle_seconds": lifecycle["auto_unload_idle_seconds"],
        }
        if lifecycle.get("state") == "failed":
            lifecycle_fields["last_error"] = (
                "model_load_failed" if lifecycle.get("last_error") is not None else None
            )
        payload.update(lifecycle_fields)
    return payload


@app.get("/v1/status", dependencies=[Depends(verify_api_key)])
async def status():
    """Real-time status with per-request details for debugging and monitoring."""
    if _model_manager is not None:
        return {
            "status": "running",
            "model_manager": {
                "memory_budget_gb": round(
                    _model_manager.memory_budget_bytes / (1024**3), 2
                ),
                "models": _model_manager.list_models(),
            },
        }
    lifecycle = _public_lifecycle_status(_get_lifecycle_status())
    if _engine is None:
        return {
            "status": "not_loaded",
            "model": _model_name,
            "residency": lifecycle,
            "requests": [],
        }

    stats = _engine.get_stats()

    # Extract batch_generator throughput when available (MLLM scheduler).
    bg = stats.get("batch_generator", {})

    return {
        "status": "running" if stats.get("running") else "stopped",
        "model": _model_name,
        "residency": lifecycle,
        "uptime_s": round(stats.get("uptime_seconds", 0), 1),
        "steps_executed": stats.get("steps_executed", 0),
        "num_running": stats.get("num_running", 0),
        "num_waiting": stats.get("num_waiting", 0),
        "total_requests_processed": stats.get("num_requests_processed", 0),
        "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
        "total_completion_tokens": stats.get("total_completion_tokens", 0),
        "generation_tps": bg.get("generation_tps", 0),
        "prompt_tps": bg.get("prompt_tps", 0),
        "metal": {
            "active_memory_gb": stats.get("metal_active_memory_gb"),
            "peak_memory_gb": stats.get("metal_peak_memory_gb"),
            "cache_memory_gb": stats.get("metal_cache_memory_gb"),
        },
        "cache": stats.get("memory_aware_cache")
        or stats.get("paged_cache")
        or stats.get("prefix_cache"),
        "requests": stats.get("requests", []),
    }


@app.get("/v1/cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Get cache statistics for debugging and monitoring."""
    engine_cache = None
    if _engine is not None and hasattr(_engine, "get_cache_stats"):
        try:
            engine_cache = _engine.get_cache_stats()
        except Exception as exc:
            engine_cache = {"error": f"engine cache stats failed: {exc}"}

    try:
        from mlx_vlm.utils import (
            get_multimodal_kv_cache_stats,
            get_pil_cache_stats,
            get_pixel_values_cache_stats,
        )

        return {
            "engine_cache": engine_cache,
            "multimodal_kv_cache": get_multimodal_kv_cache_stats(),
            "pixel_values_cache": get_pixel_values_cache_stats(),
            "pil_image_cache": get_pil_cache_stats(),
        }
    except ImportError:
        return {
            "engine_cache": engine_cache,
            "error": "Cache stats not available (mlx_vlm not loaded)",
        }


@app.delete("/v1/cache", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear all caches."""
    cleared_engine = None
    if _engine is not None and hasattr(_engine, "clear_runtime_caches"):
        try:
            cleared_engine = _engine.clear_runtime_caches()
        except Exception as exc:
            logger.warning("Failed to clear engine caches: %s", exc, exc_info=True)
            cleared_engine = {"error": str(exc)}

    try:
        from mlx_vlm.utils import (
            clear_multimodal_kv_cache,
            clear_pixel_values_cache,
        )

        clear_multimodal_kv_cache()
        clear_pixel_values_cache()
        return {
            "status": "cleared",
            "engine_cache": cleared_engine,
            "caches": ["multimodal_kv", "pixel_values", "pil_image"],
        }
    except ImportError:
        return {
            "status": "cleared",
            "engine_cache": cleared_engine,
            "error": "Cache clear not available (mlx_vlm not loaded)",
        }


@app.delete("/v1/cache/prefix", dependencies=[Depends(verify_api_key)])
async def clear_prefix_cache():
    """Clear the text prefix cache used for KV reuse in continuous batching.

    If the server was started with ``--warm-prompts``, the warm-up is
    re-run in the background after clear so the next real request still
    hits the cache. Response returns immediately without waiting for
    the re-warm to finish.
    """
    if _engine is None:
        return {"status": "no_engine"}
    cleared = False
    if hasattr(_engine, "clear_prefix_cache"):
        try:
            _engine.clear_prefix_cache()
            cleared = True
        except Exception as e:
            logger.warning(
                "[clear_prefix_cache] engine.clear_prefix_cache failed: %s",
                _sanitize_log_text(e, limit=500),
            )

    # Auto re-warm in background if warm-prompts was configured.
    rewarm_scheduled = False
    if cleared and _warm_prompts_path and hasattr(_engine, "stream_chat"):

        async def _rewarm():
            try:
                from vllm_mlx.prompt_warmup import (
                    load_warmup_file,
                    warm_prefix_cache,
                )

                prompts = load_warmup_file(_warm_prompts_path)
                result = await warm_prefix_cache(_engine, prompts)
                logger.info(
                    "[clear_prefix_cache] re-warm done: %d completed, %d skipped, %.1fs",
                    result["count"],
                    result["skipped"],
                    result["elapsed_ms"] / 1000,
                )
            except Exception as e:
                logger.warning(
                    "[clear_prefix_cache] re-warm failed: %s",
                    _sanitize_log_text(e, limit=500),
                )

        asyncio.create_task(_rewarm())
        rewarm_scheduled = True

    status = "cleared" if cleared else "not_supported"
    return {"status": status, "rewarm_scheduled": rewarm_scheduled}


@app.post(
    "/v1/requests/{request_id}/cancel",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def cancel_request(request_id: str):
    """Cancel an active or queued request.

    The request_id is the chatcmpl-xxx ID from the first SSE streaming chunk.
    """
    engine = get_engine()
    try:
        aborted = await engine.abort_request(request_id)
    except Exception as exc:
        logger.exception("Failed to cancel request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel request {request_id}: {exc}",
        ) from exc

    if not aborted:
        raise HTTPException(
            status_code=404,
            detail=f"Request not found or cancellation is unsupported: {request_id}",
        )

    logger.info("[cancel_request] accepted request_id=%s", request_id)
    return {
        "object": "request.cancel",
        "id": request_id,
        "cancelled": True,
        "model": _model_name,
    }


@app.delete(
    "/v1/requests/{request_id}",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def delete_request(request_id: str):
    """OpenAI-style alias for cancelling an active or queued request."""
    return await cancel_request(request_id)


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelsResponse:
    """List available models."""
    models = []
    if _model_manager is not None:
        models.extend(ModelInfo(id=item["id"]) for item in _model_manager.list_models())
    elif _model_name:
        models.append(ModelInfo(id=_model_name))
    if _embedding_engine is not None:
        models.append(
            ModelInfo(id=_embedding_engine.model_name, owned_by="vllm-mlx-embedding")
        )
    if _rerank_engine is not None:
        models.append(
            ModelInfo(id=_rerank_engine.model_name, owned_by="vllm-mlx-reranker")
        )
    return ModelsResponse(data=models)


# =============================================================================
# Embeddings Endpoint
# =============================================================================


@app.post(
    "/v1/embeddings",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    OpenAI-compatible embeddings API supporting single or batch inputs.

    Single text:
    ```json
    {
      "model": "mlx-community/all-MiniLM-L6-v2-4bit",
      "input": "The quick brown fox jumps over the lazy dog"
    }
    ```

    Batch of texts:
    ```json
    {
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "input": [
        "I love machine learning",
        "Deep learning is fascinating",
        "Neural networks are powerful"
      ]
    }
    ```

    Response:
    ```json
    {
      "object": "list",
      "data": [
        {"object": "embedding", "index": 0, "embedding": [0.023, -0.982, ...]},
        {"object": "embedding", "index": 1, "embedding": [0.112, -0.543, ...]},
        {"object": "embedding", "index": 2, "embedding": [0.876, 0.221, ...]}
      ],
      "model": "mlx-community/embeddinggemma-300m-6bit",
      "usage": {"prompt_tokens": 24, "total_tokens": 24}
    }
    ```

    Supported request-time models:
    - mlx-community/all-MiniLM-L6-v2-4bit (fast, compact)
    - mlx-community/embeddinggemma-300m-6bit (high quality)
    - mlx-community/bge-large-en-v1.5-4bit (best for English)
    - mlx-community/multilingual-e5-small-mlx
    - mlx-community/multilingual-e5-large-mlx
    - mlx-community/bert-base-uncased-mlx
    - mlx-community/ModernBERT-base-mlx

    Other embedding models must be pinned explicitly with --embedding-model at
    server startup.
    """
    global _embedding_engine
    tracker = _metrics.track_inference("embeddings", stream=False)

    try:
        # Resolve model name before any lazy-load path is reached.
        model_name = resolve_embedding_model_name(
            request.model,
            locked_model=_embedding_model_locked,
        )

        # Lazy-load or swap embedding engine
        load_embedding_model(model_name, lock=False, reuse_existing=True)

        # Normalise input to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        if not texts:
            raise HTTPException(status_code=400, detail="Input must not be empty")

        start_time = time.perf_counter()

        # Count tokens for usage reporting
        prompt_tokens = _embedding_engine.count_tokens(texts)

        # Generate embeddings (batch)
        embeddings = _embedding_engine.embed(texts)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Embeddings: {len(texts)} inputs, {prompt_tokens} tokens in {elapsed:.2f}s"
        )

        # Build OpenAI-compatible response with ordered indices
        data = [
            EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(embeddings)
        ]

        response = EmbeddingResponse(
            data=data,
            model=model_name,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
        tracker.finish(
            result="success",
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
        )
        return response

    except ImportError:
        tracker.finish(result="error")
        raise HTTPException(
            status_code=503,
            detail=(
                "mlx-embeddings not installed. Install with: pip install mlx-embeddings"
            ),
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception as e:
        tracker.finish(result="error")
        _log_and_raise_internal_error(
            "Embedding generation failed",
            e,
            "Embedding generation failed",
        )


# =============================================================================
# Reranking Endpoint
# =============================================================================


@app.post(
    "/v1/rerank",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """
    Rerank documents against a query using a cross-encoder model.

    Jina/Cohere-compatible reranking API. Accepts a query and a list of
    documents (strings or {text: ...} objects), returns results sorted
    by relevance score descending.
    """
    global _rerank_engine

    try:
        model_name = request.model

        # If a reranker model was pre-configured at startup, only allow that model
        if _rerank_model_locked is not None and model_name != _rerank_model_locked:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Reranker model '{model_name}' is not available. "
                    f"This server was started with --rerank-model {_rerank_model_locked}. "
                    f"Only '{_rerank_model_locked}' can be used for reranking. "
                    f"Restart the server with a different --rerank-model to use '{model_name}'."
                ),
            )

        # Validate query
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")

        # Validate documents
        if not request.documents:
            raise HTTPException(
                status_code=400, detail="Documents list must not be empty"
            )

        # Validate top_n
        if request.top_n is not None and request.top_n > len(request.documents):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"top_n ({request.top_n}) must not exceed the number of "
                    f"documents ({len(request.documents)})"
                ),
            )

        # Require --rerank-model at startup; no unconstrained lazy loading
        if _rerank_engine is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No reranker model loaded. Start the server with "
                    "--rerank-model to enable the /v1/rerank endpoint."
                ),
            )

        # Extract text from documents (handle both string and object formats)
        doc_texts = []
        original_docs = []
        for doc in request.documents:
            if isinstance(doc, str):
                doc_texts.append(doc)
                original_docs.append({"text": doc})
            elif isinstance(doc, dict) and "text" in doc:
                doc_texts.append(doc["text"])
                original_docs.append(doc)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Each document must be a string or an object with a 'text' field. "
                        f"Got: {type(doc).__name__}"
                    ),
                )

        start_time = time.perf_counter()

        # Run scoring off the event loop with concurrency limit.
        # score_pairs returns (scores, total_tokens) from the same
        # tokenization pass used for scoring — no double tokenization.
        import asyncio

        async with _rerank_engine._semaphore:
            scores, total_tokens = await asyncio.to_thread(
                _rerank_engine.score_pairs, request.query, doc_texts
            )

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Rerank: {len(doc_texts)} documents, {total_tokens} tokens in {elapsed:.2f}s"
        )

        # Build results with original index and optional document
        results = []
        for i, score in enumerate(scores):
            result = RerankResult(
                index=i,
                relevance_score=score,
                document=original_docs[i] if request.return_documents else None,
            )
            results.append(result)

        # Sort by relevance score descending
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Apply top_n limit
        if request.top_n is not None:
            results = results[: request.top_n]

        return RerankResponse(
            model=model_name,
            results=results,
            usage=RerankUsage(total_tokens=total_tokens),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MCP Endpoints
# =============================================================================


@app.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    if _mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in _mcp_manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@app.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    if _mcp_manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in _mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    return MCPServersResponse(servers=servers)


@app.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    global _mcp_executor

    if _mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    if _mcp_executor is None:
        from vllm_mlx.mcp import ToolExecutor

        _mcp_executor = ToolExecutor(_mcp_manager)

    tool_call = {
        "id": f"mcp-{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": request.tool_name,
            "arguments": request.arguments,
        },
    }
    result, _ = (await _mcp_executor.execute_tool_calls([tool_call], parallel=False))[0]

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )


# =============================================================================
# Audio Endpoints
# =============================================================================

# Global audio engines (lazy loaded)
_stt_engine = None
_tts_engine = None


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """
    Transcribe audio to text (OpenAI Whisper API compatible).

    Supported models:
    - whisper-large-v3 (multilingual, best quality)
    - whisper-large-v3-turbo (faster)
    - whisper-medium, whisper-small (lighter)
    - parakeet-tdt-0.6b-v2 (English, fastest)
    """
    global _stt_engine
    tracker = _metrics.track_inference("audio_transcriptions", stream=False)

    try:
        from .audio.stt import STTEngine  # Lazy import - optional feature

        model_name = resolve_stt_model_name(model)

        # Load engine if needed
        if _stt_engine is None or _stt_engine.model_name != model_name:
            _stt_engine = STTEngine(model_name)
            _stt_engine.load()

        # Stream uploaded file to disk under a hard size cap.
        tmp_path = await save_upload_with_limit(
            file,
            max_bytes=_max_audio_upload_bytes,
            default_suffix=".wav",
        )

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            tracker.finish(result="success")
            return result.text

        tracker.finish(result="success")
        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError:
        tracker.finish(result="error")
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception as e:
        tracker.finish(result="error")
        _log_and_raise_internal_error(
            "Transcription failed",
            e,
            "Transcription failed",
        )


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
    """
    Generate speech from text (OpenAI TTS API compatible).

    Supported models:
    - kokoro (fast, lightweight)
    - chatterbox (multilingual, expressive)
    - vibevoice (realtime)
    - voxcpm (Chinese/English)
    """
    global _tts_engine
    tracker = _metrics.track_inference("audio_speech", stream=False)

    try:
        from .audio.tts import TTSEngine  # Lazy import - optional feature

        model_name = resolve_tts_model_name(model)
        validate_tts_input_length(input, max_chars=_max_tts_input_chars)

        # Load engine if needed
        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        tracker.finish(result="success")
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError:
        tracker.finish(result="error")
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except HTTPException as exc:
        tracker.finish(result=_metrics_result_from_status(exc.status_code))
        raise
    except Exception as e:
        tracker.finish(result="error")
        _log_and_raise_internal_error(
            "TTS generation failed",
            e,
            "Speech generation failed",
        )


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from .audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}


# =============================================================================
# Streaming disconnect detection
# =============================================================================


async def _ensure_sse_terminal(
    generator: AsyncIterator[str],
    terminal_frame: str,
) -> AsyncIterator[str]:
    """Guarantee that *terminal_frame* is emitted exactly once at the end of
    *generator*, even if the generator raises mid-stream.

    If the inner generator already yields the terminal frame on its happy path,
    the wrapper detects it and avoids double-emission.  If the generator raises
    before reaching the terminal, the wrapper emits it in the ``finally`` block.
    """
    emitted = False
    try:
        async for chunk in generator:
            if chunk == terminal_frame:
                emitted = True
            yield chunk
    except Exception as e:
        logger.error(f"Streaming error, ensuring terminal frame: {e}")
    finally:
        if not emitted:
            yield terminal_frame


def _find_uvicorn_cycle(obj, depth=0, visited=None):
    """Walk through middleware wrappers to find uvicorn's RequestResponseCycle.

    This relies on uvicorn's internal ``RequestResponseCycle.disconnected``
    attribute and Starlette's middleware closure layout.  Tested against
    uvicorn 0.34-0.40 and starlette 0.44-0.46.  If either changes the
    internal layout, this function returns None and disconnect detection
    silently falls back to timeout-only behaviour.
    """
    if depth > 8:
        return None
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    # Direct hit: object has 'disconnected' bool attr (RequestResponseCycle)
    if hasattr(obj, "disconnected") and isinstance(getattr(obj, "disconnected"), bool):
        return obj

    # Check __self__ of bound methods
    self_obj = getattr(obj, "__self__", None)
    if self_obj is not None:
        result = _find_uvicorn_cycle(self_obj, depth + 1, visited)
        if result:
            return result

    # Check _receive attribute (Starlette Request -> inner receive)
    inner = getattr(obj, "_receive", None)
    if inner is not None:
        result = _find_uvicorn_cycle(inner, depth + 1, visited)
        if result:
            return result

    # Check closure cells (BaseHTTPMiddleware wrappers)
    if hasattr(obj, "__closure__") and obj.__closure__:
        for cell in obj.__closure__:
            try:
                val = cell.cell_contents
                result = _find_uvicorn_cycle(val, depth + 1, visited)
                if result:
                    return result
            except ValueError:
                pass

    return None


def _is_client_disconnected(raw_request: Request) -> bool:
    """Reliable client disconnect check.

    Starlette's ``is_disconnected()`` uses an immediately-cancelled
    ``anyio.CancelScope`` which prevents the ASGI ``receive()`` from
    executing — so it always returns False for non-streaming requests.

    This function bypasses Starlette and reads uvicorn's internal
    ``disconnected`` flag directly from the ``RequestResponseCycle``,
    walking through any middleware wrappers via closures.
    """
    if getattr(raw_request, "_is_disconnected", False):
        return True
    # Cache the cycle reference on the request for fast subsequent checks
    cycle = getattr(raw_request, "_uvicorn_cycle", None)
    if cycle is None:
        receive = getattr(raw_request, "_receive", None)
        if receive is None:
            return False
        cycle = _find_uvicorn_cycle(receive)
        # Cache even if None (to avoid repeated searches)
        raw_request._uvicorn_cycle = cycle
    if cycle is not None and cycle.disconnected:
        raw_request._is_disconnected = True
        return True
    return False


async def _disconnect_guard(
    generator: AsyncIterator[str],
    raw_request: Request,
    poll_interval: float = 0.5,
    heartbeat_interval: float = 5.0,
    cleanup=None,
    timeout: float | None = None,
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect.

    Uses asyncio racing: each __anext__() on the inner generator is
    raced against a disconnect poller.  When neither completes within
    ``heartbeat_interval`` seconds, an SSE comment is yielded as a
    heartbeat.  This forces an ASGI write which triggers broken-pipe
    detection — without heartbeats, ``is_disconnected()`` stays False
    during long prefill because no data is written to the socket.

    If *timeout* is set, the stream is forcefully stopped once the
    elapsed wall-clock time exceeds the given number of seconds.
    Without this, zombie streaming requests can run indefinitely when
    ``is_disconnected()`` never fires.

    On disconnect, the cancellation propagates to stream_outputs()
    finally-block → abort_request() → abort_prefill().
    """
    import time as _time

    _t0 = _time.monotonic()

    def _elapsed():
        return f"{_time.monotonic() - _t0:.1f}s"

    _effective_timeout = timeout or _default_timeout

    logger.info(
        f"[disconnect_guard] START poll={poll_interval}s heartbeat={heartbeat_interval}s "
        f"timeout={_effective_timeout:.0f}s"
    )

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = _is_client_disconnected(raw_request)
            if poll_count % 10 == 0 or is_disc:
                logger.info(
                    f"[disconnect_guard] poll #{poll_count} "
                    f"disconnected={is_disc} elapsed={_elapsed()}"
                )
            if is_disc:
                return

    chunk_count = 0
    heartbeat_count = 0
    disconnect_task: asyncio.Task | None = None
    anext_task: asyncio.Task | None = None
    try:
        aiter = generator.__aiter__()
        disconnect_task = asyncio.create_task(_wait_disconnect())
        anext_task = None
        while True:
            # Enforce absolute timeout for streaming requests.
            if _time.monotonic() - _t0 >= _effective_timeout:
                logger.warning(
                    f"[disconnect_guard] TIMEOUT after {_effective_timeout:.0f}s, "
                    f"{chunk_count} chunks, {heartbeat_count} heartbeats, "
                    f"elapsed={_elapsed()}"
                )
                if anext_task and not anext_task.done():
                    anext_task.cancel()
                    try:
                        await anext_task
                    except (asyncio.CancelledError, StopAsyncIteration):
                        pass
                break

            if anext_task is None:
                anext_task = asyncio.ensure_future(aiter.__anext__())

            done, _ = await asyncio.wait(
                [anext_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=heartbeat_interval,
            )

            if disconnect_task in done:
                logger.info(
                    f"[disconnect_guard] CLIENT DISCONNECTED after "
                    f"{chunk_count} chunks, {heartbeat_count} heartbeats, "
                    f"elapsed={_elapsed()}"
                )
                anext_task.cancel()
                try:
                    await anext_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
                break

            if anext_task in done:
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
                        f"[disconnect_guard] generator raised {type(exc).__name__}: {exc}, "
                        f"after {chunk_count} chunks, elapsed={_elapsed()}"
                    )
                    break
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(
                        f"[disconnect_guard] first chunk arrived, elapsed={_elapsed()}"
                    )
                yield chunk
                anext_task = None
                continue

            # Timeout — no chunk and no disconnect detected yet.
            # Send SSE comment as heartbeat to force an ASGI write.
            # If the client has disconnected, this write will fail and
            # the next is_disconnected() poll will return True.
            heartbeat_count += 1
            yield ": heartbeat\n\n"

    except GeneratorExit:
        logger.info(
            f"[disconnect_guard] GeneratorExit after {chunk_count} chunks, elapsed={_elapsed()}"
        )
    finally:
        if disconnect_task and not disconnect_task.done():
            disconnect_task.cancel()
        if anext_task and not anext_task.done():
            anext_task.cancel()
        # Close the generator so that stream_outputs() finally-block fires
        # abort_request(), removing the request from the scheduler.
        # We defer the close by 0.5s to avoid a Metal thread-safety race:
        # scheduler.step() runs in run_in_executor and may be mid-eval —
        # closing the generator immediately could trigger mlx::core::eval
        # on the main thread concurrently → Metal assertion failure.
        _gen_to_close = aiter

        async def _deferred_generator_close():
            await asyncio.sleep(0.5)
            try:
                await _gen_to_close.aclose()
            except Exception as _exc:
                logger.debug(
                    f"[disconnect_guard] deferred aclose raised "
                    f"{type(_exc).__name__}: {_exc}"
                )

        asyncio.create_task(_deferred_generator_close())
        if cleanup is not None:
            result = cleanup()
            if asyncio.iscoroutine(result):
                await result
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks, "
            f"{heartbeat_count} heartbeats, elapsed={_elapsed()}"
        )


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
    timeout_detail_seconds: float | None = None,
    cleanup_result=None,
):
    """Run a coroutine with both timeout and client disconnect detection.

    For non-streaming requests where _disconnect_guard() can't be used.
    Races the coroutine against a disconnect poller, same pattern as
    _disconnect_guard but for awaitable (non-generator) coroutines.
    """
    import time as _time

    _t0 = _time.monotonic()

    task = asyncio.ensure_future(coro)

    async def _wait_disconnect():
        poll_count = 0
        while True:
            await asyncio.sleep(poll_interval)
            poll_count += 1
            is_disc = _is_client_disconnected(raw_request)
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
            # Timeout
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise HTTPException(
                status_code=504,
                detail=(
                    "Request timed out after "
                    f"{(timeout_detail_seconds or timeout):.1f} seconds"
                ),
            )

        if disconnect_task in done:
            # Client disconnected
            logger.info(
                f"[disconnect_guard] CLIENT DISCONNECTED (non-stream) "
                f"elapsed={_time.monotonic() - _t0:.1f}s"
            )
            if task in done:
                try:
                    result = task.result()
                except (asyncio.CancelledError, Exception):
                    pass
                else:
                    if cleanup_result is not None:
                        cleanup = cleanup_result(result)
                        if asyncio.iscoroutine(cleanup):
                            await cleanup
            else:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            return None  # Signal to caller that client disconnected

        # Task completed
        return task.result()

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()


def _start_request_budget(timeout: float | None) -> tuple[float, float]:
    """Return the total timeout and absolute deadline for a request."""
    total_timeout = timeout or _default_timeout
    return total_timeout, time.monotonic() + total_timeout


def _remaining_request_timeout(total_timeout: float, deadline: float) -> float:
    """Compute remaining request budget or raise the standard timeout error."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {total_timeout:.1f} seconds",
        )
    return remaining


_active_request_contexts: dict[int, RequestModelContext] = {}


async def _acquire_default_engine_for_request(
    raw_request: Request,
    *,
    total_timeout: float,
    deadline: float,
    count_activity: bool = True,
    model: str | None = None,
) -> BaseEngine | None:
    """Acquire the engine for a request, using the model registry when active.

    When ``_model_manager`` is set (registry mode), acquires the engine for the
    requested *model* via ``_acquire_request_model``.  The resulting
    ``RequestModelContext`` is stashed in ``_active_request_contexts`` keyed by
    ``id(raw_request)`` so that the matching ``_release_default_engine`` call
    can release the lease.

    In single-model mode the behaviour is unchanged.
    """
    if _model_manager is not None and model is not None:

        async def _registry_acquire():
            ctx = await _acquire_request_model(model)
            if raw_request is not None:
                _active_request_contexts[id(raw_request)] = ctx
            return ctx.engine

        async def _registry_cleanup(_result):
            ctx = _active_request_contexts.pop(id(raw_request), None)
            if ctx is not None:
                await ctx.release()

        if raw_request is None:
            return await _registry_acquire()

        return await _wait_with_disconnect(
            _registry_acquire(),
            raw_request,
            timeout=_remaining_request_timeout(total_timeout, deadline),
            timeout_detail_seconds=total_timeout,
            cleanup_result=lambda _r: _registry_cleanup(_r),
        )

    if count_activity:
        acquire_coro = _acquire_default_engine()
        cleanup = lambda _result: _release_default_engine()
    else:
        acquire_coro = _acquire_default_engine(count_activity=False)
        cleanup = lambda _result: _release_default_engine(count_activity=False)

    if raw_request is None:
        return await acquire_coro

    return await _wait_with_disconnect(
        acquire_coro,
        raw_request,
        timeout=_remaining_request_timeout(total_timeout, deadline),
        timeout_detail_seconds=total_timeout,
        cleanup_result=cleanup,
    )


async def _release_engine_for_request(raw_request: Request | None) -> None:
    """Release the engine acquired for this request.

    In registry mode, releases the model lease stashed by
    ``_acquire_default_engine_for_request``.  In single-model mode, falls
    through to the default release path.
    """
    if raw_request is not None:
        ctx = _active_request_contexts.pop(id(raw_request), None)
        if ctx is not None:
            await ctx.release()
            return
    await _release_default_engine()


def _make_release_cleanup(raw_request: Request | None):
    """Return a cleanup callable suitable for ``_disconnect_guard``."""
    if _model_manager is not None and raw_request is not None:

        async def _cleanup():
            ctx = _active_request_contexts.pop(id(raw_request), None)
            if ctx is not None:
                await ctx.release()
            else:
                await _release_default_engine()

        return _cleanup
    return _release_default_engine


# =============================================================================
# Completion Endpoints
# =============================================================================


@app.post(
    "/v1/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create a text completion."""
    _validate_model_name(request.model)
    effective_max_tokens = _resolve_request_max_tokens(request.max_tokens)
    tracker = _metrics.track_inference("completions", stream=request.stream)

    # Handle single prompt or list of prompts
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    total_timeout, deadline = _start_request_budget(request.timeout)

    # --- Detailed request logging ---
    prompt_preview = prompts[0][:200] if prompts else "(empty)"
    prompt_len = sum(len(p) for p in prompts)
    logger.info(
        f"[REQUEST] POST /v1/completions stream={request.stream} "
        f"max_tokens={request.max_tokens} temp={request.temperature} "
        f"top_p={request.top_p} top_k={request.top_k} min_p={request.min_p} "
        f"presence_penalty={request.presence_penalty} "
        f"repetition_penalty={request.repetition_penalty} "
        f"prompt_chars={prompt_len} "
        f"prompt_preview={_sanitize_log_text(prompt_preview, limit=200)}"
    )

    # Resolve repetition penalty for completions
    comp_rep_penalty = request.repetition_penalty

    engine = await _acquire_default_engine_for_request(
        raw_request,
        total_timeout=total_timeout,
        deadline=deadline,
        model=request.model,
    )
    if engine is None:
        return Response(status_code=499)
    release_on_exit = True

    try:
        if request.stream:
            response = StreamingResponse(
                _disconnect_guard(
                    _ensure_sse_terminal(
                        stream_completion(
                            engine,
                            prompts[0],
                            request,
                            effective_max_tokens,
                            repetition_penalty=comp_rep_penalty,
                            metrics_tracker=tracker,
                        ),
                        "data: [DONE]\n\n",
                    ),
                    raw_request,
                    cleanup=_make_release_cleanup(raw_request),
                    timeout=total_timeout,
                ),
                media_type="text/event-stream",
            )
            release_on_exit = False
            return response

        # Non-streaming response with timing and timeout
        start_time = time.perf_counter()
        choices = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        for i, prompt in enumerate(prompts):
            generate_kwargs = {
                "prompt": prompt,
                "max_tokens": effective_max_tokens,
                "temperature": _resolve_temperature(request.temperature),
                "top_p": _resolve_top_p(request.top_p),
                "top_k": request.top_k or 0,
                "min_p": request.min_p or 0.0,
                "presence_penalty": request.presence_penalty or 0.0,
                "stop": request.stop,
            }
            if comp_rep_penalty is not None:
                generate_kwargs["repetition_penalty"] = comp_rep_penalty
            if request.specprefill is not None:
                generate_kwargs["specprefill"] = request.specprefill
            if request.specprefill_keep_pct is not None:
                generate_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct
            try:
                if raw_request is None:
                    output = await engine.generate(**generate_kwargs)
                else:
                    output = await _wait_with_disconnect(
                        engine.generate(**generate_kwargs),
                        raw_request,
                        timeout=_remaining_request_timeout(total_timeout, deadline),
                        timeout_detail_seconds=total_timeout,
                    )
            except HTTPException as exc:
                tracker.finish(result=_metrics_result_from_status(exc.status_code))
                raise
            if output is None:
                tracker.finish(
                    result="client_closed",
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                )
                return Response(status_code=499)  # Client closed request

            choices.append(
                CompletionChoice(
                    index=i,
                    text=output.text,
                    finish_reason=output.finish_reason,
                )
            )
            total_completion_tokens += output.completion_tokens
            total_prompt_tokens += (
                output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
            )

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        tracker.finish(
            result="success",
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )
        return CompletionResponse(
            model=_model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )
    finally:
        if release_on_exit:
            await _release_engine_for_request(raw_request)


@app.post(
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
    effective_max_tokens = _resolve_request_max_tokens(request.max_tokens)
    tracker = _metrics.track_inference("chat_completions", stream=request.stream)
    total_timeout, deadline = _start_request_budget(request.timeout)

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
        f"temp={request.temperature} top_p={request.top_p} "
        f"top_k={request.top_k} min_p={request.min_p} "
        f"presence_penalty={request.presence_penalty} "
        f"repetition_penalty={request.repetition_penalty} "
        f"msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.info(
        "[REQUEST] last user message preview: %s",
        _sanitize_log_text(last_user_preview, limit=300),
    )

    engine = await _acquire_default_engine_for_request(
        raw_request,
        total_timeout=total_timeout,
        deadline=deadline,
        model=request.model,
    )
    if engine is None:
        return Response(status_code=499)

    release_on_exit = True
    try:
        try:
            prepared = _prepare_chat_completion_invocation(
                engine,
                request,
                effective_max_tokens,
            )
        except UnsafeRemoteURLError as exc:
            tracker.finish(result="client_error")
            _raise_remote_media_http_error(exc)

        if request.stream:
            response = StreamingResponse(
                _disconnect_guard(
                    _ensure_sse_terminal(
                        stream_chat_completion(
                            engine,
                            prepared.messages,
                            request,
                            metrics_tracker=tracker,
                            **prepared.chat_kwargs,
                        ),
                        "data: [DONE]\n\n",
                    ),
                    raw_request,
                    cleanup=_make_release_cleanup(raw_request),
                    timeout=total_timeout,
                ),
                media_type="text/event-stream",
            )
            release_on_exit = False
            return response

        start_time = time.perf_counter()

        try:
            output = await _wait_with_disconnect(
                engine.chat(messages=prepared.messages, **prepared.chat_kwargs),
                raw_request,
                timeout=_remaining_request_timeout(total_timeout, deadline),
                timeout_detail_seconds=total_timeout,
            )
        except HTTPException as exc:
            tracker.finish(result=_metrics_result_from_status(exc.status_code))
            raise
        if output is None:
            tracker.finish(result="client_closed")
            return Response(status_code=499)  # Client closed request

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        reasoning_text, cleaned_text, tool_calls = _extract_reasoning_and_tool_calls(
            output.text,
            request,
            allow_reasoning=(getattr(request, "enable_thinking", None) is not False),
            engine=engine,
        )

        # Process response_format if specified (after reasoning parser cleaned the text)
        if prepared.response_format and not tool_calls:
            json_input = cleaned_text or output.text
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, prepared.response_format
            )
            if parsed_json is not None:
                # Return JSON as string
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                if prepared.json_logits_processor is not None:
                    logger.error(
                        "Constrained decoding produced invalid JSON: %s", error
                    )
                else:
                    logger.warning(f"JSON validation failed: {error}")

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else output.finish_reason

        tracker.finish(
            result="success",
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
        )
        return ChatCompletionResponse(
            model=_model_name,
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=(
                            clean_output_text(cleaned_text) if cleaned_text else None
                        ),
                        reasoning=reasoning_text,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )
    finally:
        if release_on_exit:
            await _release_engine_for_request(raw_request)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize message roles and merge consecutive same-role messages.

    1. Maps non-standard roles to standard ones (e.g. ``developer`` -> ``system``).
    2. Merges consecutive same-role messages to satisfy chat template constraints
       (Qwen 3.5, Llama, etc. require alternating roles).

    Only merges when both messages have string content. Messages with list
    content (multimodal) are left as-is to preserve image/video attachments.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        New list with normalized roles and consecutive same-role messages merged.
    """
    # OpenAI Responses API uses "developer" instead of "system".
    # Map it so chat templates don't fail and fall back to raw prefill.
    _ROLE_MAP = {"developer": "system"}

    if not messages:
        return messages

    merged = [messages[0].copy()]
    if merged[0]["role"] in _ROLE_MAP:
        merged[0]["role"] = _ROLE_MAP[merged[0]["role"]]
    for msg in messages[1:]:
        prev = merged[-1]
        role = _ROLE_MAP.get(msg["role"], msg["role"])
        if (
            role == prev["role"]
            and isinstance(prev.get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            # Merge string content with double newline separator
            prev["content"] = prev["content"] + "\n\n" + msg["content"]
            logger.debug(
                f"Merged consecutive {role} messages "
                f"({len(prev['content'])} chars total)"
            )
        else:
            copy = msg.copy()
            copy["role"] = role
            merged.append(copy)

    mapped_roles = sum(1 for m in messages if m["role"] in _ROLE_MAP)
    merged_count = len(messages) - len(merged)
    if mapped_roles or merged_count:
        parts = []
        if mapped_roles:
            parts.append(f"mapped {mapped_roles} role(s)")
        if merged_count:
            parts.append(f"merged {len(messages)} -> {len(merged)}")
        logger.info(f"Normalized messages: {', '.join(parts)}")

    return merged


def _get_engine_tokenizer(engine) -> object | None:
    """
    Return the tokenizer backing ``engine``, if exposed.

    Different engine classes store the tokenizer under different attributes.
    We try the common ones and return ``None`` if nothing matches, so that
    optional features like constrained decoding can degrade gracefully.
    """
    for attr in ("_tokenizer", "tokenizer", "_processor", "processor"):
        tok = getattr(engine, attr, None)
        if tok is not None:
            return tok
    return None


@app.post(
    "/v1/responses",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_response(request: ResponsesRequest, raw_request: Request):
    """Create a Responses API response."""
    try:
        if request.stream:
            chat_request = _responses_request_to_chat_request(request)
            _validate_remote_media_urls(chat_request.messages)
            return StreamingResponse(
                _disconnect_guard(_stream_responses_request(request), raw_request),
                media_type="text/event-stream",
            )

        response_object, _persisted_messages = await _run_responses_request(
            request, raw_request
        )
    except UnsafeRemoteURLError as exc:
        _raise_remote_media_http_error(exc)

    if response_object is None:
        return Response(status_code=499)

    return response_object


def _get_forced_tool_name(tool_choice) -> str | None:
    """Extract forced tool name from tool_choice, if any.

    Returns the function name when tool_choice is a dict like
    {"type": "function", "function": {"name": "X"}}, or None otherwise.
    """
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") != "function":
        return None
    func = tool_choice.get("function")
    if isinstance(func, dict):
        return func.get("name")
    return None


def _apply_forced_tool_choice(tool_choice, tools, messages, chat_kwargs=None):
    """Apply forced tool_choice by filtering tools and injecting instructions.

    Handles:
    - tool_choice={"type":"function","function":{"name":"X"}} -> filter + instruct
    - tool_choice="required" -> instruct model to call at least one tool

    Args:
        tool_choice: The tool_choice value from the request
        tools: List of converted tools for the template
        messages: The message list (will be copied if modified)
        chat_kwargs: Optional dict to modify (e.g. disable thinking)

    Returns:
        Tuple of (tools, messages) - potentially filtered/modified
    """
    if not tools:
        return tools, messages

    forced_name = _get_forced_tool_name(tool_choice)
    if forced_name:
        # Filter tools to only the forced function
        filtered = [t for t in tools if _tool_name(t) == forced_name]
        if not filtered:
            available = [_tool_name(t) for t in tools if _tool_name(t)]
            raise ValueError(
                f"tool_choice function '{forced_name}' not found in tools. "
                f"Available: {available}"
            )
        tools = filtered
        instruction = (
            f"[IMPORTANT INSTRUCTION] You MUST call the `{forced_name}` function. "
            f"Do NOT respond with plain text. Respond ONLY with a tool call to "
            f"`{forced_name}`. This is mandatory."
        )
        messages = _inject_json_instruction(messages, instruction)
        # Disable thinking to prevent model from reasoning its way out
        if chat_kwargs is not None:
            chat_kwargs["enable_thinking"] = False
    elif tool_choice == "required":
        instruction = (
            "[IMPORTANT INSTRUCTION] You MUST call at least one of the available "
            "tools. Do NOT respond with plain text only."
        )
        messages = _inject_json_instruction(messages, instruction)

    return tools, messages


def _tool_name(tool: dict) -> str | None:
    """Extract function name from a tool definition dict."""
    func = tool.get("function")
    if isinstance(func, dict):
        return func.get("name")
    return None


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Find existing system message
    system_idx = None
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            system_idx = i
            break

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


# =============================================================================
# Anthropic Messages API Endpoints
# =============================================================================


def _convert_anthropic_stop_reason(openai_reason: str | None) -> str:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason or "", "end_turn")


def _prepare_anthropic_endpoint_invocation(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    effective_max_tokens: int,
) -> PreparedChatInvocation:
    """Prepare Anthropic invocation and convert URL-safety errors to 400s."""
    try:
        return _prepare_anthropic_invocation(
            engine,
            openai_request,
            effective_max_tokens,
        )
    except UnsafeRemoteURLError as exc:
        _raise_remote_media_http_error(exc)


@app.post(
    "/v1/messages", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def create_anthropic_message(
    request: Request,
):
    """
    Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI format, runs inference
    through the existing engine, and converts the response back.

    Supports both streaming and non-streaming modes.
    """
    tracker = _metrics.track_inference("anthropic_messages", stream=False)

    # Parse the raw body to handle Anthropic request format.
    # Some clients (e.g. Claude Code) may send JSON with invalid escape
    # sequences like \s, \d in regex patterns within tool definitions.
    # Python's json.loads is strict per RFC 8259 and rejects these.
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        if "Invalid \\escape" in str(e):
            raw = await request.body()
            # Replace lone backslashes (not valid JSON escapes) with \\
            body = json.loads(re.sub(rb'\\(?!["\\/bfnrtu])', rb"\\\\", raw))
        else:
            raise
    anthropic_request = AnthropicRequest(**body)

    _validate_model_name(anthropic_request.model)
    effective_max_tokens = _resolve_request_max_tokens(anthropic_request.max_tokens)

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
    logger.info(
        "[REQUEST] last user message preview: %s",
        _sanitize_log_text(last_user_preview, limit=300),
    )

    # Convert Anthropic request -> OpenAI request
    openai_request = anthropic_to_openai(anthropic_request)
    total_timeout, deadline = _start_request_budget(None)
    engine = await _acquire_default_engine_for_request(
        request,
        total_timeout=total_timeout,
        deadline=deadline,
        model=openai_request.model,
    )
    if engine is None:
        return Response(status_code=499)
    release_on_exit = True
    prepared = _prepare_anthropic_endpoint_invocation(
        engine,
        openai_request,
        effective_max_tokens,
    )

    try:
        if anthropic_request.stream:
            anthropic_terminal = (
                f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            )
            response = StreamingResponse(
                _disconnect_guard(
                    _ensure_sse_terminal(
                        _stream_anthropic_messages(
                            engine,
                            openai_request,
                            anthropic_request,
                            prepared,
                            metrics_tracker=tracker,
                        ),
                        anthropic_terminal,
                    ),
                    request,
                    cleanup=_make_release_cleanup(request),
                    timeout=total_timeout,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            release_on_exit = False
            return response

        start_time = time.perf_counter()
        try:
            output = await _wait_with_disconnect(
                engine.chat(messages=prepared.messages, **prepared.chat_kwargs),
                request,
                timeout=_remaining_request_timeout(total_timeout, deadline),
                timeout_detail_seconds=total_timeout,
            )
        except HTTPException as exc:
            tracker.finish(result=_metrics_result_from_status(exc.status_code))
            raise
        if output is None:
            tracker.finish(result="client_closed")
            return Response(status_code=499)  # Client closed request

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        reasoning_text, cleaned_text, tool_calls = _extract_reasoning_and_tool_calls(
            output.text,
            openai_request,
            allow_reasoning=(
                prepared.json_logits_processor is None
                or isinstance(
                    prepared.json_logits_processor,
                    _ThinkingAwareLogitsProcessor,
                )
            ),
            engine=engine,
        )

        if prepared.response_format and not tool_calls:
            json_input = cleaned_text or output.text
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, prepared.response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
            if not is_valid:
                if prepared.json_logits_processor is not None:
                    logger.error(
                        "Constrained decoding produced invalid JSON on Anthropic endpoint: %s",
                        error,
                    )
                else:
                    logger.warning(
                        "JSON validation failed on Anthropic endpoint: %s", error
                    )

        # Clean output text
        final_content = None
        if cleaned_text:
            final_content = clean_output_text(cleaned_text)

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else output.finish_reason

        # Build Anthropic content blocks directly (with thinking support)
        content_blocks = []

        if reasoning_text:
            content_blocks.append(
                AnthropicResponseContentBlock(type="thinking", thinking=reasoning_text)
            )

        if final_content:
            content_blocks.append(
                AnthropicResponseContentBlock(type="text", text=final_content)
            )

        if tool_calls:
            for tc in tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}
                content_blocks.append(
                    AnthropicResponseContentBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=tool_input,
                    )
                )

        if not content_blocks:
            content_blocks.append(AnthropicResponseContentBlock(type="text", text=""))

        stop_reason = _convert_anthropic_stop_reason(
            "tool_calls" if tool_calls else output.finish_reason
        )

        anthropic_response = AnthropicResponse(
            model=_model_name,
            content=content_blocks,
            stop_reason=stop_reason,
            usage=AnthropicUsage(
                input_tokens=output.prompt_tokens,
                output_tokens=output.completion_tokens,
            ),
        )
        tracker.finish(
            result="success",
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
        )
        return Response(
            content=anthropic_response.model_dump_json(exclude_none=True),
            media_type="application/json",
        )
    finally:
        if release_on_exit:
            await _release_engine_for_request(request)


@app.post(
    "/v1/messages/count_tokens",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def count_anthropic_tokens(request: Request):
    """
    Count tokens for an Anthropic Messages API request.

    Uses the model's tokenizer for accurate counting.
    Claude Code calls this endpoint for token budgeting.
    Note: Don't parse via AnthropicRequest — count_tokens requests
    from Claude Code don't include max_tokens.
    """
    body = await request.json()
    request_model = body.get("model")
    if isinstance(request_model, str) and request_model:
        _validate_model_name(request_model)
    total_timeout, deadline = _start_request_budget(None)
    engine = await _acquire_default_engine_for_request(
        request,
        total_timeout=total_timeout,
        deadline=deadline,
        count_activity=False,
        model=request_model,
    )
    if engine is None:
        return Response(status_code=499)

    tokenizer = engine.tokenizer

    total_tokens = 0

    try:
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
                        # tool_use input
                        if block.get("input"):
                            total_tokens += len(
                                tokenizer.encode(json.dumps(block["input"]))
                            )
                        # tool_result content
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
    finally:
        await _release_engine_for_request(request)


def _emit_content_pieces(
    pieces: list[tuple[str, str]],
    current_block_type: str | None,
    block_index: int,
) -> tuple[list[str], str | None, int]:
    """Emit Anthropic SSE events for content pieces from the think router.

    Handles block type transitions (thinking <-> text), emitting
    content_block_start/stop/delta events as needed.

    Args:
        pieces: List of (block_type, text) from StreamingThinkRouter
        current_block_type: Current open block type, or None
        block_index: Current block index

    Returns:
        Tuple of (events, updated_block_type, updated_block_index)
    """
    events = []
    for block_type, text in pieces:
        if block_type != current_block_type:
            # Close previous block if open
            if current_block_type is not None:
                events.append(
                    f"event: content_block_stop\ndata: "
                    f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                )
                block_index += 1
            # Start new block
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
        # Emit delta
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
    prepared: PreparedChatInvocation,
    metrics_tracker=None,
) -> AsyncIterator[str]:
    """
    Stream Anthropic Messages API SSE events.

    Converts OpenAI streaming chunks to Anthropic event format:
    message_start -> content_block_start -> content_block_delta* ->
    content_block_stop -> message_delta -> message_stop

    When a reasoning parser is active, emits a ``thinking`` content block
    (index 0) for reasoning tokens and a ``text`` content block (index 1)
    for the actual response, matching the Anthropic extended thinking format.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()
    result_label = "success"
    prompt_tokens = 0

    messages = prepared.messages
    chat_kwargs = dict(prepared.chat_kwargs)

    # Emit message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": _model_name,
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

    use_reasoning = _reasoning_parser is not None and not chat_kwargs.get(
        "logits_processors"
    )

    if use_reasoning:
        _reasoning_parser.reset_state()

    # Block index tracking: with reasoning parser we use index 0 for
    # thinking and index 1 for text; without parser, index 0 for text.
    thinking_block_started = False
    text_block_started = False
    thinking_index = 0
    text_index = 1 if use_reasoning else 0

    if not use_reasoning:
        # No reasoning parser — start text block immediately
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        text_block_started = True

    # Stream content deltas
    accumulated_text = ""
    completion_tokens = 0

    # Tool call streaming suppression — prevents raw tool markup from leaking
    # as text_delta events. Mirrors the OpenAI streaming path logic.
    tool_parser = None
    tool_accumulated_text = ""
    tool_markup_possible = False
    tool_parser = _get_streaming_tool_parser(openai_request, engine)

    try:
        async for output in engine.stream_chat(messages=messages, **chat_kwargs):
            if metrics_tracker is not None:
                metrics_tracker.observe_ttft()
            delta_text = output.new_text

            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens

            # Track token counts
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            if not delta_text:
                continue

            # Filter special tokens
            filtered = SPECIAL_TOKENS_PATTERN.sub("", delta_text)
            if not filtered:
                continue

            if not use_reasoning:
                # Simple path — no reasoning parsing
                accumulated_text += filtered
                content_to_emit = filtered

                # Filter tool call markup during streaming
                if tool_parser and content_to_emit:
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible_after_delta(
                            tool_accumulated_text, content_to_emit
                        )
                    ):
                        tool_accumulated_text += content_to_emit
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += content_to_emit
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, content_to_emit
                        )
                        if tool_result is None or "tool_calls" in tool_result:
                            # Inside tool markup or tool calls detected — suppress
                            continue
                        content_to_emit = tool_result.get("content", "")
                        if content_to_emit:
                            content_to_emit = _TOOL_MARKUP_PATTERN.sub(
                                "", content_to_emit
                            )
                        if not content_to_emit:
                            continue

                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content_to_emit}})}\n\n"
                continue

            # Reasoning parser path
            previous_text = accumulated_text
            accumulated_text += filtered
            delta_msg = _reasoning_parser.extract_reasoning_streaming(
                previous_text, accumulated_text, filtered
            )

            if delta_msg is None:
                continue

            if delta_msg.reasoning:
                if not thinking_block_started:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': thinking_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                    thinking_block_started = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': thinking_index, 'delta': {'type': 'thinking_delta', 'thinking': delta_msg.reasoning}})}\n\n"

            if delta_msg.content:
                content_to_emit = delta_msg.content

                # Filter tool call markup during streaming
                if tool_parser and content_to_emit:
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible_after_delta(
                            tool_accumulated_text, content_to_emit
                        )
                    ):
                        tool_accumulated_text += content_to_emit
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += content_to_emit
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, content_to_emit
                        )
                        if tool_result is None or "tool_calls" in tool_result:
                            # Inside tool markup or tool calls detected — suppress
                            continue
                        content_to_emit = tool_result.get("content", "")
                        if content_to_emit:
                            content_to_emit = _TOOL_MARKUP_PATTERN.sub(
                                "", content_to_emit
                            )
                        if not content_to_emit:
                            continue

                if thinking_block_started and not text_block_started:
                    # Close thinking block, open text block
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_started = True
                elif not text_block_started:
                    # No thinking was emitted, start text block at index 0
                    text_index = 0
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    text_block_started = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': content_to_emit}})}\n\n"

        # Close any open thinking block that was never followed by text
        if thinking_block_started and not text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': thinking_index})}\n\n"
            # Emit empty text block so response always has text content
            text_index = thinking_index + 1
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True

        # Check for tool calls in accumulated text
        _, tool_calls = _parse_tool_calls_with_parser(
            accumulated_text,
            openai_request,
            engine=engine,
        )

        # Close text block
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_index})}\n\n"

        # If there are tool calls, emit tool_use blocks
        next_index = (text_index + 1) if text_block_started else 0
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                tool_index = next_index + i
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}

                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': tool_index, 'content_block': {'type': 'tool_use', 'id': tc.id, 'name': tc.function.name, 'input': {}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tool_input)}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

        # Determine stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"

        # Emit message_delta with stop_reason and usage
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": completion_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Anthropic messages (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Emit message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    except HTTPException as exc:
        result_label = _metrics_result_from_status(exc.status_code)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        result_label = "cancelled"
        raise
    except Exception:
        result_label = "error"
        raise
    finally:
        if metrics_tracker is not None:
            metrics_tracker.finish(
                result=result_label,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


# =============================================================================
# Streaming Helpers
# =============================================================================


async def stream_completion(
    engine: BaseEngine,
    prompt: str,
    request: CompletionRequest,
    max_tokens: int,
    repetition_penalty: float | None = None,
    metrics_tracker=None,
) -> AsyncIterator[str]:
    """Stream completion response."""
    result = "success"
    prompt_tokens = 0
    completion_tokens = 0
    generate_kwargs = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "top_k": request.top_k or 0,
        "min_p": request.min_p or 0.0,
        "presence_penalty": request.presence_penalty or 0.0,
        "stop": request.stop,
    }
    if repetition_penalty is not None:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    if request.specprefill is not None:
        generate_kwargs["specprefill"] = request.specprefill
    if request.specprefill_keep_pct is not None:
        generate_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct

    try:
        async for output in engine.stream_generate(**generate_kwargs):
            if metrics_tracker is not None:
                metrics_tracker.observe_ttft()
            prompt_tokens = (
                output.prompt_tokens
                if hasattr(output, "prompt_tokens")
                else prompt_tokens
            )
            completion_tokens = (
                output.completion_tokens
                if hasattr(output, "completion_tokens")
                else completion_tokens
            )
            data = {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": _model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": output.new_text,
                        "finish_reason": (
                            output.finish_reason if output.finished else None
                        ),
                    }
                ],
            }
            if output.finished:
                data["usage"] = get_usage(output).model_dump()
            yield f"data: {json.dumps(data)}\n\n"
    except HTTPException as exc:
        result = _metrics_result_from_status(exc.status_code)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        result = "cancelled"
        raise
    except Exception:
        result = "error"
        raise
    finally:
        yield "data: [DONE]\n\n"
        if metrics_tracker is not None:
            metrics_tracker.finish(
                result=result,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


async def stream_chat_completion(
    engine: BaseEngine,
    messages: list,
    request: ChatCompletionRequest,
    metrics_tracker=None,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    result_label = "success"

    # Check if we should include usage in the final chunk
    include_usage = request.stream_options and request.stream_options.include_usage

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=response_id,
        model=_model_name,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Track if we need to add <think> prefix for thinking models (when no reasoning parser)
    # The template adds <think> to the prompt, so the model output starts inside the think block
    is_thinking_model = (
        "nemotron" in (engine.model_name or "").lower() and not _reasoning_parser
    )
    think_prefix_sent = False

    # Reset reasoning parser state for this stream
    if _reasoning_parser:
        _reasoning_parser.reset_state()

    # Track accumulated text for reasoning parser
    accumulated_text = ""

    # Track token counts for usage reporting
    prompt_tokens = 0
    completion_tokens = 0
    last_output = None

    # Response-format streaming filter — strip markdown code fences from
    # content when client asked for JSON. Non-streaming path strips fences
    # via ``parse_json_output``; without this, streaming clients see
    # ``"```json{...}```"`` instead of ``"{...}"`` for models that wrap
    # their structured output in markdown (e.g. Gemma 4).
    fence_stripper: StreamingJsonFenceStripper | None = None
    _rf = getattr(request, "response_format", None)
    _rf_type = None
    if _rf is not None:
        _rf_type = getattr(_rf, "type", None)
        if _rf_type is None and isinstance(_rf, dict):
            _rf_type = _rf.get("type")
    if _rf_type in ("json_object", "json_schema"):
        fence_stripper = StreamingJsonFenceStripper()

    # Tool call streaming state
    tool_parser = None
    tool_accumulated_text = ""
    tool_calls_detected = False
    tool_markup_possible = False  # Fast path: skip parsing until markers appear
    tool_parser = _get_streaming_tool_parser(request, engine)

    try:
        # Stream content
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if metrics_tracker is not None:
                metrics_tracker.observe_ttft()
            delta_text = output.new_text
            last_output = output

            # Track token counts from output (updated each chunk)
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            # Use reasoning parser if enabled (skip when enable_thinking=False)
            if (
                _reasoning_parser
                and delta_text
                and request.enable_thinking is not False
            ):
                previous_text = accumulated_text
                accumulated_text += delta_text
                delta_msg = _reasoning_parser.extract_reasoning_streaming(
                    previous_text, accumulated_text, delta_text
                )

                if delta_msg is None:
                    # Skip this chunk (e.g., <think> token itself)
                    continue

                content = delta_msg.content
                reasoning = delta_msg.reasoning

                # Some models (e.g. MiniMax) wrap tool calls in <think>
                # blocks, so reasoning parser captures tool call XML as
                # reasoning while content stays None.  Redirect reasoning
                # to the content stream so the tool parser can handle it.
                if tool_parser and reasoning and not content:
                    _check = tool_accumulated_text + reasoning
                    if (
                        "<minimax:tool_call>" in _check
                        or "<tool_call>" in _check
                        or '<invoke name="' in _check
                    ):
                        content = reasoning
                        reasoning = None

                # Tool call parsing on content portion
                if tool_parser and content:
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible_after_delta(
                            tool_accumulated_text, content
                        )
                    ):
                        tool_accumulated_text += content
                        # Emit as-is; no tool markup is even possible yet.
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += content
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, content
                        )

                        if tool_result is None:
                            # Inside tool markup - suppress content output
                            if reasoning:
                                # Still emit reasoning while buffering tool call
                                chunk = ChatCompletionChunk(
                                    id=response_id,
                                    model=_model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            delta=ChatCompletionChunkDelta(
                                                reasoning=reasoning,
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                    usage=None,
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        if "tool_calls" in tool_result:
                            # Emit structured tool calls
                            tool_calls_detected = True
                            # Coerce arguments against tool schemas
                            tools = (
                                request.model_dump().get("tools")
                                if request and request.tools
                                else None
                            )
                            if tools:
                                for tc in tool_result["tool_calls"]:
                                    fn = tc.get("function", {})
                                    if "arguments" in fn and "name" in fn:
                                        fn["arguments"] = _coerce_tool_arguments(
                                            fn["arguments"], fn["name"], tools
                                        )
                            chunk = ChatCompletionChunk(
                                id=response_id,
                                model=_model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        delta=ChatCompletionChunkDelta(
                                            tool_calls=tool_result["tool_calls"],
                                            reasoning=reasoning,
                                        ),
                                        finish_reason=(
                                            "tool_calls" if output.finished else None
                                        ),
                                    )
                                ],
                                usage=get_usage(output) if output.finished else None,
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        # Normal content from tool parser
                        content = tool_result.get("content", "")
                        # Strip any leaked tool markup tags
                        if content:
                            content = _TOOL_MARKUP_PATTERN.sub("", content)

                # Strip markdown code fences when response_format is set.
                if fence_stripper is not None and not tool_calls_detected:
                    content = fence_stripper.feed(content) if content else ""
                    if output.finished:
                        flush = fence_stripper.finalize()
                        if flush:
                            content = content + flush

                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=content if content else None,
                                reasoning=reasoning,
                            ),
                            finish_reason=(
                                "tool_calls"
                                if (output.finished and tool_calls_detected)
                                else (output.finish_reason if output.finished else None)
                            ),
                        )
                    ],
                    usage=get_usage(output) if output.finished else None,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            else:
                # Standard path without reasoning parsing
                content = delta_text

                # Filter special tokens that may leak into streaming output
                if content:
                    content = SPECIAL_TOKENS_PATTERN.sub("", content)

                # Add <think> prefix on first content chunk for thinking models
                if is_thinking_model and not think_prefix_sent and content:
                    content = "<think>" + content
                    think_prefix_sent = True

                # Tool call streaming parsing
                if tool_parser and delta_text:
                    # Fast path: skip full parsing until likely tool markup appears.
                    # This preserves the cheap path for ordinary text while still
                    # allowing generic streaming tool parsing when no explicit
                    # parser flags are configured.
                    if (
                        not tool_markup_possible
                        and not _streaming_tool_markup_possible_after_delta(
                            tool_accumulated_text, delta_text
                        )
                    ):
                        tool_accumulated_text += delta_text
                        # No tool markup yet, fall through to normal chunk emission
                    else:
                        if not tool_markup_possible:
                            tool_markup_possible = True
                        tool_previous = tool_accumulated_text
                        tool_accumulated_text += delta_text
                        tool_result = tool_parser.extract_tool_calls_streaming(
                            tool_previous, tool_accumulated_text, delta_text
                        )

                        if tool_result is None:
                            # Inside tool markup - suppress output
                            continue

                        if "tool_calls" in tool_result:
                            # Emit structured tool calls
                            tool_calls_detected = True
                            # Coerce arguments against tool schemas
                            tools = (
                                request.model_dump().get("tools")
                                if request and request.tools
                                else None
                            )
                            if tools:
                                for tc in tool_result["tool_calls"]:
                                    fn = tc.get("function", {})
                                    if "arguments" in fn and "name" in fn:
                                        fn["arguments"] = _coerce_tool_arguments(
                                            fn["arguments"], fn["name"], tools
                                        )
                            chunk = ChatCompletionChunk(
                                id=response_id,
                                model=_model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        delta=ChatCompletionChunkDelta(
                                            tool_calls=tool_result["tool_calls"]
                                        ),
                                        finish_reason=(
                                            "tool_calls" if output.finished else None
                                        ),
                                    )
                                ],
                                usage=get_usage(output) if output.finished else None,
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue

                        # Normal content from tool parser
                        content = tool_result.get("content", "")
                        # Strip any leaked tool markup tags
                        if content:
                            content = _TOOL_MARKUP_PATTERN.sub("", content)

                # Strip markdown code fences when response_format is set.
                if fence_stripper is not None and not tool_calls_detected:
                    content = fence_stripper.feed(content) if content else ""
                    if output.finished:
                        flush = fence_stripper.finalize()
                        if flush:
                            content = content + flush

                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                content=content if content else None
                            ),
                            finish_reason=(
                                "tool_calls"
                                if (output.finished and tool_calls_detected)
                                else (output.finish_reason if output.finished else None)
                            ),
                        )
                    ],
                    usage=get_usage(output) if output.finished else None,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Fallback: if tool parser accumulated text but never emitted tool_calls
        # (e.g., </tool_call> never arrived, or <function= block still incomplete)
        if (
            tool_parser
            and tool_accumulated_text
            and not tool_calls_detected
            and _streaming_tool_markup_possible(tool_accumulated_text)
        ):
            final_parse_result = tool_parser.extract_tool_calls(tool_accumulated_text)
            if final_parse_result.tools_called:
                tools = (
                    request.model_dump().get("tools")
                    if request and request.tools
                    else None
                )
                tool_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=[
                                    {
                                        "index": i,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": _coerce_tool_arguments(
                                                tc["arguments"], tc["name"], tools
                                            ),
                                        },
                                    }
                                    for i, tc in enumerate(
                                        final_parse_result.tool_calls
                                    )
                                ]
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                yield f"data: {tool_chunk.model_dump_json()}\n\n"

        # Safety-net validation: if response_format was requested, verify the
        # accumulated output still parses.  When constrained decoding is active
        # this should always succeed; if it fails we log loudly (error) so we
        # notice grammar-integration regressions.  When constrained decoding was
        # *not* active (optional dep missing, incompatible tokenizer, combined
        # with tools), we log at warning level only — the prompt-only path is
        # best-effort.
        if (
            getattr(request, "response_format", None) is not None
            and not tool_calls_detected
        ):
            try:
                _, _parsed, _is_valid, _err = parse_json_output(
                    accumulated_text, request.response_format
                )
                if not _is_valid:
                    # Determine whether constrained decoding was wired up.  We
                    # passed the processor through ``kwargs`` so its presence is
                    # the signal.
                    has_constrained = any(
                        p.__class__.__name__ == "JSONSchemaLogitsProcessor"
                        for p in (kwargs.get("logits_processors") or [])
                    )
                    if has_constrained:
                        logger.error(
                            "Streaming constrained decoding produced invalid JSON: %s",
                            _err,
                        )
                    else:
                        logger.warning("Streaming JSON validation failed: %s", _err)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Streaming JSON validation raised: %s", exc)

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
                model=_model_name,
                choices=[],  # Empty choices for usage-only chunk
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
    except HTTPException as exc:
        result_label = _metrics_result_from_status(exc.status_code)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        result_label = "cancelled"
        raise
    except Exception:
        result_label = "error"
        raise
    finally:
        if metrics_tracker is not None:
            metrics_tracker.finish(
                result=result_label,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )


# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager, _mcp_executor

    try:
        from vllm_mlx.mcp import (
            MCPClientManager,
            ToolExecutor,
            ToolSandbox,
            load_mcp_config,
        )

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        sandbox = ToolSandbox(allowed_high_risk_tools=config.allowed_high_risk_tools)
        _mcp_executor = ToolExecutor(_mcp_manager, sandbox=sandbox)

        logger.info(f"MCP initialized with {len(_mcp_manager.get_all_tools())} tools")

    except ImportError:
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error("Failed to initialize MCP: %s", _sanitize_log_text(e, limit=500))
        raise


# =============================================================================
# TCP Keepalive
# =============================================================================


def _make_keepalive_http_protocol(idle=10, interval=5, count=3):
    """Create a uvicorn HTTP protocol class with aggressive TCP keepalive.

    When a client abruptly disconnects (power-off, network loss), the server
    TCP stack won't notice for ~2 hours (default keepalive).  With aggressive
    keepalive (idle=10s, interval=5s, count=3), dead connections are detected
    in ~25 seconds, letting ``_wait_with_disconnect()`` abort the request and
    stop wasting GPU cycles on tokens nobody will receive.
    """
    from uvicorn.protocols.http.h11_impl import H11Protocol

    _Base = H11Protocol

    class _KeepaliveProtocol(_Base):
        def connection_made(self, transport):
            super().connection_made(transport)
            sock = transport.get_extra_info("socket")
            if sock is None:
                return
            try:
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
                # macOS: TCP_KEEPALIVE (idle time), Linux: TCP_KEEPIDLE
                if hasattr(_socket, "TCP_KEEPALIVE"):
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPALIVE, idle)
                elif hasattr(_socket, "TCP_KEEPIDLE"):
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, idle)
                if hasattr(_socket, "TCP_KEEPINTVL"):
                    sock.setsockopt(
                        _socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, interval
                    )
                if hasattr(_socket, "TCP_KEEPCNT"):
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, count)
            except OSError:
                pass  # best-effort; some platforms may not support all options

    return _KeepaliveProtocol


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    parser = create_parser()
    args = parser.parse_args()

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter, _metrics_enabled
    global _default_temperature, _default_top_p, _default_chat_template_kwargs
    global _max_audio_upload_bytes, _max_tts_input_chars
    _api_key = args.api_key
    _default_timeout = args.timeout
    _metrics_enabled = args.enable_metrics
    _metrics.configure(enabled=args.enable_metrics)
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p
    _default_chat_template_kwargs = args.default_chat_template_kwargs
    _max_audio_upload_bytes = args.max_audio_upload_mb * 1024 * 1024
    _max_tts_input_chars = args.max_tts_input_chars

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    if args.enable_metrics:
        logger.info("  Metrics: ENABLED (/metrics, unauthenticated)")
    else:
        logger.info("  Metrics: DISABLED - Use --enable-metrics to expose /metrics")
    if args.auto_unload_idle_seconds > 0:
        logger.info(
            "  Idle auto-unload: ENABLED (%.0fs)", args.auto_unload_idle_seconds
        )
    else:
        logger.info("  Idle auto-unload: DISABLED")
    if args.trust_remote_code:
        logger.warning("  Remote code loading: ENABLED (--trust-remote-code)")
    else:
        logger.info("  Remote code loading: DISABLED (default)")
    logger.info(
        f"  Audio upload limit: {args.max_audio_upload_mb} MiB, "
        f"TTS input limit: {args.max_tts_input_chars} chars"
    )
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Initialize reasoning parser if specified
    if args.reasoning_parser:
        global _reasoning_parser, _reasoning_parser_name
        from .reasoning import get_parser

        _reasoning_parser_name = args.reasoning_parser
        parser_cls = get_parser(args.reasoning_parser)
        _reasoning_parser = parser_cls()
        logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=args.continuous_batching,
        max_tokens=args.max_tokens,
        max_request_tokens=args.max_request_tokens,
        force_mllm=args.mllm,
        trust_remote_code=args.trust_remote_code,
        auto_unload_idle_seconds=args.auto_unload_idle_seconds,
        lazy_load_model=args.lazy_load_model,
    )

    # Start server with TCP keepalive for fast dead-client detection.
    # Without this, abrupt client disconnects (power-off, network loss) take
    # 2+ hours to detect via default TCP keepalive, wasting GPU cycles.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        http=_make_keepalive_http_protocol(),
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the standalone server CLI parser."""
    parser = argparse.ArgumentParser(
        description="vllm-mlx OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: localhost; use 0.0.0.0 to expose externally)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow HuggingFace remote code execution during model/tokenizer loading",
    )
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Enable continuous batching for multiple concurrent users",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Default max tokens for generation",
    )
    parser.add_argument(
        "--max-request-tokens",
        type=int,
        default=32768,
        help="Maximum max_tokens accepted from API clients (default: 32768)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Expose Prometheus metrics on /metrics (disabled by default)",
    )
    parser.add_argument(
        "--auto-unload-idle-seconds",
        type=float,
        default=0.0,
        help="Unload the main model after this many idle seconds (0 = disabled)",
    )
    parser.add_argument(
        "--lazy-load-model",
        action="store_true",
        help="Register the main model at startup but defer loading until first request",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Pre-load an embedding model at startup (e.g. mlx-community/all-MiniLM-L6-v2-4bit)",
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default temperature for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top_p for generation when not specified in request",
    )
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=make_json_object_arg_parser("--default-chat-template-kwargs"),
        default=None,
        help=(
            "Default chat template kwargs to apply to all requests when request "
            "chat_template_kwargs is omitted or empty; empty request kwargs use "
            'existing server defaults (JSON object, e.g. {"enable_thinking": false})'
        ),
    )
    parser.add_argument(
        "--max-audio-upload-mb",
        type=int,
        default=DEFAULT_MAX_AUDIO_UPLOAD_MB,
        help="Maximum size of uploaded audio files in MiB (default: 25)",
    )
    parser.add_argument(
        "--max-tts-input-chars",
        type=int,
        default=DEFAULT_MAX_TTS_INPUT_CHARS,
        help="Maximum number of characters accepted by /v1/audio/speech (default: 4096)",
    )
    return parser


if __name__ == "__main__":
    main()
