# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for route handlers.

These functions were extracted from server.py to enable route modules
(chat, completions, anthropic) to share common logic without importing
from the monolithic server module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import HTTPException
from starlette.requests import Request

from ..api.models import (
    CompletionTokensDetails,
    FunctionCall,
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

# Tool-use system prompt (auto-injected when tools are provided and parser is active)
_TOOL_USE_SYSTEM_SUFFIX = (
    "\n\nIMPORTANT: When the user's request can be answered using the provided tools, "
    "you MUST use the appropriate tool immediately. Do NOT ask for clarification when "
    "a reasonable default exists. Do NOT explain what you will do — just do it. "
    "Be direct and concise in your responses. "
    "Do NOT think out loud or show your reasoning process. "
    "Give direct answers only — no preamble like 'The user asks...' or 'Let me think...'."
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
    """Resolve max_tokens with thinking budget for reasoning models."""
    cfg = get_config()
    base = request_value if request_value is not None else cfg.default_max_tokens
    if enable_thinking is False:
        return base
    if cfg.reasoning_parser_name and base > 0 and base < 4096:
        return base + cfg.thinking_token_budget
    return base


def _resolve_temperature(request_value: float | None) -> float:
    """Resolve temperature: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    if cfg.default_temperature is not None:
        return cfg.default_temperature
    return _FALLBACK_TEMPERATURE


def _resolve_top_p(request_value: float | None) -> float:
    """Resolve top_p: request > CLI default > fallback."""
    if request_value is not None:
        return request_value
    cfg = get_config()
    if cfg.default_top_p is not None:
        return cfg.default_top_p
    return _FALLBACK_TOP_P


# ── Usage / logprobs ───────────────────────────────────────────────


def _build_usage(output: GenerationOutput, reasoning_text: str | None) -> Usage:
    """Build Usage with reasoning token breakdown when applicable."""
    cfg = get_config()
    total_completion = output.completion_tokens
    if reasoning_text and cfg.reasoning_parser_name:
        reasoning_tokens = max(1, len(reasoning_text) // 4)
        reasoning_tokens = min(reasoning_tokens, total_completion)
        return Usage(
            prompt_tokens=output.prompt_tokens,
            completion_tokens=total_completion,
            total_tokens=output.prompt_tokens + total_completion,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
            ),
        )
    return Usage(
        prompt_tokens=output.prompt_tokens,
        completion_tokens=total_completion,
        total_tokens=output.prompt_tokens + total_completion,
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
    if not request_model:
        return

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
    output_text: str, request=None
) -> tuple[str, list | None]:
    """Parse tool calls from model output using the configured parser.

    Creates a per-call parser instance to avoid state corruption under
    concurrent BatchedEngine requests.
    """
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
        logger.warning(
            f"Failed to create tool parser '{cfg.tool_call_parser}': {e}"
        )
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

        if hasattr(cfg.engine, "_prefix_cache") and cfg.engine._prefix_cache is not None:
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
) -> AsyncIterator[str]:
    """Wrap streaming generator to abort on client disconnect."""
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
        logger.info(
            f"[disconnect_guard] CLEANUP done, {chunk_count} chunks total, elapsed={_elapsed()}"
        )


async def _wait_with_disconnect(
    coro,
    raw_request: Request,
    timeout: float,
    poll_interval: float = 0.5,
):
    """Run a coroutine with both timeout and client disconnect detection."""
    import time as _time

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

        return task.result()

    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        if not task.done():
            task.cancel()
