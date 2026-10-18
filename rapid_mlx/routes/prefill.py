# SPDX-License-Identifier: Apache-2.0
"""Explicit, output-free prefix warming for local clients."""

from __future__ import annotations

import asyncio
import copy
import time
from collections import OrderedDict
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import disk_caches
from ..api.models import Message
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import decode_inline_tool_call_arguments, extract_multimodal_content
from ..config import get_config
from ..middleware.auth import check_rate_limit_or_x_api_key, verify_api_key_or_x_api_key
from ..runtime.cache import save_prefix_cache_to_disk
from ..service.helpers import (
    _TOOL_USE_SYSTEM_SUFFIX,
    _append_tool_use_suffix,
    enforce_context_length,
    enforce_context_length_for_messages,
)

router = APIRouter(
    dependencies=[
        Depends(verify_api_key_or_x_api_key),
        Depends(check_rate_limit_or_x_api_key),
    ]
)

_guard = asyncio.Lock()
_active: asyncio.Task | None = None
_active_key: str | None = None
_active_revision = -1
_draft_revisions: OrderedDict[str, tuple[int, float]] = OrderedDict()
_REVISION_TTL_SECONDS = 300
_MAX_REVISION_KEYS = 128


class PrefillRequest(BaseModel):
    """One complete system prefix or one replaceable conversation draft."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["prefix", "draft"]
    messages: list[dict] = Field(min_length=1, max_length=64)
    tools: list[dict] = Field(default_factory=list, max_length=64)
    draft_key: str | None = Field(default=None, min_length=1, max_length=64)
    revision: int | None = Field(default=None, ge=0)
    persist: bool = False
    enable_thinking: bool | None = None
    allow_disk_cache: bool = False

    @model_validator(mode="after")
    def validate_shape(self):
        if (
            sum(len(str(message)) for message in self.messages)
            + sum(len(str(tool)) for tool in self.tools)
            > 500_000
        ):
            raise ValueError("prefill input is too large")
        for message in self.messages:
            if message.get("role") not in {
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
            }:
                raise ValueError("prefill accepts known chat roles only")
            content = message.get("content")
            if not isinstance(content, str) and not (
                message["role"] == "assistant"
                and content is None
                and isinstance(message.get("tool_calls"), list)
                and message["tool_calls"]
            ):
                raise ValueError("prefill accepts text messages and tool calls only")
        if self.kind == "prefix":
            if len(self.messages) != 1 or self.messages[0]["role"] != "system":
                raise ValueError("prefix prefill requires one system message")
            if self.draft_key is not None or self.revision is not None:
                raise ValueError("prefix prefill cannot use a draft key")
        else:
            if self.messages[-1]["role"] != "user":
                raise ValueError("a draft must end with a user message")
            if self.draft_key is None or self.revision is None:
                raise ValueError("draft_key and revision are required")
            if self.persist:
                raise ValueError("draft prefill cannot request a disk checkpoint")
        return self


def _render_inputs(
    body: PrefillRequest,
) -> tuple[list[dict], list[dict] | None, bool | None]:
    """Apply the same tool prompt and schema conversion as chat completions."""
    cfg = get_config()
    engine = cfg.engine
    if getattr(engine, "is_mllm", False):
        messages = [
            {
                key: value
                for key, value in copy.deepcopy(message).items()
                if value is not None
            }
            for message in body.messages
        ]
        if getattr(engine, "preserve_native_tool_format", False):
            decode_inline_tool_call_arguments(messages)
    else:
        messages, _, _ = extract_multimodal_content(
            cast(list[Message], body.messages),
            preserve_native_format=bool(
                getattr(engine, "preserve_native_tool_format", False)
            ),
        )
    for message in messages:
        if message["role"] == "developer":
            message["role"] = "system"
    if body.tools and getattr(cfg, "tool_call_parser", None):
        for message in messages:
            if message["role"] == "system":
                message["content"] = _append_tool_use_suffix(
                    message["content"], _TOOL_USE_SYSTEM_SUFFIX
                )
                break
        else:
            messages.insert(
                0, {"role": "system", "content": _TOOL_USE_SYSTEM_SUFFIX.strip()}
            )
    tools = convert_tools_for_template(body.tools) if body.tools else None
    thinking = False if getattr(cfg, "no_thinking", False) else body.enable_thinking
    if thinking is None and (body.tools or getattr(cfg, "reasoning_parser_name", None)):
        thinking = False
    return messages, tools, thinking


async def _warm(engine, body: PrefillRequest) -> dict:
    messages, tools, thinking = _render_inputs(body)
    stream = engine.stream_chat(
        messages=messages,
        tools=tools,
        max_tokens=1,
        temperature=0,
        enable_thinking=thinking,
    )

    async def _consume():
        last = None
        try:
            async for output in stream:
                last = output
                if output.finished:
                    break
        finally:
            await stream.aclose()
        return last

    consume = asyncio.create_task(_consume())
    try:
        # Waiting does not propagate the stream's outcome, so cancellation
        # here always belongs to this warm task, even on Python 3.10.
        await asyncio.wait({consume})
    except asyncio.CancelledError:
        consume.cancel()
        while not consume.done():
            try:
                await asyncio.wait({consume})
            except asyncio.CancelledError:
                continue
        raise
    last = consume.result()
    if last is None or not last.finished:
        raise HTTPException(status_code=503, detail="Prefill did not complete")
    result = {
        "status": "prefilled",
        "prompt_tokens": getattr(last, "prompt_tokens", 0),
        "cached_tokens": getattr(last, "cached_tokens", 0),
        "cache_saved": False,
    }
    if body.persist and not disk_caches.disabled():
        checkpoint_task = asyncio.create_task(
            asyncio.to_thread(save_prefix_cache_to_disk, 30)
        )
        try:
            result["cache_saved"] = await asyncio.shield(checkpoint_task)
        except asyncio.CancelledError:
            # A worker thread cannot be cancelled. Keep this slot until the
            # bounded save exits rather than overlap it with a replacement.
            while not checkpoint_task.done():
                try:
                    await asyncio.shield(checkpoint_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            raise
    return result


@router.post("/v1/cache/prefill")
async def prefill(body: PrefillRequest, request: Request):
    """Warm KV state without sending generated text to the caller.

    A new draft revision cancels its predecessor. One warm operation per
    process bounds speculative memory use; ordinary chat requests continue.
    ``persist`` requests an immediate cache save for fixed prefixes. A true
    ``cache_saved`` means at least one entry committed; it does not prove the
    requested entry survived a partial save. Every prefill requires explicit
    consent when the server's normal disk cache is enabled: a later shutdown
    can persist the warm KV state.
    """
    global _active, _active_key, _active_revision
    cfg = get_config()
    if not cfg.ready or cfg.draining or cfg.engine is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    if not disk_caches.disabled() and not body.allow_disk_cache:
        raise HTTPException(
            status_code=409,
            detail="Warm KV may be saved by the configured disk cache; set allow_disk_cache=true or disable disk caches",
        )

    if body.enable_thinking is None and getattr(cfg, "default_reasoning_effort", None):
        raise HTTPException(
            status_code=409,
            detail="Set enable_thinking explicitly when the server has a default reasoning effort",
        )

    try:
        messages, tools, thinking = _render_inputs(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if body.kind == "prefix":
        try:
            token_ids, _ = cast(Any, cfg.engine)._shared_system_tools_prompt(
                messages, tools, thinking, None
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Cannot render prefix") from exc
        enforce_context_length(cfg.engine, len(token_ids), max_tokens=1)
    else:
        counted = enforce_context_length_for_messages(
            cfg.engine,
            messages,
            tools=body.tools or None,
            max_tokens=1,
            enable_thinking=thinking,
        )
        if counted is None:
            raise HTTPException(status_code=400, detail="Cannot count draft tokens")

    previous = None
    draft_key = cast(str, body.draft_key)
    revision = cast(int, body.revision)
    async with _guard:
        now = time.monotonic()
        while (
            _draft_revisions
            and next(iter(_draft_revisions.values()))[1] < now - _REVISION_TTL_SECONDS
        ):
            _draft_revisions.popitem(last=False)
        if body.kind == "draft":
            prior = _draft_revisions.get(draft_key)
            if prior is not None and revision <= prior[0]:
                raise HTTPException(status_code=409, detail="Draft revision is stale")
        if _active is not None and not _active.done():
            if body.kind != "draft" or draft_key != _active_key:
                raise HTTPException(
                    status_code=429, detail="A prefill is already running"
                )
            if revision <= _active_revision:
                raise HTTPException(status_code=409, detail="Draft revision is stale")
            previous = _active
            previous.cancel()
            try:
                await asyncio.wait({previous})
            except asyncio.CancelledError:
                previous.cancel()
                while not previous.done():
                    try:
                        await asyncio.wait({previous})
                    except asyncio.CancelledError:
                        continue
                raise
            if not previous.cancelled():
                previous.result()
        task = asyncio.create_task(_warm(cfg.engine, body))
        _active = task
        _active_key = body.draft_key
        _active_revision = body.revision if body.revision is not None else -1
        if body.kind == "draft":
            _draft_revisions[draft_key] = (revision, now)
            _draft_revisions.move_to_end(draft_key)
            if len(_draft_revisions) > _MAX_REVISION_KEYS:
                _draft_revisions.popitem(last=False)

    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                break
            await asyncio.wait({task}, timeout=0.1)
        return await task
    except asyncio.CancelledError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise
    finally:
        async with _guard:
            if _active is task and task.done():
                _active = None
                _active_key = None
                _active_revision = -1
