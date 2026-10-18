# SPDX-License-Identifier: Apache-2.0
"""Explicit, output-free prefix warming for local clients."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from .. import disk_caches
from ..config import get_config
from ..middleware.auth import verify_api_key_or_x_api_key
from ..runtime.cache import save_prefix_cache_to_disk

router = APIRouter(dependencies=[Depends(verify_api_key_or_x_api_key)])

_guard = asyncio.Lock()
_active: asyncio.Task | None = None
_active_key: str | None = None
_active_revision = -1


class PrefillRequest(BaseModel):
    """One complete system prefix or one replaceable conversation draft."""

    kind: Literal["prefix", "draft"]
    messages: list[dict] = Field(min_length=1, max_length=64)
    tools: list[dict] = Field(default_factory=list, max_length=64)
    draft_key: str | None = Field(default=None, min_length=1, max_length=64)
    revision: int | None = Field(default=None, ge=0)
    persist: bool = False
    enable_thinking: bool | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if (
            sum(len(str(message)) for message in self.messages)
            + sum(len(str(tool)) for tool in self.tools)
            > 500_000
        ):
            raise ValueError("prefill input is too large")
        if not all(
            isinstance(message.get("content"), str)
            and message.get("role")
            in {"system", "developer", "user", "assistant", "tool"}
            for message in self.messages
        ):
            raise ValueError("prefill accepts text messages only")
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


async def _warm(engine, body: PrefillRequest) -> dict:
    stream = engine.stream_chat(
        messages=body.messages,
        tools=body.tools or None,
        max_tokens=1,
        temperature=0,
        enable_thinking=body.enable_thinking
        if body.enable_thinking is not None
        else (False if body.tools else None),
    )
    last = None
    try:
        async for output in stream:
            last = output
            if output.finished:
                break
    finally:
        await stream.aclose()
    result = {
        "status": "prefilled",
        "prompt_tokens": getattr(last, "prompt_tokens", 0),
        "cached_tokens": getattr(last, "cached_tokens", 0),
        "persisted": False,
    }
    if body.persist and not disk_caches.disabled():
        result["persisted"] = await asyncio.to_thread(save_prefix_cache_to_disk, 0)
    return result


@router.post("/v1/cache/prefill")
async def prefill(body: PrefillRequest, request: Request):
    """Warm KV state without sending generated text to the caller.

    A new draft revision cancels its predecessor. One warm operation per
    process bounds speculative memory use; ordinary chat requests continue.
    """
    global _active, _active_key, _active_revision
    cfg = get_config()
    if not cfg.ready or cfg.draining or cfg.engine is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    previous = None
    async with _guard:
        if _active is not None and not _active.done():
            if body.kind != "draft" or body.draft_key != _active_key:
                raise HTTPException(
                    status_code=429, detail="A prefill is already running"
                )
            if body.revision <= _active_revision:
                raise HTTPException(status_code=409, detail="Draft revision is stale")
            previous = _active
            previous.cancel()
            try:
                await previous
            except asyncio.CancelledError:
                pass
        task = asyncio.create_task(_warm(cfg.engine, body))
        _active = task
        _active_key = body.draft_key
        _active_revision = body.revision if body.revision is not None else -1

    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                break
            await asyncio.wait({task}, timeout=0.1)
        return await task
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        async with _guard:
            if _active is task:
                _active = None
                _active_key = None
                _active_revision = -1
