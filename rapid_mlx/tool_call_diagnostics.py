# SPDX-License-Identifier: Apache-2.0
"""Request-local rejected-call observations, never executable tool calls."""

import json
import logging
from contextvars import ContextVar
from functools import wraps
from threading import Lock

logger = logging.getLogger(__name__)
_rejections: ContextVar[dict | None] = ContextVar("tool_rejections", default=None)
_counter_lock = Lock()
_rejected_total = 0


def record_rejected_tool_call(name: str, raw_arguments: str) -> None:
    """Count unique rejected (name, raw arguments) pairs once per HTTP request.

    Streaming parsers revisit accumulated text. Deduplicate these observations,
    including fallback reparses, without logging model-generated arguments.
    Outside a request scope the counter counts individual parser observations.
    """
    global _rejected_total
    records = _rejections.get()
    key = (name, raw_arguments)
    if records is not None:
        if key in records:
            return
        records[key] = {
            "name": name,
            "raw_arguments": raw_arguments,
            "reason": "undeclared_tool",
        }
    with _counter_lock:
        _rejected_total += 1
    logger.warning("tool-call rejected: reason=undeclared_tool name=%r", name)


def rejected_tool_calls_total() -> int:
    with _counter_lock:
        return _rejected_total


def current_rejected_tool_calls() -> list[dict[str, str]] | None:
    records = _rejections.get()
    return list(records.values()) if records else None


def _annotate_terminal_event(event, records):
    # Preserve non-JSON/custom frames and arbitrary legal StreamingResponse
    # payload types; only complete completion-chunk objects carry diagnostics.
    if not records or not isinstance(event, (str, bytes)):
        return event
    try:
        text = event.decode("utf-8") if isinstance(event, bytes) else event
        if not text.startswith("data: {"):
            return event
        payload = json.loads(text[6:])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return event
    if payload.get("object") != "chat.completion.chunk":
        return event
    choices = payload.get("choices")
    if not isinstance(choices, list) or not any(
        isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str)
        for choice in choices
    ):
        return event
    payload["rejected_tool_calls"] = list(records.values())
    text = "data: " + json.dumps(payload) + "\n\n"
    return text.encode("utf-8") if isinstance(event, bytes) else text


async def _observe_stream(iterator, records):
    """Keep parser context across awaits, but never across a yield to the caller."""
    try:
        while True:
            token = _rejections.set(records)
            try:
                event = await anext(iterator)
            except StopAsyncIteration:
                break
            finally:
                _rejections.reset(token)
            yield _annotate_terminal_event(event, records)
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


def observe_tool_rejections(handler):
    """Attach optional diagnostics to buffered or streamed chat responses."""

    @wraps(handler)
    async def observed(*args, **kwargs):
        from starlette.responses import StreamingResponse

        records = {}
        token = _rejections.set(records)
        try:
            response = await handler(*args, **kwargs)
        finally:
            _rejections.reset(token)
        if isinstance(response, StreamingResponse):
            response.body_iterator = _observe_stream(response.body_iterator, records)
        return response

    return observed
