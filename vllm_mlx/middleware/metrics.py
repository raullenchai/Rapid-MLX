# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware that records per-request inference metrics.

Wraps responses for /v1/chat/completions and /v1/completions, parses streamed
SSE chunks (and final non-streaming JSON) to extract usage, finish_reason,
and incremental output text, then publishes the data to the
RequestRecorder for the /v1/requests endpoint and the TUI monitor.
"""

from __future__ import annotations

import asyncio
import json
import logging

from ..request_metrics import get_recorder

logger = logging.getLogger(__name__)

_TRACKED_PATHS = ("/v1/chat/completions", "/v1/completions")
_MAX_BUFFER_BYTES = 4 * 1024 * 1024  # safety cap for non-stream JSON capture


def _safe_json_loads(payload: str | bytes) -> dict | None:
    try:
        return json.loads(payload)
    except Exception:
        return None


def _extract_chat_delta(payload: dict) -> tuple[str | None, str | None]:
    """Returns (delta_text, finish_reason) for chat completion chunks/final."""
    try:
        choice = (payload.get("choices") or [None])[0]
        if not choice:
            return None, None
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text is None:
            message = choice.get("message") or {}
            text = message.get("content")
        return text, choice.get("finish_reason")
    except Exception:
        return None, None


def _extract_completion_delta(payload: dict) -> tuple[str | None, str | None]:
    """Returns (delta_text, finish_reason) for legacy completion chunks/final."""
    try:
        choice = (payload.get("choices") or [None])[0]
        if not choice:
            return None, None
        return choice.get("text"), choice.get("finish_reason")
    except Exception:
        return None, None


def _extract_usage(payload: dict) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return None, None
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )


class MetricsMiddleware:
    """Pure ASGI middleware (does not buffer the entire stream)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path not in _TRACKED_PATHS:
            await self.app(scope, receive, send)
            return

        recorder = get_recorder()
        req_id = recorder.start(surface=path)

        is_chat = path == "/v1/chat/completions"
        sse_carry = b""
        json_buffer = bytearray()
        is_sse = False
        first_token_seen = False
        last_finish_reason: str | None = None
        last_prompt_tokens: int | None = None
        last_generated_tokens: int | None = None
        running_text_tokens = 0  # rough fallback when usage is absent
        engine_gen_tps: float = 0.0
        engine_ttft: float | None = None
        engine_acceptance: float | None = None
        engine_block_size: int | None = None

        def poll_engine_stats() -> None:
            """Snapshot engine's per-request tps/ttft/acceptance during the stream."""
            nonlocal engine_gen_tps, engine_ttft
            nonlocal engine_acceptance, engine_block_size
            try:
                from ..config import get_config

                cfg = get_config()
                if cfg.engine is None:
                    return
                stats = cfg.engine.get_stats()
                for r in stats.get("requests") or []:
                    tps = r.get("tokens_per_second")
                    if tps is not None:
                        v = float(tps)
                        if v > engine_gen_tps:
                            engine_gen_tps = v
                    ttft_s = r.get("ttft_s")
                    if ttft_s is not None and engine_ttft is None:
                        try:
                            engine_ttft = float(ttft_s)
                        except Exception:
                            pass
                    accept = r.get("acceptance_ratio")
                    if accept is not None:
                        try:
                            engine_acceptance = float(accept)
                        except Exception:
                            pass
                    block = r.get("block_size")
                    if block is not None:
                        try:
                            engine_block_size = int(block)
                        except Exception:
                            pass
            except Exception:
                pass

        def handle_payload(payload: dict) -> None:
            nonlocal first_token_seen, last_finish_reason
            nonlocal last_prompt_tokens, last_generated_tokens, running_text_tokens
            if is_chat:
                text, finish = _extract_chat_delta(payload)
            else:
                text, finish = _extract_completion_delta(payload)
            if finish:
                last_finish_reason = finish
            ptoks, gtoks = _extract_usage(payload)
            if ptoks is not None:
                last_prompt_tokens = ptoks
            if gtoks is not None:
                last_generated_tokens = gtoks
            if text:
                if not first_token_seen:
                    recorder.mark_first_token(req_id)
                    first_token_seen = True
                # crude token estimate when usage events haven't arrived yet
                running_text_tokens += max(1, len(text) // 4)
                fallback_gtoks = (
                    last_generated_tokens
                    if last_generated_tokens is not None
                    else running_text_tokens
                )
                recorder.update(
                    req_id,
                    delta_text=text,
                    generated_tokens=fallback_gtoks,
                    prompt_tokens=last_prompt_tokens,
                )
            elif ptoks is not None or gtoks is not None:
                recorder.update(
                    req_id,
                    generated_tokens=last_generated_tokens,
                    prompt_tokens=last_prompt_tokens,
                )

        def consume_sse(buf: bytes) -> bytes:
            nonlocal sse_carry
            data = sse_carry + buf
            *complete, sse_carry = data.split(b"\n\n")
            for raw_event in complete:
                for line in raw_event.split(b"\n"):
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == b"[DONE]":
                        continue
                    payload = _safe_json_loads(body)
                    if isinstance(payload, dict):
                        handle_payload(payload)
            return sse_carry

        async def send_wrapper(message):
            nonlocal is_sse, sse_carry
            try:
                if message["type"] == "http.response.start":
                    headers = message.get("headers") or []
                    for name, value in headers:
                        if name.decode("latin-1").lower() == "content-type":
                            ctype = value.decode("latin-1").lower()
                            is_sse = "text/event-stream" in ctype
                            break
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"") or b""
                    more = bool(message.get("more_body", False))
                    if is_sse:
                        consume_sse(body)
                        poll_engine_stats()
                    else:
                        if len(json_buffer) < _MAX_BUFFER_BYTES:
                            json_buffer.extend(
                                body[: _MAX_BUFFER_BYTES - len(json_buffer)]
                            )
                        poll_engine_stats()
                    if not more:
                        if not is_sse and json_buffer:
                            payload = _safe_json_loads(bytes(json_buffer))
                            if isinstance(payload, dict):
                                handle_payload(payload)
                        recorder.finish(
                            req_id,
                            finish_reason=last_finish_reason,
                            prompt_tokens=last_prompt_tokens,
                            generated_tokens=last_generated_tokens,
                            non_streaming=not is_sse,
                            engine_gen_tps=engine_gen_tps if engine_gen_tps > 0 else None,
                            engine_ttft=engine_ttft,
                            acceptance_ratio=engine_acceptance,
                            block_size=engine_block_size,
                        )
            except Exception as exc:  # never let metrics break the response
                logger.debug("metrics middleware error: %s", exc)
            await send(message)

        # Periodically poll engine stats so non-streaming requests still
        # capture the engine-reported tokens_per_second / ttft while the
        # route is generating (no SSE events arrive in that case).
        poller_done = asyncio.Event()

        async def poll_loop() -> None:
            while not poller_done.is_set():
                poll_engine_stats()
                try:
                    await asyncio.wait_for(poller_done.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

        poller_task = asyncio.create_task(poll_loop())

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            recorder.finish(req_id, finish_reason="error", error=str(exc))
            raise
        finally:
            poller_done.set()
            try:
                await poller_task
            except Exception:
                pass
