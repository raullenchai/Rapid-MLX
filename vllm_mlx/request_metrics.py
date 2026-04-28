# SPDX-License-Identifier: Apache-2.0
"""In-memory recorder for completed inference requests.

Used by the metrics ASGI middleware to capture per-request stats (tokens,
finish reason, message preview, throughput) and expose them via /v1/requests
for the TUI monitor.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any

_DEFAULT_HISTORY = 100
_MAX_PREVIEW_CHARS = 240


class RequestRecorder:
    """Thread-safe ring buffer of completed request stats with a single
    "active" snapshot for the request currently being processed."""

    def __init__(self, history_limit: int = _DEFAULT_HISTORY) -> None:
        self._lock = threading.Lock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._active: dict[str, dict[str, Any]] = {}

    def start(self, surface: str) -> str:
        req_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._active[req_id] = {
                "request_id": req_id,
                "surface": surface,
                "started_at": now,
                "updated_at": now,
                "first_token_at": None,
                "last_token_at": None,
                "text_chunks": 0,
                "phase": "prefill",
                "generated_tokens": 0,
                "prompt_tokens": 0,
                "message_preview": "",
                "message_preview_source": "output",
            }
        return req_id

    def mark_first_token(self, req_id: str) -> None:
        now = time.time()
        with self._lock:
            entry = self._active.get(req_id)
            if entry is None:
                return
            if entry["first_token_at"] is None:
                entry["first_token_at"] = now
            entry["phase"] = "generation"
            entry["updated_at"] = now

    def update(
        self,
        req_id: str,
        *,
        delta_text: str | None = None,
        generated_tokens: int | None = None,
        prompt_tokens: int | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            entry = self._active.get(req_id)
            if entry is None:
                return
            entry["updated_at"] = now
            if delta_text:
                preview = (entry.get("message_preview") or "") + delta_text
                if len(preview) > _MAX_PREVIEW_CHARS:
                    preview = preview[-_MAX_PREVIEW_CHARS:]
                entry["message_preview"] = preview
                entry["last_token_at"] = now
                entry["text_chunks"] = int(entry.get("text_chunks", 0)) + 1
            if generated_tokens is not None:
                entry["generated_tokens"] = max(
                    entry.get("generated_tokens", 0), int(generated_tokens)
                )
            if prompt_tokens is not None:
                entry["prompt_tokens"] = max(
                    entry.get("prompt_tokens", 0), int(prompt_tokens)
                )

    def finish(
        self,
        req_id: str,
        *,
        finish_reason: str | None = None,
        prompt_tokens: int | None = None,
        generated_tokens: int | None = None,
        error: str | None = None,
        non_streaming: bool = False,
        engine_gen_tps: float | None = None,
        engine_ttft: float | None = None,
        acceptance_ratio: float | None = None,
        block_size: int | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            entry = self._active.pop(req_id, None)
            if entry is None:
                return

            started_at = float(entry.get("started_at") or now)
            first_at = entry.get("first_token_at")
            last_at = entry.get("last_token_at")
            text_chunks = int(entry.get("text_chunks", 0))
            elapsed = max(0.0, now - started_at)
            ttft = (first_at - started_at) if first_at else None

            ptoks = (
                int(prompt_tokens)
                if prompt_tokens is not None
                else int(entry.get("prompt_tokens") or 0)
            )
            gtoks = (
                int(generated_tokens)
                if generated_tokens is not None
                else int(entry.get("generated_tokens") or 0)
            )

            # Generation window: use (elapsed - ttft) when ttft is known
            # and meaningfully smaller than elapsed (real streaming with
            # separable prefill/decode). Fall back to total elapsed for
            # non-streaming or coalesced single-chunk streams.
            if (
                not non_streaming
                and ttft is not None
                and text_chunks > 1
                and elapsed > ttft + 0.01
            ):
                generation_window = elapsed - ttft
                prompt_tps = (ptoks / ttft) if ttft > 0.01 else 0.0
            else:
                generation_window = elapsed
                ttft = None
                prompt_tps = 0.0
            generation_tps = (
                (gtoks / generation_window) if generation_window > 0.01 else 0.0
            )

            # Engine-reported values (from scheduler) are more accurate
            # than middleware timing — prefer them when available.
            if engine_gen_tps is not None and engine_gen_tps > 0:
                generation_tps = float(engine_gen_tps)
            if engine_ttft is not None and engine_ttft > 0:
                ttft = float(engine_ttft)
                if ptoks > 0 and ttft > 0.01:
                    prompt_tps = ptoks / ttft

            record_acceptance = (
                float(acceptance_ratio) if acceptance_ratio is not None else None
            )
            record_block_size = (
                int(block_size) if block_size is not None else None
            )

            record = {
                "request_id": entry["request_id"],
                "surface": entry.get("surface", ""),
                "started_at": started_at,
                "finished_at": now,
                "elapsed": elapsed,
                "ttft": ttft,
                "prompt_tokens": ptoks,
                "generated_tokens": gtoks,
                "generation_tps": generation_tps,
                "prompt_tps": prompt_tps,
                "finish_reason": finish_reason or ("error" if error else "stop"),
                "message_preview": entry.get("message_preview") or "",
                "message_preview_source": entry.get("message_preview_source", "output"),
                "error": error,
                "acceptance_ratio": record_acceptance,
                "block_size": record_block_size,
            }
            self._entries.append(record)

    def entries(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            data = list(self._entries)
        if limit <= 0:
            return data
        return data[-limit:]

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._active:
                return None
            req_id = next(iter(self._active))
            return dict(self._active[req_id])

    def last(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._entries:
                return None
            return dict(self._entries[-1])


_recorder: RequestRecorder | None = None


def get_recorder() -> RequestRecorder:
    global _recorder
    if _recorder is None:
        _recorder = RequestRecorder()
    return _recorder
