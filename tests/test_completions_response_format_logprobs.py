# SPDX-License-Identifier: Apache-2.0
"""r10-G/H follow-up — codex round-1 MED #4 / #5.

The R10-H4 ``response_format=json_object`` fix on ``/v1/completions``
was incomplete: ``logprobs`` undid the fence-strip on the non-stream
path (the post-cleanup ``final_text`` was overwritten by the token
concatenation) and silently bypassed it on the streaming path (the
``_json_mode`` gate explicitly excluded ``want_logprobs``). The two
contracts are incompatible — ``text_offset`` is a per-token byte
cursor against ``text``, and a fence-stripped ``text`` no longer
aligns with the raw token concatenation.

Rather than ship a half-correct hybrid (silent corruption vs silent
bypass) we reject the combination with a 400 envelope on BOTH
synchronous and streaming paths. Either knob alone keeps working.

The four tests below pin all four matrix cells:

1. ``stream=false`` + ``response_format=json_object`` + ``logprobs=2`` → 400
2. ``stream=false`` + ``response_format=json_object`` (no logprobs)  → 200,
   wire ``text`` is fence-stripped (positive-path coverage so a future
   refactor that *also* skips the strip on logprobs=None is caught).
3. ``stream=true``  + ``response_format=json_object`` + ``logprobs=2`` → 400
4. ``stream=true``  + ``response_format=json_object`` (no logprobs)  → 200,
   the single buffered text delta is fence-stripped.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.completions import router as completions_router

# A fenced-JSON output that vlad r10-R1 / bo r10-R1 reproduced on the
# wire. The fence-strip helper (``extract_json_from_response``) must
# peel the markdown wrapper and the "Just the JSON.\n…" preamble down
# to the bare ``{"answer": 42}`` JSON object.
_FENCED_TEXT = 'Just the JSON.\nAnswer:\n```json\n{"answer": 42}\n```'
_CLEAN_TEXT = '{"answer": 42}'


class _FencedCompletionEngine:
    """Mock completion engine that emits the fenced-JSON shape r10-H4
    was wired to clean up. Implements both ``generate`` (sync) and
    ``stream_generate`` (SSE) so both code paths can be exercised
    without a real model load.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    async def generate(self, prompt, **kwargs):
        return GenerationOutput(
            text=_FENCED_TEXT,
            prompt_tokens=8,
            completion_tokens=12,
            finished=True,
            finish_reason="stop",
        )

    async def stream_generate(self, prompt, **kwargs):
        # Single chunk carrying the full fenced text. The streaming
        # json-mode buffer accumulates ``new_text`` across chunks and
        # runs the strip once at finalize, so one chunk is enough to
        # exercise the wire-output contract.
        yield GenerationOutput(
            text=_FENCED_TEXT,
            new_text=_FENCED_TEXT,
            prompt_tokens=8,
            completion_tokens=12,
            finished=True,
            finish_reason="stop",
        )


def _make_client() -> TestClient:
    cfg = reset_config()
    cfg.engine = _FencedCompletionEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    # ``install_exception_handlers`` is what unwraps
    # ``HTTPException(detail={"error": ...})`` into a flat top-level
    # ``error`` body. Without it FastAPI emits the legacy
    # ``{"detail": {...}}`` shape and the structured envelope assertions
    # below would mis-aim (the tests would still pass on a regression
    # that returned a bare-string detail with the wrong status code).
    install_exception_handlers(app)
    app.include_router(completions_router)
    return TestClient(app)


def _parse_sse_events(text: str) -> list[dict]:
    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# Synchronous /v1/completions
# ---------------------------------------------------------------------------


def test_sync_response_format_plus_logprobs_rejected_with_400():
    """Codex r1 MED #4: pre-fix the route returned 200 with raw fenced
    text in ``choices[0].text`` (the ``want_logprobs`` branch reset
    ``final_text`` to the token concat, undoing the strip at L404).
    Post-fix the combination is rejected with the standard
    OpenAI-shaped invalid_request envelope so the caller picks one
    knob per request."""
    client = _make_client()

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "response_format": {"type": "json_object"},
            "logprobs": 2,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "unsupported_combination"
    assert err["param"] == "response_format"
    # Message names BOTH knobs so a caller skimming the error string
    # immediately sees the resolution.
    assert "response_format" in err["message"]
    assert "logprobs" in err["message"]


def test_sync_response_format_without_logprobs_still_strips_fences():
    """Positive-path control: with ``logprobs`` omitted, the
    sync ``/v1/completions`` route returns 200 and the wire ``text``
    is the fence-stripped JSON (chat-lane parity per r10-H4).

    Catches a future refactor that over-corrects MED #4 by skipping the
    strip whenever ``response_format`` is set."""
    client = _make_client()

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "response_format": {"type": "json_object"},
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["text"] == _CLEAN_TEXT
    # Logprobs branch was not taken; payload must be absent.
    assert body["choices"][0].get("logprobs") is None


# ---------------------------------------------------------------------------
# Streaming /v1/completions
# ---------------------------------------------------------------------------


def test_stream_response_format_plus_logprobs_rejected_with_400():
    """Codex r1 MED #5: pre-fix the route returned 200 with raw fenced
    SSE deltas (the streaming ``_json_mode`` gate at L601 explicitly
    excluded ``want_logprobs``, silently bypassing the buffered fence
    strip). Post-fix the combination is rejected BEFORE the SSE
    StreamingResponse commits, so the caller sees a 400 envelope
    instead of an EventSource that emits unscrubbed text."""
    client = _make_client()

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "stream": True,
            "response_format": {"type": "json_object"},
            "logprobs": 2,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "unsupported_combination"
    assert err["param"] == "response_format"


def test_stream_response_format_without_logprobs_strips_fences_in_buffered_chunk():
    """Positive-path control for streaming: with ``logprobs`` omitted,
    the route returns 200 and the buffered single-chunk emit carries
    the fence-stripped JSON in ``choices[0].text``.

    Asserts on the consolidated chunk the json-mode branch emits at
    finalize (per the r10-H4 design note in ``completions.py`` —
    structured-output clients don't pipeline partial JSON, so legacy
    completions trade per-token granularity for a clean wire payload).
    """
    client = _make_client()

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "stream": True,
            "response_format": {"type": "json_object"},
        },
    )

    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    text_chunks = [
        e for e in events if e.get("choices") and e["choices"][0].get("text")
    ]
    assert len(text_chunks) >= 1, (
        f"expected at least one text-carrying chunk; got events={events!r}"
    )
    # The buffered emit is the LAST text-carrying chunk; its text
    # field must be the fence-stripped payload.
    final = text_chunks[-1]
    assert final["choices"][0]["text"] == _CLEAN_TEXT
    # Finish reason is on the same buffered chunk in json-mode.
    assert final["choices"][0]["finish_reason"] == "stop"
