# SPDX-License-Identifier: Apache-2.0
"""R6-H5: cross-route context-window enforcement (40K-token prompt at
``context_window=40960`` must return HTTP 400 ``context_length_exceeded``).

The 0.8.7 dogfood (Hiro R1) flagged that the four chat surfaces —
``/v1/chat/completions``, ``/v1/completions``, ``/v1/messages``, and
``/v1/responses`` — must each call the ``enforce_context_length*``
helpers BEFORE handing the request to the engine. Pre-fix the existing
``test_context_length_exceeded.py`` exercised only the helper in
isolation. These tests build a fake engine whose model exposes
``max_position_embeddings = 40960`` (matching qwen3-0.6b-8bit) and
verify the structured 400 lands on every route, so a route-level
refactor that silently drops the helper call cannot regress past
CI.

Each test uses a stub engine — no real model load, no Apple Silicon
GPU needed — and a deterministic 4-chars-per-token tokenizer so the
test can target the exact ``prompt + max_tokens > context_window``
boundary without flaky BPE drift.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config

_CONTEXT_WINDOW = 40960


class _StubArgs:
    max_position_embeddings = _CONTEXT_WINDOW


class _StubModel:
    args = _StubArgs()


class _StubTokenizer:
    """Deterministic tokenizer: 1 token per 4 chars. Used by the
    helper's ``count_prompt_tokens`` so the test can hit the exact
    boundary without depending on a real BPE."""

    model_max_length = _CONTEXT_WINDOW
    bos_token = None

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return [0] * max(1, len(text) // 4)

    # FastAPI/responses routes pass tools through ``convert_tools_for_template``;
    # the chat-template render of the user message is what we count.
    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
        **_kwargs,
    ):
        """Trivial template: join the user contents with newlines so
        the tokenizer's char/4 heuristic gets us the prompt-token
        count we want."""
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
            else:
                parts.append(content)
        return "\n".join(parts)


class _StubEngine:
    """Stub for the text-only AR engine. Exposes the minimal surface
    each route's pre-engine validation reads. ``build_prompt`` calls
    the tokenizer's chat-template path so the helper-side token count
    matches the route's own resolved prompt.
    """

    is_mllm = False
    preserve_native_tool_format = False
    supports_guided_generation = False

    def __init__(self):
        self._model = _StubModel()
        self.tokenizer = _StubTokenizer()
        self._tokenizer = self.tokenizer

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )

    async def chat(self, **kwargs):  # noqa: ARG002
        # Should never be reached on the 400 path; raising makes the
        # failure mode loud if the gate is bypassed.
        raise AssertionError("engine.chat must not be reached on the 400 path")

    async def stream_chat(self, *args, **kwargs):  # noqa: ARG002
        raise AssertionError("engine.stream_chat must not be reached on the 400 path")


def _make_app(routes: list[Any]) -> TestClient:
    cfg = reset_config()
    cfg.engine = _StubEngine()
    cfg.model_name = "qwen3-0.6b-8bit"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    cfg.reasoning_parser_name = None
    cfg.default_max_tokens = 1024
    cfg.thinking_token_budget = 0

    app = FastAPI()
    for router in routes:
        app.include_router(router)
    return TestClient(app)


def _huge_text(approx_tokens: int) -> str:
    """Build a string the stub tokenizer maps to ``approx_tokens``
    tokens (4 chars / token)."""
    return "x" * (approx_tokens * 4)


def _extract_error(body: dict) -> dict:
    """Pull the OpenAI-style error envelope out of a FastAPI response.

    The structured 400 handler in ``vllm_mlx/server.py`` unwraps
    ``HTTPException(detail={"error": {...}})`` into a top-level
    ``{"error": {...}}`` body. When the route is mounted on a bare
    FastAPI app (these tests do that to avoid the full server
    bootstrap), FastAPI's default handler wraps the same dict under
    ``"detail"`` instead. Accept both shapes so the assertion targets
    the canonical fields.
    """
    if isinstance(body.get("error"), dict):
        return body["error"]
    if isinstance(body.get("detail"), dict):
        inner = body["detail"]
        if isinstance(inner.get("error"), dict):
            return inner["error"]
        return inner
    return body


# ─── /v1/chat/completions ───────────────────────────────────────────


def test_chat_completions_rejects_over_context_window():
    """``/v1/chat/completions`` must surface the structured 400
    envelope when ``prompt + max_tokens > context_window``."""
    from vllm_mlx.routes.chat import router as chat_router

    client = _make_app([chat_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": _huge_text(41_000)}],
        "max_tokens": 16,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"
    assert err.get("type") == "invalid_request_error"
    assert str(_CONTEXT_WINDOW) in err.get("message", "")


# ─── /v1/completions ────────────────────────────────────────────────


def test_completions_rejects_over_context_window():
    """``/v1/completions`` (raw-prompt API) must enforce the same
    cap. The helper here is ``enforce_context_length_for_prompt``
    — no chat template applied."""
    from vllm_mlx.routes.completions import router as completions_router

    client = _make_app([completions_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "prompt": _huge_text(41_000),
        "max_tokens": 16,
    }
    resp = client.post("/v1/completions", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"


# ─── /v1/messages (Anthropic) ───────────────────────────────────────


def test_anthropic_messages_rejects_over_context_window():
    """``/v1/messages`` (Anthropic shape) must enforce the same cap.
    Anthropic SDKs branch on ``error.type`` so we pin the envelope
    matches the chat lane."""
    from vllm_mlx.routes.anthropic import router as anthropic_router

    client = _make_app([anthropic_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": _huge_text(41_000)}],
        "max_tokens": 16,
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"


# ─── /v1/responses ──────────────────────────────────────────────────


def test_responses_rejects_over_context_window():
    """``/v1/responses`` (OpenAI Responses API) must enforce the
    same cap. The route re-extracts multimodal content before the
    gate, so this also pins that the gate fires on the re-extracted
    text-only shape."""
    from vllm_mlx.routes.responses import router as responses_router

    client = _make_app([responses_router])

    payload = {
        "model": "qwen3-0.6b-8bit",
        "input": _huge_text(41_000),
        "max_output_tokens": 16,
    }
    resp = client.post("/v1/responses", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = _extract_error(body)
    assert err.get("code") == "context_length_exceeded"


# ─── Under-cap requests still pass through the gate ────────────────


def test_chat_completions_passes_when_within_context_window():
    """Sanity check: a prompt that fits inside ``prompt + max_tokens
    <= context_window`` must NOT be rejected. Locks in that the
    enforcement is bounded — over-strict gates would block legitimate
    long-context requests."""
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.routes.chat import router as chat_router

    # Build the app, then swap in a chat impl that returns a real
    # response (the default stub raises on .chat to catch silent
    # bypass on the 400 path).
    cfg = reset_config()
    engine = _StubEngine()

    async def _chat(**kwargs):  # noqa: ARG001
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=10,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    engine.chat = _chat  # type: ignore[assignment]

    cfg.engine = engine
    cfg.model_name = "qwen3-0.6b-8bit"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    cfg.reasoning_parser_name = None
    cfg.default_max_tokens = 1024
    cfg.thinking_token_budget = 0

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    # 30K tokens + max_tokens 16 = 30016 < 40960. Must pass.
    payload = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": _huge_text(30_000)}],
        "max_tokens": 16,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
