# SPDX-License-Identifier: Apache-2.0
"""Regression for M-16: text-only models silently accept image blocks.

Bug (Quincy 0.8.1 dogfood)
--------------------------
``POST /v1/messages`` to a text-only model with an Anthropic ``image``
content block returned HTTP 200 with the image silently dropped. The
Anthropic adapter
(``vllm_mlx/api/anthropic_adapter._convert_message_to_openai``) only
forwards ``text``/``tool_use``/``tool_result`` blocks to the OpenAI-shape
request — every other block type (``image``, ``document``) is silently
discarded. The caller sees a confident-but-empty answer about the
missing media.

Fix
---
At the route boundary in ``vllm_mlx/routes/anthropic.py``, before the
adapter runs, inspect every block in every message. If ``engine.is_mllm``
is False and we see ``image`` or ``document``, raise HTTP 400 naming the
model/incompatibility. Mirrors the same R9P1 guard the OpenAI
``/v1/chat/completions`` route already enforces (``routes/chat.py``).

Scope
-----
* ``/v1/messages`` (Anthropic): the new guard.
* ``/v1/chat/completions`` (OpenAI): pre-existing guard already covers
  ``image_url``/``image``; this file adds parity-style assertions so a
  future refactor can't silently regress either surface.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.anthropic import router as anthropic_router
from vllm_mlx.routes.chat import router as chat_router


class _TextOnlyEngine:
    """Minimal engine stub matching the text-only contract."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


class _VisionEngine(_TextOnlyEngine):
    """Same stub but advertised as MLLM. Should accept image blocks
    (passed through to the downstream multimodal pipeline)."""

    is_mllm = True


def _make_client(engine, *, include_anthropic: bool = True) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "text-only-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(chat_router)
    if include_anthropic:
        app.include_router(anthropic_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


# ----------------------------------------------------------------------
# Anthropic surface: /v1/messages
# ----------------------------------------------------------------------


def test_anthropic_text_only_model_rejects_image_block():
    """The bug repro: text-only model + image block => HTTP 400, not 200."""
    client = _make_client(_TextOnlyEngine())
    resp = client.post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                        {"type": "text", "text": "What is in this image?"},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "text-only-model" in detail
    assert "image" in detail or "document" in detail


def test_anthropic_text_only_model_rejects_document_block():
    """Anthropic ``document`` blocks (PDFs) are also dropped silently
    by the adapter; the guard must reject them too."""
    client = _make_client(_TextOnlyEngine())
    resp = client.post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQK",
                            },
                        },
                        {"type": "text", "text": "Summarise this pdf"},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "document" in detail


def test_anthropic_vision_model_accepts_image_block():
    """Sanity check: VLM-capable engine must NOT see a 400 — the route
    must propagate the image block downstream."""
    client = _make_client(_VisionEngine())
    resp = client.post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
        },
    )
    # We do not assert 200 strictly — downstream multimodal pipeline
    # may itself reject the stub payload — but the route-level
    # capability gate must NOT fire.
    if resp.status_code == 400:
        assert "does not support" not in resp.json().get("detail", ""), (
            "VLM-capable engine wrongly triggered the text-only media gate"
        )


def test_anthropic_text_only_model_accepts_text_only_content():
    """Sanity check: text-only model + text-only content => 200."""
    client = _make_client(_TextOnlyEngine())
    resp = client.post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text


def test_anthropic_text_only_model_accepts_string_content():
    """The simple-string shape (most common) keeps working unchanged."""
    client = _make_client(_TextOnlyEngine())
    resp = client.post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text


# ----------------------------------------------------------------------
# OpenAI surface: /v1/chat/completions (mirror)
# ----------------------------------------------------------------------


def test_openai_text_only_model_rejects_image_url_block():
    """The pre-existing OpenAI-side guard must keep rejecting image_url
    on text-only models. Pinned here so a refactor that drops it lights
    up alongside the new Anthropic test."""
    client = _make_client(_TextOnlyEngine(), include_anthropic=False)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        },
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "does not support" in resp.json()["detail"]


def test_openai_text_only_model_accepts_text_only_content():
    client = _make_client(_TextOnlyEngine(), include_anthropic=False)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "text-only-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
