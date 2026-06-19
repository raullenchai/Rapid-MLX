# SPDX-License-Identifier: Apache-2.0
"""VLM chat-route image handling — regression for issue #682.

Pins two contracts at the ``/v1/chat/completions`` route layer:

1. **Image content is forwarded to the MLLM engine.** A multipart user
   message with ``{"type": "image_url", ...}`` parts must reach
   ``engine.chat`` / ``engine.stream_chat`` so the engine can extract
   the image bytes itself (chat.py:577-598 builds raw ``messages``
   without flattening multimodal content for the MLLM branch).

2. **Engine-side prompt-cap errors surface as HTTP 400.** A 1920×1080
   screenshot decoded by Qwen3-VL produces ~2200+ vision tokens — past
   the default ``prefill_step_size=2048`` cap that
   ``mllm_batch_generator._process_prompts`` enforces. Pre-fix the
   scheduler swallowed the ValueError as ``finish_reason="length"`` +
   empty content; the route returned HTTP 200 + empty assistant
   message; Desktop rendered the misleading "Reached max_tokens
   before any output" error (``ChatViewModel.swift:491``).

Post-fix:
- ``mllm_scheduler._step_no_queue`` classifies "exceeds the per-batch
  cap" as a client error → ``RequestOutput.error`` is set.
- ``stream_outputs`` raises ``ValueError`` instead of yielding the
  fake-success output.
- ``routes/chat.py`` maps the marker substring to HTTP 400 so the
  client sees an actionable message instead of a generic 500.

Both tests use a stub MLLM engine; no real model is loaded, so they
run in unit-test CI without an Apple Silicon GPU.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _StubMLLMEngine:
    """Mock MLLM engine. Records the messages it received and either
    returns a canned ``GenerationOutput`` or raises a ValueError that
    mimics the engine-layer error path.
    """

    preserve_native_tool_format = False
    is_mllm = True
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, *, raise_msg: str | None = None):
        self._raise_msg = raise_msg
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        if self._raise_msg is not None:
            raise ValueError(self._raise_msg)
        return GenerationOutput(
            text="A blue background.",
            new_text="A blue background.",
            prompt_tokens=2300,
            completion_tokens=4,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        if self._raise_msg is not None:
            raise ValueError(self._raise_msg)
        text = "A blue background."
        yield GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=2300,
            completion_tokens=4,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_client(engine: _StubMLLMEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "qwen3-vl-8b-4bit"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    cfg.reasoning_parser_name = None

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


_IMAGE_DATA_URL = (
    # 1×1 transparent PNG — the route doesn't decode the bytes, the
    # stub engine doesn't either; this just exercises the message-shape
    # plumbing through the route → engine boundary.
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII="
)


def _multipart_user_message(text: str) -> dict:
    """OpenAI-style multipart user message: one text + one image_url part."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URL}},
        ],
    }


def test_chat_route_forwards_image_url_content_to_mllm_engine():
    """Multimodal user content reaches the MLLM engine intact.

    The chat route's MLLM branch (chat.py:577-598) must NOT flatten
    image_url parts out of ``messages`` before calling ``engine.chat``
    — the engine extracts images itself via
    ``extract_multimodal_content`` inside ``stream_chat`` /
    ``chat``. A refactor that ran ``extract_multimodal_content``
    twice (once at the route, once in the engine) would corrupt the
    chat template — Qwen3-VL's Jinja template expects content-list
    parts with ``{"type":"image"}`` to emit ``<|vision_start|>…<|vision_end|>``
    tokens, but the route handler would have already stripped them.

    Pin: the engine received a message whose ``content`` is a LIST
    (not a flat string), with at least one ``type=image_url`` part.
    """
    engine = _StubMLLMEngine()
    client = _make_client(engine)

    payload = {
        "model": "qwen3-vl-8b-4bit",
        "messages": [_multipart_user_message("What is this?")],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text

    # Engine got called exactly once on the non-streaming path.
    assert len(engine.chat_calls) == 1
    fwd_msgs = engine.chat_calls[0]["messages"]
    assert len(fwd_msgs) == 1

    # The route normalises pydantic Message → dict via model_dump, so
    # the content is a list of dicts (not Pydantic ContentPart models).
    msg = fwd_msgs[0]
    assert isinstance(msg, dict)
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list), (
        f"MLLM branch must forward list content, got {type(msg['content']).__name__}"
    )

    parts = msg["content"]
    types = [p.get("type") for p in parts if isinstance(p, dict)]
    assert "text" in types, f"text part missing: {types}"
    assert "image_url" in types, (
        f"image_url part dropped by route before reaching engine: {types}"
    )

    # Image URL payload survives end-to-end.
    image_part = next(p for p in parts if p.get("type") == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_chat_route_maps_per_batch_cap_error_to_http_400():
    """Engine raises "exceeds the per-batch cap" → route returns 400.

    Pre-fix this surfaced as HTTP 200 + empty assistant message +
    ``finish_reason=length`` (#682). Desktop's ChatViewModel rendered
    the misleading "Reached max_tokens before any output" error
    because finish_reason==length with zero tokens.

    Post-fix: the chat route recognises the per-batch-cap marker and
    converts to HTTP 400 with the actionable engine message
    ("downscale the image / raise --prefill-step-size").
    """
    engine = _StubMLLMEngine(
        raise_msg=(
            "Total prompt tokens (2292) exceeds the per-batch cap "
            "(2048 for 1 request(s)). For image inputs, downscale the "
            "image; for text inputs, shorten the prompt or restart "
            "the server with --prefill-step-size set higher."
        )
    )
    client = _make_client(engine)

    payload = {
        "model": "qwen3-vl-8b-4bit",
        "messages": [_multipart_user_message("这是谁")],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400, (
        f"per-batch cap error must surface as 400, got "
        f"{resp.status_code}: {resp.text[:400]}"
    )

    body = resp.json()
    detail = body.get("detail") or body.get("error", {}).get("message", "")
    assert "exceeds the per-batch cap" in detail, (
        f"detail must carry the actionable engine message; got {detail!r}"
    )
    assert "downscale the image" in detail
    assert "--prefill-step-size" in detail


def test_chat_route_maps_image_fetch_error_to_http_400_still_works():
    """The earlier #457 fix path must not regress alongside #682.

    Both ``Failed to process image`` and ``exceeds the per-batch cap``
    are now routed through the same client-error branch; this guards
    against a future refactor accidentally narrowing the mapping back
    to one of them.
    """
    engine = _StubMLLMEngine(
        raise_msg="Failed to process image: 404 Client Error",
    )
    client = _make_client(engine)

    payload = {
        "model": "qwen3-vl-8b-4bit",
        "messages": [_multipart_user_message("describe")],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
