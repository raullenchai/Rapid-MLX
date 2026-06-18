# SPDX-License-Identifier: Apache-2.0
"""Regression test for logprobs-path channel routing on harmony / gemma4.

Bug: when a client requested ``logprobs=true`` + ``top_logprobs>0`` on a
non-streaming chat completion, the route rerouted to ``engine.stream_chat``
to collect per-token logprob data and then kept ONLY the last yielded chunk
as ``output``. On channel-routed models (harmony/gpt-oss, gemma4) the
streaming iterator emits one chunk per token with a per-chunk ``channel``
field, but ``output.reasoning_text`` was never populated — so
``_finalize_content_and_reasoning`` saw an empty engine reasoning and fell
back to the text-regex parser, which leaked analysis-channel content into
``message.content`` and dropped ``reasoning_content`` entirely.

Surfaced by the iter7 onboarding sweep on gpt-oss-20b-mxfp4-q8: identical request
with vs without ``logprobs:true`` produced different channel routing — same
shape as #442 (PR #443) but on the logprobs codepath instead of truncated
output. Fix: accumulate ``new_text`` by ``channel`` while iterating the
stream and rebuild ``output.text`` / ``output.reasoning_text`` before the
finalize helper runs.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _ChannelRoutedEngine:
    """Mock engine whose stream_chat yields per-token channel-routed chunks.

    Mirrors the shape ``_stream_with_output_router`` produces on harmony:
    each chunk carries ``channel`` ∈ {"reasoning", "content"} and the new
    text for that single token.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, chunks: list[tuple[str, str]]):
        # chunks: list of (channel, new_text)
        self._chunks = chunks
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        for i, (channel, text) in enumerate(self._chunks):
            finished = i == len(self._chunks) - 1
            yield GenerationOutput(
                text=text,
                new_text=text,
                tokens=[1000 + i],
                prompt_tokens=10,
                completion_tokens=i + 1,
                finished=finished,
                finish_reason="stop" if finished else None,
                channel=channel,
            )


def _make_client(engine: _ChannelRoutedEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None  # let engine reasoning_text be authoritative
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


def test_logprobs_path_preserves_channel_split():
    """harmony-shaped stream → response must keep reasoning out of content."""
    chunks = [
        ("reasoning", "The user wants 2+2. "),
        ("reasoning", "That equals 4."),
        ("content", "4"),
    ]
    engine = _ChannelRoutedEngine(chunks)
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "2+2?"}],
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    assert msg["content"] == "4", (
        f"content must contain ONLY the content-channel text; got {msg['content']!r}"
    )
    assert msg.get("reasoning_content") == "The user wants 2+2. That equals 4.", (
        f"reasoning_content must contain reasoning-channel text; "
        f"got {msg.get('reasoning_content')!r}"
    )


def test_logprobs_path_without_channel_falls_back_unchanged():
    """Non-routed models (channel=None) keep the legacy single-chunk shape."""
    chunks = [
        (None, "hello"),
        (None, "world"),
    ]
    engine = _ChannelRoutedEngine(chunks)
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    # Non-routed legacy behavior: last chunk's ``text`` wins (route never
    # overrides because ``saw_channel`` stays False). We don't aggregate
    # per-chunk text on this path, matching pre-fix behavior.
    assert msg["content"] == "world"
    assert msg.get("reasoning_content") in (None, "")


def test_logprobs_path_reasoning_only_surfaces_via_rescue():
    """Analysis-only stream (no final channel) on the logprobs path.

    Issue #569 updated the contract for this scenario: a reasoning-only
    turn was previously emitted with ``content=null``, which agentic
    OpenAI clients (Cline / Cursor / Codex CLI) reading only the
    ``content`` field would see as an empty assistant message. The
    route-layer ``_rescue_silent_drop_from_reasoning`` now surfaces
    the reasoning trace as ``content`` while keeping
    ``reasoning_content`` populated — duplication between the two
    fields is the lesser evil vs. a silently empty response.

    What this test STILL pins (the #442 contract): reasoning channel
    text does not appear in content via the BLEED path (parser leaking
    analysis content because the channel split failed). The text we
    see in ``content`` here is the EXPLICIT rescue, not a regex leak —
    the engine-level test ``test_truncated_analysis_drops_content_
    and_recovers_reasoning`` in ``tests/test_engine_router_non_stream``
    still asserts the engine returns ``content == ""`` on the same
    input shape.
    """
    chunks = [
        ("reasoning", "thinking only, no final."),
    ]
    engine = _ChannelRoutedEngine(chunks)
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    msg = body["choices"][0]["message"]
    # #569: rescue fires — content carries the reasoning trace so the
    # assistant turn isn't silently empty for OpenAI-compat clients.
    assert msg.get("content") == "thinking only, no final.", (
        f"#569: content must surface reasoning when otherwise empty; "
        f"got {msg.get('content')!r}"
    )
    # reasoning_content stays populated unchanged.
    assert msg.get("reasoning_content") == "thinking only, no final."
