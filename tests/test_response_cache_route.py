# SPDX-License-Identifier: Apache-2.0
"""Route-level integration for the opt-in prompt-deterministic response
cache — the feature's PRIMARY integration surface.

The unit tests in ``tests/test_response_cache.py`` cover the LRU store,
the key builder, and the singleton, but none of them exercise the wiring
in ``routes/chat.py`` that actually performs the lookup + store around a
real chat request. Deleting both the lookup and store blocks from the
route would leave every unit test green. These tests fire HTTP requests
through the live FastAPI chat router with a generation-counting fake
engine and assert the end-to-end short-circuit:

* the same deterministic (greedy) request issued TWICE invokes generation
  exactly ONCE (second served from cache),
* the two responses carry equivalent content,
* a change to a render-determining input (the messages) is NOT served from
  the first request's cache entry (proves the key follows the consumed
  input, not a constant).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.response_cache import (
    configure_response_cache,
    get_response_cache,
    reset_response_cache_for_tests,
)
from vllm_mlx.routes.chat import router as chat_router


class _CountingEngine:
    """Fake engine whose ``chat()`` counts generation invocations and
    echoes a per-call-unique marker so a cached (replayed) response is
    distinguishable from a freshly-generated one.

    ``chat()`` is the method the non-streaming, non-guided chat path
    invokes (routes/chat.py ~line 2884). ``build_prompt`` is present for
    route validation callers, but the cache key is derived from the raw
    messages — a call to ``build_prompt`` here would be a SECOND render,
    which the cache deliberately avoids.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.chat_calls = 0
        self.build_prompt_calls = 0
        self.seen_messages: list[Any] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        self.build_prompt_calls += 1
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.chat_calls += 1
        self.seen_messages.append(messages)
        # Deterministic content per messages, but tag with the call index
        # so a REPLAYED (cached) response — which reuses call #1's body —
        # is distinguishable from a fresh generation.
        return GenerationOutput(
            text=f"answer (gen#{self.chat_calls})",
            raw_text=f"answer (gen#{self.chat_calls})",
            prompt_tokens=4,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
        )


def _make_client(engine: _CountingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_response_cache_for_tests()
    yield
    reset_response_cache_for_tests()


def _greedy_body(content: str) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,  # greedy → cacheable
        "max_tokens": 16,
        "stream": False,
    }


def test_repeated_deterministic_request_served_from_cache():
    """MUTATION-KILL: deleting the lookup+store blocks in routes/chat.py
    makes generation run TWICE, so ``chat_calls == 2`` and this fails.

    Two identical greedy requests → generation runs ONCE; the second is a
    cache hit with equivalent content.
    """
    engine = _CountingEngine()
    configure_response_cache(64)  # enable the cache (post-load config path)
    client = _make_client(engine)

    r1 = client.post("/v1/chat/completions", json=_greedy_body("hello there"))
    assert r1.status_code == 200, r1.text
    r2 = client.post("/v1/chat/completions", json=_greedy_body("hello there"))
    assert r2.status_code == 200, r2.text

    # (a) generation invoked exactly once — the second was served cached.
    assert engine.chat_calls == 1, (
        f"expected 1 generation, got {engine.chat_calls} — the second "
        "request was NOT served from cache (lookup/store wiring missing?)"
    )

    # (b) equivalent content across the two responses.
    c1 = r1.json()["choices"][0]["message"]["content"]
    c2 = r2.json()["choices"][0]["message"]["content"]
    assert c1 == c2 == "answer (gen#1)"

    # (c) metadata reflects a hit on the second: a distinct response object
    # (fresh id) carrying the SAME stored completion body. The id differs
    # (each hit is a distinct response), the content is the replayed one.
    assert r1.json()["id"] != r2.json()["id"]

    # The cache recorded exactly one hit and one miss.
    snap = get_response_cache().snapshot()
    assert snap["hits"] == 1
    assert snap["misses"] == 1


def test_cache_disabled_does_not_short_circuit():
    """With the cache disabled (capacity 0, the default), two identical
    greedy requests each run generation — zero behavior change."""
    engine = _CountingEngine()
    configure_response_cache(0)  # explicitly disabled
    client = _make_client(engine)

    client.post("/v1/chat/completions", json=_greedy_body("hi"))
    client.post("/v1/chat/completions", json=_greedy_body("hi"))
    assert engine.chat_calls == 2


def test_key_follows_consumed_messages_not_a_constant():
    """The cache key is derived from the request's messages (the exact
    input generation consumes), so a DIFFERENT prompt must MISS and run
    generation again — it must not collide with the first entry.

    This guards the B2 fix: keying on the consumed messages (not a second
    independent render) still discriminates distinct prompts.
    """
    engine = _CountingEngine()
    configure_response_cache(64)
    client = _make_client(engine)

    client.post("/v1/chat/completions", json=_greedy_body("first prompt"))
    client.post("/v1/chat/completions", json=_greedy_body("second prompt"))
    # Two distinct prompts → two generations (no false cache hit).
    assert engine.chat_calls == 2


def test_non_greedy_request_is_never_cached():
    """A sampled (temperature > 0) request is not deterministic, so it must
    never be short-circuited — two identical sampled requests each run
    generation."""
    engine = _CountingEngine()
    configure_response_cache(64)
    client = _make_client(engine)

    body = _greedy_body("sample me")
    body["temperature"] = 0.9  # sampling → not cacheable

    client.post("/v1/chat/completions", json=body)
    client.post("/v1/chat/completions", json=body)
    assert engine.chat_calls == 2
    # No lookup/store happened for the ineligible request.
    snap = get_response_cache().snapshot()
    assert snap["hits"] == 0
    assert snap["misses"] == 0
    assert snap["entries"] == 0
