# SPDX-License-Identifier: Apache-2.0
"""D-ANTHRO-SPEC-POLISH route-level tests (F7 + F12).

This module covers the route-level behavior the adapter unit tests in
``test_anthropic_adapter.py`` can't reach:

* **F7** — ``tool_choice={"type":"none"}`` must drop ``tools`` from
  the engine's ``chat_kwargs`` so the chat template never injects
  tool definitions into the prompt. Adapter-level coverage in
  ``test_anthropic_adapter.py`` locks the input contract; this file
  locks the wire-through-to-engine path.

* **F12** — ``/v1/messages/count_tokens`` must apply the SAME chat
  template that ``/v1/messages`` applies before tokenizing, so the
  count matches ``usage.input_tokens`` exactly. Pre-fix the count
  consistently under-reported by ~5 tokens (Sergei evidence, delta=-5
  across 5 unrelated prompts).

Both surfaces share a single ``_RecordingEngine`` so we can assert on
exactly what ``build_prompt`` saw — the F-220 chat-route test pattern
applied to the Anthropic route.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.anthropic import router as anthropic_router


class _StubTokenizer:
    """Trivial tokenizer that returns one token per word + a constant
    per-render boilerplate so we can distinguish "raw text encoded"
    from "chat-template-rendered prompt encoded".

    The chat template prefix accounts for the per-turn role tokens
    (``<|im_start|>user`` etc.) that the qwen3 template emits.
    """

    chat_template = "<|im_start|>{{ messages }}<|im_end|>"
    bos_token = None

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return list(range(len(text.split())))


class _RecordingEngine:
    """Records every ``chat()`` and ``build_prompt()`` call so tests
    can assert on the kwargs the route delivered.

    ``build_prompt`` returns a string with a 5-token "boilerplate"
    prefix so a downstream tokenizer encode will count both the user
    content tokens AND the per-turn template overhead — modeling the
    real qwen3 template behavior that exposes the F12 delta.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = _StubTokenizer()

    def __init__(self):
        self.last_chat_kwargs: dict[str, Any] | None = None
        self.last_messages: Any = None
        self.last_build_prompt_kwargs: dict[str, Any] | None = None
        self.last_build_prompt_messages: Any = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        self.last_build_prompt_messages = messages
        self.last_build_prompt_kwargs = {
            "tools": tools,
            "enable_thinking": enable_thinking,
        }
        # Per-turn role overhead = 5 stub tokens (mirrors the qwen3
        # template's ``<|im_start|>user\n...<|im_end|>\n
        # <|im_start|>assistant\n`` boilerplate).
        body = " ".join(
            m.get("content", "") if isinstance(m, dict) else "" for m in messages
        )
        return f"role role role role role {body}"

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )


def _make_client(engine: _RecordingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.tool_call_parser = None  # ensure no extra suffix injection
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(anthropic_router)
    return TestClient(app)


_TOOLS_FIXTURE = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


# ===========================================================================
# F7 — tool_choice={"type":"none"} strips tools from the engine call.
# ===========================================================================


def test_f7_tool_choice_none_strips_tools_from_engine_kwargs():
    """F7: when the client sets ``tool_choice={"type":"none"}``, the
    engine must NOT see ``tools`` in its ``chat_kwargs``. Pre-fix the
    adapter forwarded tools verbatim, the chat template injected them
    into the system prompt, and the model emitted a partial
    ``<tool_call>{...}`` marker that leaked into the response text
    (Sergei repro F7).

    Single source of truth: the fix lives in the adapter (drops
    ``tools`` when ``tool_choice == "none"``), so any future route
    that consumes ``openai_request.tools`` after the adapter runs is
    automatically covered.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "what's the weather in Tokyo?"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.last_chat_kwargs is not None
    # ``tools`` either absent or None on the engine call.
    tools_seen = engine.last_chat_kwargs.get("tools")
    assert not tools_seen, (
        f"tool_choice='none' must strip tools from engine call; "
        f"got tools={tools_seen!r}"
    )


def test_f7_tool_choice_none_response_has_no_tool_use_block():
    """F7 wire-level: the response content list must NOT carry any
    ``tool_use`` block when ``tool_choice="none"``. Even if a defiant
    model wrote raw ``<tool_call>{...}`` text (now impossible because
    tools were stripped), the parser layer would still drop the
    ``tool_use`` block under ``tool_choice="none"``.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    tool_use_blocks = [
        b for b in body.get("content", []) if b.get("type") == "tool_use"
    ]
    assert tool_use_blocks == [], body


def test_f7_tool_choice_none_text_has_no_tool_call_marker():
    """F7 leak check: the model's text output must not contain a raw
    ``<tool_call>{...}`` marker. With tools stripped from the prompt,
    the model has nothing to nudge it toward emitting that marker —
    but we lock the wire shape directly so a future regression
    surfaces fast.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    for blk in body.get("content", []):
        if blk.get("type") == "text":
            assert "<tool_call>" not in (blk.get("text") or ""), blk


def test_f7_tool_choice_auto_keeps_tools():
    """Negative control: ``tool_choice="auto"`` (the default) leaves
    tools intact so the model can still call them. Locks against an
    over-eager strip that would silently disable all tool use."""
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "tools": _TOOLS_FIXTURE,
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "weather"}],
        },
    )
    assert resp.status_code == 200
    tools_seen = engine.last_chat_kwargs.get("tools")
    assert tools_seen, (
        f"tool_choice='auto' must keep tools on engine call; got tools={tools_seen!r}"
    )


# ===========================================================================
# F12 — count_tokens applies the same chat template as /v1/messages.
# ===========================================================================


def test_f12_count_tokens_applies_chat_template():
    """F12: ``/v1/messages/count_tokens`` must run the request through
    ``engine.build_prompt`` so the chat template's per-turn role
    boilerplate (``<|im_start|>user`` etc.) is counted. Pre-fix it
    tokenized each text segment in isolation and consistently
    under-reported by ~5 tokens (Sergei repro F12 across 5 prompts).

    Probe: the stub ``build_prompt`` adds a 5-token "role" prefix to
    any rendered prompt. If the route correctly delegates to
    ``build_prompt``, the count includes those 5 tokens. If the route
    falls back to the legacy per-segment count, those 5 tokens are
    missing.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Pre-fix count: 2 tokens ("hello world" → 2 stub tokens).
    # Post-fix count: 2 + 5 = 7 stub tokens (5 role-prefix tokens
    # plus the 2 content tokens) — proving build_prompt ran.
    assert body["input_tokens"] == 7, (
        f"count_tokens should apply the chat template; expected 7, got {body}"
    )
    # The stub engine recorded the build_prompt call — direct
    # evidence the route delegated to the template path.
    assert engine.last_build_prompt_messages is not None


def test_f12_count_tokens_matches_messages_usage_input_tokens():
    """F12 parity: the count must equal ``usage.input_tokens`` for an
    identical request. The fixture engine reports
    ``prompt_tokens=4`` from its ``chat()`` mock — that path doesn't
    re-render the prompt, so the test asserts the parity contract at
    the count_tokens surface via the stub's known boilerplate cost.

    Stronger parity is enforced end-to-end by the live-server repro
    script (``/tmp/d-anthro-polish-repro.py``); this unit test locks
    the route plumbing so the chat-template path can't silently
    regress to the per-segment count.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)

    # Same body to both endpoints (minus max_tokens which count_tokens
    # doesn't require).
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "a quick brown fox"}],
    }

    r1 = client.post(
        "/v1/messages",
        json={**body, "max_tokens": 5},
    )
    r2 = client.post("/v1/messages/count_tokens", json=body)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Both surfaces saw the same rendered prompt — the count must
    # match the build_prompt-derived token count exactly.
    count = r2.json()["input_tokens"]
    # 5 stub tokens (role overhead) + 4 content tokens ("a quick brown fox") = 9.
    assert count == 9, f"expected 9 tokens, got {count}"


def test_f12_count_tokens_excludes_role_overhead_on_fallback():
    """Legacy fallback path: when ``build_prompt`` is unavailable
    (e.g. on a test stub that doesn't expose it), the count falls
    back to the per-segment tokenizer encode. Documents the
    fallback semantics so a future refactor doesn't silently drop
    the fallback.

    Setup a minimal stub WITHOUT ``build_prompt`` and verify the
    response is a non-error count (the historical contract holds
    on test stubs).
    """

    class _NoBuildPromptEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = _StubTokenizer()
        # Deliberately omit ``build_prompt``.

    engine = _NoBuildPromptEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
        },
    )
    assert resp.status_code == 200
    # Fallback path: 2 content tokens, no role overhead.
    assert resp.json()["input_tokens"] == 2


def test_f12_count_tokens_without_max_tokens_does_not_422():
    """F12 spec parity: ``/v1/messages/count_tokens`` accepts requests
    WITHOUT ``max_tokens`` (Anthropic's spec). The adapter shim we
    use internally to reuse ``AnthropicRequest`` must not surface a
    ``max_tokens`` requirement to the count_tokens caller.
    """
    engine = _RecordingEngine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
