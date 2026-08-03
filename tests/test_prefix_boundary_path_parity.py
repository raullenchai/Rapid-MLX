# SPDX-License-Identifier: Apache-2.0
"""Path-parity regression for the prefix_boundary wiring (#427).

PR #435 added per-message boundary snapshots so hybrid (Mamba+Transformer)
models can reuse cache across multi-turn conversations. The fix wired
``_compute_prefix_boundary`` into ``BatchedEngine.stream_chat()`` (the
streaming path) but missed ``chat()`` and ``generate()`` — the entry
points for non-streaming clients. pydantic_ai, smolagents and langchain
all default to ``stream:false`` so they hit those entry points and the
fix was a no-op for them, leaving the very bug fishloa reported
(opencode, multi-turn agentic, Qwen3.6 hybrid) unfixed.

This test enforces the contract that BOTH paths must compute and forward
``prefix_boundary`` for multi-message conversations. It's structured as
a path-parity assertion: anything the streaming path does for the
boundary handoff, the non-streaming path must do too.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

from vllm_mlx.engine.batched import BatchedEngine

_SENTINEL_BOUNDARY = 42


class _CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


class _StubOutput:
    """Minimal stand-in for the GenerationOutput type returned by
    ``self._engine.generate`` / ``stream_generate``. Carries the fields
    the downstream path reads after generation completes.
    """

    def __init__(self):
        self.output_text = "stub-response"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.finish_reason = "stop"
        self.usage = None
        self.text = "stub-response"
        self.new_text = "stub-response"
        self.tokens = []
        self.new_token_ids = []
        self.finished = True
        self.logprobs = None
        self.token_logprobs = None
        self.output_token_ids = []


class _StubLLMEngine:
    """Captures kwargs that ``BatchedEngine.generate`` forwards to the
    LLM engine. ``add_request`` is the actual prod surface that receives
    ``prefix_boundary`` and sets it on the Request — we record kwargs on
    ``generate`` because that's where ``BatchedEngine`` plumbs through.
    """

    def __init__(self):
        self.last_generate_kwargs: dict[str, Any] | None = None

    async def generate(self, *, prompt, sampling_params, **kwargs):
        self.last_generate_kwargs = kwargs
        return _StubOutput()

    async def add_request(self, *, prompt, sampling_params, **kwargs):
        # Streaming path goes through add_request directly (line 1021
        # in batched.py). Same kwargs surface as generate() — kept in
        # sync so either path observes the boundary.
        self.last_generate_kwargs = kwargs
        return "req-stub"

    async def stream_outputs(self, request_id):
        yield _StubOutput()


def _build_engine(
    monkeypatch, *, is_hybrid: bool = True
) -> tuple[BatchedEngine, _StubLLMEngine]:
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = False
    stub = _StubLLMEngine()
    engine._engine = stub
    monkeypatch.setattr(engine, "_apply_chat_template", lambda *a, **k: "prompt-stub")
    monkeypatch.setattr(
        engine,
        "_compute_prefix_boundary",
        lambda messages, tools=None: _SENTINEL_BOUNDARY,
    )
    monkeypatch.setattr(engine, "_is_hybrid_model", lambda: is_hybrid)
    return engine, stub


def test_chat_non_stream_forwards_prefix_boundary(monkeypatch):
    """``engine.chat()`` (non-streaming) must compute prefix_boundary
    and forward it through ``generate()`` → ``LLMEngine.generate()`` →
    ``add_request()``. Prior to the follow-up to PR #435 this path
    silently dropped the boundary, leaving fishloa's repro broken.
    """
    engine, stub = _build_engine(monkeypatch)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    asyncio.run(engine.chat(messages=messages))
    assert stub.last_generate_kwargs is not None, "generate was never called"
    assert stub.last_generate_kwargs.get("prefix_boundary") == _SENTINEL_BOUNDARY, (
        f"non-stream path dropped prefix_boundary: "
        f"forwarded kwargs={stub.last_generate_kwargs!r}"
    )


def test_stream_chat_forwards_prefix_boundary(monkeypatch):
    """``engine.stream_chat()`` (streaming) — the path PR #435 already
    wired — keeps working. Asserted here as the parity reference so a
    regression in either path is caught by the same test file.
    """
    engine, stub = _build_engine(monkeypatch)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    async def _drain():
        async for _ in engine.stream_chat(messages=messages):
            break  # one yield is enough to prove kwargs were captured

    asyncio.run(_drain())
    assert stub.last_generate_kwargs is not None, "add_request was never called"
    assert stub.last_generate_kwargs.get("prefix_boundary") == _SENTINEL_BOUNDARY, (
        f"stream path dropped prefix_boundary: "
        f"forwarded kwargs={stub.last_generate_kwargs!r}"
    )


def test_single_message_zero_boundary_both_paths(monkeypatch):
    """When ``_compute_prefix_boundary`` returns 0 (single-turn request),
    BOTH paths must end up at the downstream layer with the same value
    so neither one accidentally drifts to a different default. Asserts
    the forwarded value is exactly 0 — engine_core treats 0 as "unset"
    and skips the boundary-split logic in ``_schedule_waiting``.
    """
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = False
    stub = _StubLLMEngine()
    engine._engine = stub
    monkeypatch.setattr(engine, "_apply_chat_template", lambda *a, **k: "prompt-stub")
    monkeypatch.setattr(
        engine, "_compute_prefix_boundary", lambda messages, tools=None: 0
    )
    monkeypatch.setattr(engine, "_is_hybrid_model", lambda: True)
    messages = [{"role": "user", "content": "single"}]

    asyncio.run(engine.chat(messages=messages))
    assert stub.last_generate_kwargs.get("prefix_boundary", 0) == 0, (
        f"non-stream path: expected boundary=0 got {stub.last_generate_kwargs!r}"
    )

    stub.last_generate_kwargs = None

    async def _drain():
        async for _ in engine.stream_chat(messages=messages):
            break

    asyncio.run(_drain())
    assert stub.last_generate_kwargs.get("prefix_boundary", 0) == 0, (
        f"stream path: expected boundary=0 got {stub.last_generate_kwargs!r}"
    )


def test_non_hybrid_model_skips_boundary_both_paths(monkeypatch):
    """Pure Transformer models (gpt-oss-20b-mxfp4-q8, qwen3-coder-4bit, etc.) must NOT
    take the boundary-split path even if a multi-message conversation
    would otherwise produce ``prefix_boundary > 0``.

    Why: ``BatchGenerator.insert_segments`` empirically corrupts harmony
    tool-call channel state across multi-turn-with-tools on gpt-oss-20b-mxfp4-q8
    (pydantic_ai 6_multi_tool drops from 6/6 to 5/6 — agent loops on
    ``add(3,4)`` until ``request_limit`` exhausts). Pure Transformers
    don't need the boundary save anyway — trim+supersequence reuse
    works — so the gate is a free no-op for them. This test pins the
    gate so a future refactor can't quietly re-enable the broken path.
    """
    engine, stub = _build_engine(monkeypatch, is_hybrid=False)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    asyncio.run(engine.chat(messages=messages))
    assert stub.last_generate_kwargs.get("prefix_boundary", 0) == 0, (
        f"non-stream path: non-hybrid model leaked boundary "
        f"into kwargs={stub.last_generate_kwargs!r}"
    )

    stub.last_generate_kwargs = None

    async def _drain():
        async for _ in engine.stream_chat(messages=messages):
            break

    asyncio.run(_drain())
    assert stub.last_generate_kwargs.get("prefix_boundary", 0) == 0, (
        f"stream path: non-hybrid model leaked boundary "
        f"into kwargs={stub.last_generate_kwargs!r}"
    )


def test_nontrimmable_pure_attention_forwards_boundary_both_paths(monkeypatch):
    """Pure attention with a detected non-trimmable cache needs snapshots."""
    engine, stub = _build_engine(monkeypatch, is_hybrid=False)
    engine._scheduler_config = SimpleNamespace(hybrid_cache_entries=8)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    asyncio.run(engine.chat(messages=messages))
    assert stub.last_generate_kwargs.get("prefix_boundary") == _SENTINEL_BOUNDARY

    stub.last_generate_kwargs = None

    async def _drain():
        async for _ in engine.stream_chat(messages=messages):
            break

    asyncio.run(_drain())
    assert stub.last_generate_kwargs.get("prefix_boundary") == _SENTINEL_BOUNDARY


def test_prefix_boundary_prefers_latest_stable_message_boundary(monkeypatch):
    """Exclude the transient generation suffix, not the latest user turn."""
    engine, _ = _build_engine(monkeypatch)
    engine._compute_prefix_boundary = BatchedEngine._compute_prefix_boundary.__get__(
        engine, BatchedEngine
    )
    engine._tokenizer = _CharacterTokenizer()

    def render(messages, tools=None, *, add_generation_prompt=True, **kwargs):
        body = "|".join(str(message.get("content", "")) for message in messages)
        return body + ("|ASSISTANT_GENERATION" if add_generation_prompt else "")

    monkeypatch.setattr(engine, "_apply_chat_template", render)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]
    stable = render(messages, add_generation_prompt=False)

    assert engine._compute_prefix_boundary(messages) == len(stable)


def test_prefix_boundary_stops_before_transient_server_priming(monkeypatch):
    """A server-only reminder is absent from the client's next request."""
    engine, _ = _build_engine(monkeypatch)
    engine._compute_prefix_boundary = BatchedEngine._compute_prefix_boundary.__get__(
        engine, BatchedEngine
    )
    engine._tokenizer = _CharacterTokenizer()

    def render(messages, tools=None, *, add_generation_prompt=True, **kwargs):
        body = "|".join(
            f"{message.get('role')}:{message.get('content')}" for message in messages
        )
        return body + ("|GEN" if add_generation_prompt else "")

    monkeypatch.setattr(engine, "_apply_chat_template", render)
    stable_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "investigate"},
        {"role": "assistant", "content": "prior work"},
        {"role": "user", "content": "continue"},
    ]
    messages = [
        *stable_messages,
        {"role": "developer", "content": "temporary progress checkpoint"},
    ]

    boundary = engine._compute_prefix_boundary(
        messages, transient_message_start=len(stable_messages)
    )
    real = render(messages)
    future = render(
        [
            *stable_messages,
            {"role": "assistant", "content": "__rapid_mlx_boundary_probe__"},
        ],
        add_generation_prompt=False,
    )
    expected = 0
    for real_char, future_char in zip(real, future):
        if real_char != future_char:
            break
        expected += 1

    assert boundary == expected
    assert boundary > 0
    assert "temporary" not in real[:boundary]


def test_transient_priming_marker_is_removed_before_chat_templating():
    messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "developer",
            "content": "server-only reminder",
            "_rapid_mlx_transient_priming": True,
        },
    ]

    cleaned, transient_start = BatchedEngine._prepare_cache_stable_messages(messages)

    assert transient_start == 1
    assert cleaned == [
        {"role": "user", "content": "continue"},
        {"role": "developer", "content": "server-only reminder"},
    ]
    assert messages[-1]["_rapid_mlx_transient_priming"] is True


def test_prefix_boundary_falls_back_when_no_generation_form_is_not_prefix(
    monkeypatch,
):
    """Third-party template drift keeps the conservative dummy-user LCP."""
    engine, _ = _build_engine(monkeypatch)
    engine._compute_prefix_boundary = BatchedEngine._compute_prefix_boundary.__get__(
        engine, BatchedEngine
    )
    engine._tokenizer = _CharacterTokenizer()

    def render(messages, tools=None, *, add_generation_prompt=True, **kwargs):
        if not add_generation_prompt:
            return "not-a-prefix"
        latest = messages[-1].get("content", "")
        return f"shared-history|{latest}|generation"

    monkeypatch.setattr(engine, "_apply_chat_template", render)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]

    assert engine._compute_prefix_boundary(messages) == len("shared-history|")


def test_prefix_boundary_rejects_last_user_rewrite_on_next_turn(monkeypatch):
    engine, _ = _build_engine(monkeypatch)
    engine._compute_prefix_boundary = BatchedEngine._compute_prefix_boundary.__get__(
        engine, BatchedEngine
    )
    engine._tokenizer = _CharacterTokenizer()

    def render(messages, tools=None, *, add_generation_prompt=True, **kwargs):
        parts = []
        for index, message in enumerate(messages):
            content = str(message.get("content", ""))
            is_final_user = index == len(messages) - 1 and message.get("role") == "user"
            parts.append(f"LAST:{content}" if is_final_user else f"TURN:{content}")
        body = "shared|" + "|".join(parts)
        return body + ("|GEN" if add_generation_prompt else "")

    monkeypatch.setattr(engine, "_apply_chat_template", render)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "latest"},
    ]

    boundary = engine._compute_prefix_boundary(messages)
    assert boundary < len(render(messages, add_generation_prompt=False))
    assert boundary == len("shared|TURN:system|")
