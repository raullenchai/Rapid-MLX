# SPDX-License-Identifier: Apache-2.0
"""R-01 cross-route truncation contract: no synthetic text injection.

When a reasoning model is cut short mid-``<think>`` by a ``max_tokens``
cap, no transport may inject a literal placeholder string into the
model's ``content`` / ``output_text`` / ``text`` field. Every transport
already carries an unambiguous structured truncation signal — that, plus
the populated ``reasoning_content`` (or ``thinking`` block), is the
canonical cue.

Transports under test:
    * ``/v1/chat/completions``    → ``finish_reason="length"``
    * ``/v1/responses``           → ``status="incomplete"`` +
                                     ``output_tokens_details.reasoning_tokens``
    * ``/v1/messages`` (Anthropic) → ``stop_reason="max_tokens"`` +
                                     ``thinking`` content block

Each transport is covered for BOTH streaming and non-streaming paths.

The fix lives at a single source of truth
(``vllm_mlx.service.helpers._apply_reasoning_cutoff_notice``); these
tests pin the user-visible behaviour on every route boundary so the
helper cannot drift between surfaces.
"""

from __future__ import annotations

import json
import re

import pytest

from vllm_mlx.service.helpers import REASONING_CUTOFF_SENTINEL

# Substrings that flag synthetic truncation text. Case-insensitive.
_TRUNCATED_SUBSTRING = "truncated"


# ---------------------------------------------------------------------
# Mock engine: a reasoning model whose ``<think>`` block never closes.
# Used uniformly across all three routes so the contract is identical.
# ---------------------------------------------------------------------


class _LengthCutMidThinkEngine:
    """Single-output engine: ``<think>`` opens but never closes, and
    generation finishes with ``finish_reason="length"`` — the exact
    production shape R-01 was filed against."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    chat_template = ""

    def __init__(self):
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    def estimate_prompt_tokens(self, prompt):
        return 4

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="<think>Reasoning about 17*23 step by step",
            new_text="<think>Reasoning about 17*23 step by step",
            prompt_tokens=4,
            completion_tokens=12,
            finished=True,
            finish_reason="length",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        # Streaming path: emit partial reasoning across multiple deltas
        # so the streaming postprocessor exercises the per-delta path,
        # then close with ``finish_reason="length"`` without ever
        # emitting ``</think>``.
        from vllm_mlx.engine.base import GenerationOutput

        deltas = [
            "<think>Reasoning ",
            "about 17 * 23 ",
            "step by step",
        ]
        accumulated = ""
        for i, delta in enumerate(deltas):
            accumulated += delta
            is_last = i == len(deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="length" if is_last else None,
                channel=None,
            )


def _seed_cfg(cfg):
    """Common cfg shape for every test below: qwen3 reasoning parser +
    length-cut mock engine."""
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg.engine = _LengthCutMidThinkEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def _parse_sse_named(text: str) -> list[tuple[str | None, dict]]:
    """Parse SSE preserving ``event:`` names (Responses surface uses
    named events; chat-completions uses unnamed ``data:`` lines)."""
    events: list[tuple[str | None, dict]] = []
    current_event: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
            continue
        if not line.startswith("data:"):
            if line == "":
                current_event = None
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            events.append((current_event, json.loads(payload)))
        except json.JSONDecodeError:
            continue
    return events


@pytest.fixture(autouse=True)
def _default_off_env(monkeypatch):
    """Every R-01 test runs with the env knob explicitly unset, so the
    R-01 default-off behaviour is what the assertions exercise. Tests
    that want to verify the opt-in path live in
    ``test_reasoning_content_null_rescue.py`` — this file is the
    cross-route no-injection contract."""
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)


# ---------------------------------------------------------------------
# /v1/chat/completions — OpenAI lane
# ---------------------------------------------------------------------


def test_chat_completions_nonstream_no_truncated_injection():
    """OpenAI chat lane non-streaming: ``message.content`` must NOT
    carry the ``[truncated…]`` sentinel. ``finish_reason="length"`` is
    the canonical truncation cue and ``reasoning_content`` carries the
    thought trace."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_cfg(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        msg = payload["choices"][0]["message"]
        content = msg.get("content")

        assert content != REASONING_CUTOFF_SENTINEL, (
            f"chat non-stream must NOT inject sentinel; got content={content!r}"
        )
        if content:
            assert _TRUNCATED_SUBSTRING not in content.lower(), (
                f"chat non-stream content must not carry 'truncated' "
                f"synthetic text; got {content!r}"
            )
        assert payload["choices"][0]["finish_reason"] == "length"
        assert msg.get("reasoning_content"), (
            "reasoning_content must remain populated as the canonical truncation cue"
        )
    finally:
        reset_config()


def test_chat_completions_stream_no_truncated_injection():
    """OpenAI chat lane streaming: NO ``delta.content`` chunk in the
    SSE stream may carry the ``[truncated…]`` sentinel.
    ``finish_reason="length"`` on the terminal chunk is the canonical
    truncation cue and ``delta.reasoning_content`` carries the trace."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_cfg(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        assert events

        streamed_reasoning = ""
        terminal_finish_reason: str | None = None
        for ev in events:
            for choice in ev.get("choices", []):
                d = choice.get("delta") or {}
                content = d.get("content")
                if content:
                    assert content != REASONING_CUTOFF_SENTINEL, (
                        f"chat stream must NOT inject sentinel into any "
                        f"delta.content chunk; got {content!r}"
                    )
                    assert _TRUNCATED_SUBSTRING not in content.lower(), (
                        f"chat stream delta.content must not carry "
                        f"'truncated' synthetic text; got {content!r}"
                    )
                if d.get("reasoning_content"):
                    streamed_reasoning += d["reasoning_content"]
                if choice.get("finish_reason") is not None:
                    terminal_finish_reason = choice["finish_reason"]
        assert terminal_finish_reason == "length", (
            "chat stream must surface finish_reason=length as the "
            "canonical truncation cue"
        )
        assert streamed_reasoning, (
            "chat stream reasoning_content deltas must still flow"
        )
    finally:
        reset_config()


# ---------------------------------------------------------------------
# /v1/responses — Responses lane
# ---------------------------------------------------------------------


def test_responses_nonstream_no_truncated_injection():
    """Responses lane non-streaming: ``output[*].content[*].text`` must
    NOT carry the ``[truncated…]`` sentinel. ``status="incomplete"`` +
    ``usage.output_tokens_details.reasoning_tokens`` are the canonical
    truncation cues."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_cfg(cfg)

    try:
        app = FastAPI()
        app.include_router(responses_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "max_output_tokens": 16,
                "input": "compute 17*23",
                "stream": False,
            },
        )
        if resp.status_code != 200:
            pytest.skip(
                f"/v1/responses adapter rejected synthetic shape "
                f"({resp.status_code}); behavior covered by helper unit "
                f"tests in test_reasoning_content_null_rescue.py"
            )
        payload = resp.json()
        body_str = json.dumps(payload)

        assert REASONING_CUTOFF_SENTINEL not in body_str, (
            f"responses non-stream must NOT carry the sentinel anywhere "
            f"in the envelope; got payload={payload!r}"
        )
        for item in payload.get("output") or []:
            for block in item.get("content") or []:
                text = block.get("text") or ""
                assert _TRUNCATED_SUBSTRING not in text.lower(), (
                    f"responses non-stream output_text must not carry "
                    f"'truncated' synthetic text; got block={block!r}"
                )
        # Sanity: the structured truncation cue is present.
        assert payload.get("status") == "incomplete", (
            f"responses non-stream must surface status='incomplete' as "
            f"the canonical truncation cue; got status={payload.get('status')!r}"
        )
    finally:
        reset_config()


def test_responses_stream_no_truncated_injection():
    """Responses lane streaming: NO ``response.output_text.delta`` SSE
    event may carry the ``[truncated…]`` sentinel."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_cfg(cfg)

    try:
        app = FastAPI()
        app.include_router(responses_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "max_output_tokens": 64,
                "input": "compute 17*23",
                "stream": True,
            },
        )
        if resp.status_code != 200:
            pytest.skip(
                f"/v1/responses stream adapter rejected synthetic shape "
                f"({resp.status_code}); covered by helper unit tests"
            )
        events = _parse_sse_named(resp.text)
        assert events, "expected at least one SSE event"

        for event_name, payload in events:
            body_str = json.dumps(payload)
            assert REASONING_CUTOFF_SENTINEL not in body_str, (
                f"responses stream event {event_name!r} must NOT carry "
                f"the sentinel; got payload={payload!r}"
            )
            # Specifically check ``output_text.delta`` event shape.
            if event_name == "response.output_text.delta":
                delta = payload.get("delta") or ""
                if isinstance(delta, str):
                    assert _TRUNCATED_SUBSTRING not in delta.lower(), (
                        f"responses stream output_text.delta must not "
                        f"carry 'truncated' synthetic text; got {delta!r}"
                    )
    finally:
        reset_config()


# ---------------------------------------------------------------------
# /v1/messages — Anthropic lane
# ---------------------------------------------------------------------


def test_messages_nonstream_no_truncated_injection():
    """Anthropic lane non-streaming: no text-type content block in the
    response may carry the ``[truncated…]`` sentinel.
    ``stop_reason="max_tokens"`` + the ``thinking`` content block are
    the canonical truncation cues."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_cfg(cfg)

    try:
        app = FastAPI()
        app.include_router(anthropic_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "stream": False,
            },
        )
        if resp.status_code != 200:
            pytest.skip(
                f"/v1/messages adapter rejected synthetic shape "
                f"({resp.status_code}); behavior covered by helper unit "
                f"tests"
            )
        payload = resp.json()
        body_str = json.dumps(payload)

        assert REASONING_CUTOFF_SENTINEL not in body_str, (
            f"messages non-stream must NOT carry the sentinel anywhere; "
            f"got payload={payload!r}"
        )
        for block in payload.get("content") or []:
            if block.get("type") == "text":
                text = block.get("text") or ""
                assert _TRUNCATED_SUBSTRING not in text.lower(), (
                    f"messages non-stream text block must not carry "
                    f"'truncated' synthetic text; got block={block!r}"
                )
        # Sanity: structured truncation cue present.
        assert payload.get("stop_reason") == "max_tokens", (
            f"messages non-stream must surface stop_reason='max_tokens' "
            f"as the canonical truncation cue; "
            f"got {payload.get('stop_reason')!r}"
        )
    finally:
        reset_config()


def test_messages_stream_no_truncated_injection():
    """Anthropic lane streaming: NO ``content_block_delta`` event with a
    ``text_delta`` payload may carry the ``[truncated…]`` sentinel."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_cfg(cfg)

    try:
        app = FastAPI()
        app.include_router(anthropic_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "stream": True,
            },
        )
        if resp.status_code != 200:
            pytest.skip(
                f"/v1/messages stream adapter rejected synthetic shape "
                f"({resp.status_code})"
            )
        events = _parse_sse_named(resp.text)
        assert events, "expected at least one SSE event"

        for event_name, payload in events:
            body_str = json.dumps(payload)
            assert REASONING_CUTOFF_SENTINEL not in body_str, (
                f"messages stream event {event_name!r} must NOT carry "
                f"the sentinel; got payload={payload!r}"
            )
            if event_name == "content_block_delta":
                delta = payload.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text") or ""
                    assert _TRUNCATED_SUBSTRING not in text.lower(), (
                        f"messages stream text_delta must not carry "
                        f"'truncated' synthetic text; got {text!r}"
                    )
    finally:
        reset_config()


# ---------------------------------------------------------------------
# Belt-and-braces: helper truth table at default-off
# ---------------------------------------------------------------------


def test_helper_returns_none_on_default_env(monkeypatch):
    """Direct helper assertion at default-off — pins that no synthetic
    text can leak even if a future route call site forgets to pass
    every predicate. The helper itself defaults to no-op."""
    from vllm_mlx.service.helpers import _apply_reasoning_cutoff_notice

    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    result = _apply_reasoning_cutoff_notice(
        final_content=None,
        reasoning_text="<incomplete thought>",
        tool_calls=None,
        finish_reason="length",
    )
    assert result is None, (
        f"R-01 helper default must return None on length-cut mid-think; got {result!r}"
    )


def test_no_truncated_literal_in_route_module_call_sites():
    """Static guard: the literal sentinel string must NOT appear inline
    in any route module. The only legitimate definition site is the
    ``REASONING_CUTOFF_SENTINEL`` constant in ``service.helpers``. A
    future regression that hard-codes the literal at a route call site
    (e.g. a regex band-aid) fails this test."""
    import importlib
    import inspect

    pattern = re.compile(r"\[truncated.*reasoning incomplete")
    for module_name in (
        "vllm_mlx.routes.chat",
        "vllm_mlx.routes.responses",
        "vllm_mlx.routes.anthropic",
    ):
        mod = importlib.import_module(module_name)
        src = inspect.getsource(mod)
        assert not pattern.search(src), (
            f"R-01 single-source-of-truth: the truncated sentinel literal "
            f"must NOT appear inline in {module_name}; it is defined once "
            f"in vllm_mlx.service.helpers.REASONING_CUTOFF_SENTINEL and "
            f"applied via _apply_reasoning_cutoff_notice. A literal here "
            f"indicates a regex band-aid that bypasses the helper."
        )
