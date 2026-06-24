# SPDX-License-Identifier: Apache-2.0
"""R11-B / R11-M-F1 regression guard — streaming ``/v1/responses`` emits a
``reasoning`` output item when ``max_output_tokens`` cuts off mid-think.

Background: Mira R1 F1 captured the streaming ``/v1/responses`` path
silently shipping only three SSE events (``response.created``,
``response.in_progress``, ``response.completed``) with ``output:[]`` and
``status:"completed"`` when ``max_output_tokens`` truncated the model while
still inside ``<think>...</think>``. The non-streaming surface on the SAME
prompt correctly returned ``status:"incomplete"`` + a ``reasoning`` output
item via ``openai_to_responses``. This cross-path asymmetry was
unrecoverable by Codex CLI / openai-python clients — the streaming
``output[]`` was empty so nothing displayable reached the user.

The fix (vllm_mlx/routes/responses.py ``_stream_responses``):
  * Accumulate reasoning bytes from BOTH the channel-routed (gemma4 /
    harmony) and parser-routed (qwen3 / deepseek / glm4 / minimax) paths.
  * Emit a ``response.output_item.added`` → ``response.output_item.done``
    pair for a ``reasoning`` item before ``response.completed`` whenever
    reasoning text was accumulated.
  * Mark the reasoning item ``status:"incomplete"`` when
    ``last_finish_reason == "length"`` (mirrors the non-stream surface's
    ``_build_reasoning_output_item`` invocation under R11-B).
  * Emit ``response.completed`` with ``status:"incomplete"`` and
    ``incomplete_details:{reason:"max_output_tokens"}`` instead of
    ``status:"completed"`` whenever the engine reported
    ``finish_reason="length"``.

These tests pin those invariants and the streaming/non-streaming parity
contract.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Engine stub: streams a reasoning-only output that ends with
# ``finish_reason="length"``. Mirrors the live-server shape Mira R1 F1
# captured on Qwen3-0.6B with ``max_output_tokens=32`` — the model emitted
# ``<think>\n...thoughts...`` and was cut off before the closing tag.
# ---------------------------------------------------------------------------


class _Tokenizer:
    # Qwen3-shaped chat template: includes both ``<think>`` and the
    # ``add_generation_prompt`` marker so
    # ``service.helpers._should_start_in_thinking`` returns True. That
    # routes the parser's no-tag Case-3 stream to ``reasoning`` and lets
    # the qwen3 finalize correctly classify the cut-off thought trace as
    # ``DeltaMessage(reasoning=...)`` instead of leaking to ``content``.
    chat_template = (
        "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n"
        "{% endif %}"
    )

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None
    reasoning_text: str = ""
    matched_stop: str | None = None
    cached_tokens: int = 0


# Realistic mid-think prefix the model emits when ``<think>`` is in the
# prompt (Qwen3 chat template injects ``<think>\n``) and ``max_output_tokens``
# trips before ``</think>`` arrives. NO ``</think>`` token here.
_REASONING_CHUNKS = [
    "Okay, the user said",
    " \"Hi\". I should respond",
    " politely. Let me",
    " start with a greeting.",
]


class _ReasoningCutoffEngine:
    """Streams the reasoning chunks above, then a final chunk with
    ``finish_reason="length"`` to mimic the ``max_output_tokens`` cutoff."""

    preserve_native_tool_format = False

    def __init__(self):
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        full = "".join(_REASONING_CHUNKS)
        return _GenerationOutput(
            text="",
            raw_text=full,
            reasoning_text=full,
            prompt_tokens=4,
            completion_tokens=len(_REASONING_CHUNKS),
            finish_reason="length",
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, chunk in enumerate(_REASONING_CHUNKS):
            accumulated += chunk
            is_last = i == len(_REASONING_CHUNKS) - 1
            yield _GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4 if i == 0 else 0,
                completion_tokens=i + 1,
                # ``length`` only on the terminal chunk — the engine
                # reports the truncation signal once it runs out of
                # budget, the same shape the live MLX scheduler emits.
                finish_reason="length" if is_last else None,
            )


# ---------------------------------------------------------------------------
# Test fixture — mirrors ``tests/test_responses_sse_event_order.py``'s
# lightweight engine swap so we don't need to load a real MLX model.
# ---------------------------------------------------------------------------


_IMPORTED = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


@pytest.fixture
def responses_client(monkeypatch):
    previous_modules = {n: sys.modules.get(n, _MISSING) for n in _IMPORTED}
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _ReasoningCutoffEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    # Wire the Qwen3 reasoning parser — that's the path the bug repro
    # used (the live Mira R1 F1 server ran ``--reasoning-parser qwen3``
    # auto-configured from the model family detector).
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = "qwen3"

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


def _parse_sse(body_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_text = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_text = line[len("data:") :].strip()
        if event_name and data_text is not None:
            events.append((event_name, json.loads(data_text)))
    return events


HEADERS = {"Authorization": "Bearer test-secret"}


def _stream_payload(**overrides):
    base = {
        "model": "test-model",
        "input": "Hi",
        "stream": True,
        "max_output_tokens": 32,
    }
    base.update(overrides)
    return base


def _non_stream_payload(**overrides):
    base = {"model": "test-model", "input": "Hi", "max_output_tokens": 32}
    base.update(overrides)
    return base


# =============================================================================
# R11-B (R11-M-F1) — streaming emits a reasoning item under max_output_tokens
# =============================================================================


class TestStreamingBudgetExhaustEmitsReasoningItem:
    def test_reasoning_output_item_added_under_length_cutoff(self, responses_client):
        """Pre-fix the streaming path shipped ONLY ``response.created`` +
        ``response.in_progress`` + ``response.completed`` with ``output:[]``
        when ``</think>`` never closed within budget. Post-fix the
        accumulated reasoning bytes ship as a ``reasoning`` output item via
        the ``output_item.added`` → ``output_item.done`` event pair."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        names = [n for n, _ in events]

        # The reasoning item must be emitted via both lifecycle events.
        assert "response.output_item.added" in names, (
            f"reasoning ``output_item.added`` missing under max_output_tokens "
            f"cutoff. Events seen: {names}"
        )
        assert "response.output_item.done" in names, (
            f"reasoning ``output_item.done`` missing under max_output_tokens "
            f"cutoff. Events seen: {names}"
        )

        # Locate the added/done pair and verify the item shape.
        added = next(d for n, d in events if n == "response.output_item.added")
        done = next(d for n, d in events if n == "response.output_item.done")
        assert added["item"]["type"] == "reasoning"
        assert done["item"]["type"] == "reasoning"
        # Shared item id across added/done (SDK contract).
        assert added["item"]["id"] == done["item"]["id"]
        assert added["item"]["id"].startswith("rs_")

    def test_reasoning_item_done_status_is_incomplete(self, responses_client):
        """The reasoning item itself flips to ``status:"incomplete"`` when
        the model was still mid-think when its budget ran out — the text
        we ship is partial, flagging it as such matches the non-stream
        surface's ``_build_reasoning_output_item(status="incomplete")``
        invocation."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        done = next(d for n, d in events if n == "response.output_item.done")
        assert done["item"]["status"] == "incomplete"

    def test_completed_status_is_incomplete_with_max_output_tokens_reason(
        self, responses_client
    ):
        """Pre-fix ``response.completed`` reported ``status:"completed"``
        regardless of the underlying truncation. Post-fix:
        ``finish_reason="length"`` → ``status:"incomplete"`` +
        ``incomplete_details:{reason:"max_output_tokens"}``."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        assert completed["response"]["status"] == "incomplete"
        assert completed["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }

    def test_completed_output_array_carries_reasoning_item(self, responses_client):
        """The terminal ``response.completed.response.output[]`` array
        must include the reasoning item so SDK consumers reading the
        Response object see the same shape as the per-event walk. Pre-fix
        ``output:[]`` shipped here."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        output = completed["response"]["output"]
        assert len(output) >= 1, f"completed.output is empty: {output}"
        # Reasoning item is FIRST in ``output[]`` (mirrors the non-stream
        # ``openai_to_responses`` ordering — ``reasoning`` precedes
        # ``message`` in spec order).
        assert output[0]["type"] == "reasoning"
        assert output[0]["status"] == "incomplete"
        assert output[0]["summary"], "summary[] is empty"
        summary_text = output[0]["summary"][0]["text"]
        # The accumulated reasoning text is non-empty AND not duplicated.
        # The expected payload is the concatenation of the engine's
        # reasoning chunks (no leading ``\n`` from the chat template
        # since the test engine emits chunks without the template
        # injecting one into ``raw_text``).
        expected_reasoning = "".join(_REASONING_CHUNKS)
        assert summary_text == expected_reasoning, (
            f"summary text mismatch:\n  got: {summary_text!r}\n  "
            f"expected: {expected_reasoning!r}"
        )
        # Pre-fix duplication regression guard: the reasoning text must
        # appear EXACTLY ONCE — the original bug had finalize and the
        # in-loop accumulator both contributing the same bytes.
        assert summary_text.count("Okay, the user said") == 1, (
            f"reasoning text duplicated:\n  {summary_text!r}"
        )

    def test_reasoning_tokens_credited_under_cutoff(self, responses_client):
        """``usage.output_tokens_details.reasoning_tokens`` MUST be > 0
        when reasoning bytes were accumulated. Pre-fix the streaming path
        always reported ``reasoning_tokens=0`` because the reasoning
        accumulator didn't exist."""
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            body = "".join(resp.iter_text())
        events = _parse_sse(body)
        completed = next(d for n, d in events if n == "response.completed")
        usage = completed["response"]["usage"]
        reasoning_tokens = usage["output_tokens_details"]["reasoning_tokens"]
        assert reasoning_tokens > 0, (
            f"reasoning_tokens={reasoning_tokens} but reasoning text was "
            f"accumulated — usage credit missing."
        )


# =============================================================================
# Cross-path parity — streaming + non-streaming converge on the same shape
# =============================================================================


class TestStreamingNonStreamingParity:
    def test_same_status_and_incomplete_details(self, responses_client):
        """Stream and non-stream on the same prompt + same
        ``max_output_tokens`` must report the same ``status`` and the
        same ``incomplete_details`` block."""
        # Non-streaming call
        non_stream_resp = responses_client.client.post(
            "/v1/responses",
            json=_non_stream_payload(),
            headers=HEADERS,
        )
        assert non_stream_resp.status_code == 200
        non_stream_body = non_stream_resp.json()

        # Streaming call
        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            stream_body = "".join(resp.iter_text())
        stream_events = _parse_sse(stream_body)
        stream_completed = next(
            d for n, d in stream_events if n == "response.completed"
        )

        assert non_stream_body["status"] == "incomplete"
        assert stream_completed["response"]["status"] == "incomplete"
        assert non_stream_body["status"] == stream_completed["response"]["status"]

        assert non_stream_body.get("incomplete_details") == {
            "reason": "max_output_tokens"
        }
        assert stream_completed["response"].get("incomplete_details") == {
            "reason": "max_output_tokens"
        }

    def test_same_output_shape_under_cutoff(self, responses_client):
        """Both surfaces must ship ``output[0].type == "reasoning"`` with
        the same item ``status``. Pre-fix the streaming surface shipped
        ``output:[]`` here — exactly the cross-path asymmetry R11-M-F1
        documented."""
        non_stream_resp = responses_client.client.post(
            "/v1/responses",
            json=_non_stream_payload(),
            headers=HEADERS,
        )
        non_stream_body = non_stream_resp.json()

        with responses_client.client.stream(
            "POST",
            "/v1/responses",
            json=_stream_payload(),
            headers=HEADERS,
        ) as resp:
            stream_body = "".join(resp.iter_text())
        stream_events = _parse_sse(stream_body)
        stream_completed = next(
            d for n, d in stream_events if n == "response.completed"
        )
        stream_output = stream_completed["response"]["output"]
        non_stream_output = non_stream_body["output"]

        assert len(non_stream_output) >= 1
        assert len(stream_output) >= 1
        assert non_stream_output[0]["type"] == "reasoning"
        assert stream_output[0]["type"] == "reasoning"
        assert non_stream_output[0]["status"] == stream_output[0]["status"]
        assert non_stream_output[0]["status"] == "incomplete"
