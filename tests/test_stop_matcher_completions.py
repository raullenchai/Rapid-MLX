# SPDX-License-Identifier: Apache-2.0
"""H-03 regression: ``/v1/completions`` must honour ``stop``.

The systematic stop-matcher landed in #716 lives at the scheduler layer
(``Scheduler._process_batch_responses`` /
``MLLMScheduler._process_batch_responses``) and is exercised by
``tests/test_stop_string_enforcement.py`` already. What the user-reported
H-03 surfaced is that the OpenAI legacy completions route had no
end-to-end test asserting the wire surface: forward ``stop``, get
``finish_reason="stop"``, payload does NOT contain the matched stop
string.

These tests pin the wire-level contract so a future refactor of the
route (e.g. another spec-parity follow-up) cannot silently re-introduce
the gap that personas Petra (r2) and Sergei (r2) reported. They sit at
the route layer with a stub engine that mimics the scheduler's behaviour
(``stop`` arrives → trimmed text + ``finish_reason="stop"``).

The tests cover:
  * non-stream + stream branches,
  * the multi-alternative ``stop`` list,
  * the empty-list ``stop`` no-op (used by SDKs as the default),
  * end-to-end forwarding to ``engine.generate`` / ``engine.stream_generate``
    so the kwarg never gets dropped at the route boundary.

For the actual stop-trim semantics see
``test_stop_string_enforcement.py`` (scheduler-layer) and
``test_scheduler_stop_decoder_surface.py`` (incremental decoder).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def patched_config():
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


class _StubGenerationOutput:
    """Minimal ``GenerationOutput`` stand-in for non-streaming tests."""

    def __init__(
        self,
        text: str = "ok",
        finish_reason: str = "stop",
        matched_stop: str | None = None,
    ):
        self.text = text
        self.finish_reason = finish_reason
        self.completion_tokens = 1
        self.prompt_tokens = 1
        self.cached_tokens = 0
        self.matched_stop = matched_stop


class _StreamChunk:
    """Minimal streaming chunk mimicking the engine's ``GenerationOutput``."""

    def __init__(
        self,
        new_text: str,
        finished: bool,
        finish_reason: str | None,
        text: str = "",
        matched_stop: str | None = None,
    ):
        self.new_text = new_text
        self.text = text
        self.finished = finished
        self.finish_reason = finish_reason
        self.prompt_tokens = 1
        self.completion_tokens = 1
        self.cached_tokens = 0
        self.matched_stop = matched_stop


def _build_app(patch_cfg, monkeypatch, engine):
    """Mount /v1/completions wired to ``engine``."""
    from vllm_mlx.routes import completions as comp_route

    app = FastAPI()
    app.include_router(comp_route.router)
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)
    monkeypatch.setattr(
        comp_route, "enforce_context_length_for_prompt", lambda *_a, **_kw: None
    )
    return TestClient(app, raise_server_exceptions=False)


class _RecordingEngine:
    """Capture the kwargs passed to ``engine.generate`` / ``stream_generate``.

    Non-streaming returns a stub ``GenerationOutput`` that mimics the
    scheduler's stop-trim result. Streaming yields a short list of chunks
    where the terminal chunk carries ``finish_reason="stop"`` — the
    scheduler-side stop-string trim is exercised in
    ``test_stop_string_enforcement.py``; here we pin the route wiring.
    """

    tokenizer = None

    def __init__(
        self,
        *,
        non_stream_text: str = "before",
        non_stream_matched_stop: str | None = "STOP",
        stream_chunks: list[_StreamChunk] | None = None,
    ):
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self._non_stream_text = non_stream_text
        self._non_stream_matched_stop = non_stream_matched_stop
        self._stream_chunks = stream_chunks or [
            _StreamChunk(new_text="before", finished=False, finish_reason=None),
            _StreamChunk(
                new_text="",
                finished=True,
                finish_reason="stop",
                text="before",
                matched_stop="STOP",
            ),
        ]

    async def generate(self, *_, **kwargs):
        self.generate_calls.append(kwargs)
        return _StubGenerationOutput(
            text=self._non_stream_text,
            finish_reason="stop",
            matched_stop=self._non_stream_matched_stop,
        )

    async def stream_generate(self, *_, **kwargs):
        self.stream_calls.append(kwargs)
        for c in self._stream_chunks:
            yield c


def test_completions_non_stream_forwards_stop_to_engine(patched_config, monkeypatch):
    engine = _RecordingEngine()
    client = _build_app(patched_config, monkeypatch, engine)
    r = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "say before STOP after",
            "max_tokens": 16,
            "stop": ["STOP", "DONE"],
        },
    )
    assert r.status_code == 200, r.text
    assert len(engine.generate_calls) == 1, "engine.generate not called"
    forwarded = engine.generate_calls[0].get("stop")
    assert forwarded == ["STOP", "DONE"], (
        f"`stop` list must reach engine.generate; got {forwarded!r}"
    )
    body = r.json()
    # The engine stub returns trimmed text already; the route must
    # surface ``finish_reason="stop"`` on the choice.
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "STOP" not in body["choices"][0]["text"]


def test_completions_non_stream_omitted_stop_forwards_none(patched_config, monkeypatch):
    """Sanity: omitting ``stop`` must not synthesize an empty list at the route."""
    engine = _RecordingEngine(non_stream_matched_stop=None)
    client = _build_app(patched_config, monkeypatch, engine)
    r = client.post(
        "/v1/completions",
        json={"model": "stub-model", "prompt": "hi", "max_tokens": 4},
    )
    assert r.status_code == 200
    assert engine.generate_calls[0].get("stop") is None, (
        "omitted ``stop`` must forward as None — coercing it to ``[]`` "
        "would mask a future regression where the engine starts treating "
        "an empty list differently from None."
    )


def test_completions_non_stream_empty_stop_list_is_noop(patched_config, monkeypatch):
    """Explicit empty-list ``stop`` is the legacy SDK default — must not 400."""
    engine = _RecordingEngine(non_stream_matched_stop=None)
    client = _build_app(patched_config, monkeypatch, engine)
    r = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "hi",
            "max_tokens": 4,
            "stop": [],
        },
    )
    assert r.status_code == 200
    assert engine.generate_calls[0].get("stop") == []


def test_completions_stream_forwards_stop_to_engine(patched_config, monkeypatch):
    engine = _RecordingEngine()
    client = _build_app(patched_config, monkeypatch, engine)
    with client.stream(
        "POST",
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "say before STOP after",
            "max_tokens": 16,
            "stop": ["STOP"],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                chunks.append(line)

    assert len(engine.stream_calls) == 1
    forwarded = engine.stream_calls[0].get("stop")
    assert forwarded == ["STOP"], (
        f"stream_generate must receive ``stop`` from the route; got {forwarded!r}"
    )
    # The terminal chunk must carry ``finish_reason="stop"`` and the
    # response chunks must not echo the stop marker (scheduler-level
    # trim is responsible — we only pin the wire-shape here).
    assert any('"finish_reason": "stop"' in c for c in chunks), (
        "streaming response must include a chunk with finish_reason='stop'; "
        "Petra's H-03 repro showed finish_reason='length' with the stop "
        "string still in the output."
    )
    joined_text = "".join(c for c in chunks)
    assert "STOP" not in joined_text


def test_completions_stop_multiple_alternatives_forwarded(patched_config, monkeypatch):
    """The full alternation list must reach the engine — not just the first entry.

    Pre-#716 the matcher iterated only on first-match; with multiple
    alternatives a route that silently truncated to ``stop[:1]`` would
    let the second alternative leak through.
    """
    engine = _RecordingEngine()
    client = _build_app(patched_config, monkeypatch, engine)
    r = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "hello",
            "max_tokens": 8,
            "stop": ["\n\n", "def ", "END"],
        },
    )
    assert r.status_code == 200
    assert engine.generate_calls[0].get("stop") == ["\n\n", "def ", "END"]


def test_completions_single_element_stop_list_forwarded(patched_config, monkeypatch):
    """A single-entry ``stop`` list is the canonical SDK shape (the
    Anthropic CLI, OpenAI SDK and Petra's IDE plugin all send the
    list form). Pin that a one-element list still reaches the engine
    intact — pre-#716 it would be silently dropped if the route ever
    coerced the field to ``None`` for "empty-ish" inputs.
    """
    engine = _RecordingEngine()
    client = _build_app(patched_config, monkeypatch, engine)
    r = client.post(
        "/v1/completions",
        json={
            "model": "stub-model",
            "prompt": "hello",
            "max_tokens": 8,
            "stop": ["END"],
        },
    )
    assert r.status_code == 200
    forwarded = engine.generate_calls[0].get("stop")
    assert forwarded == ["END"], (
        f"single-element `stop` must reach the engine verbatim; got {forwarded!r}"
    )
