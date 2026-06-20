# SPDX-License-Identifier: Apache-2.0
"""Regression test for /v1/messages dropping ``stop_sequences``.

Bug: the Anthropic route's ``_resolved_sampling_kwargs`` helper forwarded
``temperature``, ``top_p``, and extended sampling params to the engine but
NOT ``stop`` — so ``stop_sequences`` from the request flowed through
``anthropic_to_openai`` into ``openai_request.stop`` and then died at the
route boundary. Engine ran uncapped, model emitted past the user's stop
tokens. Surfaced by the iter8 onboarding sweep on gpt-oss-20b-mxfp4-q8: same
prompt + ``stop_sequences:["STOPHERE"]`` returned full text including
"STOPHERE", finish_reason=end_turn. Identical prompt via /v1/chat/
completions with ``stop:["STOPHERE"]`` stopped correctly. Fix: include
``stop`` in the kwargs returned by ``_resolved_sampling_kwargs`` so both
the non-stream and stream branches forward it.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.anthropic import router as anthropic_router


class _RecordingEngine:
    """Mock engine that records the kwargs passed to ``chat`` / ``stream_chat``.

    The point of the test isn't whether the model stops — it's that the
    request's ``stop_sequences`` reaches the engine. The previous bug was
    the route dropping the field silently.

    H-03 extension: the stub can carry an arbitrary ``matched_stop`` so
    we can pin the route's translation of the scheduler-pinned matched
    bytes into Anthropic ``stop_reason="stop_sequence"`` +
    ``stop_sequence: <str>``. ``None`` mirrors the legacy behaviour
    (EOS / length / no-stop → ``stop_reason="end_turn"``).
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, matched_stop: str | None = None):
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self._matched_stop = matched_stop

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
            matched_stop=self._matched_stop,
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        yield GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
            matched_stop=self._matched_stop,
        )


def _make_client(engine: _RecordingEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None

    app = FastAPI()
    app.include_router(anthropic_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


def test_stop_sequences_forwarded_to_engine_non_stream():
    engine = _RecordingEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stop_sequences": ["STOPHERE", "ALSO_STOP"],
            "messages": [{"role": "user", "content": "say STOPHERE then more"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(engine.chat_calls) == 1
    forwarded_stop = engine.chat_calls[0]["kwargs"].get("stop")
    assert forwarded_stop == ["STOPHERE", "ALSO_STOP"], (
        f"stop_sequences must be forwarded as ``stop`` to engine.chat(); "
        f"got {forwarded_stop!r}"
    )


def test_stop_sequences_forwarded_to_engine_streaming():
    engine = _RecordingEngine()
    client = _make_client(engine)

    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "say END then more"}],
        },
    ) as resp:
        assert resp.status_code == 200
        # drain the stream
        for _ in resp.iter_lines():
            pass

    assert len(engine.stream_calls) == 1
    forwarded_stop = engine.stream_calls[0]["kwargs"].get("stop")
    assert forwarded_stop == ["END"], (
        f"stop_sequences must be forwarded as ``stop`` to engine.stream_chat(); "
        f"got {forwarded_stop!r}"
    )


def test_stop_sequences_none_when_omitted_non_stream():
    """Sanity: when client omits stop_sequences, we don't synthesize one."""
    engine = _RecordingEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    forwarded_stop = engine.chat_calls[0]["kwargs"].get("stop")
    assert forwarded_stop is None, (
        f"omitted stop_sequences must forward as None; got {forwarded_stop!r}"
    )


# ---------------------------------------------------------------------------
# H-03: matched_stop must surface on the wire
# ---------------------------------------------------------------------------
#
# Persona Sergei (r2) reported that ``messages.create(stop_sequences=["END"]
# )`` returned ``stop_reason="end_turn"`` with ``stop_sequence`` absent —
# even when the scheduler's stop-string trim DID fire. The matcher
# (PR #716) landed at the scheduler layer, but ``/v1/messages`` had no
# wire-level adapter for the Anthropic-specific ``stop_sequence``
# surface, so the engine knew which stop fired but the response never
# said so. The fix plumbs ``GenerationOutput.matched_stop`` through the
# adapter; these tests pin the wire shape on both the non-stream and
# streaming branches.


def test_stop_sequence_surfaced_in_non_stream_response():
    """When the engine reports a matched stop, the response must carry
    ``stop_reason="stop_sequence"`` AND ``stop_sequence: <the matched
    string>``. Persona Sergei's original repro: matched stop "END" →
    pre-fix returned ``stop_reason="end_turn"``, ``stop_sequence`` field
    silently dropped.
    """
    engine = _RecordingEngine(matched_stop="END")
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "say END"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "stop_sequence", (
        f"matched user stop → ``stop_reason='stop_sequence'`` per Anthropic "
        f"spec; got {body['stop_reason']!r}. Pre-fix the adapter mapped "
        f"finish_reason='stop' to 'end_turn' unconditionally."
    )
    assert body["stop_sequence"] == "END", (
        f"the matched stop bytes must surface verbatim in "
        f"``stop_sequence``; got {body.get('stop_sequence')!r}"
    )


def test_stop_sequence_absent_when_no_user_stop_matched():
    """EOS / length / no-stop terminations must still map to ``end_turn``
    with ``stop_sequence: null`` — the H-03 fix must NOT regress the
    default mapping."""
    engine = _RecordingEngine(matched_stop=None)
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stop_reason"] == "end_turn"
    # ``stop_sequence`` is excluded when None via ``exclude_none``; the
    # legacy Anthropic SDK shape allows either omission or explicit
    # null. Accept both.
    assert body.get("stop_sequence") in (None,)


def test_stop_sequence_surfaced_in_streaming_message_delta():
    """The terminal SSE ``message_delta`` must carry the matched stop
    bytes in ``delta.stop_sequence`` and set ``delta.stop_reason`` to
    ``"stop_sequence"``. Pre-fix the streaming branch hard-coded
    ``stop_sequence: None`` even when the engine reported a match.
    """
    engine = _RecordingEngine(matched_stop="END")
    client = _make_client(engine)

    saw_message_delta: list[dict] = []
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "say END"}],
        },
    ) as resp:
        assert resp.status_code == 200
        import json as _json

        data_buffer: list[str] = []
        last_event: str | None = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                last_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and last_event == "message_delta":
                payload = line[len("data:") :].strip()
                if payload:
                    try:
                        saw_message_delta.append(_json.loads(payload))
                    except Exception:
                        data_buffer.append(payload)

    assert saw_message_delta, "stream must include at least one message_delta event"
    delta = saw_message_delta[-1]["delta"]
    assert delta["stop_reason"] == "stop_sequence", (
        f"stream terminal delta must report stop_reason='stop_sequence'; "
        f"got {delta!r}. Pre-fix the streaming branch hard-coded "
        f"stop_reason='end_turn'."
    )
    assert delta["stop_sequence"] == "END", (
        f"the matched stop bytes must surface in delta.stop_sequence; "
        f"got {delta!r}. Pre-fix the streaming branch hard-coded "
        f"stop_sequence: null."
    )


def test_stop_sequence_streaming_falls_back_to_end_turn_when_no_match():
    """Streaming parity for the no-match path: ``stop_reason='end_turn'``
    + ``stop_sequence: null`` (the pre-fix default)."""
    engine = _RecordingEngine(matched_stop=None)
    client = _make_client(engine)

    saw_message_delta: list[dict] = []
    with client.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as resp:
        assert resp.status_code == 200
        import json as _json

        last_event: str | None = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                last_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and last_event == "message_delta":
                payload = line[len("data:") :].strip()
                if payload:
                    saw_message_delta.append(_json.loads(payload))

    assert saw_message_delta
    delta = saw_message_delta[-1]["delta"]
    assert delta["stop_reason"] == "end_turn"
    assert delta["stop_sequence"] is None


def test_stop_sequence_multi_alternative_surfaces_first_match():
    """When ``stop_sequences`` has multiple alternatives, the
    ``stop_sequence`` field must echo whichever bytes the scheduler
    actually matched — not the request's first alternative."""
    engine = _RecordingEngine(matched_stop="ALSO_STOP")
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stop_sequences": ["NEVER_MATCHES", "ALSO_STOP", "WAY_DOWN"],
            "messages": [{"role": "user", "content": "say ALSO_STOP"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == "ALSO_STOP"


def test_stop_sequence_empty_list_does_not_force_stop_sequence_reason():
    """Empty stop_sequences must behave like omitted — the matcher
    cannot fire, so the response must NOT be reclassified to
    ``stop_sequence``."""
    engine = _RecordingEngine(matched_stop=None)
    client = _make_client(engine)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stop_sequences": [],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "end_turn"
