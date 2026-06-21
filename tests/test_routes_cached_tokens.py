# SPDX-License-Identifier: Apache-2.0
"""Route-level integration tests for the prefix-cache reporting field.

The helper-level unit tests in ``test_server_utils.py`` and
``test_anthropic_adapter.py`` cover ``_build_usage`` / ``get_usage``
and the Anthropic adapter mapping in isolation, but they don't exercise
the actual ``/v1/chat/completions`` and ``/v1/completions`` routes.
Without these tests, a refactor that rebuilds the route's usage
assembly (e.g. drops the per-chunk ``cached_tokens`` accumulator in
``routes/chat.py`` or the ``total_cached_tokens`` accumulator in
``routes/completions.py``) silently regresses cache reporting while
the helper tests stay green. The Anthropic streaming counterpart of
this coverage is in ``test_anthropic_route_streaming.py``.
"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.routes.completions import router as completions_router


class _CacheReportingChatEngine:
    """Mock chat engine that reports prefix-cache hits on every output.

    Mirrors the production scheduler's behavior: ``cached_tokens`` is
    set once per request (at schedule time) and stays constant across
    all stream chunks. Returns ``GenerationOutput`` from both ``chat``
    (non-streaming) and ``stream_chat`` (streaming) so the same engine
    instance can drive both code paths.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, *, prompt_tokens: int, cached_tokens: int):
        self._prompt_tokens = prompt_tokens
        self._cached_tokens = cached_tokens

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="hi",
            raw_text="hi",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            cached_tokens=self._cached_tokens,
        )

    async def stream_chat(self, messages, **kwargs):
        deltas = ["hi", " there"]
        accumulated = ""
        for i, delta in enumerate(deltas):
            accumulated += delta
            is_last = i == len(deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                channel=None,
                cached_tokens=self._cached_tokens,
            )


class _CacheReportingCompletionEngine:
    """Mock completion engine. Returns a single ``GenerationOutput``
    from ``generate``; the ``/v1/completions`` route loops over the
    request's prompts and accumulates token counts across each.
    Also implements ``stream_generate`` for the SSE streaming path
    so the wire-shape regression tests can assert on the final chunk.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, *, prompt_tokens: int, cached_tokens: int):
        self._prompt_tokens = prompt_tokens
        self._cached_tokens = cached_tokens

    async def generate(self, prompt, **kwargs):
        return GenerationOutput(
            text="answer",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            cached_tokens=self._cached_tokens,
        )

    async def stream_generate(self, prompt, **kwargs):
        deltas = ["ans", "wer"]
        for i, delta in enumerate(deltas):
            is_last = i == len(deltas) - 1
            yield GenerationOutput(
                text="answer"[: (i + 1) * 3],
                new_text=delta,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                cached_tokens=self._cached_tokens,
            )


def _make_chat_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _make_completions_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(completions_router)
    return TestClient(app)


def _parse_sse_events(text: str) -> list[dict]:
    # NOTE: do NOT silently swallow ``JSONDecodeError`` here. The streaming
    # usage-block tests assert on a chunk near the end of the SSE response;
    # if an earlier chunk regresses to invalid JSON, an except-pass loop
    # would skip it and the test could still pass on the trailing valid
    # chunk — masking a real wire-format regression. Surface the parse
    # failure so the suite is the canary it claims to be.
    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


def test_chat_non_streaming_response_carries_cached_tokens():
    """End-to-end: prefix-cache hit at the engine layer surfaces as
    ``usage.prompt_tokens_details.cached_tokens`` on the
    ``/v1/chat/completions`` response. Closes the loop from
    ``GenerationOutput.cached_tokens`` → ``_build_usage`` →
    Pydantic ``Usage`` → JSON wire.
    """
    engine = _CacheReportingChatEngine(prompt_tokens=200, cached_tokens=128)
    client = _make_chat_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["usage"]["prompt_tokens"] == 200
    details = body["usage"].get("prompt_tokens_details")
    assert details is not None, (
        "prompt_tokens_details must appear on the response when the "
        "engine reported a cache hit; got usage="
        f"{body['usage']!r}"
    )
    assert details["cached_tokens"] == 128


def test_chat_streaming_with_include_usage_carries_cached_tokens_in_dedicated_chunk():
    """When ``stream_options.include_usage`` is true, the dedicated
    trailing usage chunk (empty ``choices``, populated ``usage``) must
    surface ``cached_tokens``. This is the primary OpenAI-spec way
    streaming clients consume usage — the per-chunk accumulator in
    ``routes/chat.py`` and the ``_UsageOutput`` namespace fed to
    ``_build_usage`` must both propagate the field correctly.
    """
    engine = _CacheReportingChatEngine(prompt_tokens=200, cached_tokens=128)
    client = _make_chat_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    usage_chunks = [e for e in events if not e.get("choices") and e.get("usage")]
    assert len(usage_chunks) == 1, (
        f"expected exactly one dedicated usage chunk; got {len(usage_chunks)}"
    )
    usage = usage_chunks[0]["usage"]
    details = usage.get("prompt_tokens_details")
    assert details is not None
    assert details["cached_tokens"] == 128


def test_chat_streaming_without_include_usage_omits_usage_block_entirely():
    """D-SSE-USAGE regression: when the caller does NOT pass
    ``stream_options.include_usage=true``, the OpenAI streaming spec
    requires the ``usage`` field (including its ``cached_tokens``
    subfield) to be absent from every SSE chunk. Pre-v0.8.2 the finish
    chunk carried a populated ``usage`` block — encoded here as the
    accommodation for "legacy bare clients" — which LangChain /
    AI-SDK / vercel-ai-stream parsers double-counted. The new contract
    is opt-in: clients that want cached_tokens on the streaming wire
    must pass ``include_usage=true`` and read the dedicated trailing
    chunk (covered by
    ``test_chat_streaming_with_include_usage_carries_cached_tokens_in_dedicated_chunk``).
    """
    engine = _CacheReportingChatEngine(prompt_tokens=200, cached_tokens=128)
    client = _make_chat_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    finish_events = [
        e for e in events for c in e.get("choices", []) if c.get("finish_reason")
    ]
    assert len(finish_events) == 1
    assert finish_events[0].get("usage") is None, (
        "finish chunk MUST NOT carry usage when include_usage is unset"
    )
    any_usage_key = [e for e in events if "usage" in e]
    assert any_usage_key == [], (
        f"no SSE chunk may carry the usage KEY when include_usage is "
        f"unset; got {len(any_usage_key)} chunk(s) with the key "
        f'(includes regressions to ``"usage": null``)'
    )


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


def test_completions_streaming_dedicated_usage_chunk_omits_null_detail_fields():
    """Pin the wire shape of the streaming ``/v1/completions`` dedicated
    usage chunk (emitted only when ``include_usage=true`` per
    D-SSE-USAGE): with no cache hit and no reasoning split, the JSON
    payload's ``usage`` block must NOT carry literal ``null``
    ``prompt_tokens_details`` or ``completion_tokens_details`` keys.
    ``model_dump(exclude_none=True)`` drops them; bare ``model_dump()``
    leaves them in as nulls, which some SDK accumulators trip on.
    Matches the cleanup ``routes/chat.py`` already does for its
    trailing usage chunk and locks the streaming-completions sibling
    so a future revert silently re-adds the null keys.
    """
    engine = _CacheReportingCompletionEngine(prompt_tokens=200, cached_tokens=0)
    client = _make_completions_client(engine)

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 32,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    final_chunks = [e for e in events if e.get("usage")]
    assert len(final_chunks) == 1, (
        f"expected exactly one chunk with usage; got {len(final_chunks)}"
    )
    usage = final_chunks[0]["usage"]
    assert "prompt_tokens_details" not in usage, (
        f"null prompt_tokens_details must be excluded; got {usage!r}"
    )
    assert "completion_tokens_details" not in usage, (
        f"null completion_tokens_details must be excluded; got {usage!r}"
    )
    # Sanity-check that the non-null keys survived the exclude_none.
    assert usage["prompt_tokens"] == 200
    assert usage["total_tokens"] == 202


def test_completions_streaming_dedicated_usage_chunk_includes_cached_tokens_when_hit():
    """Sibling assertion: when there IS a cache hit AND the caller
    opted in via ``stream_options.include_usage=true``, the streaming
    dedicated usage chunk DOES include ``prompt_tokens_details`` with
    the populated count — `exclude_none=True` only suppresses null
    fields, not populated ones.
    """
    engine = _CacheReportingCompletionEngine(prompt_tokens=200, cached_tokens=128)
    client = _make_completions_client(engine)

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 32,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    final_chunks = [e for e in events if e.get("usage")]
    assert len(final_chunks) == 1
    usage = final_chunks[0]["usage"]
    details = usage.get("prompt_tokens_details")
    assert details is not None
    assert details["cached_tokens"] == 128
    # No reasoning split on the completions path → completion_tokens_details
    # stays null and gets excluded.
    assert "completion_tokens_details" not in usage


def test_completions_non_streaming_response_carries_cached_tokens():
    """``/v1/completions`` accumulates ``cached_tokens`` across each
    prompt in the request batch (single prompt = single accumulator
    increment). Exercises the
    ``total_cached_tokens += output.cached_tokens`` line in
    ``routes/completions.py`` and confirms it surfaces on the response
    via the same ``PromptTokensDetails`` shape.
    """
    engine = _CacheReportingCompletionEngine(prompt_tokens=200, cached_tokens=128)
    client = _make_completions_client(engine)

    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 32,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["usage"]["prompt_tokens"] == 200
    details = body["usage"].get("prompt_tokens_details")
    assert details is not None, (
        "prompt_tokens_details must appear on the response when the "
        "engine reported a cache hit; got usage="
        f"{body['usage']!r}"
    )
    assert details["cached_tokens"] == 128


# ---------------------------------------------------------------------------
# Scheduler RequestOutput construction (the source of the field)
# ---------------------------------------------------------------------------


def test_scheduler_request_output_construction_carries_cached_tokens():
    """The single source line that injects the field on the production
    path: ``scheduler.py``'s ``_process_batch_responses`` constructs
    ``RequestOutput(..., cached_tokens=request.cached_tokens, ...)``
    from the ``Request`` whose prefix-cache lookup populated the
    field. The route-level tests above use mock engines that
    short-circuit the scheduler entirely, so without this test a
    silent removal of the source line would leave production
    responses always reporting 0 — yet every test would still pass.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from vllm_mlx.request import Request, SamplingParams
    from vllm_mlx.scheduler import Scheduler

    # Bypass ``__init__`` — it sets up BatchGenerator, prefix cache
    # tiers, detokenizer pool, etc., none of which
    # ``_process_batch_responses`` touches for this test. We populate
    # only the fields the method reads.
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._detokenizer_pool = {}
    scheduler.uid_to_request_id = {1: "req-1"}
    scheduler.running = {}
    # Minimal tokenizer for ``_decode_tokens`` (called on the
    # non-stop path); a no-op decoder is enough since the test
    # asserts only on the cached_tokens field.
    scheduler._actual_tokenizer = SimpleNamespace(
        encode=MagicMock(return_value=[1, 2]),
        decode=MagicMock(return_value="x"),
    )

    # Build a Request with a known cached_tokens count. Attach a
    # minimal decoder so the streaming-decode branch on the non-stop
    # response also works without touching the real IncrementalDecoder
    # machinery.
    request = Request(
        request_id="req-1",
        prompt="hello",
        sampling_params=SamplingParams(),
    )
    request.num_prompt_tokens = 200
    request.cached_tokens = 128
    request._decoder = SimpleNamespace(add_token=MagicMock(return_value="x"))
    scheduler.running["req-1"] = request

    # Fake mlx-lm-style response: uid maps to our request, one new
    # token, non-stop finish reason so ``_process_batch_responses``
    # builds the RequestOutput in the normal token-append path.
    fake_response = SimpleNamespace(
        uid=1,
        token=42,
        finish_reason=None,
        logprobs=None,
    )

    outputs, _ = scheduler._process_batch_responses([fake_response])
    assert len(outputs) == 1, "scheduler should emit one output for one response"
    assert outputs[0].cached_tokens == 128, (
        "scheduler must propagate request.cached_tokens into the "
        "constructed RequestOutput so downstream usage assembly can "
        "report the prefix-cache hit count"
    )
    assert outputs[0].prompt_tokens == 200
    assert outputs[0].request_id == "req-1"
