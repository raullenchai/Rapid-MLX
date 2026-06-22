# SPDX-License-Identifier: Apache-2.0
"""
r5-A bundle — Anthropic /v1/messages stream finalize regressions.

Covers four wire-shape bugs that landed together in the streaming
finalize / event-builder layer:

* C-08 CRIT — phantom text re-encoding the thinking buffer on
  ``stream + thinking + max_tokens`` truncation. The route called
  ``reasoning_parser.finalize_streaming(accumulated_raw)`` at
  end-of-stream and emitted the parser's Case-2 "no ``</think>``
  seen — fall back to content" return value as a fresh text content
  block, duplicating bytes already shipped as ``thinking_delta``.

* R-06 HIGH — terminal ``message_delta.stop_reason`` hard-coded to
  ``end_turn`` even when the engine reported ``finish_reason="length"``.
  Anthropic spec requires ``max_tokens`` so SDK helpers know whether to
  re-issue a continuation.

* R-07 HIGH — ``/no_think`` + ``max_tokens`` exhausted by the
  suppressed-thinking budget produced ``message_start → message_delta
  → message_stop`` with ZERO ``content_block_*`` events while billing
  non-zero ``output_tokens``. SDKs surface that as an empty Message
  with no valid content block.

* R-08 HIGH — Computer-Use streaming emitted the entire serialized
  tool input as ONE ``input_json_delta`` event instead of the
  spec-required progressive fragments.

All four are repro'd via the in-process FastAPI ``TestClient`` against
``vllm_mlx.routes.anthropic.router`` so the SSE wire bytes the SDK
would see are exercised end-to-end.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.routes.anthropic import (
    _split_tool_input_json,
    router,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Stub whose ``chat_template`` triggers ``_starts_thinking=True``."""

    chat_template = "{% if add_generation_prompt %}<think>\n{% endif %}"


class _PlainTokenizer:
    """Stub for the no-implicit-thinking branches (tool_use, /no_think)."""

    chat_template = ""


def _parse_sse(text: str) -> list[tuple[str | None, dict]]:
    events: list[tuple[str | None, dict]] = []
    for raw in text.split("\n\n"):
        evt_name: str | None = None
        data: dict | None = None
        for line in raw.splitlines():
            if line.startswith("event: "):
                evt_name = line[len("event: ") :]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    data = None
        if data is not None:
            events.append((evt_name, data))
    return events


def _make_client(engine, *, reasoning_parser_name: str | None = None) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.reasoning_parser_name = reasoning_parser_name
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.tool_call_parser = None
    cfg.enable_auto_tool_choice = False
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Engine stubs
# ---------------------------------------------------------------------------


class _ThinkingTruncatedEngine:
    """Yield qwen3-shaped raw reasoning tokens then truncate on length.

    Mirrors the dogfood repro from ``mei-r1.md`` C-08: the chat
    template prefixes ``<think>`` so the parser is mid-think, the
    model emits a few tokens, and ``max_tokens`` fires before
    ``</think>`` is produced. With the original code path, the
    finalize stage re-emits the buffer as a phantom text block.
    """

    preserve_native_tool_format = False
    tokenizer = _Tokenizer()

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    async def stream_chat(self, messages, **kwargs):
        for i, t in enumerate(self._deltas, start=1):
            yield SimpleNamespace(
                new_text=t,
                prompt_tokens=14,
                completion_tokens=i,
                finish_reason=None,
            )
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=14,
            completion_tokens=len(self._deltas),
            finish_reason="length",
            finished=True,
        )


class _NoThinkExhaustedEngine:
    """Yield empty deltas (suppressed thinking burned the budget) then length finish.

    Repro for R-07: ``/no_think`` + tiny ``max_tokens`` so the entire
    budget is consumed by reasoning that the parser-bypass discards,
    producing no visible content but a non-zero ``output_tokens``.
    """

    preserve_native_tool_format = False
    tokenizer = _Tokenizer()

    async def stream_chat(self, messages, **kwargs):
        for i in range(3):
            yield SimpleNamespace(
                new_text="",
                prompt_tokens=14,
                completion_tokens=i + 1,
                finish_reason=None,
            )
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=14,
            completion_tokens=3,
            finish_reason="length",
            finished=True,
        )


class _ToolCallEngine:
    """Yield a single structured tool_call so the route's tool-use path runs.

    The tool_calls payload follows the HarmonyStreamingRouter shape
    (flat ``{"name", "arguments"}`` dicts) that
    ``_parse_tool_calls_with_parser`` consumes directly.
    """

    preserve_native_tool_format = False
    tokenizer = _PlainTokenizer()

    def __init__(self, tool_input: dict, finish_reason: str = "tool_calls"):
        self._args = json.dumps(tool_input)
        self._finish_reason = finish_reason

    async def stream_chat(self, messages, **kwargs):
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=20,
            completion_tokens=8,
            finish_reason=None,
            tool_calls=[
                {
                    "id": "call_test_1",
                    "name": "computer",
                    "arguments": self._args,
                }
            ],
            channel=None,
        )
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=20,
            completion_tokens=8,
            finish_reason=self._finish_reason,
            finished=True,
            tool_calls=None,
        )


class _NaturalStopEngine:
    """Yield content that closes ``</think>`` and finishes with ``stop``.

    Guard against the C-08 fix over-suppressing — a stream that
    legitimately closed thinking and produced text must keep
    ``stop_reason="end_turn"`` and surface both blocks.
    """

    preserve_native_tool_format = False
    tokenizer = _Tokenizer()

    async def stream_chat(self, messages, **kwargs):
        for i, t in enumerate(["thinking-bytes", "</think>", "final answer"], start=1):
            yield SimpleNamespace(
                new_text=t,
                prompt_tokens=14,
                completion_tokens=i,
                finish_reason=None,
            )
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=14,
            completion_tokens=3,
            finish_reason="stop",
            finished=True,
        )


class _NoTagsLengthTruncatedEngine:
    """Yield plain content (no think tags) and truncate with ``length``.

    Codex r1 REQUIRED #2 — the dangerous over-suppression case for
    C-08. ``_FakeTokenizer`` (template prefixes ``<think>``) is
    NOT used here; the tokenizer has NO ``<think>`` opener in its
    template. The qwen3 parser's implicit-think heuristic still
    conservatively buckets the no-tag bytes as reasoning during
    streaming, and the corrective ``finalize_streaming`` pass is
    the ONLY place those bytes get promoted to text. Truncation by
    ``max_tokens`` must NOT block that correction — the C-08
    suppression predicate has to discriminate "thinking block
    actually opened" (template/model emitted ``<think>``) from
    "parser was being conservative on no-tag plain content".
    """

    preserve_native_tool_format = False
    # Critical: ``chat_template`` has NO ``<think>`` opener so
    # ``_should_start_in_thinking`` returns False — i.e. no
    # template-injected thinking block was ever opened.
    tokenizer = _PlainTokenizer()

    async def stream_chat(self, messages, **kwargs):
        deltas = ["1", ", ", "2"]
        for i, t in enumerate(deltas, start=1):
            yield SimpleNamespace(
                new_text=t,
                prompt_tokens=10,
                completion_tokens=i,
                finish_reason=None,
            )
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=10,
            completion_tokens=len(deltas),
            finish_reason="length",
            finished=True,
        )


class _ZeroOutputEngine:
    """Yield no deltas and finish with ``stop`` and zero output tokens.

    Codex r1 REQUIRED #1 — guards the R-07 synthesis from over-
    firing on benign empty completions (a legitimately-empty
    response e.g. ``max_tokens=0`` returning nothing). The fix
    predicate requires non-zero ``completion_tokens`` AND
    ``finish_reason="length"``; this engine carries neither, so
    no synthetic content_block may be emitted.
    """

    preserve_native_tool_format = False
    tokenizer = _PlainTokenizer()

    async def stream_chat(self, messages, **kwargs):
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=5,
            completion_tokens=0,
            finish_reason="stop",
            finished=True,
        )


# ---------------------------------------------------------------------------
# C-08 — phantom text suppression on truncation mid-think
# ---------------------------------------------------------------------------


def test_c08_vibethinker_preamble_length_truncation_no_duplicate_emit():
    """C-08 byte-faithful predicate covers VibeThinker's preamble-
    before-tag pattern. Codex r2 REQUIRED specifically called out
    this case: the parser streams a chatty multi-sentence preamble
    as ``thinking_delta`` BEFORE ``<think>`` ever appears, so the
    tag-heuristic version of the C-08 fix
    (``"<think>" in accumulated_raw``) would let the duplicate
    re-emit through. Byte-faithful comparison catches it.

    The mocked stream emits a ~30-char preamble (well under the
    VibeThinker 1024-char threshold) then gets truncated by length.
    With the VibeThinker parser active, those bytes stream as
    reasoning and ``finalize_streaming`` Case-1 (no tags AND short)
    returns the same bytes as content. The fixed predicate
    suppresses the duplicate emit.
    """
    # VibeThinker is registered under the ``vibethinker`` parser
    # name; if the registry doesn't surface it on this build, fall
    # back to the deepseek_r1 base parser (which exhibits the same
    # short-no-tag finalize correction shape).
    from vllm_mlx.reasoning import get_parser

    try:
        get_parser("vibethinker")
        parser_name = "vibethinker"
    except Exception:
        parser_name = "deepseek_r1"

    engine = _NoTagsLengthTruncatedEngine()
    client = _make_client(engine, reasoning_parser_name=parser_name)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 4,
            "stream": True,
            "messages": [{"role": "user", "content": "ignored"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    thinking_joined = "".join(
        d["delta"]["thinking"]
        for _, d in events
        if d.get("type") == "content_block_delta"
        and d.get("delta", {}).get("type") == "thinking_delta"
    )
    text_joined = "".join(
        d["delta"]["text"]
        for _, d in events
        if d.get("type") == "content_block_delta"
        and d.get("delta", {}).get("type") == "text_delta"
    )
    # Codex r2 NIT — guard against the vacuous-pass failure mode the
    # earlier draft had. The deepseek_r1 parser MUST route these
    # under-threshold no-tag bytes through the thinking channel
    # during streaming (its base ``extract_reasoning_streaming``
    # returns reasoning while ``_saw_any_tag`` is False and the
    # accumulated text is below ``NO_TAG_CONTENT_THRESHOLD``). If
    # the parser registry surfaces a variant that routes directly
    # to text, the test would silently pass without exercising the
    # C-08 path — assert the precondition is met.
    assert thinking_joined, (
        f"precondition failure: parser {parser_name!r} did not route any "
        f"bytes through the thinking channel during streaming, so the "
        f"C-08 duplicate-suppression branch is not actually exercised "
        f"by this test"
    )
    # And the wire MUST carry these bytes exactly once.
    assert thinking_joined not in text_joined, (
        "VibeThinker / DeepSeek-R1 preamble-before-tag length truncation: "
        "byte-faithful duplicate detected — the C-08 predicate must catch "
        "this codex r2 REQUIRED case"
    )


def test_c08_implicit_think_length_truncation_no_duplicate_emit():
    """C-08 byte-faithful predicate — when the chunk loop already
    shipped bytes as ``thinking_delta`` AND the engine truncated
    with ``length``, the finalize ``content`` correction MUST be
    suppressed regardless of which parser produced it.

    Codex r2 REQUIRED — VibeThinker / DeepSeek-R1 preamble parsers
    stream reasoning BEFORE the literal ``<think>`` opener; a length-
    truncated preamble has the SAME shape this test exercises (no
    ``<think>`` in buffer, thinking already on the wire, ``length``
    finish). Both the C-08 mei scenario AND the preamble scenario
    must converge on "do not duplicate bytes". The codex r1
    discussion of "implicit-think conservative bucketing" framed the
    correction as RIGHT, but its corrective text block is byte-equal
    to what was already shipped — emitting it would re-introduce the
    same data-duplication failure mode C-08 closes.

    The natural-stop path (``finish_reason="stop"``, exercised by the
    existing ``test_no_think_tags_yields_text_delta``) is unaffected
    because the suppression predicate gates on
    ``stream_finish_reason == "length"``.
    """
    engine = _NoTagsLengthTruncatedEngine()
    client = _make_client(engine, reasoning_parser_name="qwen3")
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 8,
            "stream": True,
            "messages": [{"role": "user", "content": "count 1 2"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    thinking_chunks = [
        d["delta"]["thinking"]
        for _, d in events
        if d.get("type") == "content_block_delta"
        and d.get("delta", {}).get("type") == "thinking_delta"
    ]
    text_chunks = [
        d["delta"]["text"]
        for _, d in events
        if d.get("type") == "content_block_delta"
        and d.get("delta", {}).get("type") == "text_delta"
    ]
    thinking_joined = "".join(thinking_chunks)
    text_joined = "".join(text_chunks)
    # The wire MUST carry these bytes exactly once. The chunk loop
    # already shipped them as thinking_delta — emitting an
    # additional text block with the same bytes is the data-
    # duplication regression. If a future refactor wants to
    # re-classify those bytes, the right place is at the streaming
    # surface (suppress the thinking_delta in the first place), NOT
    # to emit a corrective text delta on top of the thinking ones.
    if thinking_joined:
        assert thinking_joined not in text_joined, (
            "byte-faithful duplicate detected: finalize correction re-encoded "
            "already-streamed thinking as text"
        )
    # And on the truncated path the message_delta carries the
    # correct stop_reason so the client knows what happened.
    message_delta = next(d for _, d in events if d.get("type") == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens", message_delta


def test_c08_special_tokens_in_thinking_dont_break_duplicate_detection():
    """C-08 byte-faithful predicate when thinking deltas contain
    special tokens. Codex r3 REQUIRED #1 worried that the
    streaming-side accumulator and the finalize probe might
    normalize differently. They do not in current code: every
    in-loop emission site runs ``strip_special_tokens`` on the
    bytes that eventually reach ``streamed_thinking_text``, and
    the probe normalizes its content via the same helper before
    the comparison. This test pins that invariant.

    Engine delta carries a leading special token (``<|endoftext|>``)
    inside the reasoning bytes; both sides strip it identically, so
    the equality check fires and the duplicate is suppressed.
    """
    # The qwen3 parser's streaming branch hands deltas through
    # ``strip_special_tokens`` before they become ``thinking``
    # pieces (see ``vllm_mlx/routes/anthropic.py`` lines 2051 +
    # 2170). The same helper runs on the finalize probe content.
    # As long as that symmetry holds, special-token-laced thinking
    # streams converge on equality and get suppressed.
    engine = _ThinkingTruncatedEngine(["<|endoftext|>Okay", ",", " the"])
    client = _make_client(engine, reasoning_parser_name="qwen3")
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 3,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    text_chunks = [
        d["delta"]["text"]
        for _, d in events
        if d.get("type") == "content_block_delta"
        and d.get("delta", {}).get("type") == "text_delta"
    ]
    assert text_chunks == [], (
        "phantom text emitted despite special-token-laced thinking — "
        "the streaming accumulator and the finalize probe must remain "
        "byte-comparable through strip_special_tokens"
    )


def test_c08_no_phantom_text_block_on_thinking_max_tokens_truncation():
    """Stream + thinking + max_tokens must NOT re-encode the thinking
    buffer as a fresh text content_block.

    Pre-fix wire trace (the bug):
      message_start
      content_block_start(thinking)
      thinking_delta * N
      content_block_stop
      content_block_start(text)         <-- phantom
      text_delta "<all-thinking-bytes>" <-- duplicate of above
      content_block_stop
      message_delta (end_turn)
      message_stop

    Post-fix:
      message_start
      content_block_start(thinking)
      thinking_delta * N
      content_block_stop
      message_delta (max_tokens)
      message_stop
    """
    engine = _ThinkingTruncatedEngine(["Okay", ",", " the"])
    client = _make_client(engine, reasoning_parser_name="qwen3")
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 3,
            "stream": True,
            "messages": [{"role": "user", "content": "Recite A B C..."}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    block_types: list[str] = []
    thinking_chunks: list[str] = []
    text_chunks: list[str] = []
    for _, d in events:
        t = d.get("type")
        if t == "content_block_start":
            block_types.append(d["content_block"]["type"])
        elif t == "content_block_delta":
            delta = d["delta"]
            if delta.get("type") == "thinking_delta":
                thinking_chunks.append(delta.get("thinking", ""))
            elif delta.get("type") == "text_delta":
                text_chunks.append(delta.get("text", ""))
    assert thinking_chunks == ["Okay", ",", " the"], (
        "thinking bytes must reach the wire as thinking_delta events"
    )
    assert text_chunks == [], (
        "no text_delta may be emitted — phantom re-encoding regression"
    )
    assert "text" not in block_types, (
        "no fresh text content_block may open after the thinking close"
    )


# ---------------------------------------------------------------------------
# R-06 — stop_reason honors engine finish_reason
# ---------------------------------------------------------------------------


def test_r06_stop_reason_max_tokens_on_length_finish():
    """``finish_reason="length"`` must surface as
    ``message_delta.stop_reason="max_tokens"`` (Anthropic public spec).
    """
    engine = _ThinkingTruncatedEngine(["Okay", ",", " the"])
    client = _make_client(engine, reasoning_parser_name="qwen3")
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 3,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    message_delta = next(d for _, d in events if d.get("type") == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens", message_delta


def test_r06_stop_reason_end_turn_on_natural_finish():
    """``finish_reason="stop"`` (no tool_calls, no matched_stop) must
    remain ``end_turn`` so the existing contract holds for natural
    completions.
    """
    engine = _NaturalStopEngine()
    client = _make_client(engine, reasoning_parser_name="qwen3")
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    message_delta = next(d for _, d in events if d.get("type") == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn", message_delta


def test_r06_stop_reason_tool_use_wins_over_length():
    """When tool_calls survive enforcement, ``stop_reason="tool_use"``
    wins regardless of the engine's ``finish_reason`` (mutually-exclusive
    per the Anthropic spec). Prevents a regression where a future
    refactor swaps the precedence and reports ``max_tokens`` for a
    completed tool_use.

    Codex r5 [BLOCKING]: the engine MUST report
    ``finish_reason="length"`` to actually exercise the "tool_use wins
    over length" branch. An earlier draft used the tool_calls-vocab
    engine finish, which doesn't reach the length precedence check and
    would silently pass even if the precedence were inverted.
    """
    engine = _ToolCallEngine({"x": 500, "y": 300}, finish_reason="length")
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "stream": True,
            "tools": [
                {
                    "name": "computer",
                    "description": "Click",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                        },
                        "required": ["x", "y"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Click 500,300"}],
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    message_delta = next(d for _, d in events if d.get("type") == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use", message_delta


# ---------------------------------------------------------------------------
# R-07 — synthetic empty content_block when budget eats only thinking
# ---------------------------------------------------------------------------


def test_r07_does_not_synthesize_for_benign_empty_completion():
    """R-07 synthesis must NOT fire when the engine produced a
    legitimately-empty completion (zero tokens, natural finish).
    Codex r1 REQUIRED #1: pre-fix the predicate was too broad and
    fabricated a content_block for streams that had no business
    carrying one. Codex r2 NIT: use an accepted ``max_tokens`` (the
    earlier draft used ``0`` which some validation paths reject
    upstream, letting the test pass without exercising the
    predicate). ``max_tokens=1`` reaches the streaming helper and
    the engine stub still surfaces zero output + a natural
    ``finish_reason="stop"`` so the synthesis predicate's
    ``completion_tokens > 0 AND finish_reason="length"`` guards both
    fire as no-ops.
    """
    engine = _ZeroOutputEngine()
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 1,
            "stream": True,
            "messages": [{"role": "user", "content": "ignored"}],
        },
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    block_starts = [d for _, d in events if d.get("type") == "content_block_start"]
    assert block_starts == [], (
        "no synthetic content_block_start may be emitted for a "
        "benign zero-output completion"
    )
    # And the stop_reason MUST stay end_turn for natural finishes
    # (R-06 guard symmetric with the synthesis predicate above).
    message_delta = next(d for _, d in events if d.get("type") == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn", message_delta


def test_r07_empty_content_block_when_budget_exhausted_by_suppressed_thinking():
    """``/no_think`` + tiny ``max_tokens`` must NOT emit a Message with
    an empty content list while billing non-zero ``output_tokens``.

    Pre-fix:
      message_start
      message_delta (end_turn, output_tokens=3)
      message_stop

    Post-fix:
      message_start
      content_block_start(text, text="")
      content_block_stop
      message_delta (max_tokens, output_tokens=3)
      message_stop
    """
    engine = _NoThinkExhaustedEngine()
    client = _make_client(engine, reasoning_parser_name="qwen3")
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 3,
            "stream": True,
            "messages": [{"role": "user", "content": "Hello /no_think"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    block_starts = [d for _, d in events if d.get("type") == "content_block_start"]
    block_stops = [d for _, d in events if d.get("type") == "content_block_stop"]
    message_delta = next(d for _, d in events if d.get("type") == "message_delta")
    assert len(block_starts) == 1, block_starts
    assert block_starts[0]["content_block"] == {"type": "text", "text": ""}
    assert len(block_stops) == 1, block_stops
    # And the stop_reason carries the real signal a continuation
    # consumer needs (covers R-06 on the R-07 path).
    assert message_delta["delta"]["stop_reason"] == "max_tokens", message_delta
    assert message_delta["usage"]["output_tokens"] == 3, message_delta


# ---------------------------------------------------------------------------
# R-08 — progressive input_json_delta
# ---------------------------------------------------------------------------


def test_r08_input_json_delta_progressive_on_tool_use():
    """``input_json_delta`` must arrive as multiple structurally-meaningful
    fragments (not one monolithic shard) so consumers parsing-as-they-go
    receive incremental signals.
    """
    tool_input = {"x": 500, "y": 300}
    engine = _ToolCallEngine(tool_input)
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "stream": True,
            "tools": [
                {
                    "name": "computer",
                    "description": "Click",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                        },
                        "required": ["x", "y"],
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Click 500,300"}],
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    fragments: list[str] = []
    for _, d in events:
        if (
            d.get("type") == "content_block_delta"
            and d.get("delta", {}).get("type") == "input_json_delta"
        ):
            fragments.append(d["delta"]["partial_json"])
    assert len(fragments) >= 2, fragments
    # Concatenation must equal ``json.dumps(tool_input)`` so a
    # consumer accumulating fragments ends up with byte-equivalent
    # bytes to the monolithic encoding.
    assert "".join(fragments) == json.dumps(tool_input), fragments


def test_r08_input_json_delta_empty_input_emits_one_fragment():
    """Tool calls with empty ``input={}`` (the Computer-Use synthesis
    fallback that Mei caught) still emit AT LEAST one
    ``input_json_delta`` so the wire contract (every tool block has a
    delta) holds.
    """
    engine = _ToolCallEngine({})
    client = _make_client(engine)
    r = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "stream": True,
            "tools": [
                {
                    "name": "computer",
                    "description": "Click",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Click somewhere"}],
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    fragments = [
        d["delta"]["partial_json"]
        for _, d in events
        if d.get("type") == "content_block_delta"
        and d.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert fragments == ["{}"], fragments


# ---------------------------------------------------------------------------
# Helper coverage — _split_tool_input_json byte-equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        {},
        {"x": 500, "y": 300},
        {"a": [1, 2, 3], "b": {"nested": True}},
        {"k": "comma, and: colon"},
        {"unicode": "héllo", "emoji": "✨"},
        None,
        "not-a-dict",
        # Codex r1 NIT #3: non-string-keyed dicts (the
        # ``json.dumps`` coercion-to-string path) must still
        # round-trip byte-equivalent via the monolithic fallback so a
        # mis-typed tool schema doesn't silently corrupt the wire.
        {1: "x"},
        {True: "y"},
    ],
)
def test_split_tool_input_json_is_byte_equivalent(tool_input):
    """Concatenated fragments must equal ``json.dumps(tool_input)`` for
    every supported shape — clients accumulating ``partial_json`` end up
    with the same buffer the monolithic encoding would have produced.
    """
    fragments = _split_tool_input_json(tool_input)
    assert isinstance(fragments, list) and fragments
    assert "".join(fragments) == json.dumps(tool_input)
