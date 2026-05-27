# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine token-level output routing."""

from collections.abc import AsyncIterator

import pytest

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine


class FakeTokenizer:
    """Minimal tokenizer for OutputRouter detection and decoding."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self._id_to_text = {v: k for k, v in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_text.get(i, f"<UNK:{i}>") for i in ids)


# Harmony vocab IDs mirror the GPT-OSS tokenizer subset used by OutputRouter.
HARMONY_VOCAB = {
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|call|>": 200012,
    "<|endoftext|>": 200019,
    "analysis": 35644,
    "final": 17196,
    "Reason": 2,
    "ing": 3,
    "Answer": 4,
    "Fallback": 5,
}

QWEN3_VOCAB = {
    "<think>": 248068,
    "</think>": 248069,
    "Reason": 2,
    "Answer": 4,
}

GEMMA4_VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<|tool>": 46,
    "<tool|>": 47,
    "<|tool_call>": 48,
    "<tool_call|>": 49,
    "<|tool_response>": 50,
    "<tool_response|>": 51,
    '<|"|>': 52,
    "<|channel>": 100,
    "<channel|>": 101,
    "<|turn>": 105,
    "<turn|>": 106,
    "thought": 45518,
    "content": 3955,
    "final": 10218,
    "call": 6639,
    ":": 236787,
    "get": 828,
    "_": 236779,
    "weather": 19323,
    "{": 236782,
    "}": 236783,
    "city": 13319,
    "Tokyo": 89265,
}


def _make_engine(tokenizer: FakeTokenizer) -> BatchedEngine:
    engine = BatchedEngine("fake-model")
    engine._loaded = True
    engine._tokenizer = tokenizer
    engine._apply_chat_template = lambda *args, **kwargs: "prompt"
    engine._compute_prefix_boundary = lambda *args, **kwargs: 0
    return engine


async def _collect(
    outputs: AsyncIterator[GenerationOutput],
) -> list[GenerationOutput]:
    return [output async for output in outputs]


@pytest.mark.asyncio
async def test_stream_chat_routes_supported_tokenizer_channels():
    """Supported tokenizers emit channel-tagged chunks and suppress controls."""
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<|channel|>analysis<|message|>Reason",
            tokens=[200005, 35644, 200008, 2],
            finished=False,
        )
        yield GenerationOutput(
            text="",
            new_text="ing<|start|><|channel|>final<|message|>Answer",
            tokens=[3, 200006, 200005, 17196, 200008, 4],
            finished=True,
            finish_reason="stop",
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert [(o.new_text, o.channel, o.finished) for o in outputs] == [
        ("Reason", "reasoning", False),
        ("ing", "reasoning", False),
        ("Answer", "content", True),
    ]
    assert all("<|channel|>" not in output.new_text for output in outputs)
    assert all(output.logprobs is None for output in outputs)


@pytest.mark.asyncio
async def test_stream_chat_routed_outputs_preserve_cached_tokens():
    """``_make_routed_output`` + ``_routed_finish_sentinel`` must
    thread ``source.cached_tokens`` onto every routed chunk.
    OutputRouter models (Gemma 4 / harmony / gpt-oss) rebuild each
    emitted ``GenerationOutput`` from scratch — a missed propagation
    here would silently zero out the prefix-cache hit count on the
    routed flush, leaving usage at 0 while the engine actually
    served the prefix from cache. Same failure shape as PR #454 /
    #456 for ``reasoning_tokens`` and ``logprobs``.

    Drives a harmony-style stream with ``cached_tokens=128`` on every
    source chunk and asserts every routed output (content,
    reasoning, finish sentinel) carries the value through.
    """
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<|channel|>analysis<|message|>Reason",
            tokens=[200005, 35644, 200008, 2],
            finished=False,
            cached_tokens=128,
        )
        yield GenerationOutput(
            text="",
            new_text="ing<|start|><|channel|>final<|message|>Answer",
            tokens=[3, 200006, 200005, 17196, 200008, 4],
            finished=True,
            finish_reason="stop",
            cached_tokens=128,
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    # Every emitted routed output (per-token reasoning chunks, the
    # content chunk, AND the finish sentinel) must carry the
    # source's cached_tokens through.
    assert outputs, "router should emit at least one output"
    cached = [o.cached_tokens for o in outputs]
    assert cached == [128] * len(outputs), (
        f"cached_tokens must propagate to every routed output; got {cached!r}"
    )


@pytest.mark.asyncio
async def test_router_propagates_per_token_logprobs_to_routed_outputs():
    """OutputRouter must forward source.logprobs to each routed chunk.

    Regression for the v0.6.66 onboarding sweep finding: ``logprobs=true +
    top_logprobs=5`` requests against harmony/gpt-oss returned HTTP 200
    with no ``logprobs`` field in the response. Root cause: the engine's
    ``_make_routed_output`` hardcoded ``logprobs=None`` on every routed
    chunk, so ``_extract_streaming_token_logprobs`` saw
    ``chunk.logprobs is None`` and returned ``[]`` for every chunk;
    ``token_logprobs_list`` stayed empty and the route built a response
    with ``choice_logprobs=None``. PR #450 fixed a related AttributeError
    on the non-router path but couldn't surface this gap because its
    tests use single-token GenerationOutput stubs that never go through
    the router.
    """
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    sentinel_lps = ["LP-Reason", "LP-ing", "LP-Answer"]

    async def fake_stream_generate(**kwargs):
        # Single-token flushes (stream_interval=1) — exactly the shape
        # OutputRouter hits in production for content/reasoning tokens.
        yield GenerationOutput(
            text="",
            new_text="<|channel|>analysis<|message|>",
            tokens=[200005, 35644, 200008],
            finished=False,
            logprobs=None,  # control marker tokens carry no per-step lp
        )
        yield GenerationOutput(
            text="",
            new_text="Reason",
            tokens=[2],
            finished=False,
            logprobs=sentinel_lps[0],
        )
        yield GenerationOutput(
            text="",
            new_text="ing",
            tokens=[3],
            finished=False,
            logprobs=sentinel_lps[1],
        )
        yield GenerationOutput(
            text="",
            new_text="<|start|><|channel|>final<|message|>",
            tokens=[200006, 200005, 17196, 200008],
            finished=False,
            logprobs=None,
        )
        yield GenerationOutput(
            text="",
            new_text="Answer",
            tokens=[4],
            finished=True,
            finish_reason="stop",
            logprobs=sentinel_lps[2],
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    text_lps = [(o.new_text, o.logprobs) for o in outputs if o.new_text]
    assert text_lps == [
        ("Reason", sentinel_lps[0]),
        ("ing", sentinel_lps[1]),
        ("Answer", sentinel_lps[2]),
    ]


@pytest.mark.asyncio
async def test_router_propagates_logprobs_list_form_per_index():
    """When source.logprobs is a list, each routed token must pair with logprobs[i].

    The ``stream_interval > 1`` path (PR #210) accumulates per-step
    distributions into ``list[mx.array]`` paired with a multi-token
    ``tokens`` list. The router must split them back out so each routed
    chunk carries the correct per-step distribution — passing the whole
    list to every routed chunk would let ``_extract_streaming_token_logprobs``
    re-iterate the full list per chunk and emit duplicate entries.
    """
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<|channel|>final<|message|>",
            tokens=[200005, 17196, 200008],
            finished=False,
            logprobs=["LP-channel", "LP-final", "LP-message"],
        )
        yield GenerationOutput(
            text="",
            new_text="ReasoningAnswer",
            tokens=[2, 4],
            finished=True,
            finish_reason="stop",
            logprobs=["LP-Reason", "LP-Answer"],
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    text_lps = [(o.new_text, o.logprobs) for o in outputs if o.new_text]
    assert text_lps == [
        ("Reason", "LP-Reason"),
        ("Answer", "LP-Answer"),
    ]


@pytest.mark.asyncio
async def test_stream_chat_keeps_think_tag_tokenizers_on_legacy_path():
    """Think-tag routers are detected but not engine-enabled until validated."""
    engine = _make_engine(FakeTokenizer(QWEN3_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<think>Reason</think>Answer",
            tokens=[248068, 2, 248069, 4],
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert len(outputs) == 1
    assert outputs[0].new_text == "<think>Reason</think>Answer"
    assert outputs[0].channel is None


@pytest.mark.asyncio
async def test_stream_chat_routes_tool_call_channel_on_finish():
    """Truncated tool calls are drained as tool_call channel output."""
    engine = _make_engine(FakeTokenizer(GEMMA4_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<|tool_call>call:get_weather{city:Tokyo}",
            tokens=[
                48,
                6639,
                236787,
                828,
                236779,
                19323,
                236782,
                13319,
                236787,
                89265,
                236783,
            ],
            finished=True,
            finish_reason="length",
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert [(o.channel, o.finished, o.finish_reason) for o in outputs] == [
        ("tool_call", True, "length")
    ]
    assert "get_weather" in outputs[0].new_text
    assert "Tokyo" in outputs[0].new_text
    assert outputs[0].logprobs is None


# Router-allowlist families that emit Channel.TOOL_CALL as a *deferred
# aggregate* event (TokenMap has both tool_call_start and tool_call_end
# set; router buffers body tokens during RouterState.TOOL_CALL and emits
# once on the end marker with event.text = full decoded body).
#
# THIS IS THE BUG A SURFACE (v0.6.62): if the engine's single-token-flush
# optimization is applied to these events, the end-marker token's text
# clobbers the buffered body and the streaming tool call is dropped.
#
# Every family in this dict MUST be exercised by the parametrized
# `test_router_tool_call_body_preserved_single_token_flush` below. The
# structural test `test_router_allowlist_tool_call_routing_declared`
# enforces that ALL `_OUTPUT_ROUTER_ALLOWLIST` families are categorized
# either here or in `_ROUTER_FAMILIES_TOOL_CALL_AT_PARSER_LAYER` so that
# the next router-family addition cannot silently ship without coverage.
_ROUTER_FAMILIES_TOOL_CALL_AGGREGATE: dict[str, dict] = {
    "gemma4": {
        "vocab": GEMMA4_VOCAB,
        # <|tool_call>call:get_weather{city:Tokyo}<tool_call|>
        "body_tokens": [
            48,
            6639,
            236787,
            828,
            236779,
            19323,
            236782,
            13319,
            236787,
            89265,
            236783,
            49,
        ],
        "expected_substrings": [
            "<|tool_call>",
            "get_weather",
            "Tokyo",
            "<tool_call|>",
        ],
    },
}

# Router-allowlist families whose TOKENIZER-level routing does NOT emit
# Channel.TOOL_CALL — tool-call extraction happens downstream at the
# per-family ToolParser (e.g. HarmonyToolParser scanning the commentary
# channel). These families are exempt from the aggregate-body invariant
# above; their streaming coverage lives in the per-parser test files.
_ROUTER_FAMILIES_TOOL_CALL_AT_PARSER_LAYER: set[str] = {
    "harmony",  # tool calls routed via <|channel|>commentary + <|call|>;
    # extracted by HarmonyToolParser, not OutputRouter.feed()
}


@pytest.mark.parametrize("family", sorted(_ROUTER_FAMILIES_TOOL_CALL_AGGREGATE.keys()))
@pytest.mark.asyncio
async def test_router_tool_call_body_preserved_single_token_flush(family):
    """Single-token engine flush must not clobber the router's multi-token body.

    Regression: ``Channel.TOOL_CALL`` is a *deferred aggregate* — the router
    silently buffers body tokens during ``RouterState.TOOL_CALL`` and emits
    one event on the end marker with ``event.text`` carrying the full decoded
    body. The single-token-flush optimization that lets CONTENT/REASONING
    chunks reuse the scheduler's detokenized ``output.new_text`` would, if
    applied to TOOL_CALL events, override the accumulated body with just the
    end-marker token's text, silently dropping the call body. Caught
    post-v0.6.61 on gemma-4-26b — non-stream extracted a valid tool call
    from the same generation that streaming returned as bare content.

    Parametrized over every router-allowlist family that emits TOOL_CALL
    aggregate events. Adding a new family to that group (see
    ``_ROUTER_FAMILIES_TOOL_CALL_AGGREGATE``) requires extending this test.
    """
    spec = _ROUTER_FAMILIES_TOOL_CALL_AGGREGATE[family]
    vocab = spec["vocab"]
    body_tokens = spec["body_tokens"]
    expected = spec["expected_substrings"]

    engine = _make_engine(FakeTokenizer(vocab))
    _id_to_text = {v: k for k, v in vocab.items()}

    async def fake_stream_generate(**kwargs):
        for i, tid in enumerate(body_tokens):
            finished = i == len(body_tokens) - 1
            yield GenerationOutput(
                text="",
                new_text=_id_to_text[tid],
                tokens=[tid],
                finished=finished,
                finish_reason="stop" if finished else None,
            )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    tool_call_outputs = [o for o in outputs if o.channel == "tool_call"]
    assert len(tool_call_outputs) == 1, (
        f"{family}: expected exactly 1 TOOL_CALL event, got {len(tool_call_outputs)}"
    )
    body = tool_call_outputs[0].new_text
    for needle in expected:
        assert needle in body, f"{family}: {needle!r} dropped from body: {body!r}"
    assert tool_call_outputs[0].finished is True
    assert tool_call_outputs[0].finish_reason == "stop"


def test_router_allowlist_tool_call_routing_declared():
    """Every router-allowlist family must declare its tool-call routing.

    Forcing function for the Bug A class (v0.6.62): when adding a new
    family to ``_OUTPUT_ROUTER_ALLOWLIST``, the dev must categorize it as
    either (a) emitting ``Channel.TOOL_CALL`` aggregate events through
    ``OutputRouter.feed()`` — in which case the parametrized streaming
    test above must cover it — OR (b) deferring tool-call extraction to
    the per-family ``ToolParser``. The undeclared case is what let the
    single-token-flush regression ship in v0.6.61: no test enforced that
    every router family had streaming coverage for the aggregate path.
    """
    from vllm_mlx.engine.batched import _OUTPUT_ROUTER_ALLOWLIST

    declared = (
        set(_ROUTER_FAMILIES_TOOL_CALL_AGGREGATE.keys())
        | _ROUTER_FAMILIES_TOOL_CALL_AT_PARSER_LAYER
    )
    undeclared = _OUTPUT_ROUTER_ALLOWLIST - declared
    assert not undeclared, (
        f"Router-allowlist families with no declared tool-call routing: "
        f"{sorted(undeclared)}. Add each to either "
        f"_ROUTER_FAMILIES_TOOL_CALL_AGGREGATE (router emits Channel.TOOL_CALL "
        f"as a buffered aggregate event — MUST add a parametrized streaming "
        f"coverage entry too) or _ROUTER_FAMILIES_TOOL_CALL_AT_PARSER_LAYER "
        f"(tool-call extraction happens at the per-family ToolParser layer)."
    )


@pytest.mark.asyncio
async def test_stream_chat_uses_incremental_new_text_for_single_token_events():
    """Single-token routed chunks preserve scheduler detokenizer text."""
    vocab = {
        **HARMONY_VOCAB,
        "decoded-wrong": 6,
    }
    tokenizer = FakeTokenizer(vocab)
    tokenizer._id_to_text[6] = "decoded-wrong"
    engine = _make_engine(tokenizer)

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="decoded-right",
            tokens=[6],
            finished=True,
            finish_reason="stop",
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert outputs[0].new_text == "decoded-right"
    assert outputs[0].channel == "content"


@pytest.mark.asyncio
async def test_stream_chat_leaves_unsupported_tokenizer_on_legacy_path():
    """Unsupported tokenizers preserve raw chunks with channel=None."""
    engine = _make_engine(FakeTokenizer({"Hello": 1}))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="Hello",
            new_text="Hello",
            tokens=[1],
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert len(outputs) == 1
    assert outputs[0].new_text == "Hello"
    assert outputs[0].tokens == [1]
    assert outputs[0].channel is None
    assert outputs[0].finished is True


@pytest.mark.asyncio
async def test_stream_chat_falls_back_after_router_failure():
    """A mid-stream router failure disables routing for later chunks."""
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    class FailingRouter:
        def feed(self, token_id):
            raise RuntimeError("boom")

    async def fake_outputs():
        yield GenerationOutput(
            text="",
            new_text="Fallback",
            tokens=[5],
            finished=False,
            channel=None,
        )
        yield GenerationOutput(
            text="",
            new_text="Answer",
            tokens=[4],
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    outputs = await _collect(
        engine._stream_with_output_router(fake_outputs(), FailingRouter())
    )

    assert [(o.new_text, o.channel, o.finished) for o in outputs] == [
        ("Fallback", None, False),
        ("Answer", None, True),
    ]


def test_create_output_router_catches_tokenizer_property_errors():
    """Tokenizer access failures fall back to legacy parsing."""

    class BrokenTokenizerEngine(BatchedEngine):
        @property
        def tokenizer(self):
            raise RuntimeError("not loaded")

    engine = BrokenTokenizerEngine("fake-model")

    assert engine._create_output_router() is None
