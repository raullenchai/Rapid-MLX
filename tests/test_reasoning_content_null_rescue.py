# SPDX-License-Identifier: Apache-2.0
"""Length stop before the reasoning end marker (#5295).

When generation stops for ``max_tokens`` before ``</think>`` (or the
harmony final-channel marker), the unfinished output stays in the
reasoning channel and ``content`` stays empty/null on every transport,
buffered and streaming. No notice and no copy of the reasoning tail is
synthesized into ``content``; clients read the structured signals
(``finish_reason="length"``, Responses ``status="incomplete"``, Messages
``stop_reason="max_tokens"``). The former opt-out environment knobs
(``RAPID_MLX_REASONING_RESCUE`` / ``RAPID_MLX_REASONING_CUTOFF_NOTICE``)
no longer change the output; the route tests set them to ``on`` to pin
that.
"""

from __future__ import annotations

import json

import pytest

from rapid_mlx.service.helpers import _rescue_silent_drop_from_reasoning

#: Literal of the removed notice, kept only so tests can pin its absence.
_OLD_NOTICE = "[truncated — reasoning incomplete; raise max_tokens]"


@pytest.fixture
def legacy_notice_env_on(monkeypatch):
    """Set the retired knobs to ``on``; output must not change."""
    monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on")
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")


# ──────────────────────────────────────────────────────────────────────
# Parser-wide assembly tests: drive _finalize_content_and_reasoning +
# _rescue_silent_drop_from_reasoning in the same orchestration the chat
# route runs, against every ``<think>``-style parser family.
# ──────────────────────────────────────────────────────────────────────


def _finalize_route_assembly(
    *,
    raw_text: str,
    reasoning_parser,
    finish_reason: str,
    enable_thinking: bool | None = None,
    engine_reasoning_text: str = "",
):
    """Mirror the chat route's final-content assembly sequence:

    1. ``_finalize_content_and_reasoning`` — extracts reasoning/content
    2. ``_rescue_silent_drop_from_reasoning`` — issue #569 rescue

    Returns ``(final_content, reasoning_text)`` exactly as the route
    layer would set them on the AssistantMessage.
    """
    from rapid_mlx.api.utils import (
        clean_output_text,
        sanitize_output,
        strip_thinking_tags,
    )
    from rapid_mlx.service.helpers import (
        _finalize_content_and_reasoning,
        _rescue_silent_drop_from_reasoning,
    )

    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw_text,
        cleaned_text=raw_text,
        tool_calls=[],
        reasoning_parser=reasoning_parser,
        engine_reasoning_text=engine_reasoning_text,
        enable_thinking=enable_thinking,
    )
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
    final_content = _rescue_silent_drop_from_reasoning(
        final_content,
        reasoning_text,
        tool_calls=[],
        finish_reason=finish_reason,
        raw_text=raw_text,
        reasoning_is_case4=False,
    )
    return final_content, reasoning_text


def _parser_cases():
    """Every ``<think>``-tag reasoning-parser family alongside shared
    raw-text shapes that produce the four scenarios this helper cares
    about: open-only (mid-think), closed+truncated-answer, stop-cut
    mid-think, and the happy-path closed pair.

    Each entry: ``(name, parser_class, raw_open_only, raw_closed_trunc,
    raw_stop_cut_mid, raw_happy)``."""
    from rapid_mlx.reasoning.deepseek_r1_parser import (
        DeepSeekR1ReasoningParser,
        VibeThinkerReasoningParser,
    )
    from rapid_mlx.reasoning.glm4_parser import Glm4ReasoningParser
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    think_open_only = "<think>Let me think about 17*23. 17 * 20 = 340. 17 * 3 ="
    think_closed_trunc = "<think>17 * 23 = 391.</think>The answer is 39"
    think_stop_mid = "<think>The user wants weather. Let me check"
    think_happy = "<think>17*23 = 391</think>The answer is 391."

    return [
        (
            "qwen3",
            Qwen3ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "deepseek_r1",
            DeepSeekR1ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "vibethinker",
            VibeThinkerReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "glm4",
            Glm4ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
    ]


@pytest.fixture(params=_parser_cases(), ids=lambda p: p[0])
def parser_case(request):
    name, cls, raw_open_only, raw_closed_trunc, raw_stop_mid, raw_happy = request.param
    return {
        "name": name,
        "parser": cls(),
        "raw_open_only": raw_open_only,
        "raw_closed_trunc": raw_closed_trunc,
        "raw_stop_mid": raw_stop_mid,
        "raw_happy": raw_happy,
    }


class TestParserWideLengthCutMidThink:
    """Length-cut mid-think produces empty content for every reasoning
    parser family; the trace stays in ``reasoning_content``."""

    def test_length_cut_mid_think_leaves_content_empty(self, parser_case):
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_open_only"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert not content, (
            f"[{parser_case['name']}]: unfinished reasoning must not reach "
            f"content; got content={content!r}"
        )
        # ``reasoning_content`` keeps the trace.
        assert reasoning is not None and "17" in reasoning, (
            f"reasoning_content must remain populated for "
            f"{parser_case['name']}; got {reasoning!r}"
        )

    def test_length_cut_after_close_preserves_partial_answer(self, parser_case):
        """Happy(ish) path: the reasoning block CLOSED before the
        truncation point. ``content`` carries the partial answer and
        must NOT be overwritten with anything."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_closed_trunc"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert content == "The answer is 39", (
            f"length-cut AFTER reasoning-close [{parser_case['name']}]: "
            f"must preserve partial answer; got content={content!r}"
        )
        assert reasoning is not None and "391" in reasoning

    def test_stop_cut_mid_think_strict_null(self, parser_case):
        """A stop-cut mid-think never carries a synthesized notice (the
        #569 stop-finish rescue is unchanged and out of scope here)."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_stop_mid"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        assert _OLD_NOTICE not in (content or ""), (
            f"[{parser_case['name']}]: no notice on finish_reason=stop; "
            f"got content={content!r}"
        )
        assert reasoning is not None and reasoning.strip()

    def test_happy_path_full_generation_preserves_answer(self, parser_case):
        """Full closed reasoning + answer flow: content has the answer,
        reasoning has the trace."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_happy"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        assert content == "The answer is 391.", (
            f"happy-path [{parser_case['name']}]: content must equal answer; "
            f"got {content!r}"
        )
        assert reasoning is not None and "391" in reasoning


class TestRescueSilentDropFromReasoning:
    def test_1570_natural_eos_from_prompt_primed_reasoning_stays_suppressed(self):
        """Natural EOS does not turn an unclosed implicit thought into content."""
        content = _rescue_silent_drop_from_reasoning(
            final_content=None,
            reasoning_text="Okay, so I need to figure out unified memory.",
            tool_calls=None,
            finish_reason="stop",
            raw_text="Okay, so I need to figure out unified memory.",
            reasoning_is_case4=True,
            # The shared template probe recognizes that R1-Distill's shipped
            # template ignores enable_thinking=False and still primes <think>.
            prompt_thinking_active=True,
            implicit_reasoning_until_close=True,
        )
        assert content is None

    def test_1570_custom_non_priming_template_rescues_short_answer(self):
        content = _rescue_silent_drop_from_reasoning(
            final_content=None,
            reasoning_text="Unified memory is shared by the CPU and GPU.",
            tool_calls=None,
            finish_reason="stop",
            raw_text="Unified memory is shared by the CPU and GPU.",
            reasoning_is_case4=True,
            prompt_thinking_active=False,
            implicit_reasoning_until_close=True,
        )
        assert content == "Unified memory is shared by the CPU and GPU."

    def test_case4_length_without_prompt_thinking_rescues_content(self):
        """A non-thinking answer truncated by max_tokens must not be
        silently dropped just because a Case-4 parser put it in reasoning."""
        content = _rescue_silent_drop_from_reasoning(
            final_content=None,
            reasoning_text="The answer is 12",
            tool_calls=None,
            finish_reason="length",
            raw_text="The answer is 12",
            reasoning_is_case4=True,
            prompt_thinking_active=False,
        )
        assert content == "The answer is 12"

    def test_case4_length_with_prompt_thinking_stays_suppressed(self):
        """Prompt-injected max_tokens mid-think remains a structured
        reasoning truncation, not user-visible content."""
        content = _rescue_silent_drop_from_reasoning(
            final_content=None,
            reasoning_text="5+7 equals 12",
            tool_calls=None,
            finish_reason="length",
            raw_text="5+7 equals 12",
            reasoning_is_case4=True,
            prompt_thinking_active=True,
        )
        assert content is None


# ──────────────────────────────────────────────────────────────────────
# Streaming SSE: no content event on a length stop mid-think.
# ──────────────────────────────────────────────────────────────────────


class _StreamEngine:
    """Minimal streaming engine for the streaming-surface tests.

    Emits text-mode (``channel=None``) deltas so the route's reasoning
    parser is on the hot path — the test then configures a Qwen3 parser
    via ``cfg.reasoning_parser`` and the deltas start with a literal
    ``<think>`` opener so the parser's ``_saw_any_tag`` flag flips and
    the streaming rescue's truncated-think gate fires.

    Set ``include_open_tag=False`` to opt out of the ``<think>`` opener
    for tests that want to assert the plain silent-drop rescue path
    (no parser flag flip)."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(
        self,
        reasoning_deltas: list[str],
        finish_reason: str = "length",
        include_open_tag: bool = True,
    ):
        self._deltas = reasoning_deltas
        self._finish_reason = finish_reason
        self._include_open_tag = include_open_tag
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        from rapid_mlx.engine.base import GenerationOutput

        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        accumulated = ""
        deltas = list(self._deltas)
        if self._include_open_tag and deltas:
            deltas[0] = "<think>" + deltas[0]
        for i, delta in enumerate(deltas):
            accumulated += delta
            is_last = i == len(deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason=self._finish_reason if is_last else None,
                channel=None,
            )


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


def _stream_post(
    reasoning_deltas,
    finish_reason="length",
    include_open_tag=True,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _StreamEngine(
        reasoning_deltas,
        finish_reason=finish_reason,
        include_open_tag=include_open_tag,
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"

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
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
        assert resp.status_code == 200, resp.text
        return _parse_sse(resp.text)
    finally:
        reset_config()


def test_streaming_length_cut_mid_think_emits_no_content(legacy_notice_env_on):
    """When reasoning streamed but no content streamed AND
    ``finish_reason="length"``, no ``delta.content`` event is emitted,
    even with the retired notice knobs set to ``on``. Per-delta
    ``reasoning_content`` chunks still flow during the loop."""
    events = _stream_post(
        ["Let me think about ", "the weather query. ", "I should call"],
        finish_reason="length",
    )
    assert events, "expected at least the terminal SSE chunk"

    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            assert not d.get("content"), (
                f"length stop mid-think must not emit content; got delta={d!r}"
            )
    assert events[-1]["choices"][0]["finish_reason"] == "length"

    # Reasoning content stream is independent and still flows.
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            if d.get("reasoning_content"):
                streamed_reasoning += d["reasoning_content"]
    assert "weather query" in streamed_reasoning, (
        "per-delta reasoning_content chunks must still flow; "
        f"got streamed={streamed_reasoning!r}"
    )


def test_streaming_stop_cut_mid_think_no_content_d_stop_think_guard(
    legacy_notice_env_on,
):
    """SSE D-STOP-THINK regression guard: stop-string cut mid-think
    keeps ``delta.content=None`` on every chunk."""
    events = _stream_post(
        ["Buffer ", "more ", "thought"],
        finish_reason="stop",
    )

    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            assert not d.get("content"), (
                f"D-STOP-THINK: no content allowed on stop-cut mid-think; "
                f"got delta={d!r}"
            )


def test_streaming_happy_path_content_unchanged_on_length(legacy_notice_env_on):
    """Happy-path guard: when content WAS streamed during the loop, the
    assembled content stream MUST equal the original output."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.routes.chat import router as chat_router

    class _ContentEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return "PROMPT"

        async def stream_chat(self, messages, **kwargs):
            for i, txt in enumerate(["Hello ", "world."]):
                is_last = i == 1
                yield GenerationOutput(
                    text=txt,
                    new_text=txt,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=is_last,
                    finish_reason="length" if is_last else None,
                    channel=None,
                )

    cfg = reset_config()
    cfg.engine = _ContentEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
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
        events = _parse_sse(resp.text)
        streamed_content = ""
        for ev in events:
            for choice in ev.get("choices", []):
                d = choice.get("delta") or {}
                if d.get("content"):
                    streamed_content += d["content"]
        assert streamed_content == "Hello world.", (
            f"happy-path content stream must equal engine output; "
            f"got {streamed_content!r}"
        )
    finally:
        reset_config()


# ──────────────────────────────────────────────────────────────────────
# Buffered route tests on every transport.
# ──────────────────────────────────────────────────────────────────────


class _EngineLengthCutMidThink:
    """Shared mock engine for the route-wiring behavioral tests.

    Returns a single ``GenerationOutput`` with an unclosed ``<think>``
    opener and ``finish_reason="length"`` — the exact production
    failure shape of #5295. Used across chat /
    responses / anthropic e2e wiring tests so the contract is identical
    on every surface.
    """

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
        from rapid_mlx.engine.base import GenerationOutput

        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="<think>Computing 17*23 step by step",
            new_text="<think>Computing 17*23 step by step",
            prompt_tokens=4,
            completion_tokens=12,
            finished=True,
            finish_reason="length",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        yield await self.chat(messages, **kwargs)


def _seed_length_cut_engine(cfg):
    """Common cfg shape for the route-wiring behavioral tests:
    qwen3 reasoning parser + length-cut mock engine."""
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg.engine = _EngineLengthCutMidThink()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"


def test_chat_route_length_cut_mid_think_content_null(legacy_notice_env_on):
    """``/v1/chat/completions`` non-streaming: a length-cut mid-think
    reply keeps ``content`` null and the trace in ``reasoning_content``."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

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
        assert not msg.get("content"), msg
        assert payload["choices"][0]["finish_reason"] == "length"
        assert msg.get("reasoning_content") == "Computing 17*23 step by step"
    finally:
        reset_config()


def test_anthropic_route_length_cut_mid_think_thinking_only(legacy_notice_env_on):
    """``/v1/messages``: a length-cut mid-think reply carries only the
    ``thinking`` block and ``stop_reason="max_tokens"``."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

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
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["stop_reason"] == "max_tokens"
        blocks = payload.get("content") or []
        assert [b["type"] for b in blocks] == ["thinking"], payload
        assert blocks[0]["thinking"] == "Computing 17*23 step by step"
    finally:
        reset_config()


def test_responses_route_length_cut_mid_think_incomplete(legacy_notice_env_on):
    """``/v1/responses``: a length-cut mid-think reply is
    ``status="incomplete"`` with only a reasoning item."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

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
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "incomplete"
        assert payload["incomplete_details"] == {"reason": "max_output_tokens"}
        assert [item["type"] for item in payload["output"]] == ["reasoning"], payload
        assert json.dumps(payload).find("output_text") == -1
    finally:
        reset_config()
