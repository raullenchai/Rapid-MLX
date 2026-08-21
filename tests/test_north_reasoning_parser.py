# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cohere North reasoning parser (North-Mini-Code).

Regression (2026-08-20 release dogfood): serving
``mlx-community/North-Mini-Code-1.0-4bit`` shipped the raw chain of
thought and the literal ``<|END_THINKING|><|START_TEXT|>`` markers inside
``message.content`` because no parser understood North's format.
"""

import json
from pathlib import Path

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.north_parser import NorthReasoningParser

END_THINK = "<|END_THINKING|>"
START_TEXT = "<|START_TEXT|>"
END_TEXT = "<|END_TEXT|>"


@pytest.fixture
def parser() -> NorthReasoningParser:
    return NorthReasoningParser()


class TestRegistry:
    def test_north_is_registered(self):
        assert get_parser("north") is NorthReasoningParser

    def test_alias_profiles_wire_north(self):
        aliases = json.loads(
            (Path(__file__).parent.parent / "vllm_mlx" / "aliases.json").read_text()
        )
        for alias in ("north-mini-code-4bit", "north-mini-code-bf16"):
            assert aliases[alias]["reasoning_parser"] == "north"

    def test_auto_config_wires_north_for_raw_hf_paths(self):
        from vllm_mlx.model_auto_config import detect_model_config

        cfg = detect_model_config("mlx-community/North-Mini-Code-1.0-4bit")
        assert cfg is not None
        assert cfg.reasoning_parser == "north"
        assert cfg.tool_call_parser is None


class TestPromptThinkingPredicateMixedTemplate:
    def test_mixed_marker_template_uses_all_pairs(self):
        # A template whose source mentions BOTH marker families but whose
        # active branch renders only the North pair must still be
        # detected (codex on #2171: first-present-pair inspection bug).
        from vllm_mlx.service.helpers import _should_start_in_thinking

        template = (
            "{# legacy: <think></think> #}"
            "{% if add_generation_prompt %}<|START_THINKING|>{% endif %}"
        )
        assert _should_start_in_thinking(template, None, unconditional=True) is True


class TestPromptThinkingPredicate:
    def test_north_template_detected_as_prompt_thinking(self):
        from vllm_mlx.service.helpers import _should_start_in_thinking

        template = (
            "{% if add_generation_prompt %}"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>"
            "{% endif %}"
        )
        assert _should_start_in_thinking(template, None) is True
        # Explicitly disabled thinking still short-circuits to False.
        assert _should_start_in_thinking(template, False) is False

    def test_think_tag_templates_still_detected(self):
        from vllm_mlx.service.helpers import _should_start_in_thinking

        template = "{% if add_generation_prompt %}<think>\n{% endif %}"
        assert _should_start_in_thinking(template, None) is True


class TestExtractReasoning:
    def test_implicit_think_shape(self, parser):
        # The live dogfood shape: opener lives in the prompt, output is
        # cot + END_THINKING + wrapped answer.
        out = f"This is simple. Answer: 4.{END_THINK}{START_TEXT}4{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning == "This is simple. Answer: 4."
        assert content == "4"

    def test_explicit_both_markers(self, parser):
        out = f"<|START_THINKING|>plan{END_THINK}{START_TEXT}done{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning == "plan"
        assert content == "done"

    def test_no_markers_routes_to_reasoning(self, parser):
        # North templates end the prompt inside <|START_THINKING|>, so a
        # marker-free output is a truncated thought trace.
        reasoning, content = parser.extract_reasoning("half a thought")
        assert reasoning == "half a thought"
        assert content is None

    def test_direct_answer_without_thinking_block(self, parser):
        out = f"{START_TEXT}direct answer{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning is None
        assert content == "direct answer"

    def test_unterminated_text_wrapper(self, parser):
        # Truncated mid-answer: END_TEXT never arrived.
        out = f"cot{END_THINK}{START_TEXT}partial answer"
        reasoning, content = parser.extract_reasoning(out)
        assert reasoning == "cot"
        assert content == "partial answer"

    def test_bare_json_response_routes_to_content_in_json_mode(self, parser):
        # JSON-mode contract: North's template instructs structured
        # responses to emit bare JSON with no channel markers — gated on
        # the EXPLICIT request signal, not inferred from the first
        # character (codex final-round #1).
        parser.configure_request(json_mode=True)
        reasoning, content = parser.extract_reasoning('{"answer": 4}')
        assert reasoning is None
        assert content == '{"answer": 4}'
        reasoning, content = parser.extract_reasoning("\n[1, 2, 3]")
        assert reasoning is None
        assert content == "\n[1, 2, 3]"

    def test_brace_headed_thought_stays_reasoning_without_json_mode(self, parser):
        # Privacy: a chain of thought that merely opens with a brace must
        # NOT bypass the reasoning split when the request did not ask for
        # JSON output (codex final-round #1).
        reasoning, content = parser.extract_reasoning(
            '{"draft": 1} — no wait, let me reconsider'
        )
        assert content is None
        assert reasoning is not None and reasoning.startswith('{"draft"')

    def test_no_marker_leakage_in_either_channel(self, parser):
        out = f"think{END_THINK}{START_TEXT}answer{END_TEXT}"
        reasoning, content = parser.extract_reasoning(out)
        for channel in (reasoning, content):
            assert channel is not None
            assert "<|" not in channel


class TestTruncationContract:
    def test_open_in_think_before_end_marker(self, parser):
        assert parser.is_open_in_think("some unfinished thought") is True

    def test_not_open_after_end_marker(self, parser):
        assert parser.is_open_in_think(f"thought{END_THINK}answer") is False

    def test_not_open_in_direct_answer_shape(self, parser):
        assert parser.is_open_in_think(f"{START_TEXT}answer") is False

    def test_empty_is_not_open(self, parser):
        assert parser.is_open_in_think("") is False


def _stream(parser, deltas):
    parser.reset_state()
    accumulated = ""
    results = []
    for delta in deltas:
        prev = accumulated
        accumulated += delta
        msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
        if msg:
            results.append(msg)
    reasoning = "".join(m.reasoning for m in results if m.reasoning)
    content = "".join(m.content for m in results if m.content)
    return reasoning, content


class TestChatRouteStreaming:
    """Route-level regression: the live 2026-08-20 dogfood repro streamed
    the whole North output (CoT + literal markers) as ``delta.content``
    because the casual-chat auto-disable resolved thinking to False and
    the postprocessor bypassed the parser."""

    def test_stream_splits_reasoning_and_strips_markers(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class NorthPlainEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def stream_chat(self, messages, **kwargs):
                deltas = [
                    "Provide answer: 4.",
                    END_THINK,
                    START_TEXT,
                    "4",
                    END_TEXT,
                ]
                acc = ""
                for i, d in enumerate(deltas):
                    acc += d
                    yield GenerationOutput(
                        text=acc,
                        new_text=d,
                        prompt_tokens=4,
                        completion_tokens=i + 1,
                        finished=(i == len(deltas) - 1),
                        finish_reason="stop" if i == len(deltas) - 1 else None,
                    )

        cfg = reset_config()
        try:
            cfg.engine = NorthPlainEngine()
            cfg.model_name = "north-test"
            cfg.model_registry = None
            cfg.reasoning_parser = get_parser("north")()
            cfg.reasoning_parser_name = "north"
            cfg.tool_parser = None
            cfg.no_thinking = False

            app = FastAPI()
            app.include_router(chat_router)
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "north-test",
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            )
            reasoning_parts, content_parts = [], []
            for raw in resp.text.split("\n\n"):
                for line in raw.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[len("data: ") :])
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(delta["reasoning_content"])
                        if delta.get("content"):
                            content_parts.append(delta["content"])
            reasoning = "".join(reasoning_parts)
            content = "".join(content_parts)
            assert reasoning == "Provide answer: 4."
            assert content == "4"
            assert "<|" not in content
        finally:
            reset_config()


class TestChatRouteEofFlush:
    def test_route_flushes_withheld_marker_tail_at_eof(self):
        """Route-level never-drop contract (codex r3 BLOCKING): a stream
        ending in a marker-like run must surface those bytes via the
        StreamingPostProcessor EOF flush, not lose them."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class TruncatedTailEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def stream_chat(self, messages, **kwargs):
                deltas = ["cot", END_THINK, START_TEXT, "answer<|END_TE"]
                acc = ""
                for i, d in enumerate(deltas):
                    acc += d
                    yield GenerationOutput(
                        text=acc,
                        new_text=d,
                        prompt_tokens=4,
                        completion_tokens=i + 1,
                        finished=(i == len(deltas) - 1),
                        finish_reason="length" if i == len(deltas) - 1 else None,
                    )

        cfg = reset_config()
        try:
            cfg.engine = TruncatedTailEngine()
            cfg.model_name = "north-test"
            cfg.model_registry = None
            cfg.reasoning_parser = get_parser("north")()
            cfg.reasoning_parser_name = "north"
            cfg.tool_parser = None
            cfg.no_thinking = False

            app = FastAPI()
            app.include_router(chat_router)
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "north-test",
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            )
            content_parts = []
            for raw in resp.text.split("\n\n"):
                for line in raw.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[len("data: ") :])
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
            content = "".join(content_parts)
            assert content.startswith("answer")
            # The withheld tail must be flushed, not dropped.
            assert "<|END_TE" in content
        finally:
            reset_config()

    def test_route_flushes_thinking_phase_tail_at_eof(self):
        """Same never-drop contract, thinking phase (codex round-2 BLOCKING
        on this PR): a stream that ends mid-marker while STILL inside the
        thinking channel releases its withheld tail as a ``reasoning``
        event from ``finalize()``. The route's finalize loop consumed only
        ``tool_call`` and ``content`` events, so that tail vanished from
        ``reasoning_content`` — delete the ``reasoning`` branch in the
        finalize loop and this test goes red."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class TruncatedThinkingEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def stream_chat(self, messages, **kwargs):
                # Ends mid END_THINKING marker: still in thinking phase.
                deltas = ["deliberating", " about it<|END_THI"]
                acc = ""
                for i, d in enumerate(deltas):
                    acc += d
                    yield GenerationOutput(
                        text=acc,
                        new_text=d,
                        prompt_tokens=4,
                        completion_tokens=i + 1,
                        finished=(i == len(deltas) - 1),
                        finish_reason="length" if i == len(deltas) - 1 else None,
                    )

        cfg = reset_config()
        try:
            cfg.engine = TruncatedThinkingEngine()
            cfg.model_name = "north-test"
            cfg.model_registry = None
            cfg.reasoning_parser = get_parser("north")()
            cfg.reasoning_parser_name = "north"
            cfg.tool_parser = None
            cfg.no_thinking = False

            app = FastAPI()
            app.include_router(chat_router)
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "north-test",
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            )
            reasoning_parts = []
            for raw in resp.text.split("\n\n"):
                for line in raw.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[len("data: ") :])
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(delta["reasoning_content"])
            reasoning = "".join(reasoning_parts)
            assert reasoning.startswith("deliberating")
            # The withheld mid-marker tail must reach reasoning_content.
            assert "<|END_THI" in reasoning
        finally:
            reset_config()


class TestStreaming:
    def test_streaming_simple_flow(self, parser):
        reasoning, content = _stream(
            parser,
            ["think", "ing", END_THINK, START_TEXT, "ans", "wer", END_TEXT],
        )
        assert reasoning == "thinking"
        assert content == "answer"

    def test_streaming_text_marker_split_across_deltas(self, parser):
        # START_TEXT arrives split over three deltas glued to answer bytes.
        reasoning, content = _stream(
            parser,
            ["cot", END_THINK, "<|STA", "RT_TE", "XT|>an", "swer", END_TEXT],
        )
        assert reasoning == "cot"
        assert content == "answer"

    def test_streaming_end_text_split_with_glued_bytes(self, parser):
        reasoning, content = _stream(
            parser,
            ["cot", END_THINK, START_TEXT, "answer<|END_", "TEXT|>"],
        )
        assert reasoning == "cot"
        assert content == "answer"

    def test_streaming_no_marker_bytes_leak(self, parser):
        parser.reset_state()
        deltas = ["cot", END_THINK, "<|STA", "RT_TE", "XT|>a", END_TEXT]
        accumulated = ""
        for delta in deltas:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if msg and msg.content:
                assert "<|" not in msg.content

    def test_streaming_bare_json_routes_to_content(self, parser):
        # JSON-mode streaming: bare JSON with no markers must stream on
        # the content lane, not vanish into reasoning — gated on the
        # explicit request signal (codex final-round #1).
        parser.configure_request(json_mode=True)
        accumulated = ""
        rc, cc = [], []
        for delta in ['{"ans', 'wer": ', "4}"]:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if msg and msg.reasoning:
                rc.append(msg.reasoning)
            if msg and msg.content:
                cc.append(msg.content)
        assert "".join(rc) == ""
        assert "".join(cc) == '{"answer": 4}'

    def test_scalar_json_roots_route_to_content_in_json_mode(self, parser):
        # JSON permits scalar roots — "ok", 42, true, null (codex
        # final-round-2 #2).
        for doc in ('"ok"', "42", "true", "null"):
            parser.configure_request(json_mode=True)
            reasoning, content = parser.extract_reasoning(doc)
            assert reasoning is None, doc
            assert content == doc

    def test_streaming_scalar_json_routes_to_content(self, parser):
        parser.configure_request(json_mode=True)
        accumulated = ""
        cc = []
        for delta in ["tr", "ue"]:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if msg and msg.content:
                cc.append(msg.content)
        assert "".join(cc) == "true"

    def test_configure_request_resets_machine_state(self, parser):
        # The postprocessor reset path calls configure_request INSTEAD of
        # reset_state — a reused parser must not carry the previous
        # request's phase/buffer (codex final-round-2 #1).
        parser.extract_reasoning_streaming("", END_THINK, END_THINK)
        assert parser._sm_phase == "content"
        parser.configure_request(json_mode=False)
        assert parser._sm_phase == "thinking"
        assert parser._sm_buf == ""

    def test_streaming_brace_thought_stays_reasoning_without_json_mode(self, parser):
        reasoning, content = _stream(parser, ['{"draft"', ": 1} hmm"])
        assert content == ""
        assert reasoning.startswith('{"draft"')

    def test_streaming_split_end_thinking_marker(self, parser):
        # <|END_THINKING|> split across deltas must not leak marker bytes
        # into reasoning nor swallow the answer (codex r2 BLOCKING #1).
        reasoning, content = _stream(
            parser,
            ["cot<", "|END_THINKING|>", START_TEXT, "answer", END_TEXT],
        )
        assert reasoning == "cot"
        assert content == "answer"

    def test_streaming_spill_before_start_text(self, parser):
        # Channel spill before <|START_TEXT|> with no thinking closer:
        # mirror of the non-streaming ("spill", "answer") split (codex r2
        # BLOCKING #2).
        reasoning, content = _stream(
            parser,
            ["spill", START_TEXT, "answer", END_TEXT],
        )
        assert reasoning == "spill"
        assert content == "answer"

    def test_implicit_reasoning_until_close_is_set(self, parser):
        # North templates prime thinking unconditionally — same contract
        # pair as DeepSeek-R1-Distill (codex r2 BLOCKING #3).
        assert parser.implicit_reasoning_until_close is True

    def test_streaming_direct_answer_shape(self, parser):
        # No thinking block at all: <|START_TEXT|> arrives first (split
        # across deltas) — mirror of the non-streaming direct-answer
        # branch (codex r1 BLOCKING #1).
        reasoning, content = _stream(
            parser,
            ["<|STAR", "T_TEXT|>dir", "ect answer", END_TEXT],
        )
        assert reasoning == ""
        assert content == "direct answer"

    def test_streaming_direct_answer_with_leading_whitespace(self, parser):
        reasoning, content = _stream(
            parser,
            ["\n ", START_TEXT, "ok", END_TEXT],
        )
        assert reasoning == ""
        assert content == "ok"

    def test_finalize_flushes_partial_marker_tail(self, parser):
        # An answer that ends in a marker-like prefix is withheld by the
        # streaming stripper; finalize must flush it, never drop it
        # (codex r1 BLOCKING #2).
        parser.reset_state()
        accumulated = ""
        outs = []
        for delta in ["cot", END_THINK, START_TEXT, "answer<|END_TE"]:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if msg:
                outs.append(msg)
        content = "".join(m.content for m in outs if m.content)
        assert content == "answer"
        fin = parser.finalize_streaming(accumulated)
        assert fin is not None
        assert (fin.content or "").endswith("<|END_TE")

    def test_sanitize_when_thinking_disabled_is_set(self, parser):
        """North templates ignore ``enable_thinking`` and always pre-open
        the thinking channel; the postprocessor's thinking-off bypass
        (e.g. after the casual-chat auto-disable) must therefore keep the
        parser engaged — this flag is the contract the streaming gate
        checks. Live repro: with the flag absent, every streamed delta
        (CoT + literal markers) shipped as ``delta.content``."""
        assert parser.sanitize_when_thinking_disabled is True

    def test_streaming_lone_angle_in_answer_survives(self, parser):
        # A genuine "<" in the answer that never becomes a marker must be
        # flushed by the following delta, not swallowed.
        reasoning, content = _stream(
            parser,
            ["cot", END_THINK, START_TEXT, "a <", " b", END_TEXT],
        )
        assert content == "a < b"


class TestContentPhaseConsumesGenuineCloser:
    """codex round-4 MAJOR on this PR: a reasoning-cap forced close
    flips the machine to "content" while the model — which never saw
    the forged closer — still emits its genuine ``<|END_THINKING|>``
    later. Content phase stripped only TEXT wrappers, so the genuine
    closer shipped as literal visible bytes (and leaked incrementally
    when split across deltas). Content phase now consumes it as a
    structural no-op. Revert ``_strip_content_phase_markers`` /
    ``_CONTENT_MARKERS`` in the content branch to see these go red."""

    def _force_flip(self, parser):
        # The cap machinery's forced close: parser sees its own closer
        # and flips to content phase with nothing else in the buffer.
        #
        # These tests deliberately pin post-flip thought bytes
        # ("erate") ARRIVING AS CONTENT. That is the reasoning-cap
        # contract, not a leak: the cap (upstream vLLM #20859 backport)
        # reroutes over-budget bytes to content precisely so no model
        # output is ever silently dropped, and every think-tag parser
        # behaves identically post-cap (probe-verified against qwen3 —
        # same spill, same channel). A "discard until the genuine
        # closer" state would silently destroy model bytes and diverge
        # north from the whole parser family.
        msg = parser.extract_reasoning_streaming("", END_THINK, END_THINK)
        assert parser._sm_phase == "content"
        return msg

    def test_whole_genuine_closer_after_forced_flip(self, parser):
        self._force_flip(parser)
        reasoning, content = _stream_from_phase(
            parser, ["erate", END_THINK, START_TEXT, "4", END_TEXT]
        )
        assert content == "erate4", (reasoning, content)
        assert "<|" not in content

    def test_split_genuine_closer_after_forced_flip(self, parser):
        self._force_flip(parser)
        reasoning, content = _stream_from_phase(
            parser, ["erate<|END_THI", "NKING|>", "<|START_TEXT|>4<|END_TEXT|>"]
        )
        assert content == "erate4", (reasoning, content)
        assert "<|" not in content

    def test_split_closer_pending_at_eof_flushes_literally(self, parser):
        # Never-drop at EOF still wins over marker withholding: if the
        # stream dies mid-marker the held bytes are model output.
        self._force_flip(parser)
        reasoning, content = _stream_from_phase(parser, ["answer tail<|END_THI"])
        final = parser.finalize_streaming("")
        flushed = (final.content or "") if final else ""
        assert content + flushed == "answer tail<|END_THI", (content, flushed)


def _stream_from_phase(parser, deltas):
    accumulated = ""
    results = []
    for delta in deltas:
        prev = accumulated
        accumulated += delta
        msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
        if msg:
            results.append(msg)
    reasoning = "".join(m.reasoning for m in results if m.reasoning)
    content = "".join(m.content for m in results if m.content)
    return reasoning, content
