# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for the Cohere Command typed-channel detector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.reasoning import DeltaMessage, ReasoningParser, get_parser
from vllm_mlx.reasoning.cohere_command_parser import (
    ACTION_START,
    TEXT_END,
    TEXT_START,
    THINK_END,
    THINK_START,
    CohereCommand4ReasoningParser,
    _first_marker_outside_json_strings,
    _json_container_end,
    _partial_marker_suffix_length,
)


def _stream(
    text: str,
    chunks: list[str],
    *,
    json_mode: bool = False,
) -> tuple[str | None, str | None]:
    parser = CohereCommand4ReasoningParser()
    parser.configure_request(json_mode=json_mode)
    previous = ""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_reasoning_streaming(previous, current, chunk)
        previous = current
        if delta and delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta and delta.content:
            content_parts.append(delta.content)
    final = parser.finish_stream()
    if final and final.reasoning:
        reasoning_parts.append(final.reasoning)
    if final and final.content:
        content_parts.append(final.content)
    return "".join(reasoning_parts) or None, "".join(content_parts) or None


def _all_two_part_splits(text: str) -> list[list[str]]:
    return [[text], *[[text[:i], text[i:]] for i in range(1, len(text))], list(text)]


class TestRegistration:
    def test_protocol_and_legacy_names_resolve_to_same_parser(self):
        assert get_parser("cohere_command4") is CohereCommand4ReasoningParser
        assert get_parser("north") is CohereCommand4ReasoningParser

    def test_aliases_use_protocol_name(self):
        aliases = json.loads(
            (Path(__file__).parent.parent / "vllm_mlx" / "aliases.json").read_text()
        )
        for alias in ("north-mini-code-4bit", "north-mini-code-bf16"):
            assert aliases[alias]["reasoning_parser"] == "cohere_command4"

    def test_raw_checkpoint_path_uses_protocol_name(self):
        from vllm_mlx.model_auto_config import detect_model_config

        config = detect_model_config("mlx-community/North-Mini-Code-1.0-4bit")
        assert config is not None
        assert config.reasoning_parser == "cohere_command4"
        assert config.tool_call_parser == "north"


def test_shared_parser_lifecycle_defaults_and_think_boundaries():
    class StatelessParser(ReasoningParser):
        def extract_reasoning(self, model_output, **kwargs):
            return None, model_output

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            return DeltaMessage(content=delta_text)

    parser = StatelessParser()
    assert parser.reasoning_start_str is None
    assert parser.reasoning_end_str is None
    assert parser.finish_stream() is None
    assert parser.prepare_forced_reasoning_end() is None

    think_parser = get_parser("qwen3")()
    assert think_parser.reasoning_start_str == think_parser.start_token
    assert think_parser.reasoning_end_str == think_parser.end_token


def test_protocol_helpers_cover_empty_incomplete_and_escaped_inputs():
    assert _partial_marker_suffix_length("", (THINK_END,)) == 0
    assert _json_container_end("   ") is None
    escaped = '{"value":"a\\\\b"}'
    assert _json_container_end(escaped) == len(escaped)
    assert _json_container_end('{"incomplete":') is None
    quoted = '{"value":"<|START_THINKING|>"}'
    assert _first_marker_outside_json_strings(quoted, (THINK_START,)) is None
    malformed = f'{{"draft":1{THINK_END}'
    assert _first_marker_outside_json_strings(malformed, (THINK_END,)) == (
        malformed.index(THINK_END),
        THINK_END,
    )


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        (
            f"plan{THINK_END}{TEXT_START}answer{TEXT_END}",
            ("plan", "answer"),
        ),
        (
            f"{THINK_START}plan{THINK_END}{TEXT_START}answer{TEXT_END}",
            ("plan", "answer"),
        ),
        (f"{TEXT_START}direct{TEXT_END}", (None, "direct")),
        (
            f'plan{THINK_END}{ACTION_START}[{{"tool_name":"f"}}]<|END_ACTION|>',
            ("plan", f'{ACTION_START}[{{"tool_name":"f"}}]<|END_ACTION|>'),
        ),
        ("unfinished thought", ("unfinished thought", None)),
        (f"plan{THINK_END}{TEXT_START}partial", ("plan", "partial")),
    ],
)
def test_full_parse_protocol_shapes(wire, expected):
    assert CohereCommand4ReasoningParser().extract_reasoning(wire) == expected


def test_full_parse_empty_missing_text_block_and_close_without_output():
    parser = CohereCommand4ReasoningParser()
    assert parser.extract_reasoning("") == (None, None)
    assert parser._extract_text_block("plain") is None
    assert parser.extract_reasoning(f"plan{THINK_END}") == ("plan", None)


def test_boundary_properties_and_open_think_state():
    parser = CohereCommand4ReasoningParser()
    assert parser.reasoning_start_str == THINK_START
    assert parser.reasoning_end_str == THINK_END
    assert parser.start_token == THINK_START
    assert parser.end_token == THINK_END
    assert parser.is_open_in_think("") is False
    assert parser.is_open_in_think("draft") is True
    assert parser.is_open_in_think(f"draft{THINK_END}") is False
    parser.configure_request(json_mode=True)
    assert parser.is_open_in_think("draft") is False


@pytest.mark.parametrize(
    "wire",
    [
        f"plan{THINK_END}{TEXT_START}answer{TEXT_END}",
        f"{THINK_START}plan{THINK_END}{TEXT_START}answer{TEXT_END}",
        f"{TEXT_START}direct{TEXT_END}",
        f'plan{THINK_END}{ACTION_START}[{{"tool_name":"f"}}]<|END_ACTION|>',
        "unfinished thought",
        f"plan{THINK_END}{TEXT_START}partial",
    ],
)
def test_streaming_is_invariant_at_every_boundary(wire):
    expected = CohereCommand4ReasoningParser().extract_reasoning(wire)
    for chunks in _all_two_part_splits(wire):
        assert _stream(wire, chunks) == expected, chunks


@pytest.mark.parametrize(
    "document", ['{"answer":4}', "\n[1,2]", '"ok"', "42", "true", "null"]
)
def test_json_mode_routes_bare_json_to_content(document):
    parser = CohereCommand4ReasoningParser()
    assert parser.extract_reasoning(document, json_mode=True) == (None, document)
    assert parser.is_open_in_think(document) is False
    for chunks in _all_two_part_splits(document):
        assert _stream(document, chunks, json_mode=True) == (None, document)


def test_json_container_waits_for_eof_before_publishing_ambiguous_bytes():
    parser = CohereCommand4ReasoningParser()
    parser.configure_request(json_mode=True)

    first = parser.extract_reasoning_streaming("", '{"items":[1,', '{"items":[1,')
    second = parser.extract_reasoning_streaming(
        '{"items":[1,',
        '{"items":[1,2]}trailing scratch',
        "2]}trailing scratch",
    )
    final = parser.finish_stream()

    assert first is None
    assert second is None
    assert final is not None and final.content == '{"items":[1,2]}'
    assert parser.extract_reasoning_streaming("", "ignored", "ignored") is None


def test_json_shaped_reasoning_defers_to_typed_channel_at_every_boundary():
    wire = f'{{"draft":1}}{THINK_END}{TEXT_START}{{"answer":4}}{TEXT_END}'
    expected = ('{"draft":1}', '{"answer":4}')

    parser = CohereCommand4ReasoningParser()
    assert parser.extract_reasoning(wire, json_mode=True) == expected
    for chunks in _all_two_part_splits(wire):
        assert _stream(wire, chunks, json_mode=True) == expected, chunks


def test_incomplete_json_shaped_reasoning_yields_to_unquoted_protocol_markers():
    wire = f'{{"draft":1{THINK_END}{TEXT_START}{{"answer":4}}{TEXT_END}'
    expected = ('{"draft":1', '{"answer":4}')

    parser = CohereCommand4ReasoningParser()
    assert parser.extract_reasoning(wire, json_mode=True) == expected
    for chunks in _all_two_part_splits(wire):
        assert _stream(wire, chunks, json_mode=True) == expected, chunks


def test_json_container_lexer_ignores_brackets_and_escapes_inside_strings():
    wire = '{"value":"escaped \\" } ] text","nested":{"ok":true}}ignored'

    reasoning, content = _stream(wire, list(wire), json_mode=True)

    assert reasoning is None
    assert content == wire.removesuffix("ignored")


@pytest.mark.parametrize(
    "wire",
    [
        '{"value":"<|START_THINKING|>"}',
        '{"value":"<|START_TEXT|>"}',
        '\n["<|END_THINKING|>", {"action":"<|START_ACTION|>"}]',
    ],
)
def test_json_container_treats_protocol_markers_inside_strings_as_data(wire):
    parser = CohereCommand4ReasoningParser()
    assert parser.extract_reasoning(wire, json_mode=True) == (None, wire)
    for chunks in _all_two_part_splits(wire):
        assert _stream(wire, chunks, json_mode=True) == (None, wire)


def test_json_looking_thought_stays_private_without_json_request():
    document = '{"draft":1} — reconsider'
    assert CohereCommand4ReasoningParser().extract_reasoning(document) == (
        document,
        None,
    )


def test_json_mode_waits_for_protocol_evidence_before_classifying_prose():
    wire = f'draft reasoning{THINK_END}{TEXT_START}{{"answer":4}}{TEXT_END}'
    assert _stream(wire, [wire], json_mode=True) == (
        "draft reasoning",
        '{"answer":4}',
    )


def test_nonstream_orchestrator_passes_json_request_contract():
    from vllm_mlx.service.helpers import _finalize_content_and_reasoning

    document = '{"answer":4}'
    content, reasoning = _finalize_content_and_reasoning(
        raw_text=document,
        cleaned_text=document,
        tool_calls=[],
        reasoning_parser=CohereCommand4ReasoningParser(),
        json_mode=True,
    )
    assert content == document
    assert reasoning is None


def test_action_marker_is_preserved_for_downstream_tool_parser():
    action = (
        f'{ACTION_START}[{{"tool_name":"weather","parameters":{{}}}}]<|END_ACTION|>'
    )
    wire = f"check forecast{THINK_END}{action}"
    reasoning, content = _stream(wire, list(wire))
    assert reasoning == "check forecast"
    assert content == action


def test_direct_action_transition_is_preserved():
    action = f'{ACTION_START}[{{"tool_name":"f"}}]<|END_ACTION|>'
    assert _stream(action, [action]) == (None, action)


def test_whitespace_only_reasoning_emits_nothing():
    assert _stream("   ", ["   "]) == (None, None)


def test_close_without_output_drains_cleanly_at_eof():
    assert _stream(f"plan{THINK_END}", ["plan", THINK_END]) == ("plan", None)


def test_partial_output_marker_after_close_is_discarded_at_eof():
    wire = f"plan{THINK_END}<|START_TE"
    assert _stream(wire, ["plan", THINK_END, "<|START_TE"]) == ("plan", None)


def test_finish_releases_partial_marker_in_reasoning_phase():
    wire = "thinking<|END_THI"
    assert _stream(wire, ["thinking", "<|END_THI"]) == (wire, None)


def test_finish_releases_partial_text_end_as_answer_bytes():
    wire = f"plan{THINK_END}{TEXT_START}answer<|END_TE"
    assert _stream(wire, ["plan", THINK_END, TEXT_START, "answer<|END_TE"]) == (
        "plan",
        "answer<|END_TE",
    )


def test_finish_is_idempotent():
    parser = CohereCommand4ReasoningParser()
    parser.extract_reasoning_streaming("", "abc<|END_THI", "abc<|END_THI")
    assert parser.finish_stream() is not None
    assert parser.finish_stream() is None


def test_configure_request_resets_incremental_state():
    parser = CohereCommand4ReasoningParser()
    parser.extract_reasoning_streaming("", THINK_END, THINK_END)
    parser.configure_request(json_mode=True)
    message = parser.extract_reasoning_streaming("", '{"ok":', '{"ok":')
    assert message is None
    final = parser.finish_stream()
    assert final is not None
    assert final.reasoning is None
    assert final.content == '{"ok":'


def test_forced_reasoning_end_keeps_later_model_close_structural():
    parser = CohereCommand4ReasoningParser()
    first = parser.extract_reasoning_streaming("", "abcdefgh", "abcdefgh")
    assert first is not None and first.reasoning == "abcdefgh"

    parser.prepare_forced_reasoning_end()
    assert parser.extract_reasoning_streaming("", THINK_END, THINK_END) is None
    tail = f"ijkl{THINK_END}{TEXT_START}done{TEXT_END}"
    message = parser.extract_reasoning_streaming("", tail, tail)
    assert message is not None
    assert message.reasoning is None
    assert message.content == "ijkldone"


def test_forced_content_preserves_action_and_partial_marker_at_eof():
    parser = CohereCommand4ReasoningParser()
    parser.extract_reasoning_streaming("", "plan", "plan")
    parser.prepare_forced_reasoning_end()
    parser.extract_reasoning_streaming("plan", f"plan{THINK_END}", THINK_END)

    action = f'{ACTION_START}[{{"tool_name":"f"}}]<|END_ACTION|>'
    message = parser.extract_reasoning_streaming(
        "", f"visible{action}", f"visible{action}"
    )
    assert message is not None
    assert message.content == f"visible{action}"

    parser = CohereCommand4ReasoningParser()
    parser.extract_reasoning_streaming("", "plan", "plan")
    parser.prepare_forced_reasoning_end()
    parser.extract_reasoning_streaming("plan", f"plan{THINK_END}", THINK_END)
    held = "tail<|END_THI"
    first = parser.extract_reasoning_streaming("", held, held)
    final = parser.finish_stream()
    assert first is not None and first.content == "tail"
    assert final is not None and final.content == "<|END_THI"


def test_forced_close_routes_unaccounted_marker_prefix_to_content():
    parser = CohereCommand4ReasoningParser()
    prefix = "<|END_THI"
    first = parser.extract_reasoning_streaming("", f"abcd{prefix}", f"abcd{prefix}")
    assert first is not None and first.reasoning == "abcd"

    parser.prepare_forced_reasoning_end()
    forced = parser.extract_reasoning_streaming("", THINK_END, THINK_END)

    assert forced is not None
    assert forced.reasoning is None
    assert forced.content == prefix


def test_prompt_priming_detects_command_markers_and_mixed_templates():
    from vllm_mlx.service.helpers import _should_start_in_thinking

    template = (
        "{# historical <think></think> markers #}"
        "{% if add_generation_prompt %}<|START_THINKING|>{% endif %}"
    )
    assert _should_start_in_thinking(template, None, unconditional=True) is True


def test_request_flags_keep_parser_active_for_implicit_protocol():
    parser = CohereCommand4ReasoningParser()
    assert parser.sanitize_when_thinking_disabled is True
    assert parser.implicit_reasoning_until_close is True
    assert parser.reasoning_end_str == THINK_END


class TestChatRouteStreaming:
    @staticmethod
    def _read_channels(response_text: str) -> tuple[str, str]:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for event in response_text.split("\n\n"):
            for line in event.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("content"):
                        content_parts.append(delta["content"])
        return "".join(reasoning_parts), "".join(content_parts)

    @pytest.mark.parametrize(
        (
            "deltas",
            "finish_reason",
            "emit_terminal",
            "reasoning_max_tokens",
            "expected_reasoning",
            "expected_content",
        ),
        [
            (
                ["Provide answer: 4.", THINK_END, TEXT_START, "4", TEXT_END],
                "stop",
                True,
                None,
                "Provide answer: 4.",
                "4",
            ),
            (
                ["deliberating", " about it<|END_THI"],
                "length",
                True,
                None,
                "deliberating about it<|END_THI",
                None,
            ),
            (
                ["plan", THINK_END, TEXT_START, "answer<|END_TE"],
                "length",
                True,
                None,
                "plan",
                "answer<|END_TE",
            ),
            (["<|END_THI"], None, False, None, "<|END_THI", None),
            (["abcd<|END_THI"], "length", True, 1, "abcd", "<|END_THI"),
        ],
    )
    def test_server_sse_protocol_and_eof_drain(
        self,
        monkeypatch,
        deltas,
        finish_reason,
        emit_terminal,
        reasoning_max_tokens,
        expected_reasoning,
        expected_content,
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes import chat as chat_route

        # This HTTP contract belongs to the no-MLX CI lane. Admission is
        # orthogonal to protocol parsing and imports the MLX scheduler lazily,
        # so replace only that boundary with the same successful no-op used by
        # the lightweight route harnesses.
        monkeypatch.setattr(chat_route, "_check_admission_or_503", lambda _engine: None)

        class Engine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def stream_chat(self, messages, **kwargs):
                accumulated = ""
                for index, delta in enumerate(deltas):
                    accumulated += delta
                    final = emit_terminal and index == len(deltas) - 1
                    yield GenerationOutput(
                        text=accumulated,
                        new_text=delta,
                        prompt_tokens=4,
                        completion_tokens=index + 1,
                        finished=final,
                        finish_reason=finish_reason if final else None,
                    )

        config = reset_config()
        try:
            config.engine = Engine()
            config.model_name = "cohere-command-test"
            config.model_registry = None
            config.reasoning_parser = CohereCommand4ReasoningParser()
            config.reasoning_parser_name = "cohere_command4"
            config.tool_parser = None
            config.no_thinking = False

            app = FastAPI()
            app.include_router(chat_route.router)
            payload = {
                "model": "cohere-command-test",
                "messages": [{"role": "user", "content": "2+2?"}],
                "stream": True,
                "max_tokens": 100,
            }
            if reasoning_max_tokens is not None:
                payload["reasoning_max_tokens"] = reasoning_max_tokens
            response = TestClient(app).post(
                "/v1/chat/completions",
                json=payload,
            )
            assert response.status_code == 200
            reasoning, content = self._read_channels(response.text)
            assert reasoning == expected_reasoning
            if expected_content is not None:
                assert content == expected_content
        finally:
            reset_config()
