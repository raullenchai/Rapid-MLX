"""Response-format constraints must preserve system prefixes and reasoning."""

import json
import logging

import pytest

from rapid_mlx.api.tool_calling import build_json_system_prompt
from rapid_mlx.config import get_config
from rapid_mlx.service.helpers import _inject_json_instruction
from tests.test_response_format_json_schema_strict import (
    _Engine,
    _make_client,
    _payload,
)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("format_type", ["json_schema", "json_object"])
def test_guided_keeps_messages_unchanged(stream, format_type):
    engine = _Engine()
    client = _make_client(engine)
    payload = _payload(strict=True, stream=stream)
    if format_type == "json_object":
        payload["response_format"] = {"type": "json_object"}
    payload["messages"].insert(0, {"role": "system", "content": "<|think|>My rules"})
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert engine.guided_calls[0]["messages"] == payload["messages"]


def test_fallback_preserves_prefix_and_logs(caplog):
    engine = _Engine(supports_guided=False)
    client = _make_client(engine)
    payload = _payload(strict=False)
    payload["messages"].insert(0, {"role": "system", "content": "<|think|>My rules"})
    with caplog.at_level(logging.INFO):
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    content = engine.chat_calls[0]["messages"][0]["content"]
    assert content.startswith("<|think|>My rules\n\n")
    lines = [
        r.message
        for r in caplog.records
        if "prompt augmented for response_format" in r.message
    ]
    assert len(lines) == 1
    assert "unavailable" in lines[0]


@pytest.mark.parametrize("format_type", ["json_object", "json_schema"])
@pytest.mark.parametrize("strict", [False, True])
def test_instruction_allows_reasoning(format_type, strict):
    response_format = _payload(strict=strict)["response_format"]
    response_format["type"] = format_type
    instruction = build_json_system_prompt(response_format)
    assert "NO thinking" not in instruction
    assert "NO text before" not in instruction
    assert "final answer" in instruction


def test_injection_does_not_mutate_original():
    messages = [{"role": "system", "content": "<|think|>Rules"}]
    result = _inject_json_instruction(messages, "JSON please")
    assert messages[0]["content"] == "<|think|>Rules"
    assert result[0]["content"] == "<|think|>Rules\n\nJSON please"


@pytest.mark.parametrize("typed_message", [False, True])
def test_injection_appends_block_without_mutating_original(typed_message):
    from copy import deepcopy

    from rapid_mlx.api.models import Message

    blocks = [
        {"type": "text", "text": "<|think|>First rule"},
        {"type": "text", "text": "Second rule"},
    ]
    original_blocks = deepcopy(blocks)
    system = (
        Message(role="system", content=blocks)
        if typed_message
        else {"role": "system", "content": blocks}
    )
    messages = [{"role": "user", "content": "Earlier turn"}, system]
    original = deepcopy(messages)
    original_content = system.content if typed_message else system["content"]

    result = _inject_json_instruction(messages, "JSON please")

    content = result[1].content if typed_message else result[1]["content"]
    assert result[0] == messages[0]
    assert len(result) == len(messages)
    assert content == [*original_content, {"type": "text", "text": "JSON please"}]
    assert messages == original
    assert blocks == original_blocks
    assert result[1] is not system
    assert content is not original_content


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failing_method", ["encode", "decode"])
def test_reasoning_gate_tokenizer_failure_uses_logged_fallback(
    monkeypatch, caplog, stream, failing_method
):
    from unittest.mock import Mock

    from rapid_mlx.api import guided
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    class Tokenizer:
        def encode(self, text, **kwargs):
            if text == "</think>":
                if failing_method == "encode":
                    raise RuntimeError("marker encode failed")
                return [7]
            return [1, 2]

        def decode(self, ids):
            if ids == [7]:
                raise RuntimeError("marker decode failed")
            return "prompt"

    engine = _Engine(chat_text='<think>Keep this reasoning.</think>{"value":42}')
    engine.tokenizer = Tokenizer()
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    build = Mock()
    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build)
    payload = _payload(strict=True, stream=stream)
    payload["enable_thinking"] = True
    payload["messages"].insert(0, {"role": "system", "content": "<|think|>My rules"})

    with caplog.at_level(logging.DEBUG, logger="rapid_mlx.routes.chat"):
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    assert "Keep this reasoning." in response.text
    assert not engine.guided_calls
    build.assert_not_called()
    calls = engine.stream_calls if stream else engine.chat_calls
    assert len(calls) == 1
    assert calls[0]["messages"][0]["content"].startswith("<|think|>My rules\n\n")
    assert "final answer" in calls[0]["messages"][0]["content"]
    assert calls[0]["kwargs"]["enable_thinking"] is True
    assert "grammar_logits_processor" not in calls[0]["kwargs"]
    assert "Unable to resolve response_format reasoning gate" in caplog.text
    assert f"marker {failing_method} failed" in caplog.text
    fallback_logs = [
        r.message
        for r in caplog.records
        if "prompt augmented for response_format" in r.message
    ]
    assert len(fallback_logs) == 1
    assert "no supported atomic reasoning-end marker" in fallback_logs[0]


@pytest.mark.parametrize("repair_content", ['{"value":42}', '{"value":200}'])
def test_strict_repair_validates_extracted_content(monkeypatch, repair_content):
    from unittest.mock import Mock

    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.routes import chat as chat_route

    initial_content = '{"value":0}'

    class RepairingEngine(_Engine):
        async def chat(self, *, messages, **kwargs):
            content = repair_content if self.chat_calls else initial_content
            self._chat_text = f"<think>Reasoning for this attempt.</think>{content}"
            return await super().chat(messages=messages, **kwargs)

    engine = RepairingEngine(supports_guided=False)
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    validate = Mock(wraps=chat_route.validate_and_envelope)
    monkeypatch.setattr(chat_route, "validate_and_envelope", validate)
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR", "on")
    payload = _payload(strict=True)
    payload["enable_thinking"] = True

    response = client.post("/v1/chat/completions", json=payload)

    assert len(engine.chat_calls) == 2
    assert not engine.guided_calls
    assert [call.args[0] for call in validate.call_args_list] == [
        initial_content,
        repair_content,
    ]
    if repair_content == '{"value":42}':
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert json.loads(message["content"]) == json.loads(repair_content)
        assert message["reasoning_content"] == "Reasoning for this attempt."
    else:
        assert response.status_code == 422, response.text
        details = response.json()["error"]["details"]
        assert details["attempts"] == 2
        assert details["reason"] == "schema_violation"
        assert details["failing_path"] == "/value"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("gate_available", [False, True])
@pytest.mark.parametrize("explicit_thinking", [False, True])
def test_reasoning_preserved_with_schema(
    monkeypatch, stream, gate_available, explicit_thinking
):
    from rapid_mlx.api import guided
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    raw = '<think>Let me reason carefully.</think>{"value":42}'
    engine = _Engine(supports_guided=True, chat_text=raw)

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [7] if text == "</think>" else [1, 2]

        def decode(self, ids):
            return "</think>" if ids == [7] else "prompt"

    if gate_available:
        engine.tokenizer = Tokenizer()
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    processor = object()
    builds = []

    def build(tokenizer, schema, **kwargs):
        builds.append(kwargs)
        return processor

    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build)
    payload = _payload(strict=True, stream=stream)
    if explicit_thinking:
        payload["enable_thinking"] = True
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert "Let me reason carefully." in response.text
    assert not engine.guided_calls
    call = (engine.stream_calls if stream else engine.chat_calls)[0]
    assert call["kwargs"].get("enable_thinking") is (
        True if explicit_thinking else None
    )
    if gate_available:
        assert builds == [{"reasoning_end_id": 7}]
        assert call["messages"] == payload["messages"]
        assert call["kwargs"]["grammar_logits_processor"] is processor
    else:
        assert not builds
        assert "final answer" in call["messages"][0]["content"]


def test_atomic_reasoning_gate_excludes_reasoning_and_boundary(monkeypatch):
    """Drive the existing processor across a boundary and a batched tail."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np

    from rapid_mlx.api import tool_grammar

    matcher = Mock()
    matcher.get_error.return_value = None
    matcher.is_stopped.return_value = False
    matcher.is_accepting.return_value = False
    matcher.consume_token.return_value = True
    monkeypatch.setattr(tool_grammar, "get_request_matcher", lambda *args: matcher)
    monkeypatch.setattr(tool_grammar, "allocate_token_bitmask", lambda *args: object())
    monkeypatch.setattr(tool_grammar, "fill_next_token_bitmask", Mock())
    monkeypatch.setattr(
        tool_grammar, "apply_token_bitmask", lambda logits, mask: logits - 1
    )
    processor = tool_grammar.GrammarLogitsProcessor(
        SimpleNamespace(vocab_size=16), "grammar", reasoning_end_id=7
    )
    logits = np.zeros((1, 16))
    assert processor([10, 11], logits) is logits  # prompt, including cached prefix
    assert processor([10, 11, 4, 5], logits) is logits  # free reasoning
    matcher.consume_token.assert_not_called()
    masked = processor([10, 11, 4, 5, 7, 8, 9], logits)
    assert [call.args[0] for call in matcher.consume_token.call_args_list] == [8, 9]
    assert np.all(masked == -1)


@pytest.mark.parametrize("stream", [False, True])
def test_gemma_channel_reasoning_uses_logged_fallback(stream, caplog):
    from rapid_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser

    raw = '<|channel>thought\nConsider the answer.<channel|><|channel>content\n{"value":42}<channel|>'
    engine = _Engine(chat_text=raw)
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "gemma4"
    cfg.reasoning_parser = Gemma4ReasoningParser()
    payload = _payload(strict=True, stream=stream)
    payload["messages"].insert(0, {"role": "system", "content": "<|think|>My rules"})
    with caplog.at_level(logging.INFO):
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert "Consider the answer." in response.text
    call = (engine.stream_calls if stream else engine.chat_calls)[0]
    assert call["messages"][0]["content"].startswith("<|think|>My rules\n\n")
    assert call["kwargs"].get("enable_thinking") is not False
    assert "no supported atomic reasoning-end marker" in caplog.text
    assert not engine.guided_calls


@pytest.mark.parametrize("stream", [False, True])
def test_runtime_guided_failure_augments_only_fallback(stream, caplog):
    engine = _Engine(guided_raises=RuntimeError("backend unavailable"))
    client = _make_client(engine)
    payload = _payload(strict=False, stream=stream)
    payload["messages"].insert(0, {"role": "system", "content": "<|think|>My rules"})
    with caplog.at_level(logging.INFO):
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert engine.guided_calls[0]["messages"] == payload["messages"]
    call = (engine.stream_calls if stream else engine.chat_calls)[0]
    assert call["messages"][0]["content"].startswith("<|think|>My rules\n\n")
    assert (
        sum("prompt augmented for response_format" in r.message for r in caplog.records)
        == 1
    )
    assert "failed at runtime" in caplog.text
