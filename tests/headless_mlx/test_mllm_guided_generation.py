# SPDX-License-Identifier: Apache-2.0
"""Guided text decoding on vision engines, without loading model weights."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rapid_mlx.engine import batched
from rapid_mlx.engine.base import GenerationOutput
from tests.test_schema_reasoning_handoff_5702 import setup_processor as setup_processor


@pytest.mark.parametrize("available", [False, True])
def test_mllm_advertises_guided_text_support(monkeypatch, available):
    monkeypatch.setattr(batched, "HAS_GUIDED", available)
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._is_mllm = True
    assert engine.supports_guided_generation is available


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_end_id", [None, 7])
async def test_mllm_generate_with_schema_preserves_media_and_processor(
    monkeypatch, reasoning_end_id
):
    from rapid_mlx.api import guided

    monkeypatch.setattr(batched, "HAS_GUIDED", True)
    marker = object()
    tokenizer = object()
    build_processor = Mock(return_value=marker)
    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build_processor)
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._is_mllm = True
    engine._loaded = True
    engine._processor = SimpleNamespace(tokenizer=tokenizer)
    output = GenerationOutput(text='{"label":"cat"}')
    engine.chat = AsyncMock(return_value=output)
    messages = [{"role": "user", "content": "classify"}]
    images = ["data:image/png;base64,FAKE"]

    budget = object()
    assert (
        await engine.generate_with_schema(
            messages,
            {"type": "object"},
            images=images,
            max_tokens=12,
            temperature=0.2,
            top_p=0.8,
            reasoning_budget_logits_processor=budget,
            reasoning_end_id=reasoning_end_id,
            enable_thinking=reasoning_end_id is not None,
        )
        is output
    )
    kwargs = engine.chat.call_args.kwargs
    build_processor.assert_called_once_with(
        tokenizer, {"type": "object"}, reasoning_end_id=reasoning_end_id
    )
    assert kwargs["messages"] is messages
    assert kwargs["images"] is images
    assert kwargs["grammar_logits_processor"] is marker
    assert kwargs["enable_thinking"] is (reasoning_end_id is not None)
    if reasoning_end_id is None:
        assert "reasoning_budget_logits_processor" not in kwargs
    else:
        assert kwargs["reasoning_budget_logits_processor"] is budget
    assert (kwargs["max_tokens"], kwargs["temperature"], kwargs["top_p"]) == (
        12,
        0.2,
        0.8,
    )


@pytest.mark.asyncio
async def test_mllm_schema_setup_failure_never_silently_falls_back(monkeypatch):
    from rapid_mlx.api import guided

    monkeypatch.setattr(batched, "HAS_GUIDED", True)
    monkeypatch.setattr(
        guided, "build_json_schema_logits_processor", lambda *_, **__: None
    )
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._is_mllm = True
    engine._loaded = True
    engine._processor = SimpleNamespace(tokenizer=object())
    engine.chat = AsyncMock()
    with pytest.raises(RuntimeError, match="processor"):
        await engine.generate_with_schema([], {"type": "object"})
    engine.chat.assert_not_called()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", ["effort", "enabled", "disabled", "template"])
def test_mllm_reasoning_schema_uses_shared_gate(monkeypatch, strict, stream, thinking):
    """The reporter's reasoning-effort request keeps thinking and its budget."""
    from rapid_mlx.api import guided
    from rapid_mlx.config import get_config
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.routes import chat
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [7] if text == "</think>" else [1, 2]

        def decode(self, ids):
            return "</think>" if ids == [7] else "prompt"

    raw = (
        '{"value":42}'
        if thinking == "disabled"
        else '<think>Reasoning.</think>{"value":42}'
    )
    engine = _Engine(supports_guided=False, chat_text=raw)
    engine.is_mllm = True
    engine.tokenizer = Tokenizer()
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    marker, budget = object(), object()
    build = Mock(return_value=marker)
    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build)
    monkeypatch.setattr(
        chat, "_build_reasoning_budget_processor", lambda *a, **k: budget
    )
    payload = _payload(strict=strict, stream=stream)
    if thinking == "effort":
        payload["reasoning_effort"] = "high"
    elif thinking == "template":
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["enable_thinking"] = thinking == "enabled"
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert not engine.guided_calls
    call = (engine.stream_calls if stream else engine.chat_calls)[0]
    assert call["messages"] == payload["messages"]
    assert call["kwargs"]["grammar_logits_processor"] is marker
    if thinking == "disabled":
        assert not build.call_args.kwargs
        assert call["kwargs"]["enable_thinking"] is False
        assert "reasoning_budget_logits_processor" not in call["kwargs"]
    else:
        assert build.call_args.kwargs == {"reasoning_end_id": 7}
        assert call["kwargs"].get("enable_thinking") is not False
        assert call["kwargs"]["reasoning_budget_logits_processor"] is budget
        assert "Reasoning." in response.text
    assert "X-RapidMLX-Warning" not in response.headers


@pytest.mark.parametrize("stream", [False, True])
def test_mllm_non_atomic_reasoning_warns_without_disabling_thinking(
    monkeypatch, caplog, stream
):
    from rapid_mlx.api import guided
    from rapid_mlx.config import get_config
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    engine = _Engine(
        supports_guided=False, chat_text='<think>Reasoning.</think>{"value":42}'
    )
    engine.is_mllm = True
    engine.tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 2])
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    monkeypatch.setattr(guided, "is_guided_available", lambda: True)
    build = Mock()
    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build)
    payload = _payload(strict=True, stream=stream)
    payload["reasoning_effort"] = "high"
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert "atomic reasoning-end marker" in response.headers["X-RapidMLX-Warning"]
    assert "unconstrained" in caplog.text
    build.assert_not_called()
    call = (engine.stream_calls if stream else engine.chat_calls)[0]
    assert call["kwargs"].get("enable_thinking") is not False
    assert "grammar_logits_processor" not in call["kwargs"]
    assert "Reasoning." in response.text


@pytest.mark.asyncio
async def test_direct_mllm_guided_reasoning_requires_gate(monkeypatch):
    monkeypatch.setattr(batched, "HAS_GUIDED", True)
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._is_mllm = True
    engine._loaded = True
    engine.chat = AsyncMock()
    with pytest.raises(RuntimeError, match="reasoning_end_id"):
        await engine.generate_with_schema([], {"type": "object"}, enable_thinking=True)
    engine.chat.assert_not_called()


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_mllm_tools_preserve_shared_policy_and_warn(monkeypatch, strict, stream):
    from rapid_mlx.config import get_config
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    engine = _Engine()
    engine.is_mllm = True
    client = _make_client(engine)
    get_config().tool_call_parser = None
    payload = _payload(strict=strict, stream=stream)
    payload["tools"] = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    payload["tool_choice"] = "auto"
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    assert "with tools" in response.headers["X-RapidMLX-Warning"]
    assert not engine.guided_calls


def test_mllm_invalid_strict_schema_rejected_before_generation():
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    engine = _Engine()
    engine.is_mllm = True
    client = _make_client(engine)
    payload = _payload(strict=True)
    payload["response_format"]["json_schema"]["schema"] = {"type": "objct"}
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400, response.text
    assert not engine.chat_calls and not engine.stream_calls and not engine.guided_calls


def test_mllm_failed_strict_repair_retains_degradation_warning(monkeypatch, caplog):
    from rapid_mlx.api import guided
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    engine = _Engine(supports_guided=False, chat_text='{"wrong":42}')
    engine.is_mllm = True
    client = _make_client(engine)
    monkeypatch.setattr(
        guided, "build_json_schema_logits_processor", lambda *a, **k: object()
    )
    response = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert response.status_code == 422, response.text
    assert len(engine.chat_calls) == 2
    assert "repair is unconstrained" in response.headers["X-RapidMLX-Warning"]
    assert "repair is unconstrained" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parser_configured,marker_resolves", [(False, False), (True, False), (True, True)]
)
async def test_direct_mllm_implicit_thinking_resolves_configured_gate(
    monkeypatch, parser_configured, marker_resolves
):
    from rapid_mlx.api import guided
    from rapid_mlx.config import reset_config
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = reset_config()
    cfg.reasoning_parser = Qwen3ReasoningParser() if parser_configured else None
    tokenizer = SimpleNamespace(
        encode=lambda *a, **k: [7] if marker_resolves else [1, 2],
        decode=lambda ids: "</think>",
    )
    build = Mock(return_value=object())
    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build)
    monkeypatch.setattr(batched, "HAS_GUIDED", True)
    engine = batched.BatchedEngine.__new__(batched.BatchedEngine)
    engine._is_mllm = True
    engine._loaded = True
    engine._processor = SimpleNamespace(tokenizer=tokenizer)
    engine.chat = AsyncMock()
    if parser_configured and not marker_resolves:
        with pytest.raises(RuntimeError, match="atomic reasoning_end_id"):
            await engine.generate_with_schema([], {"type": "object"})
        build.assert_not_called()
        engine.chat.assert_not_called()
    else:
        await engine.generate_with_schema([], {"type": "object"})
        assert build.call_args.kwargs["reasoning_end_id"] == (
            7 if parser_configured else None
        )
        engine.chat.assert_awaited_once()
    reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["", "not JSON", '{"value":200}'])
def test_responses_guided_schema_violation_uses_validation_envelope(
    monkeypatch, stream, text
):
    from rapid_mlx.api.strict_json_schema import (
        build_violation_envelope,
        validate_and_envelope,
    )
    from rapid_mlx.config import get_config
    from rapid_mlx.middleware.auth import rate_limiter
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_responses_client,
        _responses_payload,
    )

    monkeypatch.setattr(rate_limiter, "enabled", False)
    raw = f"<think>Reasoning.</think>{text}"
    engine = _Engine(guided_text=raw)
    engine.is_mllm = True
    engine.tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: [7], decode=lambda ids: "</think>"
    )
    client = _make_responses_client(engine, rate_limiter)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"
    payload = _responses_payload(strict=True, stream=stream)
    payload["enable_thinking"] = True
    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 422, response.text
    _, details = validate_and_envelope(text, payload["text"]["format"]["schema"])
    assert response.json() == build_violation_envelope(
        details, param="text.format", attempts=1
    )
    assert engine.guided_calls[0]["kwargs"]["reasoning_end_id"] == 7


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("explicit_thinking", [False, True])
def test_mllm_strict_thinking_decodes_through_shared_gate(
    monkeypatch, setup_processor, stream, explicit_thinking
):
    """Run real grammar masks with MLLM history through chat postprocessing."""
    import json

    import numpy as np

    from rapid_mlx.api import guided
    from rapid_mlx.config import get_config
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from tests.test_response_format_json_schema_strict import (
        _Engine,
        _make_client,
        _payload,
    )

    env = setup_processor
    processor = env.build()
    reasoning = "Let me reason."
    raw = f'{reasoning}</think>\n\n{{"answer":"4"}}'

    class Engine(_Engine):
        is_mllm = True
        tokenizer = SimpleNamespace(
            encode=lambda text, **kw: env.encode(text), decode=env.decode
        )

        def decode_outputs(self, kwargs):
            assert kwargs["grammar_logits_processor"] is processor
            history = []
            for token in env.encode(raw):
                masked = processor(history, env.logits())
                # The model's intended reasoning and answer must both remain
                # sampleable; constraining token zero forbids the first letter.
                assert np.isfinite(masked[0, token])
                history.append(token)
                yield GenerationOutput(
                    text=env.decode(history),
                    new_text=env.decode([token]),
                    prompt_tokens=4,
                    completion_tokens=len(history),
                    finished=False,
                )
            processor(history, env.logits())
            assert processor._reasoning_ended
            assert processor._matcher.is_accepting()
            assert not processor._aborted
            yield GenerationOutput(
                text=raw,
                new_text="",
                prompt_tokens=4,
                completion_tokens=len(history),
                finished=True,
                finish_reason="stop",
            )

        async def stream_chat(self, messages, **kwargs):
            for output in self.decode_outputs(kwargs):
                yield output

        async def chat(self, *, messages, **kwargs):
            return list(self.decode_outputs(kwargs))[-1]

    engine = Engine()
    client = _make_client(engine)
    cfg = get_config()
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    build = Mock(return_value=processor)
    monkeypatch.setattr(guided, "build_json_schema_logits_processor", build)
    payload = _payload(strict=True, stream=stream)
    payload["response_format"]["json_schema"]["schema"] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    if explicit_thinking:
        payload["enable_thinking"] = True
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    build.assert_called_once_with(
        engine.tokenizer,
        payload["response_format"]["json_schema"]["schema"],
        reasoning_end_id=env.end_id,
    )
    if stream:
        chunks = [
            json.loads(line[6:])["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        content = "".join(chunk["delta"].get("content") or "" for chunk in chunks)
        thinking = "".join(
            chunk["delta"].get("reasoning_content") or "" for chunk in chunks
        )
        assert chunks[-1]["finish_reason"] == "stop"
    else:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        thinking = choice["message"]["reasoning_content"]
        assert choice["finish_reason"] == "stop"
    assert json.loads(content) == {"answer": "4"}
    assert thinking.strip() == reasoning
    assert not engine.guided_calls
