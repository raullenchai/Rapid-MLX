# SPDX-License-Identifier: Apache-2.0
"""MLLM reasoning budgets reuse the text lane's generation-time processor."""

from types import SimpleNamespace

import pytest

from rapid_mlx.api.reasoning_budget import ReasoningBudgetLogitsProcessor
from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.routes.chat import (
    _build_reasoning_budget_processor,
    _effective_posthoc_reasoning_cap,
)


class Tokenizer:
    chat_template = "test template"
    all_special_ids = [50, 99]

    def __init__(self, prefix):
        self.prefix = prefix
        self.renders = []

    def encode(self, text, **kwargs):
        return [{"<think>": 50, "</think>": 99}.get(text, 1)]

    def decode(self, ids, **kwargs):
        return {50: "<think>", 99: "</think>"}.get(ids[0], "token")

    def apply_chat_template(self, messages, **kwargs):
        self.renders.append((messages, kwargs))
        body = repr(messages)
        return body + (self.prefix if kwargs["add_generation_prompt"] else "")


def make_engine(prefix="assistant\n<think>\n"):
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = True
    engine._loaded = True
    engine._model_name = "mlx-community/Qwen3.5-4B-MLX-4bit"
    engine._processor = Tokenizer(prefix)
    engine._model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=SimpleNamespace(shape=(128, 8)))
            )
        )
    )
    return engine


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("seeded", [False, True])
def test_mllm_builds_budget_from_real_template(media, seeded):
    engine = make_engine("assistant\n<think>\n" if seeded else "assistant\n")
    content = "User supplied <think> is not a generation prefix"
    if media:
        content = [
            {"type": "text", "text": content},
            {"type": "image_url", "image_url": {"url": "https://example.com/image"}},
        ]
    request = SimpleNamespace(reasoning_max_tokens=2, tools=None, stop=None)
    cfg = SimpleNamespace(model_path=engine._model_name, reasoning_parser_name="qwen3")
    proc = _build_reasoning_budget_processor(
        engine, request, cfg, [{"role": "user", "content": content}], True
    )
    assert isinstance(proc, ReasoningBudgetLogitsProcessor), engine.tokenizer.renders
    assert proc._started is seeded
    assert (
        _effective_posthoc_reasoning_cap(
            {"reasoning_budget_logits_processor": proc}, request
        )
        is None
    )
    renders = engine.tokenizer.renders
    assert [kw["add_generation_prompt"] for _, kw in renders] == [True, False]
    assert all(kw["enable_thinking"] is True for _, kw in renders)
    if media:
        assert renders[0][0][-1]["content"][1]["type"] == "image"


@pytest.mark.parametrize("thinking,stop", [(False, None), (True, ["</think>"])])
def test_mllm_retains_budget_safety_gates(thinking, stop):
    engine = make_engine()
    request = SimpleNamespace(reasoning_max_tokens=2, tools=None, stop=stop)
    cfg = SimpleNamespace(model_path=engine._model_name, reasoning_parser_name="qwen3")
    assert (
        _build_reasoning_budget_processor(
            engine, request, cfg, [{"role": "user", "content": "hi"}], thinking
        )
        is None
    )
    assert not engine.tokenizer.renders


@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("surface", ["chat", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("seeded", [False, True])
def test_mllm_budget_closes_reasoning_before_answer(
    monkeypatch, surface, stream, seeded, finish
):
    """Exercise route setup and MLLM processor history without tensor operations.

    Only the numerical force mask and model sampler are replaced: the real
    budget state machine must force the close before our model emits an answer.
    """
    import json

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.config import reset_config
    from rapid_mlx.engine.base import GenerationOutput
    from rapid_mlx.mllm_batch_generator import _apply_request_logits_processors
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.routes import anthropic, chat
    from tests.test_reasoning_boundary_routes import norm
    from tests.test_response_format_json_schema_strict import _Engine

    rendered = make_engine("assistant\n<think>\n" if seeded else "assistant\n")
    processors = []

    class Engine(_Engine):
        is_mllm = True
        tokenizer = rendered.tokenizer
        _model = rendered._model
        build_prompt = rendered.build_prompt

        def decode_budget(self, kwargs):
            processor = kwargs.get("reasoning_budget_logits_processor")
            assert isinstance(processor, ReasoningBudgetLogitsProcessor)
            processors.append(processor)
            # Return the forced token id in place of constructing an MLX mask.
            monkeypatch.setattr(processor, "_force_distribution", lambda logits: 99)
            monkeypatch.setattr(processor, "_reopen_suppressor", None)
            req = SimpleNamespace(
                logits_processors=[processor], logits_processor_tokens=[]
            )
            tokens = req.logits_processor_tokens
            pieces = [] if seeded else ["<think>"]
            for _ in range(8):
                forced = _apply_request_logits_processors(req, None)
                if forced == 99:
                    token, piece = 99, "</think>"
                elif processor._ended:
                    token, piece = 20, "Answer: 4."
                elif not processor._started:
                    token, piece = 50, "<think>"
                    pieces = []
                else:
                    token, piece = 10, "Private thought. "
                tokens.append(token)
                pieces.append(piece)
                if token == 20:
                    break
            assert tokens.count(10) == 2
            assert 99 in tokens
            return "".join(pieces)

        async def chat(self, **kwargs):
            text = self.decode_budget(kwargs)
            return GenerationOutput(
                text=text,
                raw_text=text,
                finished=True,
                finish_reason=finish,
                prompt_tokens=4,
                completion_tokens=4,
            )

        async def stream_chat(self, **kwargs):
            text = self.decode_budget(kwargs)
            cumulative = ""
            for char in text:
                cumulative += char
                yield GenerationOutput(text=cumulative, new_text=char, finished=False)
            yield GenerationOutput(
                text=text,
                finished=True,
                finish_reason=finish,
                prompt_tokens=4,
                completion_tokens=4,
            )

    cfg = reset_config()
    cfg.model_name = rendered._model_name
    cfg.engine = Engine(supports_guided=False)
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    app = FastAPI()
    app.include_router((chat if surface == "chat" else anthropic).router)
    body = {
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": "Solve"}],
        "max_tokens": 64,
        "stream": stream,
    }
    if surface == "chat":
        body.update(reasoning_max_tokens=2, enable_thinking=True)
        url = "/v1/chat/completions"
    else:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2}
        url = "/v1/messages"
    try:
        with TestClient(app) as client:
            response = client.post(url, json=body)
        assert response.status_code == 200, response.text
        packets = (
            [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            if stream
            else [response.json()]
        )
        result = norm(packets, surface, stream)
        assert result["content"].strip() == "Answer: 4.", packets
        assert result["reasoning"].strip() == "Private thought. Private thought."
        assert len(processors) == 1
    finally:
        reset_config()


@pytest.mark.parametrize("mllm,budget", [(False, 2), (True, None)])
def test_anthropic_budget_attachment_keeps_unbudgeted_requests(mllm, budget):
    from rapid_mlx.config import reset_config
    from rapid_mlx.routes.anthropic import _attach_mllm_reasoning_budget

    engine = make_engine()
    engine._is_mllm = mllm
    cfg = reset_config()
    cfg.model_name = engine._model_name
    request = SimpleNamespace(reasoning_max_tokens=budget, tools=None, stop=None)
    kwargs = {"enable_thinking": True}
    try:
        _attach_mllm_reasoning_budget(
            engine, request, [{"role": "user", "content": "hi"}], kwargs
        )
        assert kwargs == {"enable_thinking": True}
        assert not engine._processor.renders
    finally:
        reset_config()
