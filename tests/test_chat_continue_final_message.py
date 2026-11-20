# SPDX-License-Identifier: Apache-2.0
"""Assistant-prefill continuation without model weights."""

import json

import jinja2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rapid_mlx.api.models import ChatCompletionRequest, ContentPart
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.prompt_host_cache import PromptHostCache
from rapid_mlx.reasoning import get_parser
from rapid_mlx.routes import chat
from rapid_mlx.utils.chat_template import (
    apply_chat_template,
    normalize_assistant_content,
)
from tests.test_rejected_tool_diagnostics import TOOLS, Engine

TEMPLATE = (
    "{% for m in messages %}{{ m.role }}:{{ m.content }}<end>{% endfor %}"
    "{% if add_generation_prompt %}assistant:<think>{% endif %}"
)
PREFILL = "Here is a solution...\n```python\n"
MESSAGES = [
    {"role": "user", "content": "Write a solution."},
    {"role": "assistant", "content": PREFILL},
]
OPTIONS = {"add_generation_prompt": False, "continue_final_message": True}


def tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=TEMPLATE)


class LegacyApplicator:
    chat_template = TEMPLATE

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return (
            jinja2.Environment()
            .from_string(self.chat_template)
            .render(messages=messages, add_generation_prompt=add_generation_prompt)
        )


def engine(tok):
    result = BatchedEngine("test-model")
    result._loaded = True
    result._tokenizer = tok
    result._is_mllm = False
    result._processor = None
    result._prompt_host_cache = PromptHostCache(enabled=True)
    return result


def test_request_defaults_and_validation():
    request = ChatCompletionRequest(model="test-model", messages=MESSAGES)
    assert request.add_generation_prompt is True
    assert request.continue_final_message is False
    assert ChatCompletionRequest(model="test-model", messages=MESSAGES, **OPTIONS)
    with pytest.raises(ValidationError, match="add_generation_prompt=False"):
        ChatCompletionRequest(
            model="test-model", messages=MESSAGES, continue_final_message=True
        )
    with pytest.raises(ValidationError, match="final assistant"):
        ChatCompletionRequest(model="test-model", messages=MESSAGES[:1], **OPTIONS)


@pytest.mark.parametrize("legacy", [False, True])
def test_closing_template_continues_after_prefill(legacy):
    tok = LegacyApplicator() if legacy else tokenizer()
    prompt = apply_chat_template(tok, MESSAGES, **OPTIONS)
    assert prompt == "user:Write a solution.<end>assistant:" + PREFILL
    assert prompt.endswith(PREFILL)
    assert apply_chat_template(tok, MESSAGES).endswith("<end>assistant:<think>")
    assert apply_chat_template(tok, MESSAGES, add_generation_prompt=False).endswith(
        PREFILL + "<end>"
    )


def test_fallback_uses_last_occurrence_and_normalized_text_parts():
    history = [
        {"role": "user", "content": PREFILL},
        {"role": "assistant", "content": [{"type": "text", "text": PREFILL}]},
    ]
    prompt = apply_chat_template(LegacyApplicator(), history, **OPTIONS)
    assert prompt == "user:" + PREFILL + "<end>assistant:" + PREFILL


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("content", ["end", "<", "<end>", "", []])
def test_continuation_boundary_excludes_terminator(legacy, content):
    tok = LegacyApplicator() if legacy else tokenizer()
    history = [*MESSAGES[:1], {"role": "assistant", "content": content}]
    assert apply_chat_template(tok, history, **OPTIONS) == (
        "user:Write a solution.<end>assistant:" + (content if content else "")
    )


@pytest.mark.parametrize("part_class", [dict, ContentPart])
def test_continuation_normalizes_multiple_text_parts(part_class):
    content = [
        part_class(type="text", text="<think>"),
        part_class(type="text", text="Start"),
    ]
    assert normalize_assistant_content(content) == "<think>Start"
    history = [*MESSAGES[:1], {"role": "assistant", "content": content}]
    assert apply_chat_template(LegacyApplicator(), history, **OPTIONS) == (
        "user:Write a solution.<end>assistant:<think>Start"
    )


def test_structural_boundary_uses_history_generation_header():
    class MatchingHeader(LegacyApplicator):
        chat_template = TEMPLATE.replace("assistant:<think>", "assistant:")

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert all(
                m["content"] in {"Write a solution.", "end", ""} for m in messages
            )
            return super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )

    history = [*MESSAGES[:1], {"role": "assistant", "content": "end"}]
    assert apply_chat_template(MatchingHeader(), history, **OPTIONS) == (
        "user:Write a solution.<end>assistant:end"
    )


def test_guarded_last_occurrence_when_probe_is_unsupported():
    class NoProbe(LegacyApplicator):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            if messages[-1]["content"].startswith("RAPID_MLX_"):
                raise ValueError("Probe content is unsupported")
            return super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )

    assert apply_chat_template(NoProbe(), MESSAGES, **OPTIONS).endswith(PREFILL)


def test_ambiguous_fallback_boundary_is_rejected():
    tok = LegacyApplicator()
    tok.chat_template = (
        "{% for m in messages %}{{ m.role }}:"
        "{% if m.content != '\u27e6continuation-boundary\u27e7' %}"
        "{{ m.content }}{% endif %}end{% endfor %}"
    )
    history = [*MESSAGES[:1], {"role": "assistant", "content": "end"}]
    with pytest.raises(ValueError, match="verifiable boundary"):
        apply_chat_template(tok, history, **OPTIONS)


def test_fallback_refuses_missing_content():
    tok = LegacyApplicator()
    tok.chat_template = "assistant:<end>"
    with pytest.raises(ValueError, match="final assistant content was not found"):
        apply_chat_template(tok, MESSAGES, **OPTIONS)


def test_render_cache_separates_continuation_and_generation_flags():
    eng = engine(tokenizer())
    closed = eng._apply_chat_template(MESSAGES, add_generation_prompt=False)
    continued = eng._apply_chat_template(MESSAGES, **OPTIONS)
    default = eng._apply_chat_template(MESSAGES)
    assert len({closed, continued, default}) == 3
    assert eng._apply_chat_template(MESSAGES, **OPTIONS) == continued
    assert eng.build_prompt(MESSAGES, **OPTIONS) == continued
    assert eng.render_request_prompt(MESSAGES, **OPTIONS) == continued


def test_client_template_kwargs_cannot_override_continuation_controls():
    tok = tokenizer()
    assert apply_chat_template(
        tok, MESSAGES, chat_template_kwargs={"continue_final_message": True}
    ).endswith("<end>assistant:<think>")
    assert apply_chat_template(
        tok,
        MESSAGES,
        **OPTIONS,
        chat_template_kwargs={
            "continue_final_message": False,
            "add_generation_prompt": True,
        },
    ).endswith(PREFILL)


class CapturingEngine(Engine):
    def __init__(self, raw):
        super().__init__(raw)
        self.tokenizer = tokenizer()
        self.options = None
        self.prompt = None

    def build_prompt(self, messages, **kwargs):
        return apply_chat_template(
            self.tokenizer,
            messages,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "add_generation_prompt",
                    "continue_final_message",
                    "enable_thinking",
                    "chat_template_kwargs",
                    "tools",
                }
            },
        )

    async def chat(self, messages, **kwargs):
        self.options = kwargs
        self.prompt = self.build_prompt(messages, **kwargs)
        return await super().chat(**kwargs)

    async def stream_chat(self, messages, **kwargs):
        self.options = kwargs
        self.prompt = self.build_prompt(messages, **kwargs)
        async for output in super().stream_chat(**kwargs):
            yield output


@pytest.fixture
def client():
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = get_parser("qwen3")()
    cfg.tool_call_parser = "glm47"
    cfg.enable_auto_tool_choice = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat.router)
    yield TestClient(app), cfg
    reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_call", [False, True])
def test_response_contains_only_new_content_outside_reasoning(
    client, stream, tool_call
):
    http, cfg = client
    raw = (
        (
            "<tool_call>run_shell_command<arg_key>command</arg_key>"
            "<arg_value>pwd</arg_value></tool_call>"
        )
        if tool_call
        else 'print("ok")\n```'
    )
    cfg.engine = CapturingEngine(raw)
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": MESSAGES,
            **OPTIONS,
            "stream": stream,
            "max_tokens": 64,
            "enable_thinking": True,
            "reasoning_max_tokens": 8,
            "tools": TOOLS,
        },
    )
    assert response.status_code == 200, response.text
    packets = (
        [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        if stream
        else [response.json()]
    )
    messages = [
        choice.get("delta", choice.get("message", {}))
        for packet in packets
        for choice in packet.get("choices", [])
    ]
    assert not any(m.get("reasoning_content") for m in messages)
    content = "".join(m.get("content") or "" for m in messages)
    if tool_call:
        assert any(m.get("tool_calls") for m in messages)
        assert PREFILL not in content
    else:
        assert content == raw
    assert cfg.engine.prompt.endswith(PREFILL)
    assert cfg.engine.options["continue_final_message"] is True
    assert cfg.engine.options["add_generation_prompt"] is False
    assert "reasoning_budget_logits_processor" not in cfg.engine.options
    assert cfg.reasoning_parser_name == "qwen3"
    assert (
        chat._template_generation_prefix(
            None,
            None,
            None,
            None,
            **{
                "continue_final_message": True,
            },
        )
        == ""
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "bad_options,bad_messages",
    [
        ({"continue_final_message": True}, MESSAGES),
        (OPTIONS, MESSAGES[:1]),
    ],
)
def test_invalid_continuation_returns_400(client, stream, bad_options, bad_messages):
    http, _ = client
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": bad_messages,
            "stream": stream,
            **bad_options,
        },
    )
    assert response.status_code == 400
    assert "continue_final_message" in response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefill", ["", [], "end", [{"type": "text", "text": "end"}]])
def test_legacy_continuation_http_boundary(client, stream, prefill):
    http, cfg = client
    cfg.engine = CapturingEngine("42")
    cfg.engine.tokenizer = LegacyApplicator()
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [*MESSAGES[:1], {"role": "assistant", "content": prefill}],
            **OPTIONS,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    assert cfg.engine.prompt == (
        "user:Write a solution.<end>assistant:"
        + ("end" if prefill in ("end", [{"type": "text", "text": "end"}]) else "")
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefill", [PREFILL, "end", ""])
def test_missing_prefill_in_legacy_template_returns_400(client, stream, prefill):
    http, cfg = client
    cfg.engine = CapturingEngine("unused")
    cfg.engine.tokenizer = LegacyApplicator()
    cfg.engine.tokenizer.chat_template = "assistant:<end>"
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [*MESSAGES[:1], {"role": "assistant", "content": prefill}],
            **OPTIONS,
            "stream": stream,
        },
    )
    assert response.status_code == 400, response.text
    assert "final assistant content was not found" in response.text


@pytest.mark.parametrize("stream", [False, True])
async def test_batched_generation_receives_continuation_prompt(stream):
    eng = engine(tokenizer())
    seen = {}

    async def generate(**kwargs):
        from rapid_mlx.engine.base import GenerationOutput

        seen.update(kwargs)
        return GenerationOutput(text="new content")

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    eng.generate = generate
    eng.stream_generate = stream_generate

    def unexpected_router():
        pytest.fail("A content continuation must not enter new-turn token routing")

    eng._create_output_router = unexpected_router
    if stream:
        outputs = [output async for output in eng.stream_chat(MESSAGES, **OPTIONS)]
        assert outputs[-1].text == "new content"
    else:
        assert (await eng.chat(MESSAGES, **OPTIONS)).text == "new content"
    assert seen["prompt"].endswith(PREFILL)
    assert "continue_final_message" not in seen
    assert "add_generation_prompt" not in seen


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize(
    "model,parser_name,generation_suffix",
    [
        ("LFM2.5-Thinking", "qwen3", ""),
        ("Qwen3", "qwen3", "<think>\n"),
        ("DeepSeek-R1-Distill-Qwen", "deepseek_r1_distill", ""),
    ],
)
@pytest.mark.parametrize(
    "prefill", ["<think>Start", "<think>Done</think>Answer:", "Answer:"]
)
def test_engine_continuation_prompt_owner(
    client,
    stream,
    chunk_size,
    finish_reason,
    model,
    parser_name,
    generation_suffix,
    prefill,
):
    """Use real engine rendering/metadata, then the route's selected owner/parser."""
    from tests.test_prompt_parse_context import Tokenizer as ByteTokenizer

    class ChatMLTokenizer(LegacyApplicator, ByteTokenizer):
        chat_template = (
            "{% for m in messages %}<|im_start|>{{ m.role }}\n"
            "{{ m.content }}<|im_end|>{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n"
            + generation_suffix
            + "{% endif %}"
        )

    http, cfg = client
    cfg.model_name = model
    cfg.reasoning_parser_name = parser_name
    cfg.reasoning_parser = get_parser(parser_name)()
    cfg.engine = eng = engine(ChatMLTokenizer())
    eng.check_admission = lambda: None  # Scheduler/GPU work is stubbed below.
    raw = "secret reasoning</think>42" if prefill == "<think>Start" else "42"
    truncated_reasoning = prefill == "<think>Start" and finish_reason == "length"
    if truncated_reasoning:
        raw = "secret reasoning"
    contexts = []

    async def generate(**kwargs):
        capture = kwargs.get("_on_prompt_prepared")
        prompt = kwargs["prompt"]
        if capture is not None:
            capture(eng.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt)
            contexts.append(capture.context)
        return GenerationOutput(
            text=raw, raw_text=raw, finished=True, finish_reason=finish_reason
        )

    async def stream_generate(**kwargs):
        await generate(**kwargs)
        for offset in range(0, len(raw), chunk_size):
            yield GenerationOutput(
                text=raw[: offset + chunk_size],
                new_text=raw[offset : offset + chunk_size],
                finished=False,
            )
        yield GenerationOutput(text=raw, finished=True, finish_reason=finish_reason)

    eng.generate = generate
    eng.stream_generate = stream_generate
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                # Historical markers must never determine continuation state.
                {"role": "user", "content": "Quote <think>fake</think>"},
                {"role": "assistant", "content": prefill},
            ],
            **OPTIONS,
            "stream": stream,
            "enable_thinking": True,
        },
    )
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
    deltas = [
        choice.get("delta", choice.get("message", {}))
        for packet in packets
        for choice in packet.get("choices", [])
    ]
    assert "".join(delta.get("content") or "" for delta in deltas) == (
        "" if truncated_reasoning else "42"
    )
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == (
        "secret reasoning" if prefill == "<think>Start" else ""
    )
    assert contexts[0].assistant_suffix == prefill
    assert contexts[0].assistant_suffix_ids == tuple(prefill.encode())
    assert contexts[0].initial_channel == (
        "reasoning"
        if prefill == "<think>Start"
        else "content"
        if prefill.startswith("<think>")
        else None
    )


def test_continuation_uses_raw_stop_matching(client):
    from rapid_mlx.service.helpers import reasoning_stop_scope_kwargs

    _, cfg = client
    cfg.engine = CapturingEngine("new content")
    request = ChatCompletionRequest(
        model="test-model", messages=MESSAGES, stop=["```"], **OPTIONS
    )
    assert reasoning_stop_scope_kwargs(cfg.engine, request) == {}


def test_continuation_after_completed_think_block():
    prefill = "<think>Already considered.</think>Here is the answer:\n"
    messages = [*MESSAGES[:1], {"role": "assistant", "content": prefill}]
    assert apply_chat_template(tokenizer(), messages, **OPTIONS).endswith(prefill)


@pytest.mark.parametrize("legacy", [False, True])
def test_template_that_never_opens_a_new_assistant_turn(legacy):
    tok = LegacyApplicator() if legacy else tokenizer()
    tok.chat_template = (
        "{% for m in messages %}{{ m.role }}:{{ m.content }}<end>{% endfor %}"
    )
    assert apply_chat_template(tok, MESSAGES).endswith(PREFILL + "<end>")
    assert apply_chat_template(tok, MESSAGES, **OPTIONS).endswith(PREFILL)


async def test_generate_keeps_channel_words_as_continuation_content():
    from types import SimpleNamespace

    eng = engine(tokenizer())
    text = "thought\nThis word belongs to the answer."

    async def generate(**kwargs):
        return SimpleNamespace(
            output_text=text,
            output_token_ids=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            cached_tokens=0,
            finish_reason="stop",
        )

    def unexpected_router():
        pytest.fail(
            "Continuation token IDs must remain outside new-turn channel detection"
        )

    eng._engine = SimpleNamespace(generate=generate)
    eng._ensure_generation_admission = lambda: (None, False)
    eng._create_output_router = unexpected_router
    output = await eng.generate("assistant:prefill", _continue_final_message=True)
    assert output.text == text
    assert not output.reasoning_text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "prefill,options,expected_reasoning",
    [
        ("<think>Let me calculate", OPTIONS, " further."),
        ("<think>Done.</think>Answer:", OPTIONS, ""),
        ("Answer:", OPTIONS, ""),
        (None, {}, " further."),
    ],
)
def test_prefill_reasoning_state(client, stream, prefill, options, expected_reasoning):
    http, cfg = client
    raw = " further.</think>42" if expected_reasoning else "42"
    cfg.engine = CapturingEngine(raw)
    messages = MESSAGES[:1]
    if prefill is not None:
        messages = [*messages, {"role": "assistant", "content": prefill}]
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": messages,
            **options,
            "stream": stream,
            "enable_thinking": True,
            "max_tokens": 64,
            "reasoning_max_tokens": 32,
        },
    )
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
    deltas = [
        choice.get("delta", choice.get("message", {}))
        for packet in packets
        for choice in packet.get("choices", [])
    ]
    # The existing non-streaming parser strips surrounding reasoning whitespace.
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == (
        expected_reasoning if stream else expected_reasoning.strip()
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == "42"


@pytest.mark.parametrize("stream", [False, True])
def test_open_prefill_keeps_posthoc_reasoning_budget(client, stream):
    http, cfg = client
    cfg.engine = CapturingEngine("abcdefghijklmnopqrstuvwxyz</think>42")
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                *MESSAGES[:1],
                {"role": "assistant", "content": "<think>Start"},
            ],
            **OPTIONS,
            "stream": stream,
            "enable_thinking": True,
            "reasoning_max_tokens": 1,
            "max_tokens": 64,
        },
    )
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
    deltas = [
        choice.get("delta", choice.get("message", {}))
        for packet in packets
        for choice in packet.get("choices", [])
    ]
    reasoning = "".join(delta.get("reasoning_content") or "" for delta in deltas)
    assert reasoning == "abcd"
    assert "".join(delta.get("content") or "" for delta in deltas).endswith("42")
    assert "reasoning_budget_logits_processor" not in cfg.engine.options


@pytest.mark.parametrize("enable_thinking", [None, False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    "prefill,seeded",
    [
        ("<think>Let me calculate", True),
        ("<think>Done.</think>Answer:", False),
        ("Answer:", False),
    ],
)
def test_continuation_decode_budget_seed(
    client, monkeypatch, enable_thinking, structured, prefill, seeded
):
    from rapid_mlx.api import reasoning_budget

    monkeypatch.setattr(reasoning_budget, "resolve_think_token_ids", lambda *_: (1, 2))
    monkeypatch.setattr(chat, "_engine_output_vocab_size", lambda _: 10)

    _, cfg = client
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            *MESSAGES[:1],
            {
                "role": "assistant",
                "content": [ContentPart(type="text", text=prefill)]
                if structured
                else prefill,
            },
        ],
        **OPTIONS,
        reasoning_max_tokens=1,
    )
    if structured:
        assert isinstance(request.messages[-1].content[0], ContentPart)
    assert chat._continuation_reasoning_prefix(request) == prefill
    processor = chat._build_reasoning_budget_processor(
        CapturingEngine("unused"), request, cfg, [], enable_thinking
    )
    assert (processor is not None) is seeded
    if seeded:
        assert processor._started
        assert (
            chat._effective_posthoc_reasoning_cap(
                {"reasoning_budget_logits_processor": processor}, request
            )
            is None
        )
        processor._commit_tail([7], 1)
        assert processor._think_count == 1
    else:
        assert chat._effective_posthoc_reasoning_cap({}, request) == 1


@pytest.mark.parametrize("parser_name", ["deepseek_r1_distill", "qwen3"])
@pytest.mark.parametrize("chunk_size", [1, 7])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_open_prefill_streaming_matches_non_streaming(
    client, parser_name, chunk_size, stream, structured
):
    http, cfg = client
    reasoning = "x" * 80
    raw = reasoning + "</think>42"

    class ChunkedEngine(CapturingEngine):
        async def stream_chat(self, messages, **kwargs):
            self.options = kwargs
            self.prompt = self.build_prompt(messages, **kwargs)
            for offset in range(0, len(self.raw), chunk_size):
                end = offset + chunk_size
                yield GenerationOutput(
                    text=self.raw[:end],
                    new_text=self.raw[offset:end],
                    prompt_tokens=4,
                    completion_tokens=min(end, len(self.raw)),
                    finished=False,
                )
            yield GenerationOutput(
                text=self.raw,
                new_text="",
                prompt_tokens=4,
                completion_tokens=len(self.raw),
                finished=True,
                finish_reason="stop",
            )

    cfg.reasoning_parser_name = parser_name
    cfg.reasoning_parser = get_parser(parser_name)()
    cfg.engine = ChunkedEngine(raw)
    if parser_name == "deepseek_r1_distill":
        cfg.engine.tokenizer.chat_template = TEMPLATE.replace(
            "assistant:<think>", "assistant:"
        )
    response = http.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                *MESSAGES[:1],
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<think>"},
                        {"type": "text", "text": "Start"},
                    ]
                    if structured
                    else "<think>Start",
                },
            ],
            **OPTIONS,
            "stream": stream,
            "enable_thinking": True,
            "max_tokens": 128,
        },
    )
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
    deltas = [
        choice.get("delta", choice.get("message", {}))
        for packet in packets
        for choice in packet.get("choices", [])
    ]
    assert (
        "".join(delta.get("reasoning_content") or "" for delta in deltas) == reasoning
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == "42"
    assert cfg.engine.prompt.endswith("assistant:<think>Start")


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_streaming_prefill_override_survives_reset(client, chunk_size):
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    _, cfg = client
    cfg.reasoning_parser_name = "deepseek_r1_distill"
    cfg.engine = CapturingEngine("")
    cfg.engine.tokenizer.chat_template = TEMPLATE.replace(
        "assistant:<think>", "assistant:"
    )
    processor = StreamingPostProcessor(
        cfg, enable_thinking=True, prompt_thinking_active=True
    )
    raw = "x" * 80 + "</think>42"
    # Exercise initialization, then reuse after a completed stream has changed
    # the parser state. The route also calls reset immediately after construction.
    for attempt in range(3):
        if attempt:
            processor.reset()
        events = []
        for offset in range(0, len(raw), chunk_size):
            events.extend(
                processor.process_chunk(
                    GenerationOutput(
                        text=raw[: offset + chunk_size],
                        new_text=raw[offset : offset + chunk_size],
                        finished=False,
                    )
                )
            )
        events.extend(processor.finalize())
        assert "".join(e.reasoning or "" for e in events if e.type == "reasoning") == (
            "x" * 80
        )
        assert "".join(e.content or "" for e in events if e.type == "content") == "42"
