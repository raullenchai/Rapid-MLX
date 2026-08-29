# SPDX-License-Identifier: Apache-2.0
"""Test that ``chat_template_kwargs`` extra keys reach ``apply_chat_template``.

Pins issue #2474: ``chat_template_kwargs["reasoning_effort"]`` was silently
dropped before the template render, so Qwen3.8 was permanently pinned to its
``xhigh`` template default. The fix threads the client-supplied dict through
route → engine → ``shared_apply_chat_template``, merging unknown keys into
``template_kwargs`` without overriding server-resolved values.
"""

from __future__ import annotations

import pytest

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.utils.chat_template import apply_chat_template


class FakeTokenizer:
    """Minimal tokenizer that records the kwargs it receives."""

    def __init__(self):
        self.received_kwargs: dict | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.received_kwargs = kwargs
        return "rendered"

    def encode(self, _text):
        return [1, 2]


def _make_applicator():
    tok = FakeTokenizer()
    # Guard against the sanitiser rejecting the fake applicator.
    tok.chat_template = "fake"
    return tok


class TestChatTemplateKwargsPassthrough:
    """``chat_template_kwargs`` extra keys are merged into template_kwargs."""

    def test_reasoning_effort_reaches_the_template(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        assert tok.received_kwargs is not None
        assert tok.received_kwargs["reasoning_effort"] == "low"

    def test_server_resolved_keys_are_not_overridden(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        assert tok.received_kwargs["enable_thinking"] is False

    def test_tokenize_and_add_generation_prompt_not_overridden(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
            chat_template_kwargs={
                "tokenize": True,
                "add_generation_prompt": True,
            },
        )
        assert tok.received_kwargs["tokenize"] is False
        assert tok.received_kwargs["add_generation_prompt"] is False

    def test_none_chat_template_kwargs_is_a_noop(self):
        tok = _make_applicator()
        apply_chat_template(tok, [{"role": "user", "content": "hi"}])
        assert tok.received_kwargs is not None
        assert "reasoning_effort" not in tok.received_kwargs

    def test_tools_key_not_overridden_by_client(self):
        tok = _make_applicator()
        server_tools = [{"type": "function", "function": {"name": "f"}}]
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            tools=server_tools,
            chat_template_kwargs={
                "tools": [{"type": "function", "function": {"name": "evil"}}]
            },
        )
        assert tok.received_kwargs["tools"] == server_tools

    def test_client_tools_key_is_never_forwarded(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={
                "tools": [{"type": "function", "function": {"name": "evil"}}]
            },
        )
        assert "tools" not in tok.received_kwargs


class TestBatchedEngineChatTemplateKwargs:
    """All BatchedEngine chat entry points preserve template kwargs."""

    def _engine(self) -> BatchedEngine:
        received: dict = {}
        engine = BatchedEngine("test-model")
        engine._loaded = True
        engine._prepare_cache_stable_messages = lambda messages: (messages, None)

        def apply_chat_template(*_args, **kwargs):
            received.update(kwargs.get("chat_template_kwargs") or {})
            return "prompt"

        engine._apply_chat_template = apply_chat_template
        engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
            prompt,
            None,
        )
        return engine, received

    async def test_chat_forwards_chat_template_kwargs(self):
        engine, received = self._engine()
        seen_kwargs = {}

        async def generate(**kwargs):
            seen_kwargs.update(kwargs)
            return "output"

        engine.generate = generate

        await engine.chat(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        assert received == {"reasoning_effort": "low"}
        assert seen_kwargs.get("chat_template_kwargs") is None

    async def test_stream_chat_forwards_chat_template_kwargs(self):
        engine, received = self._engine()
        engine._create_output_router = lambda: None
        engine.stream_generate = lambda **kwargs: kwargs

        async def route_stream(stream, _router):
            stream["consumed"] = True
            yield "output"

        engine._stream_with_output_router = route_stream

        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                chat_template_kwargs={"reasoning_effort": "medium"},
            )
        ]

        assert outputs == ["output"]
        assert received == {"reasoning_effort": "medium"}

    async def test_generate_with_schema_forwards_chat_template_kwargs(
        self, monkeypatch
    ):
        class GuidedEngine(BatchedEngine):
            @property
            def supports_guided_generation(self):
                return True

            @property
            def tokenizer(self):
                return FakeTokenizer()

        engine = GuidedEngine("test-model")
        engine._loaded = True
        engine._run_guided_generation = lambda **_kwargs: '{"ok": true}'

        received = {}

        def fake_shared_apply(tokenizer, messages, **kwargs):
            received.update(kwargs.get("chat_template_kwargs") or {})
            return kwargs.get("chat_template_kwargs")

        monkeypatch.setattr(
            "vllm_mlx.engine.batched.shared_apply_chat_template",
            fake_shared_apply,
        )

        output = await engine.generate_with_schema(
            messages=[{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        assert output.text == '{"ok": true}'
        assert received == {"reasoning_effort": "low"}

    async def test_build_prompt_forwards_chat_template_kwargs(self):
        engine = self._engine()[0]
        engine._apply_chat_template = lambda *_args, **kwargs: kwargs.get(
            "chat_template_kwargs"
        )
        engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
            prompt,
            None,
        )
        engine._loaded = True

        assert engine.build_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        ) == {"reasoning_effort": "low"}


class _RawRequest:
    def __init__(self):
        self.headers = {}

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


class _CapturingChatEngine:
    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "test-model"

    def __init__(self):
        self.chat_kwargs = {}
        self.build_prompt_kwargs = {}

    def build_prompt(
        self, messages, *, tools=None, enable_thinking=None, chat_template_kwargs=None
    ):
        self.build_prompt_kwargs.update(
            {
                "tools": tools,
                "enable_thinking": enable_thinking,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )
        raise ValueError("Chat template error: unsupported reasoning_effort")

    async def chat(self, messages, **kwargs):
        self.chat_kwargs.update(kwargs)
        return GenerationOutput(
            text="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


async def _await_direct(coro, *_args, **_kwargs):
    return await coro


def _patch_chat_route(monkeypatch, engine):
    from vllm_mlx.routes import chat

    monkeypatch.setattr(chat, "_resolve_max_tokens", lambda *_args, **_kwargs: 64)
    monkeypatch.setattr(chat, "get_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(chat, "_validate_model_name", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat, "_check_admission_or_503", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat, "_release_admission_unless_committed", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(chat, "_wait_with_disconnect", _await_direct)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(chat, "enforce_context_length_for_messages", lambda *a, **k: 1)


class TestChatRouteChatTemplateKwargs:
    """Chat route forwards the resolved template kwargs to the engine."""

    async def test_nonstreaming_route_forwards_chat_template_kwargs(self, monkeypatch):
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.routes import chat

        engine = _CapturingChatEngine()
        _patch_chat_route(monkeypatch, engine)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            stream=False,
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        await chat._create_chat_completion_impl(
            request,
            _RawRequest(),
            engine,
            _commit_state=[False],
            _admission_acquired=[False],
        )

        assert engine.chat_kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "low"
        }

    async def test_streaming_route_validates_chat_template_kwargs(self, monkeypatch):
        from vllm_mlx.api.models import ChatCompletionRequest
        from vllm_mlx.routes import chat

        engine = _CapturingChatEngine()
        _patch_chat_route(monkeypatch, engine)
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            stream=True,
            chat_template_kwargs={"reasoning_effort": "unsupported"},
        )

        with pytest.raises(Exception, match="unsupported reasoning_effort"):
            await chat._create_chat_completion_impl(
                request,
                _RawRequest(),
                engine,
                _commit_state=[False],
                _admission_acquired=[False],
            )

        assert engine.build_prompt_kwargs.get("chat_template_kwargs") == {
            "reasoning_effort": "unsupported"
        }
