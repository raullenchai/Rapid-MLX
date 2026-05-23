# SPDX-License-Identifier: Apache-2.0
"""Regression test for ``message.content`` key being dropped when empty.

Bug: ``routes/chat.py`` initialized ``final_content = None`` and only
populated it if the engine returned non-empty text. Combined with
``model_dump_json(exclude_none=True)`` on the response, the ``content``
field was silently DROPPED from the JSON response when the engine
emitted no content tokens (e.g. prefix-cache hit that sampled EOS first,
or a reasoning-only response).

OpenAI's spec requires ``message.content`` to always be present in
non-tool-call responses (empty string ``""`` when no text). Clients
keyed off ``choices[0].message.content`` (LangChain, Vercel AI SDK)
crashed with ``KeyError`` / ``AttributeError`` on the missing field.

Fix: when ``final_content is None and not tool_calls``, default to ``""``
so the key is always serialized.

Tool-call responses are exempt — OpenAI's spec permits ``content: null``
when ``tool_calls`` is populated.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _EmptyTextEngine:
    """Engine that returns finish_reason=stop with empty text (only EOS)."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="",
            new_text="",
            tokens=[2],  # EOS only
            prompt_tokens=8,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _client(engine_cls=_EmptyTextEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine_cls()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


class TestEmptyContentKeyPresent:
    """Bug C from iter15-hybrid onboarding: empty-text response missing
    ``content`` field entirely. Verify the key survives serialization."""

    def test_empty_text_response_includes_content_key(self):
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200, resp.text
            msg = resp.json()["choices"][0]["message"]
            assert "content" in msg, (
                "OpenAI spec requires 'content' to always be present in "
                "non-tool-call responses (empty string when no text). "
                "Pre-fix, this key was dropped by exclude_none=True."
            )
            assert msg["content"] == ""
        finally:
            reset_config()

    def test_empty_text_response_role_still_present(self):
        """Sanity: role must also survive serialization."""
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200
            msg = resp.json()["choices"][0]["message"]
            assert msg["role"] == "assistant"
        finally:
            reset_config()

    def test_normal_text_response_unchanged(self):
        """The fix only affects the empty-text path; normal text round-trips."""

        class _NormalEngine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="hello world",
                    new_text="hello world",
                    tokens=[1, 2, 3],
                    prompt_tokens=8,
                    completion_tokens=3,
                    finished=True,
                    finish_reason="stop",
                    channel=None,
                )

        client = _client(engine_cls=_NormalEngine)
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            assert resp.status_code == 200
            msg = resp.json()["choices"][0]["message"]
            assert msg["content"] == "hello world"
        finally:
            reset_config()
