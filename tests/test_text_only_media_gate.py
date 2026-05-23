# SPDX-License-Identifier: Apache-2.0
"""Regression test for text-only models silently dropping audio content parts.

Bug: ``vllm_mlx/routes/chat.py`` rejected ``image_url``/``image``/``video``/
``video_url`` parts on text-only models (``engine.is_mllm = False``) with a
clean HTTP 400, but ``audio_url``/``audio``/``input_audio`` were not in the
reject list. ``extract_multimodal_content`` then silently stripped the audio
part and the model hallucinated ("there is no audio attached") while the
caller saw HTTP 200.

Fix: extend the content-type reject set to include the three audio shapes
OpenAI clients send and broaden the error message to "image, video, or audio".
The R9P1 invariant ("text-only models never silently drop media") now covers
the full set.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _TextOnlyEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _client() -> TestClient:
    cfg = reset_config()
    cfg.engine = _TextOnlyEngine()
    cfg.model_name = "text-only-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _media_request(part: dict) -> dict:
    return {
        "model": "text-only-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    part,
                ],
            }
        ],
        "max_tokens": 8,
    }


class TestTextOnlyMediaRejection:
    """Each media shape must produce HTTP 400, not silent drop."""

    def test_image_url_rejected(self):
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json=_media_request(
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/i.png"},
                    }
                ),
            )
            assert resp.status_code == 400
            assert "does not support" in resp.json()["detail"]
        finally:
            reset_config()

    def test_video_url_rejected(self):
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json=_media_request(
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/v.mp4"},
                    }
                ),
            )
            assert resp.status_code == 400
            assert "does not support" in resp.json()["detail"]
        finally:
            reset_config()

    def test_audio_url_rejected(self):
        """OpenAI shape: ``{"type": "audio_url", "audio_url": {"url": "..."}}``.

        Pre-fix: HTTP 200, audio silently stripped, model hallucinated.
        Post-fix: HTTP 400 mirroring image/video.
        """
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json=_media_request(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/clip.mp3"},
                    }
                ),
            )
            assert resp.status_code == 400, resp.text
            detail = resp.json()["detail"]
            assert "does not support" in detail
            assert "audio" in detail
        finally:
            reset_config()

    def test_audio_short_form_rejected(self):
        """Alt shape some clients send: ``{"type": "audio", "audio": "..."}``."""
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json=_media_request({"type": "audio", "audio": "base64data"}),
            )
            assert resp.status_code == 400, resp.text
            assert "audio" in resp.json()["detail"]
        finally:
            reset_config()

    def test_input_audio_rejected(self):
        """OpenAI Realtime / gpt-4o-audio shape: ``input_audio``."""
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json=_media_request(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "base64data", "format": "wav"},
                    }
                ),
            )
            assert resp.status_code == 400, resp.text
            assert "audio" in resp.json()["detail"]
        finally:
            reset_config()

    def test_pure_text_passes(self):
        """Sanity: plain text content still reaches the engine."""
        client = _client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "text-only-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                },
            )
            assert resp.status_code == 200, resp.text
        finally:
            reset_config()
