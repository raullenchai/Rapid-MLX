# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-065 — bare-string ``image_url`` shorthand
on a ``type:"image_url"`` content part used to silently slip past
the schema layer and be dropped by the multimodal preprocessor.

The wire-form pre-fix:
    {"type":"image_url","image_url":"data:image/png;base64,..."}
                                     ^ bare string, NOT spec-shape

The OpenAI Chat Completions API requires
    {"type":"image_url","image_url":{"url":"..."}}
                                    ^^^^^^^^^^^^ object with required url
i.e. an object with a required ``url`` slot. Pre-fix, the
``ContentPart.image_url`` field accepted a bare string via the
``ImageUrl | dict | str | None`` union, and the multimodal
preprocessor unwrapped ``image["url"]`` only on the dict shape —
the bare-string form silently dropped the image. The model
received only the text and hallucinated ("the image is blank")
on a multimodal model; on a text-only model the request 400'd at
preprocess with a vague "model does not support image" message
that never told the client the actual mismatch (shape, not
modality).

The fix:
  * Reject bare-string ``image_url`` (resp. ``video_url`` /
    ``audio_url``) at the ``ContentPart`` model_validator AND at
    the parent ``Message`` model_validator (the dict-fallback
    arm of ``list[ContentPart] | list[dict]``).
  * Gate the reject on ``type``: only fire when the part type
    advertises the media slot (``type=="image_url"`` for the
    ``image_url`` field). A pure-text part that carries an
    unrelated ``image_url:"..."`` slot (legacy / hand-rolled
    clients) is NOT collaterally broken.
  * Mirror the rule on ``video_url`` and ``audio_url`` so the
    spec-shape contract is consistent across modalities.

Validation happens BEFORE model dispatch, so port 8045 with a
text-only model (qwen3-0.6b-8bit) is sufficient for the live
repro — the schema layer fires before the preprocessor would
get a chance to reject on modality grounds.
"""

import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import ChatCompletionRequest, ContentPart


class TestContentPartBareStringMediaRejected:
    """Direct ``ContentPart`` construction tests — pins the
    ContentPart-level model_validator contract."""

    def test_bare_string_image_url_rejected(self):
        with pytest.raises(ValidationError) as ei:
            ContentPart(
                type="image_url",
                image_url="data:image/png;base64,iVBORw0KG...",
            )
        msg = str(ei.value)
        assert "image_url must be an object" in msg
        assert "url" in msg

    def test_bare_string_video_url_rejected(self):
        with pytest.raises(ValidationError) as ei:
            ContentPart(
                type="video_url",
                video_url="https://example.com/vid.mp4",
            )
        msg = str(ei.value)
        assert "video_url must be an object" in msg

    def test_bare_string_audio_url_rejected(self):
        with pytest.raises(ValidationError) as ei:
            ContentPart(
                type="audio_url",
                audio_url="https://example.com/snd.mp3",
            )
        msg = str(ei.value)
        assert "audio_url must be an object" in msg

    def test_dict_shape_image_url_accepted(self):
        """``image_url: {"url": "..."}`` is the spec-shape and must
        continue to parse."""
        part = ContentPart(
            type="image_url",
            image_url={"url": "data:image/png;base64,iVBORw0KG..."},
        )
        assert part.type == "image_url"
        # Accept either ``ImageUrl`` model or raw dict (union may
        # resolve either way depending on shape).
        if isinstance(part.image_url, dict):
            assert part.image_url["url"].startswith("data:image/png")
        else:
            assert part.image_url.url.startswith("data:image/png")

    def test_typed_image_url_model_accepted(self):
        """Passing an already-constructed ``ImageUrl`` instance is
        also a legal wire form (in-process callers)."""
        from vllm_mlx.api.models import ImageUrl

        part = ContentPart(
            type="image_url",
            image_url=ImageUrl(url="https://example.com/x.png"),
        )
        assert part.image_url.url == "https://example.com/x.png"

    def test_text_part_with_unrelated_image_url_string_unaffected(self):
        """Codex-defensive: a content part of ``type:"text"`` that
        happens to carry an unrelated ``image_url`` string slot
        (legacy / hand-rolled clients that populate every slot) must
        NOT be rejected — the validator is gated on ``type`` so it
        only fires when the part actually advertises itself as the
        media kind."""
        part = ContentPart(
            type="text",
            text="hello",
            image_url="https://example.com/should-be-ignored.png",
        )
        assert part.type == "text"
        assert part.text == "hello"


class TestChatCompletionRequestBareStringMediaRejected:
    """End-to-end through ChatCompletionRequest — the Pydantic
    ``Message.content`` union is ``list[ContentPart] | list[dict]``,
    so a payload that fails ContentPart parsing CAN fall back to
    ``list[dict]``. The parent ``Message._validate_media_url_types``
    validator must also fire on the dict-fallback arm so the
    bare-string hazard is closed end-to-end."""

    def _build(self, content):
        return {
            "model": "qwen3-0.6b-8bit",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 5,
        }

    def test_bare_string_image_url_rejected_end_to_end(self):
        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest.model_validate(
                self._build(
                    [
                        {"type": "text", "text": "what color?"},
                        {
                            "type": "image_url",
                            "image_url": ("data:image/png;base64,iVBORw0KG..."),
                        },
                    ]
                )
            )
        msg = str(ei.value)
        assert "image_url must be an object" in msg

    def test_bare_string_video_url_rejected_end_to_end(self):
        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest.model_validate(
                self._build(
                    [
                        {
                            "type": "video_url",
                            "video_url": "https://example.com/v.mp4",
                        }
                    ]
                )
            )
        msg = str(ei.value)
        assert "video_url must be an object" in msg

    def test_dict_shape_image_url_accepted_end_to_end(self):
        req = ChatCompletionRequest.model_validate(
            self._build(
                [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/x.png"},
                    },
                ]
            )
        )
        part = req.messages[0].content[1]
        # The well-typed dict resolves to the Pydantic ContentPart
        # variant (with the inner ImageUrl model) for valid input.
        if hasattr(part, "image_url"):
            url = (
                part.image_url.url
                if hasattr(part.image_url, "url")
                else part.image_url["url"]
            )
        else:
            url = part["image_url"]["url"]
        assert url == "https://example.com/x.png"

    def test_text_only_content_unaffected(self):
        """Sanity: a pure-text content list (single string OR
        list of text parts) must still parse unchanged."""
        req = ChatCompletionRequest.model_validate(self._build("hi there"))
        assert req.messages[0].content == "hi there"

        req2 = ChatCompletionRequest.model_validate(
            self._build([{"type": "text", "text": "hi"}])
        )
        assert len(req2.messages[0].content) == 1
