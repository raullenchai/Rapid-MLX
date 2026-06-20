# SPDX-License-Identifier: Apache-2.0
"""Regression tests for H-15 — non-string ``text`` (and image-source
fields) on a content block used to slip past the schema layer and
crash inside the prompt flattener with raw
``TypeError: sequence item 0: expected str instance, X found``.

Pre-fix surface (Daniela r3 repro):
    POST /v1/chat/completions
    {"messages":[{"role":"user","content":[
        {"type":"text","text":[["deeply","nested"],"list"]}
    ]}]}
    → HTTP 500 ``{"error":{"message":"Internal server error"}}``
      with a raw ``TypeError`` in the server log.

Root cause: ``Message.content`` is declared as the union
``str | list[ContentPart] | list[dict]``. When ``text`` is a
non-string, the ``list[ContentPart]`` arm rejects (because
``ContentPart.text`` is ``str | None``), and Pydantic falls back to
the looser ``list[dict]`` arm — which silently accepts the
malformed payload. ``_join_text_parts`` then crashes when it tries
to ``"".join(...)`` the non-string ``text`` value.

Shallow nesting (``text: 123`` / ``text: {"k": "v"}``) reproduces
the same 500 — the dict-fallback path is shape-agnostic.

Fix surface (covers both OpenAI- and Anthropic-flavoured routes):
  * ``ContentPart`` gains an explicit ``_validate_text_type``
    model_validator so direct construction errors name the field
    cleanly (the ``str | None`` declaration alone surfaces as
    Pydantic's raw ``string_type`` loc trail).
  * ``Message._validate_media_url_types`` is extended to also
    reject non-string ``text`` in the dict-fallback arm — this is
    the path that actually escapes to ``_join_text_parts`` and
    500s.
  * ``AnthropicContentBlock`` gains a ``text`` field validator
    plus an image-source string-check (the same H-15 shape applied
    to ``source.data`` / ``source.url`` on Anthropic image blocks).

Each rejection produces a 400 with a stable, field-named message:
    ``content[].text must be a string (got <type>)``

Live HTTP behaviour pinned via TestClient — the FastAPI route is
constructed with the route-level handlers from
``vllm_mlx.middleware.exception_handlers`` so the envelope-shape
assertions match production.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import AnthropicContentBlock, AnthropicRequest
from vllm_mlx.api.models import ChatCompletionRequest, ContentPart, Message
from vllm_mlx.middleware.exception_handlers import _validation_error_response

# ---------------------------------------------------------------------------
# Direct Pydantic construction — pins schema-layer rejection contract.
# ---------------------------------------------------------------------------


class TestContentPartTextStrictType:
    """Direct ``ContentPart(text=…)`` construction must reject
    non-string text with the named-field message."""

    @pytest.mark.parametrize(
        ("value", "type_name"),
        [
            ([["nested"]], "list"),
            (123, "int"),
            ({"k": "v"}, "dict"),
            (12.3, "float"),
            (True, "bool"),
        ],
    )
    def test_non_string_text_rejected(self, value: Any, type_name: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ContentPart(type="text", text=value)
        msg = str(exc_info.value)
        # H-15 named-field message must be in the rejection.
        assert "content[].text must be a string" in msg, msg

    def test_string_text_accepted(self) -> None:
        cp = ContentPart(type="text", text="hello")
        assert cp.text == "hello"

    def test_null_text_accepted(self) -> None:
        # Backwards-compat: ``text=None`` is legal (empty no-op text part).
        cp = ContentPart(type="text", text=None)
        assert cp.text is None

    def test_empty_string_text_accepted(self) -> None:
        cp = ContentPart(type="text", text="")
        assert cp.text == ""


class TestMessageDictFallbackTextStrictType:
    """``Message.content`` dict-fallback arm is the path that actually
    leaked to ``_join_text_parts`` and 500'd pre-H-15. Pin the
    rejection at the parent Message level so the live route surface
    is covered (the route receives a ``Message`` after the union
    resolves)."""

    @pytest.mark.parametrize(
        ("value", "type_name"),
        [
            ([["nested"]], "list"),
            (123, "int"),
            ({"k": "v"}, "dict"),
        ],
    )
    def test_dict_fallback_non_string_text_rejected(
        self, value: Any, type_name: str
    ) -> None:
        # Build a content list whose ``text`` value would force
        # Pydantic into the ``list[dict]`` fallback arm.
        with pytest.raises(ValidationError) as exc_info:
            Message(
                role="user",
                content=[{"type": "text", "text": value}],
            )
        msg = str(exc_info.value)
        assert "content[].text must be a string" in msg, msg
        assert f"got {type_name}" in msg, msg

    def test_null_text_accepted(self) -> None:
        # ``text: null`` is a legal no-op (semantically equivalent to
        # an absent text part).
        m = Message(role="user", content=[{"type": "text", "text": None}])
        part = m.content[0]
        text = part["text"] if isinstance(part, dict) else part.text
        assert text is None

    def test_valid_text_accepted(self) -> None:
        m = Message(role="user", content=[{"type": "text", "text": "ok"}])
        part = m.content[0]
        text = part["text"] if isinstance(part, dict) else part.text
        assert text == "ok"


# ---------------------------------------------------------------------------
# Anthropic content block — `text: str | None` already 422s, but the
# H-15 fix adds an explicit field validator so the error message
# names ``content[].text`` cleanly.
# ---------------------------------------------------------------------------


class TestAnthropicContentBlockTextStrictType:
    @pytest.mark.parametrize(
        ("value", "type_name"),
        [
            ([["nested"]], "list"),
            (123, "int"),
            ({"k": "v"}, "dict"),
        ],
    )
    def test_non_string_text_rejected(self, value: Any, type_name: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AnthropicContentBlock(type="text", text=value)
        msg = str(exc_info.value)
        assert "content[].text must be a string" in msg, msg
        assert f"got {type_name}" in msg, msg

    def test_string_text_accepted(self) -> None:
        block = AnthropicContentBlock(type="text", text="hi")
        assert block.text == "hi"

    def test_null_text_accepted(self) -> None:
        block = AnthropicContentBlock(type="text", text=None)
        assert block.text is None


class TestAnthropicImageSourceStrictType:
    """Pin string-typing on ``source.data`` / ``source.url`` (the
    Anthropic-side equivalent of ``image_url.url``; F-066 sibling)."""

    @pytest.mark.parametrize("key", ["data", "url"])
    def test_non_string_source_field_rejected(self, key: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AnthropicContentBlock(
                type="image",
                source={"type": "base64", "media_type": "image/png", key: 123},
            )
        msg = str(exc_info.value)
        assert f"image source.{key} must be a string" in msg, msg

    def test_valid_source_accepted(self) -> None:
        block = AnthropicContentBlock(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgo=",
            },
        )
        assert block.source["data"] == "iVBORw0KGgo="


# ---------------------------------------------------------------------------
# Top-level request models — covers both routes via the public-facing
# parse surface (matches what FastAPI does).
# ---------------------------------------------------------------------------


class TestChatCompletionRequestTextStrictType:
    """``/v1/chat/completions`` body parse."""

    @pytest.mark.parametrize(
        ("text_value", "type_name"),
        [
            ([["deeply", "nested"], "list"], "list"),  # Daniela r3 shape
            (123, "int"),
            ({"nested": "obj"}, "dict"),
        ],
    )
    def test_non_string_text_in_content_rejected(
        self, text_value: Any, type_name: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text_value}],
                    }
                ],
            )
        msg = str(exc_info.value)
        assert "content[].text must be a string" in msg, msg
        assert f"got {type_name}" in msg, msg

    def test_valid_text_content_accepted(self) -> None:
        req = ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        assert req.messages[0].content[0].text == "hello"

    def test_valid_string_content_accepted(self) -> None:
        # The ``content: str`` arm of the union must still work.
        req = ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.messages[0].content == "hello"

    def test_multi_text_block_payload_accepted(self) -> None:
        # Multiple text parts — a legal OpenAI-spec edge case.
        req = ChatCompletionRequest(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        )
        assert req.messages[0].content[0].text == "first"
        assert req.messages[0].content[1].text == "second"

    def test_image_url_non_string_url_still_rejected(self) -> None:
        # F-066 already covered this; pin it here too so this test
        # file owns the full content-block strict-typing contract.
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": 123},
                            }
                        ],
                    }
                ],
            )
        msg = str(exc_info.value)
        assert "image_url.url must be a string" in msg, msg


class TestAnthropicRequestTextStrictType:
    """``/v1/messages`` body parse."""

    @pytest.mark.parametrize(
        ("text_value", "type_name"),
        [
            ([["nested"]], "list"),
            (123, "int"),
            ({"nested": "obj"}, "dict"),
        ],
    )
    def test_non_string_text_in_content_rejected(
        self, text_value: Any, type_name: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AnthropicRequest(
                model="claude-3-opus",
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text_value}],
                    }
                ],
            )
        msg = str(exc_info.value)
        assert "content[].text must be a string" in msg, msg
        assert f"got {type_name}" in msg, msg

    def test_valid_text_content_accepted(self) -> None:
        req = AnthropicRequest(
            model="claude-3-opus",
            max_tokens=10,
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
        assert req.messages[0].content[0].text == "hi"

    def test_valid_string_content_accepted(self) -> None:
        req = AnthropicRequest(
            model="claude-3-opus",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.messages[0].content == "hi"


# ---------------------------------------------------------------------------
# Live envelope shape — pins the 400 surface the client actually sees
# on ``/v1/chat/completions``. Uses a minimal FastAPI app wired to the
# canonical ``_validation_error_response`` so we don't depend on a
# running model.
# ---------------------------------------------------------------------------


def _make_test_app() -> FastAPI:
    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def _handler(_request, exc):  # noqa: ANN001
        return _validation_error_response(exc)

    @app.post("/v1/chat/completions")
    async def _chat(req: ChatCompletionRequest):  # noqa: ARG001
        return {"ok": True}

    @app.post("/v1/messages")
    async def _messages(req: AnthropicRequest):  # noqa: ARG001
        return {"ok": True}

    return app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(_make_test_app())


class TestLive400EnvelopeChatCompletions:
    @pytest.mark.parametrize(
        ("text_value", "type_name"),
        [
            ([["deeply", "nested"], "list"], "list"),
            (123, "int"),
            ({"nested": "obj"}, "dict"),
        ],
    )
    def test_non_string_text_returns_400_clean_envelope(
        self,
        client: TestClient,
        text_value: Any,
        type_name: str,
    ) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text_value}],
                    }
                ],
            },
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        # OpenAI-shaped envelope (matches the canonical
        # ``_validation_error_response`` output).
        assert body["error"]["type"] == "invalid_request_error"
        assert "content[].text must be a string" in body["error"]["message"]
        assert f"got {type_name}" in body["error"]["message"]
        # No naked Python type-error text leaks (the H-15 surface).
        assert "TypeError" not in body["error"]["message"]
        assert "sequence item 0" not in body["error"]["message"]

    def test_valid_text_passes_validation(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            },
        )
        assert resp.status_code == 200, resp.text

    def test_image_url_non_string_url_returns_400_clean(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": 123},
                            }
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert "image_url.url must be a string" in body["error"]["message"]


class TestLive400EnvelopeAnthropicMessages:
    @pytest.mark.parametrize(
        ("text_value", "type_name"),
        [
            ([["nested"]], "list"),
            (123, "int"),
            ({"nested": "obj"}, "dict"),
        ],
    )
    def test_non_string_text_returns_400_clean_envelope(
        self,
        client: TestClient,
        text_value: Any,
        type_name: str,
    ) -> None:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-opus",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text_value}],
                    }
                ],
            },
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "content[].text must be a string" in body["error"]["message"]
        assert f"got {type_name}" in body["error"]["message"]
        assert "TypeError" not in body["error"]["message"]
