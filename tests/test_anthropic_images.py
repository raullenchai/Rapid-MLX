# SPDX-License-Identifier: Apache-2.0
"""Images must survive the Anthropic boundary or fail before generation (#4483)."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rapid_mlx.api.anthropic_adapter import AnthropicContentError, _convert_message
from rapid_mlx.api.anthropic_models import AnthropicContentBlock, AnthropicMessage
from rapid_mlx.api.utils import extract_multimodal_content
from rapid_mlx.config import reset_config
from rapid_mlx.routes.anthropic import router

URL = "https://example.com/screenshot.png"
BASE64_SOURCE = {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}


def image(source=None):
    return {"type": "image", "source": source or BASE64_SOURCE}


@pytest.mark.parametrize(
    "source,expected",
    [
        (BASE64_SOURCE, "data:image/png;base64,aGVsbG8="),
        ({"type": "url", "url": URL}, URL),
    ],
)
def test_images_preserve_order_and_bytes(source, expected):
    blocks = [
        {"type": "text", "text": "before"},
        image(source),
        {"type": "text", "text": "between"},
        image(source),
    ]
    converted = _convert_message(AnthropicMessage(role="user", content=blocks))
    parts = converted[0].model_dump()["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url"]
    messages, images, videos = extract_multimodal_content(converted)
    assert images == [expected, expected]
    assert not videos
    assert "before" in str(messages) and "between" in str(messages)


def test_image_only_message_is_not_empty():
    converted = _convert_message(AnthropicMessage(role="user", content=[image()]))
    assert extract_multimodal_content(converted)[1] == [
        "data:image/png;base64,aGVsbG8="
    ]


@pytest.mark.parametrize("native", [False, True])
def test_tool_result_images_preserve_correlation_and_reach_extractor(native):
    converted = _convert_message(
        AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": [{"type": "text", "text": "screenshot"}, image()],
                },
                {"type": "tool_result", "tool_use_id": "toolu_456", "content": "done"},
            ],
        )
    )
    assert [m.role for m in converted] == ["tool", "tool", "user"]
    assert converted[0].tool_call_id == "toolu_123"
    assert converted[0].content == "screenshot"
    assert converted[1].tool_call_id == "toolu_456"
    assert extract_multimodal_content(converted, preserve_native_format=native)[1] == [
        "data:image/png;base64,aGVsbG8="
    ]


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"type": "file", "file_id": "secret"},
        {"type": "base64"},
        {"type": "base64", "data": "abc", "media_type": "application/pdf"},
        {"type": "base64", "data": "", "media_type": "image/png"},
        {"type": "base64", "data": "abc", "media_type": 123},
        {"type": "url"},
        {"type": "url", "url": " "},
    ],
)
def test_invalid_image_sources_are_rejected(source):
    with pytest.raises(ValidationError, match="image source"):
        AnthropicContentBlock(type="image", source=source)


@pytest.mark.parametrize("nested", [False, True])
def test_documents_never_disappear(nested):
    block = {"type": "document", "source": {"type": "base64", "data": "abc"}}
    if nested:
        block = {"type": "tool_result", "tool_use_id": "toolu_123", "content": [block]}
    with pytest.raises(
        AnthropicContentError, match="document inputs are not supported"
    ):
        _convert_message(AnthropicMessage(role="user", content=[block]))


class VisionEngine:
    preserve_native_tool_format = False
    tokenizer = None

    def __init__(self, vision=True):
        self.is_mllm = vision
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(
            text="green",
            raw_text="green",
            prompt_tokens=5,
            completion_tokens=1,
            finish_reason="stop",
            tool_calls=None,
        )

    async def stream_chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        yield SimpleNamespace(new_text="green", prompt_tokens=5, completion_tokens=1)


@pytest.fixture
def make_client():
    def make(vision=True):
        engine = VisionEngine(vision)
        cfg = reset_config()
        cfg.engine = engine
        cfg.model_name = "test-model"
        cfg.no_thinking = True
        cfg.model_registry = None
        app = FastAPI()
        app.include_router(router)
        return TestClient(app), engine

    yield make
    reset_config()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "source,expected",
    [
        (BASE64_SOURCE, "data:image/png;base64,aGVsbG8="),
        ({"type": "url", "url": URL}, URL),
    ],
)
def test_routes_forward_images(make_client, stream, nested, source, expected):
    client, engine = make_client()
    block = image(source)
    if nested:
        block = {"type": "tool_result", "tool_use_id": "toolu_123", "content": [block]}
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 20,
            "stream": stream,
            "messages": [{"role": "user", "content": [block]}],
        },
    )
    assert response.status_code == 200, response.text
    assert len(engine.calls) == 1
    messages, kwargs = engine.calls[0]
    assert extract_multimodal_content(messages)[1] == [expected]
    assert not kwargs.get("images")  # Engine extracts embedded bytes exactly once.
    assert any(isinstance(m["content"], list) for m in messages)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "kind,vision", [("image", False), ("document", False), ("document", True)]
)
def test_unsupported_media_fails_before_engine(
    make_client, stream, nested, kind, vision
):
    client, engine = make_client(vision)
    block = {"type": kind, "source": BASE64_SOURCE}
    if nested:
        block = {"type": "tool_result", "tool_use_id": "toolu_123", "content": [block]}
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 20,
            "stream": stream,
            "messages": [{"role": "user", "content": [block]}],
        },
    )
    assert response.status_code == 400, response.text
    assert kind in response.text
    assert not engine.calls


def test_count_tokens_rejects_documents(make_client):
    client, engine = make_client()
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "document", "source": BASE64_SOURCE}],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "document inputs are not supported" in response.text
    assert not engine.calls


def test_vision_preparation_decodes_native_tool_arguments():
    from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
    from rapid_mlx.api.anthropic_models import AnthropicRequest
    from rapid_mlx.routes.anthropic import _prepare_anthropic_engine_messages

    request = anthropic_to_openai(
        AnthropicRequest(
            model="test-model",
            max_tokens=20,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "screenshot",
                            "input": {"screen": 1},
                        }
                    ],
                },
                {"role": "user", "content": [image()]},
            ],
        )
    )
    engine = VisionEngine()
    engine.preserve_native_tool_format = True
    messages, images, videos = _prepare_anthropic_engine_messages(request, engine)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"screen": 1}
    assert extract_multimodal_content(messages)[1] == ["data:image/png;base64,aGVsbG8="]
    assert not images and not videos
