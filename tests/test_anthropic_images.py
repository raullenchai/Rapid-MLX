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
BASE64_SOURCE = {
    "type": "base64",
    "media_type": "image/png",
    "data": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAIklEQVR4nGNkaGAgCTCRppxhVANRgIk4ZQgwqoEYQHIoAQC7HwCg8pZY7gAAAABJRU5ErkJggg==",
}

DATA_URL = "data:image/png;base64," + BASE64_SOURCE["data"]


def image(source=None):
    return {"type": "image", "source": source or BASE64_SOURCE}


@pytest.mark.parametrize(
    "source,expected",
    [
        (BASE64_SOURCE, DATA_URL),
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
    assert extract_multimodal_content(converted)[1] == [DATA_URL]


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
        DATA_URL
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
        (BASE64_SOURCE, DATA_URL),
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
    assert extract_multimodal_content(messages)[1] == [DATA_URL]
    assert not images and not videos


@pytest.mark.parametrize("data", ["!!!", "abc", "你好", "aGVs bG8="])
def test_invalid_base64_payloads_fail_at_schema(data):
    with pytest.raises(ValidationError, match="image source.data must be valid base64"):
        AnthropicContentBlock(type="image", source={**BASE64_SOURCE, "data": data})


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/image.png",
        "/tmp/image.png",
        DATA_URL,
        "https://",
        "ftp://example.com/a.png",
    ],
)
def test_url_sources_cannot_select_local_file_loader(url):
    with pytest.raises(ValidationError, match="image source.url must be an HTTP"):
        AnthropicContentBlock(type="image", source={"type": "url", "url": url})


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/image.png", "http://169.254.169.254/image.png"]
)
def test_image_url_uses_existing_remote_media_safety_guard(monkeypatch, url):
    import socket

    from rapid_mlx.models.mllm import RemoteMediaFetchError, process_image_input

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    converted = _convert_message(
        AnthropicMessage(role="user", content=[image({"type": "url", "url": url})])
    )
    extracted = extract_multimodal_content(converted)[1]
    with pytest.raises(RemoteMediaFetchError):
        process_image_input(extracted[0])


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_nested_image_returns_400_before_generation(make_client, stream):
    client, engine = make_client()
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 20,
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [
                                image(
                                    {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "!!!",
                                    }
                                )
                            ],
                        }
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "Invalid image source in tool_result" in response.text
    assert "!!!" not in response.text
    assert not engine.calls


def test_multiple_screenshot_results_keep_explicit_image_association():
    converted = _convert_message(
        AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": [image()],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_456",
                    "content": [image({"type": "url", "url": URL})],
                },
            ],
        )
    )
    assert [m.tool_call_id for m in converted[:2]] == ["toolu_123", "toolu_456"]
    parts = converted[2].model_dump()["content"]
    assert parts[0]["text"] == "Images from tool_result toolu_123:"
    assert parts[1]["image_url"]["url"] == DATA_URL
    assert parts[2]["text"] == "Images from tool_result toolu_456:"
    assert parts[3]["image_url"]["url"] == URL


@pytest.mark.parametrize("stream", [False, True])
def test_decoder_client_errors_keep_existing_http_or_sse_contract(make_client, stream):
    from rapid_mlx.request import ClientRequestError

    client, engine = make_client()
    message = "Failed to process image: image data could not be decoded"

    async def reject_chat(*args, **kwargs):
        raise ClientRequestError(message)

    async def reject_stream(*args, **kwargs):
        raise ClientRequestError(message)
        yield  # Marks the engine contract as an async generator.

    engine.chat = reject_chat
    engine.stream_chat = reject_stream
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 20,
            "stream": stream,
            "messages": [{"role": "user", "content": [image()]}],
        },
    )
    assert response.status_code == (200 if stream else 400)
    if stream:
        assert '"type": "invalid_request_error"' in response.text
        assert "data: [DONE]" in response.text
    assert message in response.text


@pytest.mark.parametrize("tool_first", [False, True])
def test_mixed_direct_and_tool_images_keep_reply_adjacency_and_image_order(tool_first):
    direct = image({"type": "url", "url": URL})
    tool = {"type": "tool_result", "tool_use_id": "toolu_123", "content": [image()]}
    blocks = [tool, direct] if tool_first else [direct, tool]
    converted = _convert_message(AnthropicMessage(role="user", content=blocks))
    assert [m.role for m in converted] == ["tool", "user"]
    assert converted[0].tool_call_id == "toolu_123"
    expected = [DATA_URL, URL] if tool_first else [URL, DATA_URL]
    assert extract_multimodal_content(converted)[1] == expected
    assert "Images from tool_result toolu_123:" in str(converted[1].content)


@pytest.mark.parametrize("native", [False, True])
def test_tool_screenshot_captions_keep_block_order(native):
    converted = _convert_message(
        AnthropicMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": [
                        {"type": "text", "text": "Before"},
                        image(),
                        "Between",
                        image({"type": "url", "url": URL}),
                        {"type": "text", "text": "After"},
                    ],
                }
            ],
        )
    )
    assert [m.role for m in converted] == ["tool", "user"]
    assert converted[0].tool_call_id == "toolu_123"
    assert converted[0].content == "Before\nBetween\nAfter"
    parts = converted[1].model_dump(exclude_none=True)["content"]
    assert parts == [
        {"type": "text", "text": "Images from tool_result toolu_123:"},
        {"type": "text", "text": "Before"},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
        {"type": "text", "text": "Between"},
        {"type": "image_url", "image_url": {"url": URL}},
        {"type": "text", "text": "After"},
    ]
    assert extract_multimodal_content(converted, preserve_native_format=native)[1] == [
        DATA_URL,
        URL,
    ]


def test_base64_validation_uses_bounded_chunks(monkeypatch):
    import base64

    payload = "A" * (64 * 1024 + 4)
    original_decode = base64.b64decode
    sizes = []

    def bounded_decode(data, **kwargs):
        sizes.append(len(data))
        assert len(data) <= 64 * 1024
        return original_decode(data, **kwargs)

    monkeypatch.setattr(base64, "b64decode", bounded_decode)
    AnthropicContentBlock(type="image", source={**BASE64_SOURCE, "data": payload})
    assert sizes == [64 * 1024, 4]


@pytest.mark.parametrize(
    "payload",
    [
        "AAAA" * (16 * 1024 - 1) + "AA==AAAA",
        "A" * (64 * 1024) + "!!!=",
    ],
)
def test_invalid_base64_across_chunk_boundary_is_rejected(payload):
    with pytest.raises(ValidationError, match="image source.data must be valid base64"):
        AnthropicContentBlock(type="image", source={**BASE64_SOURCE, "data": payload})


def test_url_scheme_is_normalized_for_image_loader():
    source = {"type": "url", "url": "HTTPS://example.com/screenshot.png"}
    converted = _convert_message(AnthropicMessage(role="user", content=[image(source)]))
    assert extract_multimodal_content(converted)[1] == [URL]
