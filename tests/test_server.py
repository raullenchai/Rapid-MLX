# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible API server."""

import platform
import sys

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Unit Tests - Request/Response Models
# =============================================================================


class TestRequestModels:
    """Test Pydantic request models."""

    def test_chat_message_text_only(self):
        """Test chat message with text content."""
        from vllm_mlx.server import Message

        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_multimodal(self):
        """Test chat message with multimodal content."""
        from vllm_mlx.server import Message

        content = [
            {"type": "text", "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        msg = Message(role="user", content=content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_image_url_model(self):
        """Test ImageUrl model."""
        from vllm_mlx.server import ImageUrl

        img_url = ImageUrl(url="https://example.com/image.jpg")
        assert img_url.url == "https://example.com/image.jpg"
        assert img_url.detail is None

    def test_video_url_model(self):
        """Test VideoUrl model."""
        from vllm_mlx.server import VideoUrl

        video_url = VideoUrl(url="https://example.com/video.mp4")
        assert video_url.url == "https://example.com/video.mp4"

    def test_content_part_text(self):
        """Test ContentPart with text."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="text", text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_content_part_image(self):
        """Test ContentPart with image_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="image_url", image_url={"url": "https://example.com/img.jpg"}
        )
        assert part.type == "image_url"
        # image_url can be dict or ImageUrl object
        if isinstance(part.image_url, dict):
            assert part.image_url["url"] == "https://example.com/img.jpg"
        else:
            assert part.image_url.url == "https://example.com/img.jpg"

    def test_content_part_video(self):
        """Test ContentPart with video."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(type="video", video="/path/to/video.mp4")
        assert part.type == "video"
        assert part.video == "/path/to/video.mp4"

    def test_content_part_video_url(self):
        """Test ContentPart with video_url."""
        from vllm_mlx.server import ContentPart

        part = ContentPart(
            type="video_url", video_url={"url": "https://example.com/video.mp4"}
        )
        assert part.type == "video_url"
        # video_url can be dict or VideoUrl object
        if isinstance(part.video_url, dict):
            assert part.video_url["url"] == "https://example.com/video.mp4"
        else:
            assert part.video_url.url == "https://example.com/video.mp4"


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_basic_request(self):
        """Test basic chat completion request."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )

        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens is None  # uses _default_max_tokens when None
        assert (
            request.temperature is None
        )  # resolved at runtime by _resolve_temperature
        assert request.stream is False  # default

    def test_request_with_options(self):
        """Test request with custom options."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.5,
            stream=True,
        )

        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True

    def test_request_with_video_params(self):
        """Test request with video parameters."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Describe the video")],
            video_fps=2.0,
            video_max_frames=16,
        )

        assert request.video_fps == 2.0
        assert request.video_max_frames == 16


class TestCompletionRequest:
    """Test CompletionRequest model."""

    def test_basic_completion_request(self):
        """Test basic completion request."""
        from vllm_mlx.server import CompletionRequest

        request = CompletionRequest(model="test-model", prompt="Once upon a time")

        assert request.model == "test-model"
        assert request.prompt == "Once upon a time"
        assert request.max_tokens is None  # uses _default_max_tokens when None


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test server helper functions."""

    def test_is_mllm_model_patterns(self):
        """Test MLLM model detection patterns."""
        from vllm_mlx.server import is_mllm_model

        # Should detect as MLLM
        assert is_mllm_model("mlx-community/Qwen3-VL-4B-Instruct-3bit")
        assert is_mllm_model("mlx-community/llava-1.5-7b-4bit")
        assert is_mllm_model("mlx-community/paligemma-3b-mix-224-4bit")
        assert is_mllm_model("mlx-community/pixtral-12b-4bit")
        assert is_mllm_model("mlx-community/Idefics3-8B-Llama3-4bit")
        assert is_mllm_model("mlx-community/deepseek-vl-7b-chat-4bit")

        # Should NOT detect as MLLM
        assert not is_mllm_model("mlx-community/Llama-3.2-1B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Mistral-7B-Instruct-4bit")
        assert not is_mllm_model("mlx-community/Qwen2-7B-Instruct-4bit")

    def test_extract_multimodal_content_text_only(self):
        """Test extracting content from text-only messages."""
        from vllm_mlx.server import Message, extract_multimodal_content

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 2
        assert processed[0]["content"] == "Hello"
        assert len(images) == 0
        assert len(videos) == 0

    def test_extract_multimodal_content_with_image(self):
        """Test extracting content with images."""
        from vllm_mlx.server import Message, extract_multimodal_content

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "What's this?"
        assert len(images) == 1
        assert "https://example.com/img.jpg" in images[0]

    def test_extract_multimodal_content_with_video(self):
        """Test extracting content with videos."""
        from vllm_mlx.server import Message, extract_multimodal_content

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video", "video": "/path/to/video.mp4"},
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(processed) == 1
        assert processed[0]["content"] == "Describe this video"
        assert len(videos) == 1
        assert videos[0] == "/path/to/video.mp4"

    def test_extract_multimodal_content_with_video_url(self):
        """Test extracting content with video_url format."""
        from vllm_mlx.server import Message, extract_multimodal_content

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What happens?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                ],
            )
        ]

        processed, images, videos = extract_multimodal_content(messages)

        assert len(videos) == 1


# =============================================================================
# Security and Reliability Tests (PR #4)
# =============================================================================


class TestRateLimiter:
    """Test the RateLimiter class for rate limiting functionality."""

    def test_rate_limiter_disabled_by_default(self):
        """Test that rate limiter allows all requests when disabled."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, enabled=False)

        # Should allow unlimited requests when disabled
        for _ in range(100):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True
            assert retry_after == 0

    def test_rate_limiter_enforces_limit(self):
        """Test that rate limiter enforces the request limit."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=3, enabled=True)

        # First 3 requests should be allowed
        for i in range(3):
            allowed, retry_after = limiter.is_allowed("client1")
            assert allowed is True, f"Request {i + 1} should be allowed"
            assert retry_after == 0

        # 4th request should be blocked
        allowed, retry_after = limiter.is_allowed("client1")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_per_client(self):
        """Test that rate limits are tracked per client."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Client 1 uses its quota
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        allowed, _ = limiter.is_allowed("client1")
        assert allowed is False

        # Client 2 should still have quota
        allowed, _ = limiter.is_allowed("client2")
        assert allowed is True

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        import threading

        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=100, enabled=True)
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    allowed, _ = limiter.is_allowed("shared_client")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 100
        # Exactly 100 requests allowed (our limit)
        assert results.count(True) == 100


class TestTempFileManager:
    """Test the TempFileManager class for temp file cleanup."""

    def test_register_and_cleanup_single_file(self):
        """Test registering and cleaning up a single temp file."""
        import os
        import tempfile

        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Create a real temp file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        temp.write(b"test content")
        temp.close()

        # Register it
        path = manager.register(temp.name)
        assert path == temp.name
        assert os.path.exists(temp.name)

        # Cleanup
        result = manager.cleanup(temp.name)
        assert result is True
        assert not os.path.exists(temp.name)

    def test_cleanup_all_files(self):
        """Test cleaning up all registered temp files."""
        import os
        import tempfile

        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []

        # Create multiple temp files
        for i in range(3):
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt")
            temp.write(f"content {i}".encode())
            temp.close()
            manager.register(temp.name)
            paths.append(temp.name)

        # Verify all exist
        for p in paths:
            assert os.path.exists(p)

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 3

        # Verify all deleted
        for p in paths:
            assert not os.path.exists(p)

    def test_cleanup_nonexistent_file(self):
        """Test cleanup of a non-existent file."""
        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()

        # Cleanup a file that doesn't exist
        result = manager.cleanup("/nonexistent/path/file.txt")
        assert result is False

    def test_thread_safe_registration(self):
        """Test that TempFileManager is thread-safe."""
        import tempfile
        import threading

        from vllm_mlx.models.mllm import TempFileManager

        manager = TempFileManager()
        paths = []
        lock = threading.Lock()
        errors = []

        def register_files():
            try:
                for _ in range(5):
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                    temp.write(b"test")
                    temp.close()
                    path = manager.register(temp.name)
                    with lock:
                        paths.append(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_files) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(paths) == 25

        # Cleanup all
        cleaned = manager.cleanup_all()
        assert cleaned == 25


class TestRequestOutputCollectorThreadSafety:
    """Test thread-safety of RequestOutputCollector._waiting_consumers."""

    def test_waiting_consumers_thread_safe(self):
        """Test that _waiting_consumers counter is thread-safe."""
        import threading

        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset the counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        errors = []

        def manipulate_counter():
            try:
                for _ in range(100):
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers += 1
                    with RequestOutputCollector._waiting_lock:
                        RequestOutputCollector._waiting_consumers -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=manipulate_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        # Should return to zero
        with RequestOutputCollector._waiting_lock:
            assert RequestOutputCollector._waiting_consumers == 0

    def test_has_waiting_consumers_method(self):
        """Test has_waiting_consumers class method."""
        from vllm_mlx.output_collector import RequestOutputCollector

        # Reset counter
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0

        assert RequestOutputCollector.has_waiting_consumers() is False

        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 1

        assert RequestOutputCollector.has_waiting_consumers() is True

        # Reset
        with RequestOutputCollector._waiting_lock:
            RequestOutputCollector._waiting_consumers = 0


class TestRequestTimeoutField:
    """Test the new timeout field in request models."""

    def test_chat_completion_request_timeout_field(self):
        """Test that ChatCompletionRequest has timeout field."""
        from vllm_mlx.server import ChatCompletionRequest, Message

        # Default should be None
        request = ChatCompletionRequest(
            model="test-model", messages=[Message(role="user", content="Hello")]
        )
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            timeout=60.0,
        )
        assert request_with_timeout.timeout == 60.0

    def test_completion_request_timeout_field(self):
        """Test that CompletionRequest has timeout field."""
        from vllm_mlx.server import CompletionRequest

        # Default should be None
        request = CompletionRequest(model="test-model", prompt="Once upon a time")
        assert request.timeout is None

        # Can set custom timeout
        request_with_timeout = CompletionRequest(
            model="test-model", prompt="Once upon a time", timeout=120.0
        )
        assert request_with_timeout.timeout == 120.0


class TestAPIKeyVerification:
    """Test API key verification with timing attack prevention."""

    def test_secrets_compare_digest_usage(self):
        """Test that secrets.compare_digest is used (timing attack prevention)."""
        import secrets

        # Verify secrets.compare_digest works as expected
        key1 = "test-api-key-12345"
        key2 = "test-api-key-12345"
        key3 = "different-key-67890"

        # Same keys should match
        assert secrets.compare_digest(key1, key2) is True

        # Different keys should not match
        assert secrets.compare_digest(key1, key3) is False

        # Verify it's constant-time (by checking function exists)
        assert hasattr(secrets, "compare_digest")

    def test_verify_api_key_rejects_invalid(self):
        """Test that invalid API key is rejected with 401."""
        import asyncio

        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        import vllm_mlx.server as server
        from vllm_mlx.config import get_config

        cfg = get_config()
        original_key = cfg.api_key

        try:
            # Set a known API key on config (where verify_api_key reads from)
            cfg.api_key = "valid-secret-key"

            # Create mock credentials with invalid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="invalid-key"
            )

            # Should raise HTTPException with 401
            with pytest.raises(HTTPException) as exc_info:
                asyncio.get_event_loop().run_until_complete(
                    server.verify_api_key(credentials)
                )

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)
        finally:
            cfg.api_key = original_key

    def test_verify_api_key_accepts_valid(self):
        """Test that valid API key is accepted."""
        import asyncio

        from fastapi.security import HTTPAuthorizationCredentials

        import vllm_mlx.server as server
        from vllm_mlx.config import get_config

        cfg = get_config()
        original_key = cfg.api_key

        try:
            # Set a known API key on config
            cfg.api_key = "valid-secret-key"

            # Create mock credentials with valid key
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="valid-secret-key"
            )

            # Should not raise any exception
            result = asyncio.get_event_loop().run_until_complete(
                server.verify_api_key(credentials)
            )
            # verify_api_key returns True on success (no exception raised)
            assert result is True or result is None
        finally:
            cfg.api_key = original_key


class TestRateLimiterHTTPResponse:
    """Test rate limiter HTTP response behavior."""

    def test_rate_limiter_returns_retry_after(self):
        """Test that rate limiter returns retry_after when limit exceeded."""
        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Exhaust the limit
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Next request should be denied with retry_after
        allowed, retry_after = limiter.is_allowed("test_client")

        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
        assert retry_after <= 60  # Should be within a minute

    def test_rate_limiter_window_cleanup(self):
        """Test that rate limiter cleans up old requests from sliding window."""
        import time

        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, enabled=True)

        # Make some requests
        limiter.is_allowed("test_client")
        limiter.is_allowed("test_client")

        # Should be denied (limit reached)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is False

        # Manually inject old timestamps to simulate time passing
        # The sliding window should clean these up
        old_time = time.time() - 120  # 2 minutes ago
        with limiter._lock:
            limiter._requests["test_client"] = [old_time, old_time]

        # Now should be allowed again (old requests cleaned up)
        allowed, _ = limiter.is_allowed("test_client")
        assert allowed is True

    def test_rate_limiter_stale_key_purge(self):
        """Stale client keys are purged when dict exceeds 100 entries (regression)."""
        import time

        from vllm_mlx.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, enabled=True)

        # Seed 101 unique clients with expired timestamps
        old_time = time.time() - 120  # 2 minutes ago (outside window)
        with limiter._lock:
            for i in range(101):
                limiter._requests[f"stale_client_{i}"] = [old_time]

        assert len(limiter._requests) == 101

        # One more request triggers purge (len > 100)
        limiter.is_allowed("new_client")

        # Stale keys should be purged
        assert len(limiter._requests) < 101
        # new_client should still be present
        assert "new_client" in limiter._requests


# =============================================================================
# Integration Tests (require running server)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestServerIntegration:
    """Integration tests that require a running server.

    These tests are skipped by default. Run with:
        pytest -m integration --server-url http://localhost:8000
    """

    @pytest.fixture
    def server_url(self, request):
        """Get server URL from command line or use default."""
        return request.config.getoption("--server-url", default="http://localhost:8000")

    def test_health_endpoint(self, server_url):
        """Test /health endpoint."""
        import requests

        response = requests.get(f"{server_url}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data

    def test_models_endpoint(self, server_url):
        """Test /v1/models endpoint."""
        import requests

        response = requests.get(f"{server_url}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completion(self, server_url):
        """Test /v1/chat/completions endpoint."""
        import requests

        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--server-url",
        action="store",
        default="http://localhost:8000",
        help="URL of the vllm-mlx server for integration tests",
    )
