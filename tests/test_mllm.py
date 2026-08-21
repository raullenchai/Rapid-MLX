# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Multimodal Language Model (MLLM) wrapper."""

import platform
import sys
from pathlib import Path

import pytest
import requests

apple_silicon_only = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_mllm_model():
    """Return a small MLLM model for testing."""
    return "mlx-community/Qwen3-VL-4B-Instruct-3bit"


@pytest.fixture
def test_image_path(tmp_path, monkeypatch):
    """Download a real image from Wikimedia Commons for tests."""
    pytest.importorskip("PIL")
    import io

    import requests
    from PIL import Image

    # Use a small dog image from Wikimedia Commons (public domain)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        path = tmp_path / "test_image.jpg"
        img.save(path)
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        return str(path)
    except Exception:
        # Fallback to synthetic image if download fails
        img = Image.new("RGB", (320, 240), color="blue")
        path = tmp_path / "test_image.jpg"
        img.save(path)
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        return str(path)


@pytest.fixture
def test_video_path(tmp_path, monkeypatch):
    """Download a real video from Wikimedia Commons for tests."""
    import requests

    # Use a short video from Wikimedia Commons (Creative Commons)
    # This is a 3-second sample video
    url = "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.160p.webm"

    path = tmp_path / "test_video.webm"

    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        return str(path)
    except Exception:
        # Fallback to synthetic video if download fails
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))

        # Create 30 frames (1 second)
        for i in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:] = (255, 0, 0)  # Blue in BGR
            out.write(frame)

        out.release()
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        return str(path)


# =============================================================================
# Unit Tests - No Model Loading Required
# =============================================================================


@apple_silicon_only
class TestMLLMHelperFunctions:
    """Test helper functions that don't require model loading."""

    def test_is_base64_image(self):
        """Test base64 image detection."""
        from vllm_mlx.models.mllm import is_base64_image

        assert is_base64_image("data:image/png;base64,iVBORw0KGgo=")
        assert is_base64_image("data:image/jpeg;base64,/9j/4AAQSkZJRg==")
        assert not is_base64_image("https://example.com/image.jpg")
        assert not is_base64_image("/path/to/image.jpg")

    def test_is_base64_video(self):
        """Test base64 video detection."""
        from vllm_mlx.models.mllm import is_base64_video

        assert is_base64_video("data:video/mp4;base64,AAAA")
        assert is_base64_video("data:video/webm;base64,AAAA")
        assert not is_base64_video("https://example.com/video.mp4")
        assert not is_base64_video("/path/to/video.mp4")

    def test_is_url(self):
        """Test URL detection."""
        from vllm_mlx.models.mllm import is_url

        assert is_url("https://example.com/image.jpg")
        assert is_url("http://example.com/video.mp4")
        assert not is_url("/path/to/file.jpg")
        assert not is_url("data:image/png;base64,AAAA")


@apple_silicon_only
class TestVideoFrameExtraction:
    """Test video frame extraction functions."""

    def test_get_video_info(self, test_video_path):
        """Test getting video information."""
        cv2 = pytest.importorskip("cv2")

        # Use OpenCV directly since get_video_info may not be exported
        cap = cv2.VideoCapture(test_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Video from Wikimedia will have different properties
        assert total_frames > 0
        assert fps > 0
        assert width > 0
        assert height > 0

    def test_extract_video_frames_smart(self, test_video_path):
        """Test smart frame extraction."""
        pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart

        # Extract frames
        frames = extract_video_frames_smart(test_video_path, fps=2.0, max_frames=10)

        assert len(frames) > 0
        assert len(frames) <= 10
        # Check frame shape (height, width, channels)
        assert len(frames[0].shape) == 3  # Should be 3D array

    def test_extract_frames_respects_max_frames(self, test_video_path):
        """Test that max_frames limit is respected."""
        pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart

        frames = extract_video_frames_smart(test_video_path, fps=30.0, max_frames=5)

        assert len(frames) <= 5

    def test_save_frames_to_temp(self, test_video_path):
        """Test saving frames to temp files."""
        pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart, save_frames_to_temp

        frames = extract_video_frames_smart(test_video_path, fps=1.0, max_frames=2)
        paths = save_frames_to_temp(frames)

        assert len(paths) == len(frames)
        for path in paths:
            assert Path(path).exists()
            assert path.endswith(".jpg")


@apple_silicon_only
class TestImageProcessing:
    """Test image processing functions."""

    def test_process_image_input_local_file(self, test_image_path):
        """Test processing local image file."""
        from vllm_mlx.models.mllm import process_image_input

        result = process_image_input(test_image_path)
        assert Path(result).read_bytes() == Path(test_image_path).read_bytes()

    def test_process_image_input_dict_format(self, test_image_path):
        """Test processing image in dict format."""
        from vllm_mlx.models.mllm import process_image_input

        # OpenAI format
        result = process_image_input({"url": test_image_path})
        assert Path(result).exists()

    def test_prepare_images_raises_on_fetch_failure(self, monkeypatch):
        """MLLMModel._prepare_images must raise (not swallow) image-fetch errors.

        Regression for #457: image-fetch errors used to be caught with
        ``try/except + logger.warning + continue``, leaving the request to
        proceed with zero images. VLMs then returned HTTP 200 + empty
        completion + ``finish_reason=length`` (or hallucinated content from
        no input), which was indistinguishable from a model refusal.

        The fix raises ``ValueError("Failed to process image: ...")``, which
        the chat/anthropic route exception handlers convert to HTTP 400 with
        a descriptive detail.
        """
        from vllm_mlx import models
        from vllm_mlx.models.mllm import MLXMultimodalLM

        def _boom(img):
            raise RuntimeError("404 Client Error: NOT FOUND for url: http://x/y.jpg")

        monkeypatch.setattr(models.mllm, "process_image_input", _boom)

        # Construct minimally — _prepare_images doesn't touch self at all,
        # so __new__ avoids the heavy __init__ (no model load).
        instance = MLXMultimodalLM.__new__(MLXMultimodalLM)

        with pytest.raises(ValueError, match="Failed to process image"):
            instance._prepare_images(["http://x/y.jpg"])

    def test_prepare_images_propagates_first_failure(self, monkeypatch):
        """First image failure raises immediately — no partial-image fallback.

        OpenAI's GPT-4V fails strict on any image error. Mirroring that
        prevents the silent-partial-failure mode where some images load but
        others get skipped and the model answers based on incomplete input.
        """
        from vllm_mlx import models
        from vllm_mlx.models.mllm import MLXMultimodalLM

        call_count = {"n": 0}

        def _boom_on_second(img):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "/tmp/ok.jpg"
            raise RuntimeError("network down")

        monkeypatch.setattr(models.mllm, "process_image_input", _boom_on_second)
        instance = MLXMultimodalLM.__new__(MLXMultimodalLM)

        with pytest.raises(ValueError, match="Failed to process image"):
            instance._prepare_images(["ok", "bad"])

        # Failed on second image (after first succeeded) — strict semantics.
        assert call_count["n"] == 2


@apple_silicon_only
class TestVideoProcessing:
    """Test video processing functions."""

    def test_process_video_input_local_file(self, test_video_path):
        """Test processing local video file."""
        from vllm_mlx.models.mllm import process_video_input

        result = process_video_input(test_video_path)
        assert Path(result).read_bytes() == Path(test_video_path).read_bytes()

    def test_process_video_input_dict_format(self, test_video_path):
        """Test processing video in dict format."""
        from vllm_mlx.models.mllm import process_video_input

        # OpenAI format
        result = process_video_input({"url": test_video_path})
        assert Path(result).exists()

    def test_process_video_input_empty_raises(self):
        """Test that empty input raises error."""
        from vllm_mlx.models.mllm import process_video_input

        with pytest.raises(ValueError):
            process_video_input("")

        with pytest.raises(ValueError):
            process_video_input({})


class TestMediaSecurity:
    """SSRF + arbitrary-file-read hardening for image/video media inputs."""

    @staticmethod
    def _fake_getaddrinfo(host_to_ips):
        """Host-aware getaddrinfo stand-in so unit tests never hit DNS."""
        import socket as _socket

        def _fake(host, port, family=0, type=0, proto=0, flags=0):
            ips = host_to_ips.get(host) or host_to_ips.get("*", ["93.184.216.34"])
            return [
                (_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips
            ]

        return _fake

    # ---- SSRF / remote-URL guard -----------------------------------------

    def test_assert_safe_remote_url_allows_public(self, monkeypatch):
        from vllm_mlx.models.mllm import _assert_safe_remote_url

        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo({"*": ["93.184.216.34"]}),
        )
        # Should not raise.
        _assert_safe_remote_url("http://example.com/photo.jpg")

    def test_assert_safe_remote_url_blocks_loopback(self, monkeypatch):
        from vllm_mlx.models.mllm import RemoteMediaFetchError, _assert_safe_remote_url

        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo({"*": ["127.0.0.1"]}),
        )
        with pytest.raises(RemoteMediaFetchError):
            _assert_safe_remote_url("http://internal.local/i.jpg")

    def test_assert_safe_remote_url_blocks_metadata(self, monkeypatch):
        from vllm_mlx.models.mllm import RemoteMediaFetchError, _assert_safe_remote_url

        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo({"*": ["169.254.169.254"]}),
        )
        with pytest.raises(RemoteMediaFetchError):
            _assert_safe_remote_url("http://metadata.internal/latest/meta-data/")

    def test_assert_safe_remote_url_blocks_rfc1918(self, monkeypatch):
        from vllm_mlx.models.mllm import RemoteMediaFetchError, _assert_safe_remote_url

        for ip in ("10.0.0.1", "172.16.0.1", "192.168.1.1"):
            monkeypatch.setattr(
                "vllm_mlx.models.mllm.socket.getaddrinfo",
                self._fake_getaddrinfo({"*": [ip]}),
            )
            with pytest.raises(RemoteMediaFetchError):
                _assert_safe_remote_url("http://internal.example/i.jpg")

    def test_guarded_request_blocks_redirect_to_internal(self, monkeypatch):
        """A 302 to a private/metadata address must be refused, not followed."""
        from vllm_mlx.models.mllm import RemoteMediaFetchError, _guarded_request

        class _Redirect:
            is_redirect = True
            is_permanent_redirect = False
            headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

            def close(self):
                pass

            def raise_for_status(self):
                pass

        class _FakeSession:
            def request(self, method, url, **kwargs):
                return _Redirect()

            def close(self):
                pass

        monkeypatch.setattr("vllm_mlx.models.mllm.requests.Session", _FakeSession)
        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo(
                {
                    "evil.example": ["93.184.216.34"],
                    "169.254.169.254": ["169.254.169.254"],
                }
            ),
        )
        with pytest.raises(RemoteMediaFetchError):
            _guarded_request(
                "GET",
                "http://evil.example/a.jpg",
                timeout=5,
                headers={},
                stream=True,
            )

    def test_guarded_request_connects_to_validated_address(self, monkeypatch):
        from vllm_mlx.models.mllm import _close_guarded_response, _guarded_request

        requested = {}

        class _Response:
            is_redirect = False
            is_permanent_redirect = False

            def close(self):
                requested["response_closed"] = True

        class _FakeSession:
            def request(self, method, url, **kwargs):
                requested.update(method=method, url=url, kwargs=kwargs)
                return _Response()

            def close(self):
                requested["session_closed"] = True

        monkeypatch.setattr("vllm_mlx.models.mllm.requests.Session", _FakeSession)
        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo({"cdn.example": ["93.184.216.34"]}),
        )

        response = _guarded_request(
            "GET",
            "http://cdn.example/image.jpg",
            timeout=5,
            headers={},
            stream=True,
        )
        assert requested["url"] == "http://93.184.216.34/image.jpg"
        assert requested["kwargs"]["headers"]["Host"] == "cdn.example"
        _close_guarded_response(response)
        assert requested["response_closed"] is True
        assert requested["session_closed"] is True

    def test_guarded_request_retries_all_validated_addresses(self, monkeypatch):
        from vllm_mlx.models.mllm import _close_guarded_response, _guarded_request

        attempted = []

        class _Response:
            is_redirect = False
            is_permanent_redirect = False

            def close(self):
                pass

        class _FakeSession:
            def request(self, method, url, **kwargs):
                attempted.append(url)
                if "93.184.216.34" in url:
                    raise requests.ConnectionError("first address unavailable")
                return _Response()

            def close(self):
                pass

        monkeypatch.setattr("vllm_mlx.models.mllm.requests.Session", _FakeSession)
        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo({"cdn.example": ["93.184.216.34", "93.184.216.35"]}),
        )
        response = _guarded_request(
            "GET", "http://cdn.example/a.jpg", timeout=5, headers={}, stream=True
        )
        assert attempted == [
            "http://93.184.216.34/a.jpg",
            "http://93.184.216.35/a.jpg",
        ]
        _close_guarded_response(response)

    def test_guarded_request_brackets_ipv6_host_header(self, monkeypatch):
        from vllm_mlx.models.mllm import _close_guarded_response, _guarded_request

        captured = {}

        class _Response:
            is_redirect = False
            is_permanent_redirect = False

            def close(self):
                pass

        class _FakeSession:
            def request(self, method, url, **kwargs):
                captured.update(url=url, headers=kwargs["headers"])
                return _Response()

            def close(self):
                pass

        address = "2606:4700:4700::1111"
        monkeypatch.setattr("vllm_mlx.models.mllm.requests.Session", _FakeSession)
        monkeypatch.setattr(
            "vllm_mlx.models.mllm.socket.getaddrinfo",
            self._fake_getaddrinfo({"*": [address]}),
        )
        response = _guarded_request(
            "GET", f"http://[{address}]/a.jpg", timeout=5, headers={}, stream=True
        )
        assert captured["headers"]["Host"] == f"[{address}]"
        assert captured["url"] == f"http://[{address}]/a.jpg"
        _close_guarded_response(response)

    # ---- arbitrary-file-read guard --------------------------------------

    def test_local_media_blocks_non_media_extension(self, tmp_path):
        from vllm_mlx.models.mllm import process_image_input, process_video_input

        secret = tmp_path / "config.json"
        secret.write_text('{"api_key": "sekrit"}')
        with pytest.raises(ValueError):
            process_image_input(str(secret))
        with pytest.raises(ValueError):
            process_video_input(str(secret))

    def test_local_media_blocks_secret_with_media_suffix(self, tmp_path):
        from vllm_mlx.models.mllm import process_image_input, process_video_input

        disguised = tmp_path / "credentials.jpg"
        disguised.write_text("api_key=secret")
        with pytest.raises(ValueError):
            process_image_input(str(disguised))
        with pytest.raises(ValueError):
            process_video_input(str(disguised))

    def test_local_media_blocks_etc_passwd(self, monkeypatch, tmp_path):
        from vllm_mlx.models.mllm import process_image_input

        # Extension guard already covers /etc/passwd; rooting makes it airtight.
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        with pytest.raises(ValueError):
            process_image_input("/etc/passwd")

    def test_local_media_confined_to_media_root(self, monkeypatch, tmp_path):
        from vllm_mlx.models.mllm import process_image_input, process_video_input

        root = tmp_path / "media"
        root.mkdir()
        good = root / "img.jpg"
        good.write_bytes(b"\xff\xd8\xff\xe0")

        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"\xff\xd8\xff\xe0")

        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(root))
        copied = Path(process_image_input(str(good)))
        assert copied != good
        assert copied.read_bytes() == good.read_bytes()
        with pytest.raises(ValueError):
            process_image_input(str(outside))  # inside tmp but outside root
        with pytest.raises(ValueError):
            process_image_input(str(root / ".." / "outside.jpg"))  # .. escape
        with pytest.raises(ValueError):
            process_video_input(str(outside))

    def test_local_media_invalid_root_fails_closed(self, monkeypatch, tmp_path):
        from vllm_mlx.models.mllm import process_image_input, process_video_input

        image = tmp_path / "image.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0")
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path / "missing"))
        with pytest.raises(ValueError, match="MEDIA_ROOT is invalid"):
            process_image_input(str(image))
        with pytest.raises(ValueError, match="MEDIA_ROOT is invalid"):
            process_video_input(str(image))

    def test_local_media_rejects_symlink(self, monkeypatch, tmp_path):
        from vllm_mlx.models.mllm import process_image_input, process_video_input

        image = tmp_path / "image.jpg"
        image.write_bytes(b"\xff\xd8\xff\xe0")
        link = tmp_path / "link.jpg"
        link.symlink_to(image)
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        with pytest.raises(ValueError):
            process_image_input(str(link))
        with pytest.raises(ValueError):
            process_video_input(str(link))

    def test_disable_local_media_paths_env(self, monkeypatch, tmp_path):
        from vllm_mlx.models.mllm import process_image_input, process_video_input

        img = tmp_path / "ok.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        monkeypatch.setenv("RAPID_MLX_DISABLE_LOCAL_MEDIA_PATHS", "1")
        with pytest.raises(ValueError):
            process_image_input(str(img))
        with pytest.raises(ValueError):
            process_video_input(str(img))


# =============================================================================
# MLLM Model Tests
# =============================================================================


@apple_silicon_only
class TestMLLMModelInit:
    """Test MLLM model initialization (no model loading)."""

    def test_model_init(self):
        """Test model initialization."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        assert model.model_name == "test-model"
        assert not model._loaded

    def test_model_info_not_loaded(self):
        """Test model info when not loaded."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        info = model.get_model_info()

        assert info["loaded"] is False
        assert info["model_name"] == "test-model"

    def test_model_repr(self):
        """Test model string representation."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM("test-model")
        repr_str = repr(model)

        assert "MLXMultimodalLM" in repr_str
        assert "test-model" in repr_str


# =============================================================================
# Integration Tests - Require Model Loading (Slow)
# =============================================================================


@pytest.mark.slow
@apple_silicon_only
class TestMLLMImageGeneration:
    """Integration tests for MLLM image generation."""

    def test_generate_with_image(self, small_mllm_model, test_image_path):
        """Test generation with an image."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        output = model.generate(
            prompt="What animal is in this image?",
            images=[test_image_path],
            max_tokens=30,
        )

        assert output.text is not None
        assert len(output.text) > 0
        assert output.completion_tokens > 0

    def test_describe_image(self, small_mllm_model, test_image_path):
        """Test describe_image convenience method."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        description = model.describe_image(test_image_path, max_tokens=30)

        assert description is not None
        assert len(description) > 0


@pytest.mark.slow
@apple_silicon_only
class TestMLLMVideoGeneration:
    """Integration tests for MLLM video generation."""

    def test_generate_with_video(self, small_mllm_model, test_video_path):
        """Test generation with a video."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        output = model.generate(
            prompt="Describe this video.",
            videos=[test_video_path],
            video_fps=1.0,
            video_max_frames=4,
            max_tokens=20,
        )

        assert output.text is not None
        assert len(output.text) > 0

    def test_describe_video(self, small_mllm_model, test_video_path):
        """Test describe_video convenience method."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        description = model.describe_video(
            test_video_path,
            fps=1.0,
            max_frames=4,
            max_tokens=20,
        )

        assert description is not None
        assert len(description) > 0


@pytest.mark.slow
@apple_silicon_only
class TestMLLMChat:
    """Integration tests for MLLM chat interface."""

    def test_chat_with_image(self, small_mllm_model, test_image_path):
        """Test chat interface with image."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image_path},
                    {"type": "text", "text": "What animal is this?"},
                ],
            }
        ]

        output = model.chat(messages, max_tokens=30)

        assert output.text is not None
        assert len(output.text) > 0

    def test_chat_with_video(self, small_mllm_model, test_video_path):
        """Test chat interface with video."""
        pytest.importorskip("mlx_vlm")
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM(small_mllm_model)
        model.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": test_video_path},
                    {"type": "text", "text": "Describe the colors in this video."},
                ],
            }
        ]

        output = model.chat(messages, max_tokens=30, video_fps=1.0, video_max_frames=4)

        assert output.text is not None
        assert len(output.text) > 0
