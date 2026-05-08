# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Multimodal Language Model (MLLM) wrapper."""

import base64
import platform
import sys
from pathlib import Path

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
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
def test_image_path(tmp_path):
    """Download a real image from Wikimedia Commons for tests."""
    pytest.importorskip("PIL")
    import requests
    from PIL import Image
    import io

    # Use a small dog image from Wikimedia Commons (public domain)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return str(path)
    except Exception:
        # Fallback to synthetic image if download fails
        img = Image.new("RGB", (320, 240), color="blue")
        path = tmp_path / "test_image.jpg"
        img.save(path)
        return str(path)


@pytest.fixture
def test_video_path(tmp_path):
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
        return str(path)


@pytest.fixture
def test_audio_path(tmp_path):
    """Create a small local WAV file for audio input tests."""
    import wave

    path = tmp_path / "test_audio.wav"
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return str(path)


# =============================================================================
# Unit Tests - No Model Loading Required
# =============================================================================


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

    def test_is_base64_audio(self):
        """Test base64 audio detection."""
        from vllm_mlx.models.mllm import is_base64_audio

        assert is_base64_audio("data:audio/wav;base64,AAAA")
        assert is_base64_audio("data:audio/mp3;base64,AAAA")
        assert not is_base64_audio("https://example.com/audio.wav")
        assert not is_base64_audio("/path/to/audio.wav")

    def test_is_url(self):
        """Test URL detection."""
        from vllm_mlx.models.mllm import is_url

        assert is_url("https://example.com/image.jpg")
        assert is_url("http://example.com/video.mp4")
        assert not is_url("/path/to/file.jpg")
        assert not is_url("data:image/png;base64,AAAA")

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "http://127.0.0.1/private.jpg",
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost/private.jpg",
            "http://[::1]/private.jpg",
        ],
    )
    def test_validate_url_safety_rejects_unsafe_targets(self, url):
        """Test that local and private targets are rejected."""
        from vllm_mlx.models.mllm import UnsafeRemoteURLError, _validate_url_safety

        with pytest.raises(UnsafeRemoteURLError):
            _validate_url_safety(url)

    def test_validate_url_safety_allows_public_ip(self):
        """Test that public literal IPs remain allowed."""
        from vllm_mlx.models.mllm import _validate_url_safety

        _validate_url_safety("https://8.8.8.8/image.jpg")

    def test_unsafe_remote_url_error_has_safe_public_message(self):
        """Public safety errors should not disclose resolved hosts or IPs."""
        from vllm_mlx.models.mllm import UnsafeRemoteURLError, _validate_url_safety

        with pytest.raises(UnsafeRemoteURLError) as exc_info:
            _validate_url_safety("http://169.254.169.254/latest/meta-data/")

        assert "169.254.169.254" in str(exc_info.value)
        assert exc_info.value.public_message == "Remote media URL is not allowed"
        assert "169.254.169.254" not in exc_info.value.public_message

    def test_request_with_safe_redirects_blocks_unsafe_redirect(self, monkeypatch):
        """Test that redirect hops are validated before a second request."""
        from vllm_mlx.models import mllm

        calls = []

        class FakeResponse:
            def __init__(self, url, status_code, headers):
                self.url = url
                self.status_code = status_code
                self.headers = headers
                self.is_redirect = status_code in {301, 302, 303, 307, 308}
                self.is_permanent_redirect = status_code in {301, 308}

            def close(self):
                return None

        def fake_request(method, url, **kwargs):
            calls.append((method, url, kwargs["allow_redirects"]))
            return FakeResponse(
                url,
                302,
                {"location": "http://127.0.0.1/internal.jpg"},
            )

        monkeypatch.setattr(mllm.requests, "request", fake_request)

        with pytest.raises(mllm.UnsafeRemoteURLError):
            mllm._request_with_safe_redirects(
                "GET",
                "https://8.8.8.8/image.jpg",
                timeout=30,
                headers={"User-Agent": "test"},
                stream=True,
            )

        assert calls == [("GET", "https://8.8.8.8/image.jpg", False)]


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
        cv2 = pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart

        # Extract frames
        frames = extract_video_frames_smart(test_video_path, fps=2.0, max_frames=10)

        assert len(frames) > 0
        assert len(frames) <= 10
        # Check frame shape (height, width, channels)
        assert len(frames[0].shape) == 3  # Should be 3D array

    def test_extract_frames_respects_max_frames(self, test_video_path):
        """Test that max_frames limit is respected."""
        cv2 = pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart

        frames = extract_video_frames_smart(test_video_path, fps=30.0, max_frames=5)

        assert len(frames) <= 5

    def test_save_frames_to_temp(self, test_video_path):
        """Test saving frames to temp files."""
        cv2 = pytest.importorskip("cv2")
        from vllm_mlx.models.mllm import extract_video_frames_smart, save_frames_to_temp

        frames = extract_video_frames_smart(test_video_path, fps=1.0, max_frames=2)
        paths = save_frames_to_temp(frames)

        assert len(paths) == len(frames)
        for path in paths:
            assert Path(path).exists()
            assert path.endswith(".jpg")


class TestImageProcessing:
    """Test image processing functions."""

    def test_process_image_input_local_file_rejected(self, test_image_path):
        """Test that local image paths are rejected."""
        from vllm_mlx.models.mllm import process_image_input

        with pytest.raises(ValueError, match="Unsupported image input"):
            process_image_input(test_image_path)

    def test_process_image_input_dict_format_base64(self):
        """Test processing image in dict format with base64 payload."""
        from vllm_mlx.models.mllm import process_image_input

        image_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR").decode()
        result = process_image_input({"url": f"data:image/png;base64,{image_b64}"})
        assert Path(result).exists()

    def test_download_image_blocks_unsafe_url_before_request(self, monkeypatch):
        """Test that blocked image URLs fail before any request is made."""
        from vllm_mlx.models import mllm

        def fail_request(*args, **kwargs):
            raise AssertionError("requests.request should not be called")

        monkeypatch.setattr(mllm.requests, "request", fail_request)

        with pytest.raises(mllm.UnsafeRemoteURLError):
            mllm.download_image("http://169.254.169.254/latest/meta-data/")


class TestVideoProcessing:
    """Test video processing functions."""

    def test_process_video_input_local_file_rejected(self, test_video_path):
        """Test that local video paths are rejected."""
        from vllm_mlx.models.mllm import process_video_input

        with pytest.raises(ValueError, match="Unsupported video input"):
            process_video_input(test_video_path)

    def test_process_video_input_dict_format_base64(self):
        """Test processing video in dict format with base64 payload."""
        from vllm_mlx.models.mllm import process_video_input

        video_b64 = base64.b64encode(b"\x00" * 100).decode()
        result = process_video_input({"url": f"data:video/mp4;base64,{video_b64}"})
        assert Path(result).exists()

    def test_process_video_input_empty_raises(self):
        """Test that empty input raises error."""
        from vllm_mlx.models.mllm import process_video_input

        with pytest.raises(ValueError):
            process_video_input("")

        with pytest.raises(ValueError):
            process_video_input({})

    def test_download_video_blocks_unsafe_url_before_request(self, monkeypatch):
        """Test that blocked video URLs fail before any request is made."""
        from vllm_mlx.models import mllm

        def fail_request(*args, **kwargs):
            raise AssertionError("requests.request should not be called")

        monkeypatch.setattr(mllm.requests, "request", fail_request)

        with pytest.raises(mllm.UnsafeRemoteURLError):
            mllm.download_video("http://127.0.0.1/internal.mp4")


class TestAudioProcessing:
    """Test audio processing functions."""

    def test_process_audio_input_local_file(self, test_audio_path):
        """Test processing local audio file."""
        from vllm_mlx.models.mllm import process_audio_input

        result = process_audio_input(test_audio_path)
        assert result == test_audio_path

    def test_process_audio_input_dict_format(self, test_audio_path):
        """Test processing audio in dict format."""
        from pathlib import Path

        from vllm_mlx.models.mllm import process_audio_input

        result = process_audio_input({"url": test_audio_path})
        assert Path(result).exists()

    def test_collect_audio_inputs(self):
        """Audio items should be extracted and keyed by message index."""
        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = False
        model._video_native = False

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/a.wav"},
                    },
                    {"type": "text", "text": "Transcribe this"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "/tmp/b.wav"},
                    {"type": "text", "text": "And this"},
                ],
            },
        ]

        result = model._collect_audio_inputs(messages)
        assert result[0] == ["https://example.com/a.wav"]
        assert result[1] == ["/tmp/b.wav"]


# =============================================================================
# MLLM Model Tests
# =============================================================================


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
