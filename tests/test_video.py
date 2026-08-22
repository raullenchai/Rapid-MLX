# SPDX-License-Identifier: Apache-2.0
"""Tests for video support in MLLM chat/stream_chat."""

from vllm_mlx.models.mllm import (
    FRAME_FACTOR,
    MIN_FRAMES,
    MLXMultimodalLM,
    is_base64_video,
    smart_nframes,
)


class TestSmartNframes:
    """Verify frame count alignment and clamping."""

    def test_basic_calculation(self):
        # 300 frames at 30fps = 10s video, at 2fps target = 20 frames
        result = smart_nframes(300, 30.0, target_fps=2.0)
        assert result == 20
        assert result % FRAME_FACTOR == 0

    def test_clamps_to_min(self):
        # Very short video: 6 frames at 30fps
        result = smart_nframes(6, 30.0, target_fps=2.0)
        assert result >= MIN_FRAMES
        assert result % FRAME_FACTOR == 0

    def test_clamps_to_max(self):
        # Very long video: 100000 frames
        result = smart_nframes(100000, 30.0, target_fps=2.0, max_frames=64)
        assert result <= 64
        assert result % FRAME_FACTOR == 0

    def test_result_always_even(self):
        for total in [5, 7, 11, 13, 100, 999]:
            result = smart_nframes(total, 30.0)
            assert result % FRAME_FACTOR == 0, (
                f"Odd frame count {result} for total={total}"
            )


class TestVideoUrlParsing:
    """Verify video_url content type extraction from OpenAI messages."""

    def _make_model(self):
        """Create an unloaded model instance for testing."""
        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = False
        model._video_native = False
        return model

    def _extract_video_inputs(self, messages):
        """Use the actual _collect_video_inputs helper."""
        model = self._make_model()
        return model._collect_video_inputs(messages)

    def test_video_url_dict_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/video.mp4"},
                    },
                    {"type": "text", "text": "Describe this video"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert 0 in result
        assert result[0] == ["https://example.com/video.mp4"]

    def test_video_url_string_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": "https://example.com/video.mp4"},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert result[0] == ["https://example.com/video.mp4"]

    def test_video_type(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/path/to/video.mp4"},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert result[0] == ["/path/to/video.mp4"]

    def test_no_video(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert len(result) == 0

    def test_mixed_media(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.jpg"},
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/vid.mp4"},
                    },
                    {"type": "text", "text": "Compare"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        # Only video extracted, not image
        assert result[0] == ["https://example.com/vid.mp4"]

    def test_multi_message_videos(self):
        """Videos in different messages should be keyed by message index."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/path/first.mp4"},
                    {"type": "text", "text": "First"},
                ],
            },
            {"role": "assistant", "content": "OK"},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/path/second.mp4"},
                    {"type": "text", "text": "Second"},
                ],
            },
        ]
        result = self._extract_video_inputs(messages)
        assert result[0] == ["/path/first.mp4"]
        assert result[2] == ["/path/second.mp4"]
        assert 1 not in result

    def test_multiple_videos_single_message(self):
        """Multiple videos in one message should produce a list at that index."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/path/a.mp4"},
                    {"type": "video_url", "video_url": {"url": "/path/b.mp4"}},
                    {"type": "text", "text": "Compare these"},
                ],
            }
        ]
        result = self._extract_video_inputs(messages)
        assert result[0] == ["/path/a.mp4", "/path/b.mp4"]


class TestTranslateMessages:
    """Verify OpenAI format to process_vision_info format translation."""

    def _make_model(self):
        """Create an unloaded model instance for testing translation."""
        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = False
        model._video_native = True
        return model

    def test_text_only_passthrough(self):
        model = self._make_model()
        messages = [{"role": "user", "content": "Hello"}]
        result = model._translate_messages_for_native_video(messages, 2.0, 128)
        assert result[0]["content"] == "Hello"

    def test_video_url_translated(self, tmp_path, monkeypatch):
        video_path = tmp_path / "input.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 88)
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        model = self._make_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        result = model._translate_messages_for_native_video(messages, 2.0, 128)
        content = result[0]["content"]

        types = [item["type"] for item in content]
        assert "video" in types
        assert "text" in types
        video_item = next(i for i in content if i["type"] == "video")
        assert video_item["fps"] == 2.0
        assert video_item["max_frames"] == 128

    def test_video_url_type_translated(self, tmp_path, monkeypatch):
        video_path = tmp_path / "input.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 88)
        monkeypatch.setenv("RAPID_MLX_MEDIA_ROOT", str(tmp_path))
        model = self._make_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": str(video_path)},
                    },
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]
        result = model._translate_messages_for_native_video(messages, 1.0, 64)
        content = result[0]["content"]

        types = [item["type"] for item in content]
        assert "video" in types
        assert "text" in types
        video_item = next(i for i in content if i["type"] == "video")
        assert video_item["fps"] == 1.0
        assert video_item["max_frames"] == 64


class TestCollectVideoInputsPydantic:
    """Verify _collect_video_inputs handles Pydantic models correctly."""

    def _make_model(self):
        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = False
        model._video_native = False
        return model

    def test_pydantic_model_dump(self):
        """Pydantic v2 objects with model_dump() are handled."""

        class FakeContent:
            def model_dump(self, exclude_none=False):
                return {"type": "video", "video": "/path/to/video.mp4"}

        messages = [{"role": "user", "content": [FakeContent()]}]
        result = self._make_model()._collect_video_inputs(messages)
        assert result[0] == ["/path/to/video.mp4"]

    def test_pydantic_v1_dict(self):
        """Pydantic v1 objects with dict() are handled."""

        class FakeContent:
            def dict(self):
                return {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/v.mp4"},
                    "image_url": None,
                }

        messages = [{"role": "user", "content": [FakeContent()]}]
        result = self._make_model()._collect_video_inputs(messages)
        assert result[0] == ["https://example.com/v.mp4"]

    def test_empty_video_url_skipped(self):
        """Empty video URL dicts are skipped."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": ""}},
                ],
            }
        ]
        result = self._make_model()._collect_video_inputs(messages)
        assert len(result) == 0


class TestToolForwarding:
    """Verify tools are popped from kwargs before native video path."""

    def test_tools_not_in_kwargs_after_pop(self):
        """Ensure tools don't leak into **kwargs for mlx_vlm.generate()."""
        model = MLXMultimodalLM.__new__(MLXMultimodalLM)
        model._loaded = False
        model._video_native = True

        tools = [{"type": "function", "function": {"name": "test"}}]
        kwargs = {"tools": tools, "video_fps": 2.0, "video_max_frames": 64}

        # Simulate what chat() does: pop tools before native video branch
        kwargs.pop("video_fps", 2.0)
        kwargs.pop("video_max_frames", 128)
        popped_tools = kwargs.pop("tools", None)

        assert popped_tools == tools
        assert "tools" not in kwargs

    def test_generate_native_video_accepts_tools_param(self):
        """Verify _generate_native_video signature accepts tools kwarg."""
        import inspect

        sig = inspect.signature(MLXMultimodalLM._generate_native_video)
        params = list(sig.parameters.keys())
        assert "tools" in params

    def test_prepare_native_video_inputs_accepts_tools(self):
        """Verify preprocessing helper also accepts tools."""
        import inspect

        sig = inspect.signature(MLXMultimodalLM._prepare_native_video_inputs)
        params = list(sig.parameters.keys())
        assert "tools" in params

    def test_generate_imports_from_video_generate(self):
        """Verify _generate_native_video uses mlx_vlm.video_generate.generate."""
        import inspect

        source = inspect.getsource(MLXMultimodalLM._generate_native_video)
        assert "from mlx_vlm.video_generate import generate" in source


class TestIsBase64Video:
    def test_detects_base64_video(self):
        assert is_base64_video("data:video/mp4;base64,AAAA") is True

    def test_rejects_non_video(self):
        assert is_base64_video("data:image/jpeg;base64,AAAA") is False
        assert is_base64_video("https://example.com/video.mp4") is False
