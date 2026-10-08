# SPDX-License-Identifier: Apache-2.0
"""Video input degrades to a client-safe 400 when OpenCV is absent.

Rapid-MLX Desktop omits opencv-python (its macOS wheels embed a
GPL-configured FFmpeg). Frame extraction must then reject the request with a
typed, actionable message instead of an ImportError that aborts the engine.
"""

from __future__ import annotations

import sys

import pytest

from rapid_mlx.models import mllm
from rapid_mlx.request import ClientRequestError, is_media_input_error


def test_missing_opencv_is_a_client_request_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)

    with pytest.raises(ClientRequestError) as caught:
        mllm.extract_video_frames_smart("clip.mp4")

    message = str(caught.value)
    assert message == mllm.VIDEO_INPUT_UNAVAILABLE_MESSAGE
    assert "Rapid-MLX Desktop omits it" in message
    assert is_media_input_error(caught.value)
    assert isinstance(caught.value.__cause__, ImportError)
