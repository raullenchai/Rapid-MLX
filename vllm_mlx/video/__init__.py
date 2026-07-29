# SPDX-License-Identifier: Apache-2.0
"""Video support for rapid-mlx — the text→video / image→video lane.

Provides:
- VideoEngine: MLX-native LTX-2.3 backend (t2v + i2v, native audio) via the
  optional ``[video]`` extra (``pip install '.[video]'``).

This is the fourth generation lane alongside audio's TTS / STT / Music. The
concrete ``VideoEngine`` here is intended to satisfy the ``VideoEngine``
Protocol behind the ``/v1/video/generations`` route (PR #1300). See
``vllm_mlx/video/engine.py`` for details.
"""

from .engine import DEFAULT_MODEL, VideoEngine, generate_video

__all__ = [
    "VideoEngine",
    "generate_video",
    "DEFAULT_MODEL",
]
