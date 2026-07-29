# SPDX-License-Identifier: Apache-2.0
"""Video-generation lane (content-farm engine).

CONTRACT-ONLY today: :mod:`vllm_mlx.video.engine` defines the
``VideoEngine`` interface a future LTX-2.3 backend implements. The
``/v1/video/generations`` route (see :mod:`vllm_mlx.routes.video`) ships
the request/response schema now and returns HTTP 501 until a concrete
backend is registered.
"""

from .engine import VideoEngine, resolve_video_engine

__all__ = ["VideoEngine", "resolve_video_engine"]
