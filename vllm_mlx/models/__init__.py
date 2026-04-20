# SPDX-License-Identifier: Apache-2.0
"""
MLX Model wrappers for vLLM.

MLXMultimodalLM wraps mlx-vlm for vision models.
LLM models are loaded directly via mlx-lm (no wrapper needed).
"""

from vllm_mlx.models.mllm import MLXMultimodalLM

MLXVisionLanguageModel = MLXMultimodalLM

__all__ = ["MLXMultimodalLM", "MLXVisionLanguageModel"]
